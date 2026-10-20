"""Market-fitted ratings for fixtures the bookmakers have not priced yet."""

from __future__ import annotations

import math

import polars as pl

from fpl_bot.derive import team_ratings
from fpl_bot.derive.team_ratings import fit_team_ratings


def _history() -> pl.DataFrame:
    # Team 1 strong (attack 0.4, defence 0.3), team 3 weak, home advantage 0.15.
    attack = {1: 0.4, 2: 0.0, 3: -0.3}
    defence = {1: 0.3, 2: 0.0, 3: -0.2}
    rows = []
    for home in attack:
        for away in attack:
            if home == away:
                continue
            lh = math.exp(0.2 + 0.15 + attack[home] - defence[away])
            la = math.exp(0.2 + attack[away] - defence[home])
            rows.append({"home_team_id": home, "away_team_id": away, "lh": lh, "la": la,
                         "csh": 0.9 * math.exp(-la), "csa": 0.9 * math.exp(-lh), "age": 1})
    return pl.DataFrame(rows * 4)


def test_ratings_recover_strength_home_edge_and_clean_sheet_level() -> None:
    r = fit_team_ratings(_history(), [1, 2, 3])
    strong_home = r.expected_goals(1, 3, home=True)
    weak_home = r.expected_goals(3, 1, home=True)
    assert strong_home > 2 * weak_home
    assert r.expected_goals(2, 2, home=True) > r.expected_goals(2, 2, home=False)
    assert abs(r.cs_factor - 0.9) < 1e-9
    assert 0 < r.clean_sheet(0.5) < 0.95


def test_unknown_team_gets_the_league_level() -> None:
    r = fit_team_ratings(_history(), [1, 2, 3, 99])
    neutral = r.expected_goals(99, 2, home=False)
    assert r.expected_goals(3, 2, home=False) < neutral < r.expected_goals(1, 2, home=False)


def test_only_unpriced_fixtures_are_filled(monkeypatch) -> None:
    priced = pl.DataFrame({"fixture_id": [10, 10], "team_id": [1, 3],
                           "lambda_market_xg": [2.0, 0.6], "cs_prob_market": [0.5, 0.1]})
    fixtures = pl.DataFrame({"fixture_id": [10, 11], "season_id": [26, 26], "gameweek": [9, 10],
                             "home_team_id": [1, 3], "away_team_id": [3, 1]})
    sides = _history().with_columns(pl.lit(26).alias("season_id"), pl.lit(5).alias("gameweek"))

    def market(ids=None):
        return priced.filter(pl.col("fixture_id").is_in(ids)) if ids is not None else priced

    monkeypatch.setattr(team_ratings.pit, "market_xg_for_fixtures", market)
    monkeypatch.setattr(team_ratings.pit, "upcoming_fixtures",
                        lambda ids: fixtures.filter(pl.col("fixture_id").is_in(ids)))
    monkeypatch.setattr(team_ratings, "_priced_sides", lambda seasons: sides)
    team_ratings._fallback_rows.cache_clear()
    out = team_ratings.market_xg_with_fallback(26, [10, 11])
    assert out.filter(pl.col("fixture_id") == 10)["is_fallback"].to_list() == [False, False]
    filled = out.filter(pl.col("fixture_id") == 11)
    assert filled["is_fallback"].all() and filled.height == 2
    away_strong = filled.filter(pl.col("team_id") == 1)["lambda_market_xg"][0]
    home_weak = filled.filter(pl.col("team_id") == 3)["lambda_market_xg"][0]
    assert away_strong > home_weak  # Opponent strength now matters.


def test_one_unpriced_fixture_still_fits_on_the_whole_league(monkeypatch) -> None:
    """Regression: indexing only the target teams discarded almost all history."""
    fixtures = pl.DataFrame({"fixture_id": [11], "season_id": [26], "gameweek": [10],
                             "home_team_id": [1], "away_team_id": [3]})
    sides = _history().with_columns(pl.lit(26).alias("season_id"), pl.lit(5).alias("gameweek"))
    sides = sides.filter(~((pl.col("home_team_id") == 1) & (pl.col("away_team_id") == 3))
                         & ~((pl.col("home_team_id") == 3) & (pl.col("away_team_id") == 1)))
    monkeypatch.setattr(team_ratings.pit, "market_xg_for_fixtures", lambda ids=None: pl.DataFrame())
    monkeypatch.setattr(team_ratings.pit, "upcoming_fixtures", lambda ids: fixtures)
    monkeypatch.setattr(team_ratings, "_priced_sides", lambda seasons: sides)
    team_ratings._fallback_rows.cache_clear()
    out = team_ratings.market_xg_with_fallback(26, [11])
    assert out.height == 2 and out["is_fallback"].all()


def test_no_history_keeps_callers_on_defaults(monkeypatch) -> None:
    fixtures = pl.DataFrame({"fixture_id": [11], "season_id": [26], "gameweek": [10],
                             "home_team_id": [1], "away_team_id": [3]})
    monkeypatch.setattr(team_ratings.pit, "market_xg_for_fixtures", lambda ids=None: pl.DataFrame())
    monkeypatch.setattr(team_ratings.pit, "upcoming_fixtures", lambda ids: fixtures)
    monkeypatch.setattr(team_ratings, "_priced_sides", lambda seasons: pl.DataFrame())
    team_ratings._fallback_rows.cache_clear()
    assert team_ratings.market_xg_with_fallback(26, [11]).is_empty()
