"""Team attack/defence ratings for fixtures the bookmakers have not priced yet.

Live forecasts plan six gameweeks ahead, but odds exist for only the next one
or two. Unpriced fixtures used to get constant lambdas (home 1.4, away 1.2,
clean sheets 0.30/0.25): opponent-blind for most of the horizon, and the goals
model read their missing market features as zero, deflating attackers.

The ratings are fitted on the market's own expectations, so an unpriced
fixture inherits the market's view of both teams:

    log(lambda) = mu + home * is_home + attack[team] - defence[opponent]

by recency-weighted ridge least squares over this season's priced fixtures
and the previous season's last ten gameweeks. Clean-sheet chance is
exp(-lambda against), scaled to the market's level on the same rows.

Measured two gameweeks ahead on 24/25 and 25/26: lambda error against the
eventual market price halved (0.41 → 0.20, 0.33 → 0.16), and goals deviance
closed 67-81% of the gap between the constants and the market itself
(data/audits/2026-10-19/fallback_ratings.py).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import polars as pl

from fpl_bot.db import pit

RIDGE = 3.0
HALF_LIFE_GWS = 10.0
PRIOR_SEASON_FROM_GW = 29


@dataclass(frozen=True)
class TeamRatings:
    beta: np.ndarray
    index: dict[int, int]
    cs_factor: float

    def expected_goals(self, team: int, opponent: int, home: bool) -> float:
        n = len(self.index)
        value = self.beta[0] + self.beta[1] * home
        if team in self.index:
            value += self.beta[2 + self.index[team]]
        if opponent in self.index:
            value -= self.beta[2 + n + self.index[opponent]]
        return float(np.exp(value))

    def clean_sheet(self, lambda_against: float) -> float:
        return float(min(0.95, self.cs_factor * np.exp(-lambda_against)))


def fit_team_ratings(history: pl.DataFrame, teams: list[int]) -> TeamRatings:
    """Fit on rows with home_team_id, away_team_id, lh, la, csh, csa, age (gameweeks)."""
    index = {t: i for i, t in enumerate(teams)}
    n = len(teams)
    rows, target, weight, cs_ratio = [], [], [], []
    for r in history.iter_rows(named=True):
        w = 0.5 ** (max(r["age"], 0) / HALF_LIFE_GWS)
        sides = (
            (r["home_team_id"], r["away_team_id"], r["lh"], r["csh"], r["la"], 1),
            (r["away_team_id"], r["home_team_id"], r["la"], r["csa"], r["lh"], 0),
        )
        for team, opponent, lam, cs, lam_against, home in sides:
            cs_ratio.append(cs / np.exp(-lam_against))
            if team not in index or opponent not in index:
                continue
            x = np.zeros(2 + 2 * n)
            x[0], x[1] = 1.0, home
            x[2 + index[team]] = 1.0
            x[2 + n + index[opponent]] = -1.0
            rows.append(x)
            target.append(np.log(lam))
            weight.append(w)
    if not rows:
        raise ValueError("No priced fixtures to fit team ratings on.")
    X, y, w = np.array(rows), np.array(target), np.array(weight)
    penalty = np.diag([0.0, 0.0] + [RIDGE] * (2 * n))
    beta = np.linalg.solve(X.T @ (X * w[:, None]) + penalty, X.T @ (w * y))
    return TeamRatings(beta=beta, index=index, cs_factor=float(np.mean(cs_ratio)))


@lru_cache(maxsize=4)
def _priced_sides(season_ids: tuple[int, ...]) -> pl.DataFrame:
    # Cached per process: one forecast asks for the same history several times.
    market = pit.market_xg_for_fixtures()
    if market.is_empty():
        return pl.DataFrame()
    fixtures = pit.upcoming_fixtures(market["fixture_id"].unique().to_list())
    if fixtures.is_empty():
        return pl.DataFrame()
    fixtures = fixtures.filter(pl.col("season_id").is_in(season_ids))
    side = market.select("fixture_id", "team_id", "lambda_market_xg", "cs_prob_market")
    home = side.rename(
        {"team_id": "home_team_id", "lambda_market_xg": "lh", "cs_prob_market": "csh"}
    )
    away = side.rename(
        {"team_id": "away_team_id", "lambda_market_xg": "la", "cs_prob_market": "csa"}
    )
    return fixtures.join(home, on=["fixture_id", "home_team_id"]).join(
        away, on=["fixture_id", "away_team_id"]
    )


def market_xg_with_fallback(season_id: int, fixture_ids: list[int]) -> pl.DataFrame:
    """Market lambdas for `fixture_ids`, with ratings standing in for unpriced ones.

    Same columns as `pit.market_xg_for_fixtures` plus `is_fallback`.
    """
    market = pit.market_xg_for_fixtures(fixture_ids)
    priced = (
        set(market["fixture_id"].to_list()) if not market.is_empty() else set()
    )
    missing = [f for f in fixture_ids if f not in priced]
    out = (
        market.with_columns(pl.lit(False).alias("is_fallback"))
        if not market.is_empty()
        else pl.DataFrame()
    )
    if not missing:
        return out
    fallback = _fallback_rows(season_id, tuple(sorted(missing)))
    if fallback.is_empty():
        return out
    return fallback if out.is_empty() else pl.concat([out, fallback], how="vertical_relaxed")


@lru_cache(maxsize=8)
def _fallback_rows(season_id: int, missing: tuple[int, ...]) -> pl.DataFrame:
    targets = pit.upcoming_fixtures(list(missing))
    history = _priced_sides((season_id - 1, season_id))
    if targets.is_empty() or history.is_empty():
        return pl.DataFrame()
    first_gw = int(targets["gameweek"].min())
    history = history.filter(
        (pl.col("season_id") == season_id)
        | (pl.col("gameweek") >= PRIOR_SEASON_FROM_GW)
    ).with_columns(
        pl.when(pl.col("season_id") == season_id)
        .then(first_gw - pl.col("gameweek"))
        .otherwise(first_gw + 38 - pl.col("gameweek"))
        .alias("age")
    )
    # Every team in the history anchors the scale, not only the target teams.
    teams = sorted(
        set(history["home_team_id"].to_list())
        | set(history["away_team_id"].to_list())
        | set(targets["home_team_id"].to_list())
        | set(targets["away_team_id"].to_list())
    )
    try:
        ratings = fit_team_ratings(history, teams)
    except (ValueError, np.linalg.LinAlgError):
        return pl.DataFrame()  # Callers keep their neutral defaults.
    rows = []
    for t in targets.iter_rows(named=True):
        lh = ratings.expected_goals(t["home_team_id"], t["away_team_id"], home=True)
        la = ratings.expected_goals(t["away_team_id"], t["home_team_id"], home=False)
        rows += [
            {"fixture_id": t["fixture_id"], "team_id": t["home_team_id"],
             "lambda_market_xg": lh, "cs_prob_market": ratings.clean_sheet(la)},
            {"fixture_id": t["fixture_id"], "team_id": t["away_team_id"],
             "lambda_market_xg": la, "cs_prob_market": ratings.clean_sheet(lh)},
        ]
    return pl.DataFrame(rows).with_columns(pl.lit(True).alias("is_fallback"))
