"""Unit tests for the BPS simulator (Phase 2.4)."""
from __future__ import annotations

import numpy as np
import polars as pl

from fpl_bot.db.event_source import (
    EmpiricalResidualEventSource,
    minutes_to_bucket,
)
from fpl_bot.models.bps import (
    BPS_ASSIST,
    BPS_CLEAN_SHEET_GK_DEF,
    BPS_GOAL_BY_POSITION,
    BPS_GOAL_CONCEDED,
    BPS_LONG_APPEARANCE,
    BPS_PENALTY_GOAL_2025,
    BPS_SHORT_APPEARANCE,
    _bps_from_saves,
    assign_bonus_within_fixture,
    player_sub_rates,
    score_bps_known_events,
    split_p_full_by_position,
)

# ── BPS rule table correctness ───────────────────────────────────────────────


def test_60_min_appearance_is_short_bps_bucket() -> None:
    bps = score_bps_known_events(
        position="MID", minutes=60, goals=0, assists=0,
        team_clean_sheet=False, team_goals_conceded=0,
    )
    assert bps == BPS_SHORT_APPEARANCE


def test_under_60_gets_short_appearance_bonus() -> None:
    bps = score_bps_known_events(
        position="MID", minutes=59, goals=0, assists=0,
        team_clean_sheet=False, team_goals_conceded=0,
    )
    assert bps == BPS_SHORT_APPEARANCE


def test_over_60_gets_long_appearance_bonus() -> None:
    bps = score_bps_known_events(
        position="MID", minutes=61, goals=0, assists=0,
        team_clean_sheet=False, team_goals_conceded=0,
    )
    assert bps == BPS_LONG_APPEARANCE


def test_def_goal_12_bps() -> None:
    bps = score_bps_known_events(
        position="DEF", minutes=90, goals=1, assists=0,
        team_clean_sheet=True, team_goals_conceded=0,
    )
    assert bps == (
        BPS_LONG_APPEARANCE
        + BPS_GOAL_BY_POSITION["DEF"]
        + BPS_CLEAN_SHEET_GK_DEF
    )


def test_mid_goal_18_bps() -> None:
    bps = score_bps_known_events(
        position="MID", minutes=90, goals=1, assists=0,
        team_clean_sheet=False, team_goals_conceded=2,
    )
    assert bps == BPS_LONG_APPEARANCE + BPS_GOAL_BY_POSITION["MID"]


def test_fwd_goal_24_bps() -> None:
    bps = score_bps_known_events(
        position="FWD", minutes=90, goals=1, assists=1,
        team_clean_sheet=False, team_goals_conceded=1,
    )
    assert bps == BPS_LONG_APPEARANCE + BPS_GOAL_BY_POSITION["FWD"] + BPS_ASSIST


def test_2025_penalty_goal_is_12_bps_regardless_of_position() -> None:
    bps = score_bps_known_events(
        position="FWD", minutes=90, goals=1, assists=0,
        team_clean_sheet=False, team_goals_conceded=0,
        penalties_scored=1, season_id=25,
    )
    assert bps == BPS_LONG_APPEARANCE + BPS_PENALTY_GOAL_2025


def test_pre_2025_penalty_goal_keeps_positional_bps() -> None:
    bps = score_bps_known_events(
        position="FWD", minutes=90, goals=1, assists=0,
        team_clean_sheet=False, team_goals_conceded=0,
        penalties_scored=1, season_id=24,
    )
    assert bps == BPS_LONG_APPEARANCE + BPS_GOAL_BY_POSITION["FWD"]


def test_cs_only_for_gk_def_at_90_plus() -> None:
    """MID does NOT get +12 BPS for CS; GK/DEF get it only if they played 60+."""
    mid = score_bps_known_events(
        position="MID", minutes=90, goals=0, assists=0,
        team_clean_sheet=True, team_goals_conceded=0,
    )
    assert mid == BPS_LONG_APPEARANCE  # no CS bonus

    sub_def = score_bps_known_events(
        position="DEF", minutes=45, goals=0, assists=0,
        team_clean_sheet=True, team_goals_conceded=0,
    )
    assert sub_def == BPS_SHORT_APPEARANCE  # didn't play 60+, but appeared


def test_goals_conceded_penalty_for_gk_def_at_90() -> None:
    """GK/DEF lose 4 BPS per goal conceded when playing 90+ min."""
    bps = score_bps_known_events(
        position="DEF", minutes=90, goals=0, assists=0,
        team_clean_sheet=False, team_goals_conceded=3,
    )
    assert bps == BPS_LONG_APPEARANCE + 3 * BPS_GOAL_CONCEDED


def test_pre_2024_has_no_goal_conceded_bps_penalty() -> None:
    bps = score_bps_known_events(
        position="DEF", minutes=90, goals=0, assists=0,
        team_clean_sheet=False, team_goals_conceded=3, season_id=23,
    )
    assert bps == BPS_LONG_APPEARANCE


def test_saves_2_each_for_gk_when_location_unavailable() -> None:
    assert _bps_from_saves(0) == 0
    assert _bps_from_saves(2) == 4
    assert _bps_from_saves(3) == 6
    assert _bps_from_saves(6) == 12


def test_legacy_known_bps_reproduces_pre_correction_rules() -> None:
    bps = score_bps_known_events(
        position="DEF",
        minutes=90,
        goals=1,
        assists=0,
        team_clean_sheet=False,
        team_goals_conceded=2,
        saves=0,
        rules_mode="legacy",
    )
    assert bps == 6 + 24 - 2
    assert _bps_from_saves(5, rules_mode="legacy") == 2


def test_yellow_red_cards() -> None:
    bps = score_bps_known_events(
        position="MID", minutes=60, goals=0, assists=0,
        team_clean_sheet=False, team_goals_conceded=0,
        yellow_cards=1, red_cards=0,
    )
    assert bps == BPS_SHORT_APPEARANCE - 3
    bps_red = score_bps_known_events(
        position="MID", minutes=60, goals=0, assists=0,
        team_clean_sheet=False, team_goals_conceded=0,
        yellow_cards=0, red_cards=1,
    )
    assert bps_red == BPS_SHORT_APPEARANCE - 9


# ── Bonus assignment ─────────────────────────────────────────────────────────


def test_assign_bonus_strict_ranking() -> None:
    bps = np.array([100, 80, 60, 50, 40], dtype=np.float64)
    pids = np.array([1, 2, 3, 4, 5])
    bonus = assign_bonus_within_fixture(bps, pids)
    assert bonus.tolist() == [3, 2, 1, 0, 0]


def test_assign_bonus_with_top_tie() -> None:
    """Two players tied at top — both get 3, no 2 awarded, third place gets 1."""
    bps = np.array([100, 100, 70, 50, 40], dtype=np.float64)
    pids = np.array([1, 2, 3, 4, 5])
    bonus = assign_bonus_within_fixture(bps, pids)
    assert bonus.tolist() == [3, 3, 1, 0, 0]


def test_assign_bonus_with_second_tie() -> None:
    bps = np.array([100, 80, 80, 50], dtype=np.float64)
    bonus = assign_bonus_within_fixture(bps, np.arange(4))
    assert bonus.tolist() == [3, 2, 2, 0]


def test_assign_bonus_with_third_tie() -> None:
    bps = np.array([100, 80, 60, 60], dtype=np.float64)
    bonus = assign_bonus_within_fixture(bps, np.arange(4))
    assert bonus.tolist() == [3, 2, 1, 1]


def test_legacy_bonus_assigns_each_unique_score() -> None:
    bps = np.array([100, 100, 70, 50], dtype=np.float64)
    bonus = assign_bonus_within_fixture(
        bps, np.arange(4), rules_mode="legacy"
    )
    assert bonus.tolist() == [3, 3, 2, 1]


def test_assign_bonus_empty() -> None:
    bonus = assign_bonus_within_fixture(np.array([]), np.array([]))
    assert len(bonus) == 0


# ── Minutes-to-bucket mapping ────────────────────────────────────────────────


def test_minutes_to_bucket_boundaries() -> None:
    assert minutes_to_bucket(0) == 0
    assert minutes_to_bucket(1) == 30
    assert minutes_to_bucket(59) == 30
    assert minutes_to_bucket(60) == 70
    assert minutes_to_bucket(89) == 70
    assert minutes_to_bucket(90) == 90
    assert minutes_to_bucket(95) == 90  # stoppage time


# ── Per-position alpha ───────────────────────────────────────────────────────


def test_split_p_full_by_position_uses_train_data() -> None:
    """Alphas are P(60-89 | 60+); should be a fraction in [0,1]."""
    df = pl.DataFrame({
        "position_code": ["MID"] * 10,
        "minutes": [60, 70, 75, 90, 90, 90, 65, 80, 90, 75],
    })
    alphas = split_p_full_by_position(df)
    # Of 10 MID rows with minutes >= 60: minutes 60,70,75,65,80,75 = 6 are < 90
    assert abs(alphas["MID"] - 0.6) < 1e-9


def test_split_p_full_handles_empty_position() -> None:
    df = pl.DataFrame({"position_code": ["MID"] * 5, "minutes": [60, 70, 90, 75, 65]})
    alphas = split_p_full_by_position(df)
    assert "GKP" in alphas
    assert alphas["GKP"] == 0.4  # fallback


def test_player_sub_rates_use_only_earlier_full_appearances() -> None:
    """P(60-89 | 60+) from the player's own earlier 60+ games, shrunk to position."""
    import datetime as dt

    day = lambda d: dt.datetime(2026, 9, d, 15, tzinfo=dt.UTC)  # noqa: E731
    history = pl.DataFrame({
        "player_id": [1, 1, 1, 1, 2],
        "kickoff_utc": [day(1), day(8), day(15), day(22), day(1)],
        "minutes": [90, 90, 45, 70, 90],  # The 45 is not a 60+ game and is ignored.
    })
    targets = pl.DataFrame({
        "player_id": [1, 1, 2, 3],
        "fixture_id": [10, 11, 12, 13],
        "kickoff_utc": [day(15), day(29), day(29), day(29)],
        "position_code": ["MID", "MID", "DEF", "FWD"],
    })
    alphas = {"MID": 0.4, "DEF": 0.1, "FWD": 0.5}
    out = player_sub_rates(history, targets, alphas, window=20, prior_weight=2)
    rate = dict(zip(out["fixture_id"], out["alpha_60_89"], strict=True))
    # Fixture 10 sees GW 1 and 8 (two full 90s; the day-15 game is its own kickoff).
    assert abs(rate[10] - (0 + 2 * 0.4) / (2 + 2)) < 1e-9
    # Fixture 11 sees 90, 90, 70: one early substitution in three.
    assert abs(rate[11] - (1 + 2 * 0.4) / (3 + 2)) < 1e-9
    assert abs(rate[12] - (0 + 2 * 0.1) / (1 + 2)) < 1e-9
    assert abs(rate[13] - 0.5) < 1e-9  # No history: the position rate.
    assert out.height == targets.height


def test_player_sub_rates_window_keeps_recent_games() -> None:
    import datetime as dt

    kicks = [dt.datetime(2026, 1, 1, tzinfo=dt.UTC) + dt.timedelta(days=7 * i) for i in range(6)]
    history = pl.DataFrame({"player_id": [1] * 6, "kickoff_utc": kicks, "minutes": [70, 70, 70, 90, 90, 90]})
    targets = pl.DataFrame({
        "player_id": [1], "fixture_id": [1],
        "kickoff_utc": [kicks[-1] + dt.timedelta(days=7)], "position_code": ["MID"],
    })
    out = player_sub_rates(history, targets, {"MID": 0.5}, window=3, prior_weight=1)
    assert abs(out["alpha_60_89"][0] - (0 + 0.5) / (3 + 1)) < 1e-9


# ── EmpiricalResidualEventSource ─────────────────────────────────────────────


def test_residual_source_requires_fit() -> None:
    src = EmpiricalResidualEventSource()
    rng = np.random.default_rng(0)
    try:
        src.simulate_unmodeled_bps("MID", 60, rng)
        raise AssertionError("expected RuntimeError when sampling before fit")
    except RuntimeError:
        pass


def test_residual_source_fits_per_bucket_stats() -> None:
    train_df = pl.DataFrame({
        "position_code": ["MID"] * 50 + ["FWD"] * 50,
        "minutes": [70] * 50 + [90] * 50,
        "residual_bps": [5.0] * 50 + [10.0] * 50,
    })
    src = EmpiricalResidualEventSource()
    src.fit(train_df)
    assert src.n_buckets_fitted >= 2

    rng = np.random.default_rng(0)
    # MID at 70 min should sample near 5
    samples = [src.simulate_unmodeled_bps("MID", 70, rng) for _ in range(100)]
    assert abs(np.mean(samples) - 5.0) < 1.0
    # FWD at 90 min should sample near 10
    samples = [src.simulate_unmodeled_bps("FWD", 90, rng) for _ in range(100)]
    assert abs(np.mean(samples) - 10.0) < 1.0


def test_legacy_residual_source_preserves_continuous_samples() -> None:
    train_df = pl.DataFrame({
        "position_code": ["MID"] * 50,
        "minutes": [90] * 50,
        "residual_bps": list(range(50)),
    })
    src = EmpiricalResidualEventSource(integer_bps=False)
    src.fit(train_df)
    sample = src.simulate_unmodeled_bps("MID", 90, np.random.default_rng(0))
    assert not sample.is_integer()


def test_residual_source_zero_minutes_returns_zero() -> None:
    train_df = pl.DataFrame({
        "position_code": ["MID"] * 50,
        "minutes": [90] * 50,
        "residual_bps": [10.0] * 50,
    })
    src = EmpiricalResidualEventSource()
    src.fit(train_df)
    rng = np.random.default_rng(0)
    # Bench player gets 0 BPS regardless of fitted distribution
    assert src.simulate_unmodeled_bps("MID", 0, rng) == 0.0


def test_residual_source_samples_integer_bps() -> None:
    train_df = pl.DataFrame({
        "position_code": ["MID"] * 50,
        "minutes": [90] * 50,
        "residual_bps": [float(i % 7) for i in range(50)],
    })
    src = EmpiricalResidualEventSource()
    src.fit(train_df)
    rng = np.random.default_rng(0)
    samples = [src.simulate_unmodeled_bps("MID", 90, rng) for _ in range(100)]
    assert all(sample.is_integer() for sample in samples)


def test_saves_model_recovers_opponent_effect() -> None:
    from fpl_bot.models.bps import fit_saves_model

    rng = np.random.default_rng(0)
    rows, market = [], []
    for f in range(400):
        lam = float(rng.uniform(0.6, 2.8))
        market.append({"fixture_id": f, "team_id": 2, "lambda_market_xg": lam})
        rows.append({"player_id": 1, "fixture_id": f, "minutes": 90, "was_home": True,
                     "home_team_id": 1, "away_team_id": 2,
                     "saves": int(rng.poisson(np.exp(0.9 + 0.65 * np.log(lam))))})
    a, b = fit_saves_model(pl.DataFrame(rows), pl.DataFrame({"player_id": [1], "position_code": ["GKP"]}),
                           pl.DataFrame(market))
    assert abs(b - 0.65) < 0.15 and abs(a - 0.9) < 0.15


def test_opponent_saves_rates_touch_only_goalkeepers() -> None:
    from fpl_bot.models.bps import opponent_saves_rates

    players = pl.DataFrame({"fixture_id": [1, 1, 1], "is_home": [True, False, True],
                            "position_code": ["GKP", "GKP", "DEF"], "saves_rate_per_90": [3.0, 3.0, 0.0]})
    fixtures = pl.DataFrame({"fixture_id": [1], "home_team_lambda": [2.0], "away_team_lambda": [1.0]})
    out = opponent_saves_rates(players, fixtures, (np.log(3.0), 0.5))
    home_gk, away_gk, defender = out["saves_rate_per_90"].to_list()
    assert abs(home_gk - 3.0) < 1e-9          # Opponent lambda 1.0 → baseline.
    assert abs(away_gk - 3.0 * 2 ** 0.5) < 1e-9  # Facing lambda 2.0 → more saves.
    assert defender == 0.0


def test_player_card_rates_use_only_earlier_matches() -> None:
    import datetime as dt

    from fpl_bot.models.bps import CARD_PRIOR_NINETIES, player_card_rates

    day = lambda d: dt.datetime(2026, 9, d, 15, tzinfo=dt.UTC)  # noqa: E731
    history = pl.DataFrame({
        "player_id": [1, 1, 1, 2, 2],
        "kickoff_utc": [day(1), day(8), day(15), day(1), day(8)],
        "minutes": [90, 90, 90, 90, 90],
        "yellow_cards": [1, 1, 0, 0, 0],
    })
    positions = pl.DataFrame({"player_id": [1, 2], "position_code": ["MID", "MID"]})
    targets = pl.DataFrame({"player_id": [1, 1], "fixture_id": [10, 11],
                            "kickoff_utc": [day(15), day(22)], "position_code": ["MID", "MID"]})
    out = player_card_rates(history, targets, positions)
    pos_rate = 2 / 5
    m = CARD_PRIOR_NINETIES
    rates = out["yc_rate_per_90"].to_list()
    assert abs(rates[0] - (2 + m * pos_rate) / (2 + m)) < 1e-9  # Day-15 match excluded.
    assert abs(rates[1] - (2 + m * pos_rate) / (3 + m)) < 1e-9
