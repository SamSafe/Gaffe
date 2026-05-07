"""Unit tests for the BPS simulator (Phase 2.4)."""
from __future__ import annotations

import numpy as np
import polars as pl

from fpl_bot.db.event_source import (
    EmpiricalResidualEventSource,
    minutes_to_bucket,
)
from fpl_bot.models.bps import (
    BPS_60_MIN,
    BPS_ASSIST,
    BPS_CLEAN_SHEET_GK_DEF,
    BPS_GOAL_BY_POSITION,
    _bps_from_saves,
    assign_bonus_within_fixture,
    score_bps_known_events,
    split_p_full_by_position,
)

# ── BPS rule table correctness ───────────────────────────────────────────────


def test_60_min_appearance_bonus() -> None:
    bps = score_bps_known_events(
        position="MID", minutes=60, goals=0, assists=0,
        team_clean_sheet=False, team_goals_conceded=0,
    )
    assert bps == BPS_60_MIN


def test_under_60_no_appearance_bonus() -> None:
    bps = score_bps_known_events(
        position="MID", minutes=59, goals=0, assists=0,
        team_clean_sheet=False, team_goals_conceded=0,
    )
    assert bps == 0


def test_def_goal_24_bps() -> None:
    bps = score_bps_known_events(
        position="DEF", minutes=90, goals=1, assists=0,
        team_clean_sheet=True, team_goals_conceded=0,
    )
    # +6 (60 min) + 24 (DEF goal) + 12 (GK/DEF CS at 90+) = 42
    assert bps == BPS_60_MIN + BPS_GOAL_BY_POSITION["DEF"] + BPS_CLEAN_SHEET_GK_DEF


def test_mid_goal_18_bps() -> None:
    bps = score_bps_known_events(
        position="MID", minutes=90, goals=1, assists=0,
        team_clean_sheet=False, team_goals_conceded=2,
    )
    assert bps == BPS_60_MIN + BPS_GOAL_BY_POSITION["MID"]


def test_fwd_goal_12_bps() -> None:
    bps = score_bps_known_events(
        position="FWD", minutes=90, goals=1, assists=1,
        team_clean_sheet=False, team_goals_conceded=1,
    )
    # 6 + 12 + 9 = 27
    assert bps == BPS_60_MIN + BPS_GOAL_BY_POSITION["FWD"] + BPS_ASSIST


def test_cs_only_for_gk_def_at_90_plus() -> None:
    """MID does NOT get +12 BPS for CS; GK/DEF get it only if they played 60+."""
    mid = score_bps_known_events(
        position="MID", minutes=90, goals=0, assists=0,
        team_clean_sheet=True, team_goals_conceded=0,
    )
    assert mid == BPS_60_MIN  # no CS bonus

    sub_def = score_bps_known_events(
        position="DEF", minutes=45, goals=0, assists=0,
        team_clean_sheet=True, team_goals_conceded=0,
    )
    assert sub_def == 0  # didn't play 60+


def test_goals_conceded_penalty_for_gk_def_at_90() -> None:
    """GK/DEF lose 1 BPS per goal conceded when playing 90+ min."""
    bps = score_bps_known_events(
        position="DEF", minutes=90, goals=0, assists=0,
        team_clean_sheet=False, team_goals_conceded=3,
    )
    # 6 (60+ min) - 3 (goals conceded) = 3
    assert bps == BPS_60_MIN - 3


def test_saves_2_per_3_for_gk() -> None:
    assert _bps_from_saves(0) == 0
    assert _bps_from_saves(2) == 0  # 2 saves = 0 BPS
    assert _bps_from_saves(3) == 2  # 3 saves = 2 BPS
    assert _bps_from_saves(6) == 4
    assert _bps_from_saves(7) == 4  # rounds down
    assert _bps_from_saves(9) == 6


def test_yellow_red_cards() -> None:
    bps = score_bps_known_events(
        position="MID", minutes=60, goals=0, assists=0,
        team_clean_sheet=False, team_goals_conceded=0,
        yellow_cards=1, red_cards=0,
    )
    assert bps == BPS_60_MIN - 3
    bps_red = score_bps_known_events(
        position="MID", minutes=60, goals=0, assists=0,
        team_clean_sheet=False, team_goals_conceded=0,
        yellow_cards=0, red_cards=1,
    )
    assert bps_red == BPS_60_MIN - 9


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
    # top two tied at 100 → both 3; next unique (70) → 2; next (50) → 1
    assert bonus.tolist() == [3, 3, 2, 1, 0]


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
