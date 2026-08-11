"""Tests for the DefCon xPts adjustment helper."""
from __future__ import annotations

from unittest.mock import patch

import polars as pl

from fpl_bot.eval.defcon_adjustment import (
    DEFCON_THRESHOLD,
    compute_defcon_adjustments,
)


def _fake_dc_df(rows: list[tuple[int, int, int, int]]) -> pl.DataFrame:
    """rows: list of (player_id, gameweek, defensive_contribution, minutes)."""
    return pl.DataFrame(
        [
            {
                "player_id": p,
                "gameweek": g,
                "defensive_contribution": dc,
                "minutes": m,
            }
            for p, g, dc, m in rows
        ]
    )


def _fake_positions(mapping: dict[int, str]) -> pl.DataFrame:
    return pl.DataFrame(
        [{"player_id": pid, "position_code": pos} for pid, pos in mapping.items()]
    )


def test_def_trigger_rate_pit_correct():
    """A DEF who triggers every GW should get rate ≈ 1 after enough samples."""
    # 5 prior GWs all triggered (DC=12 > 10 threshold); GW6 is the target
    rows = [(100, gw, 12, 90) for gw in range(1, 7)]
    with patch(
        "fpl_bot.db.pit.defensive_contribution_per_player_per_gw",
        return_value=_fake_dc_df(rows),
    ), patch(
        "fpl_bot.db.pit.all_player_positions",
        return_value=_fake_positions({100: "DEF"}),
    ):
        adj = compute_defcon_adjustments(test_season=25, target_gws=[6])
    # GW6 prediction should be based on GW1-5 (5 triggers / 5 appearances = 1.0)
    assert adj[(100, 6)] == 2.0


def test_def_no_trigger_returns_zero():
    rows = [(100, gw, 5, 90) for gw in range(1, 7)]  # all under threshold 10
    with patch(
        "fpl_bot.db.pit.defensive_contribution_per_player_per_gw",
        return_value=_fake_dc_df(rows),
    ), patch(
        "fpl_bot.db.pit.all_player_positions",
        return_value=_fake_positions({100: "DEF"}),
    ):
        adj = compute_defcon_adjustments(test_season=25, target_gws=[6])
    assert adj[(100, 6)] == 0.0


def test_mid_threshold_is_12_not_10():
    """A MID with DC=11 (below MID threshold 12) shouldn't be counted."""
    rows = [(200, gw, 11, 90) for gw in range(1, 7)]
    with patch(
        "fpl_bot.db.pit.defensive_contribution_per_player_per_gw",
        return_value=_fake_dc_df(rows),
    ), patch(
        "fpl_bot.db.pit.all_player_positions",
        return_value=_fake_positions({200: "MID"}),
    ):
        adj = compute_defcon_adjustments(test_season=25, target_gws=[6])
    # 0 prior triggers under MID threshold 12 → rate 0
    assert adj[(200, 6)] == 0.0


def test_gkp_excluded():
    rows = [(300, gw, 15, 90) for gw in range(1, 7)]
    with patch(
        "fpl_bot.db.pit.defensive_contribution_per_player_per_gw",
        return_value=_fake_dc_df(rows),
    ), patch(
        "fpl_bot.db.pit.all_player_positions",
        return_value=_fake_positions({300: "GKP"}),
    ):
        adj = compute_defcon_adjustments(test_season=25, target_gws=[6])
    assert (300, 6) not in adj


def test_zero_minutes_dropped():
    """0-min appearances should not feed the rolling rate."""
    rows = [(100, 1, 12, 0), (100, 2, 12, 90), (100, 3, 12, 90)]
    with patch(
        "fpl_bot.db.pit.defensive_contribution_per_player_per_gw",
        return_value=_fake_dc_df(rows),
    ), patch(
        "fpl_bot.db.pit.all_player_positions",
        return_value=_fake_positions({100: "DEF"}),
    ):
        adj = compute_defcon_adjustments(
            test_season=25, target_gws=[3], min_appearances=2
        )
    # GW3 sees GW2 only (GW1 has 0 mins → dropped). 1 trigger / 1 appearance,
    # but min_appearances=2 → fallback to position rate.
    # In our 1-player fixture, position rate = 1.0 → adj should be 2.0
    assert adj[(100, 3)] == 2.0


def test_per_position_shrinkage():
    rows = [(100, gw, 12, 90) for gw in range(1, 7)]
    with patch(
        "fpl_bot.db.pit.defensive_contribution_per_player_per_gw",
        return_value=_fake_dc_df(rows),
    ), patch(
        "fpl_bot.db.pit.all_player_positions",
        return_value=_fake_positions({100: "DEF"}),
    ):
        adj = compute_defcon_adjustments(
            test_season=25,
            target_gws=[6],
            per_position_shrinkage={"DEF": 0.5},
        )
    # Full strength would be 2.0; shrinkage 0.5 → 1.0
    assert adj[(100, 6)] == 1.0


def test_new_season_starts_from_prior_season_evidence():
    """GW1 of season 26 must use season 25, not silently drop DefCon."""
    prior = [(100, gw, 12, 90) for gw in range(1, 5)]

    def _by_season(season_id: int) -> pl.DataFrame:
        return _fake_dc_df(prior) if season_id == 25 else pl.DataFrame()

    with patch(
        "fpl_bot.db.pit.defensive_contribution_per_player_per_gw",
        side_effect=_by_season,
    ), patch(
        "fpl_bot.db.pit.all_player_positions",
        return_value=_fake_positions({100: "DEF", 200: "DEF"}),
    ):
        adj = compute_defcon_adjustments(
            test_season=26,
            target_gws=[1],
            target_player_ids=[100, 200],
        )

    assert adj[(100, 1)] == 2.0
    # A promoted/new player has no personal history, so gets the prior-season
    # position fallback rather than losing the scoring rule entirely.
    assert adj[(200, 1)] == 2.0


def test_position_fallback_does_not_look_ahead_within_target_season():
    current = [(100, 1, 12, 90), (100, 2, 5, 90)]

    def _by_season(season_id: int) -> pl.DataFrame:
        return _fake_dc_df(current) if season_id == 26 else pl.DataFrame()

    with patch(
        "fpl_bot.db.pit.defensive_contribution_per_player_per_gw",
        side_effect=_by_season,
    ), patch(
        "fpl_bot.db.pit.all_player_positions",
        return_value=_fake_positions({100: "DEF", 200: "DEF"}),
    ):
        adj = compute_defcon_adjustments(
            test_season=26,
            target_gws=[1, 2],
            target_player_ids=[200],
        )

    assert adj[(200, 1)] == 0.0
    assert adj[(200, 2)] == 2.0


def test_thresholds():
    """Sanity check the thresholds retained for the 2026/27 rules."""
    assert DEFCON_THRESHOLD["DEF"] == 10
    assert DEFCON_THRESHOLD["MID"] == 12
    assert DEFCON_THRESHOLD["FWD"] == 12
    assert "GKP" not in DEFCON_THRESHOLD
