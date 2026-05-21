"""Unit tests for scenario-derived optimizer helpers."""

from __future__ import annotations

import polars as pl
import pytest

from fpl_bot.optim.scenarios import captain_haul_score_per_gw


def test_captain_haul_score_adds_mean_excess_above_threshold() -> None:
    pts_df = pl.DataFrame(
        {
            "player_id": [1, 1, 1, 2, 2, 2],
            "gameweek": [5, 5, 5, 5, 5, 5],
            "scenario_id": [0, 1, 2, 0, 1, 2],
            "xpts": [4.0, 8.0, 14.0, 7.0, 7.0, 7.0],
        }
    )

    scores = captain_haul_score_per_gw(pts_df, threshold=6.0, weight=0.5)

    assert scores[(1, 5)] == pytest.approx((4 + 8 + 14) / 3 + 0.5 * ((0 + 2 + 8) / 3))
    assert scores[(2, 5)] == pytest.approx(7.0 + 0.5)


def test_captain_haul_score_validates_non_negative_inputs() -> None:
    pts_df = pl.DataFrame(
        {
            "player_id": [1],
            "gameweek": [1],
            "scenario_id": [0],
            "xpts": [6.0],
        }
    )

    with pytest.raises(ValueError, match="threshold"):
        captain_haul_score_per_gw(pts_df, threshold=-1.0)

    with pytest.raises(ValueError, match="weight"):
        captain_haul_score_per_gw(pts_df, weight=-0.1)
