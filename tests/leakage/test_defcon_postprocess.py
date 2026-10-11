"""DefCon post-processing must not reveal who played the target gameweek."""
from __future__ import annotations

from unittest.mock import patch

import polars as pl
import pytest

from fpl_bot.optim.prediction_postprocess import apply_prediction_postprocessing

pytestmark = pytest.mark.leakage


def _postprocess(target_gw_minutes: int) -> dict[tuple[int, int], float]:
    # Player 100 triggers DefCon in GW1-5; only his GW6 outcome varies.
    rows = [
        {"player_id": 100, "gameweek": gw, "defensive_contribution": 12, "minutes": 90}
        for gw in range(1, 6)
    ]
    rows.append({
        "player_id": 100, "gameweek": 6,
        "defensive_contribution": 12 if target_gw_minutes else 0,
        "minutes": target_gw_minutes,
    })
    with patch(
        "fpl_bot.db.pit.defensive_contribution_per_player_per_gw",
        return_value=pl.DataFrame(rows),
    ), patch(
        "fpl_bot.db.pit.all_player_positions",
        return_value=pl.DataFrame({"player_id": [100], "position_code": ["DEF"]}),
    ):
        return apply_prediction_postprocessing(
            {(100, 6): 1.0},
            season_id=25,
            train_seasons=[25],
            appearance_exposure_by_pgw={(100, 6): 0.5},
            apply_suspension=False,
        )


def test_defcon_backtest_is_identical_whether_or_not_the_player_appeared():
    played, benched = _postprocess(90), _postprocess(0)
    assert played == benched
    # Rate 5/5 from earlier GWs, x2 points, DEF shrinkage 0.4, half an appearance.
    assert played[(100, 6)] == pytest.approx(1.0 + 2.0 * 0.4 * 0.5)
