"""Tests for shared prediction post-processing behavior."""
from __future__ import annotations

from unittest.mock import patch

import pytest

from fpl_bot.optim.prediction_postprocess import apply_prediction_postprocessing

# apply_prediction_postprocessing reads DefCon tuning from the dev Postgres.
pytestmark = pytest.mark.integration


def test_defcon_backtest_mode_preserves_legacy_target_gws_none() -> None:
    predictions = {(100, 6): 1.0, (100, 7): 1.0}

    with patch(
        "fpl_bot.eval.defcon_adjustment.compute_defcon_adjustments",
        return_value={(100, 6): 0.4, (100, 7): 0.8},
    ) as compute:
        out = apply_prediction_postprocessing(
            predictions,
            season_id=25,
            train_seasons=[25],
        )

    assert out[(100, 6)] == 1.4
    assert out[(100, 7)] == 1.8
    assert compute.call_args.kwargs["target_gws"] is None


def test_defcon_live_mode_extends_to_prediction_target_gws() -> None:
    predictions = {(100, 6): 1.0, (100, 7): 1.0}

    with patch(
        "fpl_bot.eval.defcon_adjustment.compute_defcon_adjustments",
        return_value={(100, 6): 0.4, (100, 7): 0.8},
    ) as compute:
        out = apply_prediction_postprocessing(
            predictions,
            season_id=25,
            train_seasons=[25],
            extend_defcon_future=True,
        )

    assert out[(100, 6)] == 1.4
    assert out[(100, 7)] == 1.8
    assert compute.call_args.kwargs["target_gws"] == [6, 7]
    assert compute.call_args.kwargs["target_player_ids"] == [100]


def test_defcon_live_mode_is_enabled_for_season_26() -> None:
    predictions = {(100, 1): 1.0}

    with patch(
        "fpl_bot.eval.defcon_adjustment.compute_defcon_adjustments",
        return_value={(100, 1): 0.4},
    ) as compute:
        out = apply_prediction_postprocessing(
            predictions,
            season_id=26,
            train_seasons=[25, 26],
            extend_defcon_future=True,
        )

    assert out[(100, 1)] == 1.4
    compute.assert_called_once()


def test_defcon_skips_untuned_seasons() -> None:
    predictions = {(100, 6): 1.0}

    with patch(
        "fpl_bot.eval.defcon_adjustment.compute_defcon_adjustments",
        return_value={(100, 6): 0.4},
    ) as compute:
        out = apply_prediction_postprocessing(
            predictions,
            season_id=24,
            train_seasons=[24],
            extend_defcon_future=True,
        )

    assert out == predictions
    compute.assert_not_called()
