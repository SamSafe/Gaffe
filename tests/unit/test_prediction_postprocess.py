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


def test_early_gw_prior_is_actually_populated_for_the_previous_season():
    """The GW1-3 cold-start prior is the mechanism that keeps a season opener
    sane, and it fails OPEN: `_apply_early_gw_prior` returns silently when the
    lookup is empty, so a broken prior is invisible in the output.

    It was broken exactly this way — the last gameweek was read from
    `dim_fixture` (the schedule, GW38) rather than from ingested results
    (GW29), so it queried five gameweeks with no data and returned {}. The
    2026-08-13 GW1 run therefore had NO prior at all, and expected Haaland to
    score 0.0027 goals.
    """
    from fpl_bot.db import pit

    played = pit.all_player_match_with_kickoff(season_ids=[25])
    if played.is_empty():
        pytest.skip("no season-25 results ingested locally")

    prior = pit.player_actual_pts_last_n_gws(25, n=5)
    assert prior, (
        "cold-start prior is empty for a season that has ingested results — "
        "the GW1-3 blend is silently inert"
    )
    assert max(prior.values()) > 0.0, "prior returned only zero-point players"
