from unittest.mock import patch

import pytest

from fpl_bot.optim.prediction_postprocess import apply_prediction_postprocessing


def test_live_defcon_is_conditional_on_playing():
    predictions = {(1, 7): 0.0, (2, 7): 1.0, (3, 7): 2.0}
    with patch(
        "fpl_bot.eval.defcon_adjustment.compute_defcon_adjustments",
        return_value=dict.fromkeys(predictions, 0.8),
    ):
        out = apply_prediction_postprocessing(
            predictions, season_id=26, train_seasons=[26], apply_suspension=False,
            appearance_exposure_by_pgw={(1, 7): 0.0, (2, 7): 0.5, (3, 7): 1.0},
        )
    assert out[(1, 7)] == 0.0
    assert out[(2, 7)] == pytest.approx(1.4)
    assert out[(3, 7)] == pytest.approx(2.8)
