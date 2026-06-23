"""News availability updates the minutes distribution before xPts simulation."""
from __future__ import annotations

import math

import polars as pl
import pytest

from fpl_bot.models.minutes import (
    apply_availability_to_minutes_predictions,
    availability_adjusted_minutes_probs,
)


def test_scalar_zero_availability_forces_zero_minutes() -> None:
    assert availability_adjusted_minutes_probs(0.1, 0.2, 0.7, 0.0) == (1.0, 0.0, 0.0)


def test_scalar_full_availability_preserves_normalized_base() -> None:
    out = availability_adjusted_minutes_probs(1.0, 2.0, 7.0, 1.0)
    assert out == pytest.approx((0.1, 0.2, 0.7))


def test_scalar_partial_availability_moves_mass_to_zero() -> None:
    out = availability_adjusted_minutes_probs(0.1, 0.2, 0.7, 0.4)
    assert out == pytest.approx((0.64, 0.08, 0.28))
    assert sum(out) == pytest.approx(1.0)


@pytest.mark.parametrize("availability", [-0.01, 1.01, math.nan, math.inf])
def test_scalar_rejects_invalid_availability(availability: float) -> None:
    with pytest.raises(ValueError):
        availability_adjusted_minutes_probs(0.1, 0.2, 0.7, availability)


def _prediction_frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "player_id": [10, 20, 30],
            "fixture_id": [100, 100, 100],
            "gameweek": [5, 5, 5],
            "p_minutes_zero": [0.1, 0.2, 0.3],
            "p_minutes_short": [0.2, 0.3, 0.2],
            "p_minutes_full": [0.7, 0.5, 0.5],
        }
    )


def test_no_signals_are_exactly_neutral() -> None:
    base = _prediction_frame()
    out = apply_availability_to_minutes_predictions(base, {})
    assert out is base
    assert out.equals(base)


def test_frame_updates_only_signalled_player_gameweeks() -> None:
    base = _prediction_frame()
    out = apply_availability_to_minutes_predictions(
        base,
        {(10, 5): 0.0, (20, 5): 0.5, (30, 6): 0.0},
    )
    by_player = {row["player_id"]: row for row in out.iter_rows(named=True)}

    assert (
        by_player[10]["p_minutes_zero"],
        by_player[10]["p_minutes_short"],
        by_player[10]["p_minutes_full"],
    ) == pytest.approx((1.0, 0.0, 0.0))
    assert (
        by_player[20]["p_minutes_zero"],
        by_player[20]["p_minutes_short"],
        by_player[20]["p_minutes_full"],
    ) == pytest.approx((0.6, 0.15, 0.25))
    # Signal is for GW6, so the GW5 row remains exactly unchanged.
    assert (
        by_player[30]["p_minutes_zero"],
        by_player[30]["p_minutes_short"],
        by_player[30]["p_minutes_full"],
    ) == (0.3, 0.2, 0.5)

    sums = out.select(
        (
            pl.col("p_minutes_zero")
            + pl.col("p_minutes_short")
            + pl.col("p_minutes_full")
        ).alias("probability_sum")
    )["probability_sum"]
    assert sums.to_list() == pytest.approx([1.0, 1.0, 1.0])


def test_frame_rejects_invalid_signal_before_joining() -> None:
    with pytest.raises(ValueError, match="invalid availability"):
        apply_availability_to_minutes_predictions(_prediction_frame(), {(10, 5): 1.2})


def test_frame_rejects_invalid_base_for_signalled_row() -> None:
    base = _prediction_frame().with_columns(
        pl.when(pl.col("player_id") == 10)
        .then(float("nan"))
        .otherwise(pl.col("p_minutes_full"))
        .alias("p_minutes_full")
    )
    with pytest.raises(ValueError, match="finite, non-negative"):
        apply_availability_to_minutes_predictions(base, {(10, 5): 0.5})
