"""Choose boosting rounds on past data, then refit on all available history."""
from __future__ import annotations

from collections.abc import Callable
from datetime import timedelta
from typing import Any, Protocol, TypeVar

import polars as pl

TRAINING_POLICY_VERSION = "chronological_v1"


class IterativeModel(Protocol):
    num_iterations: int


ModelT = TypeVar("ModelT", bound=IterativeModel)


def chronological_validation_split(
    history: pl.DataFrame,
    *,
    validation_fraction: float = 0.2,
    result_gap_hours: float = 3.0,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Hold out the latest kickoff groups inside an existing training fold.

    All players in simultaneous fixtures stay together. A three-hour gap
    prevents a preceding match's unfinished result from entering the fit.
    This function never receives the outer evaluation set.
    """
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1")
    if result_gap_hours < 0:
        raise ValueError("result_gap_hours must be non-negative")
    if "kickoff_utc" not in history.columns:
        raise ValueError("Chronological validation requires kickoff_utc")
    if history["kickoff_utc"].null_count():
        raise ValueError("Chronological validation requires non-null kickoffs")
    if not isinstance(history.schema["kickoff_utc"], pl.Datetime):
        raise ValueError("Chronological validation requires datetime kickoffs")

    kickoffs = history["kickoff_utc"].unique().sort().to_list()
    if len(kickoffs) < 2:
        raise ValueError("Chronological validation requires at least two kickoff groups")
    split_index = max(1, min(len(kickoffs) - 1, int(len(kickoffs) * (1 - validation_fraction))))
    cutoff = kickoffs[split_index]
    fit_cutoff = cutoff - timedelta(hours=result_gap_hours)
    fit = history.filter(pl.col("kickoff_utc") < fit_cutoff)
    validation = history.filter(pl.col("kickoff_utc") >= cutoff)
    if fit.is_empty() or validation.is_empty():
        raise ValueError("Not enough match history for chronological validation and result gap")
    return fit, validation


def train_with_chronological_validation(
    trainer: Callable[..., ModelT],
    history: pl.DataFrame,
    **training_options: Any,
) -> ModelT:
    """Select iterations on a train-only holdout; refit including recent rows.

    The trainer must support ``valid`` and ``num_boost_round``. The refit
    has no validation set and uses only the selected number of rounds, so
    neither live outcomes nor the outer test labels can affect stopping.
    """
    if "valid" in training_options:
        raise ValueError("Pass only training history; validation is selected internally")
    fit, validation = chronological_validation_split(history)
    selected = trainer(fit, valid=validation, **training_options)
    options = {**training_options, "num_boost_round": max(1, selected.num_iterations)}
    return trainer(history, valid=None, **options)
