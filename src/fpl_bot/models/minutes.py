"""Minutes model — train and predict (Phase 2.1).

LightGBM multi-class (3 buckets: 0, 1-59, 60+). Monotonic constraints on
features where physically defensible. Trains on the feature matrix from
`fpl_bot.features.minutes`.

PIT correctness: this module reads ONLY through `fpl_bot.features.minutes`,
which itself reads only through `fpl_bot.db.pit`. The static-import leakage
gate enforces this.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl

from fpl_bot.features.minutes import FEATURE_COLUMNS

NUM_CLASSES = 3

# Pinned LightGBM thread count for reproducible (deterministic) training —
# see models/goals.py. Fixed literal so reproducibility doesn't depend on
# the core count the OS exposes at train time.
LGBM_NUM_THREADS = 4
LABEL_COLUMN = "minutes_bucket"


def availability_adjusted_minutes_probs(
    p_zero: float,
    p_short: float,
    p_full: float,
    p_available: float,
) -> tuple[float, float, float]:
    """Condition a base minutes distribution on an availability signal.

    ``p_available`` represents the probability that the player is available
    for selection, independently of the base model's ordinary rotation risk.
    If unavailable, the player is in the zero-minutes bucket. If available,
    the normalized base distribution applies.

    This function is intentionally pure so it can be tested and reused by
    future news/status calibrators without importing live or database code.
    """
    values = (p_zero, p_short, p_full, p_available)
    if not all(math.isfinite(v) for v in values):
        raise ValueError("minutes and availability probabilities must be finite")
    if not 0.0 <= p_available <= 1.0:
        raise ValueError("p_available must be in [0, 1]")
    if any(v < 0.0 for v in (p_zero, p_short, p_full)):
        raise ValueError("base minutes probabilities must be non-negative")

    total = p_zero + p_short + p_full
    if total <= 0.0:
        raise ValueError("base minutes probabilities must have positive mass")

    base_zero = p_zero / total
    base_short = p_short / total
    base_full = p_full / total
    adjusted = (
        (1.0 - p_available) + p_available * base_zero,
        p_available * base_short,
        p_available * base_full,
    )
    # Normalize once more to absorb harmless floating-point drift.
    adjusted_total = sum(adjusted)
    return (
        adjusted[0] / adjusted_total,
        adjusted[1] / adjusted_total,
        adjusted[2] / adjusted_total,
    )


def apply_availability_to_minutes_predictions(
    predictions: pl.DataFrame,
    availability_by_pgw: dict[tuple[int, int], float] | None,
) -> pl.DataFrame:
    """Apply availability only to explicitly signalled player-gameweeks.

    ``predictions`` must contain ``player_id``, ``gameweek`` and the three
    minutes-probability columns. With no signals, the original DataFrame is
    returned unchanged so historical/backtest predictions stay neutral.
    """
    if not availability_by_pgw or predictions.is_empty():
        return predictions

    required = {
        "player_id",
        "gameweek",
        "p_minutes_zero",
        "p_minutes_short",
        "p_minutes_full",
    }
    missing = required.difference(predictions.columns)
    if missing:
        raise ValueError(f"minutes predictions missing required columns: {sorted(missing)}")

    signal_rows: list[dict[str, int | float]] = []
    for (player_id, gameweek), p_available in sorted(availability_by_pgw.items()):
        if not math.isfinite(p_available) or not 0.0 <= p_available <= 1.0:
            raise ValueError(
                f"invalid availability for player={player_id}, gw={gameweek}: "
                f"{p_available!r}"
            )
        signal_rows.append(
            {
                "player_id": int(player_id),
                "gameweek": int(gameweek),
                "_news_p_available": float(p_available),
            }
        )

    signals = pl.DataFrame(signal_rows)
    out = predictions.join(signals, on=["player_id", "gameweek"], how="left")
    has_signal = pl.col("_news_p_available").is_not_null()
    base_total = (
        pl.col("p_minutes_zero")
        + pl.col("p_minutes_short")
        + pl.col("p_minutes_full")
    )
    invalid_component = (
        (pl.col("p_minutes_zero") < 0.0)
        | (pl.col("p_minutes_short") < 0.0)
        | (pl.col("p_minutes_full") < 0.0)
        | ~pl.col("p_minutes_zero").is_finite()
        | ~pl.col("p_minutes_short").is_finite()
        | ~pl.col("p_minutes_full").is_finite()
    )
    invalid_base = out.filter(has_signal & ((base_total <= 0.0) | invalid_component))
    if not invalid_base.is_empty():
        raise ValueError(
            "signalled base minutes probabilities must be finite, non-negative, "
            "and have positive mass"
        )

    available = pl.col("_news_p_available")
    out = out.with_columns(
        pl.when(has_signal)
        .then((1.0 - available) + available * pl.col("p_minutes_zero") / base_total)
        .otherwise(pl.col("p_minutes_zero"))
        .alias("p_minutes_zero"),
        pl.when(has_signal)
        .then(available * pl.col("p_minutes_short") / base_total)
        .otherwise(pl.col("p_minutes_short"))
        .alias("p_minutes_short"),
        pl.when(has_signal)
        .then(available * pl.col("p_minutes_full") / base_total)
        .otherwise(pl.col("p_minutes_full"))
        .alias("p_minutes_full"),
    )
    return out.drop("_news_p_available")

# Monotonic constraints: +1 monotonic non-decreasing in P(starter), -1 non-increasing,
# 0 unconstrained. The constraint applies to the model's RAW output, which for
# multiclass is per-class, but LightGBM applies the same vector to all classes;
# we keep monotonicity only on features where a single sign makes sense across
# all three classes (intuitively: "more minutes recently → less likely 0, more
# likely 60+"). LightGBM doesn't natively handle per-class monotonicity, so we
# apply the constraint at the multi-class level — interpreted as monotonicity in
# *expected* minutes. Verified empirically by the calibration plots.
MONOTONIC_FEATURE_SIGNS: dict[str, int] = {
    "min_last_1": 1,
    "min_last_3": 1,
    "min_last_5": 1,
    "min_last_10": 1,
    "start60_rate_3": 1,
    "start60_rate_5": 1,
    "start60_rate_10": 1,
    "bucket_last_1": 1,
    "season_match_count": 1,
}


@dataclass
class TrainedMinutesModel:
    booster: lgb.Booster
    feature_names: list[str]
    num_iterations: int

    def predict_proba(self, X: pl.DataFrame) -> np.ndarray:
        """Returns (N, 3) array of class probabilities."""
        arr = X.select(self.feature_names).to_pandas().to_numpy()
        return self.booster.predict(arr, num_iteration=self.num_iterations)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.booster.save_model(str(path), num_iteration=self.num_iterations)

    @classmethod
    def load(cls, path: Path, feature_names: list[str]) -> TrainedMinutesModel:
        booster = lgb.Booster(model_file=str(path))
        return cls(
            booster=booster,
            feature_names=feature_names,
            num_iterations=booster.best_iteration or booster.num_trees(),
        )


def train_minutes_model(
    train: pl.DataFrame,
    valid: pl.DataFrame | None = None,
    *,
    num_leaves: int = 31,
    learning_rate: float = 0.05,
    min_data_in_leaf: int = 50,
    num_boost_round: int = 1000,
    early_stopping_rounds: int = 50,
    seed: int = 42,
) -> TrainedMinutesModel:
    """Train a 3-class LightGBM minutes model on a feature DataFrame."""
    feature_names = FEATURE_COLUMNS
    monotone_constraints = [MONOTONIC_FEATURE_SIGNS.get(f, 0) for f in feature_names]

    train_df = train.select([*feature_names, LABEL_COLUMN]).drop_nulls(LABEL_COLUMN)
    X_train = train_df.select(feature_names).to_pandas().to_numpy()
    y_train = train_df.select(LABEL_COLUMN).to_pandas().to_numpy().ravel()

    train_set = lgb.Dataset(
        X_train, label=y_train, feature_name=feature_names, free_raw_data=False
    )

    valid_sets = [train_set]
    valid_names = ["train"]
    if valid is not None and not valid.is_empty():
        valid_df = valid.select([*feature_names, LABEL_COLUMN]).drop_nulls(LABEL_COLUMN)
        X_valid = valid_df.select(feature_names).to_pandas().to_numpy()
        y_valid = valid_df.select(LABEL_COLUMN).to_pandas().to_numpy().ravel()
        valid_set = lgb.Dataset(
            X_valid,
            label=y_valid,
            feature_name=feature_names,
            reference=train_set,
            free_raw_data=False,
        )
        valid_sets.append(valid_set)
        valid_names.append("valid")

    params = {
        "objective": "multiclass",
        "num_class": NUM_CLASSES,
        "metric": "multi_logloss",
        "num_leaves": num_leaves,
        "learning_rate": learning_rate,
        "min_data_in_leaf": min_data_in_leaf,
        "monotone_constraints": monotone_constraints,
        "verbosity": -1,
        "seed": seed,
        # Reproducible training: without these, multi-threaded histogram
        # construction is float-nondeterministic, so regenerating the
        # prediction cache shifts backtest points by tens of pts and masks
        # real model changes. `deterministic`+`force_row_wise` keep
        # multi-threading but fix the result — but only for a CONSTANT
        # num_threads, so pin it (else a throttling machine varies it).
        "deterministic": True,
        "force_row_wise": True,
        "num_threads": LGBM_NUM_THREADS,
    }

    callbacks: list = [lgb.log_evaluation(period=0)]  # suppress per-iter logs
    if valid is not None and not valid.is_empty():
        callbacks.append(lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False))

    booster = lgb.train(
        params,
        train_set,
        num_boost_round=num_boost_round,
        valid_sets=valid_sets,
        valid_names=valid_names,
        callbacks=callbacks,
    )

    return TrainedMinutesModel(
        booster=booster,
        feature_names=feature_names,
        num_iterations=booster.best_iteration or booster.num_trees(),
    )
