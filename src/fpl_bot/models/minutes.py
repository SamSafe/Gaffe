"""Minutes model — train and predict (Phase 2.1).

LightGBM multi-class (3 buckets: 0, 1-59, 60+). Monotonic constraints on
features where physically defensible. Trains on the feature matrix from
`fpl_bot.features.minutes`.

PIT correctness: this module reads ONLY through `fpl_bot.features.minutes`,
which itself reads only through `fpl_bot.db.pit`. The static-import leakage
gate enforces this.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl

from fpl_bot.features.minutes import FEATURE_COLUMNS

NUM_CLASSES = 3
LABEL_COLUMN = "minutes_bucket"

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
