"""Price-change model — Phase 3.5 multinomial classifier.

5-class LightGBM (Δ ∈ {-2, -1, 0, +1, +2}) with class_weight handling for
the ~92% Δ=0 majority. Predicts a full PMF; the MILP integrator consumes
E[Δ] = Σ_k k · P(Δ=k) to get an expected next-GW price.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl

LABEL_COLUMN = "price_delta_tenths"
NUM_CLASSES = 5
CLASSES = (-2, -1, 0, 1, 2)
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}


@dataclass
class TrainedPriceChangeModel:
    booster: lgb.Booster
    feature_names: list[str]
    num_iterations: int

    def predict_pmf(self, X: pl.DataFrame) -> np.ndarray:
        """Returns (n_rows, 5) array of per-class probabilities."""
        arr = X.select(self.feature_names).to_pandas().to_numpy()
        return self.booster.predict(arr, num_iteration=self.num_iterations)

    def predict_expected_delta(self, X: pl.DataFrame) -> np.ndarray:
        """Expected Δ in tenths: Σ_k k · P(Δ=k). MILP consumes this directly."""
        pmf = self.predict_pmf(X)
        weights = np.array(CLASSES, dtype=np.float64)
        return pmf @ weights

    def predict_p_rise(self, X: pl.DataFrame) -> np.ndarray:
        """Marginal P(Δ > 0)."""
        pmf = self.predict_pmf(X)
        return pmf[:, CLASS_TO_IDX[1]] + pmf[:, CLASS_TO_IDX[2]]

    def predict_p_fall(self, X: pl.DataFrame) -> np.ndarray:
        """Marginal P(Δ < 0)."""
        pmf = self.predict_pmf(X)
        return pmf[:, CLASS_TO_IDX[-1]] + pmf[:, CLASS_TO_IDX[-2]]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.booster.save_model(str(path), num_iteration=self.num_iterations)


def _label_to_class_idx(arr: np.ndarray) -> np.ndarray:
    """Map ∈ {-2,-1,0,1,2} → {0,1,2,3,4}."""
    out = np.empty_like(arr, dtype=np.int8)
    for c, idx in CLASS_TO_IDX.items():
        out[arr == c] = idx
    return out


def train_price_change_model(
    train: pl.DataFrame,
    feature_names: list[str],
    valid: pl.DataFrame | None = None,
    *,
    num_leaves: int = 63,
    learning_rate: float = 0.05,
    min_data_in_leaf: int = 100,
    num_boost_round: int = 1500,
    early_stopping_rounds: int = 100,
    seed: int = 42,
) -> TrainedPriceChangeModel:
    needed = [*feature_names, LABEL_COLUMN]
    train_df = train.select(needed).drop_nulls(LABEL_COLUMN)
    X_train = train_df.select(feature_names).to_pandas().to_numpy()
    y_train_raw = train_df.select(LABEL_COLUMN).to_pandas().to_numpy().ravel()
    y_train = _label_to_class_idx(y_train_raw)

    # Class-balanced sample weights: w_c = total / (num_classes × n_c)
    counts = np.bincount(y_train, minlength=NUM_CLASSES).astype(np.float64)
    counts[counts == 0] = 1.0  # avoid div-by-zero on rare absent classes
    class_weights = (len(y_train) / (NUM_CLASSES * counts))
    sample_weight = class_weights[y_train]

    train_set = lgb.Dataset(
        X_train,
        label=y_train,
        weight=sample_weight,
        feature_name=feature_names,
        free_raw_data=False,
    )

    valid_sets = [train_set]
    valid_names = ["train"]
    if valid is not None and not valid.is_empty():
        v = valid.select(needed).drop_nulls(LABEL_COLUMN)
        X_valid = v.select(feature_names).to_pandas().to_numpy()
        y_valid_raw = v.select(LABEL_COLUMN).to_pandas().to_numpy().ravel()
        y_valid = _label_to_class_idx(y_valid_raw)
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
        "learning_rate": learning_rate,
        "num_leaves": num_leaves,
        "min_data_in_leaf": min_data_in_leaf,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.9,
        "bagging_freq": 5,
        "seed": seed,
        "verbose": -1,
    }
    callbacks = [lgb.log_evaluation(0)]
    if valid is not None and not valid.is_empty():
        callbacks.append(lgb.early_stopping(early_stopping_rounds, verbose=False))
    booster = lgb.train(
        params,
        train_set,
        num_boost_round=num_boost_round,
        valid_sets=valid_sets,
        valid_names=valid_names,
        callbacks=callbacks,
    )
    return TrainedPriceChangeModel(
        booster=booster,
        feature_names=feature_names,
        num_iterations=booster.best_iteration or num_boost_round,
    )
