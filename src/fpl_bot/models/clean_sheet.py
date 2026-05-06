"""Clean-sheet model — train and predict (Phase 2.3).

LightGBM binary classification with `init_score = logit(market_cs_prob)`.
The booster learns the *residual* in log-odds space; final prediction is
`sigmoid(logit_market + raw_score)`. If the residual learns zero, predictions
equal the market.

Per design: ship the residual model only if it beats market-only by ≥ 0.005
absolute Brier on ≥ 3 of 4 walk-forward folds (gate decided in eval module).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl

LABEL_COLUMN = "clean_sheet"
INIT_SCORE_COLUMN = "logit_market_cs_prob"

# Monotonic constraints applied to the static feature subset. Team_id one-hot
# columns (added per-fold by the encoder) are unconstrained.
MONOTONIC_FEATURE_SIGNS: dict[str, int] = {
    "team_lambda_market_xg": -1,  # high own-attack λ → open game → less CS
    "opponent_lambda_market_xg": -1,  # tougher opponent → less CS
    "team_cs_rate_last_3": 1,
    "team_cs_rate_last_5": 1,
    "team_cs_rate_last_10": 1,
    "team_goals_conceded_last_3": -1,
    "team_goals_conceded_last_5": -1,
    "team_goals_conceded_last_10": -1,
}


@dataclass
class TrainedCleanSheetModel:
    booster: lgb.Booster
    feature_names: list[str]
    num_iterations: int

    def predict_cs_prob(
        self, X: pl.DataFrame, market_cs_prob: np.ndarray
    ) -> np.ndarray:
        """Returns blended CS probability: sigmoid(logit(market) + raw_score)."""
        arr = X.select(self.feature_names).to_pandas().to_numpy()
        raw = self.booster.predict(
            arr, num_iteration=self.num_iterations, raw_score=True
        )
        eps = 1e-4
        market = np.clip(market_cs_prob, eps, 1 - eps)
        logit_market = np.log(market / (1 - market))
        return 1.0 / (1.0 + np.exp(-(logit_market + raw)))

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.booster.save_model(str(path), num_iteration=self.num_iterations)


def train_clean_sheet_model(
    train: pl.DataFrame,
    feature_names: list[str],
    valid: pl.DataFrame | None = None,
    *,
    num_leaves: int = 31,
    learning_rate: float = 0.05,
    min_data_in_leaf: int = 50,
    num_boost_round: int = 1000,
    early_stopping_rounds: int = 50,
    seed: int = 42,
) -> TrainedCleanSheetModel:
    """Train the residual binary model with logit(market) as init_score."""
    monotone_constraints = [
        MONOTONIC_FEATURE_SIGNS.get(f, 0) for f in feature_names
    ]

    needed = [*feature_names, LABEL_COLUMN, INIT_SCORE_COLUMN]
    train_df = (
        train.select(needed).drop_nulls(LABEL_COLUMN).drop_nulls(INIT_SCORE_COLUMN)
    )
    X_train = train_df.select(feature_names).to_pandas().to_numpy()
    y_train = (
        train_df.select(LABEL_COLUMN).to_pandas().to_numpy().astype(np.int8).ravel()
    )
    init_train = train_df.select(INIT_SCORE_COLUMN).to_pandas().to_numpy().ravel()

    train_set = lgb.Dataset(
        X_train,
        label=y_train,
        feature_name=feature_names,
        free_raw_data=False,
        init_score=init_train,
    )

    valid_sets = [train_set]
    valid_names = ["train"]
    if valid is not None and not valid.is_empty():
        v = (
            valid.select(needed)
            .drop_nulls(LABEL_COLUMN)
            .drop_nulls(INIT_SCORE_COLUMN)
        )
        X_valid = v.select(feature_names).to_pandas().to_numpy()
        y_valid = (
            v.select(LABEL_COLUMN).to_pandas().to_numpy().astype(np.int8).ravel()
        )
        init_valid = v.select(INIT_SCORE_COLUMN).to_pandas().to_numpy().ravel()
        valid_set = lgb.Dataset(
            X_valid,
            label=y_valid,
            feature_name=feature_names,
            reference=train_set,
            free_raw_data=False,
            init_score=init_valid,
        )
        valid_sets.append(valid_set)
        valid_names.append("valid")

    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "num_leaves": num_leaves,
        "learning_rate": learning_rate,
        "min_data_in_leaf": min_data_in_leaf,
        "monotone_constraints": monotone_constraints,
        "verbosity": -1,
        "seed": seed,
    }

    callbacks: list = [lgb.log_evaluation(period=0)]
    if valid is not None and not valid.is_empty():
        callbacks.append(
            lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False)
        )

    booster = lgb.train(
        params,
        train_set,
        num_boost_round=num_boost_round,
        valid_sets=valid_sets,
        valid_names=valid_names,
        callbacks=callbacks,
    )

    return TrainedCleanSheetModel(
        booster=booster,
        feature_names=feature_names,
        num_iterations=booster.best_iteration or booster.num_trees(),
    )


# ── Metrics ───────────────────────────────────────────────────────────────────


def brier_score(y_true: np.ndarray, p_pred: np.ndarray) -> float:
    return float(np.mean((p_pred - y_true.astype(float)) ** 2))


def binary_log_loss(y_true: np.ndarray, p_pred: np.ndarray) -> float:
    eps = 1e-9
    p = np.clip(p_pred, eps, 1 - eps)
    y = y_true.astype(float)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def binary_ece(y_true: np.ndarray, p_pred: np.ndarray, n_bins: int = 10) -> float:
    """Expected calibration error for binary predictions, equally spaced bins."""
    bins = np.linspace(0, 1, n_bins + 1)
    n = len(p_pred)
    ece = 0.0
    for b in range(n_bins):
        mask = (p_pred >= bins[b]) & (p_pred < bins[b + 1])
        if b == n_bins - 1:
            mask = mask | (p_pred == 1.0)
        if mask.sum() == 0:
            continue
        avg_pred = float(p_pred[mask].mean())
        avg_actual = float(y_true[mask].mean())
        ece += (mask.sum() / n) * abs(avg_pred - avg_actual)
    return float(ece)


def market_only_baseline(market_cs_prob: np.ndarray) -> np.ndarray:
    """The 'do nothing' baseline — emit the market probability directly."""
    return market_cs_prob


def rolling_cs5_baseline(df: pl.DataFrame) -> np.ndarray:
    """Naive baseline: predict the team's last-5-match CS rate."""
    return df["team_cs_rate_last_5"].fill_null(0.3).cast(pl.Float64).to_numpy()
