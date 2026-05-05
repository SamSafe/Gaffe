"""Goals + assists per-90 model — train and predict (Phase 2.2).

LightGBM Poisson regression with `init_score = log(minutes/90)` offset.
`exp(model_output)` is the predicted per-90 rate; multiply by expected
minutes to get expected goals/assists for the fixture.

Same module covers both targets — the only difference is the target column
name and the monotonic-constraint feature signs.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl

from fpl_bot.features.goals import FEATURE_COLUMNS

LABEL_COLUMNS = {"goals": "goals", "assists": "assists"}

# +1 monotonic non-decreasing (more of this feature → higher rate)
GOALS_MONOTONIC: dict[str, int] = {
    "xg_per_90_last_3": 1,
    "xg_per_90_last_5": 1,
    "xg_per_90_last_10": 1,
    "npxg_per_90_last_3": 1,
    "npxg_per_90_last_5": 1,
    "npxg_per_90_last_10": 1,
    "shots_per_90_last_3": 1,
    "shots_per_90_last_5": 1,
    "shots_per_90_last_10": 1,
    "team_lambda_market_xg": 1,
    "opponent_lambda_market_xg": -1,  # tougher opponent → lower goals
    "is_penalty_taker": 1,
}

ASSISTS_MONOTONIC: dict[str, int] = {
    "xa_per_90_last_3": 1,
    "xa_per_90_last_5": 1,
    "xa_per_90_last_10": 1,
    "key_passes_per_90_last_3": 1,
    "key_passes_per_90_last_5": 1,
    "key_passes_per_90_last_10": 1,
    "team_lambda_market_xg": 1,
    "opponent_lambda_market_xg": -1,
}


@dataclass
class TrainedPer90Model:
    booster: lgb.Booster
    feature_names: list[str]
    target: str  # 'goals' or 'assists'
    num_iterations: int

    def predict_per_90(self, X: pl.DataFrame) -> np.ndarray:
        """Returns predicted per-90 rate (= exp(model output))."""
        arr = X.select(self.feature_names).to_pandas().to_numpy()
        # init_score is encoded into model_output; without per-row offset at
        # predict time we get exp(score) which IS the per-90 rate
        score = self.booster.predict(arr, num_iteration=self.num_iterations, raw_score=True)
        return np.exp(score)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.booster.save_model(str(path), num_iteration=self.num_iterations)


def train_per_90_model(
    train: pl.DataFrame,
    target: str,
    valid: pl.DataFrame | None = None,
    *,
    weighted: bool = True,
    num_leaves: int = 31,
    learning_rate: float = 0.05,
    min_data_in_leaf: int = 50,
    num_boost_round: int = 1000,
    early_stopping_rounds: int = 50,
    seed: int = 42,
) -> TrainedPer90Model:
    """Train a Poisson per-90 model. target ∈ {'goals', 'assists'}."""
    if target not in LABEL_COLUMNS:
        raise ValueError(f"target must be 'goals' or 'assists'; got {target!r}")
    label_col = LABEL_COLUMNS[target]
    monotone = GOALS_MONOTONIC if target == "goals" else ASSISTS_MONOTONIC
    monotone_constraints = [monotone.get(f, 0) for f in FEATURE_COLUMNS]

    needed_cols = [*FEATURE_COLUMNS, label_col, "minutes_factor"]
    train_df = train.select(needed_cols).drop_nulls(label_col).drop_nulls("minutes_factor")

    X_train = train_df.select(FEATURE_COLUMNS).to_pandas().to_numpy()
    y_train = train_df.select(label_col).to_pandas().to_numpy().ravel()
    mins_train = train_df.select("minutes_factor").to_pandas().to_numpy().ravel()
    init_train = np.log(np.clip(mins_train, 1e-6, None))
    weight_train = mins_train if weighted else None

    train_set = lgb.Dataset(
        X_train,
        label=y_train,
        feature_name=FEATURE_COLUMNS,
        free_raw_data=False,
        init_score=init_train,
        weight=weight_train,
    )

    valid_sets = [train_set]
    valid_names = ["train"]
    if valid is not None and not valid.is_empty():
        v = valid.select(needed_cols).drop_nulls(label_col).drop_nulls("minutes_factor")
        X_valid = v.select(FEATURE_COLUMNS).to_pandas().to_numpy()
        y_valid = v.select(label_col).to_pandas().to_numpy().ravel()
        mins_valid = v.select("minutes_factor").to_pandas().to_numpy().ravel()
        init_valid = np.log(np.clip(mins_valid, 1e-6, None))
        valid_set = lgb.Dataset(
            X_valid,
            label=y_valid,
            feature_name=FEATURE_COLUMNS,
            reference=train_set,
            free_raw_data=False,
            init_score=init_valid,
            weight=mins_valid if weighted else None,
        )
        valid_sets.append(valid_set)
        valid_names.append("valid")

    params = {
        "objective": "poisson",
        "metric": "poisson",
        "num_leaves": num_leaves,
        "learning_rate": learning_rate,
        "min_data_in_leaf": min_data_in_leaf,
        "monotone_constraints": monotone_constraints,
        "verbosity": -1,
        "seed": seed,
    }

    callbacks: list = [lgb.log_evaluation(period=0)]
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

    return TrainedPer90Model(
        booster=booster,
        feature_names=FEATURE_COLUMNS,
        target=target,
        num_iterations=booster.best_iteration or booster.num_trees(),
    )


def poisson_deviance(
    y_true: np.ndarray, y_pred_rate_per_90: np.ndarray, minutes_factor: np.ndarray
) -> float:
    """Mean Poisson deviance per fixture, with per-row offset.

    y_true: integer counts (goals or assists)
    y_pred_rate_per_90: predicted per-90 rate
    minutes_factor: minutes / 90 for each row (the offset)
    """
    expected = y_pred_rate_per_90 * minutes_factor
    expected = np.clip(expected, 1e-9, None)
    # 2 * (y * log(y/μ) - (y - μ)), with 0 * log(0) = 0
    with np.errstate(divide="ignore", invalid="ignore"):
        term = np.where(
            y_true > 0,
            y_true * np.log(y_true / expected) - (y_true - expected),
            -(y_true - expected),
        )
    return float(2.0 * term.mean())


def baseline_rolling_xg5_per_90(
    df: pl.DataFrame, *, target: str
) -> np.ndarray:
    """Naive baseline: predicted rate = rolling-5-start xG/90 for goals,
    or rolling-5-start xA/90 for assists. Uses last_5 columns directly.

    Returns a (N,) array of per-90 rates.
    """
    col = "xg_per_90_last_5" if target == "goals" else "xa_per_90_last_5"
    return df[col].fill_null(0.0).cast(pl.Float64).to_numpy()
