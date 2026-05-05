"""Phase 2.1 diagnostics — ablation table, calibration plots, feature importance.

Acceptance-gate artifacts (per `docs/design/phase2_1_minutes_model.md` §10).
Outputs land under `docs/design/phase2_1_results/`.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.metrics import log_loss

from fpl_bot.eval.minutes_eval import (
    WALK_FORWARD_FOLDS,
    _drop_unusable,
    _ece,
)
from fpl_bot.features.minutes import FEATURE_COLUMNS, build_feature_table
from fpl_bot.models.minutes import (
    LABEL_COLUMN,
    MONOTONIC_FEATURE_SIGNS,
    NUM_CLASSES,
    train_minutes_model,
)

RESULTS_DIR = Path("docs/design/phase2_1_results")

FEATURE_GROUPS: dict[str, list[str]] = {
    "rolling_minutes": ["min_last_1", "min_last_3", "min_last_5", "min_last_10"],
    "start60_rates": ["start60_rate_3", "start60_rate_5", "start60_rate_10"],
    "bucket_history": ["bucket_last_1"],
    "timing": ["days_since_last_match", "days_into_season", "gameweek"],
    "season_context": ["season_match_count"],
    "position": ["pos_GKP", "pos_DEF", "pos_MID", "pos_FWD"],
}


@dataclass
class AblationRow:
    group: str
    full_log_loss: float
    ablated_log_loss: float
    delta: float


def _train_with_feature_subset(
    train: pl.DataFrame,
    valid: pl.DataFrame,
    features: list[str],
) -> tuple[float, np.ndarray]:
    """Train on a feature subset and return (log_loss, probs) on valid."""
    import lightgbm as lgb

    train_df = train.select([*features, LABEL_COLUMN]).drop_nulls(LABEL_COLUMN)
    X_train = train_df.select(features).to_pandas().to_numpy()
    y_train = train_df.select(LABEL_COLUMN).to_pandas().to_numpy().ravel()
    valid_df = valid.select([*features, LABEL_COLUMN]).drop_nulls(LABEL_COLUMN)
    X_valid = valid_df.select(features).to_pandas().to_numpy()
    y_valid = valid_df.select(LABEL_COLUMN).to_pandas().to_numpy().ravel()

    train_set = lgb.Dataset(X_train, label=y_train, feature_name=features, free_raw_data=False)
    valid_set = lgb.Dataset(
        X_valid, label=y_valid, feature_name=features, reference=train_set, free_raw_data=False
    )
    params = {
        "objective": "multiclass",
        "num_class": NUM_CLASSES,
        "metric": "multi_logloss",
        "num_leaves": 31,
        "learning_rate": 0.05,
        "min_data_in_leaf": 50,
        "monotone_constraints": [MONOTONIC_FEATURE_SIGNS.get(f, 0) for f in features],
        "verbosity": -1,
        "seed": 42,
    }
    booster = lgb.train(
        params,
        train_set,
        num_boost_round=1000,
        valid_sets=[train_set, valid_set],
        valid_names=["train", "valid"],
        callbacks=[
            lgb.log_evaluation(period=0),
            lgb.early_stopping(stopping_rounds=50, verbose=False),
        ],
    )
    probs = booster.predict(X_valid, num_iteration=booster.best_iteration)
    return float(log_loss(y_valid, probs, labels=list(range(NUM_CLASSES)))), probs


def run_ablation(
    test_season: int = 24,
    feature_df: pl.DataFrame | None = None,
) -> list[AblationRow]:
    """Single-fold ablation on the most data-rich fold (default: 2024/25 test)."""
    fold = next(f for f in WALK_FORWARD_FOLDS if f["test"] == test_season)

    if feature_df is None:
        all_seasons = sorted(set(fold["train"] + [fold["test"]]))
        feature_df = build_feature_table(season_ids=all_seasons)
    feature_df = _drop_unusable(feature_df)

    train = feature_df.filter(pl.col("season_id").is_in(fold["train"]))
    valid = feature_df.filter(pl.col("season_id") == fold["test"])

    full_ll, _ = _train_with_feature_subset(train, valid, FEATURE_COLUMNS)

    rows: list[AblationRow] = []
    for group_name, group_feats in FEATURE_GROUPS.items():
        ablated = [f for f in FEATURE_COLUMNS if f not in group_feats]
        ablated_ll, _ = _train_with_feature_subset(train, valid, ablated)
        rows.append(
            AblationRow(
                group=group_name,
                full_log_loss=full_ll,
                ablated_log_loss=ablated_ll,
                delta=ablated_ll - full_ll,
            )
        )
    return rows


def format_ablation(rows: list[AblationRow]) -> str:
    if not rows:
        return "(no ablation rows)"
    rows_sorted = sorted(rows, key=lambda r: -r.delta)
    lines = [
        "feature group     | full ll  | w/o group ll | delta (higher = group more important)",
        "------------------+----------+--------------+--------------------------------------",
    ]
    for r in rows_sorted:
        lines.append(
            f"{r.group:<17} | {r.full_log_loss:.4f}   | {r.ablated_log_loss:.4f}       | {r.delta:+.4f}"
        )
    return "\n".join(lines)


def plot_calibration_for_fold(
    test_season: int,
    out_path: Path,
    n_bins: int = 10,
) -> Path:
    """Per-class reliability diagram for one fold; saves a 3-subplot PNG."""
    import matplotlib.pyplot as plt

    fold = next(f for f in WALK_FORWARD_FOLDS if f["test"] == test_season)
    all_seasons = sorted(set(fold["train"] + [fold["test"]]))
    feature_df = _drop_unusable(build_feature_table(season_ids=all_seasons))
    train = feature_df.filter(pl.col("season_id").is_in(fold["train"]))
    valid = feature_df.filter(pl.col("season_id") == fold["test"])

    model = train_minutes_model(train, valid=valid)
    probs = model.predict_proba(valid)
    y = valid.select(LABEL_COLUMN).to_pandas().to_numpy().ravel()
    eces = _ece(probs, y, n_bins=n_bins)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    class_labels = ["B_0 = 0 min", "B_S = 1-59 min", "B_F = 60+ min"]
    for c, ax in enumerate(axes):
        p = probs[:, c]
        actual = (y == c).astype(np.float32)
        bins = np.linspace(0, 1, n_bins + 1)
        bin_centers, bin_actual, bin_count = [], [], []
        for b in range(n_bins):
            mask = (p >= bins[b]) & (p < bins[b + 1])
            if b == n_bins - 1:
                mask = mask | (p == 1.0)
            if mask.sum() == 0:
                continue
            bin_centers.append(float(p[mask].mean()))
            bin_actual.append(float(actual[mask].mean()))
            bin_count.append(int(mask.sum()))

        ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, label="perfect")
        sizes = [50 + 200 * c / max(bin_count) for c in bin_count]
        ax.scatter(bin_centers, bin_actual, s=sizes, alpha=0.7)
        ax.plot(bin_centers, bin_actual, linewidth=1.2)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("predicted probability")
        ax.set_ylabel("empirical frequency")
        ax.set_title(f"{class_labels[c]}  ECE={eces[c]:.3f}")
        ax.grid(alpha=0.3)
        ax.legend(loc="upper left", fontsize=8)

    fig.suptitle(
        f"Minutes model calibration — test season 20{test_season}/{(test_season+1)%100:02d}"
        f"  (n={len(y)})"
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def feature_importance_table(test_season: int = 24) -> str:
    """LightGBM gain-based feature importance for a fold's model."""
    fold = next(f for f in WALK_FORWARD_FOLDS if f["test"] == test_season)
    all_seasons = sorted(set(fold["train"] + [fold["test"]]))
    feature_df = _drop_unusable(build_feature_table(season_ids=all_seasons))
    train = feature_df.filter(pl.col("season_id").is_in(fold["train"]))
    valid = feature_df.filter(pl.col("season_id") == fold["test"])
    model = train_minutes_model(train, valid=valid)

    importance = model.booster.feature_importance(importance_type="gain")
    pairs = sorted(
        zip(model.feature_names, importance, strict=False), key=lambda kv: -kv[1]
    )
    total = sum(importance)
    lines = [
        "feature                | gain     | share",
        "-----------------------+----------+------",
    ]
    for name, gain in pairs:
        share = gain / total if total > 0 else 0
        lines.append(f"{name:<22} | {gain:>8.0f} | {share:.1%}")
    return "\n".join(lines)
