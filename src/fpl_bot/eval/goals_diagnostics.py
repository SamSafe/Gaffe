"""Phase 2.2 diagnostics — ablation + calibration plots + feature importance.

Acceptance-gate close-out per docs/design/phase2_2_goals_assists.md.
Outputs land under `docs/design/phase2_2_results/`.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

from fpl_bot.eval.goals_eval import WALK_FORWARD_FOLDS, _drop_unusable
from fpl_bot.features.goals import FEATURE_COLUMNS, build_feature_table
from fpl_bot.models.goals import (
    LABEL_COLUMNS,
    poisson_deviance,
    train_per_90_model,
)

RESULTS_DIR = Path("docs/design/phase2_2_results")

FEATURE_GROUPS: dict[str, list[str]] = {
    "xg_rolling": ["xg_per_90_last_3", "xg_per_90_last_5", "xg_per_90_last_10"],
    "npxg_rolling": ["npxg_per_90_last_3", "npxg_per_90_last_5", "npxg_per_90_last_10"],
    "xa_rolling": ["xa_per_90_last_3", "xa_per_90_last_5", "xa_per_90_last_10"],
    "shots_rolling": ["shots_per_90_last_3", "shots_per_90_last_5", "shots_per_90_last_10"],
    "key_passes_rolling": [
        "key_passes_per_90_last_3",
        "key_passes_per_90_last_5",
        "key_passes_per_90_last_10",
    ],
    "chain_buildup": ["xg_chain_per_90_last_5", "xg_buildup_per_90_last_5"],
    "market_xg": ["team_lambda_market_xg", "opponent_lambda_market_xg"],
    "venue": ["was_home"],
    "penalty_taker": ["is_penalty_taker"],
    "role_mismatch": ["role_mismatch"],
    "timing": ["days_into_season", "gameweek"],
    "position": ["pos_GKP", "pos_DEF", "pos_MID", "pos_FWD"],
}


@dataclass
class AblationRow:
    target: str
    group: str
    full_deviance: float
    ablated_deviance: float
    delta: float


def _train_with_subset(
    train: pl.DataFrame, valid: pl.DataFrame, target: str, features: list[str]
) -> tuple[float, np.ndarray]:
    """Train a Poisson model on a feature subset; return (deviance, predicted rates)."""
    import lightgbm as lgb

    label_col = LABEL_COLUMNS[target]
    needed = [*features, label_col, "minutes_factor"]
    train_df = train.select(needed).drop_nulls(label_col).drop_nulls("minutes_factor")
    valid_df = valid.select(needed).drop_nulls(label_col).drop_nulls("minutes_factor")

    X_t = train_df.select(features).to_pandas().to_numpy()
    y_t = train_df.select(label_col).to_pandas().to_numpy().ravel()
    m_t = train_df.select("minutes_factor").to_pandas().to_numpy().ravel()
    init_t = np.log(np.clip(m_t, 1e-6, None))

    X_v = valid_df.select(features).to_pandas().to_numpy()
    y_v = valid_df.select(label_col).to_pandas().to_numpy().ravel()
    m_v = valid_df.select("minutes_factor").to_pandas().to_numpy().ravel()
    init_v = np.log(np.clip(m_v, 1e-6, None))

    train_set = lgb.Dataset(
        X_t, label=y_t, feature_name=features, free_raw_data=False, init_score=init_t, weight=m_t
    )
    valid_set = lgb.Dataset(
        X_v,
        label=y_v,
        feature_name=features,
        reference=train_set,
        free_raw_data=False,
        init_score=init_v,
        weight=m_v,
    )
    params = {
        "objective": "poisson",
        "metric": "poisson",
        "num_leaves": 31,
        "learning_rate": 0.05,
        "min_data_in_leaf": 50,
        "verbosity": -1,
        "seed": 42,
    }
    booster = lgb.train(
        params,
        train_set,
        num_boost_round=1000,
        valid_sets=[train_set, valid_set],
        valid_names=["train", "valid"],
        callbacks=[lgb.log_evaluation(period=0), lgb.early_stopping(50, verbose=False)],
    )
    score = booster.predict(X_v, num_iteration=booster.best_iteration, raw_score=True)
    rate = np.exp(score)
    return poisson_deviance(y_v, rate, m_v), rate


def run_ablation(
    test_season: int = 24,
    targets: tuple[str, ...] = ("goals", "assists"),
) -> list[AblationRow]:
    """Single-fold ablation for goals + assists on the most data-rich fold."""
    fold = next(f for f in WALK_FORWARD_FOLDS if f["test"] == test_season)
    all_seasons = sorted(set(fold["train"] + [fold["test"]]))
    feature_df = build_feature_table(season_ids=all_seasons)

    rows: list[AblationRow] = []
    for target in targets:
        df = _drop_unusable(feature_df, target)
        train = df.filter(pl.col("season_id").is_in(fold["train"]))
        valid = df.filter(pl.col("season_id") == fold["test"])

        full_dev, _ = _train_with_subset(train, valid, target, FEATURE_COLUMNS)
        for group_name, group_feats in FEATURE_GROUPS.items():
            ablated_features = [f for f in FEATURE_COLUMNS if f not in group_feats]
            ablated_dev, _ = _train_with_subset(train, valid, target, ablated_features)
            rows.append(
                AblationRow(
                    target=target,
                    group=group_name,
                    full_deviance=full_dev,
                    ablated_deviance=ablated_dev,
                    delta=ablated_dev - full_dev,
                )
            )
    return rows


def format_ablation(rows: list[AblationRow]) -> str:
    if not rows:
        return "(no ablation rows)"
    lines = [
        "target   | feature group       | full dev | ablated  | delta (higher = group more important)",
        "---------+---------------------+----------+----------+----------------------------------------",
    ]
    for target in sorted({r.target for r in rows}):
        for r in sorted([r for r in rows if r.target == target], key=lambda x: -x.delta):
            lines.append(
                f"{r.target:<8} | {r.group:<19} | {r.full_deviance:.4f}   | "
                f"{r.ablated_deviance:.4f}   | {r.delta:+.4f}"
            )
        lines.append("---------+---------------------+----------+----------+----------------------------------------")
    return "\n".join(lines)


def feature_importance_table(test_season: int = 24, target: str = "goals") -> str:
    """LightGBM gain-based feature importance for a fold's model."""
    fold = next(f for f in WALK_FORWARD_FOLDS if f["test"] == test_season)
    all_seasons = sorted(set(fold["train"] + [fold["test"]]))
    feature_df = _drop_unusable(build_feature_table(season_ids=all_seasons), target)
    train = feature_df.filter(pl.col("season_id").is_in(fold["train"]))
    valid = feature_df.filter(pl.col("season_id") == fold["test"])
    model = train_per_90_model(train, target=target, valid=valid)

    importance = model.booster.feature_importance(importance_type="gain")
    pairs = sorted(
        zip(model.feature_names, importance, strict=False), key=lambda kv: -kv[1]
    )
    total = sum(importance)
    lines = [
        f"target: {target}",
        "feature                       | gain     | share",
        "------------------------------+----------+------",
    ]
    for name, gain in pairs:
        share = gain / total if total > 0 else 0
        lines.append(f"{name:<29} | {gain:>8.0f} | {share:.1%}")
    return "\n".join(lines)


def plot_calibration_for_fold(
    test_season: int,
    target: str,
    out_path: Path,
    n_bins: int = 10,
) -> Path:
    """Calibration plot: per-decile mean predicted rate vs mean empirical rate."""
    import matplotlib.pyplot as plt

    fold = next(f for f in WALK_FORWARD_FOLDS if f["test"] == test_season)
    all_seasons = sorted(set(fold["train"] + [fold["test"]]))
    feature_df = _drop_unusable(build_feature_table(season_ids=all_seasons), target)
    train = feature_df.filter(pl.col("season_id").is_in(fold["train"]))
    valid = feature_df.filter(pl.col("season_id") == fold["test"])

    model = train_per_90_model(train, target=target, valid=valid)
    pred_rate = model.predict_per_90(valid)
    label_col = LABEL_COLUMNS[target]
    y = valid.select(label_col).to_pandas().to_numpy().ravel().astype(np.float64)
    minutes_factor = valid.select("minutes_factor").to_pandas().to_numpy().ravel()

    expected = pred_rate * minutes_factor

    # Bin by predicted expected (per-fixture); within bin, compare mean predicted to mean actual
    order = np.argsort(expected)
    e_sorted = expected[order]
    y_sorted = y[order]
    n = len(expected)
    bin_size = max(1, n // n_bins)

    bin_pred = []
    bin_actual = []
    bin_n = []
    for b in range(n_bins):
        start = b * bin_size
        end = (b + 1) * bin_size if b < n_bins - 1 else n
        if start >= end:
            continue
        bin_pred.append(float(e_sorted[start:end].mean()))
        bin_actual.append(float(y_sorted[start:end].mean()))
        bin_n.append(end - start)

    fig, ax = plt.subplots(figsize=(7, 6))
    max_x = max(max(bin_pred), max(bin_actual)) * 1.05 if bin_pred else 1.0
    ax.plot([0, max_x], [0, max_x], "k--", linewidth=0.8, label="perfect calibration")
    sizes = [40 + 200 * c / max(bin_n) for c in bin_n]
    ax.scatter(bin_pred, bin_actual, s=sizes, alpha=0.7, color="tab:blue")
    ax.plot(bin_pred, bin_actual, linewidth=1.2, color="tab:blue")
    ax.set_xlabel("mean predicted (per-fixture expected)")
    ax.set_ylabel("mean empirical")
    ax.set_xlim(0, max_x)
    ax.set_ylim(0, max_x)
    ax.set_title(
        f"{target} calibration  test=20{test_season}/{(test_season+1)%100:02d}  "
        f"n={n}  decile bins, dot size ∝ bin count"
    )
    ax.grid(alpha=0.3)
    ax.legend()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path
