"""Walk-forward CV + calibration + baseline comparison for the minutes model."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl
from sklearn.metrics import log_loss

from fpl_bot.features.minutes import build_feature_table
from fpl_bot.models.minutes import LABEL_COLUMN, train_minutes_model

WALK_FORWARD_FOLDS: list[dict] = [
    {"train": [19, 20], "test": 21},
    {"train": [19, 20, 21], "test": 22},
    {"train": [19, 20, 21, 22], "test": 23},
    {"train": [19, 20, 21, 22, 23], "test": 24},
]


@dataclass
class FoldResult:
    test_season: int
    log_loss_model: float
    log_loss_baseline_rolling3: float
    log_loss_baseline_position: float
    n_train: int
    n_test: int
    ece_per_class: list[float]


def _drop_unusable(df: pl.DataFrame) -> pl.DataFrame:
    """Drop rows lacking labels or with no history (first match per player)."""
    return df.filter(
        pl.col(LABEL_COLUMN).is_not_null() & pl.col("min_last_1").is_not_null()
    )


def _baseline_rolling3(df: pl.DataFrame) -> np.ndarray:
    """Predict by historical bucket distribution from the last 3 GWs (per row).

    Implementation: weighted vote of bucket_last_1 (which is the prev-row bucket)
    smoothed by start60_rate_3 (which encodes recent 60+ frequency). Returns
    (N, 3) probability matrix that beats position-marginal but not the model.
    """
    n = len(df)
    probs = np.zeros((n, 3), dtype=np.float32)
    bucket_last_1 = df["bucket_last_1"].fill_null(1).to_numpy()
    start60_rate_3 = df["start60_rate_3"].fill_null(0.5).to_numpy()
    for i in range(n):
        # Distribute mass: 0.6 weight on previous bucket, 0.4 on rate-shifted
        probs[i, int(bucket_last_1[i])] += 0.6
        probs[i, 2] += 0.4 * float(start60_rate_3[i])
        probs[i, 0] += 0.4 * (1 - float(start60_rate_3[i])) * 0.4
        probs[i, 1] += 0.4 * (1 - float(start60_rate_3[i])) * 0.6
    probs = probs / probs.sum(axis=1, keepdims=True)
    return probs


def _baseline_position_marginal(train: pl.DataFrame, test: pl.DataFrame) -> np.ndarray:
    """For each test row, predict the bucket distribution from the train set
    among players of the same position. Floor baseline."""
    train_filt = train.filter(pl.col(LABEL_COLUMN).is_not_null())
    pos_marg: dict[str, np.ndarray] = {}
    for pos in ("GKP", "DEF", "MID", "FWD"):
        col = f"pos_{pos}"
        sub = train_filt.filter(pl.col(col) == 1).select(LABEL_COLUMN)
        if sub.is_empty():
            pos_marg[pos] = np.array([1 / 3, 1 / 3, 1 / 3], dtype=np.float32)
            continue
        labels = sub.to_pandas().to_numpy().ravel()
        counts = np.bincount(labels, minlength=3).astype(np.float32)
        pos_marg[pos] = counts / counts.sum()

    overall = np.full(3, 1 / 3, dtype=np.float32)
    n = len(test)
    out = np.zeros((n, 3), dtype=np.float32)
    for i, row in enumerate(test.iter_rows(named=True)):
        if row["pos_GKP"]:
            out[i] = pos_marg["GKP"]
        elif row["pos_DEF"]:
            out[i] = pos_marg["DEF"]
        elif row["pos_MID"]:
            out[i] = pos_marg["MID"]
        elif row["pos_FWD"]:
            out[i] = pos_marg["FWD"]
        else:
            out[i] = overall
    return out


def _ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> list[float]:
    """Per-class expected calibration error (binned)."""
    out = []
    for c in range(probs.shape[1]):
        p = probs[:, c]
        y = (labels == c).astype(np.float32)
        bins = np.linspace(0, 1, n_bins + 1)
        ece_c = 0.0
        n = len(p)
        for b in range(n_bins):
            mask = (p >= bins[b]) & (p < bins[b + 1])
            if b == n_bins - 1:
                mask = mask | (p == 1.0)
            if mask.sum() == 0:
                continue
            avg_pred = float(p[mask].mean())
            avg_actual = float(y[mask].mean())
            ece_c += (mask.sum() / n) * abs(avg_pred - avg_actual)
        out.append(ece_c)
    return out


def run_walk_forward_cv(
    folds: list[dict] | None = None,
    *,
    feature_df: pl.DataFrame | None = None,
) -> list[FoldResult]:
    """Train and evaluate one model per walk-forward fold; return per-fold metrics."""
    folds_to_use = folds or WALK_FORWARD_FOLDS
    if feature_df is None:
        all_seasons = sorted(
            set(s for f in folds_to_use for s in f["train"] + [f["test"]])
        )
        feature_df = build_feature_table(season_ids=all_seasons)
    feature_df = _drop_unusable(feature_df)

    results: list[FoldResult] = []
    for fold in folds_to_use:
        train = feature_df.filter(pl.col("season_id").is_in(fold["train"]))
        test = feature_df.filter(pl.col("season_id") == fold["test"])
        if train.is_empty() or test.is_empty():
            continue

        model = train_minutes_model(train, valid=test)
        probs_model = model.predict_proba(test)
        y_test = test.select(LABEL_COLUMN).to_pandas().to_numpy().ravel()

        ll_model = log_loss(y_test, probs_model, labels=[0, 1, 2])
        probs_b1 = _baseline_rolling3(test)
        ll_b1 = log_loss(y_test, probs_b1, labels=[0, 1, 2])
        probs_b2 = _baseline_position_marginal(train, test)
        ll_b2 = log_loss(y_test, probs_b2, labels=[0, 1, 2])

        results.append(
            FoldResult(
                test_season=fold["test"],
                log_loss_model=float(ll_model),
                log_loss_baseline_rolling3=float(ll_b1),
                log_loss_baseline_position=float(ll_b2),
                n_train=len(train),
                n_test=len(test),
                ece_per_class=_ece(probs_model, y_test),
            )
        )

    return results


def format_results(results: list[FoldResult]) -> str:
    if not results:
        return "(no folds completed)"
    lines = [
        "season  | n_train | n_test  | model    | rolling3 | pos_marg | ece_0  ece_1  ece_2",
        "--------+---------+---------+----------+----------+----------+--------------------",
    ]
    for r in results:
        ece_s = "  ".join(f"{e:.3f}" for e in r.ece_per_class)
        lines.append(
            f" 20{r.test_season}/{(r.test_season+1)%100:02d} "
            f"| {r.n_train:>7} | {r.n_test:>7} "
            f"| {r.log_loss_model:.4f}  | {r.log_loss_baseline_rolling3:.4f}   "
            f"| {r.log_loss_baseline_position:.4f}   | {ece_s}"
        )
    return "\n".join(lines)
