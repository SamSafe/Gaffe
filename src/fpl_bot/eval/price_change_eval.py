"""Walk-forward CV for the Phase 3.5 price-change predictor.

Gates (all must pass for the predictor to ship; if it fails, the cost-basis
sell-tax fix ships alone — still a Phase 3.5 win):

1. Calibration: marginal Brier ≤ 0.04 on rise direction (P(Δ>0) vs actual)
   and on fall direction (P(Δ<0) vs actual) on ≥ 3 of 4 walk-forward folds.

2. Top-k precision: among players flagged P(rise) ≥ 0.5 on a given GW
   boundary, ≥ 60% should actually rise. Symmetrically for falls.

3. Multiclass log-loss: reported only (no hard threshold). The argmax
   accuracy is dominated by the trivial "predict 0" baseline so we don't
   gate on it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from fpl_bot.features.price_change import (
    FEATURE_COLUMNS,
    LABEL_COLUMN,
    build_feature_table,
)
from fpl_bot.models.price_change import (
    CLASS_TO_IDX,
    _label_to_class_idx,
    train_price_change_model,
)


@dataclass
class FoldResult:
    test_season: int
    train_seasons: list[int]
    n_train: int
    n_test: int
    multi_logloss: float
    brier_rise: float       # binary Brier on actual{Δ>0} vs P(Δ>0)
    brier_fall: float       # binary Brier on actual{Δ<0} vs P(Δ<0)
    top_k_precision_rise: float  # of P(rise)≥0.5 flagged, fraction that rose
    top_k_precision_fall: float  # of P(fall)≥0.5 flagged, fraction that fell
    n_flagged_rise: int
    n_flagged_fall: int
    n_actual_rise: int
    n_actual_fall: int


@dataclass
class WalkForwardResult:
    folds: list[FoldResult] = field(default_factory=list)

    @property
    def gate1_calibration_pass(self) -> bool:
        ok = sum(
            1 for f in self.folds if f.brier_rise <= 0.04 and f.brier_fall <= 0.04
        )
        return ok >= 3 and len(self.folds) >= 4

    @property
    def gate2_topk_precision_pass(self) -> bool:
        # Each fold must hit ≥0.6 on both rise and fall flags. Folds with
        # zero flags get a free pass (no signal claimed).
        for f in self.folds:
            if f.n_flagged_rise > 0 and f.top_k_precision_rise < 0.6:
                return False
            if f.n_flagged_fall > 0 and f.top_k_precision_fall < 0.6:
                return False
        return True

    @property
    def all_predictor_gates_pass(self) -> bool:
        return self.gate1_calibration_pass and self.gate2_topk_precision_pass


def _multiclass_logloss(pmf: np.ndarray, y_true_idx: np.ndarray) -> float:
    eps = 1e-12
    p = np.clip(pmf[np.arange(len(y_true_idx)), y_true_idx], eps, 1 - eps)
    return float(-np.mean(np.log(p)))


def _brier(probs: np.ndarray, actual: np.ndarray) -> float:
    return float(np.mean((probs - actual) ** 2))


def run_walk_forward_cv(
    folds: list[dict] | None = None,
    *,
    save_models_dir: Path | None = None,
) -> WalkForwardResult:
    """Default folds: train [19,20]→test 21; [19,20,21]→22; ...; [19..23]→24."""
    if folds is None:
        folds = [
            {"train": [19, 20], "test": 21},
            {"train": [19, 20, 21], "test": 22},
            {"train": [19, 20, 21, 22], "test": 23},
            {"train": [19, 20, 21, 22, 23], "test": 24},
        ]

    out = WalkForwardResult()
    for fold in folds:
        train_df = build_feature_table(fold["train"])
        test_df = build_feature_table([fold["test"]])
        if train_df.is_empty() or test_df.is_empty():
            continue

        model = train_price_change_model(train_df, FEATURE_COLUMNS, valid=test_df)

        if save_models_dir is not None:
            model.save(save_models_dir / f"fold_{fold['test']}.lgb")

        pmf = model.predict_pmf(test_df)  # (N, 5)
        y_true = test_df.select(LABEL_COLUMN).to_pandas().to_numpy().ravel()
        y_idx = _label_to_class_idx(y_true)

        logloss = _multiclass_logloss(pmf, y_idx)

        # Marginal P(rise) = P(Δ=1) + P(Δ=2); marginal P(fall) similarly.
        p_rise = pmf[:, CLASS_TO_IDX[1]] + pmf[:, CLASS_TO_IDX[2]]
        p_fall = pmf[:, CLASS_TO_IDX[-1]] + pmf[:, CLASS_TO_IDX[-2]]
        actual_rise = (y_true > 0).astype(np.float64)
        actual_fall = (y_true < 0).astype(np.float64)

        brier_rise = _brier(p_rise, actual_rise)
        brier_fall = _brier(p_fall, actual_fall)

        # Top-k precision at threshold 0.5
        flagged_rise = p_rise >= 0.5
        flagged_fall = p_fall >= 0.5
        n_flagged_rise = int(flagged_rise.sum())
        n_flagged_fall = int(flagged_fall.sum())
        prec_rise = (
            float(actual_rise[flagged_rise].mean()) if n_flagged_rise > 0 else 1.0
        )
        prec_fall = (
            float(actual_fall[flagged_fall].mean()) if n_flagged_fall > 0 else 1.0
        )

        out.folds.append(
            FoldResult(
                test_season=fold["test"],
                train_seasons=list(fold["train"]),
                n_train=train_df.height,
                n_test=test_df.height,
                multi_logloss=logloss,
                brier_rise=brier_rise,
                brier_fall=brier_fall,
                top_k_precision_rise=prec_rise,
                top_k_precision_fall=prec_fall,
                n_flagged_rise=n_flagged_rise,
                n_flagged_fall=n_flagged_fall,
                n_actual_rise=int(actual_rise.sum()),
                n_actual_fall=int(actual_fall.sum()),
            )
        )

    return out


def format_results(result: WalkForwardResult) -> str:
    lines = ["Phase 3.5 — Price-change predictor walk-forward CV", ""]
    lines.append(
        f"{'season':<10} {'n_test':>7} {'logloss':>8} "
        f"{'brier_rise':>11} {'brier_fall':>11} "
        f"{'prec_rise':>10} {'prec_fall':>10} "
        f"{'flag_rise':>9} {'flag_fall':>9}"
    )
    for f in result.folds:
        lines.append(
            f"20{f.test_season:02d}/{(f.test_season+1)%100:02d}    "
            f"{f.n_test:>7} {f.multi_logloss:>8.4f} "
            f"{f.brier_rise:>11.4f} {f.brier_fall:>11.4f} "
            f"{f.top_k_precision_rise:>10.3f} {f.top_k_precision_fall:>10.3f} "
            f"{f.n_flagged_rise:>9} {f.n_flagged_fall:>9}"
        )

    lines.append("")
    lines.append(
        f"Gate 1 (Brier ≤ 0.04 on rise+fall on ≥3/4 folds): "
        f"{'PASS' if result.gate1_calibration_pass else 'FAIL'}"
    )
    lines.append(
        f"Gate 2 (top-k precision ≥ 0.6 when flags exist): "
        f"{'PASS' if result.gate2_topk_precision_pass else 'FAIL'}"
    )
    lines.append("")
    lines.append(
        f"ALL PREDICTOR GATES PASS: {result.all_predictor_gates_pass}"
    )
    return "\n".join(lines)
