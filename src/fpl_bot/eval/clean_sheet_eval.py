"""Walk-forward CV for the clean-sheet model (Phase 2.3).

Per Phase 0 §4.4 + Phase 2.3 design: the residual model is shipped only if
it beats the market-only baseline by ≥ 0.005 absolute Brier on ≥ 3 of 4
walk-forward folds. Otherwise the production "model" is just `market_cs_prob`.
This module both trains+evaluates AND emits the gate decision.
"""
from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from fpl_bot.features.clean_sheet import (
    INIT_SCORE_COLUMN,
    LABEL_COLUMN,
    MARKET_PROB_COLUMN,
    build_feature_table,
    feature_names_with_team_id,
    make_team_id_encoder,
)
from fpl_bot.models.clean_sheet import (
    binary_ece,
    binary_log_loss,
    brier_score,
    market_only_baseline,
    rolling_cs5_baseline,
    train_clean_sheet_model,
)

WALK_FORWARD_FOLDS: list[dict] = [
    {"train": [19, 20], "test": 21},
    {"train": [19, 20, 21], "test": 22},
    {"train": [19, 20, 21, 22], "test": 23},
    {"train": [19, 20, 21, 22, 23], "test": 24},
]

GATE_DELTA_THRESHOLD = 0.005  # absolute Brier improvement vs market
GATE_MIN_FOLDS_PASSING = 3  # of 4


@dataclass
class CSFoldResult:
    test_season: int
    n_train: int
    n_test: int
    brier_model: float
    brier_market: float
    brier_rolling5: float
    log_loss_model: float
    log_loss_market: float
    ece_model: float
    ece_market: float
    delta_brier_market_minus_model: float


def _drop_unusable(df: pl.DataFrame) -> pl.DataFrame:
    return df.filter(
        pl.col(LABEL_COLUMN).is_not_null()
        & pl.col(INIT_SCORE_COLUMN).is_not_null()
        & pl.col(MARKET_PROB_COLUMN).is_not_null()
    )


def run_walk_forward_cv(
    folds: list[dict] | None = None,
    *,
    feature_df: pl.DataFrame | None = None,
) -> list[CSFoldResult]:
    folds_to_use = folds or WALK_FORWARD_FOLDS
    if feature_df is None:
        all_seasons = sorted(
            set(s for f in folds_to_use for s in f["train"] + [f["test"]])
        )
        feature_df = build_feature_table(season_ids=all_seasons)
    feature_df = _drop_unusable(feature_df)

    results: list[CSFoldResult] = []
    for fold in folds_to_use:
        train = feature_df.filter(pl.col("season_id").is_in(fold["train"]))
        test = feature_df.filter(pl.col("season_id") == fold["test"])
        if train.is_empty() or test.is_empty():
            continue

        # Fold-internal team_id encoder (CRITICAL: train-only fit)
        seen_teams = set(train["team_id"].unique().to_list())
        encoder = make_team_id_encoder(seen_teams)
        train_enc = encoder.transform(train)
        test_enc = encoder.transform(test)
        feature_names = feature_names_with_team_id(encoder)

        model = train_clean_sheet_model(train_enc, feature_names, valid=test_enc)

        market_test = test_enc[MARKET_PROB_COLUMN].to_numpy()
        y_test = test_enc[LABEL_COLUMN].to_numpy().astype(int)

        p_model = model.predict_cs_prob(test_enc, market_test)
        p_market = market_only_baseline(market_test)
        p_rolling5 = rolling_cs5_baseline(test_enc)

        brier_m = brier_score(y_test, p_model)
        brier_market = brier_score(y_test, p_market)

        results.append(
            CSFoldResult(
                test_season=fold["test"],
                n_train=len(train),
                n_test=len(test),
                brier_model=brier_m,
                brier_market=brier_market,
                brier_rolling5=brier_score(y_test, p_rolling5),
                log_loss_model=binary_log_loss(y_test, p_model),
                log_loss_market=binary_log_loss(y_test, p_market),
                ece_model=binary_ece(y_test, p_model),
                ece_market=binary_ece(y_test, p_market),
                delta_brier_market_minus_model=brier_market - brier_m,
            )
        )
    return results


def gate_decision(results: list[CSFoldResult]) -> dict:
    """Acceptance-gate decision: ship residual or fall back to market-only."""
    deltas = [r.delta_brier_market_minus_model for r in results]
    pass_count = sum(1 for d in deltas if d >= GATE_DELTA_THRESHOLD)
    passed = pass_count >= GATE_MIN_FOLDS_PASSING and len(results) >= GATE_MIN_FOLDS_PASSING
    return {
        "passed": passed,
        "production_model": "residual" if passed else "market_only",
        "fold_deltas": deltas,
        "fold_pass_count": pass_count,
        "threshold": GATE_DELTA_THRESHOLD,
        "min_folds_passing": GATE_MIN_FOLDS_PASSING,
    }


def format_results(results: list[CSFoldResult]) -> str:
    if not results:
        return "(no folds completed)"
    lines = [
        "season  | n_train | n_test | brier_m | brier_mkt | brier_r5 | ll_m   | ll_mkt | ece_m | ece_mkt | delta",
        "--------+---------+--------+---------+-----------+----------+--------+--------+-------+---------+-------",
    ]
    for r in results:
        lines.append(
            f" 20{r.test_season}/{(r.test_season+1)%100:02d} "
            f"| {r.n_train:>7} | {r.n_test:>6} "
            f"| {r.brier_model:.4f}  | {r.brier_market:.4f}    | {r.brier_rolling5:.4f}   "
            f"| {r.log_loss_model:.4f} | {r.log_loss_market:.4f} "
            f"| {r.ece_model:.3f} | {r.ece_market:.3f}   | {r.delta_brier_market_minus_model:+.4f}"
        )
    decision = gate_decision(results)
    lines.append("")
    lines.append(
        f"Gate: residual must beat market by ≥ {decision['threshold']} Brier in ≥ "
        f"{decision['min_folds_passing']}/{len(results)} folds. "
        f"Folds passing: {decision['fold_pass_count']}/{len(results)}. "
        f"PASSED: {decision['passed']}. Production model: {decision['production_model']}."
    )
    return "\n".join(lines)
