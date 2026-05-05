"""Walk-forward CV for goals & assists per-90 models (Phase 2.2)."""
from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from fpl_bot.features.goals import build_feature_table
from fpl_bot.models.goals import (
    LABEL_COLUMNS,
    baseline_rolling_xg5_per_90,
    poisson_deviance,
    train_per_90_model,
)

WALK_FORWARD_FOLDS: list[dict] = [
    {"train": [19, 20], "test": 21},
    {"train": [19, 20, 21], "test": 22},
    {"train": [19, 20, 21, 22], "test": 23},
    {"train": [19, 20, 21, 22, 23], "test": 24},
]


@dataclass
class GoalsFoldResult:
    test_season: int
    target: str
    weighted: bool
    deviance_model: float
    deviance_baseline: float
    n_train: int
    n_test: int


def _drop_unusable(df: pl.DataFrame, target: str) -> pl.DataFrame:
    label_col = LABEL_COLUMNS[target]
    return df.filter(
        pl.col(label_col).is_not_null()
        & pl.col("minutes_factor").is_not_null()
        & pl.col("xg_per_90_last_5").is_not_null()  # require some history
    )


def run_walk_forward_cv(
    folds: list[dict] | None = None,
    targets: tuple[str, ...] = ("goals", "assists"),
    weighted_options: tuple[bool, ...] = (True, False),
    *,
    feature_df: pl.DataFrame | None = None,
) -> list[GoalsFoldResult]:
    """Train and evaluate per-90 models for goals and assists across folds,
    in both minutes-weighted and unweighted variants per Phase 2.2 review."""
    folds_to_use = folds or WALK_FORWARD_FOLDS
    if feature_df is None:
        all_seasons = sorted(set(s for f in folds_to_use for s in f["train"] + [f["test"]]))
        feature_df = build_feature_table(season_ids=all_seasons)

    results: list[GoalsFoldResult] = []
    for target in targets:
        target_df = _drop_unusable(feature_df, target)
        for fold in folds_to_use:
            train = target_df.filter(pl.col("season_id").is_in(fold["train"]))
            test = target_df.filter(pl.col("season_id") == fold["test"])
            if train.is_empty() or test.is_empty():
                continue

            y_test = test.select(LABEL_COLUMNS[target]).to_pandas().to_numpy().ravel()
            mins_test = test.select("minutes_factor").to_pandas().to_numpy().ravel()
            baseline_rate = baseline_rolling_xg5_per_90(test, target=target)
            dev_baseline = poisson_deviance(y_test, baseline_rate, mins_test)

            for weighted in weighted_options:
                model = train_per_90_model(train, target=target, valid=test, weighted=weighted)
                pred_rate = model.predict_per_90(test)
                dev_model = poisson_deviance(y_test, pred_rate, mins_test)
                results.append(
                    GoalsFoldResult(
                        test_season=fold["test"],
                        target=target,
                        weighted=weighted,
                        deviance_model=dev_model,
                        deviance_baseline=dev_baseline,
                        n_train=len(train),
                        n_test=len(test),
                    )
                )
    return results


def format_results(results: list[GoalsFoldResult]) -> str:
    if not results:
        return "(no folds completed)"
    lines = [
        "target   | weighted | season  | n_train | n_test  | model    | baseline | rel diff",
        "---------+----------+---------+---------+---------+----------+----------+---------",
    ]
    for r in sorted(results, key=lambda x: (x.target, not x.weighted, x.test_season)):
        rel = (r.deviance_baseline - r.deviance_model) / max(r.deviance_baseline, 1e-9)
        lines.append(
            f"{r.target:<8} | {'yes ' if r.weighted else 'no  '}     "
            f"| 20{r.test_season}/{(r.test_season+1)%100:02d} "
            f"| {r.n_train:>7} | {r.n_test:>7} "
            f"| {r.deviance_model:.4f}   | {r.deviance_baseline:.4f}   "
            f"| {rel:+.1%}"
        )
    return "\n".join(lines)
