"""Phase 2.3 diagnostics — ablation, feature importance, calibration plots,
subgroup ECE.

Even though the residual model failed the acceptance gate (production model
is `market_only`), the diagnostics matter:
  - Confirm the market is well-calibrated everywhere (the actual prod model)
  - Document why the residual found no lift (feature importance shows it)
  - Subgroup ECE: per market_cs_prob decile + per top-6 team

Outputs land under `docs/design/phase2_3_results/`.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

from fpl_bot.db import pit
from fpl_bot.eval.clean_sheet_eval import WALK_FORWARD_FOLDS, _drop_unusable
from fpl_bot.features.clean_sheet import (
    LABEL_COLUMN,
    MARKET_PROB_COLUMN,
    build_feature_table,
    feature_names_with_team_id,
    make_team_id_encoder,
)
from fpl_bot.models.clean_sheet import (
    binary_ece,
    brier_score,
    train_clean_sheet_model,
)

RESULTS_DIR = Path("docs/design/phase2_3_results")

# Top-6 EPL clubs by FPL convention; full names match dim_team.full_name
TOP6_TEAMS = ("Arsenal", "Liverpool", "Man City", "Man Utd", "Spurs", "Chelsea")

FEATURE_GROUPS: dict[str, list[str]] = {
    "market_lambdas": ["team_lambda_market_xg", "opponent_lambda_market_xg"],
    "rolling_cs": [
        "team_cs_rate_last_3",
        "team_cs_rate_last_5",
        "team_cs_rate_last_10",
    ],
    "rolling_goals_conceded": [
        "team_goals_conceded_last_3",
        "team_goals_conceded_last_5",
        "team_goals_conceded_last_10",
    ],
    "rolling_goals_for": ["team_goals_for_last_5"],
    "venue": ["was_home"],
    "timing": ["days_since_last_match", "days_into_season", "gameweek"],
}


@dataclass
class AblationRow:
    group: str
    full_brier: float
    ablated_brier: float
    delta: float


def _train_with_subset(
    train: pl.DataFrame,
    valid: pl.DataFrame,
    feature_subset: list[str],
) -> tuple[float, np.ndarray]:
    model = train_clean_sheet_model(train, feature_subset, valid=valid)
    market_test = valid[MARKET_PROB_COLUMN].to_numpy()
    y_test = valid[LABEL_COLUMN].to_numpy().astype(int)
    p = model.predict_cs_prob(valid, market_test)
    return brier_score(y_test, p), p


def run_ablation(test_season: int = 24) -> list[AblationRow]:
    fold = next(f for f in WALK_FORWARD_FOLDS if f["test"] == test_season)
    all_seasons = sorted(set(fold["train"] + [fold["test"]]))
    feature_df = _drop_unusable(build_feature_table(season_ids=all_seasons))
    train = feature_df.filter(pl.col("season_id").is_in(fold["train"]))
    valid = feature_df.filter(pl.col("season_id") == fold["test"])

    seen_teams = set(train["team_id"].unique().to_list())
    encoder = make_team_id_encoder(seen_teams)
    train = encoder.transform(train)
    valid = encoder.transform(valid)
    full_features = feature_names_with_team_id(encoder)

    full_brier, _ = _train_with_subset(train, valid, full_features)

    rows: list[AblationRow] = []
    for group_name, group_feats in FEATURE_GROUPS.items():
        ablated = [f for f in full_features if f not in group_feats]
        ablated_brier, _ = _train_with_subset(train, valid, ablated)
        rows.append(
            AblationRow(
                group=group_name,
                full_brier=full_brier,
                ablated_brier=ablated_brier,
                delta=ablated_brier - full_brier,
            )
        )
    return rows


def format_ablation(rows: list[AblationRow]) -> str:
    if not rows:
        return "(no ablation rows)"
    lines = [
        "feature group           | full brier | ablated   | delta (higher = group more important)",
        "------------------------+------------+-----------+----------------------------------------",
    ]
    for r in sorted(rows, key=lambda x: -x.delta):
        lines.append(
            f"{r.group:<23} | {r.full_brier:.4f}     | {r.ablated_brier:.4f}    | {r.delta:+.4f}"
        )
    return "\n".join(lines)


def feature_importance_table(test_season: int = 24) -> str:
    fold = next(f for f in WALK_FORWARD_FOLDS if f["test"] == test_season)
    all_seasons = sorted(set(fold["train"] + [fold["test"]]))
    feature_df = _drop_unusable(build_feature_table(season_ids=all_seasons))
    train = feature_df.filter(pl.col("season_id").is_in(fold["train"]))
    valid = feature_df.filter(pl.col("season_id") == fold["test"])

    seen_teams = set(train["team_id"].unique().to_list())
    encoder = make_team_id_encoder(seen_teams)
    train = encoder.transform(train)
    valid = encoder.transform(valid)
    feature_names = feature_names_with_team_id(encoder)

    model = train_clean_sheet_model(train, feature_names, valid=valid)
    importance = model.booster.feature_importance(importance_type="gain")
    pairs = sorted(
        zip(model.feature_names, importance, strict=False), key=lambda kv: -kv[1]
    )
    total = sum(importance)
    lines = [
        "feature                       | gain     | share",
        "------------------------------+----------+------",
    ]
    for name, gain in pairs:
        share = gain / total if total > 0 else 0
        lines.append(f"{name:<29} | {gain:>8.0f} | {share:.1%}")
    return "\n".join(lines)


def subgroup_ece_table(test_season: int = 24) -> str:
    """ECE by market_cs_prob decile + per-top-6-team. Reports for both
    market_only (the production model) and residual (for diagnostic record)."""
    fold = next(f for f in WALK_FORWARD_FOLDS if f["test"] == test_season)
    all_seasons = sorted(set(fold["train"] + [fold["test"]]))
    feature_df = _drop_unusable(build_feature_table(season_ids=all_seasons))
    train = feature_df.filter(pl.col("season_id").is_in(fold["train"]))
    valid = feature_df.filter(pl.col("season_id") == fold["test"])

    seen_teams = set(train["team_id"].unique().to_list())
    encoder = make_team_id_encoder(seen_teams)
    train_enc = encoder.transform(train)
    valid_enc = encoder.transform(valid)
    feature_names = feature_names_with_team_id(encoder)

    model = train_clean_sheet_model(train_enc, feature_names, valid=valid_enc)
    market_test = valid_enc[MARKET_PROB_COLUMN].to_numpy()
    y_test = valid_enc[LABEL_COLUMN].to_numpy().astype(int)
    p_market = market_test
    p_model = model.predict_cs_prob(valid_enc, market_test)

    lines = [f"Subgroup ECE — test season 20{test_season}/{(test_season+1)%100:02d}", ""]

    # Decile bins on market_cs_prob
    deciles = np.quantile(p_market, np.linspace(0, 1, 11))
    deciles[-1] = deciles[-1] + 1e-9  # ensure max is included in last bin
    lines.append("market_cs_prob decile | n   | ece_market | ece_model")
    lines.append("----------------------+-----+------------+----------")
    for d in range(10):
        lo, hi = deciles[d], deciles[d + 1]
        mask = (p_market >= lo) & (p_market < hi)
        if mask.sum() == 0:
            continue
        ece_mkt = binary_ece(y_test[mask], p_market[mask], n_bins=5)
        ece_mdl = binary_ece(y_test[mask], p_model[mask], n_bins=5)
        lines.append(
            f"  D{d+1} [{lo:.3f}, {hi:.3f})  | {int(mask.sum()):>3} "
            f"| {ece_mkt:.3f}     | {ece_mdl:.3f}"
        )

    lines.append("")
    lines.append("top-6 team        | n   | ece_market | ece_model")
    lines.append("------------------+-----+------------+----------")
    name_to_team_id = pit.team_id_by_full_name([fold["test"]])
    team_id_arr = valid_enc["team_id"].to_numpy()
    for team_name in TOP6_TEAMS:
        tid = name_to_team_id.get((fold["test"], team_name))
        if tid is None:
            lines.append(f"  {team_name:<15} | (not in season {fold['test']})")
            continue
        mask = team_id_arr == tid
        if mask.sum() == 0:
            continue
        ece_mkt = binary_ece(y_test[mask], p_market[mask], n_bins=5)
        ece_mdl = binary_ece(y_test[mask], p_model[mask], n_bins=5)
        lines.append(
            f"  {team_name:<15} | {int(mask.sum()):>3} "
            f"| {ece_mkt:.3f}     | {ece_mdl:.3f}"
        )

    return "\n".join(lines)


def plot_calibration_for_fold(
    test_season: int,
    out_path: Path,
    n_bins: int = 10,
) -> Path:
    """Side-by-side calibration: market (production) vs residual model."""
    import matplotlib.pyplot as plt

    fold = next(f for f in WALK_FORWARD_FOLDS if f["test"] == test_season)
    all_seasons = sorted(set(fold["train"] + [fold["test"]]))
    feature_df = _drop_unusable(build_feature_table(season_ids=all_seasons))
    train = feature_df.filter(pl.col("season_id").is_in(fold["train"]))
    valid = feature_df.filter(pl.col("season_id") == fold["test"])

    seen_teams = set(train["team_id"].unique().to_list())
    encoder = make_team_id_encoder(seen_teams)
    train_enc = encoder.transform(train)
    valid_enc = encoder.transform(valid)
    feature_names = feature_names_with_team_id(encoder)

    model = train_clean_sheet_model(train_enc, feature_names, valid=valid_enc)
    market_test = valid_enc[MARKET_PROB_COLUMN].to_numpy()
    y_test = valid_enc[LABEL_COLUMN].to_numpy().astype(int)
    p_model = model.predict_cs_prob(valid_enc, market_test)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, p_pred, label in (
        (axes[0], market_test, "market_only (production)"),
        (axes[1], p_model, "residual (failed gate)"),
    ):
        bins = np.linspace(0, 1, n_bins + 1)
        bin_pred, bin_actual, bin_n = [], [], []
        for b in range(n_bins):
            mask = (p_pred >= bins[b]) & (p_pred < bins[b + 1])
            if b == n_bins - 1:
                mask = mask | (p_pred == 1.0)
            if mask.sum() == 0:
                continue
            bin_pred.append(float(p_pred[mask].mean()))
            bin_actual.append(float(y_test[mask].mean()))
            bin_n.append(int(mask.sum()))
        ece = binary_ece(y_test, p_pred, n_bins=n_bins)
        ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, label="perfect")
        sizes = [40 + 200 * c / max(bin_n) for c in bin_n] if bin_n else []
        ax.scatter(bin_pred, bin_actual, s=sizes, alpha=0.7)
        ax.plot(bin_pred, bin_actual, linewidth=1.2)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("predicted CS probability")
        ax.set_ylabel("empirical CS rate")
        ax.set_title(f"{label}\nECE={ece:.3f}")
        ax.grid(alpha=0.3)
        ax.legend(loc="upper left", fontsize=8)

    fig.suptitle(
        f"Clean-sheet calibration — test season 20{test_season}/{(test_season+1)%100:02d} "
        f"(n={len(y_test)})"
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path
