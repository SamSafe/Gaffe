"""Phase 6 v3 diagnostic: per-fold prediction-quality comparison.

Loads each fold's cached predictions (the eval_df produced by
`eval/xpts_eval._run_one_fold`) and computes a comparable set of
calibration metrics. The output answers: "is 25/26 structurally
different from earlier folds, or just a hard season?"

Metrics per fold:
- Overall MAE of e_xpts vs total_points
- Mean predicted, mean actual, overall bias
- Brier score for P(xPts ≥ 6)
- Per-position bias (where the mis-calibration concentrates)
- Decile calibration (over-/under-prediction by xPts strata)
- Per-GW bias series (does drift grow / shrink over the season?)
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
from sqlalchemy import select

from fpl_bot.db import pit
from fpl_bot.db.models import DimFixture
from fpl_bot.db.session import session_scope

HAUL_THRESHOLD = 6
POSITIONS = ("GKP", "DEF", "MID", "FWD")

FOLDS = [
    {"train": [19, 20], "test": 21},
    {"train": [19, 20, 21], "test": 22},
    {"train": [19, 20, 21, 22], "test": 23},
    {"train": [19, 20, 21, 22, 23], "test": 24},
    {"train": [19, 20, 21, 22, 23, 24], "test": 25},
]


@dataclass(frozen=True)
class FoldDiagnostics:
    test_season: int
    n_rows: int
    n_gws: int
    mean_pred: float
    mean_actual: float
    bias: float
    mae: float
    brier_ge6: float
    haul_rate: float
    per_position_bias: dict[str, float]
    per_position_mae: dict[str, float]
    decile_bias: list[tuple[float, float]]  # [(mean_pred, mean_actual)] per decile
    per_gw_bias: list[tuple[int, float, int]]  # [(gw, bias, n)]


def _load_fold_eval_df(
    test_season: int,
    train_seasons: list[int],
    cache_dir: Path = Path("data/cache/xpts_predictions"),
) -> pl.DataFrame | None:
    """Load cached eval_df for a fold; returns None if absent."""
    train_str = "_".join(str(s) for s in train_seasons)
    path = cache_dir / f"season_{test_season}_train_{train_str}.parquet"
    if not path.exists():
        return None
    return pl.read_parquet(path)


def _fixture_to_gw(test_season: int) -> dict[int, int]:
    with session_scope() as s:
        rows = s.execute(
            select(DimFixture.fixture_id, DimFixture.gameweek).where(
                DimFixture.season_id == test_season
            )
        ).all()
    return {int(r.fixture_id): int(r.gameweek) for r in rows}


def compute_fold_diagnostics(
    test_season: int,
    train_seasons: list[int],
    cache_dir: Path = Path("data/cache/xpts_predictions"),
) -> FoldDiagnostics | None:
    eval_df = _load_fold_eval_df(test_season, train_seasons, cache_dir)
    if eval_df is None or eval_df.is_empty():
        return None

    # Join gameweek + position
    fid_to_gw = _fixture_to_gw(test_season)
    df = eval_df.with_columns(
        pl.col("fixture_id")
        .replace_strict(fid_to_gw, default=0)
        .alias("gameweek")
    ).filter(pl.col("gameweek") > 0)
    pos_df = pit.all_player_positions()
    df = df.join(pos_df, on="player_id", how="left").with_columns(
        pl.col("position_code").fill_null("MID")
    )

    y = df["total_points"].to_numpy().astype(np.float64)
    pred = df["e_xpts"].to_numpy().astype(np.float64)
    pos_arr = df["position_code"].to_numpy()
    y_ge6 = (y >= HAUL_THRESHOLD).astype(int)
    has_brier = "p_xpts_ge_6" in df.columns
    p_ge6 = df["p_xpts_ge_6"].to_numpy().astype(np.float64) if has_brier else None

    # Overall metrics
    mae = float(np.mean(np.abs(y - pred)))
    bias = float(pred.mean() - y.mean())
    brier = float(np.mean((p_ge6 - y_ge6) ** 2)) if p_ge6 is not None else float("nan")

    # Per-position
    pp_bias: dict[str, float] = {}
    pp_mae: dict[str, float] = {}
    for p in POSITIONS:
        mask = pos_arr == p
        if not mask.any():
            continue
        pp_bias[p] = float(pred[mask].mean() - y[mask].mean())
        pp_mae[p] = float(np.mean(np.abs(y[mask] - pred[mask])))

    # Decile calibration (10 buckets by predicted xPts)
    order = np.argsort(pred)
    e_sorted = pred[order]
    y_sorted = y[order]
    n = len(y_sorted)
    n_bins = 10
    bin_size = max(1, n // n_bins)
    decile_bias: list[tuple[float, float]] = []
    for b in range(n_bins):
        start = b * bin_size
        end = (b + 1) * bin_size if b < n_bins - 1 else n
        if start >= end:
            continue
        decile_bias.append(
            (float(e_sorted[start:end].mean()), float(y_sorted[start:end].mean()))
        )

    # Per-GW bias
    per_gw: list[tuple[int, float, int]] = []
    gw_groups = (
        df.group_by("gameweek")
        .agg(
            pl.col("e_xpts").mean().alias("pred_mean"),
            pl.col("total_points").cast(pl.Float64).mean().alias("actual_mean"),
            pl.col("e_xpts").count().alias("n"),
        )
        .sort("gameweek")
    )
    for r in gw_groups.iter_rows(named=True):
        per_gw.append(
            (int(r["gameweek"]), float(r["pred_mean"] - r["actual_mean"]), int(r["n"]))
        )

    n_gws = df["gameweek"].n_unique()

    return FoldDiagnostics(
        test_season=test_season,
        n_rows=int(n),
        n_gws=int(n_gws),
        mean_pred=float(pred.mean()),
        mean_actual=float(y.mean()),
        bias=bias,
        mae=mae,
        brier_ge6=brier,
        haul_rate=float(y_ge6.mean()),
        per_position_bias=pp_bias,
        per_position_mae=pp_mae,
        decile_bias=decile_bias,
        per_gw_bias=per_gw,
    )


def format_cross_fold_table(folds: list[FoldDiagnostics]) -> str:
    """Side-by-side comparison across folds."""
    if not folds:
        return "(no folds available)"
    cols = [f"fold {f.test_season}" for f in folds]
    lines: list[str] = []
    lines.append("Per-fold prediction quality")
    lines.append("=" * 60)
    lines.append("")

    def row(label: str, vals: list[str]) -> str:
        return f"  {label:<22} | " + " | ".join(f"{v:>9}" for v in vals)

    lines.append(row("", cols))
    lines.append("  " + "-" * 22 + "-+-" + "-+-".join(["-" * 9 for _ in cols]))
    lines.append(row("n_rows", [f"{f.n_rows:,}" for f in folds]))
    lines.append(row("n_gws", [f"{f.n_gws}" for f in folds]))
    lines.append(row("mean predicted", [f"{f.mean_pred:.3f}" for f in folds]))
    lines.append(row("mean actual", [f"{f.mean_actual:.3f}" for f in folds]))
    lines.append(row("bias (pred-actual)", [f"{f.bias:+.3f}" for f in folds]))
    lines.append(row("MAE", [f"{f.mae:.3f}" for f in folds]))
    lines.append(row("Brier(≥6)", [f"{f.brier_ge6:.4f}" for f in folds]))
    lines.append(row("haul rate", [f"{f.haul_rate:.4f}" for f in folds]))
    lines.append("")
    lines.append("Per-position bias (pred − actual):")
    lines.append("")
    lines.append(row("", cols))
    for pos in POSITIONS:
        lines.append(
            row(
                pos,
                [
                    f"{f.per_position_bias.get(pos, float('nan')):+.3f}"
                    for f in folds
                ],
            )
        )
    lines.append("")
    lines.append("Per-position MAE:")
    lines.append("")
    lines.append(row("", cols))
    for pos in POSITIONS:
        lines.append(
            row(
                pos,
                [f"{f.per_position_mae.get(pos, float('nan')):.3f}" for f in folds],
            )
        )
    return "\n".join(lines)


def format_decile_table(fold: FoldDiagnostics) -> str:
    lines = [
        f"Fold {fold.test_season} — decile calibration (pred vs actual, sorted by pred)",
        "",
        " decile | mean_pred | mean_actual | bias",
        " -------+-----------+-------------+--------",
    ]
    for i, (p, a) in enumerate(fold.decile_bias, start=1):
        lines.append(f" D{i:<5} | {p:>9.3f} | {a:>11.3f} | {p-a:+.3f}")
    return "\n".join(lines)


def format_per_gw_bias(fold: FoldDiagnostics, top_n: int = 10) -> str:
    """Show the top-N most-biased GWs for a fold."""
    sorted_gws = sorted(fold.per_gw_bias, key=lambda x: abs(x[1]), reverse=True)
    lines = [
        f"Fold {fold.test_season} — top-{top_n} GWs by |bias|",
        "",
        " GW | bias    | n",
        " ---+---------+--------",
    ]
    for gw, bias, n in sorted_gws[:top_n]:
        lines.append(f" {gw:>2} | {bias:+.3f} | {n}")
    return "\n".join(lines)


def run_all_folds(out_dir: Path | None = None) -> list[FoldDiagnostics]:
    """Run diagnostics across all available folds, write artifacts."""
    folds: list[FoldDiagnostics] = []
    for f in FOLDS:
        diag = compute_fold_diagnostics(f["test"], f["train"])
        if diag is None:
            print(f"  (no cache for fold {f['test']}; skipped)")
            continue
        folds.append(diag)

    table = format_cross_fold_table(folds)
    print(table)
    print()
    for d in folds:
        print(format_decile_table(d))
        print()
        print(format_per_gw_bias(d, top_n=10))
        print()

    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "cross_fold_summary.txt").write_text(table + "\n")
        for d in folds:
            (out_dir / f"fold_{d.test_season}_decile.txt").write_text(
                format_decile_table(d) + "\n"
            )
            (out_dir / f"fold_{d.test_season}_per_gw_bias.txt").write_text(
                format_per_gw_bias(d, top_n=38) + "\n"
            )
        print(f"  artifacts in {out_dir}")

    return folds
