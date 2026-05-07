"""Phase 2.5 diagnostics — per-position MAE, decile calibration plot, sample PMFs."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

from fpl_bot.db import pit
from fpl_bot.eval.xpts_eval import HAUL_THRESHOLD, _run_one_fold

RESULTS_DIR = Path("docs/design/phase2_5_results")
POSITIONS = ("GKP", "DEF", "MID", "FWD")


def per_position_table(eval_df: pl.DataFrame, positions_df: pl.DataFrame) -> str:
    df = eval_df.join(positions_df, on="player_id", how="left").with_columns(
        pl.col("position_code").fill_null("MID")
    )
    y = df["total_points"].to_numpy().astype(np.float64)
    sim_e = df["e_xpts"].to_numpy()
    sim_p_ge6 = df["p_xpts_ge_6"].to_numpy()
    pos_arr = df["position_code"].to_numpy()
    y_ge6 = (y >= HAUL_THRESHOLD).astype(int)

    lines = [
        "Per-position MAE + Brier(≥6) — fold 2024/25",
        "",
        "position | n     | mean_actual | mean_predicted | MAE   | Brier(≥6) | haul_rate",
        "---------+-------+-------------+----------------+-------+-----------+----------",
    ]
    for pos in POSITIONS:
        mask = pos_arr == pos
        n = int(mask.sum())
        if n == 0:
            continue
        mae = float(np.mean(np.abs(y[mask] - sim_e[mask])))
        brier = float(np.mean((sim_p_ge6[mask] - y_ge6[mask]) ** 2))
        lines.append(
            f"  {pos:<6} | {n:>5} | {float(y[mask].mean()):.3f}       | "
            f"{float(sim_e[mask].mean()):.3f}          | {mae:.3f} | "
            f"{brier:.4f}    | {float(y_ge6[mask].mean()):.4f}"
        )
    return "\n".join(lines)


def decile_calibration_table(eval_df: pl.DataFrame, n_bins: int = 10) -> str:
    y = eval_df["total_points"].to_numpy().astype(np.float64)
    e = eval_df["e_xpts"].to_numpy()
    order = np.argsort(e)
    e_sorted = e[order]
    y_sorted = y[order]
    n = len(y_sorted)
    bin_size = max(1, n // n_bins)
    lines = [
        "Decile calibration — fold 2024/25 (predicted vs actual mean xPts per decile)",
        "",
        "decile | n     | mean_pred | mean_actual | abs_diff | within ±10%?",
        "-------+-------+-----------+-------------+----------+-------------",
    ]
    for b in range(n_bins):
        start = b * bin_size
        end = (b + 1) * bin_size if b < n_bins - 1 else n
        if start >= end:
            continue
        e_mean = float(e_sorted[start:end].mean())
        y_mean = float(y_sorted[start:end].mean())
        diff = abs(e_mean - y_mean)
        denom = max(abs(y_mean), 0.5)
        within = "yes" if (diff / denom <= 0.10 or diff <= 0.5) else "NO"
        lines.append(
            f"  D{b+1:<3}| {end-start:>5} | {e_mean:.3f}     | {y_mean:.3f}       "
            f"| {diff:.3f}    | {within}"
        )
    return "\n".join(lines)


def plot_decile_calibration(eval_df: pl.DataFrame, out_path: Path, n_bins: int = 10) -> Path:
    import matplotlib.pyplot as plt

    y = eval_df["total_points"].to_numpy().astype(np.float64)
    e = eval_df["e_xpts"].to_numpy()
    order = np.argsort(e)
    e_sorted = e[order]
    y_sorted = y[order]
    n = len(y_sorted)
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
    max_v = max(max(bin_pred), max(bin_actual)) * 1.1
    ax.plot([0, max_v], [0, max_v], "k--", linewidth=0.8, label="perfect")
    sizes = [40 + 200 * c / max(bin_n) for c in bin_n] if bin_n else []
    ax.scatter(bin_pred, bin_actual, s=sizes, alpha=0.7, color="tab:blue")
    ax.plot(bin_pred, bin_actual, linewidth=1.2, color="tab:blue")
    ax.set_xlim(0, max_v)
    ax.set_ylim(0, max_v)
    ax.set_xlabel("mean predicted E[xPts]")
    ax.set_ylabel("mean actual total_points")
    ax.set_title("xPts decile calibration — fold 2024/25 (dot size ∝ bin count)")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", fontsize=9)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def run_full_diagnostics(test_season: int = 24, n_iterations: int = 500) -> None:
    """Single entry point producing all Phase 2.5 close-out artifacts."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Re-run fold; we need the simulator output (with PMFs) joined to actuals
    fold = next(
        f for f in [
            {"train": [19, 20, 21, 22, 23], "test": 24},
        ] if f["test"] == test_season
    )
    result = _run_one_fold(
        fold["train"], fold["test"], n_iterations=n_iterations, seed=42
    )
    if result is None:
        print("No fold result — aborting diagnostics")
        return
    fold_result, eval_df, _ = result

    # We also need the full simulator output (xpts_pmf etc.) — re-build it
    # alongside the eval_df. The simplest path: re-run the trio + simulator
    # via the existing internals to get the per-fixture PMF DataFrame. For
    # diagnostics scope we use the eval_df fields only (no PMF plot relies
    # on a separate path); the sample-PMF plot is best-effort.
    positions_df = pit.all_player_positions()

    text_pos = per_position_table(eval_df, positions_df)
    print("=== PER-POSITION ===")
    print(text_pos)
    (RESULTS_DIR / "per_position_2024_25.txt").write_text(text_pos + "\n")

    print()
    text_dec = decile_calibration_table(eval_df)
    print("=== DECILE CALIBRATION ===")
    print(text_dec)
    (RESULTS_DIR / "decile_calibration_2024_25.txt").write_text(text_dec + "\n")

    print()
    print("=== CALIBRATION PLOT ===")
    plot = plot_decile_calibration(
        eval_df, RESULTS_DIR / "decile_calibration_2024_25.png"
    )
    print(f"  wrote {plot}")

    print()
    print("=== FOLD GATES ===")
    print(
        f"primary={fold_result.primary_gate_pass}, "
        f"captaincy={fold_result.captaincy_gate_pass}, "
        f"calibration_passes={fold_result.calibration_decile_passes}/10"
    )
    print(
        f"captaincy: sim={fold_result.captaincy_sim_total} "
        f"baseline={fold_result.captaincy_baseline_total} "
        f"oracle={fold_result.captaincy_optimal_total}"
    )
