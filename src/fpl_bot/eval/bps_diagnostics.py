"""Phase 2.4 diagnostics — per-position ECE, per-tier Brier, calibration plots,
no-residual comparison.

Runs on a single most-data-rich fold per the 2.1/2.2/2.3 pattern. Outputs
land under `docs/design/phase2_4_results/`.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

from fpl_bot.db import pit
from fpl_bot.db.event_source import EmpiricalResidualEventSource
from fpl_bot.eval.bps_eval import (
    WALK_FORWARD_FOLDS,
    _ece,
    _train_goals_or_assists_predictor,
    _train_minutes_predictor,
)
from fpl_bot.features.bps import (
    assemble_fold_predictions,
    baseline_top3_by_xpts,
    fixture_inputs_iter,
)
from fpl_bot.models.bps import (
    BPSSimulator,
    fit_residual_dataset,
    split_p_full_by_position,
)

RESULTS_DIR = Path("docs/design/phase2_4_results")
POSITIONS = ("GKP", "DEF", "MID", "FWD")


@dataclass
class _DiagFoldData:
    """Holds the predictions + actuals needed by every diagnostic table."""

    eval_df: pl.DataFrame
    positions_test: list[str]
    no_residual_predictions: pl.DataFrame  # simulator with event_source.simulate_unmodeled_bps ≡ 0


class _ZeroResidualEventSource:
    """Sentinel: returns 0 BPS contribution for unmodeled events. Used for the
    no-residual comparison only."""

    def simulate_unmodeled_bps(self, position: str, minutes_played: int, rng) -> float:
        return 0.0

    def fit(self, train_df: pl.DataFrame) -> None:
        return None


def _run_fold_for_diagnostics(
    test_season: int = 24,
    n_iterations: int = 200,
    seed: int = 42,
) -> _DiagFoldData:
    """Replays the per-fold pipeline once for the chosen test_season and also
    runs a no-residual variant. Returns both prediction sets joined to actuals.
    """
    fold = next(f for f in WALK_FORWARD_FOLDS if f["test"] == test_season)
    train_seasons = fold["train"]

    minutes_pred = _train_minutes_predictor(train_seasons, test_season)
    goals_pred = _train_goals_or_assists_predictor(
        train_seasons, test_season, target="goals"
    )
    assists_pred = _train_goals_or_assists_predictor(
        train_seasons, test_season, target="assists"
    )

    positions_df = pit.all_player_positions()
    train_pm = pit.all_player_match_with_kickoff(season_ids=train_seasons)
    residual_df = fit_residual_dataset(train_pm, positions_df)
    fitted_source = EmpiricalResidualEventSource()
    fitted_source.fit(residual_df)
    zero_source = _ZeroResidualEventSource()

    train_pm_for_alphas = train_pm.join(positions_df, on="player_id", how="left").drop_nulls(
        "position_code"
    )
    alphas = split_p_full_by_position(train_pm_for_alphas)

    prepared = assemble_fold_predictions(
        test_season=test_season,
        minutes_predictions=minutes_pred,
        goals_predictions=goals_pred,
        assists_predictions=assists_pred,
        alphas_by_position=alphas,
    )

    def _simulate_all(source) -> pl.DataFrame:
        sim = BPSSimulator(event_source=source, n_iterations=n_iterations, seed=seed)
        rows = []
        for fix_inputs in fixture_inputs_iter(prepared):
            r = sim.simulate_fixture(fix_inputs)
            if not r.is_empty():
                rows.append(r.with_columns(pl.lit(fix_inputs.fixture_id).alias("fixture_id")))
        return pl.concat(rows, how="vertical_relaxed") if rows else pl.DataFrame()

    full_predictions = _simulate_all(fitted_source)
    no_res_predictions = _simulate_all(zero_source)

    test_pm = pit.all_player_match_with_kickoff(season_ids=[test_season])
    actual = test_pm.select("player_id", "fixture_id", "bonus", "minutes").rename(
        {"bonus": "actual_bonus"}
    )

    top3 = baseline_top3_by_xpts(prepared.player_predictions, prepared.fixtures)

    eval_df = (
        full_predictions.rename({
            "p_bonus_0": "sim_p0",
            "p_bonus_1": "sim_p1",
            "p_bonus_2": "sim_p2",
            "p_bonus_3": "sim_p3",
        })
        .join(top3.select(
            "player_id", "fixture_id",
            pl.col("p_bonus_0").alias("top3_p0"),
            pl.col("p_bonus_1").alias("top3_p1"),
            pl.col("p_bonus_2").alias("top3_p2"),
            pl.col("p_bonus_3").alias("top3_p3"),
        ), on=["player_id", "fixture_id"], how="left")
        .join(actual, on=["player_id", "fixture_id"], how="inner")
        .drop_nulls("actual_bonus")
        .join(positions_df, on="player_id", how="left")
        .with_columns(pl.col("position_code").fill_null("MID"))
    )

    positions_test = eval_df["position_code"].to_list()
    return _DiagFoldData(
        eval_df=eval_df,
        positions_test=positions_test,
        no_residual_predictions=no_res_predictions,
    )


# ── Per-position ECE table ────────────────────────────────────────────────────


def per_position_ece_table(data: _DiagFoldData) -> str:
    eval_df = data.eval_df
    y = eval_df.select("actual_bonus").to_pandas().to_numpy().ravel().astype(int)
    sim_p_pos = (1.0 - eval_df["sim_p0"]).to_numpy()
    sim_p_3 = eval_df["sim_p3"].to_numpy()
    y_pos = (y > 0).astype(int)
    y_3 = (y == 3).astype(int)
    pos_arr = np.array(data.positions_test)

    lines = [
        "Per-position calibration on P(bonus > 0) and P(bonus = 3) — fold 2024/25",
        "",
        "position | n     | bonus_pos_rate | sim_mean_p_pos | ece_pos | sim_mean_p_3 | ece_3",
        "---------+-------+----------------+----------------+---------+--------------+------",
    ]
    for pos in POSITIONS:
        mask = pos_arr == pos
        n = int(mask.sum())
        if n == 0:
            continue
        rate = float(y_pos[mask].mean())
        mean_p_pos = float(sim_p_pos[mask].mean())
        ece_pos = _ece(y_pos[mask], sim_p_pos[mask])
        mean_p_3 = float(sim_p_3[mask].mean())
        ece_3 = _ece(y_3[mask], sim_p_3[mask])
        lines.append(
            f"  {pos:<6} | {n:>5} | {rate:.4f}         | {mean_p_pos:.4f}         "
            f"| {ece_pos:.3f}   | {mean_p_3:.4f}       | {ece_3:.3f}"
        )
    return "\n".join(lines)


# ── Per-bonus-tier Brier ──────────────────────────────────────────────────────


def per_tier_brier_table(data: _DiagFoldData) -> str:
    eval_df = data.eval_df
    y = eval_df.select("actual_bonus").to_pandas().to_numpy().ravel().astype(int)

    lines = [
        "Per-bonus-tier Brier — fold 2024/25 (lower = better)",
        "",
        "tier | n_actual | sim_brier | top3_brier | sim - top3",
        "-----+----------+-----------+------------+-----------",
    ]
    for k in (1, 2, 3):
        target = (y == k).astype(int)
        sim_p = eval_df[f"sim_p{k}"].to_numpy()
        top3_p = eval_df[f"top3_p{k}"].fill_null(0.0).to_numpy()
        sim_brier = float(np.mean((sim_p - target) ** 2))
        top3_brier = float(np.mean((top3_p - target) ** 2))
        lines.append(
            f"   {k} | {int(target.sum()):>8} | {sim_brier:.4f}    "
            f"| {top3_brier:.4f}     | {sim_brier - top3_brier:+.4f}"
        )
    return "\n".join(lines)


# ── No-residual comparison ────────────────────────────────────────────────────


def residual_value_table(data: _DiagFoldData) -> str:
    """Compare full simulator (with empirical residual) to a no-residual variant.
    Quantifies the value the residual source adds.
    """
    eval_df = data.eval_df
    no_res = data.no_residual_predictions

    if no_res.is_empty():
        return "(no_residual run produced no rows)"

    joined = (
        no_res.rename({
            "p_bonus_0": "no_res_p0",
            "p_bonus_3": "no_res_p3",
        })
        .select("player_id", "fixture_id", "no_res_p0", "no_res_p3")
        .join(eval_df, on=["player_id", "fixture_id"], how="inner")
    )
    y = joined.select("actual_bonus").to_pandas().to_numpy().ravel().astype(int)
    y_pos = (y > 0).astype(int)
    y_3 = (y == 3).astype(int)

    sim_p_pos = (1.0 - joined["sim_p0"]).to_numpy()
    no_res_p_pos = (1.0 - joined["no_res_p0"]).to_numpy()
    sim_p_3 = joined["sim_p3"].to_numpy()
    no_res_p_3 = joined["no_res_p3"].to_numpy()

    lines = [
        "Residual source value — fold 2024/25 (lower Brier = better)",
        "",
        "metric          | with_residual | no_residual | residual_lift",
        "----------------+---------------+-------------+--------------",
        f"Brier P(>0)     | {float(np.mean((sim_p_pos - y_pos) ** 2)):.4f}        "
        f"| {float(np.mean((no_res_p_pos - y_pos) ** 2)):.4f}      "
        f"| {float(np.mean((no_res_p_pos - y_pos) ** 2)) - float(np.mean((sim_p_pos - y_pos) ** 2)):+.4f}",
        f"Brier P(=3)     | {float(np.mean((sim_p_3 - y_3) ** 2)):.4f}        "
        f"| {float(np.mean((no_res_p_3 - y_3) ** 2)):.4f}      "
        f"| {float(np.mean((no_res_p_3 - y_3) ** 2)) - float(np.mean((sim_p_3 - y_3) ** 2)):+.4f}",
        f"ECE P(>0)       | {_ece(y_pos, sim_p_pos):.4f}        "
        f"| {_ece(y_pos, no_res_p_pos):.4f}      | -",
    ]
    return "\n".join(lines)


# ── Calibration plot ──────────────────────────────────────────────────────────


def plot_calibration_per_position(
    data: _DiagFoldData,
    out_path: Path,
    n_bins: int = 10,
) -> Path:
    """Per-position reliability diagram for P(bonus > 0). 2x2 subplots."""
    import matplotlib.pyplot as plt

    eval_df = data.eval_df
    y = eval_df.select("actual_bonus").to_pandas().to_numpy().ravel().astype(int)
    y_pos = (y > 0).astype(int)
    sim_p_pos = (1.0 - eval_df["sim_p0"]).to_numpy()
    pos_arr = np.array(data.positions_test)

    fig, axes = plt.subplots(2, 2, figsize=(10, 9))
    bins = np.linspace(0, 1, n_bins + 1)
    for ax, pos in zip(axes.flat, POSITIONS, strict=False):
        mask = pos_arr == pos
        if mask.sum() == 0:
            ax.set_title(f"{pos}: no data")
            continue
        p = sim_p_pos[mask]
        y_p = y_pos[mask]
        bin_pred, bin_actual, bin_n = [], [], []
        for b in range(n_bins):
            m = (p >= bins[b]) & (p < bins[b + 1])
            if b == n_bins - 1:
                m = m | (p == 1.0)
            if m.sum() == 0:
                continue
            bin_pred.append(float(p[m].mean()))
            bin_actual.append(float(y_p[m].mean()))
            bin_n.append(int(m.sum()))
        ece_p = _ece(y_p, p)
        ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, label="perfect")
        sizes = [40 + 200 * c / max(bin_n) for c in bin_n] if bin_n else []
        ax.scatter(bin_pred, bin_actual, s=sizes, alpha=0.7)
        ax.plot(bin_pred, bin_actual, linewidth=1.2)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("predicted P(bonus > 0)")
        ax.set_ylabel("empirical P(bonus > 0)")
        ax.set_title(f"{pos}: n={int(mask.sum())}, ECE={ece_p:.3f}")
        ax.grid(alpha=0.3)
        ax.legend(loc="upper left", fontsize=8)

    fig.suptitle("BPS simulator calibration per position — fold 2024/25")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    import matplotlib.pyplot as _plt

    _plt.close(fig)
    return out_path


def run_full_diagnostics(test_season: int = 24, n_iterations: int = 200) -> None:
    """Single entrypoint to produce all Phase 2.4 close-out artifacts."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    data = _run_fold_for_diagnostics(test_season=test_season, n_iterations=n_iterations)

    pos_table = per_position_ece_table(data)
    print("=== PER-POSITION ECE ===")
    print(pos_table)
    (RESULTS_DIR / f"per_position_ece_20{test_season}_{(test_season+1)%100:02d}.txt").write_text(
        pos_table + "\n"
    )

    print()
    tier_table = per_tier_brier_table(data)
    print("=== PER-TIER BRIER ===")
    print(tier_table)
    (RESULTS_DIR / f"per_tier_brier_20{test_season}_{(test_season+1)%100:02d}.txt").write_text(
        tier_table + "\n"
    )

    print()
    res_table = residual_value_table(data)
    print("=== RESIDUAL SOURCE VALUE ===")
    print(res_table)
    (RESULTS_DIR / f"residual_value_20{test_season}_{(test_season+1)%100:02d}.txt").write_text(
        res_table + "\n"
    )

    print()
    print("=== CALIBRATION PLOT ===")
    plot = plot_calibration_per_position(
        data,
        out_path=RESULTS_DIR
        / f"calibration_per_position_20{test_season}_{(test_season+1)%100:02d}.png",
    )
    print(f"  wrote {plot}")
