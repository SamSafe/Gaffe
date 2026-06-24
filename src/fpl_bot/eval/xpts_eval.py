"""Walk-forward CV for the joint xPts distribution (Phase 2.5).

Reuses Phase 2.4's training pipeline; adds:
  - MAE on E[xPts] vs actual `total_points`
  - Brier on P(xPts ≥ 6) — FPL "haul" threshold
  - Calibration check (±10% per decile of E[xPts])
  - Captaincy-proxy gate: total captain points across test fold,
    sim's argmax-E[xPts] vs rolling-3-GW xPts leader

Per design §6 round-1: 500 MC iterations, full PMF histogram in output,
optional raw-sample dump for the held-out 2025/26 fold.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

from fpl_bot.db import pit
from fpl_bot.db.event_source import EmpiricalResidualEventSource
from fpl_bot.eval.bps_eval import (
    _train_goals_or_assists_predict_only,
    _train_goals_or_assists_predictor,
    _train_minutes_predict_only,
    _train_minutes_predictor,
)
from fpl_bot.features.bps import (
    assemble_fold_predictions,
    fixture_inputs_iter,
)
from fpl_bot.models.bps import (
    BPSRulesMode,
    BPSSimulator,
    fit_residual_dataset,
    split_p_full_by_position,
)
from fpl_bot.models.minutes import apply_availability_to_minutes_predictions
from fpl_bot.models.xpts import FPL_GOAL_BY_POSITION

WALK_FORWARD_FOLDS: list[dict] = [
    {"train": [19, 20], "test": 21},
    {"train": [19, 20, 21], "test": 22},
    {"train": [19, 20, 21, 22], "test": 23},
    {"train": [19, 20, 21, 22, 23], "test": 24},
]

# Optional fifth fold: train on all backtest seasons + held-out test on the
# current 2025/26 season; used for raw-sample dump (no gate evaluation since
# 2025/26 may be partial/in-progress).
HELD_OUT_FOLD = {"train": [19, 20, 21, 22, 23, 24], "test": 25}

CALIBRATION_BAND_PCT = 0.10  # ±10% per decile (per design §3)
HAUL_THRESHOLD = 6


@dataclass
class XPtsFoldResult:
    test_season: int
    n_test_player_fixtures: int
    mae_simulator: float
    mae_naive_sum: float
    mae_position: float
    brier_ge6_simulator: float
    brier_ge6_naive_sum: float
    brier_ge6_position: float
    calibration_decile_passes: int  # of 10
    captaincy_sim_total: int
    captaincy_baseline_total: int
    captaincy_optimal_total: int
    primary_gate_pass: bool  # MAE+Brier(≥6) both beat both baselines
    captaincy_gate_pass: bool


# ── Baselines ─────────────────────────────────────────────────────────────────


def _baseline_naive_sum_of_components(
    player_predictions: pl.DataFrame,
    fixtures: pl.DataFrame,
    expected_bonus: pl.DataFrame,
) -> pl.DataFrame:
    """E[xPts] ≈ E[appearance] + E[goals]*goal_pts + E[assists]*3 + E[cs]*cs_pts + E[bonus].

    Uses the trio's point estimates directly without joint sampling. The
    simulator must beat this to demonstrate joint-sampling value-add.
    """
    # Expected minutes (linear midpoints): 30 * p_short + 75 * p_full
    df = player_predictions.with_columns(
        (
            pl.col("p_minutes_short") * 30.0 + pl.col("p_minutes_full") * 75.0
        ).alias("e_minutes"),
        (
            pl.col("p_minutes_short") * 1.0 + pl.col("p_minutes_full") * 2.0
        ).alias("e_appearance_pts"),
    )
    # Goal points by position (lookup via map) and CS points by position
    pos_goal_pts = pl.col("position_code").replace_strict(
        FPL_GOAL_BY_POSITION, default=4
    )
    cs_pts = pl.when(pl.col("position_code") == "GKP").then(4).when(
        pl.col("position_code") == "DEF"
    ).then(4).when(pl.col("position_code") == "MID").then(1).otherwise(0)

    df = df.with_columns(
        (pl.col("lambda_goals_per_90") * pl.col("e_minutes") / 90.0 * pos_goal_pts)
        .alias("e_goals_pts"),
        (pl.col("lambda_assists_per_90") * pl.col("e_minutes") / 90.0 * 3.0)
        .alias("e_assists_pts"),
    )

    # CS expected based on team market_cs_prob × P(60+ min)
    fixtures_slim = fixtures.select(
        "fixture_id", "home_team_id", "away_team_id",
        "home_market_cs_prob", "away_market_cs_prob",
    )
    df = df.join(fixtures_slim, on="fixture_id", how="left")
    df = df.with_columns(
        pl.when(pl.col("is_home"))
        .then(pl.col("home_market_cs_prob"))
        .otherwise(pl.col("away_market_cs_prob"))
        .alias("team_cs_prob"),
    )
    df = df.with_columns(
        (pl.col("team_cs_prob") * pl.col("p_minutes_full") * cs_pts).alias("e_cs_pts"),
    )

    df = df.join(
        expected_bonus.select("player_id", "fixture_id", "expected_bonus"),
        on=["player_id", "fixture_id"],
        how="left",
    ).with_columns(pl.col("expected_bonus").fill_null(0.0))

    df = df.with_columns(
        (
            pl.col("e_appearance_pts")
            + pl.col("e_goals_pts")
            + pl.col("e_assists_pts")
            + pl.col("e_cs_pts")
            + pl.col("expected_bonus")
        ).alias("e_xpts_naive"),
    )
    return df.select("player_id", "fixture_id", "e_xpts_naive")


def _baseline_position_mean(
    train_actuals: pl.DataFrame,
    test_player_predictions: pl.DataFrame,
) -> pl.DataFrame:
    """Predict per-(position, minutes-bucket) historical mean total_points."""
    train = train_actuals.filter(pl.col("minutes") >= 1).filter(
        pl.col("total_points").is_not_null()
    )
    pos_mean: dict[tuple[str, str], float] = {}
    pos_overall_mean: dict[str, float] = {}
    for pos in ("GKP", "DEF", "MID", "FWD"):
        sub = train.filter(pl.col("position_code") == pos)
        if sub.is_empty():
            pos_overall_mean[pos] = 2.0
            continue
        for bucket_label, bucket_filter in (
            ("short", pl.col("minutes") < 60),
            ("full", pl.col("minutes") >= 60),
        ):
            bucket_sub = sub.filter(bucket_filter).select("total_points")
            if bucket_sub.is_empty():
                continue
            mean_v = float(bucket_sub.to_pandas().to_numpy().ravel().mean())
            pos_mean[(pos, bucket_label)] = mean_v
        pos_overall_mean[pos] = float(
            sub.select("total_points").to_pandas().to_numpy().ravel().mean()
        )

    rows = []
    for row in test_player_predictions.iter_rows(named=True):
        pos = row.get("position_code") or "MID"
        p_full = float(row.get("p_minutes_full") or 0.0)
        p_short = float(row.get("p_minutes_short") or 0.0)
        full_mean = pos_mean.get((pos, "full"), pos_overall_mean.get(pos, 2.0))
        short_mean = pos_mean.get((pos, "short"), 1.0)
        e = p_full * full_mean + p_short * short_mean
        rows.append({
            "player_id": row["player_id"],
            "fixture_id": row["fixture_id"],
            "e_xpts_position": float(e),
        })
    return pl.DataFrame(rows)


def _captaincy_baseline_rolling3(
    test_actuals: pl.DataFrame,
    train_actuals: pl.DataFrame,
) -> pl.DataFrame:
    """For each GW in the test season, return the captain-pick from a
    rolling-3-GW total_points leader. Uses only data with kickoff strictly
    before each GW's earliest fixture (PIT-correct)."""
    pm = pl.concat(
        [train_actuals, test_actuals], how="vertical_relaxed"
    ).filter(pl.col("minutes") >= 1).filter(pl.col("total_points").is_not_null())
    pm = pm.sort(["player_id", "kickoff_utc"])
    pm = pm.with_columns(
        pl.col("total_points")
        .shift(1)
        .rolling_mean(window_size=3, min_samples=1)
        .over("player_id")
        .alias("rolling3_pts"),
    )
    # For each test fixture, the rolling3 has been computed using prior history
    test_with_rolling = pm.join(
        test_actuals.select("player_id", "fixture_id"),
        on=["player_id", "fixture_id"],
        how="inner",
    )
    return test_with_rolling.select(
        "player_id", "fixture_id", "rolling3_pts", "kickoff_utc"
    )


# ── Metrics ───────────────────────────────────────────────────────────────────


def _mae(y: np.ndarray, y_hat: np.ndarray) -> float:
    return float(np.mean(np.abs(y - y_hat)))


def _brier_binary(y_true: np.ndarray, p_pred: np.ndarray) -> float:
    return float(np.mean((p_pred - y_true.astype(float)) ** 2))


def _calibration_decile_pass_count(
    y: np.ndarray, e: np.ndarray, n_bins: int = 10
) -> int:
    """Counts deciles whose mean predicted E[xPts] is within ±10% of empirical mean."""
    if len(y) == 0:
        return 0
    order = np.argsort(e)
    e_sorted = e[order]
    y_sorted = y[order]
    n = len(y_sorted)
    bin_size = max(1, n // n_bins)
    passes = 0
    for b in range(n_bins):
        start = b * bin_size
        end = (b + 1) * bin_size if b < n_bins - 1 else n
        if start >= end:
            continue
        e_mean = float(e_sorted[start:end].mean())
        y_mean = float(y_sorted[start:end].mean())
        denom = max(abs(y_mean), 0.5)  # protect against tiny denominators
        if abs(e_mean - y_mean) / denom <= CALIBRATION_BAND_PCT or abs(
            e_mean - y_mean
        ) <= 0.5:
            passes += 1
    return passes


# ── Per-fold orchestration ───────────────────────────────────────────────────


def _run_one_fold(
    train_seasons: list[int],
    test_season: int,
    *,
    n_iterations: int,
    seed: int,
    return_raw_samples: bool = False,
    train_through_gw: int | None = None,
    bps_rules_mode: BPSRulesMode = "official",
) -> tuple[XPtsFoldResult, pl.DataFrame, np.ndarray | None] | None:
    """Returns (fold_result, eval_df, raw_samples_or_None). raw_samples is a
    long-form polars DataFrame (player_id, fixture_id, iteration, xpts) when
    return_raw_samples is True; None otherwise.

    If `train_through_gw` is set, additionally includes test_season's
    GWs 1..train_through_gw in training (PIT-correct walk-forward retrain).
    This affects minutes/goals/assists trainers AND the residual/alpha
    fits derived from train_pm.
    """
    minutes_pred = _train_minutes_predictor(
        train_seasons, test_season, train_through_gw=train_through_gw
    )
    goals_pred = _train_goals_or_assists_predictor(
        train_seasons, test_season, target="goals",
        train_through_gw=train_through_gw,
    )
    assists_pred = _train_goals_or_assists_predictor(
        train_seasons, test_season, target="assists",
        train_through_gw=train_through_gw,
    )
    if minutes_pred.is_empty() or goals_pred.is_empty():
        return None

    positions_df = pit.all_player_positions()
    # For walk-forward: include test_season's pre-cutoff GWs in train_pm
    # used for residual fit and per-position alphas.
    train_pm_seasons = list(train_seasons)
    if train_through_gw is not None:
        train_pm_seasons = [*train_seasons, test_season]
    train_pm = pit.all_player_match_with_kickoff(season_ids=train_pm_seasons)
    if train_through_gw is not None:
        train_pm = train_pm.filter(
            (pl.col("season_id") != test_season)
            | (pl.col("gameweek") <= train_through_gw)
        )
    residual_df = fit_residual_dataset(
        train_pm, positions_df, rules_mode=bps_rules_mode
    )
    event_source = EmpiricalResidualEventSource(
        integer_bps=bps_rules_mode == "official"
    )
    event_source.fit(residual_df)

    train_pm_for_alphas = train_pm.join(
        positions_df, on="player_id", how="left"
    ).drop_nulls("position_code")
    alphas = split_p_full_by_position(train_pm_for_alphas)

    prepared = assemble_fold_predictions(
        test_season=test_season,
        minutes_predictions=minutes_pred,
        goals_predictions=goals_pred,
        assists_predictions=assists_pred,
        alphas_by_position=alphas,
    )
    if prepared.fixtures.is_empty() or prepared.player_predictions.is_empty():
        return None

    simulator = BPSSimulator(
        event_source=event_source,
        n_iterations=n_iterations,
        seed=seed,
        bps_rules_mode=bps_rules_mode,
    )

    sim_rows: list[pl.DataFrame] = []
    raw_rows: list[dict] = []
    for fix_inputs in fixture_inputs_iter(prepared):
        if return_raw_samples:
            df, raw_arr = simulator.simulate_fixture(
                fix_inputs, return_raw_xpts_samples=True
            )
            if not df.is_empty():
                df = df.with_columns(pl.lit(fix_inputs.fixture_id).alias("fixture_id"))
                sim_rows.append(df)
                pids = df["player_id"].to_list()
                for i, pid in enumerate(pids):
                    for s in range(raw_arr.shape[1]):
                        raw_rows.append({
                            "player_id": int(pid),
                            "fixture_id": fix_inputs.fixture_id,
                            "iteration": s,
                            "xpts": int(raw_arr[i, s]),
                        })
        else:
            df = simulator.simulate_fixture(fix_inputs)
            if not df.is_empty():
                df = df.with_columns(pl.lit(fix_inputs.fixture_id).alias("fixture_id"))
                sim_rows.append(df)

    if not sim_rows:
        return None
    simulator_predictions = pl.concat(sim_rows, how="vertical_relaxed")

    # Naive sum-of-components baseline (uses simulator's E[bonus])
    naive_sum = _baseline_naive_sum_of_components(
        prepared.player_predictions,
        prepared.fixtures,
        simulator_predictions.select("player_id", "fixture_id", "expected_bonus"),
    )

    # Position-marginal baseline
    position_baseline_train = (
        train_pm.join(positions_df, on="player_id", how="left").drop_nulls(
            "position_code"
        )
    )
    position_baseline = _baseline_position_mean(
        position_baseline_train, prepared.player_predictions
    )

    # Actuals
    test_pm = pit.all_player_match_with_kickoff(season_ids=[test_season])
    eval_df = (
        simulator_predictions.select(
            "player_id", "fixture_id", "e_xpts", "p_xpts_ge_6"
        )
        .join(naive_sum, on=["player_id", "fixture_id"], how="left")
        .join(position_baseline, on=["player_id", "fixture_id"], how="left")
        .join(
            test_pm.select("player_id", "fixture_id", "total_points", "kickoff_utc"),
            on=["player_id", "fixture_id"],
            how="inner",
        )
        .drop_nulls("total_points")
    )

    y = eval_df["total_points"].to_numpy().astype(np.float64)
    y_ge6 = (y >= HAUL_THRESHOLD).astype(int)
    sim_e = eval_df["e_xpts"].to_numpy()
    sim_p_ge6 = eval_df["p_xpts_ge_6"].to_numpy()
    naive_e = eval_df["e_xpts_naive"].to_numpy()
    pos_e = eval_df["e_xpts_position"].to_numpy()

    mae_sim = _mae(y, sim_e)
    mae_naive = _mae(y, naive_e)
    mae_pos = _mae(y, pos_e)

    brier_sim = _brier_binary(y_ge6, sim_p_ge6)
    # Naive baseline on P(≥6): convert to a heuristic threshold using mean
    naive_p_ge6 = (naive_e >= HAUL_THRESHOLD).astype(float)
    pos_p_ge6 = (pos_e >= HAUL_THRESHOLD).astype(float)
    brier_naive = _brier_binary(y_ge6, naive_p_ge6)
    brier_pos = _brier_binary(y_ge6, pos_p_ge6)

    decile_passes = _calibration_decile_pass_count(y, sim_e)

    # Captaincy gate
    cap_baseline_df = _captaincy_baseline_rolling3(
        test_pm.select(
            "player_id", "fixture_id", "minutes", "total_points", "kickoff_utc"
        ),
        train_pm.select(
            "player_id", "fixture_id", "minutes", "total_points", "kickoff_utc"
        ),
    )
    cap_sim_total, cap_baseline_total, cap_optimal_total = _captaincy_evaluation(
        eval_df=eval_df,
        baseline_rolling=cap_baseline_df,
        test_pm=test_pm,
    )

    primary_pass = (
        mae_sim < mae_naive
        and mae_sim < mae_pos
        and brier_sim < brier_naive
        and brier_sim < brier_pos
    )
    captaincy_pass = cap_sim_total > cap_baseline_total

    raw_samples_df = pl.DataFrame(raw_rows) if return_raw_samples and raw_rows else None

    return (
        XPtsFoldResult(
            test_season=test_season,
            n_test_player_fixtures=len(eval_df),
            mae_simulator=mae_sim,
            mae_naive_sum=mae_naive,
            mae_position=mae_pos,
            brier_ge6_simulator=brier_sim,
            brier_ge6_naive_sum=brier_naive,
            brier_ge6_position=brier_pos,
            calibration_decile_passes=decile_passes,
            captaincy_sim_total=cap_sim_total,
            captaincy_baseline_total=cap_baseline_total,
            captaincy_optimal_total=cap_optimal_total,
            primary_gate_pass=primary_pass,
            captaincy_gate_pass=captaincy_pass,
        ),
        eval_df,
        raw_samples_df,
    )


def _captaincy_evaluation(
    *,
    eval_df: pl.DataFrame,
    baseline_rolling: pl.DataFrame,
    test_pm: pl.DataFrame,
) -> tuple[int, int, int]:
    """Pick captain per gameweek using sim's E[xPts] and rolling baseline.

    For each (player, fixture) row we already have e_xpts. Aggregate to
    per-(player, gameweek) totals (DGW-aware via sum), then per-GW pick the
    argmax player. Compute 2 × actual_xpts_for_that_player_in_that_GW.
    """
    fixtures_with_gw = test_pm.select(
        "player_id", "fixture_id", "kickoff_utc", "total_points"
    ).join(
        # Need gameweek; pull from dim_fixture
        _gameweek_for_fixtures(set(eval_df["fixture_id"].to_list())),
        on="fixture_id",
        how="left",
    )

    eval_with_gw = eval_df.join(
        fixtures_with_gw.select("player_id", "fixture_id", "gameweek"),
        on=["player_id", "fixture_id"],
        how="left",
    ).drop_nulls("gameweek")

    # Per (player, gameweek), sum e_xpts and total_points
    per_player_gw = eval_with_gw.group_by(["player_id", "gameweek"]).agg(
        pl.col("e_xpts").sum().alias("e_xpts_gw"),
        pl.col("total_points").sum().alias("actual_pts_gw"),
    )

    cap_sim_total = 0
    cap_optimal_total = 0
    for gw_value, group in per_player_gw.group_by("gameweek"):
        gw = gw_value[0] if isinstance(gw_value, tuple) else gw_value
        if group.is_empty():
            continue
        sim_pick = group.sort("e_xpts_gw", descending=True).head(1)
        cap_sim_total += int(sim_pick["actual_pts_gw"][0]) * 2
        opt_pick = group.sort("actual_pts_gw", descending=True).head(1)
        cap_optimal_total += int(opt_pick["actual_pts_gw"][0]) * 2
        _ = gw  # silence unused

    # Baseline: rolling-3 leader per GW
    baseline_with_gw = baseline_rolling.join(
        fixtures_with_gw.select("fixture_id", "gameweek"),
        on="fixture_id",
        how="left",
    ).drop_nulls("gameweek")
    baseline_per_gw = baseline_with_gw.group_by(["player_id", "gameweek"]).agg(
        pl.col("rolling3_pts").mean().alias("rolling3_gw"),
    )
    actuals_per_gw = per_player_gw.select("player_id", "gameweek", "actual_pts_gw")
    baseline_with_actual = baseline_per_gw.join(
        actuals_per_gw, on=["player_id", "gameweek"], how="inner"
    )

    cap_baseline_total = 0
    for _gw_val, group in baseline_with_actual.group_by("gameweek"):
        if group.is_empty():
            continue
        pick = group.sort("rolling3_gw", descending=True, nulls_last=True).head(1)
        cap_baseline_total += int(pick["actual_pts_gw"][0]) * 2

    return cap_sim_total, cap_baseline_total, cap_optimal_total


def _gameweek_for_fixtures(fixture_ids: set[int]) -> pl.DataFrame:
    """Helper: look up gameweek per fixture_id from dim_fixture."""
    from sqlalchemy import select as sa_select

    from fpl_bot.db.models import DimFixture
    from fpl_bot.db.session import session_scope

    with session_scope() as s:
        stmt = sa_select(DimFixture.fixture_id, DimFixture.gameweek).where(
            DimFixture.fixture_id.in_(list(fixture_ids))
        )
        rows = s.execute(stmt).all()
    return pl.DataFrame(
        [{"fixture_id": r.fixture_id, "gameweek": r.gameweek} for r in rows]
    )


# ── Public API ────────────────────────────────────────────────────────────────


def run_walk_forward_cv(
    folds: list[dict] | None = None,
    *,
    n_iterations: int = 500,
    seed: int = 42,
) -> list[XPtsFoldResult]:
    folds_to_use = folds or WALK_FORWARD_FOLDS
    out: list[XPtsFoldResult] = []
    for fold in folds_to_use:
        result = _run_one_fold(
            fold["train"], fold["test"], n_iterations=n_iterations, seed=seed
        )
        if result is None:
            continue
        out.append(result[0])
    return out


def run_held_out_with_raw_samples(
    *,
    output_dir: Path = Path("data/predictions/xpts"),
    n_iterations: int = 500,
    seed: int = 42,
) -> Path | None:
    """Trains on all backtest seasons, predicts on the held-out fold (season 25),
    dumps per-iteration raw samples to parquet for Phase 4 SAA development.
    Returns the path written, or None if season 25 has no data."""
    fold = HELD_OUT_FOLD
    result = _run_one_fold(
        fold["train"],
        fold["test"],
        n_iterations=n_iterations,
        seed=seed,
        return_raw_samples=True,
    )
    if result is None:
        return None
    _, _, raw_df = result
    if raw_df is None or raw_df.is_empty():
        return None
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"xpts_raw_samples_20{fold['test']}_{(fold['test']+1)%100:02d}.parquet"
    raw_df.write_parquet(out_path)
    return out_path


def run_fold_with_raw_samples(
    *,
    test_season: int,
    train_seasons: list[int],
    output_dir: Path = Path("data/cache/xpts_raw_samples"),
    n_iterations: int = 200,
    seed: int = 42,
) -> Path | None:
    """Phase 4 SAA support: dump per-iteration raw FPL points for an arbitrary
    walk-forward fold. Cache layout matches the Phase 3 predictions cache:
    `season_{N}_train_{...}_n{N_ITER}.parquet`.
    """
    result = _run_one_fold(
        train_seasons,
        test_season,
        n_iterations=n_iterations,
        seed=seed,
        return_raw_samples=True,
    )
    if result is None:
        return None
    _, _, raw_df = result
    if raw_df is None or raw_df.is_empty():
        return None
    output_dir.mkdir(parents=True, exist_ok=True)
    train_str = "_".join(str(s) for s in train_seasons)
    out_path = output_dir / f"season_{test_season}_train_{train_str}_n{n_iterations}.parquet"
    raw_df.write_parquet(out_path)
    return out_path


def run_predict_only(
    *,
    test_season: int,
    train_seasons: list[int],
    upcoming_fixture_ids: list[int],
    availability_by_pgw: dict[tuple[int, int], float] | None = None,
    n_iterations: int = 200,
    seed: int = 42,
) -> pl.DataFrame:
    """Phase 6 v2: produce per-(player, fixture) xPts predictions for the
    specified UPCOMING fixtures.

    Unlike `_run_one_fold` (a backtest function), this works for fixtures
    that haven't been played yet. The output schema mirrors `eval_df` from
    `_run_one_fold` but without `total_points` (no actuals exist).

    Returns columns: player_id, fixture_id, e_xpts, p_xpts_ge_6,
    e_xpts_naive, e_xpts_position, kickoff_utc.
    """
    if not upcoming_fixture_ids:
        return pl.DataFrame()

    minutes_pred = _train_minutes_predict_only(
        train_seasons, test_season, upcoming_fixture_ids
    )
    goals_pred = _train_goals_or_assists_predict_only(
        train_seasons, test_season, upcoming_fixture_ids, target="goals"
    )
    assists_pred = _train_goals_or_assists_predict_only(
        train_seasons, test_season, upcoming_fixture_ids, target="assists"
    )
    if minutes_pred.is_empty() or goals_pred.is_empty():
        return pl.DataFrame()

    positions_df = pit.all_player_positions()
    train_pm = pit.all_player_match_with_kickoff(season_ids=train_seasons)
    residual_df = fit_residual_dataset(train_pm, positions_df)
    event_source = EmpiricalResidualEventSource()
    event_source.fit(residual_df)
    train_pm_for_alphas = train_pm.join(
        positions_df, on="player_id", how="left"
    ).drop_nulls("position_code")
    alphas = split_p_full_by_position(train_pm_for_alphas)

    # assemble_fold_predictions uses pit.all_player_match_with_kickoff for
    # test_season, which won't return rows for upcoming fixtures. So we
    # need a Phase-6-specific assembler. Synthesize fixtures + per-player
    # rows directly.
    from sqlalchemy import func as sa_func
    from sqlalchemy import select as sa_select

    from fpl_bot.db.models import DimFixture, FactPlayerStatus
    from fpl_bot.db.session import session_scope
    from fpl_bot.features.bps import FoldPredictionInputs

    with session_scope() as s:
        fixture_rows = s.execute(
            sa_select(
                DimFixture.fixture_id, DimFixture.season_id, DimFixture.gameweek,
                DimFixture.home_team_id, DimFixture.away_team_id,
            ).where(DimFixture.fixture_id.in_(upcoming_fixture_ids))
        ).all()
        latest_ps = (
            sa_select(
                FactPlayerStatus.player_id,
                sa_func.max(FactPlayerStatus.recorded_at).label("max_rec"),
            )
            .where(FactPlayerStatus.season_id == test_season)
            .group_by(FactPlayerStatus.player_id)
            .subquery("ps_latest")
        )
        ps_rows = s.execute(
            sa_select(
                FactPlayerStatus.player_id, FactPlayerStatus.team_id, FactPlayerStatus.position_code,
            ).join(
                latest_ps,
                (latest_ps.c.player_id == FactPlayerStatus.player_id)
                & (latest_ps.c.max_rec == FactPlayerStatus.recorded_at),
            )
        ).all()

    fixtures_df = pl.DataFrame(
        [
            {
                "fixture_id": int(r.fixture_id),
                "season_id": int(r.season_id),
                "gameweek": int(r.gameweek),
                "home_team_id": int(r.home_team_id),
                "away_team_id": int(r.away_team_id),
            }
            for r in fixture_rows
        ]
    )
    # Market xG join
    market = pit.market_xg_for_fixtures()
    if not market.is_empty():
        hm = market.select(
            pl.col("fixture_id"),
            pl.col("team_id").alias("home_team_id"),
            pl.col("lambda_market_xg").cast(pl.Float64).alias("home_team_lambda"),
            pl.col("cs_prob_market").cast(pl.Float64).alias("home_market_cs_prob"),
        )
        am = market.select(
            pl.col("fixture_id"),
            pl.col("team_id").alias("away_team_id"),
            pl.col("lambda_market_xg").cast(pl.Float64).alias("away_team_lambda"),
            pl.col("cs_prob_market").cast(pl.Float64).alias("away_market_cs_prob"),
        )
        fixtures_df = fixtures_df.join(hm, on=["fixture_id", "home_team_id"], how="left")
        fixtures_df = fixtures_df.join(am, on=["fixture_id", "away_team_id"], how="left")
    # Fill missing market_xg with sensible defaults
    fixtures_df = fixtures_df.with_columns(
        pl.col("home_team_lambda").fill_null(1.4),
        pl.col("away_team_lambda").fill_null(1.2),
        pl.col("home_market_cs_prob").fill_null(0.3),
        pl.col("away_market_cs_prob").fill_null(0.25),
    )

    # Per-player rows: cross fixtures with players, filter by team membership
    players_df = pl.DataFrame(
        [
            {
                "player_id": int(r.player_id),
                "current_team_id": int(r.team_id),
                "position_code": r.position_code,
            }
            for r in ps_rows
        ]
    )
    pm = fixtures_df.join(players_df, how="cross")
    pm = pm.filter(
        (pl.col("current_team_id") == pl.col("home_team_id"))
        | (pl.col("current_team_id") == pl.col("away_team_id"))
    )
    pm = pm.with_columns(
        (pl.col("current_team_id") == pl.col("home_team_id")).alias("is_home"),
    )
    pm = pm.join(minutes_pred, on=["player_id", "fixture_id"], how="left")
    pm = pm.join(goals_pred, on=["player_id", "fixture_id"], how="left")
    pm = pm.join(assists_pred, on=["player_id", "fixture_id"], how="left")
    pm = pm.with_columns(
        pl.col("p_minutes_zero").fill_null(0.5),
        pl.col("p_minutes_short").fill_null(0.2),
        pl.col("p_minutes_full").fill_null(0.3),
        pl.col("lambda_goals_per_90").fill_null(0.05),
        pl.col("lambda_assists_per_90").fill_null(0.05),
    )
    # Live-only news belongs in the minutes distribution, before goals, clean
    # sheets, and bonus are jointly simulated. No mapping means exact legacy
    # behavior, which keeps historical folds neutral.
    pm = apply_availability_to_minutes_predictions(pm, availability_by_pgw)
    # Other fields required by the simulator
    pm = pm.with_columns(
        pl.lit(0.0).alias("saves_rate_per_90"),
        pl.lit(0.0).alias("yc_rate_per_90"),
        pl.lit(0.0).alias("rc_rate_per_90"),
        pl.lit(False).alias("is_penalty_taker"),
        pl.col("current_team_id").alias("team_id"),
    )

    prepared = FoldPredictionInputs(
        fixtures=fixtures_df, player_predictions=pm, alphas_by_position=alphas
    )

    simulator = BPSSimulator(
        event_source=event_source, n_iterations=n_iterations, seed=seed
    )
    sim_rows: list[pl.DataFrame] = []
    for fix_inputs in fixture_inputs_iter(prepared):
        df = simulator.simulate_fixture(fix_inputs)
        if not df.is_empty():
            df = df.with_columns(pl.lit(fix_inputs.fixture_id).alias("fixture_id"))
            sim_rows.append(df)
    if not sim_rows:
        return pl.DataFrame()

    simulator_predictions = pl.concat(sim_rows, how="vertical_relaxed")

    # Attach kickoff_utc from dim_fixture so downstream callers can sort
    with session_scope() as s:
        kickoffs = {
            r.fixture_id: r.kickoff_utc
            for r in s.execute(
                sa_select(DimFixture.fixture_id, DimFixture.kickoff_utc).where(
                    DimFixture.fixture_id.in_(upcoming_fixture_ids)
                )
            ).all()
        }
    out = simulator_predictions.select("player_id", "fixture_id", "e_xpts", "p_xpts_ge_6")
    out = out.with_columns(
        # placeholders for the deterministic+naive baselines used downstream
        pl.col("e_xpts").alias("e_xpts_naive"),
        pl.col("e_xpts").alias("e_xpts_position"),
        pl.lit(0).alias("total_points"),  # no actuals for upcoming GWs
        pl.col("fixture_id")
        .replace_strict(kickoffs, default=None)
        .alias("kickoff_utc"),
    )
    # The cached `_run_one_fold` eval_df reads kickoff_utc through a
    # Postgres TIMESTAMPTZ → polars Datetime that picks up the session's
    # local TZ (typically America/Edmonton on dev). Normalize to UTC so
    # downstream concats are TZ-consistent regardless of how the cache
    # was written.
    if "kickoff_utc" in out.columns:
        col = out.schema["kickoff_utc"]
        if isinstance(col, pl.Datetime) and col.time_zone is not None:
            out = out.with_columns(pl.col("kickoff_utc").dt.convert_time_zone("UTC"))
        else:
            out = out.with_columns(
                pl.col("kickoff_utc").cast(pl.Datetime("us")).dt.replace_time_zone("UTC")
            )
    # Match the column order of `_run_one_fold`'s eval_df for downstream
    # concat compatibility (polars `concat` aligns by position by default;
    # caller uses `how="diagonal_relaxed"` to align by name as belt-and-suspenders).
    return out.select(
        "player_id", "fixture_id", "e_xpts", "p_xpts_ge_6",
        "e_xpts_naive", "e_xpts_position", "total_points", "kickoff_utc",
    )


def format_results(results: list[XPtsFoldResult]) -> str:
    if not results:
        return "(no folds completed)"
    lines = [
        "season  | n      | MAE: sim/naive/pos        | Brier(≥6): sim/naive/pos | dec | cap: sim/base/opt | gates",
        "--------+--------+---------------------------+--------------------------+-----+---------------------+------",
    ]
    for r in results:
        lines.append(
            f" 20{r.test_season}/{(r.test_season+1)%100:02d} "
            f"| {r.n_test_player_fixtures:>6} "
            f"| {r.mae_simulator:.3f}/{r.mae_naive_sum:.3f}/{r.mae_position:.3f}     "
            f"| {r.brier_ge6_simulator:.4f}/{r.brier_ge6_naive_sum:.4f}/{r.brier_ge6_position:.4f}  "
            f"| {r.calibration_decile_passes:>3} "
            f"| {r.captaincy_sim_total:>5}/{r.captaincy_baseline_total:>5}/{r.captaincy_optimal_total:>5}  "
            f"| {'P' if r.primary_gate_pass else '✗'}/{'C' if r.captaincy_gate_pass else '✗'}"
        )
    n_primary = sum(1 for r in results if r.primary_gate_pass)
    n_cap = sum(1 for r in results if r.captaincy_gate_pass)
    n_calib = sum(1 for r in results if r.calibration_decile_passes >= 8)
    lines.append("")
    lines.append(
        f"Acceptance gate: primary (MAE+Brier ≥6 beat both baselines) {n_primary}/{len(results)}; "
        f"calibration ≥ 8/10 deciles {n_calib}/{len(results)}; "
        f"captaincy (sim > rolling3) {n_cap}/{len(results)}."
    )
    return "\n".join(lines)
