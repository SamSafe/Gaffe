"""Walk-forward CV for the BPS simulator (Phase 2.4).

Per design §4: simulator is REQUIRED (cannot fall back to a heuristic in
production). Acceptance gate validates the simulator works:
  - Brier on P(bonus > 0) beats naive_event_proxy by ≥ 5% relative
  - Brier on P(bonus = 3) beats top3_by_xpts
  - ECE < 0.05 per position
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from fpl_bot.db import pit
from fpl_bot.db.event_source import EmpiricalResidualEventSource
from fpl_bot.features.bps import (
    assemble_fold_predictions,
    baseline_naive_event_proxy,
    baseline_position_marginal,
    baseline_top3_by_xpts,
    fixture_inputs_iter,
)
from fpl_bot.features.goals import (
    build_feature_table as build_goals_feature_table,
)
from fpl_bot.features.goals import (
    build_prediction_feature_table as build_goals_prediction_table,
)
from fpl_bot.features.minutes import (
    build_feature_table as build_minutes_feature_table,
)
from fpl_bot.features.minutes import (
    build_prediction_feature_table as build_minutes_prediction_table,
)
from fpl_bot.models.bps import (
    AssistSamplingMode,
    BPSRulesMode,
    BPSSimulator,
    fit_residual_dataset,
    split_p_full_by_position,
)
from fpl_bot.models.goals import LABEL_COLUMNS as GOALS_LABEL_COLUMNS
from fpl_bot.models.goals import train_per_90_model
from fpl_bot.models.minutes import train_minutes_model

WALK_FORWARD_FOLDS: list[dict] = [
    {"train": [19, 20], "test": 21},
    {"train": [19, 20, 21], "test": 22},
    {"train": [19, 20, 21, 22], "test": 23},
    {"train": [19, 20, 21, 22, 23], "test": 24},
]

GATE_PRIMARY_THRESHOLD_REL = 0.05  # 5% relative Brier improvement on P(bonus>0)


@dataclass
class BPSFoldResult:
    test_season: int
    n_test_player_fixtures: int
    # Brier on P(bonus > 0)
    brier_pos_simulator: float
    brier_pos_naive: float
    brier_pos_top3: float
    brier_pos_position: float
    # Brier on P(bonus = 3)
    brier_3_simulator: float
    brier_3_top3: float
    # Calibration
    ece_simulator_overall: float
    ece_simulator_by_position: dict[str, float]
    # Gate signals
    primary_rel_improvement: float  # vs naive_event_proxy on P(bonus>0)
    gate_primary_pass: bool
    gate_secondary_pass: bool


def _brier(y_true: np.ndarray, p_pred: np.ndarray) -> float:
    return float(np.mean((p_pred - y_true.astype(float)) ** 2))


def _ece(y_true: np.ndarray, p_pred: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0, 1, n_bins + 1)
    n = len(p_pred)
    if n == 0:
        return 0.0
    ece = 0.0
    for b in range(n_bins):
        mask = (p_pred >= bins[b]) & (p_pred < bins[b + 1])
        if b == n_bins - 1:
            mask = mask | (p_pred == 1.0)
        if mask.sum() == 0:
            continue
        avg_pred = float(p_pred[mask].mean())
        avg_actual = float(y_true[mask].mean())
        ece += (mask.sum() / n) * abs(avg_pred - avg_actual)
    return float(ece)


def _train_minutes_predictor(
    train_seasons: list[int],
    test_season: int,
    *,
    train_through_gw: int | None = None,
) -> pl.DataFrame:
    """Train minutes predictor and predict on test_season fixtures.

    If `train_through_gw` is set, additionally includes test_season's
    GWs 1..train_through_gw in the training set. Predictions are still
    generated for ALL of test_season (the caller filters as needed).
    Used by the walk-forward harness to PIT-correctly retrain at chunk
    boundaries during the season.
    """
    df = build_minutes_feature_table(season_ids=[*train_seasons, test_season])
    df = df.filter(pl.col("min_last_1").is_not_null())
    train_mask = pl.col("season_id").is_in(train_seasons)
    if train_through_gw is not None:
        train_mask = train_mask | (
            (pl.col("season_id") == test_season)
            & (pl.col("gameweek") <= train_through_gw)
        )
    train = df.filter(train_mask).filter(pl.col("minutes_bucket").is_not_null())
    test = df.filter(pl.col("season_id") == test_season)
    if train.is_empty() or test.is_empty():
        return pl.DataFrame()
    model = train_minutes_model(train, valid=test)
    probs = model.predict_proba(test)
    return test.select("player_id", "fixture_id").with_columns(
        pl.Series("p_minutes_zero", probs[:, 0]),
        pl.Series("p_minutes_short", probs[:, 1]),
        pl.Series("p_minutes_full", probs[:, 2]),
    )


def _train_goals_or_assists_predictor(
    train_seasons: list[int],
    test_season: int,
    *,
    target: str,
    train_through_gw: int | None = None,
) -> pl.DataFrame:
    """Train goals/assists predictor; see `_train_minutes_predictor` for
    the `train_through_gw` semantics."""
    df = build_goals_feature_table(season_ids=[*train_seasons, test_season])
    label_col = GOALS_LABEL_COLUMNS[target]
    df = df.filter(pl.col(label_col).is_not_null()).filter(
        pl.col("minutes_factor").is_not_null()
    )
    train_mask = pl.col("season_id").is_in(train_seasons)
    if train_through_gw is not None:
        train_mask = train_mask | (
            (pl.col("season_id") == test_season)
            & (pl.col("gameweek") <= train_through_gw)
        )
    train = df.filter(train_mask)
    test = df.filter(pl.col("season_id") == test_season)
    if train.is_empty() or test.is_empty():
        return pl.DataFrame()
    model = train_per_90_model(train, target=target, valid=test, weighted=True)
    rate = model.predict_per_90(test)
    out_col = f"lambda_{target}_per_90"
    return test.select("player_id", "fixture_id").with_columns(
        pl.Series(out_col, rate)
    )


# ─── Phase 6 v2: predict-only variants ────────────────────────────────────


def _train_minutes_predict_only(
    train_seasons: list[int],
    test_season: int,
    upcoming_fixture_ids: list[int],
) -> pl.DataFrame:
    """Train minutes model on train_seasons, predict on upcoming fixtures.

    Unlike _train_minutes_predictor, the test set is synthesized from
    upcoming-fixture metadata (no played-fixture data needed).
    """
    train_df = build_minutes_feature_table(season_ids=train_seasons)
    train_df = train_df.filter(pl.col("min_last_1").is_not_null()).filter(
        pl.col("minutes_bucket").is_not_null()
    )
    test_df = build_minutes_prediction_table(
        test_season=test_season, upcoming_fixture_ids=upcoming_fixture_ids
    )
    if train_df.is_empty() or test_df.is_empty():
        return pl.DataFrame()
    model = train_minutes_model(train_df, valid=train_df)
    probs = model.predict_proba(test_df)
    return test_df.select("player_id", "fixture_id").with_columns(
        pl.Series("p_minutes_zero", probs[:, 0]),
        pl.Series("p_minutes_short", probs[:, 1]),
        pl.Series("p_minutes_full", probs[:, 2]),
    )


def _train_goals_or_assists_predict_only(
    train_seasons: list[int],
    test_season: int,
    upcoming_fixture_ids: list[int],
    *,
    target: str,
) -> pl.DataFrame:
    """Train goals/assists model on train_seasons, predict on upcoming fixtures."""
    train_df = build_goals_feature_table(season_ids=train_seasons)
    label_col = GOALS_LABEL_COLUMNS[target]
    train_df = train_df.filter(pl.col(label_col).is_not_null()).filter(
        pl.col("minutes_factor").is_not_null()
    )
    test_df = build_goals_prediction_table(
        test_season=test_season, upcoming_fixture_ids=upcoming_fixture_ids
    )
    if train_df.is_empty() or test_df.is_empty():
        return pl.DataFrame()
    model = train_per_90_model(train_df, target=target, valid=train_df, weighted=True)
    rate = model.predict_per_90(test_df)
    out_col = f"lambda_{target}_per_90"
    return test_df.select("player_id", "fixture_id").with_columns(
        pl.Series(out_col, rate)
    )


def run_walk_forward_cv(
    folds: list[dict] | None = None,
    *,
    n_iterations: int = 200,
    seed: int = 42,
    bps_rules_mode: BPSRulesMode = "official",
    assist_sampling_mode: AssistSamplingMode = "independent",
) -> list[BPSFoldResult]:
    folds_to_use = folds or WALK_FORWARD_FOLDS
    results: list[BPSFoldResult] = []

    positions_df = pit.all_player_positions()

    for fold in folds_to_use:
        train_seasons = fold["train"]
        test_season = fold["test"]

        # 1. Train trio + predict on test fold
        minutes_pred = _train_minutes_predictor(train_seasons, test_season)
        goals_pred = _train_goals_or_assists_predictor(
            train_seasons, test_season, target="goals"
        )
        assists_pred = _train_goals_or_assists_predictor(
            train_seasons, test_season, target="assists"
        )
        if minutes_pred.is_empty() or goals_pred.is_empty():
            continue

        # 2. Fit residual on train fold using actual events
        train_pm = pit.all_player_match_with_kickoff(season_ids=train_seasons)
        residual_df = fit_residual_dataset(
            train_pm, positions_df, rules_mode=bps_rules_mode
        )
        event_source = EmpiricalResidualEventSource(
            integer_bps=bps_rules_mode == "official"
        )
        event_source.fit(residual_df)

        # 3. Compute alpha = P(60-89 | 60+) per position from train data
        train_pm_for_alphas = train_pm.join(
            positions_df, on="player_id", how="left"
        ).drop_nulls("position_code")
        alphas = split_p_full_by_position(train_pm_for_alphas)

        # 4. Assemble fold prediction inputs
        prepared = assemble_fold_predictions(
            test_season=test_season,
            minutes_predictions=minutes_pred,
            goals_predictions=goals_pred,
            assists_predictions=assists_pred,
            alphas_by_position=alphas,
        )
        if prepared.fixtures.is_empty() or prepared.player_predictions.is_empty():
            continue

        # 5. Run simulator per fixture
        simulator = BPSSimulator(
            event_source=event_source,
            n_iterations=n_iterations,
            seed=seed,
            bps_rules_mode=bps_rules_mode,
            assist_sampling_mode=assist_sampling_mode,
        )
        sim_rows: list[pl.DataFrame] = []
        for fix_inputs in fixture_inputs_iter(prepared):
            sim = simulator.simulate_fixture(fix_inputs)
            if not sim.is_empty():
                sim = sim.with_columns(pl.lit(fix_inputs.fixture_id).alias("fixture_id"))
                sim_rows.append(sim)
        if not sim_rows:
            continue
        simulator_predictions = pl.concat(sim_rows, how="vertical_relaxed")

        # 6. Baselines
        naive = baseline_naive_event_proxy(
            prepared.player_predictions, prepared.fixtures
        )
        top3 = baseline_top3_by_xpts(
            prepared.player_predictions, prepared.fixtures
        )
        position_baseline_train = (
            train_pm.join(positions_df, on="player_id", how="left").drop_nulls(
                "position_code"
            )
        )
        position_baseline = baseline_position_marginal(
            position_baseline_train, prepared.player_predictions
        )

        # 7. Actuals from fact_player_match
        test_pm = pit.all_player_match_with_kickoff(season_ids=[test_season])
        actual = test_pm.select("player_id", "fixture_id", "bonus", "minutes").rename(
            {"bonus": "actual_bonus"}
        )

        # 8. Join all predictions to actuals
        eval_df = (
            simulator_predictions.rename({
                "p_bonus_0": "sim_p0",
                "p_bonus_1": "sim_p1",
                "p_bonus_2": "sim_p2",
                "p_bonus_3": "sim_p3",
            })
            .join(naive.select(
                "player_id", "fixture_id",
                pl.col("p_bonus_0").alias("naive_p0"),
                pl.col("p_bonus_1").alias("naive_p1"),
                pl.col("p_bonus_2").alias("naive_p2"),
                pl.col("p_bonus_3").alias("naive_p3"),
            ), on=["player_id", "fixture_id"], how="left")
            .join(top3.select(
                "player_id", "fixture_id",
                pl.col("p_bonus_0").alias("top3_p0"),
                pl.col("p_bonus_3").alias("top3_p3"),
            ), on=["player_id", "fixture_id"], how="left")
            .join(position_baseline.select(
                "player_id", "fixture_id",
                pl.col("p_bonus_0").alias("posb_p0"),
            ), on=["player_id", "fixture_id"], how="left")
            .join(actual, on=["player_id", "fixture_id"], how="inner")
            .drop_nulls("actual_bonus")
        )

        # 9. Compute metrics
        y = eval_df.select("actual_bonus").to_pandas().to_numpy().ravel().astype(int)
        y_pos = (y > 0).astype(int)
        y_3 = (y == 3).astype(int)

        sim_p_pos = (1.0 - eval_df["sim_p0"]).to_numpy()
        naive_p_pos = (1.0 - eval_df["naive_p0"].fill_null(1.0)).to_numpy()
        top3_p_pos = (1.0 - eval_df["top3_p0"].fill_null(1.0)).to_numpy()
        posb_p_pos = (1.0 - eval_df["posb_p0"].fill_null(1.0)).to_numpy()

        sim_p_3 = eval_df["sim_p3"].to_numpy()
        top3_p_3 = eval_df["top3_p3"].fill_null(0.0).to_numpy()

        brier_pos_sim = _brier(y_pos, sim_p_pos)
        brier_pos_naive = _brier(y_pos, naive_p_pos)
        brier_pos_top3 = _brier(y_pos, top3_p_pos)
        brier_pos_position = _brier(y_pos, posb_p_pos)
        brier_3_sim = _brier(y_3, sim_p_3)
        brier_3_top3 = _brier(y_3, top3_p_3)

        # ECE per position on P(bonus>0)
        positions_test = (
            eval_df.join(positions_df, on="player_id", how="left")
            .with_columns(pl.col("position_code").fill_null("MID"))
        )["position_code"].to_list()
        ece_overall = _ece(y_pos, sim_p_pos)
        ece_by_pos: dict[str, float] = {}
        for pos in ("GKP", "DEF", "MID", "FWD"):
            mask = np.array([p == pos for p in positions_test])
            if mask.sum() == 0:
                continue
            ece_by_pos[pos] = _ece(y_pos[mask], sim_p_pos[mask])

        rel_imp = (brier_pos_naive - brier_pos_sim) / max(brier_pos_naive, 1e-9)
        gate_primary_pass = rel_imp >= GATE_PRIMARY_THRESHOLD_REL
        gate_secondary_pass = brier_3_sim < brier_3_top3

        results.append(
            BPSFoldResult(
                test_season=test_season,
                n_test_player_fixtures=len(eval_df),
                brier_pos_simulator=brier_pos_sim,
                brier_pos_naive=brier_pos_naive,
                brier_pos_top3=brier_pos_top3,
                brier_pos_position=brier_pos_position,
                brier_3_simulator=brier_3_sim,
                brier_3_top3=brier_3_top3,
                ece_simulator_overall=ece_overall,
                ece_simulator_by_position=ece_by_pos,
                primary_rel_improvement=rel_imp,
                gate_primary_pass=gate_primary_pass,
                gate_secondary_pass=gate_secondary_pass,
            )
        )

    return results


def format_results(results: list[BPSFoldResult]) -> str:
    if not results:
        return "(no folds completed)"
    lines = [
        "season  | n      | brier_pos: sim/naive/top3/posM | brier_3: sim/top3 | ece_overall | gate_p | gate_s",
        "--------+--------+--------------------------------+-------------------+-------------+--------+-------",
    ]
    for r in results:
        lines.append(
            f" 20{r.test_season}/{(r.test_season+1)%100:02d} "
            f"| {r.n_test_player_fixtures:>6} "
            f"| {r.brier_pos_simulator:.4f}/{r.brier_pos_naive:.4f}/{r.brier_pos_top3:.4f}/{r.brier_pos_position:.4f} "
            f"| {r.brier_3_simulator:.4f}/{r.brier_3_top3:.4f}    "
            f"| {r.ece_simulator_overall:.3f}       "
            f"| {'✓' if r.gate_primary_pass else '✗'}      "
            f"| {'✓' if r.gate_secondary_pass else '✗'}"
        )
    n_primary = sum(1 for r in results if r.gate_primary_pass)
    n_secondary = sum(1 for r in results if r.gate_secondary_pass)
    lines.append("")
    lines.append(
        f"Acceptance gate: primary (sim beats naive by ≥ 5% on P(bonus>0)) "
        f"passed {n_primary}/{len(results)} folds; "
        f"secondary (sim beats top3 on P(bonus=3)) passed {n_secondary}/{len(results)}."
    )
    return "\n".join(lines)
