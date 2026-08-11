"""Feature/prediction assembly for the BPS simulator (Phase 2.4).

Given trained Phase 2.1-2.3 models and a target test fold, build the
per-fixture FixtureInputs structures the BPS simulator consumes. PIT-routed:
imports only the public APIs from features/manual_overrides + db.pit, and
the trained models themselves.

v1 Tier-B rates (saves, cards) use position-level priors rather than
per-player rolling rates — a v1.1 enhancement candidate (the empirical
residual absorbs per-player deviation in v1).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from fpl_bot.db import pit
from fpl_bot.features import manual_overrides
from fpl_bot.models.bps import FixtureInputs

# v1 league-average Tier-B rates per 90 minutes
SAVES_PER_90_GK = 3.0  # roughly 2.7 average; round to 3 for simplicity
YC_RATE_PER_90 = 0.13
RC_RATE_PER_90 = 0.005


@dataclass
class FoldPredictionInputs:
    """All the per-(player, fixture) predictions needed by the BPS simulator."""

    fixtures: pl.DataFrame  # one row per fixture: fixture_id, season_id, gw, home/away_team, lambdas, cs_prob
    player_predictions: pl.DataFrame  # player_id, fixture_id, position_code, is_home, p_zero, p_short, p_full, lambda_g_per_90, lambda_a_per_90, is_penalty_taker
    alphas_by_position: dict[str, float]


def assemble_fold_predictions(
    *,
    test_season: int,
    minutes_predictions: pl.DataFrame,
    goals_predictions: pl.DataFrame,
    assists_predictions: pl.DataFrame,
    alphas_by_position: dict[str, float],
) -> FoldPredictionInputs:
    """Joins the trio of model predictions for a test season with market xG +
    positions + penalty taker flags. Returns the structured input the BPS
    simulator consumes per fixture.

    Inputs are model output DataFrames with the following columns:
      minutes_predictions: player_id, fixture_id, p_minutes_zero, p_minutes_short, p_minutes_full
      goals_predictions:   player_id, fixture_id, lambda_goals_per_90
      assists_predictions: player_id, fixture_id, lambda_assists_per_90
    """
    # Fixtures and team lambdas
    fixtures_raw = pit.all_player_match_with_kickoff(season_ids=[test_season])
    if fixtures_raw.is_empty():
        return FoldPredictionInputs(
            fixtures=pl.DataFrame(),
            player_predictions=pl.DataFrame(),
            alphas_by_position=alphas_by_position,
        )

    fixtures = (
        fixtures_raw.select(
            "fixture_id", "season_id", "gameweek", "home_team_id", "away_team_id"
        )
        .unique()
        .sort("fixture_id")
    )

    market = pit.market_xg_for_fixtures()
    if market.is_empty():
        # Fall back: zero-info lambdas; simulator will produce mostly residual-driven output
        fixtures = fixtures.with_columns(
            pl.lit(1.4).alias("home_team_lambda"),
            pl.lit(1.2).alias("away_team_lambda"),
            pl.lit(0.3).alias("home_market_cs_prob"),
            pl.lit(0.25).alias("away_market_cs_prob"),
        )
    else:
        home_market = market.select(
            pl.col("fixture_id"),
            pl.col("team_id").alias("home_team_id"),
            pl.col("lambda_market_xg").alias("home_team_lambda"),
            pl.col("cs_prob_market").alias("home_market_cs_prob"),
        )
        away_market = market.select(
            pl.col("fixture_id"),
            pl.col("team_id").alias("away_team_id"),
            pl.col("lambda_market_xg").alias("away_team_lambda"),
            pl.col("cs_prob_market").alias("away_market_cs_prob"),
        )
        fixtures = fixtures.join(
            home_market, on=["fixture_id", "home_team_id"], how="left"
        ).join(away_market, on=["fixture_id", "away_team_id"], how="left")

    # Cast Decimal/Numeric columns to float to avoid downstream type issues
    for c in (
        "home_team_lambda",
        "away_team_lambda",
        "home_market_cs_prob",
        "away_market_cs_prob",
    ):
        if c in fixtures.columns:
            fixtures = fixtures.with_columns(pl.col(c).cast(pl.Float64))

    # Per-player frame: keep one row per (player_id, fixture_id), retain is_home
    pm = fixtures_raw.select(
        "player_id", "fixture_id", "was_home", "home_team_id", "away_team_id"
    ).rename({"was_home": "is_home"})
    pm = pm.unique(subset=["player_id", "fixture_id"])

    # Positions (current snapshot — same approximation as Phase 2.1/2.2)
    positions = pit.all_player_positions()
    if positions.is_empty():
        pm = pm.with_columns(pl.lit("MID").alias("position_code"))
    else:
        pm = pm.join(positions, on="player_id", how="left").with_columns(
            pl.col("position_code").fill_null("MID")
        )

    # Predictions joins
    pm = pm.join(minutes_predictions, on=["player_id", "fixture_id"], how="left")
    pm = pm.join(goals_predictions, on=["player_id", "fixture_id"], how="left")
    pm = pm.join(assists_predictions, on=["player_id", "fixture_id"], how="left")

    # Fill missing predictions with conservative defaults (e.g. position-mean)
    pm = pm.with_columns(
        pl.col("p_minutes_zero").fill_null(0.5),
        pl.col("p_minutes_short").fill_null(0.2),
        pl.col("p_minutes_full").fill_null(0.3),
        pl.col("lambda_goals_per_90").fill_null(0.05),
        pl.col("lambda_assists_per_90").fill_null(0.05),
    )

    # Penalty taker flags
    pen_takers = _resolved_pk_taker_set(test_season)
    pm = pm.with_columns(
        pl.col("player_id").is_in(list(pen_takers)).cast(pl.Int8).alias("is_penalty_taker")
    )

    # v1 Tier-B rates: position-level constants
    pm = pm.with_columns(
        pl.when(pl.col("position_code") == "GKP")
        .then(SAVES_PER_90_GK)
        .otherwise(0.0)
        .alias("saves_rate_per_90"),
        pl.lit(YC_RATE_PER_90).alias("yc_rate_per_90"),
        pl.lit(RC_RATE_PER_90).alias("rc_rate_per_90"),
    )

    return FoldPredictionInputs(
        fixtures=fixtures,
        player_predictions=pm,
        alphas_by_position=alphas_by_position,
    )


def _resolved_pk_taker_set(season_id: int) -> set[int]:
    """Returns the set of stable player_ids designated as primary PK takers
    for the given season across all teams, per configs/set_piece_takers.yaml."""
    raw = manual_overrides.set_piece_takers_raw().get(season_id, {})
    teams = pit.team_id_by_full_name([season_id])
    scoped_players = pit.player_id_by_season_team_web_name([season_id])
    pids: set[int] = set()
    for team_full_name, team_info in raw.items():
        if not isinstance(team_info, dict):
            continue
        taker = team_info.get("penalty")
        if not taker:
            continue
        team_id = teams.get((season_id, team_full_name))
        if team_id is None:
            continue
        pid = scoped_players.get((season_id, team_id, taker))
        if pid is not None:
            pids.add(pid)
    return pids


def fixture_inputs_iter(prepared: FoldPredictionInputs):
    """Generator yielding one FixtureInputs per fixture in the test fold."""
    if prepared.fixtures.is_empty() or prepared.player_predictions.is_empty():
        return
    for f_row in prepared.fixtures.iter_rows(named=True):
        fid = f_row["fixture_id"]
        players_for_fixture = prepared.player_predictions.filter(
            pl.col("fixture_id") == fid
        )
        if players_for_fixture.is_empty():
            continue
        yield FixtureInputs(
            fixture_id=fid,
            season_id=f_row["season_id"],
            gameweek=f_row["gameweek"],
            home_team_id=f_row["home_team_id"],
            away_team_id=f_row["away_team_id"],
            home_team_lambda=float(f_row.get("home_team_lambda") or 1.4),
            away_team_lambda=float(f_row.get("away_team_lambda") or 1.2),
            home_market_cs_prob=float(f_row.get("home_market_cs_prob") or 0.3),
            away_market_cs_prob=float(f_row.get("away_market_cs_prob") or 0.25),
            players=players_for_fixture,
            alphas_by_position=prepared.alphas_by_position,
        )


# ── Baselines for evaluation ──────────────────────────────────────────────────


def baseline_naive_event_proxy(
    player_predictions: pl.DataFrame,
    fixtures: pl.DataFrame,
    *,
    n_bonus_per_fixture: int = 3,
) -> pl.DataFrame:
    """Predict bonus from a linear combination of expected goals/assists/CS.

    Computes a 'bonus_proxy_score' per player; assigns top-3 in each fixture
    to bonus tiers 3/2/1, with the rest at 0. Returns one row per
    (player_id, fixture_id) with p_bonus_0..3 (single-realization assignment
    treated as deterministic).
    """
    GOAL_BPS_AVG = 18.0  # MID-weighted average across positions
    ASSIST_BPS = 9.0
    CS_BPS = 12.0

    # Expected minutes ~ p_zero*0 + p_short*30 + p_full*75 (mid of 60-90)
    df = player_predictions.with_columns(
        (
            pl.col("p_minutes_short") * 30 + pl.col("p_minutes_full") * 75
        ).alias("exp_minutes_30plus"),
    )

    df = df.with_columns(
        (
            pl.col("lambda_goals_per_90") * pl.col("exp_minutes_30plus") / 90.0
        ).alias("exp_goals"),
        (
            pl.col("lambda_assists_per_90") * pl.col("exp_minutes_30plus") / 90.0
        ).alias("exp_assists"),
    )

    # CS expected from the fixture's home/away prob
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

    # Bonus proxy: weight each contribution by typical BPS value
    df = df.with_columns(
        (
            pl.col("exp_goals") * GOAL_BPS_AVG
            + pl.col("exp_assists") * ASSIST_BPS
            + pl.col("team_cs_prob") * CS_BPS * (
                pl.col("position_code").is_in(["GKP", "DEF"]).cast(pl.Float64)
            )
        ).alias("bonus_proxy_score")
    )

    return _assign_top_n_bonus(df, score_col="bonus_proxy_score", n=n_bonus_per_fixture)


def baseline_top3_by_xpts(
    player_predictions: pl.DataFrame,
    fixtures: pl.DataFrame,
    *,
    n_bonus_per_fixture: int = 3,
) -> pl.DataFrame:
    """Top-3-by-xPts heuristic: rank by an FPL-points-style proxy, top 3 get
    bonus 3/2/1.

    Approximation: 4*P(60+) + 4*goals_pts + 3*assists_pts + 4*P(CS for GK/DEF).
    """
    df = player_predictions.with_columns(
        pl.when(pl.col("position_code") == "FWD")
        .then(4)
        .when(pl.col("position_code") == "MID")
        .then(5)
        .when(pl.col("position_code") == "DEF")
        .then(6)
        .otherwise(6)
        .alias("goal_pts_per_goal"),
    )

    df = df.with_columns(
        (pl.col("p_minutes_full") * 2.0).alias("appearance_pts"),
        (
            pl.col("lambda_goals_per_90") * (pl.col("p_minutes_full") * 75 + pl.col("p_minutes_short") * 30) / 90.0 * pl.col("goal_pts_per_goal")
        ).alias("goals_xpts"),
        (
            pl.col("lambda_assists_per_90") * (pl.col("p_minutes_full") * 75 + pl.col("p_minutes_short") * 30) / 90.0 * 3.0
        ).alias("assists_xpts"),
    )

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
        (
            pl.col("appearance_pts")
            + pl.col("goals_xpts")
            + pl.col("assists_xpts")
            + pl.col("team_cs_prob") * 4.0 * pl.col("position_code").is_in(["GKP", "DEF"]).cast(pl.Float64)
        ).alias("xpts_score")
    )

    return _assign_top_n_bonus(df, score_col="xpts_score", n=n_bonus_per_fixture)


def _assign_top_n_bonus(
    df: pl.DataFrame, *, score_col: str, n: int = 3
) -> pl.DataFrame:
    """For each fixture, rank players by score_col descending; top 3 get
    deterministic bonus 3/2/1. Returns columns: player_id, fixture_id,
    p_bonus_0..p_bonus_3 ∈ {0, 1}, expected_bonus."""
    rows = []
    for fid, group in df.group_by("fixture_id"):
        sub = group.sort(score_col, descending=True, nulls_last=True)
        for rank, row in enumerate(sub.iter_rows(named=True)):
            bonus = 0
            if rank == 0:
                bonus = 3
            elif rank == 1:
                bonus = 2
            elif rank == 2:
                bonus = 1
            rows.append(
                {
                    "player_id": row["player_id"],
                    "fixture_id": int(fid[0]) if isinstance(fid, tuple) else int(fid),
                    "p_bonus_0": 1.0 if bonus == 0 else 0.0,
                    "p_bonus_1": 1.0 if bonus == 1 else 0.0,
                    "p_bonus_2": 1.0 if bonus == 2 else 0.0,
                    "p_bonus_3": 1.0 if bonus == 3 else 0.0,
                    "expected_bonus": float(bonus),
                }
            )
    return pl.DataFrame(rows)


def baseline_position_marginal(
    train_actuals: pl.DataFrame,
    test_player_predictions: pl.DataFrame,
) -> pl.DataFrame:
    """Floor baseline: predict per-(position, minutes-bucket) historical bonus
    distribution. train_actuals must have position_code, minutes, bonus.
    Returns one row per test (player_id, fixture_id) with predicted p_bonus_*.
    """
    # Compute marginal P(bonus = k | position, minutes_bucket >= 60)
    train = train_actuals.filter(pl.col("minutes") >= 60)
    pos_dist: dict[str, np.ndarray] = {}
    for pos in ("GKP", "DEF", "MID", "FWD"):
        sub = train.filter(pl.col("position_code") == pos).select("bonus")
        if sub.is_empty():
            pos_dist[pos] = np.array([0.85, 0.08, 0.05, 0.02])
            continue
        b = sub.to_pandas().to_numpy().ravel().astype(int)
        counts = np.bincount(b, minlength=4)[:4].astype(float)
        if counts.sum() == 0:
            pos_dist[pos] = np.array([0.85, 0.08, 0.05, 0.02])
        else:
            pos_dist[pos] = counts / counts.sum()

    # P(any bonus | minutes < 60) is much lower; approximate as a 5x dilution
    # vs the 60+ rate (most cameos don't get bonus).
    rows = []
    for row in test_player_predictions.iter_rows(named=True):
        pos = row.get("position_code") or "MID"
        p_full = float(row.get("p_minutes_full") or 0.0)
        p_short = float(row.get("p_minutes_short") or 0.0)
        # Mix: bonus dist when 60+ played, else mostly p_bonus_0
        full_dist = pos_dist[pos]
        short_dist = np.array([0.97, 0.02, 0.01, 0.00])
        zero_dist = np.array([1.0, 0.0, 0.0, 0.0])
        p_zero = max(0.0, 1.0 - p_full - p_short)
        mixed = p_full * full_dist + p_short * short_dist + p_zero * zero_dist
        if mixed.sum() > 0:
            mixed = mixed / mixed.sum()
        rows.append(
            {
                "player_id": row["player_id"],
                "fixture_id": row["fixture_id"],
                "p_bonus_0": float(mixed[0]),
                "p_bonus_1": float(mixed[1]),
                "p_bonus_2": float(mixed[2]),
                "p_bonus_3": float(mixed[3]),
                "expected_bonus": float(np.dot(mixed, [0, 1, 2, 3])),
            }
        )
    return pl.DataFrame(rows)
