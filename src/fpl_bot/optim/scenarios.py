"""Phase 4 SAA: load per-iteration raw FPL points and build per-scenario
(player, gameweek) point dictionaries for the MILP objective.

Raw samples format (from `eval/xpts_eval.run_fold_with_raw_samples`):
    columns: player_id (i64), fixture_id (i32), iteration (i64), xpts (i64)

For SAA we need:
    pts[(player_id, gameweek, scenario_id)] → float

Aggregation rules:
- DGW handling: sum xpts across fixture_ids that share the same gameweek.
- Scenario subsampling: take the first `n_scenarios` of the available iterations
  (deterministic; iteration 0..n-1). v1 = 50 scenarios from the 200-iter dump.
"""
from __future__ import annotations

from pathlib import Path

import polars as pl
from sqlalchemy import select

from fpl_bot.db.models import DimFixture
from fpl_bot.db.session import session_scope


def cache_path_for_fold(
    test_season: int,
    train_seasons: list[int],
    n_iterations: int = 200,
    cache_dir: Path = Path("data/cache/xpts_raw_samples"),
) -> Path:
    train_str = "_".join(str(s) for s in train_seasons)
    return cache_dir / f"season_{test_season}_train_{train_str}_n{n_iterations}.parquet"


def load_raw_samples(
    test_season: int,
    train_seasons: list[int],
    n_iterations: int = 200,
    cache_dir: Path = Path("data/cache/xpts_raw_samples"),
) -> pl.DataFrame:
    """Load the raw-samples DataFrame for a fold. Adds a `gameweek` column via
    join with dim_fixture.
    """
    path = cache_path_for_fold(test_season, train_seasons, n_iterations, cache_dir)
    if not path.exists():
        raise FileNotFoundError(
            f"Raw samples not cached for fold (season={test_season}, "
            f"train={train_seasons}) at {path}. Run "
            f"scripts/phase4_dump_raw_samples.py first."
        )
    raw = pl.read_parquet(path)
    # Attach gameweek
    fids = raw["fixture_id"].unique().to_list()
    with session_scope() as s:
        rows = s.execute(
            select(DimFixture.fixture_id, DimFixture.gameweek).where(
                DimFixture.fixture_id.in_(fids)
            )
        ).all()
    fid_to_gw = {r.fixture_id: r.gameweek for r in rows}
    raw = raw.with_columns(
        pl.col("fixture_id").replace_strict(fid_to_gw, default=0).alias("gameweek")
    )
    return raw


def aggregate_pts_by_player_gw_scenario(
    raw: pl.DataFrame,
    n_scenarios: int,
) -> pl.DataFrame:
    """Aggregate raw (player, fixture, iteration) → (player, gameweek, scenario).

    DGW: sum xpts across fixtures in the same (player, gameweek, iteration).
    Subsample scenarios: take the first n_scenarios iterations (deterministic).
    """
    if n_scenarios <= 0:
        raise ValueError("n_scenarios must be positive")
    # Subsample first n_scenarios iterations (deterministic; matches the
    # simulator's iteration seeding via BPSSimulator(seed=42)).
    sub = raw.filter(pl.col("iteration") < n_scenarios)
    # Aggregate to (player_id, gameweek, iteration)
    agg = sub.group_by(["player_id", "gameweek", "iteration"]).agg(
        pl.col("xpts").sum().alias("xpts")
    )
    return agg.rename({"iteration": "scenario_id"})


def captain_lower_quantile_per_gw(
    pts_df: pl.DataFrame,
    quantile: float = 0.25,
) -> dict[tuple[int, int], float]:
    """Compute the lower-quantile xPts per (player_id, gameweek) across
    scenarios. Used as the captain reward signal to bias the MILP toward
    minutes-reliable, low-downside captains rather than high-mean / high-
    variance "boom or bust" picks.

    `pts_df` is the output of `aggregate_pts_by_player_gw_scenario`:
        columns: player_id, gameweek, scenario_id, xpts

    Q25 by default — high enough to ignore freak 0-min outliers, low enough
    to penalize variance. Returns a per-(player, gw) dict.
    """
    if not (0.0 < quantile < 1.0):
        raise ValueError(f"quantile must be in (0,1); got {quantile}")
    q = (
        pts_df.group_by(["player_id", "gameweek"])
        .agg(pl.col("xpts").quantile(quantile).alias("q_xpts"))
    )
    return {
        (int(r["player_id"]), int(r["gameweek"])): float(r["q_xpts"])
        for r in q.iter_rows(named=True)
    }


def captain_haul_score_per_gw(
    pts_df: pl.DataFrame,
    *,
    threshold: float = 6.0,
    weight: float = 1.0,
) -> dict[tuple[int, int], float]:
    """Compute a captain-specific score that rewards upside in simulations.

    Mean xPts is still the base score. The haul term adds
    `weight * E[max(xPts - threshold, 0)]`, so the value stays in points units
    while favoring players whose simulated distribution contains capturable
    ceiling outcomes.
    """
    if threshold < 0:
        raise ValueError(f"threshold must be non-negative; got {threshold}")
    if weight < 0:
        raise ValueError(f"weight must be non-negative; got {weight}")

    score = (
        pts_df.group_by(["player_id", "gameweek"])
        .agg(
            pl.col("xpts").mean().alias("mean_xpts"),
            pl.when(pl.col("xpts") > threshold)
            .then(pl.col("xpts") - threshold)
            .otherwise(0.0)
            .mean()
            .alias("mean_excess_haul"),
        )
        .with_columns(
            (
                pl.col("mean_xpts")
                + float(weight) * pl.col("mean_excess_haul")
            ).alias("captain_score")
        )
    )
    return {
        (int(r["player_id"]), int(r["gameweek"])): float(r["captain_score"])
        for r in score.iter_rows(named=True)
    }


def make_pts_dict(
    pts_df: pl.DataFrame,
    candidates: list[int],
    horizon_gws: list[int],
    n_scenarios: int,
) -> dict[tuple[int, int, int], float]:
    """Build the (player_id, gw, scenario_id) → pts dict scoped to candidates
    and horizon. Missing entries default to 0.0 (player has no fixture that GW,
    e.g., BGW)."""
    candidate_set = set(candidates)
    gw_set = set(horizon_gws)
    scenario_ids = list(range(n_scenarios))

    df = pts_df.filter(
        pl.col("player_id").is_in(candidate_set)
        & pl.col("gameweek").is_in(gw_set)
        & pl.col("scenario_id").is_in(scenario_ids)
    )
    out: dict[tuple[int, int, int], float] = {}
    for r in df.iter_rows(named=True):
        out[(int(r["player_id"]), int(r["gameweek"]), int(r["scenario_id"]))] = float(
            r["xpts"]
        )
    return out
