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
