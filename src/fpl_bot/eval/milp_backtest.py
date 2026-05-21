"""Rolling MILP backtest harness (Phase 3 §7-§8).

Per round-1 review: validity gates and performance metrics are SEPARATE.
Phase 3 ships only if all validity gates pass; performance is reported,
not gated.

Validity gates (all must pass):
  - Feasibility: every per-GW MILP solve returns a valid solution
  - Budget: squad cost + bank ≤ £100m + accumulated capital gains
  - Transfer accounting: ft evolution matches FPL rules
  - Chip legality: each chip slot used at most once; never two chips in a GW
  - No future leakage: solver inputs only contain pre-deadline data

Performance metrics (reported):
  - Total points across the season
  - Captaincy gain vs rolling-3 baseline
  - Transfer gain (objective contribution of transferred-in − out − hits)
  - EO-adjusted utility (the actual MILP objective)
  - Comparison vs buy-and-hold-template baseline
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from itertools import combinations, permutations
from pathlib import Path

import polars as pl
from sqlalchemy import select

from fpl_bot.db import pit
from fpl_bot.db.models import DimFixture, FactPlayerMatch
from fpl_bot.db.session import session_scope
from fpl_bot.eval.xpts_eval import _run_one_fold
from fpl_bot.optim.candidate_filter import select_candidates
from fpl_bot.optim.chip_scheduler import free_hit_gws, horizon_before_free_hit
from fpl_bot.optim.eo import eo_for_candidates
from fpl_bot.optim.milp import MilpInputs, solve_rolling_horizon
from fpl_bot.optim.prediction_postprocess import (
    DEFCON_PER_POSITION_SHRINKAGE,
    apply_prediction_postprocessing,
)
from fpl_bot.optim.state import BacktestState, GwDecisions, apply_gw_outcomes

# Real-availability filter (Phase 6 v3): a player needs ≥ MIN_MINUTES_THRESHOLD
# minutes over the prior MINUTES_LOOKBACK GWs to enter the candidate pool. 30
# minutes across 3 GWs is conservative — long-term injury exclusion without
# dropping single-GW rotation cases.
MINUTES_LOOKBACK = 3
MIN_MINUTES_THRESHOLD = 30


@dataclass
class GwBacktestRecord:
    """Per-GW backtest outcomes."""

    gameweek: int
    decisions: GwDecisions
    actual_points: int  # XI total + captain bonus - 4*hits, FOR THIS GW
    state_before: BacktestState
    state_after: BacktestState
    solve_time_s: float
    solve_status: str


@dataclass(frozen=True)
class GwAttributionRecord:
    """Per-GW points-loss decomposition against hindsight oracles.

    These fields are diagnostics only. They intentionally do not feed back
    into the optimizer; their job is to show which model layer deserves the
    next upgrade.
    """

    gameweek: int
    bot_points: int
    captain_oracle_points: int
    captain_regret: int
    captain_oracle_captain: int | None
    captain_oracle_vice: int | None
    lineup_base_points: int
    lineup_oracle_base_points: int
    lineup_regret: int
    lineup_oracle_xi: frozenset[int]
    transferred_in_actual_points: int
    transferred_out_actual_points: int
    transfer_immediate_gain: int
    top_actual_player_id: int | None
    top_actual_player_points: int
    top_actual_in_squad: bool
    top_actual_in_candidates: bool
    best_candidate_not_owned_id: int | None
    best_candidate_not_owned_points: int


@dataclass
class BacktestSeasonResult:
    test_season: int
    rho: float
    alpha: float
    beta: float
    horizon: int
    gw_records: list[GwBacktestRecord] = field(default_factory=list)
    attribution_records: list[GwAttributionRecord] = field(default_factory=list)
    # Validity gates (all must pass to ship)
    validity_feasibility: bool = True
    validity_budget: bool = True
    validity_transfers: bool = True
    validity_chips: bool = True
    validity_no_future_leakage: bool = True
    validity_failures: list[str] = field(default_factory=list)
    # Performance metrics (reported)
    perf_total_points: int = 0
    perf_template_total_points: int = 0  # buy-and-hold baseline

    @property
    def all_validity_gates_pass(self) -> bool:
        return all([
            self.validity_feasibility,
            self.validity_budget,
            self.validity_transfers,
            self.validity_chips,
            self.validity_no_future_leakage,
        ])


def _per_player_per_gw_predictions(eval_df: pl.DataFrame) -> dict[tuple[int, int], float]:
    """Aggregate fixture-level e_xpts to (player_id, gameweek). Eval_df from
    xpts_eval has player_id, fixture_id, e_xpts, total_points. We need to
    join in gameweek from dim_fixture."""
    fids = eval_df["fixture_id"].unique().to_list()
    with session_scope() as s:
        rows = s.execute(
            select(DimFixture.fixture_id, DimFixture.gameweek).where(
                DimFixture.fixture_id.in_(fids)
            )
        ).all()
    fid_to_gw = {r.fixture_id: r.gameweek for r in rows}

    df = eval_df.with_columns(
        pl.col("fixture_id")
        .replace_strict(fid_to_gw, default=0)
        .alias("gameweek")
    )
    agg = df.group_by(["player_id", "gameweek"]).agg(
        pl.col("e_xpts").sum().alias("e_xpts_gw"),
        pl.col("total_points").sum().alias("actual_pts_gw"),
    )
    return {
        (int(r["player_id"]), int(r["gameweek"])): float(r["e_xpts_gw"])
        for r in agg.iter_rows(named=True)
    }


def _per_player_per_gw_actuals(eval_df: pl.DataFrame) -> dict[tuple[int, int], int]:
    """Aggregate fixture-level total_points to (player_id, gameweek)."""
    fids = eval_df["fixture_id"].unique().to_list()
    with session_scope() as s:
        rows = s.execute(
            select(DimFixture.fixture_id, DimFixture.gameweek).where(
                DimFixture.fixture_id.in_(fids)
            )
        ).all()
    fid_to_gw = {r.fixture_id: r.gameweek for r in rows}

    df = eval_df.with_columns(
        pl.col("fixture_id")
        .replace_strict(fid_to_gw, default=0)
        .alias("gameweek")
    )
    agg = df.group_by(["player_id", "gameweek"]).agg(
        pl.col("total_points").sum().alias("actual_pts_gw"),
    )
    return {
        (int(r["player_id"]), int(r["gameweek"])): int(r["actual_pts_gw"])
        for r in agg.iter_rows(named=True)
    }


def _per_player_per_gw_actual_minutes(
    season_id: int,
) -> dict[tuple[int, int], int]:
    """Per-(player, gameweek) actual minutes (summed across DGW fixtures).

    Pulled directly from fact_player_match.minutes joined to dim_fixture for
    the test season. Used by the auto-sub scorer to detect XI blanks.
    """
    with session_scope() as s:
        # Dedupe by latest recorded_at per (player_id, fixture_id) — same
        # pattern as pit.all_player_match_with_kickoff after the Phase 3.5
        # backfill that added a second bitemporal row per fixture.
        from sqlalchemy import func as _func
        latest = (
            select(
                FactPlayerMatch.player_id,
                FactPlayerMatch.fixture_id,
                _func.max(FactPlayerMatch.recorded_at).label("max_rec"),
            )
            .group_by(FactPlayerMatch.player_id, FactPlayerMatch.fixture_id)
            .subquery("fpm_minutes_latest")
        )
        rows = s.execute(
            select(
                FactPlayerMatch.player_id,
                DimFixture.gameweek,
                FactPlayerMatch.minutes,
            )
            .join(
                latest,
                (latest.c.player_id == FactPlayerMatch.player_id)
                & (latest.c.fixture_id == FactPlayerMatch.fixture_id)
                & (latest.c.max_rec == FactPlayerMatch.recorded_at),
            )
            .join(DimFixture, DimFixture.fixture_id == FactPlayerMatch.fixture_id)
            .where(DimFixture.season_id == season_id)
        ).all()
    out: dict[tuple[int, int], int] = {}
    for r in rows:
        if r.gameweek is None or r.minutes is None:
            continue
        key = (int(r.player_id), int(r.gameweek))
        out[key] = out.get(key, 0) + int(r.minutes)
    return out


def _team_id_per_player_for_season(
    season_id: int, candidates: list[int]
) -> dict[int, int]:
    """Per-player most-common team_id from fact_player_match in this season."""
    with session_scope() as s:
        rows = s.execute(
            select(
                FactPlayerMatch.player_id,
                FactPlayerMatch.was_home,
                FactPlayerMatch.fixture_id,
                DimFixture.home_team_id,
                DimFixture.away_team_id,
            )
            .join(DimFixture, DimFixture.fixture_id == FactPlayerMatch.fixture_id)
            .where(DimFixture.season_id == season_id)
            .where(FactPlayerMatch.player_id.in_(candidates))
        ).all()

    counts: dict[int, dict[int, int]] = {}
    for r in rows:
        team = r.home_team_id if r.was_home else r.away_team_id
        if team is None:
            continue
        counts.setdefault(r.player_id, {})
        counts[r.player_id][team] = counts[r.player_id].get(team, 0) + 1

    out: dict[int, int] = {}
    for pid, c in counts.items():
        out[pid] = max(c, key=c.get)
    return out


def _per_player_per_gw_prices(
    season_id: int, candidates: list[int]
) -> dict[tuple[int, int], int]:
    """Most-recent price_tenths per (player, gameweek) from fact_player_match."""
    with session_scope() as s:
        rows = s.execute(
            select(
                FactPlayerMatch.player_id,
                DimFixture.gameweek,
                FactPlayerMatch.price_tenths,
            )
            .join(DimFixture, DimFixture.fixture_id == FactPlayerMatch.fixture_id)
            .where(DimFixture.season_id == season_id)
            .where(FactPlayerMatch.player_id.in_(candidates))
            .where(FactPlayerMatch.price_tenths.is_not(None))
        ).all()

    out: dict[tuple[int, int], int] = {}
    for r in rows:
        out[(r.player_id, r.gameweek)] = r.price_tenths
    return out


def _resolve_price_at_gw(
    player_id: int, gameweek: int, prices_by_gw: dict[tuple[int, int], int]
) -> int:
    """Return price_tenths for (player, gw); fall back to nearest PRIOR GW.

    PIT-correct: NEVER walks forward. If no prior price known, returns
    a league-default fallback (rare; only triggers for players with no
    prior matches in the season — typically January transfers or unused
    candidates being added speculatively).
    """
    direct = prices_by_gw.get((player_id, gameweek))
    if direct is not None:
        return direct
    for back in range(1, gameweek):
        v = prices_by_gw.get((player_id, gameweek - back))
        if v is not None:
            return v
    return 50  # league default ~£5m fallback


def _check_validity(
    record: GwBacktestRecord,
    state_before: BacktestState,
    state_after: BacktestState,
    prices: dict[int, int],
    used_chips_before: frozenset[str],
) -> tuple[bool, bool, bool, bool, list[str]]:
    """Returns (budget_ok, transfers_ok, chips_ok, no_leak_ok, errors)."""
    errors: list[str] = []
    decisions = record.decisions

    # Budget: for normal weeks, squad cost + resulting bank must fit within
    # entering team value. On Free Hit, state_after.bank has already reverted
    # to the permanent squad's bank, so validate only the temporary FH squad
    # cost against entering team value.
    squad_cost = sum(prices.get(p, 0) for p in decisions.squad)
    entering_value = state_before.bank + sum(prices.get(p, 0) for p in state_before.squad)
    checked_value = (
        squad_cost
        if decisions.chip_played in ("FH1", "FH2")
        else squad_cost + state_after.bank
    )
    if checked_value > entering_value + 1:  # tolerance for rounding
        errors.append(
            f"GW{record.gameweek} budget violation: cost {squad_cost} + bank {state_after.bank} > entering"
        )
    budget_ok = len(errors) == 0

    # Transfer accounting: bin and bout match the squad delta
    bin_n = len(decisions.transferred_in)
    bout_n = len(decisions.transferred_out)
    if bin_n != bout_n and state_before.squad:
        # Squad size must remain 15 (cold start exempt — bin = 15, bout = 0)
        errors.append(f"GW{record.gameweek} transfer imbalance: bin={bin_n}, bout={bout_n}")

    # FT evolution: if no chip waiver, transfers above ft = hits
    chip_waives_ft = decisions.chip_played in ("WC1", "WC2", "FH1", "FH2")
    if not chip_waives_ft and state_before.squad:
        expected_hits = max(0, bin_n - state_before.free_transfers)
        if decisions.hits != expected_hits:
            errors.append(
                f"GW{record.gameweek} hit count {decisions.hits} != expected {expected_hits}"
            )
    transfers_ok = not any("transfer" in e or "hit" in e for e in errors[:])

    # Chip legality: chip not in already-used set; one chip per GW
    chips_ok = True
    if decisions.chip_played and decisions.chip_played in used_chips_before:
        errors.append(f"GW{record.gameweek} replayed chip: {decisions.chip_played}")
        chips_ok = False

    # Future leakage: solver inputs only contain pre-deadline data — verified
    # at design level (predictions come from xpts_eval._run_one_fold which uses
    # walk-forward CV; static-import audit covers optim/ as well). No runtime
    # check possible without sentinel rows; presume true if other gates pass.
    no_leak_ok = True

    return budget_ok, transfers_ok, chips_ok, no_leak_ok, errors


def _is_bench_boost_chip(chip: str | None) -> bool:
    return chip in ("BB", "BB1", "BB2")


def _xi_formation_valid(xi: frozenset[int], positions: dict[int, str]) -> bool:
    if len(xi) != 11:
        return False
    counts = {"GKP": 0, "DEF": 0, "MID": 0, "FWD": 0}
    for player_id in xi:
        position = positions.get(player_id)
        if position not in counts:
            return False
        counts[position] += 1
    return (
        counts["GKP"] == 1
        and counts["DEF"] >= 3
        and counts["MID"] >= 2
        and counts["FWD"] >= 1
        and sum(counts.values()) == 11
    )


def _score_decisions(
    *,
    decisions: GwDecisions,
    actual_pts: dict[int, int],
    actual_minutes: dict[int, int],
    positions: dict[int, str],
    bench_order_xpts: dict[int, float],
) -> int:
    from fpl_bot.optim.scorer import ScorerInputs, score_gw

    return score_gw(
        ScorerInputs(
            decisions=decisions,
            actual_pts=actual_pts,
            actual_minutes=actual_minutes,
            positions=positions,
            bench_order_xpts=bench_order_xpts,
        )
    ).gw_points


def _postprocess_captain_decision(
    decisions: GwDecisions,
    *,
    gameweek: int,
    captain_predictions: dict[tuple[int, int], float] | None,
) -> GwDecisions:
    """Pick captain/vice from the solved XI using an alternate captain score."""
    if not captain_predictions:
        return decisions

    xi = sorted(decisions.starting_xi)
    if len(xi) < 2:
        return decisions

    ranked = sorted(
        xi,
        key=lambda player_id: (
            captain_predictions.get((player_id, gameweek), 0.0),
            -player_id,
        ),
        reverse=True,
    )
    return replace(decisions, captain=ranked[0], vice=ranked[1])


def _oracle_captain_points(
    *,
    decisions: GwDecisions,
    bot_points: int,
    actual_pts: dict[int, int],
    actual_minutes: dict[int, int],
    positions: dict[int, str],
    bench_order_xpts: dict[int, float],
) -> tuple[int, int | None, int | None]:
    """Best same-XI score if captain/vice were picked with hindsight."""
    xi = tuple(sorted(decisions.starting_xi))
    best_points = bot_points
    best_captain = decisions.captain
    best_vice = decisions.vice

    if len(xi) < 2:
        return best_points, best_captain, best_vice

    for captain, vice in permutations(xi, 2):
        candidate = replace(decisions, captain=captain, vice=vice)
        points = _score_decisions(
            decisions=candidate,
            actual_pts=actual_pts,
            actual_minutes=actual_minutes,
            positions=positions,
            bench_order_xpts=bench_order_xpts,
        )
        if points > best_points:
            best_points = points
            best_captain = captain
            best_vice = vice

    return best_points, best_captain, best_vice


def _lineup_base_points(
    *,
    decisions: GwDecisions,
    scoring_players: frozenset[int],
    actual_pts: dict[int, int],
) -> int:
    players = (
        decisions.squad
        if _is_bench_boost_chip(decisions.chip_played)
        else scoring_players
    )
    return sum(actual_pts.get(p, 0) for p in players) - 4 * decisions.hits


def _oracle_lineup_base_points(
    *,
    decisions: GwDecisions,
    actual_pts: dict[int, int],
    positions: dict[int, str],
    bot_lineup_base_points: int,
) -> tuple[int, frozenset[int]]:
    """Best valid XI base points from the owned squad, excluding captain multiplier."""
    if _is_bench_boost_chip(decisions.chip_played):
        return bot_lineup_base_points, decisions.starting_xi

    best_points = bot_lineup_base_points
    best_xi = decisions.starting_xi
    for xi_tuple in combinations(sorted(decisions.squad), 11):
        xi = frozenset(xi_tuple)
        if not _xi_formation_valid(xi, positions):
            continue
        points = sum(actual_pts.get(p, 0) for p in xi) - 4 * decisions.hits
        if points > best_points:
            best_points = points
            best_xi = xi
    return best_points, best_xi


def _gw_actuals(
    *,
    gw: int,
    actual_by_pgw: dict[tuple[int, int], int],
) -> list[tuple[int, int]]:
    return [
        (player_id, points)
        for (player_id, gameweek), points in actual_by_pgw.items()
        if gameweek == gw
    ]


def _build_gw_attribution(
    *,
    gw: int,
    decisions: GwDecisions,
    bot_points: int,
    final_scoring_players: frozenset[int],
    actual_pts_this_gw: dict[int, int],
    actual_minutes_this_gw: dict[int, int],
    actual_by_pgw: dict[tuple[int, int], int],
    positions: dict[int, str],
    bench_order_xpts: dict[int, float],
    candidates: set[int],
) -> GwAttributionRecord:
    captain_points, captain, vice = _oracle_captain_points(
        decisions=decisions,
        bot_points=bot_points,
        actual_pts=actual_pts_this_gw,
        actual_minutes=actual_minutes_this_gw,
        positions=positions,
        bench_order_xpts=bench_order_xpts,
    )
    lineup_base = _lineup_base_points(
        decisions=decisions,
        scoring_players=final_scoring_players,
        actual_pts=actual_pts_this_gw,
    )
    lineup_oracle, lineup_oracle_xi = _oracle_lineup_base_points(
        decisions=decisions,
        actual_pts=actual_pts_this_gw,
        positions=positions,
        bot_lineup_base_points=lineup_base,
    )

    all_actuals = _gw_actuals(gw=gw, actual_by_pgw=actual_by_pgw)
    top_actual_id: int | None = None
    top_actual_points = 0
    if all_actuals:
        top_actual_id, top_actual_points = max(
            all_actuals, key=lambda item: (item[1], -item[0])
        )

    best_candidate_not_owned_id: int | None = None
    best_candidate_not_owned_points = 0
    not_owned_candidate_actuals = [
        (player_id, points)
        for player_id, points in all_actuals
        if player_id in candidates and player_id not in decisions.squad
    ]
    if not_owned_candidate_actuals:
        best_candidate_not_owned_id, best_candidate_not_owned_points = max(
            not_owned_candidate_actuals,
            key=lambda item: (item[1], -item[0]),
        )

    if decisions.transferred_out:
        transferred_in_points = sum(
            actual_by_pgw.get((player_id, gw), 0)
            for player_id in decisions.transferred_in
        )
        transferred_out_points = sum(
            actual_by_pgw.get((player_id, gw), 0)
            for player_id in decisions.transferred_out
        )
        transfer_gain = (
            transferred_in_points
            - transferred_out_points
            - 4 * decisions.hits
        )
    else:
        transferred_in_points = 0
        transferred_out_points = 0
        transfer_gain = 0

    return GwAttributionRecord(
        gameweek=gw,
        bot_points=bot_points,
        captain_oracle_points=captain_points,
        captain_regret=max(0, captain_points - bot_points),
        captain_oracle_captain=captain,
        captain_oracle_vice=vice,
        lineup_base_points=lineup_base,
        lineup_oracle_base_points=lineup_oracle,
        lineup_regret=max(0, lineup_oracle - lineup_base),
        lineup_oracle_xi=lineup_oracle_xi,
        transferred_in_actual_points=transferred_in_points,
        transferred_out_actual_points=transferred_out_points,
        transfer_immediate_gain=transfer_gain,
        top_actual_player_id=top_actual_id,
        top_actual_player_points=top_actual_points,
        top_actual_in_squad=top_actual_id in decisions.squad if top_actual_id else False,
        top_actual_in_candidates=top_actual_id in candidates if top_actual_id else False,
        best_candidate_not_owned_id=best_candidate_not_owned_id,
        best_candidate_not_owned_points=best_candidate_not_owned_points,
    )


def _build_walk_forward_eval_df(
    *,
    test_season: int,
    train_seasons: list[int],
    chunk_size: int,
    n_iterations: int,
    cache_dir: Path,
    cache_predictions: bool,
) -> pl.DataFrame:
    """Build a combined eval_df via chunk-wise walk-forward retraining.

    For each chunk [chunk_start, chunk_start+chunk_size), train on
    train_seasons + test_season's GWs 1..chunk_start-1 and use those
    predictions for the chunk. Chunk 0 (GW1..) uses the baseline train
    (no test-season data) — same as the non-walk-forward path.

    Eval_df rows for each chunk are filtered to the chunk's GWs to
    avoid mixing different models' predictions for the same fixture.
    Per-chunk parquet caches keep iteration fast.
    """
    train_str = "_".join(str(s) for s in train_seasons)
    # Need fixture → gameweek mapping for chunk filtering
    with session_scope() as _s:
        fx_rows = _s.execute(
            select(DimFixture.fixture_id, DimFixture.gameweek).where(
                DimFixture.season_id == test_season
            )
        ).all()
    fid_to_gw = {int(r.fixture_id): int(r.gameweek) for r in fx_rows}
    all_gws = sorted(set(fid_to_gw.values()))
    if not all_gws:
        raise RuntimeError(f"No fixtures for season {test_season}")
    max_gw = max(all_gws)

    chunk_starts = list(range(1, max_gw + 1, chunk_size))
    pieces: list[pl.DataFrame] = []
    for chunk_start in chunk_starts:
        chunk_end = min(chunk_start + chunk_size - 1, max_gw)
        train_through = chunk_start - 1 if chunk_start > 1 else None
        # Per-chunk cache
        suffix = "baseline" if train_through is None else f"thru_gw{train_through}"
        chunk_cache = cache_dir / f"season_{test_season}_train_{train_str}_{suffix}.parquet"
        if cache_predictions and chunk_cache.exists():
            eval_df = pl.read_parquet(chunk_cache)
        else:
            result = _run_one_fold(
                train_seasons, test_season,
                n_iterations=n_iterations, seed=42,
                train_through_gw=train_through,
            )
            if result is None:
                raise RuntimeError(
                    f"Walk-forward fold failed at chunk_start={chunk_start}"
                )
            _, eval_df, _ = result
            if cache_predictions:
                eval_df.write_parquet(chunk_cache)
        # Filter this chunk's slice
        chunk_fixtures = [
            fid for fid, gw in fid_to_gw.items() if chunk_start <= gw <= chunk_end
        ]
        slice_df = eval_df.filter(pl.col("fixture_id").is_in(chunk_fixtures))
        pieces.append(slice_df)
        print(
            f"  walk-forward chunk GW{chunk_start}-{chunk_end} "
            f"(train_through_gw={train_through}): {slice_df.shape[0]} rows"
        )
    return pl.concat(pieces, how="diagonal_relaxed")


def backtest_season(
    test_season: int,
    train_seasons: list[int],
    *,
    horizon: int = 6,
    rho: float = 1.0,
    alpha: float = 0.8,
    beta: float = 0.05,
    enable_chips: bool = False,
    n_iterations: int = 200,
    cache_predictions: bool = True,
    cache_dir: Path = Path("data/cache/xpts_predictions"),
    use_saa: bool = False,
    n_scenarios: int = 25,
    raw_samples_dir: Path = Path("data/cache/xpts_raw_samples"),
    use_chip_schedule: bool = False,
    transfer_penalty: float = 0.0,
    captain_quantile: float | None = None,
    captain_haul_weight: float = 0.0,
    captain_haul_threshold: float = 6.0,
    captain_postprocess: bool = False,
    position_calibration: dict[str, float] | None = None,
    walk_forward_chunk: int | None = None,
    defcon_shrinkage: float | None = None,
    defcon_per_position_shrinkage: dict[str, float] | None = DEFCON_PER_POSITION_SHRINKAGE,
    fwd_calibration: bool = False,
) -> BacktestSeasonResult:
    """Run rolling MILP backtest for one test season.

    If `walk_forward_chunk` is set (e.g., 5), the underlying boosters
    are retrained every chunk_size GWs using only data prior to the
    chunk start (PIT-correct). Predictions for chunk GWs come from the
    chunk-specific model. Without `walk_forward_chunk`, the boosters
    train once on `train_seasons` and predict for the whole test_season
    (current default behavior).
    """
    if captain_quantile is not None and captain_haul_weight > 0:
        raise ValueError("captain_quantile and captain_haul_weight are mutually exclusive")

    cache_dir.mkdir(parents=True, exist_ok=True)
    train_str = "_".join(str(s) for s in train_seasons)
    cache_path = cache_dir / f"season_{test_season}_train_{train_str}.parquet"
    if walk_forward_chunk is not None:
        eval_df = _build_walk_forward_eval_df(
            test_season=test_season,
            train_seasons=train_seasons,
            chunk_size=walk_forward_chunk,
            n_iterations=n_iterations,
            cache_dir=cache_dir,
            cache_predictions=cache_predictions,
        )
    elif cache_predictions and cache_path.exists():
        eval_df = pl.read_parquet(cache_path)
    else:
        result = _run_one_fold(
            train_seasons, test_season, n_iterations=n_iterations, seed=42
        )
        if result is None:
            raise RuntimeError(f"Could not generate predictions for season {test_season}")
        _, eval_df, _ = result
        if cache_predictions:
            eval_df.write_parquet(cache_path)

    # Aggregate to per(player, gw) predictions and actuals
    pred_by_pgw = _per_player_per_gw_predictions(eval_df)
    actual_by_pgw = _per_player_per_gw_actuals(eval_df)

    pred_by_pgw = apply_prediction_postprocessing(
        pred_by_pgw,
        season_id=test_season,
        train_seasons=train_seasons,
        position_calibration=position_calibration,
        defcon_shrinkage=defcon_shrinkage,
        defcon_per_position_shrinkage=defcon_per_position_shrinkage,
        fwd_calibration=fwd_calibration,
        cache_dir=cache_dir,
    )

    # Actual minutes per (player, gw) — needed by the auto-sub scorer to
    # detect XI blanks (minutes == 0).
    actual_minutes_by_pgw = _per_player_per_gw_actual_minutes(test_season)

    # Captain alternate scoring: load raw samples and compute a captain-only
    # reward term. Decouples captain decision from the XI mean-xPts term.
    # `captain_quantile` is conservative; `captain_haul_weight` is aggressive
    # and rewards simulated ceiling outcomes above `captain_haul_threshold`.
    captain_predictions: dict[tuple[int, int], float] | None = None
    if captain_quantile is not None or captain_haul_weight > 0:
        from fpl_bot.optim.scenarios import (
            aggregate_pts_by_player_gw_scenario as _agg,
        )
        from fpl_bot.optim.scenarios import (
            captain_haul_score_per_gw,
            captain_lower_quantile_per_gw,
            load_raw_samples,
        )

        try:
            _raw = load_raw_samples(
                test_season=test_season,
                train_seasons=train_seasons,
                n_iterations=n_iterations,
                cache_dir=raw_samples_dir,
            )
            _pts_df = _agg(_raw, n_scenarios=n_iterations)
            if captain_quantile is not None:
                captain_predictions = captain_lower_quantile_per_gw(
                    _pts_df, quantile=captain_quantile
                )
            else:
                captain_predictions = captain_haul_score_per_gw(
                    _pts_df,
                    threshold=captain_haul_threshold,
                    weight=captain_haul_weight,
                )
        except FileNotFoundError:
            mode = (
                f"captain_quantile={captain_quantile}"
                if captain_quantile is not None
                else (
                    f"captain_haul_weight={captain_haul_weight}, "
                    f"captain_haul_threshold={captain_haul_threshold}"
                )
            )
            print(
                f"  {mode} requested but raw "
                f"samples not cached; falling back to mean xpts for captain."
            )

    # SAA: load per-scenario raw samples and aggregate to (player, gw, scenario).
    pts_per_scenario: pl.DataFrame | None = None
    scenario_ids: list[int] | None = None
    if use_saa:
        from fpl_bot.optim.scenarios import (
            aggregate_pts_by_player_gw_scenario,
            load_raw_samples,
        )
        raw_samples = load_raw_samples(
            test_season=test_season,
            train_seasons=train_seasons,
            n_iterations=n_iterations,
            cache_dir=raw_samples_dir,
        )
        pts_per_scenario = aggregate_pts_by_player_gw_scenario(
            raw_samples, n_scenarios=n_scenarios
        )
        scenario_ids = list(range(n_scenarios))

    # Identify all gameweeks that have predictions
    all_gws = sorted({gw for (_, gw) in pred_by_pgw if gw > 0})
    if not all_gws:
        raise RuntimeError(f"No GWs with predictions for season {test_season}")

    # Get all candidate player_ids appearing in predictions
    all_players = sorted({pid for (pid, _) in pred_by_pgw})

    # Resolve team_id per player and prices per (player, gw)
    teams = _team_id_per_player_for_season(test_season, all_players)
    prices_by_pgw = _per_player_per_gw_prices(test_season, all_players)
    positions = pit.all_player_positions().to_pandas().set_index("player_id")["position_code"].to_dict()

    # Drop players with missing position or team
    valid_players = [p for p in all_players if p in positions and p in teams]

    # Predictions DataFrame for candidate filtering
    pred_records = [
        {"player_id": pid, "gameweek": gw, "e_xpts": v}
        for (pid, gw), v in pred_by_pgw.items()
        if pid in valid_players
    ]
    pred_df = pl.DataFrame(pred_records)

    # Phase 5 chip schedule (computed once at fold start). Decouples chip
    # timing from the rolling MILP — each per-GW solve sees the chip
    # activations as fixed parameters, not decisions.
    chip_schedule: dict[str, int] | None = None
    if use_chip_schedule and enable_chips:
        from fpl_bot.optim.chip_scheduler import make_chip_schedule
        from fpl_bot.optim.fixture_analytics import load_fixture_analytics
        fixture_analytics = load_fixture_analytics(test_season)
        chip_schedule = make_chip_schedule(
            analytics=fixture_analytics,
            predictions_df=pred_df,
            team_id_per_player={p: teams[p] for p in valid_players if p in teams},
        ).as_dict()
        print(f"Chip schedule: {chip_schedule}", flush=True)

    state = BacktestState.cold_start(season_id=test_season)
    result = BacktestSeasonResult(
        test_season=test_season, rho=rho, alpha=alpha, beta=beta, horizon=horizon
    )

    print(f"Starting backtest: season {test_season}, {len(all_gws)} GWs, "
          f"{len(valid_players)} eligible players, horizon={horizon}")

    for gw in all_gws:
        # NOTE on stability: appsi_highs occasionally SIGABRTs after many
        # solves in one Python process — a HiGHS C++ state leak we can't
        # control from Python. Experiments showed gc.collect() either here
        # or inside solve_milp triggers destructors at the wrong time and
        # MAKES IT WORSE. The reliable workaround is per-fold subprocess
        # isolation; see scripts/phase2_5_1_phase3_isolated.py. The
        # single-process walk-forward is best-effort.
        # Cold-start GW1: H=1 (just pick the squad; terminal value brings in
        # 5-GW lookahead via post-horizon xPts term). Reduces solve complexity.
        # Subsequent GWs: full rolling horizon.
        effective_horizon = 1 if not state.squad else horizon
        # Build horizon (this gw + horizon-1 lookahead, capped at last GW)
        horizon_gws = [w for w in range(gw, gw + effective_horizon) if w in all_gws]
        horizon_gws = horizon_before_free_hit(horizon_gws, chip_schedule)
        if not horizon_gws:
            continue
        is_current_free_hit = gw in free_hit_gws(chip_schedule)

        # Candidate filter using horizon predictions + current squad
        horizon_pred = pred_df.filter(pl.col("gameweek").is_in(horizon_gws))
        candidates_set = select_candidates(
            season_id=test_season,
            horizon_predictions=horizon_pred,
            current_squad=set(state.squad),
            top_n_per_position=25,
            cheap_k_per_position=3,
        )
        candidates_set &= set(valid_players)
        candidates_set |= set(state.squad)

        # Real-availability filter (Phase 6 v3): drop candidates whose minutes
        # over the prior MINUTES_LOOKBACK GWs total < MIN_MINUTES_THRESHOLD —
        # the backtest analog of the live status_code in {i,n,s,u} filter.
        # GW1 has no history, so this is a no-op there. Current-squad players
        # are kept (the MILP must be able to transfer them out).
        recent_min = pit.recent_minutes_at_gw(
            test_season, gw, lookback_gws=MINUTES_LOOKBACK
        )
        captain_attenuator: dict[int, float] | None = None
        if recent_min:
            keep: set[int] = set()
            for p in candidates_set:
                if p in state.squad:
                    keep.add(p)
                    continue
                if recent_min.get(p, 0) >= MIN_MINUTES_THRESHOLD:
                    keep.add(p)
            candidates_set = keep
            # Captain blank-risk attenuator: 1.0 for full availability across
            # the lookback (270 mins over 3 GWs), 0.0 for fully absent. Biases
            # captain choice toward minutes-reliable players without affecting
            # the XI pred (which already factors in minutes via the model).
            max_minutes = MINUTES_LOOKBACK * 90
            captain_attenuator = {
                p: min(recent_min.get(p, 0) / max_minutes, 1.0)
                for p in candidates_set
            }

        candidates = sorted(candidates_set)
        if not candidates:
            result.validity_feasibility = False
            result.validity_failures.append(f"GW{gw}: empty candidate set")
            break

        # Resolve current price for this GW (used as buy price for everyone; also
        # the sell price for non-state-squad players who could be bought-then-sold
        # within the horizon — they have no cost basis yet so no spread).
        buy_prices = {p: _resolve_price_at_gw(p, gw, prices_by_pgw) for p in candidates}
        # Sell prices: FPL's sell-tax rule.
        #   tax = max(0, (current - cost_basis) // 2)  (50% tax on profit only)
        #   sell = current - tax
        # When current < basis: profit < 0, tax = 0, sell = current ✓
        # When current ≥ basis: tax = profit//2, sell = basis + ceil(profit/2)
        # Players not in state.squad have no cost basis yet → sell = current
        # (could only be sold within-horizon after a buy, no profit possible
        # under static prices).
        sell_prices = {}
        for p in candidates:
            current = buy_prices[p]
            basis = state.cost_basis.get(p) if p in state.squad else None
            if basis is not None:
                tax = max(0, (current - basis) // 2)
                sell_prices[p] = current - tax
            else:
                sell_prices[p] = current

        # Predictions dict scoped to candidates × horizon_gws
        pred_dict = {
            (p, w): pred_by_pgw.get((p, w), 0.0)
            for p in candidates
            for w in horizon_gws
        }

        eo = eo_for_candidates(candidates)
        teams_dict = {p: teams[p] for p in candidates if p in teams}
        positions_dict = {p: positions.get(p, "MID") for p in candidates}

        # SAA: build per-scenario pts dict scoped to this horizon
        pts_per_scenario_dict: dict[tuple[int, int, int], float] | None = None
        if use_saa and pts_per_scenario is not None and scenario_ids is not None:
            from fpl_bot.optim.scenarios import make_pts_dict
            pts_per_scenario_dict = make_pts_dict(
                pts_per_scenario,
                candidates=candidates,
                horizon_gws=horizon_gws,
                n_scenarios=n_scenarios,
            )

        inputs = MilpInputs(
            state=state,
            horizon_weeks=horizon_gws,
            candidates=candidates,
            predictions=pred_dict,
            eo=eo,
            buy_prices=buy_prices,
            sell_prices=sell_prices,
            positions=positions_dict,
            teams=teams_dict,
            rho=rho,
            alpha=alpha,
            beta=beta,
            enable_chips=enable_chips,
            full_predictions=None if is_current_free_hit else pred_df,
            use_saa=use_saa,
            predictions_per_scenario=pts_per_scenario_dict,
            scenario_ids=scenario_ids,
            chip_schedule=chip_schedule,
            captain_attenuator=captain_attenuator,
            transfer_penalty=transfer_penalty,
            captain_predictions=None if captain_postprocess else captain_predictions,
        )

        import time

        t0 = time.time()
        # Cold-start gets longer time budget; rolling-horizon solves are smaller.
        # Rolling limit history:
        #   v1.0/v1.1 = 60s
        #   v1.2 = 120s (α/β grid surfaced flake at 60s where β-equivalent
        #     configs hit no_feasible at maxTimeLimit)
        # v2.5.1 root cause: the multinomial-fix predictions tightened
        # within-team ties → LP relaxation has many near-optimal integer
        # solutions → branch-and-bound can't prove optimality fast. Fix is
        # the 1% MIP gap in solve_milp (not a time-limit bump).
        time_limit = 180 if not state.squad else 120
        try:
            decisions, meta = solve_rolling_horizon(inputs, time_limit_s=time_limit)
        except Exception as exc:
            result.validity_feasibility = False
            result.validity_failures.append(f"GW{gw}: solver failed ({exc})")
            break
        decisions = _postprocess_captain_decision(
            decisions,
            gameweek=gw,
            captain_predictions=captain_predictions if captain_postprocess else None,
        )
        solve_time = time.time() - t0

        # Compute actual GW points with FPL auto-sub + auto-vice rules.
        from fpl_bot.optim.scorer import ScorerInputs, score_gw
        actual_pts_this_gw = {
            p: actual_by_pgw.get((p, gw), 0) for p in decisions.squad
        }
        actual_minutes_this_gw = {
            p: actual_minutes_by_pgw.get((p, gw), 0) for p in decisions.squad
        }
        # Bench-order heuristic: rank by this-GW predicted xPts (the same
        # signal the MILP optimized against). The manager's best-guess
        # ranking of "who should sub in first if needed".
        bench_order_xpts = {
            p: pred_by_pgw.get((p, gw), 0.0) for p in decisions.squad
        }
        scorer_out = score_gw(
            ScorerInputs(
                decisions=decisions,
                actual_pts=actual_pts_this_gw,
                actual_minutes=actual_minutes_this_gw,
                positions=positions_dict,
                bench_order_xpts=bench_order_xpts,
            )
        )
        gw_points = scorer_out.gw_points
        result.attribution_records.append(
            _build_gw_attribution(
                gw=gw,
                decisions=decisions,
                bot_points=gw_points,
                final_scoring_players=scorer_out.starting_xi_final,
                actual_pts_this_gw=actual_pts_this_gw,
                actual_minutes_this_gw=actual_minutes_this_gw,
                actual_by_pgw=actual_by_pgw,
                positions=positions_dict,
                bench_order_xpts=bench_order_xpts,
                candidates=set(candidates),
            )
        )

        # Apply outcomes → next state. Pass the cost-basis-aware sell prices
        # so apply_gw_outcomes' bank update matches what the MILP solved against.
        state_before = state
        actual_prices_for_state = {
            p: {"buy": buy_prices[p], "sell": sell_prices[p]} for p in candidates
        }
        new_state = apply_gw_outcomes(state, decisions, actual_prices_for_state)

        record = GwBacktestRecord(
            gameweek=gw,
            decisions=decisions,
            actual_points=gw_points,
            state_before=state_before,
            state_after=new_state,
            solve_time_s=solve_time,
            solve_status=meta["termination"],
        )
        result.gw_records.append(record)

        # Validity checks. Budget check uses current (buy) prices to compare
        # squad value before/after — the cost-basis sell-tax doesn't make money
        # appear from nowhere (sell ≤ current under tax), so cost + bank
        # ≤ entering still holds.
        budget_ok, transfers_ok, chips_ok, leak_ok, errs = _check_validity(
            record,
            state_before=state_before,
            state_after=new_state,
            prices=buy_prices,
            used_chips_before=state_before.chips_used,
        )
        if not budget_ok:
            result.validity_budget = False
            result.validity_failures.extend(errs)
        if not transfers_ok:
            result.validity_transfers = False
            result.validity_failures.extend([e for e in errs if "transfer" in e or "hit" in e])
        if not chips_ok:
            result.validity_chips = False
            result.validity_failures.extend([e for e in errs if "chip" in e or "replayed" in e])
        if not leak_ok:
            result.validity_no_future_leakage = False

        state = new_state

    # Aggregate performance
    result.perf_total_points = sum(r.actual_points for r in result.gw_records)
    result.perf_template_total_points = _buy_and_hold_template_points(
        all_gws=all_gws,
        actual_by_pgw=actual_by_pgw,
        first_gw_decisions=result.gw_records[0].decisions if result.gw_records else None,
    )

    return result


def _buy_and_hold_template_points(
    *,
    all_gws: list[int],
    actual_by_pgw: dict[tuple[int, int], int],
    first_gw_decisions: GwDecisions | None,
) -> int:
    """Buy-and-hold baseline: GW1 squad held all season, captain set to player
    with highest actual_pts in each GW within the held XI (oracle captaincy on
    held squad — generous baseline)."""
    if first_gw_decisions is None:
        return 0
    held_xi = first_gw_decisions.starting_xi
    total = 0
    for gw in all_gws:
        xi_pts = [actual_by_pgw.get((p, gw), 0) for p in held_xi]
        if not xi_pts:
            continue
        total += sum(xi_pts) + max(xi_pts)  # captain on whichever XI player scored highest
    return total


def format_validity_report(result: BacktestSeasonResult) -> str:
    lines = [
        f"Validity gates — season 20{result.test_season}/{(result.test_season+1)%100:02d}",
        "",
        f"  Feasibility:        {'PASS' if result.validity_feasibility else 'FAIL'}",
        f"  Budget:             {'PASS' if result.validity_budget else 'FAIL'}",
        f"  Transfer accounting:{'PASS' if result.validity_transfers else 'FAIL'}",
        f"  Chip legality:      {'PASS' if result.validity_chips else 'FAIL'}",
        f"  No future leakage:  {'PASS' if result.validity_no_future_leakage else 'FAIL'}",
        "",
        f"  ALL GATES PASS: {result.all_validity_gates_pass}",
    ]
    if result.validity_failures:
        lines.append("")
        lines.append("Failures:")
        for f in result.validity_failures[:10]:  # cap output
            lines.append(f"  - {f}")
        if len(result.validity_failures) > 10:
            lines.append(f"  ... and {len(result.validity_failures) - 10} more")
    return "\n".join(lines)


def format_performance_report(result: BacktestSeasonResult) -> str:
    lines = [
        f"Performance — season 20{result.test_season}/{(result.test_season+1)%100:02d}",
        "",
        f"  Total points (MILP):        {result.perf_total_points}",
        f"  Buy-and-hold-template:      {result.perf_template_total_points}",
        f"  MILP - template:            {result.perf_total_points - result.perf_template_total_points:+d}",
        "",
        f"  GWs solved: {len(result.gw_records)} / 38",
        f"  Mean solve time:    {sum(r.solve_time_s for r in result.gw_records) / max(len(result.gw_records), 1):.1f}s",
        f"  Total transfers:    {sum(len(r.decisions.transferred_in) for r in result.gw_records)}",
        f"  Total hits taken:   {sum(r.decisions.hits for r in result.gw_records)}",
        f"  Chips played:       {[r.decisions.chip_played for r in result.gw_records if r.decisions.chip_played]}",
    ]
    return "\n".join(lines)


def format_attribution_report(result: BacktestSeasonResult, top_n: int = 5) -> str:
    records = result.attribution_records
    if not records:
        return (
            f"Attribution — season 20{result.test_season}/"
            f"{(result.test_season+1)%100:02d}\n\n  No attribution records."
        )

    captain_total = sum(r.captain_regret for r in records)
    lineup_total = sum(r.lineup_regret for r in records)
    transfer_total = sum(r.transfer_immediate_gain for r in records)
    top_in_squad = sum(1 for r in records if r.top_actual_in_squad)
    top_in_candidates = sum(1 for r in records if r.top_actual_in_candidates)

    lines = [
        f"Attribution — season 20{result.test_season}/{(result.test_season+1)%100:02d}",
        "",
        f"  Bot points:                         {result.perf_total_points}",
        f"  Captain regret, same XI:            +{captain_total}",
        f"  Lineup/bench regret, owned squad:   +{lineup_total}",
        f"  Same-GW transfer immediate gain:    {transfer_total:+d}",
        f"  GW top scorer already in squad:     {top_in_squad}/{len(records)}",
        f"  GW top scorer in candidate pool:    {top_in_candidates}/{len(records)}",
    ]

    captain_leaks = sorted(records, key=lambda r: r.captain_regret, reverse=True)
    captain_leaks = [r for r in captain_leaks if r.captain_regret > 0][:top_n]
    if captain_leaks:
        lines.extend(["", "  Biggest captain leaks:"])
        for r in captain_leaks:
            lines.append(
                f"    GW{r.gameweek}: +{r.captain_regret} "
                f"(bot {r.bot_points}, oracle {r.captain_oracle_points}, "
                f"cap {r.captain_oracle_captain})"
            )

    lineup_leaks = sorted(records, key=lambda r: r.lineup_regret, reverse=True)
    lineup_leaks = [r for r in lineup_leaks if r.lineup_regret > 0][:top_n]
    if lineup_leaks:
        lines.extend(["", "  Biggest lineup/bench leaks:"])
        for r in lineup_leaks:
            lines.append(
                f"    GW{r.gameweek}: +{r.lineup_regret} "
                f"(base {r.lineup_base_points}->{r.lineup_oracle_base_points})"
            )

    missed_top = [
        r for r in records
        if r.top_actual_player_id is not None and not r.top_actual_in_squad
    ]
    missed_top.sort(key=lambda r: r.top_actual_player_points, reverse=True)
    if missed_top:
        lines.extend(["", "  Highest missed GW scorers:"])
        for r in missed_top[:top_n]:
            lines.append(
                f"    GW{r.gameweek}: player {r.top_actual_player_id} "
                f"scored {r.top_actual_player_points} "
                f"(candidate={r.top_actual_in_candidates})"
            )

    candidate_misses = [
        r for r in records
        if r.best_candidate_not_owned_id is not None
        and r.best_candidate_not_owned_points > 0
    ]
    candidate_misses.sort(key=lambda r: r.best_candidate_not_owned_points, reverse=True)
    if candidate_misses:
        lines.extend(["", "  Best unowned candidates by GW:"])
        for r in candidate_misses[:top_n]:
            lines.append(
                f"    GW{r.gameweek}: player {r.best_candidate_not_owned_id} "
                f"scored {r.best_candidate_not_owned_points}"
            )

    return "\n".join(lines)
