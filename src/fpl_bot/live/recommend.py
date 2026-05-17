"""Phase 6 live recommend pipeline.

End-to-end: read state → build candidates → apply status filter → run
MILP → emit recommendation files (markdown + JSON).

Reuses every Phase 1 → Phase 5 component. The only new logic vs the
backtest harness is the status override path: injured players are dropped
from the candidate set; doubtful players have their xPts attenuated by
chance_of_playing/100 before the MILP sees them.
"""
from __future__ import annotations

import datetime as dt
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import polars as pl

from fpl_bot.db import pit
from fpl_bot.eval.milp_backtest import (
    _per_player_per_gw_predictions,
    _per_player_per_gw_prices,
    _resolve_price_at_gw,
    _team_id_per_player_for_season,
)
from fpl_bot.eval.xpts_eval import _run_one_fold
from fpl_bot.live.state_builder import (
    LiveStatusOverrides,
    load_status_overrides,
    load_user_state,
)
from fpl_bot.optim.candidate_filter import select_candidates
from fpl_bot.optim.chip_scheduler import make_chip_schedule
from fpl_bot.optim.eo import eo_for_candidates
from fpl_bot.optim.fixture_analytics import load_fixture_analytics
from fpl_bot.optim.milp import MilpInputs, solve_rolling_horizon

OUTPUT_ROOT = Path("data/live/recommendations")


@dataclass
class RecommendationContext:
    """All inputs needed to write a recommendation."""

    season_id: int
    gameweek: int
    team_id: int
    state_before: dict  # serialized BacktestState fields
    chip_schedule: dict[str, int]
    overrides: dict  # excluded_player_ids, xpts_attenuator
    horizon_gws: list[int]
    candidates_count: int


def _attenuate_predictions(
    pred_by_pgw: dict[tuple[int, int], float],
    overrides: LiveStatusOverrides,
) -> dict[tuple[int, int], float]:
    """Multiply pred by attenuator for doubtful players; drop excluded."""
    out: dict[tuple[int, int], float] = {}
    for (pid, gw), v in pred_by_pgw.items():
        if pid in overrides.excluded_player_ids:
            continue
        mult = overrides.xpts_attenuator.get(pid, 1.0)
        out[(pid, gw)] = v * mult
    return out


def generate_recommendation(
    *,
    season_id: int,
    gameweek: int,
    team_id: int,
    train_seasons: list[int],
    horizon: int = 6,
    rho: float = 1.0,
    alpha: float = 0.8,
    beta: float = 0.0,
    n_iterations: int = 500,
    cache_predictions: bool = True,
    cache_dir: Path = Path("data/cache/xpts_predictions"),
    output_dir: Path | None = None,
) -> tuple[Path, Path]:
    """Run the full Phase 6 pipeline and write recommendation files.

    Returns (markdown_path, json_path).
    """
    # 1. Load or generate per-(player, gw) predictions for the test season
    cache_path = cache_dir / (
        f"season_{season_id}_train_{'_'.join(str(s) for s in train_seasons)}.parquet"
    )
    if cache_predictions and cache_path.exists():
        eval_df = pl.read_parquet(cache_path)
    else:
        result = _run_one_fold(
            train_seasons, season_id, n_iterations=n_iterations, seed=42
        )
        if result is None:
            raise RuntimeError(f"Could not generate predictions for season {season_id}")
        _, eval_df, _ = result
        if cache_predictions:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            eval_df.write_parquet(cache_path)

    pred_by_pgw = _per_player_per_gw_predictions(eval_df)
    all_gws = sorted({gw for (_, gw) in pred_by_pgw if gw > 0})
    all_players = sorted({pid for (pid, _) in pred_by_pgw})

    # 2. Load user state from latest snapshot
    state = load_user_state(season_id=season_id, gameweek=gameweek, team_id=team_id)

    # 3. Status overrides — applied BEFORE candidate filter
    overrides = load_status_overrides(candidate_player_ids=all_players)
    pred_by_pgw_filtered = _attenuate_predictions(pred_by_pgw, overrides)

    # 4. Team / position / price resolution (same as backtest)
    teams = _team_id_per_player_for_season(season_id, all_players)
    prices_by_pgw = _per_player_per_gw_prices(season_id, all_players)
    positions = (
        pit.all_player_positions()
        .to_pandas()
        .set_index("player_id")["position_code"]
        .to_dict()
    )
    valid_players = [
        p
        for p in all_players
        if p in positions and p in teams and p not in overrides.excluded_player_ids
    ]

    # 5. Chip schedule + DGW set (Phase 5)
    fixture_analytics = load_fixture_analytics(season_id)
    pred_df = pl.DataFrame(
        [
            {"player_id": pid, "gameweek": gw, "e_xpts": v}
            for (pid, gw), v in pred_by_pgw_filtered.items()
            if pid in valid_players
        ]
    )
    chip_schedule_obj = make_chip_schedule(
        analytics=fixture_analytics,
        predictions_df=pred_df,
        team_id_per_player={p: teams[p] for p in valid_players if p in teams},
    )
    chip_schedule = chip_schedule_obj.as_dict()

    # 6. Candidate set for the current GW horizon
    horizon_gws = [w for w in range(gameweek, gameweek + horizon) if w in all_gws]
    if not horizon_gws:
        raise RuntimeError(
            f"No predictions for any GW in horizon starting at GW{gameweek}"
        )
    horizon_pred = pred_df.filter(pl.col("gameweek").is_in(horizon_gws))
    candidates_set = select_candidates(
        season_id=season_id,
        horizon_predictions=horizon_pred,
        current_squad=set(state.squad),
        top_n_per_position=25,
        cheap_k_per_position=3,
    )
    candidates_set &= set(valid_players)
    candidates_set |= set(state.squad)  # always keep current squad
    candidates = sorted(candidates_set)

    # 7. Prices + sell-tax for state.squad players
    buy_prices = {p: _resolve_price_at_gw(p, gameweek, prices_by_pgw) for p in candidates}
    sell_prices: dict[int, int] = {}
    for p in candidates:
        current = buy_prices[p]
        basis = state.cost_basis.get(p) if p in state.squad else None
        if basis is not None:
            tax = max(0, (current - basis) // 2)
            sell_prices[p] = current - tax
        else:
            sell_prices[p] = current

    pred_dict = {
        (p, w): pred_by_pgw_filtered.get((p, w), 0.0)
        for p in candidates
        for w in horizon_gws
    }
    eo = eo_for_candidates(candidates)
    teams_dict = {p: teams[p] for p in candidates if p in teams}
    positions_dict = {p: positions.get(p, "MID") for p in candidates}

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
        enable_chips=True,
        full_predictions=pred_df,
        chip_schedule=chip_schedule,
    )

    decisions, meta = solve_rolling_horizon(inputs, time_limit_s=180)

    # 8. Write outputs
    if output_dir is None:
        output_dir = OUTPUT_ROOT / f"season_{season_id}" / f"gw_{gameweek}"
    output_dir.mkdir(parents=True, exist_ok=True)

    md_path = output_dir / "recommendation.md"
    json_path = output_dir / "recommendation.json"

    ctx = RecommendationContext(
        season_id=season_id,
        gameweek=gameweek,
        team_id=team_id,
        state_before={
            "squad": sorted(state.squad),
            "bank_tenths": state.bank,
            "free_transfers": state.free_transfers,
            "chips_used": sorted(state.chips_used),
        },
        chip_schedule=chip_schedule,
        overrides={
            "excluded_player_ids": sorted(overrides.excluded_player_ids),
            "xpts_attenuator": overrides.xpts_attenuator,
        },
        horizon_gws=horizon_gws,
        candidates_count=len(candidates),
    )

    _write_json_sidecar(
        json_path,
        ctx,
        decisions,
        meta,
        positions_dict,
        buy_prices,
        sell_prices,
    )
    _write_markdown(
        md_path,
        ctx,
        decisions,
        meta,
        positions_dict,
        buy_prices,
        sell_prices,
        pred_dict,
    )

    return md_path, json_path


def _write_json_sidecar(
    path: Path,
    ctx: RecommendationContext,
    decisions,
    meta: dict,
    positions: dict[int, str],
    buy_prices: dict[int, int],
    sell_prices: dict[int, int],
) -> None:
    out = {
        "generated_at": dt.datetime.now(dt.UTC).isoformat(),
        "context": asdict(ctx),
        "decisions": {
            "gameweek": decisions.gameweek,
            "squad": sorted(decisions.squad),
            "starting_xi": sorted(decisions.starting_xi),
            "captain": decisions.captain,
            "vice": decisions.vice,
            "transferred_in": sorted(decisions.transferred_in),
            "transferred_out": sorted(decisions.transferred_out),
            "chip_played": decisions.chip_played,
            "hits": decisions.hits,
            "objective_value": decisions.objective_value,
        },
        "solve_meta": meta,
    }
    path.write_text(json.dumps(out, indent=2, default=str))


def _player_label(pid: int, web_names: dict[int, str], positions: dict[int, str]) -> str:
    name = web_names.get(pid, str(pid))
    pos = positions.get(pid, "?")
    return f"{name} ({pos})"


def _write_markdown(
    path: Path,
    ctx: RecommendationContext,
    decisions,
    meta: dict,
    positions: dict[int, str],
    buy_prices: dict[int, int],
    sell_prices: dict[int, int],
    pred_dict: dict[tuple[int, int], float],
) -> None:
    # Resolve web_names from dim_player
    from sqlalchemy import select

    from fpl_bot.db.models import DimPlayer
    from fpl_bot.db.session import session_scope
    pids = set(decisions.squad) | set(decisions.transferred_in) | set(decisions.transferred_out)
    with session_scope() as s:
        rows = s.execute(select(DimPlayer).where(DimPlayer.player_id.in_(pids))).scalars().all()
    web_names = {r.player_id: r.web_name for r in rows}

    # Group by position for display
    def _by_pos(pids_set: set[int]) -> dict[str, list[int]]:
        out = {"GKP": [], "DEF": [], "MID": [], "FWD": []}
        for p in pids_set:
            pos = positions.get(p, "MID")
            if pos in out:
                out[pos].append(p)
        for k in out:
            out[k].sort(key=lambda p: pred_dict.get((p, ctx.gameweek), 0.0), reverse=True)
        return out

    squad_by_pos = _by_pos(set(decisions.squad))
    xi_by_pos = _by_pos(set(decisions.starting_xi))
    bench = sorted(
        set(decisions.squad) - set(decisions.starting_xi),
        key=lambda p: pred_dict.get((p, ctx.gameweek), 0.0),
        reverse=True,
    )

    lines: list[str] = []
    lines.append(f"# GW {ctx.gameweek} Recommendation — season 20{ctx.season_id}/{(ctx.season_id+1)%100:02d}")
    lines.append("")
    lines.append(f"Generated: {dt.datetime.now(dt.UTC).isoformat()}")
    lines.append(f"Solver: {meta.get('termination', 'unknown')} (obj={meta.get('objective', 'NA'):.2f})")
    lines.append("")
    lines.append("## Squad (15 players)")
    for pos in ("GKP", "DEF", "MID", "FWD"):
        ids = squad_by_pos.get(pos, [])
        if ids:
            names = " · ".join(_player_label(p, web_names, positions) for p in ids)
            lines.append(f"- **{pos}**: {names}")
    lines.append("")
    lines.append("## Starting XI")
    for pos in ("GKP", "DEF", "MID", "FWD"):
        ids = xi_by_pos.get(pos, [])
        if ids:
            names = " · ".join(_player_label(p, web_names, positions) for p in ids)
            lines.append(f"- **{pos}**: {names}")
    lines.append("")
    lines.append("## Bench (in order)")
    for i, p in enumerate(bench, start=1):
        lines.append(f"{i}. {_player_label(p, web_names, positions)}  (xPts: {pred_dict.get((p, ctx.gameweek), 0.0):.2f})")
    lines.append("")

    cap_name = web_names.get(decisions.captain, str(decisions.captain)) if decisions.captain else "—"
    vice_name = web_names.get(decisions.vice, str(decisions.vice)) if decisions.vice else "—"
    lines.append(f"## Captain: **{cap_name}** (vice: {vice_name})")
    lines.append("")

    lines.append("## Transfers")
    if decisions.transferred_in or decisions.transferred_out:
        for p in sorted(decisions.transferred_out):
            lines.append(f"- OUT: {_player_label(p, web_names, positions)}  (sell £{sell_prices.get(p, 0)/10:.1f}m)")
        for p in sorted(decisions.transferred_in):
            lines.append(f"- IN:  {_player_label(p, web_names, positions)}  (buy £{buy_prices.get(p, 0)/10:.1f}m)")
        lines.append(f"- Hits: **{decisions.hits}** (× -4 pts = {-4 * decisions.hits} pts)")
    else:
        lines.append("- No transfers.")
    lines.append("")

    chip_played = decisions.chip_played or "—"
    lines.append(f"## Chip: **{chip_played}**")
    lines.append("")
    lines.append("## Season chip plan")
    for slot, gw in ctx.chip_schedule.items():
        marker = " ← this GW" if gw == ctx.gameweek else ""
        lines.append(f"- {slot}: GW{gw}{marker}")
    lines.append("")

    if ctx.overrides["excluded_player_ids"] or ctx.overrides["xpts_attenuator"]:
        lines.append("## Live status overrides applied")
        if ctx.overrides["excluded_player_ids"]:
            lines.append(f"- Excluded (injured/suspended/unavailable): {len(ctx.overrides['excluded_player_ids'])} players")
        if ctx.overrides["xpts_attenuator"]:
            lines.append(f"- Attenuated (doubtful): {len(ctx.overrides['xpts_attenuator'])} players")

    path.write_text("\n".join(lines))
