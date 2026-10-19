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
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import polars as pl

from fpl_bot.db import pit
from fpl_bot.eval.milp_backtest import (
    _per_player_per_gw_appearances,
    _per_player_per_gw_predictions,
    _per_player_per_gw_prices,
    _resolve_price_at_gw,
    _team_id_per_player_for_season,
)
from fpl_bot.eval.xpts_eval import run_predict_only
from fpl_bot.features import manual_overrides
from fpl_bot.live.market_state import apply_current_market, selling_price
from fpl_bot.live.state_builder import (
    LiveStatusOverrides,
    load_status_overrides,
    load_user_sell_prices,
    load_user_state,
)
from fpl_bot.models.validation import TRAINING_POLICY_VERSION
from fpl_bot.optim.candidate_filter import select_candidates
from fpl_bot.optim.chip_scheduler import (
    free_hit_gws,
    horizon_before_free_hit,
    make_chip_schedule,
)
from fpl_bot.optim.eo import eo_for_candidates
from fpl_bot.optim.fixture_analytics import load_fixture_analytics
from fpl_bot.optim.milp import MilpInputs, solve_rolling_horizon
from fpl_bot.optim.prediction_postprocess import apply_prediction_postprocessing
from fpl_bot.optim.state import (
    ALL_CHIP_SLOTS,
    CHIPS_FIRST_HALF,
    SECOND_HALF_FIRST_GW,
    BacktestState,
)

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
    *,
    news_attenuator: dict[tuple[int, int], float] | None = None,
    unexclude_returning: set[int] | None = None,
) -> dict[tuple[int, int], float]:
    """Multiply pred by attenuator for doubtful players; drop excluded.

    Phase 7 2.1: `news_attenuator` is a per-(player, gw) multiplier from
    parsed news text (e.g., 0.0 for the GWs a player is out, no entry for
    GWs after their return-date). `unexclude_returning` lists players who
    are currently in excluded_player_ids but have a return-date inside the
    horizon — we KEEP them as candidates so the MILP can pick them for
    post-return GWs.
    """
    out: dict[tuple[int, int], float] = {}
    unexcl = unexclude_returning or set()
    for (pid, gw), v in pred_by_pgw.items():
        if pid in overrides.excluded_player_ids and pid not in unexcl:
            continue
        mult = overrides.xpts_attenuator.get(pid, 1.0)
        if news_attenuator is not None:
            mult *= news_attenuator.get((pid, gw), 1.0)
        out[(pid, gw)] = v * mult
    return out


def generate_recommendation(
    *,
    season_id: int,
    gameweek: int,
    team_id: int,
    train_seasons: list[int],
    horizon: int = 6,
    rho: float = 0.0,
    alpha: float = 0.8,
    beta: float = 0.0,
    n_iterations: int = 500,
    cache_predictions: bool = True,
    cache_dir: Path = Path("data/cache/xpts_predictions"),
    output_dir: Path | None = None,
    state_override: BacktestState | None = None,
) -> tuple[Path, Path]:
    """Run the full Phase 6 pipeline and write recommendation files.

    Returns (markdown_path, json_path).
    ``cache_predictions`` is retained for caller compatibility; live horizons
    always regenerate because historical evaluation caches lack freshness keys.
    """
    # 1. Generate fresh predictions from upcoming-fixture feature rows.
    eval_df: pl.DataFrame | None = None
    # Historical evaluation caches are not live forecasts: their keys do not
    # identify current results, odds, news, overrides, or feature code. Always
    # regenerate the live horizon rather than silently serving stale values.

    # A live cache miss needs only upcoming predictions. Running a historical
    # evaluation fold here trains the stack twice and cannot fill future GWs.

    pred_by_pgw: dict[tuple[int, int], float] = {}
    if eval_df is not None and not eval_df.is_empty():
        pred_by_pgw = _per_player_per_gw_predictions(eval_df)

    cached_gws = sorted({gw for (_, gw) in pred_by_pgw if gw > 0})
    target_horizon_gws = list(range(gameweek, min(39, gameweek + horizon)))
    from fpl_bot.live.news_extract import (
        build_pred_attenuator,
        latest_news_per_player,
        return_gw_for_date,
    )

    # The current extractor emits explicit availability masks (principally
    # pre-return 0.0). Feed them into upcoming-fixture minutes probabilities;
    # the later xPts attenuator remains as a fallback for cached predictions.
    news_attenuator = build_pred_attenuator(
        season_id=season_id, horizon_gws=target_horizon_gws
    )
    missing_gws = [w for w in target_horizon_gws if w not in cached_gws]

    if missing_gws:
        # Phase 6 v2: predict-only for upcoming fixtures
        from sqlalchemy import select as sa_select

        from fpl_bot.db.models import DimFixture
        from fpl_bot.db.session import session_scope
        with session_scope() as s:
            upcoming_ids = [
                int(r[0])
                for r in s.execute(
                    sa_select(DimFixture.fixture_id).where(
                        (DimFixture.season_id == season_id)
                        & (DimFixture.gameweek.in_(missing_gws))
                    )
                ).all()
            ]
        if upcoming_ids:
            # Phase 8 shadow log lands beside the recommendation for this GW.
            shadow_dir = output_dir or (
                OUTPUT_ROOT / f"season_{season_id}" / f"gw_{gameweek}"
            )
            # Manual team-news overrides (configs/expected_minutes_overrides.yaml).
            # Resolved here so an unresolvable name fails before the expensive
            # solve rather than after it.
            expected_minutes = manual_overrides.resolve_expected_minutes(
                season_id=season_id,
                gameweeks=missing_gws,
                web_name_to_player_id=pit.web_name_to_player_id(),
            )
            if expected_minutes:
                console_note = (
                    f"applying {len(expected_minutes)} manual expected-minutes override(s)"
                )
                print(console_note)
            predict_df = run_predict_only(
                test_season=season_id,
                train_seasons=train_seasons,
                upcoming_fixture_ids=upcoming_ids,
                availability_by_pgw=news_attenuator,
                expected_minutes_by_pgw=expected_minutes,
                n_iterations=n_iterations,
                seed=42,
                prop_shadow_out=shadow_dir / "player_prop_shadow.parquet",
            )
            if not predict_df.is_empty():
                if eval_df is None or eval_df.is_empty():
                    eval_df = predict_df
                else:
                    # Normalize kickoff_utc TZ so both paths concat cleanly
                    # (cache typically has session-TZ Datetime; predict_df
                    # is UTC). Push both to UTC.
                    if "kickoff_utc" in eval_df.columns:
                        col = eval_df.schema["kickoff_utc"]
                        if isinstance(col, pl.Datetime) and col.time_zone is not None:
                            eval_df = eval_df.with_columns(
                                pl.col("kickoff_utc").dt.convert_time_zone("UTC")
                            )
                    eval_df = pl.concat(
                        [eval_df, predict_df], how="diagonal_relaxed"
                    )
                # Refresh pred_by_pgw with combined data
                pred_by_pgw = _per_player_per_gw_predictions(eval_df)

    if not pred_by_pgw:
        raise RuntimeError(f"Could not generate predictions for season {season_id}")

    pred_by_pgw = apply_prediction_postprocessing(
        pred_by_pgw,
        season_id=season_id,
        train_seasons=train_seasons,
        cache_dir=cache_dir,
        appearance_exposure_by_pgw=(
            _per_player_per_gw_appearances(eval_df) if eval_df is not None else None
        ),
    )

    all_gws = sorted({gw for (_, gw) in pred_by_pgw if gw > 0})
    all_players = sorted({pid for (pid, _) in pred_by_pgw})

    # 2. Load user state from latest snapshot
    if state_override is not None and (state_override.season_id != season_id or state_override.gameweek != gameweek):
        raise ValueError("The supplied bot state must match the forecast season and gameweek")
    state = state_override or load_user_state(season_id=season_id, gameweek=gameweek, team_id=team_id)
    exact_sell_prices = {} if state_override is not None else load_user_sell_prices(
        season_id=season_id, gameweek=gameweek, team_id=team_id
    )

    # 3. Status overrides — applied BEFORE candidate filter.
    # Phase 7 2.1: parse FPL news text for return dates; players currently
    # excluded but returning inside the horizon are un-excluded (kept as
    # candidates) with their pre-return GW pred zeroed.
    overrides = load_status_overrides(candidate_player_ids=all_players)
    horizon_gws_for_news = target_horizon_gws
    # Un-exclude players who have a return-date strictly inside the horizon
    unexclude_returning: set[int] = set()
    for ex in latest_news_per_player(season_id=season_id):
        if ex.return_date is None or ex.out_for_season:
            continue
        rg = return_gw_for_date(ex.return_date, season_id)
        if rg is None or rg > max(horizon_gws_for_news):
            continue
        if ex.player_id in overrides.excluded_player_ids:
            unexclude_returning.add(ex.player_id)
    pred_by_pgw_filtered = _attenuate_predictions(
        pred_by_pgw,
        overrides,
        news_attenuator=news_attenuator,
        unexclude_returning=unexclude_returning,
    )

    # Current bootstrap is authoritative for live clubs and purchase prices,
    # including owned players absent from the prediction table.
    market_players = sorted(set(all_players) | set(state.squad))
    teams = _team_id_per_player_for_season(season_id, market_players)
    prices_by_pgw = _per_player_per_gw_prices(season_id, market_players)
    status_team_price = pit.latest_team_and_price_per_player(season_id)
    apply_current_market(teams, prices_by_pgw, status_team_price, target_horizon_gws)
    positions = (
        pit.all_player_positions()
        .to_pandas()
        .set_index("player_id")["position_code"]
        .to_dict()
    )
    valid_players = [
        p
        for p in all_players
        if p in positions and p in teams
        and (p not in overrides.excluded_player_ids or p in unexclude_returning)
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
    horizon_gws = horizon_before_free_hit(horizon_gws, chip_schedule)
    if not horizon_gws:
        raise RuntimeError(
            f"No predictions for any GW in horizon starting at GW{gameweek}"
        )
    is_current_free_hit = gameweek in free_hit_gws(chip_schedule)
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

    # 7. Prices + sell-tax for state.squad players. Phase 7 2.2: project
    # sell_prices to horizon-end using the price-change predictor so the
    # MILP's terminal-value chain rewards holding rising players. buy_prices
    # stay at current (we pay today).
    buy_prices = {p: _resolve_price_at_gw(p, gameweek, prices_by_pgw) for p in candidates}
    sell_prices: dict[int, int] = {}
    try:
        from fpl_bot.live.price_predict import (
            predict_one_step_price_deltas,
            projected_horizon_sell_price,
        )
        deltas = predict_one_step_price_deltas(
            season_id=season_id,
            gameweek=gameweek,
            candidates=list(candidates),
            train_seasons=train_seasons,
        )
        price_predict_available = True
    except Exception as exc:
        # Fall back to flat sell_prices if the predictor can't run (no
        # training data, missing features). Logged so it's visible in
        # the smoke output.
        print(f"  price-change predictor unavailable: {exc}; sell_prices = current")
        deltas = {p: 0.0 for p in candidates}
        price_predict_available = False
    for p in candidates:
        current = buy_prices[p]
        delta = deltas.get(p, 0.0)
        projected_horizon_end = (
            projected_horizon_sell_price(current, delta, horizon)
            if price_predict_available
            else current
        )
        basis = state.cost_basis.get(p) if p in state.squad else None
        if basis is not None:
            if price_predict_available:
                sell_prices[p] = selling_price(projected_horizon_end, basis)
            else:
                sell_prices[p] = exact_sell_prices.get(p, selling_price(current, basis))
        else:
            sell_prices[p] = projected_horizon_end

    pred_dict = {
        (p, w): pred_by_pgw_filtered.get((p, w), 0.0)
        for p in candidates
        for w in horizon_gws
    }
    eo = eo_for_candidates(candidates) if rho else {}
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
        full_predictions=None if is_current_free_hit else pred_df,
        chip_schedule=chip_schedule,
    )

    decisions, meta = solve_rolling_horizon(inputs, time_limit_s=180)
    if decisions.transferred_in and decisions.chip_played not in {"WC1", "WC2", "FH1", "FH2"}:
        meta["hold_comparison"] = _hold_comparison(inputs, meta)

    meta["rho"] = rho
    meta["alpha"] = alpha
    meta["beta"] = beta
    from sqlalchemy import func as sa_func
    from sqlalchemy import select

    from fpl_bot.db.models import DimFixture, FactMarketXg, FactPlayerMatch
    from fpl_bot.db.session import session_scope

    with session_scope() as s:
        market_coverage = s.execute(
            select(
                DimFixture.gameweek,
                sa_func.count(sa_func.distinct(DimFixture.fixture_id)),
                sa_func.count(sa_func.distinct(FactMarketXg.fixture_id)),
            ).outerjoin(FactMarketXg, FactMarketXg.fixture_id == DimFixture.fixture_id)
            .where(DimFixture.season_id == season_id, DimFixture.gameweek.in_(target_horizon_gws))
            .group_by(DimFixture.gameweek).order_by(DimFixture.gameweek)
        ).all()
        latest_results_gw = s.scalar(
            select(sa_func.max(DimFixture.gameweek))
            .join(FactPlayerMatch, FactPlayerMatch.fixture_id == DimFixture.fixture_id)
            .where(DimFixture.season_id == season_id)
        )
    meta["data_quality"] = {
        "training_policy": TRAINING_POLICY_VERSION,
        "completed_results_through_gw": latest_results_gw,
        "simulation_iterations": n_iterations,
        "train_seasons": train_seasons,
        "bookmaker_coverage": {
            str(gw): {"fixtures": int(total), "priced_fixtures": int(priced)}
            for gw, total, priced in market_coverage
        },
        "gameweeks_with_market_fallback": [
            int(gw) for gw, total, priced in market_coverage if priced < total
        ],
    }

    # 8. Write outputs
    if output_dir is None:
        output_dir = OUTPUT_ROOT / f"season_{season_id}" / f"gw_{gameweek}"
    output_dir.mkdir(parents=True, exist_ok=True)

    if eval_df is not None:
        eval_df.write_parquet(output_dir / "fixture_predictions.parquet")
    forecast_rows = [
        {"player_id": pid, "gameweek": gw, "expected_points": value}
        for (pid, gw), value in sorted(pred_by_pgw_filtered.items())
        if gw in target_horizon_gws
    ]
    (output_dir / "player_predictions.json").write_text(
        json.dumps(forecast_rows, indent=2)
    )

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
        pred_dict,
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


def _hold_comparison(inputs: MilpInputs, meta: dict) -> dict | None:
    """Modelled horizon value of this week's moves over rolling the transfer.

    Both solves stop within CBC's 1% relative gap, so differences inside that
    tolerance do not separate the two plans.
    """
    try:
        _, hold_meta = solve_rolling_horizon(
            replace(inputs, hold_first_week=True), time_limit_s=120
        )
    except RuntimeError:
        return None
    plan, hold = float(meta["objective"]), float(hold_meta["objective"])
    return {
        "plan_objective": round(plan, 2),
        "hold_objective": round(hold, 2),
        "gain": round(plan - hold, 2),
        "tolerance": round(0.01 * max(abs(plan), abs(hold)), 2),
        "weeks": len(inputs.horizon_weeks),
    }


def _write_json_sidecar(
    path: Path,
    ctx: RecommendationContext,
    decisions,
    meta: dict,
    positions: dict[int, str],
    buy_prices: dict[int, int],
    sell_prices: dict[int, int],
    pred_dict: dict[tuple[int, int], float],
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
            "bench_order": sorted(
                decisions.squad - decisions.starting_xi,
                key=lambda p: (-pred_dict.get((p, ctx.gameweek), 0.0), p),
            ),
            "expected_points_by_player": {
                str(p): pred_dict.get((p, ctx.gameweek), 0.0)
                for p in sorted(decisions.squad)
            },
            "expected_gw_points": sum(
                pred_dict.get((p, ctx.gameweek), 0.0)
                for p in (decisions.squad if decisions.chip_played in ("BB1", "BB2") else decisions.starting_xi)
            ) + (2 if decisions.chip_played in ("TC1", "TC2") else 1)
            * pred_dict.get((decisions.captain, ctx.gameweek), 0.0) - 4 * decisions.hits,
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
    # Same key as the JSON `bench_order` that retrospectives score against.
    bench = sorted(
        set(decisions.squad) - set(decisions.starting_xi),
        key=lambda p: (-pred_dict.get((p, ctx.gameweek), 0.0), p),
    )

    lines: list[str] = []
    lines.append(f"# GW {ctx.gameweek} Recommendation — season 20{ctx.season_id}/{(ctx.season_id+1)%100:02d}")
    lines.append("")
    lines.append(f"Generated: {dt.datetime.now(dt.UTC).isoformat()}")
    lines.append(f"Solver: {meta.get('termination', 'unknown')} (obj={meta.get('objective', 'NA'):.2f})")
    quality = meta.get("data_quality", {})
    if quality:
        lines.append(f"Completed results through GW{quality['completed_results_through_gw']}.")
        fallback = quality.get("gameweeks_with_market_fallback", [])
        if fallback:
            lines.append("Bookmaker coverage is incomplete for " + ", ".join(f"GW{gw}" for gw in fallback) + "; those forecasts use market defaults.")
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
    # The scheduler plans all eight slots from the fixture calendar alone; it
    # does not know which ones the manager has already spent. The MILP does
    # (it fixes an already-used slot's activation to 0), so an unmarked plan
    # line here is purely a reporting lie — it reads as "you still have a
    # Bench Boost banked for GW19" when that chip is gone. Mark them.
    chips_used = set(ctx.state_before.get("chips_used") or ())
    for slot, gw in ctx.chip_schedule.items():
        if slot in chips_used:
            lines.append(f"- {slot}: ~~GW{gw}~~ — ALREADY PLAYED, not available")
            continue
        marker = " ← this GW" if gw == ctx.gameweek else ""
        lines.append(f"- {slot}: GW{gw}{marker}")
    # The scheduler places a Free Hit only on a known blank GW, so a banked
    # chip can be absent from the plan entirely. Silence there reads as
    # "nothing to decide" while the chip quietly expires.
    for slot in ALL_CHIP_SLOTS:
        if slot in chips_used or slot in ctx.chip_schedule:
            continue
        expires = SECOND_HALF_FIRST_GW - 1 if slot in CHIPS_FIRST_HALF else 38
        if ctx.gameweek <= expires:
            lines.append(
                f"- {slot}: unscheduled, still available — lost if unused by GW{expires}"
            )
    lines.append("")

    if ctx.overrides["excluded_player_ids"] or ctx.overrides["xpts_attenuator"]:
        lines.append("## Live status overrides applied")
        if ctx.overrides["excluded_player_ids"]:
            lines.append(f"- Excluded (injured/suspended/unavailable): {len(ctx.overrides['excluded_player_ids'])} players")
        if ctx.overrides["xpts_attenuator"]:
            lines.append(f"- Attenuated (doubtful): {len(ctx.overrides['xpts_attenuator'])} players")

    path.write_text("\n".join(lines))
