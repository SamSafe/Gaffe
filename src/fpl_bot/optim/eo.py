"""Effective ownership for the MILP ρ-EO penalty.

Phase 7 1.3: when a LiveFPL `fact_eo_snapshot` row exists for a player at
rank_band='top10k', we use it (effective_ownership column already includes
captaincy weight). Otherwise fall back to the original Phase 3 v1
approximation: overall selected_by_percent from `fact_player_status`.

For backtest GWs (where we don't have historical EO snapshots), v1 uses
TODAY's snapshot — known approximation documented in Phase 3 §1. Live
deployment uses current snapshot directly.

Blend: when both sources are available for a player, the result is
TOP10K_BLEND_WEIGHT × top10k_EO + (1 − TOP10K_BLEND_WEIGHT) × overall.
The ρ-EO term penalizes alignment with template; top10k is the cohort
the bot wants to beat, so weighting it heavily makes the bot more
differential against the top template.
"""
from __future__ import annotations

import polars as pl
from sqlalchemy import func, select

from fpl_bot.db.models import FactEoSnapshot, FactPlayerStatus
from fpl_bot.db.session import session_scope

# Heuristic: top-10k EO ≈ overall_ownership × CAPTAIN_FACTOR for premiums,
# plus a baseline. For Phase 3 v1 we keep it simple — return overall ownership
# directly as the EO proxy. The MILP's ρ-tunability handles calibration.
DEFAULT_CAPTAIN_BUMP = 1.0  # multiplier on overall ownership; ρ tuning absorbs misfit

# Blend weight on top10k vs overall EO when both are available. Heavier on
# top10k since the ρ-EO term in the MILP penalizes alignment with template.
TOP10K_BLEND_WEIGHT = 0.7


def overall_eo_snapshot(season_id: int | None = None) -> pl.DataFrame:
    """Returns latest-snapshot overall EO per player.

    Columns: player_id, eo (in [0, 100] same units as
    fact_player_status.selected_by_percent).

    `season_id` restricts the snapshot to one season. Live callers leave it
    None (the newest snapshot IS the live season). A BACKTEST must pass its
    test season: without it the newest row wins globally, so a 2025-26
    backtest would be scored against 2026-27 ownership — a template built
    from a different player population, with promoted and relegated clubs
    swapped out. That is not a mild approximation, it is the wrong template.
    """
    with session_scope() as s:
        latest_q = select(
            FactPlayerStatus.player_id,
            func.max(FactPlayerStatus.recorded_at).label("max_rec"),
        )
        if season_id is not None:
            latest_q = latest_q.where(FactPlayerStatus.season_id == season_id)
        latest = latest_q.group_by(FactPlayerStatus.player_id).subquery()
        rows_q = select(
            FactPlayerStatus.player_id,
            FactPlayerStatus.selected_by_percent,
        ).join(
            latest,
            (latest.c.player_id == FactPlayerStatus.player_id)
            & (latest.c.max_rec == FactPlayerStatus.recorded_at),
        )
        if season_id is not None:
            rows_q = rows_q.where(FactPlayerStatus.season_id == season_id)
        rows = s.execute(rows_q).all()

    if not rows:
        return pl.DataFrame()
    return pl.DataFrame(
        [
            {
                "player_id": r.player_id,
                "eo": float(r.selected_by_percent or 0.0) * DEFAULT_CAPTAIN_BUMP,
            }
            for r in rows
        ]
    )


def latest_top10k_eo(season_id: int | None = None) -> dict[int, float]:
    """Return {player_id: top10k_effective_ownership_fraction} for the most
    recent (event_time) LiveFPL snapshot. Empty when no rows exist.

    `season_id` scopes the snapshot to one season — see `overall_eo_snapshot`
    for why a backtest must pass it.
    """
    with session_scope() as s:
        latest_q = (
            select(
                FactEoSnapshot.player_id,
                func.max(FactEoSnapshot.event_time).label("max_t"),
            )
            .where(FactEoSnapshot.rank_band == "top10k")
            .where(FactEoSnapshot.source == "livefpl")
        )
        if season_id is not None:
            latest_q = latest_q.where(FactEoSnapshot.season_id == season_id)
        latest = latest_q.group_by(FactEoSnapshot.player_id).subquery()
        rows_q = (
            select(
                FactEoSnapshot.player_id,
                FactEoSnapshot.effective_ownership,
            ).join(
                latest,
                (latest.c.player_id == FactEoSnapshot.player_id)
                & (latest.c.max_t == FactEoSnapshot.event_time),
            )
            .where(FactEoSnapshot.rank_band == "top10k")
            .where(FactEoSnapshot.source == "livefpl")
        )
        if season_id is not None:
            rows_q = rows_q.where(FactEoSnapshot.season_id == season_id)
        rows = s.execute(rows_q).all()
    return {int(r.player_id): float(r.effective_ownership) / 100.0 for r in rows}


def eo_for_candidates(
    candidate_player_ids: list[int],
    season_id: int | None = None,
) -> dict[int, float]:
    """Map of player_id → effective_ownership fraction in [0, 1].

    Blends LiveFPL top10k EO with overall `selected_by_percent` when both
    are available. Falls back to overall alone when LiveFPL hasn't snapshot
    the player. Players with no signal → 0.0 (treated as anti-template).

    Pass `season_id` from a backtest so the template comes from the season
    being tested rather than from today.
    """
    overall_df = overall_eo_snapshot(season_id)
    overall: dict[int, float] = {}
    if not overall_df.is_empty():
        sub = overall_df.filter(pl.col("player_id").is_in(candidate_player_ids))
        overall = {int(r["player_id"]): float(r["eo"]) / 100.0 for r in sub.iter_rows(named=True)}
    top10k = latest_top10k_eo(season_id)
    out: dict[int, float] = {}
    for pid in candidate_player_ids:
        o = overall.get(pid)
        t = top10k.get(pid)
        if t is not None and o is not None:
            out[pid] = TOP10K_BLEND_WEIGHT * t + (1 - TOP10K_BLEND_WEIGHT) * o
        elif t is not None:
            out[pid] = t
        elif o is not None:
            out[pid] = o
        else:
            out[pid] = 0.0
    return out
