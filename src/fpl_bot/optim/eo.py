"""Effective ownership via `fpl_api_approx` per Phase 0 round-2 fallback.

Approximation: top-10k EO ≈ overall ownership × captain-percentage heuristic.
The captain term is approximated as a fixed multiplier (the "engagement"
factor) applied to overall ownership: high-priced premiums are
disproportionately captained by top managers.

For backtest GWs (where we don't have historical EO snapshots in
fact_player_status), v1 uses TODAY's snapshot — known approximation
documented in Phase 3 §1. Live deployment uses current snapshot directly.
"""
from __future__ import annotations

import polars as pl
from sqlalchemy import func, select

from fpl_bot.db.models import FactPlayerStatus
from fpl_bot.db.session import session_scope

# Heuristic: top-10k EO ≈ overall_ownership × CAPTAIN_FACTOR for premiums,
# plus a baseline. For Phase 3 v1 we keep it simple — return overall ownership
# directly as the EO proxy. The MILP's ρ-tunability handles calibration.
DEFAULT_CAPTAIN_BUMP = 1.0  # multiplier on overall ownership; ρ tuning absorbs misfit


def overall_eo_snapshot() -> pl.DataFrame:
    """Returns latest-snapshot overall EO per player.

    Columns: player_id, eo (in [0, 100] same units as
    fact_player_status.selected_by_percent).
    """
    with session_scope() as s:
        latest = (
            select(
                FactPlayerStatus.player_id,
                func.max(FactPlayerStatus.recorded_at).label("max_rec"),
            )
            .group_by(FactPlayerStatus.player_id)
            .subquery()
        )
        rows = s.execute(
            select(
                FactPlayerStatus.player_id,
                FactPlayerStatus.selected_by_percent,
            ).join(
                latest,
                (latest.c.player_id == FactPlayerStatus.player_id)
                & (latest.c.max_rec == FactPlayerStatus.recorded_at),
            )
        ).all()

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


def eo_for_candidates(
    candidate_player_ids: list[int],
) -> dict[int, float]:
    """Map of player_id → eo value (decimal in [0, 1])."""
    df = overall_eo_snapshot()
    if df.is_empty():
        return dict.fromkeys(candidate_player_ids, 0.0)
    df = df.filter(pl.col("player_id").is_in(candidate_player_ids))
    out: dict[int, float] = {}
    for r in df.iter_rows(named=True):
        # selected_by_percent is in [0, 100]; convert to fraction
        out[r["player_id"]] = float(r["eo"]) / 100.0
    # Players with no snapshot → 0 EO (treat as anti-template differential)
    for pid in candidate_player_ids:
        out.setdefault(pid, 0.0)
    return out
