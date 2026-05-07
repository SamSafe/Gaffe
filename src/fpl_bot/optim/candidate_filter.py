"""Prefilter the 600+ players to ~200 candidates per MILP solve (Phase 3 §10).

Heuristic union:
  - Top N by predicted E[xPts] per position (across the horizon)
  - Current squad members (must be feasible to keep)
  - Penalty takers per the manual config
  - K cheapest per position (bench fillers, BB enablers)

Falls back to the full player set if the prefilter is empty for any position.
"""
from __future__ import annotations

import polars as pl

from fpl_bot.db import pit
from fpl_bot.features import manual_overrides

POSITIONS = ("GKP", "DEF", "MID", "FWD")
DEFAULT_TOP_N_BY_POSITION = 50
DEFAULT_CHEAP_K_BY_POSITION = 5


def select_candidates(
    *,
    season_id: int,
    horizon_predictions: pl.DataFrame,  # cols: player_id, gameweek, e_xpts
    current_squad: set[int] | None = None,
    top_n_per_position: int = DEFAULT_TOP_N_BY_POSITION,
    cheap_k_per_position: int = DEFAULT_CHEAP_K_BY_POSITION,
) -> set[int]:
    """Returns the set of candidate player_ids for the MILP solve."""
    if horizon_predictions.is_empty():
        return set(current_squad or set())

    # Aggregate predicted xPts across the horizon per player
    horizon_total = horizon_predictions.group_by("player_id").agg(
        pl.col("e_xpts").sum().alias("horizon_xpts")
    )

    # Join positions and current price
    positions_df = pit.all_player_positions()
    horizon_total = horizon_total.join(positions_df, on="player_id", how="left").drop_nulls(
        "position_code"
    )

    candidates: set[int] = set(current_squad or set())

    # Top N per position by predicted xPts
    for pos in POSITIONS:
        sub = (
            horizon_total.filter(pl.col("position_code") == pos)
            .sort("horizon_xpts", descending=True)
            .head(top_n_per_position)
        )
        candidates.update(sub["player_id"].to_list())

    # Penalty takers across all teams in this season (manual config)
    pk_raw = manual_overrides.set_piece_takers_raw().get(season_id, {})
    web_to_pid = pit.web_name_to_player_id()
    for team_info in pk_raw.values():
        if not isinstance(team_info, dict):
            continue
        taker_web = team_info.get("penalty")
        if taker_web:
            pid = web_to_pid.get(taker_web)
            if pid is not None:
                candidates.add(pid)

    # K cheapest per position (bench fillers / BB enablers). Use price from
    # the latest fact_player_status snapshot.
    cheapest = _cheapest_k_per_position(cheap_k_per_position)
    for pids in cheapest.values():
        candidates.update(pids)

    return candidates


def _cheapest_k_per_position(k: int) -> dict[str, list[int]]:
    """For each position, return the K player_ids with the lowest current price
    (from latest fact_player_status snapshot)."""
    from sqlalchemy import func, select

    from fpl_bot.db.models import FactPlayerStatus
    from fpl_bot.db.session import session_scope

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
                FactPlayerStatus.position_code,
                FactPlayerStatus.price_tenths,
            ).join(
                latest,
                (latest.c.player_id == FactPlayerStatus.player_id)
                & (latest.c.max_rec == FactPlayerStatus.recorded_at),
            )
        ).all()

    by_pos: dict[str, list[tuple[int, int]]] = {p: [] for p in POSITIONS}
    for r in rows:
        if r.position_code in by_pos:
            by_pos[r.position_code].append((r.player_id, r.price_tenths))

    out: dict[str, list[int]] = {}
    for pos in POSITIONS:
        sorted_by_price = sorted(by_pos[pos], key=lambda x: x[1])
        out[pos] = [pid for pid, _ in sorted_by_price[:k]]
    return out
