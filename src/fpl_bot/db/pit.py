"""Point-in-time query layer. The ONLY sanctioned read path for downstream code.

Every function takes an explicit `as_of` datetime and translates it to
`WHERE recorded_at <= as_of` plus a window pick of the latest row per natural
key. Modules under fpl_bot.features / fpl_bot.models / fpl_bot.scenarios may
ONLY import from this module to access fact data.

See docs/design/phase0.md §3.3 for the full API surface.
"""
from __future__ import annotations

import datetime as dt
from typing import Any

import polars as pl
from sqlalchemy import select

from fpl_bot.db.models import (
    FactPlayerMatch,
    FactPlayerMatchEvent,
    FactPlayerStatus,
)
from fpl_bot.db.session import session_scope


def player_status_as_of(player_id: int, as_of: dt.datetime) -> dict[str, Any] | None:
    """Latest player_status row with recorded_at <= as_of."""
    with session_scope() as s:
        row = s.execute(
            select(FactPlayerStatus)
            .where(FactPlayerStatus.player_id == player_id)
            .where(FactPlayerStatus.recorded_at <= as_of)
            .order_by(FactPlayerStatus.recorded_at.desc())
            .limit(1)
        ).scalar_one_or_none()
        if row is None:
            return None
        return {
            "player_id": row.player_id,
            "season_id": row.season_id,
            "position_code": row.position_code,
            "team_id": row.team_id,
            "price_tenths": row.price_tenths,
            "status_code": row.status_code,
            "news": row.news,
            "chance_of_playing_next_round": row.chance_of_playing_next_round,
            "selected_by_percent": float(row.selected_by_percent)
            if row.selected_by_percent is not None
            else None,
            "event_time": row.event_time,
            "recorded_at": row.recorded_at,
        }


def squad_market_as_of(season_id: int, as_of: dt.datetime) -> pl.DataFrame:
    """All players' latest status as of `as_of`. NOT IMPLEMENTED — Phase 1B."""
    raise NotImplementedError("squad_market_as_of: Phase 1B (needs window query helper)")


def player_match_history(player_id: int, before: dt.datetime) -> pl.DataFrame:
    """All completed matches for a player with kickoff < before."""
    with session_scope() as s:
        rows = s.execute(
            select(FactPlayerMatch)
            .where(FactPlayerMatch.player_id == player_id)
            .where(FactPlayerMatch.recorded_at <= before)
        ).scalars().all()
    if not rows:
        return pl.DataFrame()
    return pl.DataFrame(
        [
            {
                "player_id": r.player_id,
                "fixture_id": r.fixture_id,
                "minutes": r.minutes,
                "goals": r.goals,
                "assists": r.assists,
                "clean_sheet": r.clean_sheet,
                "total_points": r.total_points,
            }
            for r in rows
        ]
    )


def all_player_match_with_kickoff(
    season_ids: list[int] | None = None,
) -> pl.DataFrame:
    """Bulk fetch: every fact_player_match × dim_fixture row, time-keyed by kickoff_utc.

    Used for backtest feature building. Time-cutoff filtering for PIT correctness
    is the caller's responsibility — typically polars window functions over
    per-player time-sorted groupings (shift / rolling) that ensure each row's
    features use only data from earlier rows.
    """
    from fpl_bot.db.models import DimFixture as _DimFixture

    with session_scope() as s:
        stmt = (
            select(
                FactPlayerMatch.player_id,
                FactPlayerMatch.fixture_id,
                FactPlayerMatch.minutes,
                FactPlayerMatch.goals,
                FactPlayerMatch.assists,
                FactPlayerMatch.clean_sheet,
                FactPlayerMatch.total_points,
                _DimFixture.season_id,
                _DimFixture.gameweek,
                _DimFixture.kickoff_utc,
                _DimFixture.home_team_id,
                _DimFixture.away_team_id,
            ).join(_DimFixture, _DimFixture.fixture_id == FactPlayerMatch.fixture_id)
        )
        if season_ids is not None:
            stmt = stmt.where(_DimFixture.season_id.in_(season_ids))
        rows = s.execute(stmt).all()

    if not rows:
        return pl.DataFrame()
    return pl.DataFrame(
        [
            {
                "player_id": r.player_id,
                "fixture_id": r.fixture_id,
                "minutes": r.minutes,
                "goals": r.goals,
                "assists": r.assists,
                "clean_sheet": r.clean_sheet,
                "total_points": r.total_points,
                "season_id": r.season_id,
                "gameweek": r.gameweek,
                "kickoff_utc": r.kickoff_utc,
                "home_team_id": r.home_team_id,
                "away_team_id": r.away_team_id,
            }
            for r in rows
        ]
    )


def all_player_positions() -> pl.DataFrame:
    """Latest position_code per player from fact_player_status (current snapshot).

    Treated as static metadata — positions change rarely. Returns
    columns: player_id, position_code.
    """
    from sqlalchemy import func as _func

    with session_scope() as s:
        latest = (
            select(
                FactPlayerStatus.player_id,
                _func.max(FactPlayerStatus.recorded_at).label("max_rec"),
            )
            .group_by(FactPlayerStatus.player_id)
            .subquery()
        )
        rows = s.execute(
            select(FactPlayerStatus.player_id, FactPlayerStatus.position_code)
            .join(
                latest,
                (latest.c.player_id == FactPlayerStatus.player_id)
                & (latest.c.max_rec == FactPlayerStatus.recorded_at),
            )
        ).all()

    if not rows:
        return pl.DataFrame()
    return pl.DataFrame(
        [{"player_id": r.player_id, "position_code": r.position_code} for r in rows]
    )


def market_xg_for_fixture(fixture_id: int, as_of: dt.datetime) -> tuple[float, float]:
    """Return (home_lambda, away_lambda). NOT IMPLEMENTED — Phase 1B (Dixon-Coles)."""
    raise NotImplementedError("market_xg_for_fixture: Phase 1B (Dixon-Coles inverter)")


def eo_as_of(
    season_id: int,
    gw: int,
    rank_band: str,
    as_of: dt.datetime,
) -> pl.DataFrame:
    """Effective ownership for all players in `rank_band`. NOT IMPLEMENTED — Phase 1B."""
    raise NotImplementedError("eo_as_of: Phase 1B (after LiveFPL ingest)")


def match_event_history(player_id: int, before: dt.datetime) -> pl.DataFrame:
    """All fbref event-count rows for a player before `before`."""
    with session_scope() as s:
        rows = s.execute(
            select(FactPlayerMatchEvent)
            .where(FactPlayerMatchEvent.player_id == player_id)
            .where(FactPlayerMatchEvent.recorded_at <= before)
        ).scalars().all()
    if not rows:
        return pl.DataFrame()
    return pl.DataFrame(
        [
            {
                "player_id": r.player_id,
                "fixture_id": r.fixture_id,
                "tackles_won": r.tackles_won,
                "interceptions": r.interceptions,
                "recoveries": r.recoveries,
                "blocks": r.blocks,
                "dribbles_completed": r.dribbles_completed,
                "big_chances_created": r.big_chances_created,
                "passes_completed": r.passes_completed,
            }
            for r in rows
        ]
    )


def bps_rules_for_season(season_id: int) -> pl.DataFrame:
    """BPS rules for a given season. NOT IMPLEMENTED — Phase 2.4 (BPS sim)."""
    raise NotImplementedError("bps_rules_for_season: Phase 2.4")


def penalty_taker_as_of(team_id: int, as_of: dt.datetime) -> pl.DataFrame:
    """Ordered penalty-taker list for a team, manual override-aware. NOT IMPLEMENTED — Phase 1B."""
    raise NotImplementedError("penalty_taker_as_of: Phase 1B")
