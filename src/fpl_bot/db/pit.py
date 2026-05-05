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
                FactPlayerMatch.was_home,
                FactPlayerMatch.price_tenths,
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
                "was_home": r.was_home,
                "price_tenths": r.price_tenths,
                "season_id": r.season_id,
                "gameweek": r.gameweek,
                "kickoff_utc": r.kickoff_utc,
                "home_team_id": r.home_team_id,
                "away_team_id": r.away_team_id,
            }
            for r in rows
        ]
    )


def understat_player_match_history(
    season_ids: list[int] | None = None,
) -> pl.DataFrame:
    """Bulk fetch of fact_understat_player_match aggregated to latest recorded_at per
    (understat_player_id, match_date). Optional filter to fixtures in season_ids
    (skips rows with NULL fixture_id since season is unresolvable for them).
    """
    from sqlalchemy import func as _func

    from fpl_bot.db.models import (
        DimFixture as _DimFixture,
    )
    from fpl_bot.db.models import (
        FactUnderstatPlayerMatch as _FUPM,
    )

    with session_scope() as s:
        latest = (
            select(
                _FUPM.understat_player_id,
                _FUPM.match_date,
                _func.max(_FUPM.recorded_at).label("max_rec"),
            )
            .group_by(_FUPM.understat_player_id, _FUPM.match_date)
            .subquery()
        )
        stmt = select(
            _FUPM.understat_player_id,
            _FUPM.match_date,
            _FUPM.fixture_id,
            _FUPM.player_id,
            _FUPM.position,
            _FUPM.minutes,
            _FUPM.goals,
            _FUPM.shots,
            _FUPM.xg,
            _FUPM.xa,
            _FUPM.key_passes,
            _FUPM.npg,
            _FUPM.npxg,
            _FUPM.xg_chain,
            _FUPM.xg_buildup,
        ).join(
            latest,
            (latest.c.understat_player_id == _FUPM.understat_player_id)
            & (latest.c.match_date == _FUPM.match_date)
            & (latest.c.max_rec == _FUPM.recorded_at),
        )
        if season_ids is not None:
            stmt = stmt.join(_DimFixture, _DimFixture.fixture_id == _FUPM.fixture_id).where(
                _DimFixture.season_id.in_(season_ids)
            )
        rows = s.execute(stmt).all()

    if not rows:
        return pl.DataFrame()
    return pl.DataFrame(
        [
            {
                "understat_player_id": r.understat_player_id,
                "match_date": r.match_date,
                "fixture_id": r.fixture_id,
                "player_id": r.player_id,
                "position_us": r.position,
                "minutes_us": r.minutes,
                "goals_us": r.goals,
                "shots": r.shots,
                "xg": float(r.xg) if r.xg is not None else None,
                "xa": float(r.xa) if r.xa is not None else None,
                "key_passes": r.key_passes,
                "npg": r.npg,
                "npxg": float(r.npxg) if r.npxg is not None else None,
                "xg_chain": float(r.xg_chain) if r.xg_chain is not None else None,
                "xg_buildup": float(r.xg_buildup) if r.xg_buildup is not None else None,
            }
            for r in rows
        ]
    )


def market_xg_for_fixtures(
    fixture_ids: list[int] | None = None,
) -> pl.DataFrame:
    """Latest fact_market_xg row per (fixture_id, team_id). Returns
    columns: fixture_id, team_id, lambda, cs_prob.
    """
    from sqlalchemy import func as _func

    from fpl_bot.db.models import FactMarketXg as _FMX

    with session_scope() as s:
        latest = (
            select(
                _FMX.fixture_id,
                _FMX.team_id,
                _func.max(_FMX.source_recorded_at).label("max_src"),
            )
            .group_by(_FMX.fixture_id, _FMX.team_id)
            .subquery()
        )
        stmt = select(_FMX.fixture_id, _FMX.team_id, _FMX.lambda_, _FMX.cs_prob).join(
            latest,
            (latest.c.fixture_id == _FMX.fixture_id)
            & (latest.c.team_id == _FMX.team_id)
            & (latest.c.max_src == _FMX.source_recorded_at),
        )
        if fixture_ids is not None:
            stmt = stmt.where(_FMX.fixture_id.in_(fixture_ids))
        rows = s.execute(stmt).all()

    if not rows:
        return pl.DataFrame()
    return pl.DataFrame(
        [
            {
                "fixture_id": r.fixture_id,
                "team_id": r.team_id,
                "lambda_market_xg": float(r.lambda_),
                "cs_prob_market": float(r.cs_prob),
            }
            for r in rows
        ]
    )


def web_name_to_player_id() -> dict[str, int]:
    """Map of FPL web_name → stable player_id (FPL code). Latest snapshot."""
    from fpl_bot.db.models import DimPlayer as _DimPlayer

    with session_scope() as s:
        rows = s.execute(select(_DimPlayer.player_id, _DimPlayer.web_name)).all()
    return {r.web_name: r.player_id for r in rows if r.web_name}


def team_id_by_full_name(season_ids: list[int] | None = None) -> dict[tuple[int, str], int]:
    """Map of (season_id, full_name) → team_id."""
    from fpl_bot.db.models import DimTeam as _DimTeam

    with session_scope() as s:
        stmt = select(_DimTeam.season_id, _DimTeam.full_name, _DimTeam.team_id)
        if season_ids is not None:
            stmt = stmt.where(_DimTeam.season_id.in_(season_ids))
        rows = s.execute(stmt).all()
    return {(r.season_id, r.full_name): r.team_id for r in rows}


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
