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

    Dedupes by latest recorded_at per (player_id, fixture_id) — needed because
    re-ingest creates a new bitemporal row each time (e.g., the Phase 3.5
    transfers backfill added a second row per fixture). Same pattern as
    `understat_player_match_history` and `eo_as_of`.

    Used for backtest feature building. Time-cutoff filtering for PIT correctness
    is the caller's responsibility — typically polars window functions over
    per-player time-sorted groupings (shift / rolling) that ensure each row's
    features use only data from earlier rows.
    """
    from sqlalchemy import func as _func

    from fpl_bot.db.models import DimFixture as _DimFixture

    with session_scope() as s:
        latest = (
            select(
                FactPlayerMatch.player_id,
                FactPlayerMatch.fixture_id,
                _func.max(FactPlayerMatch.recorded_at).label("max_rec"),
            )
            .group_by(FactPlayerMatch.player_id, FactPlayerMatch.fixture_id)
            .subquery("fpm_latest")
        )
        stmt = (
            select(
                FactPlayerMatch.player_id,
                FactPlayerMatch.fixture_id,
                FactPlayerMatch.minutes,
                FactPlayerMatch.goals,
                FactPlayerMatch.assists,
                FactPlayerMatch.clean_sheet,
                FactPlayerMatch.goals_conceded,
                FactPlayerMatch.saves,
                FactPlayerMatch.yellow_cards,
                FactPlayerMatch.red_cards,
                FactPlayerMatch.bonus,
                FactPlayerMatch.bps,
                FactPlayerMatch.total_points,
                FactPlayerMatch.was_home,
                FactPlayerMatch.price_tenths,
                FactPlayerMatch.transfers_in,
                FactPlayerMatch.transfers_out,
                FactPlayerMatch.transfers_balance,
                FactPlayerMatch.selected,
                _DimFixture.season_id,
                _DimFixture.gameweek,
                _DimFixture.kickoff_utc,
                _DimFixture.home_team_id,
                _DimFixture.away_team_id,
            )
            .join(
                latest,
                (latest.c.player_id == FactPlayerMatch.player_id)
                & (latest.c.fixture_id == FactPlayerMatch.fixture_id)
                & (latest.c.max_rec == FactPlayerMatch.recorded_at),
            )
            .join(_DimFixture, _DimFixture.fixture_id == FactPlayerMatch.fixture_id)
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
                "goals_conceded": r.goals_conceded,
                "saves": r.saves,
                "yellow_cards": r.yellow_cards,
                "red_cards": r.red_cards,
                "bonus": r.bonus,
                "bps": r.bps,
                "total_points": r.total_points,
                "was_home": r.was_home,
                "price_tenths": r.price_tenths,
                "transfers_in": r.transfers_in,
                "transfers_out": r.transfers_out,
                "transfers_balance": r.transfers_balance,
                "selected": r.selected,
                "season_id": r.season_id,
                "gameweek": r.gameweek,
                "kickoff_utc": r.kickoff_utc,
                "home_team_id": r.home_team_id,
                "away_team_id": r.away_team_id,
            }
            for r in rows
        ],
        schema_overrides={
            "transfers_in": pl.Int64,
            "transfers_out": pl.Int64,
            "transfers_balance": pl.Int64,
            "selected": pl.Int64,
        },
        infer_schema_length=None,
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
    *,
    as_of: dt.datetime | None = None,
) -> pl.DataFrame:
    """Latest fact_market_xg row per (fixture_id, team_id). Returns
    columns: fixture_id, team_id, lambda, cs_prob.
    """
    if as_of is not None and as_of.utcoffset() is None:
        raise ValueError("as_of must be timezone-aware")
    from sqlalchemy import func as _func

    from fpl_bot.db.models import FactMarketXg as _FMX

    with session_scope() as s:
        latest_stmt = select(
            _FMX.fixture_id,
            _FMX.team_id,
            _func.max(_FMX.source_recorded_at).label("max_src"),
        )
        if as_of is not None:
            latest_stmt = latest_stmt.where(_FMX.source_recorded_at <= as_of)
        latest = latest_stmt.group_by(_FMX.fixture_id, _FMX.team_id).subquery()
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


def upcoming_fixtures(fixture_ids: list[int]) -> pl.DataFrame:
    """Phase 6 v2: pull dim_fixture metadata for arbitrary fixture_ids.

    Used by the predict-only feature builders to synthesize per-(player,
    fixture) rows for fixtures that haven't been played yet (no
    fact_player_match rows exist for them).

    Returns columns: fixture_id, season_id, gameweek, kickoff_utc,
    home_team_id, away_team_id.
    """
    if not fixture_ids:
        return pl.DataFrame()
    from fpl_bot.db.models import DimFixture as _DimFixture
    with session_scope() as s:
        rows = s.execute(
            select(
                _DimFixture.fixture_id,
                _DimFixture.season_id,
                _DimFixture.gameweek,
                _DimFixture.kickoff_utc,
                _DimFixture.home_team_id,
                _DimFixture.away_team_id,
            ).where(_DimFixture.fixture_id.in_(fixture_ids))
        ).all()
    if not rows:
        return pl.DataFrame()
    return pl.DataFrame(
        [
            {
                "fixture_id": int(r.fixture_id),
                "season_id": int(r.season_id),
                "gameweek": int(r.gameweek),
                "kickoff_utc": r.kickoff_utc,
                "home_team_id": int(r.home_team_id),
                "away_team_id": int(r.away_team_id),
            }
            for r in rows
        ]
    )


def fixture_gameweeks_for_season(season_id: int) -> dict[int, int]:
    """Map fixture_id to gameweek for one season."""
    from fpl_bot.db.models import DimFixture as _DimFixture

    with session_scope() as s:
        rows = s.execute(
            select(_DimFixture.fixture_id, _DimFixture.gameweek).where(
                _DimFixture.season_id == season_id
            )
        ).all()
    return {int(r.fixture_id): int(r.gameweek) for r in rows}


def season_player_status_snapshot(season_id: int) -> pl.DataFrame:
    """Phase 6 v2: latest (player_id, team_id, position_code) per player for
    a season, taken from fact_player_status (most recent recorded_at).

    Used by predict-only feature builders to identify "eligible players"
    for an upcoming fixture (those currently registered to one of the
    fixture's teams).

    Returns columns: player_id, team_id, position_code.
    """
    from sqlalchemy import func as _func

    from fpl_bot.db.models import FactPlayerStatus as _FPS
    with session_scope() as s:
        latest = (
            select(
                _FPS.player_id,
                _func.max(_FPS.recorded_at).label("max_rec"),
            )
            .where(_FPS.season_id == season_id)
            .group_by(_FPS.player_id)
            .subquery("ps_latest")
        )
        rows = s.execute(
            select(_FPS.player_id, _FPS.team_id, _FPS.position_code)
            .join(
                latest,
                (latest.c.player_id == _FPS.player_id)
                & (latest.c.max_rec == _FPS.recorded_at),
            )
        ).all()
    if not rows:
        return pl.DataFrame()
    return pl.DataFrame(
        [
            {
                "player_id": int(r.player_id),
                "team_id": int(r.team_id),
                "position_code": r.position_code,
            }
            for r in rows
        ]
    )


def all_player_status_snapshots(season_id: int) -> pl.DataFrame:
    """All stored FPL status snapshots for a season.

    Returns raw snapshot rows for offline feature builders that need repeated
    intra-week observations. The caller is responsible for PIT-safe ordering
    and lag/label construction.
    """
    with session_scope() as s:
        rows = s.execute(
            select(
                FactPlayerStatus.player_id,
                FactPlayerStatus.season_id,
                FactPlayerStatus.recorded_at,
                FactPlayerStatus.price_tenths,
                FactPlayerStatus.selected_by_percent,
            ).where(FactPlayerStatus.season_id == season_id)
        ).all()
    if not rows:
        return pl.DataFrame()
    return pl.DataFrame(
        [
            {
                "player_id": int(r.player_id),
                "season_id": int(r.season_id),
                "recorded_at": r.recorded_at,
                "price_tenths": int(r.price_tenths),
                "selected_by_percent": float(r.selected_by_percent or 0.0),
            }
            for r in rows
        ]
    )


def defensive_contribution_per_player_per_gw(
    season_id: int,
) -> pl.DataFrame:
    """Per-(player_id, gameweek) defensive_contribution stat for a season,
    deduped to the latest recorded_at per (player, fixture). DGW handling:
    sum across same-GW fixtures (one player can have 2 matches in a GW).

    Returns columns: player_id, gameweek, defensive_contribution, minutes.
    Only rows with non-null defensive_contribution are returned (the column
    is only populated for FPL 25/26 onward).

    Used by the DefCon adjustment helper to predict per-(player, gw) DefCon
    points via PIT-correct rolling-rate estimate.
    """
    from sqlalchemy import func as _func

    from fpl_bot.db.models import DimFixture as _DimFixture

    with session_scope() as s:
        latest = (
            select(
                FactPlayerMatch.player_id,
                FactPlayerMatch.fixture_id,
                _func.max(FactPlayerMatch.recorded_at).label("max_rec"),
            )
            .group_by(FactPlayerMatch.player_id, FactPlayerMatch.fixture_id)
            .subquery("fpm_latest_dc")
        )
        rows = s.execute(
            select(
                FactPlayerMatch.player_id,
                _DimFixture.gameweek,
                FactPlayerMatch.defensive_contribution,
                FactPlayerMatch.minutes,
            )
            .join(
                latest,
                (latest.c.player_id == FactPlayerMatch.player_id)
                & (latest.c.fixture_id == FactPlayerMatch.fixture_id)
                & (latest.c.max_rec == FactPlayerMatch.recorded_at),
            )
            .join(_DimFixture, _DimFixture.fixture_id == FactPlayerMatch.fixture_id)
            .where(_DimFixture.season_id == season_id)
            .where(FactPlayerMatch.defensive_contribution.isnot(None))
        ).all()
    if not rows:
        return pl.DataFrame()
    df = pl.DataFrame(
        [
            {
                "player_id": int(r.player_id),
                "gameweek": int(r.gameweek),
                "defensive_contribution": int(r.defensive_contribution),
                "minutes": int(r.minutes or 0),
            }
            for r in rows
        ]
    )
    # DGW: sum across same-GW fixtures
    return df.group_by(["player_id", "gameweek"]).agg(
        pl.col("defensive_contribution").sum(),
        pl.col("minutes").sum(),
    )


def player_actual_pts_last_n_gws(
    season_id: int, n: int = 5
) -> dict[int, float]:
    """Mean `total_points` per fixture across the LAST `n` finished GWs of
    `season_id`. Used as a GW1 cold-start prior for the next season.

    Cross-season identity: `player_id` is the stable FPL `code` (not the
    per-season element id), so a player_id in season N+1 = same player in
    season N. Promoted teams' players don't appear and naturally fall back
    to the model's default xPts.

    Dedupes by latest recorded_at per (player_id, fixture_id), same pattern
    as `all_player_match_with_kickoff`.
    """
    from sqlalchemy import func as _func

    from fpl_bot.db.models import DimFixture as _DimFixture

    with session_scope() as s:
        max_gw = s.execute(
            select(_func.max(_DimFixture.gameweek)).where(
                _DimFixture.season_id == season_id
            )
        ).scalar()
        if max_gw is None:
            return {}
        lo = max(1, int(max_gw) - n + 1)
        latest = (
            select(
                FactPlayerMatch.player_id,
                FactPlayerMatch.fixture_id,
                _func.max(FactPlayerMatch.recorded_at).label("max_rec"),
            )
            .group_by(FactPlayerMatch.player_id, FactPlayerMatch.fixture_id)
            .subquery("fpm_latest_prior")
        )
        rows = s.execute(
            select(
                FactPlayerMatch.player_id,
                FactPlayerMatch.total_points,
            )
            .join(
                latest,
                (latest.c.player_id == FactPlayerMatch.player_id)
                & (latest.c.fixture_id == FactPlayerMatch.fixture_id)
                & (latest.c.max_rec == FactPlayerMatch.recorded_at),
            )
            .join(_DimFixture, _DimFixture.fixture_id == FactPlayerMatch.fixture_id)
            .where(_DimFixture.season_id == season_id)
            .where(_DimFixture.gameweek >= lo)
            .where(_DimFixture.gameweek <= int(max_gw))
        ).all()
    by_pid: dict[int, list[int]] = {}
    for r in rows:
        by_pid.setdefault(int(r.player_id), []).append(int(r.total_points or 0))
    return {pid: sum(pts) / len(pts) for pid, pts in by_pid.items()}


def recent_minutes_at_gw(
    season_id: int, target_gw: int, lookback_gws: int = 3
) -> dict[int, int]:
    """Sum of `minutes` per player across the last `lookback_gws` finished GWs
    strictly before `target_gw` for `season_id`.

    PIT-correct by construction: filters on gameweek < target_gw. Used by the
    backtest as a real-availability proxy when historical fact_player_status
    is unavailable — a player with zero minutes across the lookback window is
    effectively unavailable (injured / dropped / suspended).

    Dedupes by latest recorded_at per (player_id, fixture_id) — same pattern
    as `all_player_match_with_kickoff`.

    Returns: player_id → total minutes in window. Players who appear in no
    fixtures in the window are omitted (caller treats missing as 0).
    """
    from sqlalchemy import func as _func

    from fpl_bot.db.models import DimFixture as _DimFixture

    if lookback_gws < 1 or target_gw < 2:
        return {}
    lo = max(1, target_gw - lookback_gws)
    with session_scope() as s:
        latest = (
            select(
                FactPlayerMatch.player_id,
                FactPlayerMatch.fixture_id,
                _func.max(FactPlayerMatch.recorded_at).label("max_rec"),
            )
            .group_by(FactPlayerMatch.player_id, FactPlayerMatch.fixture_id)
            .subquery("fpm_latest_rmin")
        )
        rows = s.execute(
            select(
                FactPlayerMatch.player_id,
                FactPlayerMatch.minutes,
            )
            .join(
                latest,
                (latest.c.player_id == FactPlayerMatch.player_id)
                & (latest.c.fixture_id == FactPlayerMatch.fixture_id)
                & (latest.c.max_rec == FactPlayerMatch.recorded_at),
            )
            .join(_DimFixture, _DimFixture.fixture_id == FactPlayerMatch.fixture_id)
            .where(_DimFixture.season_id == season_id)
            .where(_DimFixture.gameweek >= lo)
            .where(_DimFixture.gameweek < target_gw)
        ).all()
    out: dict[int, int] = {}
    for r in rows:
        out[int(r.player_id)] = out.get(int(r.player_id), 0) + int(r.minutes or 0)
    return out


def season_start_kickoff(season_id: int) -> dt.datetime | None:
    """Phase 6 v2: min(kickoff_utc) across dim_fixture for a season.

    Used to compute `days_into_season` for upcoming fixtures.
    """
    from sqlalchemy import func as _func

    from fpl_bot.db.models import DimFixture as _DimFixture
    with session_scope() as s:
        return s.execute(
            select(_func.min(_DimFixture.kickoff_utc)).where(
                _DimFixture.season_id == season_id
            )
        ).scalar()
