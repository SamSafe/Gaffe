"""Phase 7 2.1 — extract structured return dates from FPL `news` text.

FPL's bootstrap-static endpoint includes a `news` text field per player
(already ingested into `fact_player_status.news`). For currently-
unavailable players (status in {i, s, u}) FPL frequently posts an
"Expected back DD MMM" line. Parsing that line gives us a per-(player,
gw) availability mask we can use in the MILP — pre-empt transfers for
returning players without waiting for them to hit status='a'.

Patterns we extract (highest signal first):
  "Expected back DD MMM"           → return_date = that date
  "Expected back DD MMM/early Jun" → return_date = first date in expr
  "out for season"                 → return_date = far future
  "Unknown return date"            → no return date (treat as out for horizon)

Anything else → no override (the existing status filter still applies).
"""
from __future__ import annotations

import datetime as dt
import re
from dataclasses import dataclass

import polars as pl
from sqlalchemy import func, select

from fpl_bot.db.models import DimFixture, FactPlayerStatus
from fpl_bot.db.session import session_scope

# Compiled patterns
_EXPECTED_BACK_RE = re.compile(
    r"Expected back\s+(\d{1,2})\s+([A-Za-z]+)", re.IGNORECASE
)
_OUT_FOR_SEASON_RE = re.compile(
    r"out for (the )?season|until next season|season-ending",
    re.IGNORECASE,
)

_MONTH_NAMES = {
    name.lower(): num
    for num, name in enumerate(
        [
            "",
            "Jan", "Feb", "Mar", "Apr", "May", "Jun",
            "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
        ]
    )
    if name
}


@dataclass(frozen=True)
class NewsExtraction:
    """Per-player structured info parsed from FPL news text."""

    player_id: int
    raw_news: str
    return_date: dt.date | None  # None = unknown / no parseable date
    out_for_season: bool


def _parse_dd_mon(day_str: str, mon_str: str, today: dt.date) -> dt.date | None:
    """Parse 'DD Mon' to a date in the most-likely upcoming year.

    FPL doesn't include the year. If parsed month is before today's month,
    assume next year; otherwise assume current year.
    """
    try:
        day = int(day_str)
    except ValueError:
        return None
    month = _MONTH_NAMES.get(mon_str[:3].lower())
    if not month or not (1 <= day <= 31):
        return None
    year = today.year
    candidate = dt.date(year, month, day) if month >= today.month else dt.date(year + 1, month, day)
    return candidate


def extract_news_from_status_rows(
    rows: list[tuple[int, str | None]],
    *,
    today: dt.date | None = None,
) -> list[NewsExtraction]:
    """Parse a list of (player_id, news_text) tuples into structured records."""
    if today is None:
        today = dt.datetime.now(dt.UTC).date()
    out: list[NewsExtraction] = []
    for pid, news in rows:
        if not news:
            continue
        out_for_season = bool(_OUT_FOR_SEASON_RE.search(news))
        match = _EXPECTED_BACK_RE.search(news)
        return_date: dt.date | None = None
        if match:
            return_date = _parse_dd_mon(match.group(1), match.group(2), today)
        out.append(
            NewsExtraction(
                player_id=pid,
                raw_news=news,
                return_date=return_date,
                out_for_season=out_for_season,
            )
        )
    return out


def latest_news_per_player(
    season_id: int | None = None,
    *,
    as_of: dt.datetime | None = None,
    today: dt.date | None = None,
) -> list[NewsExtraction]:
    """Pull the latest eligible news snapshot for every player.

    ``as_of`` is optional for the live path, but required for any historical
    replay so a later status update cannot leak backwards across a deadline.
    """
    if as_of is not None and as_of.utcoffset() is None:
        raise ValueError("as_of must be timezone-aware")
    with session_scope() as s:
        latest_stmt = select(
            FactPlayerStatus.player_id,
            func.max(FactPlayerStatus.recorded_at).label("max_rec"),
        )
        if season_id is not None:
            latest_stmt = latest_stmt.where(FactPlayerStatus.season_id == season_id)
        if as_of is not None:
            latest_stmt = latest_stmt.where(FactPlayerStatus.recorded_at <= as_of)
        latest = (
            latest_stmt
            .group_by(FactPlayerStatus.player_id)
            .subquery()
        )
        rows_stmt = (
            select(FactPlayerStatus.player_id, FactPlayerStatus.news)
            .join(
                latest,
                (latest.c.player_id == FactPlayerStatus.player_id)
                & (latest.c.max_rec == FactPlayerStatus.recorded_at),
            )
            .where(FactPlayerStatus.news.isnot(None))
        )
        if season_id is not None:
            rows_stmt = rows_stmt.where(FactPlayerStatus.season_id == season_id)
        rows = s.execute(rows_stmt).all()
    extraction_date = today
    if extraction_date is None and as_of is not None:
        extraction_date = as_of.astimezone(dt.UTC).date()
    return extract_news_from_status_rows(
        [(int(r.player_id), r.news) for r in rows],
        today=extraction_date,
    )


def return_gw_for_date(
    return_date: dt.date,
    season_id: int,
) -> int | None:
    """First GW whose first fixture starts on or after `return_date`."""
    with session_scope() as s:
        rows = s.execute(
            select(DimFixture.gameweek, func.min(DimFixture.kickoff_utc).label("gw_start"))
            .where(DimFixture.season_id == season_id)
            .group_by(DimFixture.gameweek)
            .order_by(DimFixture.gameweek)
        ).all()
    for gw, start in rows:
        if start is None:
            continue
        if start.date() >= return_date:
            return int(gw)
    return None


def build_pred_attenuator(
    *,
    season_id: int,
    horizon_gws: list[int],
    today: dt.date | None = None,
    as_of: dt.datetime | None = None,
) -> dict[tuple[int, int], float]:
    """Per-(player_id, gw) multiplicative pred attenuator from news text.

    For each player with a parsed return_date:
      - GWs strictly before the return-GW → 0.0 (force pred to 0)
      - GWs ≥ return-GW → no override (1.0 implied by absence from dict)

    For out-for-season players → 0.0 for all horizon GWs.

    Caller multiplies their `pred_dict[(p, w)]` by the attenuator before
    passing into the MILP. Players currently excluded by status_code that
    HAVE a future return-GW in horizon should be UN-excluded by the caller
    so the MILP can consider transferring them in for post-return GWs.
    """
    extractions = latest_news_per_player(
        season_id=season_id,
        as_of=as_of,
        today=today,
    )
    out: dict[tuple[int, int], float] = {}
    for ex in extractions:
        if ex.out_for_season:
            for gw in horizon_gws:
                out[(ex.player_id, gw)] = 0.0
            continue
        if ex.return_date is None:
            continue
        return_gw = return_gw_for_date(ex.return_date, season_id)
        if return_gw is None:
            # Return date is post-season — treat as out for the horizon
            for gw in horizon_gws:
                out[(ex.player_id, gw)] = 0.0
            continue
        for gw in horizon_gws:
            if gw < return_gw:
                out[(ex.player_id, gw)] = 0.0
    return out


def news_summary_df(season_id: int | None = None) -> pl.DataFrame:
    """Diagnostic helper — pretty-print the latest news extractions."""
    rows = latest_news_per_player(season_id=season_id)
    return pl.DataFrame(
        [
            {
                "player_id": r.player_id,
                "raw_news": r.raw_news,
                "return_date": r.return_date.isoformat() if r.return_date else None,
                "out_for_season": r.out_for_season,
            }
            for r in rows
        ]
    )
