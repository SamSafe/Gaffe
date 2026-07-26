"""football-data.co.uk historical odds ingest (Phase 1B source #2).

Free CSV downloads of EPL match results + bookmaker odds back to 1993/94.
robots.txt: Disallow empty (all paths allowed). License (notes.txt): free for
non-commercial use with attribution.

Two-layer per §2.3:
  fetch_raw_footballdata(season) → data/raw/footballdata/{date}/E0_{season}.csv
  parse_raw_footballdata(raw_path, season_id) → fact_odds rows, joined to
    dim_fixture via (kickoff date, home_team_id, away_team_id)

Markets captured: 1X2 (Bet365, Pinnacle, William Hill) + totals 2.5
(Bet365, Pinnacle). Sufficient for Dixon-Coles inversion (§2.2).
"""
from __future__ import annotations

import datetime as dt
from hashlib import sha256
from pathlib import Path

import httpx
import polars as pl
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert

from fpl_bot.config import settings
from fpl_bot.db.models import DimFixture, DimTeam, FactOdds
from fpl_bot.db.session import session_scope
from fpl_bot.ingest.audit import audit_fetch

FOOTBALL_DATA_BASE = "https://www.football-data.co.uk/mmz4281"

# football-data.co.uk team name → FPL short_name. Includes alt names ("Tottenham"
# vs "Spurs"). Update when new teams are promoted.
FD_TO_FPL_SHORT: dict[str, str] = {
    "Arsenal": "ARS",
    "Aston Villa": "AVL",
    "Bournemouth": "BOU",
    "Brentford": "BRE",
    "Brighton": "BHA",
    "Burnley": "BUR",
    "Chelsea": "CHE",
    "Coventry": "COV",
    "Coventry City": "COV",
    "Crystal Palace": "CRY",
    "Everton": "EVE",
    "Fulham": "FUL",
    "Hull": "HUL",
    "Hull City": "HUL",
    "Ipswich": "IPS",
    "Leeds": "LEE",
    "Leicester": "LEI",
    "Liverpool": "LIV",
    "Luton": "LUT",
    "Man City": "MCI",
    "Man United": "MUN",
    "Newcastle": "NEW",
    "Norwich": "NOR",
    "Nott'm Forest": "NFO",
    "Sheffield United": "SHU",
    "Southampton": "SOU",
    "Spurs": "TOT",
    "Tottenham": "TOT",
    "Sunderland": "SUN",
    "Watford": "WAT",
    "West Brom": "WBA",
    "West Ham": "WHU",
    "Wolves": "WOL",
}

# (bookmaker, market, home_col, draw_col, away_col)
ONEX2_COLS: list[tuple[str, str, str, str, str]] = [
    ("B365", "1X2", "B365H", "B365D", "B365A"),
    ("PS", "1X2", "PSH", "PSD", "PSA"),
    ("WH", "1X2", "WHH", "WHD", "WHA"),
]

# (bookmaker, market, over_col, under_col)
TOTALS_COLS: list[tuple[str, str, str, str]] = [
    ("B365", "totals_2.5", "B365>2.5", "B365<2.5"),
    ("P", "totals_2.5", "P>2.5", "P<2.5"),
]


def _season_url(season: str) -> str:
    """'2024-25' → 'https://www.football-data.co.uk/mmz4281/2425/E0.csv'."""
    start, end = season.split("-")
    code = start[-2:] + end
    return f"{FOOTBALL_DATA_BASE}/{code}/E0.csv"


def _today_dir(source: str) -> Path:
    today = dt.datetime.now(dt.UTC).strftime("%Y-%m-%d")
    d = settings.raw_dir / source / today
    d.mkdir(parents=True, exist_ok=True)
    return d


def fetch_raw_footballdata(season: str) -> Path:
    """Download one season's E0.csv from football-data.co.uk."""
    url = _season_url(season)
    raw_path = _today_dir("footballdata") / f"E0_{season}.csv"

    with audit_fetch(source="footballdata", url=url, user_agent=settings.user_agent) as audit:
        with httpx.Client(
            headers={"User-Agent": settings.user_agent},
            timeout=settings.request_timeout_seconds,
        ) as client:
            r = client.get(url)
            audit.response_code = r.status_code
            r.raise_for_status()
            payload = r.content

        audit.byte_size = len(payload)
        audit.content_hash = sha256(payload).hexdigest()
        audit.raw_path = str(raw_path)

        if raw_path.exists() and sha256(raw_path.read_bytes()).hexdigest() == audit.content_hash:
            audit.parse_status = "skipped_unchanged"
        else:
            raw_path.write_bytes(payload)

    return raw_path


def parse_raw_footballdata(raw_path: Path, season_id: int) -> dict[str, int]:
    """Parse CSV → fact_odds. Joins to dim_fixture via (date, home, away)."""
    df = pl.read_csv(raw_path, ignore_errors=True, infer_schema_length=10000)

    short_to_team_id, fixture_lookup = _build_lookups(season_id)

    counts = {
        "fact_odds": 0,
        "skipped_unmapped_team": 0,
        "skipped_no_fixture": 0,
        "skipped_bad_date": 0,
    }

    with session_scope() as s:
        for row in df.iter_rows(named=True):
            home_short = FD_TO_FPL_SHORT.get(row.get("HomeTeam") or "")
            away_short = FD_TO_FPL_SHORT.get(row.get("AwayTeam") or "")
            if home_short is None or away_short is None:
                counts["skipped_unmapped_team"] += 1
                continue

            home_id = short_to_team_id.get(home_short)
            away_id = short_to_team_id.get(away_short)
            if home_id is None or away_id is None:
                counts["skipped_unmapped_team"] += 1
                continue

            match_date = _parse_fd_date(row.get("Date"))
            if match_date is None:
                counts["skipped_bad_date"] += 1
                continue

            fixture_match = fixture_lookup.get((match_date, home_id, away_id))
            if fixture_match is None:
                counts["skipped_no_fixture"] += 1
                continue
            fixture_id, event_time = fixture_match

            # football-data publishes closing prices, not timestamped quote
            # histories. Use kickoff as the documented closing approximation.
            quote_time = event_time

            for bk, market, hc, dc, ac in ONEX2_COLS:
                for sel, col in (("home", hc), ("draw", dc), ("away", ac)):
                    n = _insert_odds(
                        s, fixture_id, bk, market, sel, event_time, quote_time, row.get(col)
                    )
                    counts["fact_odds"] += n

            for bk, market, oc, uc in TOTALS_COLS:
                for sel, col in (("over", oc), ("under", uc)):
                    n = _insert_odds(
                        s, fixture_id, bk, market, sel, event_time, quote_time, row.get(col)
                    )
                    counts["fact_odds"] += n

    return counts


def _build_lookups(
    season_id: int,
) -> tuple[
    dict[str, int],
    dict[tuple[dt.date, int, int], tuple[int, dt.datetime]],
]:
    """Returns team ids and fixture id/commencement lookup."""
    with session_scope() as s:
        team_rows = s.execute(
            select(DimTeam.team_id, DimTeam.short_name).where(DimTeam.season_id == season_id)
        ).all()
        fixture_rows = (
            s.execute(select(DimFixture).where(DimFixture.season_id == season_id))
            .scalars()
            .all()
        )

    short_to_team_id = {t.short_name: t.team_id for t in team_rows}
    fixture_lookup = {
        (f.kickoff_utc.date(), f.home_team_id, f.away_team_id): (
            f.fixture_id,
            f.kickoff_utc,
        )
        for f in fixture_rows
    }
    return short_to_team_id, fixture_lookup


def _parse_fd_date(value: object) -> dt.date | None:
    if not value:
        return None
    s = str(value)
    parts = s.split("/")
    if len(parts) != 3:
        return None
    fmt = "%d/%m/%y" if len(parts[2]) == 2 else "%d/%m/%Y"
    try:
        return dt.datetime.strptime(s, fmt).date()
    except ValueError:
        return None


def _insert_odds(
    s,
    fixture_id: int,
    bookmaker: str,
    market: str,
    selection: str,
    event_time: dt.datetime,
    quote_time: dt.datetime,
    raw_value: object,
) -> int:
    if raw_value is None or raw_value == "":
        return 0
    try:
        decimal_odds = float(raw_value)
    except (TypeError, ValueError):
        return 0
    if decimal_odds <= 1.0:
        return 0  # invalid or scratched market
    stmt = pg_insert(FactOdds).values(
        fixture_id=fixture_id,
        bookmaker=bookmaker,
        market=market,
        selection=selection,
        event_time=event_time,
        quote_time=quote_time,
        decimal_odds=decimal_odds,
    )
    stmt = stmt.on_conflict_do_nothing(
        index_elements=["fixture_id", "bookmaker", "market", "selection", "quote_time"]
    )
    s.execute(stmt)
    return 1
