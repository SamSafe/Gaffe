"""the-odds-api.com live bookmaker odds (Phase 1B source #6 / Phase 7 1.2b).

Honest API with documented free tier (500 requests/month). Used for the
weekly live run; football-data.co.uk covers historical/closed odds.

Compliance: §2.5 free-tier API with explicit key. No scraping. The API
key is supplied via the FPL_BOT_ODDS_API_KEY env var. If missing, fetch
raises a clear error rather than silently failing.

Markets captured (Phase 7 1.2b):
  h2h     → market="1X2"        selection in (home, draw, away)
  spreads → market="ah_{point}" selection in (home, away); point=signed
  totals  → market="totals_{p}" selection in (over, under)

Each (region × market) is one API request; default 3 markets × 1 region
= 3 requests per snapshot. EPL has ~10 fixtures/week, so a weekly pull
costs 3 requests against the 500/month free tier.

Bookmaker code is `OA_{api_key}` (e.g. "OA_pinnacle") to avoid collision
with football-data's post-match "PS" Pinnacle reading.

Endpoint reference: https://the-odds-api.com/liveapi/guides/v4/
"""
from __future__ import annotations

import datetime as dt
import json
from hashlib import sha256
from pathlib import Path
from typing import Any

import httpx
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert

from fpl_bot.config import settings
from fpl_bot.db.models import DimFixture, DimTeam, FactOdds
from fpl_bot.db.session import session_scope
from fpl_bot.ingest.audit import audit_fetch

ODDS_API_BASE = "https://api.the-odds-api.com/v4"
EPL_SPORT_KEY = "soccer_epl"
DEFAULT_MARKETS = "h2h,spreads,totals"
DEFAULT_REGIONS = "uk"
DEFAULT_BOOKMAKERS = "pinnacle,betfair_ex_uk,williamhill,bet365"
DEFAULT_ODDS_FORMAT = "decimal"

# The Odds API team name → FPL short_name (season 25 names; extend on
# promotion). Includes alternate naming for some clubs since the API has
# been inconsistent over time.
OA_TO_FPL_SHORT: dict[str, str] = {
    "Arsenal": "ARS",
    "Aston Villa": "AVL",
    "AFC Bournemouth": "BOU",
    "Bournemouth": "BOU",
    "Brentford": "BRE",
    "Brighton and Hove Albion": "BHA",
    "Brighton & Hove Albion": "BHA",
    "Burnley": "BUR",
    "Chelsea": "CHE",
    "Crystal Palace": "CRY",
    "Everton": "EVE",
    "Fulham": "FUL",
    "Leeds United": "LEE",
    "Leeds": "LEE",
    "Liverpool": "LIV",
    "Manchester City": "MCI",
    "Manchester United": "MUN",
    "Newcastle United": "NEW",
    "Newcastle": "NEW",
    "Nottingham Forest": "NFO",
    "Sheffield United": "SHU",
    "Sunderland": "SUN",
    "Tottenham Hotspur": "TOT",
    "Tottenham": "TOT",
    "West Ham United": "WHU",
    "West Ham": "WHU",
    "Wolverhampton Wanderers": "WOL",
    "Wolves": "WOL",
    "Ipswich Town": "IPS",
    "Luton Town": "LUT",
}


def fetch_raw_oddsapi(
    sport: str = EPL_SPORT_KEY,
    markets: str = DEFAULT_MARKETS,
    regions: str = DEFAULT_REGIONS,
    bookmakers: str | None = DEFAULT_BOOKMAKERS,
) -> Path:
    """Fetch live EPL odds. Requires FPL_BOT_ODDS_API_KEY env var.

    Writes a sidecar `*.meta.json` with the quota-headers from the
    response (remaining/used) so we can monitor the 500/mo cap.
    """
    api_key = settings.odds_api_key
    if not api_key:
        raise RuntimeError(
            "the-odds-api requires a free-tier API key. "
            "Set FPL_BOT_ODDS_API_KEY in your environment "
            "(register at https://the-odds-api.com)."
        )

    params = [
        f"regions={regions}",
        f"markets={markets}",
        f"oddsFormat={DEFAULT_ODDS_FORMAT}",
        f"apiKey={api_key}",
    ]
    if bookmakers:
        params.append(f"bookmakers={bookmakers}")
    url = f"{ODDS_API_BASE}/sports/{sport}/odds?{'&'.join(params)}"
    today = dt.datetime.now(dt.UTC).strftime("%Y-%m-%d")
    raw_dir = settings.raw_dir / "oddsapi" / today
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_path = raw_dir / f"{sport}_{markets.replace(',', '+')}.json"

    audit_url = url.replace(api_key, "***REDACTED***")
    remaining = used = None
    with audit_fetch(source="oddsapi", url=audit_url, user_agent=settings.user_agent) as audit:
        with httpx.Client(
            headers={"User-Agent": settings.user_agent},
            timeout=settings.request_timeout_seconds,
        ) as client:
            r = client.get(url)
            audit.response_code = r.status_code
            r.raise_for_status()
            payload = r.content
            remaining = r.headers.get("x-requests-remaining")
            used = r.headers.get("x-requests-used")
        audit.byte_size = len(payload)
        audit.content_hash = sha256(payload).hexdigest()
        audit.raw_path = str(raw_path)
        if raw_path.exists() and sha256(raw_path.read_bytes()).hexdigest() == audit.content_hash:
            audit.parse_status = "skipped_unchanged"
        else:
            raw_path.write_bytes(payload)
    meta_path = raw_path.with_suffix(".meta.json")
    meta_path.write_text(
        json.dumps(
            {
                "fetched_at": dt.datetime.now(dt.UTC).isoformat(),
                "sport": sport,
                "markets": markets,
                "regions": regions,
                "bookmakers": bookmakers,
                "remaining_requests": remaining,
                "used_requests": used,
            },
            indent=2,
        )
    )
    return raw_path


def parse_raw_oddsapi(raw_path: Path, season_id: int) -> dict[str, int]:
    """Parse the JSON payload → fact_odds rows.

    Joins to dim_fixture via (commence_date → kickoff date, mapped
    home/away team names → dim_team.team_id). Idempotent: ON CONFLICT
    on the snapshot PK updates only an exact duplicate quote. Later pulls are
    preserved as distinct rows via `quote_time`.
    """
    payload = json.loads(raw_path.read_text())
    counts = {
        "fact_odds": 0,
        "events_in_payload": len(payload),
        "skipped_unmapped_team": 0,
        "skipped_no_fixture": 0,
        "skipped_no_outcome": 0,
    }
    short_to_team_id, fixture_lookup = _build_lookups(season_id)
    fetched_at = _raw_fetched_at(raw_path)

    with session_scope() as s:
        for event in payload:
            home_short = OA_TO_FPL_SHORT.get(event.get("home_team") or "")
            away_short = OA_TO_FPL_SHORT.get(event.get("away_team") or "")
            if home_short is None or away_short is None:
                counts["skipped_unmapped_team"] += 1
                continue
            home_id = short_to_team_id.get(home_short)
            away_id = short_to_team_id.get(away_short)
            if home_id is None or away_id is None:
                counts["skipped_unmapped_team"] += 1
                continue

            commence_ts = event.get("commence_time")
            if not commence_ts:
                counts["skipped_no_outcome"] += 1
                continue
            event_time = dt.datetime.fromisoformat(
                commence_ts.replace("Z", "+00:00")
            ).astimezone(dt.UTC)
            fixture_id = fixture_lookup.get((event_time.date(), home_id, away_id))
            if fixture_id is None:
                counts["skipped_no_fixture"] += 1
                continue

            for bk in event.get("bookmakers", []) or []:
                bk_code = f"OA_{bk['key']}"
                for market in bk.get("markets", []) or []:
                    m_key = market.get("key")
                    quote_time = _parse_api_timestamp(
                        market.get("last_update") or bk.get("last_update")
                    ) or fetched_at
                    for outcome in market.get("outcomes", []) or []:
                        sel, market_code = _outcome_to_selection(
                            m_key, outcome,
                            event.get("home_team"),
                            event.get("away_team"),
                        )
                        if sel is None:
                            counts["skipped_no_outcome"] += 1
                            continue
                        price = outcome.get("price")
                        if price is None:
                            counts["skipped_no_outcome"] += 1
                            continue
                        try:
                            decimal_odds = float(price)
                        except (TypeError, ValueError):
                            counts["skipped_no_outcome"] += 1
                            continue
                        stmt = (
                            pg_insert(FactOdds)
                            .values(
                                fixture_id=fixture_id,
                                bookmaker=bk_code,
                                market=market_code,
                                selection=sel,
                                event_time=event_time,
                                quote_time=quote_time,
                                decimal_odds=decimal_odds,
                            )
                            .on_conflict_do_update(
                                index_elements=[
                                    "fixture_id",
                                    "bookmaker",
                                    "market",
                                    "selection",
                                    "quote_time",
                                ],
                                set_={"decimal_odds": decimal_odds},
                            )
                        )
                        s.execute(stmt)
                        counts["fact_odds"] += 1
    return counts


def _parse_api_timestamp(value: object) -> dt.datetime | None:
    """Parse an Odds API ISO timestamp as UTC; return None when malformed."""
    if not value:
        return None
    try:
        parsed = dt.datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.UTC)
    return parsed.astimezone(dt.UTC)


def _raw_fetched_at(raw_path: Path) -> dt.datetime:
    """Read the audited fetch time, falling back to the raw file mtime."""
    meta_path = raw_path.with_suffix(".meta.json")
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text())
        except (json.JSONDecodeError, OSError):
            meta = {}
        parsed = _parse_api_timestamp(meta.get("fetched_at"))
        if parsed is not None:
            return parsed
    return dt.datetime.fromtimestamp(raw_path.stat().st_mtime, tz=dt.UTC)


def _outcome_to_selection(
    market_key: str | None,
    outcome: dict[str, Any],
    home_team_name: str | None,
    away_team_name: str | None,
) -> tuple[str | None, str]:
    """Translate one Odds API outcome → (selection, market_code).

    Returns (None, ...) when the outcome is unrecognized (caller skips).
    """
    name = outcome.get("name") or ""
    if market_key == "h2h":
        if name == "Draw":
            return ("draw", "1X2")
        if name == home_team_name:
            return ("home", "1X2")
        if name == away_team_name:
            return ("away", "1X2")
        return (None, "1X2")
    if market_key == "spreads":
        point = outcome.get("point")
        if point is None:
            return (None, "ah")
        sel = (
            "home" if name == home_team_name
            else ("away" if name == away_team_name else None)
        )
        if sel is None:
            return (None, "ah")
        # Canonicalize the market to the HOME handicap so both complementary
        # outcomes share one market key. The API reports opposite signed
        # points on the home and away outcome rows.
        home_point = float(point) if sel == "home" else -float(point)
        return (sel, f"ah_{home_point:g}")
    if market_key == "totals":
        point = outcome.get("point")
        if point is None:
            return (None, "totals")
        lower = name.lower()
        sel = "over" if lower == "over" else ("under" if lower == "under" else None)
        if sel is None:
            return (None, "totals")
        return (sel, f"totals_{point}")
    return (None, market_key or "unknown")


def _build_lookups(
    season_id: int,
) -> tuple[dict[str, int], dict[tuple[dt.date, int, int], int]]:
    with session_scope() as s:
        team_rows = s.execute(
            select(DimTeam.team_id, DimTeam.short_name).where(
                DimTeam.season_id == season_id
            )
        ).all()
        fixture_rows = (
            s.execute(select(DimFixture).where(DimFixture.season_id == season_id))
            .scalars()
            .all()
        )
    short_to_team_id = {t.short_name: t.team_id for t in team_rows}
    fixture_lookup = {
        (f.kickoff_utc.date(), f.home_team_id, f.away_team_id): f.fixture_id
        for f in fixture_rows
    }
    return short_to_team_id, fixture_lookup
