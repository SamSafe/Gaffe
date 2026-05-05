"""the-odds-api.com live bookmaker odds (Phase 1B source #6).

Honest API with documented free tier (500 requests/month). Used in Phase 6
for the weekly live run; not strictly required for Phase 1–5 backtests
since football-data.co.uk covers historical odds.

Compliance: §2.5 free-tier API with explicit key. No scraping. The API
key is supplied via the FPL_BOT_ODDS_API_KEY env var. If missing, fetch
raises a clear error rather than silently failing.

Endpoint reference: https://the-odds-api.com/liveapi/guides/v4/
"""
from __future__ import annotations

import datetime as dt
import json
from hashlib import sha256
from pathlib import Path

import httpx

from fpl_bot.config import settings
from fpl_bot.ingest.audit import audit_fetch

ODDS_API_BASE = "https://api.the-odds-api.com/v4"
EPL_SPORT_KEY = "soccer_epl"
DEFAULT_MARKETS = "h2h,totals"  # 1X2 + over/under
DEFAULT_REGIONS = "uk,eu"
DEFAULT_ODDS_FORMAT = "decimal"


def fetch_raw_oddsapi(
    sport: str = EPL_SPORT_KEY,
    markets: str = DEFAULT_MARKETS,
    regions: str = DEFAULT_REGIONS,
) -> Path:
    """Fetch live EPL odds. Requires FPL_BOT_ODDS_API_KEY env var."""
    api_key = settings.odds_api_key
    if not api_key:
        raise RuntimeError(
            "the-odds-api requires a free-tier API key. "
            "Set FPL_BOT_ODDS_API_KEY in your environment "
            "(register at https://the-odds-api.com)."
        )

    url = (
        f"{ODDS_API_BASE}/sports/{sport}/odds"
        f"?regions={regions}&markets={markets}"
        f"&oddsFormat={DEFAULT_ODDS_FORMAT}&apiKey={api_key}"
    )
    today = dt.datetime.now(dt.UTC).strftime("%Y-%m-%d")
    raw_dir = settings.raw_dir / "oddsapi" / today
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_path = raw_dir / f"{sport}_{markets.replace(',', '+')}.json"

    audit_url = url.replace(api_key, "***REDACTED***")
    with audit_fetch(source="oddsapi", url=audit_url, user_agent=settings.user_agent) as audit:
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


def parse_raw_oddsapi(raw_path: Path) -> dict[str, int]:
    """Parser deferred — Phase 6 weekly live run.

    The payload is a JSON list of upcoming matches with bookmaker odds in
    structured format (per the-odds-api v4 guide). Match-to-fixture
    resolution at parse time uses (commence_time → kickoff date, home/away
    team name → dim_team) — same pattern as football-data.
    """
    payload = json.loads(raw_path.read_text())
    return {
        "matches_in_payload": len(payload),
        "fact_odds_written": 0,  # parser body filled in for Phase 6
        "status": "parser_deferred_to_phase_6",
    }


