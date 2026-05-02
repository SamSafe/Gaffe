"""FPL official API ingestion.

Endpoints (unofficial but stable):
  - bootstrap-static: players, teams, current GW, ownership, prices
  - fixtures: full season fixture list
  - event/{id}/live: live points (post-GW)
  - element-summary/{id}: per-player history (Phase 1B)

Two-layer per §2.3:
  fetch_raw_fpl_api(endpoint)  → data/raw/fpl_api/{date}/{endpoint}.json + audit row
  parse_raw_fpl_api(endpoint, raw_path) → typed inserts into dim_*/fact_* tables
"""
from __future__ import annotations

import datetime as dt
import json
from hashlib import sha256
from pathlib import Path

import httpx
from sqlalchemy.dialects.postgresql import insert as pg_insert

from fpl_bot.config import settings
from fpl_bot.db.models import (
    DimFixture,
    DimFixtureSeasonXref,
    DimPlayer,
    DimPlayerSeasonXref,
    DimTeam,
    FactPlayerStatus,
)
from fpl_bot.db.session import session_scope
from fpl_bot.ingest.audit import audit_fetch

FPL_BASE = "https://fantasy.premierleague.com/api"
ENDPOINTS: dict[str, str] = {
    "bootstrap-static": f"{FPL_BASE}/bootstrap-static/",
    "fixtures": f"{FPL_BASE}/fixtures/",
}


def _today_dir(source: str) -> Path:
    today = dt.datetime.now(dt.UTC).strftime("%Y-%m-%d")
    d = settings.raw_dir / source / today
    d.mkdir(parents=True, exist_ok=True)
    return d


def fetch_raw_fpl_api(endpoint: str) -> Path:
    """Fetch one FPL endpoint, write to disk, log to audit. Returns the raw path.

    Idempotent: if the upstream payload is byte-identical to what's already on
    disk for today, marks parse_status='skipped_unchanged' and does not rewrite.
    """
    if endpoint not in ENDPOINTS:
        raise ValueError(f"Unknown FPL endpoint: {endpoint!r}")
    url = ENDPOINTS[endpoint]
    raw_path = _today_dir("fpl_api") / f"{endpoint}.json"

    with audit_fetch(source="fpl_api", url=url, user_agent=settings.user_agent) as audit:
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


def parse_raw_fpl_api(endpoint: str, raw_path: Path, season_id: int) -> dict[str, int]:
    """Parse a raw payload into typed records. Returns counts per table."""
    payload = json.loads(raw_path.read_text())
    if endpoint == "bootstrap-static":
        return _parse_bootstrap_static(payload, season_id=season_id)
    if endpoint == "fixtures":
        return _parse_fixtures(payload, season_id=season_id)
    raise ValueError(f"No parser for endpoint: {endpoint!r}")


def _parse_bootstrap_static(payload: dict, season_id: int) -> dict[str, int]:
    teams_raw = payload["teams"]
    elements_raw = payload["elements"]
    element_types = {et["id"]: et["singular_name_short"] for et in payload["element_types"]}

    # Snapshot timestamp: FPL doesn't expose payload generation time; use ingest time.
    # event_time = recorded_at for this source; consumer treats as "we observed this at T".
    now_ts = dt.datetime.now(dt.UTC)
    counts = {"dim_team": 0, "dim_player": 0, "fact_player_status": 0}

    with session_scope() as s:
        for t in teams_raw:
            stmt = pg_insert(DimTeam).values(
                team_id=t["id"],
                season_id=season_id,
                short_name=t["short_name"],
                full_name=t["name"],
                promoted=False,  # populated separately when promoted-team list is known
            )
            stmt = stmt.on_conflict_do_update(
                index_elements=["team_id", "season_id"],
                set_={"short_name": stmt.excluded.short_name, "full_name": stmt.excluded.full_name},
            )
            s.execute(stmt)
            counts["dim_team"] += 1

        for e in elements_raw:
            stable_id = e["code"]  # FPL `code` is stable across seasons; `id` is not
            element_id = e["id"]   # per-season element id, used at ingest time only

            stmt = pg_insert(DimPlayer).values(
                player_id=stable_id,
                web_name=e["web_name"],
                first_name=e.get("first_name"),
                last_name=e.get("second_name"),
            )
            stmt = stmt.on_conflict_do_update(
                index_elements=["player_id"],
                set_={
                    "web_name": stmt.excluded.web_name,
                    "first_name": stmt.excluded.first_name,
                    "last_name": stmt.excluded.last_name,
                },
            )
            s.execute(stmt)
            counts["dim_player"] += 1

            xref_stmt = pg_insert(DimPlayerSeasonXref).values(
                season_id=season_id,
                fpl_element_id=element_id,
                player_id=stable_id,
            )
            xref_stmt = xref_stmt.on_conflict_do_update(
                index_elements=["season_id", "fpl_element_id"],
                set_={"player_id": xref_stmt.excluded.player_id},
            )
            s.execute(xref_stmt)
            counts["dim_player_season_xref"] = counts.get("dim_player_season_xref", 0) + 1

            s.add(
                FactPlayerStatus(
                    player_id=stable_id,
                    season_id=season_id,
                    position_code=element_types[e["element_type"]][:3].upper(),
                    team_id=e["team"],
                    price_tenths=e["now_cost"],
                    status_code=e["status"],
                    news=e.get("news") or None,
                    chance_of_playing_next_round=e.get("chance_of_playing_next_round"),
                    selected_by_percent=float(e["selected_by_percent"])
                    if e.get("selected_by_percent")
                    else None,
                    event_time=now_ts,
                )
            )
            counts["fact_player_status"] += 1

    return counts


def _parse_fixtures(payload: list[dict], season_id: int) -> dict[str, int]:
    counts = {"dim_fixture": 0, "dim_fixture_season_xref": 0}
    with session_scope() as s:
        for f in payload:
            if f.get("kickoff_time") is None:
                continue  # unscheduled fixture; skip until kickoff is set
            stable_id = f["code"]    # stable across seasons
            per_season_id = f["id"]  # per-season identifier
            kickoff = dt.datetime.fromisoformat(f["kickoff_time"].replace("Z", "+00:00"))

            stmt = pg_insert(DimFixture).values(
                fixture_id=stable_id,
                season_id=season_id,
                gameweek=f["event"] or 0,
                kickoff_utc=kickoff,
                home_team_id=f["team_h"],
                away_team_id=f["team_a"],
                finished=bool(f.get("finished", False)),
            )
            stmt = stmt.on_conflict_do_update(
                index_elements=["fixture_id"],
                set_={
                    "gameweek": stmt.excluded.gameweek,
                    "kickoff_utc": stmt.excluded.kickoff_utc,
                    "finished": stmt.excluded.finished,
                },
            )
            s.execute(stmt)
            counts["dim_fixture"] += 1

            xref_stmt = pg_insert(DimFixtureSeasonXref).values(
                season_id=season_id,
                fpl_fixture_id=per_season_id,
                fixture_id=stable_id,
            )
            xref_stmt = xref_stmt.on_conflict_do_update(
                index_elements=["season_id", "fpl_fixture_id"],
                set_={"fixture_id": xref_stmt.excluded.fixture_id},
            )
            s.execute(xref_stmt)
            counts["dim_fixture_season_xref"] += 1
    return counts


def latest_raw_for_today(endpoint: str) -> Path | None:
    """Find today's raw file for an endpoint; useful for parse-only runs."""
    today_dir = _today_dir("fpl_api")
    p = today_dir / f"{endpoint}.json"
    return p if p.exists() else None
