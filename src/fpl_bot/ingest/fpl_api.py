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
from base64 import urlsafe_b64decode
from hashlib import sha256
from http.cookies import CookieError, SimpleCookie
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
    FactUserTeamSnapshot,
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


def fetch_my_team(team_id: int, gameweek: int) -> Path:
    """Phase 6: pull the user's team state for one GW.

    Endpoint: `entry/{team_id}/event/{gameweek}/picks/`
    Returns dict with per-pick `element` (FPL element_id), `position`,
    `is_captain`, `is_vice_captain`, `multiplier`, and an `entry_history`
    block with `bank` / `event_transfers` / ... and `picks` array.

    For the CURRENT GW (deadline not yet passed), use `my-team/{team_id}/`
    instead — but that requires an authenticated session. v1 uses the
    public picks endpoint for the last finished GW + we evolve state
    locally. Trade-off documented in design §2.1.
    """
    url = f"{FPL_BASE}/entry/{team_id}/event/{gameweek}/picks/"
    raw_path = _today_dir("fpl_api") / f"my_team_{team_id}_gw_{gameweek}.json"
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
        raw_path.write_bytes(payload)
    return raw_path


def fetch_current_my_team(team_id: int) -> Path:
    """Fetch the authenticated current-squad endpoint.

    This endpoint is the only FPL source that exposes exact purchase price,
    selling price, bank and free-transfer state before the deadline. It
    requires a current bearer token from the browser's `X-API-Authorization`
    request header. A legacy full Cookie header is also accepted and used as
    a fallback source when it contains an `access_token` cookie.
    """
    headers = _authenticated_fpl_headers(
        access_token=settings.fpl_access_token,
        cookie_header=settings.fpl_cookie,
    )
    url = f"{FPL_BASE}/my-team/{team_id}/"
    raw_path = _today_dir("fpl_api") / f"my_team_auth_{team_id}.json"
    with audit_fetch(source="fpl_api", url=url, user_agent=settings.user_agent) as audit:
        with httpx.Client(
            headers={"User-Agent": settings.user_agent, **headers},
            timeout=settings.request_timeout_seconds,
        ) as client:
            r = client.get(url)
            audit.response_code = r.status_code
            if r.status_code in (401, 403):
                raise RuntimeError(
                    "FPL rejected the my-team authentication. Refresh the logged-in "
                    "FPL page, copy the complete X-API-Authorization request header, "
                    "and set it as FPL_BOT_FPL_ACCESS_TOKEN in .env."
                )
            r.raise_for_status()
            payload = r.content
        audit.byte_size = len(payload)
        audit.content_hash = sha256(payload).hexdigest()
        audit.raw_path = str(raw_path)
        raw_path.write_bytes(payload)
    return raw_path


def _access_token_from_cookie(cookie_header: str | None) -> str | None:
    """Extract an access_token cookie without logging or persisting its value."""
    if not cookie_header:
        return None
    jar = SimpleCookie()
    try:
        jar.load(cookie_header)
    except CookieError:
        return None
    morsel = jar.get("access_token")
    return morsel.value if morsel and morsel.value else None


def _jwt_expiry(token: str) -> dt.datetime | None:
    """Read an unverified JWT expiry solely to produce an early useful error."""
    parts = token.split(".")
    if len(parts) != 3:
        return None
    try:
        payload_segment = parts[1]
        padding = "=" * (-len(payload_segment) % 4)
        payload = json.loads(urlsafe_b64decode(payload_segment + padding))
        expiry = payload.get("exp")
        if not isinstance(expiry, (int, float)):
            return None
        return dt.datetime.fromtimestamp(expiry, tz=dt.UTC)
    except (ValueError, TypeError, json.JSONDecodeError):
        return None


def _authenticated_fpl_headers(
    *,
    access_token: str | None,
    cookie_header: str | None,
    now: dt.datetime | None = None,
) -> dict[str, str]:
    """Build private-FPL headers and reject an obviously expired bearer token."""
    token = (access_token or _access_token_from_cookie(cookie_header) or "").strip()
    if token.lower().startswith("bearer "):
        token = token[7:].strip()
    if not token:
        raise RuntimeError(
            "Authenticated my-team ingest requires FPL_BOT_FPL_ACCESS_TOKEN. "
            "Copy the X-API-Authorization request header from a freshly loaded, "
            "logged-in fantasy.premierleague.com page."
        )

    expiry = _jwt_expiry(token)
    current = now or dt.datetime.now(dt.UTC)
    if expiry is not None and expiry <= current:
        raise RuntimeError(
            f"The FPL access token expired at {expiry.isoformat()}. Refresh the "
            "logged-in FPL page and replace FPL_BOT_FPL_ACCESS_TOKEN with the "
            "fresh X-API-Authorization request header."
        )

    headers = {"X-API-Authorization": f"Bearer {token}"}
    if cookie_header:
        headers["Cookie"] = cookie_header
    return headers


def _coerce_int(value: object, default: int = 0) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _latest_status_price(player_id: int) -> int:
    with session_scope() as s:
        stmt = (
            FactPlayerStatus.__table__.select()
            .where(FactPlayerStatus.player_id == player_id)
            .order_by(FactPlayerStatus.recorded_at.desc())
            .limit(1)
        )
        row = s.execute(stmt).first()
    return int(row.price_tenths) if row is not None else 50


def _bank_and_free_transfers(payload: dict) -> tuple[int, int]:
    """Extract bank + remaining free transfers from either payload shape.

    Authenticated `my-team` exposes a `transfers` block. The public historical
    picks endpoint does not expose the current FT allowance; for that shape we
    use a conservative default of 1 and let manual overrides correct it.
    """
    transfers = payload.get("transfers") or {}
    entry_history = payload.get("entry_history") or {}

    bank = _coerce_int(
        transfers.get("bank", entry_history.get("bank", 0)),
        default=0,
    )

    if "free_transfers" in transfers:
        return bank, max(0, _coerce_int(transfers.get("free_transfers"), default=1))
    if "free_transfers" in entry_history:
        return bank, max(0, _coerce_int(entry_history.get("free_transfers"), default=1))

    limit = transfers.get("limit")
    made = transfers.get("made")
    if limit is not None:
        return bank, max(0, _coerce_int(limit, default=1) - _coerce_int(made, default=0))

    return bank, 1


def _chips_used_from_payload(payload: dict) -> list[str]:
    """Best-effort chip extraction.

    The public endpoint only carries `active_chip` for that GW. Authenticated
    payloads vary across seasons, so this recognizes common active/played
    shapes but leaves exact season-slot correction to the live override file
    when needed.
    """
    out: list[str] = []
    active_chip = payload.get("active_chip")
    if active_chip:
        out.append(str(active_chip))

    for chip in payload.get("chips", []) or []:
        if not isinstance(chip, dict):
            continue
        name = chip.get("name")
        if not name:
            continue
        status = str(chip.get("status_for_entry") or chip.get("status") or "").lower()
        if status == "active":
            out.append(str(name))

    # Keep order but drop duplicates.
    deduped: list[str] = []
    for name in out:
        if name not in deduped:
            deduped.append(name)
    return deduped


def parse_my_team(
    raw_path: Path,
    season_id: int,
    gameweek: int,
    team_id: int,
) -> int:
    """Parse a my-team payload into fact_user_team_snapshot rows.

    Resolves per-season FPL element_id → stable player_id via
    `dim_player_season_xref`. Selling/purchase prices come from the
    `picks` array (FPL exposes both in the my-team auth endpoint; the
    public picks endpoint only has current price). For v1 we treat
    `selling_price` = `purchase_price` = current price (consistent
    with our static-price MILP simplification from Phase 3 v1.0); a
    later v2 can pull these from the authenticated my-team endpoint.

    Returns the count of rows inserted.
    """
    payload = json.loads(raw_path.read_text())
    picks = payload.get("picks", [])
    chips_used = _chips_used_from_payload(payload)
    bank_tenths, free_transfers = _bank_and_free_transfers(payload)

    with session_scope() as s:
        # Resolve element_id → stable player_id
        xref_rows = s.execute(
            DimPlayerSeasonXref.__table__.select().where(
                DimPlayerSeasonXref.season_id == season_id
            )
        ).all()
    element_to_pid = {r.fpl_element_id: r.player_id for r in xref_rows}

    n = 0
    with session_scope() as s:
        for pick in picks:
            element_id = pick["element"]
            player_id = element_to_pid.get(element_id)
            if player_id is None:
                continue  # promoted player not in xref yet; skip
            current_price = _latest_status_price(player_id)
            purchase_price = _coerce_int(
                pick.get("purchase_price", pick.get("purchase_price_tenths")),
                default=current_price,
            )
            selling_price = _coerce_int(
                pick.get("selling_price", pick.get("selling_price_tenths")),
                default=current_price,
            )

            ins = pg_insert(FactUserTeamSnapshot).values(
                season_id=season_id,
                gameweek=gameweek,
                team_id=team_id,
                player_id=player_id,
                purchase_price_tenths=purchase_price,
                selling_price_tenths=selling_price,
                multiplier=int(pick.get("multiplier", 1)),
                is_captain=bool(pick.get("is_captain", False)),
                is_vice=bool(pick.get("is_vice_captain", False)),
                position=int(pick.get("position", 0)),
                bank_tenths=int(bank_tenths),
                free_transfers=int(free_transfers),
                chips_used_json=json.dumps(chips_used),
            )
            s.execute(ins)
            n += 1
    return n
