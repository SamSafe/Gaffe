"""Player-prop (anytime-goalscorer) ingest — The Odds API per-event endpoint.

Phase 8. The match-odds ingest (`ingest/oddsapi.py`) uses the cheap
league-wide `/odds` endpoint; player props are only available per event, so
this module lists events first (free) and then pulls one market per event.

Credit accounting, measured against the live API on 2026-07-26:
  - `/sports/{sport}/events` costs **0** credits.
  - `/events/{id}/odds` costs 1 credit per market per region **when a
    bookmaker returns prices**; a response with `bookmakers: []` costs **0**.
So probing early in the week is free, and a full GW1-sized pull (10 fixtures,
1 market, 1 region) costs ~10 of the 500 free monthly credits.

Name resolution is the failure mode worth designing around: bookmakers write
"Bukayo Saka" where FPL has web_name "Saka". Candidates are therefore
restricted to the two clubs actually in the fixture before any surname match
is attempted, which removes nearly all of the ambiguity that makes fuzzy
matching dangerous. Anything still ambiguous is skipped and counted, never
guessed — a wrong player_id would silently move goal rate to the wrong player.

See docs/design/phase8_player_prop_odds.md.
"""
from __future__ import annotations

import datetime as dt
import json
import re
import unicodedata
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

import httpx
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert

from fpl_bot.config import settings
from fpl_bot.db.models import DimFixture, DimPlayer, FactPlayerOdds, FactPlayerStatus
from fpl_bot.db.session import session_scope
from fpl_bot.ingest.audit import audit_fetch
from fpl_bot.ingest.oddsapi import OA_TO_FPL_SHORT, ODDS_API_BASE

EPL_SPORT_KEY = "soccer_epl"
ANYTIME_MARKET = "player_goal_scorer_anytime"
# Soccer player props on the free tier are carried by US books; `uk` returns
# nothing for this market. Kept explicit because it drives credit cost.
DEFAULT_PROP_REGIONS = "us"
# Our canonical market code in fact_player_odds.
MARKET_CODE = "anytime_goalscorer"
# Only pull events kicking off within this many days — one gameweek's worth.
DEFAULT_HORIZON_DAYS = 8
# Hard ceiling on per-event requests in one run, so a double gameweek (or a
# mis-set horizon) cannot quietly drain the monthly credit budget.
DEFAULT_MAX_EVENTS = 14


# ─── Fetch ─────────────────────────────────────────────────────────────────


def fetch_raw_player_props(
    sport: str = EPL_SPORT_KEY,
    regions: str = DEFAULT_PROP_REGIONS,
    horizon_days: int = DEFAULT_HORIZON_DAYS,
    max_events: int = DEFAULT_MAX_EVENTS,
) -> Path:
    """Fetch anytime-goalscorer props for upcoming events. Returns the raw path.

    Writes one JSON file containing the per-event payloads, plus a
    `*.meta.json` sidecar recording the remaining credit quota (as
    `ingest/oddsapi.py` does) and which events were requested.
    """
    api_key = settings.odds_api_key
    if not api_key:
        raise RuntimeError(
            "the-odds-api requires a free-tier API key. "
            "Set FPL_BOT_ODDS_API_KEY in your environment "
            "(register at https://the-odds-api.com)."
        )

    today = dt.datetime.now(dt.UTC).strftime("%Y-%m-%d")
    raw_dir = settings.raw_dir / "oddsapi" / today
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_path = raw_dir / f"{sport}_player_props.json"

    events_url = f"{ODDS_API_BASE}/sports/{sport}/events"
    remaining = used = None
    now = dt.datetime.now(dt.UTC)
    horizon = now + dt.timedelta(days=horizon_days)

    with httpx.Client(
        headers={"User-Agent": settings.user_agent},
        timeout=settings.request_timeout_seconds,
    ) as client:
        with audit_fetch(
            source="oddsapi_props",
            url=events_url,
            user_agent=settings.user_agent,
        ) as audit:
            r = client.get(events_url, params={"apiKey": api_key})
            audit.response_code = r.status_code
            r.raise_for_status()
            events = r.json()
            audit.byte_size = len(r.content)
            audit.content_hash = sha256(r.content).hexdigest()

        upcoming = [
            e
            for e in events
            if (ct := _parse_api_timestamp(e.get("commence_time"))) is not None
            and now <= ct <= horizon
        ]
        upcoming.sort(key=lambda e: e["commence_time"])
        selected = upcoming[:max_events]

        event_payloads: list[dict[str, Any]] = []
        for event in selected:
            odds_url = f"{ODDS_API_BASE}/sports/{sport}/events/{event['id']}/odds"
            audit_url = f"{odds_url}?markets={ANYTIME_MARKET}&regions={regions}"
            with audit_fetch(
                source="oddsapi_props",
                url=audit_url,
                user_agent=settings.user_agent,
            ) as audit:
                r = client.get(
                    odds_url,
                    params={
                        "apiKey": api_key,
                        "regions": regions,
                        "markets": ANYTIME_MARKET,
                        "oddsFormat": "decimal",
                    },
                )
                audit.response_code = r.status_code
                r.raise_for_status()
                payload = r.json()
                audit.byte_size = len(r.content)
                audit.content_hash = sha256(r.content).hexdigest()
                audit.raw_path = str(raw_path)
            event_payloads.append(payload)
            remaining = r.headers.get("x-requests-remaining", remaining)
            used = r.headers.get("x-requests-used", used)

    raw_path.write_text(json.dumps(event_payloads, indent=2))
    priced = sum(1 for p in event_payloads if p.get("bookmakers"))
    raw_path.with_suffix(".meta.json").write_text(
        json.dumps(
            {
                "fetched_at": now.isoformat(),
                "sport": sport,
                "markets": ANYTIME_MARKET,
                "regions": regions,
                "horizon_days": horizon_days,
                "events_upcoming": len(upcoming),
                "events_requested": len(selected),
                "events_with_prices": priced,
                "remaining_requests": remaining,
                "used_requests": used,
            },
            indent=2,
        )
    )
    return raw_path


# ─── Name resolution (pure) ────────────────────────────────────────────────


@dataclass(frozen=True)
class PlayerRow:
    """Minimal player identity needed to resolve a bookmaker's spelling."""

    player_id: int
    web_name: str
    first_name: str | None
    last_name: str | None
    team_id: int | None


def normalize_name(value: str) -> str:
    """Casefold, strip accents and punctuation, collapse whitespace.

    "Ødegaard" and "Odegaard", "M.Salah" and "M Salah" must compare equal —
    bookmakers and FPL disagree on all of these.
    """
    decomposed = unicodedata.normalize("NFKD", value)
    ascii_only = "".join(c for c in decomposed if not unicodedata.combining(c))
    # Keep the 'ø'→'o' class of mapping working: NFKD leaves some letters
    # untouched, so fold the remaining non-ASCII letters explicitly.
    ascii_only = ascii_only.replace("ø", "o").replace("Ø", "O")
    ascii_only = ascii_only.replace("ł", "l").replace("Ł", "L")
    ascii_only = ascii_only.replace("đ", "d").replace("Đ", "D")
    cleaned = re.sub(r"[^\w\s]", " ", ascii_only)
    return " ".join(cleaned.casefold().split())


class NameIndex:
    """Lookup structures for resolving a bookmaker player name to a player_id."""

    def __init__(self, players: list[PlayerRow]) -> None:
        self._by_team: dict[int | None, list[PlayerRow]] = {}
        for p in players:
            self._by_team.setdefault(p.team_id, []).append(p)
        self._players = players
        self._tokens: dict[int, frozenset[str]] = {
            p.player_id: frozenset(
                normalize_name(
                    f"{p.first_name or ''} {p.last_name or ''} {p.web_name}"
                ).split()
            )
            for p in players
        }

    def candidates(self, team_ids: set[int] | None) -> list[PlayerRow]:
        if team_ids is None:
            return self._players
        out: list[PlayerRow] = []
        for team_id in team_ids:
            out.extend(self._by_team.get(team_id, []))
        return out

    def tokens(self, player: PlayerRow) -> frozenset[str]:
        """Every name token FPL knows for this player, normalized."""
        return self._tokens.get(player.player_id, frozenset())


def resolve_player_name(
    source_name: str,
    index: NameIndex,
    team_ids: set[int] | None = None,
) -> int | None:
    """Bookmaker player name → FPL player_id, or None when not confidently resolved.

    Tried in descending confidence, always within the fixture's two clubs when
    `team_ids` is given:
      1. full name ("first last") exact
      2. web_name exact
      3. surname exact against `last_name` or `web_name`, unique
      4. surname + first initial, unique
      5. token subset — every token of the bookmaker's name appears among the
         player's tokens, unique

    Rule 5 exists because FPL stores full Iberian/Brazilian surnames while books
    use the common short form: "Gabriel Martinelli" against first_name
    "Gabriel" / last_name "Martinelli Silva" matches on nothing above. It stays
    safe by still demanding a unique hit.

    Note the precedence that resolves Arsenal's two Gabriels: a bare "Gabriel"
    hits rule 2 (web_name) and lands on Gabriel Magalhães, whose FPL web_name is
    exactly "Gabriel", while "Gabriel Martinelli" falls through to rule 5. This
    is intended — FPL's web_name follows the same broadcast convention the books
    do — but it is a real precedence decision, not an accident.

    Returns None on ambiguity rather than picking one. A wrong player_id would
    silently transplant a striker's goal rate onto a teammate, which is worse
    than having no prop for that player at all.
    """
    target = normalize_name(source_name)
    if not target:
        return None
    candidates = index.candidates(team_ids)
    if not candidates:
        return None

    def unique(matches: list[PlayerRow]) -> int | None:
        ids = {m.player_id for m in matches}
        return ids.pop() if len(ids) == 1 else None

    full = [
        c
        for c in candidates
        if normalize_name(f"{c.first_name or ''} {c.last_name or ''}") == target
    ]
    if (hit := unique(full)) is not None:
        return hit

    web = [c for c in candidates if normalize_name(c.web_name) == target]
    if (hit := unique(web)) is not None:
        return hit

    target_tokens = target.split()
    surname = target_tokens[-1]
    by_surname = [
        c
        for c in candidates
        if surname in {normalize_name(c.last_name or ""), normalize_name(c.web_name)}
    ]
    if (hit := unique(by_surname)) is not None:
        return hit

    if len(target_tokens) >= 2 and by_surname:
        initial = target_tokens[0][:1]
        narrowed = [
            c
            for c in by_surname
            if normalize_name(c.first_name or "").startswith(initial)
        ]
        if (hit := unique(narrowed)) is not None:
            return hit

    target_token_set = set(target_tokens)
    by_tokens = [c for c in candidates if target_token_set <= index.tokens(c)]
    if (hit := unique(by_tokens)) is not None:
        return hit
    return None


# ─── Parse ─────────────────────────────────────────────────────────────────


def parse_raw_player_props(raw_path: Path, season_id: int) -> dict[str, int]:
    """Parse per-event prop payloads → `fact_player_odds` rows.

    Idempotent on (fixture, player, bookmaker, market, quote_time): a re-parse
    of the same pull updates the price in place, while a later pull lands as a
    new snapshot.
    """
    payloads = json.loads(raw_path.read_text())
    counts = {
        "fact_player_odds": 0,
        "events_in_payload": len(payloads),
        "events_with_prices": 0,
        "skipped_unmapped_team": 0,
        "skipped_no_fixture": 0,
        "skipped_unresolved_player": 0,
        "skipped_no_outcome": 0,
    }
    fetched_at = _raw_fetched_at(raw_path)
    counts["events_with_prices"] = sum(1 for e in payloads if e.get("bookmakers"))
    if not counts["events_with_prices"]:
        # Normal until ~2-3 days before kickoff. Return without opening a
        # transaction or building lookups — there is nothing to insert, and a
        # weekly probe should be free in every sense.
        return counts

    index, short_to_team_id, fixture_lookup = _build_lookups(season_id)
    unresolved: set[str] = set()

    with session_scope() as s:
        for event in payloads:
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

            event_time = _parse_api_timestamp(event.get("commence_time"))
            if event_time is None:
                counts["skipped_no_outcome"] += 1
                continue
            fixture_id = fixture_lookup.get((event_time.date(), home_id, away_id))
            if fixture_id is None:
                counts["skipped_no_fixture"] += 1
                continue
            team_ids = {home_id, away_id}

            for bk in event.get("bookmakers", []) or []:
                bk_code = f"OA_{bk['key']}"
                for market in bk.get("markets", []) or []:
                    if market.get("key") != ANYTIME_MARKET:
                        continue
                    quote_time = (
                        _parse_api_timestamp(market.get("last_update"))
                        or _parse_api_timestamp(bk.get("last_update"))
                        or fetched_at
                    )
                    for outcome in market.get("outcomes", []) or []:
                        parsed = _parse_prop_outcome(outcome)
                        if parsed is None:
                            counts["skipped_no_outcome"] += 1
                            continue
                        source_name, decimal_odds = parsed
                        player_id = resolve_player_name(source_name, index, team_ids)
                        if player_id is None:
                            counts["skipped_unresolved_player"] += 1
                            unresolved.add(source_name)
                            continue
                        s.execute(
                            pg_insert(FactPlayerOdds)
                            .values(
                                fixture_id=fixture_id,
                                player_id=player_id,
                                bookmaker=bk_code,
                                market=MARKET_CODE,
                                quote_time=quote_time,
                                event_time=event_time,
                                decimal_odds=decimal_odds,
                                source_player_name=source_name,
                            )
                            .on_conflict_do_update(
                                index_elements=[
                                    "fixture_id",
                                    "player_id",
                                    "bookmaker",
                                    "market",
                                    "quote_time",
                                ],
                                set_={"decimal_odds": decimal_odds},
                            )
                        )
                        counts["fact_player_odds"] += 1

    if unresolved:
        # Surfaced rather than logged-and-forgotten: a systematic naming change
        # at one book shows up here as a block of misses.
        counts["distinct_unresolved_names"] = len(unresolved)
        raw_path.with_suffix(".unresolved.json").write_text(
            json.dumps(sorted(unresolved), indent=2)
        )
    return counts


def _parse_prop_outcome(outcome: dict[str, Any]) -> tuple[str, float] | None:
    """Extract (player name, decimal odds) from one anytime-scorer outcome.

    The Odds API puts the player in `description` and Yes/No in `name` for
    player props, but has historically also emitted the player in `name` with
    no description. Only the "Yes" side is stored — the derive step inverts
    P(scores) directly.
    """
    name = (outcome.get("name") or "").strip()
    description = (outcome.get("description") or "").strip()
    if description:
        if name and name.casefold() not in {"yes", "over"}:
            return None
        source_name = description
    else:
        if not name or name.casefold() in {"yes", "no", "over", "under"}:
            return None
        source_name = name
    price = outcome.get("price")
    try:
        decimal_odds = float(price)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if decimal_odds <= 1.0:
        return None
    return source_name, decimal_odds


def _build_lookups(
    season_id: int,
) -> tuple[NameIndex, dict[str, int], dict[tuple[dt.date, int, int], int]]:
    """(name index, short_name→team_id, (kickoff date, home, away)→fixture_id)."""
    from fpl_bot.db.models import DimTeam

    with session_scope() as s:
        team_rows = s.execute(
            select(DimTeam.short_name, DimTeam.team_id).where(
                DimTeam.season_id == season_id
            )
        ).all()
        fixture_rows = s.execute(
            select(
                DimFixture.fixture_id,
                DimFixture.kickoff_utc,
                DimFixture.home_team_id,
                DimFixture.away_team_id,
            ).where(DimFixture.season_id == season_id)
        ).all()
        # Latest status row per player gives the club they are at now, which is
        # what scopes name resolution to the two teams in the fixture.
        status_rows = s.execute(
            select(
                FactPlayerStatus.player_id,
                FactPlayerStatus.team_id,
                FactPlayerStatus.recorded_at,
            ).where(FactPlayerStatus.season_id == season_id)
        ).all()
        player_rows = s.execute(
            select(
                DimPlayer.player_id,
                DimPlayer.web_name,
                DimPlayer.first_name,
                DimPlayer.last_name,
            )
        ).all()

    latest_team: dict[int, tuple[dt.datetime, int]] = {}
    for row in status_rows:
        prev = latest_team.get(row.player_id)
        if prev is None or row.recorded_at > prev[0]:
            latest_team[row.player_id] = (row.recorded_at, row.team_id)

    players = [
        PlayerRow(
            player_id=r.player_id,
            web_name=r.web_name,
            first_name=r.first_name,
            last_name=r.last_name,
            team_id=latest_team.get(r.player_id, (None, None))[1],
        )
        for r in player_rows
    ]
    short_to_team_id = {r.short_name: r.team_id for r in team_rows}
    fixture_lookup = {
        (r.kickoff_utc.astimezone(dt.UTC).date(), r.home_team_id, r.away_team_id): r.fixture_id
        for r in fixture_rows
    }
    return NameIndex(players), short_to_team_id, fixture_lookup


def _parse_api_timestamp(value: object) -> dt.datetime | None:
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
