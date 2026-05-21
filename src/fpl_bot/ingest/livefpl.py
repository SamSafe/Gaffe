"""LiveFPL top-10k EO ingest (Phase 7 1.3).

Previously deferred (the site was suspected to be a JS SPA), but the
`/EO` endpoint actually renders the full top-10k vs overall EO table
inline as static HTML — fetchable with a simple GET. This module wires
it up.

Source page: https://www.livefpl.net/EO

Columns per row (table#main):
  ID (hidden) | Type "n-POS" (hidden) | web_name | team
  Top10k_EO% | Top10k_(C)% | Top10k_(TC)% | Overall_EO% | Overall_(C)% | Overall_(TC)% | Score

We capture top10k + overall as two `fact_eo_snapshot` rows per player
per snapshot, with source="livefpl".

Compliance: robots.txt is unrestricted. User-Agent identifies us.

Column convention: LiveFPL's "EO" already follows the standard FPL
effective-ownership formula (ownership% + captaincy%). So:
  effective_ownership = LiveFPL "EO"  (the headline figure)
  captaincy_pct       = LiveFPL "(C)"
  ownership_pct       = LiveFPL "EO" − LiveFPL "(C)"  (back out raw ownership)
"""
from __future__ import annotations

import datetime as dt
import re
from hashlib import sha256
from pathlib import Path

import httpx
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert

from fpl_bot.config import settings
from fpl_bot.db.models import DimPlayer, DimPlayerSeasonXref, DimTeam, FactEoSnapshot
from fpl_bot.db.session import session_scope
from fpl_bot.ingest.audit import audit_fetch

LIVEFPL_EO_URL = "https://www.livefpl.net/EO"

# Row matcher: each player row begins with <tr style="color:black">.
_ROW_RE = re.compile(r'<tr style="color:black">(.*?)</tr>', re.DOTALL)
_TD_RE = re.compile(r"<td[^>]*>(.*?)</td>", re.DOTALL)


def fetch_raw_livefpl() -> Path:
    """Download the LiveFPL /EO HTML page → data/raw/livefpl/{date}/EO.html."""
    today = dt.datetime.now(dt.UTC).strftime("%Y-%m-%d")
    raw_dir = settings.raw_dir / "livefpl" / today
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_path = raw_dir / "EO.html"

    with audit_fetch(
        source="livefpl", url=LIVEFPL_EO_URL, user_agent=settings.user_agent
    ) as audit:
        with httpx.Client(
            headers={"User-Agent": settings.user_agent},
            timeout=settings.request_timeout_seconds,
            follow_redirects=True,
        ) as client:
            r = client.get(LIVEFPL_EO_URL)
            audit.response_code = r.status_code
            r.raise_for_status()
            payload = r.content
        audit.byte_size = len(payload)
        audit.content_hash = sha256(payload).hexdigest()
        audit.raw_path = str(raw_path)
        raw_path.write_bytes(payload)
    return raw_path


def parse_raw_livefpl(
    raw_path: Path,
    *,
    season_id: int,
    gameweek: int,
    event_time: dt.datetime | None = None,
) -> dict[str, int]:
    """Parse the EO HTML → `fact_eo_snapshot` rows (one per rank_band per player).

    Returns counts dict. Idempotent: ON CONFLICT on the composite PK
    (player, season, gw, rank_band, source, event_time) does nothing —
    re-runs at the same event_time are no-ops.
    """
    if event_time is None:
        event_time = dt.datetime.now(dt.UTC).replace(second=0, microsecond=0)
    html = raw_path.read_text()
    counts = {
        "rows_seen": 0,
        "fact_eo_snapshot": 0,
        "skipped_unmapped_player": 0,
        "skipped_parse_error": 0,
    }
    web_to_pid, _ = _build_lookups(season_id)
    rows = _ROW_RE.findall(html)
    counts["rows_seen"] = len(rows)

    with session_scope() as s:
        for row_html in rows:
            cells = _TD_RE.findall(row_html)
            if len(cells) < 10:
                counts["skipped_parse_error"] += 1
                continue
            # cells: [id, type, web_name, team, top10k_eo, top10k_c, top10k_tc,
            #         overall_eo, overall_c, overall_tc, score]
            web_name = _clean(cells[2])
            try:
                t10_eo = _pct(cells[4])
                t10_c = _pct(cells[5])
                ov_eo = _pct(cells[7])
                ov_c = _pct(cells[8])
            except ValueError:
                counts["skipped_parse_error"] += 1
                continue
            pid = web_to_pid.get(web_name)
            if pid is None:
                counts["skipped_unmapped_player"] += 1
                continue
            for band, livefpl_eo, c_pct in (
                ("top10k", t10_eo, t10_c),
                ("overall", ov_eo, ov_c),
            ):
                # LiveFPL's "EO" already follows the standard formula
                # (ownership + captaincy). Back out raw ownership.
                ownership = max(0.0, livefpl_eo - c_pct)
                stmt = (
                    pg_insert(FactEoSnapshot)
                    .values(
                        player_id=pid,
                        season_id=season_id,
                        gameweek=gameweek,
                        rank_band=band,
                        source="livefpl",
                        event_time=event_time,
                        ownership_pct=ownership,
                        captaincy_pct=c_pct,
                        effective_ownership=livefpl_eo,
                        provenance_url=LIVEFPL_EO_URL,
                    )
                    .on_conflict_do_nothing(
                        index_elements=[
                            "player_id",
                            "season_id",
                            "gameweek",
                            "rank_band",
                            "source",
                            "event_time",
                        ],
                    )
                )
                s.execute(stmt)
                counts["fact_eo_snapshot"] += 1
    return counts


def _clean(html_fragment: str) -> str:
    """Strip tags + collapse whitespace from a TD inner content."""
    text = re.sub(r"<[^>]+>", "", html_fragment)
    return text.strip()


def _pct(html_fragment: str) -> float:
    """Convert '27.85%' to 27.85. Empty / '-' → 0.0."""
    text = _clean(html_fragment)
    if not text or text in ("-", ""):
        return 0.0
    text = text.replace("%", "").strip()
    return float(text)


def _build_lookups(
    season_id: int,
) -> tuple[dict[str, int], dict[str, str]]:
    """(web_name → player_id, full_team_name → short_name) for this season."""
    with session_scope() as s:
        player_rows = s.execute(
            select(DimPlayer.player_id, DimPlayer.web_name)
            .join(
                DimPlayerSeasonXref,
                DimPlayerSeasonXref.player_id == DimPlayer.player_id,
            )
            .where(DimPlayerSeasonXref.season_id == season_id)
        ).all()
        team_rows = s.execute(
            select(DimTeam.full_name, DimTeam.short_name).where(
                DimTeam.season_id == season_id
            )
        ).all()
    return (
        {p.web_name: int(p.player_id) for p in player_rows},
        {t.full_name: t.short_name for t in team_rows},
    )
