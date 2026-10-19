"""Per-gameweek transfers, ownership and price for the current season.

The live event feed that fills ``fact_player_match`` in season omits these
fields, but the minutes model's market features need them for every training
row. Each player's ``element-summary`` history carries them for every round,
so one pass over the squad lists fills every missing gameweek at once.

Rows are written as new bitemporal copies of the latest player/fixture row,
the same pattern as the Phase 3.5 transfers backfill, so the PIT reader's
latest-row rule picks them up without losing the match statistics.
"""

from __future__ import annotations

import datetime as dt
import json
import time
from pathlib import Path

import httpx
import polars as pl
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert

from fpl_bot.config import settings
from fpl_bot.db import pit
from fpl_bot.db.models import DimFixtureSeasonXref, DimPlayerSeasonXref, FactPlayerMatch
from fpl_bot.db.session import session_scope
from fpl_bot.ingest.fpl_api import FPL_BASE, _today_dir

MARKET_FIELDS = ("transfers_in", "transfers_out", "transfers_balance", "selected", "price_tenths")


def missing_market_gameweeks(season_id: int, tolerance: float = 0.2) -> list[int]:
    """Gameweeks where most rows still lack transfers/ownership.

    A few rows stay empty for good (players who left the league have no
    element-summary), so a small remainder must not trigger a fresh pass.
    """
    rows = pit.all_player_match_with_kickoff(season_ids=[season_id])
    if rows.is_empty():
        return []
    share = rows.group_by("gameweek").agg(pl.col("selected").is_null().mean().alias("missing"))
    return sorted(share.filter(pl.col("missing") > tolerance)["gameweek"].to_list())


def fetch_raw_element_summaries(element_ids: list[int], pause_s: float = 0.1) -> Path:
    """Fetch each player's FPL history into one raw file. ~800 polite requests."""
    raw_path = _today_dir("fpl_api") / "element_summaries.json"
    histories: dict[str, list[dict]] = {}
    with httpx.Client(
        headers={"User-Agent": settings.user_agent},
        timeout=settings.request_timeout_seconds,
    ) as client:
        for element_id in element_ids:
            r = client.get(f"{FPL_BASE}/element-summary/{element_id}/")
            r.raise_for_status()
            histories[str(element_id)] = r.json().get("history", [])
            time.sleep(pause_s)
    raw_path.write_text(
        json.dumps({"fetched_at": dt.datetime.now(dt.UTC).isoformat(), "history": histories})
    )
    return raw_path


def parse_raw_element_summaries(raw_path: Path, season_id: int) -> dict[str, int]:
    """Write market fields onto the latest player/fixture rows that lack them."""
    histories = json.loads(raw_path.read_text())["history"]
    with session_scope() as s:
        element_to_pid = {
            int(r.fpl_element_id): int(r.player_id)
            for r in s.execute(
                select(DimPlayerSeasonXref).where(DimPlayerSeasonXref.season_id == season_id)
            ).scalars()
        }
        fixture_to_id = {
            int(r.fpl_fixture_id): int(r.fixture_id)
            for r in s.execute(
                select(DimFixtureSeasonXref).where(DimFixtureSeasonXref.season_id == season_id)
            ).scalars()
        }
    values = {}
    for element_id, rows in histories.items():
        pid = element_to_pid.get(int(element_id))
        for row in rows:
            fid = fixture_to_id.get(int(row["fixture"]))
            if pid is None or fid is None:
                continue
            values[(pid, fid)] = {
                "transfers_in": row["transfers_in"],
                "transfers_out": row["transfers_out"],
                "transfers_balance": row["transfers_balance"],
                "selected": row["selected"],
                "price_tenths": row["value"],
            }
    counts = {"history_rows": len(values), "fact_player_match": 0}
    columns = [c.name for c in FactPlayerMatch.__table__.columns if c.name != "recorded_at"]
    with session_scope() as s:
        latest = {}
        for row in s.execute(
            select(FactPlayerMatch)
            .where(FactPlayerMatch.fixture_id.in_({fid for _, fid in values}))
            .order_by(FactPlayerMatch.recorded_at)
        ).scalars():
            latest[(row.player_id, row.fixture_id)] = row
        inserts = []
        for key, row in latest.items():
            market = values.get(key)
            if market is None or row.selected is not None:
                continue
            inserts.append({**{c: getattr(row, c) for c in columns}, **market})
        if inserts:
            s.execute(pg_insert(FactPlayerMatch), inserts)
        counts["fact_player_match"] = len(inserts)
    return counts


def backfill_market_history(season_id: int, element_ids: list[int]) -> dict[str, int]:
    """Fill every missing gameweek in one pass, or do nothing if none are missing."""
    missing = missing_market_gameweeks(season_id)
    if not missing:
        return {"missing_gameweeks": 0, "fact_player_match": 0}
    counts = parse_raw_element_summaries(fetch_raw_element_summaries(element_ids), season_id)
    return {"missing_gameweeks": len(missing), **counts}
