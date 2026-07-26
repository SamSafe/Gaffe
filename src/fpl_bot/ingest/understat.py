"""Understat per-match aggregates via vaastav's mirror (Phase 1B source #3).

Compliance: understat.com's robots.txt disallows all paths for all UAs, so
we cannot scrape it directly. Instead we read vaastav's redistributed
mirror at:
    data/raw/vaastav/Fantasy-Premier-League/data/{season}/understat/{Name}_{id}.csv

Cost vs direct scrape: per-match aggregate only — no shot coordinates,
situation, or body-part. Sufficient for §4.2 rolling xG/90 features;
insufficient for set-piece-taker derivation (which becomes a manual list).

Two-layer per §2.3:
  fetch_raw_understat(season): no-op shim that audits the read of the
    vaastav mirror (enforces explicit dependency)
  parse_raw_understat_season(season): per-CSV ingestion + name/date
    matching to populate dim_player.understat_id and fixture_id link
"""
from __future__ import annotations

import datetime as dt
import re
from pathlib import Path

import polars as pl
from sqlalchemy import select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert

from fpl_bot.config import settings
from fpl_bot.db.models import (
    DimFixture,
    DimPlayer,
    DimTeam,
    FactUnderstatPlayerMatch,
)
from fpl_bot.db.session import session_scope
from fpl_bot.ingest.audit import audit_fetch
from fpl_bot.ingest.vaastav import SEASON_ID_FROM_FOLDER

# Understat full team name → FPL full name (matches dim_team.full_name).
US_TO_FPL_FULL_NAME = {
    "Arsenal": "Arsenal",
    "Aston Villa": "Aston Villa",
    "Bournemouth": "Bournemouth",
    "Brentford": "Brentford",
    "Brighton": "Brighton",
    "Burnley": "Burnley",
    "Chelsea": "Chelsea",
    "Coventry": "Coventry City",
    "Coventry City": "Coventry City",
    "Crystal Palace": "Crystal Palace",
    "Everton": "Everton",
    "Fulham": "Fulham",
    "Hull": "Hull City",
    "Hull City": "Hull City",
    "Ipswich": "Ipswich Town",
    "Leeds": "Leeds",
    "Leicester": "Leicester",
    "Liverpool": "Liverpool",
    "Luton": "Luton",
    "Manchester City": "Man City",
    "Manchester United": "Man Utd",
    "Newcastle United": "Newcastle",
    "Norwich": "Norwich",
    "Nottingham Forest": "Nott'm Forest",
    "Sheffield United": "Sheffield Utd",
    "Southampton": "Southampton",
    "Sunderland": "Sunderland",
    "Tottenham": "Spurs",
    "Watford": "Watford",
    "West Bromwich Albion": "West Brom",
    "West Ham": "West Ham",
    "Wolverhampton Wanderers": "Wolves",
}

# FPL renames some clubs between seasons, so a single mapping above cannot
# match `dim_team.full_name` for every season a club appears in (e.g. IPS was
# "Ipswich" in 2024-25 and "Ipswich Town" in 2026-27). The fixture lookup
# spans all seasons, so try these alternates before giving up on a match.
FPL_FULL_NAME_ALTERNATES: dict[str, tuple[str, ...]] = {
    "Ipswich Town": ("Ipswich",),
    "Coventry City": ("Coventry",),
    "Hull City": ("Hull",),
}


def _fpl_full_name_candidates(understat_name: str | None) -> tuple[str, ...]:
    """FPL full names an Understat team name may correspond to, best first."""
    if not understat_name:
        return ()
    primary = US_TO_FPL_FULL_NAME.get(understat_name)
    if primary is None:
        return ()
    return (primary, *FPL_FULL_NAME_ALTERNATES.get(primary, ()))

# Filename: "First_Last_understatid.csv"; allow accents/hyphens in name.
FILENAME_RE = re.compile(r"^(.+?)_(\d+)\.csv$")


def _vaastav_understat_dir(season_folder: str) -> Path:
    return (
        settings.raw_dir
        / "vaastav"
        / "Fantasy-Premier-League"
        / "data"
        / season_folder
        / "understat"
    )


def fetch_raw_understat(season_folder: str) -> Path:
    """No-op shim that audits the read of vaastav's bundled understat mirror.

    Direct understat.com fetching is blocked by robots.txt. The actual files
    are pulled via `fpl-bot ingest vaastav --raw-only`.
    """
    path = _vaastav_understat_dir(season_folder)
    if not path.exists():
        raise FileNotFoundError(
            f"vaastav understat dir missing for {season_folder}; "
            "run `fpl-bot ingest vaastav --raw-only` first"
        )
    with audit_fetch(
        source="understat_via_vaastav",
        url=f"local-mirror://{path}",
        user_agent=settings.user_agent,
    ) as audit:
        audit.raw_path = str(path)
        audit.byte_size = sum(f.stat().st_size for f in path.glob("*.csv"))
        audit.response_code = 200
        audit.parse_status = "ok"
    return path


def parse_raw_understat_season(season_folder: str) -> dict[str, int]:
    """Ingest one season's understat per-match aggregates.

    Note: vaastav's per-season understat folder contains each player's FULL
    career history (not just that season's matches). We ingest all of it
    because cross-league xG history is useful for cold-start of new signings.
    Fixture-linking spans ALL ingested seasons; non-EPL matches stay NULL.
    """
    if season_folder not in SEASON_ID_FROM_FOLDER:
        raise ValueError(f"Unknown season folder: {season_folder!r}")
    understat_dir = _vaastav_understat_dir(season_folder)
    if not understat_dir.exists():
        raise FileNotFoundError(f"vaastav understat dir missing: {understat_dir}")

    csv_files = sorted(understat_dir.glob("*.csv"))
    fixture_lookup, name_to_player_id, last_to_player_id = _build_lookups()

    counts = {
        "files": 0,
        "rows": 0,
        "linked_to_player": 0,
        "linked_to_fixture": 0,
        "unmatched_player": 0,
        "unmatched_fixture": 0,
    }

    with session_scope() as s:
        for csv_path in csv_files:
            m = FILENAME_RE.match(csv_path.name)
            if not m:
                continue
            counts["files"] += 1
            name_underscored, understat_id_str = m.groups()
            understat_id = int(understat_id_str)
            display_name = name_underscored.replace("_", " ")

            # Match to dim_player by full name; fall back to last-name only
            player_id = name_to_player_id.get(display_name)
            if player_id is None:
                last = display_name.rsplit(" ", 1)[-1].lower()
                player_id = last_to_player_id.get(last)
            if player_id is not None:
                counts["linked_to_player"] += 1
                s.execute(
                    update(DimPlayer)
                    .where(DimPlayer.player_id == player_id)
                    .values(understat_id=understat_id)
                )
            else:
                counts["unmatched_player"] += 1

            df = pl.read_csv(csv_path, ignore_errors=True, infer_schema_length=200)
            for row in df.iter_rows(named=True):
                date_str = row.get("date")
                if not date_str:
                    continue
                try:
                    match_date = dt.datetime.strptime(str(date_str)[:10], "%Y-%m-%d").date()
                except ValueError:
                    continue

                h_us = row.get("h_team")
                a_us = row.get("a_team")
                fixture_id: int | None = None
                for home_full in _fpl_full_name_candidates(h_us):
                    for away_full in _fpl_full_name_candidates(a_us):
                        fixture_id = fixture_lookup.get((match_date, home_full, away_full))
                        if fixture_id is not None:
                            break
                    if fixture_id is not None:
                        break
                if fixture_id is not None:
                    counts["linked_to_fixture"] += 1
                else:
                    counts["unmatched_fixture"] += 1

                stmt = pg_insert(FactUnderstatPlayerMatch).values(
                    understat_player_id=understat_id,
                    match_date=match_date,
                    understat_match_id=row.get("id"),
                    h_team=h_us,
                    a_team=a_us,
                    fixture_id=fixture_id,
                    player_id=player_id,
                    position=row.get("position"),
                    minutes=_int_or_none(row.get("time")),
                    goals=_int_or_none(row.get("goals")),
                    shots=_int_or_none(row.get("shots")),
                    xg=_float_or_none(row.get("xG")),
                    xa=_float_or_none(row.get("xA")),
                    key_passes=_int_or_none(row.get("key_passes")),
                    npg=_int_or_none(row.get("npg")),
                    npxg=_float_or_none(row.get("npxG")),
                    xg_chain=_float_or_none(row.get("xGChain")),
                    xg_buildup=_float_or_none(row.get("xGBuildup")),
                    assists=_int_or_none(row.get("assists")),
                )
                stmt = stmt.on_conflict_do_nothing(
                    index_elements=[
                        "understat_player_id",
                        "match_date",
                        "recorded_at",
                    ]
                )
                s.execute(stmt)
                counts["rows"] += 1

    return counts


def _int_or_none(v: object) -> int | None:
    if v is None or v == "":
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _float_or_none(v: object) -> float | None:
    if v is None or v == "":
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _build_lookups() -> tuple[
    dict[tuple[dt.date, str, str], int],
    dict[str, int],
    dict[str, int],
]:
    """Returns ((date, home_full, away_full)→fixture_id, full_name→player_id, last_lower→player_id).

    Fixture lookup spans ALL ingested seasons: vaastav's per-season understat
    folder contains each player's full career, so a 2024-25 ingest needs to
    match matches from 2019-2024 too. Joined to dim_team to use the stable
    full_name (per-season team_id is irrelevant for matching).
    """
    from sqlalchemy import alias

    home = alias(DimTeam, "th")
    away = alias(DimTeam, "ta")
    with session_scope() as s:
        fixture_rows = s.execute(
            select(
                DimFixture.fixture_id,
                DimFixture.kickoff_utc,
                home.c.full_name.label("home_name"),
                away.c.full_name.label("away_name"),
            )
            .join(
                home,
                (home.c.team_id == DimFixture.home_team_id)
                & (home.c.season_id == DimFixture.season_id),
            )
            .join(
                away,
                (away.c.team_id == DimFixture.away_team_id)
                & (away.c.season_id == DimFixture.season_id),
            )
        ).all()
        player_rows = s.execute(
            select(DimPlayer.player_id, DimPlayer.first_name, DimPlayer.last_name)
        ).all()

    fixture_lookup = {
        (f.kickoff_utc.date(), f.home_name, f.away_name): f.fixture_id
        for f in fixture_rows
    }
    name_to_player_id: dict[str, int] = {}
    last_to_player_id: dict[str, int] = {}
    for p in player_rows:
        if p.first_name and p.last_name:
            name_to_player_id[f"{p.first_name} {p.last_name}"] = p.player_id
            last_to_player_id.setdefault(p.last_name.lower(), p.player_id)
    return fixture_lookup, name_to_player_id, last_to_player_id
