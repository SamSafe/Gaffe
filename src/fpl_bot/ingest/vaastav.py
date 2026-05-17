"""Vaastav historical FPL backfill — https://github.com/vaastav/Fantasy-Premier-League.

License: MIT public repo. Compliance: §2.5 — git clone only, no scraping.

Two-layer per §2.3:
  fetch_raw_vaastav() → git clone/pull to data/raw/vaastav/Fantasy-Premier-League/
  parse_raw_vaastav_season(season_folder) → ingest one season's teams, players
    (+ xref), fixtures (+ xref), and GW-by-GW player-match data via xref translation
"""
from __future__ import annotations

import datetime as dt
import subprocess
from pathlib import Path

import polars as pl
from sqlalchemy.dialects.postgresql import insert as pg_insert

from fpl_bot.config import settings
from fpl_bot.db.models import (
    DimFixture,
    DimFixtureSeasonXref,
    DimPlayer,
    DimPlayerSeasonXref,
    DimTeam,
    FactPlayerMatch,
)
from fpl_bot.db.session import session_scope
from fpl_bot.ingest.audit import audit_fetch

VAASTAV_REPO_URL = "https://github.com/vaastav/Fantasy-Premier-League.git"

# Map vaastav folder name → season_id (2-digit year of season start).
SEASON_ID_FROM_FOLDER: dict[str, int] = {
    f"{2000 + y}-{(y + 1) % 100:02d}": y for y in range(16, 30)
}


def _vaastav_dir() -> Path:
    return settings.raw_dir / "vaastav" / "Fantasy-Premier-League"


def fetch_raw_vaastav() -> Path:
    """Clone or fast-forward the vaastav repo. Idempotent. Returns the worktree path."""
    repo_dir = _vaastav_dir()
    repo_dir.parent.mkdir(parents=True, exist_ok=True)

    with audit_fetch(
        source="vaastav", url=VAASTAV_REPO_URL, user_agent=settings.user_agent
    ) as audit:
        if (repo_dir / ".git").exists():
            cmd = ["git", "-C", str(repo_dir), "pull", "--ff-only"]
        else:
            cmd = ["git", "clone", "--depth", "1", VAASTAV_REPO_URL, str(repo_dir)]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        audit.response_code = 200 if result.returncode == 0 else 500
        audit.raw_path = str(repo_dir)
        audit.byte_size = (
            sum(f.stat().st_size for f in repo_dir.rglob("*") if f.is_file())
            if repo_dir.exists()
            else 0
        )
        if result.returncode != 0:
            raise RuntimeError(f"git failed: {result.stderr.strip()}")
    return repo_dir


def parse_raw_vaastav_season(season_folder: str) -> dict[str, int]:
    """Ingest one season's data from the cloned repo."""
    if season_folder not in SEASON_ID_FROM_FOLDER:
        raise ValueError(f"Unknown season folder: {season_folder!r}")
    season_id = SEASON_ID_FROM_FOLDER[season_folder]
    season_path = _vaastav_dir() / "data" / season_folder
    if not season_path.exists():
        raise FileNotFoundError(f"Season path missing: {season_path}")

    counts: dict[str, int] = {"season_id": season_id}
    counts["dim_team"] = _ingest_teams(season_path, season_id)
    element_to_code = _ingest_players(season_path, season_id)
    counts["dim_player"] = len(element_to_code)
    counts["dim_player_season_xref"] = len(element_to_code)
    fixture_to_code = _ingest_fixtures(season_path, season_id)
    counts["dim_fixture"] = len(fixture_to_code)
    counts["dim_fixture_season_xref"] = len(fixture_to_code)
    counts["fact_player_match"] = _ingest_gw_data(
        season_path, season_id, element_to_code, fixture_to_code
    )
    return counts


def _ingest_teams(season_path: Path, season_id: int) -> int:
    csv_path = season_path / "teams.csv"
    if not csv_path.exists():
        return 0
    df = pl.read_csv(csv_path, ignore_errors=True)
    n = 0
    with session_scope() as s:
        for row in df.iter_rows(named=True):
            stmt = pg_insert(DimTeam).values(
                team_id=row["id"],
                season_id=season_id,
                short_name=row.get("short_name") or row["name"][:3].upper(),
                full_name=row["name"],
                promoted=False,
            )
            stmt = stmt.on_conflict_do_nothing(index_elements=["team_id", "season_id"])
            s.execute(stmt)
            n += 1
    return n


def _ingest_players(season_path: Path, season_id: int) -> dict[int, int]:
    """Write dim_player + xref. Returns map: per-season element_id → stable code."""
    csv_path = season_path / "players_raw.csv"
    if not csv_path.exists():
        return {}
    df = pl.read_csv(csv_path, ignore_errors=True)
    element_to_code: dict[int, int] = {}
    with session_scope() as s:
        for row in df.iter_rows(named=True):
            element_id = row["id"]
            code = row["code"]
            element_to_code[element_id] = code

            stmt = pg_insert(DimPlayer).values(
                player_id=code,
                web_name=row.get("web_name") or "",
                first_name=row.get("first_name"),
                last_name=row.get("second_name"),
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

            xref = pg_insert(DimPlayerSeasonXref).values(
                season_id=season_id,
                fpl_element_id=element_id,
                player_id=code,
            )
            xref = xref.on_conflict_do_update(
                index_elements=["season_id", "fpl_element_id"],
                set_={"player_id": xref.excluded.player_id},
            )
            s.execute(xref)
    return element_to_code


def _ingest_fixtures(season_path: Path, season_id: int) -> dict[int, int]:
    """Write dim_fixture + xref. Returns map: per-season fixture_id → stable code."""
    csv_path = season_path / "fixtures.csv"
    if not csv_path.exists():
        return {}
    df = pl.read_csv(csv_path, ignore_errors=True)
    fixture_to_code: dict[int, int] = {}
    with session_scope() as s:
        for row in df.iter_rows(named=True):
            kickoff = row.get("kickoff_time")
            if not kickoff:
                continue
            kickoff_dt = dt.datetime.fromisoformat(str(kickoff).replace("Z", "+00:00"))
            per_season_id = row["id"]
            code = row["code"]
            fixture_to_code[per_season_id] = code

            stmt = pg_insert(DimFixture).values(
                fixture_id=code,
                season_id=season_id,
                gameweek=row.get("event") or 0,
                kickoff_utc=kickoff_dt,
                home_team_id=row["team_h"],
                away_team_id=row["team_a"],
                finished=bool(row.get("finished", False)),
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

            xref = pg_insert(DimFixtureSeasonXref).values(
                season_id=season_id,
                fpl_fixture_id=per_season_id,
                fixture_id=code,
            )
            xref = xref.on_conflict_do_update(
                index_elements=["season_id", "fpl_fixture_id"],
                set_={"fixture_id": xref.excluded.fixture_id},
            )
            s.execute(xref)
    return fixture_to_code


def _ingest_gw_data(
    season_path: Path,
    season_id: int,
    element_to_code: dict[int, int],
    fixture_to_code: dict[int, int],
) -> int:
    gws_dir = season_path / "gws"
    if not gws_dir.exists():
        return 0
    # Per-GW files are named gw1.csv .. gw38.csv. merged_gw.csv duplicates them.
    gw_files = sorted(
        f for f in gws_dir.glob("gw*.csv") if "merged" not in f.name.lower()
    )
    total = 0
    for gw_file in gw_files:
        df = pl.read_csv(gw_file, ignore_errors=True, infer_schema_length=10000)
        total += _ingest_one_gw_df(df, element_to_code, fixture_to_code)
    return total


def _ingest_one_gw_df(
    df: pl.DataFrame,
    element_to_code: dict[int, int],
    fixture_to_code: dict[int, int],
) -> int:
    n = 0
    with session_scope() as s:
        for row in df.iter_rows(named=True):
            element_id = row.get("element")
            fixture_per_season_id = row.get("fixture")
            if element_id is None or fixture_per_season_id is None:
                continue
            player_id = element_to_code.get(element_id)
            fixture_id = fixture_to_code.get(fixture_per_season_id)
            if player_id is None or fixture_id is None:
                continue

            was_home = row.get("was_home")
            if was_home is not None and not isinstance(was_home, bool):
                # vaastav stores 'True'/'False' strings in some seasons
                was_home = str(was_home).strip().lower() in ("true", "1")
            stmt = pg_insert(FactPlayerMatch).values(
                player_id=player_id,
                fixture_id=fixture_id,
                minutes=row.get("minutes"),
                goals=row.get("goals_scored"),
                assists=row.get("assists"),
                clean_sheet=bool(row.get("clean_sheets") or 0),
                goals_conceded=row.get("goals_conceded"),
                saves=row.get("saves"),
                yellow_cards=row.get("yellow_cards"),
                red_cards=row.get("red_cards"),
                bonus=row.get("bonus"),
                bps=row.get("bps"),
                total_points=row.get("total_points"),
                was_home=was_home,
                price_tenths=row.get("value"),
                transfers_in=row.get("transfers_in"),
                transfers_out=row.get("transfers_out"),
                transfers_balance=row.get("transfers_balance"),
                selected=row.get("selected"),
            )
            # Idempotent within a transaction. `recorded_at` is server-side
            # `now()` which is stable across all rows in the same transaction,
            # so a (player, fixture) appearing twice in the same season's
            # merged gw{N}.csv files (e.g., postponed fixtures show up in two
            # gw files) would otherwise PK-conflict. A re-ingest in a fresh
            # transaction WILL produce a new bitemporal row (different
            # recorded_at).
            stmt = stmt.on_conflict_do_nothing(
                index_elements=["player_id", "fixture_id", "recorded_at"]
            )
            s.execute(stmt)
            n += 1
    return n
