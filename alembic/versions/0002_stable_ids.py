"""Stable global IDs for dim_player and dim_fixture (round-4 design fix).

FPL per-season element/fixture `id` is not stable across seasons; the FPL
`code` field is. This migration rekeys dim_player and dim_fixture on `code`
and adds xref tables for ingest-time translation.

Smoke-test fact rows are truncated; they will be re-ingested with stable IDs.

Revision ID: 0002
Revises: 0001
Create Date: 2026-05-02
"""
from __future__ import annotations

from typing import Sequence

from alembic import op

revision: str = "0002"
down_revision: str | None = "0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # Truncate fact rows that reference per-season player/fixture ids.
    # No real data lost — only smoke-test rows from the initial FPL ingest.
    op.execute("TRUNCATE fact_player_status")
    op.execute("TRUNCATE fact_player_match")
    op.execute("TRUNCATE fact_match_result")
    op.execute("TRUNCATE fact_player_match_event")
    op.execute("TRUNCATE fact_eo_snapshot")

    # Rekey dim_player and dim_fixture on stable FPL `code`.
    op.execute("DROP TABLE dim_player")
    op.execute("DROP TABLE dim_fixture")

    op.execute(
        """
        CREATE TABLE dim_player (
            player_id     INT PRIMARY KEY,    -- FPL `code` (stable across seasons)
            web_name      TEXT NOT NULL,
            first_name    TEXT,
            last_name     TEXT,
            understat_id  INT,
            fbref_id      TEXT
        )
        """
    )
    op.execute(
        """
        CREATE TABLE dim_fixture (
            fixture_id    INT PRIMARY KEY,    -- FPL fixture `code` (stable across seasons)
            season_id     SMALLINT NOT NULL,
            gameweek      SMALLINT NOT NULL,
            kickoff_utc   TIMESTAMPTZ NOT NULL,
            home_team_id  SMALLINT NOT NULL,
            away_team_id  SMALLINT NOT NULL,
            finished      BOOLEAN NOT NULL DEFAULT FALSE
        )
        """
    )

    # Xref tables: per-season FPL id ↔ stable code
    op.execute(
        """
        CREATE TABLE dim_player_season_xref (
            season_id       SMALLINT NOT NULL,
            fpl_element_id  INT NOT NULL,
            player_id       INT NOT NULL,
            PRIMARY KEY (season_id, fpl_element_id)
        )
        """
    )
    op.execute("CREATE INDEX ix_player_xref_player ON dim_player_season_xref (player_id)")

    op.execute(
        """
        CREATE TABLE dim_fixture_season_xref (
            season_id       SMALLINT NOT NULL,
            fpl_fixture_id  INT NOT NULL,
            fixture_id      INT NOT NULL,
            PRIMARY KEY (season_id, fpl_fixture_id)
        )
        """
    )
    op.execute("CREATE INDEX ix_fixture_xref_fixture ON dim_fixture_season_xref (fixture_id)")


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS dim_fixture_season_xref")
    op.execute("DROP TABLE IF EXISTS dim_player_season_xref")
    op.execute("DROP TABLE IF EXISTS dim_fixture")
    op.execute("DROP TABLE IF EXISTS dim_player")
    op.execute(
        """
        CREATE TABLE dim_player (
            player_id     INT PRIMARY KEY,
            web_name      TEXT NOT NULL,
            first_name    TEXT,
            last_name     TEXT,
            understat_id  INT,
            fbref_id      TEXT
        )
        """
    )
    op.execute(
        """
        CREATE TABLE dim_fixture (
            fixture_id    INT PRIMARY KEY,
            season_id     SMALLINT NOT NULL,
            gameweek      SMALLINT NOT NULL,
            kickoff_utc   TIMESTAMPTZ NOT NULL,
            home_team_id  SMALLINT NOT NULL,
            away_team_id  SMALLINT NOT NULL,
            finished      BOOLEAN NOT NULL DEFAULT FALSE
        )
        """
    )
