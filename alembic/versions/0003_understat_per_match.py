"""Replace fact_understat_shot with fact_understat_player_match (round-5 design fix).

Understat's robots.txt disallows all paths; we cannot scrape per-shot data
directly. Vaastav's repo redistributes per-match Understat aggregates
(goals, shots, xG, xA, key_passes, xG_chain, xG_buildup) but NOT shot-level
coordinates / situation. The schema is reshaped accordingly.

Revision ID: 0003
Revises: 0002
Create Date: 2026-05-04
"""
from __future__ import annotations

from typing import Sequence

from alembic import op

revision: str = "0003"
down_revision: str | None = "0002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("DROP TABLE IF EXISTS fact_understat_shot")
    op.execute(
        """
        CREATE TABLE fact_understat_player_match (
            understat_player_id  INT NOT NULL,
            match_date           DATE NOT NULL,
            recorded_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
            understat_match_id   INT,
            h_team               TEXT,
            a_team               TEXT,
            fixture_id           INT,                  -- nullable; resolved at ingest
            player_id            INT,                  -- nullable; resolved at ingest (stable code)
            position             TEXT,
            minutes              SMALLINT,
            goals                SMALLINT,
            shots                SMALLINT,
            xg                   NUMERIC(7,5),
            xa                   NUMERIC(7,5),
            key_passes           SMALLINT,
            npg                  SMALLINT,
            npxg                 NUMERIC(7,5),
            xg_chain             NUMERIC(7,5),
            xg_buildup           NUMERIC(7,5),
            assists              SMALLINT,
            PRIMARY KEY (understat_player_id, match_date, recorded_at)
        )
        """
    )
    op.execute(
        "CREATE INDEX ix_understat_pm_pit "
        "ON fact_understat_player_match (player_id, match_date, recorded_at DESC)"
    )
    op.execute(
        "CREATE INDEX ix_understat_pm_fixture "
        "ON fact_understat_player_match (fixture_id, recorded_at DESC) "
        "WHERE fixture_id IS NOT NULL"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_understat_pm_fixture")
    op.execute("DROP INDEX IF EXISTS ix_understat_pm_pit")
    op.execute("DROP TABLE IF EXISTS fact_understat_player_match")
    op.execute(
        """
        CREATE TABLE fact_understat_shot (
            shot_id          TEXT PRIMARY KEY,
            fixture_id       INT,
            player_id        INT,
            minute           SMALLINT,
            xg               NUMERIC(5,4),
            xa_for_assister  NUMERIC(5,4),
            result           TEXT,
            situation        TEXT,
            shot_type        TEXT,
            body_part        TEXT,
            is_set_piece     BOOLEAN,
            recorded_at      TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """
    )
