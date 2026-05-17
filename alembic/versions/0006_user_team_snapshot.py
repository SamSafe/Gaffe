"""Add fact_user_team_snapshot for Phase 6 live state.

One row per (season_id, gameweek, team_id, player_id, recorded_at) per pull.
Stores the user's FPL team state for the live recommend pipeline.

Revision ID: 0006
Revises: 0005
Create Date: 2026-05-16
"""
from __future__ import annotations

from typing import Sequence

from alembic import op

revision: str = "0006"
down_revision: str | None = "0005"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE fact_user_team_snapshot (
            season_id SMALLINT NOT NULL,
            gameweek SMALLINT NOT NULL,
            team_id INTEGER NOT NULL,
            player_id INTEGER NOT NULL,
            recorded_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            purchase_price_tenths SMALLINT NOT NULL,
            selling_price_tenths SMALLINT NOT NULL,
            multiplier SMALLINT NOT NULL,
            is_captain BOOLEAN NOT NULL DEFAULT FALSE,
            is_vice BOOLEAN NOT NULL DEFAULT FALSE,
            position SMALLINT NOT NULL,
            bank_tenths INTEGER NOT NULL,
            free_transfers SMALLINT NOT NULL,
            chips_used_json TEXT NOT NULL DEFAULT '[]',
            PRIMARY KEY (season_id, gameweek, team_id, player_id, recorded_at)
        );
        """
    )


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS fact_user_team_snapshot")
