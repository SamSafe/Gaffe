"""Add was_home and price_tenths to fact_player_match (Phase 2.2 prep).

vaastav's per-GW CSVs include both fields. Re-ingest required after migration.

Revision ID: 0004
Revises: 0003
Create Date: 2026-05-04
"""
from __future__ import annotations

from typing import Sequence

from alembic import op

revision: str = "0004"
down_revision: str | None = "0003"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("ALTER TABLE fact_player_match ADD COLUMN was_home BOOLEAN")
    op.execute("ALTER TABLE fact_player_match ADD COLUMN price_tenths SMALLINT")


def downgrade() -> None:
    op.execute("ALTER TABLE fact_player_match DROP COLUMN price_tenths")
    op.execute("ALTER TABLE fact_player_match DROP COLUMN was_home")
