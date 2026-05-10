"""Add transfers_in/out/balance and selected to fact_player_match (Phase 3.5).

Source: vaastav gw{N}.csv per-GW snapshot. Powers the Phase 3.5 price-change
predictor. Re-ingest required after migration.

Revision ID: 0005
Revises: 0004
Create Date: 2026-05-09
"""
from __future__ import annotations

from typing import Sequence

from alembic import op

revision: str = "0005"
down_revision: str | None = "0004"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("ALTER TABLE fact_player_match ADD COLUMN transfers_in INTEGER")
    op.execute("ALTER TABLE fact_player_match ADD COLUMN transfers_out INTEGER")
    op.execute("ALTER TABLE fact_player_match ADD COLUMN transfers_balance INTEGER")
    op.execute("ALTER TABLE fact_player_match ADD COLUMN selected INTEGER")


def downgrade() -> None:
    op.execute("ALTER TABLE fact_player_match DROP COLUMN selected")
    op.execute("ALTER TABLE fact_player_match DROP COLUMN transfers_balance")
    op.execute("ALTER TABLE fact_player_match DROP COLUMN transfers_out")
    op.execute("ALTER TABLE fact_player_match DROP COLUMN transfers_in")
