"""Add defensive_contribution + components to fact_player_match (Phase 7).

FPL 2025/26 added the Defensive Contribution Points rule:
  - DEF: +2 pts if defensive_contribution >= 10 in a match
  - MID/FWD: +2 pts if defensive_contribution >= 12 in a match

Source: vaastav 2025-26/gws/gw{N}.csv. Only available for 25/26 (vaastav
introduced these columns in this season). Earlier seasons' rows will have
NULL values.

`defensive_contribution` is FPL's per-position threshold metric (NOT a
simple sum of components — for DEFs it's tackles + CBI; for MID/FWD it's
tackles + CBI + recoveries).

Re-ingest 25/26 vaastav after this migration.

Revision ID: 0007
Revises: 0006
Create Date: 2026-05-18
"""
from __future__ import annotations

from typing import Sequence

from alembic import op

revision: str = "0007"
down_revision: str | None = "0006"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute(
        "ALTER TABLE fact_player_match ADD COLUMN defensive_contribution SMALLINT"
    )
    op.execute("ALTER TABLE fact_player_match ADD COLUMN tackles SMALLINT")
    op.execute("ALTER TABLE fact_player_match ADD COLUMN recoveries SMALLINT")
    op.execute(
        "ALTER TABLE fact_player_match "
        "ADD COLUMN clearances_blocks_interceptions SMALLINT"
    )


def downgrade() -> None:
    op.execute(
        "ALTER TABLE fact_player_match DROP COLUMN clearances_blocks_interceptions"
    )
    op.execute("ALTER TABLE fact_player_match DROP COLUMN recoveries")
    op.execute("ALTER TABLE fact_player_match DROP COLUMN tackles")
    op.execute("ALTER TABLE fact_player_match DROP COLUMN defensive_contribution")
