"""Record the pre-deadline transfer counters on each player status snapshot.

Before a deadline, bootstrap-static's `transfers_in_event` / `transfers_out_event`
count transfers INTO the coming gameweek (checked on the GW7 deadline: 771k six
hours out for De Cuyper, 921k final in element-summary; both reset after the
deadline). With ownership, they let the minutes model see this week's crowd
reaction to team news, matching the historical per-gameweek fields in
fact_player_match. `selected` is the owner count (percent x total managers).

Revision ID: 0011
Revises: 0010
Create Date: 2026-10-19
"""
from __future__ import annotations

from typing import Sequence

from alembic import op

revision: str = "0011"
down_revision: str | None = "0010"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_COLUMNS = ("transfers_in_event", "transfers_out_event", "selected")


def upgrade() -> None:
    for col in _COLUMNS:
        op.execute(f"ALTER TABLE fact_player_status ADD COLUMN {col} INTEGER")


def downgrade() -> None:
    for col in reversed(_COLUMNS):
        op.execute(f"ALTER TABLE fact_player_status DROP COLUMN {col}")
