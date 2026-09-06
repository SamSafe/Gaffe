"""Add FPL's own expected-stats columns to fact_player_match.

The goals/assists models source every attacking rolling feature (xG, xA,
shots, key passes, xGChain, xGBuildup) from Understat, mirrored by vaastav
under `data/{season}/understat/`. **vaastav stopped publishing that folder
after 2024-25.** Seasons 25 and 26 therefore have zero Understat rows, and
`features/goals.build_feature_table` fills every one of those features with
0.0 — silently, because the join is a left join onto a table that simply has
no rows for the season.

Measured on 2026-27 GW2: 100% of rows had `xg_per_90_last_3 == 0`. The models
were running blind on every attacking feature for the whole live season.

FPL's own API carries Opta-derived expected stats and is a first-party source
we already ingest:
  - historical: vaastav `gws/gw{N}.csv` has `expected_goals`,
    `expected_assists`, `expected_goal_involvements`, `expected_goals_conceded`
    and `starts` from 2022-23 onward.
  - live: `event/{GW}/live` returns the same fields per player per gameweek.

Against Understat on the 22-24 overlap (23,536 player-matches) FPL xG has
r = 0.89 (spearman 0.90) and runs ~0.83x the Understat scale; xA is weaker
(r = 0.71). It is a slightly less predictive but *live* replacement.

`starts` is included because it is a cleaner rotation signal than
`minutes >= 60` and the minutes model can use it later.

Backfill seasons 22+ by re-ingesting vaastav after this migration.

Revision ID: 0010
Revises: 0009
Create Date: 2026-08-29
"""
from __future__ import annotations

from typing import Sequence

from alembic import op

revision: str = "0010"
down_revision: str | None = "0009"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

# Per-match expected values are small (< ~5) but we keep Understat's
# precision so the two sources stay directly comparable in diagnostics.
_XG_COLUMNS = (
    "expected_goals",
    "expected_assists",
    "expected_goal_involvements",
    "expected_goals_conceded",
)


def upgrade() -> None:
    for col in _XG_COLUMNS:
        op.execute(f"ALTER TABLE fact_player_match ADD COLUMN {col} NUMERIC(7, 5)")
    op.execute("ALTER TABLE fact_player_match ADD COLUMN starts SMALLINT")


def downgrade() -> None:
    op.execute("ALTER TABLE fact_player_match DROP COLUMN starts")
    for col in reversed(_XG_COLUMNS):
        op.execute(f"ALTER TABLE fact_player_match DROP COLUMN {col}")
