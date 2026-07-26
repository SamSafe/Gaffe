"""Player-prop (anytime-goalscorer) odds table — Phase 8.

Kept separate from `fact_odds`: the natural key needs a player, and the source
is a different endpoint (The Odds API per-event, mostly US books) with its own
coverage and liquidity profile, which we want to be able to audit and weight
independently of the match market.

`quote_time` is part of the key, mirroring `fact_odds` after 0008, so repeated
pre-deadline pulls accumulate as distinct snapshots instead of overwriting —
without that, point-in-time reconstruction is impossible.

Revision ID: 0009
Revises: 0008
Create Date: 2026-07-26
"""
from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0009"
down_revision: str | None = "0008"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE fact_player_odds (
            fixture_id INTEGER NOT NULL,
            player_id INTEGER NOT NULL,
            bookmaker TEXT NOT NULL,
            market TEXT NOT NULL,
            quote_time TIMESTAMPTZ NOT NULL,
            event_time TIMESTAMPTZ NOT NULL,
            decimal_odds NUMERIC(8, 4) NOT NULL,
            source_player_name TEXT NOT NULL,
            recorded_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            PRIMARY KEY (fixture_id, player_id, bookmaker, market, quote_time)
        )
        """
    )
    op.execute(
        "CREATE INDEX ix_fact_player_odds_fixture_quote "
        "ON fact_player_odds (fixture_id, quote_time DESC)"
    )
    op.execute("CREATE INDEX ix_fact_player_odds_player ON fact_player_odds (player_id)")


def downgrade() -> None:
    op.execute("DROP TABLE fact_player_odds")
