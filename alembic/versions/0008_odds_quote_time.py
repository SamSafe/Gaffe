"""Preserve distinct bookmaker quote snapshots.

`fact_odds.event_time` historically stored fixture commencement and was also
part of the primary key. Repeated live pulls therefore updated the same row.
Add `quote_time` and key snapshots by the bookmaker/market update timestamp.

Existing rows are historical closing prices or a single live snapshot, so the
only honest backfill is their existing event time.

Revision ID: 0008
Revises: 0007
Create Date: 2026-06-23
"""
from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0008"
down_revision: str | None = "0007"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("ALTER TABLE fact_odds ADD COLUMN quote_time TIMESTAMPTZ")
    op.execute("UPDATE fact_odds SET quote_time = event_time")
    # Older Odds API rows encoded each selection's own signed handicap, which
    # split the two outcomes across `ah_-x` and `ah_+x`. Canonicalize to the
    # home-team line so a complete two-way market can be selected atomically.
    op.execute(
        """
        UPDATE fact_odds
        SET market = 'ah_' || ((-substring(market FROM 4)::numeric)::text)
        WHERE market LIKE 'ah_%' AND selection = 'away'
        """
    )
    op.execute("ALTER TABLE fact_odds ALTER COLUMN quote_time SET NOT NULL")
    op.execute("ALTER TABLE fact_odds DROP CONSTRAINT fact_odds_pkey")
    op.execute(
        "ALTER TABLE fact_odds ADD PRIMARY KEY "
        "(fixture_id, bookmaker, market, selection, quote_time)"
    )
    op.execute(
        "CREATE INDEX ix_fact_odds_fixture_quote_time "
        "ON fact_odds (fixture_id, quote_time DESC)"
    )


def downgrade() -> None:
    # Multiple quote snapshots collapse under the old event-time key. Keep the
    # latest quote deterministically before restoring the legacy constraint.
    op.execute(
        """
        DELETE FROM fact_odds older
        USING fact_odds newer
        WHERE older.fixture_id = newer.fixture_id
          AND older.bookmaker = newer.bookmaker
          AND older.market = newer.market
          AND older.selection = newer.selection
          AND older.event_time = newer.event_time
          AND older.quote_time < newer.quote_time
        """
    )
    op.execute("DROP INDEX ix_fact_odds_fixture_quote_time")
    op.execute("ALTER TABLE fact_odds DROP CONSTRAINT fact_odds_pkey")
    op.execute(
        "ALTER TABLE fact_odds ADD PRIMARY KEY "
        "(fixture_id, bookmaker, market, selection, event_time)"
    )
    op.execute("ALTER TABLE fact_odds DROP COLUMN quote_time")
