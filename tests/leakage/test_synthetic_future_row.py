"""Synthetic future-row leakage test (§3.4 / §7.6).

Inserts a sentinel `fact_player_status` row with recorded_at far in the future,
then asserts the PIT API at a present-time `as_of` cutoff never returns it.
A positive-control test confirms the sentinel IS visible when as_of is *past*
the future timestamp — proves the test machinery isn't trivially broken.

Marked @pytest.mark.integration — skipped if PostgreSQL is unreachable.
"""
from __future__ import annotations

import datetime as dt

import pytest
from sqlalchemy import delete, text

from fpl_bot.db import pit
from fpl_bot.db.models import FactPlayerStatus
from fpl_bot.db.session import engine, session_scope

# Real player code (Salah). Using a known existing player avoids spurious
# "no row for player_id" passes that aren't actually testing leakage.
SENTINEL_PLAYER_ID = 118748
SENTINEL_NEWS = "__LEAKAGE_TEST_SENTINEL__"
FUTURE_TS = dt.datetime(3000, 1, 1, tzinfo=dt.UTC)
FAR_FUTURE_AS_OF = dt.datetime(3001, 1, 1, tzinfo=dt.UTC)


def _db_available() -> bool:
    try:
        with engine.connect() as c:
            c.execute(text("SELECT 1"))
        return True
    except Exception:
        return False


@pytest.fixture
def sentinel_row():
    """Insert a future-timestamped sentinel; clean up before AND after.

    Cleanup-before guards against leftover sentinels from a crashed prior run.
    Cleanup targets `news = SENTINEL_NEWS` so it can never touch real data.
    """
    if not _db_available():
        pytest.skip("PostgreSQL not available; integration test skipped")

    def _purge() -> None:
        with session_scope() as s:
            s.execute(
                delete(FactPlayerStatus).where(FactPlayerStatus.news == SENTINEL_NEWS)
            )

    _purge()
    with session_scope() as s:
        s.add(
            FactPlayerStatus(
                player_id=SENTINEL_PLAYER_ID,
                recorded_at=FUTURE_TS,
                event_time=FUTURE_TS,
                season_id=99,
                position_code="MID",
                team_id=99,
                price_tenths=9999,
                status_code="x",
                news=SENTINEL_NEWS,
            )
        )
    try:
        yield
    finally:
        _purge()


@pytest.mark.integration
@pytest.mark.leakage
def test_pit_filters_future_row_at_present_as_of(sentinel_row: None) -> None:
    """Core invariant: PIT must never return a row with recorded_at > as_of."""
    now = dt.datetime.now(dt.UTC)
    result = pit.player_status_as_of(SENTINEL_PLAYER_ID, now)
    assert result is not None, (
        "Salah should have at least one historical row from prior FPL/vaastav ingest"
    )
    assert result["recorded_at"] <= now, (
        f"PIT returned a row with recorded_at={result['recorded_at']} > as_of={now} — leakage"
    )
    assert result["news"] != SENTINEL_NEWS, (
        "Future-dated sentinel leaked through PIT at a present-time as_of"
    )


@pytest.mark.integration
@pytest.mark.leakage
def test_pit_returns_sentinel_when_as_of_is_in_far_future(sentinel_row: None) -> None:
    """Positive control: as_of past the sentinel timestamp must surface it.

    Confirms the test setup actually inserted the row and the PIT query is
    capable of returning it when the cutoff permits — protects against a
    silent pass where the sentinel was never written or the query is broken.
    """
    result = pit.player_status_as_of(SENTINEL_PLAYER_ID, FAR_FUTURE_AS_OF)
    assert result is not None
    assert result["news"] == SENTINEL_NEWS, (
        f"Expected sentinel news at as_of={FAR_FUTURE_AS_OF}; got {result['news']!r}"
    )
    assert result["recorded_at"] == FUTURE_TS, (
        f"Expected sentinel recorded_at={FUTURE_TS}; got {result['recorded_at']}"
    )
