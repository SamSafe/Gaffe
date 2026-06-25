"""Shared pytest configuration.

Tests marked ``@pytest.mark.integration`` require a live PostgreSQL database.
When none is reachable — e.g. in CI, which runs ``pytest tests/`` without a DB
service — those tests are *skipped* rather than left to error on a refused
connection. Locally (with the populated dev Postgres up) they run normally.

This centralizes the ad-hoc ``_db_available()`` guards that some integration
modules already carry, so simply marking a module/test ``integration`` is now
enough to get the correct skip-when-no-DB behavior.
"""
from __future__ import annotations

import functools


@functools.lru_cache(maxsize=1)
def _db_available() -> bool:
    try:
        from sqlalchemy import text

        from fpl_bot.db.session import engine

        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception:
        return False


def pytest_collection_modifyitems(config, items):
    """Skip integration tests when no PostgreSQL is reachable."""
    if _db_available():
        return
    import pytest

    skip_db = pytest.mark.skip(reason="PostgreSQL not available; integration test skipped")
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip_db)
