"""Leakage gate for the minutes feature builder.

Beyond the static-import check (already in test_static_imports.py), this
file adds a dynamic check: rows from later kickoffs must not influence
features for earlier kickoffs. Done by computing features for the full
corpus, then re-computing on a strict subset truncated to an earlier
cutoff, and asserting the overlapping rows match exactly.
"""
from __future__ import annotations

import pytest
from sqlalchemy import text

from fpl_bot.db.session import engine
from fpl_bot.features.minutes import (
    FEATURE_COLUMNS,
    ROTATION_FEATURE_COLUMNS,
    build_feature_table,
)


def _db_available() -> bool:
    try:
        with engine.connect() as c:
            c.execute(text("SELECT 1"))
        return True
    except Exception:
        return False


@pytest.mark.integration
@pytest.mark.leakage
def test_features_are_pit_stable_across_corpus_truncation() -> None:
    """Features for season 22 rows must be identical whether we build over
    [19,20,21,22] or [19,20,21,22,23,24]. Adding later seasons must not change
    earlier-season feature values."""
    if not _db_available():
        pytest.skip("PostgreSQL not available; integration test skipped")

    full = build_feature_table(season_ids=[19, 20, 21, 22, 23, 24])
    truncated = build_feature_table(season_ids=[19, 20, 21, 22])

    if full.is_empty() or truncated.is_empty():
        pytest.skip("No corpus data; ingest before running")

    # Compare feature values for season 22 between the two builds
    import polars as pl

    full_22 = full.filter(pl.col("season_id") == 22).sort(["player_id", "fixture_id"])
    trunc_22 = truncated.filter(pl.col("season_id") == 22).sort(
        ["player_id", "fixture_id"]
    )

    assert len(full_22) == len(trunc_22), (
        f"Row count mismatch for season 22: full={len(full_22)} vs truncated={len(trunc_22)}"
    )

    for col in [*FEATURE_COLUMNS, *ROTATION_FEATURE_COLUMNS, "minutes_bucket"]:
        a = full_22[col].to_list()
        b = trunc_22[col].to_list()
        assert a == b, (
            f"Feature `{col}` differs between full-corpus and truncated build "
            f"for season 22 — adding future seasons leaked into earlier features"
        )
