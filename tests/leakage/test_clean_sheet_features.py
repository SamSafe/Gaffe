"""Leakage gate for the clean-sheet feature builder.

Features for season N must be byte-identical whether built over [N-2, N] or
[N-2, N+M]. Adding later seasons must not change earlier-season feature values.
"""
from __future__ import annotations

import pytest
from sqlalchemy import text

from fpl_bot.db.session import engine
from fpl_bot.features.clean_sheet import FEATURE_COLUMNS, build_feature_table


def _db_available() -> bool:
    try:
        with engine.connect() as c:
            c.execute(text("SELECT 1"))
        return True
    except Exception:
        return False


@pytest.mark.integration
@pytest.mark.leakage
def test_clean_sheet_features_pit_stable_across_truncation() -> None:
    if not _db_available():
        pytest.skip("PostgreSQL not available; integration test skipped")

    full = build_feature_table(season_ids=[19, 20, 21, 22, 23, 24])
    truncated = build_feature_table(season_ids=[19, 20, 21, 22])

    if full.is_empty() or truncated.is_empty():
        pytest.skip("No corpus data; ingest before running")

    import polars as pl

    full_22 = full.filter(pl.col("season_id") == 22).sort(
        ["team_id", "fixture_id"]
    )
    trunc_22 = truncated.filter(pl.col("season_id") == 22).sort(
        ["team_id", "fixture_id"]
    )

    assert len(full_22) == len(trunc_22), (
        f"Row count mismatch for season 22: "
        f"full={len(full_22)} vs truncated={len(trunc_22)}"
    )

    # Check the most leak-prone columns: rolling team-level features and label
    leak_cols = [
        c for c in FEATURE_COLUMNS
        if c.startswith(("team_cs_rate_", "team_goals_conceded_", "team_goals_for_"))
    ]
    leak_cols += ["clean_sheet", "goals_conceded"]

    for col in leak_cols:
        a = full_22[col].to_list()
        b = trunc_22[col].to_list()
        assert a == b, (
            f"Feature `{col}` differs between full-corpus and truncated build "
            f"for season 22 — adding future seasons leaked into earlier features"
        )
