"""Unit tests for the clean-sheet feature builder."""
from __future__ import annotations

import polars as pl
import pytest
from sqlalchemy import text

from fpl_bot.db.session import engine
from fpl_bot.features.clean_sheet import (
    FEATURE_COLUMNS,
    LABEL_COLUMN,
    build_feature_table,
    feature_names_with_team_id,
    make_team_id_encoder,
)


def _db_available() -> bool:
    try:
        with engine.connect() as c:
            c.execute(text("SELECT 1"))
        return True
    except Exception:
        return False


def test_team_id_encoder_known_team_one_hot() -> None:
    encoder = make_team_id_encoder({1, 2, 3})
    df = pl.DataFrame({"team_id": [1, 2, 3, 1]})
    out = encoder.transform(df)
    assert out["team_id_1"].to_list() == [1, 0, 0, 1]
    assert out["team_id_2"].to_list() == [0, 1, 0, 0]
    assert out["team_id_3"].to_list() == [0, 0, 1, 0]


def test_team_id_encoder_unseen_team_all_zero() -> None:
    """Promoted teams not seen in train get all-zero one-hot — the model
    falls back on rolling/market features. This is the key correctness
    invariant for fold-internal encoding."""
    encoder = make_team_id_encoder({1, 2})
    df = pl.DataFrame({"team_id": [1, 99]})  # 99 was never seen in train
    out = encoder.transform(df)
    assert out["team_id_1"].to_list() == [1, 0]
    assert out["team_id_2"].to_list() == [0, 0]
    assert "team_id_99" not in out.columns


def test_team_id_encoder_empty_train_set() -> None:
    """Edge case: encoder fitted with no team_ids — transform must not
    add any columns and must not error."""
    encoder = make_team_id_encoder(set())
    df = pl.DataFrame({"team_id": [1, 2, 3]})
    out = encoder.transform(df)
    # No team_id_<X> cols added; original frame unchanged
    assert out.columns == ["team_id"]


def test_feature_names_with_team_id_concatenation() -> None:
    encoder = make_team_id_encoder({5, 7})
    names = feature_names_with_team_id(encoder)
    # Static features come first, team_id one-hots last
    assert names[: len(FEATURE_COLUMNS)] == FEATURE_COLUMNS
    assert names[-2:] == ["team_id_5", "team_id_7"]


@pytest.mark.integration
def test_build_feature_table_team_fixture_row_count() -> None:
    """Each fixture must produce exactly 2 team-fixture rows (one per team)."""
    if not _db_available():
        pytest.skip("PostgreSQL not available; integration test skipped")

    df = build_feature_table(season_ids=[24])
    if df.is_empty():
        pytest.skip("No corpus data; ingest before running")

    fixture_team_pairs = df.select(["fixture_id", "team_id"]).unique()
    assert len(fixture_team_pairs) == len(df), "duplicate (team_id, fixture_id) rows"

    # 2 rows per fixture
    rows_per_fixture = df.group_by("fixture_id").len().select("len")
    assert rows_per_fixture.n_unique() == 1, (
        f"every fixture should have exactly 2 team-rows; got distinct counts: "
        f"{rows_per_fixture.unique().to_list()}"
    )
    assert rows_per_fixture[0, 0] == 2


@pytest.mark.integration
def test_clean_sheet_label_consistency() -> None:
    """clean_sheet (boolean) must equal goals_conceded == 0 within every team-fixture."""
    if not _db_available():
        pytest.skip("PostgreSQL not available; integration test skipped")

    df = build_feature_table(season_ids=[24])
    if df.is_empty():
        pytest.skip("No corpus data; ingest before running")

    derived = (df["goals_conceded"] == 0).cast(pl.Int8).to_list()
    actual = df[LABEL_COLUMN].to_list()
    assert derived == actual, "clean_sheet flag must equal (goals_conceded == 0)"


@pytest.mark.integration
def test_market_cs_prob_in_unit_interval() -> None:
    """market_cs_prob must be a probability in [0, 1] for all linked rows."""
    if not _db_available():
        pytest.skip("PostgreSQL not available; integration test skipped")

    df = build_feature_table(season_ids=[24])
    if df.is_empty():
        pytest.skip("No corpus data; ingest before running")

    market_probs = df["market_cs_prob"].drop_nulls().to_list()
    assert all(0.0 <= p <= 1.0 for p in market_probs)
