"""Feature-correctness unit tests for the minutes feature builder.

Synthetic-DataFrame tests that don't require a live DB; they exercise the
windowing/labeling logic directly on a small known-good input.
"""
from __future__ import annotations

import datetime as dt

import polars as pl
import pytest

from fpl_bot.features.minutes import (
    ROTATION_FEATURE_COLUMNS,
    _label_bucket,
    _rotation_feature_frames,
    feature_columns_for_mode,
)


def _rotation_rows() -> pl.DataFrame:
    cores = [
        (25, 101, [1, 2, 3]),
        (25, 102, [1, 2, 4]),
        (25, 103, [1, 4, 5]),
        (25, 104, [1, 5, 6]),
        (26, 201, [7, 8, 9]),
    ]
    rows = []
    for offset, (season_id, fixture_id, core) in enumerate(cores):
        for player_id in range(1, 10):
            rows.append(
                {
                    "season_id": season_id,
                    "fixture_id": fixture_id,
                    "kickoff_utc": dt.datetime(2025, 8, 1, tzinfo=dt.UTC)
                    + dt.timedelta(days=offset * 7),
                    "player_id": player_id,
                    "minutes": 90 if player_id in core else 0,
                    "was_home": True,
                    "home_team_id": 1,
                    "away_team_id": 2,
                }
            )
    return pl.DataFrame(rows)


def test_label_bucket_boundaries() -> None:
    """0 → 0, 1 → 1, 59 → 1, 60 → 2, 90 → 2."""
    df = pl.DataFrame({"minutes": [0, 1, 59, 60, 90]})
    df = df.with_columns(_label_bucket(pl.col("minutes")).alias("b"))
    assert df["b"].to_list() == [0, 1, 1, 2, 2]


def test_label_bucket_handles_null() -> None:
    df = pl.DataFrame({"minutes": [None, 30, None]})
    df = df.with_columns(_label_bucket(pl.col("minutes")).alias("b"))
    # NULL minutes should propagate as NULL bucket (filtered upstream)
    out = df["b"].to_list()
    assert out[0] is None
    assert out[1] == 1
    assert out[2] is None


def test_rotation_features_exclude_current_fixture_and_reset_by_season() -> None:
    fixture_features, latest = _rotation_feature_frames(_rotation_rows())
    season_25 = fixture_features.filter(pl.col("season_id") == 25).sort(
        "fixture_id"
    )

    assert season_25["team_core_churn_3"].to_list()[:2] == [None, None]
    assert season_25["team_core_churn_3"][2] == pytest.approx(1 / 3)
    assert season_25["team_core_churn_5"][3] == pytest.approx(1 / 3)
    season_26_first = fixture_features.filter(pl.col("season_id") == 26).row(
        0, named=True
    )
    assert season_26_first["team_core_churn_3"] is None

    latest_25 = latest.filter(pl.col("season_id") == 25).row(0, named=True)
    assert latest_25["team_core_churn_3"] == pytest.approx(1 / 3)
    assert latest_25["team_core_churn_5"] == pytest.approx(1 / 3)


def test_rotation_features_are_unchanged_by_future_fixture() -> None:
    raw = _rotation_rows().filter(pl.col("season_id") == 25)
    full, _ = _rotation_feature_frames(raw)
    truncated, _ = _rotation_feature_frames(
        raw.filter(pl.col("fixture_id") <= 103)
    )

    assert full.filter(pl.col("fixture_id") <= 103).to_dicts() == truncated.to_dicts()


def test_rotation_feature_mode_is_explicit() -> None:
    baseline = feature_columns_for_mode("baseline")
    rotation = feature_columns_for_mode("rotation")

    assert all(column not in baseline for column in ROTATION_FEATURE_COLUMNS)
    assert rotation == [*baseline, *ROTATION_FEATURE_COLUMNS]


def test_no_raw_db_imports_in_minutes_features() -> None:
    """The features module must not import sqlalchemy or db.models / db.session.

    This is also covered by tests/leakage/test_static_imports.py, but having a
    module-local assertion makes the contract obvious to anyone editing this file.
    """
    import ast
    import pathlib

    here = pathlib.Path(__file__).parent.parent.parent
    src = here / "src" / "fpl_bot" / "features" / "minutes.py"
    tree = ast.parse(src.read_text())
    forbidden = {"sqlalchemy", "fpl_bot.db.models", "fpl_bot.db.session", "psycopg"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert not any(
                    alias.name == f or alias.name.startswith(f + ".") for f in forbidden
                ), f"forbidden import: {alias.name}"
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            assert not any(
                mod == f or mod.startswith(f + ".") for f in forbidden
            ), f"forbidden import-from: {mod}"


@pytest.mark.parametrize(
    "minutes,expected",
    [(0, 0), (1, 1), (45, 1), (59, 1), (60, 2), (75, 2), (90, 2), (95, 2)],
)
def test_label_bucket_individual(minutes: int, expected: int) -> None:
    df = pl.DataFrame({"m": [minutes]})
    df = df.with_columns(_label_bucket(pl.col("m")).alias("b"))
    assert df["b"][0] == expected
