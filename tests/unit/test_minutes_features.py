"""Feature-correctness unit tests for the minutes feature builder.

Synthetic-DataFrame tests that don't require a live DB; they exercise the
windowing/labeling logic directly on a small known-good input.
"""
from __future__ import annotations

import polars as pl
import pytest

from fpl_bot.features.minutes import _label_bucket


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
