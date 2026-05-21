"""Unit tests for intra-week price-snapshot feature scaffolding."""
from __future__ import annotations

import datetime as dt

import polars as pl

from fpl_bot.features import price_snapshot as ps


def test_snapshot_features_build_player_local_deltas() -> None:
    rows = pl.DataFrame(
        [
            {
                "player_id": 2,
                "season_id": 99,
                "recorded_at": dt.datetime(2026, 8, 1, 12, tzinfo=dt.UTC),
                "price_tenths": 70,
                "selected_by_percent": 20.0,
            },
            {
                "player_id": 1,
                "season_id": 99,
                "recorded_at": dt.datetime(2026, 8, 1, 0, tzinfo=dt.UTC),
                "price_tenths": 55,
                "selected_by_percent": 10.0,
            },
            {
                "player_id": 1,
                "season_id": 99,
                "recorded_at": dt.datetime(2026, 8, 1, 12, tzinfo=dt.UTC),
                "price_tenths": 56,
                "selected_by_percent": 12.5,
            },
            {
                "player_id": 1,
                "season_id": 99,
                "recorded_at": dt.datetime(2026, 8, 2, 0, tzinfo=dt.UTC),
                "price_tenths": 56,
                "selected_by_percent": 14.0,
            },
        ]
    )

    out = ps.build_snapshot_features_from_rows(rows)
    p1 = out.filter(pl.col("player_id") == 1).sort("recorded_at")

    assert p1["selected_pct_delta_1"].to_list() == [0.0, 2.5, 1.5]
    assert p1["price_delta_since_prev_snapshot"].to_list() == [0, 1, 0]
    assert p1[ps.LABEL_COLUMN].to_list() == [1, 0, None]
    assert p1["hours_since_prev_snapshot"].to_list() == [0.0, 12.0, 12.0]


def test_snapshot_features_empty_input_returns_empty() -> None:
    rows = pl.DataFrame(
        schema={
            "player_id": pl.Int64,
            "season_id": pl.Int64,
            "recorded_at": pl.Datetime,
            "price_tenths": pl.Int64,
            "selected_by_percent": pl.Float64,
        }
    )
    out = ps.build_snapshot_features_from_rows(rows)
    assert out.is_empty()
