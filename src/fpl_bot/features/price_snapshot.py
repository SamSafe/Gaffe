"""Intra-week price-snapshot features for a future price-change v2.

The Phase 3.5 GW-end price predictor failed because its transfer signal
arrived too late. This module uses repeated FPL bootstrap-static snapshots
already stored in `fact_player_status` to build earlier, intra-week signals
without scraping or paid APIs.

It is scaffolding only: the live recommender keeps the failed Phase 3.5 model
disabled by default until these features pass walk-forward gates.
"""
from __future__ import annotations

import polars as pl

from fpl_bot.db import pit

FEATURE_COLUMNS = [
    "price_tenths",
    "selected_by_percent",
    "selected_pct_delta_1",
    "price_delta_since_prev_snapshot",
    "hours_since_prev_snapshot",
]
LABEL_COLUMN = "price_delta_next_snapshot"


def build_snapshot_features_from_rows(rows: pl.DataFrame) -> pl.DataFrame:
    """Build PIT-ordered features from raw status-snapshot rows.

    Required columns: player_id, season_id, recorded_at, price_tenths,
    selected_by_percent. The label is the next observed snapshot's price
    delta, so rows with no next snapshot naturally have NULL labels.
    """
    if rows.is_empty():
        return rows
    df = rows.sort(["player_id", "recorded_at"])
    df = df.with_columns(
        pl.col("selected_by_percent").cast(pl.Float64).fill_null(0.0),
        pl.col("price_tenths").cast(pl.Int16),
        pl.col("recorded_at").cast(pl.Datetime),
    )
    df = df.with_columns(
        pl.col("selected_by_percent")
        .shift(1)
        .over("player_id")
        .alias("_selected_prev"),
        pl.col("price_tenths").shift(1).over("player_id").alias("_price_prev"),
        pl.col("price_tenths").shift(-1).over("player_id").alias("_price_next"),
        pl.col("recorded_at").shift(1).over("player_id").alias("_recorded_prev"),
    )
    df = df.with_columns(
        (pl.col("selected_by_percent") - pl.col("_selected_prev"))
        .fill_null(0.0)
        .alias("selected_pct_delta_1"),
        (pl.col("price_tenths") - pl.col("_price_prev"))
        .fill_null(0)
        .alias("price_delta_since_prev_snapshot"),
        (pl.col("_price_next") - pl.col("price_tenths")).alias(LABEL_COLUMN),
        (
            (
                pl.col("recorded_at").cast(pl.Datetime)
                - pl.col("_recorded_prev").cast(pl.Datetime)
            ).dt.total_seconds()
            / 3600.0
        )
        .fill_null(0.0)
        .alias("hours_since_prev_snapshot"),
    )
    return df.drop(
        ["_selected_prev", "_price_prev", "_price_next", "_recorded_prev"]
    )


def build_feature_table(season_id: int) -> pl.DataFrame:
    """Build snapshot features from all stored FPL status snapshots."""
    return build_snapshot_features_from_rows(
        pit.all_player_status_snapshots(season_id)
    )
