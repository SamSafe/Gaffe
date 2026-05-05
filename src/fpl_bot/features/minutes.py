"""Feature builder for the minutes model (Phase 2.1).

PIT-routed: imports only `fpl_bot.db.pit`. The static-import leakage gate
enforces this; the synthetic-future-row gate validates that adding a row
with kickoff later than `as_of` does not change features for earlier rows.

V1 scope (per docs/design/phase2_1_minutes_model.md addendum): trains
without `fact_player_status` fields (only one snapshot exists). Live-only
fields can be appended at predict time without retraining.
"""
from __future__ import annotations

import polars as pl

from fpl_bot.db import pit

POSITION_CODES = ("GKP", "DEF", "MID", "FWD")


def _label_bucket(minutes: pl.Expr) -> pl.Expr:
    """Maps actual minutes to bucket: 0 → 0, 1-59 → 1, 60+ → 2. NULL propagates."""
    return (
        pl.when(minutes.is_null())
        .then(None)
        .when(minutes == 0)
        .then(0)
        .when(minutes < 60)
        .then(1)
        .otherwise(2)
        .cast(pl.Int8, strict=False)
    )


def build_feature_table(season_ids: list[int] | None = None) -> pl.DataFrame:
    """Build the per-(player, fixture) feature matrix.

    Each row's features are derived strictly from earlier-kickoff matches for
    the same player; PIT correctness is enforced by polars `shift(1).over()`
    excluding the current row before any rolling computation.

    Returns columns:
        player_id, fixture_id, season_id, gameweek, kickoff_utc, minutes
        + features (see V1 set in design doc)
        + minutes_bucket (label, 0/1/2)
    """
    raw = pit.all_player_match_with_kickoff(season_ids=season_ids)
    if raw.is_empty():
        return raw

    # Drop rows missing minutes (cancelled / unfinished fixtures)
    raw = raw.filter(pl.col("minutes").is_not_null())

    # Sort by player + time so shift/rolling operate in chronological order
    df = raw.sort(by=["player_id", "kickoff_utc"])

    # Label
    df = df.with_columns(_label_bucket(pl.col("minutes")).alias("minutes_bucket"))

    # Per-player history features. shift(1) excludes the current row.
    df = df.with_columns(
        pl.col("minutes").shift(1).over("player_id").alias("min_last_1"),
        pl.col("minutes_bucket").shift(1).over("player_id").alias("bucket_last_1"),
    )
    df = df.with_columns(
        pl.col("min_last_1")
        .rolling_mean(window_size=3, min_samples=1)
        .over("player_id")
        .alias("min_last_3"),
        pl.col("min_last_1")
        .rolling_mean(window_size=5, min_samples=1)
        .over("player_id")
        .alias("min_last_5"),
        pl.col("min_last_1")
        .rolling_mean(window_size=10, min_samples=1)
        .over("player_id")
        .alias("min_last_10"),
    )
    # Rate of 60+ starts in trailing windows
    df = df.with_columns(
        (pl.col("min_last_1") >= 60).cast(pl.Float32).alias("started_60_prev")
    )
    df = df.with_columns(
        pl.col("started_60_prev")
        .rolling_mean(window_size=3, min_samples=1)
        .over("player_id")
        .alias("start60_rate_3"),
        pl.col("started_60_prev")
        .rolling_mean(window_size=5, min_samples=1)
        .over("player_id")
        .alias("start60_rate_5"),
        pl.col("started_60_prev")
        .rolling_mean(window_size=10, min_samples=1)
        .over("player_id")
        .alias("start60_rate_10"),
    )

    # Days since last match (NULL on first match for a player)
    df = df.with_columns(
        (
            (
                pl.col("kickoff_utc").cast(pl.Datetime)
                - pl.col("kickoff_utc").shift(1).cast(pl.Datetime).over("player_id")
            ).dt.total_seconds()
            / 86400.0
        ).alias("days_since_last_match")
    )

    # Days into season (relative to the season's first kickoff)
    df = df.with_columns(
        pl.col("kickoff_utc").min().over("season_id").alias("season_start_utc")
    )
    df = df.with_columns(
        (
            (
                pl.col("kickoff_utc").cast(pl.Datetime)
                - pl.col("season_start_utc").cast(pl.Datetime)
            ).dt.total_seconds()
            / 86400.0
        ).alias("days_into_season")
    )

    # Player's prior match count this season (proxy for "established starter")
    df = df.with_columns(
        pl.int_range(0, pl.len()).over(["player_id", "season_id"]).alias(
            "season_match_count"
        )
    )

    # Position one-hot (static metadata — current snapshot from fact_player_status)
    positions = pit.all_player_positions()
    if not positions.is_empty():
        df = df.join(positions, on="player_id", how="left")
    else:
        df = df.with_columns(pl.lit(None).alias("position_code"))
    for code in POSITION_CODES:
        df = df.with_columns(
            (pl.col("position_code") == code).cast(pl.Int8).alias(f"pos_{code}")
        )

    # Drop helper cols not needed downstream
    df = df.drop(["started_60_prev", "season_start_utc", "position_code"])

    return df


FEATURE_COLUMNS: list[str] = [
    "min_last_1",
    "min_last_3",
    "min_last_5",
    "min_last_10",
    "start60_rate_3",
    "start60_rate_5",
    "start60_rate_10",
    "bucket_last_1",
    "days_since_last_match",
    "days_into_season",
    "gameweek",
    "season_match_count",
    "pos_GKP",
    "pos_DEF",
    "pos_MID",
    "pos_FWD",
]
