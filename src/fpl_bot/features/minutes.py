"""Feature builder for the minutes model (Phase 2.1).

PIT-routed: imports only `fpl_bot.db.pit`. The static-import leakage gate
enforces this; the synthetic-future-row gate validates that adding a row
with kickoff later than `as_of` does not change features for earlier rows.

V1 scope (per docs/design/phase2_1_minutes_model.md addendum): trains
without `fact_player_status` fields (only one snapshot exists). Live-only
fields can be appended at predict time without retraining.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Literal

import numpy as np
import polars as pl

from fpl_bot.db import pit

POSITION_CODES = ("GKP", "DEF", "MID", "FWD")
MinutesFeatureMode = Literal["baseline", "rotation"]
ROTATION_FEATURE_COLUMNS = ["team_core_churn_3", "team_core_churn_5"]


def _rotation_feature_frames(
    raw: pl.DataFrame,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Return PIT fixture features and latest post-fixture rates per team."""
    required = {
        "season_id",
        "fixture_id",
        "kickoff_utc",
        "player_id",
        "minutes",
        "was_home",
        "home_team_id",
        "away_team_id",
    }
    if raw.is_empty() or not required.issubset(raw.columns):
        return pl.DataFrame(), pl.DataFrame()

    resolved = raw.with_columns(
        pl.when(pl.col("was_home"))
        .then(pl.col("home_team_id"))
        .otherwise(pl.col("away_team_id"))
        .alias("team_id")
    )
    fixture_cores = (
        resolved.group_by(
            ["season_id", "team_id", "fixture_id", "kickoff_utc"]
        )
        .agg(
            pl.col("player_id")
            .filter(pl.col("minutes") >= 60)
            .alias("core_player_ids")
        )
        .sort(["season_id", "team_id", "kickoff_utc", "fixture_id"])
    )

    previous_core: dict[tuple[int, int], set[int]] = {}
    churn_history: dict[tuple[int, int], list[float]] = defaultdict(list)
    feature_rows: list[dict[str, int | float | None]] = []
    for row in fixture_cores.iter_rows(named=True):
        season_id = int(row["season_id"])
        team_id = int(row["team_id"])
        key = (season_id, team_id)
        history = churn_history[key]
        feature_rows.append(
            {
                "season_id": season_id,
                "team_id": team_id,
                "fixture_id": int(row["fixture_id"]),
                "team_core_churn_3": (
                    float(np.mean(history[-3:])) if history else None
                ),
                "team_core_churn_5": (
                    float(np.mean(history[-5:])) if history else None
                ),
            }
        )

        current_core = {int(pid) for pid in (row["core_player_ids"] or [])}
        prior_core = previous_core.get(key)
        if prior_core:
            retained = len(prior_core.intersection(current_core))
            history.append(1.0 - retained / len(prior_core))
        previous_core[key] = current_core

    latest_rows = [
        {
            "season_id": season_id,
            "team_id": team_id,
            "team_core_churn_3": float(np.mean(history[-3:])),
            "team_core_churn_5": float(np.mean(history[-5:])),
        }
        for (season_id, team_id), history in churn_history.items()
        if history
    ]
    return pl.DataFrame(feature_rows), pl.DataFrame(latest_rows)


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

    rotation_features, _ = _rotation_feature_frames(raw)
    raw = raw.with_columns(
        pl.when(pl.col("was_home"))
        .then(pl.col("home_team_id"))
        .otherwise(pl.col("away_team_id"))
        .alias("_team_id_resolved")
    )
    if not rotation_features.is_empty():
        raw = raw.join(
            rotation_features.rename({"team_id": "_team_id_resolved"}),
            on=["season_id", "_team_id_resolved", "fixture_id"],
            how="left",
        )
    else:
        raw = raw.with_columns(
            pl.lit(None, dtype=pl.Float64).alias("team_core_churn_3"),
            pl.lit(None, dtype=pl.Float64).alias("team_core_churn_5"),
        )

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
    df = df.drop(
        [
            "started_60_prev",
            "season_start_utc",
            "position_code",
            "_team_id_resolved",
        ]
    )

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


def feature_columns_for_mode(mode: MinutesFeatureMode) -> list[str]:
    if mode == "baseline":
        return list(FEATURE_COLUMNS)
    if mode == "rotation":
        return [*FEATURE_COLUMNS, *ROTATION_FEATURE_COLUMNS]
    raise ValueError(f"unknown minutes feature mode: {mode}")


def build_prediction_feature_table(
    test_season: int,
    upcoming_fixture_ids: list[int],
) -> pl.DataFrame:
    """Phase 6 v2: feature rows for UPCOMING fixtures (no played data yet).

    Per-player rolling features come from their LATEST played match in
    `test_season`. New fixture rows synthesize team/position/days_into_season
    from dim_fixture + fact_player_status.
    """
    if not upcoming_fixture_ids:
        return pl.DataFrame()

    history = build_feature_table(season_ids=[test_season])
    if history.is_empty():
        per_player_rolling = pl.DataFrame()
    else:
        per_player_rolling = (
            history.sort(["player_id", "kickoff_utc"])
            .group_by("player_id", maintain_order=True)
            .last()
        )

    fixtures_df = pit.upcoming_fixtures(upcoming_fixture_ids)
    players_df = pit.season_player_status_snapshot(test_season).rename(
        {"team_id": "current_team_id"}
    )
    season_start = pit.season_start_kickoff(test_season)

    if fixtures_df.is_empty() or players_df.is_empty():
        return pl.DataFrame()

    cross = fixtures_df.join(players_df, how="cross")
    cross = cross.filter(
        (pl.col("current_team_id") == pl.col("home_team_id"))
        | (pl.col("current_team_id") == pl.col("away_team_id"))
    )

    # Position one-hot
    for code in POSITION_CODES:
        cross = cross.with_columns(
            (pl.col("position_code") == code).cast(pl.Int8).alias(f"pos_{code}")
        )

    # Rolling features from history
    rolling_cols = [
        "min_last_1", "min_last_3", "min_last_5", "min_last_10",
        "start60_rate_3", "start60_rate_5", "start60_rate_10",
        "bucket_last_1", "days_since_last_match", "season_match_count",
    ]
    if not per_player_rolling.is_empty():
        keep = ["player_id"] + [c for c in rolling_cols if c in per_player_rolling.columns]
        cross = cross.join(per_player_rolling.select(keep), on="player_id", how="left")
    for c in rolling_cols:
        if c in cross.columns:
            cross = cross.with_columns(pl.col(c).fill_null(0))
        else:
            cross = cross.with_columns(pl.lit(0).alias(c))

    raw_history = pit.all_player_match_with_kickoff(season_ids=[test_season])
    _, latest_rotation = _rotation_feature_frames(raw_history)
    if not latest_rotation.is_empty():
        latest_rotation = latest_rotation.rename({"team_id": "current_team_id"})
        cross = cross.join(
            latest_rotation.select(
                "season_id", "current_team_id", *ROTATION_FEATURE_COLUMNS
            ),
            on=["season_id", "current_team_id"],
            how="left",
        )
    else:
        cross = cross.with_columns(
            pl.lit(None, dtype=pl.Float64).alias("team_core_churn_3"),
            pl.lit(None, dtype=pl.Float64).alias("team_core_churn_5"),
        )

    if season_start is not None:
        cross = cross.with_columns(
            (
                (pl.col("kickoff_utc").cast(pl.Datetime) - pl.lit(season_start).cast(pl.Datetime))
                .dt.total_seconds()
                / 86400.0
            ).alias("days_into_season")
        )
    else:
        cross = cross.with_columns(pl.lit(0.0).alias("days_into_season"))

    cross = cross.with_columns(
        pl.lit(None, dtype=pl.Int16).alias("minutes"),
        pl.lit(None, dtype=pl.Int8).alias("minutes_bucket"),
    )

    return cross
