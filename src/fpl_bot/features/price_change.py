"""Feature builder for the Phase 3.5 price-change predictor.

Per-(player_id, gw_close_boundary) row predicting the discrete price delta
Δ ∈ {-2, -1, 0, +1, +2} that occurs at the boundary from gw{N} close to
gw{N+1} open. Computed PIT-correctly: features for the boundary at end of
GW N use only data from GWs ≤ N.

Source: `pit.all_player_match_with_kickoff` (now surfacing the four
Phase-3.5-backfilled columns: transfers_in, transfers_out, transfers_balance,
selected). Plus `pit.all_player_positions` for position one-hot.
"""
from __future__ import annotations

import polars as pl

from fpl_bot.db import pit

# Per-season approximation of total managers in game. Within-season fluctuation
# is ~flat post-GW3 (~10M for 24/25). v2 candidate: time-vary.
_TOTAL_MANAGERS_BY_SEASON: dict[int, int] = {
    19: 7_000_000,    # 2019-20
    20: 7_500_000,    # 2020-21
    21: 8_000_000,    # 2021-22
    22: 9_000_000,    # 2022-23
    23: 10_000_000,   # 2023-24
    24: 11_000_000,   # 2024-25
    25: 11_500_000,   # 2025-26
}
_DEFAULT_TOTAL_MANAGERS = 10_000_000

LABEL_COLUMN = "price_delta_tenths"
FEATURE_COLUMNS = [
    "net_transfer_rate",
    "net_transfer_rate_lag1",
    "transfers_in_rate",
    "transfers_out_rate",
    "selected_rate",
    "current_price_tenths",
    "price_lag1_tenths",
    "gw_in_season",
    "pos_GKP",
    "pos_DEF",
    "pos_MID",
    "pos_FWD",
]


def build_feature_table(season_ids: list[int] | None = None) -> pl.DataFrame:
    """Build per-(player_id, gw_boundary) rows.

    Boundary "N" = end of GW N (= prediction target: price change occurring
    between GW N close and GW N+1 open). Features at boundary N use data
    from GW N close (transfers_in/out/balance/selected over that GW).

    PIT correctness: columns are computed from gw{N}.csv-derived bitemporal
    rows; the rolling/shift uses `over(player_id).sort_by(gameweek)` to
    ensure no row sees future data.
    """
    pm = pit.all_player_match_with_kickoff(season_ids=season_ids)
    if pm.is_empty():
        return pm

    # One physical fixture per (player, fixture). For DGW players have 2 rows
    # in the same gameweek — for the price-change predictor we want one row
    # per (player, gameweek) using the player's GW-end snapshot. transfers
    # and selected are per-GW snapshots and identical across DGW fixtures, so
    # take any one row (use max kickoff to deterministically pick the later).
    pm = (
        pm.sort(["player_id", "gameweek", "kickoff_utc"])
        .group_by(["player_id", "season_id", "gameweek"], maintain_order=True)
        .agg(
            pl.col("transfers_in").last(),
            pl.col("transfers_out").last(),
            pl.col("transfers_balance").last(),
            pl.col("selected").last(),
            pl.col("price_tenths").last(),
            pl.col("kickoff_utc").last(),
        )
    )

    # Drop rows where transfers data is missing (older seasons / 2018-19 csv
    # didn't ingest cleanly).
    pm = pm.filter(pl.col("transfers_in").is_not_null())

    # Total-managers normalizer per season
    pm = pm.with_columns(
        pl.col("season_id")
        .map_elements(
            lambda s: _TOTAL_MANAGERS_BY_SEASON.get(s, _DEFAULT_TOTAL_MANAGERS),
            return_dtype=pl.Int64,
        )
        .alias("_total_managers")
    )

    # Sort by player + gw for shift-based features
    pm = pm.sort(["player_id", "season_id", "gameweek"])

    # Rates: rate = count / total_managers
    pm = pm.with_columns(
        (pl.col("transfers_in") / pl.col("_total_managers")).alias("transfers_in_rate"),
        (pl.col("transfers_out") / pl.col("_total_managers")).alias("transfers_out_rate"),
        (pl.col("transfers_balance") / pl.col("_total_managers")).alias(
            "net_transfer_rate"
        ),
        (pl.col("selected") / pl.col("_total_managers")).alias("selected_rate"),
        pl.col("price_tenths").alias("current_price_tenths"),
    )

    # Lag-1 features: prior gameweek's net transfer rate and price.
    # shift within-(player_id, season_id) so cross-season rows aren't confused.
    pm = pm.with_columns(
        pl.col("net_transfer_rate")
        .shift(1)
        .over(["player_id", "season_id"])
        .alias("net_transfer_rate_lag1"),
        pl.col("price_tenths")
        .shift(1)
        .over(["player_id", "season_id"])
        .alias("price_lag1_tenths"),
    )

    # Label: price_tenths change from THIS GW close to NEXT GW close.
    # Δ = next_price - current_price. Done via shift(-1).
    pm = pm.with_columns(
        pl.col("price_tenths")
        .shift(-1)
        .over(["player_id", "season_id"])
        .alias("_next_price_tenths")
    )
    pm = pm.with_columns(
        (pl.col("_next_price_tenths") - pl.col("price_tenths")).alias(LABEL_COLUMN)
    )

    # Drop rows where label is missing (last GW of season for each player)
    pm = pm.filter(pl.col(LABEL_COLUMN).is_not_null())
    # Clip ±2 (rare ±3 events in raw data are likely data hiccups)
    pm = pm.with_columns(pl.col(LABEL_COLUMN).clip(-2, 2))

    # gw_in_season feature
    pm = pm.with_columns(pl.col("gameweek").alias("gw_in_season"))

    # Position one-hot from dim_player. Inner-join drops players without a
    # position record (~1% of fact_player_match rows; these are typically
    # players whose dim_player row predates the position-code backfill).
    pos = pit.all_player_positions()
    pm = pm.join(pos, on="player_id", how="inner")
    pm = pm.with_columns(
        (pl.col("position_code") == "GKP").cast(pl.Int8).alias("pos_GKP"),
        (pl.col("position_code") == "DEF").cast(pl.Int8).alias("pos_DEF"),
        (pl.col("position_code") == "MID").cast(pl.Int8).alias("pos_MID"),
        (pl.col("position_code") == "FWD").cast(pl.Int8).alias("pos_FWD"),
    )

    # Final select
    keep = [
        "player_id",
        "season_id",
        "gameweek",
        LABEL_COLUMN,
        *FEATURE_COLUMNS,
    ]
    return pm.select(keep)
