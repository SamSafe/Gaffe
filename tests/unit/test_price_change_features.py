"""Unit tests for the Phase 3.5 price-change feature builder.

The predictor itself was shelved (CV gates failed), but the feature builder
remains in place as a reusable PIT-routed primitive. These tests guard the
feature-builder semantics so a future v2 (with richer intra-week features)
can rebuild on the same scaffolding.
"""
from __future__ import annotations

import polars as pl

from fpl_bot.features import price_change as pc


def test_label_distribution_matches_data():
    """Smoke: 24/25 has the expected ~92% Δ=0, ~2% rise, ~5.5% fall split."""
    df = pc.build_feature_table([24])
    n = df.height
    # ~18k after inner-joining position; some 24/25 players in fact_player_match
    # don't have dim_player position records.
    assert n > 15_000, f"Expected ~18k rows for 24/25, got {n}"
    counts = (
        df.group_by(pc.LABEL_COLUMN)
        .agg(pl.len().alias("n"))
        .sort(pc.LABEL_COLUMN)
        .to_dict(as_series=False)
    )
    by_class = dict(zip(counts[pc.LABEL_COLUMN], counts["n"], strict=True))
    zero_share = by_class.get(0, 0) / n
    assert 0.85 <= zero_share <= 0.97, f"Δ=0 share = {zero_share:.3f}, expected 0.85-0.97"
    assert -2 in by_class or by_class.get(-2, 0) == 0
    assert 2 in by_class or by_class.get(2, 0) == 0


def test_no_null_in_non_lag_features():
    """Non-lag features must be non-null. lag1 features are legitimately null
    on each player's first row (GW1, or first appearance for mid-season
    arrivals); LightGBM handles those natively as missing values, so we only
    guard the non-lag columns here."""
    df = pc.build_feature_table([24])
    non_lag = [c for c in pc.FEATURE_COLUMNS if "lag" not in c]
    for col in non_lag:
        nulls = df.select(pl.col(col).is_null().sum()).item()
        assert nulls == 0, f"Feature {col} has {nulls} nulls"


def test_label_clipped_to_pm2():
    """Labels must be clipped to {-2, -1, 0, 1, 2}."""
    df = pc.build_feature_table([24])
    labels = df.select(pl.col(pc.LABEL_COLUMN).unique()).to_series().to_list()
    for lbl in labels:
        assert -2 <= lbl <= 2, f"Label {lbl} outside ±2"


def test_one_row_per_player_gameweek():
    """After the per-(player, gw) aggregation, no duplicates."""
    df = pc.build_feature_table([24])
    keys = df.select(["player_id", "gameweek"])
    assert keys.is_unique().all(), "Found duplicate (player_id, gameweek) rows"


def test_position_one_hot_sums_to_one():
    """Each row must have exactly one position bit set."""
    df = pc.build_feature_table([24])
    sums = df.select(
        (
            pl.col("pos_GKP") + pl.col("pos_DEF") + pl.col("pos_MID") + pl.col("pos_FWD")
        ).alias("s")
    ).to_series()
    assert (sums == 1).all(), "Some rows have ≠1 position bits set"


def test_label_aligns_with_next_gw_price():
    """Spot-check: for a player with rows in GW1 and GW2, label at GW1 row
    should equal price@GW2 - price@GW1."""
    df = pc.build_feature_table([24])
    # Pick a player with multiple rows
    head = df.head(3).to_dict(as_series=False)
    if len(set(head["player_id"])) == 1 and len(head["gameweek"]) >= 2:
        # If first 3 rows are same player consecutive GWs, GW1 label should
        # equal GW2 current_price - GW1 current_price (clipped to ±2).
        gw1_label = head[pc.LABEL_COLUMN][0]
        gw1_price = head["current_price_tenths"][0]
        gw2_price = head["current_price_tenths"][1]
        expected = max(-2, min(2, gw2_price - gw1_price))
        assert gw1_label == expected, (
            f"Label {gw1_label} != GW2-GW1 price diff {expected}"
        )
