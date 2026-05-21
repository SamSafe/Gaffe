"""Phase 7 2.2 — live price-change predictor wire-in.

Loads the Phase 3.5 multinomial price-change model and predicts per-player
horizon-end price for a live recommend. Result is fed into the MILP's
sell_prices and terminal-value chain so the bot rewards holding rising
players (and penalizes falling ones) when picking transfers.

Backtest path is unaffected: there we already know realized prices via
`_resolve_price_at_gw`. Predictions only fill the future-price gap that
exists only for the upcoming GW(s).
"""
from __future__ import annotations

from functools import lru_cache

import polars as pl

from fpl_bot.config import settings
from fpl_bot.features.price_change import (
    FEATURE_COLUMNS,
    build_feature_table,
)
from fpl_bot.models.price_change import (
    LABEL_COLUMN,
    TrainedPriceChangeModel,
    train_price_change_model,
)


@lru_cache(maxsize=4)
def _trained_model_for(train_seasons_key: tuple[int, ...]) -> TrainedPriceChangeModel:
    """Train (and cache) the model on the given train seasons."""
    if not settings.enable_shelved_price_predictor:
        raise RuntimeError(
            "Phase 3.5 price-change predictor is disabled by default because "
            "its walk-forward gates failed; validate v2 snapshot features "
            "before enabling FPL_BOT_ENABLE_SHELVED_PRICE_PREDICTOR."
        )
    train_df = build_feature_table(list(train_seasons_key))
    train_df = train_df.drop_nulls(LABEL_COLUMN)
    return train_price_change_model(train_df, FEATURE_COLUMNS)


def predict_one_step_price_deltas(
    *,
    season_id: int,
    gameweek: int,
    candidates: list[int],
    train_seasons: list[int],
) -> dict[int, float]:
    """Predict E[Δ price] in tenths for each candidate between GW close
    and GW+1 open. Uses the latest available row per candidate (GW < target).

    For LIVE use at upcoming GW N: the latest row available is GW N-1, which
    captures full-GW transfer activity ending at the GW N kickoff. That is
    exactly the feature snapshot the model was trained against.
    """
    model = _trained_model_for(tuple(sorted(train_seasons)))
    # Build features for test_season and pick the latest row per player
    feats = build_feature_table([season_id])
    if feats.is_empty():
        return dict.fromkeys(candidates, 0.0)
    feats = feats.filter(pl.col("player_id").is_in(candidates))
    # Latest row strictly before target GW (lookback safety)
    feats = feats.filter(pl.col("gameweek") < gameweek)
    if feats.is_empty():
        return dict.fromkeys(candidates, 0.0)
    feats = (
        feats.sort(["player_id", "gameweek"])
        .group_by("player_id", maintain_order=True)
        .agg([pl.col(c).last() for c in FEATURE_COLUMNS])
    )
    feats_X = feats.select(["player_id", *FEATURE_COLUMNS])
    deltas = model.predict_expected_delta(feats_X.select(FEATURE_COLUMNS))
    out: dict[int, float] = {
        int(pid): float(d)
        for pid, d in zip(feats_X["player_id"].to_list(), deltas, strict=True)
    }
    for pid in candidates:
        out.setdefault(pid, 0.0)
    return out


def projected_horizon_sell_price(
    current_price_tenths: int,
    expected_delta_per_gw: float,
    horizon_weeks: int,
    *,
    cap_per_gw_tenths: float = 0.6,
) -> int:
    """Compound a one-step E[Δ] estimate across `horizon_weeks` to give a
    horizon-end sell price.

    Cap per-GW absolute delta to ±0.6 tenths because the multinomial model
    can over-extrapolate on noisy weeks; FPL's actual per-GW move is bounded
    to ±2 tenths but most moves are 0/±1 — capping at 0.6 limits the model's
    influence per step.
    """
    capped = max(-cap_per_gw_tenths, min(cap_per_gw_tenths, expected_delta_per_gw))
    projected = current_price_tenths + capped * horizon_weeks
    return max(1, round(projected))
