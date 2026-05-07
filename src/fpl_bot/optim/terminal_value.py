"""Terminal value V_T(s_H) for the rolling-horizon MILP (Phase 3 §6).

V_T = α · sum_p x_{p,H} · xPts_next_5(p) + β · sum_p sell_price_p · x_{p,H}

  - First term: expected post-horizon points (5 GWs after the horizon end).
    For each player in the squad at horizon-tip, sum the predicted E[xPts]
    over the next 5 fixtures we have predictions for; fall back to a
    position-mean × number-of-fixtures estimate when unavailable.
  - Second term: squad market value at horizon-tip — captures optionality
    of holding rising-price assets. v1 uses sell_price (which equals
    buy_price under static-price assumption).
"""
from __future__ import annotations

import polars as pl

# Defaults — tunable on fold 24/25 per Phase 3 §6
ALPHA_DEFAULT = 0.8
BETA_DEFAULT = 0.05  # per-£1m of squad value; multiplied through tenths-of-millions

# Position-mean fallback per-GW xPts (used when post-horizon predictions are missing)
POSITION_MEAN_XPTS = {"GKP": 2.0, "DEF": 2.5, "MID": 2.5, "FWD": 3.0}


def compute_post_horizon_xpts(
    *,
    candidates: list[int],
    horizon_end_gw: int,
    full_predictions: pl.DataFrame,  # cols: player_id, gameweek, e_xpts
    positions: dict[int, str],
    n_lookahead_gws: int = 5,
) -> dict[int, float]:
    """For each candidate, sum predicted E[xPts] over GWs (horizon_end_gw+1 ..
    horizon_end_gw+n_lookahead_gws). Missing values fall back to position-mean."""
    target_gws = list(range(horizon_end_gw + 1, horizon_end_gw + 1 + n_lookahead_gws))
    if full_predictions.is_empty():
        return {
            pid: POSITION_MEAN_XPTS.get(positions.get(pid, "MID"), 2.5)
            * n_lookahead_gws
            for pid in candidates
        }

    sub = full_predictions.filter(pl.col("gameweek").is_in(target_gws)).filter(
        pl.col("player_id").is_in(candidates)
    )
    summed = (
        sub.group_by("player_id")
        .agg(pl.col("e_xpts").sum().alias("future_xpts"))
        .to_pandas()
        .set_index("player_id")["future_xpts"]
        .to_dict()
    )

    out: dict[int, float] = {}
    for pid in candidates:
        if pid in summed:
            out[pid] = float(summed[pid])
        else:
            # Fallback: position-mean × number of GWs (no fixtures-counted yet
            # in v1; assume one fixture per GW)
            pos = positions.get(pid, "MID")
            out[pid] = POSITION_MEAN_XPTS.get(pos, 2.5) * n_lookahead_gws
    return out


def terminal_value_coefficients(
    *,
    candidates: list[int],
    horizon_end_gw: int,
    full_predictions: pl.DataFrame,
    positions: dict[int, str],
    sell_prices_tenths: dict[int, int],
    alpha: float = ALPHA_DEFAULT,
    beta: float = BETA_DEFAULT,
) -> dict[int, float]:
    """Returns per-player terminal-value coefficient. Multiplied by x_{p,H}
    in the MILP objective to give V_T(s_H)."""
    future_xpts = compute_post_horizon_xpts(
        candidates=candidates,
        horizon_end_gw=horizon_end_gw,
        full_predictions=full_predictions,
        positions=positions,
    )
    out: dict[int, float] = {}
    for pid in candidates:
        post = future_xpts.get(pid, 0.0)
        sell_tenths = sell_prices_tenths.get(pid, 0)
        # beta is per-£1m; sell_tenths is in tenths-of-millions ⇒ /10
        out[pid] = alpha * post + beta * (sell_tenths / 10.0)
    return out
