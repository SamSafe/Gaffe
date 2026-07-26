"""Player-prop (anytime-goalscorer) odds → implied goal rate.

Phase 8 ceiling-raiser. The bot already inverts the MATCH market (1X2 +
O/U 2.5) into a team goal rate via Dixon-Coles (`derive/dixon_coles.py`).
This module does the player-level analogue: invert a bookmaker's
anytime-goalscorer probability into a per-90 goal rate that can replace or
blend with the model's `lambda_goals_per_90`.

**Why this is a ceiling-raiser.** The bot derives per-player goal rates from
xG-shares + team λ — a model approximation. The anytime-scorer market prices
the same quantity directly and *also* bakes in expected minutes, rotation
risk, role and form. It's the bot's "edge = the odds" thesis extended from
the team to the player level, aimed squarely at the one persistent weakness
(elite-FWD / top-decile under-prediction).

**The minutes double-counting pitfall** (the reason this module exists as
pure, tested logic). The market anytime prob `P` is UNCONDITIONAL — it
already integrates over the chance the player is benched/subbed. The BPS
simulator, however, samples minutes per iteration and scales the per-90
rate by the sampled minutes. So feeding the market-implied *fixture* rate
straight in would apply minutes twice. We divide the implied fixture goals
by the player's expected minutes fraction to recover a per-90 rate the
simulator can re-scale without double-counting.

This module is pure (no I/O); the ingest + live wiring land in 26/27 when
the per-event odds endpoint can be exercised against real fixtures. See
docs/design/phase8_player_prop_odds.md.
"""
from __future__ import annotations

import datetime as dt
import math
import statistics
from dataclasses import dataclass
from typing import Any

import polars as pl

# Anytime-scorer "yes" prices carry a bookmaker margin. Soccer goalscorer
# two-way markets (yes/no) typically run ~6-8% overround on the pair; with
# only the "yes" side we remove a flat half-margin as a first approximation.
DEFAULT_ANYTIME_MARGIN = 0.07
# Clamp the implied per-90 rate so a thin/short-minutes line can't explode it.
MAX_IMPLIED_RATE = 3.0
# Floor on the expected-minutes divisor, for the same reason.
MIN_MINUTES_FRACTION = 0.25


def devig_anytime_prob(yes_decimal_odds: float, *, margin: float = DEFAULT_ANYTIME_MARGIN) -> float:
    """Raw anytime-scorer 'yes' decimal odds → de-vigged P(scores >= 1).

    With only the 'yes' price available we approximate the fair probability by
    shaving half the typical two-way margin off the implied prob.
    """
    if yes_decimal_odds <= 1.0:
        return 0.0
    p_raw = 1.0 / yes_decimal_odds
    p_fair = p_raw * (1.0 - margin / 2.0)
    return min(max(p_fair, 0.0), 0.99)


def anytime_prob_to_fixture_goals(p_anytime: float) -> float:
    """P(scores >= 1 in the fixture) -> expected goals in the fixture (Poisson).

    Under goals ~ Poisson(mu): P(>=1) = 1 - exp(-mu)  =>  mu = -ln(1 - P).
    `mu` is UNCONDITIONAL (already integrates expected minutes).
    """
    p = min(max(p_anytime, 0.0), 0.99)
    return -math.log(1.0 - p)


def market_implied_per90_rate(
    p_anytime: float,
    expected_minutes_fraction: float,
    *,
    min_minutes_fraction: float = MIN_MINUTES_FRACTION,
    max_rate: float = MAX_IMPLIED_RATE,
) -> float:
    """Anytime-scorer prob -> per-90 goal rate for the simulator.

    The simulator re-scales the per-90 rate by per-iteration sampled minutes,
    so we divide the unconditional implied fixture goals by the player's
    expected minutes fraction (E[minutes]/90) to avoid double-counting
    minutes. `min_minutes_fraction` floors the divisor so a low-minutes
    player's rate isn't blown up to absurdity.
    """
    mu_fixture = anytime_prob_to_fixture_goals(p_anytime)
    denom = max(expected_minutes_fraction, min_minutes_fraction)
    return min(mu_fixture / denom, max_rate)


def blend_goal_rate(model_rate: float, market_rate: float, *, market_weight: float) -> float:
    """Convex blend of the model and market per-90 rates.

    `market_weight` in [0, 1]; 1.0 = trust the market fully, 0.0 = model only.
    A blend (rather than full replacement) hedges noisy / US-book-only props
    and players the market lists thinly. The weight is a tunable to be set
    once live anytime-scorer data exists to validate against (26/27).
    """
    w = min(max(market_weight, 0.0), 1.0)
    return (1.0 - w) * model_rate + w * market_rate


# ─── Consensus across bookmakers ───────────────────────────────────────────


@dataclass(frozen=True)
class AnytimeConsensus:
    """De-vigged P(scores) for one (fixture, player), pooled across books."""

    p_anytime: float
    n_books: int
    latest_quote_time: dt.datetime


def consensus_anytime_probs(
    rows: list[Any],
    *,
    as_of: dt.datetime | None = None,
    margin: float = DEFAULT_ANYTIME_MARGIN,
) -> dict[tuple[int, int], AnytimeConsensus]:
    """`fact_player_odds` rows → {(fixture_id, player_id): consensus}.

    Mirrors `dixon_coles._consensus_market_probs`: take each bookmaker's
    LATEST eligible quote, de-vig within that book, and only then pool across
    books. `as_of` enforces point-in-time correctness — a quote published after
    the deadline must never inform that deadline's decision.

    Pooling uses the median: player-prop coverage on the free tier is a handful
    of US books, where one stale or mispriced line moves a mean materially more
    than it moves a median.
    """
    if as_of is not None and as_of.utcoffset() is None:
        raise ValueError("as_of must be timezone-aware")

    latest: dict[tuple[int, int, str], tuple[dt.datetime, float]] = {}
    for row in rows:
        try:
            odds = float(row.decimal_odds)
        except (TypeError, ValueError):
            continue
        if odds <= 1.0 or not math.isfinite(odds):
            continue
        quote_time = row.quote_time
        if quote_time.utcoffset() is None:
            quote_time = quote_time.replace(tzinfo=dt.UTC)
        if as_of is not None and quote_time > as_of:
            continue
        key = (int(row.fixture_id), int(row.player_id), str(row.bookmaker))
        prev = latest.get(key)
        if prev is None or quote_time > prev[0]:
            latest[key] = (quote_time, odds)

    pooled: dict[tuple[int, int], list[tuple[dt.datetime, float]]] = {}
    for (fixture_id, player_id, _bk), (quote_time, odds) in latest.items():
        pooled.setdefault((fixture_id, player_id), []).append(
            (quote_time, devig_anytime_prob(odds, margin=margin))
        )

    out: dict[tuple[int, int], AnytimeConsensus] = {}
    for key, entries in pooled.items():
        probs = [p for _t, p in entries]
        out[key] = AnytimeConsensus(
            p_anytime=statistics.median(probs),
            n_books=len(probs),
            latest_quote_time=max(t for t, _p in entries),
        )
    return out


# ─── Applying the market rate to model predictions ─────────────────────────

# Expected minutes from the 4-bucket minutes model, matching the linear
# midpoints used by eval/xpts_eval (30 * p_short + 75 * p_full).
_E_MINUTES = pl.col("p_minutes_short") * 30.0 + pl.col("p_minutes_full") * 75.0


def attach_market_goal_rates(
    player_predictions: pl.DataFrame,
    consensus: dict[tuple[int, int], AnytimeConsensus],
    *,
    market_weight: float,
    min_books: int = 1,
) -> pl.DataFrame:
    """Add market-implied goal rates to a predictions frame and blend them in.

    Requires `player_id`, `fixture_id`, `lambda_goals_per_90`, and the minutes
    bucket probabilities. Adds:
      - `lambda_goals_per_90_model` — the untouched model rate (kept so the
        shadow comparison can be logged even at market_weight 0)
      - `market_goal_rate_per_90`, `market_p_anytime`, `market_n_books`
      - `lambda_goals_per_90` — blended in place

    With `market_weight = 0.0` the blend is a no-op by construction, so the
    market columns can be logged and validated for weeks before any
    recommendation depends on them. That ordering is deliberate: this feature
    has no backtest (no free historical player props), so the only honest
    validation is live accuracy measured while it is inert.

    Rows without a prop keep the model rate untouched — absence of a prop is
    not evidence of a low goal rate, it usually just means the books listed a
    shorter market.
    """
    if player_predictions.is_empty():
        return player_predictions

    df = player_predictions.with_columns(
        pl.col("lambda_goals_per_90").alias("lambda_goals_per_90_model")
    )
    if not consensus:
        return df.with_columns(
            pl.lit(None, dtype=pl.Float64).alias("market_p_anytime"),
            pl.lit(None, dtype=pl.Float64).alias("market_goal_rate_per_90"),
            pl.lit(0, dtype=pl.Int32).alias("market_n_books"),
        )

    props = pl.DataFrame(
        [
            {
                "fixture_id": fixture_id,
                "player_id": player_id,
                "market_p_anytime": c.p_anytime,
                "market_n_books": c.n_books,
            }
            for (fixture_id, player_id), c in consensus.items()
            if c.n_books >= min_books
        ],
        schema={
            "fixture_id": pl.Int64,
            "player_id": pl.Int64,
            "market_p_anytime": pl.Float64,
            "market_n_books": pl.Int32,
        },
    )
    df = df.with_columns(
        pl.col("fixture_id").cast(pl.Int64), pl.col("player_id").cast(pl.Int64)
    ).join(props, on=["fixture_id", "player_id"], how="left")

    # Vectorized equivalent of `market_implied_per90_rate`; the two are pinned
    # to agree by test_market_rate_expression_matches_scalar_function.
    p_clamped = pl.col("market_p_anytime").clip(0.0, 0.99)
    mu_fixture = -((1.0 - p_clamped).log())
    denom = (_E_MINUTES / 90.0).clip(lower_bound=MIN_MINUTES_FRACTION)
    df = df.with_columns(
        pl.when(pl.col("market_p_anytime").is_not_null())
        .then((mu_fixture / denom).clip(upper_bound=MAX_IMPLIED_RATE))
        .otherwise(None)
        .alias("market_goal_rate_per_90")
    )
    w = min(max(market_weight, 0.0), 1.0)
    df = df.with_columns(
        pl.when(pl.col("market_goal_rate_per_90").is_not_null())
        .then(
            (1.0 - w) * pl.col("lambda_goals_per_90_model")
            + w * pl.col("market_goal_rate_per_90")
        )
        .otherwise(pl.col("lambda_goals_per_90_model"))
        .alias("lambda_goals_per_90"),
        pl.col("market_n_books").fill_null(0),
    )
    return df
