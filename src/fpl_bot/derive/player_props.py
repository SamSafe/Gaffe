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

import math

# Anytime-scorer "yes" prices carry a bookmaker margin. Soccer goalscorer
# two-way markets (yes/no) typically run ~6-8% overround on the pair; with
# only the "yes" side we remove a flat half-margin as a first approximation.
DEFAULT_ANYTIME_MARGIN = 0.07
# Clamp the implied per-90 rate so a thin/short-minutes line can't explode it.
MAX_IMPLIED_RATE = 3.0


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
    min_minutes_fraction: float = 0.25,
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
