"""Unit tests for the Phase 8 anytime-goalscorer → goal-rate logic.

These pin the inversion math and, critically, the minutes-double-counting
guard — the subtle correctness property of wiring market player props into a
simulator that re-samples minutes.
"""
from __future__ import annotations

import math

import pytest

from fpl_bot.derive.player_props import (
    anytime_prob_to_fixture_goals,
    blend_goal_rate,
    devig_anytime_prob,
    market_implied_per90_rate,
)


def test_devig_shaves_margin_off_implied_prob():
    # 2.0 decimal => 0.5 raw; half of a 7% margin removed => 0.5*(1-0.035).
    p = devig_anytime_prob(2.0, margin=0.07)
    assert p == pytest.approx(0.5 * (1 - 0.035))


def test_devig_degenerate_odds():
    assert devig_anytime_prob(1.0) == 0.0
    assert devig_anytime_prob(0.5) == 0.0


def test_anytime_prob_to_goals_inverts_poisson():
    # mu = -ln(1-P); round-trip P = 1 - exp(-mu).
    for p in (0.1, 0.35, 0.6):
        mu = anytime_prob_to_fixture_goals(p)
        assert (1.0 - math.exp(-mu)) == pytest.approx(p)


def test_anytime_prob_monotonic_and_clamped():
    assert anytime_prob_to_fixture_goals(0.0) == 0.0
    assert anytime_prob_to_fixture_goals(0.2) < anytime_prob_to_fixture_goals(0.5)
    # P clamped at 0.99 so mu stays finite.
    assert math.isfinite(anytime_prob_to_fixture_goals(1.0))


def test_per90_rate_undoes_minutes_so_sim_rescaling_recovers_fixture_goals():
    # The whole point: implied per-90 * expected-minutes-fraction == the
    # unconditional fixture goals, so the simulator's minutes scaling doesn't
    # double-count.
    p, mins = 0.45, 0.8
    rate = market_implied_per90_rate(p, mins)
    assert rate * mins == pytest.approx(anytime_prob_to_fixture_goals(p))


def test_per90_rate_floors_low_minutes_divisor():
    # A 10%-minutes player shouldn't get an exploded rate; divisor floored.
    rate = market_implied_per90_rate(0.3, 0.10, min_minutes_fraction=0.25)
    mu = anytime_prob_to_fixture_goals(0.3)
    assert rate == pytest.approx(mu / 0.25)


def test_per90_rate_clamped_to_max():
    rate = market_implied_per90_rate(0.99, 0.25, max_rate=3.0)
    assert rate == 3.0


def test_blend_endpoints_and_midpoint():
    assert blend_goal_rate(0.4, 1.0, market_weight=0.0) == pytest.approx(0.4)
    assert blend_goal_rate(0.4, 1.0, market_weight=1.0) == pytest.approx(1.0)
    assert blend_goal_rate(0.4, 1.0, market_weight=0.5) == pytest.approx(0.7)


def test_blend_weight_clamped():
    assert blend_goal_rate(0.4, 1.0, market_weight=5.0) == pytest.approx(1.0)
    assert blend_goal_rate(0.4, 1.0, market_weight=-1.0) == pytest.approx(0.4)
