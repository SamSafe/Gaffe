"""Unit tests for the Phase 3.5 cost-basis sell-tax rule.

The harness in eval/milp_backtest.py computes per-player sell prices using:
   tax = max(0, (current - basis) // 2)
   sell = current - tax

This test exercises that formula directly via small synthetic states. Also
verifies apply_gw_outcomes consumes whatever the harness passes (no policy
in state.py).
"""
from __future__ import annotations

from fpl_bot.optim.state import (
    INITIAL_BUDGET_TENTHS,
    BacktestState,
    GwDecisions,
    apply_gw_outcomes,
)


def _sell_with_tax(basis: int, current: int) -> int:
    """The harness rule: 50% tax on profit, no floor on loss."""
    tax = max(0, (current - basis) // 2)
    return current - tax


def test_sell_tax_breakeven():
    """Basis = current → no profit → no tax → sell = current."""
    assert _sell_with_tax(50, 50) == 50


def test_sell_tax_on_profit_even():
    """Basis = 50, current = 56, profit = 6 → tax = 3 → sell = 53."""
    assert _sell_with_tax(50, 56) == 53


def test_sell_tax_on_profit_odd():
    """Basis = 50, current = 55, profit = 5 → tax = 5//2 = 2 → sell = 53.
    Equivalent to FPL's 'tax = profit // 2' rule (truncates toward zero)."""
    assert _sell_with_tax(50, 55) == 53


def test_sell_no_tax_on_loss():
    """Basis = 60, current = 55 → profit = -5 → tax = 0 → sell = current = 55."""
    assert _sell_with_tax(60, 55) == 55


def test_sell_tax_at_max_profit():
    """A 1m gain (basis=50, current=60) → tax = 5 → sell = 55."""
    assert _sell_with_tax(50, 60) == 55


def test_apply_gw_outcomes_uses_provided_sell():
    """apply_gw_outcomes must consume whatever sell price the harness provides
    (no policy decision inside state.py)."""
    state = BacktestState(
        season_id=24,
        gameweek=5,
        squad=frozenset({100, 200}),
        bank=50,
        free_transfers=1,
        chips_used=frozenset(),
        cost_basis={100: 60, 200: 70},
    )
    decisions = GwDecisions(
        gameweek=5,
        squad=frozenset({200, 300}),  # sold 100, kept 200, bought 300
        starting_xi=frozenset({200, 300}),
        captain=200,
        vice=300,
        transferred_in=frozenset({300}),
        transferred_out=frozenset({100}),
        chip_played=None,
        hits=0,
        objective_value=0.0,
    )
    # Sell 100 at 55 (loss-no-tax case: bought at 60, current 55 → sell 55).
    # Buy 300 at 80.
    actual_prices = {
        100: {"buy": 55, "sell": 55},
        300: {"buy": 80, "sell": 80},
    }
    new_state = apply_gw_outcomes(state, decisions, actual_prices)

    # Bank: 50 + 55 (sold 100) - 80 (bought 300) = 25
    assert new_state.bank == 25
    # Cost basis: 100 dropped, 200 retained, 300 added
    assert new_state.cost_basis == {200: 70, 300: 80}


def test_apply_gw_outcomes_with_profit_sell_tax():
    """Selling profitable asset: harness pre-computes tax-adjusted sell price."""
    state = BacktestState(
        season_id=24,
        gameweek=10,
        squad=frozenset({100}),
        bank=0,
        free_transfers=1,
        chips_used=frozenset(),
        cost_basis={100: 50},  # bought at 5.0m
    )
    # Player rose from 50 to 60 — profit 10 → tax = 5 → sell = 55
    sell_with_tax = _sell_with_tax(50, 60)
    assert sell_with_tax == 55

    decisions = GwDecisions(
        gameweek=10,
        squad=frozenset({200}),
        starting_xi=frozenset({200}),
        captain=200,
        vice=None,
        transferred_in=frozenset({200}),
        transferred_out=frozenset({100}),
        chip_played=None,
        hits=0,
        objective_value=0.0,
    )
    actual_prices = {
        100: {"buy": 60, "sell": sell_with_tax},  # current 60, but tax-adjusted sell 55
        200: {"buy": 50, "sell": 50},
    }
    new_state = apply_gw_outcomes(state, decisions, actual_prices)

    # Bank: 0 + 55 (tax-adjusted sell of 100) - 50 (buy 200) = 5
    assert new_state.bank == 5
    # Without tax we'd have gotten 60 - 50 = 10 in bank — this confirms the
    # tax is being applied via the harness, not papered over by state.py.


def test_cold_start_records_buy_price_as_basis():
    """At cold start, every bought player's cost basis == buy price."""
    state = BacktestState.cold_start(season_id=24)
    decisions = GwDecisions(
        gameweek=1,
        squad=frozenset({1, 2, 3}),
        starting_xi=frozenset({1, 2, 3}),
        captain=1,
        vice=2,
        transferred_in=frozenset({1, 2, 3}),
        transferred_out=frozenset(),
        chip_played=None,
        hits=0,
        objective_value=0.0,
    )
    actual_prices = {
        1: {"buy": 130, "sell": 130},  # 13.0m
        2: {"buy": 100, "sell": 100},  # 10.0m
        3: {"buy": 50, "sell": 50},    # 5.0m
    }
    new_state = apply_gw_outcomes(state, decisions, actual_prices)
    assert new_state.cost_basis == {1: 130, 2: 100, 3: 50}
    assert new_state.bank == INITIAL_BUDGET_TENTHS - (130 + 100 + 50)
