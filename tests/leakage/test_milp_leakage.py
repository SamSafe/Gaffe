"""Leakage test for the MILP backtest harness (Phase 3).

Asserts the price-walk in `_resolve_price_at_gw` only walks BACKWARD —
forward walking would leak future prices into earlier-GW solves.
"""
from __future__ import annotations

from fpl_bot.eval.milp_backtest import _resolve_price_at_gw


def test_price_walk_only_backward() -> None:
    """If we have prices at GW 3 and GW 7 but ask for GW 5, the resolver must
    return the GW 3 (prior) price, NOT the GW 7 (future) price."""
    prices = {
        (100, 3): 80,  # prior price
        (100, 7): 95,  # future price
    }
    resolved = _resolve_price_at_gw(player_id=100, gameweek=5, prices_by_gw=prices)
    assert resolved == 80, "must use prior GW price, not future"


def test_price_walk_direct_hit() -> None:
    """If price exists for the exact GW, return it directly."""
    prices = {
        (100, 5): 85,
        (100, 4): 80,
        (100, 6): 90,
    }
    assert _resolve_price_at_gw(100, 5, prices) == 85


def test_price_walk_falls_back_to_default_when_no_prior() -> None:
    """No prior GW price → returns league-default fallback (NOT a future price)."""
    prices = {
        (100, 10): 80,  # only future price available
    }
    resolved = _resolve_price_at_gw(player_id=100, gameweek=5, prices_by_gw=prices)
    assert resolved == 50, "no prior data → default fallback, not future leak"


def test_price_walk_player_isolation() -> None:
    """Resolution scoped to the requested player_id only."""
    prices = {
        (100, 3): 80,
        (200, 5): 95,  # different player; must not contaminate
    }
    resolved = _resolve_price_at_gw(player_id=100, gameweek=5, prices_by_gw=prices)
    assert resolved == 80
    resolved2 = _resolve_price_at_gw(player_id=999, gameweek=5, prices_by_gw=prices)
    assert resolved2 == 50  # default fallback for unknown player


def test_price_walk_finds_nearest_prior() -> None:
    prices = {
        (100, 1): 75,
        (100, 4): 82,
        (100, 9): 90,
    }
    # Asking for GW 7: must return GW 4's 82, not GW 9's 90
    assert _resolve_price_at_gw(100, 7, prices) == 82
