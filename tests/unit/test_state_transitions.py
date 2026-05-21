"""Unit tests for BacktestState + apply_gw_outcomes (Phase 3)."""
from __future__ import annotations

from fpl_bot.optim.state import (
    INITIAL_BUDGET_TENTHS,
    BacktestState,
    GwDecisions,
    apply_gw_outcomes,
)


def test_cold_start_state() -> None:
    s = BacktestState.cold_start(season_id=24)
    assert s.gameweek == 1
    assert s.bank == INITIAL_BUDGET_TENTHS
    assert s.free_transfers == 1
    assert len(s.squad) == 0
    assert len(s.chips_used) == 0


def test_chips_available_first_half_unused() -> None:
    s = BacktestState.cold_start(season_id=24)
    avail_gw1 = s.chips_available_for_gw(1)
    # 25/26: first-half slots are WC1/FH1/BB1/TC1; second-half slots not eligible.
    assert "WC1" in avail_gw1
    assert "FH1" in avail_gw1
    assert "BB1" in avail_gw1
    assert "TC1" in avail_gw1
    assert "WC2" not in avail_gw1
    assert "FH2" not in avail_gw1
    assert "BB2" not in avail_gw1
    assert "TC2" not in avail_gw1


def test_chips_available_second_half_unused() -> None:
    s = BacktestState.cold_start(season_id=24)
    avail_gw20 = s.chips_available_for_gw(20)
    assert "WC2" in avail_gw20
    assert "FH2" in avail_gw20
    assert "BB2" in avail_gw20
    assert "TC2" in avail_gw20
    assert "WC1" not in avail_gw20
    assert "FH1" not in avail_gw20
    assert "BB1" not in avail_gw20
    assert "TC1" not in avail_gw20


def test_chips_used_excluded() -> None:
    s = BacktestState(
        season_id=24,
        gameweek=10,
        chips_used=frozenset({"WC1", "TC1"}),
    )
    avail = s.chips_available_for_gw(10)
    assert "WC1" not in avail
    assert "TC1" not in avail
    assert "FH1" in avail  # still available
    assert "BB1" in avail


def test_apply_gw_outcomes_no_transfers() -> None:
    """No transfers in GW: bank unchanged; FT increments by 1 capped at 5."""
    s = BacktestState(
        season_id=24,
        gameweek=5,
        squad=frozenset({100, 200, 300}),
        bank=50,
        free_transfers=2,
    )
    decisions = GwDecisions(
        gameweek=5,
        squad=frozenset({100, 200, 300}),
        starting_xi=frozenset({100, 200}),
        captain=100,
        vice=200,
        transferred_in=frozenset(),
        transferred_out=frozenset(),
        chip_played=None,
        hits=0,
        objective_value=0.0,
    )
    new_state = apply_gw_outcomes(s, decisions, actual_prices={})
    assert new_state.bank == 50
    assert new_state.free_transfers == 3  # 2 + 1
    assert new_state.gameweek == 6


def test_apply_gw_outcomes_one_transfer() -> None:
    """Sub one player out and one in. Bank reflects the cost diff."""
    s = BacktestState(
        season_id=24,
        gameweek=5,
        squad=frozenset({100, 200}),
        bank=10,
        free_transfers=1,
        cost_basis={100: 80, 200: 70},
    )
    decisions = GwDecisions(
        gameweek=5,
        squad=frozenset({200, 300}),
        starting_xi=frozenset({200, 300}),
        captain=200,
        vice=300,
        transferred_in=frozenset({300}),
        transferred_out=frozenset({100}),
        chip_played=None,
        hits=0,
        objective_value=0.0,
    )
    actual_prices = {300: {"buy": 90, "sell": 90}}
    new_state = apply_gw_outcomes(s, decisions, actual_prices)
    # Sold 100 @ 80, bought 300 @ 90 → bank = 10 + 80 - 90 = 0
    assert new_state.bank == 0
    # FT consumed 1 of 1 → ft_next = min(5, 1 + 1 - 1) = 1
    assert new_state.free_transfers == 1
    assert 300 in new_state.cost_basis
    assert 100 not in new_state.cost_basis


def test_apply_gw_outcomes_chip_waives_ft_consumption() -> None:
    """WC chip means transfers don't consume FT count for next-GW carry-over."""
    s = BacktestState(
        season_id=24,
        gameweek=5,
        squad=frozenset({100, 200}),
        bank=10,
        free_transfers=1,
        cost_basis={100: 80, 200: 70},
    )
    decisions = GwDecisions(
        gameweek=5,
        squad=frozenset({200, 300}),
        starting_xi=frozenset({200, 300}),
        captain=200,
        vice=300,
        transferred_in=frozenset({300}),
        transferred_out=frozenset({100}),
        chip_played="WC1",
        hits=0,
        objective_value=0.0,
    )
    new_state = apply_gw_outcomes(s, decisions, actual_prices={300: {"buy": 90, "sell": 90}})
    # WC waives FT consumption: ft_next = min(5, 1 + 1 - 0) = 2
    assert new_state.free_transfers == 2
    assert "WC1" in new_state.chips_used


def test_apply_gw_outcomes_free_hit_reverts_squad_and_bank() -> None:
    """FH transfers are temporary: next-GW state keeps the pre-FH squad."""
    s = BacktestState(
        season_id=24,
        gameweek=18,
        squad=frozenset({100, 200}),
        bank=10,
        free_transfers=1,
        cost_basis={100: 80, 200: 70},
    )
    decisions = GwDecisions(
        gameweek=18,
        squad=frozenset({300, 400}),
        starting_xi=frozenset({300, 400}),
        captain=300,
        vice=400,
        transferred_in=frozenset({300, 400}),
        transferred_out=frozenset({100, 200}),
        chip_played="FH1",
        hits=0,
        objective_value=0.0,
    )
    new_state = apply_gw_outcomes(
        s,
        decisions,
        actual_prices={
            100: {"sell": 80},
            200: {"sell": 70},
            300: {"buy": 90},
            400: {"buy": 60},
        },
    )
    assert new_state.squad == s.squad
    assert new_state.bank == s.bank
    assert new_state.cost_basis == s.cost_basis
    assert new_state.free_transfers == 2
    assert "FH1" in new_state.chips_used


def test_apply_gw_outcomes_ft_caps_at_5() -> None:
    """FT count caps at 5 even with multi-week buildup."""
    s = BacktestState(
        season_id=24,
        gameweek=5,
        squad=frozenset({100}),
        bank=0,
        free_transfers=5,
    )
    decisions = GwDecisions(
        gameweek=5,
        squad=frozenset({100}),
        starting_xi=frozenset({100}),
        captain=100,
        vice=None,
        transferred_in=frozenset(),
        transferred_out=frozenset(),
        chip_played=None,
        hits=0,
        objective_value=0.0,
    )
    new_state = apply_gw_outcomes(s, decisions, actual_prices={})
    assert new_state.free_transfers == 5  # min(5, 5+1-0) = 5


def test_apply_gw_outcomes_advances_gameweek() -> None:
    s = BacktestState(season_id=24, gameweek=10)
    decisions = GwDecisions(
        gameweek=10,
        squad=frozenset(),
        starting_xi=frozenset(),
        captain=None,
        vice=None,
        transferred_in=frozenset(),
        transferred_out=frozenset(),
        chip_played=None,
        hits=0,
        objective_value=0.0,
    )
    new_state = apply_gw_outcomes(s, decisions, actual_prices={})
    assert new_state.gameweek == 11
