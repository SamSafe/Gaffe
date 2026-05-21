"""Unit tests for rolling-backtest validity checks."""
from __future__ import annotations

from fpl_bot.eval.milp_backtest import GwBacktestRecord, _check_validity
from fpl_bot.optim.state import BacktestState, GwDecisions


def test_free_hit_budget_check_uses_temporary_budget_not_reverted_bank() -> None:
    state_before = BacktestState(
        season_id=24,
        gameweek=6,
        squad=frozenset({1, 2}),
        bank=20,
    )
    state_after = BacktestState(
        season_id=24,
        gameweek=7,
        squad=frozenset({1, 2}),
        bank=20,
        chips_used=frozenset({"FH1"}),
    )
    decisions = GwDecisions(
        gameweek=6,
        squad=frozenset({3, 4}),
        starting_xi=frozenset({3, 4}),
        captain=3,
        vice=4,
        transferred_in=frozenset({3, 4}),
        transferred_out=frozenset({1, 2}),
        chip_played="FH1",
        hits=0,
        objective_value=0.0,
    )
    record = GwBacktestRecord(
        gameweek=6,
        decisions=decisions,
        actual_points=0,
        state_before=state_before,
        state_after=state_after,
        solve_time_s=0.0,
        solve_status="ok",
    )

    budget_ok, transfers_ok, chips_ok, leak_ok, errors = _check_validity(
        record,
        state_before=state_before,
        state_after=state_after,
        prices={1: 490, 2: 490, 3: 500, 4: 490},
        used_chips_before=frozenset(),
    )

    assert budget_ok
    assert transfers_ok
    assert chips_ok
    assert leak_ok
    assert errors == []
