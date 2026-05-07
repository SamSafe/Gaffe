"""Unit tests for the MILP build — constraint correctness on small problems."""
from __future__ import annotations

import pytest

from fpl_bot.optim.milp import MilpInputs, solve_rolling_horizon
from fpl_bot.optim.state import BacktestState


def _toy_inputs(
    *,
    candidates: list[int],
    positions: dict[int, str],
    teams: dict[int, int],
    prices: dict[int, int],
    horizon: list[int],
    state: BacktestState,
    rho: float = 0.0,
) -> MilpInputs:
    pred = {(p, w): float(prices[p]) / 50.0 for p in candidates for w in horizon}
    eo = dict.fromkeys(candidates, 0.0)
    return MilpInputs(
        state=state,
        horizon_weeks=horizon,
        candidates=candidates,
        predictions=pred,
        eo=eo,
        buy_prices=prices,
        sell_prices=prices,
        positions=positions,
        teams=teams,
        rho=rho,
        alpha=0.0,
        beta=0.0,
    )


def _make_balanced_pool(n_teams: int = 20, per_team: int = 15) -> tuple[list, dict, dict, dict]:
    """Make a synthetic candidate pool covering n_teams with per-team players
    distributed across positions to satisfy 2-5-5-3 globally."""
    candidates: list[int] = []
    positions: dict[int, str] = {}
    teams: dict[int, int] = {}
    prices: dict[int, int] = {}
    pid = 1
    pos_pattern = (["GKP"] * 2 + ["DEF"] * 5 + ["MID"] * 5 + ["FWD"] * 3)
    base_price = 40
    for t in range(1, n_teams + 1):
        for i in range(per_team):
            pos = pos_pattern[i % len(pos_pattern)]
            candidates.append(pid)
            positions[pid] = pos
            teams[pid] = t
            prices[pid] = base_price + (i * 3)
            pid += 1
    return candidates, positions, teams, prices


@pytest.mark.integration  # uses HiGHS solver
def test_cold_start_pick_15_within_budget() -> None:
    candidates, positions, teams, prices = _make_balanced_pool()
    state = BacktestState.cold_start(season_id=24)
    inputs = _toy_inputs(
        candidates=candidates,
        positions=positions,
        teams=teams,
        prices=prices,
        horizon=[1],
        state=state,
    )
    decisions, meta = solve_rolling_horizon(inputs, time_limit_s=30)
    assert meta["termination"] == "optimal"
    assert len(decisions.squad) == 15
    pos_count = {p: sum(1 for s in decisions.squad if positions[s] == p) for p in ("GKP", "DEF", "MID", "FWD")}
    assert pos_count == {"GKP": 2, "DEF": 5, "MID": 5, "FWD": 3}
    total_cost = sum(prices[p] for p in decisions.squad)
    assert total_cost <= 1000


@pytest.mark.integration
def test_cold_start_xi_valid_formation() -> None:
    candidates, positions, teams, prices = _make_balanced_pool()
    state = BacktestState.cold_start(season_id=24)
    inputs = _toy_inputs(
        candidates=candidates,
        positions=positions,
        teams=teams,
        prices=prices,
        horizon=[1],
        state=state,
    )
    decisions, _ = solve_rolling_horizon(inputs, time_limit_s=30)
    assert len(decisions.starting_xi) == 11
    xi_pos = {p: sum(1 for s in decisions.starting_xi if positions[s] == p) for p in ("GKP", "DEF", "MID", "FWD")}
    assert xi_pos["GKP"] == 1
    assert xi_pos["DEF"] >= 3
    assert xi_pos["MID"] >= 2
    assert xi_pos["FWD"] >= 1
    # All XI in squad
    assert decisions.starting_xi.issubset(decisions.squad)


@pytest.mark.integration
def test_cold_start_captain_in_xi() -> None:
    candidates, positions, teams, prices = _make_balanced_pool()
    inputs = _toy_inputs(
        candidates=candidates,
        positions=positions,
        teams=teams,
        prices=prices,
        horizon=[1],
        state=BacktestState.cold_start(season_id=24),
    )
    decisions, _ = solve_rolling_horizon(inputs, time_limit_s=30)
    assert decisions.captain is not None
    assert decisions.captain in decisions.starting_xi
    assert decisions.vice is not None
    assert decisions.vice in decisions.starting_xi
    assert decisions.captain != decisions.vice


@pytest.mark.integration
def test_cold_start_per_team_cap_holds() -> None:
    candidates, positions, teams, prices = _make_balanced_pool()
    inputs = _toy_inputs(
        candidates=candidates,
        positions=positions,
        teams=teams,
        prices=prices,
        horizon=[1],
        state=BacktestState.cold_start(season_id=24),
    )
    decisions, _ = solve_rolling_horizon(inputs, time_limit_s=30)
    team_counts: dict[int, int] = {}
    for p in decisions.squad:
        team_counts[teams[p]] = team_counts.get(teams[p], 0) + 1
    assert all(c <= 3 for c in team_counts.values()), team_counts


@pytest.mark.integration
def test_cold_start_no_hits() -> None:
    """Cold start should never incur hits — first GW picks are free."""
    candidates, positions, teams, prices = _make_balanced_pool()
    inputs = _toy_inputs(
        candidates=candidates,
        positions=positions,
        teams=teams,
        prices=prices,
        horizon=[1],
        state=BacktestState.cold_start(season_id=24),
    )
    decisions, _ = solve_rolling_horizon(inputs, time_limit_s=30)
    assert decisions.hits == 0
