"""Unit tests for the MILP build — constraint correctness on small problems."""
from __future__ import annotations

import pytest

from fpl_bot.optim.milp import MilpInputs, build_milp, solve_rolling_horizon
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


def test_gw19_gw20_free_hits_cannot_both_activate() -> None:
    candidates, positions, teams, prices = _make_balanced_pool()
    inputs = _toy_inputs(
        candidates=candidates,
        positions=positions,
        teams=teams,
        prices=prices,
        horizon=[19, 20],
        state=BacktestState.cold_start(season_id=26),
    )
    inputs.enable_chips = True

    model = build_milp(inputs)

    constraint = model.con_no_boundary_consecutive_free_hits
    assert constraint.upper() == 1
    assert "z_fh[19]" in str(constraint.body)
    assert "z_fh[20]" in str(constraint.body)


def test_gw1_with_a_prepicked_squad_restructures_without_hits() -> None:
    """FPL grants unlimited free transfers until the season's first deadline.

    A manager who pre-picked a squad still holds a state with 1 free transfer,
    so before this was handled the live GW1 solve made exactly ONE transfer and
    called everything else a -4 hit. Verified against the real 26/27 opener.
    """
    candidates, positions, teams, prices = _make_balanced_pool()
    # Own the 15 most expensive players, then make the cheap ones score best so
    # a full restructure is clearly optimal if (and only if) it is free.
    owned = sorted(candidates, key=lambda p: -prices[p])[:15]
    state = BacktestState(
        season_id=26,
        gameweek=1,
        squad=frozenset(owned),
        bank=1000,
        free_transfers=1,
        cost_basis={p: prices[p] for p in owned},
    )
    inputs = _toy_inputs(
        candidates=candidates,
        positions=positions,
        teams=teams,
        prices=prices,
        horizon=[1],
        state=state,
    )
    inputs.predictions.update(
        {(p, 1): (100.0 if p not in owned else 0.0) for p in candidates}
    )
    decisions, _ = solve_rolling_horizon(inputs, time_limit_s=60)
    assert decisions.hits == 0, "GW1 transfers must never cost a hit"
    assert len(decisions.transferred_in) > 1, (
        "GW1 should be free to restructure beyond a single free transfer"
    )


def test_gw2_still_charges_hits_beyond_free_transfers() -> None:
    """The GW1 exemption must not leak into the rest of the season."""
    candidates, positions, teams, prices = _make_balanced_pool()
    owned = sorted(candidates, key=lambda p: -prices[p])[:15]
    state = BacktestState(
        season_id=26,
        gameweek=2,
        squad=frozenset(owned),
        bank=1000,
        free_transfers=1,
        cost_basis={p: prices[p] for p in owned},
    )
    inputs = _toy_inputs(
        candidates=candidates,
        positions=positions,
        teams=teams,
        prices=prices,
        horizon=[2],
        state=state,
    )
    inputs.predictions.update(
        {(p, 2): (100.0 if p not in owned else 0.0) for p in candidates}
    )
    decisions, _ = solve_rolling_horizon(inputs, time_limit_s=60)
    assert decisions.hits > 0, "beyond GW1, extra transfers must still cost hits"
