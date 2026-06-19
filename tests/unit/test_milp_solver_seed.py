"""Unit tests for the CBC solver wrapper's reproducibility seed.

These exercise `solve_milp` directly against a tiny hand-built Pyomo model
(no DB / no `build_milp`), so they only validate the seed plumbing and that
CBC still solves correctly with the seed flags present. The seed's effect on
large-model run-to-run variance can only be shown via a multi-fold backtest,
not here.
"""

from __future__ import annotations

import pyomo.environ as pyo

from fpl_bot.optim.milp import DEFAULT_CBC_SEED, solve_milp


def _toy_knapsack() -> pyo.ConcreteModel:
    """Small binary knapsack with a unique optimum.

    Maximize 3a + 5b + 4c s.t. 2a + 3b + 2c <= 4, all binary.
    Optimal: b=c=1 (weight 5? no) -> enumerate: best feasible is a=c=1
    (value 7, weight 4) vs b+? b=1 weight3 leaves 1 -> only b (value 5),
    or c=1 weight2 + a=1 weight2 -> value 7. So optimum = 7 at a=c=1.
    """
    m = pyo.ConcreteModel()
    m.I = pyo.Set(initialize=["a", "b", "c"])
    value = {"a": 3, "b": 5, "c": 4}
    weight = {"a": 2, "b": 3, "c": 2}
    m.x = pyo.Var(m.I, domain=pyo.Binary)
    m.obj = pyo.Objective(
        expr=sum(value[i] * m.x[i] for i in m.I), sense=pyo.maximize
    )
    m.cap = pyo.Constraint(expr=sum(weight[i] * m.x[i] for i in m.I) <= 4)
    return m


def test_solve_milp_with_seed_finds_optimum():
    meta = solve_milp(_toy_knapsack(), time_limit_s=10, random_seed=DEFAULT_CBC_SEED)
    assert meta["termination"] in ("optimal", "feasible")
    assert meta["objective"] == 7.0


def test_solve_milp_without_seed_still_solves():
    # random_seed=None restores the unseeded CBC invocation; must still solve.
    meta = solve_milp(_toy_knapsack(), time_limit_s=10, random_seed=None)
    assert meta["termination"] in ("optimal", "feasible")
    assert meta["objective"] == 7.0


def test_seeded_solves_are_reproducible():
    m1 = solve_milp(_toy_knapsack(), time_limit_s=10, random_seed=12345)
    m2 = solve_milp(_toy_knapsack(), time_limit_s=10, random_seed=12345)
    assert m1["objective"] == m2["objective"]
    assert m1["termination"] == m2["termination"]
