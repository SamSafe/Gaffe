"""Regression guards for the reproducibility fixes (commits bc6b0b1 + 81763fb).

If these fail, prediction-cache regeneration has become non-deterministic
again — which silently reintroduces tens of points of backtest noise and
makes model A/B comparisons untrustworthy (the lesson that killed both the
finishing-multiplier and FDR experiments). Keep them green.
"""
from __future__ import annotations

import numpy as np
import polars as pl

from fpl_bot.db.event_source import EventSource
from fpl_bot.features.goals import FEATURE_COLUMNS
from fpl_bot.models.bps import BPSSimulator, FixtureInputs
from fpl_bot.models.goals import train_per_90_model


class _ZeroEventSource(EventSource):
    """No-op residual source so the test isolates the sampling layer."""

    def simulate_unmodeled_bps(self, position, minutes_played, rng):
        return 0.0


def _make_inputs(players: pl.DataFrame | None = None) -> FixtureInputs:
    if players is None:
        rows = []
        for i in range(11):
            rows.append(_player_row(1000 + i, team_id=1, is_home=True, lam_g=0.4))
        for i in range(11):
            rows.append(_player_row(2000 + i, team_id=2, is_home=False, lam_g=0.25))
        players = pl.DataFrame(rows)
    return FixtureInputs(
        fixture_id=7, season_id=24, gameweek=1, home_team_id=1, away_team_id=2,
        home_team_lambda=1.6, away_team_lambda=1.1,
        home_market_cs_prob=0.3, away_market_cs_prob=0.4,
        players=players,
        alphas_by_position={"GKP": 0.4, "DEF": 0.4, "MID": 0.4, "FWD": 0.4},
    )


def _player_row(pid: int, *, team_id: int, is_home: bool, lam_g: float) -> dict:
    return {
        "player_id": pid, "position_code": "MID", "team_id": team_id, "is_home": is_home,
        "p_minutes_zero": 0.1, "p_minutes_short": 0.1, "p_minutes_full": 0.8,
        "lambda_goals_per_90": lam_g, "lambda_assists_per_90": 0.1,
        "saves_rate_per_90": 0.0, "yc_rate_per_90": 0.1, "rc_rate_per_90": 0.0,
        "is_penalty_taker": False,
    }


def test_simulate_fixture_is_reproducible():
    """Same inputs → identical predictions (guards per-fixture seeded RNG)."""
    sim = BPSSimulator(event_source=_ZeroEventSource(), n_iterations=300, seed=42)
    a = sim.simulate_fixture(_make_inputs()).sort("player_id")
    b = sim.simulate_fixture(_make_inputs()).sort("player_id")
    assert np.array_equal(a["e_xpts"].to_numpy(), b["e_xpts"].to_numpy())
    assert np.array_equal(a["p_xpts_ge_6"].to_numpy(), b["p_xpts_ge_6"].to_numpy())


def test_simulate_fixture_is_player_order_independent():
    """Shuffling the input player rows must not change per-player results —
    guards the players.sort('player_id') in simulate_fixture."""
    base = _make_inputs()
    shuffled = base.players.sample(fraction=1.0, shuffle=True, seed=99)
    sim = BPSSimulator(event_source=_ZeroEventSource(), n_iterations=300, seed=42)
    a = sim.simulate_fixture(base).sort("player_id")
    b = sim.simulate_fixture(_make_inputs(players=shuffled)).sort("player_id")
    assert a["player_id"].to_list() == b["player_id"].to_list()
    assert np.array_equal(a["e_xpts"].to_numpy(), b["e_xpts"].to_numpy())


def _synth_goals_df(n: int = 800, seed: int = 0) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    data: dict = {c: rng.random(n) for c in FEATURE_COLUMNS}
    data["goals"] = rng.poisson(0.3, n).astype(np.int64)
    data["minutes_factor"] = rng.uniform(0.5, 1.0, n)
    return pl.DataFrame(data)


def test_goals_model_training_is_reproducible():
    """Training twice on identical data → identical predictions (guards the
    LightGBM deterministic / force_row_wise / pinned num_threads params)."""
    df = _synth_goals_df()
    r1 = train_per_90_model(df, target="goals", valid=df, weighted=True).predict_per_90(df)
    r2 = train_per_90_model(df, target="goals", valid=df, weighted=True).predict_per_90(df)
    assert np.array_equal(r1, r2)
