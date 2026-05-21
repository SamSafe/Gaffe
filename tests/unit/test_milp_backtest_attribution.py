"""Unit tests for rolling-backtest points-loss attribution."""

from __future__ import annotations

from fpl_bot.eval.milp_backtest import (
    _build_gw_attribution,
    _postprocess_captain_decision,
)
from fpl_bot.optim.scorer import ScorerInputs, score_gw
from fpl_bot.optim.state import GwDecisions

SQUAD = frozenset({1, 2, 11, 12, 13, 14, 15, 21, 22, 23, 24, 25, 31, 32, 33})
XI = frozenset({1, 11, 12, 13, 14, 21, 22, 23, 24, 31, 32})
POSITIONS = {
    1: "GKP",
    2: "GKP",
    11: "DEF",
    12: "DEF",
    13: "DEF",
    14: "DEF",
    15: "DEF",
    21: "MID",
    22: "MID",
    23: "MID",
    24: "MID",
    25: "MID",
    31: "FWD",
    32: "FWD",
    33: "FWD",
}


def _decisions(*, captain: int = 21, vice: int = 22, chip: str | None = None) -> GwDecisions:
    return GwDecisions(
        gameweek=3,
        squad=SQUAD,
        starting_xi=XI,
        captain=captain,
        vice=vice,
        transferred_in=frozenset(),
        transferred_out=frozenset(),
        chip_played=chip,
        hits=0,
        objective_value=0.0,
    )


def _score(decisions: GwDecisions, points: dict[int, int]) -> int:
    minutes = {player_id: 90 for player_id in SQUAD}
    return score_gw(
        ScorerInputs(
            decisions=decisions,
            actual_pts=points,
            actual_minutes=minutes,
            positions=POSITIONS,
            bench_order_xpts={player_id: 1.0 for player_id in SQUAD},
        )
    ).gw_points


def _attribution(decisions: GwDecisions, points: dict[int, int]):
    minutes = {player_id: 90 for player_id in SQUAD}
    return _build_gw_attribution(
        gw=3,
        decisions=decisions,
        bot_points=_score(decisions, points),
        final_scoring_players=decisions.starting_xi,
        actual_pts_this_gw=points,
        actual_minutes_this_gw=minutes,
        actual_by_pgw={(player_id, 3): pts for player_id, pts in points.items()},
        positions=POSITIONS,
        bench_order_xpts={player_id: 1.0 for player_id in SQUAD},
        candidates=set(SQUAD),
    )


def test_attribution_measures_same_xi_captain_regret() -> None:
    points = {player_id: 2 for player_id in SQUAD}
    points[21] = 4
    points[23] = 12

    record = _attribution(_decisions(captain=21, vice=22), points)

    assert record.captain_oracle_captain == 23
    assert record.captain_regret == 8
    assert record.lineup_regret == 0


def test_attribution_measures_owned_squad_lineup_regret() -> None:
    points = {player_id: 2 for player_id in SQUAD}
    points[25] = 15

    record = _attribution(_decisions(captain=21, vice=22), points)

    assert 25 in record.lineup_oracle_xi
    assert record.lineup_regret == 13


def test_bench_boost_has_no_lineup_regret() -> None:
    points = {player_id: 2 for player_id in SQUAD}
    points[25] = 15
    points[33] = 13

    record = _attribution(_decisions(captain=21, vice=22, chip="BB"), points)

    assert record.lineup_regret == 0
    assert record.lineup_oracle_xi == XI


def test_postprocess_captain_decision_uses_alternate_score_within_xi() -> None:
    decisions = _decisions(captain=21, vice=22)
    updated = _postprocess_captain_decision(
        decisions,
        gameweek=3,
        captain_predictions={
            (23, 3): 9.0,
            (24, 3): 8.0,
            (25, 3): 20.0,  # bench player: ignored
        },
    )

    assert updated.captain == 23
    assert updated.vice == 24
    assert updated.starting_xi == decisions.starting_xi
    assert updated.squad == decisions.squad
