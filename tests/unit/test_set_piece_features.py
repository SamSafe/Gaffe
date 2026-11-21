"""Tests for the Phase 7 2.3 set-piece role resolvers."""
from __future__ import annotations

import runpy
from datetime import date
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from fpl_bot.db import pit

from fpl_bot.features import manual_overrides
from fpl_bot.features.bps import _resolved_pk_taker_set
from fpl_bot.features.goals import _resolved_pk_takers, _resolved_set_piece_players


def test_committed_snapshot_matches_observations():
    root = Path(__file__).resolve().parents[2]
    render = runpy.run_path(str(root / "scripts/gen_set_piece_takers.py"))["render"]
    assert render() == (root / "configs/set_piece_takers.yaml").read_text()
    earlier = yaml.safe_load(render(date(2026, 10, 1)))
    assert earlier[26]["Fulham"]["corner"] == ["Bobb"]
    assert manual_overrides.corner_takers(26, "Fulham") == ["Iwobi", "Bobb"]


def test_unknown_team_returns_empty():
    takers = manual_overrides.corner_takers(24, "NotATeam")
    assert takers == []


def test_unknown_season_returns_empty():
    takers = manual_overrides.corner_takers(99, "Arsenal")
    assert takers == []


def test_duplicate_web_names_resolve_within_season_and_team():
    raw = {
        26: {
            "Chelsea": {
                "penalty": "Palmer",
                "direct_fk": [],
                "corner": ["Palmer"],
            },
            "Ipswich Town": {
                "penalty": "Palmer",
                "direct_fk": ["Palmer"],
                "corner": [],
            },
        }
    }
    teams = {(26, "Chelsea"): 6, (26, "Ipswich Town"): 12}
    players = {
        (26, 6, "Palmer"): 244851,
        (26, 12, "Palmer"): 112520,
    }
    with patch(
        "fpl_bot.features.goals.manual_overrides.set_piece_takers_raw",
        return_value=raw,
    ), patch(
        "fpl_bot.features.goals.pit.team_id_by_full_name",
        return_value=teams,
    ), patch(
        "fpl_bot.features.goals.pit.player_id_by_season_team_web_name",
        return_value=players,
    ):
        penalty = _resolved_pk_takers([26]).sort("player_team_id")
        set_pieces = _resolved_set_piece_players([26]).sort("player_team_id")
    with patch(
        "fpl_bot.features.bps.manual_overrides.set_piece_takers_raw",
        return_value=raw,
    ), patch(
        "fpl_bot.features.bps.pit.team_id_by_full_name",
        return_value=teams,
    ), patch(
        "fpl_bot.features.bps.pit.player_id_by_season_team_web_name",
        return_value=players,
    ):
        bps_penalty_takers = _resolved_pk_taker_set(26)

    assert penalty["pk_taker_player_id"].to_list() == [244851, 112520]
    assert bps_penalty_takers == {244851, 112520}
    assert set_pieces.select(
        "player_id", "is_corner_taker", "is_fk_taker"
    ).rows() == [
        (244851, 1, 0),
        (112520, 0, 1),
    ]


def test_role_overrides_have_stable_player_ids():
    overrides = manual_overrides.position_role_overrides()
    assert overrides
    assert all(isinstance(info.get("player_id"), int) for info in overrides.values())


@pytest.mark.integration
def test_configured_takers_resolve_for_every_configured_season():
    raw = manual_overrides.set_piece_takers_raw()
    teams = pit.team_id_by_full_name(list(raw))
    players = pit.player_id_by_season_team_web_name(list(raw))
    for season, block in raw.items():
        for team, roles in block.items():
            team_id = teams[season, team]
            for value in roles.values():
                for name in [value] if isinstance(value, str) else value:
                    assert (season, team_id, name) in players, (season, team, name)
