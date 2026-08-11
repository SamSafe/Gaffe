"""Tests for the Phase 7 2.3 set-piece role resolvers."""
from __future__ import annotations

from unittest.mock import patch

from fpl_bot.features import manual_overrides
from fpl_bot.features.bps import _resolved_pk_taker_set
from fpl_bot.features.goals import _resolved_pk_takers, _resolved_set_piece_players


def test_yaml_loads():
    """Sanity: the set_piece_takers.yaml is parseable + indexed by season."""
    raw = manual_overrides.set_piece_takers_raw()
    assert isinstance(raw, dict)
    assert 24 in raw, "season 24 must be configured (used as training data)"


def test_season_24_arsenal_has_corner_takers():
    raw = manual_overrides.set_piece_takers_raw()
    arsenal = raw[24].get("Arsenal")
    assert arsenal is not None
    assert "Rice" in arsenal.get("corner", [])
    assert "Saka" in arsenal.get("direct_fk", [])


def test_season_25_stub_present():
    """A stub entry must exist for season 25 (Phase 7 2.3 placeholder)
    even if empty, so the resolver doesn't error.
    """
    raw = manual_overrides.set_piece_takers_raw()
    assert 25 in raw


def test_corner_takers_helper():
    takers = manual_overrides.corner_takers(24, "Arsenal")
    assert "Rice" in takers


def test_direct_fk_helper():
    takers = manual_overrides.direct_fk_takers(24, "Liverpool")
    assert "M.Salah" in takers


def test_penalty_helper():
    pk = manual_overrides.penalty_taker(24, "Liverpool")
    assert pk == "M.Salah"


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
