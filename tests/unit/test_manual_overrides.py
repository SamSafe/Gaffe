"""Sanity tests for hand-curated config loaders."""
from __future__ import annotations

from fpl_bot.features.manual_overrides import (
    actual_role,
    corner_takers,
    direct_fk_takers,
    is_role_mismatch,
    penalty_taker,
    position_role_overrides,
    set_piece_takers_raw,
)


def test_set_piece_takers_loads_at_least_one_season() -> None:
    raw = set_piece_takers_raw()
    assert raw, "set_piece_takers.yaml is missing or empty"
    assert 26 in raw


def test_observed_penalty_takers() -> None:
    assert penalty_taker(26, "Ipswich Town") == "Clarke"
    assert penalty_taker(26, "Chelsea") == "Palmer"
    assert penalty_taker(26, "Fulham") is None


def test_unknown_team_or_season_returns_none() -> None:
    assert penalty_taker(99, "Liverpool") is None
    assert penalty_taker(24, "Madeup FC") is None


def test_direct_fk_and_corner_lists_are_lists() -> None:
    fks = direct_fk_takers(26, "Ipswich Town")
    assert isinstance(fks, list)
    assert "Enciso" in fks
    corners = corner_takers(26, "Arsenal")
    assert isinstance(corners, list)


def test_position_role_overrides_known_cases() -> None:
    assert is_role_mismatch("M.Salah")
    assert actual_role("M.Salah") == "ST"
    assert is_role_mismatch("Palmer")
    assert actual_role("Palmer") == "F9"
    overrides = position_role_overrides()
    for web_name, info in overrides.items():
        assert "fpl_position" in info, f"{web_name} missing fpl_position"
        assert "actual_role" in info, f"{web_name} missing actual_role"


def test_unknown_player_is_not_mismatch() -> None:
    assert not is_role_mismatch("__nonexistent_player__")
    assert actual_role("__nonexistent_player__") is None
