"""Tests for the Phase 7 2.3 set-piece role resolvers."""
from __future__ import annotations

from fpl_bot.features import manual_overrides


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
