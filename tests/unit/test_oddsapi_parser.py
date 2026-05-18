"""Tests for the Odds API outcome → selection mapper."""
from __future__ import annotations

from fpl_bot.ingest.oddsapi import _outcome_to_selection


def test_h2h_home_draw_away():
    home = "Arsenal"
    away = "Burnley"
    assert _outcome_to_selection("h2h", {"name": "Arsenal", "price": 1.5}, home, away) == ("home", "1X2")
    assert _outcome_to_selection("h2h", {"name": "Draw", "price": 4.0}, home, away) == ("draw", "1X2")
    assert _outcome_to_selection("h2h", {"name": "Burnley", "price": 7.0}, home, away) == ("away", "1X2")


def test_h2h_unknown_name_returns_none():
    sel, _ = _outcome_to_selection(
        "h2h", {"name": "Some Random Team", "price": 1.0}, "Arsenal", "Burnley"
    )
    assert sel is None


def test_spreads_includes_point():
    home, away = "Arsenal", "Burnley"
    sel, market = _outcome_to_selection(
        "spreads", {"name": "Arsenal", "price": 1.94, "point": -2.5}, home, away
    )
    assert sel == "home"
    assert market == "ah_-2.5"
    sel, market = _outcome_to_selection(
        "spreads", {"name": "Burnley", "price": 1.96, "point": 2.5}, home, away
    )
    assert sel == "away"
    assert market == "ah_2.5"


def test_spreads_missing_point():
    sel, _ = _outcome_to_selection(
        "spreads", {"name": "Arsenal", "price": 1.94}, "Arsenal", "Burnley"
    )
    assert sel is None


def test_totals_over_under():
    sel, market = _outcome_to_selection(
        "totals", {"name": "Over", "price": 1.85, "point": 3.5}, "Arsenal", "Burnley"
    )
    assert sel == "over"
    assert market == "totals_3.5"
    sel, market = _outcome_to_selection(
        "totals", {"name": "Under", "price": 2.05, "point": 3.5}, "Arsenal", "Burnley"
    )
    assert sel == "under"


def test_unknown_market_returns_none_selection():
    sel, market = _outcome_to_selection(
        "btts", {"name": "Yes", "price": 1.8}, "Arsenal", "Burnley"
    )
    assert sel is None
    assert market == "btts"


def test_h2h_case_sensitive_team_name():
    """Confirm we don't accidentally fuzzy-match — names must match exactly."""
    sel, _ = _outcome_to_selection(
        "h2h", {"name": "arsenal", "price": 1.5}, "Arsenal", "Burnley"
    )
    assert sel is None
