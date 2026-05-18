"""Tests for the LiveFPL EO HTML parser."""
from __future__ import annotations

from fpl_bot.ingest.livefpl import _clean, _pct


def test_clean_strips_tags_and_whitespace():
    assert _clean("  <span>Haaland</span> ") == "Haaland"
    assert _clean("Salah") == "Salah"
    assert _clean("") == ""


def test_pct_with_value():
    assert _pct("27.85%") == 27.85
    assert _pct(" 0.5% ") == 0.5
    assert _pct("100%") == 100.0


def test_pct_empty_or_dash_returns_zero():
    assert _pct("") == 0.0
    assert _pct("-") == 0.0


def test_pct_strips_tags_too():
    assert _pct("<span>15.10%</span>") == 15.1


def test_pct_invalid_raises():
    import pytest
    with pytest.raises(ValueError):
        _pct("not a number")
