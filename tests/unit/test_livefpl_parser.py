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


def test_int_or_none_reads_the_hidden_element_id_cell():
    from fpl_bot.ingest.livefpl import _int_or_none

    assert _int_or_none("154") == 154
    assert _int_or_none("<span> 301 </span>") == 301
    # A missing or non-numeric id must fall through to the name path, not raise.
    assert _int_or_none("") is None
    assert _int_or_none("-") is None


def _db_available() -> bool:
    from sqlalchemy import text

    from fpl_bot.db.session import engine

    try:
        with engine.connect() as c:
            c.execute(text("SELECT 1"))
        return True
    except Exception:
        return False


def test_name_lookup_refuses_to_guess_between_duplicate_web_names():
    """A shared surname must never resolve to an arbitrary one of its owners.

    LiveFPL rows were keyed on web_name alone, so for a duplicated surname
    whichever row the query returned last silently won. In 2026-27 that gave
    Cole Palmer's top-10k EO to Alex Palmer, a promoted-club backup keeper —
    leaving the most-templated midfielder in the game looking like a free
    differential to the MILP's rho-EO term.
    """
    import pytest

    if not _db_available():
        pytest.skip("PostgreSQL not available; integration test skipped")

    from sqlalchemy import select

    from fpl_bot.db.models import DimPlayer, DimPlayerSeasonXref
    from fpl_bot.db.session import session_scope
    from fpl_bot.ingest.livefpl import _build_lookups

    by_element, by_name, _ = _build_lookups(26)
    if not by_element:
        pytest.skip("season 26 players not ingested locally")

    with session_scope() as s:
        rows = s.execute(
            select(DimPlayer.web_name, DimPlayer.player_id)
            .join(
                DimPlayerSeasonXref,
                DimPlayerSeasonXref.player_id == DimPlayer.player_id,
            )
            .where(DimPlayerSeasonXref.season_id == 26)
        ).all()

    counts: dict[str, int] = {}
    for r in rows:
        counts[r.web_name] = counts.get(r.web_name, 0) + 1
    duplicated = {n for n, c in counts.items() if c > 1}
    assert duplicated, "expected 2026-27 to contain duplicate web_names"

    leaked = duplicated & set(by_name)
    assert not leaked, f"ambiguous web_names still resolve to one player: {sorted(leaked)}"

    # Every registered player is reachable by their exact element id.
    assert len(by_element) == len(rows)
