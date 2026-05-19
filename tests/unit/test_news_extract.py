"""Tests for the Phase 7 2.1 FPL news extractor."""
from __future__ import annotations

import datetime as dt

from fpl_bot.live.news_extract import extract_news_from_status_rows


def test_expected_back_dd_mon():
    today = dt.date(2026, 5, 18)
    rows = [(100, "Knee injury - Expected back 25 May")]
    out = extract_news_from_status_rows(rows, today=today)
    assert len(out) == 1
    assert out[0].player_id == 100
    assert out[0].return_date == dt.date(2026, 5, 25)
    assert not out[0].out_for_season


def test_expected_back_wraps_to_next_year_when_month_already_past():
    today = dt.date(2026, 8, 1)
    rows = [(200, "Hamstring - Expected back 15 Jun")]
    out = extract_news_from_status_rows(rows, today=today)
    # June < August → assume next year
    assert out[0].return_date == dt.date(2027, 6, 15)


def test_out_for_season_flag():
    today = dt.date(2026, 1, 1)
    rows = [
        (300, "Achilles - Out for the season"),
        (301, "ACL - season-ending injury"),
        (302, "Knock - Expected back 15 Jan"),
    ]
    out = extract_news_from_status_rows(rows, today=today)
    by_pid = {r.player_id: r for r in out}
    assert by_pid[300].out_for_season is True
    assert by_pid[301].out_for_season is True
    assert by_pid[302].out_for_season is False


def test_unknown_return_no_match():
    today = dt.date(2026, 5, 18)
    rows = [(400, "Knee injury - Unknown return date")]
    out = extract_news_from_status_rows(rows, today=today)
    assert out[0].return_date is None
    assert out[0].out_for_season is False


def test_empty_news_dropped():
    rows = [(500, None), (501, ""), (502, "Knock - Expected back 25 May")]
    out = extract_news_from_status_rows(rows, today=dt.date(2026, 5, 18))
    assert len(out) == 1
    assert out[0].player_id == 502


def test_garbled_date_returns_none():
    today = dt.date(2026, 5, 18)
    rows = [(600, "Knee - Expected back 99 Foo")]
    out = extract_news_from_status_rows(rows, today=today)
    assert out[0].return_date is None
