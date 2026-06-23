"""Tests for the Phase 7 2.1 FPL news extractor."""
from __future__ import annotations

import datetime as dt
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from fpl_bot.live.news_extract import (
    extract_news_from_status_rows,
    latest_news_per_player,
)


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


def test_latest_news_historical_replay_applies_pit_cutoff_and_season() -> None:
    statements = []

    class FakeResult:
        def all(self):
            return [SimpleNamespace(player_id=700, news="Expected back 15 Jun")]

    class FakeSession:
        def execute(self, statement):
            statements.append(statement)
            return FakeResult()

    @contextmanager
    def fake_session_scope():
        yield FakeSession()

    as_of = dt.datetime(2026, 8, 1, 12, tzinfo=dt.UTC)
    with patch("fpl_bot.live.news_extract.session_scope", fake_session_scope):
        out = latest_news_per_player(season_id=25, as_of=as_of)

    assert out[0].return_date == dt.date(2027, 6, 15)
    sql = str(statements[0])
    assert "fact_player_status.season_id" in sql
    assert "fact_player_status.recorded_at <=" in sql


def test_latest_news_rejects_naive_as_of() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        latest_news_per_player(as_of=dt.datetime(2026, 8, 1, 12))
