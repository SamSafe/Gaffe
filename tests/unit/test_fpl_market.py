"""Current-season transfer history: when a backfill pass is needed."""

from __future__ import annotations

import polars as pl

from fpl_bot.ingest import fpl_market


def test_only_mostly_empty_gameweeks_trigger_a_pass(monkeypatch):
    rows = pl.DataFrame({
        "gameweek": [1] * 10 + [2] * 10,
        # GW1: one departed player stays empty for good. GW2: not yet backfilled.
        "selected": [5] * 9 + [None] + [None] * 10,
    })
    monkeypatch.setattr(fpl_market.pit, "all_player_match_with_kickoff", lambda season_ids: rows)
    assert fpl_market.missing_market_gameweeks(26) == [2]


def test_no_pass_when_nothing_is_missing(monkeypatch):
    monkeypatch.setattr(
        fpl_market.pit, "all_player_match_with_kickoff", lambda season_ids: pl.DataFrame()
    )
    fetched = []
    monkeypatch.setattr(fpl_market, "fetch_raw_element_summaries", fetched.append)
    assert fpl_market.backfill_market_history(26, [1, 2]) == {
        "missing_gameweeks": 0,
        "fact_player_match": 0,
    }
    assert not fetched
