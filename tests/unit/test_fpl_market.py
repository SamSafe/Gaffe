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


def test_failed_request_keeps_what_was_fetched(tmp_path, monkeypatch):
    import json

    import httpx

    class Client:
        def __init__(self, **kwargs):
            self.calls = 0

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def get(self, url):
            self.calls += 1
            request = httpx.Request("GET", url)
            if self.calls == 3:
                return httpx.Response(429, request=request)
            return httpx.Response(200, json={"history": [{"round": 1}]}, request=request)

    monkeypatch.setattr(fpl_market.httpx, "Client", Client)
    monkeypatch.setattr(fpl_market, "_today_dir", lambda source: tmp_path)
    raw = json.loads(fpl_market.fetch_raw_element_summaries([1, 2, 3, 4], pause_s=0).read_text())
    assert raw["complete"] is False
    assert sorted(raw["history"]) == ["1", "2"]


def test_transfer_completion_curve_scales_partial_weeks():
    from fpl_bot.db.pit import MARKET_MAX_HOURS, transfer_completion

    assert transfer_completion(0) == 1.0
    assert abs(transfer_completion(6) - 0.82) < 1e-9
    assert abs(transfer_completion(36) - 0.71) < 1e-9  # Halfway between 6h and 66h.
    assert transfer_completion(500) == transfer_completion(MARKET_MAX_HOURS)
