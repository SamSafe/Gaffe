"""Point-in-time bookmaker consensus and per-market de-vigging."""
from __future__ import annotations

import datetime as dt
from types import SimpleNamespace

import pytest
from sqlalchemy import inspect

from fpl_bot.db.models import FactOdds
from fpl_bot.db.pit import market_xg_for_fixtures
from fpl_bot.derive.dixon_coles import (
    _consensus_market_probs,
    _fair_binary_probability,
    _model_market_probabilities,
    fit_lambdas_from_markets,
)


def _row(
    bookmaker: str,
    market: str,
    selection: str,
    odds: float,
    quote_time: dt.datetime,
):
    return SimpleNamespace(
        bookmaker=bookmaker,
        market=market,
        selection=selection,
        decimal_odds=odds,
        quote_time=quote_time,
        event_time=dt.datetime(2026, 8, 15, 14, tzinfo=dt.UTC),
    )


def _h2h_snapshot(
    bookmaker: str,
    quote_time: dt.datetime,
    odds: tuple[float, float, float],
):
    return [
        _row(bookmaker, "1X2", selection, price, quote_time)
        for selection, price in zip(("home", "draw", "away"), odds, strict=True)
    ]


def test_fact_odds_primary_key_uses_quote_time_not_event_time() -> None:
    primary_keys = {column.name for column in inspect(FactOdds).primary_key}
    assert primary_keys == {"fixture_id", "bookmaker", "market", "selection", "quote_time"}


def test_devigs_each_book_before_averaging_consensus() -> None:
    quote_time = dt.datetime(2026, 8, 1, 12, tzinfo=dt.UTC)
    rows = [
        *_h2h_snapshot("sharp", quote_time, (2.0, 4.0, 4.0)),
        *_h2h_snapshot("soft", quote_time, (1.8, 3.6, 3.6)),
    ]

    result = _consensus_market_probs(rows)

    # Both books imply the same de-vigged 50/25/25 distribution despite
    # carrying different margins. Averaging raw reciprocals would not.
    assert result["markets"]["1X2"] == pytest.approx(
        {"home": 0.5, "draw": 0.25, "away": 0.25}
    )
    assert result["bookmaker_counts"]["1X2"] == 2


def test_latest_complete_snapshot_selected_at_cutoff() -> None:
    t1 = dt.datetime(2026, 8, 1, 10, tzinfo=dt.UTC)
    t2 = dt.datetime(2026, 8, 1, 12, tzinfo=dt.UTC)
    rows = [
        *_h2h_snapshot("book", t1, (2.0, 4.0, 4.0)),
        *_h2h_snapshot("book", t2, (1.5, 4.0, 12.0)),
    ]

    early = _consensus_market_probs(rows, as_of=t1)
    latest = _consensus_market_probs(rows, as_of=t2)

    assert early["markets"]["1X2"] == pytest.approx(
        {"home": 0.5, "draw": 0.25, "away": 0.25}
    )
    assert latest["markets"]["1X2"] == pytest.approx(
        {"home": 2 / 3, "draw": 0.25, "away": 1 / 12}
    )
    assert early["max_quote_time"] == t1
    assert latest["max_quote_time"] == t2


def test_incomplete_or_cross_timestamp_market_is_not_mixed() -> None:
    t1 = dt.datetime(2026, 8, 1, 10, tzinfo=dt.UTC)
    t2 = dt.datetime(2026, 8, 1, 11, tzinfo=dt.UTC)
    rows = [
        _row("book", "1X2", "home", 2.0, t1),
        _row("book", "1X2", "draw", 4.0, t1),
        _row("book", "1X2", "away", 4.0, t2),
    ]

    result = _consensus_market_probs(rows)

    assert "1X2" not in result["markets"]


def test_totals_and_canonical_handicap_are_complete_two_way_markets() -> None:
    quote_time = dt.datetime(2026, 8, 1, 12, tzinfo=dt.UTC)
    rows = [
        _row("book", "totals_3.25", "over", 1.95, quote_time),
        _row("book", "totals_3.25", "under", 1.95, quote_time),
        _row("book", "ah_-0.75", "home", 1.9, quote_time),
        _row("book", "ah_-0.75", "away", 2.0, quote_time),
    ]

    result = _consensus_market_probs(rows)

    assert sum(result["markets"]["totals_3.25"].values()) == pytest.approx(1.0)
    assert sum(result["markets"]["ah_-0.75"].values()) == pytest.approx(1.0)


def test_consensus_rejects_naive_cutoff() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        _consensus_market_probs([], as_of=dt.datetime(2026, 8, 1, 12))


def test_market_xg_pit_rejects_naive_cutoff_before_query() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        market_xg_for_fixtures(as_of=dt.datetime(2026, 8, 1, 12))


def test_asian_quarter_total_accounts_for_half_loss() -> None:
    outcomes = [(2.0, 0.25), (3.0, 0.50), (4.0, 0.25)]

    ordinary_three = _fair_binary_probability(outcomes, line=3.0, side="high")
    over_three_quarter = _fair_binary_probability(outcomes, line=3.25, side="high")

    assert ordinary_three == pytest.approx(0.5)
    # At exactly three goals, over 3.25 is half push / half loss.
    assert over_three_quarter == pytest.approx(1 / 3)


def test_multi_market_fit_recovers_synthetic_goal_rates() -> None:
    requested = {
        "1X2": {},
        "totals_2.5": {},
        "totals_3.25": {},
        "ah_-0.75": {},
    }
    observed = _model_market_probabilities(1.8, 1.1, requested, max_goals=14)

    lam_h, lam_a, loss = fit_lambdas_from_markets(observed, max_goals=14)

    assert lam_h == pytest.approx(1.8, abs=0.01)
    assert lam_a == pytest.approx(1.1, abs=0.01)
    assert loss < 1e-8
