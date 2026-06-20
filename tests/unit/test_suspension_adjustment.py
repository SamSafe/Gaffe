"""Unit tests for domestic-suspension carry-over (pure card accounting)."""

from __future__ import annotations

import polars as pl

from fpl_bot.eval.suspension_adjustment import suspensions_from_frame


def _frame(rows: list[dict]) -> pl.DataFrame:
    return pl.DataFrame(
        rows,
        schema={
            "player_id": pl.Int64,
            "gameweek": pl.Int64,
            "yellow_cards": pl.Int64,
            "red_cards": pl.Int64,
        },
    )


def test_red_card_bans_next_gameweek():
    df = _frame([{"player_id": 1, "gameweek": 5, "yellow_cards": 0, "red_cards": 1}])
    assert suspensions_from_frame(df) == {(1, 6)}


def test_five_yellows_one_match_ban():
    rows = [
        {"player_id": 1, "gameweek": g, "yellow_cards": 1, "red_cards": 0}
        for g in range(1, 6)  # 5th yellow in GW5 (<= cutoff 19)
    ]
    assert suspensions_from_frame(_frame(rows)) == {(1, 6)}


def test_tenth_yellow_two_match_ban():
    rows = [
        {"player_id": 1, "gameweek": g, "yellow_cards": 1, "red_cards": 0}
        for g in range(1, 11)  # 5th in GW5, 10th in GW10
    ]
    # 5th -> ban GW6; 10th -> 2-match ban GW11, GW12.
    assert suspensions_from_frame(_frame(rows)) == {(1, 6), (1, 11), (1, 12)}


def test_yellow_after_cutoff_does_not_ban():
    # 5th yellow only reached in GW20, past the GW19 cutoff -> no ban.
    rows = [
        {"player_id": 1, "gameweek": g, "yellow_cards": 1, "red_cards": 0}
        for g in (4, 8, 12, 16, 20)
    ]
    assert suspensions_from_frame(_frame(rows)) == set()


def test_no_cards_no_ban():
    df = _frame([{"player_id": 1, "gameweek": 3, "yellow_cards": 0, "red_cards": 0}])
    assert suspensions_from_frame(df) == set()


def test_double_gameweek_cards_fold_together():
    # Two matches in GW7 with a yellow each -> reaches 5th & 6th together.
    rows = [
        {"player_id": 1, "gameweek": g, "yellow_cards": 1, "red_cards": 0}
        for g in (1, 2, 3)
    ] + [
        {"player_id": 1, "gameweek": 7, "yellow_cards": 1, "red_cards": 0},
        {"player_id": 1, "gameweek": 7, "yellow_cards": 1, "red_cards": 0},
    ]
    # cum goes 3 -> 5 crossing the 5-threshold in GW7 -> ban GW8.
    assert suspensions_from_frame(_frame(rows)) == {(1, 8)}


def test_players_are_independent():
    df = _frame(
        [
            {"player_id": 1, "gameweek": 5, "yellow_cards": 0, "red_cards": 1},
            {"player_id": 2, "gameweek": 5, "yellow_cards": 1, "red_cards": 0},
        ]
    )
    assert suspensions_from_frame(df) == {(1, 6)}
