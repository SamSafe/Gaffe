import datetime as dt

import polars as pl
import pytest

from fpl_bot.features import goals, minutes


@pytest.fixture
def history(monkeypatch):
    start = dt.datetime(2026, 8, 22, 14, tzinfo=dt.UTC)
    frame = pl.DataFrame([
        {
            "player_id": player, "fixture_id": fixture, "season_id": 26,
            "gameweek": fixture, "kickoff_utc": start + dt.timedelta(days=7 * (fixture - 1)),
            "minutes": mins, "goals": 0, "assists": 0, "goals_conceded": 1,
            "was_home": True, "home_team_id": 1, "away_team_id": 2,
            "expected_goals": xg, "expected_assists": 0.1,
        }
        for player, fixture, mins, xg in [(1, 1, 90, 0.1), (1, 2, 60, 0.8), (2, 1, 30, 0.2), (2, 2, 0, 0.0)]
    ])
    monkeypatch.setattr(minutes.pit, "all_player_match_with_kickoff", lambda **kwargs: frame)
    monkeypatch.setattr(minutes.pit, "all_player_positions", lambda: pl.DataFrame({
        "player_id": [1, 2], "position_code": ["FWD", "MID"],
    }))
    monkeypatch.setattr(goals.pit, "understat_player_match_history", pl.DataFrame)
    monkeypatch.setattr(goals.pit, "market_xg_for_fixtures", pl.DataFrame)
    monkeypatch.setattr(goals, "_resolved_pk_takers", lambda _: pl.DataFrame())
    monkeypatch.setattr(goals, "_resolved_set_piece_players", lambda _: pl.DataFrame())
    monkeypatch.setattr(goals, "_resolved_role_mismatch_player_ids", set)
    return frame


def test_live_minutes_include_latest_result_without_changing_historical_rows(history):
    historical = minutes.build_feature_table([26]).sort(["player_id", "fixture_id"])
    advanced = minutes.build_feature_table([26], include_latest_match=True)
    assert advanced.filter(pl.col("fixture_id") > 0).sort(["player_id", "fixture_id"]).equals(historical)
    latest = advanced.filter(pl.col("fixture_id") == -1).sort("player_id")
    assert latest["minutes_bucket"].to_list() == [None, None]
    assert latest["min_last_1"].to_list() == [60, 0]
    assert latest["min_last_3"].to_list() == [75.0, 15.0]
    assert latest["start60_rate_3"].to_list() == [1.0, 0.0]
    assert latest["season_match_count"].to_list() == [2, 2]


def test_live_attack_includes_latest_appearance_and_keeps_labels_empty(history):
    historical = goals.build_feature_table([26]).sort(["player_id", "fixture_id"])
    advanced = goals.build_feature_table([26], include_latest_match=True)
    assert advanced.filter(pl.col("fixture_id") > 0).sort(["player_id", "fixture_id"]).equals(historical)
    latest = advanced.filter(pl.col("fixture_id") == -1).sort("player_id")
    assert latest["goals"].to_list() == [None, None]
    assert latest["minutes_factor"].to_list() == [None, None]
    assert latest["xg_per_90_last_3"][0] == pytest.approx((0.1 + 0.8 * 90 / 60) / 2)
    assert latest["xg_per_90_last_3"][1] == pytest.approx(0.2 * 90 / 30)


def _forecast_gw4(history, monkeypatch, schedule):
    kickoff = dt.datetime(2026, 9, 12, 14, tzinfo=dt.UTC)
    monkeypatch.setattr(minutes.pit, "upcoming_fixtures", lambda _: pl.DataFrame({
        "fixture_id": [4], "season_id": [26], "gameweek": [4],
        "kickoff_utc": [kickoff], "home_team_id": [1], "away_team_id": [2],
    }))
    monkeypatch.setattr(minutes.pit, "season_player_status_snapshot", lambda _: pl.DataFrame({
        "player_id": [1, 2], "team_id": [1, 1], "position_code": ["FWD", "MID"],
    }))
    monkeypatch.setattr(minutes.pit, "season_start_kickoff", lambda _: history["kickoff_utc"].min())
    monkeypatch.setattr(minutes.pit, "team_fixture_kickoffs", lambda _: schedule)
    return minutes.build_prediction_feature_table(26, [4]).sort("player_id")


def test_upcoming_rest_days_use_target_kickoff_without_intervening_fixture(history, monkeypatch):
    schedule = pl.DataFrame({
        "team_id": [1, 1, 1], "fixture_id": [1, 2, 4],
        "kickoff_utc": [history["kickoff_utc"].min() + dt.timedelta(days=d) for d in (0, 7, 21)],
    })
    forecast = _forecast_gw4(history, monkeypatch, schedule)
    assert forecast["days_since_last_match"].to_list() == [14.0, 14.0]
    assert forecast["season_match_count"].to_list() == [2, 2]


def test_horizon_rest_and_match_count_follow_the_club_schedule(history, monkeypatch):
    """GW3 is scheduled but not yet ingested: GW4 rest is 7 days, not 14."""
    schedule = pl.DataFrame({
        "team_id": [1, 1, 1, 1], "fixture_id": [1, 2, 3, 4],
        "kickoff_utc": [history["kickoff_utc"].min() + dt.timedelta(days=d) for d in (0, 7, 14, 21)],
    })
    forecast = _forecast_gw4(history, monkeypatch, schedule)
    assert forecast["days_since_last_match"].to_list() == [7.0, 7.0]
    assert forecast["season_match_count"].to_list() == [3, 3]


def test_live_team_form_includes_latest_fixture(history):
    before = goals._team_rolling_form([26]).sort("fixture_id")
    inclusive = goals._team_rolling_form([26], include_latest_match=True).sort("fixture_id")
    assert before["team_goals_conceded_last_5"].to_list() == [None, 1.0]
    assert inclusive["team_goals_conceded_last_5"].to_list() == [1.0, 1.0]
