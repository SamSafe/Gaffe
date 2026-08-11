"""Tests for the Phase 5 heuristic chip scheduler."""
from __future__ import annotations

import polars as pl

from fpl_bot.optim.chip_scheduler import (
    COLD_START_GWS,
    ChipSchedule,
    free_hit_gws,
    horizon_before_free_hit,
    make_chip_schedule,
)
from fpl_bot.optim.fixture_analytics import SeasonFixtureAnalytics


def _make_analytics_with_bgw_dgw() -> SeasonFixtureAnalytics:
    """Synthetic season: 38 GWs, 4 teams, BGW18 (2 teams blank) and DGW34 (2 teams DGW)."""
    fixture_count: dict[tuple[int, int], int] = {}
    gws_per_team: dict[int, set[int]] = {t: set() for t in (1, 2, 3, 4)}
    all_gws = set()
    # Normal GWs: all 4 teams play
    for gw in range(1, 39):
        all_gws.add(gw)
        if gw == 18:
            # BGW18: only teams 1 and 2 play; teams 3, 4 blank
            fixture_count[(1, gw)] = 1
            fixture_count[(2, gw)] = 1
            gws_per_team[1].add(gw)
            gws_per_team[2].add(gw)
        elif gw == 34:
            # DGW34: teams 3, 4 play twice; teams 1, 2 play once
            for t, count in [(1, 1), (2, 1), (3, 2), (4, 2)]:
                fixture_count[(t, gw)] = count
                gws_per_team[t].add(gw)
        else:
            for t in (1, 2, 3, 4):
                fixture_count[(t, gw)] = 1
                gws_per_team[t].add(gw)
    return SeasonFixtureAnalytics(
        season_id=99,
        gws_per_team={t: sorted(gs) for t, gs in gws_per_team.items()},
        fixture_count=fixture_count,
        all_gws=sorted(all_gws),
    )


def _make_predictions(team_id_per_player: dict[int, int]) -> pl.DataFrame:
    """Uniform predictions: every player 3 xPts per GW. DGW players get 6
    (double scoring at GW34). Used to verify TC picks the DGW player."""
    rows = []
    for pid, team in team_id_per_player.items():
        for gw in range(1, 39):
            xpts = 6.0 if gw == 34 and team in (3, 4) else 3.0
            rows.append({"player_id": pid, "gameweek": gw, "e_xpts": xpts})
    return pl.DataFrame(rows)


def test_fh_picks_max_bgw_gw():
    a = _make_analytics_with_bgw_dgw()
    team_id_per_player = {101: 1, 102: 2, 103: 3, 104: 4}
    preds = _make_predictions(team_id_per_player)
    sched = make_chip_schedule(
        analytics=a, predictions_df=preds, team_id_per_player=team_id_per_player
    )
    # BGW18 is the only BGW in first half → FH1 should land there.
    assert sched.fh1 == 18, f"FH1 expected 18, got {sched.fh1}"


def test_no_fh_chip_when_no_bgw():
    """Synthetic season with no BGW in second half — fh2 should be None."""
    fixture_count = {(t, gw): 1 for gw in range(1, 39) for t in (1, 2, 3, 4)}
    a = SeasonFixtureAnalytics(
        season_id=99,
        gws_per_team={t: list(range(1, 39)) for t in (1, 2, 3, 4)},
        fixture_count=fixture_count,
        all_gws=list(range(1, 39)),
    )
    team_id_per_player = {101: 1, 102: 2, 103: 3, 104: 4}
    preds = _make_predictions(team_id_per_player)
    sched = make_chip_schedule(
        analytics=a, predictions_df=preds, team_id_per_player=team_id_per_player
    )
    # No BGW anywhere → both FH1 and FH2 should be None
    assert sched.fh1 is None
    assert sched.fh2 is None


def test_tc_picks_dgw_player():
    a = _make_analytics_with_bgw_dgw()
    team_id_per_player = {101: 1, 102: 2, 103: 3, 104: 4}
    preds = _make_predictions(team_id_per_player)
    sched = make_chip_schedule(
        analytics=a, predictions_df=preds, team_id_per_player=team_id_per_player
    )
    # DGW is at GW34 (second half) → TC2 should land there.
    assert sched.tc2 == 34, f"TC2 expected 34 (DGW), got {sched.tc2}"


def test_bb_picks_non_colliding_in_each_half():
    a = _make_analytics_with_bgw_dgw()
    team_id_per_player = {101: 1, 102: 2, 103: 3, 104: 4}
    preds = _make_predictions(team_id_per_player)
    sched = make_chip_schedule(
        analytics=a, predictions_df=preds, team_id_per_player=team_id_per_player
    )
    # TC2 took GW34 (priority FH > TC > BB). BB2 falls back via xPts proxy.
    # Verify BB1/BB2 are assigned and don't collide with FH/TC.
    assert sched.bb1 is not None
    assert sched.bb2 is not None
    assert sched.bb1 != sched.fh1
    assert sched.bb2 != sched.tc2


def test_wc1_plays_before_fh1():
    a = _make_analytics_with_bgw_dgw()
    team_id_per_player = {101: 1, 102: 2, 103: 3, 104: 4}
    preds = _make_predictions(team_id_per_player)
    sched = make_chip_schedule(
        analytics=a, predictions_df=preds, team_id_per_player=team_id_per_player
    )
    # FH1 at GW18 → WC1 should land at GW17.
    assert sched.wc1 == 17, f"WC1 expected 17, got {sched.wc1}"


def test_chips_dont_collide():
    a = _make_analytics_with_bgw_dgw()
    team_id_per_player = {101: 1, 102: 2, 103: 3, 104: 4}
    preds = _make_predictions(team_id_per_player)
    sched = make_chip_schedule(
        analytics=a, predictions_df=preds, team_id_per_player=team_id_per_player
    )
    chips_gws = [
        v
        for v in (
            sched.wc1, sched.fh1, sched.bb1, sched.tc1,
            sched.wc2, sched.fh2, sched.bb2, sched.tc2,
        )
        if v is not None
    ]
    assert len(chips_gws) == len(set(chips_gws)), f"chip collision: {chips_gws}"


def test_as_dict_skips_none():
    sched = ChipSchedule(
        wc1=8, fh1=18, bb1=None, tc1=None,
        wc2=None, fh2=None, bb2=25, tc2=None,
    )
    d = sched.as_dict()
    assert d == {"WC1": 8, "FH1": 18, "BB2": 25}


def test_horizon_stops_before_future_free_hit():
    sched = {"FH1": 18, "BB1": 19}
    assert horizon_before_free_hit([15, 16, 17, 18, 19], sched) == [15, 16, 17]


def test_horizon_keeps_only_current_free_hit_gw():
    sched = {"FH1": 18}
    assert horizon_before_free_hit([18, 19, 20], sched) == [18]
    assert free_hit_gws(sched) == {18}


def test_no_chip_is_scheduled_inside_the_cold_start_window():
    """GW1-3 predictions are near-flat (no current-season rolling features), so
    a chip scheduled there is spent on noise. Measured on the 26/27 opener: TC
    landed on GW1 with a £4.0m defender as captain.

    Predictions here peak at GW1 precisely to bait the scheduler into it.
    """
    fixture_count = {(t, gw): 1 for gw in range(1, 39) for t in (1, 2, 3, 4)}
    a = SeasonFixtureAnalytics(
        season_id=99,
        gws_per_team={t: list(range(1, 39)) for t in (1, 2, 3, 4)},
        fixture_count=fixture_count,
        all_gws=list(range(1, 39)),
    )
    team_id_per_player = {101: 1, 102: 2, 103: 3, 104: 4}
    rows = [
        {
            "player_id": pid,
            "gameweek": gw,
            # GW1 looks best on paper; it is exactly the trap.
            "e_xpts": 99.0 if gw == 1 else 3.0,
        }
        for pid in team_id_per_player
        for gw in range(1, 39)
    ]
    sched = make_chip_schedule(
        analytics=a,
        predictions_df=pl.DataFrame(rows),
        team_id_per_player=team_id_per_player,
    )
    for slot, gw in sched.as_dict().items():
        assert gw is None or gw > COLD_START_GWS, (
            f"{slot} scheduled at GW{gw}, inside the cold-start window"
        )


def test_free_hits_are_not_scheduled_in_both_gw19_and_gw20():
    """The two half-season FH slots cannot be used in consecutive GWs."""
    teams = (1, 2, 3, 4)
    fixture_count: dict[tuple[int, int], int] = {}
    gws_per_team: dict[int, list[int]] = {team: [] for team in teams}
    for gw in range(1, 39):
        playing = (1,) if gw in {19, 20} else ((1, 2) if gw == 21 else teams)
        for team in playing:
            fixture_count[(team, gw)] = 1
            gws_per_team[team].append(gw)
    analytics = SeasonFixtureAnalytics(
        season_id=26,
        gws_per_team=gws_per_team,
        fixture_count=fixture_count,
        all_gws=list(range(1, 39)),
    )
    team_id_per_player = {101: 1, 102: 2, 103: 3, 104: 4}
    schedule = make_chip_schedule(
        analytics=analytics,
        predictions_df=_make_predictions(team_id_per_player),
        team_id_per_player=team_id_per_player,
    )

    assert schedule.fh1 == 19
    assert schedule.fh2 == 21
