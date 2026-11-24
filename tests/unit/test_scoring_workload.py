import datetime as dt

import polars as pl

from fpl_bot.live.workload import appearances, comparison
from fpl_bot.models.bps import score_bps_known_events
from fpl_bot.models.xpts import score_fpl_points
from fpl_bot.web.store import Store, write_json


def test_keeper_rules_across_seasons():
    args = dict(position="GKP", minutes=70, goals=1, assists=0, team_clean_sheet=True,
                team_goals_conceded=0, saves=3, penalties_saved=1)
    assert score_fpl_points(**args, season_id=23) == 18
    assert score_fpl_points(**args, season_id=26) == 22
    args.update(goals=0, team_clean_sheet=False, team_goals_conceded=2)
    assert score_bps_known_events(**args, season_id=26) == 6 - 8 + 6 + 9


def test_extra_time_and_deadline(tmp_path):
    now = dt.datetime.now(dt.UTC)
    summary = {"header":{"competitions":[{"status":{"type":{"completed":True,"detail":"AET"}}}]},
               "rosters":[{"roster":[{"athlete":{"displayName":"Test Player"}, "starter":True}]}]}
    bootstrap = {"elements":[{"first_name":"Test","second_name":"Player","web_name":"Player","code":1}]}
    played = appearances(summary, bootstrap)
    assert played == [{"player_id":1,"minutes":120}]
    store = Store(tmp_path,season=26)
    log = store.web / "workload/26/match.json"
    write_json(log, {"kickoff":(now-dt.timedelta(days=1)).isoformat(),
                     "recorded_at":now.isoformat(),"players":played})
    path = tmp_path/"forecast.parquet"
    pl.DataFrame([dict(player_id=1,fixture_id=5,gameweek=3,kickoff_utc=now+dt.timedelta(days=1),
                       p_minutes_zero=.1,p_minutes_short=.1,p_minutes_full=.8,expected_minutes=63.,e_xpts=4.)]).write_parquet(path)
    row = comparison(store,path,3,now+dt.timedelta(hours=1))[0]
    assert row["workload_minutes"] == 120 and row["expected_minutes_news"] < 63
    assert not comparison(store,path,3,now)
