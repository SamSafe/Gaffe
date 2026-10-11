import json

import polars as pl

from fpl_bot.db import pit
from fpl_bot.live import retrospective


def test_retrospective_uses_saved_bench_order_without_hindsight(tmp_path, monkeypatch):
    decisions = {
        "gameweek": 7, "squad": list(range(1, 16)), "starting_xi": list(range(1, 12)),
        "captain": 9, "vice": 10, "transferred_in": [], "transferred_out": [],
        "chip_played": None, "hits": 0, "objective_value": 50.0,
        "bench_order": [13, 14, 15, 12],
    }
    (tmp_path / "recommendation.json").write_text(json.dumps({"decisions": decisions}))
    points = dict.fromkeys(range(1, 16), 2)
    points.update({5: 0, 13: 1, 14: 10})
    played = dict.fromkeys(range(1, 16), 90)
    played[5] = 0
    monkeypatch.setattr(retrospective, "_per_player_actuals_for_gw", lambda *_: (points, played))
    positions = ["GKP", "DEF", "DEF", "DEF", "MID", "MID", "MID", "MID", "FWD", "FWD", "FWD", "GKP", "MID", "DEF", "FWD"]
    monkeypatch.setattr(pit, "all_player_positions", lambda: pl.DataFrame({
        "player_id": list(range(1, 16)), "position_code": positions,
    }))
    result = retrospective.compute_retrospective(season_id=26, gameweek=7, team_id=718959, output_dir=tmp_path)
    actuals = json.loads(result.read_text())
    assert actuals["auto_subs"] == [[5, 13]]
    assert actuals["gw_points"] == 23
    assert actuals["bench_order_source"] == "saved_recommendation"
