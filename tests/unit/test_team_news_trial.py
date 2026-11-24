import datetime as dt
from xml.etree import ElementTree

import polars as pl
import pytest
from fastapi.testclient import TestClient

from fpl_bot.ingest.team_news import parse_feed
from fpl_bot.live.team_news import (
    NewsReport,
    NewsTrial,
    active_reports,
    extract_limit,
    record_report,
)
from fpl_bot.models.bps import BPSSimulator
from fpl_bot.web.app import create_app
from fpl_bot.web.schemas import Preferences
from fpl_bot.web.store import Store, write_json
from fpl_bot.web.team_news import news_dashboard
from tests.unit.test_simulation_determinism import _make_inputs, _player_row, _ZeroEventSource


@pytest.fixture
def setup(tmp_path):
    now = dt.datetime.now(dt.UTC)
    season = now.year - 2000
    store = Store(tmp_path, season=season)
    store.save_preferences(Preferences(team_id=123))
    folder = store.team_dir(123)
    event = {
        "id": 2,
        "deadline_time": (now + dt.timedelta(hours=1)).isoformat(),
        "data_checked": False,
    }
    bootstrap = {
        "events": [{"id": 1, "deadline_time": (now - dt.timedelta(days=7)).isoformat()}, event],
        "teams": [
            {"id": 1, "name": "Home", "short_name": "H"},
            {"id": 2, "name": "Away", "short_name": "A"},
        ],
        "elements": [
            {
                "code": i,
                "id": i + 10,
                "team": 1 if i != 3 else 2,
                "first_name": "Test",
                "second_name": f"Player{i}",
            }
            for i in (1, 2, 3)
        ],
    }
    fixtures = [{"event": 2, "code": 7, "id": 70, "team_h": 1, "team_a": 2}]
    for name, data in (
        ("bootstrap", bootstrap),
        ("fixtures", fixtures),
        ("entry", {"id": 123}),
        ("meta", {"refreshed_at": now.isoformat()}),
    ):
        write_json(folder / (name + ".json"), data)
    return store, folder, bootstrap, fixtures, now


def sample_report(now, **kw):
    return NewsReport.model_validate(
        {
            "fixture_id": 7,
            "player_id": 1,
            "replacement_id": 2,
            "source": "https://example.org/news",
            "quote": "He is limited to 30 minutes.",
            "published_at": now - dt.timedelta(minutes=3),
            **kw,
        }
    )


def test_cutoff_revisions_and_club_validation(setup):
    store, folder, b, fixtures, now = setup
    event = b["events"][-1]

    def read(at):
        return active_reports(store, event, b, fixtures, folder, at)

    row = record_report(store, sample_report(now), b, fixtures, now)
    assert row["cap"] == 30 and read(now)[0]["id"] == row["id"]
    assert not read(now - dt.timedelta(seconds=1))
    with pytest.raises(ValueError, match="teammate"):
        record_report(store, sample_report(now, replacement_id=3), b, fixtures, now)
    with pytest.raises(ValueError, match="past"):
        record_report(
            store, sample_report(now, published_at=now + dt.timedelta(seconds=1)), b, fixtures, now
        )
    later = now + dt.timedelta(minutes=1)
    record_report(store, sample_report(now, withdraw=True), b, fixtures, later)
    assert not read(later) and read(now)
    with pytest.raises(ValueError, match="open gameweek"):
        record_report(store, sample_report(now), b, fixtures, now + dt.timedelta(hours=1))


def test_automatic_and_ambiguous_feed(setup):
    _, folder, b, fixtures, now = setup
    event = b["events"][-1]
    published = (now - dt.timedelta(minutes=5)).strftime("%a, %d %b %Y %H:%M:%S GMT")

    def rss(description):
        return ElementTree.fromstring(
            f"<rss><channel><item><description>{description}</description>"
            f"<link>https://www.bbc.com/sport/football/articles/x</link><pubDate>{published}</pubDate>"
            "</item></channel></rss>"
        ).findall("./channel/item")

    fact = rss("Test Player1 will miss the Premier League game against Away.")
    rows = parse_feed(fact, b, fixtures, event, now)
    assert len(rows) == 1 and rows[0]["cap"] == 0
    draft = parse_feed(rss("Test Player1 has an injury."), b, fixtures, event, now)
    assert len(draft) == 1 and draft[0]["cap"] is None
    assert not parse_feed(fact, b, fixtures, event, now + dt.timedelta(hours=1))
    assert extract_limit("He is not limited to 30 minutes.") is None
    b["elements"][0].update(news="Limited to 30 minutes", news_added=now.isoformat())
    store = setup[0]
    assert active_reports(store, event, b, fixtures, folder, now)[0]["cap"] == 30
    assert not active_reports(store, event, b, [*fixtures, {**fixtures[0], "code": 8}], folder, now)


def test_joint_simulator_tracks_cap_and_replacement_without_changing_baseline():
    rows = [
        _player_row(pid, team_id=team, is_home=team == 1, lam_g=0.2)
        for pid, team in [(1, 1), (2, 1), (3, 2)]
    ]
    for row in rows:
        absent = row["player_id"] == 2
        row.update(
            p_minutes_zero=float(absent),
            p_minutes_short=0.0,
            p_minutes_full=float(not absent),
            alpha_60_89=0.0,
        )
    inputs = _make_inputs(pl.DataFrame(rows))
    sim = BPSSimulator(event_source=_ZeroEventSource(), n_iterations=24)
    draws = []
    baseline = sim.simulate_fixture(inputs, track_minutes=True, record_draws=draws)
    assert baseline["e_xpts"].equals(sim.simulate_fixture(inputs)["e_xpts"])
    news = NewsTrial(
        [{"fixture_id": 7, "player_id": 1, "cap": 20, "replacement_id": 2, "recorded_at": "a"}]
    )
    constraints = news.constraints(inputs, {})
    news.compare(sim, inputs, baseline, constraints, draws)
    primary, replacement = sorted(news.rows, key=lambda r: r["player_id"])
    assert primary["expected_minutes_news"] == 20 and primary["p_minutes_short_news"] == 1
    assert replacement["expected_minutes_news"] == 70 and replacement["p_minutes_full_news"] == 1
    frame = baseline.with_columns(
        pl.lit(7).alias("fixture_id"), (1 - pl.col("p_minutes_zero")).alias("p_appearance")
    )
    adjusted = news.adjusted_frame(frame)
    assert adjusted.filter(pl.col("player_id") == 2)["p_appearance"].item() == 1
    assert not news.constraints(inputs, {(2, inputs.gameweek): 65})


def test_finalized_fixture_scoring_and_archive_cutoff(setup):
    store, folder, b, fixtures, now = setup
    gw = b["events"][-1]
    gw["data_checked"] = True
    gw["deadline_time"] = (now - dt.timedelta(minutes=1)).isoformat()
    fixtures[0]["finished"] = True
    write_json(folder / "bootstrap.json", b)
    write_json(folder / "fixtures.json", fixtures)
    write_json(
        folder / "live_gw2.json",
        {
            "elements": [
                {
                    "id": 11,
                    "stats": {"minutes": 80, "total_points": 4},
                    "explain": [
                        {
                            "fixture": 70,
                            "stats": [{"identifier": "minutes", "value": 20, "points": 1}],
                        },
                        {
                            "fixture": 71,
                            "stats": [{"identifier": "minutes", "value": 60, "points": 2}],
                        },
                    ],
                }
            ]
        },
    )
    row = {
        "role": "reported",
        "player_id": 1,
        "fixture_id": 7,
        "expected_minutes": 80,
        "expected_minutes_news": 20,
        "p_minutes_zero": 0,
        "p_minutes_short": 0,
        "p_minutes_full": 1,
        "p_minutes_zero_news": 0,
        "p_minutes_short_news": 1,
        "p_minutes_full_news": 0,
        "e_xpts": 4,
        "e_xpts_news": 1,
    }
    for offset in (-10, -5, 0):
        run = now + dt.timedelta(minutes=offset)
        archive = (
            store.root
            / f"data/live/recommendations/season_{store.season}/team_123/gw_2/archive/{offset}"
        )
        write_json(
            archive / "recommendation.json",
            {
                "generated_at": run.isoformat(),
                "context": {"team_id": 123, "season_id": store.season, "gameweek": 2},
            },
        )
        write_json(archive / "complete.json", {"completed": True})
        write_json(
            archive / "team_news_trial.json",
            {
                "deadline": gw["deadline_time"],
                "finished_at": run.isoformat(),
                "reports": [{}],
                "shortlist": {"entered": [1, 2], "left": [3]},
                "rows": [{**row, "expected_minutes_news": 20 if offset == -5 else 90}],
            },
        )
    dashboard = news_dashboard(store)
    assert dashboard["history"][0]["entrant_points"] == {1: 4, 2: None}
    scores = {r["arm"]: r for r in dashboard["scores"]}
    assert scores["news"]["n"] == 1 and scores["news"]["minutes_mae"] == 0
    assert scores["baseline"]["points_mae"] == 3 and scores["news"]["brier"] == 0
    gw["data_checked"] = False
    write_json(folder / "bootstrap.json", b)
    assert not news_dashboard(store)["scores"]


def test_api_requires_token_and_serves_news_module(setup, monkeypatch):
    store, _, _, _, now = setup
    with TestClient(create_app(store)) as client:
        assert client.get("/api/team-news").json()["games"][0]["id"] == 7
        assert client.get("/assets/team-news.js").status_code == 200
        report = sample_report(now).model_dump(mode="json")
        assert client.post("/api/team-news", json=report).status_code == 403
        monkeypatch.setattr("fpl_bot.web.app.dashboard", lambda store: {})
        token = client.get("/api/dashboard").json()["csrf_token"]
        assert (
            client.post("/api/team-news", json=report, headers={"x-gaffe-token": token}).status_code
            == 200
        )
        assert client.get("/api/team-news").json()["reports"][0]["cap"] == 30
