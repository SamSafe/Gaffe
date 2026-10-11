"""Browser API contracts with real-shaped FPL payloads and no database dependency."""

from __future__ import annotations

import datetime as dt
import json
from unittest.mock import Mock

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from fpl_bot.web.app import create_app
from fpl_bot.web.schemas import PlanRequest, Preferences
from fpl_bot.web.service import dashboard, evaluate_plan
from fpl_bot.web.store import Store, write_json
from fpl_bot.web.worker import reconstruct_state


@pytest.fixture
def populated(tmp_path, monkeypatch):
    from fpl_bot.config import settings

    monkeypatch.setattr(settings, "team_id", 718959)
    now = dt.datetime.now(dt.UTC)
    first = now - dt.timedelta(days=20)
    store = Store(tmp_path, season=first.year - 2000)
    store.save_preferences(Preferences(team_id=718959))
    folder = store.team_dir(718959)
    events = [
        {
            "id": i,
            "deadline_time": (now + dt.timedelta(days=days)).isoformat(),
            "data_checked": i == 1,
            "average_entry_score": 40,
        }
        for i, days in [(1, -20), (2, -10), (3, 5)]
    ]
    positions = [1, 1] + [2] * 5 + [3] * 5 + [4] * 3
    elements = [
        {
            "id": i,
            "code": 1000 + i,
            "web_name": f"Player {i}",
            "element_type": pos,
            "team": i % 5 + 1,
            "now_cost": 50,
            "status": "a",
            "news": "",
            "total_points": i,
        }
        for i, pos in enumerate(positions, 1)
    ]
    elements.extend(
        [
            {
                "id": 16,
                "code": 1016,
                "web_name": "New Defender",
                "element_type": 2,
                "team": 4,
                "now_cost": 55,
                "status": "a",
                "news": "",
                "total_points": 20,
            },
            {
                "id": 17,
                "code": 1017,
                "web_name": "New Midfielder",
                "element_type": 3,
                "team": 5,
                "now_cost": 55,
                "status": "a",
                "news": "",
                "total_points": 20,
            },
        ]
    )
    bootstrap = {
        "events": events,
        "elements": elements,
        "teams": [{"id": i, "name": f"Club {i}", "short_name": f"C{i}"} for i in range(1, 6)],
    }
    picks = [
        {
            "element": i,
            "position": i,
            "purchase_price": 46,
            "selling_price": 48,
            "is_captain": i == 15,
            "is_vice_captain": i == 14,
        }
        for i in range(1, 16)
    ]
    state = {
        "gameweek": 3,
        "picks": picks,
        "bank_tenths": 30,
        "free_transfers": 1,
        "chips_used": [],
        "cost_basis": {str(1000 + i): 46 for i in range(1, 16)},
    }
    for name, payload in [
        ("bootstrap", bootstrap),
        ("entry", {"id": 718959, "name": "Test Team", "summary_overall_points": 85}),
        ("state", state),
        (
            "history",
            {
                "current": [
                    {"event": 1, "points": 50, "total_points": 46, "event_transfers_cost": 4},
                    {"event": 2, "points": 39, "total_points": 85},
                ]
            },
        ),
        (
            "bot_vs_team",
            {
                "rows": [
                    {
                        "gameweek": 1,
                        "bot_net_points": 40,
                        "recommendation_generated_at": (first - dt.timedelta(hours=1)).isoformat(),
                    }
                ]
            },
        ),
    ]:
        write_json(folder / f"{name}.json", payload)
    rec_dir = tmp_path / f"data/live/recommendations/season_{store.season}/gw_3"
    rec = {
        "generated_at": now.isoformat(),
        "context": {"season_id": store.season, "team_id": 718959, "gameweek": 3},
        "decisions": {
            "squad": list(range(1001, 1016)),
            "starting_xi": list(range(1001, 1012)),
            "captain": 1010,
            "vice": 1009,
            "bench_order": list(range(1012, 1016)),
        },
    }
    write_json(rec_dir / "recommendation.json", rec)
    write_json(
        rec_dir / "player_predictions.json",
        [{"player_id": p["code"], "gameweek": 3, "expected_points": p["id"] / 5} for p in elements],
    )
    return store, folder, bootstrap, state, rec_dir


def test_partial_comparison_uses_net_points_and_final_results(populated):
    data = dashboard(populated[0])
    assert data["team"]["points"] == 85
    assert data["performance"]["covered_weeks"] == 1
    assert data["performance"]["lead"] == 6  # 50 minus 4 hit, versus 40.
    assert data["performance"]["rows"][1]["provisional"]
    assert data["performance"]["rows"][1]["bot_points"] is None


def test_post_deadline_plan_is_excluded(populated):
    store, folder, bootstrap, _, _ = populated
    write_json(
        folder / "bot_vs_team.json",
        {
            "rows": [
                {
                    "gameweek": 1,
                    "bot_net_points": 999,
                    "recommendation_generated_at": bootstrap["events"][1]["deadline_time"],
                }
            ]
        },
    )
    assert dashboard(store)["performance"]["covered_weeks"] == 0


def test_live_entry_points_take_precedence_over_lagging_history(populated):
    store, folder, _, _, _ = populated
    write_json(
        folder / "entry.json",
        {
            "id": 718959,
            "current_event": 2,
            "summary_event_points": 54,
            "summary_overall_points": 100,
        },
    )
    row = dashboard(store)["performance"]["rows"][-1]
    assert row["team_points"] == 54
    assert row["total_points"] == 100
    assert row["provisional"]
    assert row["bot_points"] is None


def test_incomplete_forecast_archive_is_never_published(populated):
    store, _, _, _, rec_dir = populated
    rec = json.loads((rec_dir / "recommendation.json").read_text())
    rec["generated_at"] = (dt.datetime.now(dt.UTC) + dt.timedelta(hours=1)).isoformat()
    archive = (
        store.root
        / f"data/live/recommendations/season_{store.season}/team_718959/gw_3/archive/unfinished"
    )
    write_json(archive / "recommendation.json", rec)
    assert len(store.recommendations(718959)) == 1
    write_json(archive / "complete.json", {"completed": True})
    assert len(store.recommendations(718959)) == 2


def test_plan_accounts_for_half_profit_and_captain(populated):
    result = evaluate_plan(
        populated[0], PlanRequest(transfers=[{"out": 1003, "incoming": 1016}], captain=1016)
    )
    assert result["plan"]["bank"] == 30 + 48 - 55
    assert result["plan"]["captain"] == 1016
    assert result["plan"]["hits"] == 0
    assert len(result["plan"]["starting_xi"]) == 11
    assert len(result["plan"]["bench_order"]) == 4


def test_multiple_transfers_charge_four_point_hits(populated):
    result = evaluate_plan(
        populated[0],
        PlanRequest(transfers=[{"out": 1003, "incoming": 1016}, {"out": 1009, "incoming": 1017}]),
    )
    assert result["plan"]["hits"] == 4
    assert result["plan"]["net_points"] == round(result["plan"]["expected_points"] - 4, 2)


@pytest.mark.parametrize(
    "transfers,message",
    [
        ([{"out": 1016, "incoming": 1003}], "Sell players you own"),
        ([{"out": 1003, "incoming": 1016}, {"out": 1003, "incoming": 1017}], "only once"),
        ([{"out": 1001, "incoming": 1016}], "2 goalkeepers"),
        ([{"out": 1004, "incoming": 1016}], "three players"),
        ([{"out": 1003, "incoming": 999999}], "could not be found"),
    ],
)
def test_invalid_plans_rejected(populated, transfers, message):
    with pytest.raises(ValueError, match=message):
        evaluate_plan(populated[0], PlanRequest(transfers=transfers))


def test_insufficient_budget_rejected(populated):
    store, folder, _, state, _ = populated
    state["bank_tenths"] = 0
    write_json(folder / "state.json", state)
    with pytest.raises(ValueError, match="exceeds your budget"):
        evaluate_plan(store, PlanRequest(transfers=[{"out": 1003, "incoming": 1016}]))


def test_estimated_sale_price_cannot_authorize_transfer_budget(populated):
    store, folder, _, state, _ = populated
    state["approximate_prices"] = [1003]
    write_json(folder / "state.json", state)
    with pytest.raises(ValueError, match="Exact sale prices are unavailable"):
        evaluate_plan(store, PlanRequest(transfers=[{"out": 1003, "incoming": 1016}]))


def test_missing_forecast_does_not_become_zero_point_plan(populated):
    store, _, _, _, rec_dir = populated
    write_json(rec_dir / "player_predictions.json", [])
    assert dashboard(store)["hold"] is None
    with pytest.raises(ValueError, match="generate a forecast"):
        evaluate_plan(store, PlanRequest())


def test_stale_squad_cannot_be_planned(populated):
    store, folder, _, state, _ = populated
    state["gameweek"] = 2
    write_json(folder / "state.json", state)
    assert dashboard(store)["hold"] is None


def test_team_switch_never_uses_other_teams_snapshot_or_forecast(populated):
    store = populated[0]
    store.save_preferences(Preferences(team_id=42))
    data = dashboard(store)
    assert data["team"]["id"] == 42
    assert data["team"]["points"] is None
    assert data["current"]["squad"] == []
    assert data["recommendation"]["decisions"] is None


def test_snapshot_pointer_reads_only_completed_matching_team(populated):
    store, folder, bootstrap, _, _ = populated
    snapshot = folder / "snapshots" / "2026T1234"
    write_json(snapshot / "entry.json", {"id": 718959})
    write_json(snapshot / "bootstrap.json", bootstrap)
    write_json(folder / "current.json", {"snapshot": "2026T1234"})
    assert store.snapshot_dir(718959) == snapshot
    write_json(folder / "current.json", {"snapshot": "../../other"})
    assert store.snapshot_dir(718959) == folder


def test_api_mutations_require_same_origin_token(populated):
    with TestClient(create_app(populated[0])) as client:
        assert client.post("/api/plan", json={}).status_code == 403
        response = client.get("/api/dashboard")
        token = response.json()["csrf_token"]
        assert response.headers["cache-control"] == "no-store"
        headers = {"X-Gaffe-Token": token}
        assert client.post("/api/plan", headers=headers, json={}).status_code == 200
        assert (
            client.post(
                "/api/plan", headers={**headers, "Origin": "https://evil.example"}, json={}
            ).status_code
            == 403
        )
        assert client.get("/api/dashboard", headers={"host": "evil.example"}).status_code == 400
        assert client.get("/api/jobs/../../etc/passwd").status_code == 404
        assert client.get("/assets/../store.py").status_code == 404


def test_browser_payload_never_contains_credentials(populated, monkeypatch):
    from fpl_bot.config import settings

    monkeypatch.setattr(settings, "odds_api_key", "TEST_SECRET_ODDS")
    monkeypatch.setattr(settings, "fpl_access_token", "TEST_SECRET_TOKEN")
    monkeypatch.setattr(
        settings, "database_url", "postgresql://name:TEST_SECRET_PASSWORD@localhost/db"
    )
    with TestClient(create_app(populated[0])) as client:
        content = client.get("/api/dashboard").text
        assert "TEST_SECRET" not in content
        assert client.get("/").status_code == 200
        assert client.get("/assets/app.js").status_code == 200


def test_preferences_validation_and_busy_guard(populated):
    with TestClient(create_app(populated[0])) as client:
        token = client.get("/api/dashboard").json()["csrf_token"]
        headers = {"X-Gaffe-Token": token}
        assert (
            client.put(
                "/api/preferences", headers=headers, json={"timezone": "Bad/Zone"}
            ).status_code
            == 422
        )
        assert (
            client.put("/api/preferences", headers=headers, json={"team_id": -1}).status_code == 422
        )
        assert (
            client.put(
                "/api/preferences", headers=headers, json={"command": "arbitrary"}
            ).status_code
            == 422
        )
        client.app.state.jobs.list = Mock(return_value=[{"status": "running"}])
        assert (
            client.put("/api/preferences", headers=headers, json={"team_id": 42}).status_code == 409
        )


def test_public_state_preserves_costs_and_free_transfers_through_wildcard(populated):
    _, _, bootstrap, state, _ = populated
    history = {
        "current": [
            {"event": 1, "event_transfers": 15},
            {"event": 2, "event_transfers": 0},
            {"event": 3, "event_transfers": 1},
            {"event": 4, "event_transfers": 2},
            {"event": 5, "event_transfers": 15},
            {"event": 6, "event_transfers": 1},
        ],
        "chips": [{"event": 1, "name": "bboost"}, {"event": 5, "name": "wildcard"}],
    }
    transfers = [
        {"event": 6, "element_in": 3, "element_in_cost": 47, "time": "a"},
        {"event": 7, "element_in": 3, "element_in_cost": 99, "time": "b"},
    ]
    rebuilt = reconstruct_state(
        bootstrap, history, transfers, state["picks"], state["cost_basis"], {}
    )
    assert rebuilt["free_transfers"] == 2
    assert rebuilt["chips_used"] == ["BB1", "WC1"]
    assert rebuilt["cost_basis"]["1003"] == 47
    assert rebuilt["picks"][2]["selling_price"] == 48
    assert not rebuilt["approximate_prices"]


def test_unknown_purchase_price_is_flagged(populated):
    _, _, bootstrap, state, _ = populated
    result = reconstruct_state(bootstrap, {"current": [{"event": 1}]}, [], state["picks"], {}, {})
    assert len(result["approximate_prices"]) == 15


def test_job_restart_marks_unfinished_job_failed(tmp_path):
    from fpl_bot.web.jobs import JobManager

    store = Store(tmp_path)
    write_json(
        store.web / "jobs" / "old.json",
        {"id": "old", "status": "running", "created_at": "2026-01-01"},
    )
    manager = JobManager(store)
    assert manager.list()[0]["status"] == "failed"
    assert "restarted" in manager.list()[0]["message"]


def test_background_job_rejects_overlap_and_hides_stderr(tmp_path, monkeypatch):
    import threading

    from fpl_bot.web.jobs import BusyError, JobManager
    from fpl_bot.web.schemas import JobRequest

    release = threading.Event()
    process = Mock()

    def output():
        yield "API_KEY=SHOULD_NEVER_REACH_THE_BROWSER\n"
        yield 'GAFFE_PROGRESS {"progress": 10, "message": "Safe progress"}\n'
        release.wait(5)

    process.stdout = output()
    process.wait.return_value = 0
    monkeypatch.setattr("fpl_bot.web.jobs.subprocess.Popen", Mock(return_value=process))
    manager = JobManager(Store(tmp_path))
    job = manager.start(JobRequest(kind="refresh"), Preferences(team_id=718959))
    with pytest.raises(BusyError):
        manager.start(JobRequest(kind="recommend"), Preferences(team_id=718959))
    release.set()
    manager.thread.join(5)
    result = manager.get(job["id"])
    assert result["status"] == "completed"
    assert "SHOULD_NEVER" not in json.dumps(manager.list())
