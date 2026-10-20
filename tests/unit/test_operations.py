"""Prospective decisions, scheduling boundaries and paired remote access."""

from __future__ import annotations

import datetime as dt
from unittest.mock import Mock

import httpx
import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from fpl_bot.live.ledger import Ledger
from fpl_bot.optim.state import BacktestState
from fpl_bot.web.automation import Automation
from fpl_bot.web.phone import COOKIE, PhoneAccess
from fpl_bot.web.schemas import Preferences
from fpl_bot.web.store import Store, read_json, write_json


@pytest.fixture
def scenario(tmp_path):
    now = dt.datetime.now(dt.UTC)
    store = Store(tmp_path, season=26)
    store.save_preferences(Preferences(team_id=718959))
    positions = [1, 1] + [2] * 5 + [3] * 5 + [4] * 3
    elements = [
        dict(id=i, code=i, element_type=pos, team=i % 5, now_cost=50)
        for i, pos in enumerate(positions, 1)
    ]
    elements.append(dict(id=16, code=16, element_type=2, team=3, now_cost=53))
    bootstrap = {
        "events": [
            dict(
                id=gw,
                deadline_time=(now + dt.timedelta(days=gw - 6)).isoformat(),
                data_checked=False,
            )
            for gw in range(7, 10)
        ],
        "elements": elements,
    }
    state = BacktestState(
        26, 7, frozenset(range(1, 16)), 10, 2, frozenset({"WC1"}), {i: 45 for i in range(1, 16)}
    )
    xi = [1, 3, 4, 5, 8, 9, 10, 11, 12, 13, 14]
    rec = dict(
        generated_at=now.isoformat(),
        context=dict(
            gameweek=7,
            state_before=dict(
                squad=sorted(state.squad), bank_tenths=10, free_transfers=2, chips_used=["WC1"]
            ),
        ),
        decisions=dict(
            squad=sorted(state.squad),
            starting_xi=xi,
            captain=13,
            vice=14,
            bench_order=[2, 15, 6, 7],
            transferred_in=[],
            transferred_out=[],
            hits=0,
            chip_played=None,
        ),
    )
    folder = store.team_dir(718959)
    write_json(folder / "bootstrap.json", bootstrap)
    write_json(folder / "entry.json", {"id": 718959})
    ledger = Ledger(store, 718959)
    ledger.seed(state, now, bootstrap["events"][0]["deadline_time"])
    return store, ledger, bootstrap, rec, now, folder


def test_freeze_preserves_own_state_and_odd_profit_rounding(scenario):
    _, ledger, bootstrap, rec, now, _ = scenario
    rec["decisions"]["squad"].remove(3)
    rec["decisions"]["squad"].append(16)
    rec["decisions"]["starting_xi"].remove(3)
    rec["decisions"]["starting_xi"].append(16)
    rec["decisions"].update(transferred_out=[3], transferred_in=[16])
    ledger.propose(rec, bootstrap, now)
    assert ledger.freeze(now)
    assert not ledger.freeze(now)
    state = ledger.read()["state"]
    assert state["bank"] == 4  # Buy 53; sell 45+floor((50-45)/2)=47.
    assert state["free_transfers"] == 2
    assert state["gameweek"] == 8
    assert 16 in state["squad"] and 3 not in state["squad"]
    ledger.seed(BacktestState.cold_start(26), now, bootstrap["events"][0]["deadline_time"])
    assert ledger.read()["state"] == state


def test_chain_rejects_modified_frozen_decision(scenario):
    _, ledger, bootstrap, rec, now, _ = scenario
    ledger.propose(rec, bootstrap, now)
    ledger.freeze(now)
    payload = read_json(ledger.path)
    payload["records"][0]["frozen"]["decisions"]["captain"] = 14
    write_json(ledger.path, payload)
    with pytest.raises(ValueError, match="chain failed"):
        ledger.read()


def test_no_draft_before_deadline_is_missing_and_never_scored(scenario):
    _, ledger, bootstrap, _, now, folder = scenario
    after = now + dt.timedelta(days=1, seconds=1)
    ledger.reconcile(bootstrap, folder, after)
    data = ledger.read()
    assert data["records"][0]["status"] == "missed"
    assert "No forecast" in data["records"][0]["reason"]
    assert data["state"]["free_transfers"] == 3
    assert ledger.summary([])["covered_weeks"] == 0


def test_pre_deadline_draft_freezes_at_catch_up(scenario):
    # The PC slept through the final run: the last draft saved before the deadline counts.
    _, ledger, bootstrap, rec, now, folder = scenario
    ledger.propose(rec, bootstrap, now)
    after = now + dt.timedelta(days=1, seconds=1)
    with pytest.raises(ValueError, match="after its deadline"):
        ledger.freeze(after)
    ledger.reconcile(bootstrap, folder, after)
    data = ledger.read()
    record = data["records"][0]
    assert record["status"] == "frozen"
    assert record["frozen"]["late_freeze"] is True
    assert record["frozen"]["saved_at"] < record["frozen"]["deadline"]
    assert data["state"]["gameweek"] == 8 and data["state"]["free_transfers"] == 3
    bootstrap["events"][0]["data_checked"] = True
    live = {"elements": [dict(id=i, stats=dict(total_points=2, minutes=90)) for i in range(1, 16)]}
    write_json(folder / "live_gw7.json", live)
    ledger.reconcile(bootstrap, folder, after)
    assert ledger.read()["records"][0]["result"]["points"] == 24


def test_edited_draft_is_missed_not_frozen(scenario):
    _, ledger, bootstrap, rec, now, folder = scenario
    ledger.propose(rec, bootstrap, now)
    payload = read_json(ledger.path)
    payload["records"][0]["draft"]["decisions"]["captain"] = 14
    write_json(ledger.path, payload)
    ledger.reconcile(bootstrap, folder, now + dt.timedelta(days=1, seconds=1))
    record = ledger.read()["records"][0]
    assert record["status"] == "missed" and "verified" in record["reason"]


def test_legacy_unhashed_draft_is_missed(scenario):
    _, ledger, bootstrap, rec, now, folder = scenario
    ledger.propose(rec, bootstrap, now)
    payload = read_json(ledger.path)
    del payload["records"][0]["draft_hash"]
    write_json(ledger.path, payload)
    ledger.reconcile(bootstrap, folder, now + dt.timedelta(days=1, seconds=1))
    assert ledger.read()["records"][0]["status"] == "missed"


@pytest.mark.parametrize("chip,expected", [(None, 24), ("TC1", 26), ("BB1", 32), ("FH1", 24)])
def test_final_scoring_and_chip_state(scenario, chip, expected):
    _, ledger, bootstrap, rec, now, folder = scenario
    rec["decisions"]["chip_played"] = chip
    ledger.propose(rec, bootstrap, now)
    ledger.freeze(now)
    live = {"elements": [dict(id=i, stats=dict(total_points=2, minutes=90)) for i in range(1, 16)]}
    write_json(folder / "live_gw7.json", live)
    ledger.reconcile(bootstrap, folder, now + dt.timedelta(days=1))
    assert ledger.read()["records"][0]["status"] == "frozen"
    bootstrap["events"][0]["data_checked"] = True
    ledger.reconcile(bootstrap, folder, now + dt.timedelta(days=1))
    assert ledger.read()["records"][0]["result"]["points"] == expected
    summary = ledger.summary([dict(gameweek=7, team_points=30, provisional=False)])
    assert summary["lead"] == 30 - expected and summary["covered_weeks"] == 1
    assert ledger.read()["state"]["bank"] == 10


def test_incomplete_live_points_remain_unscored(scenario):
    _, ledger, bootstrap, rec, now, folder = scenario
    ledger.propose(rec, bootstrap, now)
    ledger.freeze(now)
    bootstrap["events"][0]["data_checked"] = True
    write_json(folder / "live_gw7.json", {"elements": []})
    ledger.reconcile(bootstrap, folder, now)
    assert ledger.read()["records"][0]["status"] == "frozen"


def test_pairing_is_once_secure_and_revocable(tmp_path):
    phone = PhoneAccess(Store(tmp_path, season=26))
    phone.state = "local"
    phone.addresses = ["127.0.0.1", "10.0.0.10"]
    link = phone.pair_link()
    nonce = link["url"].rsplit("/", 1)[1]
    assert "svg" in link["qr_svg"]
    with TestClient(
        phone.gateway(), base_url="https://testserver", follow_redirects=False
    ) as client:
        assert client.get("/api/dashboard").status_code == 401
        assert client.get("/", headers={"host": "evil.example"}).status_code == 400
        response = client.get("/pair/" + nonce)
        assert response.status_code == 303
        assert all(
            s in response.headers["set-cookie"].lower()
            for s in ["secure", "httponly", "samesite=strict"]
        )
        raw = client.cookies.get(COOKIE)
        assert phone.paired(raw) and not phone.paired(raw, "evil.example")
        assert client.get("/pair/" + nonce).status_code == 401
        assert client.get("/api/access/pair").status_code == 403
        assert client.get("/foo/%2e%2e/api/access").status_code == 400
        assert (
            client.post("/api/jobs", headers={"origin": "https://evil.example"}).status_code == 403
        )
        proxy = Mock()

        async def request(*args, **kwargs):
            assert kwargs["headers"]["x-gaffe-device"] == "paired"
            assert "origin" not in kwargs["headers"]
            return httpx.Response(200, json={"status": "ok"})

        proxy.request = request
        client.app.state.client = proxy
        assert client.get("/health").json() == {"status": "ok"}
        # Restore a client to allow the gateway's lifespan cleanup.
        client.app.state.client = httpx.AsyncClient()
        phone.revoke()
        assert client.get("/api/dashboard").status_code == 401


def test_expired_pairing_link_is_rejected(tmp_path):
    phone = PhoneAccess(Store(tmp_path, season=26))
    import hashlib

    phone.pairings[hashlib.sha256(b"old").hexdigest()] = dt.datetime.now(dt.UTC) - dt.timedelta(
        seconds=1
    )
    assert phone.consume("old") is None


class Jobs:
    def __init__(self):
        self.jobs = []
        self.requests = []

    def list(self):
        return self.jobs

    def get(self, id):
        return next((j for j in self.jobs if j["id"] == id), None)

    def start(self, request, prefs):
        job = {"id": str(len(self.jobs)), "status": "running"}
        self.jobs.append(job)
        self.requests.append(request)
        return job


def test_scheduler_disabled_and_single_job(scenario):
    store, _, _, _, now, _ = scenario
    jobs = Jobs()
    auto = Automation(store, jobs)
    auto.tick(now)
    assert not jobs.jobs
    store.save_preferences(store.preferences().model_copy(update={"automation_enabled": True}))
    auto.tick(now)
    assert jobs.requests[0].kind == "cycle"
    auto.tick(now)
    assert len(jobs.jobs) == 1


def test_scheduler_final_forecast_then_freeze(scenario):
    store, ledger, bootstrap, rec, now, _ = scenario
    ledger.propose(rec, bootstrap, now)
    store.save_preferences(
        store.preferences().model_copy(
            update={"automation_enabled": True, "automation_odds": False}
        )
    )
    jobs = Jobs()
    auto = Automation(store, jobs)
    deadline = dt.datetime.fromisoformat(bootstrap["events"][0]["deadline_time"])
    auto.tick(deadline - dt.timedelta(minutes=60))
    assert jobs.requests[0].kind == "cycle" and jobs.requests[0].freeze_bot
    jobs.jobs[0]["status"] = "completed"
    auto.tick(deadline - dt.timedelta(minutes=10))
    assert jobs.requests[1].kind == "freeze"


@pytest.mark.parametrize("quiet,expected", [(None, []), (("23:00", "08:00"), ["cycle"])])
def test_quiet_hours_move_the_forecast_to_the_evening(scenario, quiet, expected):
    # A 09:00 deadline puts the 07:30 final run inside 23:00-08:00 quiet hours, so the
    # pre-deadline cycle runs from 22:15 the evening before instead of only 6h ahead.
    store, ledger, bootstrap, rec, now, folder = scenario
    ledger.propose(rec, bootstrap, now)
    zone = dt.timezone(dt.timedelta(hours=-6))
    deadline = (
        (now + dt.timedelta(days=2))
        .astimezone(zone)
        .replace(hour=9, minute=0, second=0, microsecond=0)
    )
    bootstrap["events"][0]["deadline_time"] = deadline.isoformat()
    write_json(folder / "bootstrap.json", bootstrap)
    update = {"automation_enabled": True, "automation_odds": False, "timezone": "Etc/GMT+6"}
    if quiet:
        update.update(quiet_start=quiet[0], quiet_end=quiet[1])
    store.save_preferences(store.preferences().model_copy(update=update))
    jobs = Jobs()
    Automation(store, jobs).tick(deadline - dt.timedelta(hours=10))
    assert [r.kind for r in jobs.requests if r.kind == "cycle"] == expected


def test_scheduler_failed_initialization_backs_off(scenario):
    store, _, _, _, now, _ = scenario
    store.save_preferences(store.preferences().model_copy(update={"automation_enabled": True}))
    jobs = Jobs()
    auto = Automation(store, jobs)
    auto.tick(now)
    jobs.jobs[0]["status"] = "failed"
    auto.tick(now + dt.timedelta(seconds=60))
    assert sum(r.kind == "cycle" for r in jobs.requests) == 1
    jobs.jobs[-1]["status"] = "completed"
    auto.tick(now + dt.timedelta(minutes=17))
    assert sum(r.kind == "cycle" for r in jobs.requests) == 2


def test_service_keeps_virtualenv_interpreter(tmp_path, monkeypatch):
    from fpl_bot.web import service_install

    root = tmp_path / "project"
    root.mkdir()
    monkeypatch.setattr(service_install.Path, "home", lambda: tmp_path)
    monkeypatch.setattr(service_install.sys, "executable", str(root / ".venv/bin/python"))
    monkeypatch.setattr(service_install.shutil, "which", lambda _: "/usr/bin/systemctl")
    runner = Mock()
    monkeypatch.setattr(service_install.subprocess, "run", runner)
    unit = service_install.install(root)
    assert str(root / ".venv/bin/python") in unit.read_text()
    assert runner.call_count == 3


def test_partial_markets_use_finite_shared_defaults(monkeypatch):
    import polars as pl

    from fpl_bot.features import bps

    fixtures = pl.DataFrame(
        dict(
            fixture_id=[1, 2],
            season_id=[26, 26],
            gameweek=[1, 1],
            home_team_id=[10, 10],
            away_team_id=[20, 20],
            player_id=[1, 1],
            was_home=[True, True],
        )
    )
    monkeypatch.setattr(bps.pit, "all_player_match_with_kickoff", lambda **kw: fixtures)
    monkeypatch.setattr(
        bps.pit,
        "market_xg_for_fixtures",
        lambda: pl.DataFrame(
            dict(fixture_id=[1], team_id=[10], lambda_market_xg=[1.8], cs_prob_market=[0.4])
        ),
    )
    monkeypatch.setattr(bps.pit, "all_player_positions", pl.DataFrame)
    monkeypatch.setattr(bps, "_resolved_pk_taker_set", lambda *a: set())
    minutes = pl.DataFrame(
        dict(
            player_id=[1, 1],
            fixture_id=[1, 2],
            p_minutes_zero=[0.0, 0.0],
            p_minutes_short=[0.0, 0.0],
            p_minutes_full=[1.0, 1.0],
        )
    )
    goals = pl.DataFrame(dict(player_id=[1, 1], fixture_id=[1, 2], lambda_goals_per_90=[0.2, 0.2]))
    assists = pl.DataFrame(
        dict(player_id=[1, 1], fixture_id=[1, 2], lambda_assists_per_90=[0.1, 0.1])
    )
    prepared = bps.assemble_fold_predictions(
        test_season=26,
        minutes_predictions=minutes,
        goals_predictions=goals,
        assists_predictions=assists,
        alphas_by_position={},
    )
    assert prepared.fixtures["home_team_lambda"].to_list() == [1.8, 1.4]
    assert prepared.fixtures["away_market_cs_prob"].to_list() == [0.25, 0.25]
    baseline = bps.baseline_naive_event_proxy(prepared.player_predictions, prepared.fixtures)
    assert baseline.height == 2
    assert all(
        baseline[c].is_finite().all() for c in ["p_bonus_0", "p_bonus_1", "p_bonus_2", "p_bonus_3"]
    )


def test_preferences_validate_quiet_hours_and_topic():
    assert Preferences(quiet_start="23:00", quiet_end="07:00", notify_topic="").notify_topic is None
    for bad in ({"quiet_start": "25:00"}, {"notify_topic": "short"}, {"notify_topic": "a/b" * 6}):
        with pytest.raises(ValueError):
            Preferences(**bad)


def test_alerts_send_forecast_once_and_each_missed_week(scenario, monkeypatch):
    from fpl_bot.web import notify

    store, ledger, bootstrap, rec, now, folder = scenario
    sent = []
    monkeypatch.setattr(notify, "send", lambda topic, title, body, tags=None: sent.append(title))
    rec["context"].update(team_id=718959, season_id=26)
    rec["decisions"].update(expected_gw_points=55.2)
    path = store.root / "data/live/recommendations/season_26/team_718959/gw_7/recommendation.json"
    write_json(path, rec)
    notify.after_job(store, {"status": "completed"}, "cycle")
    assert sent == []  # No topic, no traffic.
    store.save_preferences(
        store.preferences().model_copy(update={"notify_topic": "gaffe-0123456789ab"})
    )
    notify.after_job(store, {"status": "completed"}, "cycle")
    notify.after_job(store, {"status": "completed"}, "cycle")
    assert sent == ["GW7 forecast ready"]
    ledger.reconcile(bootstrap, folder, now + dt.timedelta(days=1, seconds=1))
    notify.after_job(store, {"status": "completed"}, "refresh")
    notify.after_job(store, {"status": "completed"}, "refresh")
    assert sent == ["GW7 forecast ready", "Bot missed GW7"]
    notify.after_job(store, {"status": "failed"}, "cycle")
    assert sent[-1] == "Scheduled forecast failed"


def test_hold_sentence_flags_gains_inside_solver_tolerance():
    from fpl_bot.web.notify import hold_sentence

    assert "fair call" in hold_sentence({"gain": 2.0, "tolerance": 4.4, "weeks": 6})
    assert "fair call" not in hold_sentence({"gain": 9.0, "tolerance": 4.4, "weeks": 6})


def test_snapshot_retention_keeps_recent_and_pre_deadline(tmp_path):
    from fpl_bot.web.worker import link_or_copy, prune_snapshots

    base = dt.datetime(2026, 10, 1, tzinfo=dt.UTC)
    stamps = [(base + dt.timedelta(hours=h)).strftime("%Y%m%dT%H%M%S%f") for h in range(40)]
    for stamp in stamps:
        write_json(tmp_path / "snapshots" / stamp / "meta.json", {})
    deadline = (base + dt.timedelta(hours=10, minutes=30)).isoformat()
    prune_snapshots(tmp_path, {"events": [{"deadline_time": deadline}]}, keep=5)
    left = sorted(p.name for p in (tmp_path / "snapshots").iterdir())
    assert left == [stamps[10], *stamps[-5:]]
    source = tmp_path / "snapshots" / stamps[-1] / "meta.json"
    link_or_copy(source, tmp_path / "linked.json")
    assert (tmp_path / "linked.json").stat().st_ino == source.stat().st_ino


def test_manual_forecast_drafts_the_bot_plan_without_failing(scenario, monkeypatch):
    from fpl_bot.web import worker

    store, *_ = scenario
    calls, messages = [], []
    monkeypatch.setattr(worker, "progress", lambda value, message: messages.append(message))
    monkeypatch.setattr(worker, "bot", lambda *args: calls.append(args))
    worker.draft_bot_plan(store, 718959, 6)
    assert calls == [(store, 718959, 6)]

    def broken(*args):
        raise RuntimeError("solver missing")

    monkeypatch.setattr(worker, "bot", broken)
    worker.draft_bot_plan(store, 718959, 6)
    assert "not updated" in messages[-1]
    worker.draft_bot_plan(store, 999, 6)  # No ledger for this team: nothing to draft.
    assert len(calls) == 1


def test_goalscorer_odds_failure_never_fails_the_refresh(tmp_path, monkeypatch):
    from fpl_bot.ingest import oddsapi_props
    from fpl_bot.web import worker

    parsed, messages = [], []
    monkeypatch.setattr(worker, "progress", lambda value, message: messages.append(message))
    monkeypatch.setattr(oddsapi_props, "fetch_raw_player_props", lambda: tmp_path / "raw.json")
    monkeypatch.setattr(
        oddsapi_props, "parse_raw_player_props", lambda raw, season_id: parsed.append(season_id)
    )
    worker.collect_player_props(Store(tmp_path, season=26))
    assert parsed == [26]

    def offline():
        raise httpx.ConnectError("offline")

    monkeypatch.setattr(oddsapi_props, "fetch_raw_player_props", offline)
    worker.collect_player_props(Store(tmp_path, season=26))
    assert "unavailable" in messages[-1]


def test_forecast_refreshes_only_stale_odds(tmp_path, monkeypatch):
    from fpl_bot.config import settings
    from fpl_bot.web import worker

    store = Store(tmp_path, season=26)
    now = dt.datetime(2026, 10, 23, 12, tzinfo=dt.UTC)
    calls, messages = [], []
    monkeypatch.setattr(worker, "refresh_bookmaker_odds", lambda s: calls.append(s))
    monkeypatch.setattr(worker, "progress", lambda value, message: messages.append(message))
    monkeypatch.setattr(worker, "latest_odds_recorded_at", lambda: now - dt.timedelta(hours=7))
    monkeypatch.setattr(settings, "odds_api_key", None)
    worker.ensure_fresh_odds(store, now)
    assert calls == []  # No key, no requests.
    monkeypatch.setattr(settings, "odds_api_key", "key")
    worker.ensure_fresh_odds(store, now)
    assert calls == [store]
    monkeypatch.setattr(worker, "latest_odds_recorded_at", lambda: now - dt.timedelta(hours=2))
    worker.ensure_fresh_odds(store, now)
    assert len(calls) == 1  # Fresh enough.

    def offline(s):
        raise httpx.ConnectError("offline")

    monkeypatch.setattr(worker, "latest_odds_recorded_at", lambda: None)
    monkeypatch.setattr(worker, "refresh_bookmaker_odds", offline)
    worker.ensure_fresh_odds(store, now)
    assert "last saved odds" in messages[-1]
