"""Local news entry, forecast archives, and scoring after FPL finalizes results."""

from __future__ import annotations

import datetime as dt
from collections import defaultdict

from fpl_bot.live.team_news import active_reports
from fpl_bot.web.store import read_json, timestamp


def match_context(store):
    snapshot = store.snapshot_dir(store.preferences().team_id)
    bootstrap = store.bootstrap(snapshot)
    fixtures = read_json(snapshot / "fixtures.json", []) if snapshot else []
    return snapshot, bootstrap, fixtures


def forecast_reports(store, event, snapshot):
    if not snapshot:
        return []
    return active_reports(
        store,
        event,
        read_json(snapshot / "bootstrap.json", {}),
        read_json(snapshot / "fixtures.json", []),
        snapshot,
        dt.datetime.now(dt.UTC),
    )


def actual_for_fixture(element, fixture_id):
    if not element:
        return None
    for match in element.get("explain", []):
        if match["fixture"] == fixture_id:
            stats = match["stats"]
            minutes = sum(s["value"] for s in stats if s["identifier"] == "minutes")
            return minutes, sum(s["points"] for s in stats)
    stats = element.get("stats", {})
    if stats.get("minutes") == 0 and stats.get("total_points") == 0:
        return 0, 0
    return None


def news_dashboard(store):
    snapshot, bootstrap, fixtures = match_context(store)
    now = dt.datetime.now(dt.UTC)
    events = {e["id"]: e for e in bootstrap.get("events", [])}
    event = next((e for e in events.values() if timestamp(e["deadline_time"]) > now), None)
    teams = {t["id"]: t["short_name"] for t in bootstrap.get("teams", [])}
    games = [
        {
            "id": f["code"],
            "clubs": [f["team_h"], f["team_a"]],
            "name": f"{teams.get(f['team_h'], '?')} vs {teams.get(f['team_a'], '?')}",
        }
        for f in fixtures
        if event and f.get("event") == event["id"] and not f.get("started")
    ]
    reports = forecast_reports(store, event, snapshot) if event else []
    latest = {}
    for path, rec in store.recommendations(store.preferences().team_id):
        if path.parent.parent.name != "archive" or path.parent.name.endswith("-previous"):
            continue
        gw = rec["context"]["gameweek"]
        trial = read_json(path.parent / "team_news_trial.json")
        if not trial or gw not in events:
            continue
        cutoff = min(timestamp(trial["deadline"]), timestamp(events[gw]["deadline_time"]))
        finished = timestamp(trial["finished_at"])
        if finished < cutoff and (
            gw not in latest or finished > timestamp(latest[gw]["finished_at"])
        ):
            latest[gw] = trial

    errors = defaultdict(list)
    history = []
    stable = {p["code"]: p["id"] for p in bootstrap.get("elements", [])}
    final_fixtures = {f["code"]: f for f in fixtures if f.get("finished")}
    for gw, trial in sorted(latest.items(), reverse=True):
        actuals = {}
        if snapshot and events[gw].get("data_checked"):
            live = read_json(snapshot / f"live_gw{gw}.json", {})
            actuals = {e["id"]: e for e in live.get("elements", [])}
        for row in trial["rows"] + trial.get("workload", []):
            f = final_fixtures.get(row["fixture_id"])
            if not f or f["event"] != gw:
                continue
            observed = actual_for_fixture(actuals.get(stable.get(row["player_id"])), f["id"])
            if observed is None:
                continue
            minutes, pts = observed
            bucket = 0 if minutes == 0 else 1 if minutes < 60 else 2
            for arm in ("", "_news"):
                probs = [row["p_minutes_" + k + arm] for k in ("zero", "short", "full")]
                errors[row["role"], arm].append(
                    (
                        sum((p - int(i == bucket)) ** 2 for i, p in enumerate(probs)),
                        abs(row["expected_minutes" + arm] - minutes),
                        abs(row["e_xpts" + arm] - pts) if row["role"] != "short rest" else None,
                    )
                )
        if trial["reports"]:
            entrants = {}
            for pid in trial.get("shortlist", {}).get("entered", []):
                stats = actuals.get(stable.get(pid), {}).get("stats", {})
                entrants[pid] = stats.get("total_points")
            history.append({"gw": gw, **trial, "entrant_points": entrants})
    scores = []
    for (role, arm), values in errors.items():
        scores.append(
            {
                "role": role,
                "arm": "news" if arm else "baseline",
                "n": len(values),
                "brier": sum(v[0] for v in values) / len(values),
                "minutes_mae": sum(v[1] for v in values) / len(values),
                "points_mae": sum(v[2] for v in values) / len(values) if role != "short rest" else None,
            }
        )
    feed = read_json(snapshot / "team_news_feed.json", {}) if snapshot else {}
    inbox = [
        r for r in feed.get("rows", []) if event and r["gw"] == event["id"] and r["cap"] is None
    ]
    return {
        "games": games,
        "reports": reports,
        "history": history,
        "scores": scores,
        "inbox": inbox,
        "feed_status": feed.get("status", "Refresh data to check BBC Sport"),
    }
