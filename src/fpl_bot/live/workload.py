"""Prospective comparison of short-rest minutes after non-league matches."""
import datetime as dt
import re
import unicodedata
from collections import defaultdict

import httpx
import polars as pl

from fpl_bot.web.store import read_json, timestamp, write_json

BASE = "https://site.api.espn.com/apis/site/v2/sports/soccer"
COMPETITIONS = ("uefa.champions", "uefa.europa", "uefa.europa.conf", "eng.fa", "eng.league_cup",
                "uefa.nations", "fifa.friendly", "fifa.worldq.uefa", "fifa.worldq.conmebol",
                "fifa.worldq.concacaf", "fifa.worldq.afc", "fifa.worldq.caf", "fifa.worldq.ofc")


def name_key(value):
    return "".join(c for c in unicodedata.normalize("NFKD", value.casefold()) if c.isalnum())


def appearances(summary, bootstrap):
    game = summary["header"]["competitions"][0]
    status = game["status"]["type"]
    if not status.get("completed"):
        return []
    detail = str(status).lower()
    duration = 120 if "extra" in detail or "aet" in detail else 90
    # Shoot-outs can follow either 90 or 120 minutes. Use explicit period evidence.
    if "penalt" in detail and not any(
        e.get("period", {}).get("number") in (3, 4) for e in summary.get("keyEvents", [])
    ):
        return []
    if "penalt" in detail:
        duration = 120
    lookup = defaultdict(set)
    for p in bootstrap["elements"]:
        for name in (p["first_name"] + " " + p["second_name"], p["web_name"]):
            if len(name.split()) >= 2:
                lookup[name_key(name)].add(p["code"])
    rows = []
    for team in summary.get("rosters", []):
        for r in team["roster"]:
            ids = lookup.get(name_key(r["athlete"]["displayName"]), set())
            if len(ids) != 1 or not (r.get("starter") or r.get("subbedIn")):
                continue
            changes = []
            for play in r.get("plays", []):
                if play.get("substitution") or play.get("redCard"):
                    clock = re.match(r"(\d+)", play["clock"]["displayValue"])
                    if clock:
                        changes.append(min(int(clock[1]), duration))
            changes.sort()
            if not r["starter"] and not changes:
                continue
            start = 0 if r["starter"] else changes.pop(0)
            if r.get("subbedOut") and not changes:
                continue
            end = changes[0] if changes else duration
            rows.append({"player_id": next(iter(ids)), "minutes": max(0, end-start)})
    return rows


def capture(store, client, competition, match_id, bootstrap):
    if not re.fullmatch(r"[a-z0-9._-]+", competition) or not str(match_id).isdigit():
        raise ValueError("Invalid competition or match ID")
    path = store.web / "workload" / str(store.season) / f"{competition}-{match_id}.json"
    if path.is_file():
        return
    summary = client.get(f"{BASE}/{competition}/summary", params={"event":match_id}).raise_for_status().json()
    game = summary["header"]["competitions"][0]
    if game["status"]["type"].get("completed"):
        write_json(path, {"competition":competition, "kickoff":game["date"],
                          "recorded_at":dt.datetime.now(dt.UTC).isoformat(),
                          "players":appearances(summary, bootstrap)})


def refresh_workload(store, client, bootstrap, now):
    for league in COMPETITIONS:
        for days_back in range(5):
            date = (now-dt.timedelta(days=days_back)).strftime("%Y%m%d")
            try:
                board = client.get(f"{BASE}/{league}/scoreboard", params={"dates":date}, timeout=8).raise_for_status().json()
                for event in board["events"]:
                    if event["status"]["type"].get("completed"):
                        capture(store, client, league, event["id"], bootstrap)
            except (httpx.HTTPError, ValueError, KeyError):
                continue


def comparison(store, path, gw, deadline):
    cutoff = dt.datetime.now(dt.UTC)
    if cutoff >= deadline or not path.exists():
        return []
    history = defaultdict(list)
    for file in (store.web / "workload" / str(store.season)).glob("*.json"):
        game = read_json(file)
        when = timestamp(game["kickoff"])
        if not deadline-dt.timedelta(days=5) < when < timestamp(game["recorded_at"]) <= cutoff:
            continue
        for p in game["players"]:
            history[p["player_id"]].append((when, p["minutes"]))
    rows = []
    for r in pl.read_parquet(path).filter(pl.col("gameweek")==gw).iter_rows(named=True):
        minutes = sum(m for t,m in history[r["player_id"]] if dt.timedelta(0)<r["kickoff_utc"]-t<=dt.timedelta(hours=96))
        if minutes < 75:
            continue
        # Candidate hypothesis: move 10% of full-appearance probability to a cameo.
        shift = 0.1*r["p_minutes_full"]
        rows.append({**{k:r[k] for k in ("fixture_id","player_id","expected_minutes","p_minutes_zero","p_minutes_short","p_minutes_full","e_xpts")},
            "role":"short rest", "workload_minutes":minutes, "expected_minutes_news":r["expected_minutes"]-45*shift,
            "p_minutes_zero_news":r["p_minutes_zero"],"p_minutes_short_news":r["p_minutes_short"]+shift,
            "p_minutes_full_news":r["p_minutes_full"]-shift})
    return rows
