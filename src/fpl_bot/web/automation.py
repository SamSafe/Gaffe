"""Deadline-aware persistent scheduling, sharing the app's single-job boundary."""

from __future__ import annotations

import datetime as dt
import threading

from fpl_bot.live.ledger import Ledger
from fpl_bot.web.jobs import ACTIVE, BusyError
from fpl_bot.web.schemas import JobRequest
from fpl_bot.web.store import Store, read_json, timestamp, write_json


class Automation:
    def __init__(self, store: Store, manager):
        self.store, self.manager = store, manager
        self.stop_event = threading.Event()
        self.thread = None
        self.path = store.web / "automation.json"

    def status(self):
        data = read_json(self.path, {})
        return {
            "enabled": self.store.preferences().automation_enabled,
            "message": data.get(
                "message", "Enable scheduled updates to run the weekly cycle automatically."
            ),
            "last_check": data.get("last_check"),
            "next_run": data.get("next_run"),
            "lead_hours": 6,
            "final_lead_minutes": 90,
            "fallback_freeze_minutes": 15,
            "refresh_hours": 6,
            "state": data.get("state", "idle"),
        }

    def start(self):
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def close(self):
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=3)

    def _loop(self):
        while not self.stop_event.is_set():
            try:
                self.tick(dt.datetime.now(dt.UTC))
            except Exception:
                data = read_json(self.path, {})
                data.update(
                    message="Scheduling paused by a data error. Refresh your team and inspect the latest job.",
                    state="error",
                )
                write_json(self.path, data)
            self.stop_event.wait(60)

    def tick(self, now: dt.datetime):
        prefs = self.store.preferences()
        state = read_json(self.path, {"slots": {}})
        state.setdefault("slots", {})
        state["last_check"] = now.isoformat()
        if not prefs.automation_enabled or not prefs.team_id:
            state.update(state="disabled", message="Scheduled updates are disabled.")
            write_json(self.path, state)
            return
        jobs = self.manager.list()
        if any(j["status"] in ACTIVE for j in jobs):
            state.update(state="running", message="Waiting for the current job to finish.")
            write_json(self.path, state)
            return
        # Commit completed slots; failures retry after 15 minutes without flooding API quota.
        for slot in state["slots"].values():
            if slot.get("status") != "running":
                continue
            job = self.manager.get(slot["job_id"])
            if job and job["status"] not in ACTIVE:
                slot["status"] = job["status"]
                if job["status"] == "failed":
                    slot["retry_at"] = (now + dt.timedelta(minutes=15)).isoformat()
        folder = self.store.snapshot_dir(prefs.team_id)
        bootstrap = self.store.bootstrap(folder)
        events = bootstrap.get("events", [])
        ledger = Ledger(self.store, prefs.team_id).read()
        target = next((e for e in events if timestamp(e["deadline_time"]) > now), None)
        keybase = f"{self.store.season}:{prefs.team_id}"
        candidates = []
        if ledger:
            for record in ledger["records"]:
                gw = record["gameweek"]
                event = next((e for e in events if e["id"] == gw), {})
                deadline = timestamp(event.get("deadline_time"))
                if (
                    record.get("status") == "draft"
                    and event.get("deadline_time")
                    and deadline - dt.timedelta(minutes=15) <= now < deadline
                ):
                    candidates.append((f"freeze:{keybase}:{gw}", JobRequest(kind="freeze")))
                if record.get("status") == "frozen" and event.get("data_checked"):
                    candidates.append((f"settle:{keybase}:{gw}", JobRequest(kind="refresh")))
            current = next((e for e in events if e["id"] == ledger["state"]["gameweek"]), None)
            if current and timestamp(current["deadline_time"]) <= now:
                candidates.append(
                    (f"catchup:{keybase}:{current['id']}", JobRequest(kind="refresh"))
                )
        if (not ledger or not ledger["records"]) and (target or not events):
            candidates.append((f"initialize:{keybase}", JobRequest(kind="cycle")))
        if target:
            deadline = timestamp(target["deadline_time"])
            seconds = (deadline - now).total_seconds()
            if 20 * 60 < seconds <= 90 * 60:
                candidates.append(
                    (
                        f"final:{keybase}:{target['id']}",
                        JobRequest(
                            kind="cycle", refresh_odds=prefs.automation_odds, freeze_bot=True
                        ),
                    )
                )
            elif 90 * 60 < seconds <= 6 * 3600:
                candidates.append(
                    (
                        f"early:{keybase}:{target['id']}",
                        JobRequest(kind="cycle", refresh_odds=prefs.automation_odds),
                    )
                )
            state["next_run"] = (
                deadline - dt.timedelta(hours=6)
                if seconds > 6 * 3600
                else deadline - dt.timedelta(minutes=90)
            ).isoformat()
        # Hourly while results are provisional or a frozen bot week awaits settlement.
        provisional = any(
            not e.get("data_checked") and timestamp(e["deadline_time"]) <= now for e in events
        )
        waiting = bool(ledger and any(r.get("status") == "frozen" for r in ledger["records"]))
        interval = 3600 if provisional or waiting else 6 * 3600
        bucket = int(now.timestamp()) // interval
        refresh_at = dt.datetime.fromtimestamp((bucket + 1) * interval, dt.UTC)
        planned = timestamp(state.get("next_run"))
        state["next_run"] = (
            min(planned, refresh_at).isoformat() if planned > now else refresh_at.isoformat()
        )
        if target or provisional or waiting:
            candidates.append(
                (f"refresh:{keybase}:{interval}:{bucket}", JobRequest(kind="refresh"))
            )
        for key, request in candidates:
            slot = state["slots"].get(key, {})
            if slot.get("status") == "completed" or timestamp(slot.get("retry_at")) > now:
                continue
            if request.refresh_odds:
                from fpl_bot.config import settings

                request.refresh_odds = bool(settings.odds_api_key)
            try:
                job = self.manager.start(request, prefs)
            except BusyError:
                break
            state["slots"][key] = {
                "job_id": job["id"],
                "status": "running",
                "started_at": now.isoformat(),
            }
            state.update(
                state="running",
                message=f"Scheduled {request.kind} started for team {prefs.team_id}.",
            )
            break
        else:
            state.update(
                state="waiting",
                message="Watching results and the next deadline. Latest scheduled work is complete.",
            )
        write_json(self.path, state)
