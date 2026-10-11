"""One isolated worker at a time; durable progress with no raw logs in the API."""

from __future__ import annotations

import datetime as dt
import json
import os
import signal
import subprocess
import sys
import threading
import uuid
from contextlib import suppress

from fpl_bot.web.schemas import JobRequest, Preferences
from fpl_bot.web.store import Store, read_json, write_json

ACTIVE = {"queued", "running"}


class BusyError(ValueError):
    pass


class JobManager:
    def __init__(self, store: Store):
        self.store = store
        self.lock = threading.Lock()
        self.process = None
        self.thread = None
        self.stopping = False
        for job in self.list():
            if job["status"] in ACTIVE:
                job.update(status="failed", message="The app restarted. Run this job again.")
                self._save(job)

    def _save(self, job):
        job["updated_at"] = dt.datetime.now(dt.UTC).isoformat()
        write_json(self.store.web / "jobs" / f"{job['id']}.json", job)

    def list(self) -> list[dict]:
        jobs = [read_json(p) for p in (self.store.web / "jobs").glob("*.json")]
        return sorted((j for j in jobs if j), key=lambda j: j["created_at"], reverse=True)[:20]

    def get(self, job_id: str) -> dict | None:
        try:
            uuid.UUID(job_id)
        except ValueError:
            return None
        return read_json(self.store.web / "jobs" / f"{job_id}.json")

    def start(self, request: JobRequest, prefs: Preferences) -> dict:
        if prefs.team_id is None:
            raise ValueError("Set a team ID in Settings before starting a job.")
        with self.lock:
            if any(job["status"] in ACTIVE for job in self.list()):
                raise BusyError("A job is already running. Wait for it to finish.")
            job = {
                "id": str(uuid.uuid4()),
                "kind": request.kind,
                "team_id": prefs.team_id,
                "status": "queued",
                "progress": 0,
                "message": "Starting…",
                "created_at": dt.datetime.now(dt.UTC).isoformat(),
            }
            self._save(job)
            self.thread = threading.Thread(
                target=self._run, args=(job, request, prefs), daemon=True
            )
            self.thread.start()
            return job.copy()

    def _run(self, job, request, prefs):
        command = [
            sys.executable,
            "-m",
            "fpl_bot.web.worker",
            request.kind,
            "--team-id",
            str(prefs.team_id),
            "--season",
            str(self.store.season),
            "--horizon",
            str(prefs.horizon),
        ]
        if request.refresh_odds:
            command.append("--refresh-odds")
        if request.freeze_bot:
            command.append("--freeze-bot")
        job.update(status="running")
        self._save(job)
        try:
            self.process = subprocess.Popen(
                command,
                cwd=self.store.root,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                bufsize=1,
                start_new_session=True,
            )
            for line in self.process.stdout:
                if not line.startswith("GAFFE_PROGRESS "):
                    continue
                update = json.loads(line.removeprefix("GAFFE_PROGRESS "))
                # Worker emits fixed, safe user messages, never exceptions or credentials.
                job.update(progress=update["progress"], message=update["message"])
                self._save(job)
            code = self.process.wait()
            if code:
                job.update(status="failed")
                if self.stopping:
                    job["message"] = "The app stopped. Run this job again."
                elif job["progress"] != -1:
                    job["message"] = "The worker stopped. Check the database setup and retry."
                job["progress"] = 0
            else:
                job.update(
                    status="completed", progress=100, message="Ready. Your dashboard is up to date."
                )
        except (OSError, ValueError):
            job.update(
                status="failed",
                progress=0,
                message="Could not start the worker. Check the Python installation.",
            )
        finally:
            self.process = None
            self._save(job)

    def close(self):
        self.stopping = True
        if self.process and self.process.poll() is None:
            if os.name == "posix":
                with suppress(ProcessLookupError):
                    os.killpg(self.process.pid, signal.SIGTERM)
            else:
                self.process.terminate()
        if self.thread:
            self.thread.join(timeout=5)
