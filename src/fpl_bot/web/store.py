"""Small atomic file store, with read-only fallback to existing audited artifacts."""

from __future__ import annotations

import datetime as dt
import json
import os
import tempfile
from pathlib import Path

from fpl_bot.config import settings
from fpl_bot.web.schemas import Preferences


def read_json(path: Path, default=None):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return default


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".gaffe-", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(value, stream, indent=2)
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def timestamp(value: str | None) -> dt.datetime:
    try:
        parsed = dt.datetime.fromisoformat((value or "").replace("Z", "+00:00"))
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=dt.UTC)
    except ValueError:
        return dt.datetime.min.replace(tzinfo=dt.UTC)


class Store:
    def __init__(self, root: Path = Path("."), season: int | None = None):
        self.root = root.resolve()
        self.season = season or settings.current_season_id
        self.web = self.root / "data/web"

    def preferences(self) -> Preferences:
        data = read_json(self.web / "preferences.json", {})
        return Preferences.model_validate({"team_id": settings.team_id, **data})

    def save_preferences(self, value: Preferences) -> None:
        write_json(self.web / "preferences.json", value.model_dump())

    def team_dir(self, team_id: int) -> Path:
        return self.web / f"season_{self.season}" / f"team_{team_id}"

    def snapshot_dir(self, team_id: int | None) -> Path | None:
        if team_id is None:
            return None
        current = self.team_dir(team_id)
        pointer = read_json(current / "current.json", {})
        stamp = pointer.get("snapshot", "")
        if stamp and stamp.isalnum():
            snapshot = current / "snapshots" / stamp
            if read_json(snapshot / "entry.json", {}).get("id") == team_id:
                return snapshot
        if read_json(current / "entry.json", {}).get("id") == team_id:
            return current
        for folder in sorted((self.root / "data/audits").glob("*"), reverse=True):
            if read_json(folder / "entry.json", {}).get("id") != team_id:
                continue
            bootstrap = read_json(folder / "bootstrap.json", {})
            events = bootstrap.get("events", [])
            if events and timestamp(events[0].get("deadline_time")).year == 2000 + self.season:
                return folder
        return None

    def bootstrap(self, snapshot: Path | None) -> dict:
        candidates = list((self.root / "data/raw/fpl_api").glob("*/bootstrap-static.json"))
        if snapshot:
            candidates.append(snapshot / "bootstrap.json")
        candidates = [p for p in candidates if p.exists()]
        for path in sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True):
            data = read_json(path, {})
            events = data.get("events", [])
            if events and timestamp(events[0].get("deadline_time")).year == 2000 + self.season:
                return data
        return {}

    def recommendations(self, team_id: int | None) -> list[tuple[Path, dict]]:
        if team_id is None:
            return []
        base = self.root / f"data/live/recommendations/season_{self.season}"
        paths = list(base.glob("gw_*/recommendation.json"))
        paths += list(base.glob(f"team_{team_id}/gw_*/recommendation.json"))
        paths += list(base.glob(f"team_{team_id}/gw_*/archive/*/recommendation.json"))
        result = []
        for path in paths:
            if "archive" in path.parts and not read_json(path.parent / "complete.json", {}).get(
                "completed"
            ):
                continue
            data = read_json(path, {})
            context = data.get("context", {})
            if context.get("team_id") == team_id and context.get("season_id") == self.season:
                result.append((path, data))
        return sorted(result, key=lambda item: timestamp(item[1].get("generated_at")))
