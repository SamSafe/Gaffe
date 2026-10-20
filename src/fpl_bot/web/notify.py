"""Optional phone alerts through a private ntfy.sh topic. Nothing is sent without one.

Alerts carry player names and the deadline only: no team ID, bank or credentials.
Anyone who knows the topic can read it, so the topic itself is the secret.
"""

from __future__ import annotations

import datetime as dt
from zoneinfo import ZoneInfo

import httpx

from fpl_bot.config import settings
from fpl_bot.web.store import Store, read_json, timestamp, write_json

NTFY = "https://ntfy.sh"


def send(topic: str, title: str, message: str, tags: list[str] | None = None) -> None:
    # JSON publishing keeps accented names intact; ntfy headers are ASCII-only.
    response = httpx.post(
        NTFY,
        json={"topic": topic, "title": title, "message": message, "tags": tags or []},
        headers={"User-Agent": settings.user_agent},
        timeout=10,
    )
    response.raise_for_status()


def forecast_message(store: Store, team_id: int, timezone: str) -> tuple[str, str] | None:
    """Title and body for the newest saved forecast, or None without one."""
    recs = store.recommendations(team_id)
    if not recs:
        return None
    _, rec = recs[-1]
    d, gw = rec["decisions"], rec["context"]["gameweek"]
    bootstrap = store.bootstrap(store.snapshot_dir(team_id))
    names = {e["code"]: e.get("web_name", str(e["code"])) for e in bootstrap.get("elements", [])}
    position = {e["code"]: e.get("element_type") for e in bootstrap.get("elements", [])}
    event = next((e for e in bootstrap.get("events", []) if e["id"] == gw), {})
    name = lambda p: names.get(p, str(p))  # noqa: E731
    # Like-for-like swaps: pair each sale with a purchase in the same position.
    remaining = list(d.get("transferred_in", []))
    pairs = []
    for out in d.get("transferred_out", []):
        same = [p for p in remaining if position.get(p) == position.get(out)]
        incoming = (same or remaining or [None])[0]
        if incoming in remaining:
            remaining.remove(incoming)
        pairs.append((out, incoming))
    moves = "; ".join(f"{name(o)} → {name(i)}" for o, i in pairs) or "Hold (roll the transfer)"
    hits = d.get("hits", 0)
    lines = [
        f"Transfers: {moves}" + (f" (−{4 * hits} hit)" if hits else ""),
        f"Captain {name(d['captain'])}, vice {name(d['vice'])}",
        f"Expected {d.get('expected_gw_points', 0):.1f} pts this week",
    ]
    hold = rec.get("solve_meta", {}).get("hold_comparison")
    if hold:
        lines.append(hold_sentence(hold))
    if event.get("deadline_time"):
        local = timestamp(event["deadline_time"]).astimezone(ZoneInfo(timezone))
        lines.append(f"Deadline {local:%a %d %b %H:%M %Z}")
    return f"GW{gw} forecast ready", "\n".join(lines)


def hold_sentence(hold: dict) -> str:
    gain, tolerance = hold["gain"], hold["tolerance"]
    text = f"Plan vs rolling the transfer: {gain:+.1f} xPts over {hold['weeks']} GWs"
    if abs(gain) <= tolerance:
        return text + f" (within solver tolerance ±{tolerance:.1f}; holding is a fair call)"
    return text


def after_job(store: Store, job: dict, kind: str) -> None:
    """Alert on finished scheduled cycles and newly missed bot weeks. Never raises."""
    prefs = store.preferences()
    if not prefs.notify_topic or not prefs.team_id:
        return
    path = store.web / "notified.json"
    seen = read_json(path, {"forecasts": [], "missed": []})
    try:
        if kind == "cycle" and job["status"] == "failed":
            send(
                prefs.notify_topic,
                "Scheduled forecast failed",
                "Open Gaffe and check the latest job. It retries in 15 minutes.",
                ["warning"],
            )
        elif kind == "cycle":
            message = forecast_message(store, prefs.team_id, prefs.timezone)
            # The two pre-deadline runs often agree; only repeat when the advice changed.
            if message and message[1] not in seen["forecasts"][-1:]:
                send(prefs.notify_topic, *message, ["soccer"])
                seen["forecasts"] = [*seen["forecasts"][-9:], message[1]]
        from fpl_bot.live.ledger import Ledger

        ledger = Ledger(store, prefs.team_id).read() or {"records": []}
        for record in ledger["records"]:
            key = f"{store.season}:{record['gameweek']}"
            if record.get("status") == "missed" and key not in seen["missed"]:
                send(
                    prefs.notify_topic,
                    f"Bot missed GW{record['gameweek']}",
                    record.get("reason", "No decision was saved before the deadline."),
                    ["warning"],
                )
                seen["missed"].append(key)
    except (httpx.HTTPError, ValueError, KeyError):
        return
    finally:
        seen["checked_at"] = dt.datetime.now(dt.UTC).isoformat()
        write_json(path, seen)
