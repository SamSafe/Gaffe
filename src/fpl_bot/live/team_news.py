"""Deadline-scoped, sourced minutes limits for the prospective news experiment."""

from __future__ import annotations

import datetime as dt
import re
from dataclasses import dataclass, field
from uuid import uuid4

import polars as pl
from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, HttpUrl

from fpl_bot.web.store import read_json, timestamp, write_json


def extract_limit(quote: str) -> int | None:
    limits = re.findall(
        r"\b(?:limited to|no more than|at most|maximum of|capped at)\s+(\d{1,2})\s*(?:minutes?|mins?)\b",
        quote,
        re.I,
    )
    if len(limits) != 1 or re.search(
        r"\b(?:not|isn't|wasn't|last|previously|might|could|if)\b", quote, re.I
    ):
        return None
    cap = int(limits[0])
    return cap if cap < 90 else None


class NewsReport(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    player_id: int = Field(gt=0)
    fixture_id: int = Field(gt=0)
    replacement_id: int | None = Field(default=None, gt=0)
    source: HttpUrl
    published_at: AwareDatetime
    quote: str = Field(min_length=3, max_length=1200)
    cap: int | None = Field(default=None, ge=0, le=89)
    withdraw: bool = False


def record_report(store, report, bootstrap, fixtures, now):
    event = next(
        (e for e in bootstrap.get("events", []) if timestamp(e["deadline_time"]) > now), None
    )
    f = next((f for f in fixtures if f["code"] == report.fixture_id), None)
    if not event or not f or f["event"] != event["id"] or f.get("started"):
        raise ValueError("Select a fixture in the next open gameweek.")
    if report.published_at > now:
        raise ValueError("Publication time must be in the past.")
    clubs = {p["code"]: p["team"] for p in bootstrap["elements"]}
    club = clubs.get(report.player_id)
    if club not in (f["team_h"], f["team_a"]):
        raise ValueError("Choose a player from this fixture.")
    sub = report.replacement_id
    if sub is not None and (sub == report.player_id or clubs.get(sub) != club):
        raise ValueError("The replacement must be a different teammate.")
    cap = report.cap if report.cap is not None else extract_limit(report.quote)
    if cap is None and not report.withdraw:
        raise ValueError("Enter a confirmed minutes limit, or zero for a confirmed absence.")
    row = {
        **report.model_dump(mode="json"),
        "cap": cap,
        "id": uuid4().hex,
        "recorded_at": now.isoformat(),
        "season": store.season,
        "gw": event["id"],
        "deadline": event["deadline_time"],
        "element_id": f["id"],
    }
    write_json(
        store.web / "team_news" / str(store.season) / str(event["id"]) / (row["id"] + ".json"), row
    )
    return row


def active_reports(store, event, bootstrap, fixtures, snapshot, now):
    if not event or now >= timestamp(event["deadline_time"]):
        return []
    current = {
        f["code"]: f for f in fixtures if f.get("event") == event["id"] and not f.get("started")
    }
    clubs = {p["code"]: p["team"] for p in bootstrap.get("elements", [])}
    by_player = {}
    observed = (
        timestamp(read_json(snapshot / "meta.json", {}).get("refreshed_at")) if snapshot else None
    )
    previous = max(
        (timestamp(e["deadline_time"]) for e in bootstrap["events"] if e["id"] < event["id"]),
        default=dt.datetime.min.replace(tzinfo=dt.UTC),
    )
    if observed and previous < observed <= now:
        for p in bootstrap.get("elements", []):
            quote = p.get("news") or ""
            cap = extract_limit(quote)
            if cap is None or not previous < timestamp(p.get("news_added")) <= observed:
                continue
            games = [f for f in current.values() if p["team"] in (f["team_h"], f["team_a"])]
            if len(games) == 1:
                f = games[0]
                by_player[f["code"], p["code"]] = {
                    "player_id": p["code"],
                    "fixture_id": f["code"],
                    "element_id": f["id"],
                    "gw": event["id"],
                    "cap": cap,
                    "replacement_id": None,
                    "published_at": p["news_added"],
                    "recorded_at": observed.isoformat(),
                    "source": "https://fantasy.premierleague.com/api/bootstrap-static/",
                    "quote": quote,
                    "withdraw": False,
                }
    feed = read_json(snapshot / "team_news_feed.json", {}) if snapshot else {}
    for r in sorted(feed.get("rows", []), key=lambda r: timestamp(r["published_at"])):
        if (
            r["gw"] == event["id"]
            and r["cap"] is not None
            and previous < timestamp(r["published_at"]) <= timestamp(r["recorded_at"]) <= now
        ):
            key = (r["fixture_id"], r["player_id"])
            if key not in by_player or timestamp(r["published_at"]) > timestamp(
                by_player[key]["published_at"]
            ):
                by_player[key] = r
    folder = store.web / "team_news" / str(store.season) / str(event["id"])
    saved = [read_json(path) for path in folder.glob("*.json")]
    for row in sorted(saved, key=lambda r: (r["recorded_at"], r["id"])):
        if timestamp(row["recorded_at"]) <= now and timestamp(row["published_at"]) <= now:
            by_player[row["fixture_id"], row["player_id"]] = row
    return [
        r
        for (fid, pid), r in by_player.items()
        if not r["withdraw"]
        and fid in current
        and clubs.get(pid) in (current[fid]["team_h"], current[fid]["team_a"])
        and (not r["replacement_id"] or clubs.get(r["replacement_id"]) == clubs.get(pid))
    ]


@dataclass
class NewsTrial:
    reports: list[dict]
    deadline: dt.datetime | None = None
    rows: list[dict] = field(default_factory=list)
    updates: list[pl.DataFrame] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    shortlist: dict = field(default_factory=dict)
    decisions: dict = field(default_factory=dict)

    def constraints(self, inputs, protected, availability=None):
        rows = [r for r in self.reports if r["fixture_id"] == inputs.fixture_id]
        constraints = []
        used = set()
        players = dict(inputs.players.select("player_id", "team_id").iter_rows())
        for r in sorted(rows, key=lambda r: r["recorded_at"], reverse=True):
            pid, sub = r["player_id"], r["replacement_id"]
            ids = {pid, sub} if sub else {pid}
            if (
                ids & used
                or not ids.issubset(players)
                or (sub and players[pid] != players[sub])
                or any((p, inputs.gameweek) in protected for p in ids)
                or (sub and (availability or {}).get((sub, inputs.gameweek)) == 0)
            ):
                self.skipped.append(
                    f"Fixture {inputs.fixture_id}, player {pid}: conflicting or unavailable input"
                )
                continue
            constraints.append((pid, r["cap"], sub))
            used.update(ids)
        return constraints

    def compare(self, simulator, inputs, baseline, constraints, draws):
        adjusted = simulator.simulate_fixture(
            inputs, minute_caps=constraints, track_minutes=True, replay_draws=draws
        )
        metrics = [
            "e_xpts",
            "p_minutes_zero",
            "p_minutes_short",
            "p_minutes_full",
            "expected_minutes",
        ]
        pair = baseline.select("player_id", *metrics).join(
            adjusted.select("player_id", *metrics), on="player_id", suffix="_news"
        )
        roles = {p: "reported" for p, _, _ in constraints}
        roles.update({sub: "replacement" for _, _, sub in constraints if sub})
        for row in pair.iter_rows(named=True):
            if row["player_id"] in roles:
                self.rows.append(
                    {**row, "fixture_id": inputs.fixture_id, "role": roles[row["player_id"]]}
                )
        self.updates.append(
            pair.select(
                "player_id",
                pl.lit(inputs.fixture_id).alias("fixture_id"),
                pl.col("e_xpts_news").alias("news_xpts"),
                (pl.col("p_minutes_zero") - pl.col("p_minutes_zero_news")).alias(
                    "appearance_delta"
                ),
                pl.col("player_id").is_in(list(roles)).alias("affected"),
            )
        )

    def adjusted_frame(self, baseline):
        return (
            baseline.join(
                pl.concat(self.updates, how="vertical_relaxed"),
                on=["player_id", "fixture_id"],
                how="left",
            )
            .with_columns(
                pl.coalesce("news_xpts", "e_xpts").alias("e_xpts"),
                pl.when(pl.col("affected").fill_null(False))
                .then((pl.col("p_appearance") + pl.col("appearance_delta")).clip(0, 1))
                .otherwise(pl.col("p_appearance"))
                .alias("p_appearance"),
            )
            .drop("news_xpts", "appearance_delta", "affected")
        )
