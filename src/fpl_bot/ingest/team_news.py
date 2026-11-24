"""Read BBC Sport football RSS for attributable, fixture-specific team news."""

from __future__ import annotations

import datetime as dt
import re
import unicodedata
from email.utils import parsedate_to_datetime
from urllib.parse import urlsplit
from xml.etree import ElementTree

import httpx

from fpl_bot.live.team_news import extract_limit
from fpl_bot.web.store import timestamp

FEED = "https://feeds.bbci.co.uk/sport/football/rss.xml"
NEWS_TERMS = re.compile(
    r"\b(injur\w*|ruled out|will miss|limited to|fitness|bench|minutes|sidelined|unavailable)\b",
    re.I,
)


def normalized(text):
    return " ".join(
        "".join(
            c
            for c in unicodedata.normalize("NFKD", text.casefold())
            if not unicodedata.combining(c)
        ).split()
    )


def parse_feed(items, bootstrap, fixtures, event, observed):
    clubs = {t["id"]: normalized(t["name"]) for t in bootstrap["teams"]}
    gw = event["id"]
    start = max(
        (timestamp(e["deadline_time"]) for e in bootstrap["events"] if e["id"] < gw),
        default=observed - dt.timedelta(days=7),
    )
    deadline = timestamp(event["deadline_time"])
    games = [f for f in fixtures if f.get("event") == gw and not f.get("started")]
    rows = []
    for item in items:
        link = (item.findtext("link") or "").strip()
        try:
            url = urlsplit(link)
        except ValueError:
            continue
        if url.scheme != "https" or url.hostname not in {
            "www.bbc.com",
            "www.bbc.co.uk",
            "bbc.com",
            "bbc.co.uk",
        }:
            continue
        try:
            published = parsedate_to_datetime(item.findtext("pubDate") or "")
        except (ValueError, TypeError, OverflowError):
            continue
        if published.tzinfo is None or not start < published <= observed < deadline:
            continue
        for quote in [item.findtext("description") or "", item.findtext("title") or ""]:
            if not NEWS_TERMS.search(quote):
                continue
            text = normalized(quote)
            for p in bootstrap["elements"]:
                full = normalized(f"{p.get('first_name', '')} {p.get('second_name', '')}")
                if len(full.split()) < 2 or not re.search(rf"\b{re.escape(full)}\b", text):
                    continue
                for f in games:
                    if p["team"] not in (f["team_h"], f["team_a"]):
                        continue
                    opponent = clubs[f["team_a"] if p["team"] == f["team_h"] else f["team_h"]]
                    # Auto-apply only a direct claim in the same sentence that
                    # names the player, Premier League competition and opponent.
                    cap = None
                    for sentence in re.split(r"[.!?;]", text):
                        if "premier league" not in sentence or not re.search(
                            rf"\b{re.escape(opponent)}\b", sentence
                        ):
                            continue
                        if (
                            re.search(
                                r"\b(if|could|might|last|previously|denied|denies|whether|reports that|but)\b",
                                sentence,
                            )
                            or "?" in text
                        ):
                            continue
                        match = (
                            r"(?:the |their )?(?:upcoming |next )?premier league "
                            rf"(?:game|match|fixture|clash) (?:against|with|at) {re.escape(opponent)}\b"
                        )
                        subject = rf"\b{re.escape(full)} "
                        if re.search(
                            subject
                            + r"(?:will miss|is ruled out of|has been ruled out of) "
                            + match,
                            sentence,
                        ):
                            cap = 0
                        elif re.search(
                            subject
                            + r"(?:is|will be) limited to \d{1,2} minutes (?:in|for) "
                            + match,
                            sentence,
                        ):
                            cap = extract_limit(sentence)
                    rows.append(
                        {
                            "gw": gw,
                            "fixture_id": f["code"],
                            "element_id": f["id"],
                            "player_id": p["code"],
                            "replacement_id": None,
                            "source": link,
                            "quote": quote[:1200],
                            "cap": cap,
                            "withdraw": False,
                            "published_at": published.isoformat(),
                            "recorded_at": observed.isoformat(),
                        }
                    )
    unique = {}
    for row in rows:
        key = row["source"], row["fixture_id"], row["player_id"]
        if key not in unique or (unique[key]["cap"] is None and row["cap"] is not None):
            unique[key] = row
    return list(unique.values())


def capture_feed(client, bootstrap, fixtures, event, previous=None):
    if not event:
        return {"status": "No open gameweek", "rows": []}
    eligible = {f["code"] for f in fixtures if f.get("event") == event["id"] and not f.get("started")}
    prior = [r for r in (previous or {}).get("rows", []) if r["gw"] == event["id"] and r["fixture_id"] in eligible]
    try:
        response = client.get(FEED, timeout=8)
        response.raise_for_status()
        observed = dt.datetime.now(dt.UTC)
        items = ElementTree.fromstring(response.content).findall("./channel/item")
        refreshed = {(item.findtext("link") or "").strip() for item in items}
        rows = [r for r in prior if r["source"] not in refreshed]
        rows.extend(parse_feed(items, bootstrap, fixtures, event, observed))
        return {"status": "BBC Sport checked", "observed_at": observed.isoformat(), "rows": rows}
    except (httpx.HTTPError, ElementTree.ParseError):
        return {"status": "BBC Sport unavailable at last refresh", "rows": prior}
