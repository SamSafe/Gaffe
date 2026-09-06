"""Compare the bot's squad against what the best-performing managers actually own.

LiveFPL's top-10k EO is a rank-band aggregate. This goes narrower and reads the
elite tail directly: the overall league (id 314) standings give the top N entries,
and each entry's `event/<gw>/picks` is public. Aggregating those gives true
ownership and captaincy among managers who are actually winning this season,
which is the cohort the rho-EO term is meant to measure the bot against.

Usage: uv run python scripts/top_manager_holdings.py [--gameweek N] [--top N]
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.request

import sqlalchemy as sa

from fpl_bot.db.models import DimPlayer, DimPlayerSeasonXref
from fpl_bot.db.session import session_scope

UA = {"User-Agent": "fpl-bot/0.1 (research; non-commercial)"}
BASE = "https://fantasy.premierleague.com/api"


def _get(url: str) -> dict:
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gameweek", type=int, required=True)
    ap.add_argument("--top", type=int, default=50)
    ap.add_argument("--season-id", type=int, default=26)
    args = ap.parse_args()

    standings = _get(f"{BASE}/leagues-classic/314/standings/?page_standings=1")
    entries = [
        (e["entry"], e["entry_name"], e["total"])
        for e in standings["standings"]["results"][: args.top]
    ]

    owned: dict[int, int] = {}
    capt: dict[int, int] = {}
    n_ok = 0
    for entry_id, _, _ in entries:
        try:
            picks = _get(f"{BASE}/entry/{entry_id}/event/{args.gameweek}/picks/")
        except Exception:
            continue
        n_ok += 1
        for p in picks["picks"]:
            el = int(p["element"])
            owned[el] = owned.get(el, 0) + 1
            if int(p["multiplier"]) > 1:
                capt[el] = capt.get(el, 0) + 1
        time.sleep(0.4)  # be polite to a free public API

    # element id -> our player_id / name
    with session_scope() as s:
        rows = s.execute(
            sa.select(
                DimPlayerSeasonXref.fpl_element_id,
                DimPlayer.player_id,
                DimPlayer.web_name,
            )
            .join(DimPlayer, DimPlayer.player_id == DimPlayerSeasonXref.player_id)
            .where(DimPlayerSeasonXref.season_id == args.season_id)
        ).all()
    el_to_pid = {int(r.fpl_element_id): int(r.player_id) for r in rows}
    el_to_name = {int(r.fpl_element_id): r.web_name for r in rows}

    print(f"Sampled {n_ok}/{len(entries)} of the overall top {args.top}, GW{args.gameweek}\n")
    print(f"{'player':<20}{'owned':>7}{'capt':>7}{'EO%':>8}")
    ranked = sorted(owned.items(), key=lambda kv: -kv[1])
    for el, n in ranked[:30]:
        eo = 100.0 * (n + capt.get(el, 0)) / max(n_ok, 1)
        print(f"{el_to_name.get(el, el):<20}{100*n/max(n_ok,1):>6.0f}%"
              f"{100*capt.get(el,0)/max(n_ok,1):>6.0f}%{eo:>8.1f}")

    out = {
        "gameweek": args.gameweek,
        "sampled": n_ok,
        "eo_by_player_id": {
            str(el_to_pid[el]): 100.0 * (n + capt.get(el, 0)) / max(n_ok, 1)
            for el, n in owned.items()
            if el in el_to_pid
        },
    }
    path = f"data/live/top{args.top}_holdings_gw{args.gameweek}.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
