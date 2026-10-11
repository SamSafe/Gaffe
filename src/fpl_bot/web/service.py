"""Dashboard assembly and instant legal what-if planning, without DB access."""

from __future__ import annotations

import datetime as dt
from collections import Counter

from fpl_bot.config import settings
from fpl_bot.live.market_state import selling_price
from fpl_bot.web.schemas import PlanRequest
from fpl_bot.web.store import Store, read_json, timestamp

POSITIONS = {1: "GKP", 2: "DEF", 3: "MID", 4: "FWD"}


def optimal_lineup(squad: list[int], players: dict[int, dict], captain: int | None = None) -> dict:
    groups = {
        pos: sorted(
            (p for p in squad if players[p]["position"] == pos),
            key=lambda p: (-players[p]["xpts"], p),
        )
        for pos in POSITIONS.values()
    }
    best = None
    for defenders in range(3, 6):
        for midfielders in range(2, 6):
            forwards = 10 - defenders - midfielders
            if not 1 <= forwards <= 3:
                continue
            counts = {"GKP": 1, "DEF": defenders, "MID": midfielders, "FWD": forwards}
            if any(len(groups[pos]) < count for pos, count in counts.items()):
                continue
            xi = [p for pos, count in counts.items() for p in groups[pos][:count]]
            if captain is not None and captain not in xi:
                # Include the chosen captain when comparing formations.
                pos = players[captain]["position"]
                selected = [p for p in xi if players[p]["position"] == pos]
                if not selected:
                    continue
                xi.remove(selected[-1])
                xi.append(captain)
            cap = captain or max(xi, key=lambda p: players[p]["xpts"])
            value = sum(players[p]["xpts"] for p in xi) + players[cap]["xpts"]
            if best is None or value > best[0]:
                best = (value, xi, cap)
    if best is None:
        raise ValueError("A valid starting XI could not be formed.")
    points, xi, cap = best
    bench = sorted(
        set(squad) - set(xi), key=lambda p: (players[p]["position"] == "GKP", -players[p]["xpts"])
    )
    vice = max((p for p in xi if p != cap), key=lambda p: players[p]["xpts"])
    return {
        "squad": squad,
        "starting_xi": xi,
        "captain": cap,
        "vice": vice,
        "bench_order": bench,
        "expected_points": round(points, 2),
    }


def _comparison(store, folder, recommendations, bootstrap, players):
    from fpl_bot.optim.scorer import ScorerInputs, score_gw
    from fpl_bot.optim.state import GwDecisions

    events = {e["id"]: e for e in bootstrap.get("events", [])}
    rows = {}
    audit = read_json(folder / "bot_vs_team.json", {}) if folder else {}
    for row in audit.get("rows", []):
        gw = row["gameweek"]
        if (
            events.get(gw, {}).get("data_checked")
            and timestamp(row.get("recommendation_generated_at")).year >= 2000
            and timestamp(row.get("recommendation_generated_at"))
            < timestamp(events[gw]["deadline_time"])
        ):
            rows[gw] = row
    element_ids = {p["element_id"]: pid for pid, p in players.items()}
    for _path, rec in recommendations:
        gw = rec["context"]["gameweek"]
        event = events.get(gw, {})
        d = rec.get("decisions", {})
        if (
            not event.get("data_checked")
            or timestamp(rec.get("generated_at")).year < 2000
            or not d.get("bench_order")
            or timestamp(rec.get("generated_at")) >= timestamp(event.get("deadline_time"))
        ):
            continue
        # Preserve the first saved pre-deadline decision, including its original bench.
        if gw in rows:
            continue
        live = read_json(folder / f"live_gw{gw}.json", {}) if folder else {}
        if not live or any(p not in players for p in d["squad"]):
            continue
        points, minutes = {}, {}
        for element in live.get("elements", []):
            pid = element_ids.get(element["id"])
            if pid is not None:
                points[pid] = element["stats"]["total_points"]
                minutes[pid] = element["stats"]["minutes"]
        decision = GwDecisions(
            gameweek=gw,
            squad=frozenset(d["squad"]),
            starting_xi=frozenset(d["starting_xi"]),
            captain=d["captain"],
            vice=d.get("vice"),
            transferred_in=frozenset(d.get("transferred_in", [])),
            transferred_out=frozenset(d.get("transferred_out", [])),
            chip_played=d.get("chip_played"),
            hits=d.get("hits", 0),
            objective_value=0,
        )
        scored = score_gw(
            ScorerInputs(
                decision,
                points,
                minutes,
                {p: v["position"] for p, v in players.items()},
                {p: float(4 - i) for i, p in enumerate(d["bench_order"])},
            )
        )
        rows[gw] = {
            "gameweek": gw,
            "bot_net_points": scored.gw_points,
            "recommendation_generated_at": rec["generated_at"],
        }
    return rows


def dashboard(store: Store) -> dict:
    prefs = store.preferences()
    folder = store.snapshot_dir(prefs.team_id)
    bootstrap = store.bootstrap(folder)
    entry = read_json(folder / "entry.json", {}) if folder else {}
    history = read_json(folder / "history.json", {}) if folder else {}
    events = bootstrap.get("events", [])
    now = dt.datetime.now(dt.UTC)
    upcoming = next((e for e in events if timestamp(e.get("deadline_time")) > now), None)
    gw = upcoming["id"] if upcoming else None
    recs = store.recommendations(prefs.team_id)
    matches = [(p, r) for p, r in recs if r["context"]["gameweek"] == gw]
    rec_path, rec = matches[-1] if matches else (None, {})
    saved_state = read_json(folder / "state.json", {}) if folder else {}
    if not saved_state and folder and gw:
        payload = read_json(folder / f"current_team_for_gw{gw}.json", {})
        state = read_json(folder / "state_reconstruction.json", {}).get(f"state_for_gw{gw}", {})
        saved_state = {
            **state,
            "picks": payload.get("picks", []),
            "gameweek": gw,
            "source": "Audited public snapshot",
        }
    # A future recommendation records a usable fallback state, but no invented lineup.
    state = (
        saved_state
        if saved_state.get("gameweek") == gw
        else rec.get("context", {}).get("state_before", {})
    )
    forecasts = read_json(rec_path.parent / "player_predictions.json", []) if rec_path else []
    by_player = {}
    for row in forecasts:
        by_player.setdefault(row["player_id"], {})[str(row["gameweek"])] = row["expected_points"]
    teams = {t["id"]: t for t in bootstrap.get("teams", [])}
    prices = {str(p.get("element")): p for p in saved_state.get("picks", [])}
    basis = state.get("cost_basis", {})
    players = {}
    for player in bootstrap.get("elements", []):
        pid = player["code"]
        pick = prices.get(str(player["id"]), {})
        purchase = pick.get("purchase_price", basis.get(str(pid)))
        sale = selling_price(player["now_cost"], purchase) if purchase is not None else None
        if pid in saved_state.get("approximate_prices", []):
            sale = None
        horizon = by_player.get(pid, {})
        team = teams.get(player["team"], {})
        players[pid] = {
            "id": pid,
            "element_id": player["id"],
            "name": player["web_name"],
            "position": POSITIONS[player["element_type"]],
            "club_id": player["team"],
            "club": team.get("short_name", "?"),
            "club_name": team.get("name", ""),
            "price": player["now_cost"],
            "selling_price": sale,
            "status": player.get("status"),
            "news": player.get("news", ""),
            "availability": player.get("chance_of_playing_next_round"),
            "total_points": player.get("total_points", 0),
            "forecast": horizon,
            "xpts": float(horizon.get(str(gw), 0)),
            "forecast_available": str(gw) in horizon,
        }
    picks = sorted(saved_state.get("picks", []), key=lambda p: p["position"])
    current_ids = [p for p in state.get("squad", []) if p in players]
    element_map = {v["element_id"]: p for p, v in players.items()}
    if picks:
        current_ids = [element_map[p["element"]] for p in picks if p["element"] in element_map]
    current = {
        "squad": current_ids,
        "starting_xi": [],
        "bench_order": [],
        "captain": None,
        "vice": None,
    }
    if picks:
        current.update(
            starting_xi=[
                element_map[p["element"]] for p in picks[:11] if p["element"] in element_map
            ],
            bench_order=[
                element_map[p["element"]] for p in picks[11:] if p["element"] in element_map
            ],
            captain=next(
                (element_map.get(p["element"]) for p in picks if p.get("is_captain")), None
            ),
            vice=next(
                (element_map.get(p["element"]) for p in picks if p.get("is_vice_captain")), None
            ),
        )
    state_current = not saved_state or saved_state.get("gameweek") == gw
    hold = (
        optimal_lineup(current_ids, players)
        if state_current
        and len(current_ids) == 15
        and all(players[p]["forecast_available"] for p in current_ids)
        else None
    )
    comparison = _comparison(store, folder, recs, bootstrap, players)
    event_map = {e["id"]: e for e in events}
    performance = []
    for row in history.get("current", []):
        event = event_map.get(row["event"], {})
        bot = comparison.get(row["event"], {}).get("bot_net_points")
        # FPL's weekly points exclude transfer costs; compare net to net.
        provisional = not event.get("data_checked", False)
        live_summary = provisional and entry.get("current_event") == row["event"]
        weekly_points = (
            entry.get("summary_event_points", row["points"]) if live_summary else row["points"]
        )
        total = (
            entry.get("summary_overall_points", row["total_points"])
            if live_summary
            else row["total_points"]
        )
        net = weekly_points - row.get("event_transfers_cost", 0)
        performance.append(
            {
                "gameweek": row["event"],
                "team_points": net,
                "total_points": total,
                "rank": entry.get("summary_overall_rank", row.get("overall_rank"))
                if live_summary
                else row.get("overall_rank"),
                "average": event.get("average_entry_score"),
                "bot_points": bot,
                "lead": net - bot if bot is not None else None,
                "provisional": provisional,
                "points_source": "Live entry summary" if live_summary else "Gameweek history",
                "hits": row.get("event_transfers_cost", 0),
            }
        )
    covered = [r for r in performance if r["bot_points"] is not None]
    meta = read_json(folder / "meta.json", {}) if folder else {}
    if not meta and folder:
        meta = {"refreshed_at": folder.name, "source": "Saved audit"}
    decisions = rec.get("decisions")
    quality = rec.get("solve_meta", {}).get("data_quality", {})
    warnings = []
    if not folder:
        warnings.append("Refresh public data to load this team's squad and history.")
    if folder and not forecasts:
        warnings.append("Generate a forecast for the next deadline to unlock planning.")
    if saved_state and saved_state.get("gameweek") != gw:
        warnings.append("Squad snapshot is from an earlier deadline. Refresh before planning.")
    latest_final = max((e["id"] for e in events if e.get("data_checked")), default=0)
    if rec and quality.get("completed_results_through_gw", 0) < latest_final:
        warnings.append(
            f"This forecast predates finalized GW{latest_final} results. Generate a new forecast."
        )
    if quality.get("gameweeks_with_market_fallback"):
        warnings.append(
            "Later weeks use market fallback: GW "
            + ", ".join(map(str, quality["gameweeks_with_market_fallback"]))
        )
    if saved_state.get("approximate_prices"):
        warnings.append(
            "Some purchase prices are estimated; transfer budgets need checking on FPL."
        )
    return {
        "preferences": prefs.model_dump(),
        "season": store.season,
        "team": {
            "id": prefs.team_id,
            "name": entry.get("name", "Your FPL team"),
            "manager": " ".join(
                filter(None, [entry.get("player_first_name"), entry.get("player_last_name")])
            ),
            "points": entry.get("summary_overall_points"),
            "rank": entry.get("summary_overall_rank"),
            "bank": state.get("bank_tenths"),
            "free_transfers": state.get("free_transfers"),
            "chips_used": state.get("chips_used", []),
            "source": state.get("source", "Saved recommendation"),
        },
        "next_event": {
            "gameweek": gw,
            "deadline": upcoming.get("deadline_time") if upcoming else None,
        },
        "players": list(players.values()),
        "current": current,
        "hold": hold,
        "recommendation": {
            "generated_at": rec.get("generated_at"),
            "decisions": decisions,
            "quality": quality,
            "horizon": rec.get("context", {}).get("horizon_gws", []),
        },
        "performance": {
            "rows": performance,
            "covered_weeks": len(covered),
            "team_compared": sum(r["team_points"] for r in covered),
            "bot_compared": sum(r["bot_points"] for r in covered),
            "lead": sum(r["lead"] for r in covered),
        },
        "freshness": meta,
        "warnings": warnings,
        "connections": {
            "odds": bool(settings.odds_api_key),
            "private_fpl": bool(settings.fpl_access_token or settings.fpl_cookie),
            "snapshot": bool(folder),
            "forecast": bool(forecasts),
        },
    }


def evaluate_plan(store: Store, request: PlanRequest) -> dict:
    data = dashboard(store)
    if not data["hold"]:
        raise ValueError("Refresh the squad and generate a forecast before planning.")
    players = {p["id"]: p for p in data["players"]}
    squad = set(data["current"]["squad"])
    if data["team"]["bank"] is None or data["team"]["free_transfers"] is None:
        raise ValueError("Bank and free transfers are missing. Refresh your team first.")
    outs = [t.out for t in request.transfers]
    ins = [t.incoming for t in request.transfers]
    if len(set(outs)) != len(outs) or len(set(ins)) != len(ins):
        raise ValueError("Each player can appear only once in a transfer plan.")
    if not set(outs) <= squad or set(ins) & squad:
        raise ValueError("Sell players you own and buy players outside your current squad.")
    if any(p not in players for p in ins):
        raise ValueError("An incoming player could not be found.")
    if any(players[p]["selling_price"] is None for p in outs):
        raise ValueError("Exact sale prices are unavailable. Refresh the team before selling.")
    if any(not players[p]["forecast_available"] for p in ins):
        raise ValueError("An incoming player has no forecast. Generate a fresh recommendation.")
    bank = (
        data["team"]["bank"]
        + sum(players[p]["selling_price"] for p in outs)
        - sum(players[p]["price"] for p in ins)
    )
    if bank < 0:
        raise ValueError(f"This plan exceeds your budget by £{-bank / 10:.1f}m.")
    planned = sorted((squad - set(outs)) | set(ins))
    if Counter(players[p]["position"] for p in planned) != {"GKP": 2, "DEF": 5, "MID": 5, "FWD": 3}:
        raise ValueError("Keep 2 goalkeepers, 5 defenders, 5 midfielders and 3 forwards.")
    if max(Counter(players[p]["club_id"] for p in planned).values()) > 3:
        raise ValueError("A squad can contain at most three players from one club.")
    if request.captain is not None and request.captain not in planned:
        raise ValueError("The captain must belong to the planned squad.")
    hits = 4 * max(0, len(ins) - data["team"]["free_transfers"])
    lineup = optimal_lineup(planned, players, request.captain)
    lineup.update(
        hits=hits,
        bank=bank,
        net_points=round(lineup["expected_points"] - hits, 2),
        gain=round(lineup["expected_points"] - hits - data["hold"]["expected_points"], 2),
    )
    return {
        "plan": lineup,
        "hold": data["hold"],
        "note": "One-week expected XI + captain points, after hits. Auto-subs and chips are not simulated in this instant comparison.",
    }
