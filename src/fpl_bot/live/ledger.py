"""Independent prospective bot, with immutable pre-deadline choices and exact state."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
from collections import Counter

from fpl_bot.live.market_state import selling_price
from fpl_bot.optim.scorer import ScorerInputs, score_gw
from fpl_bot.optim.state import BacktestState, GwDecisions, apply_gw_outcomes
from fpl_bot.web.store import Store, read_json, timestamp, write_json


def encode_state(state: BacktestState) -> dict:
    return {
        "season_id": state.season_id,
        "gameweek": state.gameweek,
        "squad": sorted(state.squad),
        "bank": state.bank,
        "free_transfers": state.free_transfers,
        "chips_used": sorted(state.chips_used),
        "cost_basis": {str(p): v for p, v in state.cost_basis.items()},
    }


def decode_state(value: dict) -> BacktestState:
    return BacktestState(
        season_id=value["season_id"],
        gameweek=value["gameweek"],
        squad=frozenset(value["squad"]),
        bank=value["bank"],
        free_transfers=value["free_transfers"],
        chips_used=frozenset(value["chips_used"]),
        cost_basis={int(p): v for p, v in value["cost_basis"].items()},
    )


def decision(value: dict, gw: int) -> GwDecisions:
    return GwDecisions(
        gameweek=gw,
        squad=frozenset(value["squad"]),
        starting_xi=frozenset(value["starting_xi"]),
        captain=value["captain"],
        vice=value["vice"],
        transferred_in=frozenset(value.get("transferred_in", [])),
        transferred_out=frozenset(value.get("transferred_out", [])),
        chip_played=value.get("chip_played"),
        hits=value.get("hits", 0),
        objective_value=value.get("objective_value", 0),
    )


def digest(value: dict) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


class Ledger:
    def __init__(self, store: Store, team_id: int):
        self.store, self.team_id = store, team_id
        self.path = store.team_dir(team_id) / "bot" / "ledger.json"

    def read(self) -> dict | None:
        data = read_json(self.path)
        if data:
            previous = digest(data["initial_state"])
            for record in data["records"]:
                if not record.get("frozen"):
                    continue
                payload = record["frozen"]
                if payload["previous_hash"] != previous or digest(payload) != record["hash"]:
                    raise ValueError("The bot ledger's saved decision chain failed verification.")
                previous = record["hash"]
        return data

    def save(self, data: dict):
        write_json(self.path, data)

    def seed(self, state: BacktestState, now: dt.datetime, deadline: str):
        if self.read():
            return self.read()
        if now >= timestamp(deadline):
            raise ValueError("A new bot must start before its first deadline.")
        if len(state.squad) != 15 or set(state.cost_basis) != set(state.squad):
            raise ValueError("The bot requires a complete squad and exact purchase prices.")
        data = {
            "version": 1,
            "team_id": self.team_id,
            "season_id": state.season_id,
            "started_at": now.isoformat(),
            "start_gameweek": state.gameweek,
            "initial_state": encode_state(state),
            "state": encode_state(state),
            "records": [],
        }
        self.save(data)
        return data

    def propose(self, rec: dict, bootstrap: dict, now: dt.datetime):
        data = self.read()
        if not data:
            raise ValueError("Start the independent bot before generating its plan.")
        state = decode_state(data["state"])
        gw = rec["context"]["gameweek"]
        event = next(e for e in bootstrap["events"] if e["id"] == gw)
        if (
            gw != state.gameweek
            or now >= timestamp(event["deadline_time"])
            or timestamp(rec["generated_at"]) > now
        ):
            raise ValueError("Bot proposals must use its current state before the deadline.")
        if timestamp(rec["generated_at"]).year < 2000:
            raise ValueError("The forecast must have a valid saved timestamp.")
        before = rec["context"]["state_before"]
        if (
            set(before["squad"]) != set(state.squad)
            or before["bank_tenths"] != state.bank
            or before["free_transfers"] != state.free_transfers
            or set(before["chips_used"]) != set(state.chips_used)
        ):
            raise ValueError("The proposal was generated from another squad state.")
        record = next((r for r in data["records"] if r["gameweek"] == gw), None)
        if record and record.get("frozen"):
            raise ValueError("This bot decision is already frozen.")
        elements = {e["code"]: e for e in bootstrap["elements"]}
        d = decision(rec["decisions"], gw)
        self._validate(d, state, elements)
        proposed = {
            "generated_at": rec["generated_at"],
            "saved_at": now.isoformat(),
            "deadline": event["deadline_time"],
            "state_before": encode_state(state),
            "decisions": rec["decisions"],
            "prices": {str(p): elements[p]["now_cost"] for p in state.squad | d.squad},
            "positions": {
                str(p): {1: "GKP", 2: "DEF", 3: "MID", 4: "FWD"}[elements[p]["element_type"]]
                for p in d.squad
            },
            "element_ids": {str(elements[p]["id"]): p for p in d.squad},
            "source_sha256": digest(rec),
        }
        if not record:
            record = {"gameweek": gw, "status": "draft"}
            data["records"].append(record)
        # Hashed at save time so a catch-up freeze can prove the draft is unchanged.
        record.update(draft=proposed, draft_hash=digest(proposed))
        self.save(data)

    @staticmethod
    def _validate(d: GwDecisions, state: BacktestState, elements: dict):
        if len(d.squad) != 15 or len(d.starting_xi) != 11 or not d.starting_xi <= d.squad:
            raise ValueError("The bot's squad or starting XI is incomplete.")
        if d.captain not in d.starting_xi or d.vice not in d.starting_xi or d.captain == d.vice:
            raise ValueError("The bot's captain and vice must be distinct starters.")
        if not state.squad | d.squad <= set(elements):
            raise ValueError("Market prices are missing for a bot holding.")
        if Counter(elements[p]["element_type"] for p in d.squad) != {1: 2, 2: 5, 3: 5, 4: 3}:
            raise ValueError("The bot squad has invalid positions.")
        xi = Counter(elements[p]["element_type"] for p in d.starting_xi)
        if xi[1] != 1 or xi[2] < 3 or xi[3] < 2 or xi[4] < 1:
            raise ValueError("The bot starting formation is invalid.")
        if max(Counter(elements[p]["team"] for p in d.squad).values()) > 3:
            raise ValueError("The bot squad exceeds the club limit.")
        if d.transferred_in != d.squad - state.squad or d.transferred_out != state.squad - d.squad:
            raise ValueError("The bot transfer accounting does not match its squad.")
        exempt = d.chip_played in {"WC1", "WC2", "FH1", "FH2"}
        if d.hits != (0 if exempt else max(0, len(d.transferred_in) - state.free_transfers)):
            raise ValueError("The bot hit accounting is invalid.")
        if d.chip_played and d.chip_played not in state.chips_available_for_gw(d.gameweek):
            raise ValueError("The bot chip is unavailable.")
        budget = state.bank + sum(
            selling_price(elements[p]["now_cost"], state.cost_basis[p]) for p in d.transferred_out
        )
        if sum(elements[p]["now_cost"] for p in d.transferred_in) > budget:
            raise ValueError("The bot plan is no longer affordable.")

    def freeze(self, now: dt.datetime):
        data = self.read()
        if not data:
            return False
        gw = data["state"]["gameweek"]
        record = next((r for r in data["records"] if r["gameweek"] == gw), None)
        if not record or not record.get("draft"):
            return False
        if now >= timestamp(record["draft"]["deadline"]):
            raise ValueError("A bot decision cannot be frozen after its deadline.")
        self._freeze(data, record, now, late=False)
        self.save(data)
        return True

    @staticmethod
    def _late_freeze_problem(record: dict) -> str | None:
        """Why the last saved draft cannot stand in for a missed final freeze, if anything."""
        draft = record.get("draft")
        if not draft:
            return "No forecast was saved before the deadline (the computer may have been off)."
        if record.get("draft_hash") != digest(draft):
            return "The saved draft could not be verified, so it was not used."
        deadline = timestamp(draft["deadline"])
        if not (
            timestamp(draft["generated_at"]) < deadline and timestamp(draft["saved_at"]) < deadline
        ):
            return "The saved draft was not completed before the deadline."
        return None

    def _freeze(self, data: dict, record: dict, now: dt.datetime, late: bool):
        draft = record["draft"]
        gw = record["gameweek"]
        state = decode_state(data["state"])
        d = decision(draft["decisions"], gw)
        bench = draft["decisions"].get("bench_order", [])
        if len(bench) != 4 or set(bench) != d.squad - d.starting_xi:
            raise ValueError("A saved bench order is required before freezing.")
        previous = next(
            (r["hash"] for r in reversed(data["records"]) if r.get("hash")),
            digest(data["initial_state"]),
        )
        frozen = {**draft, "frozen_at": now.isoformat(), "previous_hash": previous}
        if late:
            frozen["late_freeze"] = True
        prices = {
            int(p): {"buy": v, "sell": selling_price(v, state.cost_basis.get(int(p), v))}
            for p, v in draft["prices"].items()
        }
        after = apply_gw_outcomes(state, d, prices)
        if after.bank < 0:
            raise ValueError("The bot transfer budget cannot be negative.")
        record.update(
            frozen=frozen, hash=digest(frozen), status="frozen", state_after=encode_state(after)
        )
        data["state"] = encode_state(after)

    def reconcile(self, bootstrap: dict, folder, now: dt.datetime):
        data = self.read()
        if not data:
            return
        events = {e["id"]: e for e in bootstrap["events"]}
        # A verified draft saved before the deadline is frozen late. Without one, a missed
        # deadline carries the last squad and accrues FT, but never gains a fabricated score.
        while (event := events.get(data["state"]["gameweek"])) and timestamp(
            event["deadline_time"]
        ) <= now:
            gw = data["state"]["gameweek"]
            record = next((r for r in data["records"] if r["gameweek"] == gw), None)
            if record is None:
                record = {"gameweek": gw}
                data["records"].append(record)
            reason = self._late_freeze_problem(record)
            if reason is None:
                try:
                    self._freeze(data, record, now, late=True)
                    continue
                except ValueError as exc:
                    reason = f"The saved draft could not be frozen: {exc}"
            record.update(status="missed", reason=reason)
            data["state"]["gameweek"] += 1
            data["state"]["free_transfers"] = min(5, data["state"]["free_transfers"] + 1)
        for record in data["records"]:
            gw = record["gameweek"]
            if record.get("status") != "frozen" or not events.get(gw, {}).get("data_checked"):
                continue
            live = read_json(folder / f"live_gw{gw}.json", {})
            if not live:
                continue
            saved = record["frozen"]
            pts, mins = {}, {}
            for element in live.get("elements", []):
                pid = saved["element_ids"].get(str(element["id"]))
                if pid is not None:
                    pts[pid] = element["stats"]["total_points"]
                    mins[pid] = element["stats"]["minutes"]
            d = decision(saved["decisions"], gw)
            if not d.squad <= set(pts):
                continue
            scored = score_gw(
                ScorerInputs(
                    d,
                    pts,
                    mins,
                    {int(p): v for p, v in saved["positions"].items()},
                    {p: 4 - i for i, p in enumerate(saved["decisions"]["bench_order"])},
                )
            )
            record.update(
                status="scored",
                result={
                    "points": scored.gw_points,
                    "captain_final": scored.captain_final,
                    "auto_subs": scored.auto_subs,
                    "scored_at": now.isoformat(),
                },
            )
        self.save(data)

    def summary(self, user_rows: list[dict]) -> dict:
        data = self.read()
        if not data:
            return {"active": False, "rows": [], "covered_weeks": 0}
        users = {r["gameweek"]: r for r in user_rows}
        rows = []
        for record in data["records"]:
            gw = record["gameweek"]
            user = users.get(gw, {})
            bot = record.get("result", {}).get("points")
            user_points = user.get("team_points") if not user.get("provisional", True) else None
            rows.append(
                {
                    "gameweek": gw,
                    "status": record["status"],
                    "bot_points": bot,
                    "team_points": user_points,
                    "lead": user_points - bot
                    if user_points is not None and bot is not None
                    else None,
                    "frozen_at": record.get("frozen", {}).get("frozen_at"),
                    "late_freeze": record.get("frozen", {}).get("late_freeze", False),
                    "reason": record.get("reason"),
                    "hash": record.get("hash"),
                }
            )
        covered = [r for r in rows if r["lead"] is not None]
        latest = data["records"][-1] if data["records"] else {}
        return {
            "active": True,
            "start_gameweek": data["start_gameweek"],
            "started_at": data["started_at"],
            "state": data["state"],
            "rows": rows,
            "covered_weeks": len(covered),
            "team_points": sum(r["team_points"] for r in covered),
            "bot_points": sum(r["bot_points"] for r in covered),
            "lead": sum(r["lead"] for r in covered),
            "missing_weeks": [r["gameweek"] for r in rows if r["status"] == "missed"],
            "latest_decision": latest.get("frozen", latest.get("draft", {})).get("decisions"),
            "latest_status": latest.get("status"),
            "chain_verified": True,
        }
