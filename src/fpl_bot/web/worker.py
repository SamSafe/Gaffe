"""Public refresh and model workers, isolated from the interactive server."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import shutil
import sys

import httpx
import yaml
from sqlalchemy.exc import OperationalError

from fpl_bot.config import settings
from fpl_bot.web.store import Store, read_json, timestamp, write_json

FPL = "https://fantasy.premierleague.com/api"


class UserError(ValueError):
    """Only fixed, explicitly safe worker messages can reach the browser."""


def progress(value: int, message: str):
    print("GAFFE_PROGRESS " + json.dumps({"progress": value, "message": message}), flush=True)


def next_event(bootstrap):
    now = dt.datetime.now(dt.UTC)
    return next((e for e in bootstrap["events"] if timestamp(e["deadline_time"]) > now), None)


def reconstruct_state(bootstrap, history, transfers, picks, previous_basis, opening_prices):
    """Public endpoints omit cost basis and FT. Reconstruct and flag unknown costs."""
    from fpl_bot.live.market_state import selling_price

    ft = 1
    chips = {c["event"]: c["name"] for c in history.get("chips", [])}
    for row in sorted(history.get("current", []), key=lambda r: r["event"]):
        if row["event"] == 1:
            ft = 1
        else:
            exempt = chips.get(row["event"]) in {"wildcard", "freehit"}
            ft = min(5, ft + 1 - (0 if exempt else min(ft, row.get("event_transfers", 0))))
    chip_names = {"wildcard": "WC", "freehit": "FH", "bboost": "BB", "3xc": "TC"}
    used = [
        chip_names[c["name"]] + ("1" if c["event"] <= 19 else "2")
        for c in history.get("chips", [])
        if c["name"] in chip_names
    ]
    elements = {e["id"]: e for e in bootstrap["elements"]}
    # Transfers in a Free Hit week are temporary and cannot set permanent cost basis.
    latest_buy = {}
    last_gw = max((row["event"] for row in history.get("current", [])), default=0)
    for transfer in sorted(transfers, key=lambda t: t.get("time", "")):
        if transfer["event"] > last_gw:
            continue
        if chips.get(transfer["event"]) != "freehit":
            latest_buy[transfer["element_in"]] = transfer["element_in_cost"]
    result, basis, approximate = [], {}, []
    for pick in picks:
        element = elements[pick["element"]]
        pid = str(element["code"])
        purchase = latest_buy.get(
            element["id"], previous_basis.get(pid, opening_prices.get(element["id"]))
        )
        if purchase is None:
            purchase = element["now_cost"]
            approximate.append(element["code"])
        sale = selling_price(element["now_cost"], purchase)
        result.append({**pick, "purchase_price": purchase, "selling_price": sale})
        basis[pid] = purchase
    return {
        "picks": result,
        "free_transfers": ft,
        "chips_used": used,
        "cost_basis": basis,
        "approximate_prices": approximate,
        "source": "Public FPL history",
    }


def link_or_copy(source, destination):
    # Snapshot files are only ever replaced atomically, so sharing an inode is safe.
    try:
        destination.hardlink_to(source)
    except OSError:
        shutil.copy2(source, destination)


def prune_snapshots(folder, bootstrap, keep: int = 12):
    """Keep the newest snapshots plus the last one before each deadline, for audits."""
    stamps = sorted(p.name for p in (folder / "snapshots").iterdir() if p.is_dir())
    taken = {s: dt.datetime.strptime(s, "%Y%m%dT%H%M%S%f").replace(tzinfo=dt.UTC) for s in stamps}
    retained = set(stamps[-keep:])
    for event in bootstrap.get("events", []):
        deadline = timestamp(event.get("deadline_time"))
        before = [s for s in stamps if taken[s] < deadline]
        if before:
            retained.add(before[-1])
    for stamp in stamps:
        if stamp not in retained:
            shutil.rmtree(folder / "snapshots" / stamp, ignore_errors=True)


def refresh(store: Store, team_id: int, refresh_odds: bool):
    from fpl_bot.ingest import fpl_api

    folder = store.team_dir(team_id)
    previous = store.snapshot_dir(team_id)
    previous_state = read_json(previous / "state.json", {}) if previous else {}
    basis = previous_state.get("cost_basis", {})
    basis = {
        p: v for p, v in basis.items() if int(p) not in previous_state.get("approximate_prices", [])
    }
    if not basis and previous:
        reconstruction = read_json(previous / "state_reconstruction.json", {})
        states = [v for k, v in reconstruction.items() if k.startswith("state_for_gw")]
        basis = states[-1].get("cost_basis", {}) if states else {}
    progress(5, "Fetching current players and fixtures…")
    with httpx.Client(
        timeout=settings.request_timeout_seconds,
        headers={"User-Agent": settings.user_agent},
        follow_redirects=True,
    ) as client:

        def get(endpoint):
            response = client.get(f"{FPL}/{endpoint}/")
            response.raise_for_status()
            return response.json()

        bootstrap = get("bootstrap-static")
        fixtures = get("fixtures")
        if (
            not bootstrap.get("events")
            or timestamp(bootstrap["events"][0]["deadline_time"]).year != 2000 + store.season
        ):
            raise UserError(
                "The public feed belongs to another season. Update the configured season before refreshing."
            )
        entry, history, transfers = (
            get(f"entry/{team_id}"),
            get(f"entry/{team_id}/history"),
            get(f"entry/{team_id}/transfers"),
        )
        target = next_event(bootstrap)
        latest = max((r["event"] for r in history.get("current", [])), default=0)
        if not latest:
            raise UserError("This entry has no current squad yet.")
        target = target or {"id": 39}
        # A Free Hit snapshot is temporary. Use the preceding permanent squad.
        chip = next((c["name"] for c in history.get("chips", []) if c["event"] == latest), None)
        pick_gw = latest - 1 if chip == "freehit" else latest
        payload = get(f"entry/{team_id}/event/{pick_gw}/picks")
        opening = {}
        progress(15, "Reconstructing bank, free transfers and purchase prices…")
        for pick in payload["picks"]:
            element = next(e for e in bootstrap["elements"] if e["id"] == pick["element"])
            if str(element["code"]) not in basis and not any(
                t["element_in"] == pick["element"] for t in transfers
            ):
                summary = get(f"element-summary/{pick['element']}")
                rows = summary.get("history", [])
                if rows:
                    opening[pick["element"]] = rows[0]["value"]
        state = reconstruct_state(bootstrap, history, transfers, payload["picks"], basis, opening)
        bank_row = next(r for r in history["current"] if r["event"] == pick_gw)
        state.update(gameweek=target["id"], bank_tenths=bank_row["bank"])
        # Publish an entire completed snapshot atomically by pointer after all HTTP requests.
        stamp = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%S%f")
        snapshot = folder / "snapshots" / stamp
        for name, data in [
            ("bootstrap", bootstrap),
            ("fixtures", fixtures),
            ("entry", entry),
            ("history", history),
            ("transfers", transfers),
            ("state", state),
        ]:
            write_json(snapshot / f"{name}.json", data)
        if previous and (previous / "bot_vs_team.json").exists():
            shutil.copy2(previous / "bot_vs_team.json", snapshot / "bot_vs_team.json")
        for event in bootstrap["events"]:
            if not event.get("data_checked"):
                continue
            gw = event["id"]
            old = previous / f"live_gw{gw}.json" if previous else None
            if old and old.exists() and gw < latest - 1:
                link_or_copy(old, snapshot / old.name)
            else:
                write_json(snapshot / f"live_gw{gw}.json", get(f"event/{gw}/live"))
        write_json(
            snapshot / "meta.json",
            {
                "refreshed_at": dt.datetime.now(dt.UTC).isoformat(),
                "source": "Public FPL API",
                "season_id": store.season,
            },
        )
        write_json(folder / "current.json", {"snapshot": stamp})
        prune_snapshots(folder, bootstrap)
    progress(45, "Saving finalized results to the model database…")
    sync_path = store.web / f"model_sync_{store.season}.json"
    sync = read_json(sync_path, {})
    for endpoint, filename in [("bootstrap-static", "bootstrap"), ("fixtures", "fixtures")]:
        fpl_api.parse_raw_fpl_api(endpoint, snapshot / f"{filename}.json", season_id=store.season)
    for event in bootstrap["events"]:
        gw = event["id"]
        gw_fixtures = [f for f in fixtures if f.get("event") == gw]
        counts = {}
        for fixture in gw_fixtures:
            for club in (fixture["team_h"], fixture["team_a"]):
                counts[club] = counts.get(club, 0) + 1
        if (
            event.get("data_checked")
            and gw_fixtures
            and all(f["finished"] for f in gw_fixtures)
            and max(counts.values()) == 1
        ):
            import hashlib

            signature = hashlib.sha256((snapshot / f"live_gw{gw}.json").read_bytes()).hexdigest()
            if sync.get(str(gw)) != signature:
                fpl_api.parse_raw_fpl_event_live_matches(
                    snapshot / f"live_gw{gw}.json", season_id=store.season, gameweek=gw
                )
                sync[str(gw)] = signature
                write_json(sync_path, sync)
    # The live feed omits transfers and ownership; the minutes model trains on them.
    from fpl_bot.ingest.fpl_market import backfill_market_history, missing_market_gameweeks

    if missing_market_gameweeks(store.season):
        progress(55, "Recording last gameweek's transfers and ownership…")
        try:
            backfill_market_history(store.season, [e["id"] for e in bootstrap["elements"]])
        except httpx.HTTPError:
            progress(58, "Transfer history was unavailable. It will be retried next refresh.")
    enriched = {
        "picks": state["picks"],
        "transfers": {"bank": state["bank_tenths"], "free_transfers": state["free_transfers"]},
        "chips": history.get("chips", []),
    }
    write_json(snapshot / "model_team.json", enriched)
    fpl_api.parse_my_team(
        snapshot / "model_team.json", season_id=store.season, gameweek=target["id"], team_id=team_id
    )
    from fpl_bot.live.ledger import Ledger

    Ledger(store, team_id).reconcile(bootstrap, snapshot, dt.datetime.now(dt.UTC))
    if refresh_odds:
        refresh_bookmaker_odds(store)
    progress(
        95,
        "Public snapshot ready. New transfers made after the last deadline may not be visible yet.",
    )


# Bookmaker odds are the model's edge; older ones miss team news and line moves.
ODDS_MAX_AGE = dt.timedelta(hours=6)


def refresh_bookmaker_odds(store: Store):
    """Match odds (~3 API credits), derived team xG, then goalscorer odds."""
    from fpl_bot.derive.dixon_coles import derive_market_xg_for_season
    from fpl_bot.ingest import oddsapi

    progress(80, "Refreshing bookmaker odds…")
    oddsapi.parse_raw_oddsapi(oddsapi.fetch_raw_oddsapi(), season_id=store.season)
    derive_market_xg_for_season(store.season)
    collect_player_props(store)


def latest_odds_recorded_at() -> dt.datetime | None:
    from sqlalchemy import func, select

    from fpl_bot.db.models import FactOdds
    from fpl_bot.db.session import session_scope

    with session_scope() as s:
        return s.scalar(select(func.max(FactOdds.recorded_at)))


def ensure_fresh_odds(store: Store, now: dt.datetime | None = None):
    """Before a forecast, refresh odds older than ODDS_MAX_AGE when a key is set.

    A failed refresh never blocks the forecast; it runs on the last saved odds.
    """
    if not settings.odds_api_key:
        return
    now = now or dt.datetime.now(dt.UTC)
    latest = latest_odds_recorded_at()
    if latest is not None and now - latest < ODDS_MAX_AGE:
        return
    try:
        refresh_bookmaker_odds(store)
    except (httpx.HTTPError, RuntimeError, ValueError):
        progress(
            12, "Bookmaker odds could not be refreshed. The forecast uses the last saved odds."
        )


def collect_player_props(store: Store):
    """Record anytime-goalscorer odds (~10 API credits per gameweek).

    They carry zero weight in forecasts. The history is collected so that
    market-implied availability can later be validated against actual minutes.
    """
    from fpl_bot.ingest import oddsapi_props

    progress(88, "Recording goalscorer odds…")
    try:
        raw = oddsapi_props.fetch_raw_player_props()
        oddsapi_props.parse_raw_player_props(raw, season_id=store.season)
    except Exception:
        progress(90, "Goalscorer odds were unavailable. Match odds are saved.")


def recommend(store: Store, team_id: int, horizon: int):
    from fpl_bot.live.recommend import generate_recommendation

    folder = store.snapshot_dir(team_id)
    bootstrap = store.bootstrap(folder)
    target = next_event(bootstrap) if bootstrap else None
    if not target:
        raise UserError("Refresh public data before generating a recommendation.")
    state = read_json(folder / "state.json", {}) if folder else {}
    if state.get("approximate_prices"):
        raise UserError(
            "Some purchase prices are unknown. Add your exact cost basis using the live runbook before generating a budget-safe recommendation."
        )
    if state.get("gameweek") == target["id"]:
        overrides = (
            yaml.safe_load(settings.live_state_overrides_path.read_text())
            if settings.live_state_overrides_path.exists()
            else {}
        )
        overrides = overrides or {}
        overrides.setdefault(store.season, {}).setdefault(target["id"], {})[team_id] = {
            k: state[k] for k in ("bank_tenths", "free_transfers", "cost_basis", "chips_used")
        }
        overrides_path = store.team_dir(team_id) / "overrides.yaml"
        overrides_path.parent.mkdir(parents=True, exist_ok=True)
        overrides_path.write_text(yaml.safe_dump(overrides))
        settings.live_state_overrides_path = overrides_path
    progress(15, "Training on finalized matches and simulating upcoming fixtures…")
    destination = (
        store.root
        / f"data/live/recommendations/season_{store.season}/team_{team_id}/gw_{target['id']}"
    )
    # Build off to the side; a failed solve must never replace a usable forecast.
    stamp = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%S%f")
    staging = destination / "archive" / stamp
    generate_recommendation(
        season_id=store.season,
        gameweek=target["id"],
        team_id=team_id,
        train_seasons=settings.train_season_ids,
        horizon=horizon,
        n_iterations=1000,
        output_dir=staging,
    )
    write_json(staging / "complete.json", {"completed": True})
    progress(95, "Saving the optimized squad and forecast…")
    # Archive existing artifacts before replacement. Historical primary remains intact.
    if (destination / "recommendation.json").exists():
        old = destination / "archive" / (stamp + "-previous")
        old.mkdir(parents=True, exist_ok=True)
        for file in destination.iterdir():
            if file.is_file():
                shutil.copy2(file, old / file.name)
    for file in staging.iterdir():
        if file.is_file():
            shutil.copy2(file, destination / file.name)


def bot(store: Store, team_id: int, horizon: int, freeze: bool = False):
    from fpl_bot.live.ledger import Ledger, decode_state
    from fpl_bot.live.recommend import generate_recommendation
    from fpl_bot.optim.state import BacktestState

    folder = store.snapshot_dir(team_id)
    bootstrap = store.bootstrap(folder)
    target = next_event(bootstrap) if bootstrap else None
    if not folder or not target:
        raise UserError("Refresh a current squad before starting the bot.")
    ledger = Ledger(store, team_id)
    if not ledger.read():
        snapshot = read_json(folder / "state.json", {})
        if snapshot.get("gameweek") != target["id"] or snapshot.get("approximate_prices"):
            raise UserError("The bot needs a refreshed current squad and exact purchase prices.")
        elements = {e["id"]: e["code"] for e in bootstrap["elements"]}
        state = BacktestState(
            season_id=store.season,
            gameweek=target["id"],
            squad=frozenset(elements[p["element"]] for p in snapshot["picks"]),
            bank=snapshot["bank_tenths"],
            free_transfers=snapshot["free_transfers"],
            chips_used=frozenset(snapshot["chips_used"]),
            cost_basis={int(p): v for p, v in snapshot["cost_basis"].items()},
        )
        ledger.seed(state, dt.datetime.now(dt.UTC), target["deadline_time"])
    ledger.reconcile(bootstrap, folder, dt.datetime.now(dt.UTC))
    state = decode_state(ledger.read()["state"])
    if state.gameweek != target["id"]:
        progress(95, "The current bot decision is frozen. Its next squad is carried independently.")
        return
    progress(20, "Generating the independent bot plan from its own bank, squad and chips…")
    stamp = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%S%f")
    output = store.team_dir(team_id) / "bot" / "forecasts" / f"gw_{state.gameweek}" / stamp
    generate_recommendation(
        season_id=store.season,
        gameweek=state.gameweek,
        team_id=team_id,
        train_seasons=settings.train_season_ids,
        horizon=horizon,
        n_iterations=1000,
        output_dir=output,
        state_override=state,
    )
    rec = read_json(output / "recommendation.json")
    ledger.propose(rec, bootstrap, dt.datetime.now(dt.UTC))
    if freeze:
        ledger.freeze(dt.datetime.now(dt.UTC))
    progress(
        95,
        "Independent bot plan saved"
        + (
            " and frozen before the deadline."
            if freeze
            else ". The last plan saved before the deadline becomes its decision."
        ),
    )


def draft_bot_plan(store: Store, team_id: int, horizon: int):
    """A manual forecast also drafts the independent bot's plan.

    Without the always-on scheduler, this keeps the bot deciding whenever the
    app is opened before a deadline. Its failure never fails your forecast.
    """
    from fpl_bot.live.ledger import Ledger

    try:
        if Ledger(store, team_id).read():
            bot(store, team_id, horizon)
    except Exception:
        progress(95, "Your forecast is ready. The independent bot plan was not updated this time.")


def freeze_bot(store: Store, team_id: int):
    from fpl_bot.live.ledger import Ledger

    folder = store.snapshot_dir(team_id)
    if not folder:
        raise UserError("Refresh the team before settling the bot.")
    ledger = Ledger(store, team_id)
    now = dt.datetime.now(dt.UTC)
    ledger.reconcile(store.bootstrap(folder), folder, now)
    if ledger.freeze(now):
        progress(95, "The bot's saved decision is now frozen before its deadline.")
    else:
        progress(95, "Bot results and deadline state checked.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("kind", choices=["refresh", "recommend", "bot", "cycle", "freeze"])
    parser.add_argument("--team-id", type=int, required=True)
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--horizon", type=int, default=6)
    parser.add_argument("--refresh-odds", action="store_true")
    parser.add_argument("--freeze-bot", action="store_true")
    args = parser.parse_args()
    store = Store(season=args.season)
    try:
        if args.kind == "refresh":
            refresh(store, args.team_id, args.refresh_odds)
        elif args.kind == "recommend":
            ensure_fresh_odds(store)
            recommend(store, args.team_id, args.horizon)
            draft_bot_plan(store, args.team_id, args.horizon)
        elif args.kind == "bot":
            bot(store, args.team_id, args.horizon, args.freeze_bot)
        elif args.kind == "freeze":
            freeze_bot(store, args.team_id)
        else:
            refresh(store, args.team_id, args.refresh_odds)
            recommend(store, args.team_id, args.horizon)
            bot(store, args.team_id, args.horizon, args.freeze_bot)
    except UserError as exc:
        progress(-1, str(exc))
        return 1
    except OperationalError:
        message = (
            "Public data saved, but the model database is unavailable. Check PostgreSQL and database migrations, then refresh again."
            if args.kind == "refresh"
            else "The model database is unavailable. Check PostgreSQL and database migrations, then retry."
        )
        progress(-1, message)
        return 1
    except httpx.HTTPStatusError as exc:
        code = exc.response.status_code
        progress(-1, f"FPL returned HTTP {code}. Check the team ID and try again later.")
        return 1
    except httpx.HTTPError:
        progress(-1, "Could not reach FPL. Check the connection and retry.")
        return 1
    except Exception:
        # No exception text: upstream errors can contain connection URLs and API keys.
        progress(
            -1,
            "The model job could not finish. Check database migrations, historical data and solver installation in the runbook, then retry.",
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
