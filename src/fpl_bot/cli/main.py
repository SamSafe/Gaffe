"""Phase 1 developer CLI — smoke commands per §8.

Examples:
  fpl-bot ingest fpl --raw-only
  fpl-bot ingest fpl --parse-only --season-id 24
  fpl-bot pit player-status --player-id 1 --as-of 2024-08-16T18:30:00Z
  fpl-bot leakage-check
  fpl-bot ingest-audit --source fpl_api
"""
from __future__ import annotations

import datetime as dt
import subprocess
import sys
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from fpl_bot.config import settings
from fpl_bot.db import pit
from fpl_bot.derive import dixon_coles
from fpl_bot.ingest import audit as audit_module
from fpl_bot.ingest import footballdata, fpl_api, livefpl, oddsapi, understat, vaastav

app = typer.Typer(no_args_is_help=True, add_completion=False)
console = Console()

_SUPPORTED_SOURCES = {"fpl", "vaastav", "footballdata", "understat", "oddsapi", "livefpl"}


@app.command()
def ingest(
    source: Annotated[str, typer.Argument(help="Source: fpl | vaastav (more in 1B)")],
    raw_only: Annotated[bool, typer.Option("--raw-only")] = False,
    parse_only: Annotated[bool, typer.Option("--parse-only")] = False,
    season_id: Annotated[int, typer.Option("--season-id", help="For 'fpl' parse")] = 25,
    season_folder: Annotated[
        str | None,
        typer.Option("--season-folder", help="For 'vaastav' parse, e.g. 2024-25"),
    ] = None,
) -> None:
    """Run an ingestion. Default: fetch then parse. Use --raw-only or --parse-only."""
    if source not in _SUPPORTED_SOURCES:
        console.print(
            f"[red]Source '{source}' not yet implemented. Available: {_SUPPORTED_SOURCES}[/red]"
        )
        raise typer.Exit(1)
    if raw_only and parse_only:
        console.print("[red]--raw-only and --parse-only are mutually exclusive[/red]")
        raise typer.Exit(1)

    if source == "fpl":
        endpoints = ["bootstrap-static", "fixtures"]
        for ep in endpoints:
            if not parse_only:
                console.print(f"[blue]fetch_raw_fpl_api({ep!r})[/blue]")
                path = fpl_api.fetch_raw_fpl_api(ep)
                console.print(f"  → {path}")
            if not raw_only:
                path = fpl_api.latest_raw_for_today(ep)
                if path is None:
                    console.print(
                        f"[yellow]No raw payload found for {ep}; run without --parse-only first.[/yellow]"
                    )
                    continue
                console.print(f"[blue]parse_raw_fpl_api({ep!r}, season_id={season_id})[/blue]")
                counts = fpl_api.parse_raw_fpl_api(ep, path, season_id=season_id)
                for k, v in counts.items():
                    console.print(f"  {k}: {v}")

    elif source == "vaastav":
        if not parse_only:
            console.print("[blue]fetch_raw_vaastav() — git clone/pull[/blue]")
            path = vaastav.fetch_raw_vaastav()
            console.print(f"  → {path}")
        if not raw_only:
            if season_folder is None:
                console.print(
                    "[red]vaastav parse requires --season-folder, e.g. --season-folder 2024-25[/red]"
                )
                raise typer.Exit(1)
            console.print(f"[blue]parse_raw_vaastav_season({season_folder!r})[/blue]")
            counts = vaastav.parse_raw_vaastav_season(season_folder)
            for k, v in counts.items():
                console.print(f"  {k}: {v}")

    elif source == "footballdata":
        if season_folder is None:
            console.print(
                "[red]footballdata requires --season-folder, e.g. --season-folder 2024-25[/red]"
            )
            raise typer.Exit(1)
        if not parse_only:
            console.print(f"[blue]fetch_raw_footballdata({season_folder!r})[/blue]")
            path = footballdata.fetch_raw_footballdata(season_folder)
            console.print(f"  → {path}")
        if not raw_only:
            path = (
                settings.raw_dir / "footballdata"
                / dt.datetime.now(dt.UTC).strftime("%Y-%m-%d")
                / f"E0_{season_folder}.csv"
            )
            if not path.exists():
                console.print(
                    f"[yellow]No raw payload found at {path}; run without --parse-only first.[/yellow]"
                )
                return
            console.print(
                f"[blue]parse_raw_footballdata(season_id={season_id})[/blue]"
            )
            counts = footballdata.parse_raw_footballdata(path, season_id=season_id)
            for k, v in counts.items():
                console.print(f"  {k}: {v}")

    elif source == "livefpl":
        if season_folder is None:
            console.print(
                "[red]livefpl requires --season-folder (current season, e.g. 2025-26)[/red]"
            )
            raise typer.Exit(1)
        season_id_for_parse = int(season_folder.split("-")[0]) - 2000
        if not parse_only:
            console.print("[blue]fetch_raw_livefpl() — scraping LiveFPL /EO[/blue]")
            path = livefpl.fetch_raw_livefpl()
            console.print(f"  → {path}")
        if not raw_only:
            today = dt.datetime.now(dt.UTC).strftime("%Y-%m-%d")
            raw_path = settings.raw_dir / "livefpl" / today / "EO.html"
            if not raw_path.exists():
                console.print(
                    f"[yellow]No raw payload at {raw_path}; run without --parse-only first.[/yellow]"
                )
                raise typer.Exit(1)
            # GW from FPL bootstrap: pick the next-up event. For now require it
            # via CLI later; for the smoke we hardcode discovery via FPL API.
            import json as _json
            import urllib.request as _ur
            with _ur.urlopen(
                "https://fantasy.premierleague.com/api/bootstrap-static/", timeout=20
            ) as r:
                bootstrap = _json.loads(r.read())
            current_gw = next(
                (e["id"] for e in bootstrap["events"] if e.get("is_next") or e.get("is_current")),
                None,
            )
            if current_gw is None:
                console.print("[yellow]Could not infer current GW from bootstrap[/yellow]")
                raise typer.Exit(1)
            console.print(
                f"[blue]parse_raw_livefpl(season_id={season_id_for_parse}, gw={current_gw})[/blue]"
            )
            counts = livefpl.parse_raw_livefpl(
                raw_path, season_id=season_id_for_parse, gameweek=current_gw
            )
            for k, v in counts.items():
                console.print(f"  {k}: {v}")

    elif source == "oddsapi":
        # Live pre-deadline odds via the-odds-api.com. Needs FPL_BOT_ODDS_API_KEY.
        if not parse_only:
            console.print("[blue]fetch_raw_oddsapi() — pulling live EPL odds[/blue]")
            path = oddsapi.fetch_raw_oddsapi()
            console.print(f"  → {path}")
            meta = path.with_suffix(".meta.json")
            if meta.exists():
                import json as _json
                m = _json.loads(meta.read_text())
                console.print(
                    f"  quota: remaining={m.get('remaining_requests')}  "
                    f"used={m.get('used_requests')}"
                )
        if not raw_only:
            if season_folder is None:
                console.print(
                    "[red]oddsapi parse requires --season-folder (current season, e.g. 2025-26)[/red]"
                )
                raise typer.Exit(1)
            season_id_for_parse = int(season_folder.split("-")[0]) - 2000
            today = dt.datetime.now(dt.UTC).strftime("%Y-%m-%d")
            raw_path = (
                settings.raw_dir / "oddsapi" / today / "soccer_epl_h2h+spreads+totals.json"
            )
            if not raw_path.exists():
                console.print(
                    f"[yellow]No raw payload at {raw_path}; run without --parse-only first.[/yellow]"
                )
                raise typer.Exit(1)
            console.print(f"[blue]parse_raw_oddsapi(season_id={season_id_for_parse})[/blue]")
            counts = oddsapi.parse_raw_oddsapi(raw_path, season_id=season_id_for_parse)
            for k, v in counts.items():
                console.print(f"  {k}: {v}")

    elif source == "understat":
        if season_folder is None:
            console.print(
                "[red]understat requires --season-folder, e.g. --season-folder 2024-25[/red]"
            )
            raise typer.Exit(1)
        if not parse_only:
            console.print(
                f"[blue]fetch_raw_understat({season_folder!r}) — vaastav-mirror shim[/blue]"
            )
            path = understat.fetch_raw_understat(season_folder)
            console.print(f"  → {path}")
        if not raw_only:
            console.print(f"[blue]parse_raw_understat_season({season_folder!r})[/blue]")
            counts = understat.parse_raw_understat_season(season_folder)
            for k, v in counts.items():
                console.print(f"  {k}: {v}")


derive_app = typer.Typer(help="Derived computations from raw fact tables.")
app.add_typer(derive_app, name="derive")


@derive_app.command("market-xg")
def derive_market_xg(
    season_id: Annotated[int, typer.Option("--season-id")],
) -> None:
    """Run Dixon-Coles inversion on fact_odds → fact_market_xg for one season."""
    console.print(f"[blue]derive_market_xg_for_season(season_id={season_id})[/blue]")
    counts = dixon_coles.derive_market_xg_for_season(season_id)
    for k, v in counts.items():
        console.print(f"  {k}: {v}")


pit_app = typer.Typer(help="PIT (point-in-time) query smoke commands.")
app.add_typer(pit_app, name="pit")


@pit_app.command("player-status")
def pit_player_status(
    player_id: Annotated[int, typer.Option("--player-id")],
    as_of: Annotated[dt.datetime, typer.Option("--as-of")],
) -> None:
    """Show the latest player_status row with recorded_at <= as_of."""
    if as_of.tzinfo is None:
        as_of = as_of.replace(tzinfo=dt.UTC)
    result = pit.player_status_as_of(player_id, as_of)
    if result is None:
        console.print(f"[yellow]No player_status for player_id={player_id} as of {as_of}[/yellow]")
        raise typer.Exit(2)
    for k, v in result.items():
        console.print(f"  {k}: {v}")


live_app = typer.Typer(help="Phase 6 live-run commands (ingest + recommend).")
app.add_typer(live_app, name="live")


@live_app.command("ingest")
def live_ingest(
    team_id: Annotated[int, typer.Option("--team-id", envvar="FPL_TEAM_ID")],
    gameweek: Annotated[int, typer.Option("--gameweek")],
    season_id: Annotated[int, typer.Option("--season-id")] = 25,
) -> None:
    """Pull bootstrap-static + fixtures + my-team for the user's live state."""
    from fpl_bot.ingest import fpl_api

    console.print("[blue]bootstrap-static[/blue]")
    bs_raw = fpl_api.fetch_raw_fpl_api("bootstrap-static")
    bs_counts = fpl_api.parse_raw_fpl_api("bootstrap-static", bs_raw, season_id=season_id)
    for k, v in bs_counts.items():
        console.print(f"  {k}: {v}")

    console.print("[blue]fixtures[/blue]")
    fx_raw = fpl_api.fetch_raw_fpl_api("fixtures")
    fx_counts = fpl_api.parse_raw_fpl_api("fixtures", fx_raw, season_id=season_id)
    for k, v in fx_counts.items():
        console.print(f"  {k}: {v}")

    console.print(f"[blue]my-team team_id={team_id} gw={gameweek}[/blue]")
    if settings.fpl_cookie:
        console.print("[blue]authenticated current my-team[/blue]")
        mt_raw = fpl_api.fetch_current_my_team(team_id=team_id)
    else:
        console.print("[yellow]FPL_BOT_FPL_COOKIE not set; using public historical picks endpoint[/yellow]")
        mt_raw = fpl_api.fetch_my_team(team_id=team_id, gameweek=gameweek)
    n = fpl_api.parse_my_team(
        mt_raw, season_id=season_id, gameweek=gameweek, team_id=team_id
    )
    console.print(f"  fact_user_team_snapshot: {n}")


@live_app.command("recommend")
def live_recommend(
    team_id: Annotated[int, typer.Option("--team-id", envvar="FPL_TEAM_ID")],
    gameweek: Annotated[int, typer.Option("--gameweek")],
    season_id: Annotated[int, typer.Option("--season-id")] = 25,
    train_seasons: Annotated[
        str, typer.Option("--train-seasons", help="comma-separated")
    ] = "19,20,21,22,23,24",
) -> None:
    """Run Phase 5 stack + status overrides; emit markdown + JSON recommendation."""
    from fpl_bot.live.recommend import generate_recommendation

    ts = [int(s) for s in train_seasons.split(",") if s.strip()]
    md_path, json_path = generate_recommendation(
        season_id=season_id, gameweek=gameweek, team_id=team_id, train_seasons=ts
    )
    console.print(f"[green]wrote {md_path}[/green]")
    console.print(f"[green]wrote {json_path}[/green]")


@live_app.command("retrospective")
def live_retrospective(
    team_id: Annotated[int, typer.Option("--team-id", envvar="FPL_TEAM_ID")],
    gameweek: Annotated[int, typer.Option("--gameweek")],
    season_id: Annotated[int, typer.Option("--season-id")] = 25,
) -> None:
    """Post-GW: pull actuals (assumes bootstrap-static re-ingest after GW close),
    apply auto-sub scorer to the prior recommendation, write actuals.json."""
    from fpl_bot.live.retrospective import compute_retrospective

    p = compute_retrospective(season_id=season_id, gameweek=gameweek, team_id=team_id)
    console.print(f"[green]wrote {p}[/green]")


@app.command("leakage-check")
def leakage_check() -> None:
    """Run the leakage test suite (currently: static-import audit)."""
    r = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/leakage/", "-v", "--no-header"],
        check=False,
    )
    raise typer.Exit(r.returncode)


@app.command("ingest-audit")
def ingest_audit_cmd(
    source: Annotated[str | None, typer.Option("--source")] = None,
    limit: Annotated[int, typer.Option("--limit")] = 25,
) -> None:
    """Show recent ingest audit rows (compliance review)."""
    rows = audit_module.recent_audit(source=source, limit=limit)
    if not rows:
        console.print("[yellow]No audit rows found.[/yellow]")
        return
    table = Table(title=f"ingest_audit (last {len(rows)})")
    for col in ["audit_id", "source", "request_ts", "response_code", "byte_size", "parse_status"]:
        table.add_column(col)
    for r in rows:
        table.add_row(
            str(r["audit_id"]),
            r["source"],
            r["request_ts"].isoformat(),
            str(r["response_code"]),
            str(r["byte_size"]),
            r["parse_status"] or "",
        )
    console.print(table)


if __name__ == "__main__":
    app()
