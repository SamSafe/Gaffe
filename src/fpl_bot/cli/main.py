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
from fpl_bot.ingest import audit as audit_module
from fpl_bot.ingest import footballdata, fpl_api, vaastav

app = typer.Typer(no_args_is_help=True, add_completion=False)
console = Console()

_SUPPORTED_SOURCES = {"fpl", "vaastav", "footballdata"}


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
