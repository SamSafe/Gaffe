"""Loaders for hand-curated config files (Phase 0 §4.2 + §4.6 pragma).

The configs encode player-level priors that we cannot derive from data we
have free legal access to:
  - Set-piece takers (penalty, direct FK, corner) per team per season
  - Position role overrides for canonical FPL-vs-actual-role mismatches

Both files live under `configs/`. They are read at feature-build time and
joined into the feature pipeline by `fpl_bot.features.goals` (Phase 2.2).

Set-piece identifiers are FPL `web_name`, resolved within season and team at
the call site. Role overrides also carry stable FPL ``player_id`` codes so a
duplicate display name cannot select the wrong player. Resolution stays out
of this module to keep it data-source-free and trivially unit-testable.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

CONFIG_DIR = Path(__file__).parent.parent.parent.parent / "configs"


def _load_yaml(name: str) -> dict[str, Any]:
    path = CONFIG_DIR / name
    if not path.exists():
        return {}
    with path.open() as f:
        loaded = yaml.safe_load(f)
    return loaded or {}


@lru_cache(maxsize=1)
def set_piece_takers_raw() -> dict[int, dict[str, dict[str, Any]]]:
    """Returns {season_id: {team_short_name: {penalty: web_name, direct_fk: [...], corner: [...]}}}.

    Keys are integers (season_id, e.g. 24 for 2024-25) per the yaml structure.
    """
    raw = _load_yaml("set_piece_takers.yaml")
    return {int(k): v for k, v in raw.items()}


def penalty_taker(season_id: int, team_short_name: str) -> str | None:
    """Return the web_name of the primary PK taker, or None if unconfigured."""
    season = set_piece_takers_raw().get(season_id, {})
    team = season.get(team_short_name)
    if not team:
        return None
    return team.get("penalty")


def direct_fk_takers(season_id: int, team_short_name: str) -> list[str]:
    season = set_piece_takers_raw().get(season_id, {})
    return season.get(team_short_name, {}).get("direct_fk", [])


def corner_takers(season_id: int, team_short_name: str) -> list[str]:
    season = set_piece_takers_raw().get(season_id, {})
    return season.get(team_short_name, {}).get("corner", [])


@lru_cache(maxsize=1)
def expected_minutes_overrides_raw() -> dict[int, dict[int, dict[str, Any]]]:
    """Returns {season_id: {gameweek: {web_name or player_id: expected_minutes}}}.

    The human-judgement channel for team news — press conferences, lineup
    leaks, rotation calls — which the minutes model structurally cannot see.
    See docs/design/phase9_minutes.md.
    """
    raw = _load_yaml("expected_minutes_overrides.yaml")
    out: dict[int, dict[int, dict[str, Any]]] = {}
    for season, gw_block in (raw or {}).items():
        if not isinstance(gw_block, dict):
            continue
        season_out: dict[int, dict[str, Any]] = {}
        for gw, players in gw_block.items():
            if isinstance(players, dict):
                season_out[int(gw)] = players
        if season_out:
            out[int(season)] = season_out
    return out


def resolve_expected_minutes(
    season_id: int,
    gameweeks: list[int],
    web_name_to_player_id: dict[str, int],
) -> dict[tuple[int, int], float]:
    """Resolve the override config to {(player_id, gameweek): expected_minutes}.

    Keys may be a numeric FPL `player_id` (unambiguous) or a `web_name`
    (ergonomic). An unresolvable name RAISES rather than being skipped: the
    operator wrote it down because they know something the model does not, and
    silently dropping it would leave them believing the bot had been told.
    """
    season_block = expected_minutes_overrides_raw().get(season_id, {})
    resolved: dict[tuple[int, int], float] = {}
    unresolved: list[str] = []
    for gw in gameweeks:
        for key, minutes in (season_block.get(gw) or {}).items():
            key_str = str(key)
            if key_str.lstrip("-").isdigit():
                player_id = int(key_str)
            else:
                player_id = web_name_to_player_id.get(key_str)  # type: ignore[assignment]
                if player_id is None:
                    unresolved.append(f"{key_str} (season {season_id}, gw {gw})")
                    continue
            try:
                value = float(minutes)
            except (TypeError, ValueError):
                raise ValueError(
                    f"expected_minutes_overrides: non-numeric minutes for "
                    f"{key_str!r} (season {season_id}, gw {gw}): {minutes!r}"
                ) from None
            if not 0.0 <= value <= 90.0:
                raise ValueError(
                    f"expected_minutes_overrides: minutes for {key_str!r} must be "
                    f"between 0 and 90, got {value}"
                )
            resolved[(player_id, gw)] = value
    if unresolved:
        raise ValueError(
            "expected_minutes_overrides: could not resolve these names to a "
            f"player: {unresolved}. Use the exact FPL web_name or a numeric "
            "player_id — an override that does not apply is worse than none."
        )
    return resolved


@lru_cache(maxsize=1)
def position_role_overrides() -> dict[str, dict[str, Any]]:
    """Return role metadata, including stable ``player_id`` where known."""
    raw = _load_yaml("position_role_overrides.yaml")
    return raw.get("players", {}) or {}


def is_role_mismatch(web_name: str) -> bool:
    return web_name in position_role_overrides()


def actual_role(web_name: str) -> str | None:
    info = position_role_overrides().get(web_name)
    return info.get("actual_role") if info else None
