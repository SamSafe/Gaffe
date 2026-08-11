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
def position_role_overrides() -> dict[str, dict[str, Any]]:
    """Return role metadata, including stable ``player_id`` where known."""
    raw = _load_yaml("position_role_overrides.yaml")
    return raw.get("players", {}) or {}


def is_role_mismatch(web_name: str) -> bool:
    return web_name in position_role_overrides()


def actual_role(web_name: str) -> str | None:
    info = position_role_overrides().get(web_name)
    return info.get("actual_role") if info else None
