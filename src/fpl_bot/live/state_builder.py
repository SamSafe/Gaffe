"""Phase 6: build a BacktestState + status overrides from live data.

Pulls the latest `fact_user_team_snapshot` row per (season, team, player)
to reconstruct the user's current squad / bank / FT / chips / cost basis.
Applies FPL status_code overrides:
  - `i` (injured), `n` (not in squad), `s` (suspended), `u` (unavailable)
    → exclude from candidate pool (returned as excluded_player_ids)
  - `d` (doubtful, chance_of_playing < 100) → attenuator dict scales xPts
  - `a` (available) → no override
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import yaml
from sqlalchemy import func as sa_func
from sqlalchemy import select

from fpl_bot.config import settings
from fpl_bot.db.models import FactPlayerStatus, FactUserTeamSnapshot
from fpl_bot.db.session import session_scope
from fpl_bot.optim.state import SECOND_HALF_FIRST_GW, BacktestState

# status_code outcomes
DROP_STATUSES: frozenset[str] = frozenset({"i", "n", "s", "u"})


@dataclass(frozen=True)
class LiveStatusOverrides:
    """Per-player status filter outputs."""

    excluded_player_ids: frozenset[int]  # drop from candidate pool entirely
    xpts_attenuator: dict[int, float]  # player_id → multiplier in [0.0, 1.0]


_CHIP_NAME_MAP = {
    "wildcard": "WC",
    "freehit": "FH",
    "bboost": "BB",
    "3xc": "TC",
}


def _as_int_key_map(value: Any) -> dict[int, Any]:
    if not isinstance(value, dict):
        return {}
    out: dict[int, Any] = {}
    for k, v in value.items():
        try:
            out[int(k)] = v
        except (TypeError, ValueError):
            continue
    return out


def _live_state_override(
    *, season_id: int, gameweek: int, team_id: int
) -> dict[str, Any]:
    """Read optional local live-state overrides.

    Accepted YAML shape:

    season_id:
      gameweek:
        team_id:
          bank_tenths: 61
          free_transfers: 2
          chips_used: [WC1, BB2]
          cost_basis: {118748: 125}
          squad: [118748, ...]   # optional full replacement
    """
    path = settings.live_state_overrides_path
    if not path.exists():
        return {}
    raw = yaml.safe_load(path.read_text()) or {}
    season_block = _as_int_key_map(raw).get(season_id, {})
    gw_block = _as_int_key_map(season_block).get(gameweek, {})
    team_block = _as_int_key_map(gw_block).get(team_id, {})
    return team_block if isinstance(team_block, dict) else {}


def _apply_live_state_override(
    state: BacktestState,
    *,
    season_id: int,
    gameweek: int,
    team_id: int,
) -> BacktestState:
    override = _live_state_override(
        season_id=season_id, gameweek=gameweek, team_id=team_id
    )
    if not override:
        return state

    squad = state.squad
    if "squad" in override:
        squad = frozenset(int(p) for p in override.get("squad") or [])

    cost_basis = dict(state.cost_basis)
    raw_basis = override.get("cost_basis") or override.get("purchase_prices")
    if isinstance(raw_basis, dict):
        for pid, price in raw_basis.items():
            try:
                cost_basis[int(pid)] = int(price)
            except (TypeError, ValueError):
                continue

    chips_used = state.chips_used
    if "chips_used" in override:
        chips_used = frozenset(str(c) for c in (override.get("chips_used") or []))

    return BacktestState(
        season_id=state.season_id,
        gameweek=state.gameweek,
        squad=squad,
        bank=int(override.get("bank_tenths", state.bank)),
        free_transfers=int(override.get("free_transfers", state.free_transfers)),
        chips_used=chips_used,
        cost_basis=cost_basis,
    )


def _resolve_chip_history(
    *,
    season_id: int,
    team_id: int,
    through_gw: int,
) -> frozenset[str]:
    """Return the set of 8-slot chip codes the user has consumed in
    season `season_id` through (and including) `through_gw`.

    Walks every snapshot for this (season, team) where `gameweek <=
    through_gw`, deduplicating by gameweek (snapshots are per-player so
    chips_used_json is replicated). Each (chip_name, gw) maps to
    `{WC,FH,BB,TC}{1 if gw < 20 else 2}` per the 25/26 ruleset.
    """
    with session_scope() as s:
        rows = s.execute(
            select(
                FactUserTeamSnapshot.gameweek,
                FactUserTeamSnapshot.chips_used_json,
            )
            .where(
                (FactUserTeamSnapshot.season_id == season_id)
                & (FactUserTeamSnapshot.team_id == team_id)
                & (FactUserTeamSnapshot.gameweek <= through_gw)
            )
            .distinct()
        ).all()

    resolved: set[str] = set()
    for gw, chips_json in rows:
        chips = json.loads(chips_json) if chips_json else []
        for raw_name in chips:
            base = _CHIP_NAME_MAP.get(str(raw_name).lower())
            if base is None:
                continue
            suffix = "1" if int(gw) < SECOND_HALF_FIRST_GW else "2"
            resolved.add(f"{base}{suffix}")
    return frozenset(resolved)


def load_user_state(
    *,
    season_id: int,
    gameweek: int,
    team_id: int,
) -> BacktestState:
    """Build a BacktestState from the most recent fact_user_team_snapshot
    AT OR BEFORE the target gameweek.

    Note: the FPL `entry/{team_id}/event/{gw}/picks/` endpoint returns 404
    for not-yet-finished GWs. For an UPCOMING-GW recommend we therefore
    fall back to the latest available snapshot (typically the prior GW's
    picks) — that's the freshest state we can observe pre-deadline. The
    returned BacktestState has `gameweek = target gameweek` (so the MILP
    solves for the right horizon) but its squad / bank / FT come from the
    most recent snapshot ≤ target gameweek.

    At GW1 of a new season there is no prior snapshot to fall back to and none
    can be created pre-deadline without private authentication, so this
    returns `BacktestState.cold_start` — no squad, £100.0m, which is exactly
    the state the backtest solves GW1 from. That fallback is deliberately
    restricted to GW1: inventing an empty £100m squad for any later gameweek
    would silently discard a real team, so those still raise.
    """
    with session_scope() as s:
        # Find the latest snapshot gameweek at or before target
        max_gw_row = s.execute(
            select(sa_func.max(FactUserTeamSnapshot.gameweek)).where(
                (FactUserTeamSnapshot.season_id == season_id)
                & (FactUserTeamSnapshot.team_id == team_id)
                & (FactUserTeamSnapshot.gameweek <= gameweek)
            )
        ).scalar()
        if max_gw_row is None:
            if gameweek == 1:
                return _apply_live_state_override(
                    BacktestState.cold_start(season_id),
                    season_id=season_id,
                    gameweek=gameweek,
                    team_id=team_id,
                )
            raise ValueError(
                f"No user-team snapshot for season={season_id}, "
                f"team_id={team_id} at or before gw={gameweek}. "
                f"Run `fpl-bot live ingest --season-id {season_id} --gameweek N` "
                f"for a FINISHED gameweek (the picks endpoint 404s for upcoming "
                f"ones), or set FPL_BOT_FPL_ACCESS_TOKEN to read your current squad "
                f"directly. Only GW1 can be solved with no prior squad."
            )
        snapshot_gw = int(max_gw_row)

        # Subquery: latest recorded_at per player in this (season, snapshot_gw, team)
        latest = (
            select(
                FactUserTeamSnapshot.player_id,
                sa_func.max(FactUserTeamSnapshot.recorded_at).label("max_rec"),
            )
            .where(
                (FactUserTeamSnapshot.season_id == season_id)
                & (FactUserTeamSnapshot.gameweek == snapshot_gw)
                & (FactUserTeamSnapshot.team_id == team_id)
            )
            .group_by(FactUserTeamSnapshot.player_id)
            .subquery("uts_latest")
        )
        rows = s.execute(
            select(FactUserTeamSnapshot)
            .join(
                latest,
                (latest.c.player_id == FactUserTeamSnapshot.player_id)
                & (latest.c.max_rec == FactUserTeamSnapshot.recorded_at),
            )
            .where(
                (FactUserTeamSnapshot.season_id == season_id)
                & (FactUserTeamSnapshot.gameweek == snapshot_gw)
                & (FactUserTeamSnapshot.team_id == team_id)
            )
        ).scalars().all()

    if not rows:
        raise ValueError(
            f"No user-team snapshot for season={season_id}, gw={snapshot_gw}, "
            f"team_id={team_id}. Run `fpl-bot live ingest` first."
        )

    squad_ids: set[int] = set()
    cost_basis: dict[int, int] = {}
    bank_tenths = 0
    free_transfers = 1
    for r in rows:
        squad_ids.add(int(r.player_id))
        cost_basis[int(r.player_id)] = int(r.purchase_price_tenths)
        bank_tenths = int(r.bank_tenths)
        free_transfers = int(r.free_transfers)

    # Chips: scan ALL past snapshots ≤ snapshot_gw to find each GW's
    # active_chip. Map (chip_name, gw) → 8-slot code. The fact's
    # chips_used_json carries this season's active_chip for the snapshot's
    # gameweek only (set by ingest_user_team from `active_chip` payload).
    chips_used_resolved = _resolve_chip_history(
        season_id=season_id,
        team_id=team_id,
        through_gw=snapshot_gw,
    )

    state = BacktestState(
        season_id=season_id,
        gameweek=gameweek,
        squad=frozenset(squad_ids),
        bank=bank_tenths,
        free_transfers=free_transfers,
        chips_used=frozenset(chips_used_resolved),
        cost_basis=cost_basis,
    )
    return _apply_live_state_override(
        state, season_id=season_id, gameweek=gameweek, team_id=team_id
    )


def load_user_sell_prices(
    *,
    season_id: int,
    gameweek: int,
    team_id: int,
) -> dict[int, int]:
    """Latest exact sell prices from the user-team snapshot.

    Returns an empty dict when only older public snapshots are available or
    before `live ingest` has been run. Authenticated `my-team` ingest fills
    `selling_price_tenths`; public ingest stores current price as an
    approximation.
    """
    with session_scope() as s:
        max_gw_row = s.execute(
            select(sa_func.max(FactUserTeamSnapshot.gameweek)).where(
                (FactUserTeamSnapshot.season_id == season_id)
                & (FactUserTeamSnapshot.team_id == team_id)
                & (FactUserTeamSnapshot.gameweek <= gameweek)
            )
        ).scalar()
        if max_gw_row is None:
            return {}
        snapshot_gw = int(max_gw_row)
        latest = (
            select(
                FactUserTeamSnapshot.player_id,
                sa_func.max(FactUserTeamSnapshot.recorded_at).label("max_rec"),
            )
            .where(
                (FactUserTeamSnapshot.season_id == season_id)
                & (FactUserTeamSnapshot.gameweek == snapshot_gw)
                & (FactUserTeamSnapshot.team_id == team_id)
            )
            .group_by(FactUserTeamSnapshot.player_id)
            .subquery("uts_sell_latest")
        )
        rows = s.execute(
            select(
                FactUserTeamSnapshot.player_id,
                FactUserTeamSnapshot.selling_price_tenths,
            )
            .join(
                latest,
                (latest.c.player_id == FactUserTeamSnapshot.player_id)
                & (latest.c.max_rec == FactUserTeamSnapshot.recorded_at),
            )
            .where(
                (FactUserTeamSnapshot.season_id == season_id)
                & (FactUserTeamSnapshot.gameweek == snapshot_gw)
                & (FactUserTeamSnapshot.team_id == team_id)
            )
        ).all()
    return {int(r.player_id): int(r.selling_price_tenths) for r in rows}


def load_status_overrides(
    *,
    candidate_player_ids: list[int],
) -> LiveStatusOverrides:
    """Build the live status filter for the MILP candidate pool.

    Pulls the LATEST fact_player_status row per player_id (regardless of
    season — status is a live-snapshot signal). Players with status_code
    in DROP_STATUSES go into `excluded_player_ids`. Players with
    status_code == 'd' (doubtful) get an `xpts_attenuator[p] = cop/100`
    where cop = chance_of_playing_next_round (default 75 if missing).
    """
    if not candidate_player_ids:
        return LiveStatusOverrides(
            excluded_player_ids=frozenset(),
            xpts_attenuator={},
        )

    with session_scope() as s:
        latest = (
            select(
                FactPlayerStatus.player_id,
                sa_func.max(FactPlayerStatus.recorded_at).label("max_rec"),
            )
            .where(FactPlayerStatus.player_id.in_(candidate_player_ids))
            .group_by(FactPlayerStatus.player_id)
            .subquery("ps_latest")
        )
        rows = s.execute(
            select(FactPlayerStatus)
            .join(
                latest,
                (latest.c.player_id == FactPlayerStatus.player_id)
                & (latest.c.max_rec == FactPlayerStatus.recorded_at),
            )
            .where(FactPlayerStatus.player_id.in_(candidate_player_ids))
        ).scalars().all()

    excluded: set[int] = set()
    attenuator: dict[int, float] = {}
    for r in rows:
        pid = int(r.player_id)
        if r.status_code in DROP_STATUSES:
            excluded.add(pid)
            continue
        if r.status_code == "d":
            cop = r.chance_of_playing_next_round
            if cop is None:
                cop = 75  # FPL convention when "doubtful" lacks specific %
            attenuator[pid] = max(0.0, min(1.0, cop / 100.0))

    return LiveStatusOverrides(
        excluded_player_ids=frozenset(excluded),
        xpts_attenuator=attenuator,
    )
