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

from sqlalchemy import func as sa_func
from sqlalchemy import select

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
            raise ValueError(
                f"No user-team snapshot for season={season_id}, "
                f"team_id={team_id} at or before gw={gameweek}. "
                f"Run `fpl-bot live ingest` first."
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

    return BacktestState(
        season_id=season_id,
        gameweek=gameweek,
        squad=frozenset(squad_ids),
        bank=bank_tenths,
        free_transfers=free_transfers,
        chips_used=frozenset(chips_used_resolved),
        cost_basis=cost_basis,
    )


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
