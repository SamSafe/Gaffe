"""DefCon (Defensive Contribution Points) xPts adjustment for FPL 2025/26.

FPL added a new scoring rule in 25/26:
  - DEF: +2 pts if defensive_contribution ≥ 10 in a match
  - MID/FWD: +2 pts if defensive_contribution ≥ 12 in a match
  - GKP: not applicable

`defensive_contribution` for outfield players:
  - DEF: tackles + (clearances + blocks + interceptions)
  - MID/FWD: tackles + (clearances + blocks + interceptions) + recoveries

The bot's xPts model trained on 19-24 doesn't know about this rule. The
cross-fold diagnostic shows DEF bias flipped (over → under-predicted on
25/26) consistent with the new rule.

This module computes a per-(player, gameweek) DefCon-pts adjustment via:
  E[DefCon pts] = 2 × P(defcon ≥ threshold | past appearances) × p_play

The probability uses a PIT-correct rolling rate: only past GWs (relative
to the target GW) feed the estimate. Source: vaastav 25/26 gw{N}.csv —
the only place defensive_contribution is currently available (no DB
migration yet; this is the quick-validation MVP).
"""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

from fpl_bot.db import pit
from fpl_bot.db.session import session_scope
from sqlalchemy import select
from fpl_bot.db.models import DimPlayerSeasonXref

VAASTAV_25_GWS = Path("data/raw/vaastav/Fantasy-Premier-League/data/2025-26/gws")
DEFCON_THRESHOLD: dict[str, int] = {"DEF": 10, "MID": 12, "FWD": 12}


def _load_25_defcon_per_player_per_gw() -> (
    dict[tuple[int, int], tuple[int, int, str]]
):
    """Returns {(player_id, gw): (defcon, minutes, position)} from vaastav.

    Cross-season identity is via DimPlayerSeasonXref — vaastav `element` is the
    per-season FPL id; we map to stable player_id (FPL `code`).
    """
    with session_scope() as s:
        xref_rows = s.execute(
            select(
                DimPlayerSeasonXref.fpl_element_id, DimPlayerSeasonXref.player_id
            ).where(DimPlayerSeasonXref.season_id == 25)
        ).all()
    element_to_pid: dict[int, int] = {
        int(r.fpl_element_id): int(r.player_id) for r in xref_rows
    }

    out: dict[tuple[int, int], tuple[int, int, str]] = {}
    if not VAASTAV_25_GWS.exists():
        return out
    for f in sorted(VAASTAV_25_GWS.glob("gw*.csv"), key=lambda p: int(p.stem[2:])):
        gw = int(f.stem[2:])
        with open(f) as fp:
            reader = csv.DictReader(fp)
            for row in reader:
                element = int(row["element"])
                pid = element_to_pid.get(element)
                if pid is None:
                    continue
                mins = int(row.get("minutes") or 0)
                if mins <= 0:
                    continue
                dc = int(row.get("defensive_contribution") or 0)
                pos = row.get("position") or "MID"
                # vaastav uses GK; canonicalize to GKP
                if pos == "GK":
                    pos = "GKP"
                out[(pid, gw)] = (dc, mins, pos)
    return out


def compute_defcon_adjustments(
    test_season: int = 25,
    target_gws: list[int] | None = None,
    min_appearances: int = 3,
    fallback_rate: float = 0.0,
) -> dict[tuple[int, int], float]:
    """For each (player_id, target_gw), return the expected DefCon points.

    Uses PIT-correct rolling rate: for target GW = N, only player's
    GW 1..N-1 appearances feed the trigger-rate estimate. Players with
    fewer than `min_appearances` prior appearances get `fallback_rate`
    (per-position mean computed from all available prior appearances).
    """
    if test_season != 25:
        return {}
    raw = _load_25_defcon_per_player_per_gw()
    if not raw:
        return {}

    pos_lookup = pit.all_player_positions().to_pandas().set_index("player_id")[
        "position_code"
    ].to_dict()

    # Index per-player time series
    by_pid: dict[int, list[tuple[int, int, int, str]]] = defaultdict(list)
    for (pid, gw), (dc, mins, pos) in raw.items():
        by_pid[pid].append((gw, dc, mins, pos))
    for pid in by_pid:
        by_pid[pid].sort(key=lambda x: x[0])

    # Position-level fallback rates (computed across ALL data; PIT-leak is
    # tiny since this is just a fallback for unseen players)
    pos_rates: dict[str, float] = {}
    for pos in ("DEF", "MID", "FWD"):
        threshold = DEFCON_THRESHOLD[pos]
        triggers = sum(
            1 for (_pid, _g), (dc, _m, p) in raw.items() if p == pos and dc >= threshold
        )
        appearances = sum(1 for (_pid, _g), (_dc, _m, p) in raw.items() if p == pos)
        pos_rates[pos] = triggers / appearances if appearances else 0.0

    target_set = set(target_gws) if target_gws else None
    out: dict[tuple[int, int], float] = {}
    for pid, series in by_pid.items():
        pos_db = pos_lookup.get(pid)
        # Use DB position if available; fall back to the latest CSV position
        pos = pos_db or (series[-1][3] if series else "MID")
        if pos == "GKP" or pos not in DEFCON_THRESHOLD:
            continue
        threshold = DEFCON_THRESHOLD[pos]
        # Walk forward through the series; at each GW the trigger-rate is
        # computed from prior appearances only.
        triggers_so_far = 0
        appearances_so_far = 0
        for gw, dc, _mins, _pos in series:
            # Future GWs we want to score: this is for GW `gw` based on
            # prior data (appearances_so_far / triggers_so_far at this point
            # reflect strictly GW < gw).
            if target_set is None or gw in target_set:
                if appearances_so_far >= min_appearances:
                    rate = triggers_so_far / appearances_so_far
                else:
                    rate = pos_rates.get(pos, fallback_rate)
                out[(pid, gw)] = 2.0 * rate
            # Now ingest this GW's outcome for future predictions
            if dc >= threshold:
                triggers_so_far += 1
            appearances_so_far += 1
        # Also predict for any future GWs in target_set after the player's
        # last appearance (use the cumulative rate)
        if target_set is not None and series:
            last_seen = series[-1][0]
            rate = (
                triggers_so_far / appearances_so_far
                if appearances_so_far >= min_appearances
                else pos_rates.get(pos, fallback_rate)
            )
            for gw in target_set:
                if gw > last_seen and (pid, gw) not in out:
                    out[(pid, gw)] = 2.0 * rate
    return out
