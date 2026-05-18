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
  E[DefCon pts] = 2 × P(defcon ≥ threshold | past appearances)

The probability uses a PIT-correct rolling rate: only past GWs (relative
to the target GW) feed the estimate. Source: `pit.defensive_contribution
_per_player_per_gw` (Phase 7 productionized — was CSV-direct in the MVP).
"""
from __future__ import annotations

from collections import defaultdict

import polars as pl

from fpl_bot.db import pit

DEFCON_THRESHOLD: dict[str, int] = {"DEF": 10, "MID": 12, "FWD": 12}


def compute_defcon_adjustments(
    test_season: int = 25,
    target_gws: list[int] | None = None,
    min_appearances: int = 3,
    fallback_rate: float = 0.0,
    per_position_shrinkage: dict[str, float] | None = None,
) -> dict[tuple[int, int], float]:
    """For each (player_id, target_gw), return the expected DefCon points.

    Uses PIT-correct rolling rate: for target GW = N, only player's
    GW 1..N-1 appearances feed the trigger-rate estimate. Players with
    fewer than `min_appearances` prior appearances get the per-position
    mean as their rate.

    `per_position_shrinkage` (dict like {"DEF": 0.4, "MID": 0.3, "FWD": 0.3})
    scales the additive adjustment per position, since the joint xPts model
    already captures part of DefCon implicitly via rolling-pts features. A
    single-value global shrinkage is applied by the caller.

    Filters: appearances with minutes==0 are dropped (a 0-min "appearance"
    has no defensive_contribution by construction).
    """
    df = pit.defensive_contribution_per_player_per_gw(season_id=test_season)
    if df.is_empty():
        return {}
    df = df.filter(pl.col("minutes") > 0)

    pos_lookup = (
        pit.all_player_positions()
        .to_pandas()
        .set_index("player_id")["position_code"]
        .to_dict()
    )

    # Index per-player time series
    by_pid: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for r in df.iter_rows(named=True):
        by_pid[int(r["player_id"])].append(
            (int(r["gameweek"]), int(r["defensive_contribution"]))
        )
    for pid in by_pid:
        by_pid[pid].sort()

    # Position-level fallback rates (used when a player has too few prior
    # appearances to estimate their own rate).
    pos_appearances: dict[str, int] = defaultdict(int)
    pos_triggers: dict[str, int] = defaultdict(int)
    for pid, series in by_pid.items():
        pos = pos_lookup.get(pid)
        if pos not in DEFCON_THRESHOLD:
            continue
        threshold = DEFCON_THRESHOLD[pos]
        for _gw, dc in series:
            pos_appearances[pos] += 1
            if dc >= threshold:
                pos_triggers[pos] += 1
    pos_rates: dict[str, float] = {
        p: pos_triggers[p] / pos_appearances[p] if pos_appearances[p] else 0.0
        for p in DEFCON_THRESHOLD
    }

    target_set = set(target_gws) if target_gws else None
    out: dict[tuple[int, int], float] = {}
    for pid, series in by_pid.items():
        pos = pos_lookup.get(pid)
        if pos is None or pos not in DEFCON_THRESHOLD:
            continue
        threshold = DEFCON_THRESHOLD[pos]
        shrink = (
            per_position_shrinkage.get(pos, 1.0)
            if per_position_shrinkage
            else 1.0
        )
        triggers_so_far = 0
        appearances_so_far = 0
        for gw, dc in series:
            if target_set is None or gw in target_set:
                if appearances_so_far >= min_appearances:
                    rate = triggers_so_far / appearances_so_far
                else:
                    rate = pos_rates.get(pos, fallback_rate)
                out[(pid, gw)] = 2.0 * rate * shrink
            if dc >= threshold:
                triggers_so_far += 1
            appearances_so_far += 1
        # Cover future GWs after the player's last observed appearance
        if target_set is not None and series:
            last_seen = series[-1][0]
            rate = (
                triggers_so_far / appearances_so_far
                if appearances_so_far >= min_appearances
                else pos_rates.get(pos, fallback_rate)
            )
            for gw in target_set:
                if gw > last_seen and (pid, gw) not in out:
                    out[(pid, gw)] = 2.0 * rate * shrink
    return out
