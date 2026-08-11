"""DefCon (Defensive Contribution Points) xPts adjustment for FPL 2025/26+.

FPL added a new scoring rule in 25/26:
  - DEF: +2 pts if defensive_contribution ≥ 10 in a match
  - MID/FWD: +2 pts if defensive_contribution ≥ 12 in a match
  - GKP: not applicable

`defensive_contribution` for outfield players:
  - DEF: tackles + (clearances + blocks + interceptions)
  - MID/FWD: tackles + (clearances + blocks + interceptions) + recoveries

The bot's xPts model trained on pre-25/26 data doesn't know about this rule. The
cross-fold diagnostic shows DEF bias flipped (over → under-predicted on
25/26) consistent with the new rule.

This module computes a per-(player, gameweek) DefCon-pts adjustment via:
  E[DefCon pts] = 2 × P(defcon ≥ threshold | past appearances)

The probability uses a PIT-correct rolling rate: completed earlier DefCon
seasons plus only past GWs in the target season feed the estimate. This lets a
new season start from the prior season's evidence without looking ahead.
Source: `pit.defensive_contribution_per_player_per_gw`.
"""
from __future__ import annotations

from collections import defaultdict

import polars as pl

from fpl_bot.db import pit

DEFCON_THRESHOLD: dict[str, int] = {"DEF": 10, "MID": 12, "FWD": 12}


def compute_defcon_adjustments(
    test_season: int = 25,
    target_gws: list[int] | None = None,
    target_player_ids: list[int] | None = None,
    min_appearances: int = 3,
    fallback_rate: float = 0.0,
    per_position_shrinkage: dict[str, float] | None = None,
    source_seasons: list[int] | None = None,
) -> dict[tuple[int, int], float]:
    """For each (player_id, target_gw), return the expected DefCon points.

    For target GW N, completed prior DefCon seasons and target-season GWs
    1..N-1 feed the trigger-rate estimate. Players with fewer than
    `min_appearances` get a point-in-time position mean. With explicit
    ``target_gws`` and ``target_player_ids``, players who have no Premier
    League history (for example promoted-club signings) receive that fallback.

    `per_position_shrinkage` (dict like {"DEF": 0.4, "MID": 0.3, "FWD": 0.3})
    scales the additive adjustment per position, since the joint xPts model
    already captures part of DefCon implicitly via rolling-pts features. A
    single-value global shrinkage is applied by the caller.

    Filters: appearances with minutes==0 are dropped (a 0-min "appearance"
    has no defensive_contribution by construction).
    """
    if test_season < 25:
        return {}
    seasons = source_seasons or list(range(25, test_season + 1))
    seasons = sorted({season for season in seasons if 25 <= season <= test_season})
    if test_season not in seasons:
        seasons.append(test_season)

    frames: list[pl.DataFrame] = []
    for season_id in seasons:
        frame = pit.defensive_contribution_per_player_per_gw(season_id=season_id)
        if frame.is_empty():
            continue
        frames.append(
            frame.filter(pl.col("minutes") > 0).with_columns(
                pl.lit(season_id, dtype=pl.Int16).alias("season_id")
            )
        )
    if not frames:
        return {}

    positions = pit.all_player_positions()
    if positions.is_empty():
        return {}
    pos_lookup = {
        int(row["player_id"]): row["position_code"]
        for row in positions.iter_rows(named=True)
    }

    # Completed prior seasons are a starting prior for every target GW.
    historic_player: dict[int, list[int]] = defaultdict(lambda: [0, 0])
    historic_position: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    # Current-season events are kept by GW so each target uses only earlier GWs.
    current_player: dict[int, list[tuple[int, int]]] = defaultdict(list)
    current_position: dict[str, list[tuple[int, int]]] = defaultdict(list)

    for row in pl.concat(frames, how="diagonal_relaxed").iter_rows(named=True):
        pid = int(row["player_id"])
        pos = pos_lookup.get(pid)
        if pos not in DEFCON_THRESHOLD:
            continue
        triggered = int(int(row["defensive_contribution"]) >= DEFCON_THRESHOLD[pos])
        season_id = int(row["season_id"])
        if season_id < test_season:
            historic_player[pid][0] += 1
            historic_player[pid][1] += triggered
            historic_position[pos][0] += 1
            historic_position[pos][1] += triggered
        else:
            gw = int(row["gameweek"])
            current_player[pid].append((gw, triggered))
            current_position[pos].append((gw, triggered))

    for events in (*current_player.values(), *current_position.values()):
        events.sort()

    if target_gws is None:
        target_pairs = [
            (pid, gw)
            for pid, events in current_player.items()
            for gw, _triggered in events
        ]
    else:
        pids = (
            set(target_player_ids)
            if target_player_ids is not None
            else set(historic_player) | set(current_player)
        )
        target_pairs = [(pid, gw) for pid in pids for gw in sorted(set(target_gws))]

    def _past_counts(events: list[tuple[int, int]], target_gw: int) -> tuple[int, int]:
        appearances = triggers = 0
        for gw, triggered in events:
            if gw >= target_gw:
                break
            appearances += 1
            triggers += triggered
        return appearances, triggers

    position_rates: dict[tuple[str, int], float] = {}
    out: dict[tuple[int, int], float] = {}
    for pid, gw in target_pairs:
        pos = pos_lookup.get(pid)
        if pos not in DEFCON_THRESHOLD:
            continue
        past_apps, past_triggers = _past_counts(current_player.get(pid, []), gw)
        player_apps = historic_player[pid][0] + past_apps
        player_triggers = historic_player[pid][1] + past_triggers
        if player_apps >= min_appearances:
            rate = player_triggers / player_apps
        else:
            rate_key = (pos, gw)
            if rate_key not in position_rates:
                pos_apps, pos_triggers = _past_counts(current_position.get(pos, []), gw)
                pos_apps += historic_position[pos][0]
                pos_triggers += historic_position[pos][1]
                position_rates[rate_key] = (
                    pos_triggers / pos_apps if pos_apps else fallback_rate
                )
            rate = position_rates[rate_key]
        shrink = (
            per_position_shrinkage.get(pos, 1.0)
            if per_position_shrinkage
            else 1.0
        )
        out[(pid, gw)] = 2.0 * rate * shrink
    return out
