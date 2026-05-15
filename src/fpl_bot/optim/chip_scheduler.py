"""Phase 5 heuristic chip scheduler.

Decides each chip's GW slot ONCE at backtest start, then forces the
rolling MILP to play chips at those GWs. Decoupled from the MILP — no
joint optimization, just informed heuristics.

The four heuristics (per phase5_chip_dp.md §2):
- FH1 / FH2: play at GW with most TEAMS blank in each half-season.
- TC: play at GW with the highest-xPts DGW-eligible captain.
- BB: play at GW with the most teams in DGW.
- WC1 / WC2: play 1 GW before the chosen FH (to restructure squad for FH).

v1 uses TEAM-level (not squad-level) DGW/BGW to sidestep the
chicken-and-egg of needing a squad before scheduling. The FH GW chosen
by team-level BGW is empirically a strong proxy for squad-level BGW for
generic top-3-per-team squads.

Open question §7.4: chip collisions. v1 priority FH > TC > BB > WC.
If TC's best GW is the same as FH's, TC moves to its 2nd-best GW.
"""
from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from fpl_bot.optim.fixture_analytics import (
    SeasonFixtureAnalytics,
    first_half_gws,
    second_half_gws,
)


@dataclass(frozen=True)
class ChipSchedule:
    """Per-chip GW assignment. None = chip held / not played this season."""

    wc1: int | None
    fh1: int | None
    wc2: int | None
    fh2: int | None
    bb: int | None
    tc: int | None

    def as_dict(self) -> dict[str, int]:
        """Return only the chips that ARE scheduled (skip None)."""
        return {
            slot: gw
            for slot, gw in (
                ("WC1", self.wc1),
                ("FH1", self.fh1),
                ("WC2", self.wc2),
                ("FH2", self.fh2),
                ("BB", self.bb),
                ("TC", self.tc),
            )
            if gw is not None
        }


def _team_blank_count(analytics: SeasonFixtureAnalytics, gw: int) -> int:
    """Number of teams with zero fixtures in this GW."""
    if gw not in analytics.all_gws:
        return 0
    return sum(
        1
        for team in analytics.gws_per_team
        if analytics.fixture_count.get((team, gw), 0) == 0
    )


def _team_dgw_count(analytics: SeasonFixtureAnalytics, gw: int) -> int:
    """Number of teams with ≥ 2 fixtures in this GW."""
    return sum(
        1
        for team in analytics.gws_per_team
        if analytics.fixture_count.get((team, gw), 0) >= 2
    )


def _best_fh_gw(analytics: SeasonFixtureAnalytics, gws: list[int]) -> int | None:
    """GW with the most teams blank. Returns None if no BGW (no team blank
    anywhere in the window) — that signals "no FH-worthy GW".
    """
    scored = [(gw, _team_blank_count(analytics, gw)) for gw in gws]
    scored = [(gw, n) for gw, n in scored if n > 0]
    if not scored:
        return None
    return max(scored, key=lambda x: x[1])[0]


def _best_bb_gw(
    analytics: SeasonFixtureAnalytics, gws: list[int], excluded: set[int]
) -> int | None:
    """GW with the most teams in DGW, excluding `excluded` (chip collisions)."""
    scored = [
        (gw, _team_dgw_count(analytics, gw))
        for gw in gws
        if gw not in excluded
    ]
    scored = [(gw, n) for gw, n in scored if n > 0]
    if not scored:
        return None
    return max(scored, key=lambda x: x[1])[0]


def _best_tc_gw(
    analytics: SeasonFixtureAnalytics,
    predictions_df: pl.DataFrame,
    gws: list[int],
    excluded: set[int],
    team_id_per_player: dict[int, int],
) -> int | None:
    """GW with the highest-xPts captain among DGW-eligible candidates.

    For each GW with at least one team in DGW, find the highest
    aggregate-fixture-xPts player whose team has DGW. Pick the GW with
    the highest such captain xPts.

    `predictions_df` columns: player_id, gameweek, e_xpts (already summed
    across DGW fixtures via Phase 3's `_per_player_per_gw_predictions`).
    """
    best_gw, best_score = None, -1.0
    for gw in gws:
        if gw in excluded:
            continue
        dgw_teams = {
            t
            for t in analytics.gws_per_team
            if analytics.fixture_count.get((t, gw), 0) >= 2
        }
        if not dgw_teams:
            continue
        # Filter predictions to this GW + DGW-team players
        eligible_players = {
            pid for pid, tid in team_id_per_player.items() if tid in dgw_teams
        }
        if not eligible_players:
            continue
        gw_preds = predictions_df.filter(
            (pl.col("gameweek") == gw)
            & pl.col("player_id").is_in(eligible_players)
        )
        if gw_preds.is_empty():
            continue
        top_xpts = float(gw_preds["e_xpts"].max())
        if top_xpts > best_score:
            best_score = top_xpts
            best_gw = gw
    return best_gw


def make_chip_schedule(
    *,
    analytics: SeasonFixtureAnalytics,
    predictions_df: pl.DataFrame,
    team_id_per_player: dict[int, int],
) -> ChipSchedule:
    """Compute the chip schedule for one season.

    Priority order for collision resolution: FH > TC > BB > WC.
    """
    fh_gws = first_half_gws(analytics)
    sh_gws = second_half_gws(analytics)
    used: set[int] = set()

    # FH first (highest priority)
    fh1 = _best_fh_gw(analytics, fh_gws)
    if fh1 is not None:
        used.add(fh1)
    fh2 = _best_fh_gw(analytics, sh_gws)
    if fh2 is not None:
        used.add(fh2)

    # TC: any GW (the most discretionary). Prefer DGW captains.
    tc = _best_tc_gw(
        analytics, predictions_df, analytics.all_gws, used, team_id_per_player
    )
    if tc is None:
        # Fallback: GW with highest captain xPts overall
        per_gw_max = (
            predictions_df.filter(~pl.col("gameweek").is_in(list(used)))
            .group_by("gameweek")
            .agg(pl.col("e_xpts").max().alias("max_e_xpts"))
            .sort("max_e_xpts", descending=True)
        )
        tc = int(per_gw_max[0, "gameweek"]) if not per_gw_max.is_empty() else None
    if tc is not None:
        used.add(tc)

    # BB: any GW with DGW teams
    bb = _best_bb_gw(analytics, analytics.all_gws, used)
    if bb is None:
        # Fallback: GW with highest aggregate predicted xPts across top-4 bench candidates
        # (rough: take the cheapest 4 players by avg-pts across the season)
        # Even simpler: GW with highest total xPts across all players
        per_gw_total = (
            predictions_df.filter(~pl.col("gameweek").is_in(list(used)))
            .group_by("gameweek")
            .agg(pl.col("e_xpts").sum().alias("total"))
            .sort("total", descending=True)
        )
        bb = int(per_gw_total[0, "gameweek"]) if not per_gw_total.is_empty() else None
    if bb is not None:
        used.add(bb)

    # WC1 / WC2: play just before FH (or fallback). WC must land on a real
    # GW (in fh_gws / sh_gws) and not collide with another chip. Strategy:
    # pick the latest non-colliding GW that is strictly < FH GW within
    # the half. If no FH, default to GW 8 / GW 27 (FPL community classic
    # WC slots).
    def _pick_wc(half_gws: list[int], fh_gw: int | None, default: int) -> int | None:
        candidates = (
            [g for g in half_gws if g < fh_gw] if fh_gw else half_gws
        )
        # Prefer GW closest-but-before-FH.
        candidates = sorted(candidates, reverse=True)
        for cand in candidates:
            if cand not in used:
                return cand
        # Last-resort default
        if default in half_gws and default not in used:
            return default
        return None

    wc1 = _pick_wc(fh_gws, fh1, default=8)
    if wc1 is not None:
        used.add(wc1)
    wc2 = _pick_wc(sh_gws, fh2, default=27)
    if wc2 is not None:
        used.add(wc2)

    return ChipSchedule(wc1=wc1, fh1=fh1, wc2=wc2, fh2=fh2, bb=bb, tc=tc)
