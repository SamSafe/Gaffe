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

# Gameweeks whose predictions are too weak to base a chip on. Before ~GW4 there
# are no current-season rolling features, so the joint-xPts prior returns
# near-identical small values for everyone. Measured on the 26/27 opener:
# expected minutes ≈ 12 for a nailed starter, and the resulting TC pick was a
# £4.0m defender. Burning a once-per-half chip on that noise is strictly worse
# than holding it. BB already avoided early GWs for a related reason; this makes
# the rule explicit and extends it to TC.
COLD_START_GWS = 3


@dataclass(frozen=True)
class ChipSchedule:
    """Per-chip GW assignment for FPL 2025/26 (8-slot ruleset).

    Each chip type — Wildcard, Free Hit, Bench Boost, Triple Captain —
    can be played once per season half (first half = GW1-19, second = 20-38).
    None = chip held / not played this season.
    """

    wc1: int | None
    fh1: int | None
    bb1: int | None
    tc1: int | None
    wc2: int | None
    fh2: int | None
    bb2: int | None
    tc2: int | None

    def as_dict(self) -> dict[str, int]:
        """Return only the chips that ARE scheduled (skip None)."""
        return {
            slot: gw
            for slot, gw in (
                ("WC1", self.wc1),
                ("FH1", self.fh1),
                ("BB1", self.bb1),
                ("TC1", self.tc1),
                ("WC2", self.wc2),
                ("FH2", self.fh2),
                ("BB2", self.bb2),
                ("TC2", self.tc2),
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


def _best_fh_gw(
    analytics: SeasonFixtureAnalytics,
    gws: list[int],
    excluded: set[int] | None = None,
) -> int | None:
    """GW with the most teams blank. Returns None if no BGW (no team blank
    anywhere in the window) — that signals "no FH-worthy GW".
    """
    excluded = excluded or set()
    scored = [
        (gw, _team_blank_count(analytics, gw))
        for gw in gws
        if gw not in excluded
    ]
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
    """Compute the chip schedule for one season (FPL 2025/26 8-slot ruleset).

    Each chip type — WC, FH, BB, TC — gets ONE slot per season half
    (1st half = GW1-19, 2nd half = GW20-38).

    Priority order for collision resolution within a half: FH > TC > BB > WC.
    """
    fh_gws = first_half_gws(analytics)
    sh_gws = second_half_gws(analytics)
    used: set[int] = set()

    def _pick_tc_in_half(half_gws: list[int]) -> int | None:
        # Never triple-captain out of the cold-start window: the prior cannot
        # tell a premium striker from a £4.0m defender there.
        half_gws = [g for g in half_gws if g > COLD_START_GWS]
        if not half_gws:
            return None
        # Prefer DGW captains; fall back to highest captain xPts overall.
        tc = _best_tc_gw(
            analytics, predictions_df, half_gws, used, team_id_per_player
        )
        if tc is None:
            per_gw_max = (
                predictions_df.filter(pl.col("gameweek").is_in(half_gws))
                .filter(~pl.col("gameweek").is_in(list(used)))
                .group_by("gameweek")
                .agg(pl.col("e_xpts").max().alias("max_e_xpts"))
                .sort(["max_e_xpts", "gameweek"], descending=[True, False])
            )
            tc = int(per_gw_max[0, "gameweek"]) if not per_gw_max.is_empty() else None
        return tc

    def _pick_bb_in_half(half_gws: list[int]) -> int | None:
        half_gws = [g for g in half_gws if g > COLD_START_GWS]
        if not half_gws:
            return None
        bb = _best_bb_gw(analytics, half_gws, used)
        if bb is None:
            # No DGW in this half. The joint xPts prior biases early-season GWs
            # (no rolling features → position defaults inflate GW1 totals), and
            # playing BB at a cold-start GW locks in £-on-bench instead of XI.
            # Prefer the LATEST non-colliding GW — the squad has had time to be
            # transfer-tuned for bench depth.
            candidates = [g for g in half_gws if g not in used]
            bb = max(candidates) if candidates else None
        return bb

    # FH first (highest priority — most timing-sensitive)
    fh1 = _best_fh_gw(analytics, fh_gws)
    if fh1 is not None:
        used.add(fh1)
    # FPL does not permit Free Hits in consecutive GWs. With one FH per half,
    # the only possible cross-slot violation is FH1 in GW19 + FH2 in GW20.
    fh2_excluded = {20} if fh1 == 19 else set()
    fh2 = _best_fh_gw(analytics, sh_gws, excluded=fh2_excluded)
    if fh2 is not None:
        used.add(fh2)

    # TC1, TC2: best DGW captain in each half
    tc1 = _pick_tc_in_half(fh_gws)
    if tc1 is not None:
        used.add(tc1)
    tc2 = _pick_tc_in_half(sh_gws)
    if tc2 is not None:
        used.add(tc2)

    # BB1, BB2: best DGW (most teams) in each half
    bb1 = _pick_bb_in_half(fh_gws)
    if bb1 is not None:
        used.add(bb1)
    bb2 = _pick_bb_in_half(sh_gws)
    if bb2 is not None:
        used.add(bb2)

    # WC1 / WC2: pick the latest non-colliding GW before the corresponding
    # FH (so WC restructures squad just in time for FH). min_gw guards
    # against early-season WC1 (FPL convention: ≥ GW4) and out-of-window
    # WC2 (≥ GW21).
    def _pick_wc(
        half_gws: list[int], fh_gw: int | None, default: int, min_gw: int,
    ) -> int | None:
        if not half_gws:
            return None
        candidates = [g for g in half_gws if g >= min_gw]
        if fh_gw is not None:
            candidates = [g for g in candidates if g < fh_gw]
        candidates = [g for g in candidates if g not in used]
        if candidates:
            return max(candidates)
        if (
            default in half_gws
            and default >= min_gw
            and default not in used
            and (fh_gw is None or default < fh_gw)
        ):
            return default
        return None

    wc1 = _pick_wc(fh_gws, fh1, default=8, min_gw=4)
    if wc1 is not None:
        used.add(wc1)
    wc2 = _pick_wc(sh_gws, fh2, default=27, min_gw=21)
    if wc2 is not None:
        used.add(wc2)

    return ChipSchedule(
        wc1=wc1, fh1=fh1, bb1=bb1, tc1=tc1,
        wc2=wc2, fh2=fh2, bb2=bb2, tc2=tc2,
    )


def free_hit_gws(chip_schedule: dict[str, int] | None) -> set[int]:
    """Free Hit GWs in a schedule dict."""
    if not chip_schedule:
        return set()
    return {
        gw
        for slot, gw in chip_schedule.items()
        if slot in {"FH1", "FH2"}
    }


def horizon_before_free_hit(
    horizon_gws: list[int],
    chip_schedule: dict[str, int] | None,
) -> list[int]:
    """Avoid modeling a future Free Hit as a persistent squad.

    The current MILP has one squad state variable, so it cannot represent a
    temporary FH squad and the reverted permanent squad in the same horizon.
    If a future FH lies inside the horizon, stop the lookahead before it. If
    FH is the current GW, keep only that GW; the caller should also suppress
    terminal value for the solve.
    """
    if not horizon_gws:
        return horizon_gws
    fhs = free_hit_gws(chip_schedule)
    if not fhs:
        return horizon_gws
    first_w = horizon_gws[0]
    future_fhs = sorted(gw for gw in fhs if gw in horizon_gws)
    if not future_fhs:
        return horizon_gws
    first_fh = future_fhs[0]
    if first_fh == first_w:
        return [first_w]
    return [gw for gw in horizon_gws if gw < first_fh]
