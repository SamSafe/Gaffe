"""Phase 7 (Tier 3) — domestic-suspension carry-over.

The xPts stack predicts minutes from rolling form; it has no notion of a
player serving a suspension, so it happily predicts a normal return for a
banned player and the MILP fields/captains them for a guaranteed 0. This
module derives, point-in-time, the (player_id, gameweek) pairs where a
player is serving a ban, so prediction post-processing can zero them.

**Triggers** (Premier League domestic rules, 38-GW season):
  - a red card  -> at least a 1-match ban;
  - 5 yellows reached by GW19  -> 1-match ban;
  - 10 yellows reached by GW32 -> 2-match ban;
  - 15 yellows reached         -> 3-match ban.

Cards are read from `fact_player_match`. A ban triggered by cards in
gameweek `g` is served from `g+1` onward, so every emitted (player, gw)
entry depends only on matches strictly before `gw` — i.e. it is
point-in-time correct even though the whole season is scanned at once.

**Deliberately conservative.** We cannot tell a straight red from a
violent-conduct red (3-match) in the data, so reds get the minimum
1-match ban — under-banning (degrades to current behaviour) is far
cheaper than over-banning (benches an available player). The ban is
served in the next gameweek `g+k`; double/blank gameweeks are not
resolved against each team's fixture list (rare; a mismatch just lands on
a gameweek the player has no prediction row for, a harmless no-op).
"""
from __future__ import annotations

import polars as pl

from fpl_bot.db import pit

# yellow-card count -> (ban length in matches, latest gameweek the rule
# applies). Reaching the count after the cutoff carries no ban.
YELLOW_THRESHOLDS: tuple[tuple[int, int, int], ...] = (
    (5, 1, 19),
    (10, 2, 32),
    (15, 3, 38),
)


def compute_suspended_player_gws(test_season: int) -> set[tuple[int, int]]:
    """Return {(player_id, gameweek)} pairs where the player is suspended."""
    df = pit.all_player_match_with_kickoff(season_ids=[test_season])
    if df.is_empty():
        return set()
    return suspensions_from_frame(df)


def suspensions_from_frame(df: pl.DataFrame) -> set[tuple[int, int]]:
    """Pure card-accounting core (DB-free, unit-testable).

    `df` needs columns: player_id, gameweek, yellow_cards, red_cards.
    """
    # Collapse to one row per (player, gameweek); summing cards folds any
    # double-gameweek matches together before threshold accounting.
    per_gw = (
        df.group_by(["player_id", "gameweek"])
        .agg(
            pl.col("yellow_cards").fill_null(0).sum().alias("yc"),
            pl.col("red_cards").fill_null(0).sum().alias("rc"),
        )
        .sort(["player_id", "gameweek"])
    )

    suspended: set[tuple[int, int]] = set()
    for pid, sub in per_gw.group_by("player_id", maintain_order=True):
        player_id = pid[0] if isinstance(pid, tuple) else pid
        cum_yellows = 0
        for row in sub.iter_rows(named=True):
            gw = int(row["gameweek"])
            prev_cum = cum_yellows
            cum_yellows += int(row["yc"])

            ban_matches = 1 if int(row["rc"]) > 0 else 0
            for count, length, cutoff_gw in YELLOW_THRESHOLDS:
                if prev_cum < count <= cum_yellows and gw <= cutoff_gw:
                    ban_matches = max(ban_matches, length)

            for k in range(1, ban_matches + 1):
                suspended.add((int(player_id), gw + k))

    return suspended
