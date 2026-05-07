"""FPL points rule table + score function (Phase 2.5).

Per-iteration FPL points are computed alongside per-iteration BPS in the same
Monte Carlo loop (same RNG draws → consistent bonus and xPts samples). This
module defines the FPL-points side; the simulator wires the two together.

FPL 2024/25 rules. Diverges from BPS in:
  - Goals: smaller integer values (4-6 vs 12-24 BPS)
  - Cards: -1/-3 (vs -3/-9 BPS)
  - Saves: 1 per 3 (vs 2 per 3 BPS)
  - Goals conceded: -1 per 2 (vs -1 per 1 BPS)
"""
from __future__ import annotations

# FPL points constants (2024/25)
FPL_GOAL_BY_POSITION: dict[str, int] = {"GKP": 6, "DEF": 6, "MID": 5, "FWD": 4}
FPL_ASSIST = 3
FPL_CS_BY_POSITION: dict[str, int] = {"GKP": 4, "DEF": 4, "MID": 1, "FWD": 0}
FPL_YELLOW = -1
FPL_RED = -3
FPL_OWN_GOAL = -2
FPL_PEN_MISS = -2

# Histogram range (covers ~99.9% of realistic xPts; outliers clip)
XPTS_HIST_MIN = -5
XPTS_HIST_MAX = 25
XPTS_HIST_BINS = XPTS_HIST_MAX - XPTS_HIST_MIN + 1  # 31 bins, integer-indexed

# FPL "haul" thresholds — community-standard captaincy-decision tail probabilities
HAUL_THRESHOLDS: tuple[int, ...] = (2, 6, 10, 15)


def fpl_appearance_pts(minutes: int) -> int:
    if minutes <= 0:
        return 0
    if minutes < 60:
        return 1
    return 2


def fpl_saves_pts(saves: int) -> int:
    """+1 per 3 saves (GK only — caller filters by position)."""
    if saves <= 0:
        return 0
    return saves // 3


def fpl_goals_conceded_pts(goals_conceded: int) -> int:
    """-1 per 2 goals conceded (GK/DEF only — caller filters by position)."""
    if goals_conceded <= 0:
        return 0
    return -(goals_conceded // 2)


def score_fpl_points(
    *,
    position: str,
    minutes: int,
    goals: int,
    assists: int,
    team_clean_sheet: bool,
    team_goals_conceded: int,
    saves: int = 0,
    yellow_cards: int = 0,
    red_cards: int = 0,
    own_goals: int = 0,
    penalties_missed_or_saved: int = 0,
    bonus: int = 0,
) -> int:
    """Total FPL points for one player-fixture realization."""
    pts = fpl_appearance_pts(minutes)
    pts += goals * FPL_GOAL_BY_POSITION.get(position, 4)
    pts += assists * FPL_ASSIST
    if minutes >= 60 and team_clean_sheet:
        pts += FPL_CS_BY_POSITION.get(position, 0)
    if position in ("GKP", "DEF"):
        pts += fpl_goals_conceded_pts(team_goals_conceded)
        if position == "GKP":
            pts += fpl_saves_pts(saves)
    pts += yellow_cards * FPL_YELLOW
    pts += red_cards * FPL_RED
    pts += own_goals * FPL_OWN_GOAL
    pts += penalties_missed_or_saved * FPL_PEN_MISS
    pts += bonus
    return pts
