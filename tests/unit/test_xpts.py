"""Unit tests for the FPL points rule table (Phase 2.5)."""
from __future__ import annotations

from fpl_bot.models.xpts import (
    FPL_ASSIST,
    FPL_CS_BY_POSITION,
    FPL_GOAL_BY_POSITION,
    FPL_OWN_GOAL,
    FPL_PEN_MISS,
    FPL_RED,
    FPL_YELLOW,
    XPTS_HIST_BINS,
    XPTS_HIST_MAX,
    XPTS_HIST_MIN,
    fpl_appearance_pts,
    fpl_goals_conceded_pts,
    fpl_saves_pts,
    score_fpl_points,
)


# ── Appearance points ────────────────────────────────────────────────────────


def test_appearance_points_boundaries() -> None:
    assert fpl_appearance_pts(0) == 0
    assert fpl_appearance_pts(1) == 1
    assert fpl_appearance_pts(59) == 1
    assert fpl_appearance_pts(60) == 2
    assert fpl_appearance_pts(90) == 2


# ── Saves and goals-conceded ─────────────────────────────────────────────────


def test_saves_one_per_three() -> None:
    assert fpl_saves_pts(0) == 0
    assert fpl_saves_pts(2) == 0
    assert fpl_saves_pts(3) == 1
    assert fpl_saves_pts(5) == 1
    assert fpl_saves_pts(6) == 2
    assert fpl_saves_pts(9) == 3


def test_goals_conceded_negative_per_two() -> None:
    assert fpl_goals_conceded_pts(0) == 0
    assert fpl_goals_conceded_pts(1) == 0  # zero penalty for first goal
    assert fpl_goals_conceded_pts(2) == -1
    assert fpl_goals_conceded_pts(3) == -1
    assert fpl_goals_conceded_pts(4) == -2


# ── Full FPL points score ────────────────────────────────────────────────────


def test_mid_haul_with_goal_assist_cs() -> None:
    """MID 90 min, 1 goal, 1 assist, CS, 0 cards, 1 bonus."""
    pts = score_fpl_points(
        position="MID",
        minutes=90,
        goals=1,
        assists=1,
        team_clean_sheet=True,
        team_goals_conceded=0,
        bonus=1,
    )
    # 2 (appearance) + 5 (MID goal) + 3 (assist) + 1 (MID CS) + 1 (bonus) = 12
    assert pts == 2 + FPL_GOAL_BY_POSITION["MID"] + FPL_ASSIST + FPL_CS_BY_POSITION["MID"] + 1


def test_def_haul_with_goal_cs_3_bonus() -> None:
    pts = score_fpl_points(
        position="DEF",
        minutes=90,
        goals=1,
        assists=0,
        team_clean_sheet=True,
        team_goals_conceded=0,
        bonus=3,
    )
    # 2 + 6 + 0 + 4 + 0 - 0 + 3 = 15
    assert pts == 2 + 6 + 4 + 3


def test_gk_clean_sheet_with_saves_and_one_pen_save() -> None:
    """GK 90 min, CS, 6 saves. Penalty save isn't a separate FPL points field
    (only -2 for missed pen by takers, which doesn't apply to GK saves);
    pen_save in the FPL system gives +5 separately, but that's not in score_fpl_points
    yet — keeping the test honest about current behavior."""
    pts = score_fpl_points(
        position="GKP",
        minutes=90,
        goals=0,
        assists=0,
        team_clean_sheet=True,
        team_goals_conceded=0,
        saves=6,
        bonus=0,
    )
    # 2 (appearance) + 4 (GK CS) + 2 (6 saves = 2 pts) + 0 = 8
    assert pts == 2 + 4 + 2


def test_def_concedes_3_no_cs() -> None:
    pts = score_fpl_points(
        position="DEF",
        minutes=90,
        goals=0,
        assists=0,
        team_clean_sheet=False,
        team_goals_conceded=3,
        bonus=0,
    )
    # 2 (appearance) + 0 (no goal) + 0 (no CS) + (-1 for goals conceded ≥ 2) = 1
    assert pts == 2 + (-1)


def test_yellow_card() -> None:
    pts = score_fpl_points(
        position="MID",
        minutes=60,
        goals=0,
        assists=0,
        team_clean_sheet=False,
        team_goals_conceded=0,
        yellow_cards=1,
    )
    assert pts == 2 + FPL_YELLOW


def test_red_card_short_appearance() -> None:
    pts = score_fpl_points(
        position="FWD",
        minutes=30,
        goals=0,
        assists=0,
        team_clean_sheet=False,
        team_goals_conceded=0,
        red_cards=1,
    )
    assert pts == 1 + FPL_RED


def test_own_goal_negative() -> None:
    pts = score_fpl_points(
        position="DEF",
        minutes=90,
        goals=0,
        assists=0,
        team_clean_sheet=False,
        team_goals_conceded=1,
        own_goals=1,
    )
    # 2 (appearance) + 0 + 0 + 0 + (-2 own goal) = 0
    assert pts == 2 + FPL_OWN_GOAL


def test_pen_miss_negative_for_taker() -> None:
    pts = score_fpl_points(
        position="MID",
        minutes=90,
        goals=0,
        assists=0,
        team_clean_sheet=False,
        team_goals_conceded=2,
        penalties_missed_or_saved=1,
    )
    assert pts == 2 + FPL_PEN_MISS


def test_zero_minutes_zero_points() -> None:
    """A bench player who didn't come on gets 0 FPL points."""
    pts = score_fpl_points(
        position="MID",
        minutes=0,
        goals=0,
        assists=0,
        team_clean_sheet=True,
        team_goals_conceded=0,
        bonus=0,
    )
    assert pts == 0


def test_short_appearance_no_cs_bonus_for_def() -> None:
    """DEF with <60 min on a CS team gets only appearance points, no CS bonus."""
    pts = score_fpl_points(
        position="DEF",
        minutes=45,
        goals=0,
        assists=0,
        team_clean_sheet=True,
        team_goals_conceded=0,
    )
    assert pts == 1  # only appearance points


# ── Histogram constants ──────────────────────────────────────────────────────


def test_histogram_constants_consistent() -> None:
    """XPTS_HIST_BINS = XPTS_HIST_MAX - XPTS_HIST_MIN + 1."""
    assert XPTS_HIST_BINS == XPTS_HIST_MAX - XPTS_HIST_MIN + 1
    assert XPTS_HIST_MIN == -5
    assert XPTS_HIST_MAX == 25
    assert XPTS_HIST_BINS == 31
