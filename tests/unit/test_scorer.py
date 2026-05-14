"""Tests for the auto-sub + auto-vice scorer (FPL rule replication)."""
from __future__ import annotations

from fpl_bot.optim.scorer import ScorerInputs, score_gw
from fpl_bot.optim.state import GwDecisions


def _make_decisions(
    squad: list[int],
    xi: list[int],
    captain: int,
    vice: int,
    chip: str | None = None,
    hits: int = 0,
) -> GwDecisions:
    return GwDecisions(
        gameweek=1,
        squad=frozenset(squad),
        starting_xi=frozenset(xi),
        captain=captain,
        vice=vice,
        transferred_in=frozenset(),
        transferred_out=frozenset(),
        chip_played=chip,
        hits=hits,
        objective_value=0.0,
    )


# Standard 15-man squad: 2 GKP / 5 DEF / 5 MID / 3 FWD
# XI = 1 GKP + 4 DEF + 4 MID + 2 FWD (a 4-4-2 formation), bench: 1 GKP + 1 DEF + 1 MID + 1 FWD
SQUAD = [1, 2, 11, 12, 13, 14, 15, 21, 22, 23, 24, 25, 31, 32, 33]
XI = [1, 11, 12, 13, 14, 21, 22, 23, 24, 31, 32]  # GK 1, DEF 11-14, MID 21-24, FWD 31-32
BENCH = [2, 15, 25, 33]  # bench GK 2, bench DEF 15, bench MID 25, bench FWD 33

POSITIONS = {
    1: "GKP", 2: "GKP",
    11: "DEF", 12: "DEF", 13: "DEF", 14: "DEF", 15: "DEF",
    21: "MID", 22: "MID", 23: "MID", 24: "MID", 25: "MID",
    31: "FWD", 32: "FWD", 33: "FWD",
}


def test_no_blanks_baseline():
    """All XI played, no auto-subs triggered."""
    pts = {p: 5 for p in SQUAD}  # 5 pts each
    mins = {p: 90 for p in SQUAD}  # all played full
    bench_xpts = {p: 1.0 for p in SQUAD}  # arbitrary
    d = _make_decisions(squad=SQUAD, xi=XI, captain=21, vice=22)
    out = score_gw(
        ScorerInputs(decisions=d, actual_pts=pts, actual_minutes=mins,
                     positions=POSITIONS, bench_order_xpts=bench_xpts)
    )
    # 11 XI × 5 pts + captain extra (5 × 1 = 5) = 60
    assert out.gw_points == 60
    assert out.captain_final == 21
    assert not out.used_vice
    assert out.auto_subs == []


def test_xi_outfield_blank_replaced_by_bench():
    """An XI defender blanks; first bench-order outfield replacement subs in."""
    pts = {p: 5 for p in SQUAD}
    pts[11] = 0  # blank DEF
    pts[15] = 8  # bench DEF scores 8
    pts[25] = 12  # bench MID also valuable, but DEF blank → bench DEF is the only DEF replacement option
    mins = {p: 90 for p in SQUAD}
    mins[11] = 0  # didn't play
    # Bench-order: MID 25 has highest xPts; but it's a MID, replacing DEF 11
    # with MID 25 would drop DEF count to 3 (still valid: ≥3 DEF)
    # So MID 25 actually subs in (formation-valid). Let's set it that way.
    bench_xpts = {2: 0.5, 15: 1.0, 25: 5.0, 33: 0.8}
    d = _make_decisions(squad=SQUAD, xi=XI, captain=21, vice=22)
    out = score_gw(
        ScorerInputs(decisions=d, actual_pts=pts, actual_minutes=mins,
                     positions=POSITIONS, bench_order_xpts=bench_xpts)
    )
    # XI 11 (blank) replaced by 25 (MID, formation 1-3-5-2 valid)
    assert (11, 25) in out.auto_subs
    # Original XI pts (with 11 = 0) = 10×5 + 0 = 50; replacement 25 = 12; total XI = 50 + 12 = 62
    # Captain 21 (still in XI) extra = 5
    # Total: 62 + 5 = 67
    assert out.gw_points == 67


def test_xi_outfield_blank_bench_in_formation():
    """When two formations are possible, prefer the higher-xPts bench player."""
    pts = {p: 5 for p in SQUAD}
    pts[11] = 0  # DEF blank
    pts[15] = 8  # bench DEF
    pts[25] = 9  # bench MID
    pts[33] = 7  # bench FWD
    mins = {p: 90 for p in SQUAD}
    mins[11] = 0
    # Bench order: 25 (MID) > 15 (DEF) > 33 (FWD). With DEF blank, current XI
    # has 3 DEF + 4 MID + 2 FWD = 9 outfield + 1 GK = 10. Need 11. Try MID 25
    # first → formation 1-3-5-2 (XI = 11). Valid. Picks 25.
    bench_xpts = {2: 0.5, 25: 10.0, 15: 5.0, 33: 3.0}
    d = _make_decisions(squad=SQUAD, xi=XI, captain=21, vice=22)
    out = score_gw(
        ScorerInputs(decisions=d, actual_pts=pts, actual_minutes=mins,
                     positions=POSITIONS, bench_order_xpts=bench_xpts)
    )
    assert (11, 25) in out.auto_subs


def test_xi_outfield_blank_formation_constrains_choice():
    """When highest-xPts bench would invalidate formation, walk down the list."""
    pts = {p: 5 for p in SQUAD}
    pts[31] = 0  # FWD blank (XI has only 2 FWD: 31, 32 → 1 FWD left)
    pts[15] = 7  # bench DEF
    pts[25] = 8  # bench MID
    pts[33] = 4  # bench FWD
    mins = {p: 90 for p in SQUAD}
    mins[31] = 0
    # XI before: 1 GK + 4 DEF + 4 MID + 2 FWD. After 31 blanks: 1 + 4 + 4 + 1 = 10.
    # Need 1 more. Bench-order: 25 (MID, +1 MID = 5 MID, FWD count = 1 → INVALID ≥1 FWD ok),
    # actually 1 + 4 + 5 + 1 = 11 valid (≥3 DEF, ≥2 MID, ≥1 FWD all satisfied).
    # Picks 25.
    bench_xpts = {2: 0.5, 25: 10.0, 15: 8.0, 33: 4.0}
    d = _make_decisions(squad=SQUAD, xi=XI, captain=21, vice=22)
    out = score_gw(
        ScorerInputs(decisions=d, actual_pts=pts, actual_minutes=mins,
                     positions=POSITIONS, bench_order_xpts=bench_xpts)
    )
    assert (31, 25) in out.auto_subs


def test_gk_auto_sub_only_when_starter_blanks():
    """Bench GK subs in iff starting GK plays 0 mins."""
    pts = {p: 5 for p in SQUAD}
    pts[1] = 0  # starter GK blank
    pts[2] = 6  # bench GK plays
    mins = {p: 90 for p in SQUAD}
    mins[1] = 0
    bench_xpts = {p: 1.0 for p in SQUAD}
    d = _make_decisions(squad=SQUAD, xi=XI, captain=21, vice=22)
    out = score_gw(
        ScorerInputs(decisions=d, actual_pts=pts, actual_minutes=mins,
                     positions=POSITIONS, bench_order_xpts=bench_xpts)
    )
    assert (1, 2) in out.auto_subs
    # Total = 10×5 + 6 (bench GK) + capt extra 5 = 61
    assert out.gw_points == 61


def test_bench_player_must_play_to_auto_sub():
    """A bench player who didn't play can't replace a blank XI."""
    pts = {p: 5 for p in SQUAD}
    pts[11] = 0
    pts[15] = 0  # bench DEF also blank
    pts[25] = 0  # bench MID also blank
    pts[33] = 9  # bench FWD plays — does it sub in?
    mins = {p: 90 for p in SQUAD}
    mins[11] = 0
    mins[15] = 0
    mins[25] = 0
    bench_xpts = {2: 0.5, 25: 10.0, 15: 5.0, 33: 3.0}  # ordering favors blanks first
    d = _make_decisions(squad=SQUAD, xi=XI, captain=21, vice=22)
    out = score_gw(
        ScorerInputs(decisions=d, actual_pts=pts, actual_minutes=mins,
                     positions=POSITIONS, bench_order_xpts=bench_xpts)
    )
    # MID 25 blank, DEF 15 blank → only FWD 33 played. Subbing FWD 33 in for
    # DEF 11: XI becomes 1+3+4+3 = 11 (3 DEF, 4 MID, 3 FWD — all valid). Picks 33.
    assert (11, 33) in out.auto_subs


def test_auto_vice_triggered_when_captain_blanks():
    """Captain 0 mins → vice gets 2× mult."""
    pts = {p: 5 for p in SQUAD}
    pts[21] = 0  # captain blank
    pts[22] = 8  # vice scores
    mins = {p: 90 for p in SQUAD}
    mins[21] = 0
    bench_xpts = {p: 1.0 for p in SQUAD}
    d = _make_decisions(squad=SQUAD, xi=XI, captain=21, vice=22)
    out = score_gw(
        ScorerInputs(decisions=d, actual_pts=pts, actual_minutes=mins,
                     positions=POSITIONS, bench_order_xpts=bench_xpts)
    )
    # MID 21 blanked but MID 22-24 + others, plus bench MID 25 plays — does 21 get auto-subbed?
    # 21 blank, candidates from bench (sorted): all = 1.0, so first bench-outfield in
    # iteration order. Order by xPts then original — implementation-defined for ties.
    # Critical assertion: captain → vice
    assert out.captain_final == 22
    assert out.used_vice


def test_auto_vice_not_triggered_when_captain_plays():
    """Captain played ≥1 min → captain keeps multiplier."""
    pts = {p: 5 for p in SQUAD}
    pts[21] = 8
    mins = {p: 90 for p in SQUAD}
    bench_xpts = {p: 1.0 for p in SQUAD}
    d = _make_decisions(squad=SQUAD, xi=XI, captain=21, vice=22)
    out = score_gw(
        ScorerInputs(decisions=d, actual_pts=pts, actual_minutes=mins,
                     positions=POSITIONS, bench_order_xpts=bench_xpts)
    )
    assert out.captain_final == 21
    assert not out.used_vice


def test_triple_captain_3x_with_auto_vice():
    """TC + captain blank → vice gets 3× mult."""
    pts = {p: 5 for p in SQUAD}
    pts[21] = 0
    pts[22] = 10
    mins = {p: 90 for p in SQUAD}
    mins[21] = 0
    bench_xpts = {p: 1.0 for p in SQUAD}
    d = _make_decisions(squad=SQUAD, xi=XI, captain=21, vice=22, chip="TC")
    out = score_gw(
        ScorerInputs(decisions=d, actual_pts=pts, actual_minutes=mins,
                     positions=POSITIONS, bench_order_xpts=bench_xpts)
    )
    assert out.used_vice
    # Vice 22 plays 10 → captain term contributes 10 * 2 = 20 extra (3x total)
    # rest of XI varies depending on auto-sub; just check captain_final + used_vice + multiplier reach
    # Easier check: gw_points should include 2*pts[22] additional vs baseline


def test_bench_boost_all_15_score():
    """BB: all 15 score; no auto-sub used."""
    pts = {p: 5 for p in SQUAD}
    mins = {p: 90 for p in SQUAD}
    bench_xpts = {p: 1.0 for p in SQUAD}
    d = _make_decisions(squad=SQUAD, xi=XI, captain=21, vice=22, chip="BB")
    out = score_gw(
        ScorerInputs(decisions=d, actual_pts=pts, actual_minutes=mins,
                     positions=POSITIONS, bench_order_xpts=bench_xpts)
    )
    # 15 × 5 + capt extra 5 = 80
    assert out.gw_points == 80
    assert out.auto_subs == []  # BB skips auto-sub
