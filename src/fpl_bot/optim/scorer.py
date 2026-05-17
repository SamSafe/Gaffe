"""Scorer that mirrors FPL's auto-substitution and auto-vice rules.

Used by the backtest harness to compute the actual GW points a manager
would have received from a given (squad, XI, captain, vice, bench-order,
chip) decision tuple — given the actual minutes / points realized.

FPL rules (verified against 2024/25 official):

1. **Auto-sub**: when an XI outfielder plays 0 minutes, the FIRST bench
   outfielder (in bench order) who plays > 0 minutes auto-subs in, IF
   the swap keeps the formation valid (≥3 DEF, ≥2 MID, ≥1 FWD; XI = 11).
2. **GK auto-sub**: the bench GK auto-subs in only if the starting GK
   plays 0 minutes.
3. **Auto-vice**: if the captain plays 0 minutes, the vice gets the
   captain's multiplier (2× normally, 3× under TC). If both blank, no
   multiplier is applied.
4. **Bench Boost**: when BB is active, all 15 players score; no auto-sub
   needed since bench scores natively.
5. **Triple Captain**: captain mult becomes 3× instead of 2×; vice
   inherits 3× under auto-vice.

Bench order: the MILP picks the bench (15 - 11 = 4 players) but doesn't
sort them. We sort by descending predicted xPts in the relevant GW — the
manager's best guess at "who should sub in first if needed".
"""
from __future__ import annotations

from dataclasses import dataclass

from fpl_bot.optim.state import GwDecisions

XI_DEF_MIN = 3
XI_MID_MIN = 2
XI_FWD_MIN = 1


@dataclass(frozen=True)
class ScorerInputs:
    """Per-GW inputs needed to apply the auto-sub / auto-vice scorer."""

    decisions: GwDecisions  # MILP-picked squad/XI/captain/vice/chip
    actual_pts: dict[int, int]  # player_id → actual FPL points (incl. bonus) for this GW
    actual_minutes: dict[int, int]  # player_id → actual minutes (0 = blank)
    positions: dict[int, str]  # player_id → GKP/DEF/MID/FWD
    bench_order_xpts: dict[int, float]  # player_id → predicted xPts (for bench-order sort)


@dataclass(frozen=True)
class ScorerOutputs:
    """Per-GW scorer output."""

    gw_points: int
    starting_xi_final: frozenset[int]  # XI after auto-sub applied
    captain_final: int | None  # captain after auto-vice
    auto_subs: list[tuple[int, int]]  # (out_player, in_player) list
    used_vice: bool


def _played(player_id: int, minutes: dict[int, int]) -> bool:
    return minutes.get(player_id, 0) > 0


def _formation_counts(
    xi: set[int], positions: dict[int, str]
) -> dict[str, int]:
    out = {"GKP": 0, "DEF": 0, "MID": 0, "FWD": 0}
    for p in xi:
        pos = positions.get(p)
        if pos in out:
            out[pos] += 1
    return out


def _formation_valid_after_swap(
    xi: set[int], in_player: int, out_player: int, positions: dict[int, str]
) -> bool:
    """Would replacing out_player with in_player keep the formation legal?"""
    new_xi = (xi - {out_player}) | {in_player}
    counts = _formation_counts(new_xi, positions)
    return (
        counts["GKP"] == 1
        and counts["DEF"] >= XI_DEF_MIN
        and counts["MID"] >= XI_MID_MIN
        and counts["FWD"] >= XI_FWD_MIN
        and sum(counts.values()) == 11
    )


def _sort_bench(
    bench: set[int],
    positions: dict[int, str],
    bench_order_xpts: dict[int, float],
) -> tuple[int | None, list[int]]:
    """Return (bench_gk, sorted outfield bench).

    Outfield bench is sorted by descending xPts — the manager's best
    guess at "who should sub in first if needed".
    """
    bench_gk: int | None = None
    outfield: list[int] = []
    for p in bench:
        if positions.get(p) == "GKP":
            bench_gk = p
        else:
            outfield.append(p)
    outfield.sort(key=lambda p: bench_order_xpts.get(p, 0.0), reverse=True)
    return bench_gk, outfield


def score_gw(inputs: ScorerInputs) -> ScorerOutputs:
    """Apply FPL auto-sub + auto-vice rules to compute realized GW points.

    Order of operations (matches FPL):
    1. Determine which XI players played (minutes > 0).
    2. GK auto-sub: bench GK replaces starting GK if starter blanked.
    3. Outfield auto-sub: walk bench in order; for each XI blank, find
       the first bench outfielder who played AND whose position keeps
       the formation valid.
    4. Captain → vice fallback if captain didn't play.
    5. Sum points; apply captain/TC/BB multipliers as appropriate.
    """
    d = inputs.decisions
    pts = inputs.actual_pts
    mins = inputs.actual_minutes
    positions = inputs.positions

    xi = set(d.starting_xi)
    squad = set(d.squad)
    bench = squad - xi

    bench_gk, bench_outfield = _sort_bench(bench, positions, inputs.bench_order_xpts)

    # Bench Boost: all 15 score — no auto-sub
    auto_subs: list[tuple[int, int]] = []
    if d.chip_played in ("BB", "BB1", "BB2"):
        gw_points = sum(pts.get(p, 0) for p in squad)
        captain = d.captain
        used_vice = False
        if captain is not None:
            if not _played(captain, mins) and d.vice is not None and _played(d.vice, mins):
                captain = d.vice
                used_vice = True
            # BB doesn't trigger TC; multiplier is 2 (no TC under BB).
            multiplier = 2
            gw_points += pts.get(captain, 0) * (multiplier - 1) if _played(captain, mins) else 0
        gw_points -= 4 * d.hits
        return ScorerOutputs(
            gw_points=gw_points,
            starting_xi_final=frozenset(xi),
            captain_final=captain,
            auto_subs=[],
            used_vice=used_vice,
        )

    # Step 2: GK auto-sub
    xi_after = set(xi)
    gk_in_xi = next((p for p in xi if positions.get(p) == "GKP"), None)
    if (
        gk_in_xi is not None
        and not _played(gk_in_xi, mins)
        and bench_gk is not None
        and _played(bench_gk, mins)
    ):
        xi_after.discard(gk_in_xi)
        xi_after.add(bench_gk)
        auto_subs.append((gk_in_xi, bench_gk))

    # Step 3: outfield auto-subs (only for outfielders, walk bench in order)
    used_bench: set[int] = set()
    for blank_out in [p for p in xi if positions.get(p) != "GKP" and not _played(p, mins)]:
        if blank_out not in xi_after:
            continue  # already removed (e.g., GK case, but GK case is handled above)
        for candidate_in in bench_outfield:
            if candidate_in in used_bench:
                continue
            if not _played(candidate_in, mins):
                continue
            if _formation_valid_after_swap(
                xi_after, candidate_in, blank_out, positions
            ):
                xi_after.discard(blank_out)
                xi_after.add(candidate_in)
                used_bench.add(candidate_in)
                auto_subs.append((blank_out, candidate_in))
                break

    # Step 4: captain → vice fallback (mins-based check on ORIGINAL captain
    # per FPL rules; the auto-sub above doesn't change the captain pick)
    captain = d.captain
    used_vice = False
    if (
        captain is not None
        and not _played(captain, mins)
        and d.vice is not None
        and _played(d.vice, mins)
    ):
        captain = d.vice
        used_vice = True

    # Step 5: tally points
    gw_points = sum(pts.get(p, 0) for p in xi_after)
    if captain is not None and _played(captain, mins):
        multiplier = 3 if d.chip_played in ("TC", "TC1", "TC2") else 2
        gw_points += pts.get(captain, 0) * (multiplier - 1)
    gw_points -= 4 * d.hits

    return ScorerOutputs(
        gw_points=gw_points,
        starting_xi_final=frozenset(xi_after),
        captain_final=captain,
        auto_subs=auto_subs,
        used_vice=used_vice,
    )
