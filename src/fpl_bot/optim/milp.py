"""Rolling-horizon FPL MILP — Pyomo + HiGHS (Phase 3 §2-§6).

Decision variables, constraints, and objective per docs/design/phase3_deterministic_milp.md.

v1 scope: deterministic (uses E[xPts]); chips integrated as MILP variables
greedily within horizon (no Phase 5 chip-DP coupling); FH simplified — when
played the harness resets the squad after that GW.
"""
from __future__ import annotations

from dataclasses import dataclass

import polars as pl
from pyomo.environ import (
    Binary,
    ConcreteModel,
    Constraint,
    NonNegativeIntegers,
    NonNegativeReals,
    Objective,
    Set,
    Var,
    maximize,
    value,
)

from fpl_bot.optim.state import (
    ALL_CHIP_SLOTS,
    SECOND_HALF_FIRST_GW,
    BacktestState,
    GwDecisions,
)
from fpl_bot.optim.terminal_value import terminal_value_coefficients

POSITIONS = ("GKP", "DEF", "MID", "FWD")
SQUAD_SIZE = 15
XI_SIZE = 11
POSITION_SQUAD_COUNTS = {"GKP": 2, "DEF": 5, "MID": 5, "FWD": 3}
PER_TEAM_CAP = 3
HIT_COST = 4

# Big-M for chip-suppression of hits. 15 max transfers × 4 hit cost = 60; use 100 for safety.
HIT_BIGM = 100


@dataclass
class MilpInputs:
    """All the parameters the MILP needs for one rolling solve."""

    state: BacktestState
    horizon_weeks: list[int]  # absolute GW numbers, e.g. [5, 6, 7, 8, 9, 10]
    candidates: list[int]
    predictions: dict[tuple[int, int], float]  # (player_id, gw) → E[xPts]
    eo: dict[int, float]  # player_id → EO ∈ [0, 1]
    buy_prices: dict[int, int]  # tenths
    sell_prices: dict[int, int]  # tenths
    positions: dict[int, str]
    teams: dict[int, int]
    rho: float = 1.0
    alpha: float = 0.8
    beta: float = 0.05
    full_predictions: pl.DataFrame | None = None  # for terminal value lookahead
    # v1.0: chip decision variables disabled by default. The bilinear chip
    # auxiliaries (TC, BB) plus FH/WC big-M linearization roughly double the
    # MILP size and cause time-limit hits at H=6 + 200 candidates. Chips are
    # deferred to Phase 3.5 alongside the season-long chip-DP from Phase 5.
    # When False, captain mult is 2× (not 3× via TC), no BB bench scoring,
    # WC/FH unavailable.
    enable_chips: bool = False
    # Phase 4 SAA: when use_saa=True, objective uses sample-average over
    # scenarios from predictions_per_scenario instead of the deterministic
    # `predictions` dict. Set scenario_ids to the list of scenario indices
    # to average over (e.g., list(range(50))). `predictions` is still
    # required (used for candidate-filtering and as a fallback if a
    # (p, w, s) key is missing).
    use_saa: bool = False
    predictions_per_scenario: (
        dict[tuple[int, int, int], float] | None
    ) = None  # (player_id, gw, scenario_id) → pts
    scenario_ids: list[int] | None = None
    # Phase 5 chip-DP: when set, the MILP forces chip activations to the
    # pre-decided GW (1 at chip_schedule[slot], 0 at all other GWs in the
    # horizon). Decouples chip timing from the rolling MILP entirely.
    # Format: {"FH1": 18, "TC": 26, ...} (subset of slots; missing slots
    # default to "not yet played, MILP can still pick within horizon"
    # — but typically the scheduler covers all 6 slots).
    chip_schedule: dict[str, int] | None = None


def _gw_in_first_half(gw: int) -> bool:
    return gw < SECOND_HALF_FIRST_GW


def _chip_slot_for_chip_in_gw(chip_type: str, gw: int) -> str:
    """Map ('WC', gw=15) → 'WC1' (first half) etc."""
    if chip_type == "WC":
        return "WC1" if _gw_in_first_half(gw) else "WC2"
    if chip_type == "FH":
        return "FH1" if _gw_in_first_half(gw) else "FH2"
    return chip_type  # BB, TC


def build_milp(inputs: MilpInputs) -> ConcreteModel:
    state = inputs.state
    H = inputs.horizon_weeks
    P = inputs.candidates
    pred = inputs.predictions
    eo = inputs.eo
    pos = inputs.positions
    teams = inputs.teams
    buy = inputs.buy_prices
    sell = inputs.sell_prices

    m = ConcreteModel()
    m.P = Set(initialize=P)
    m.W = Set(initialize=H, ordered=True)

    # ── Decision variables ───────────────────────────────────────────────────
    m.x = Var(m.P, m.W, domain=Binary)
    m.y = Var(m.P, m.W, domain=Binary)
    m.c = Var(m.P, m.W, domain=Binary)
    m.v = Var(m.P, m.W, domain=Binary)
    m.bin = Var(m.P, m.W, domain=Binary)
    m.bout = Var(m.P, m.W, domain=Binary)
    m.ft = Var(m.W, domain=NonNegativeIntegers, bounds=(0, 5))
    m.ht = Var(m.W, domain=NonNegativeIntegers)
    m.bank = Var(m.W, domain=NonNegativeReals)
    # Chip variables: created always but pinned to 0 when chips disabled (v1.0).
    m.z_wc = Var(m.W, domain=Binary)
    m.z_fh = Var(m.W, domain=Binary)
    m.z_bb = Var(m.W, domain=Binary)
    m.z_tc = Var(m.W, domain=Binary)
    # Weekly chip bonus aggregates (replaces per-(player, week) bilinear aux).
    # tc_bonus[w] = z_tc[w] * sum_p c[p,w] * pred[p,w]    (captain extra 1× under TC)
    # bb_bonus[w] = z_bb[w] * sum_p (x[p,w] - y[p,w]) * pred[p,w]   (bench points under BB)
    # This shrinks the model from O(P*W) chip binaries to O(W) reals — at H=6,
    # P=200 that's ~2400 binaries → 12 reals. Big-M linearization is exact
    # because pred[p,w] is a parameter (the products c*pred and (x-y)*pred are
    # linear in the squad/XI/captain decision vars).
    # v1.0 simplification: chip bonuses bypass EO adjustment (captain choice is
    # still EO-aware via the c term in the main rolling+eo objective; the EXTRA
    # 1× under TC just isn't EO-discounted). Documented as v1.1 candidate.
    m.tc_bonus = Var(m.W, domain=NonNegativeReals)
    m.bb_bonus = Var(m.W, domain=NonNegativeReals)
    if not inputs.enable_chips:
        for w in H:
            m.z_wc[w].fix(0)
            m.z_fh[w].fix(0)
            m.z_bb[w].fix(0)
            m.z_tc[w].fix(0)
            m.tc_bonus[w].fix(0.0)
            m.bb_bonus[w].fix(0.0)
    elif inputs.chip_schedule is not None:
        # Phase 5+: chip schedule forces activations to pre-decided GWs.
        # 25/26 ruleset: 8 slots — WC1/WC2/FH1/FH2/BB1/BB2/TC1/TC2.
        # Each chip TYPE (wc, fh, bb, tc) has one shared m.z_* var per week;
        # the schedule supplies up to 2 target GWs per type (one per half).
        schedule = inputs.chip_schedule
        used_chips_before = set(state.chips_used)

        def _targets_for_type(slot_names: tuple[str, ...]) -> list[int]:
            return [
                schedule[s]
                for s in slot_names
                if s in schedule and s not in used_chips_before
            ]

        wc_targets = _targets_for_type(("WC1", "WC2"))
        fh_targets = _targets_for_type(("FH1", "FH2"))
        bb_targets = _targets_for_type(("BB1", "BB2"))
        tc_targets = _targets_for_type(("TC1", "TC2"))

        for w in H:
            m.z_wc[w].fix(1 if w in wc_targets else 0)
            m.z_fh[w].fix(1 if w in fh_targets else 0)
            m.z_bb[w].fix(1 if w in bb_targets else 0)
            m.z_tc[w].fix(1 if w in tc_targets else 0)

    # ── Squad shape ─────────────────────────────────────────────────────────
    def _squad_size(m, w):
        return sum(m.x[p, w] for p in m.P) == SQUAD_SIZE

    m.con_squad_size = Constraint(m.W, rule=_squad_size)

    def _per_position(m, w, position):
        return (
            sum(m.x[p, w] for p in m.P if pos.get(p) == position)
            == POSITION_SQUAD_COUNTS[position]
        )

    m.con_gkp = Constraint(m.W, rule=lambda m, w: _per_position(m, w, "GKP"))
    m.con_def = Constraint(m.W, rule=lambda m, w: _per_position(m, w, "DEF"))
    m.con_mid = Constraint(m.W, rule=lambda m, w: _per_position(m, w, "MID"))
    m.con_fwd = Constraint(m.W, rule=lambda m, w: _per_position(m, w, "FWD"))

    # Per-team cap
    distinct_teams = sorted({teams.get(p, -1) for p in P})
    m.TEAMS = Set(initialize=distinct_teams)

    def _team_cap(m, w, t):
        return sum(m.x[p, w] for p in m.P if teams.get(p) == t) <= PER_TEAM_CAP

    m.con_team_cap = Constraint(m.W, m.TEAMS, rule=_team_cap)

    # ── Starting XI / formation ─────────────────────────────────────────────
    def _xi_size(m, w):
        return sum(m.y[p, w] for p in m.P) == XI_SIZE

    m.con_xi_size = Constraint(m.W, rule=_xi_size)

    def _xi_subset(m, p, w):
        return m.y[p, w] <= m.x[p, w]

    m.con_xi_subset = Constraint(m.P, m.W, rule=_xi_subset)

    def _xi_one_gk(m, w):
        return sum(m.y[p, w] for p in m.P if pos.get(p) == "GKP") == 1

    m.con_xi_gk = Constraint(m.W, rule=_xi_one_gk)

    def _xi_def_min(m, w):
        return sum(m.y[p, w] for p in m.P if pos.get(p) == "DEF") >= 3

    m.con_xi_def = Constraint(m.W, rule=_xi_def_min)

    def _xi_mid_min(m, w):
        return sum(m.y[p, w] for p in m.P if pos.get(p) == "MID") >= 2

    m.con_xi_mid = Constraint(m.W, rule=_xi_mid_min)

    def _xi_fwd_min(m, w):
        return sum(m.y[p, w] for p in m.P if pos.get(p) == "FWD") >= 1

    m.con_xi_fwd = Constraint(m.W, rule=_xi_fwd_min)

    # ── Captain / vice ───────────────────────────────────────────────────────
    # Phase 3 baseline: single captain per week (deterministic / SAA first-stage).
    # In SAA mode, m.c[p, w] is the FIRST-STAGE captain for w=1 only; for w≥2
    # the actual captain is m.c_s[p, w, s] (scenario-conditional). We still
    # need m.c[p, w] to be defined for w≥2 (Pyomo doesn't allow conditional
    # var indexing without set tricks) — we fix it to 0 below.
    def _one_captain(m, w):
        if inputs.use_saa and w != H[0]:
            # In SAA mode, m.c is meaningful only at w=1; force 0 elsewhere.
            return sum(m.c[p, w] for p in m.P) == 0
        return sum(m.c[p, w] for p in m.P) == 1

    m.con_one_captain = Constraint(m.W, rule=_one_captain)

    def _one_vice(m, w):
        return sum(m.v[p, w] for p in m.P) == 1

    m.con_one_vice = Constraint(m.W, rule=_one_vice)

    def _capt_in_xi(m, p, w):
        return m.c[p, w] <= m.y[p, w]

    m.con_capt_in_xi = Constraint(m.P, m.W, rule=_capt_in_xi)

    def _vice_in_xi(m, p, w):
        return m.v[p, w] <= m.y[p, w]

    m.con_vice_in_xi = Constraint(m.P, m.W, rule=_vice_in_xi)

    def _capt_vice_distinct(m, p, w):
        return m.c[p, w] + m.v[p, w] <= 1

    m.con_capt_vice = Constraint(m.P, m.W, rule=_capt_vice_distinct)

    # Phase 4 SAA: scenario-conditional captain for w ≥ 2.
    if inputs.use_saa:
        if inputs.predictions_per_scenario is None or inputs.scenario_ids is None:
            raise ValueError(
                "use_saa=True requires predictions_per_scenario and scenario_ids"
            )
        m.S = Set(initialize=inputs.scenario_ids, ordered=True)
        m.c_s = Var(m.P, m.W, m.S, domain=Binary)

        first_w = H[0]

        # For w=1: c_s[p,1,s] = c[p,1] for all s (non-anticipative; first-stage)
        def _capt_first_stage_link(m, p, s):
            return m.c_s[p, first_w, s] == m.c[p, first_w]

        m.con_capt_first_stage_link = Constraint(
            m.P, m.S, rule=_capt_first_stage_link
        )

        # For w≥2: scenario-conditional captain rules
        def _one_capt_s(m, w, s):
            return sum(m.c_s[p, w, s] for p in m.P) == 1

        m.con_one_capt_s = Constraint(m.W, m.S, rule=_one_capt_s)

        def _capt_s_in_xi(m, p, w, s):
            return m.c_s[p, w, s] <= m.y[p, w]

        m.con_capt_s_in_xi = Constraint(m.P, m.W, m.S, rule=_capt_s_in_xi)

    # ── Transfers ────────────────────────────────────────────────────────────
    initial_squad = state.squad
    first_w = H[0]

    def _transfer_balance(m, p, w):
        if w == first_w:
            prev = 1 if p in initial_squad else 0
            return m.x[p, w] - prev == m.bin[p, w] - m.bout[p, w]
        # Find prior week
        prev_w = H[H.index(w) - 1]
        return m.x[p, w] - m.x[p, prev_w] == m.bin[p, w] - m.bout[p, w]

    m.con_transfer_balance = Constraint(m.P, m.W, rule=_transfer_balance)

    def _bin_bout_disjoint(m, p, w):
        return m.bin[p, w] + m.bout[p, w] <= 1

    m.con_bin_bout = Constraint(m.P, m.W, rule=_bin_bout_disjoint)

    # ── Free transfers and hits ─────────────────────────────────────────────
    def _ft_initial(m):
        return m.ft[first_w] == state.free_transfers

    if state.squad:
        m.con_ft_initial = Constraint(rule=_ft_initial)
    # Cold start: leave ft[first_w] free; we constrain hits[first_w] = 0 below.

    def _ft_evolution(m, w):
        if w == first_w:
            return Constraint.Skip
        prev_w = H[H.index(w) - 1]
        # ft_w = min(5, ft_{w-1} + 1 - transfers_used_{w-1})
        # Linearize: introduce ft_uncapped >= ft_{w-1} + 1 - sum(bin), ft <= 5, ft <= ft_uncapped
        # Skip the min cap via ft_w = ft_{prev} + 1 - transfers but bound by [0, 5]
        # We bound m.ft already at [0, 5], and add: ft_w <= ft_{prev} + 1 - sum(bout_{prev_w}) - (-M * chip_waiver)
        # WC and FH waive the consumption. Simplification: when chip waives, we let ft_w be free in [0, 5].
        wc_or_fh = m.z_wc[prev_w] + m.z_fh[prev_w]
        # If no chip waive: ft_w == ft_prev + 1 - transfers_in_prev (with cap at 5 absorbed by domain bounds)
        return (
            m.ft[w] <= m.ft[prev_w] + 1 - sum(m.bout[p, prev_w] for p in m.P) + 100 * wc_or_fh
        )

    m.con_ft_evolution = Constraint(m.W, rule=_ft_evolution)

    def _ft_above_zero(m, w):
        if w == first_w:
            return Constraint.Skip
        prev_w = H[H.index(w) - 1]
        wc_or_fh = m.z_wc[prev_w] + m.z_fh[prev_w]
        return (
            m.ft[w] >= m.ft[prev_w] + 1 - sum(m.bout[p, prev_w] for p in m.P) - 100 * (1 + wc_or_fh)
        )

    m.con_ft_above_zero = Constraint(m.W, rule=_ft_above_zero)

    def _hits(m, w):
        # Cold start: GW1 picks are free (no FPL hit penalty); ht[first_w] = 0
        if w == first_w and not state.squad:
            return m.ht[w] == 0
        return m.ht[w] >= sum(m.bout[p, w] for p in m.P) - m.ft[w] - HIT_BIGM * (
            m.z_wc[w] + m.z_fh[w]
        )

    m.con_hits = Constraint(m.W, rule=_hits)

    # ── Budget evolution ─────────────────────────────────────────────────────
    # Unified bank evolution: bank[w] = (entering bank) + sells - buys
    # where entering bank is state.bank for first_w, bank[w-1] otherwise.
    # For cold start, bin == x for picked players and bout == 0, so this
    # collapses to bank[first_w] = state.bank - sum_x_costs (squad value).

    def _bank_evolution(m, w):
        sells = sum(sell.get(p, 0) * m.bout[p, w] for p in m.P)
        buys = sum(buy.get(p, 0) * m.bin[p, w] for p in m.P)
        if w == first_w:
            return m.bank[w] == state.bank + sells - buys
        prev_w = H[H.index(w) - 1]
        return m.bank[w] == m.bank[prev_w] + sells - buys

    m.con_bank_evolution = Constraint(m.W, rule=_bank_evolution)

    # ── Chip availability ────────────────────────────────────────────────────
    def _chip_one_per_gw(m, w):
        return m.z_wc[w] + m.z_fh[w] + m.z_bb[w] + m.z_tc[w] <= 1

    m.con_chip_one_per_gw = Constraint(m.W, rule=_chip_one_per_gw)

    # Per-slot at-most-once across the horizon (with first-half / second-half restrictions)
    def _build_per_slot_constraint(m, slot: str):
        if slot in state.chips_used:
            # Force all activations to 0
            for w in H:
                if (slot == "WC1" and _gw_in_first_half(w)) or (slot == "WC2" and not _gw_in_first_half(w)):
                    m.z_wc[w].setub(0)
                elif (slot == "FH1" and _gw_in_first_half(w)) or (slot == "FH2" and not _gw_in_first_half(w)):
                    m.z_fh[w].setub(0)
                elif (slot == "BB1" and _gw_in_first_half(w)) or (slot == "BB2" and not _gw_in_first_half(w)):
                    m.z_bb[w].setub(0)
                elif (slot == "TC1" and _gw_in_first_half(w)) or (slot == "TC2" and not _gw_in_first_half(w)):
                    m.z_tc[w].setub(0)
            return
        # Slot is available: limit horizon activations to ≤ 1 in the eligible window
        if slot == "WC1":
            eligible = [w for w in H if _gw_in_first_half(w)]
            if eligible:
                m.add_component(
                    "con_wc1",
                    Constraint(expr=sum(m.z_wc[w] for w in eligible) <= 1),
                )
            for w in H:
                if not _gw_in_first_half(w):
                    m.z_wc[w].setub(0) if "WC2" in state.chips_used else None
        elif slot == "WC2":
            eligible = [w for w in H if not _gw_in_first_half(w)]
            if eligible:
                m.add_component(
                    "con_wc2",
                    Constraint(expr=sum(m.z_wc[w] for w in eligible) <= 1),
                )
        elif slot == "FH1":
            eligible = [w for w in H if _gw_in_first_half(w)]
            if eligible:
                m.add_component(
                    "con_fh1",
                    Constraint(expr=sum(m.z_fh[w] for w in eligible) <= 1),
                )
        elif slot == "FH2":
            eligible = [w for w in H if not _gw_in_first_half(w)]
            if eligible:
                m.add_component(
                    "con_fh2",
                    Constraint(expr=sum(m.z_fh[w] for w in eligible) <= 1),
                )
        elif slot == "BB1":
            eligible = [w for w in H if _gw_in_first_half(w)]
            if eligible:
                m.add_component(
                    "con_bb1",
                    Constraint(expr=sum(m.z_bb[w] for w in eligible) <= 1),
                )
        elif slot == "BB2":
            eligible = [w for w in H if not _gw_in_first_half(w)]
            if eligible:
                m.add_component(
                    "con_bb2",
                    Constraint(expr=sum(m.z_bb[w] for w in eligible) <= 1),
                )
        elif slot == "TC1":
            eligible = [w for w in H if _gw_in_first_half(w)]
            if eligible:
                m.add_component(
                    "con_tc1",
                    Constraint(expr=sum(m.z_tc[w] for w in eligible) <= 1),
                )
        elif slot == "TC2":
            eligible = [w for w in H if not _gw_in_first_half(w)]
            if eligible:
                m.add_component(
                    "con_tc2",
                    Constraint(expr=sum(m.z_tc[w] for w in eligible) <= 1),
                )

    if inputs.enable_chips:
        for slot in ALL_CHIP_SLOTS:
            _build_per_slot_constraint(m, slot)

    # Weekly chip-bonus big-M linearization — only added when chips are enabled.
    # M_TC: safe upper bound on a single player's xPts in one GW (typical max ~15).
    # M_BB: safe upper bound on bench (4 players) xPts in one GW (typical max ~30).
    M_TC = 30.0
    M_BB = 60.0

    # Helper: captain ref per (p, w, s). For SAA, uses scenario-conditional
    # c_s[p, w, s] for w≥2 and c[p, w=1] for w=1 (which equals c_s[p, 1, s]
    # by the non-anticipativity constraint). For deterministic, just c[p, w].
    def _capt_ref(m, p, w, s):
        if inputs.use_saa and w != H[0]:
            return m.c_s[p, w, s]
        return m.c[p, w]

    if inputs.enable_chips and not inputs.use_saa:
        # Deterministic: per-week chip bonuses (no scenario index)
        def _tc_bonus_le_capt(m, w):
            return m.tc_bonus[w] <= sum(m.c[p, w] * pred.get((p, w), 0.0) for p in m.P)
        m.con_tc_bonus_le_capt = Constraint(m.W, rule=_tc_bonus_le_capt)

        def _tc_bonus_le_z(m, w):
            return m.tc_bonus[w] <= M_TC * m.z_tc[w]
        m.con_tc_bonus_le_z = Constraint(m.W, rule=_tc_bonus_le_z)

        def _tc_bonus_ge(m, w):
            return m.tc_bonus[w] >= sum(m.c[p, w] * pred.get((p, w), 0.0) for p in m.P) - M_TC * (1 - m.z_tc[w])
        m.con_tc_bonus_ge = Constraint(m.W, rule=_tc_bonus_ge)

        def _bb_bonus_le_bench(m, w):
            return m.bb_bonus[w] <= sum((m.x[p, w] - m.y[p, w]) * pred.get((p, w), 0.0) for p in m.P)
        m.con_bb_bonus_le_bench = Constraint(m.W, rule=_bb_bonus_le_bench)

        def _bb_bonus_le_z(m, w):
            return m.bb_bonus[w] <= M_BB * m.z_bb[w]
        m.con_bb_bonus_le_z = Constraint(m.W, rule=_bb_bonus_le_z)

        def _bb_bonus_ge(m, w):
            return m.bb_bonus[w] >= sum(
                (m.x[p, w] - m.y[p, w]) * pred.get((p, w), 0.0) for p in m.P
            ) - M_BB * (1 - m.z_bb[w])
        m.con_bb_bonus_ge = Constraint(m.W, rule=_bb_bonus_ge)

    elif inputs.enable_chips and inputs.use_saa:
        # SAA: per-(week, scenario) chip bonuses. tc_bonus_s and bb_bonus_s
        # supersede the per-week tc_bonus/bb_bonus (which we fix to 0 below).
        pts_s = inputs.predictions_per_scenario
        m.tc_bonus_s = Var(m.W, m.S, domain=NonNegativeReals)
        m.bb_bonus_s = Var(m.W, m.S, domain=NonNegativeReals)
        for w in H:
            m.tc_bonus[w].fix(0.0)
            m.bb_bonus[w].fix(0.0)

        def _tc_bonus_s_le_capt(m, w, s):
            return m.tc_bonus_s[w, s] <= sum(
                _capt_ref(m, p, w, s) * pts_s.get((p, w, s), 0.0) for p in m.P
            )
        m.con_tc_bonus_s_le_capt = Constraint(m.W, m.S, rule=_tc_bonus_s_le_capt)

        def _tc_bonus_s_le_z(m, w, s):
            return m.tc_bonus_s[w, s] <= M_TC * m.z_tc[w]
        m.con_tc_bonus_s_le_z = Constraint(m.W, m.S, rule=_tc_bonus_s_le_z)

        def _tc_bonus_s_ge(m, w, s):
            return m.tc_bonus_s[w, s] >= sum(
                _capt_ref(m, p, w, s) * pts_s.get((p, w, s), 0.0) for p in m.P
            ) - M_TC * (1 - m.z_tc[w])
        m.con_tc_bonus_s_ge = Constraint(m.W, m.S, rule=_tc_bonus_s_ge)

        def _bb_bonus_s_le_bench(m, w, s):
            return m.bb_bonus_s[w, s] <= sum(
                (m.x[p, w] - m.y[p, w]) * pts_s.get((p, w, s), 0.0) for p in m.P
            )
        m.con_bb_bonus_s_le_bench = Constraint(m.W, m.S, rule=_bb_bonus_s_le_bench)

        def _bb_bonus_s_le_z(m, w, s):
            return m.bb_bonus_s[w, s] <= M_BB * m.z_bb[w]
        m.con_bb_bonus_s_le_z = Constraint(m.W, m.S, rule=_bb_bonus_s_le_z)

        def _bb_bonus_s_ge(m, w, s):
            return m.bb_bonus_s[w, s] >= sum(
                (m.x[p, w] - m.y[p, w]) * pts_s.get((p, w, s), 0.0) for p in m.P
            ) - M_BB * (1 - m.z_bb[w])
        m.con_bb_bonus_s_ge = Constraint(m.W, m.S, rule=_bb_bonus_s_ge)

    # ── Objective ────────────────────────────────────────────────────────────
    # Main term: (y + c) * pred - rho * eo * (y + c) * pred  (EO-adjusted)
    # Chip bonus: tc_bonus[w] + bb_bonus[w]  (weekly big-M, no EO adjust in v1)
    # Hits: -4 * ht (suppressed under WC/FH via HIT_BIGM)
    # Terminal value V_T applied to squad at last horizon week.
    last_w = H[-1]

    # Terminal value coefficients (per player at horizon-tip)
    if inputs.full_predictions is not None:
        term_coefs = terminal_value_coefficients(
            candidates=P,
            horizon_end_gw=last_w,
            full_predictions=inputs.full_predictions,
            positions=pos,
            sell_prices_tenths=sell,
            alpha=inputs.alpha,
            beta=inputs.beta,
        )
    else:
        term_coefs = dict.fromkeys(P, 0.0)

    def _obj_deterministic(m):
        # Per (player, week): main multiplier mult_main = y + c (XI + captain extra)
        # gets EO-adjusted. Chip bonuses (tc_bonus, bb_bonus) are weekly aggregates
        # added to the objective WITHOUT EO adjustment in v1.0 (small effect; see
        # tc_bonus/bb_bonus comment near var definition).
        rolling = sum(
            (m.y[p, w] + m.c[p, w]) * pred.get((p, w), 0.0)
            for p in m.P
            for w in m.W
        )
        eo_term = sum(
            inputs.rho
            * eo.get(p, 0.0)
            * pred.get((p, w), 0.0)
            * (m.y[p, w] + m.c[p, w])
            for p in m.P
            for w in m.W
        )
        chip_bonus = sum(m.tc_bonus[w] + m.bb_bonus[w] for w in m.W)
        hits = sum(HIT_COST * m.ht[w] for w in m.W)
        terminal = sum(term_coefs.get(p, 0.0) * m.x[p, last_w] for p in m.P)
        return rolling - eo_term + chip_bonus - hits + terminal

    def _obj_saa(m):
        # SAA: average over scenarios. XI is first-stage (single y[p,w]);
        # captain is scenario-conditional via _capt_ref (uses m.c[p,1] for w=1,
        # m.c_s[p,w,s] for w≥2). Per-scenario chip bonuses live in m.tc_bonus_s
        # and m.bb_bonus_s.
        pts_s = inputs.predictions_per_scenario
        S = inputs.scenario_ids
        n_s = len(S)
        # Pre-compute per-(p, w) mean pts: this is what the FIRST-STAGE XI
        # sees (it's a single decision across scenarios, so its objective
        # contribution is the scenario-mean pts).
        mean_pts: dict[tuple[int, int], float] = {}
        for p in m.P:
            for w in m.W:
                mean_pts[(p, w)] = sum(pts_s.get((p, w, s), 0.0) for s in S) / n_s

        rolling_xi = sum(
            m.y[p, w] * mean_pts[(p, w)]
            for p in m.P
            for w in m.W
        )
        eo_xi = sum(
            inputs.rho * eo.get(p, 0.0) * mean_pts[(p, w)] * m.y[p, w]
            for p in m.P
            for w in m.W
        )
        # Captain contribution: per-scenario.
        # For w=1: c[p,1] is first-stage, mean_pts is its objective contribution.
        # For w≥2: c_s[p,w,s] varies per scenario, multiply by pts_s[p,w,s].
        first_w = H[0]
        rolling_capt = sum(
            m.c[p, first_w] * mean_pts[(p, first_w)] for p in m.P
        )
        eo_capt = sum(
            inputs.rho * eo.get(p, 0.0) * mean_pts[(p, first_w)] * m.c[p, first_w]
            for p in m.P
        )
        for w in m.W:
            if w == first_w:
                continue
            for s in S:
                for p in m.P:
                    pts_pws = pts_s.get((p, w, s), 0.0)
                    rolling_capt += (1.0 / n_s) * m.c_s[p, w, s] * pts_pws
                    eo_capt += (1.0 / n_s) * inputs.rho * eo.get(p, 0.0) * pts_pws * m.c_s[p, w, s]

        # Per-scenario chip bonus averaged (only when chips are enabled —
        # otherwise tc_bonus_s / bb_bonus_s aren't built).
        if inputs.enable_chips:
            chip_bonus_saa = sum(
                (1.0 / n_s) * (m.tc_bonus_s[w, s] + m.bb_bonus_s[w, s])
                for w in m.W
                for s in S
            )
        else:
            chip_bonus_saa = 0.0
        hits = sum(HIT_COST * m.ht[w] for w in m.W)
        terminal = sum(term_coefs.get(p, 0.0) * m.x[p, last_w] for p in m.P)
        return rolling_xi + rolling_capt - eo_xi - eo_capt + chip_bonus_saa - hits + terminal

    if inputs.use_saa:
        m.objective = Objective(rule=_obj_saa, sense=maximize)
    else:
        m.objective = Objective(rule=_obj_deterministic, sense=maximize)

    return m


def _cbc_binary_path() -> str:
    """Resolve the CBC binary bundled with `pulp` (added to deps for Phase 4
    solver swap). Caching one-off; cheap to look up each call."""
    import pulp
    return pulp.PULP_CBC_CMD().path


def solve_milp(
    model: ConcreteModel,
    *,
    time_limit_s: int = 120,
    mip_rel_gap: float = 0.01,
) -> dict:
    """Solve the MILP via CBC (subprocess on LP file). Returns metadata.

    Accepts feasible-but-not-optimal solutions (we load the incumbent into
    the Pyomo model). Only fails if no feasible solution was found.

    **Why CBC and not HiGHS**: Phase 4 SAA at |S|=10+ triggered HiGHS
    SIGABRTs / heap corruption (documented across v1.4 / Phase 4 v0.2).
    The bug is inside the HiGHS C++ library at v1.14.0, not the wrapper.
    Gurobi's free license caps at 2000 vars (our SAA models are ~20k+).
    CBC is a decades-stable open-source solver, bundled with PuLP, and
    invoked via subprocess so any internal state is fully isolated per
    solve. Slower than HiGHS optimal but rock-solid.
    """
    import contextlib
    import os
    import re
    import subprocess
    import tempfile

    from pyomo.repn.plugins.lp_writer import LPWriter

    with tempfile.NamedTemporaryFile(
        suffix=".lp", delete=False, mode="w"
    ) as f:
        lp_path = f.name
        writer = LPWriter()
        write_info = writer.write(model, f, symbolic_solver_labels=True)
    sm = write_info.symbol_map

    sol_path = lp_path + ".sol"

    try:
        cbc_path = _cbc_binary_path()
        # CBC CLI: timeLimit + ratio (gap), solve, write solution.
        cmd = [
            cbc_path,
            lp_path,
            "seconds",
            str(time_limit_s),
            "ratio",
            str(mip_rel_gap),
            "solve",
            "solu",
            sol_path,
            "quit",
        ]
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=float(time_limit_s) + 30.0,
        )
        if result.returncode != 0 or not os.path.exists(sol_path):
            return {
                "status": "no_feasible_solution",
                "termination": f"cbc_rc={result.returncode}",
                "objective": None,
            }

        # Parse CBC solution format:
        #   Header line: "Optimal - objective value 30.0"
        #               or "Stopped on time - objective value ..."
        #               or "Infeasible"
        # Then: each var on its own line: "<idx> <name> <value> <reduced_cost>"
        with open(sol_path) as f:
            lines = f.read().splitlines()
        if not lines:
            return {
                "status": "no_feasible_solution",
                "termination": "empty_sol_file",
                "objective": None,
            }
        header = lines[0]
        m_obj = re.search(r"objective value (-?\d+\.?\d*)", header)
        objective = float(m_obj.group(1)) if m_obj else None
        if "Infeasible" in header or "Unbounded" in header or objective is None:
            return {
                "status": "no_feasible_solution",
                "termination": header.split(" -")[0].strip(),
                "objective": None,
            }
        is_optimal = "Optimal" in header
        termination = "optimal" if is_optimal else "feasible"

        # Parse col values (each line: "<idx> <name> <value> <rc>")
        col_values: dict[str, float] = {}
        for line in lines[1:]:
            parts = line.split()
            if len(parts) < 3:
                continue
            try:
                _ = int(parts[0])
            except ValueError:
                continue
            name = parts[1]
            try:
                val = float(parts[2])
            except ValueError:
                continue
            col_values[name] = val

        # Map CBC column values back to Pyomo Vars via symbol_map. CBC may
        # return LP-solve numerics (e.g. 0.9999... for a binary 1); round
        # to the var's domain so Pyomo doesn't W1001-warn and downstream
        # `value > 0.5`-style threshold checks behave as expected. CBC
        # omits vars with value 0 from the solution file, so default to 0.
        from pyomo.core.base.var import VarData
        from pyomo.environ import Binary, NonNegativeIntegers
        for sym_name, obj_ref in sm.bySymbol.items():
            if not isinstance(obj_ref, VarData):
                continue
            raw = col_values.get(sym_name, 0.0)
            dom = obj_ref.domain
            if dom is Binary:
                obj_ref.value = 1 if raw > 0.5 else 0
            elif dom is NonNegativeIntegers:
                obj_ref.value = round(raw)
            else:
                obj_ref.value = raw

        return {
            "status": "ok",
            "termination": termination,
            "objective": objective,
        }
    except subprocess.TimeoutExpired:
        return {
            "status": "no_feasible_solution",
            "termination": "subprocess_timeout",
            "objective": None,
        }
    finally:
        with contextlib.suppress(OSError):
            os.unlink(lp_path)
        with contextlib.suppress(OSError):
            os.unlink(sol_path)


def extract_decisions(
    model: ConcreteModel,
    inputs: MilpInputs,
) -> GwDecisions:
    """Pull the first-week decisions out of the solved MILP.

    Phase 3 v1 is greedy-rolling — we apply ONLY the first-week decisions and
    re-solve next GW with updated state.
    """
    first_w = inputs.horizon_weeks[0]

    squad_first = frozenset(p for p in inputs.candidates if value(model.x[p, first_w]) > 0.5)
    xi = frozenset(p for p in inputs.candidates if value(model.y[p, first_w]) > 0.5)
    capts = [p for p in inputs.candidates if value(model.c[p, first_w]) > 0.5]
    vices = [p for p in inputs.candidates if value(model.v[p, first_w]) > 0.5]
    captain = capts[0] if capts else None
    vice = vices[0] if vices else None
    transferred_in = frozenset(
        p for p in inputs.candidates if value(model.bin[p, first_w]) > 0.5
    )
    transferred_out = frozenset(
        p for p in inputs.candidates if value(model.bout[p, first_w]) > 0.5
    )

    chip_played: str | None = None
    if value(model.z_wc[first_w]) > 0.5:
        chip_played = "WC1" if _gw_in_first_half(first_w) else "WC2"
    elif value(model.z_fh[first_w]) > 0.5:
        chip_played = "FH1" if _gw_in_first_half(first_w) else "FH2"
    elif value(model.z_bb[first_w]) > 0.5:
        chip_played = "BB1" if _gw_in_first_half(first_w) else "BB2"
    elif value(model.z_tc[first_w]) > 0.5:
        chip_played = "TC1" if _gw_in_first_half(first_w) else "TC2"

    hits = round(value(model.ht[first_w]))

    return GwDecisions(
        gameweek=first_w,
        squad=squad_first,
        starting_xi=xi,
        captain=captain,
        vice=vice,
        transferred_in=transferred_in,
        transferred_out=transferred_out,
        chip_played=chip_played,
        hits=hits,
        objective_value=float(value(model.objective)),
    )


def solve_rolling_horizon(
    inputs: MilpInputs,
    *,
    time_limit_s: int = 120,
) -> tuple[GwDecisions, dict]:
    """End-to-end: build, solve, extract first-week decisions."""
    model = build_milp(inputs)
    meta = solve_milp(model, time_limit_s=time_limit_s)
    if meta["termination"] not in ("optimal", "feasible"):
        raise RuntimeError(f"MILP failed: {meta}")
    decisions = extract_decisions(model, inputs)
    return decisions, meta
