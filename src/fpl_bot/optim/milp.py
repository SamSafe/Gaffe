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
    SolverFactory,
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
    def _one_captain(m, w):
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
                elif slot == "BB":
                    m.z_bb[w].setub(0)
                elif slot == "TC":
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
        elif slot == "BB":
            m.add_component(
                "con_bb_once",
                Constraint(expr=sum(m.z_bb[w] for w in H) <= 1),
            )
        elif slot == "TC":
            m.add_component(
                "con_tc_once",
                Constraint(expr=sum(m.z_tc[w] for w in H) <= 1),
            )

    if inputs.enable_chips:
        for slot in ALL_CHIP_SLOTS:
            _build_per_slot_constraint(m, slot)

    # Weekly chip-bonus big-M linearization — only added when chips are enabled.
    # M_TC: safe upper bound on a single player's xPts in one GW (typical max ~15).
    # M_BB: safe upper bound on bench (4 players) xPts in one GW (typical max ~30).
    M_TC = 30.0
    M_BB = 60.0
    if inputs.enable_chips:
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

    def _obj(m):
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

    m.objective = Objective(rule=_obj, sense=maximize)

    return m


def solve_milp(
    model: ConcreteModel,
    *,
    time_limit_s: int = 120,
    mip_rel_gap: float = 0.01,
) -> dict:
    """Solve the MILP. Returns metadata about the solve.

    Accepts feasible-but-not-optimal solutions (e.g., on time-limit hit) —
    HiGHS will have loaded the incumbent. Only fails if no feasible solution
    was found at all.

    `mip_rel_gap` (default 1%): HiGHS stops as soon as it has an incumbent
    within this relative gap of the LP relaxation upper bound. The Phase 2.5
    independent-goal sampler produced predictions where the LP relaxation
    had loose ties; HiGHS could prove optimality at the default gap. Phase
    2.5.1's multinomial sampler tightens within-team prediction ties, making
    the integer search much harder to prove optimal — but a 1% gap is well
    below our prediction noise (the rho/eo/lambda parameters carry far more
    uncertainty than 1% of objective). 1% solves an order of magnitude
    faster on the tied folds.
    """
    solver = SolverFactory("appsi_highs")
    # appsi_highs ignores `solver.options[...]` for HiGHS-specific keys; pass
    # them through `highs_options` instead. (Discovered after `solver.options
    # ["mip_rel_gap"] = 0.01` was silently ignored.)
    solver.highs_options["time_limit"] = float(time_limit_s)
    solver.highs_options["mip_rel_gap"] = float(mip_rel_gap)
    solver.config.load_solution = False
    result = solver.solve(model, tee=False)

    termination = str(result.solver.termination_condition)
    found_feasible = (
        result.solver.best_feasible_objective is not None
        if hasattr(result.solver, "best_feasible_objective")
        else False
    )

    if termination == "optimal" or found_feasible:
        # Load the (optimal or incumbent) solution into the model
        solver.load_vars()
        return {
            "status": "ok",
            "termination": termination if termination == "optimal" else "feasible",
            "objective": float(value(model.objective)),
        }
    return {
        "status": "no_feasible_solution",
        "termination": termination,
        "objective": None,
    }


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
        chip_played = "BB"
    elif value(model.z_tc[first_w]) > 0.5:
        chip_played = "TC"

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
