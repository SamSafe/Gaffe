# Phase 3 — Deterministic Rolling-Horizon MILP

Status: **v1.0 shipped.** Implementation complete; all validity gates pass on fold 2024/25.

Round-1 review resolutions:
- Static prices for v1; price-change predictor deferred to Phase 3.5.
- Pyomo + HiGHS; no solver change needed.
- Horizon: **4 GWs** for v1.0 (cut from 6 for solver tractability — see §1 v1.0 simplifications).
- Walk-forward folds 21-24 with separated validity vs performance gates.
- v1.0 implementation simplifications (forced by solver-time constraints):
  - **Chips disabled** (z_wc/z_fh/z_bb/z_tc all pinned to 0). Bilinear chip auxiliaries roughly doubled MILP size and caused time-limit hits at H=6 + 200 candidates. Chips deferred to Phase 3.5 alongside the season-long chip-DP from Phase 5.
  - **Cold-start GW1 uses H=1** (not H=4) — picks the squad freely with terminal value providing 5-GW lookahead. Reduces cold-start MILP size by ~4×.
  - **Candidate pool reduced** to top-25-per-position + cheap-K-per-position + PK takers + current squad (~125 candidates per solve).
  - **Feasible-but-not-optimal solutions accepted** when solver hits time limit; only "no feasible found" is a failure.

Per Phase 0 §5: solve a rolling-horizon MILP that consumes the Phase 2.5 joint-xPts predictions, the EO data layer, and the chip rules; outputs squad / starting XI / captain-vice / transfer / chip decisions over the horizon, re-solved each gameweek.

This phase is the first architectural pivot from prediction to optimization. v1 is **deterministic** — uses E[xPts] from Phase 2.5 directly. Phase 4 will add stochastic scenario sampling.

---

## 1. Scope and v1 simplifications (review-confirmed)

| Decision | v1 choice | Rationale |
|---|---|---|
| **Solver** | Pyomo + HiGHS | Phase 0 spec; switch only if a concrete issue surfaces |
| **Horizon** | 6 gameweeks default, configurable to 8 | Keeps solve times short during development |
| **Prices** | **Static** (current/last-known buy and sell price held constant over horizon) | FPL price-change algorithm is partially opaque; deferring price prediction to Phase 3.5 |
| **Predictions** | Mean E[xPts] from `data/predictions/xpts/` (Phase 2.5 output) | No per-scenario sampling; that's Phase 4 |
| **EO** | `fpl_api_approx` (overall ownership × captain-pct heuristic) | LiveFPL/FPLStatistics are deferred; round-2 fallback |
| **Objective** | EO-adjusted xPts (Stage A) with ρ tunable | Round-1 confirmed; ρ default 1.0, sweep {0.7, 1.0, 1.2} |
| **Chips** | Integrated as decision variables | Per Phase 0 §5; chip-timing DP is Phase 5 (soft-coupled) |
| **Chip rules** | 2024/25 ruleset (banked FT cap of 5, second WC + second FH per season) | Round-2 confirmed |

---

## 2. Sets and parameters

### Sets
- $P$: players in scope (filtered to candidate set; see §10)
- $T$: teams (1..20 per season)
- $W = \{1, \ldots, H\}$: gameweeks in horizon (default $H=6$)
- $\Pi = \{\text{GKP}, \text{DEF}, \text{MID}, \text{FWD}\}$: positions

### Parameters (per (player, gameweek))
- $\text{pts}_{p,w}$: predicted E[xPts] from Phase 2.5
- $\text{eo}_{p,w}$: effective ownership (overall × captain-pct)
- $\text{price}^{\text{buy}}_p, \text{price}^{\text{sell}}_p$: static for v1 (constant across $W$)
- $\text{pos}_p \in \Pi$: from `dim_player` / `fact_player_status`
- $\text{team}_p$: stable team_id for that player in the current season

### State (carried between rolling-horizon solves)
- $\text{ft}_0$: free transfers entering the first GW of the horizon
- $\text{bank}_0$: money in bank entering the horizon
- $\text{squad}_0 \subseteq P$: 15-player squad entering the horizon
- $\text{chips\_used}$: subset of $\{\text{WC}_1, \text{WC}_2, \text{FH}_1, \text{FH}_2, \text{BB}, \text{TC}\}$

---

## 3. Decision variables

| Variable | Domain | Meaning |
|---|---|---|
| $x_{p,w}$ | $\{0,1\}$ | $p$ in 15-man squad in GW $w$ |
| $y_{p,w}$ | $\{0,1\}$ | $p$ in starting XI in GW $w$ |
| $c_{p,w}$ | $\{0,1\}$ | $p$ captained |
| $v_{p,w}$ | $\{0,1\}$ | $p$ vice-captained |
| $b^{\text{in}}_{p,w}, b^{\text{out}}_{p,w}$ | $\{0,1\}$ | transferred in / out at start of $w$ |
| $\text{ft}_w$ | $\{0,1,\ldots,5\}$ | free transfers banked entering $w$ |
| $\text{ht}_w$ | $\mathbb{Z}_{\geq 0}$ | hits taken in $w$ |
| $\text{bank}_w$ | $\mathbb{R}_{\geq 0}$ | money in bank entering $w$ (in tenths of millions) |
| $z^{\text{WC}}_w, z^{\text{FH}}_w, z^{\text{BB}}_w, z^{\text{TC}}_w$ | $\{0,1\}$ | chip activations |

Total ≈ $|P| \times H \times 6$ binary + a handful of integer/continuous = ~22k binary for $|P|=600, H=6$. HiGHS should handle.

---

## 4. Constraints

### Squad shape
$\sum_{p \in P} x_{p,w} = 15 \quad \forall w$ ; per-position counts (2/5/5/3 GKP/DEF/MID/FWD) ; per-team $\leq 3$.

### Starting XI
$\sum_p y_{p,w} = 11$ ; $y_{p,w} \leq x_{p,w}$ ; valid formations (1 GKP, ≥3 DEF, ≥2 MID, ≥1 FWD).

### Captain / vice
$\sum_p c_{p,w} = 1$ ; $\sum_p v_{p,w} = 1$ ; $c_{p,w} + v_{p,w} \leq 1$ ; $c, v \leq y$.

### Transfer balance
$x_{p,w} - x_{p,w-1} = b^{\text{in}}_{p,w} - b^{\text{out}}_{p,w}$ ; $b^{\text{in}}_{p,w} + b^{\text{out}}_{p,w} \leq 1$.

### Free-transfer evolution and hits
$\text{ft}_w = \min(5, \text{ft}_{w-1} + 1 - \sum_p b^{\text{out}}_{p,w-1})$ (linearized via auxiliaries).
$\text{ht}_w \geq \sum_p b^{\text{out}}_{p,w} - \text{ft}_w$ ; $\text{ht}_w \geq 0$.

### Budget
$\sum_p \text{price}^{\text{buy}}_p \cdot x_{p,w} + \text{bank}_w = 1000 + \text{capital\_gains}_w$ (tenths-of-millions; £100m initial budget).
$\text{bank}_w = \text{bank}_{w-1} + \sum_p \text{price}^{\text{sell}}_p \cdot b^{\text{out}}_{p,w} - \sum_p \text{price}^{\text{buy}}_p \cdot b^{\text{in}}_{p,w}$.

### Chip semantics
- $\sum_w z^{\text{WC}}_w \leq 1$ per WC slot (two slots per season; first half ≤ GW19, second ≥ GW20). Same for FH.
- $\sum_w z^{\text{BB}}_w \leq 1$, $\sum_w z^{\text{TC}}_w \leq 1$ per season.
- At most one chip per GW: $z^{\text{WC}}_w + z^{\text{FH}}_w + z^{\text{BB}}_w + z^{\text{TC}}_w \leq 1$.
- WC: $\text{ht}_w \cdot (1 - z^{\text{WC}}_w)$ enters objective (suppresses hits this week; big-M linearization).
- FH: temporary squad change reverts to $w-1$ squad next week (separate $x^{\text{FH}}_{p,w}$ parallel variable when $z^{\text{FH}}_w = 1$).
- BB: bench scores too — multiplier becomes $\text{mult}_{p,w} = y_{p,w} + (x_{p,w} - y_{p,w}) \cdot z^{\text{BB}}_w + c_{p,w}$.
- TC: captain multiplier becomes 3 — linearize $c_{p,w} \cdot z^{\text{TC}}_w$ via auxiliary.

---

## 5. Objective (deterministic)

$$
\max \;\; \sum_{w \in W} \sum_{p \in P} \big(\text{mult}_{p,w} - \rho \cdot \text{eo}_{p,w}\big) \cdot \text{pts}_{p,w}
\;-\; 4 \cdot \sum_w \text{ht}_w \cdot (1 - z^{\text{WC}}_w)
\;+\; V_T(s_H)
$$

where
$$
\text{mult}_{p,w} = y_{p,w} + c_{p,w} + 2 \cdot c_{p,w} \cdot z^{\text{TC}}_w + (x_{p,w} - y_{p,w}) \cdot z^{\text{BB}}_w
$$
(captain doubles to 2× by default and 3× under TC; bench scores under BB).

Bilinear terms $c \cdot z^{\text{TC}}$ and $(x-y) \cdot z^{\text{BB}}$ linearized with auxiliary binaries and big-M.

ρ tunable: default **1.0**; backtest sweep over **{0.7, 1.0, 1.2}** (Stage A per round-1 confirmation; Stage B threshold-chasing deferred to Phase 4).

---

## 6. Terminal value $V_T(s_H)$

Per Phase 0 §5.5:
$$
V_T(s_H) = \alpha \sum_p x_{p,H} \cdot \widehat{\text{pts}}_{p,\,H+1:H+5} \;+\; \beta \sum_p \text{price}^{\text{sell}}_p \cdot x_{p,H}
$$

- $\widehat{\text{pts}}_{p, H+1:H+5}$: simplified 5-GW post-horizon forecast. Use the same Phase 2.5 predictions for any future fixtures available; for fixtures beyond what's modeled, fall back to position-mean × fixture-difficulty-adjusted xPts/90.
- $\alpha, \beta$: tuned via backtest. Initial guesses: $\alpha = 0.8$, $\beta = 0.05$ (per-£1m-of-squad-value). Sweep both in {0.5, 0.8, 1.0} × {0.0, 0.05, 0.1} on one fold, pick by total-points lift.

---

## 7. Backtest structure

Walk-forward by season (test on 21/22, 22/23, 23/24, 24/25). Within each test season:

1. **GW1 cold start**: solve a one-shot 15-pick MILP with budget 1000, no transfers, 1 free transfer, no chips used.
2. **Per GW (2..38)**:
   - Read state (squad, bank, ft, chips_used) from end of previous GW
   - Solve rolling 6-GW MILP starting from current GW with current state
   - Apply first-GW decisions (transfers, lineup, captain, chip)
   - Run actual fixtures (use realized minutes/goals/assists/CS from `fact_player_match`; recompute actual points using same FPL rule table from Phase 2.5)
   - Update state: bank ← bank + (sells - buys); ft ← min(5, ft + 1 - transfers_used); mark chip used; squad ← squad + transfers
3. **End-of-season**: total points = sum of (lineup_points + captain_bonus_points − hit_points) across 38 GWs.

This is rolling backtest, not one-shot full-season optimization.

### Backtest performance projection

- Per-GW MILP solve: ~10-60s with HiGHS at $|P|=200$ candidates after prefilter (see §10)
- 38 GWs × 4 folds × ~30s = ~75 min per (ρ, α, β) configuration. Tractable for a small grid sweep; heavy if we expand.

---

## 8. Acceptance gate before Phase 4

Two distinct gate categories per round-1 review (separate to avoid distortions from static-price / FWD / no-news limitations):

### Optimizer validity gates (ALL MUST PASS) — fold 2024/25

- [x] **Feasibility**: every per-GW MILP solve returns a valid solution. **38/38 solves succeeded, mean 2.0s per solve.**
- [x] **Budget correctness**: cost + bank conservation holds on every solved GW. **Pass after fixing apply_gw_outcomes to use current price (not cost basis) for sells under static-price v1 — matches what the MILP solves against.**
- [x] **Transfer accounting**: hits = max(0, transfers_in - free_transfers); FT evolution caps at 5. **Pass on all 38 GWs.**
- [x] **Chip legality**: chips disabled in v1.0; trivially passes.
- [x] **No future leakage**: predictions from xpts_eval walk-forward CV; prices walk only BACKWARD via `_resolve_price_at_gw`. **Pass; covered by 5 leakage tests in `tests/leakage/test_milp_leakage.py`.**

> **Architecture note on optim/ vs leakage discipline.** The static-import leakage audit applies to `features/`, `models/`, `scenarios/` — these consume time-sensitive data and must route through PIT. `optim/` is outside guarded packages because its DB reads are current-snapshot static metadata (current prices for the GW being solved, current positions, current EO via fpl_api_approx). The MILP itself processes already-PIT-routed predictions; runtime leakage risk is in price lookup, fixed by `_resolve_price_at_gw` walking only backward.

### Performance metrics (REPORTED, NOT GATED INDIVIDUALLY) — fold 2024/25

| Metric | Value |
|---|---|
| MILP total points | **2318** |
| Buy-and-hold-template | 1915 |
| **MILP - template** | **+403 (+21.0%)** |
| GWs solved | 38 / 38 |
| Mean solve time | 2.0 s |
| Total transfers | 52 |
| Hits taken | 0 |
| Chips played | none (disabled in v1.0) |

The MILP outperforms the buy-and-hold-template baseline by 21% on a single fold, which is meaningful even given the known prediction-layer limitations (FWD under-prediction, no live-only features, static prices). Full walk-forward across 4 folds + ρ/α/β tuning is the remaining tuning work for Phase 3.0 finalization (deferred to a follow-up commit).

**Phase 4 readiness signal**: validity gates pass. Phase 4 (stochastic) can proceed in parallel with Phase 3.5 (chip-DP coupling, dynamic prices, full ρ/α/β tuning).

---

## 9. Storage / API

Per-GW solve output saved as Parquet under `data/optim/milp/{run_ts}/season_{N}/gw_{W}.parquet`:

| Column | Description |
|---|---|
| `player_id` | stable FPL code |
| `gameweek` | GW the decisions apply to |
| `in_squad` | bool |
| `in_xi` | bool |
| `is_captain`, `is_vice` | bool |
| `transferred_in`, `transferred_out` | bool |
| `objective_contribution` | float (per-player contribution to objective for accounting) |

Plus a header file `gw_{W}_meta.json`:
```json
{
  "objective_value": <float>,
  "bank_after": <int>,
  "ft_after": <int>,
  "chips_used_this_gw": ["WC" | "FH" | ...],
  "ht_count": <int>,
  "solve_time_s": <float>,
  "solve_status": "optimal" | "feasible" | ...
}
```

Public API:
```python
from fpl_bot.optim.milp import solve_rolling_horizon, BacktestState

decisions, new_state = solve_rolling_horizon(
    state: BacktestState,
    predictions: pl.DataFrame,
    horizon: int = 6,
    rho: float = 1.0,
)
```

---

## 10. Code structure

```
src/fpl_bot/
├── optim/
│   ├── milp.py                 # core MILP build + solve
│   ├── state.py                # BacktestState dataclass + transitions
│   ├── terminal_value.py       # V_T computation
│   ├── candidate_filter.py     # prefilter to ~200 candidate players (see below)
│   └── eo.py                   # `fpl_api_approx` EO computation from fact_player_status
└── eval/
    └── milp_backtest.py        # rolling backtest harness; emits validity + perf metrics
tests/
├── unit/
│   ├── test_milp_constraints.py    # constraint correctness on small synthetic problems
│   └── test_state_transitions.py   # FT evolution, bank update, chip flagging
└── leakage/
    └── test_milp_leakage.py        # solver inputs only contain pre-deadline data
```

### Candidate filter

Naive $|P| \approx 600$ produces a large MILP. Heuristic prefilter to ~200 candidates per solve:
- Top 50 by E[xPts] per position
- Plus current squad members (ensures rebuild moves are feasible)
- Plus all penalty takers and known set-piece designators
- Plus 5-deepest-price-tier per position (cheap enablers)

This loses optimality vs. global $|P|$ but keeps solve times tractable. Validity gates verify feasibility — if the prefilter ever rejects all candidates for a position, the next iteration uses full $|P|$.

---

## 11. Open questions for review

1. **Chip-DP integration scope.** Phase 0 §6 specifies a separate season-long chip-timing DP feeding shadow values into the rolling MILP. v1 ships chips as decision variables in the rolling MILP **without** the DP coupling (chip-timing decisions made greedily within the 6-GW horizon). The chip-DP is Phase 5. Confirm OK.
2. **Initial squad for backtest GW1.** Solve a "free-rein" 15-pick MILP at GW1 with no current squad? Or seed from a real top-10k template if obtainable? v1 default: free-rein MILP. The template-seeding option is a follow-up if backtest results are weird in early-season.
3. **Top-10k EO data freshness.** Currently only `fpl_api_approx`. Phase 4 cares more about EO accuracy than Phase 3 (the marginal-rank objective is robust to small EO errors). v1 accepts the approximation. Re-evaluate when LiveFPL becomes available again.
4. **Backtest computational scope.** 38 GWs × 4 folds × ~30s = ~75 min per (ρ, α, β) config. Sweep size for v1: only ρ on the cleanest fold (24/25), then full backtest at the picked ρ. α, β tuning grid only on 24/25 unless a clear signal emerges. Confirm acceptable.

---

## 12. Visible post-v1 follow-ups (do NOT block Phase 3)

These were called out in the Phase 0–2 review and remain visible because each one materially affects Phase 3 squad/captain/bench decisions:

1. **Multinomial / team-score-conditioned goal sampling.** Priority follow-up. Independent goal sampling under-predicts within-fixture *concentration*, which hurts FWD bonus and captaincy at the top end. The MILP's captaincy and FWD-pick decisions are downstream of this. Not blocking Phase 3 but re-running Phase 2.4/2.5 with the fix would tighten Phase 3 numbers.

2. **Cameo prior in the minutes model.** Phase 2.5 D1 anomaly: predicted ~0% playing for genuine benchwarmers but cameos do happen. A small additive $p_{\text{short}}$ floor per position would fix this. The MILP's bench decisions (esp. for chip planning — Bench Boost values bench depth) are downstream.

3. **Price-change predictor (Phase 3.5).** v1 uses static prices. Backtests over 38 GWs with static prices systematically under-value rising-price assets and over-value falling-price assets. Phase 3.5 builds the predictor from `transfers_in/out` data we already have in vaastav.

4. **Live-only features (status_code, news, chance_of_playing).** Available at predict time, not in training data. Currently a soft-override path at predict time. Could be quantified once `fact_player_status` accumulates multiple snapshots over the live season.

5. **EO data quality (LiveFPL/FPLStatistics scraping).** `fpl_api_approx` is the production EO source until a clean public endpoint emerges. Phase 4 stochastic optimization is more sensitive to this than Phase 3.

---

**End of Phase 3 design. Awaiting review.**
