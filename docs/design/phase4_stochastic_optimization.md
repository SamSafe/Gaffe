# Phase 4 — Stochastic Optimization (Design)

Status: **v0.1 scaffolding shipped; full walk-forward blocked on HiGHS stability (Phase 3 v1.4 issue re-amplified).**

§9 game-mechanics audit confirms Phase 3 MILP correctly models all transfer / hit / chip / FT / budget / squad / XI / captain / DGW / BGW / cost-basis mechanics. Phase 4 inherits all this unchanged — it's purely an objective-layer pivot.

**What shipped**:
1. Raw-samples dump pipeline (`xpts_eval.run_fold_with_raw_samples`) — all 4 walk-forward folds cached at `data/cache/xpts_raw_samples/`.
2. Scenario loader (`optim/scenarios.py`) with PIT-correct GW aggregation and deterministic subsampling.
3. MILP SAA branch (`optim/milp.py` `use_saa=True`): scenario-conditional captain `c_s[p,w,s]` for w≥2, first-stage `c[p,1]` linked via non-anticipativity, per-scenario big-M chip bonuses `tc_bonus_s` / `bb_bonus_s`.
4. `backtest_season` wired with `use_saa` / `n_scenarios` parameters.
5. Subprocess-isolated walk-forward script (`scripts/phase4_saa_walkforward.py`).

**Tractability sweep** (fold 21 GW2, cold-start state populated separately):

| |S| | chips off | chips on |
|---:|---:|---:|
| 5   | 8.7s | 24.6s |
| 10  | 11.3s | **9.3s** |
| 25  | 10.7s | 89.9s |

|S|=10 chips-on is comparable to Phase 3 deterministic. Single-solve correctness verified.

**v1.0 update** (CBC swap, ships): replaced HiGHS entirely with CBC via subprocess. CBC is bundled with `pulp` (`pulp.PULP_CBC_CMD().path` resolves the binary). On Phase 5 deterministic + chip-DP fold 24, CBC mean solve = 1.3s vs HiGHS ~3.5s — **2.5× faster**. Crucially, CBC handles SAA at |S|=25 without crashes (HiGHS 1.14.0 SIGABRT'd at |S|≥10 regardless of subprocess isolation). The earlier `_highspy_worker.py` is removed.

**v0.2 attempt** (kept for history): switched from Pyomo `appsi_highs` to `highspy` direct via LP-file roundtrip with subprocess-per-solve isolation. Still SIGABRTed on specific MILPs — confirming the bug is in HiGHS itself, not the wrapper. Superseded by CBC swap above.

**v1.0 SAA walk-forward at |S|=25 + chip-DP + CBC** (the production-scale Phase 4 test, unblocked by the CBC swap):

| Fold | SAA |S|=25 | Phase 5 v1 det. | Δ |
|---|---:|---:|---:|
| 21/22 | 2428 | 2555 | **−127** |
| 22/23 | 2370 | 2402 | **−32** |
| 23/24 | 2353 | 2491 | **−138** |
| 24/25 | 2428 | 2584 | **−156** |

All 5 validity gates pass on all 4 folds. Mean solve 7-11s/GW.

**Honest result**: SAA |S|=25 is **−113 pts/fold WORSE** than deterministic. More scenarios did help vs |S|=3 (avg +116/fold improvement) — Jensen's inequality bites correctly — but the **deterministic argmax of `E[xPts]` already captures most of the captain-decision value**. The scenario-conditional second-stage (captain only at w≥2) doesn't deliver enough additional optimization power to overcome SAA's noise.

Earlier |S|=3 baseline (HiGHS-blocked context):

| Fold | SAA |S|=3 | SAA |S|=25 | Δ (more scenarios) |
|---|---:|---:|---:|
| 21/22 | 2359 | 2428 | +69 |
| 22/23 | 2104 | 2370 | +266 |
| 23/24 | 2392 | 2353 | −39 |
| 24/25 | 2258 | 2428 | +170 |

Phase 4 v1.0 SAA does not ship as production default. **CBC swap stays** (real improvement: 2.5× faster + bulletproof stability). SAA scaffolding stays on disk for future v2 (scenario-conditional XI for w≥2, true mean-variance objective, or chance-constrained Stage B).

**Why scenario-conditional captain alone underperforms**:
1. Captain choice is **dominated** by 1-2 obvious picks per GW (high mean + high variance). The MILP's deterministic E[xPts] argmax already picks these — there's no untapped optionality at the captain level.
2. The constraint `c[p,w,s] ≤ y[p,w]` requires the captain to be in the **first-stage XI**. Without scenario-conditional XI (too expensive in MILP size), captain-only flexibility is bounded.
3. SAA introduces objective noise proportional to 1/√|S|. At |S|=25 noise ≈ 20% of one scenario's range, which can swamp the modest captain-hedging gain.

**Path forward (post-v1.0)**: a TRUE Phase 4 win needs either (a) scenario-conditional XI for w≥2, or (b) a non-linear aggregation (CVaR, threshold chance-constraint) — excluded by round-1 review.

**v2 attempt (scenario-conditional XI for w≥2 + captain)**: tested on fold 24 with CBC at |S|=25 + chip-DP. Implemented per phase4 §2.3-2.5 design: `y_s[p,w,s]` and `c_s[p,w,s]` for w≥2, with full XI shape constraints per (w,s); non-anticipativity for w=1; BB bonus uses scenario-conditional XI. Result:

| | total pts | mean solve | wall |
|---|---:|---:|---:|
| Deterministic + chip-DP | **2584** | 1.3s | 49s |
| SAA v1 (capt-only) \|S\|=25 | 2428 | 7.3s | ~5min |
| SAA v2 (XI+capt) \|S\|=25 | **2428** | 48.3s | 31min |

**v2 gives the same total points as v1 but is 7× slower**. The MILP's optimal squad/XI/captain doesn't change when given XI scenario-flexibility. Why: the deterministic argmax of E[xPts] already finds the "obvious" XI, and the chip-DP fixes WC/FH/BB/TC slots → there's effectively no untapped within-week flexibility for SAA to exploit. The constraint `y_s[p,w,s] ≤ x[p,w]` (XI ⊆ first-stage squad) means the XI per scenario can only re-permute already-bought players, which the deterministic mean-based pick has already optimized to.

**Both v1 and v2 are reverted from production.** Phase 4 SAA scaffolding stays on disk; the only meaningful follow-up is option (b) — a non-linear aggregation like CVaR — which round-1 explicitly excluded. Marking Phase 4 v1.0 as **architecturally complete, no production gain available at this scope**.

Replace the deterministic E[xPts] objective in the Phase 3 MILP with a sample-average approximation (SAA) over scenarios drawn from the Phase 2.5 joint-xPts simulator. This is the architectural pivot called out in Phase 0 §5: same MILP structure, but the objective becomes an average over |S| scenarios, and decisions can be split into first-stage (current GW, non-anticipative) and second-stage (future GWs, optionally scenario-conditional).

Per Phase 0 §1.2 and the round-1 review: **only Stage A** (EO-adjusted SAA) ships in v1. Stage B (threshold-chasing late-season `max P(season_total ≥ τ)`) is deferred — to be revisited if late-season backtests show insufficient variance-seeking.

---

## 1. Scope and v1 simplifications

| Decision | v1 choice | Rationale |
|---|---|---|
| **Objective** | EO-adjusted SAA (Stage A only) | Per round-1 review. Stage B deferred. |
| **Scenarios** | 50 per rolling solve (from a pool of 200 raw samples per (player, GW)) | 50 is a tractable starting point; scale to 100-200 only if MILP solve times allow. |
| **First-stage decisions** | This-GW decisions only (squad, XI, captain, transfers, chip) | Per Phase 0 §5.4. Non-anticipative — same across all scenarios. |
| **Second-stage decisions** | **Scenario-conditional CAPTAIN only** for w ≥ 2 (`c[p,w,s]`); XI stays first-stage (`y[p,w]`); first-stage `c[p,1]` shared | **Critical correction**: aggregated second-stage with linear objective is mathematically IDENTICAL to deterministic Phase 3 with scenario means — no value-add. The captain is the highest-variance lever in FPL (TC + 2× mult); scenario-conditional captain hedges that variance directly. Keeping XI first-stage limits MILP growth to ~4× (vs ~10× if XI were also scenario-conditional). Squad / transfers / chip activations STAY first-stage (commit before kickoff per FPL rules). |
| **Squad / XI / transfers / chips** | First-stage (single set across all scenarios) | XI commits at deadline; rest are obvious. |
| **Source of scenarios** | Raw per-iteration sample dump from `eval/xpts_eval.run_held_out_with_raw_samples` | Already wired. Need to extend to all walk-forward folds, not just held-out. |
| **Chips** | Same MILP variables as Phase 3 v1.1; chip BONUS now per-scenario (TC captain pts vary, BB bench points vary) | Chip *decisions* stay first-stage (binary `z_*[w]`); chip *bonuses* become scenario-averaged like xPts. |
| **EO penalty** | Same `rho * EO_p * mult * pts` term, but `pts` is now per-scenario | EO itself is a population statistic (not scenario-dependent). |
| **Horizon** | H=6 same as Phase 3 v1.2 | Larger H × more scenarios = exponential blow-up. |
| **Solver** | Pyomo + HiGHS via appsi (same as Phase 3); MIP gap 1% | Validated on Phase 3. Same SIGABRT caveat — subprocess isolation for multi-fold runs. |

---

## 2. Mathematical formulation

### 2.1 Sets
- $P$: candidate players (filtered)
- $W = \{1, \ldots, H\}$: gameweeks in the rolling horizon
- $S$: scenarios, $|S| = 50$
- $T$: distinct teams in candidate pool

### 2.2 Parameters (per scenario)
- $\text{pts}_{p,w,s}$: per-scenario sample of player $p$'s actual FPL points at GW $w$ in scenario $s$ (from `xpts_raw_samples_{N}_{N+1}.parquet`)
- $\text{eo}_p$: effective ownership (population, not scenario-dependent)
- $\rho$: EO-penalty coefficient (default 1.0, per Phase 3)
- $\alpha, \beta$: terminal-value coefficients (per Phase 3; α=0.8, β=0.0 from v1.2 tuning — may re-tune)

### 2.3 Decision variables

**First-stage** (single set across all scenarios) — Phase 3 baseline unchanged:
- $x_{p,w}$: squad (binary, all $w$)
- $y_{p,w}, v_{p,w}$: XI and vice (binary, all $w$)
- $c_{p,1}$: captain at the current GW (binary, w=1 only — first-stage)
- $\text{bin}_{p,w}, \text{bout}_{p,w}$: transfer indicators (binary, all $w$)
- $\text{ft}_w, \text{ht}_w$: FT count, hits (NonNegativeIntegers, all $w$)
- $\text{bank}_w$: bank value (NonNegativeReals, all $w$)
- $z^{WC}_w, z^{FH}_w, z^{BB}_w, z^{TC}_w$: chip activation (binary, all $w$)

**Second-stage** (scenario-conditional) — captain only, $w \geq 2$:
- $c_{p,w,s}$: per-scenario captain for w≥2 (binary)
- Constraint: $c_{p,w,s} \leq y_{p,w}$ (captain ∈ first-stage XI per scenario)
- Constraint: $\sum_p c_{p,w,s} = 1$ per (w, s) (one captain)

**Chip bonuses** (scenario-conditional, weekly big-M):
- $\text{tc\_bonus}_{w,s}$ (continuous ≥ 0): big-M ties to $z^{TC}_w$ and $\sum_p \tilde{c}_{p,w,s} \cdot \text{pts}_{p,w,s}$, where $\tilde{c}_{p,w,s} = c_{p,1}$ if $w=1$ else $c_{p,w,s}$.
- $\text{bb\_bonus}_{w,s}$ (continuous ≥ 0): big-M ties to $z^{BB}_w$ and $\sum_p (x_{p,w} - y_{p,w}) \cdot \text{pts}_{p,w,s}$. Note: bench points vary per scenario only via $\text{pts}$ (since x, y are first-stage); summation gives scenario-specific bench value.

### 2.4 Objective

Let $\tilde{c}_{p,w,s} = c_{p,1}$ if $w=1$ else $c_{p,w,s}$ (first-stage for w=1, scenario-conditional for w≥2).

XI is first-stage (single $y_{p,w}$). EO penalty uses the same $\rho \cdot \text{eo}$ as Phase 3 but is now averaged over scenarios via `pts[p,w,s]`.

$$
\max \;\; \frac{1}{|S|} \sum_{s \in S} \sum_{w \in W} \sum_{p \in P} \big( y_{p,w} + \tilde{c}_{p,w,s} \big) \cdot (1 - \rho \cdot \text{eo}_p) \cdot \text{pts}_{p,w,s} + \frac{1}{|S|} \sum_{s, w} \big( \text{tc\_bonus}_{w,s} + \text{bb\_bonus}_{w,s} \big) - 4 \sum_w \text{ht}_w + V_T(s_H)
$$

**Why this differs from deterministic**: the scenario-conditional captain $c_{p,w,s}$ for w≥2 lets the optimizer pick a different captain per scenario. Combined with the first-stage XI (which must include all candidate captains), the squad selection naturally gravitates toward XIs with multiple high-variance captaincy options — which is exactly the hedging behavior we want.

Mathematical sanity check (vs equivalence to deterministic):
- XI is first-stage → its contribution is $\sum_w \sum_p y_{p,w} \cdot \overline{\text{pts}_{p,w}}$ (averaged pts) — same as deterministic.
- Captain w=1 is first-stage → $\sum_p c_{p,1} \cdot \overline{\text{pts}_{p,1}}$ — same as deterministic.
- Captain w≥2 is scenario-conditional → $\frac{1}{|S|} \sum_s \max_p c_{p,w,s} \cdot \text{pts}_{p,w,s}$ given the XI constraint — **strictly different** from deterministic because $E_S[\max_p \text{pts}_{p,w,s}] \geq \max_p E_S[\text{pts}_{p,w,s}]$ (Jensen). The captaincy adaptive bonus IS the Phase 4 value-add.

**Effective complexity**: 
- First-stage: ~Phase 3 baseline (≈ 4-5k binaries)
- Second-stage captain: |P| × (|W|-1) × |S| ≈ 111 × 5 × 25 ≈ 14k new binaries
- Chip-bonus per scenario: 2 × |W| × |S| ≈ 300 continuous + 900 big-M constraints
- Total: ~18-20k binaries — about 4-5× Phase 3 baseline

Tractability mitigations:
- Start with **|S|=25**.
- 1% MIP gap (Phase 3 v1.2 fix).
- If still too slow: drop |S| to 10-15 and proceed.

### 2.5 Constraints

All Phase 3 first-stage constraints are unchanged:
- Squad shape (15 players, position counts 2-5-5-3, per-team ≤ 3) per $w$
- XI shape (11, ≥1 GKP, ≥3 DEF, ≥2 MID, ≥1 FWD; XI subset of squad) per $w$
- Vice rules per $w$ (vice ∈ XI; capt ≠ vice for w=1; vice unconstrained for w≥2 since not used)
- Transfer balance (bin/bout = squad delta) per $w$
- Bank evolution (cost-basis sell-tax from Phase 3.5)
- FT evolution + hits (with WC/FH chip waiver)
- Chip legality (one per GW, slot-half-of-season constraints)
- $c_{p,1}$ rules: $\sum_p c_{p,1} = 1$, $c_{p,1} \leq y_{p,1}$

Second-stage constraints (per scenario $s$, $w \geq 2$):
- $\sum_p c_{p,w,s} = 1$ (one captain per scenario per w)
- $c_{p,w,s} \leq y_{p,w}$ (captain ∈ first-stage XI)

Per-scenario chip bonus big-M:
- $\text{tc\_bonus}_{w,s} \leq \sum_p \tilde{c}_{p,w,s} \cdot \text{pts}_{p,w,s}$
- $\text{tc\_bonus}_{w,s} \leq M_{TC} \cdot z^{TC}_w$
- $\text{tc\_bonus}_{w,s} \geq \sum_p \tilde{c}_{p,w,s} \cdot \text{pts}_{p,w,s} - M_{TC} \cdot (1 - z^{TC}_w)$
- Same triple for $\text{bb\_bonus}_{w,s}$ with $\sum_p (x_{p,w} - y_{p,w}) \cdot \text{pts}_{p,w,s}$ on the RHS.

### 2.6 Stage B (deferred to v2)

Not implemented in v1. If/when added, would be a separate optimizer path activated when `distance_to_target_rank > threshold`. Formulation per Phase 0 §1.2: maximize $\mathbb{E}[\mathbf{1}\{\text{season\_total} \geq \tau\}]$ via binary indicators + big-M's.

---

## 3. Tractability analysis

Phase 3 v1.2 baseline (H=6 + chips):
- Mean solve 7-9s per GW; some GWs hit 30-50s
- ~111-114 candidates per fold-week
- Default HiGHS settings + 1% MIP gap

Phase 4 v1 expected:
- Same candidate count (no change to candidate filter)
- Same constraint count (chip bonuses aggregated)
- Objective is a single linear sum (1/|S|) × Σ_s (...) — coefficient-wise larger but still O(|P| × |W|) terms

The added cost is in **parameter loading**: 50 scenarios × 600 players × 6 GWs = 180k pts values per solve, vs Phase 3's 3.6k. Mostly memory not solver complexity.

**Solve-time prediction**: 1.5-3× slower than Phase 3 deterministic per GW. Target ≤ 30s mean. If hits ≥ 60s consistently, **drop |S| to 25**.

**Tractability gate**: ≤ 80s mean solve per rolling GW on fold 24/25.

---

## 4. Scenarios pipeline

### 4.1 What exists
- `eval/xpts_eval.run_held_out_with_raw_samples()` dumps per-(player, fixture, iteration) raw FPL points to `data/predictions/xpts/xpts_raw_samples_{N}_{N+1}.parquet`
- Implemented for the held-out season 25 only.

### 4.2 What needs adding
- Extend the raw-samples dump to ALL walk-forward folds (21, 22, 23, 24). Files: `xpts_raw_samples_21_22.parquet` etc.
- New PIT helper or direct cache file path: `data/cache/xpts_raw_samples/season_{N}_train_{...}.parquet`
- v1 keeps n_iterations=200 (matches Phase 2.5); downsample to 50 in the SAA loader.

### 4.3 Caching strategy
- Cache key: `(test_season, train_seasons_hash, n_iterations, seed)`
- File: `data/cache/xpts_raw_samples/season_{N}_train_{...}_n200.parquet`
- Loader: `_load_raw_samples_for_fold(test_season, train_seasons) → DataFrame`
- Scenario subsample: deterministic by seed within the loader (first 50 of 200, or random subset; v1 = first 50)

---

## 5. Acceptance gates

Three categories, mirroring Phase 3's structure:

### 5.1 Optimizer validity gates (ALL MUST PASS) — fold 2024/25
- [ ] Feasibility: every per-GW SAA MILP returns a valid incumbent.
- [ ] Budget correctness: cost + bank conservation (cost-basis sell-tax v3.5 rule).
- [ ] Transfer accounting: hits = max(0, transfers_in - FT); FT cap at 5.
- [ ] Chip legality: per-slot windows, one chip per GW.
- [ ] No future leakage: scenarios from walk-forward CV; prices walk only backward.

### 5.2 Tractability gate
- [ ] Mean rolling-GW solve time ≤ 30s on fold 24/25 with |S|=50. (If ≥ 60s, drop |S| to 25 and retry. If still ≥ 60s with |S|=25, halt — flag for follow-up.)

### 5.3 Performance gate (REPORTED — not hard-gated)
Phase 0 §1.3 stretch target: stochastic beats deterministic by ≥ 30 pts/season on average across folds 21-24.

Reported numbers per fold:
- MILP total points (stochastic Phase 4)
- MILP total points (deterministic Phase 3 v3.5) — for comparison
- Buy-and-hold-template
- Δ stochastic vs deterministic; Δ vs template

---

## 6. Code structure

### 6.1 New files
- `src/fpl_bot/optim/scenarios.py` — scenario-loading API. Functions:
  - `load_raw_samples(test_season, train_seasons, n_scenarios)` → DataFrame of (player_id, gameweek, scenario_id, pts)
  - `make_scenario_pts_dict(samples_df, candidates, horizon_gws)` → dict[(p, w, s) → pts]
- `src/fpl_bot/eval/saa_backtest.py` — stochastic equivalent of `eval/milp_backtest.py`. Mirrors `backtest_season` API but takes `n_scenarios` and routes through the SAA-aware MILP.
- `scripts/phase4_saa_walkforward.py` — subprocess-isolated runner (same pattern as `phase2_5_1_phase3_isolated.py`)
- `tests/unit/test_scenario_pts.py` — scenario-pts dict semantics, sampling determinism
- `docs/design/phase4_results/` — per-fold artifacts

### 6.2 Modified files
- `src/fpl_bot/optim/milp.py` — add `MilpInputs.predictions_per_scenario: dict[(int, int, int), float] | None`. When set, objective uses SAA; when None, falls back to current deterministic path. Bool flag `use_saa` for clarity.
- `src/fpl_bot/eval/xpts_eval.py` — extend `run_held_out_with_raw_samples` to take an arbitrary fold; rename to `run_fold_with_raw_samples`. Add cache write path.

### 6.3 Reuse from Phase 3
Everything except the objective changes:
- Cost-basis sell-tax (state.py)
- Cold-start H=1
- Candidate filter
- Weekly-aggregate chip bonuses
- Terminal value coefficients
- HiGHS appsi + 1% MIP gap

---

## 7. Build order

1. **Extend raw-sample dumps** for folds 21-24 (re-runs Phase 2.5 simulator on each; ~30 min total). Cache to `data/cache/xpts_raw_samples/`.
2. **Implement `optim/scenarios.py`** loader; smoke-test that the per-(p, w, s) dict builds for fold 24/25.
3. **Add SAA branch to `optim/milp.py`** behind `MilpInputs.use_saa=True`. Smoke-test: single MILP solve on cold-start GW1 of fold 24 with |S|=50.
4. **Implement `eval/saa_backtest.py`** mirroring `milp_backtest.py`. Smoke-test full fold 24/25 rolling solve.
5. **Tractability check**: measure mean solve time on fold 24/25. If ≤ 30s/GW, proceed; if ≥ 60s/GW, drop to |S|=25.
6. **Run full walk-forward** via subprocess-isolated harness across folds 21-24.
7. **Compare vs Phase 3 v3.5 baseline**: report Δ per fold.
8. **Commit + design doc final status.**

---

## 8. Open questions for review

1. **Number of scenarios**: 50 is a guess. Stochastic-programming literature suggests 100-500 for convergence; FPL's effectively-bounded points outcome may need fewer. Recommend: ship |S|=50 v1, run a |S| ∈ {25, 50, 100} sweep on fold 24/25 in §6 (the tractability check), pick the smallest that doesn't lose ≥ 5% of the |S|=200 ceiling.

2. **Scenario-conditional vs aggregated second-stage**: v1 aggregates (single squad/XI/captain decisions for all future GWs). True scenario-conditional would let the optimizer adapt to each scenario in weeks 2-H, but blow up the variable count by |S|. **Recommend aggregated v1**; scenario-conditional is a v2 candidate.

3. **Chip-bonus scenario-averaging**: aggregate the chip bonuses across scenarios so they stay weekly (Phase 3 v1.1 structure). Alternative: per-scenario `tc_bonus[w,s]` which captures the variance of captain's haul-vs-blank under TC. **Recommend aggregated v1 for tractability**; per-scenario chip bonuses are a v2 candidate.

4. **Captain (c) constrained to be in XI (y), but in stochastic — single c across all scenarios**: this is correct under "first-stage decisions are non-anticipative" — the captain is announced before the GW so it's deterministic. No issue.

5. **Subprocess isolation**: same SIGABRT issue from Phase 3 v1.4 applies. SAA likely makes it worse (larger objective, more memory). **Use subprocess-per-fold from day 1**; don't bother with single-process multi-fold.

6. **Stage B threshold-chasing**: skipped in v1. The trigger condition (`distance_to_target_rank > threshold`) requires a top-10k EO data source we don't have — `fpl_api_approx` is too coarse. Defer until LiveFPL/FPLStatistics is in. Even with the data, Stage B adds chance-constrained reformulation complexity that wants its own design phase.

---

## 9. Game-mechanics audit (Phase 3 inherited; Phase 4 inherits all)

### Modeled correctly in Phase 3 (carried into Phase 4 unchanged)
- **Free transfers (FT)**: 1 FT/GW, banked up to cap of 5 (2024/25 rule). MILP variable `ft[w]` ∈ {0,...,5} with explicit evolution constraint. Optimizer can choose to skip a transfer this GW to save FT for next — **"down-the-road" FT-saving is supported**.
- **Hits**: `ht[w] ≥ max(0, transfers_in[w] - ft[w])`; objective subtracts `4·ht`. Optimizer trades hit cost against expected gain.
- **Wildcard (WC1/WC2)**: `z_wc[w]` waives the hit cost via big-M; FT consumption is also waived. One per half-season; per-slot constraints enforce season-half windows.
- **Free Hit (FH1/FH2)**: `z_fh[w]` waives hit/FT consumption for one GW; one per half-season.
  - ⚠ **Known v1 simplification**: the MILP doesn't force the squad to revert to pre-FH next GW. Within a single rolling solve, an FH played at the horizon tip can leave the next solve seeing the FH squad. Mitigation: harness keeps `state.squad` frozen at pre-FH for that case. Phase 5 chip-DP coupling will resolve this cleanly.
- **Bench Boost (BB)**: weekly-aggregate `bb_bonus[w]` = Σ_p (x[p,w] − y[p,w]) · pts[p,w] under z_bb. Bench players score in that GW.
- **Triple Captain (TC)**: weekly-aggregate `tc_bonus[w]` = Σ_p c[p,w] · pts[p,w] under z_tc. Captain gets 3× instead of 2×.
- **Squad/XI shape**: 15 players (2-5-5-3 by position), per-team ≤ 3, XI = 11 with formation rules (1 GKP, ≥3 DEF, ≥2 MID, ≥1 FWD), captain + vice ∈ XI.
- **Budget**: cost + bank conserved via `bank_evolution` constraint (Phase 3 v1.0); cost-basis sell-tax via FPL's `sell = current − max(0, profit//2)` rule (Phase 3.5).
- **Captain triggers TC mult on points layer**: in `backtest_season`, GW points = XI points + capt_pts × (multiplier-1); multiplier=3 under TC, else 2. Bench points added when BB played.
- **DGW (Double Gameweek)**: predictions and actuals aggregate across same-GW fixtures via `_per_player_per_gw_predictions/actuals` using `pl.col("e_xpts").sum()` over fixture_ids. ✓ Correct.
- **BGW (Blank Gameweek)**: a (player, GW) with no fixture has no prediction row → pts contribution is 0. The MILP can still hold the player in squad (their `x[p,w]=1` is fine without points).
- **Terminal value V_T(s_H)**: at horizon tip, `α · post-horizon-xPts + β · sell_price` per held player. 5-GW lookahead beyond the H=6 explicit horizon → effectively 11 GWs of look-ahead per rolling solve.

### Known v1 limitations (carried into Phase 4)
These cost real points but were deemed non-blocking in Phase 3. Phase 4 inherits them:

1. **Auto-substitutions NOT modeled**. In real FPL, if an XI player gets 0 minutes, the first bench player (in bench-order, position-eligible) auto-subs in. The MILP harness computes GW points as Σ(starting_xi) without auto-sub adjustment, so XI blanks just give 0 — they don't trigger bench replacement. Effect: under-estimates actual scored points by 1-3% on average (FPL community-validated).
2. **Auto-vice NOT modeled**. If captain gets 0 minutes, vice receives the 2× multiplier. Harness applies `2 × captain_pts` unconditionally, so a 0-min captain gives 0 captain bonus instead of `2 × vice_pts`. Effect: missed multiplier on captain-blank GWs (~10% of GWs).
3. **Bench order NOT decided by MILP**. Currently the MILP picks the bench (`x − y`) but doesn't assign bench positions 1/2/3. Affects auto-sub priority — a v2 deliverable.
4. **No injury / status overrides in backtest**. `fact_player_status` (status_code, chance_of_playing) is a live-only signal — vaastav historical CSVs don't have it. Backtest treats every player as available; live mode has a soft-override path.
5. **Static buy prices over horizon**. Price-change predictor was shelved in Phase 3.5 (failed CV gates). Buy prices are held at current value across the horizon; sells use cost-basis sell-tax which is exact.
6. **No yellow-card-ban / red-card-ban tracking**. 5 YCs → 1-game ban; red card → 1+ game ban depending on offence. Effect: predictions can over-rate banned players; minor.

### What Phase 4 SAA does NOT add or change
- All the above limitations carry through to Phase 4 (it's an objective-only change).
- The auto-sub modelling gap is the largest unmodeled value-leak (1-3% of actual pts). Phase 4 should include a stretch task: implement auto-sub in the backtest **scorer** (not the MILP — would still pick the same squad, just measure real-FPL actuals more accurately). Tagged as a Phase 4 stretch in §10 below.

### Down-the-road decision-making (the user's stated concern)
The MILP IS capable of making "save FT now to spend later", "take a hit now to capture early-season form", "hold a rising-price asset for budget headroom", and "skip transferring an injured player who's likely to be replaced cheaply next GW" type decisions, via:
- The H=6 rolling horizon: each per-GW solve plans 6 GWs ahead, then commits only GW1 decisions.
- The terminal value V_T: incentive to leave the squad in a high-value, high-future-xPts state at horizon-tip.
- Phase 4 SAA: makes those forward-looking decisions ROBUST across 50 scenarios rather than just the mean — so "save FT" decisions account for scenario-variance in next-GW xPts.

Limitations on this:
- Greedy-rolling chip coupling: chips are picked per rolling solve, can lead to suboptimal chip timing across the season. Phase 5 chip-DP resolves.
- The aggregated second-stage in v1 means weeks 2-H decisions are fixed across scenarios — a player who's borderline-injured can't be picked-only-in-scenarios-where-they-play. Phase 4 v2 with scenario-conditional second-stage addresses.

---

## 10. Visible post-4.0 follow-ups (do NOT block)

1. **Stage B threshold-chasing**: per round-1, only if/when late-season backtests show insufficient variance-seeking. Needs top-10k EO data.
2. **Scenario-conditional second-stage**: let weeks 2-H adapt per scenario. Big MILP growth but theoretically tighter optimization.
3. **Per-scenario chip bonuses**: capture chip-haul variance (esp. TC under high-EO captain).
4. **Importance sampling**: oversample tail scenarios (huge hauls, blanks) for better variance estimates on chip decisions.
5. **Phase 5 chip-DP coupling**: still pending. Could compose cleanly with SAA — chip-DP picks slot, SAA picks players given chip choice.
6. **Auto-substitution modelling in the SCORER** (Phase 4 stretch). MILP picks the same squad/XI; the harness's `gw_points` computation gains the auto-sub rule: for XI players with `actual_minutes == 0`, replace with the first bench-eligible player (in bench-order) who scored. This bumps reported totals 1-3% closer to real FPL. The MILP itself doesn't change.
7. **Auto-vice modelling in the SCORER** (Phase 4 stretch). If captain `actual_minutes == 0`, apply 2× multiplier to vice instead. Same as #6 — scorer-side fix only.
8. **Bench order optimization** (v2 — MILP-side). Add bench-position decision variables (slot 1/2/3) so the MILP can prioritize bench-eligible auto-sub candidates.
9. **Live-mode status overrides**: production path that respects `chance_of_playing` and `status_code` at predict time. Not relevant for historical backtest but blocks deployment.
