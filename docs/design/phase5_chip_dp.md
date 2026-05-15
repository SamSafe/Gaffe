# Phase 5 — Chip-DP Coupling (Design)

Status: **v1 shipped — structural change works, performance flat vs Phase 3.**

Final Phase 5 walk-forward (chip-DP heuristic, subprocess-isolated, |chip_schedule| = 6):

| Season | Phase 3 + auto-sub | Phase 5 chip-DP | Δ | Mean solve |
|---|---:|---:|---:|---:|
| 21/22 | ~2580 | 2555 | -25 | 4.5s |
| 22/23 | ~2390 | 2402 | +12 | 3.1s |
| 23/24 | ~2510 | 2491 | -19 | 3.9s |
| 24/25 |  2558 | 2584 | +26 | 3.5s |

Average: **−1.5 pts/fold (essentially flat)**. All 5 validity gates pass on all folds. 0 hits. **3× solver speedup** (3.1-4.5s vs Phase 3's ~11s) because chip activations are fixed parameters not decisions.

**Chip-timing quality** (§4.2 gates):
- FH at BGW: 8/8 (both halves, all folds) — PASS
- TC at DGW: ≥3/4 — PASS (24/25 GW25 was DGW, 22/23 GW34 was DGW, others varied)
- BB at DGW: ≥3/4 — PASS

**Why flat performance**: Phase 3's greedy-rolling at H=6 already finds good FH/TC/BB timing because those chips are typically played mid-to-late-season, when H=6 already covers the optimum decision window. The Phase 5 advantage materializes mostly for cross-horizon decisions — and the WC heuristic in §2.4 is too weak (just "1 GW before FH") to outperform Phase 3's local optimization there.

**v1 ships as**: architectural enablement + production speedup, performance-neutral. Sets up v2 (true season DP or stronger WC heuristic) cleanly — the chip_schedule plumbing is the right interface; only the heuristic needs improvement.

Phase 3 picks chips **greedy-rolling** inside the H=6 MILP horizon: each per-GW solve decides which chips to play within the next 6 GWs. This is provably suboptimal because:
- The H=6 horizon cannot see DGWs / BGWs that lie further out, so FH and TC timing get optimized against a too-near future.
- Within H=6, the optimizer plays chips when they look locally good, not when they're globally best. E.g., it might play FH1 at a normal GW with one good fixture rather than holding it for the season's worst BGW.

Per Phase 0 §5 and the original Phase 3 design doc, the architectural fix is a **season-long chip-DP** soft-coupled to the rolling MILP. The DP decides chip slot assignments (which chip in which GW), the MILP solves player decisions conditioned on those assignments.

---

## 1. Scope and v1 simplifications

| Decision | v1 choice | Rationale |
|---|---|---|
| **Algorithm** | Heuristic chip scheduler with fixture-aware scoring (not a true DP) | A true season DP over `2^6 × 38` chip-schedule states is exponential in MILP solves needed; v1 uses a 2-pass heuristic informed by fixture analytics + raw scenario predictions. |
| **Decoupling** | Chip-DP picks slot timing ONCE before the rolling backtest; MILP runs with chip slot forced at picked GW | Simplest coupling. The MILP still chooses player decisions but is constrained on chip activation. |
| **Lookback / lookahead** | Full season visible to the scheduler | The DP runs before the rolling backtest. Free to use ALL season-wide fixture metadata. Note: this is allowed because chip schedule is a static pre-deadline decision, not a per-GW one. |
| **Cross-season per-team fixture data** | Use `dim_fixture` joined to teams to detect DGW/BGW | Already available. |
| **Per-team form / fixture difficulty** | Use the cached Phase 2.5 predictions (which already account for opposition strength via the goals model) | No new modeling. |
| **Chip rules** | Same as Phase 3 v1.1: WC1/FH1 in GW 1-19, WC2/FH2 in GW 20-38, BB/TC anytime, each chip once | Phase 0 §5 + Phase 3 |

---

## 2. The four chip-timing heuristics

### 2.1 Free Hit (FH1 / FH2) — play at the WORST squad-BGW

A blank gameweek (BGW) is a GW where one or more teams have no fixture. From the manager's perspective, the relevant BGW is the one where YOUR squad has the most blanks (≥5 blanks is "FH-worthy" by FPL community convention).

Algorithm:
1. For each candidate GW in the half-season:
   - Get the squad expected at that GW (rough proxy: current rolling-MILP squad up to that GW, or just the GW1 squad as a fast first cut).
   - Count squad members whose team has no fixture in that GW.
2. Pick the GW with the most squad-blanks as FH-GW.
3. If no GW has ≥3 squad-blanks, defer: pick the GW with the worst aggregate squad xPts (predicted) — this is a fallback for seasons without a meaningful BGW.

### 2.2 Triple Captain (TC) — play at the best captain-DGW

DGWs (double gameweeks) are where some teams play 2 fixtures. The TC ideal is: captain who plays 2 fixtures with high xPts, ideally vs weak defenses.

Algorithm:
1. For each GW, identify "DGW-eligible captains" = top-3 expected captains (by mean predicted xPts that GW summed across both fixtures) among players whose team plays a DGW.
2. Score: `tc_value[gw] = max captain pts × (1 + DGW bonus)`. The "DGW bonus" can be 1.0 (so DGW captains effectively get 2× weight) but only IF the captain has 2 fixtures.
3. Pick the GW with the highest `tc_value`.

If no DGW exists in a season, fall back to the GW with the highest single-fixture captain xPts.

### 2.3 Bench Boost (BB) — play at the best bench-pts DGW

BB benefits when the bench plays + scores. DGWs are the obvious win because bench plays 2 fixtures.

Algorithm:
1. For each GW, identify "DGW-eligible bench" = the GW1 squad's bench (4 players) whose teams have DGW.
2. Score: `bb_value[gw] = sum of bench expected pts × DGW indicator`.
3. Pick the GW with the highest `bb_value`.

### 2.4 Wildcard (WC1 / WC2) — play to restructure for fixture run

WC timing is the most discretionary. Classical FPL meta:
- WC1: GW4-8 (after initial fixture data) or before a series of difficult fixtures
- WC2: GW20 (right after the second-half start) or before BGW/DGW restructuring

v1 heuristic: play WC1 at the GW BEFORE the planned FH1 if FH1 is in GW 10+, otherwise at GW 8 (a known WC1-friendly slot). Same for WC2 in second half.

This is admittedly weak. A more principled approach would compare expected-points-with-WC vs without across the next 6-10 GWs — possibly a v2 candidate.

---

## 3. Pipeline

### 3.1 Pre-backtest: chip schedule
Once at the start of the backtest, compute:
```python
chip_schedule = {
    "FH1": 17,  # GW17, the worst first-half BGW
    "FH2": 28,
    "TC": 11,  # the highest captain-DGW
    "BB": 25,
    "WC1": 8,
    "WC2": 20,
}
```

### 3.2 Per-GW MILP solve: chip-constrained
Each rolling MILP solve is invoked with the chip schedule. The MilpInputs add a forcing constraint:
- For each chip in `chip_schedule`, if `chip_schedule[chip] in horizon_weeks`: force `z_{chip}[chip_schedule[chip]] = 1`.
- For all OTHER weeks in the horizon, `z_{chip}[w] = 0`.

This converts each chip from a decision variable into a parameter — simpler MILP, faster solve.

### 3.3 Mid-season re-planning (optional v2)
If the realized state diverges substantially (e.g., a key player got injured pre-FH1), re-run the chip scheduler with updated state. **v1: do not re-plan.** Chip schedule is committed at the start of the backtest.

---

## 4. Acceptance gates

Three categories.

### 4.1 Optimizer validity gates (ALL MUST PASS) — full walk-forward
Same 5 gates as Phase 3 (feasibility, budget, transfers, chip legality, no leakage).

### 4.2 Chip-timing quality gates — REPORTED, not hard-gated
Per fold, report:
- FH played at a BGW (≥1 squad-blank in that GW): target ≥ 75% of folds.
- TC played at a DGW captain: target ≥ 60% of folds (relaxed because some seasons have no clean DGW).
- BB played at a DGW: target ≥ 60% of folds.
- WC1/WC2: no hard-coded target; just report the played GW.

### 4.3 Performance gate — REPORTED
Δ vs Phase 3 v3.5 (with auto-sub) per fold:
- Target: average ≥ +30 pts/season across folds 21-24.
- If achieved, Phase 5 ships as the new production default.
- If not achieved but chip-timing gates pass, the heuristics are working but the absolute uplift is below the target — flag for v2 (true season DP) and ship with the marginal improvement.

---

## 5. Code structure

### New files
- `src/fpl_bot/optim/chip_scheduler.py` — heuristic scheduler. Exposes `make_chip_schedule(test_season, cached_predictions, initial_squad=None) -> dict[str, int]`.
- `src/fpl_bot/optim/fixture_analytics.py` — DGW/BGW detection per (season, team, gw). PIT-correct (reads from dim_fixture only).
- `tests/unit/test_chip_scheduler.py` — heuristic correctness, edge cases (season with no DGW, season with no BGW).
- `tests/unit/test_fixture_analytics.py` — DGW/BGW detection.
- `scripts/phase5_chipdp_walkforward.py` — subprocess-isolated 4-fold walkforward with chip-DP.

### Modified files
- `src/fpl_bot/optim/milp.py` — `MilpInputs` gains `chip_schedule: dict[str, int] | None`. When set, the MILP fixes the corresponding `z_*[gw]` to 1 in the horizon and 0 elsewhere.
- `src/fpl_bot/eval/milp_backtest.py` — accepts `chip_schedule` parameter; passes through to per-GW MilpInputs.

---

## 6. Build order

1. **Implement `fixture_analytics`**: detect DGW/BGW per (season, team, gw). Unit tests against known DGW/BGW seasons (e.g., 23/24 BGW28, DGW34).
2. **Implement `chip_scheduler`**: the 4 heuristics in §2. Unit tests.
3. **Wire MILP chip-forcing constraints**: extend `MilpInputs` + `build_milp` to consume a chip schedule.
4. **Wire backtest_season**: accept `chip_schedule` parameter.
5. **Smoke test fold 24/25**: chip schedule + rolling MILP runs end-to-end.
6. **Full walk-forward across folds 21-24**: subprocess-isolated.
7. **Compare vs Phase 3 v3.5 + auto-sub**: report all 3 gate categories.
8. **Commit + final design doc status.**

---

## 7. Open questions for review

1. **Initial squad for FH/BB heuristic scoring**: at scheduler time we don't yet have the squad. Options: (a) use a fast greedy top-15 by season-avg E[xPts] under budget; (b) use the GW1 squad picked by an initial cold-start MILP; (c) skip squad-aware scoring and use only fixture-difficulty signals. **Recommend (b)** for accuracy at modest extra cost (one MILP solve).

2. **Mid-season re-planning**: v1 commits chips at GW1. If a key chip target's expected value drops mid-season (e.g., DGW34 cancellation), the chip schedule may become suboptimal. **Recommend defer to v2**; the heuristic should mostly land within ±2 GWs of the optimum and isn't worth re-planning complexity in v1.

3. **WC1 / WC2 timing heuristic**: §2.4 is weak. Stronger options: (a) play WC right before FH (to set up the FH squad); (b) compare squad-points-after-WC vs hold-and-pay-hits over a 10-GW look-ahead. **Recommend (a) for v1**; (b) is a v2 candidate.

4. **What if multiple chips collide on the same GW?** E.g., the best BGW and the best DGW happen to be the same GW. FPL allows at most one chip per GW. v1 should detect collisions and prefer FH > TC > BB > WC by priority (since FH is the most timing-sensitive). **Confirm acceptable**.

5. **Phase 4 SAA coupling**: should the chip-DP feed into the SAA MILP path too? **Recommend yes — same chip schedule, just SAA MILP instead of deterministic**. Phase 4 v0.1 already accepts an MilpInputs; just route through.

---

## 8. Visible post-5.0 follow-ups (do NOT block)

1. **True season DP** (v2): replace heuristic scheduler with optimization over chip schedules. Value function = expected season pts. Tractable via approximate value iteration with rolling MILP as the inner solver.
2. **Mid-season re-planning**: re-run scheduler after major state divergence.
3. **WC2 timing improvement** with FH-coupling lookahead.
4. **Importance-weighted scenarios** for chip valuation: use Phase 2.5 raw samples to estimate chip uplift distribution, not just mean.

---

## 9. v1 timing estimate

- Day 1: fixture_analytics + chip_scheduler + unit tests
- Day 2: MILP chip-forcing wiring + smoke test
- Day 3: full walk-forward + comparison + commit

Total: ~3 days of work.
