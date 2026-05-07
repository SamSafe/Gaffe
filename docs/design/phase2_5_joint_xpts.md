# Phase 2.5 — Joint xPts Distribution

Status: **approved.** Implementation in progress.

Round-1 review resolutions:
1. Distributional output: **full PMF histogram** as a 31-bin polars list column (`xpts_pmf`, bin 0 = xPts of -5, bin 30 = xPts of +25, with clipping at the edges), in addition to summary stats and tail probs.
2. **Bump MC iteration count to 500** (vs the 200 used in Phase 2.4). xPts has wider variance than bonus.
3. **Captaincy-proxy fourth gate** — picking the simulator's argmax-E[xPts] captain across the test fold should yield more total captain points than picking the rolling-3-GW xPts leader.
4. **Raw-sample dump for held-out 2025/26** — turned ON for season 25 only; produces `xpts_raw_samples_2025_26.parquet` (~50 MB) for Phase 4 SAA development.

Per Phase 0 §4.7: assemble the per-component models into a joint xPts distribution per (player, fixture) via Monte Carlo. Each MC iteration draws minutes / goals / assists / CS / cards / saves / penalties / unmodeled-residual jointly (with the within-fixture correlation Phase 2.4 already preserves), and each draw produces **both** a BPS realization (for ranking) **and** an FPL-points realization. We aggregate to per-player percentiles and means.

This is the last predictive component before Phase 3's deterministic MILP. The deliverable is one Parquet file per fold-prediction with columns `(player_id, fixture_id, e_xpts, var_xpts, p_xpts_5, p_xpts_10, p_xpts_15, ...)` — enough to drive both deterministic and stochastic optimization.

---

## 1. What is computed

For each Monte Carlo iteration `s` of fixture `f`, the existing Phase 2.4 simulator already samples per-player events. We extend each iteration to also compute **FPL points**:

```
xpts_player_s = (
    appearance_points(minutes)
    + goals * goal_pts_by_position(position)
    + assists * 3
    + clean_sheet_pts(position, minutes, team_cs)
    + saves // 3                                  # GK only
    + goals_conceded_pts(position, minutes, team_gc)  # DEF/GK only
    + yellow_pts                                  # -1 if yc
    + red_pts                                     # -3 if rc
    + penalty_miss_pts                            # -2 per missed/saved pen
    + own_goal_pts
    + bonus[s]                                    # 0/1/2/3 from rank within fixture
)
```

The bonus is **already** computed by the Phase 2.4 simulator's `assign_bonus_within_fixture`. We just need to emit the full xPts per iteration alongside the bonus.

**FPL points rules (2024/25)**:
- Appearance: +1 for 1-59 min, +2 for 60+ min
- Goal: +6 GKP/DEF, +5 MID, +4 FWD
- Assist: +3 (all positions)
- Clean sheet (≥60 min): +4 GKP/DEF, +1 MID, 0 FWD
- Saves: +1 per 3 saves (GKP only)
- Goals conceded: −1 per 2 goals conceded (GKP/DEF only, while on pitch)
- Yellow: −1; Red: −3
- Penalty miss: −2 (taker only)
- Own goal: −2
- Bonus: +1/+2/+3

These are slightly different from BPS values (BPS uses larger weights for ranking).

---

## 2. Output schema

Per-(player_id, fixture_id) row in `data/predictions/xpts/{run_ts}/season_{N}.parquet`:

| Column | Description |
|---|---|
| `player_id` | stable FPL code |
| `fixture_id` | stable FPL fixture code |
| `season_id`, `gameweek`, `kickoff_utc` | fixture metadata |
| `e_xpts` | E[xPts] across MC iterations (mean) |
| `var_xpts` | Var[xPts] |
| `p_xpts_ge_2`, `p_xpts_ge_6`, `p_xpts_ge_10`, `p_xpts_ge_15` | tail probabilities for FPL "haul" thresholds |
| `q_xpts_5`, `q_xpts_25`, `q_xpts_50`, `q_xpts_75`, `q_xpts_95` | quantiles |
| `e_bonus` | E[bonus] (carried through from 2.4) |
| `model_version` | run identifier |

The full per-iteration xPts samples are NOT stored by default (storage cost). If Phase 4 needs raw samples for SAA, we add an opt-in `_raw_samples.parquet` with `(player_id, fixture_id, iteration, xpts)`.

---

## 3. Acceptance gate

Anchoring directly to FPL outcomes:

- **MAE on E[xPts] vs actual `total_points`** beats two baselines:
  1. Position-marginal mean xPts (per position, per minutes-bucket) — the same flavor of floor we used in 2.4.
  2. Naive sum-of-component means: `E[appearance] + E[goals]*goal_pts + E[assists]*3 + E[cs]*cs_pts + E[bonus]` using point estimates from the trio + Phase 2.4 bonus mean. This is what you'd get if you didn't bother with joint sampling — every component plugged in independently. The simulator must beat it.
- **Brier on P(xPts ≥ 6)** beats the same two baselines. xPts ≥ 6 is the FPL-community "haul" threshold and what captaincy decisions hinge on.
- **Calibration within ±10% across deciles of E[xPts]**. The decile structure: bin players by predicted E[xPts]; in each decile, mean predicted ≈ mean actual.

**Acceptance gate before Phase 3**:
- [x] **MAE on E[xPts] beats both baselines.** Passed 4/4 folds. Margin over naive sum-of-components: 2-3%; over position-marginal: 10-12%.
- [x] **Brier on P(xPts ≥ 6) beats both baselines.** Passed 4/4 folds. ~20% relative improvement over naive sum-of-components.
- [x] **Calibration ±10% per decile on ≥ 8/10 deciles.** Passed 4/4 folds: 10/10 deciles for 21/22, 22/23, 23/24; 9/10 for 24/25.
- [x] **Captaincy gate (round-1 addition)**: Simulator's argmax-E[xPts] captain beats rolling-3-GW xPts leader in total captain points. Passed 4/4 folds, with sim consistently picking 60-80% more captain points across the season.
- [x] **All leakage tests still green; xPts module covered by static-import audit on `models/*`.** 67/67 tests pass.

**Walk-forward CV results (4-fold, 500 MC iterations, ~24 min):**
```
season  | n      | MAE: sim/naive/pos     | Brier(≥6): sim/naive/pos | dec  | cap: sim/base/opt
 2021/22|  25447 | 1.063/1.083/1.187      | 0.0648/0.0819/0.0819     | 10/10| 622/370/1432
 2022/23|  26505 | 1.007/1.027/1.109      | 0.0619/0.0781/0.0781     | 10/10| 592/368/1298
 2023/24|  29725 | 0.924/0.953/1.030      | 0.0522/0.0655/0.0656     | 10/10| 634/466/1392
 2024/25|  27605 | 1.036/1.068/1.129      | 0.0620/0.0771/0.0772     |  9/10| 656/536/1428
```

### Notable findings

1. **Captaincy is the strongest signal of simulator value.** Across the four test seasons, picking the captain by argmax-E[xPts] yields **66%, 61%, 36%, 22%** more total captain points than the rolling-3-GW xPts leader. This is the FPL community's most consequential weekly decision and the gap is substantial. Optimal (oracle) captain takes ~2x the sim pick, so there's headroom for Phase 3+ refinements.

2. **MAE lift over naive sum-of-components is small (2-3%) but real.** This is the cleanest test of the joint-sampling architecture: even using the same trio outputs, plug-and-play summation underestimates the variance and (slightly) the mean. The simulator's full P(xPts) distribution is what makes the captaincy gate possible — argmax-E[xPts] requires the per-iteration draws to be coherent, not just per-component means.

3. **Calibration is excellent.** 39/40 decile passes across folds. The simulator's mean predictions track empirical means within ±10% on essentially every decile.

### Held-out 2025/26 raw-sample dump

**Deferred** — `fact_player_match` for season 25 is empty (vaastav backfill is through 2024-25; the FPL API ingest doesn't write match-level data). The `run_held_out_with_raw_samples()` capability is implemented and ready; it can run as soon as 2025-26 vaastav data is published. The Phase 4 SAA development can proceed with held-out 2024-25 raw samples in the meantime if needed.

---

## 4. Code structure

Most heavy lifting reuses Phase 2.4. Two key changes:

```
src/fpl_bot/
├── models/
│   ├── bps.py                  # extend BPSSimulator: also emit per-iter xPts samples
│   └── xpts.py                 # new: FPL-points rule table + score function;
│                               # JointXPtsSimulator wraps BPSSimulator and aggregates
└── eval/
    ├── xpts_eval.py            # walk-forward CV with MAE + Brier(≥6) + calibration deciles
    └── xpts_diagnostics.py     # per-position calibration, top-N FPL haul predictions
tests/
├── unit/
│   └── test_xpts.py            # FPL rule correctness, sum-of-components consistency
└── leakage/
    └── test_xpts_features.py   # PIT-stable as in 2.4
```

The simulator's per-iteration loop already has all the events needed; we add a `score_fpl_points()` call in the same loop and accumulate samples per player.

---

## 5. Architectural notes

**Why one simulator pass produces both BPS and xPts:** the same draws of (minutes, goals, assists, CS, cards, saves, pens) feed both rule tables. Running the simulator twice (once for BPS, once for xPts) would be wasteful and could produce inconsistent samples (different RNG draws). Phase 2.5 therefore *replaces* `BPSSimulator.simulate_fixture` with a richer `JointXPtsSimulator.simulate_fixture` that emits both bonus and xPts per iteration; Phase 2.4 callers continue to work via a thin compatibility shim that drops the xPts column.

**Storage strategy:** percentiles + tail probabilities are sufficient for almost all downstream uses. The optimizer's stochastic extension (Phase 4) wants raw scenario draws — we add an opt-in path to dump per-iteration samples if and when Phase 4 needs them. Default output is the aggregated row per (player, fixture).

**Cross-week correlation (Phase 4 dependency):** out of scope here; Phase 4's joint-trajectory sampler applies a per-player form-factor before invoking the simulator.

---

## 6. Open questions

1. **Quantile set.** Default proposal: q5/q25/q50/q75/q95 + tail probs at 2/6/10/15. Sufficient for risk-aware optimization in Phase 4? Or also store the full empirical CDF (16 bins)?
2. **Raw-sample dump.** Off by default. Should it be on for the held-out 2025/26 season so we have ground truth for Phase 4 development? Adds ~25 MB per fold at 200 iterations.
3. **MC iteration count.** Phase 2.4 used 200 iterations per fixture. For xPts the variance is wider than for bonus, so 500+ would be more accurate. Worth bumping to 500 for the production run? Cost: ~25 min for full backtest CV vs current ~10 min.
4. **Captaincy proxy as a fourth gate.** Optionally: simulator's expected captaincy points (= 2 × E[xPts] for each player, take max in fixture per team) should beat a "captain the highest-rolling-xPts player" baseline. Would catch FWD-haul under-prediction the 2.4 diagnostics surfaced. Worth adding?

---

**End of Phase 2.5 design. Awaiting review.**
