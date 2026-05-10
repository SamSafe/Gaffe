# Phase 2.5.1 — Multinomial Goal Sampling (Design)

Status: **shipped (multinomial fix + MIP gap fix); fold 24/25 backtest_season wrapper instability documented as follow-up.**

What ships:
1. **Multinomial goal allocation** in `src/fpl_bot/models/bps.py`: replaces the per-player independent `Poisson(λ_p · m/90)` for goals with a team-conditional `Multinomial(team_score, p_share)` draw. Guarantees Σ player_goals = team_score per iteration. The explicit penalty mechanism is kept only for the missed-pen BPS deduction (scored pens are absorbed in λ_p via Phase 2.2 training data — no longer added on top).
2. **HiGHS MIP gap fix** in `src/fpl_bot/optim/milp.py`: discovered that `appsi_highs` ignores `solver.options[mip_rel_gap]` and requires `solver.highs_options[...]` instead. Set 1% relative gap (well below prediction noise). Without this, the multinomial-fix predictions' tighter LP-relaxation ties caused HiGHS to time out and SIGABRT on certain GWs.
3. **9 unit tests** in `tests/unit/test_multinomial_goal_sampling.py` covering the team-total invariant, edge cases (zero score, zero weights, bench players), proportional allocation, no-double-count for PK takers.

Acceptance gates (per §3 of the original design):
- [x] All 4 mandatory Phase 2.5 gates STILL PASS on walk-forward (folds 21-24): primary 4/4, calibration ≥8/10 deciles 4/4 (24/25 improved 9/10 → 10/10), captaincy 4/4.
- [x] Sampling correctness gate: Σ player_goals == team_score across 10k iterations (PASS).
- [x] FWD bias diagnostic — see `docs/design/phase2_5_1_results/walkforward_cv.txt` for the full comparison.

Phase 3 walk-forward (downstream check, NOT a Phase 2.5.1 acceptance gate):
- Folds 21/22/23 cleanly pass all 5 validity gates with multinomial + 1% MIP gap: 2550/+884, 2365/+608, 2487/+528.
- Fold 24/25 standalone-per-GW path: all 38 GWs solve optimal (total 2538 pts), but the `backtest_season` wrapper crashes mid-loop with SIGABRT in HiGHS — only on this fold, only via this entry point. Both subprocess isolation and proper MIP gap make folds 21-23 pass; fold 24 wrapper-instability is documented as a Phase 3 v1.4 follow-up. Multinomial fix is NOT the root cause (per-GW path proves the MILPs all solve); the trigger is some interaction between `backtest_season`'s per-GW lifecycle and HiGHS internal state on a specific 24/25 fixture. Investigation deferred — does not block Phase 2.5.1 ship since the prediction-layer is the actual deliverable.

The Phase 2.5 joint xPts simulator (`src/fpl_bot/models/bps.py:212-434`) correctly samples per-fixture team scores from Dixon-Coles market λ, then samples each player's goals **independently** from per-player Poisson(λ_p · minutes_p / 90). This breaks the within-fixture team-total constraint — a single MC iteration can produce home_score=2 (from team Poisson) while the home players' independent goal samples sum to 5. The Phase 2.5 review and `bps.py:10` flagged this as the priority v2 follow-up:
> Acknowledged as bonus-concentration distortion; PRIORITY V2 candidate is Multinomial(team_score, normalized_λ).

This phase implements that fix and re-runs the Phase 2.5 walk-forward to confirm the four gates still pass.

---

## 1. The bug, concretely

In `bps.py` per MC iteration:

```python
h_score = Poisson(inputs.home_team_lambda)        # line 275
a_score = Poisson(inputs.away_team_lambda)        # line 276

# THEN for every player on-pitch (lines 297-321):
for i in players:
    minutes_factor = minutes[i] / 90.0
    goals[i] = Poisson(λ_goals[i] * minutes_factor)   # ← independent per player
    assists[i] = Poisson(λ_assists[i] * minutes_factor)
```

The team-level `h_score` / `a_score` is used **only** for the BPS goals-conceded calculation (the *opponent's* team score determines GC). The per-player `goals[i]` are never reconciled with `h_score` / `a_score`. Effects:

- **Inflated goal counts**: E[Σ_i goals_i] under independent Poisson = Σ_i (λ_p · m_p / 90), which is calibrated on average to the team total via the per-player goals model — so the *expectation* matches. But the **variance** of the sum is too high (independent Poissons sum to a higher-variance distribution than constrained sampling), producing more high-goal tail outcomes per player than reality.
- **Bonus concentration distortion**: BPS heavily weighs goals (54-72 BPS each, depending on position). Within-fixture rank-based bonus allocation depends on whose BPS is highest. Inflated goal tails → inflated BPS tails → players who score 2+ goals in a single MC iteration capture *more* bonus than they should, biasing expected bonus upward for goal-scoring positions (FWD, attacking MID).
- **FWD under-prediction observation** (`phase2_5_joint_xpts.md:115`): mean predicted 1.685 vs actual 1.816 for FWD on 24/25 — 7% undershoot. The diagnostic ties this to the independent-sampling distortion via bonus over-concentration on a few high-tail outcomes that don't actually happen.

---

## 2. The fix

Replace the per-player independent Poisson **for goals only** with a two-step team-conditional sampler:

1. Sample team scores from Dixon-Coles λ (already done).
2. For each team, allocate that team's sampled score across team players via Multinomial weighted by each player's expected goal share `w_p = λ_goals[p] · minutes_factor[p]`.

Pseudocode replacing `bps.py:275-321`:

```python
h_score = rng.poisson(inputs.home_team_lambda)
a_score = rng.poisson(inputs.away_team_lambda)

# Per-team goal allocation
def _allocate(team_players, team_score):
    weights = [λ_goals[p] * minutes_factor[p] for p in team_players]
    total_w = sum(weights)
    if total_w == 0 or team_score == 0:
        return {p: 0 for p in team_players}
    probs = [w / total_w for w in weights]
    counts = rng.multinomial(team_score, probs)
    return dict(zip(team_players, counts))

goals = {}
goals.update(_allocate(home_players, h_score))
goals.update(_allocate(away_players, a_score))

# Assists, saves, cards etc. STAY independent (per-player Poisson) —
# they're not subject to the team-total constraint in the same way.
# (Assists ARE bounded by teammates' goals, but that's a v3 refinement.)
```

**Key design choices:**

| Decision | v1 choice | Rationale |
|---|---|---|
| **Goals only?** | Yes — assists / saves / cards stay independent | Goals are the BPS-dominant event and the visible distortion. Assists also have a team-level constraint (≤ goals scored by teammates) but the second-order effect on bonus is much smaller. v2 candidate. |
| **Multinomial weights** | `λ_p · minutes_factor[p]` | Same per-player rate the independent model uses. Players with 0 minutes get 0 weight → 0 goals (correct). |
| **Edge case: zero total weight** | All players get 0 goals (even if team_score > 0) | Happens for fully-benched teams; not a real fixture. Guard prevents division by zero. |
| **Own goals** | Ignored — sampler treats team_score as scored-by-our-team | FPL's "own goals" stat is recorded separately (penalty against the conceding team's defender). Frequency ~1% of goals. v2 candidate. |
| **Penalty conversions** | Not separately modeled — λ_p already absorbs each player's expected penalty conversions | Phase 2.2 goals model is total goals (including penalties). PK takers get higher λ_p naturally. |

---

## 3. Acceptance gates

### Mandatory: all four Phase 2.5 gates must STILL pass

1. **MAE on E[xPts] beats both baselines** (≥ 4/4 folds beat naive sum-of-components and position-marginal). Phase 2.5 baseline: ~2-3% margin over naive.
2. **Brier on P(xPts ≥ 6) beats both baselines** (≥ 4/4 folds). Phase 2.5 baseline: ~20% relative improvement over naive.
3. **Calibration within ±10% across E[xPts] deciles** (≥ 8/10 deciles, ≥ 4/4 folds). Phase 2.5 baseline: 9-10 of 10 deciles per fold.
4. **Captaincy gate** (argmax-E[xPts] captain beats rolling-3-GW leader). Phase 2.5 baseline: 4/4 folds.

### New: sampling correctness gate (unit test)

Over N=10,000 MC iterations of a synthetic fixture with `home_team_lambda=2.5`, `away_team_lambda=1.2`, and 11 home players + 11 away players each with `minutes=90, λ_goals=...`:
- For each iteration, `Σ_i goals_i for home_players == h_score` (exact equality, by construction).
- Same for away.
- Pass: 100% of iterations satisfy both equalities.

This is a structural guarantee from the multinomial; the test guards against future regressions.

### Diagnostic: distortion measurements (REPORTED, not gated)

- **FWD MAE on 24/25**: expect a measurable improvement vs Phase 2.5's 7% undershoot (target: ≤ 4%).
- **Per-decile calibration**: report all 10 deciles per fold. Expect tightening especially in deciles 9-10 (high-xPts haul outcomes) where bonus concentration matters most.
- **Captaincy uplift**: expect captain points per season to improve marginally on at least 2 of 4 folds (since captaincy is dominated by FWDs/attacking MIDs whose goal sampling tightens).
- **MAE bias by position**: report mean(predicted - actual) per position per fold. Target: |bias| ≤ 5% for FWD, ≤ 3% for other positions on 24/25.

---

## 4. Code changes

### `src/fpl_bot/models/bps.py`
The BPSSimulator's per-iteration loop changes inside `simulate_fixture`:
- Replace lines 297-321's per-player `goals[i] = Poisson(λ_p · m/90)` with a per-team Multinomial draw.
- Need to know which players are on which team (home/away). Already in `FixtureInputs.players` via `was_home` or team-id.
- Update the simulator constructor to accept the per-fixture home/away player split if it isn't already.

### Update `bps.py:10` header comment
Remove the "PRIORITY V2 candidate" line; replace with "v2 multinomial fix shipped in Phase 2.5.1."

### Tests (new)
- `tests/unit/test_multinomial_goal_sampling.py` — 5-6 tests:
  - Sampling correctness gate (10k MC, sums equal team scores)
  - Edge case: team_score=0 → all players get 0 goals
  - Edge case: zero total weights → all players get 0 goals (no DivisionByZero)
  - Bench players (minutes=0) get probability 0 → never sampled
  - Multi-rounds: PK takers (high λ_p) get more goals on average than non-PK teammates with same minutes
  - Distribution match: empirical per-player goal mean over 10k iters ≈ team_λ × p_share to within 5%

### Re-run walk-forward CV
- Re-run `eval/xpts_eval.py` walk-forward across folds 21-24
- Compare metrics vs Phase 2.5 archived results (already in `docs/design/phase2_5_results/`)
- Save new diagnostics under `docs/design/phase2_5_1_results/`

### Predictions cache invalidation
Phase 3's MILP backtest caches predictions at `data/cache/xpts_predictions/season_{N}_train_{...}.parquet`. After this fix, those caches are stale. Two options:
- **A**: invalidate (rename cache dir → `_pre_multinomial`); regenerate on next backtest run. Phase 3 walk-forward at H=6+chips needs a re-run to reflect new predictions.
- **B**: leave caches; they reflect Phase 2.5 v1 predictions; new caches under a different path.

**Recommend A**: cleaner mental model. Re-run Phase 3 walk-forward at the end of Phase 2.5.1 to confirm captain/FWD picks improve.

---

## 5. Build order

1. **Implement multinomial sampler** in `bps.py:simulate_fixture`. Smoke-test with a single synthetic fixture to confirm structural correctness.
2. **Write the unit tests**. Sampling correctness gate must pass.
3. **Re-run walk-forward CV** for Phase 2.5 (folds 21-24). Confirm all 4 mandatory gates still pass; report diagnostics.
4. **Invalidate predictions cache**, re-run Phase 3 walk-forward at picked (ρ=1.0, α=0.8, β=0.0, H=6+chips, cost-basis sell-tax). Compare per-fold totals to v3.5 baseline.
5. **Update Phase 2.5 design doc** to reference Phase 2.5.1 fix; mark the v2 candidate item as resolved.
6. **Two commits**: (a) sampler change + tests + Phase 2.5.1 design doc + Phase 2.5 results delta; (b) Phase 3 re-walkforward results.

---

## 6. Open questions

1. **Assists in the same fix?** Assists are bounded above by goals: a player can only be credited an assist for a teammate's goal. Independent assist sampling can produce assists > total team goals. Smaller distortion than goals (assists are fewer BPS each: 9 BPS vs 54-72), but real. **Recommend: ship goals-only fix in v1; assists v2 candidate.**

2. **Joint per-team allocation across goals AND assists?** A more correct sampler: sample team_goals → multinomial-allocate to scorers → for each scored goal, multinomial-allocate an assist among non-scorer teammates. This introduces dependencies between scorer and assister. **Recommend: not in v1; flagged as v3.**

3. **How to handle on-field substitutions during the match?** Current sim treats a player as on-pitch for `minutes_factor[p]` of the match. The multinomial weight `λ_p · m/90` already scales by minutes. A 30-minute sub gets ⅓ the weight of a 90-min starter at the same per-90 rate, which is correct on average. **No change needed.**

4. **Re-tune Phase 3 ρ/α/β at H=6 after the fix?** If the fix moves predictions meaningfully (FWD bias improvement, etc.), the picked ρ=1.0, α=0.8 from Phase 3 v1.2 might no longer be optimal. **Recommend: run a quick ρ sweep on fold 24/25 after Phase 2.5.1 lands; if signal, full re-tune. If picked unchanged, no action.**

5. **Backwards-compatibility / feature flag?** The simulator change is a single function. There's no compelling reason to keep the old (broken) sampler available. **Recommend: replace outright. The cache invalidation is the only "feature flag" needed.**

---

## 7. Visible post-2.5.1 follow-ups (do NOT block)

1. **Multinomial assists** (per §6.1).
2. **Goal-then-assist joint sampler** (per §6.2).
3. **Own-goals model** — currently absorbed into team_score with no per-player attribution. ~1% of goals; small impact.
4. **Penalty-shootout BPS** — separate from open-play; not modeled. Cup competitions only — irrelevant to PL.
