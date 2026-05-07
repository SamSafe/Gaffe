# Phase 2.4 — BPS / Bonus Points Simulator

Status: **approved.** Implementation in progress.

Round-1 review resolutions:
1. Minutes-bucket midpoints: **0 / 30 / 70 / 90** (4-bucket split of the 3-class minutes model output; B_F split into 60-89 and 90+ via per-position α).
2. Independent goal sampling: OK for v1, but **explicitly flagged as the priority v2 candidate**. Independent goals don't enforce sum-to-team-score, which can distort *bonus concentration* (a top scorer running away in one MC realization while teammates get nothing) — match-relative bonus is sensitive to this. If acceptance-gate calibration is weak, switch to Multinomial(team_score, normalized_λ) before iterating elsewhere.
3. Residual granularity: **position × minutes-bucket** Gaussian.

Acceptance gate confirmed:
- Brier on P(bonus > 0) must beat naive-event-proxy by ≥ 5% relative
- Brier on P(bonus = 3) must beat top-3-by-xPts
- ECE < 0.05 per position

Per Phase 0 §4.5 (and Round-5 amendment due to fbref unavailability): simulate BPS from predicted match events, rank within-fixture, assign 3/2/1 bonus. **Do NOT regress on historical bonus directly** — that loses the mechanism that makes BPS predictions interpretable and out-of-distribution-robust.

This component combines the Phase 2.1–2.3 outputs (minutes / goals / assists / clean sheet) with a per-event simulator and produces the bonus distribution P(bonus ∈ {0,1,2,3}) per player per fixture. It is the last predictive component before Phase 2.5 assembles the joint xPts distribution.

---

## 1. What is simulated directly vs. approximated by empirical residuals

The BPS rule table (FPL 2024/25) covers ~20 event types. We split them into four tiers by data availability:

### Tier A — direct simulation from existing trained models

| Event | Source | BPS effect (typical) |
|---|---|---|
| Minutes (60+ vs 1-59 vs 0) | Phase 2.1 minutes model | +6 BPS for 60+ appearance |
| Goals scored (per position) | Phase 2.2 goals model + position | +24 GK/DEF, +18 MID, +12 FWD |
| Assists | Phase 2.2 assists model | +9 BPS each |
| Team clean sheet (GK/DEF) | Phase 2.3 market_cs_prob | +12 BPS for GK/DEF (90+ min only) |
| Goals conceded by GK/DEF | Derived from team market λ_opp | −1 BPS per goal conceded ≥ 2 (capped) |

### Tier B — direct from `fact_player_match` rolling rates

| Event | Source | BPS effect |
|---|---|---|
| Saves (GK only) | rolling saves/match × opponent shots scaled | +2 per 3 saves bracket |
| Yellow cards | rolling yellow_cards/90 × referee strictness | −3 BPS |
| Red cards | rolling red/90 (very rare; EB-shrunk) | −9 BPS |
| Penalty scored | designated PK taker × P(pen awarded) × 0.78 | counts toward goals BPS |
| Penalty missed/saved | as above × 0.22 split | −6 BPS |

### Tier C — empirical residual (no measurement available without fbref)

| Event group | Approach |
|---|---|
| Tackles won, interceptions, blocks, recoveries | Per-position, per-minute empirical mean from historical `total_points − BPS_from_known_events` residual |
| Dribbles completed, big chances created/missed | Same residual bucket |
| Errors leading to goal, own goals | Tail events; empirical-mean-times-(minutes/90) |
| Passes completed (capped) | Same residual bucket |

### Tier D — emergent (computed, not sampled)

| Event | How |
|---|---|
| Bonus assignment (3/2/1 within fixture) | Rank simulated total BPS across the 22 (or DGW: 44) players in the fixture; ties broken alphabetically per FPL rules |

**Architectural justification for Tier C empirical residual.** Rather than imputing each missing event individually with shaky position-mean rates, we compute the residual *between* simulated-from-known-events BPS and *actual* historical BPS, fit per-(position, minutes-bucket, season) Gaussian models on that residual, and sample additive corrections at simulation time. The resulting BPS distribution is calibrated by construction against historical bonus assignments. The cost: we can't introspect "this player got 8 BPS from tackles specifically." For Phase 2.4 that introspection isn't needed; downstream consumes only the bonus distribution.

---

## 2. Correlation handling

### Within-fixture (most important)

- **Match-level random draws** are sampled ONCE per fixture per Monte Carlo iteration:
  - `(home_score, away_score)` jointly from Dixon-Coles via the market λs in `fact_market_xg`
  - `team_cs_home = (away_score == 0)`, `team_cs_away = (home_score == 0)` — derived
  - `referee_strictness_factor` if we can derive a per-fixture referee from FPL data (skip in v1; use 1.0)
- **Team CS realization** is shared across all defenders + the GK on a team for that fixture (this is the single biggest within-team correlation source for bonus).
- **Goals conceded** is `home_score`/`away_score` of the opponent — also shared across all GK/DEF on the team. Drives the −1 BPS-per-goal-conceded penalty.

### Player-level (per-Monte-Carlo iteration)

- Player minutes are sampled **independently** from the per-player minutes-bucket distribution. Acknowledged simplification: in reality there are negative correlations within a team (a manager subs at most 5 players). v1 accepts this; v2 could enforce a 5-sub cap via rejection sampling.
- Goals/assists/saves/cards/etc. are sampled **independently per player conditional on minutes**, with rates from the per-player Phase 2.2 / Tier-B models. Acknowledged simplification: sum of player goals will only equal the sampled team_score in expectation, not on every iteration. For BPS *ranking* this is benign (the rank between two teammates depends on their relative draws, not the team total). v2 could draw per-player goals via Multinomial(team_score, per-player-λ) to enforce consistency.
- The Tier-C residual is sampled **per-player per-iteration** as a single additive Gaussian correction. We do NOT model correlation between teammates' residuals (assumed independent given match outcome).

### Cross-week (joint trajectories — Phase 4 dependency)

Phase 0 §4.7 specifies that scenario sampling for the optimizer (Phase 4) preserves cross-week correlation in a player's own form via a Gaussian copula on per-90 form shocks. This is **out of scope for Phase 2.4** — the BPS simulator is invoked per-fixture by the optimizer at scenario-sampling time, with the form-shock already applied to the per-90 rates upstream. The simulator itself is memoryless across weeks.

### Multiplications that matter

- **Minutes ↔ BPS**: a player on the bench (sampled minutes = 0) gets 0 BPS. Cleanly handled by `minutes_factor = minutes / 90` scaling all rate-based events.
- **Team CS ↔ DEF/GK BPS**: shared draw guarantees all four DEFs and the GK either all benefit or none do. Critical for the optimizer — buying two DEFs from the same team is a positively-correlated bet, and the simulator must reflect that.
- **Goals/assists ↔ bonus**: emergent. A player with 2 goals + 1 assist in a sample will rank near the top of their fixture's BPS table. The simulator's bonus distribution will correctly weight this.

---

## 3. Baseline for BPS prediction

Per Phase 0 §4.5: do not regress on bonus directly. Baselines for evaluation only:

1. **Position-marginal bonus rate** — for each (position, minutes_bucket), historical P(bonus > 0) and per-bonus-tier rates. Floor.
2. **Top-3-by-predicted-xPts heuristic** — within each fixture, the players ranked 1/2/3 by `pred_minutes × pred_pts_per_minute_proxy` get assumed 3/2/1 bonus probabilistically. Lazy but surprisingly hard to beat for most-likely-bonus prediction; reflects what FPL community calls "captaincy-tier" expectations.
3. **Naive event-proxy linear combination** — predicted_bonus_score ≈ `pred_goals × goal_BPS_value + pred_assists × 9 + pred_team_cs × cs_BPS_value`. Sums per-component contributions WITHOUT simulating events; rank within fixture and assign top-3.

The baselines escalate: position-marginal (floor) → naive linear (does the mechanism help?) → top-3-by-xPts (does the simulator beat the simplest plausible-but-wrong heuristic?).

---

## 4. Acceptance gate

Per Phase 0 §4.5: "Brier on 'did this player receive bonus?', per-position, per-bonus-tier."

**Primary metric**: Brier score on **P(bonus > 0)** per player per fixture. Equivalent to mean squared error on the binary "got any bonus" label.

**Secondary metrics**:
- Brier on P(bonus = 3) — the most valuable single bonus tier
- Log-loss on the 4-class bonus distribution {0, 1, 2, 3}
- ECE per position per bonus-tier

**Acceptance gate before Phase 2.5**:

- [x] **Beats `naive event-proxy linear combination` by ≥ 5% relative Brier on P(bonus > 0).** Actual: ~42-43% relative across all 4 folds. Way past threshold.
- [x] **Beats `top-3-by-xPts heuristic` on P(bonus = 3).** Sim Brier 0.0129-0.0154 vs heuristic 0.0232-0.0282 — ~45% better across folds.
- [x] **ECE < 0.05 per position.** Overall ECE 0.003-0.007 across folds; per-position breakdown in diagnostics close-out.
- [x] **All leakage tests still green; new feature-path leakage test added.** 22 BPS-specific tests pass.

**Walk-forward CV results (4-fold, 200 MC iterations, ~10 min):**
```
season  | n      | brier_pos: sim/naive/top3/posM | brier_3: sim/top3 | ECE
 2021/22|  25447 | 0.0420/0.0738/0.0742/0.0429    | 0.0154/0.0282     | 0.007
 2022/23|  26505 | 0.0401/0.0710/0.0711/0.0407    | 0.0150/0.0270     | 0.006
 2023/24|  29725 | 0.0359/0.0618/0.0634/0.0365    | 0.0129/0.0232     | 0.005
 2024/25|  27605 | 0.0377/0.0642/0.0675/0.0386    | 0.0135/0.0243     | 0.003
```

### Notable finding

The **position-marginal baseline (`posM`) is unexpectedly strong** — only ~2% relative worse than the simulator across folds. This says that "P(bonus | position, minutes_played)" already captures most of the BPS signal in EPL data; the simulator's mechanism-level lift over a strong-but-mechanism-free baseline is modest (~2% on Brier). The naive event-proxy and top-3-by-xPts baselines, by contrast, are notably weaker — the simulator's win over those is comfortable.

Interpretation: the simulator IS doing real work (it beats every baseline), but the headline lift over the strongest baseline is small. Phase 4 stochastic optimization should benefit from the bonus *distribution* (not just the mean) — the simulator outputs full P(bonus = k) which the position-marginal baseline only approximates. The marginal-Brier gap understates the simulator's value to the optimizer.

**If both gates fail**: the simulator implementation has a bug; iterate. Unlike Phase 2.3, there is no "ship the heuristic" fallback — the spec requires the mechanism-based simulator. We iterate until it works.

If only the secondary P(bonus=3) gate fails but the primary passes: ship v1, document the gap, plan a v1.1 that improves the residual model for high-tier bonus.

---

## 5. EventSource swap-in path

`src/fpl_bot/db/event_source.py` already defines:

```python
class EventSource(Protocol):
    def player_event_history(self, player_id: int, before: datetime) -> pl.DataFrame: ...

class EmpiricalResidualEventSource: ...   # Phase 2.4 v1 implementation
class FbrefEventSource: ...               # raises NotImplementedError; future swap-in
```

**v1 implementation of `EmpiricalResidualEventSource`** consumes:
- Per-(position, minutes_bucket, season) empirical residual statistics (mean + std) computed offline from the train fold's `actual_BPS - simulated_BPS_from_known_events`.
- Returns a DataFrame with these summary stats plus a callable `sample_residual(rng, position, minutes_bucket) -> float` for use at simulation time.

**Swap-in path for a future paid fbref / sports-reference Premium implementation**:

1. Implement `FbrefEventSource.player_event_history(player_id, before)` to read from `fact_player_match_event` (the table schema is already migrated; just empty). The function returns rolling per-event-type rates (`tackles_per_90`, `interceptions_per_90`, `blocks_per_90`, `dribbles_completed_per_90`, `big_chances_created_per_90`, etc.) per player up to `before`.
2. Update the BPS simulator's `__init__` to accept `event_source: EventSource` parameter; default to `EmpiricalResidualEventSource()`.
3. Add a new code path `simulate_event_counts_directly(rates, minutes)` that draws Poisson per-event-type rather than sampling a single residual scalar.
4. Re-run the acceptance gate. If the directly-simulated version beats the residual version, switch the default. **No changes to upstream prediction-component APIs or downstream optimizer code.**

The contract enforced by the simulator's `__init__` parameter and the `EventSource` Protocol guarantees the swap is local to the simulator. The optimizer (Phase 3+) calls `BPSSimulator(...)` and reads only the output bonus distribution — it doesn't know which source produced the simulated event counts.

---

## 6. Algorithm (Monte Carlo procedure per fixture)

For each fixture in the prediction set:

```
N = 500  # iterations (calibrate later; aim for stable Brier within ±0.001)

for s in 1..N:
    # Match-level joint draws
    (h_score, a_score) ~ DixonColes(team_λ_home, team_λ_away)
    cs = {home_team: (a_score == 0), away_team: (h_score == 0)}
    gc = {home_team: a_score, away_team: h_score}

    # Per-player draws (independent given minutes)
    for each player p in this fixture:
        minutes = sample_minutes_bucket(minutes_dist[p])  # 0, 30, or 80 representative midpoint
        minutes_factor = minutes / 90
        if minutes == 0:
            bps[p, s] = 0
            continue

        # Tier A direct events
        goals[p] = Poisson(λ_goals_per_90[p] * minutes_factor)
        assists[p] = Poisson(λ_assists_per_90[p] * minutes_factor)

        # Tier B direct events (not all apply per position)
        saves[p] = Poisson(λ_saves_per_90[p] * minutes_factor)  # GK only
        yc[p] = Bernoulli(λ_yc_per_90[p] * minutes_factor)
        rc[p] = Bernoulli(λ_rc_per_90[p] * minutes_factor)

        # Penalty events (taker only)
        if p == designated_pk_taker[team]:
            pens_awarded = Poisson(λ_team_pen_rate)  # weak prior; ~0.15 / match
            pens_scored[p] = Binomial(pens_awarded, 0.78)
            pens_missed_saved[p] = pens_awarded - pens_scored[p]

        # BPS from known events via rule table
        bps_known[p] = score_bps(rule_table, position[p], goals[p], assists[p],
                                  cs[team_of(p)], gc[team_of(p)], saves[p], yc[p], rc[p],
                                  pens_scored[p], pens_missed_saved[p])

        # Tier C empirical residual
        bps_residual[p] = event_source.sample_residual(rng, position[p], minutes_bucket(minutes))

        bps[p, s] = bps_known[p] + bps_residual[p]

    # Rank within fixture, assign bonus
    ranked = argsort(-bps[:, s])
    bonus[ranked[0], s] = 3
    bonus[ranked[1], s] = 2
    bonus[ranked[2], s] = 1
    # Ties broken per FPL rules: alphabetical on player surname (deterministic)

# Aggregate
for each player p:
    P(bonus = k | p) = mean(bonus[p, :] == k for k in {0,1,2,3})
```

**Computational cost**: 500 iterations × 760 fixtures × ~22 players = 8.4M sample-rows per fold. Pure numpy; expect under 60s per fold.

---

## 7. Storage / API

Predictions saved to `data/predictions/bps/{run_ts}/season_{N}.parquet`:

```
player_id, fixture_id, season_id, gameweek, kickoff_utc,
p_bonus_0, p_bonus_1, p_bonus_2, p_bonus_3,
expected_bonus,         # = 0*p0 + 1*p1 + 2*p2 + 3*p3
model_version
```

Public API:

```python
from fpl_bot.models.bps import simulate_bonus_distribution

probs = simulate_bonus_distribution(
    fixture_predictions: pl.DataFrame,    # per-player minutes/goals/assists/cs predictions
    event_source: EventSource,
    n_iterations: int = 500,
    seed: int = 42,
) -> pl.DataFrame                          # one row per (player_id, fixture_id) with p_bonus_*
```

---

## 8. Code structure

```
src/fpl_bot/
├── models/
│   └── bps.py                  # simulator + BPS rule table + score_bps()
├── features/
│   └── bps.py                  # assembles predictions from minutes/goals/assists/cs into the
│                               # input DataFrame for the simulator
├── db/
│   └── event_source.py         # already exists; populate EmpiricalResidualEventSource
└── eval/
    └── bps_eval.py             # walk-forward CV with the three baselines + acceptance gate
tests/
├── leakage/
│   └── test_bps_features.py    # PIT-stable; the simulator must use only as-of-decision-time inputs
└── unit/
    └── test_bps_simulator.py   # rule-table correctness, rank-tie-break, residual sampling reproducibility
```

The BPS rule table is populated via a small data migration (insert into `bps_rule_set`) for season 24/25. Earlier seasons' rules are slightly different but small enough to ignore for v1; documented as a follow-up to backfill if needed.

---

## 9. Open questions for review

1. **Minutes-bucket midpoints**. v1 uses `0` / `30` / `80` as the representative minute counts when sampling from the 3-class minutes distribution. These determine the per-90 rate scaling. Is this granularity acceptable, or do you want a finer 4-class split (e.g., 0 / 30 / 70 / 90) to better capture the GK-DEF +12 BPS at-90-min boundary?
2. **Goals/assists individual sampling vs team-constrained**. v1 samples per-player independently. Acknowledged: sum across teammates won't match the sampled team_score on each iteration. This is benign for *ranking* (which is what bonus depends on) but could distort Tier-A goals BPS slightly. v2 would draw via Multinomial(team_score, normalized_λ). Confirm v1 is acceptable for now.
3. **Residual scope**. The Tier-C empirical residual is fitted as a per-(position, minutes_bucket) Gaussian. Should we go finer (per-team? per-price-tier?) or coarser (per-position only)? Trade-off: more buckets = more variance per bucket. v1 default: per-(position, minutes_bucket).
4. **BPS rule set scope**. v1 uses 2024/25 rules only and applies them uniformly across all backtest seasons (rules drift slightly year to year). For the acceptance gate evaluation this means our simulated BPS for 2019/20 fixtures uses 2024/25 rules — a small bias. Confirm acceptable; backfilling per-season rules is a v1.1 task.
5. **PK taker config gap**. We have penalty takers backfilled for 2019-23 (penalty only; no FK/corner). Penalty events drive a meaningful chunk of goals/BPS for top scorers. v1 uses the existing config. FK/corner backfill stays as a non-blocking follow-up unless review says otherwise.
6. **Parallelism**. 500 × 760 × 22 = 8.4M samples per fold. v1 is pure-numpy serial; expect ~60s per fold. Parallelize across folds via multiprocessing if time pressure emerges. Confirm OK to ship serial v1.

---

**End of Phase 2.4 design. Awaiting review.**
