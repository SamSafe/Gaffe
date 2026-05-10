# Phase 3.5 — Dynamic Prices (Design)

Status: **partially shipped — cost-basis sell-tax LIVE, predictor SHELVED.**

Acceptance gates fired during implementation:
- Predictor walk-forward CV (folds 21-24, multinomial 5-class LightGBM with class-balanced weights): **all three gates FAIL.** brier_fall ≥ 0.05 across all folds (worse than the naive 5.5% base-rate predictor at ~0.049); top-k precision 0.32-0.42 means 2/3 of flagged rises/falls are wrong. Diagnosis: GW-end transfers_in/out is too late a signal — by the time we observe it, FPL's daily price-update threshold has already triggered (or not). v2 candidate: intra-week transfer snapshots.
- Per the §3.4 fallback path: predictor is **shelved** (code stays under `models/price_change/` and `eval/price_change_eval.py` for v2 to reuse the scaffolding); the cost-basis sell-tax change ships alone.

What ships in Phase 3.5:
1. **Schema/ingest**: 4 new columns on `fact_player_match` (`transfers_in/out/balance`, `selected`); vaastav backfill for seasons 19-24 (2018-19 errored on UTF-8 in raw CSV; not in our walk-forward train set).
2. **PIT primitive update**: `pit.all_player_match_with_kickoff` now dedupes by latest `recorded_at` per (player, fixture) — needed because the backfill creates a 2nd bitemporal row.
3. **Cost-basis sell-tax in the MILP harness**: `eval/milp_backtest._resolve_price_at_gw` now computes `sell_price[p] = current[p] - max(0, (current[p] - cost_basis[p])//2)` for state-squad players. `apply_gw_outcomes` is now policy-free w.r.t. price (consumes whatever the harness provides).
4. **Predictor scaffolding** (shelved but on disk): `features/price_change.py`, `models/price_change.py`, `eval/price_change_eval.py`, plus 14 unit tests.

Phase 3 v1.0/v1.1 used **static prices**: buy_price = sell_price = current price held constant over the rolling horizon, and the harness used the current price for sells (matching what the MILP saw). This is a known systematic distortion called out in the Phase 3 §12 follow-ups:
> Backtests over 38 GWs with static prices systematically under-value rising-price assets and over-value falling-price assets.

Phase 3.5 fixes this by:
1. Predicting per-(player, GW) price *changes* from `transfers_in/out` data.
2. Reactivating the cost-basis machinery in `apply_gw_outcomes` (already built into `BacktestState.cost_basis` forward-compatibly) so that sell prices reflect what was *paid*, not what the asset is *currently worth*.
3. Re-introducing buy/sell spread in the MILP (FPL charges a 50% sell-tax on profits — currently disabled under static-price v1).

---

## 1. Scope and v1 simplifications

| Decision | v1 choice | Rationale |
|---|---|---|
| **Prediction target** | Per-(player, GW boundary) discrete price change Δ ∈ {−2, −1, 0, +1, +2} (tenths) | Matches FPL's reality: prices move ±0.1m at most per night; ±0.2m is rare but possible across a long international break |
| **Prediction horizon** | One step per GW boundary | Multi-step predictions cascade error; the rolling-horizon MILP can stack 1-step predictions for the H=6 horizon |
| **Model class** | Multinomial logit (5-class) with class-imbalance handling | Class 0 is ~95% of rows; need calibrated probabilities not just argmax. LightGBM `objective="multiclass"` with class_weight is the candidate |
| **Features** | net-transfer-rate (transfers_balance / total managers), 7-day rolling rate, ownership level, position, days-since-last-price-change, fixture-difficulty-of-next-fixture | Reverse-engineered approximations of FPL's opaque algorithm |
| **Sell-tax modelling** | FPL exact rule: sell_price = cost_basis + floor((current - cost_basis) / 2) when current > cost_basis else current | Matches game rule. Cost basis already tracked in state.py |
| **Backtest carrier** | walk-forward folds 21-24, same as Phase 3 | Reuse the existing harness; minimal infra change |

---

## 2. Data ingest (small extension)

### Currently ingested (vaastav.py)
- `fact_player_match.price_tenths` (the `value` column from `gw*.csv`)

### Add to ingest
Extend `vaastav.py` to also capture from `gw*.csv`:
- `transfers_in` (count this GW)
- `transfers_out` (count this GW)
- `transfers_balance` (= in − out, redundant but stored for convenience)
- `selected` (raw count of managers owning the player at GW close)

These four columns become new fields on `fact_player_match` (or — if we want time-series within a GW — a new `fact_player_transfers` table; v1 says **just add to fact_player_match**, GW-end snapshot only).

**Backfill scope**: 2018/19 through 2024/25, all GWs, all players. Should be ~6 seasons × ~38 GWs × ~600 players = ~140k rows. Trivial.

**Schema migration**: 4 new nullable columns on `fact_player_match`. PIT primitives in `db/pit.py` then surface them via the existing `all_player_match_with_kickoff` (one-line extension).

---

## 3. Predictor

### 3.1 Label
`price_delta_at_gw_boundary` per (player_id, gw_close → gw_close+1):
- = +1 if price rises by 0.1m
- = +2 if price rises by 0.2m (rare)
- = -1 if drops by 0.1m
- = -2 if drops by 0.2m (rare)
- = 0 otherwise (~95% of cases)

Computed from consecutive `price_tenths` values across GW boundaries.

### 3.2 Features
Per (player_id, target_gw_boundary), all PIT-correct (computed from data ≤ end of source GW):

| Feature | Definition |
|---|---|
| `net_transfer_rate` | `transfers_balance / total_managers_in_game` (latest GW close) |
| `net_transfer_rate_lag1` | Same, prior GW |
| `transfers_in_rate` | `transfers_in / total_managers` |
| `transfers_out_rate` | `transfers_out / total_managers` |
| `selected_rate` | `selected / total_managers` (ownership level) |
| `position_code` | one-hot {GKP, DEF, MID, FWD} |
| `current_price_tenths` | absolute price (rises rarer at high prices) |
| `price_lag1` | price at prior GW close (for staleness check) |
| `gw_in_season` | absolute gameweek number (FPL throttles changes near deadline edges) |

**`total_managers_in_game`**: a single per-season scalar (~10M for 24/25). The FPL price-change algorithm normalizes by this, but our raw `transfers_balance` count is already what matters for relative thresholds. Use a constant-per-season approximation; document as a v2 candidate to time-vary.

### 3.3 Model
LightGBM `objective="multiclass"`, `num_class=5`, `class_weight="balanced"` to handle the 95/5 imbalance. Same walk-forward CV harness as `eval/goals_eval.py` and `eval/clean_sheet_eval.py`. Calibration matters more than argmax accuracy.

### 3.4 Acceptance gate

Three gates, ALL MUST PASS:

1. **Calibration**: Brier on the +1/-1 vs not-zero binary marginals ≤ 0.04 on ≥ 3 of 4 walk-forward folds. (Tighter than CS because price-change is more deterministic given the right signals.)

2. **Top-k precision**: of the players the predictor flags as P(rise) ≥ 0.5 on a given GW boundary, ≥ 60% should actually rise. Symmetrically for falls. Measured per-fold.

3. **MILP integration smoke**: with the predictor wired into `MilpInputs.buy_prices` and `sell_prices` (sell uses cost-basis + sell-tax rule), the rolling MILP fold 24/25 must:
   - Continue to pass all 5 validity gates from Phase 3
   - Show ≥ +5 pts/season improvement over static-price v1.1 baseline (small, since prediction is the new variable; we just need a non-regression signal that dynamic prices aren't actively hurting)

If any gate fails: predictor is shelved as `models/price_change/`; production stays on static prices. The MILP code path that consumes dynamic prices is gated behind a `USE_PRICE_PREDICTOR` flag.

---

## 4. MILP integration

### Currently
`apply_gw_outcomes` uses `actual_prices[p]["sell"] = current_price` (Phase 3 v1 simplification). MILP sees `buy = sell = current_price`. Cost basis is tracked but unused.

### Phase 3.5 changes
1. **Per-GW price predictions** flow into `MilpInputs.buy_prices` and `sell_prices`. The predictor outputs E[price next GW] = current + Σ_k k·P(Δ=k); the MILP's static-buy-over-horizon assumption stays (all buy_prices in the horizon use the GW1 price), but each rolling solve gets fresh predictions.
2. **Cost-basis sell pricing** in the harness: `sell_price[p] = cost_basis[p] + max(0, floor((current[p] - cost_basis[p]) / 2))` — restores the game's actual sell-tax rule. The MILP's `sell_prices[p]` parameter should match what the harness will use.
3. **Cost basis carry-over** in `apply_gw_outcomes`: already implemented, just needs to be the active code path again (remove the v1.0 static-price override).

### What does NOT change
- The MILP formulation itself (variables, constraints, objective): unchanged.
- The bank-evolution constraint: unchanged (still `bank[w] = entering + sells - buys`; the dollar amounts just use the new sell-tax rule).
- The candidate filter, EO, terminal value: unchanged.

This is genuinely a **drop-in pricing change**. Most of the complexity is in the predictor, not the MILP.

---

## 5. Backtest plan

### Pre-3.5 housekeeping (cheap, ship before predictor work)
A — α/β re-tuning at H=6 + chips on fold 24/25 (already run) showed α/β are NOT inert at H=6 — the grid spans 2481–2562 pts. Best passing config: **α=0.8, β=0.0**.
B — That same sweep surfaced a tractability flake: 3 of 9 configs hit the rolling-solve 60s time limit on a single early GW with no feasible solution found, while β-equivalent configs pass. Bump rolling-solve `time_limit_s` from 60 → 120 in `eval/milp_backtest.py`. Cold-start stays at 180s.

These two items ship as a Phase 3 v1.2 housekeeping commit *before* any 3.5 work:
1. Bump `time_limit_s = 60 → 120` for rolling solves.
2. Re-run full walk-forward at picked α=0.8, β=0.0 to confirm the +81 absolute pts/season uplift on fold 24/25 generalizes; expectation: similar uplift on folds 21-23.
3. Update Phase 3 design doc walk-forward result table.

### Phase 3.5 backtest plan (after predictor passes its 3 gates)
1. Re-run full walk-forward (folds 21-24, H=6 + chips, ρ=1.0, α=0.8, β=0.0) with dynamic prices.
2. Report Δ vs Phase 3 v1.2 (static prices, α=0.8) per fold — expectation: +5 to +30 pts/season uplift, primarily from no longer over-paying for price-falling assets.
3. Re-run α/β grid at H=6 + chips + dynamic prices on fold 24/25 — the value of holding rising-price assets (β term) should now bite, since β was inert under static prices.

Validity gates (from Phase 3) must continue to pass.

---

## 6. Open questions — decisions

1. **Class granularity**: **5-class** {-2,-1,0,+1,+2} with `class_weight="balanced"`. Keeps the rare ±2 tail; class_weight handles imbalance. The MILP integration consumes E[Δ] = Σ_k k·P(Δ=k) so any miscalibration on the ±2 tail damps out in the expectation.

2. **Total-managers normalizer**: **per-season constant**. Within-season fluctuation is ~flat post-GW3 (~10M for 24/25); the relative ranking that matters for FPL's threshold is preserved. v2 candidate to time-vary.

3. **Calibration evaluation**: **report all three** — multiclass log-loss, marginal Brier per direction (P(Δ=+k) for k=1,2 each), top-k precision at threshold 0.5. Argmax accuracy is dominated by the "predict 0" trivial baseline so we don't gate on it.

4. **Reciprocal-fold leakage check**: vaastav's `gw{N}.csv` snapshots `selected` at *post-GW close*, which is the boundary the price change is measured at. Features for predicting Δ at boundary {N → N+1} use only data from gw{N}.csv (closed) — this is naturally PIT. Leakage test: assert features for season 22 are byte-identical when computed over [19,20,21,22] vs [19,20,21,22,23,24].

5. **Cost basis at cold start**: confirmed correct in `state.py` — `cost_basis[p] = buy_price` for all GW1 picks (set in `apply_gw_outcomes`). Test coverage already exists in `tests/unit/test_state_transitions.py::test_cold_start_cost_basis`. (If test isn't there: add it.)

6. **Sell-tax floor on losses**: **confirmed** — FPL applies sell-tax only on profits. The formula `sell = cost_basis + max(0, floor((current - cost_basis)/2))` correctly returns `current` when `current ≤ cost_basis`. No sell-tax on a falling asset.

---

## 7. Code structure

### New files
- `src/fpl_bot/features/price_change.py` — feature builder (PIT-correct, mirrors clean_sheet.py shape)
- `src/fpl_bot/models/price_change.py` — LightGBM multiclass + predict API
- `src/fpl_bot/eval/price_change_eval.py` — walk-forward CV, computes 3 gates
- `src/fpl_bot/eval/price_change_diagnostics.py` — per-class confusion, calibration plots, feature importance
- `tests/unit/test_price_change_features.py` — label/feature correctness
- `tests/leakage/test_price_change_features.py` — PIT-stable across truncation
- `docs/design/phase3_5_results/` — diagnostic artifacts

### Modified files
- `src/fpl_bot/ingest/vaastav.py` — add 4 column extraction from gw*.csv
- `src/fpl_bot/db/models.py` — add 4 columns to `fact_player_match`
- `src/fpl_bot/db/pit.py` — surface 4 columns in `all_player_match_with_kickoff`
- `src/fpl_bot/optim/state.py` — un-comment / activate the cost-basis sell-price logic in `apply_gw_outcomes`
- `src/fpl_bot/eval/milp_backtest.py` — switch from `current_price` for sells to the cost-basis + sell-tax rule; add `USE_PRICE_PREDICTOR` flag wiring
- `docs/design/phase3_deterministic_milp.md` — add a "v1.2 superseded by Phase 3.5" line near the static-price simplifications block

### Migration
A new Alembic-style migration adds 4 nullable columns to `fact_player_match`. Backfill is a single `vaastav.py` re-run with `--reingest=transfers` flag.

---

## 8. Build order

1. **Ingest extension**: 4 columns to schema + vaastav.py + backfill. Run leakage tests; confirm new columns are PIT-stable.
2. **Feature builder**: `features/price_change.py`. Smoke-test that table builds for one season.
3. **Model**: LightGBM multiclass; quick train on 19-23, predict on 24, sanity-check confusion matrix.
4. **Walk-forward eval**: full CV with 3 gates. **Decision point**: gates pass → wire into MILP; gates fail → shelve predictor and ship cost-basis sell-tax change alone (still a Phase 3.5 win because it removes the static-sell distortion).
5. **MILP integration**: activate cost-basis sell logic + (if predictor passes) wire predictor outputs into buy/sell prices.
6. **Re-run walk-forward H=6 + chips with dynamic prices**: report Δ vs static.
7. **α/β re-tune at H=6 + chips + dynamic prices** on fold 24/25.
8. **Diagnostics + commit + design doc → status: shipped.**

---

## 9. Visible post-3.5 follow-ups (do NOT block)

1. **Time-varying total-managers normalizer**: as season progresses, total manager count creeps up. Ignore for v1; revisit in v2.
2. **Live-only price signals**: FPL API provides `cost_change_event` (the change *during* the current GW). Could be a feature for live predictions but not for historical training. Add when live-loop is built.
3. **Multi-step predictions**: 1-step is what we ship. Stacked 2-3 step might add value for long horizons; defer until v1 MILP integration is shipped.
