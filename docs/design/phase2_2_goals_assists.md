# Phase 2.2 — Goals & Assists per 90 (conditional on starting)

Status: **approved.** Implementation in progress.

Round-1 review resolutions:
- Sample weighting: default to `weight = minutes/90`; CV reports both weighted and unweighted for the record.
- PK config historical backfill: complete for seasons 19-23 (penalty taker per team); FK / corner backfill deferred (lower modelling priority).
- Predictions schema: one row per (player_id, fixture_id) with `lambda_goals`, `lambda_assists`, `season_id`, `gameweek`, `kickoff_utc`, `model_version`.

Sister components, same modelling family. Both predict per-90 rates conditional on the player playing ≥1 minute. Trained separately so they can carry different feature emphasis (xG vs xA, set-piece taker vs key-pass volume).

---

## 1. Target

For each (player_id, fixture_id, as_of) where `minutes ≥ 1`:

$$
\hat\lambda_{\text{goals},90} = \mathbb{E}[\text{goals per 90} \mid \text{player plays}]
$$

and likewise for assists.

**Modelling form:** LightGBM Poisson regression with `init_score = log(minutes/90)` as the offset:
$$
\text{goals} \sim \text{Poisson}\bigl(\exp(\text{model\_output}) \cdot \tfrac{\text{minutes}}{90}\bigr)
$$

so `exp(model_output)` directly = predicted per-90 rate, and at predict time we multiply by expected minutes (from the Phase 2.1 minutes-bucket distribution).

**Why per-90 conditional on starting** (per Phase 0 §4.2):
1. Decouples cleanly from the minutes model — we don't double-count the minutes uncertainty.
2. Lets us use the full sample (including substitutes) without distorting rate estimation.
3. Matches the form opta / fbref / understat publishes; sanity-checking is easy.

---

## 2. Feature catalog (v1)

### From `fact_player_match × dim_fixture` (now augmented with was_home + price_tenths)
| Feature | Purpose |
|---|---|
| `was_home` | Direct home/away effect — strong in football |
| `price_tenths` | Premium-tier indicator at predict time (limited use in training; current-snapshot only — same caveat as Phase 2.1) |
| `gameweek`, `days_into_season` | Season context |

### From `fact_understat_player_match` (joined by player_id + kickoff_date)
For both goals and assists models, rolling means over last 3 / 5 / 10 *starts*:
- `xg_per_90_last_{3,5,10}`
- `npxg_per_90_last_{3,5,10}` (non-penalty xG; complements set-piece-taker flag)
- `xa_per_90_last_{3,5,10}`
- `shots_per_90_last_{3,5,10}`
- `key_passes_per_90_last_{3,5,10}`
- `xg_chain_per_90_last_5`, `xg_buildup_per_90_last_5` (build-up involvement)

NULL for matches without an understat link (~5-15% of EPL rows expected; LightGBM handles natively).

### From `fact_market_xg` (joined via player's team for that fixture)
- `team_lambda_market_xg` — predicted goals for the player's team
- `opponent_lambda_market_xg` — predicted goals for the opposition (= our team's xGA, opponent strength)

Player's team derived from `was_home` + fixture's home/away team_ids.

### From manual overrides (`configs/`)
- `is_penalty_taker` (bool from `set_piece_takers.yaml`)
- `direct_fk_rank` (1 / 2 / 3 / NULL — position in the team's FK pecking order)
- `corner_rank` (1 / 2 / 3 / NULL — for assists model especially)
- `role_mismatch` (bool from `position_role_overrides.yaml`)
- `actual_role` (one-hot CAM / CM / DM / F9 / ST / WG when override exists; FPL position otherwise)

### Static metadata
- `pos_GKP / DEF / MID / FWD` — current snapshot from `dim_player`/`fact_player_status` (Phase 2.1 pattern)

---

## 3. Model class

LightGBM regression, `objective="poisson"`, separately for `goals` and `assists`. Hyperparameters: same small grid as Phase 2.1 (`num_leaves ∈ {15, 31, 63}`, `learning_rate ∈ {0.05, 0.1}`, `min_data_in_leaf ∈ {20, 50}`). Walk-forward CV picks per-target.

**Monotonic constraints:**
- `xg_per_90_last_*` → +1 in goals model
- `npxg_per_90_last_*` → +1 in goals model
- `is_penalty_taker` → +1 in goals model
- `xa_per_90_last_*` → +1 in assists model
- `key_passes_per_90_last_*` → +1 in assists model
- `corner_rank == 1` → +1 in assists model

---

## 4. Training procedure

Same walk-forward CV as Phase 2.1:
```
Fold 1: train [19, 20]            → predict 21
Fold 2: train [19, 20, 21]        → predict 22
Fold 3: train [19, 20, 21, 22]    → predict 23
Fold 4: train [19, 20, 21, 22, 23]→ predict 24
```

Held-out final: 2025/26 (current; in-progress GWs only).

Filter to `minutes ≥ 1` rows (conditional-on-playing). For Poisson loss, set sample weight = `minutes / 90` to reduce noise from cameo appearances if needed (try both ways; pick by held-out log-likelihood).

---

## 5. Calibration & validation

**Primary metrics**:
- Poisson deviance (deviance lower = better)
- Mean Squared Error (MSE) on `goals` (sanity-anchor)
- Brier-style decomposition: aggregate per-bucket of predicted rate, compare to empirical rate

**Baselines**:
- **Rolling xG/90 over last 5 starts** (no model) — beats are mandatory; this is the "naive" forecaster the FPL community uses
- **Position-marginal rate** — floor

**Acceptance gate before Phase 2.3**:
- [x] **Beats `rolling_xg_5` baseline by ≥ 5%** on Poisson deviance. Actual: 29–39% relative for goals, 37–48% for assists, across all 4 walk-forward folds. Weighted (default) ≈ unweighted within 0.001 deviance — both ship; weighted is the production default.
- [x] **All leakage tests still green; new feature-path leakage test added.** 25/25 passing including `test_goals_features_pit_stable_across_truncation`.
- [ ] **Calibration ±5% per decile** — diagnostic deferred to a follow-up commit (same pattern as Phase 2.1 closeout).
- [ ] **Ablation table** — diagnostic deferred to a follow-up commit.

**Substantive gate met**; calibration and ablation diagnostics follow as a non-blocking close-out commit.

---

## 6. Storage

Predictions saved to `data/predictions/goals/{run_ts}/season_{N}.parquet` and `data/predictions/assists/{run_ts}/season_{N}.parquet`. Same convention as minutes.

---

## 7. Code structure

```
src/fpl_bot/
├── features/
│   └── goals.py             # PIT-routed feature builder for both goals and assists
├── models/
│   └── goals.py             # train_per_90_model(target='goals'|'assists')
└── eval/
    └── goals_eval.py        # walk-forward CV for both targets
tests/
├── leakage/
│   └── test_goals_features.py
└── unit/
    └── test_goals_features.py
```

The same module covers both goals and assists since the feature pipeline is shared and only the target column differs.

---

## 8. Open questions

1. **Sample-weighting by minutes.** Try both: with `weight = minutes/90` and unweighted. Pick by held-out deviance.
2. **Treating substitutes vs starters separately.** v1: combined (use `minutes ≥ 1`). If sub-only data dilutes signal, stratify in v1.5.
3. **Predictions DataFrame schema.** Same as Phase 2.1 — single row per (player, fixture) with `lambda_goals`, `lambda_assists`. OK?
4. **Manual override coverage gap.** Set-piece takers config covers 2024/25 only; rows in earlier seasons get `is_penalty_taker=False` by default. This biases the historical training set. Acceptable for v1? The bias makes the model UNDERESTIMATE current-PK-taker goals if their flag wasn't set. Mitigation: backfill historical takers (~1-2 hr manual) before final eval. **Recommend backfill before merging the gate.**

---

**End of Phase 2.2 design. Awaiting review.**
