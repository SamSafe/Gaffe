# Phase 2.1 — Minutes Model

Status: **approved.** Implementation in progress.

> **Data-availability addendum (post-approval):** `fact_player_status` was only snapshotted at the FPL ingest run, not GW-by-GW historically. So the live-only fields (`status_code`, `chance_of_playing_next_round`, `news`, `selected_by_percent`) are absent for backtest training rows. We use them at PREDICT time only. Similarly, vaastav's per-GW `value` (price) and `was_home` are not yet ingested into `fact_player_match`; v1 trains without them. Adding them to the schema is a follow-up; the v1 feature list below already excludes the deferred fields. Position is from the current `dim_player`/`fact_player_status` snapshot — treated as effectively static metadata (positions change rarely; the leakage cost is negligible vs. losing the feature entirely).

The minutes model is the bottleneck of the xPts cascade — every other component (goals, assists, CS, BPS) is conditional on the player starting, so a 5pp error here dominates downstream errors. Spec emphasizes disproportionate effort.

---

## 1. Target

Multi-class classification per `(player_id, fixture_id, as_of)`:

$$
\text{minutes\_bucket} \in \{\,B_0=0,\; B_S=[1,59],\; B_F=[60,90+]\,\}
$$

Bucket choice rationale: FPL appearance points are step-functions exactly at these breakpoints (0 → 0 pts, 1–59 → 1 pt, 60+ → 2 pts). Sticking with 3-class per Phase 0 §4.1; finer-grained (e.g., separating 90-min from 60-89) is a Phase 2.5 refinement if calibration in $B_F$ proves heterogeneous.

Output: full categorical distribution $P(B \mid \text{features})$, not just argmax. Downstream BPS and joint-trajectory sampling consumes the distribution.

---

## 2. Feature catalog

Each feature is point-in-time-routed through `fpl_bot.db.pit`. **Leakage cutoff** = the latest data the feature is allowed to use given an `as_of` timestamp; if any feature reads past this, it leaks.

### Available now (from Phase 1B corpus)

| Feature | Source | Leakage cutoff | Notes |
|---|---|---|---|
| `min_last_1`, `min_last_3`, `min_last_5`, `min_last_10` | `fact_player_match` × `dim_fixture.kickoff_utc` | latest fixture with `kickoff_utc < as_of` | Multiple windows; let LightGBM choose. |
| `min_60_streak` | as above | as above | Consecutive-60+ runs are a strong starter signal. |
| `started_last_gw` | as above | as above | Bool collapse of `min_last_1 ≥ 60`. |
| `days_since_last_match` | `dim_fixture.kickoff_utc` | as above | Capped at 21 days (international break ceiling). |
| `matches_last_7d`, `matches_last_14d` | `dim_fixture` | as above | Fixture congestion. |
| `position` | `fact_player_status.position_code` | latest `recorded_at ≤ as_of` | One-hot {GKP, DEF, MID, FWD}. |
| `price_tenths` | `fact_player_status` | as above | Premium players are nearly-always-starters. |
| `price_tier` | derived | as above | Decile within position; smooths price noise. |
| `status_code` | `fact_player_status` | as above | One-hot {a, d, i, n, s, u}. |
| `chance_of_playing_next_round` | `fact_player_status` | as above | NULL → −1 sentinel. |
| `news_has_keyword_*` | `fact_player_status.news` | as above | Keyword features: `injured`, `doubtful`, `suspended`, `fitness`, `illness`, `knock`. |
| `days_into_season` | derived from kickoff vs season start | n/a (deterministic) | Captures preseason rotation, end-of-season rest. |
| `team_strength` | derived from `fact_market_xg` rolling | latest market_xg row before kickoff | Stronger teams rotate more in cup-week fixtures. |

### Deferred (data unavailable or low-yield)

| Feature | Why deferred |
|---|---|
| Manager rotation rate | No reliable free source for manager-team-tenure mapping (same data-sourcing class as fbref). Scope-out for v1; revisit if a clean source emerges. Treated as a known-missing feature, NOT silently absent. |
| UEFA midweek fixture flag | Requires cup/Europe schedules. Phase 2.5 refinement: enumerate UCL/UEL fixture dates per club from a small hand-curated config. |
| Player-specific fitness model (returning from injury) | Would help but needs dose-response data we don't have. Captured loosely via `chance_of_playing_next_round`. |

---

## 3. Model class

**LightGBM multi-class** (`objective='multiclass'`, `num_class=3`).

**Monotonic constraints** where they're physically defensible:
- `chance_of_playing_next_round` → monotonic non-decreasing in $P(B_F)$
- `min_last_1` → monotonic non-decreasing in $P(B_F)$
- `min_60_streak` → monotonic non-decreasing in $P(B_F)$

Why monotonic constraints matter: gradient-boosted models otherwise learn non-monotonic dips in small data regimes (e.g., a player with 80% chance of playing predicted lower than one with 70%, due to feature interactions). Constraints are cheap insurance.

**Hyperparameters**: small grid sweep only — `num_leaves ∈ {15, 31, 63}`, `learning_rate ∈ {0.05, 0.1}`, `min_data_in_leaf ∈ {20, 50}`. Pick by walk-forward CV log loss. Defer Bayesian opt; not where the gains are.

---

## 4. Training procedure

**Walk-forward by season**, no leakage by construction:

```
Fold 1: train [19/20, 20/21]                     →  predict 21/22
Fold 2: train [19/20, 20/21, 21/22]              →  predict 22/23
Fold 3: train [19/20, 20/21, 21/22, 22/23]       →  predict 23/24
Fold 4: train [19/20, 20/21, 21/22, 22/23, 23/24]→  predict 24/25
```

Within the predicted season, predictions are made at each GW's deadline using only data with `kickoff_utc < deadline`. PIT-enforced.

**Held-out final evaluation**: 2025/26 (current season; report metrics on completed GWs only, never used for training).

---

## 5. Calibration

LightGBM multiclass output is approximately calibrated under multinomial loss but not guaranteed. Per Phase 0 §7.2 acceptance gate: **ECE < 0.05 per subgroup** before allowing Phase 3 to proceed.

Procedure:
1. Train model on fold-train data
2. Reliability diagram per class on fold-test data, decile-binned
3. If ECE ≥ 0.05 in any subgroup: fit isotonic regression on a stratified calibration slice (held out from training)
4. Per-subgroup breakdowns: position, price tier, season half, manager tenure-bucket (when available)

---

## 6. Validation

- **Primary metrics**: multi-class log loss, mean Brier score, ECE per class.
- **Baselines**:
  - `rolling_3_mean`: predicted bucket = empirical distribution over last 3 GWs of buckets for that player. Sanity-check baseline; the model must beat it materially.
  - `position_marginal`: predict the position-level marginal distribution. Beats it trivially; defines the floor.
  - `last_gw_outcome`: predict last-GW bucket as a one-hot. Baseline for "blindly extrapolate."
- **Ablations**: drop each feature group and measure log-loss delta. Specifically watch for:
  - Removing rolling-minutes block → expect large drop
  - Removing price/position → expect modest drop
  - Removing news-keyword block → small but should be positive
- **Per-manager breakdown**: for clubs where we have manager-tenure data (top 6 manually annotated), report metrics under each manager. Diagnoses whether the model is averaging across rotation regimes.
- **Cold-start subset**: report metrics restricted to GW1 of each season + promoted-team players + new signings. Headline metrics dilute these failure modes.

---

## 7. Schema additions

Minimal. Predictions stored as parquet files under `data/predictions/minutes/{run_ts}/season_{N}.parquet` keyed on `(player_id, fixture_id)` with columns `p_zero, p_short, p_full, model_version`. No DB table for now; promote to DB if/when downstream consumers need it.

A small new `dim_manager_tenure` table is needed only if/when manager rotation features come back online. **Not creating it in 2.1.**

---

## 8. PIT API additions

```python
def minutes_features(player_id: int, fixture_id: int, as_of: datetime) -> dict[str, Any]: ...
def predicted_minutes_dist(player_id: int, fixture_id: int) -> dict[str, float]: ...
```

`minutes_features` builds the feature row from PIT data. `predicted_minutes_dist` reads from the parquet. Both leak-tested.

---

## 9. Code structure

```
src/fpl_bot/
├── features/
│   └── minutes.py                # PIT-routed feature builders
├── models/
│   └── minutes.py                # train_minutes_model, predict_minutes
├── eval/
│   └── minutes_eval.py           # walk-forward CV, calibration, ablation
└── cli/
    └── (existing main.py extended with `train minutes` and `predict minutes` subcommands)
tests/
├── leakage/
│   └── test_minutes_features.py  # static + synthetic-future-row variants for the feature path
└── unit/
    └── test_minutes_features.py  # feature correctness on small fixtures
```

`features/minutes.py` MUST import only from `fpl_bot.db.pit` — the static-import leakage gate enforces this.

---

## 10. Acceptance gate before Phase 2.2

- [ ] Beats `rolling_3_mean` baseline on log loss by ≥ 5% (relative), full corpus
- [ ] ECE < 0.05 per subgroup (position, price tier, season half)
- [ ] Calibration plots committed under `docs/design/phase2_1_results/`
- [ ] Ablation table committed
- [ ] All leakage tests still green; new feature-path leakage tests added

If any gate fails, iterate before moving to goals/assists models.

---

## 11. Open questions for review

1. **3-class vs 4-class buckets.** Sticking with {0, 1-59, 60+} per Phase 0; a fourth class for {90} (full-90 vs subbed-off) would help BPS calibration. Defer to Phase 2.5 unless you want it now.
2. **Manager rotation deferral.** Confirmed by user direction this turn. The minutes model proceeds without it; designing the EvaluationContext to accept it later as an additive feature.
3. **Storage of predictions.** Parquet under `data/predictions/`, not DB. OK?
4. **Hyperparameter scope.** Small grid sweep. Defer Bayesian/Optuna. OK?

---

**End of Phase 2.1 design. Awaiting review.**
