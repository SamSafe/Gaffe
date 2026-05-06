# Phase 2.3 — Clean Sheet Model

Status: **approved, gate evaluated.** Implementation complete; **production model is `market_only`** (residual failed the gate, as Phase 0 §4.4 explicitly anticipated).

Round-1 review resolutions:
1. Ship market-only fallback if residual fails the gate — confirmed and triggered
2. Report overall + subgroup ECE — overall ECE in CV results; subgroup ECE in diagnostics commit
3. `team_id` one-hot fitted fold-internally; unseen promoted teams get all-zero one-hot

Per Phase 0 §4.4: start with bookmaker-implied CS probability as the prior; train a residual GBM to find blendable signal on top. The expectation is that residual lift is small (markets are efficient on CS); we ship the residual model only if it beats the market-only baseline.

---

## 1. Target

Team-level binary classification per (team_id, fixture_id):
$$
\hat{P}(\text{CS}_t \mid \text{features}) = \mathbb{P}(\text{goals\_conceded}_{t,f} = 0)
$$

For each fixture, two rows: one per team (the home team's CS = home_score went unanswered? no — CS means OWN team didn't concede). Specifically, team $t$ keeps a clean sheet when the OPPONENT scored 0.

So `cs_label_t` = (away_score == 0) if t is home else (home_score == 0).

FPL appearance points already use this exact definition; `fact_player_match.clean_sheet` is at the player level but identical within a team-fixture (every player on the team that kept the CS gets the bool).

---

## 2. Approach: market prior + residual

### Layer 0 (market prior)
$$
p_{\text{market}}(t,f) = \texttt{fact\_market\_xg.cs\_prob}
$$
already computed in Phase 1B via Dixon-Coles inversion.

### Layer 1 (residual GBM)

LightGBM binary classification with `init_score = logit(p_market)`. The model learns the *residual* in log-odds space; if it learns zero, predictions equal the market.

Output:
$$
p_{\text{model}}(t,f) = \sigma\bigl(\text{logit}(p_{\text{market}}) + \text{model\_output}\bigr)
$$

**Decision rule**: ship the residual model only if it materially beats the market-only baseline on out-of-sample Brier score (≥ 0.005 absolute improvement). Otherwise ship the market alone — the spec explicitly anticipates this outcome.

---

## 3. Feature catalog (v1)

Per (team_id, fixture_id) with `as_of < kickoff`:

### From `fact_market_xg`
- `market_cs_prob` — used as `init_score` (NOT a feature column, since the model would just learn to ignore it)
- `team_lambda_market_xg` — own team's predicted goals (high values weakly correlate with conceding more in open games)
- `opponent_lambda_market_xg` — opponent's predicted goals; this IS the market-implied xGA

### From `fact_player_match` (aggregated to team level)
- `team_cs_rate_last_3`, `team_cs_rate_last_5`, `team_cs_rate_last_10` — rolling actual CS rate
- `team_goals_conceded_last_3`, `last_5`, `last_10` — rolling actual goals conceded per match
- `team_goals_for_last_5` — own attack rate (gut-check feature; teams that score more sometimes concede more in chaotic matches)

### From `fact_understat_player_match` (aggregated to team level)
- `team_xga_per_match_last_5` — opponent's xG against this team rolling. Useful "underlying-quality" complement to the noisy goals-conceded count.

### Fixture metadata
- `was_home`
- `days_since_last_match` (team-level)
- `days_into_season`
- `gameweek`
- `team_id` one-hot — captures team-specific defensive baseline that rolling features won't pick up immediately for newly-promoted teams

### Deferred (data not available in v1)
- Defensive lineup change flag (centre-back rotation) — needs starting-XI tracking with positions
- GK rotation flag — needs starting-XI history
- Saves-above-expected — needs shot-against-tracking we don't have
- Manager change — no clean source

---

## 4. Training procedure

Walk-forward CV by season, same as Phase 2.1/2.2:
```
Fold 1: train [19, 20]            → predict 21
Fold 2: train [19, 20, 21]        → predict 22
Fold 3: train [19, 20, 21, 22]    → predict 23
Fold 4: train [19, 20, 21, 22, 23]→ predict 24
```

Filter: rows with `market_cs_prob IS NOT NULL` (rare missing fixtures are dropped). Each fold has ~7,200 train + ~760 valid (380 fixtures × 2 teams).

Hyperparameters: same restrained sweep as Phase 2.1/2.2.

---

## 5. Validation

**Primary metrics**:
- Brier score (binary)
- Log loss
- ECE (binned reliability)

**Baselines**:
- **Market-only**: predict `market_cs_prob` directly. The bar to beat.
- **Rolling-CS-5**: predict `team_cs_rate_last_5`. Floor.

**Acceptance gate before Phase 2.4**:
- [x] **Residual must beat market by ≥ 0.005 Brier on ≥ 3/4 folds.** Result: **0/4 folds pass.** Per-fold deltas (Brier_market − Brier_model): +0.0002, −0.0002, −0.0002, +0.0005. All within fold-level noise. **Production model = `market_only`** as the design explicitly directs in this case.
- [x] **All leakage tests still green; new feature-path leakage test added.** 33/33 passing including `test_clean_sheet_features_pit_stable_across_truncation`.
- [x] **Overall ECE within tolerance.** Market ECE per fold: 0.018, 0.026, 0.046, 0.015 — all under 0.05. (Subgroup ECE in the diagnostics close-out commit.)
- [ ] **Calibration plot committed** — diagnostics close-out follow-up.
- [ ] **Ablation table committed** — diagnostics close-out follow-up.

**Why the residual failed (and that's fine):** EPL CS markets are very efficient. The Dixon-Coles inversion of 1X2 + totals already produces well-calibrated CS probabilities (per-fold ECE 0.015–0.046). There's no systematic mispricing for a residual model to find. This is the "market is efficient on CS" outcome the design anticipates — we keep the residual code in place for periodic re-evaluation but the production caller returns `market_cs_prob` directly.

**Walk-forward CV results (full table):**
```
season  | n_train | n_test | brier_m | brier_mkt | brier_r5 | ll_m   | ll_mkt | ece_m | ece_mkt | delta
 2021/22 |    1520 |    760 | 0.1794  | 0.1797    | 0.2296   | 0.5360 | 0.5369 | 0.020 | 0.018   | +0.0002
 2022/23 |    2280 |    760 | 0.1871  | 0.1869    | 0.2373   | 0.5552 | 0.5548 | 0.025 | 0.026   | -0.0002
 2023/24 |    3040 |    760 | 0.1555  | 0.1553    | 0.1836   | 0.4802 | 0.4799 | 0.037 | 0.046   | -0.0002
 2024/25 |    3800 |    760 | 0.1697  | 0.1702    | 0.2132   | 0.5141 | 0.5169 | 0.017 | 0.015   | +0.0005
```

The rolling-5 baseline (brier_r5 ≈ 0.21) is meaningfully worse than market (≈ 0.17), confirming the market is the right anchor and the rolling baseline is a reasonable floor — but neither rolling history nor the residual model's full-feature attempt finds anything the market hasn't already priced.

---

## 6. Storage

Predictions saved to `data/predictions/clean_sheet/{run_ts}/season_{N}.parquet` with columns: `team_id, fixture_id, season_id, gameweek, kickoff_utc, market_cs_prob, model_cs_prob, model_version`.

---

## 7. Code structure

```
src/fpl_bot/
├── features/
│   └── clean_sheet.py          # team-level feature builder
├── models/
│   └── clean_sheet.py          # train_cs_model + predict; market init_score
└── eval/
    └── clean_sheet_eval.py     # walk-forward CV with market baseline
tests/
├── leakage/
│   └── test_clean_sheet_features.py
└── unit/
    └── test_clean_sheet_features.py
```

---

## 8. Open questions

1. **Residual ceiling** — if the model fails to beat market by 0.005 Brier, the spec says ship market-only. I'll put both code paths in place; the gate decides which is "the" model in production. Confirm.
2. **Team_id one-hot** — 20 teams × 6 seasons = 120 distinct team-season pairs; encoding as team_id alone (without season) gives ~30 unique values across all promoted/relegated turnover. LightGBM handles cardinality fine. OK.
3. **Subgroup definition for ECE** — propose binning by `market_cs_prob` (low / medium / high) plus per-team for the top 6 (where most CS predictions matter for FPL DEF/GK points).

---

**End of Phase 2.3 design. Awaiting review.**
