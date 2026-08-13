# Phase 9 — Expected minutes

**Status:** 📏 measured, not yet built. Step A (baseline) is complete and is
recorded below; it reshaped the plan substantially, killing the step everyone
would reach for first.

## Motivation

Every failure of the 2026/27 season opener traced back to expected minutes, not
to goals or clean sheets. The live path predicted **4.9 expected minutes for
Haaland**, which collapsed all attacking output and left the optimizer ranking
forwards at 0.28 predicted points against defenders' 1.32. Minutes are the
multiplier on every attacking term, so an error there dominates everything else.

Products like Fantasy Football Hub invest heavily here, with analysts adjusting
expected minutes by hand. This phase asks what of that is worth copying, and
what our market-based approach already covers.

## Does the odds signal already carry team news?

Partly — and the part it misses is the dangerous one.

- **Team level: yes.** If a striker is ruled out, the 1X2 and O/U prices move,
  the Dixon-Coles inversion lowers that team's λ, and every player's ceiling
  falls with it. This is handled.
- **Player level: no, and it is worse than not knowing.** We take the correctly
  reduced team λ and split it using xG shares that still weight the absent
  player fully. He receives a large slice of a slightly smaller pie. The
  team-level correction actively misdirects us about who benefits.

The market's player-level news channel is the **anytime-goalscorer prop**, which
is unconditional: it prices `P(plays) × P(scores | plays)` jointly, so a rested
player's price lengthens sharply. Phase 8 already ingests these (217 rows for
GW1 2026/27). Two catches: the prop conflates minutes with scoring rate, and our
minutes de-scaling clamps most of it away when expected minutes are wrong.

## Step A — baseline (COMPLETE, 2026-08-13)

Using the existing `eval/minutes_eval.run_walk_forward_cv` harness.

**The model is strong in aggregate.** Multiclass log loss against two baselines:

| test season | model | rolling-3 baseline | position-marginal |
|---|---|---|---|
| 2021/22 | 0.529 | 1.510 | 1.040 |
| 2022/23 | 0.516 | 1.403 | 1.049 |
| 2023/24 | 0.473 | 1.304 | 1.037 |
| 2024/25 | 0.505 | 1.381 | 1.016 |

ECE per class ≤ 0.021 — well calibrated. This is not a weak model.

**But the season opener is materially harder**, measured by splitting the test
fold by gameweek:

| fold | GW1 | GW2-3 | GW4+ |
|---|---|---|---|
| season 23 | **0.780** (n=545) | 0.520 | 0.465 |
| season 24 | **0.846** (n=543) | 0.532 | 0.496 |

GW1 log loss is ~70% worse than mid-season, converging by GW2-3. Real, bounded,
and still comfortably better than the position-marginal baseline (~1.02) — a
degradation, not a collapse. The catastrophic live behaviour was the
feature-plumbing bug fixed in `af141e4`, not the model.

Supporting facts:
- GW1 rows **are** trainable: 1,013 exist across seasons 22-24, because a
  player's opener has last season's finale as its previous match — provided the
  feature table spans seasons.
- Summer-length gaps are rare in training: `days_since_last_match` in 60-100
  days is **1.19%** of rows (median gap is 7 days). The model has seen the
  regime, but thinly.

## Step B — availability as model features: BLOCKED

The obvious idea is to feed `status_code` and `chance_of_playing_next_round`
into the minutes model, which currently cannot see them at all. **This is not
possible with data we have free access to**, and building it would recreate the
exact train/serve skew fixed in `f97f905`:

- `fact_player_status` is a *current-snapshot* table, not history. Season 25
  holds snapshots only from 2 May - 17 May 2026; season 26 only from the 26/27
  preseason. No training season has any coverage.
- vaastav's per-gameweek files carry `minutes` and `starts` but no availability
  fields.
- vaastav's `players_raw.csv` does carry `chance_of_playing_*` and `status`, but
  as a single end-of-season snapshot per player. Using it would leak May's
  injury status into a GW1 prediction.

This vindicates the existing architecture: applying news and status as a
**live-only post-hoc attenuator** to the minutes distribution
(`apply_availability_to_minutes_predictions`) is not a shortcut, it is the only
correct option given the data. Do not "fix" it by training on these fields.

Worth noting: the weekly `ingest fpl` writes `fact_player_status` rows, so
availability history is accumulating now. By 2027/28 there should be a full
season of it, at which point step B becomes genuinely testable.

## Remaining candidates, in value order

**C. Manual team-news override (cheap, no validation needed).** Extend
`configs/live_state_overrides.yaml` with per-player expected minutes. This is
precisely what FFH's analysts supply, it is the one channel that beats any
model — a human watches the press conference and the bot does not — and it is
opt-in, so it carries no downside risk.

**D. Market-implied availability from player props (the novel one).** Divide the
market's unconditional `P(scores)` by our model's conditional
`P(scores | plays)` to recover an implied availability multiplier, then test
whether it predicts realised minutes better than the model alone. This is the
only route to a *validated* availability signal, since the market prices it and
outcomes accrue weekly. Requires actuals, so it starts after GW1 closes.

**E. Season-opener treatment (cheap, directly testable against step A).** The
1,013 GW1 rows and the 1.19% long-gap slice are thin. Candidates: an explicit
`is_season_opener` feature, upweighting long-gap rows, or a dedicated opener
model. The step-A table above is the benchmark any such change must beat.

## Validation gates

In order, per the project's hard-won rule that accuracy — not points — decides:

1. Minutes accuracy: log loss and ECE, reported **split by GW1 / GW2-3 / GW4+**,
   against the step-A table.
2. Downstream xPts accuracy: per-position bias and MAE.
3. Points, last and never alone. The FDR and FWD-calibration reversals both
   scored well on points with zero accuracy gain.
