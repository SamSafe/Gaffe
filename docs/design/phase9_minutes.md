# Phase 9 — Expected minutes

**Status:** step A measured; **step B′ (crowd transfers as a news proxy)
shipped 2026-10-19** for next-gameweek forecasts. Step B (FPL status fields)
remains blocked for lack of history.

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

## Step B′ — this week's transfers as a news proxy (SHIPPED, 2026-10-19)

Step B is blocked because status history does not exist. Managers' reactions
to the same news do: `fact_player_match` carries each gameweek's
`transfers_in/out/balance`, `selected` and price back to 2019/20. FPL
attributes a gameweek's transfers to the window **before** its deadline:
`selected[N] - selected[N-1]` matches `transfers_balance[N]` to a median of
~5k against ~60k typical balances (2023/24-2025/26), while the post-deadline
alignment misses by ~57k. So they are known before kickoff.

Features (`MARKET_FEATURE_COLUMNS`): log owners, out/in/net transfers as a
share of owners going into the window, and price.

Minutes log loss (train 19..N-1):

| Slice | 2024/25 | 2025/26 |
|---|---:|---:|
| All rows | 0.507 → 0.466 | 0.461 → 0.418 |
| Regular starters (start60_rate_5 ≥ 0.8) | 0.565 → 0.518 | 0.556 → 0.512 |
| Regulars after a DNP | 0.858 → 0.712 | 0.863 → 0.724 |
| Season openers | 0.854 → 0.743 | 0.793 → 0.655 |

Expected-minutes error for regulars after a DNP fell by ~7 minutes. Longer
history features (20-game rates, DNP streaks) added little and inconsistently.

Joint xPts, paired simulations (same seeds), each fixture scored one week
ahead; squared error with 90% week-block interval of the change:

| Season | Baseline | Market | Change CI | Spearman |
|---|---:|---:|---|---|
| 2024/25 | 4.162 | 4.085 | −0.103 … −0.053 | 0.689 → 0.720 |
| 2025/26 | 3.600 | 3.503 | −0.123 … −0.074 | 0.733 → 0.755 |
| 2026/27 GW1-6 | 4.619 | 4.497 | −0.220 … −0.047 | 0.697 → 0.723 |

Every position improved in 24/25 and 25/26; the top-20-per-week slice
improved in all three. Scripts and logs: `data/audits/2026-10-19/
minutes_features.py`, `market_xpts.py`.

**Live use and limits.**
- Only the coming gameweek has transfers. The live trainer fits a second
  minutes model on these features for those rows; later weeks keep the
  baseline. Multi-week MILP backtests must never use "market" mode (it would
  read future weeks' transfers), so backtest points cannot show this gain.
- Live counts come from bootstrap-static's `transfers_in_event/out_event`,
  stored on each status snapshot (migration 0011) and used only when the
  snapshot falls after the previous gameweek's first kickoff and before this
  gameweek's deadline (90 minutes before first kickoff, when FPL resets the
  counters). Owners going in are the exact previous-gameweek count.
- Counts arrive over the week. Against element-summary totals, 2026/27
  snapshots held 14% (GW4, 151 hours out), 10% (GW7, 146h), 60% (GW2, 66h)
  and 82% (GW7, 6h) of the final transfers, in and out alike. Snapshots more
  than 72 hours out are not used; inside that window counts are scaled up by
  the interpolated completion (`pit.transfer_completion`, at most 1.7x).
- The live event feed omits these fields; `ingest/fpl_market` refills them
  from each player's element-summary once per finished gameweek.
- The doubtful-flag multiplier (xPts × chance_of_playing/100) is skipped for
  the market-priced week when the snapshot is within 12 hours of the
  deadline. Managers react to the same flag, so it double
  counted: on the 37 doubtful player-weeks of 2026/27 GW2/GW4 (the weeks with
  a snapshot inside their own window), actual P(plays) was 0.405; market
  minutes alone predicted 0.378 (Brier 0.187), market × chance 0.270 (Brier
  0.202). The multiplier did help the baseline model (Brier 0.244 → 0.228),
  so later weeks keep it. Small sample; direction and mechanism agree.
  Checked live-faithfully for the week after a flag (pre-flag features scored
  against the next week's minutes, 37 rows), the multiplier still edged Brier
  (0.278 → 0.261) while over-shrinking the mean (0.296 vs actual 0.432):
  inconclusive, so the horizon rule is unchanged
  (`data/audits/2026-10-19/doubtful_{check,horizon}.py`).

## Remaining candidates, in value order

**C. Manual team-news override — SHIPPED 2026-08-13.**
`configs/expected_minutes_overrides.yaml` takes per-player expected minutes per
gameweek, keyed by `web_name` or numeric `player_id`. This is precisely what
FFH's analysts supply, and it is the one channel that beats any model — a human
watches the press conference and the bot does not.

Design points worth keeping:
- Values are **expected minutes, not probabilities**, converted to the three
  model buckets preserving the expectation exactly (`minutes_to_bucket_probs`).
  Above 75 it saturates, since the 60+ bucket is then already certain.
- It sets the **whole distribution**, so it expresses rotation and cameos
  ("expect 30 minutes"), which the pre-existing news attenuator cannot — that
  one only shifts mass toward "did not play".
- Applied **last**, beating both the model and the news attenuator.
- An unresolvable name **raises**. A silently-dropped override would leave the
  operator believing the bot had been told something it had not, which is worse
  than having no override channel at all.
- The committed file is empty and a test asserts it stays that way, so nobody
  inherits a previous gameweek's overrides.

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
