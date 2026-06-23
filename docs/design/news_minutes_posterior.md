# News-adjusted minutes posterior

## Hypothesis

Availability news should modify the minutes distribution before joint-xPts
simulation. Applying the same signal to final xPts loses clean-sheet threshold,
appearance, bonus, captaincy-tail, and autosub effects.

For an availability probability `a` and normalized base minutes probabilities
`(p0, p_short, p_full)`:

```text
p0'      = (1 - a) + a * p0
p_short' = a * p_short
p_full'  = a * p_full
```

This first version uses only explicit FPL-news availability masks already
extracted by `live/news_extract.py`. It does not infer probabilities from free
text and does not claim historical performance lift.

## Scope

1. Add a pure, reusable probability update in the minutes-model layer.
2. Apply explicit news availability to upcoming-fixture minutes predictions
   before the BPS/xPts simulator.
3. Preserve the existing final-xPts attenuation as a compatibility fallback
   for cached predictions that cannot be re-simulated.
4. Record prospective live comparisons before learning phrase/source weights.

## Acceptance gates

- Probabilities remain finite, within `[0, 1]`, and sum to one.
- `a=0` produces `(1, 0, 0)`; `a=1` preserves the normalized base model.
- Missing news signals are bit-for-bit neutral.
- Invalid availability values fail explicitly rather than being silently
  clipped.
- Existing unit and leakage tests do not regress.
- Prior-season predictions/backtests are unchanged when no historical news
  signal is supplied.

Historical news snapshots do not exist, so a backtest improvement is not an
acceptance gate for this change. The honest performance test is prospective:
persist base and news-adjusted minutes distributions at each deadline, then
score both against realized minutes using multiclass log loss, Brier score,
calibration, and downstream FPL points.

## Baseline before implementation

Run on 2026-06-23:

- 172 tests passed.
- 10 tests skipped because PostgreSQL was unavailable.
- 9 failures and 13 errors were database-dependent baseline failures caused by
  PostgreSQL being unreachable; no source changes were present.

## Validation result

Completed on 2026-06-23:

- Full suite: **217 passed**.
- Changed-file lint and Python compilation: passed.
- No-signal neutrality: exact identity across 14 cached prediction artifacts
  covering 332,712 player-fixture rows.
- Historical backtest code does not receive an availability mapping and is
  therefore unchanged. No performance lift is claimed.

Frozen cached component metrics for future comparisons:

| Test season | Rows | Mean predicted | Mean actual | MAE | Brier (≥6) |
|---|---:|---:|---:|---:|---:|
| 2021/22 | 25,447 | 1.2070 | 1.2345 | 1.0674 | 0.0646 |
| 2022/23 | 26,505 | 1.1860 | 1.1969 | 1.0084 | 0.0619 |
| 2023/24 | 29,725 | 1.0867 | 1.0520 | 0.9328 | 0.0523 |
| 2024/25 | 27,605 | 1.1627 | 1.2009 | 1.0425 | 0.0622 |
| 2025/26 | 22,343 | 1.1083 | 1.1784 | 0.9716 | 0.0565 |

These figures are reference baselines, not evidence that news improves the
model. News performance requires deadline snapshots and prospective outcomes.
