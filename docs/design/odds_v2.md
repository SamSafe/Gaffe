# Odds v2 — point-in-time snapshots and richer market fitting

## Hypothesis

Bookmaker odds are the prediction stack's strongest measured signal, but the
current implementation loses information in two places:

1. live quotes use fixture commencement as the primary-key timestamp, so a
   later fetch overwrites the earlier quote;
2. derivation averages raw implied probabilities across bookmakers and uses
   only 1X2 plus the 2.5 total, despite ingesting Asian handicaps and other
   totals lines.

The expected gain is better-calibrated team goal distributions and honest
pre-deadline replay. Extra model capacity is not part of this phase.

## Observed baseline

The 2026-05-18 raw payload contains bookmaker and market `last_update`
timestamps. Across 13 events it includes:

- 33 head-to-head markets;
- 22 totals markets at 2.5, 2.75, 3.0, 3.25, and 3.5;
- 11 Asian-handicap markets;
- exchange lay prices that are currently ignored.

The legacy fitter is independent Poisson, not Dixon–Coles-corrected. It forms
the consensus by averaging `1 / decimal_odds` before removing overround, then
fits home win, draw, and over-2.5 probabilities.

## Work plan

1. Add an explicit `quote_time` to `fact_odds` and make it part of the primary
   key. Keep `event_time` as fixture commencement and `recorded_at` as local
   ingest time.
2. Populate live `quote_time` from market/bookmaker `last_update`; use a
   documented closing-time approximation for historical football-data rows.
3. Select the latest complete market per bookmaker at or before an optional
   cutoff. Never combine selections from different snapshots.
4. Remove overround within each bookmaker market before building a consensus.
5. Extend the candidate fitter to score every supported totals and handicap
   line, while retaining a safe legacy fallback when only 1X2/2.5 exists.
6. Keep the candidate behind an explicit derivation mode until validation
   passes.

## Acceptance gates

- Repeated live fetches create distinct quote snapshots.
- All derived rows are reproducible at a supplied `as_of` cutoff.
- Incomplete bookmaker markets are excluded, not partially averaged.
- Every de-vigged market sums to one within numerical tolerance.
- Candidate market-reconstruction loss beats or matches legacy on the stored
  live payload.
- Historical score/CS log loss and Brier do not regress materially across
  prior seasons.
- Same-seed downstream xPts/MILP comparisons are reported for every available
  prior-season fold before changing the default.
- Full unit, integration, migration, and leakage suites pass.

Historical football-data prices are closing odds. They remain unsuitable for
claiming a fully point-in-time backtest; this phase makes live snapshots honest
and labels the historical approximation rather than hiding it.

## Validation results

### Snapshot/data integrity

- Migration `0008` applied to 32,221 existing odds rows; zero null quote times.
- Re-parsing the stored 2026-05-18 live payload created true pre-kickoff quote
  snapshots distinct from fixture commencement.
- Re-parsing the identical payload again left the row count unchanged (240
  Odds-API rows before and after).
- Handicap outcomes now share one home-line market key, so both selections are
  selected atomically.

### Market reconstruction

On the 11 mapped fixtures in the stored live payload, fitting all available
1X2, totals, and handicap markets reduced mean squared reconstruction error:

```text
legacy  0.00019413
all     0.00007719   (-60.2%)
```

### Historical component metrics

Closing-odds evaluation across seven seasons showed essentially neutral mixed
results. `all` minus production exact-score NLL ranged from -0.00146 to
+0.00215; clean-sheet Brier changes were similarly small and mixed.

| Season | Production NLL | All NLL | Production CS Brier | All CS Brier |
|---|---:|---:|---:|---:|
| 2019/20 | 2.84173 | 2.84148 | 0.18538 | 0.18522 |
| 2020/21 | 2.96362 | 2.96577 | 0.19752 | 0.19765 |
| 2021/22 | 2.88911 | 2.89003 | 0.17967 | 0.17972 |
| 2022/23 | 2.98569 | 2.98464 | 0.18689 | 0.18673 |
| 2023/24 | 3.04971 | 3.04825 | 0.15528 | 0.15514 |
| 2024/25 | 2.94371 | 2.94560 | 0.17023 | 0.17062 |
| 2025/26 | 2.86964 | 2.86936 | 0.18057 | 0.18057 |

### Same-seed downstream decision gate

| Fold | Production | All-market | Absolute delta | Validity |
|---|---:|---:|---:|---|
| 2024/25 | 2497 | 2423 | **-74** | pass/pass |
| 2025/26 | 1591 | 1649 | **+58** | pass/pass |

The richer fitter is not stable across folds, so it remains opt-in via
`--fit-mode all`. The accepted production changes are quote snapshot
preservation, point-in-time cutoffs, canonical market pairing, and per-book
de-vigging. Production retains the conservative 1X2 + totals-2.5 fitter.

Final validation: **228 tests passed** against PostgreSQL schema revision
`0008`; changed-file lint, compilation, and whitespace checks passed.
