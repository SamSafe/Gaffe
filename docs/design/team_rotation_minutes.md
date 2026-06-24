# Point-in-time team rotation features

## Scope

Historical manager identity and announced starting lineups are not available
consistently in the current corpus. The backtestable approximation is a team
rotation propensity derived from prior player minutes. A player with at least
60 minutes is treated as part of that fixture's core lineup.

For each `(season_id, team_id, fixture_id)` in kickoff order:

1. Build the set of players who recorded at least 60 minutes.
2. Measure realized churn versus the previous fixture as the fraction of the
   previous core absent from the current core.
3. Before appending the current realized churn, expose the mean of the prior
   three and five churn observations as model features.

The current lineup and minutes label never enter the current row's features.
Team history resets at each season boundary so reused FPL team identifiers and
promoted/relegated clubs cannot contaminate one another.

## Comparison path

`baseline` ignores the new columns and reproduces the current minutes model.
`rotation` trains on the two new features. Both modes use the same rows,
LightGBM seed/thread settings, odds, simulator, and downstream optimizer.

The upcoming-fixture path computes the latest team rates after all completed
fixtures and joins them by the player's current team. It does not reuse the
feature row from a player's last match, which would be one observation stale.

## Frozen baselines

| Test season | Minutes log loss | ECE class 0/1/2 | Joint xPts MAE | Haul Brier |
|---|---:|---|---:|---:|
| 2024/25 | 0.5058 | 0.016 / 0.015 / 0.012 | 1.04197 | 0.061999 |
| 2025/26 | 0.4647 | 0.018 / 0.013 / 0.017 | 0.97334 | 0.056471 |

The joint-xPts baselines use production independent assist sampling, official
BPS rules, 500 Monte Carlo iterations, and seed 42.

## Acceptance gates

- Unit tests cover churn arithmetic, season reset, insufficient-history nulls,
  and latest-rate calculation for an upcoming fixture.
- Leakage tests prove adding future fixtures does not change earlier rotation
  features.
- Minutes log loss improves in at least one held-out season and does not
  regress by more than 0.002 in the other; no ECE class regresses by more than
  0.01.
- Joint-xPts MAE and haul Brier improve or remain within 0.005 and 0.0005,
  respectively, in both 2024/25 and 2025/26.
- Downstream season backtests pass all validity gates and do not materially
  regress against the deterministic baseline.
- Full unit, integration, database, and leakage suites pass.

If the feature misses these gates, `baseline` remains the production default
and the result is documented rather than tuned on the held-out seasons.

## Interim results

| Test season | Mode | Minutes log loss | Joint xPts MAE | P(xPts >= 6) Brier | Joint gates |
|---|---|---:|---:|---:|---|
| 2024/25 | baseline | 0.5058 | 1.04197 | 0.061999 | pass |
| 2024/25 | rotation | 0.5048 | 1.04326 | 0.061954 | pass |
| 2025/26 | baseline | 0.4647 | 0.97334 | 0.056471 | pass |
| 2025/26 | rotation | 0.4639 | 0.97035 | 0.056132 | pass |

The feature improves minutes log loss and haul Brier in both seasons. Joint
MAE improves by 0.0030 in 2025/26 and regresses by 0.0013 in 2024/25, within
the frozen tolerance. Calibration and all joint model gates pass.

The full regression run passes 245 tests, including database and leakage
coverage, and lint passes. Downstream MILP results remain required before a
promotion decision or commit; the first solve attempt was blocked by the
execution approval quota before the process started.
