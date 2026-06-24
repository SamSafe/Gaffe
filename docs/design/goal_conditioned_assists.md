# Goal-conditioned assist allocation

## Problem

The joint xPts simulator samples team goals first and allocates them to
players, but still samples every player's assists from an independent Poisson
distribution. That permits impossible states:

- assists in a match where the player's team scores zero goals;
- more team assists than team goals;
- a player assisting the same goal they score.

The existing independent sampler remains available as a shadow mode so the
candidate can be compared with identical odds, BPS rules, folds, iterations,
and seeds.

## Candidate mechanism

For each simulated team:

1. Sample the team score from the market goal lambda.
2. Allocate goals to on-field players using the existing goal-rate and
   sampled-minutes weights.
3. Compute the lineup's conditional expected assists as
   `sum(lambda_assists_per_90 * minutes / 90)`.
4. Set the probability that each goal is assisted to the lineup expected
   assists divided by the market team-goal lambda, clipped to `[0, 1]`.
   This preserves the assist model's expected team total when it is feasible.
5. For each goal, sample assisted/unassisted. Allocate an assisted goal among
   on-field players using assist-rate and minutes weights, excluding that
   goal's scorer.

The exclusion is per goal, so a player may assist a teammate in the same match
in which they also score. Zero-minute players have zero weight.

## Point-in-time behavior

The probability uses only predictions fitted on the training fold, sampled
minutes, and the pre-match market goal lambda. It does not estimate a rate
from the held-out season. Historical assisted-goal ratios are used only as a
diagnostic, not as candidate input.

## Frozen baselines

Both modes use official BPS rules, committed legacy-consensus bookmaker odds,
500 Monte Carlo iterations, and seed 42.

| Test season | xPts MAE | P(xPts >= 6) Brier | MILP points | Validity |
|---|---:|---:|---:|---|
| 2024/25 | 1.04197 | 0.061999 | 2431 | pass |
| 2025/26 | 0.97334 | 0.056471 | 1554 | pass |

## Acceptance gates

- Actual sampler tests prove, on every generated event set:
  `sum(team assists) <= team goals`, no assists at zero goals, no assist by a
  goal's scorer, and no event for a zero-minute player.
- The independent shadow mode reproduces the pre-change simulator path.
- Same-seed 2024/25 and 2025/26 xPts MAE and haul Brier improve or remain
  within noise (maximum tolerated regression: `0.005` MAE and `0.0005`
  Brier). At least one held-out season must improve for a performance claim.
- Both downstream backtests pass every validity gate and neither has an
  unexplained regression beyond the documented roughly 30-point solver noise.
- Focused, full, database, and leakage tests pass.

Promotion is evidence-driven. If the structural candidate misses the metric
gates, production stays on independent sampling and the conditioned mode
remains available for live-season shadow comparison.

## Results and decision

| Test season | Mode | xPts MAE | P(xPts >= 6) Brier | MILP points | Template | Gates |
|---|---|---:|---:|---:|---:|---|
| 2024/25 | independent | 1.04197 | 0.061999 | 2431-2442 | 2243 | pass |
| 2024/25 | conditioned | 1.04039 | 0.061884 | 2388 | 1953 | pass |
| 2025/26 | independent | 0.97334 | 0.056471 | 1554 | 1530 | pass |
| 2025/26 | conditioned | 0.96980 | 0.056134 | 1587 | 1541 | pass |

The candidate improves MAE and Brier in both held-out seasons and adds 33
downstream points in 2025/26. However, its 2024/25 backtest is 43-54 points
below two independent-sampler runs. The candidate result repeated exactly in
the sequential variance audit, while the independent result moved by 11
points. This exceeds the roughly 30-point solver-noise allowance.

Decision: do not promote. `independent` remains the production default;
`goal_conditioned` is retained as an explicit shadow mode because it enforces
the correct event invariants and improves the more statistically stable
player-fixture metrics. Revisit with live 2026/27 observations or a broader
multi-fold optimizer comparison.
