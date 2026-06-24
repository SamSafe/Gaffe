# BPS rule correction and season versioning

## Trigger

Official-rule verification found that the simulator's mechanistic BPS layer
does not match FPL rules:

- positional goal values are reversed (`DEF=24`, `FWD=12` in code versus
  official `DEF=12`, `FWD=24`);
- short appearances omit 3 BPS;
- goalkeeper saves use `2 BPS per three saves` instead of `2 per save` for
  2024/25;
- goals conceded deduct 1 BPS instead of 4;
- tied bonus groups do not consume ranking places correctly;
- from 2025/26, penalty goals are 12 BPS regardless of position.

Primary sources:

- https://fantasy.premierleague.com/help/rules
- https://www.premierleague.com/en/news/4362127
- https://www.premierleague.com/en/news/4059044

This is a mechanism correction, not a calibration multiplier. It is a plausible
structural explanation for the persistent forward bonus underprediction.

## Implementation scope

1. Version directly modelled BPS rules by `season_id` where rules differ.
2. Correct appearance, non-penalty goal, save, and conceded-goal BPS.
3. Distinguish sampled penalty goals for 2025/26 BPS without changing FPL goal
   points or the team-goal invariant.
4. Allocate tied bonus using competition ranks: a two-player tie for first
   consumes ranks 1 and 2, so the next group receives third-place bonus.
5. Continue fitting the empirical residual after subtracting corrected known
   BPS; unavailable spatial save/tackle details stay in that residual.

## Acceptance gates

- Exhaustive official tie examples pass.
- Known-event examples match official values for 2024/25 and 2025/26.
- Team/player goal invariants remain unchanged.
- BPS Brier and per-position calibration improve or remain within noise across
  prior-season folds; forward calibration is the primary target.
- Joint-xPts MAE/Brier do not regress materially.
- Same-seed 2024/25 and 2025/26 downstream backtests pass all validity gates
  and do not show an unexplained material regression.
- Full unit, integration, and leakage suites pass.

## Controlled evaluation

The comparison path reproduces the pre-correction known-event table, bonus
ranking, continuous residual sampling, and zero-minute ranking behavior. Both
modes use the same committed bookmaker consensus, model inputs, 500 Monte
Carlo iterations, and seed 42.

| Test season | Mode | xPts MAE | P(xPts >= 6) Brier | MILP points | Template | Gates |
|---|---|---:|---:|---:|---:|---|
| 2024/25 | legacy | 1.04030 | 0.061990 | 2396 | 2335 | pass |
| 2024/25 | official | 1.04197 | 0.061999 | 2431 | 2243 | pass |
| 2025/26 | legacy | 0.96976 | 0.056345 | 1572 | 1541 | pass |
| 2025/26 | official | 0.97334 | 0.056471 | 1554 | 1530 | pass |

The official rules do not produce a component-metric lift: legacy is better
by 0.0017-0.0036 MAE and at most 0.00013 Brier. Those differences are below
0.4% and are not material. Downstream results are mixed (+35 in 2024/25 and
-18 in 2025/26), within the backtest's documented season-level solver noise,
and every validity gate passes.

Decision: use `official` in production because it matches the mechanism and
does not materially regress either held-out season. Retain `legacy` only as an
explicit comparison mode. Do not claim this phase as a performance
improvement; its value is removing a known structural error before future BPS
features are added.
