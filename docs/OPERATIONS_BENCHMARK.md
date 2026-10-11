# Corrected development benchmark — 11 October 2026

Both folds were regenerated from chronological training, then evaluated for all
38 gameweeks. Configuration: horizon 6, rho 0, alpha 0.8, beta 0, 200 joint
simulations with seed 42, no chips, and the live model's default captain
objective. Historical seasons use their preceding seasons for training.

| Test season | Train seasons | Model points | Buy-and-hold points | Difference | Joint xPts MAE | Naive MAE | Haul Brier |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2024/25 | 2019/20–2023/24 | 2,508 | 2,104 | +404 | 1.0400 | 1.0690 | 0.06191 |
| 2025/26 | 2019/20–2024/25 | 2,331 | 1,994 | +337 | 0.9494 | 0.9687 | 0.05501 |

Both runs passed feasibility, budget, transfer and chip accounting. Each passed
the existing predictive and captain gates and all ten calibration deciles.
Captain-selection points were 680 versus 516 for 2024/25, and 492 versus 358 for
2025/26; this is a per-week predictor diagnostic, not the rolling squad score.

These are development benchmarks, **not a verified leakage-free estimate of
live performance**. Closing historical odds can postdate FPL deadlines; later
horizon feature rows can include subsequently observed results; player position
and club metadata can reflect later information. The seasons were examined in
earlier development. The harness now keeps the provenance gate failed instead
of assuming accounting checks establish data timing. Time-limited solver
incumbents can also vary with machine load.

Corrections included FPL's floor-rounded half-profit selling rule, captain
objective alignment with live defaults, and finite shared fallback probabilities
where market data covers only part of the fixture set. The latter fixes a NaN
baseline metric; simulator missing-market defaults already existed.

Reproduce from a populated database:

```bash
uv run python scripts/operations_benchmark.py --season 24
uv run python scripts/operations_benchmark.py --season 25
```

Detailed metrics, every weekly squad, bank, hits, captain, solver status and
accounting failures are saved under
`data/audits/2026-10-11/operations/season_N/`. Raw artifacts stay outside git.
JSON output rejects non-finite metrics.

The prospective bot starts from team 718959's current GW7 state, rather than
inventing earlier decisions. It retains its own holdings, purchase prices, bank,
free transfers and chips, freezes timestamped decisions before deadlines, and
scores finalized actuals with its saved bench and vice captain. Its comparison
has no completed coverage at initialization. This is the evidence needed to
establish whether the refined model actually beats the manager.

To improve the historical claim further, rebuild every horizon from an explicit
prediction-time feature cutoff, retain deadline-specific odds and club/position
snapshots, and reserve an untouched future evaluation period. The prospective
ledger supplies that future period without substituting development results for
it.
