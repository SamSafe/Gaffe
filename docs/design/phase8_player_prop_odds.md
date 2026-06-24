# Phase 8 — Player-prop (anytime-goalscorer) odds

**Status:** ⚙️ core logic built + unit-tested (`derive/player_props.py`,
`tests/unit/test_player_props.py`); ingest + live wiring **deferred to the
26/27 season** (cannot be developed or validated off-season — see
Constraints). This is a **ceiling-raiser**, not an incremental tweak.

## Motivation

Every backtestable model feature this project has tried that *duplicates*
the team-odds signal (finishing skill, FDR, FWD isotonic calibration) was
measured to add no prediction accuracy — because the bot already extracts
its edge from the match market (Dixon-Coles inversion of 1X2 + O/U 2.5 →
team λ). The remaining headroom is in **information the pre-match
match-odds market doesn't expose at the player level**.

The anytime-goalscorer market is exactly that: a direct, market-priced
probability that *this player* scores in *this fixture*. It already
integrates expected minutes, rotation risk, role (penalties/set-pieces) and
form — the things the bot currently *approximates* by allocating team λ
across players via xG-shares. Using it is the bot's validated thesis ("the
odds are the edge") extended from team to player level, aimed at the one
persistent weakness the cross-fold diagnostic keeps showing: elite-FWD /
top-decile under-prediction.

## Architecture

```
event-odds endpoint (The Odds API, per-fixture)         [NEW ingest, deferred]
  → player_goal_scorer_anytime "yes" prices
  → resolve player name → player_id (fuzzy, reuse web_name map)
  → fact_player_odds(fixture_id, player_id, p_anytime, recorded_at)   [NEW table]
        │
        ▼
derive/player_props.py  (BUILT THIS SESSION, pure + tested)
  devig_anytime_prob         strip ~half the two-way margin
  anytime_prob_to_fixture_goals   P(≥1) = 1−e^(−μ)  ⇒  μ = −ln(1−P)
  market_implied_per90_rate  μ / E[minutes_fraction]   ← un-double-count minutes
  blend_goal_rate            convex blend with model rate (market_weight)
        │
        ▼
live predict-only path (run_predict_only / _train_goals_or_assists_predict_only)
  → override/blend lambda_goals_per_90 for players with a market prop   [NEW wiring, deferred]
  → existing BPSSimulator consumes it unchanged
```

It mirrors the existing team-λ flow (`derive/dixon_coles.py`): invert market
probabilities to a rate, feed the simulator. No simulator changes needed.

### The minutes double-counting pitfall (why the core is its own module)

The anytime prob `P` is **unconditional** — it already prices in the chance
the player is benched/subbed. But `BPSSimulator` samples minutes per
iteration and multiplies the per-90 rate by sampled minutes. Feeding the
implied *fixture* rate straight in would apply minutes twice. So
`market_implied_per90_rate` divides the implied fixture goals
`μ = −ln(1−P)` by the player's expected minutes fraction, giving a per-90
rate the simulator can re-scale without double-counting. This is the subtle
correctness property; it is unit-tested now
(`test_per90_rate_undoes_minutes_so_sim_rescaling_recovers_fixture_goals`).

## Constraints (the honest part)

- **Live-only, not backtestable.** There is no free historical
  anytime-scorer data (The Odds API historical endpoint is paid + 10×
  credits; football-data.co.uk has no player props). So this feature
  *cannot* be validated on the 2019–2025 folds — the same situation as the
  Tier-2 news / EO / price-predictor features, which shipped as live-only
  infrastructure by design.
- **Off-season blocks even ingest development.** The per-event odds endpoint
  has no EPL fixtures (let alone player props) until ~Aug 2026, so the
  ingest + name-resolution can't be exercised against real responses now.
  Building it blind would be error-prone; hence deferred.
- **US-book coverage.** Soccer player props on the free tier are currently
  US sportsbooks — less sharp/liquid than Pinnacle. The `blend_goal_rate`
  weight (start conservative, e.g. 0.3–0.5) hedges this rather than fully
  replacing the model rate.
- **Credit budget.** Per-event requests cost more than the 3-credit team
  pull: ~10 EPL fixtures × 1 market ≈ 10 credits/week ≈ 40/month of the 500
  free-tier credits. Affordable for a weekly run.

## Validation plan (26/27)

Because there's no backtest, validate the **mechanism** live, the same way
the project judges any model change — accuracy, not points:

1. Each GW, log both the model's `lambda_goals_per_90` and the
   market-implied rate per player with a prop.
2. Over a rolling window, compare calibration of each against actual goals
   (per-position bias / MAE, and especially top-decile/FWD calibration —
   the target weakness).
3. Tune `market_weight` to the blend that minimises out-of-sample error;
   ship only if the market signal demonstrably improves player-goal
   calibration over the model allocation.
4. Guard against the squad-reshuffle trap (Phase 7 FDR lesson): a recommend
   change is only "better" if it improves player-goal accuracy, not just
   because the squad moved.

## What shipped this session

- `derive/player_props.py` — the four pure inversion/blend functions, with
  the minutes-double-count guard.
- `tests/unit/test_player_props.py` — 9 tests pinning the math and the guard.
- This design doc.

Deferred (26/27, needs live data): the `fact_player_odds` table + migration,
the `event-odds` ingest with player-name resolution, and the predict-only
wiring + `market_weight` tuning.
