# Phase 8 — Player-prop (anytime-goalscorer) odds

**Status:** 🟡 fully built and wired, shipped **inert** (`market_weight = 0.0`)
as of 2026-07-26. The table, ingest, consensus, and predict-only blending all
exist and are tested; what does not exist yet is *evidence*, because no
bookmaker has priced this market for 26/27 yet. Every gameweek now writes a
model-vs-market shadow log, and the weight stays at 0 until that log says the
market rate is better calibrated. This is a **ceiling-raiser**, not an
incremental tweak — which is exactly why it does not get to change
recommendations on the strength of its story alone.

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
   market-implied rate per player with a prop. **Done automatically**: every
   `live recommend` writes
   `data/live/recommendations/season_26/gw_<GW>/player_prop_shadow.parquet`
   with both rates, `market_p_anytime`, `market_n_books` and `e_minutes`.
2. Over a rolling window, compare calibration of each against actual goals
   (per-position bias / MAE, and especially top-decile/FWD calibration —
   the target weakness).
3. Tune `market_weight` to the blend that minimises out-of-sample error;
   ship only if the market signal demonstrably improves player-goal
   calibration over the model allocation.
4. Guard against the squad-reshuffle trap (Phase 7 FDR lesson): a recommend
   change is only "better" if it improves player-goal accuracy, not just
   because the squad moved.

## What shipped (June 2026)

- `derive/player_props.py` — the four pure inversion/blend functions, with
  the minutes-double-count guard.
- `tests/unit/test_player_props.py` — 9 tests pinning the math and the guard.
- This design doc.

## What shipped (2026-07-26, season rollover)

- `fact_player_odds` + migration `0009`. Keyed by `quote_time` like
  `fact_odds`, so repeated pre-deadline pulls accumulate as snapshots.
- `ingest/oddsapi_props.py` — the per-event ingest. Lists events (free),
  pulls one market per event inside a horizon, and hard-caps requests per run
  so a double gameweek cannot drain the credit budget.
- Name resolution, five rules deep, scoped to the two clubs in the fixture and
  refusing to guess on ambiguity. Unresolved bookmaker spellings are written to
  a `.unresolved.json` sidecar so a systematic naming change at one book shows
  up as a visible block of misses rather than silent under-coverage.
- `derive.player_props.consensus_anytime_probs` — per-book latest quote,
  de-vig within book, then **median** across books (the free tier's US books
  are thin enough that one stale line moves a mean materially).
  `attach_market_goal_rates` blends into `lambda_goals_per_90`.
- `pit.player_prop_odds_rows` with `as_of` support, so the consensus is
  point-in-time correct.
- Wired into `run_predict_only`, applied *after* the news/availability minutes
  adjustment (the market rate is de-scaled by our expected minutes, which the
  simulator re-applies, so the two must be the same minutes).
- `settings.player_prop_market_weight`, default **0.0**.
- 32 tests in `tests/unit/test_player_prop_ingest.py`.

### Measured facts about the API (2026-07-26)

- `/sports/soccer_epl/events` costs **0** credits and already lists GW1.
- `player_goal_scorer_anytime` is a valid market key (a bogus key 422s).
- An event with no prices returns `bookmakers: []` and costs **0** credits —
  so the weekly probe is free until books actually put the market up, which
  for soccer is typically ~2-3 days before kickoff.
- Both DB-side joins were verified against real 26/27 data: the ARS-COV GW1
  fixture resolves, and 16 of 17 realistic bookmaker spellings resolved to the
  right player (the 17th, Trossard, correctly returned None — he is not in the
  26/27 game).

## Still open

- **No priced payload has ever been parsed.** The priced path is covered only
  by a synthetic payload built to the shape the live API returned. First real
  test is ~2026-08-19. Expect to fix something on the first real pull.
- **`market_weight` is 0**, so this currently changes nothing about any
  recommendation. Raising it requires the shadow-log comparison below.
- Coverage rate is unknown: how many of a fixture's players US books actually
  price is not something the empty responses can tell us.

### The cold-start minutes interaction (found 2026-07-26, measured)

Running the full pipeline on GW1 with injected prices exposed a real problem
for *early-season* use specifically. At GW1 the minutes model has no 26/27 data
and predicted `e_minutes ≈ 11.7` for Saka — a nailed starter. Two consequences:

1. The per-90 divisor hits its `min_minutes_fraction = 0.25` floor, so the
   minutes identity that justifies this whole approach
   (`rate × minutes_fraction == μ_fixture`) **stops holding**: we divide by
   0.25 but the simulator multiplies by 0.13, delivering roughly half the
   market's implied fixture goals. The floor is a deliberate guard against
   exploding a thin line, but at GW1 nearly every player may hit it, which
   turns a per-player safety clamp into a systematic ~50% attenuation.
2. The logged per-90 rates are not comparable to each other while minutes are
   cold (model 0.065 vs market 2.46 for the same player). **Any shadow-log
   analysis must therefore compare at fixture level** —
   `rate × e_minutes / 90` against actual goals — which is exactly why
   `e_minutes` is in the log.

Neither is urgent while `market_weight` is 0, and both mostly self-resolve once
the minutes model has ~5 played GWs. But it means the first three gameweeks of
shadow data are the least trustworthy part of the sample, and that raising the
weight on early-season evidence would be a mistake. If props ever do get a
non-zero weight, revisit whether the floor should scale with how much
current-season minutes data exists.
