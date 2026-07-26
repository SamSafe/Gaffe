# Gaffe

A Fantasy Premier League decision bot: a probabilistic prediction stack
(expected points per player per gameweek) feeding a mixed-integer linear
program (MILP) that picks the squad, starting XI, captain, transfers, and
chip timing under FPL's budget/formation/transfer rules.

---

## ⚠️ Honesty notes (read these first)

**On AI authorship.** Most of the code here was written by AI coding
agents — 83 of 92 commits carry a `Co-Authored-By: Claude` trailer, mostly
Claude Code (Claude Opus) with a few later changes by OpenAI Codex. They
were a fast way to turn decisions into working code, but the decisions
were the work: the architecture, the modelling approach, what to build and
what to cut, and — most of all — reading the results honestly and telling
a real improvement from a flattering-but-spurious one (see the methodology
note below; that judgement is where the project lives or dies). So the
design docs, code, and tests are AI-generated, but the direction, the
domain calls, and the validation are human.

**On performance — it is not elite.** Honest expectation from the 25/26
backtest: the bot lands around **top 150-300k** of ~11M managers —
comparable to a good-but-not-great human, fully automated. It does **not**
reach top-20k. Two reasons, measured rather than asserted:

- The bot's edge over a naive buy-and-hold baseline comes almost entirely
  from **bookmaker odds**. With odds disabled it _loses_ to buy-and-hold
  (measured: −70 pts over 29 GWs). Its skill is essentially "read the
  betting market and optimize around it." The betting market is efficient,
  so that caps the ceiling.
- The flattering historical-fold backtest numbers (~2400-2560 pts/season)
  are partly inflated by an 8-chip ruleset applied to seasons that only
  had ~5 chips, and by closing-odds look-ahead. The clean 25/26 test (with
  the correct ruleset) is the honest one.

**On backtest reproducibility (and a methodology trap).** The backtest is
now bit-reproducible: the CBC solver runs with a pinned seed, and LightGBM
training and the Monte-Carlo simulator are deterministic (`deterministic`/
`force_row_wise`/pinned threads; per-fixture seeded RNG; players sorted
before sampling). Same config → same result, every run. This replaced an
earlier regime where regeneration varied by tens of points and silently
masked real model changes.

Reproducibility is *not* the same as a meaningful difference, though. The
MILP picks one of many near-equal-expected-value squads within its 1% MIP
gap, so a small feature change can deterministically reshuffle the squad
into one that happens to fit a given season's actual results — a real,
reproducible points swing that does **not** generalize. The honest test of
a prediction-model change is therefore whether it improves prediction
**accuracy** (per-position bias / MAE / decile calibration), not whether
the backtest points went up. Measured the hard way: a fixture-difficulty
feature scored +57 on 25/26 with *zero* accuracy gain (flat MAE on two
folds) → reverted as spurious; an isotonic FWD calibration likewise.
Suspension carry-over, by contrast, both improved points (+68/+86) and had
a clear mechanism (banned players score 0), so it shipped.

**It has not been run live for a full season yet.** Everything here is
validated on historical backtests. The real test is 2026/27.

---

## How it works

```
ingest ──► prediction models ──► joint xPts simulator ──► MILP optimizer ──► recommendation
(FPL,      (minutes, goals,      (Monte-Carlo combine     (squad/XI/captain/
 odds,      assists, clean        per-player outcomes      transfers/chips under
 xG, EO)    sheet, BPS)           into points PMF)         budget + FPL rules)
```

1. **Ingest** (`ingest/`) — FPL API (squad, prices, status, news), vaastav
   historical match data, Understat xG/xA, football-data.co.uk closing
   odds, The Odds API live odds, LiveFPL top-10k effective ownership.
2. **Prediction** (`features/`, `models/`) — LightGBM models for minutes
   (4-bucket), goals & assists (Poisson per-90 on xG + team form +
   set-piece roles), clean sheets (market-prob baseline), and a BPS/bonus
   simulator. Bookmaker-implied team λ (Dixon-Coles inversion) is a core
   feature.
3. **Joint xPts** (`eval/xpts_eval.py`, `models/bps.py`) — a Monte-Carlo
   simulator combines per-player outcome distributions into an expected-
   points distribution per (player, gameweek).
4. **Post-processing** (`optim/prediction_postprocess.py`) — early-season
   cold-start prior, DefCon (FPL 25/26 defensive-contribution points),
   domestic-suspension carry-over (zero a banned player's GW), and optional
   calibration. Shared by backtest and live so they don't drift.
5. **Optimize** (`optim/milp.py`) — a rolling-horizon Pyomo/CBC MILP picks
   the value-maximizing legal squad, with a heuristic chip scheduler
   (`optim/chip_scheduler.py`) and an EO-aware template-differential term.
6. **Live** (`live/`) — pulls current state and emits a markdown + JSON
   recommendation per gameweek; see [`docs/RUNBOOK_live.md`](docs/RUNBOOK_live.md).

The project was built in numbered phases; design docs live in
[`docs/design/`](docs/design/) (phase 0 = data/schema, phases 2.x =
prediction models, phase 3 = MILP, phase 5 = chips, phase 6 = live run,
phase 7 = odds/DefCon/EO + model improvements). Phase 4 (stochastic
sample-average MILP) was tried and shelved — same results as the
deterministic version, ~7× slower.

## Tech stack

Python ≥3.11 · PostgreSQL + SQLAlchemy + Alembic · polars · LightGBM ·
Pyomo + CBC · scipy · pydantic-settings · typer/rich · pytest. Managed
with `uv`.

## Repo layout

```
src/fpl_bot/
  cli/        typer CLI entrypoint (`gaffe ...`)
  db/         SQLAlchemy models + point-in-time (PIT) query layer
  ingest/     data-source ingest (fpl, vaastav, understat, footballdata, oddsapi, livefpl)
  derive/     Dixon-Coles odds → market xG
  features/   feature builders (minutes, goals, clean_sheet, bps, price)
  models/     LightGBM trainers + BPS simulator + calibration
  eval/       walk-forward backtest harness, joint-xPts eval, diagnostics
  optim/      MILP, chip scheduler, EO, prediction post-processing
  live/       live recommend pipeline + state builder
configs/      hand-curated set-piece takers + optional live-state overrides
docs/         design docs + live runbook
tests/        unit + leakage (point-in-time) tests
```

## Quickstart

```bash
uv sync --all-extras
uv run alembic upgrade head           # needs a PostgreSQL DB (see fpl_bot/config.py)
uv run gaffe ingest fpl             # pull current FPL bootstrap
uv run pytest tests/ -q               # 265 unit + leakage tests (DB-backed
                                      # ones are marked `integration` and
                                      # auto-skip when no Postgres is up)
```

Running it for a real gameweek: follow [`docs/RUNBOOK_live.md`](docs/RUNBOOK_live.md).

Secrets/config go in a gitignored `.env` (env-prefix `FPL_BOT_`): e.g.
`FPL_BOT_ODDS_API_KEY` (The Odds API free tier), optionally
`FPL_BOT_FPL_COOKIE` for exact bank/transfer/price state.

## Data sources & compliance

Research / non-commercial use only. Sources are public and free:
FPL public + authenticated-self endpoints, vaastav's redistributed FPL
data, Understat (via vaastav mirror), football-data.co.uk, The Odds API
(free tier, own key), LiveFPL. Each ingest writes an audit row (URL,
timestamp, hash). Respect each source's terms if you reuse this.

## Status & limitations

- **Validated** on walk-forward backtests (seasons 2019/20 → 2025/26),
  with point-in-time leakage tests guarding feature construction.
- **Not** validated live over a full season.
- Known gaps the bot can't see: real-time lineup leaks, press-conference
  nuance beyond the FPL `news` field, manager rotation intent. These are
  the main edge a skilled human retains over it.
- A shelved Phase 3.5 price-change predictor is disabled by default (it
  failed its walk-forward gates).

## Disclaimer

Not financial advice; FPL involves no real-money stakes but this makes no
guarantees about your rank. Use the recommendations as a strong starting
point to adjust with your own judgement, not as gospel.
