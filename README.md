# fpl-bot

A Fantasy Premier League decision bot: a probabilistic prediction stack
(expected points per player per gameweek) feeding a mixed-integer linear
program (MILP) that picks the squad, starting XI, captain, transfers, and
chip timing under FPL's budget/formation/transfer rules.

---

## ⚠️ Honesty notes (read these first)

**On AI authorship.** This project was built almost entirely by AI coding
agents under human direction. 69 of 72 commits carry a `Co-Authored-By:
Claude` trailer; the bulk was written by Claude Code (Claude Opus), with
some later changes by OpenAI Codex. A human (the repo owner) set the
goals, made the calls, reviewed output, and ran the experiments — but the
design docs, code, tests, and most of this README are AI-generated. If
you're evaluating this as a portfolio piece or auditing it for
correctness, weight it accordingly: treat it as AI-pair-programmed work,
not hand-authored.

**On performance — it is not elite.** Honest expectation from the 25/26
backtest: the bot lands around **top 150-300k** of ~11M managers —
comparable to a good-but-not-great human, fully automated. It does **not**
reach top-20k. Two reasons, measured rather than asserted:

- The bot's edge over a naive buy-and-hold baseline comes almost entirely
  from **bookmaker odds**. With odds disabled it *loses* to buy-and-hold
  (measured: −70 pts over 29 GWs). Its skill is essentially "read the
  betting market and optimize around it." The betting market is efficient,
  so that caps the ceiling.
- The flattering historical-fold backtest numbers (~2400-2560 pts/season)
  are partly inflated by an 8-chip ruleset applied to seasons that only
  had ~5 chips, and by closing-odds look-ahead. The clean 25/26 test (with
  the correct ruleset) is the honest one.

**On backtest noise.** The MILP solver (CBC at a 1% MIP gap) is
non-deterministic across runs: the same configuration varies by roughly
±28 points over a season. Any backtest comparison under ~30 pts is within
noise and shouldn't be over-read.

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
   cold-start prior, DefCon (FPL 25/26 defensive-contribution points), and
   optional calibration. Shared by backtest and live so they don't drift.
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
  cli/        typer CLI entrypoint (`fpl-bot ...`)
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
uv run fpl-bot ingest fpl             # pull current FPL bootstrap
uv run pytest tests/ -q               # 194 unit + leakage tests
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
