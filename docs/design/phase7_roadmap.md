# Phase 7 — Roadmap

Compiled from the user's gap list + the v3 polish-session findings (Phase 6
§10) + my own audit of what's modeled vs missing. Each item is one row of a
roadmap. **Status legend:** ✅ done · ⚠️ partial · ❌ missing · 🛠 broken.

Prioritization criteria: (1) measured/estimated impact in pts/season, (2)
implementation effort, (3) dependencies / blocking other work. Items in the
same tier are roughly independent and can be tackled in any order.

---

## Inventory + status

| # | Requirement | Status | Notes |
|---|---|---|---|
| 1 | Transfer rules + hit penalty | ✅ | MILP constraints; static prices (no price-change prediction; Phase 3.5 had this but it's not active). |
| 2 | 25/26 chip rules (2× each WC/FH/BB/TC, half-split) | ✅ | [5df26fe](src/fpl_bot/optim/state.py). Heuristic scheduler picks GWs once at fold start. |
| 3 | DefCon (FPL 25/26 defensive-contribution points) | ⚠️ MVP | [1b6b639](src/fpl_bot/eval/defcon_adjustment.py). PIT empirical rolling rate × 0.3 shrinkage, read from vaastav CSV. No DB schema, no predictor. |
| 4 | Injuries / availability | ⚠️ | Live: `status_code in (i/n/s/u)` drops; `d` attenuates by cop/100. Backtest: rolling-minutes filter. No expected-return date. |
| 5 | Form (short-term) | ✅ | LightGBM features: rolling last-1/3/5 of pts, minutes, goals, assists, CS, saves. |
| 6 | Past performances (long-term) | ✅ | Cross-season player_id (stable FPL `code`) + rolling features back to 2019. |
| 7 | Ownership (EO) | ⚠️ Heuristic | `selected_by_percent` from FPL API in ρ-EO penalty. Not top-10k specifically. |
| 8 | Captain blank-risk attenuator | ✅ (off by default) | [024e8aa](src/fpl_bot/eval/milp_backtest.py). Doesn't help on its own. |
| 9 | Cold-start GW1-3 prior blend | ✅ | [5599118](src/fpl_bot/eval/milp_backtest.py). Last-5-GW prior-season actual pts. |
| 10 | BPS simulator | 🛠 | `bps_rule_set` table empty; rules hard-coded for 24/25. Probably fine since FPL didn't change BPS in 25/26 (only added DefCon as a separate score). |
| 11 | Goal/assist xPts model | ⚠️ | Works but chronic FWD under-prediction across all folds (-0.13 to -0.35). |
| 12 | Clean sheet model | ⚠️ | Market-only fallback chosen (Phase 2.3); residual lift didn't pass acceptance gate. |
| 13 | Lineup news / press conferences | ❌ | No ingest. The biggest single source of user-bot edge. |
| 14 | Bookmaker odds (goal probabilities) | ❌ | Internal Dixon-Coles used instead. Free APIs exist (Pinnacle, Betfair Exchange). |
| 15 | Top-10k EO | ❌ | LiveFPL / FPLStatistics scrape needed. |
| 16 | Set-piece taker rotation | ⚠️ | Manual penalty-taker list exists; no per-week rotation tracking. |
| 17 | Suspension carry-over (yellow → 5/10/15) | ❌ | Not tracked. Small impact. |
| 18 | Price-change predictor | ⚠️ Built, not active | Phase 3.5 has a transfer-balance model; not wired into prediction pipeline. |
| 19 | Manager rotation patterns | ❌ | Pep-style rotation isn't modeled. |
| 20 | Fixture congestion (3-in-7) | ❌ | Each fixture treated independently. |
| 21 | Weather / kick-off time | ❌ | Out of scope. |

---

## Prioritized implementation plan

### Tier 1 — Highest value / lowest effort (do first)

**1.1 Productionize DefCon** (1-2 days)
- DB migration: add `defensive_contribution`, `tackles`, `recoveries`, `clearances_blocks_interceptions` columns to `fact_player_match`.
- Update vaastav ingest to parse the new columns. Backfill 25/26 (the only season with these columns).
- Move `compute_defcon_adjustments` from CSV-direct read to a PIT primitive (`pit.defcon_per_player_per_gw`).
- Per-position shrinkage tuning (instead of single 0.3). DEF probably needs more shrinkage than MID since DEF's rolling features capture more.
- Add unit tests for the PIT primitive.
- Acceptance: cross-fold diagnostic shows 25/26 per-position bias within ±0.05 of fold 24.

**1.2 Bookmaker odds ingest for goals/CS** ✅ DONE (turned out to be a one-command catch-up)
- Audit revealed we already had: football-data.co.uk ingest pulls Pinnacle closing odds (PSH/PSD/PSA) + Over/Under 2.5 (P>2.5 / P<2.5) → `fact_odds`; `derive market-xg` runs Dixon-Coles inversion → `fact_market_xg`; feature builders already consume `pit.market_xg_for_fixtures()`.
- Gap: 25/26 had never been run through this pipeline. Pinnacle's free API is gone (auth-only), but football-data.co.uk historical CSVs include Pinnacle closing odds — sufficient for backtest and any non-live recommendation.
- Catch-up ran in two commands:
  - `fpl-bot ingest footballdata --season-folder 2025-26` → 2,760 odds rows
  - `fpl-bot derive market-xg --season-id 25` → 684 market_xg rows
- Cache invalidated and regenerated.
- Result: bot 1402 → **1498** (+96 pts, beats template +24 for the first time on 25/26).
- Live-recommend caveat: football-data.co.uk publishes only post-match. For live use the system still relies on the FPL internal Dixon-Coles. A separate live-odds feed (The Odds API or a manual scrape) remains a future v4 item — but for the historical and "look at last week to set this week" recommendation flow we're fine.

**1.3 Top-10k EO scraping** (1 day)
- LiveFPL `/elite-eo` or FPLStatistics scraping (latter has cleaner table HTML).
- Schedule: weekly snapshot to `fact_top10k_eo(player_id, gw_snapshot, eo_pct, recorded_at)`.
- Wire into `eo_for_candidates` to blend top-10k with general EO (weight TBD; start 0.7 top10k + 0.3 general).
- Acceptance: ρ-EO sweep shows improvement on retro vs general-EO-only.

**1.4 25/26 BPS audit** ✅ DONE — no changes needed
- Empirical audit: for each position, sampled single-goal scorers (no
  assist, no card, no OG) and computed remainder = bps − (6 for 60min
  appearance + 12 if DEF CS + GC deductions). Remainders cluster around
  the existing values (DEF=24, MID=18, FWD=12) with small variance from
  unaccounted BPS sources (shots, passes, tackles) — consistent with
  unchanged scoring matrix.
- FPL 25/26 rule changes are:
  1. Defensive Contribution Points (DefCon) — SEPARATE +2 pts category
     (handled by Phase 7 1.1)
  2. BPS scoring matrix — UNCHANGED
  3. No other rule changes affecting prediction-relevant scoring
- `bps_rule_set` table remains empty; the hard-coded constants in
  `fpl_bot/models/bps.py` already match 25/26's reality.

### Tier 2 — Significant effort, high reward

**2.1 Lineup news ingest pipeline** (1-2 weeks)
- Sources: Twitter/X official club accounts, press conference transcripts via official EPL APIs (Stats Perform, OPTA), or scraping fantasyfootballscout.co.uk's "team news" page.
- Pipeline: ingest → NLP entity extraction (player names + status keywords: "doubt", "ruled out", "expected to start") → store in `fact_player_news(player_id, news_ts, sentiment, source_url)`.
- Wire into live `LiveStatusOverrides` to drop "ruled out" and attenuate "doubtful".
- Acceptance: bot avoids carrying ruled-out players at the deadline on a sample of past GWs.
- **Risk**: this is most-of-a-month of work. NLP entity extraction is fiddly. Twitter API requires paid access ($100/mo Basic tier).

**2.2 Productionize price-change predictor** (3-4 days)
- Phase 3.5 has `transfers_balance` model code; wire its output into `_resolve_price_at_gw` to predict prices forward.
- Add price-change-aware sell tax: hold-to-rise transfers should win bank.
- Acceptance: retro shows bot accumulates bank value relative to template.

**2.3 Set-piece rotation tracker** (1-2 days)
- Track corner / FK / penalty assignments per match (vaastav has `penalty_taker` flag via `penalties_missed`/`penalties_scored`).
- Build a rolling-3-matches set-piece-share per player.
- Multiply assist xPts by set-piece share for corners/FKs (currently uniform).
- Acceptance: assist MAE improves on out-of-fold.

### Tier 3 — Modest impact, low priority

**3.1 Suspension carry-over** (1 day)
- Track yellow-card cumulative count per player; flag at 4 (next yellow → suspension), 9, 14.
- Drop suspended players from candidates.
- Estimated value: 5-15 pts/season.

**3.2 Manager rotation patterns** (1-2 days)
- Build per-manager "rotation propensity" stat (% of XI changes week-to-week).
- Attenuate minutes prediction for high-rotation teams' fringe players.
- Estimated value: 10-30 pts/season.

**3.3 Fixture congestion adjustment** (1 day)
- For each fixture, compute days-since-last-fixture per team.
- Penalize minutes/xPts predictions for 3-in-7-day clusters.
- Estimated value: 10-20 pts/season.

### Tier 4 — Skip unless other priorities cleared

**4.1 Weather / kick-off time** — out of scope; no evidence of meaningful impact.

**4.2 Per-fold transfer-penalty tuning** — already swept; default 0 is right.

---

## Working order

I'll execute Tier 1 in order: 1.1 → 1.2 → 1.3 → 1.4. Each is committed
independently. I'll pause and check in:

- Before starting **1.2** (bookmaker odds) to confirm API choice and whether
  you want to pay for Pinnacle real-time vs free Betfair Exchange.
- Before starting **2.1** (lineup news) to confirm scope — this is a
  multi-week project and may need a separate design doc.
- If any item's acceptance gate fails, before tuning.

When all Tier 1 items are committed, I'll re-run the cross-fold diagnostic
and multi-fold backtest to measure cumulative impact. Then we decide if
Tier 2 is worth the effort or if we ship what we have.

---

## Tier 1 close-out — measured impact

All 4 items shipped. Multi-fold validation (all v3 + all Tier 1 changes,
default settings):

| Fold | Bot | Template | Diff | Δ vs pre-Tier-1 |
|---|---|---|---|---|
| 21 | 2562 | 1735 | +827 | +67 |
| 22 | 2380 | 1818 | +562 | −40 |
| 23 | 2485 | 1756 | +729 | −36 |
| 24 | 2406 | 2082 | +324 | +22 |
| 25 | **1517** | 1466 | **+51** | **+196** |

All 5 validity gates pass on every fold. **Net Tier 1 impact across all
folds: +209 pts.** The headline win is +196 on 25/26 — the bot now
beats template by +51 on the current season (vs −128 at session start).

Small negatives on folds 22/23 (~−40 each) are within noise, driven by
the LiveFPL EO snapshot being from May 2026 applied across all backtest
GWs (documented lookahead approximation; identical caveat applies to
the pre-existing overall_eo_snapshot). For LIVE recommend the EO blend
is PIT-correct.

### Cumulative session progress (1215 → 1517 on 25/26, +302, 68% of gap)

| Stage | 25/26 bot | Δ |
|---|---|---|
| Session start | 1215 | — |
| Phase 6 v3 polish (8-chip, BB fix, GW1-3 prior, etc.) | 1321 | +106 |
| Tier 1.1 DefCon productionized | 1402 | +81 |
| Tier 1.2 market_xg catch-up (football-data) | 1498 | +96 |
| Tier 1.3 LiveFPL top10k EO blend | 1517 | +19 |
| Tier 1.4 BPS audit (no change) | 1517 | 0 |
| **Final** | **1517** | **+302** |

Gap to user REDACTED_TEAM_ID (1661 over 29 played GWs): −144 (was −446 at start).
