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

**1.2 Bookmaker odds ingest for goals/CS** (2-3 days)
- Source: Betfair Exchange JSON API (free, requires app key registration) OR Pinnacle (free, no registration). Decide based on coverage of full season schedule pre-deadline.
- Ingest: pre-deadline pull of Match Odds + Over/Under goal markets per fixture.
- New table `fact_bookmaker_odds(fixture_id, market, runner, price, recorded_at)`.
- Conversion: implied λ_home / λ_away from goal markets — replaces or blends with Dixon-Coles `lambda_market_xg`.
- Acceptance: per-fold Brier(≥6) improvement on out-of-fold validation. Target: −0.005 absolute Brier on at least 3 of 4 folds.

**1.3 Top-10k EO scraping** (1 day)
- LiveFPL `/elite-eo` or FPLStatistics scraping (latter has cleaner table HTML).
- Schedule: weekly snapshot to `fact_top10k_eo(player_id, gw_snapshot, eo_pct, recorded_at)`.
- Wire into `eo_for_candidates` to blend top-10k with general EO (weight TBD; start 0.7 top10k + 0.3 general).
- Acceptance: ρ-EO sweep shows improvement on retro vs general-EO-only.

**1.4 25/26 BPS audit** (half day — quick verification, possibly already correct)
- Confirm whether FPL changed BPS scoring matrix in 25/26 (vs the DefCon addition which is a separate score). If unchanged, no work needed; document and close.
- If changed: populate `bps_rule_set` table with 25/26 deltas; wire `score_bps_known_events` to look up by season_id.

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
