# Phase 6 — Live Run (Design)

Status: **v1 shipped (architecture complete); v2 needed for upcoming-GW recommend.**

**v1 ships** (8d9bf14 + d6a0791):
- Schema (`fact_user_team_snapshot`), ingest extension (`fetch_my_team` + `parse_my_team`), state builder, status filter (`i/n/s/u` drop + `d` attenuation), recommend pipeline, retrospective scorer, `fpl-bot live {ingest,recommend,retrospective}` CLI.
- 141/141 tests pass. Live smoke verified: `fpl-bot live ingest` pulls bootstrap-static + fixtures + user-team picks cleanly.
- Bonus: idempotency fix on vaastav ingest (d6a0791) — unblocks 25/26 historical data ingest.

**v2 gap** surfaced during live smoke: the recommend pipeline reuses `_run_one_fold` (a backtest function) which produces predictions ONLY for GWs that have:
1. fact_player_match rows (played fixtures), AND
2. understat data (typically lagged by a few days post-match).

For the **upcoming GW** (deadline-time recommend), neither (1) nor (2) is true yet. Result: `_run_one_fold` returns predictions for the most-recent-played GW range (e.g., 25/26 produced predictions for GW1-29), and the recommend pipeline raises `RuntimeError: No predictions for any GW in horizon starting at GW{N}` when N > last-predicted GW.

**v2 scope** (documented but not implemented):
- Add a `predict-only mode` to the feature builders (`features/minutes.py`, `features/goals.py`, `features/clean_sheet.py`) that produces per-(player, fixture) feature rows for UPCOMING fixtures using only rolling-from-past features (rolling features `shift(1).over(player_id)` work for unplayed fixtures because they look at PRIOR rows; the blocker is the `minutes >= 1` filter that drops unplayed fixtures from the feature table).
- New entry point `eval/xpts_eval.run_predict_only(test_season, train_seasons, fixture_ids)` that:
  - trains models on train_seasons (unchanged),
  - builds prediction-mode feature rows for the specified UPCOMING fixture_ids,
  - applies trained models,
  - runs the joint xPts simulator,
  - returns eval_df WITHOUT actuals (no `total_points` column — just predictions).
- Update `live/recommend.py` to use `run_predict_only` when the test_season's most-recent prediction GW < target gameweek.

Estimated effort: 1-2 days. Touches 3 feature builders + xpts_eval entry point + Phase 6 recommend wiring + tests.

**Workaround in the meantime**: live recommend works for any GW where backtest predictions are available (e.g., 25/26 GW1-29 currently). Useful for retrospective comparison ("what would the bot have recommended last GW?") but not for the upcoming deadline.

The bot is mature: 2584 on fold 24/25 backtest, all 5 validity gates pass, full Phase 1 → Phase 5 stack stable. Phase 6 ships it as a live production tool — pull the current FPL season state pre-deadline, run the rolling MILP, emit a human-actionable recommendation.

Per Phase 0 §10: weekly live run is the cap of the project. Backtest validates the methodology; Phase 6 actually scores points in the 25/26 season.

---

## 1. Scope

| Decision | v1 choice | Rationale |
|---|---|---|
| **Trigger** | CLI invocation (manual at first; cron later) | Avoids deployment infrastructure overhead in v1; user runs `fpl-bot live recommend` before each deadline. |
| **Solver stack** | Phase 5 deterministic + chip-DP + auto-sub, CBC backend | Best-validated config (2584 on 24/25). No SAA (v1 underperformed; v2 same result 7× slower). |
| **Live overrides** | Status-code-based candidate filter + chance-of-playing minutes attenuation | Phase 0 §12 flagged this as the live-only path; never used in backtest because vaastav has no historical status. |
| **Cold start vs mid-season join** | Both supported | First GW of season: full cold-start MILP. Mid-season: read current squad from FPL API's user-team endpoint (or YAML config) and resolve cost basis. |
| **Output format** | Markdown + JSON sidecar | Human reads the markdown; JSON for retrospective tooling. |
| **Persistence** | Per-GW recommendation + post-GW actuals → `data/live/recommendations/season_25/gw_{N}/` | Lets us compute live-vs-backtest gap at season end. |
| **Multi-user** | v1 = single user (the dev), team_id from config | Multi-user / web UI is out of scope. |

---

## 2. Architecture

### 2.1 Live-data ingest (mostly exists; minor extension)

Existing `ingest/fpl_api.py` already pulls and parses:
- `bootstrap-static` → `dim_player`, `dim_team`, `fact_player_status` (with `status_code`, `chance_of_playing_this_round`, `chance_of_playing_next_round`)
- `fixtures` → `dim_fixture`

Needs adding:
- **User-team endpoint** (`my-team/{team_id}/` or `entry/{team_id}/event/{gw}/picks/`): pull the current squad + bank + free transfers + chips used so far
- **Cost-basis ingest**: vaastav-style `purchase_price` per player from the user-team endpoint (FPL API exposes `cost`, `selling_price`, `multiplier`)

### 2.2 State construction

Build a `BacktestState` (same dataclass we use in backtest) from live ingest:
- `season_id = 25`
- `gameweek = current_gw` (from `bootstrap-static.events.is_next`)
- `squad = frozenset(player_ids)` from user-team picks
- `bank = bank_tenths` from user-team
- `free_transfers = ft_count` from user-team
- `chips_used = frozenset({...})` from user-team `chips` array
- `cost_basis = {player_id: purchase_price}` from user-team picks

This is just the existing `BacktestState` populated from live data instead of cold-start.

### 2.3 Status-code filtering

FPL `status_code` values:
- `a` = available
- `d` = doubtful (chance_of_playing < 100%)
- `i` = injured
- `n` = not in squad / not yet (loan / etc.)
- `s` = suspended
- `u` = unavailable (long-term, retired)

v1 policy:
- `i`, `n`, `s`, `u` → **exclude from candidate pool** entirely (don't even consider buying)
- `d` → keep, but **attenuate predicted xPts by chance_of_playing/100** before MILP solve
- `a` → no override

### 2.4 Solver invocation

Single rolling-MILP solve at the current GW with H=6:
- Chip-DP runs once at the start (or annually) to set the chip schedule for the rest of the season
- MILP runs with picked chip schedule + status overrides + current state

### 2.5 Output

Markdown report `data/live/recommendations/season_25/gw_{N}/recommendation.md`:

```markdown
# GW {N} Recommendation — generated {ISO timestamp}

## Squad (15 players)
GKP: Raya · Pope
DEF: Trippier · Saliba · Gabriel · Estupiñán · Robertson
MID: Saka · Palmer · Salah · Fernandes · Foden
FWD: Haaland · Watkins · Solanke

## Starting XI (11)
GKP: Raya
DEF: Trippier · Saliba · Gabriel · Robertson
MID: Saka · Palmer · Salah · Fernandes
FWD: Haaland · Watkins
Bench (in order): Pope · Estupiñán · Foden · Solanke

## Captain: Haaland (vice: Salah)

## Transfers
OUT: Player X (selling at £X.Xm)
IN:  Player Y (buying at £Y.Ym)
Hits: 0 (within FT budget)

## Chip
None this GW. Next planned: FH1 at GW17.

## Expected GW points: ~62.4 (top scenario)
```

JSON sidecar `recommendation.json`:
```json
{
  "season_id": 25, "gameweek": N,
  "squad": [...],
  "starting_xi": [...],
  "captain": <id>, "vice": <id>,
  "bench_order": [...],
  "transfers_in": [...], "transfers_out": [...],
  "hits": 0,
  "chip_played": null,
  "objective_value": ...,
  "predicted_gw_points": ...,
  "chip_schedule": {...}
}
```

### 2.6 Post-GW retrospective

After each GW closes:
- Pull actual results via `bootstrap-static` (now includes `total_points` for finished GW)
- Compute realized GW points using the same scorer (auto-sub + auto-vice)
- Append to `data/live/recommendations/season_25/gw_{N}/actuals.json` and to a season-rolling `season.csv`

End-of-season: compute live vs backtest delta. If live MUCH lower → model degradation in the wild; investigate features that don't generalize.

---

## 3. CLI surface

New commands under `fpl-bot live`:

```
fpl-bot live ingest        # pulls bootstrap-static + fixtures + user-team
fpl-bot live recommend     # builds state, runs MILP, writes recommendation
fpl-bot live retrospective # pulls finished-GW actuals, writes actuals.json
fpl-bot live status        # quick CLI status: current GW, deadline, last recommendation
```

All commands take an optional `--season 25` flag (defaults to current).

---

## 4. Acceptance gates

All MUST PASS at v1 ship:

1. **Live ingest end-to-end**: `fpl-bot live ingest` pulls and parses without error; `fact_player_status` and user-team table get fresh rows.
2. **Recommend produces valid output**: `fpl-bot live recommend` emits a recommendation file with all five validity gates of the underlying MILP solve passing.
3. **Status override actually filters**: a doubtful player (`status_code = d`, `chance_of_playing < 50`) gets attenuated; an injured player (`status_code = i`) is dropped from the candidate set.
4. **Cold start path works**: invoking at GW1 with no prior squad cold-starts correctly.
5. **Mid-season path works**: invoking at GW N>1 with a populated `state.squad` correctly evolves to GW N+1.

Reported (not gated):
- First-week recommendation diff vs a sensible template / popular pick.

---

## 5. Code structure

### New files
- `src/fpl_bot/live/state_builder.py` — builds BacktestState from FPL API user-team payload + status filter.
- `src/fpl_bot/live/recommend.py` — top-level: ingest → state → MILP → write recommendation.
- `src/fpl_bot/live/retrospective.py` — post-GW actuals fetch + comparison.
- `src/fpl_bot/live/output.py` — markdown + JSON writers.
- `src/fpl_bot/cli/live.py` — new typer subcommand group.
- `tests/unit/test_live_state_builder.py` — synthetic FPL API payload → BacktestState parsing.
- `tests/unit/test_status_override.py` — i/n/s/u dropped, d attenuated, a unchanged.

### Modified files
- `src/fpl_bot/ingest/fpl_api.py` — add `fetch_my_team(team_id, gw)` → user-team endpoint, parse to a dataclass.
- `src/fpl_bot/cli/main.py` — wire the new `live` subcommand group.
- `src/fpl_bot/db/models.py` — add a small `fact_user_team_snapshot` table (per (season_id, gameweek, team_id, recorded_at) bitemporal) so we have an audit trail of live state at each pull.

---

## 6. Build order

1. **Live ingest extension**: `fetch_my_team` + new schema table. Smoke against the real FPL API for a known team_id.
2. **State builder**: payload → BacktestState. Unit tests with synthetic payloads.
3. **Status override**: candidate filter + xPts attenuator. Unit tests.
4. **Recommend pipeline**: end-to-end CLI invocation that writes markdown + JSON.
5. **Retrospective pipeline**: post-GW actuals fetch + scorer integration.
6. **Live smoke**: run `fpl-bot live recommend` against the current 25/26 GW for a real `team_id`. Sanity-check the picked squad.
7. **Commit + design doc final status.**

---

## 7. Open questions

1. **`team_id` source**: hard-code in `.env`? Pass via CLI flag every call? Both? Recommend: env var `FPL_TEAM_ID`, optional CLI override.

2. **Multi-rolling-solve cadence**: should live recommend re-run the chip-DP scheduler each call, or fix it once per season? Recommend: re-run per call (cheap, captures fixture updates).

3. **Live `n_iterations` for the simulator**: backtest used 200. Live can use more (200-500) since wall time per GW solve is now ~1s with CBC. Recommend 500.

4. **Status `d` (doubtful) attenuation curve**: `chance_of_playing` returns {25, 50, 75} typically. Linear `xPts × p/100` or step-function (≥75 → 100%, 50 → 75%, etc.)? Recommend: linear `p/100` for simplicity; revisit if backtest live-replay shows bias.

5. **Cold-start mid-season**: if joining the bot mid-season without a `team_id`, build a fresh squad. Use the existing cold-start path (H=1, candidate filter, full optimal pick). Recommend: support both `--cold-start` and `--from-team-id` modes.

6. **Free-hit squad reversion**: if the bot recommends FH, next GW's recommendation must use the PRE-FH squad. The user-team endpoint will show the pre-FH squad after the FH-GW closes (FPL auto-reverts), so this should "just work". Confirm during live smoke.

---

## 8. Visible post-v1 follow-ups

1. **Cron / scheduled runs**: GitHub Actions or systemd timer to auto-pull and recommend before each deadline.
2. **Slack / email notifications**: push the markdown to a notification channel.
3. **Multi-user / web UI**: Streamlit dashboard with team_id input.
4. **Live `chance_of_playing` blend with model minutes prediction**: instead of just attenuating xPts, fold the FPL signal into the Phase 2.1 minutes model at predict time.
5. **Top-10k EO scraping** (LiveFPL / FPLStatistics): per Phase 0 §1.2, top-10k EO would tighten the ρ-EO term. Deferred from Phase 0 due to scraping difficulty.
6. **Late-deadline pull**: re-run within ~1 hour of deadline to catch last-minute lineups / injuries.

---

## 9. Effort estimate

- Day 1: schema + ingest extension + state builder + tests
- Day 2: recommend pipeline + output formatters + CLI
- Day 3: retrospective + live smoke + commit

Roughly **2-3 days** of focused work.
