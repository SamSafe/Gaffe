# Live Weekly Runbook (26/27 season)

## Audit update — 2026-10-11

The local database has now been refreshed through finalized GW5. GW6 is still
in progress. The default training seasons now include 26; only finalized
matches should be ingested into that training set. For team 718959, the audited
GW7 override records £2.8m bank, two free transfers, BB1/TC1/WC1 already used,
and purchase prices reconstructed from public transfer and player histories.
Refresh this state after any new transfer; it is a snapshot, not account sync.

Live recommendations now always regenerate their horizon. Historical parquet
caches are retained for backtest research and cannot supply a fresh live
forecast. Outputs also save `fixture_predictions.parquet`,
`player_predictions.json`, predicted GW points, and the selected bench order.
Retrospectives consume that saved bench order rather than sorting by actual
points. Older artifacts without an order use an explicitly labelled fallback.

The live feature builders include the latest completed result. Rest days and
the season match count follow the club's fixture schedule, including fixtures
not yet ingested, because training rows exist for every club fixture (unused
players included). Live saves/cards use the same priors as historical
simulation. Defensive-contribution forecasts are multiplied by expected
fixture appearances in both live and backtest runs. Corrected purchase-price
overrides also govern selling values, so public snapshots cannot grant the
full profit on a rise. A `cost_basis` entry carries forward to later
gameweeks until a newer block replaces it; bank, free transfers and chips
apply only to their own gameweek. Update the price after re-buying a player.

At this audit, all GW7 and GW8 fixtures have market-xG coverage. GW9–12 have
none and use market defaults; the recommendation now reports that coverage.
Run again after GW6 is finalized, and refresh odds/news near the GW7 deadline
on **17 October, 04:00 America/Edmonton**. Detailed results and remaining model
validation requirements are in `docs/AUDIT_2026-10-11_team_718959.md`.

Core prediction trainers now choose boosting rounds using the most recent
20% of distinct kickoff groups inside the training history, with a three-hour
gap for match results. They then refit on all completed training rows using
the chosen round count. No future evaluation labels choose the stopping point.
Recommendation metadata records `training_policy: chronological_v1`.

How to run the bot for a real gameweek. Follow top-to-bottom each week,
ideally ~3-6 hours before the deadline (after team news + once odds are
posted, before lineups leak).

## One-time, at season start — status for 2026-27: STRUCTURALLY READY (2026-08-11)

Kept as the checklist to repeat each August. The latest successful bootstrap
contains all 20 clubs, 577 registered players, and all 380 fixtures. The
authenticated current squad is also stored locally. No schema migration is
needed for the new signings or FPL's position changes: the bootstrap ingest
updates those dimension/status rows. The transfer window remains open through
1 September, so repeat steps 3 and 5 before GW1 and after deadline-day moves.
The GW1 deadline in the current fixture feed is 21 August at 17:30 UTC.

Live decision feeds are deliberately still pending: season-26 odds,
market-xG, player-prop, and EO tables currently contain zero rows. Pull them
in the weekly cycle once bookmakers/LiveFPL publish coverage; until then, a
recommendation is only a cold-start fallback and should not drive the final
squad.

0. **Update the season defaults** in `fpl_bot/config.py`:
   `current_season_id` (26 for 2026-27) and `train_seasons`. Every CLI
   `--season-id` / `--train-seasons` default reads from these, so a stale
   value here silently writes the new season's data into last season's rows
   — the single most damaging season-rollover mistake. Overridable per-shell
   via `FPL_BOT_CURRENT_SEASON_ID` / `FPL_BOT_TRAIN_SEASONS`.

1. **Add promoted clubs to the ingest name maps.** Three separate maps key
   external team names to FPL identifiers, and an unmapped club silently
   drops *all* its odds — which is the bot's main edge — with no error:
   - `ingest/oddsapi.py` `OA_TO_FPL_SHORT` (The Odds API name → short_name)
   - `ingest/footballdata.py` `FD_TO_FPL_SHORT`
   - `ingest/understat.py` `US_TO_FPL_FULL_NAME` (→ `dim_team.full_name`)

   FPL also *renames* clubs between seasons (IPS was `Ipswich` in 24/25,
   `Ipswich Town` in 26/27), so add any new form to
   `understat.FPL_FULL_NAME_ALTERNATES`. Verify with the check in
   "Season-start verification" below. For 26/27: COV, HUL added.

2. **Confirm DB is migrated:**
   ```bash
   uv run alembic upgrade head
   ```

3. **Ingest the new-season FPL bootstrap** (players, teams, fixtures, prices):
   ```bash
   uv run gaffe ingest fpl --season-id 26
   ```

4. **Decide `train_seasons`.** Until 26/27 has ~5+ played GWs, train on
   everything through 25/26 (`19,20,21,22,23,24,25` — the preseason setting).
   Once 26/27 has enough played GWs (≈GW6+), append 26 in `config.py`. The
   cold-start GW1-3 prior automatically uses the most-recent NON-test
   season's last-5-GW actuals (see `eval/milp_backtest._build...prior`).

5. **Regenerate set-piece takers** from the FPL API's own order fields:
   ```bash
   uv run python scripts/gen_set_piece_takers.py --season-id 26
   ```
   Paste the emitted block into `configs/set_piece_takers.yaml` (replacing
   any existing block for that season). This beats hand-curation: FPL
   maintains `penalties_order` / `direct_freekicks_order` /
   `corners_and_indirect_freekicks_order` itself. Re-run it a few times
   during the season — the pre-season list is FPL's guess for unsettled
   teams, and promoted clubs' orders move once they actually play. Without a
   block, `is_corner_taker` / `is_fk_taker` are 0 (as they were all of
   25/26 — harmless but inactive). The checked-in season-26 block was last
   regenerated from the 2026-08-11 bootstrap and resolves all 20 clubs.

6. **Prediction caches:** live recommendations always regenerate their
   horizon. Research backtest and raw-sample caches now include the training
   policy version as well as season/train IDs, so legacy stopping-policy
   caches are ignored. They do not track every feature-code change. Keep historical reference variants and
   regenerate only the research key being evaluated after a model change;
   never blanket-delete `data/cache/xpts_predictions/`.

### Check the PREVIOUS season is complete before trusting GW1-3

The GW1-3 prior is built from the last 5 gameweeks of the previous season, so a
truncated previous season silently degrades it — and the local vaastav clone is
a git checkout that only advances when `ingest vaastav` is run. On 2026-08-13
the checkout was 4 months stale, season 25 held results only to GW29, and the
prior was drawn from **February** football instead of May. Re-ingesting moved
Palmer's prior from 9.0 to 3.0 and dropped him out of the recommended XI
entirely — a materially different GW1 squad.

```bash
uv run gaffe ingest vaastav --season-folder 2025-26   # git pull + reparse
uv run python -c "
from fpl_bot.db import pit
gws = sorted({int(r) for r in pit.all_player_match_with_kickoff(season_ids=[25])['gameweek']})
print('previous season gameweeks with results:', min(gws), '-', max(gws), f'({len(gws)} of 38)')
"
```

Expect 1-38. Anything less means the prior is sampling the wrong months.

### Season-start verification

Catches the two silent failures above — an unmapped club and unresolved
set-piece names:

```bash
uv run python -c "
from fpl_bot.db import pit
from fpl_bot.ingest.oddsapi import OA_TO_FPL_SHORT
from fpl_bot.ingest.footballdata import FD_TO_FPL_SHORT
from fpl_bot.features.goals import _resolved_pk_takers
shorts = set(pit.team_short_names(26))
for name, m in (('oddsapi', OA_TO_FPL_SHORT), ('footballdata', FD_TO_FPL_SHORT)):
    print(name, 'unmapped clubs:', sorted(shorts - set(m.values())) or 'none')
print('pk takers resolved:', _resolved_pk_takers([26]).height, '/ 20')
"
```

### 2026/27 rules audit

- There are again two sets of Wildcard, Free Hit, Bench Boost, and Triple
  Captain. The first set expires after GW19 and cannot carry over. Free Hit
  cannot be played in consecutive gameweeks; the scheduler and MILP now both
  block the GW19/GW20 boundary case.
- Up to five free transfers can still be banked. There is no AFCON free-
  transfer top-up this season.
- Defensive Contribution thresholds are unchanged (DEF 10; MID/FWD 12 for
  +2 points). Season 26 now starts from season-25 player/position evidence and
  then updates point-in-time as new matches arrive.
- The Bonus Points System changed again: being tackled no longer loses BPS,
  CBI earns one BPS per three actions, and goalkeeper save/penalty-save values
  changed. The simulator models its known aggregate events but still uses a
  historical empirical residual for actions we cannot observe separately;
  treat small bonus differences as approximate until live diagnostics exist.
- A gameweek is finalized at 09:00 UK time on the day after its final match.
  Do not run the retrospective before then.

Official references: [chips](https://www.premierleague.com/en/news/4679879/whats-happening-with-fpl-chips-in-202627),
[general changes](https://www.premierleague.com/en/news/4679873/all-you-need-to-know-about-changes-to-fpl-for-202627),
[DefCon](https://www.premierleague.com/en/news/4361991/whats-happening-with-defensive-contribution-points-in-202627-fantasy), and
[BPS](https://www.premierleague.com/en/news/4679946/whats-new-in-202627-fantasy-changes-to-bonus-points-system).

## First three gameweeks — expect the bot at its weakest

> **UPDATE (2026-08-13, after fixing the cold-start prior):** the collapse
> below was largely a BUG, not just early-season data poverty.
> `pit.player_actual_pts_last_n_gws` read the last gameweek from `dim_fixture`
> (the schedule, GW38) instead of from ingested results (GW29), so it queried
> five empty gameweeks, returned `{}`, and `_apply_early_gw_prior` silently did
> nothing — the GW1-3 prior has been inert. With it fixed, Haaland goes 0.46 ->
> 3.50 xPts, the captain moves from a £4m defender to Palmer, and the objective
> rises 234 -> 285. What remains IS genuine early-season weakness: at GW1 the
> blend is 70% "last season's last-5 form" (a noisy 5-match sample), and
> players with no PL history — promoted clubs, overseas signings — have no
> prior at all and still inherit the model's defender bias. Treat GW1-3 output
> as plausible-but-weak rather than broken.

> **ORIGINAL VERDICT (2026-08-13, before the prior fix): do not use the bot's
> GW1 recommendation.** Ingesting odds did not fix the season-opener inversion. The
> minutes model is the binding constraint: it predicted **4.9 expected minutes
> for Haaland**, which collapses every attacking return. The model expects
> Haaland to score **0.0027** goals in GW1; the bookmakers price him at **57%
> to score**. Across all 217 priced players, mean model goal expectation is
> **0.0029 vs the market's 0.207 — off by ~70×**. Clean-sheet points survive
> because they read the market probability directly, so defenders outrank
> forwards 1.32 to 0.28 and the optimizer benches Haaland to captain a
> defender. Player props cannot rescue it either: with expected minutes that
> low the market rate hits its `min_minutes_fraction` floor and ~80% of the
> signal is clamped away. Pick GW1 yourself; revisit the bot around GW5 once
> the minutes model has real current-season data.

**Measured on an earlier GW1 dry run (2026-07-26, before any odds existed):** the
bot produced a £76.0m squad of £4-6m players, left **£24m unspent**, and ranked
cheap defenders ~5× above elite forwards (mean xPts: DEF 1.62 vs FWD 0.32;
Thiaw 2.96 vs Haaland 0.30). Two separate causes, worth telling apart:

- **No odds existed yet**, so every fixture fell back to the hardcoded
  defaults (home λ 1.4, away 1.2, CS 0.30/0.25). That is the bot's documented
  no-odds floor — the regime the README measures as *losing* to buy-and-hold.
  Defenders keep a default clean-sheet value while attackers' goal rates
  collapse, hence the inversion. This should largely resolve once
  `ingest oddsapi` + `derive market-xg` have run for the gameweek.
- **The minutes model is cold**: ~12 expected minutes for a nailed starter,
  which suppresses all attacking returns. That one does *not* resolve until
  26/27 has played gameweeks.

So: **re-run the GW1 recommendation after the first odds pull and judge it
then**. Do not take a pre-odds recommendation seriously, and sanity-check the
GW1 squad by hand regardless — leaving £24m unspent is the tell that the
prediction spread, not the optimizer, is the weak link.

The structural reasons behind this:

- No 26/27 played GWs means no rolling xG/xA or team form; predictions lean
  on the cold-start prior (25/26 last-5-GW actuals) plus market λ.
- Promoted clubs (26/27: COV, HUL, IPS) have **no** PL history at all, and
  neither do incoming transfers from abroad. They are near-invisible to the
  model regardless of the prior — not "predicted to be bad", simply unseen.
- Odds are the main live signal that *does* work at GW1, so getting the
  odds pull in (step 2 of the weekly cycle) matters more than usual.

Treat the GW1 output as a budget-allocation sketch, not a squad.

GW1 mechanics that are now handled, for reference: with no prior squad and no
access token, `live recommend --gameweek 1` solves from a cold-start state (no squad,
£100.0m) instead of erroring, clubs and prices come from the FPL bootstrap
snapshot rather than from played matches, and the chip scheduler will not
assign any chip inside GW1-3 (`chip_scheduler.COLD_START_GWS`) — it previously
triple-captained a £4.0m defender in GW1.

## Every gameweek (the weekly cycle)

Run in this order. Each step is idempotent. `--season-id` and
`--train-seasons` are passed explicitly below even though they now default
from `config.py` — a wrong season id here corrupts data silently, so it is
worth seeing the value you are writing to.

```bash
# 1. Refresh FPL state — prices, status_code, news, your squad.
uv run gaffe ingest fpl --season-id 26
uv run gaffe live ingest --season-id 26 --team-id <YOUR_TEAM_ID> --gameweek <GW>

# 2. Pull pre-deadline bookmaker odds (The Odds API, ~3 requests).
#    Needs FPL_BOT_ODDS_API_KEY in .env (500 req/mo free tier).
uv run gaffe ingest oddsapi --season-folder 2026-27

# 2b. Anytime-goalscorer player props (Phase 8). Free while no book has
#     priced the market, ~1 credit per priced fixture once they have.
uv run gaffe ingest oddsapi-props --season-folder 2026-27

# 3. Pull top-10k effective ownership (LiveFPL scrape).
uv run gaffe ingest livefpl --season-folder 2026-27

# 4. (When vaastav publishes the played GWs) refresh match data for
#    rolling features + DefCon. vaastav lags ~a few days post-match.
uv run gaffe ingest vaastav --parse-only --season-folder 2026-27

# 5. Derive market_xg from the freshly-ingested odds.
uv run gaffe derive market-xg --season-id 26

# 6. Generate the recommendation. Drop the trailing ",26" from
#    --train-seasons until 26/27 has ~5+ played GWs (see season-start step 4).
uv run gaffe live recommend \
    --season-id 26 --team-id <YOUR_TEAM_ID> --gameweek <GW> \
    --train-seasons 19,20,21,22,23,24,25,26
```

The default uses the cross-fold-validated conservative fitter. For shadow
comparison only, `--fit-mode all` also consumes alternate totals and Asian
handicap lines; it is not the production default because historical downstream
results were mixed.

### Running without account authentication (current setup, 2026-09-06)

Private `my-team` ingest is **retired for now**. `FPL_BOT_FPL_COOKIE` is
commented out in `.env` and no access token is set, so `gaffe live ingest`
takes the public `entry/<id>/event/<gw>/picks` path. That endpoint 404s for an
upcoming gameweek, so ingest the LAST FINISHED gameweek and let the recommend
step roll it forward:

```bash
uv run gaffe live ingest --season-id 26 --team-id <ID> --gameweek <LAST_FINISHED_GW>
```

What the public path does NOT carry, and must therefore be maintained by hand
in `configs/live_state_overrides.yaml` before each recommend:

- **`free_transfers`** — not exposed anywhere publicly. Track it from
  `entry/<id>/history/`: each row's `event_transfers` is what was spent that
  week, and FPL grants +1 per GW capped at 5. Getting this wrong is the one
  error that silently changes the recommendation, because the MILP will take
  free transfers it does not have rather than a hit.
- **`bank_tenths`** — available as `entry_history.bank` on the picks payload,
  and the ingest does store it, so override it only if it looks stale.
- **per-player `cost_basis`** — the public ingest records CURRENT price as the
  purchase price, so sell values drift from truth as prices move. Harmless
  early; revisit if squad value and the sum of purchase prices diverge by more
  than ~£0.5m.

`chips_used` needs no override: it resolves from the snapshot's own
`chips_used_json` (GW1 `bboost` → `BB1`).

To restore authenticated ingest instead,
open browser developer tools on a logged-in `fantasy.premierleague.com` page,
open the Network panel, reload, and select a private `api/my-team/<id>` request.
Copy its `X-API-Authorization` request-header value into the gitignored `.env`:

```dotenv
FPL_BOT_FPL_ACCESS_TOKEN=Bearer <short-lived-token>
```

If the browser shows no authorization header, copy only the `access_token`
cookie value from DevTools' Application/Storage cookie view into the same env
variable. A full Cookie header in `FPL_BOT_FPL_COOKIE` remains a compatibility
fallback, but contains more credentials than the bot needs. Never commit or
share either value. Tokens expire; refresh the logged-in page and replace the
value when authentication fails. FPL entry IDs can also change between
seasons, so confirm `FPL_BOT_TEAM_ID` in `.env` each August. Without private
authentication, the bot uses the public historical picks endpoint and
`configs/live_state_overrides.yaml` can patch bank, free transfers, chips used,
and cost basis locally.

Output lands in `data/live/recommendations/season_26/gw_<GW>/recommendation.md`.

## After the gameweek closes — do this every week, not optionally

Wait until 09:00 UK time on the day after the final match, when FPL finalizes
the gameweek, then run:

```bash
uv run gaffe ingest fpl --season-id 26   # refresh finalized status + fixtures
uv run gaffe live retrospective --season-id 26 --team-id <YOUR_TEAM_ID> --gameweek <GW>
```

The retrospective prefers vaastav's per-match rows when available and otherwise
pulls FPL's official `event/<GW>/live` feed, so it works during the mirror's usual
publication lag. A finalized single-fixture gameweek also lands in
`fact_player_match` for rolling features; double-gameweek aggregates are used only
for scoring because they cannot be split safely by fixture. The command appends
`actuals.json` with realized points (auto-sub + auto-vice scored) and records which
source was used.

26/27 is the bot's first live season, so this log is the only unbiased
evidence about it that will ever exist — every other number in this repo
comes from a backtest of a season the model was developed against. Skipping a
week leaves a permanent hole in it.

## How differential is the bot, really? — measured 2026-09-06

`scripts/top_manager_holdings.py` reads the overall league (id 314) top N
entries' public picks and aggregates true ownership and captaincy among
managers actually winning this season. Against the GW4 recommendation:

| | |
|---|---|
| bot squad's total top-50 EO | **172%** |
| a top-50 manager's own squad totals | ~1500% |
| bot's captain (Ampadu) owned by | **0%** of the top 50 |
| Haaland: owned / captained in top 50 | **98% / 94%** (EO 192%) |
| Isak: owned in top 50 | **94%** |

The bot is not "differential" in the usual sense — it is close to disjoint
from the winning cohort, and it is not currently beating them. Being without
a 192%-EO asset is a per-gameweek rank bet of roughly twice that player's
score. Read this table before accepting a recommendation, and treat a squad
total under ~400% as a flag that the rho-EO term is dominating rather than
tilting.

Run it each week:

```bash
uv run python scripts/top_manager_holdings.py --gameweek <LAST_FINISHED_GW> --top 50
```

## What the bot does / doesn't know — read before trusting blindly

**Uses (pre-deadline, legitimate):**
- Bookmaker odds (The Odds API) for goal/CS probability — the bot's single
  largest edge. Without odds the bot LOSES to buy-and-hold (measured: −70).
- Rolling player xG/xA (Understat), team form, market λ.
- FPL `news` return-dates ("Expected back DD MMM") — pre-empts transfers
  for returning players. Explicit pre-return availability now updates the
  minutes distribution before goals/CS/bonus simulation; final-xPts
  attenuation remains only as a fallback for cached predictions.
- status_code (i/n/s/u → drop, d → attenuate by chance_of_playing).
- Top-10k EO (LiveFPL) for the template-differential ρ term. LiveFPL rows are
  keyed on FPL's own element id (its hidden first column), NOT on web_name.
  Name keying silently mis-assigned every duplicated surname — 2026-27 has
  fifteen, and `Palmer` resolved to Alex Palmer (a backup keeper) instead of
  Cole Palmer, so the most-templated midfielder in the game read as a free
  differential and the MILP bought and captained him. Fixed 2026-09-06; the
  parse output's `resolved_by_name_fallback` count should stay at 0.

**Does NOT know (override manually if you have better info):**
- Last-minute lineup leaks / press-conference nuance beyond the FPL `news`
  field. This is the main gap vs a skilled human.
- Manager rotation intent (who'll be rested).
- Set-piece role changes not yet in the yaml.

**Honest expectation (per the 25/26 backtest):** the bot performs at roughly
top-150-300k level — comparable to a good-but-not-elite human, fully
automated. It does NOT reach top-20k; its edge is the bookmaker's edge, and
the market is efficient. Treat the recommendation as a strong default to
adjust with your own team-news read, not gospel.

## Reading the "Season chip plan" block

The scheduler plans all eight slots from the fixture calendar alone and does
not know which chips have been spent; the MILP does, and fixes a spent slot's
activation to zero. A spent slot is now printed struck through and marked
ALREADY PLAYED — treat any unmarked line as the live plan.

Two things that block on the calendar rather than on the model:

- **TC and BB want a double gameweek, and DGWs do not exist yet.** They are
  created later by postponements, so early in a season the first half contains
  none at all. Both slots therefore fall back to "latest non-colliding GW in
  the half" — which reads as TC1/BB1 parked at GW17-19. That is deferral, not
  a recommendation: as soon as a real DGW enters the calendar the scheduler
  claims it. Do not read a parked chip as "the bot wants to play it there".
- TC previously used a different fallback — the argmax of per-GW max xPts —
  which was doubly wrong, because the prediction table only covers the current
  6-week horizon and early-season xPts is inflated by thin rolling features. It
  put TC1 on GW4 of 2026-27 with no DGW anywhere in GW1-19, i.e. it would have
  burned the chip immediately. Fixed 2026-09-06.

## Player props are NOT ready to switch on — measured 2026-09-06

`player_prop_market_weight` must stay 0.0 until the market rate is normalized.
The de-vig in `derive/player_props.py` shaves a flat half-margin
(`DEFAULT_ANYTIME_MARGIN = 0.07`), which is right for a two-way yes/no market
and far too small for a full anytime-scorer player list.

Measured on 2026-27 GW2, summing de-vigged `P(scores >= 1)` per fixture:

| quantity | value |
|---|---|
| mean sum of P(scores) per fixture | **6.57** |
| a fair book should sum to (expected distinct scorers) | ~2.3-2.8 |
| implied residual overround | **~2.5x** |
| GW2: market expected goals vs actual | 55.9 vs **29** |
| GW1: market expected goals vs actual | 39.6 vs **21** |

So blending the market rate in at any positive weight inflates goal
expectations across the board by roughly 2.5x. Fix the normalization first —
scale each fixture's player probabilities so they sum to that fixture's own
market expected goals, which `fact_market_xg` already carries — then re-measure
before touching the weight.

For contrast, the model's own rate errs the other way: it expected 3.9 goals
where 29 happened. That was the Understat blindness (below), and the shadow
log shows the repair landing — distinct model rates per gameweek went 76/217
(GW1) and 99/567 (GW2) to **296/329** (GW4), with the top rate rising from
0.09 to 0.77 goals/90. A league whose best striker scores 0.09 per 90 was the
symptom; GW4 is the first gameweek where the attacking model is discriminating
between players at all.

## Phase 8 player props — inert on purpose

`gaffe ingest oddsapi-props` is in the weekly cycle, but
`player_prop_market_weight` is **0.0**, so anytime-goalscorer odds currently
change nothing about the recommendation. They are recorded, inverted to a
per-90 goal rate, and logged beside the model's own rate in
`data/live/recommendations/season_26/gw_<GW>/player_prop_shadow.parquet`.

That is deliberate. The feature cannot be backtested — there is no free
historical player-prop data — so the only honest validation is live, and the
project's own hard-won rule is that a model change earns its place by
improving *accuracy*, not by sounding right (see the FDR and FWD-calibration
reversals in the README). Accumulate several gameweeks of shadow logs, compare
both rates against actual goals (per-position bias/MAE, and especially FWD and
top-decile calibration), and only then raise the weight via
`FPL_BOT_PLAYER_PROP_MARKET_WEIGHT`. Start around 0.3 if it wins.

One caveat when you read those logs: in GW1-3 the minutes model is cold
(measured: `e_minutes ≈ 12` for a nailed starter at GW1), which both distorts
the market rate and makes per-90 rates incomparable. Compare at fixture level
(`rate × e_minutes / 90`) and discount the first three gameweeks — see the
cold-start section in the design doc.

Two things to check on the first genuinely priced pull (~2 days before the GW1
deadline — no bookmaker has priced 26/27 yet, so nothing has parsed a real
payload):
- `skipped_unresolved_player` in the parse output, plus the
  `.unresolved.json` sidecar. A block of misses means a book changed its
  naming, not that those players are unpriced.
- How many players per fixture are actually priced. Unknown so far.

## Quota / cost notes

- The Odds API: 500 requests/month free. Weekly pull = 3 requests
  (h2h + spreads + totals × 1 region), plus ~1 per fixture for player props
  once books price them (~10/GW). Call it ~55/month — still comfortable.
  Listing events and unpriced events both cost 0.
- FPL API: free. Exact current-squad ingest needs your own short-lived access
  token, not a paid API key.
- LiveFPL + FFS + football-data: free, no key.
- Check `data/raw/oddsapi/<date>/*.meta.json` for remaining quota.

## Troubleshooting

- **"No predictions for GW N"**: the prediction cache doesn't cover GW N.
  The live recommend falls back to the predict-only path automatically; if
  it errors, confirm `ingest fpl` ran (fixtures must exist in dim_fixture).
- **"No user-team snapshot"**: run `gaffe live ingest` first; for an
  upcoming GW it falls back to the latest snapshot at-or-before that GW.
- **market_xg empty for the season**: run steps 2 + 5 (ingest oddsapi →
  derive market-xg). Without it the bot loses its main edge.
- **Recommendation looks stale / ignores a new injury**: confirm
  `ingest fpl` ran today (status_code + news are snapshot at ingest time).
