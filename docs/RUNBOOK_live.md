# Live Weekly Runbook (26/27 season)

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
   everything through 25/26 (`19,20,21,22,23,24,25` — the current default).
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

6. **Prediction caches: nothing to do at rollover.** The cache filename
   embeds the test season *and* the train-season list
   (`season_26_train_19_20_21_22_23_24_25.parquet`, see
   `live/recommend.py`), so a new season or a changed `train_seasons`
   writes a new key and cannot serve last season's predictions. Do **not**
   blanket-delete `data/cache/xpts_predictions/` — it holds the historical
   backtest baselines (including the kept `.pre_teamform` /
   `.bps_legacy_current_odds` reference variants) at ~12 min/fold to rebuild.

   The real staleness risk is different: changing *model or feature code*
   while the season/train key stays the same silently reuses the old
   predictions. After such a change, delete only the affected key(s):
   ```bash
   rm -f data/cache/xpts_predictions/season_26_train_19_20_21_22_23_24_25.parquet
   ```

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

**Measured on a real GW1 dry run (2026-07-26, before any odds existed):** the
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
    --train-seasons 19,20,21,22,23,24,25
```

The default uses the cross-fold-validated conservative fitter. For shadow
comparison only, `--fit-mode all` also consumes alternate totals and Asian
handicap lines; it is not the production default because historical downstream
results were mixed.

For exact current bank, free transfers, purchase prices, and sell prices,
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
uv run gaffe ingest fpl --season-id 26   # actuals land in bootstrap-static
uv run gaffe live retrospective --season-id 26 --team-id <YOUR_TEAM_ID> --gameweek <GW>
```
Appends `actuals.json` with realized points (auto-sub + auto-vice scored).

26/27 is the bot's first live season, so this log is the only unbiased
evidence about it that will ever exist — every other number in this repo
comes from a backtest of a season the model was developed against. Skipping a
week leaves a permanent hole in it.

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
- Top-10k EO (LiveFPL) for the template-differential ρ term.

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
