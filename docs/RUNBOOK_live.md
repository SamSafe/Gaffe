# Live Weekly Runbook (26/27 season)

How to run the bot for a real gameweek. Follow top-to-bottom each week,
ideally ~3-6 hours before the deadline (after team news + once odds are
posted, before lineups leak).

## One-time, at season start — status for 2026-27: DONE (2026-07-26)

Kept as the checklist to repeat each August. Everything below is already
done for 26/27 unless marked otherwise.

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
   25/26 — harmless but inactive).

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

## First three gameweeks — expect the bot at its weakest

GW1-3 is the one stretch where the model is structurally blind, so weight
your own judgement much more heavily:

- No 26/27 played GWs means no rolling xG/xA or team form; predictions lean
  on the cold-start prior (25/26 last-5-GW actuals) plus market λ.
- Promoted clubs (26/27: COV, HUL, IPS) have **no** PL history at all, and
  neither do incoming transfers from abroad. They are near-invisible to the
  model regardless of the prior — not "predicted to be bad", simply unseen.
- Odds are the main live signal that *does* work at GW1, so getting the
  odds pull in (step 2 of the weekly cycle) matters more than usual.

Treat the GW1 output as a budget-allocation sketch, not a squad.

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

For exact current bank, free transfers, purchase prices, and sell prices,
set `FPL_BOT_FPL_COOKIE` to the full logged-in `fantasy.premierleague.com`
Cookie header before `live ingest`. Without it, the bot uses the public
historical picks endpoint and `configs/live_state_overrides.yaml` can patch
bank, free transfers, chips used, and cost basis locally.

Output lands in `data/live/recommendations/season_26/gw_<GW>/recommendation.md`.

## After the gameweek closes — do this every week, not optionally

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

## Quota / cost notes

- The Odds API: 500 requests/month free. Weekly pull = 3 requests
  (h2h + spreads + totals × 1 region). ~12/month — comfortable.
- FPL API: free. Exact current-squad ingest needs your own logged-in cookie,
  not a paid API key.
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
