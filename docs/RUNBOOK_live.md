# Live Weekly Runbook (26/27 season)

How to run the bot for a real gameweek. Follow top-to-bottom each week,
ideally ~3-6 hours before the deadline (after team news + once odds are
posted, before lineups leak).

## One-time, at season start (August 2026)

1. **Confirm DB is migrated:**
   ```bash
   uv run alembic upgrade head
   ```

2. **Ingest the new-season FPL bootstrap** (players, teams, fixtures, prices):
   ```bash
   uv run gaffe ingest fpl
   ```

3. **Decide `train_seasons`.** Until 26/27 has ~5+ played GWs, train on
   everything through 25/26:
   ```
   --train-seasons 19,20,21,22,23,24,25
   ```
   Once 26/27 has enough played GWs (≈GW6+), append 26. The cold-start
   GW1-3 prior automatically uses the most-recent NON-test season's
   last-5-GW actuals (see `eval/milp_backtest._build...prior`).

4. **Populate set-piece takers** for 26/27 in `configs/set_piece_takers.yaml`
   under a `26:` block (penalty / direct_fk / corner per team). Until this
   is filled, `is_corner_taker` / `is_fk_taker` features are 0 for 26/27 —
   harmless but inactive.

5. **Clean stale prediction caches** when train_seasons changes:
   ```bash
   rm -f data/cache/xpts_predictions/*.parquet
   rm -f data/cache/xpts_raw_samples/*.parquet
   ```
   (They're rebuilt on next backtest/recommend; stale caches silently serve
   old predictions.)

## Every gameweek (the weekly cycle)

Run in this order. Each step is idempotent.

```bash
# 1. Refresh FPL state — prices, status_code, news, your squad.
uv run gaffe ingest fpl
uv run gaffe live ingest --team-id <YOUR_TEAM_ID> --gameweek <GW>

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

# 6. Generate the recommendation.
uv run gaffe live recommend \
    --team-id <YOUR_TEAM_ID> --gameweek <GW> \
    --train-seasons 19,20,21,22,23,24,25,26
```

For exact current bank, free transfers, purchase prices, and sell prices,
set `FPL_BOT_FPL_COOKIE` to the full logged-in `fantasy.premierleague.com`
Cookie header before `live ingest`. Without it, the bot uses the public
historical picks endpoint and `configs/live_state_overrides.yaml` can patch
bank, free transfers, chips used, and cost basis locally.

Output lands in `data/live/recommendations/season_26/gw_<GW>/recommendation.md`.

## After the gameweek closes (optional, for tracking)

```bash
uv run gaffe live retrospective --team-id <YOUR_TEAM_ID> --gameweek <GW>
```
Appends `actuals.json` with realized points (auto-sub + auto-vice scored).

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
