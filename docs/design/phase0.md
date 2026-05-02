# Phase 0 — Design Document

Status: **approved (rounds 1–2 review complete).** Ready to enter Phase 1. No production code is written yet. Pseudocode, SQL DDL sketches, and mathematical notation only.

---

## 1. Objective and success criteria

### 1.1 The split objective

The user has clarified: *prediction* is calibrated to expected points; *optimization* targets rank ("placement"). These are different objectives and the architecture must reflect that.

- **Layer 1 (prediction)** is a forecasting problem. Output the marginal distribution of FPL points per (player, gameweek), and a sampler over joint trajectories. Train and evaluate with proper scoring rules: log loss, Brier, calibration. Models are unbiased estimators of E[points] and P(points ∈ bucket); they do not "know" they will be used for rank-chasing.

- **Layer 2 (optimization)** is a portfolio-selection problem under uncertainty. Maximizing E[season points] is the wrong objective if the goal is rank, because:
  - Rank movement depends on **score relative to the field**, not absolute score.
  - Late in the season, when behind a target threshold (e.g., top-10k), the rank-utility is *variance-seeking*: a gambler who needs +30 points should prefer (50% chance of +60, 50% chance of 0) over (100% chance of +20). Mean-variance and CVaR objectives, as listed in the Phase 4 spec, point in the *opposite* direction (risk-averse) and would actively harm rank.

### 1.2 Proposed optimization objective

I propose a two-stage formulation:

**Stage A — within-gameweek, marginal rank-progress (default for early/mid-season):**

$$
\max \; \mathbb{E}\!\left[ \sum_{w \in W} \sum_{p \in P} \big(\text{mult}_{p,w} - \text{EO}_{p,w}\big) \cdot \text{pts}_{p,w} \right] - 4 \sum_{w} \text{ht}_w + V_T(s_H)
$$

where $\text{EO}_{p,w}$ is the *effective ownership* of player $p$ in gameweek $w$ (the field's expected captaincy- and chip-weighted ownership share), $\text{mult}_{p,w}$ is your own multiplier (0 if benched, 1 if XI, 2 if captain, 3 if triple-captained), and $V_T$ is the terminal value at the horizon. $(\text{mult}_{p,w} - \text{EO}_{p,w}) \cdot \text{pts}_{p,w}$ is the marginal points-vs-field for player $p$. This is a linear approximation that maximizes expected rank progress at the margin and is solver-friendly.

**Stage B — late-season threshold-chasing (optional, switched on when distance to target rank exceeds threshold):**

$$
\max \; \mathbb{P}\!\left( \text{season\_total} \geq \tau \right)
$$

solved via scenario sampling and a chance-constrained reformulation. Linearizable as $\max \mathbb{E}[\mathbf{1}\{\text{season\_total} \geq \tau\}]$ over the Layer 1 scenario set; binary indicators and big-M's are added to the MILP.

**Why two stages and not a single rank-utility:** the full $\mathbb{E}[\text{utility}(\text{rank})]$ is non-concave in scores and intractable directly. Stage A is the standard linear approximation used in the FPL analytics community ("EO-adjusted xPts") and is what tools like LiveFPL implicitly optimize. Stage B is the regime-switch refinement that buys you variance-seeking behavior when behind. I am open to dropping Stage B if you want to keep the optimizer simpler in the early phases — it can be added in Phase 4.

**Resolved in round 1:** EO-adjusted expected points is the default; CVaR-on-the-upside is not pursued.

**Risk-preference parameter $\rho$.** The EO term is scalable: $(\text{mult}_{p,w} - \rho \cdot \text{EO}_{p,w}) \cdot \text{pts}_{p,w}$, with $\rho \in [0, 1.5]$.
- $\rho = 0$: pure E[points] — template-conforming, no rank awareness.
- $\rho = 1$: pure rank-marginal — points-vs-field at the margin.
- $\rho > 1$: extra differential-seeking (anti-template).

Per the user's "good portion of risk per choice" directive, **default $\rho = 1.0$ with a backtest sweep over $\{0.7, 1.0, 1.2\}$** to pick the season-rank optimum. This is the principled lever for risk-vs-template trade-off; it's tunable per gameweek if late-season threshold play warrants escalation, but a single season-wide value is fine for v1.

**Implication for the data plan:** the EO term means we need ownership data per player per gameweek, captured *before kickoff*, point-in-time. FPL API exposes this in `bootstrap-static` (`selected_by_percent`) and per-element history. We also need top-10k ownership, which is *not* in the official API. Workarounds:
- Approximate top-10k EO with overall EO weighted by captain percentage.
- Scrape from third-party trackers (LiveFPL, FPLStatistics) — adds a fragile dependency. Recommend deferring until Phase 4 and starting with overall-EO.

### 1.3 Success criteria for the system

- Prediction: each component model beats a rolling-average baseline on log loss / Brier, with calibration deviation < 0.05 in each decile, on out-of-sample seasons.
- Optimization: full backtest across 2019/20–2023/24 finishes inside top-10k pace (median rank ≤ 10k) in at least 3 of 5 seasons, with the stochastic variant beating deterministic by ≥ 30 season points on average.
- These are stretch targets, not contractual; revisit after Phase 6 results.

---

## 2. Data sources and ingestion plan

### 2.1 Sources

| Source | Content | Cost | Cadence | Notes |
|---|---|---|---|---|
| FPL official API (`fantasy.premierleague.com/api/`) | Players, teams, fixtures, prices, ownership, gameweek live scores | Free, no auth | Hourly (prices), per-GW (results) | Endpoints: `bootstrap-static`, `fixtures`, `event/{id}/live`, `element-summary/{id}`, `entry/{team_id}/event/{gw}/picks`. Unofficial but stable. |
| vaastav/Fantasy-Premier-League (GitHub) | Historical FPL snapshots, GW-by-GW, 2016/17 onward | Free | Backfill once, weekly thereafter | Used for backtest training set. Cleaned, opinionated structure. |
| Understat | Shot-level xG, xA, key passes; multi-league | Free, scrape | Weekly post-GW | Best xG source for backtests. Has API-like JSON endpoints embedded in HTML. |
| fbref (StatsBomb-derived) | Detailed match/player stats — tackles, interceptions, blocks, recoveries, dribbles, big chances created/missed, errors, passes completed; position role indicators | Free, scrape | Weekly post-GW | Required for full BPS event simulation (§4.5–§4.6). EPL coverage from 2017/18 onward. Rate-limit politely. |
| football-data.co.uk | Historical bookmaker odds (1X2, totals, BTTS, AH) for EPL since 1993 | Free CSV | Backfill once, weekly | The backtest odds source. No clean-sheet market directly. |
| the-odds-api.com (free tier) | Live bookmaker odds (1X2, totals, spreads) | Free, 500 req/mo | Weekly pre-GW | Live odds for current season. 1 GW × 1 fetch covers ~10 fixtures in 1 request. ~38 GW × 1 ≈ 38 calls/season — well under cap. |
| LiveFPL (livefpl.net) | Top-10k effective ownership and captaincy snapshots per GW | Free, scrape | Weekly pre-GW (T-1h ideal) | Required for the EO term in the rank-utility objective (§1.2). Fragile HTML scrape; cache raw snapshots and validate column shapes. Fallback: FPLStatistics. |

### 2.2 Clean-sheet derivation

No free source provides clean-sheet odds directly. Derive them:

1. From bookmaker 1X2 + totals (and BTTS where present), fit a **Dixon-Coles bivariate Poisson** model per fixture to recover $(\lambda_{\text{home}}, \lambda_{\text{away}})$ goal-rate parameters.
2. $\mathbb{P}(\text{home CS}) = \mathbb{P}(\text{away goals} = 0) = e^{-\lambda_{\text{away}}}$, similarly for away CS, with the Dixon-Coles low-score correction.
3. This gives a *bookmaker-implied* clean sheet probability that we can blend with our own model in the prediction layer.

This adds a small inversion step but is well-trodden in football-modeling literature (Dixon & Coles 1997; Karlis & Ntzoufras 2003).

### 2.3 Cadence and ingestion pipeline

Per gameweek:
- **Pre-GW (T-24h to T-1h):** fetch FPL bootstrap-static, fixtures, latest odds, latest news. Snapshot prices and ownership. This is the data the optimizer is allowed to use.
- **Post-GW:** fetch live scores, BPS breakdowns, Understat shot-level data, fbref match reports. This becomes training data for the next iteration.

**Two-layer ingestion (per source, mandatory):**

1. `fetch_raw_<source>` — pulls the upstream payload (HTTP / CSV / scrape) and writes to `data/raw/{source}/{YYYY-MM-DD}/{filename}`. Writes nothing to the database except an audit row in `ingest_audit` (URL, response code, content hash, retrieved_at). Idempotent: re-runs on the same day overwrite the same path with content-hash check.
2. `parse_raw_<source>` — reads from `data/raw/{source}/...`, parses to typed records, writes to staging/fact tables. Reads zero network. Re-runnable any time.

This split is non-negotiable for the four scrape sources (LiveFPL, FPLStatistics, Understat, fbref) where parser fragility is the primary failure mode: a parser bug should never force us to re-hit a third-party site. Raw payload retention is permanent (gitignored, but kept on disk).

```
data/raw/
├── fpl_api/2024-08-16/bootstrap-static.json
├── fbref/2024-08-16/match_3578123.html
├── livefpl/2024-08-16/eo-gw1.html
├── understat/2024-08-16/match_27845.html
└── footballdata/E0_2023-24.csv
```

### 2.4 Terms of service

- FPL API: unofficial but historically tolerated. No write actions in this design.
- Understat / fbref: scraping permitted with reasonable rate limits.
- football-data.co.uk: free for non-commercial use.
- the-odds-api: free tier explicitly sanctioned within request cap.

### 2.5 Scraping compliance policy

Applies to all scraped sources (LiveFPL, FPLStatistics, Understat, fbref).

**Mandatory before first fetch from a new source:**
- Read `robots.txt` for the host. If a path is disallowed, do not fetch it.
- Read the site's Terms of Service. Note any commercial-use restrictions in the source module's docstring; this project is non-commercial personal use.
- Make a single low-rate exploratory fetch to confirm structure; do not iterate against the site during exploration.

**Runtime conduct:**
- Identify with a clear, descriptive User-Agent: `fpl-bot/0.x (+contact-email-or-repo-url)`.
- Honor `Crawl-delay` from robots.txt and any HTTP 429 responses with exponential backoff.
- Cache raw HTML aggressively; never re-fetch unchanged content (content-hash check before write).
- Per-source request budget: ≤ 1 request/min sustained for HTML scrape sources. LiveFPL: once per gameweek (T-1h ideal). fbref/Understat: post-GW only, sequential, 2s minimum spacing.
- Never bypass paywalls, login walls, or anti-bot protections. If a source moves behind one of these, fall back; do not work around it.

**Failure handling and fallback chain:**
- If a source returns 4xx/5xx for > 3 consecutive attempts, log and **fall back** rather than retry-storm:
  - LiveFPL → FPLStatistics → `fpl_api_approx` (overall EO from FPL API + captain-pct heuristic).
  - fbref → mark fixture's detailed events as missing; BPS sim degrades to empirical residual for that fixture only.
  - Understat → fail loudly; we cannot proceed without xG.
- All ingested rows tag `source` and (where applicable) `provenance_url`; downstream code can filter by source quality tier.

**Audit log (`ingest_audit` table):** every fetch records URL, request timestamp, response code, content hash, byte size, parse status. Queryable so we can demonstrate compliance retrospectively. DDL added below in §3.2.

**Posture if a source's policy changes:** if upstream tightens its policy or starts returning anti-bot challenges, the default is to disable that source's `fetch_raw_*` job until reviewed. Parsing of already-cached raw data continues. The system never escalates to evade detection.

---

## 3. Database schema (PostgreSQL) with point-in-time correctness

### 3.1 Design principle

Every fact we ingest is recorded as an append-only event with two timestamps:
- `event_time`: when the fact was true in the world (e.g., when the price changed).
- `recorded_at`: when we observed and stored it.

A point-in-time query for "what was true as of gameweek N kickoff" filters `recorded_at <= gw_n_kickoff` and picks the latest event per natural key. This is bitemporal-lite — we don't carry full `valid_from`/`valid_to` ranges since FPL data doesn't need true bitemporal joins, just as-of-`recorded_at`.

**Why not snapshot tables per GW:** snapshotting throws away intra-GW state changes (price moves, news updates) which matter for live decisions. Append-only preserves everything; views provide the GW-snapshot abstraction.

**Leakage discipline:** *no downstream feature pipeline reads any table directly.* All reads go through the PIT query layer (`fpl_bot.db.pit`), which mandates an `as_of` parameter. The leakage test suite (Phase 1) confirms by introspection that no model/feature module imports raw tables.

**Cross-season identity (round-4 design fix).** FPL's per-season element `id` and per-season fixture `id` reset each season — Salah's element_id varied from 191 to 308 across six seasons. To support multi-season backfill, `dim_player.player_id` and `dim_fixture.fixture_id` store the **FPL `code` field** (stable across seasons; e.g., Salah=118748, fixture=2561895), not the per-season `id`. Two xref tables (`dim_player_season_xref`, `dim_fixture_season_xref`) translate per-season `id` → stable code at ingest time. All fact tables reference the stable code; no fact-table schema change required.

### 3.2 Core tables (DDL sketches)

```sql
-- Dimension tables (slowly-changing)
CREATE TABLE dim_team (
  team_id        SMALLINT PRIMARY KEY,
  short_name     TEXT NOT NULL,
  full_name      TEXT NOT NULL,
  season_id      SMALLINT NOT NULL,  -- composite PK in practice; promoted teams change
  promoted       BOOLEAN NOT NULL,
  PRIMARY KEY (team_id, season_id)
);

CREATE TABLE dim_player (
  player_id      INT PRIMARY KEY,         -- FPL `code` (stable across seasons), NOT per-season element id
  web_name       TEXT NOT NULL,
  first_name     TEXT,
  last_name      TEXT,
  understat_id   INT,                     -- nullable; backfilled by name match
  fbref_id       TEXT
);

CREATE TABLE dim_fixture (
  fixture_id     INT PRIMARY KEY,         -- FPL fixture `code` (stable across seasons)
  season_id      SMALLINT NOT NULL,
  gameweek       SMALLINT NOT NULL,
  kickoff_utc    TIMESTAMPTZ NOT NULL,
  home_team_id   SMALLINT NOT NULL,
  away_team_id   SMALLINT NOT NULL,
  finished       BOOLEAN NOT NULL DEFAULT FALSE
);

-- Translation tables: per-season FPL `id` ↔ stable global `code` (= dim_player.player_id / dim_fixture.fixture_id)
CREATE TABLE dim_player_season_xref (
  season_id       SMALLINT NOT NULL,
  fpl_element_id  INT NOT NULL,           -- the per-season `id` from bootstrap-static
  player_id       INT NOT NULL,           -- stable `code` (FK to dim_player.player_id, not enforced)
  PRIMARY KEY (season_id, fpl_element_id)
);
CREATE INDEX ix_player_xref_player ON dim_player_season_xref (player_id);

CREATE TABLE dim_fixture_season_xref (
  season_id           SMALLINT NOT NULL,
  fpl_fixture_id      INT NOT NULL,       -- the per-season `id` from fixtures endpoint
  fixture_id          INT NOT NULL,       -- stable `code` (FK to dim_fixture.fixture_id, not enforced)
  PRIMARY KEY (season_id, fpl_fixture_id)
);
CREATE INDEX ix_fixture_xref_fixture ON dim_fixture_season_xref (fixture_id);

-- Append-only event tables (the bitemporal-lite core)
CREATE TABLE fact_player_status (
  player_id      INT NOT NULL,
  season_id      SMALLINT NOT NULL,
  position_code  CHAR(3) NOT NULL,       -- GKP/DEF/MID/FWD per FPL
  team_id        SMALLINT NOT NULL,
  price_tenths   SMALLINT NOT NULL,      -- 50 = £5.0m
  status_code    CHAR(1) NOT NULL,       -- a/d/i/n/s/u
  news           TEXT,
  chance_of_playing_next_round  SMALLINT,
  selected_by_percent           NUMERIC(5,2),
  event_time     TIMESTAMPTZ NOT NULL,
  recorded_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
  PRIMARY KEY (player_id, recorded_at)
);
CREATE INDEX ix_player_status_pit ON fact_player_status (player_id, recorded_at DESC);

CREATE TABLE fact_match_result (
  fixture_id     INT NOT NULL,
  home_score     SMALLINT,
  away_score     SMALLINT,
  finished       BOOLEAN NOT NULL,
  recorded_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
  PRIMARY KEY (fixture_id, recorded_at)
);

CREATE TABLE fact_player_match (
  player_id      INT NOT NULL,
  fixture_id     INT NOT NULL,
  minutes        SMALLINT,
  goals          SMALLINT,
  assists        SMALLINT,
  clean_sheet    BOOLEAN,
  goals_conceded SMALLINT,
  saves          SMALLINT,
  yellow_cards   SMALLINT,
  red_cards      SMALLINT,
  bonus          SMALLINT,
  bps            SMALLINT,
  total_points   SMALLINT,
  recorded_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
  PRIMARY KEY (player_id, fixture_id, recorded_at)
);

CREATE TABLE fact_understat_shot (
  shot_id        TEXT PRIMARY KEY,        -- understat-provided
  fixture_id     INT,                     -- nullable; matched via fuzzy join
  player_id      INT,
  minute         SMALLINT,
  xg             NUMERIC(5,4),
  xa_for_assister NUMERIC(5,4),
  result         TEXT,
  situation      TEXT,
  shot_type      TEXT,
  body_part      TEXT,
  is_set_piece   BOOLEAN,
  recorded_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE fact_odds (
  fixture_id     INT NOT NULL,
  bookmaker      TEXT NOT NULL,
  market         TEXT NOT NULL,           -- '1X2', 'totals_2.5', 'btts', etc.
  selection      TEXT NOT NULL,           -- 'home', 'over', 'yes', etc.
  decimal_odds   NUMERIC(8,4) NOT NULL,
  event_time     TIMESTAMPTZ NOT NULL,    -- when the line was quoted
  recorded_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
  PRIMARY KEY (fixture_id, bookmaker, market, selection, event_time)
);

-- Derived: bookmaker-implied team xG per fixture (Dixon-Coles inversion)
CREATE TABLE fact_market_xg (
  fixture_id     INT NOT NULL,
  team_id        SMALLINT NOT NULL,
  lambda         NUMERIC(6,4) NOT NULL,
  cs_prob        NUMERIC(6,5) NOT NULL,
  source_recorded_at TIMESTAMPTZ NOT NULL,  -- pin to the odds snapshot it was derived from
  recorded_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
  PRIMARY KEY (fixture_id, team_id, source_recorded_at)
);

-- Effective ownership snapshots (multiple rank bands; primary use is top-10k for the optimizer EO term)
CREATE TABLE fact_eo_snapshot (
  player_id              INT NOT NULL,
  season_id              SMALLINT NOT NULL,
  gameweek               SMALLINT NOT NULL,
  rank_band              TEXT NOT NULL,            -- 'top10k' | 'top100k' | 'overall'
  ownership_pct          NUMERIC(6,3) NOT NULL,    -- % of band owning
  captaincy_pct          NUMERIC(6,3) NOT NULL,    -- % of band captaining
  effective_ownership    NUMERIC(6,3) NOT NULL,    -- ownership + captaincy + chip-weighted
  source                 TEXT NOT NULL,            -- 'livefpl' | 'fplstatistics' | 'fpl_api_approx'
  provenance_url         TEXT,                     -- canonical fetch URL for audit
  event_time             TIMESTAMPTZ NOT NULL,     -- snapshot time, must be < kickoff
  recorded_at            TIMESTAMPTZ NOT NULL DEFAULT now(),
  PRIMARY KEY (player_id, season_id, gameweek, rank_band, source, event_time)
);
CREATE INDEX ix_eo_snapshot_pit ON fact_eo_snapshot (player_id, season_id, gameweek, rank_band, recorded_at DESC);

-- Per-player per-match detailed event counts (from fbref) — feeds the BPS simulator
CREATE TABLE fact_player_match_event (
  player_id              INT NOT NULL,
  fixture_id             INT NOT NULL,
  passes_completed       SMALLINT,
  tackles_won            SMALLINT,
  interceptions          SMALLINT,
  blocks                 SMALLINT,
  recoveries             SMALLINT,
  dribbles_completed     SMALLINT,
  big_chances_created    SMALLINT,
  big_chances_missed     SMALLINT,
  penalties_won          SMALLINT,
  penalties_conceded     SMALLINT,
  penalties_saved        SMALLINT,
  penalties_missed       SMALLINT,
  errors_leading_to_goal SMALLINT,
  own_goals              SMALLINT,
  offsides               SMALLINT,
  was_fouled             SMALLINT,
  fouls_committed        SMALLINT,
  recorded_at            TIMESTAMPTZ NOT NULL DEFAULT now(),
  PRIMARY KEY (player_id, fixture_id, recorded_at)
);

-- BPS rule set, versioned by season (rules change between seasons; capture each year)
CREATE TABLE bps_rule_set (
  season_id       SMALLINT NOT NULL,
  event_code      TEXT NOT NULL,            -- e.g. 'mid_goal', 'tackle_won', 'pass_completed'
  position_filter CHAR(3),                  -- nullable; null = applies to all positions
  bps_value       SMALLINT NOT NULL,        -- can be negative
  cap_per_match   SMALLINT,                 -- nullable; some events are capped
  notes           TEXT,
  PRIMARY KEY (season_id, event_code, position_filter)
);

-- Penalty-taker dim with manual override support (see §4.6)
-- Derived layer auto-fills from rolling penalty-taken history;
-- manual layer takes precedence when override_set = TRUE.
CREATE TABLE dim_penalty_taker (
  team_id        SMALLINT NOT NULL,
  player_id      INT NOT NULL,
  rank_in_order  SMALLINT NOT NULL,         -- 1 = primary taker, 2 = secondary, etc.
  source         TEXT NOT NULL,             -- 'derived' | 'manual'
  override_set   BOOLEAN NOT NULL DEFAULT FALSE,
  valid_from     TIMESTAMPTZ NOT NULL,
  valid_to       TIMESTAMPTZ,               -- nullable = current
  recorded_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
  PRIMARY KEY (team_id, rank_in_order, valid_from, source)
);

-- Ingest audit log (per §2.5). Every fetch from any source writes one row here
-- regardless of whether parsing succeeds. Used for compliance demonstration and replay.
CREATE TABLE ingest_audit (
  audit_id        BIGSERIAL PRIMARY KEY,
  source          TEXT NOT NULL,            -- 'fpl_api' | 'livefpl' | 'fbref' | ...
  url             TEXT NOT NULL,
  request_ts      TIMESTAMPTZ NOT NULL,
  response_code   SMALLINT,
  byte_size       INT,
  content_hash    TEXT,                     -- sha256 of raw payload
  raw_path        TEXT,                     -- path under data/raw/ where stored
  parse_status    TEXT,                     -- 'pending' | 'ok' | 'failed' | 'skipped_unchanged'
  parse_error     TEXT,
  user_agent      TEXT,
  recorded_at     TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX ix_ingest_audit_source_ts ON ingest_audit (source, request_ts DESC);
```

### 3.3 Point-in-time query layer

A thin Python module (`fpl_bot.db.pit`) exposes one entry per logical query:

```python
def player_status_as_of(player_id: int, as_of: datetime) -> PlayerStatus: ...
def squad_market_as_of(season_id: int, as_of: datetime) -> pl.DataFrame: ...
def player_match_history(player_id: int, before: datetime) -> pl.DataFrame: ...
def market_xg_for_fixture(fixture_id: int, as_of: datetime) -> tuple[float, float]: ...
def eo_as_of(season_id: int, gw: int, rank_band: str, as_of: datetime) -> pl.DataFrame: ...
    # rank_band ∈ {'top10k','top100k','overall'}; falls back across sources if primary missing
def match_event_history(player_id: int, before: datetime) -> pl.DataFrame: ...        # fbref event counts
def bps_rules_for_season(season_id: int) -> pl.DataFrame: ...
def penalty_taker_as_of(team_id: int, as_of: datetime) -> pl.DataFrame: ...           # ordered list, manual override-aware
```

Every function takes an explicit `as_of` and translates to `WHERE recorded_at <= as_of` plus a window function picking the latest row per natural key. This is the *only* sanctioned read path.

### 3.4 Leakage tests (deliverable in Phase 1)

- **Static check**: AST-walk all modules under `fpl_bot.features` and `fpl_bot.models`, fail if any imports `sqlalchemy`, raw connection objects, or non-PIT modules.
- **Dynamic check**: synthetic dataset with sentinel rows whose `recorded_at` is after a target `as_of`. Run feature pipeline; assert no sentinel value appears in the output.
- **Round-trip check**: replay a real backtest GW, materialize all features, then re-ingest with one extra hour of "future" data. Features must be byte-identical.

---

## 4. Predictive components — features and leakage analysis

For each component I list: target, model class, feature groups, and a leakage row per group. Treat any feature as leaky until I can name the cutoff that defines it.

### 4.1 Minutes model

- **Target**: 3-class — {0, 1–59, 60+} minutes per (player, fixture).
- **Model**: LightGBM multi-class with monotonic constraints where defensible (e.g., `chance_of_playing_next_round` should be monotonic in P(60+)).
- **Why this is the bottleneck**: minutes drive the entire xPts cascade. A 5pp error here dominates downstream errors in goals/assists/CS.

| Feature group | Examples | Leakage cutoff | Notes |
|---|---|---|---|
| Recent minutes | rolling minutes over last 1/3/5/10 GWs | last completed GW < as_of | DGW handling: weight by fixture, not by week. |
| Days since last match | scalar | from `dim_fixture.kickoff_utc < as_of` | UEFA midweek games count. |
| Fixture congestion | matches in last 7d, 14d (PL + Europe + cup) | as above | Source: cup/Europe schedules; small auxiliary table. |
| Manager rotation rate | player's rotation freq under current manager when fixture-density ≥ k | from completed fixtures only | **Leakage trap**: if a manager changes mid-season, the prior data must be partitioned. Manager identity per team per date needed (small mapping table). |
| Price tier | current price decile within position | `fact_player_status` as_of | Price as proxy for "premium starter." |
| Injury / availability | `status_code`, `chance_of_playing_next_round`, `news` keywords | `fact_player_status` as_of | News text → keyword features (suspended, injured, fitness, illness). |
| Days into season | scalar | derived | Captures preseason rotation and end-of-season rest. |
| Position | from `fact_player_status` | as of | GKPs are nearly deterministic. |
| Team's UEFA midweek flag | bool | from cup schedule | Rotation predictor. |

**Calibration validation**: per-bucket calibration plot, log loss vs three baselines: (a) "always predict 60+ if started last GW," (b) rolling-mean predicted-prob, (c) a simple logistic on minutes-last-GW + status. Per-manager breakdown to surface rotation patterns the model misses.

### 4.2 Goals per 90 (conditional on starting)

- **Target**: goals per 90, conditional on minutes ≥ 1. Modeled as Negative Binomial GLM or LightGBM with Poisson loss; pick after calibration test.
- **Why conditional on starting**: decouples the minutes uncertainty (already modeled) from scoring rate; avoids double-counting.

| Feature group | Examples | Leakage cutoff |
|---|---|---|
| Player rolling xG/90 | last 5/10/20 PL appearances, exponentially-weighted | Understat shots with kickoff < as_of |
| Player rolling shots/90 | total, in-box, big-chances | as above |
| Opponent xGA/90 (defensive strength) | adjusted for opponent strength via Elo-style team rating | finished fixtures only |
| Home/away | bool | fixture metadata |
| Set-piece taker | bool: penalty/free-kick/corner taker | derived from Understat `situation` field |
| Position role mismatch | flag if FPL position ≠ actual playing role (from fbref position cluster) | fbref data as_of |
| Bookmaker team λ | derived from Dixon-Coles inversion | `fact_market_xg` as_of |
| Player share of team shots | rolling | as above |

**Position role mismatch — the alpha note**: FPL's `position_code` lags reality. A "MID" who plays as a striker (Salah-style false-9 cases) has a higher goals/90 prior than the MID baseline. I propose a manually-curated mismatch list in Phase 2.1 and a learned cluster (k-means on fbref touch heatmap embeddings) as a Phase 2.5 refinement. **Both must be PIT-safe**: don't use a player's *current* role to backfill their *historical* prior.

### 4.3 Assists per 90 (conditional on starting)

Symmetric to 4.2, with xA replacing xG. Two extra features:
- Teammate finishing quality (rolling team conversion rate above/below xG) — captures Salah/Mané-era inflation.
- Set-piece deliveries from this player.

### 4.4 Clean sheet model

Spec calls for "start by directly using bookmaker clean sheet odds." We don't have CS markets for free, so:

1. **Layer 0**: Dixon-Coles-derived $\mathbb{P}(\text{CS})$ from 1X2 + totals.
2. **Layer 1 (residual model)**: GBM predicting $\text{CS\_actual} - \mathbb{P}_{\text{market}}(\text{CS})$ from team-level features:
   - Rolling team xGA over various windows.
   - GK quality proxy (rolling saves above expected, if computable from Understat).
   - Defensive lineup change indicator (centre-back rotation).
   - Travel / congestion for the defending team.

If the residual model has no out-of-sample lift, drop it and use market only. **Honest expectation**: market odds in EPL are very efficient; residual lift is likely small (< 0.5 pp Brier improvement). I'd rather find that out and document it than assume otherwise.

### 4.5 Bonus points (full BPS simulation)

Per round-1 directive: full event-level simulation, no empirical-residual shortcut. Plan:

1. Maintain a versioned **BPS rule table** in `bps_rule_set` (rules change between seasons; capture each year and select via `season_id` at simulation time).
2. Each event type contributing to BPS is modeled as a per-player rate model conditional on minutes (see §4.6 for the inventory). Sample event counts per player per fixture per scenario.
3. Score each sampled fixture by applying BPS rules → integer BPS per player. Respect per-match caps (e.g., passes-completed cap) and conditional rules (e.g., position-specific goal weights) exactly.
4. Within each fixture, rank players by simulated BPS, apply FPL tie-break ladder, assign 3 / 2 / 1 bonus.
5. Calibrate against historical bonus assignments: Brier on "did this player receive bonus?" per position and per bonus tier; mean bias by player price tier.

**Why this is meaningfully harder than the simplified version we rejected:** ~10 additional component models (§4.6) at the per-player per-match level, each with its own feature pipeline and calibration check. The BPS rules also have caps and conditional rewards that must be respected exactly — an off-by-one in the cap logic silently distorts the bonus distribution.

**Data dependency:** fbref detailed match logs cover EPL from 2017/18 onward. Pre-2017/18 evaluation will degrade to empirical-residual mode for events fbref doesn't have; the 2019/20–2023/24 evaluation window is unaffected.

### 4.6 Underlying event models (BPS components and direct-points events)

Beyond minutes / goals / assists / clean sheet, the following per-player per-match event models feed both direct FPL points (cards, saves, penalty-related) and the BPS simulator (§4.5):

| Event | Model class | Position scope | Direct-pts impact | BPS impact |
|---|---|---|---|---|
| Yellow / red cards | Poisson with referee-strictness rolling | All | −1 / −3 | yes |
| Saves | Poisson on opponent shots × save-rate (rolling, EB-shrunk) | GK | 1 per 3 saves | yes |
| Tackles won | Poisson per 90, position-conditional rate | DEF/MID/FWD (decreasing) | none | yes |
| Interceptions | Poisson per 90 | DEF/MID heavy | none | yes |
| Recoveries | Poisson per 90 | All | none | yes |
| Blocks | Poisson per 90 | DEF heavy | none | yes |
| Dribbles completed | Poisson per 90 | MID/FWD | none | yes |
| Big chances created | Poisson per 90, correlated residual w/ xA model | MID/FWD | none | yes |
| Big chances missed | Poisson conditional on shots taken | MID/FWD | none | yes (negative) |
| Penalties scored | $\mathbb{P}(\text{pen awarded}) \times \mathbb{P}(\text{is taker}) \times 0.78$ scoring rate | Designated taker | yes (counts toward goals) | yes |
| Penalties missed/saved | as above × $(1 - 0.78)$ split into miss vs save | Taker / opposing GK | −2 missed; +5 GK saved | yes |
| Own goals | Empirical league-average rate × position prior | All | −2 | yes |
| Errors leading to goal | Rolling per 90, EB-shrunk to position prior | All | none | yes (negative) |
| Passes completed | Negative Binomial; reflected for capped BPS contribution | All | none | yes (capped) |

**Common feature template** (mirrors §4.2): rolling rates over 1/3/5/10 GW windows with exponential weights; opponent-strength adjustment; home/away; minutes-conditioned; position fixed effect. PIT-routed via `match_event_history`.

**Empirical-Bayes shrinkage** for rare events (penalties, own goals, errors): per-player rate $\hat\lambda_p = \frac{n_p + \alpha}{m_p + \beta}$ where $(\alpha, \beta)$ come from the position-level prior fit on all historical data ≤ as_of. Critical to avoid one-fluke-event blowing up a player's prior.

**Cross-event correlation (within-fixture):** tackles, interceptions, recoveries, and blocks are highly positively correlated within-player. Sample them jointly via a shared "defensive-engagement" latent draw per player per fixture. Independent sampling produces unrealistically smooth bonus distributions.

**Penalty-taker designation:** an additional small derived table `dim_penalty_taker(team_id, player_id, valid_from, valid_to)` maintained from rolling penalty-taken history with manual override support. Without this, the penalty-points term is misspecified.

### 4.7 Joint xPts distribution

Components are conditionally independent given fixture and lineup state (a strong but standard assumption). Monte Carlo:
1. For each player p, fixture f, sample minutes class.
2. If 60+, sample goals, assists, CS-flag (CS via team-level joint sample), saves, cards.
3. Simulate BPS from event sample → bonus.
4. Sum to FPL points per the rule table.

Cross-player correlation within a fixture is preserved by sharing the sampled `(home_score, away_score)` between teammates (for CS, conceded) and via shared lineup random draws (e.g., set-piece taker availability flips both penalties and direct free-kicks).

Cross-week correlation in a player's *own* form is preserved by sampling a per-player latent "form factor" per scenario and applying it as a multiplicative shock to that player's xG/xA across the horizon. **This is the joint-trajectory requirement from the user spec** — we are not sampling each week independently. Implementation: a low-dimensional Gaussian copula over (player × week) form-shocks, fit on residuals from the per-90 models.

---

## 5. MILP formulation (rolling horizon)

### 5.1 Sets and parameters

- $P$: players in scope. $T$: teams. $W = \{1, \ldots, H\}$: gameweeks in horizon (H = 6–8). $S$: scenarios (deterministic uses a single mean trajectory; stochastic samples 100–500).
- Position partitions $P_{\text{GKP}}, P_{\text{DEF}}, P_{\text{MID}}, P_{\text{FWD}}$.
- $\text{price}_{p,w}^{\text{buy}}, \text{price}_{p,w}^{\text{sell}}$: predicted buy/sell prices (FPL has a buy/sell spread on price changes).
- $\text{pts}_{p,w,s}$: scenario sample of player points.
- $\text{EO}_{p,w}$: effective ownership prior (point-in-time, latest available before GW $w$).
- $\text{ft}_0, \text{bank}_0, \text{squad}_0$: initial state.

### 5.2 Decision variables (per scenario where stochastic, indexed by $s$ on points-bearing terms only — first-stage decisions are scenario-independent; bench order and captain may be scenario-conditional in a two-stage formulation, but for tractability the deterministic-first-stage version uses a single set of decisions)

| Variable | Domain | Meaning |
|---|---|---|
| $x_{p,w}$ | $\{0,1\}$ | $p$ in 15-man squad in GW $w$ |
| $y_{p,w}$ | $\{0,1\}$ | $p$ in starting XI in GW $w$ |
| $c_{p,w}$ | $\{0,1\}$ | $p$ captained |
| $v_{p,w}$ | $\{0,1\}$ | $p$ vice-captained |
| $b^{\text{in}}_{p,w}$ | $\{0,1\}$ | transferred in at start of GW $w$ |
| $b^{\text{out}}_{p,w}$ | $\{0,1\}$ | transferred out at start of GW $w$ |
| $\text{ft}_w$ | $\{0,1,\ldots,5\}$ | free transfers banked entering GW $w$ |
| $\text{ht}_w$ | $\mathbb{Z}_{\geq 0}$ | hits taken in GW $w$ |
| $\text{bank}_w$ | $\mathbb{R}_{\geq 0}$ | money in bank entering GW $w$ |
| $z^{\text{WC}}_w, z^{\text{FH}}_w, z^{\text{BB}}_w, z^{\text{TC}}_w$ | $\{0,1\}$ | chip activations |

### 5.3 Constraints

**Squad shape:**
$$
\sum_{p \in P} x_{p,w} = 15, \quad \sum_{p \in P_{\text{GKP}}} x_{p,w} = 2, \quad \sum_{p \in P_{\text{DEF}}} x_{p,w} = 5, \quad \sum_{p \in P_{\text{MID}}} x_{p,w} = 5, \quad \sum_{p \in P_{\text{FWD}}} x_{p,w} = 3
$$

**Per-team cap (≤ 3):**
$$
\sum_{p \in P_t} x_{p,w} \leq 3 \quad \forall t, w
$$

**Starting XI and formation:**
$$
\sum_{p \in P} y_{p,w} = 11, \quad y_{p,w} \leq x_{p,w}, \quad \sum_{p \in P_{\text{GKP}}} y_{p,w} = 1
$$
$$
\sum_{p \in P_{\text{DEF}}} y_{p,w} \geq 3, \quad \sum_{p \in P_{\text{MID}}} y_{p,w} \geq 2, \quad \sum_{p \in P_{\text{FWD}}} y_{p,w} \geq 1
$$

**Captain / vice:**
$$
\sum_p c_{p,w} = 1, \quad \sum_p v_{p,w} = 1, \quad c_{p,w} + v_{p,w} \leq 1, \quad c_{p,w} \leq y_{p,w}, \quad v_{p,w} \leq y_{p,w}
$$

**Transfer balance:**
$$
x_{p,w} - x_{p,w-1} = b^{\text{in}}_{p,w} - b^{\text{out}}_{p,w}, \quad b^{\text{in}}_{p,w} + b^{\text{out}}_{p,w} \leq 1
$$

**Free transfer evolution (linearized, ignoring chips):**
$$
\text{ft}_w = \min\!\left( 5, \; \text{ft}_{w-1} + 1 - \sum_p b^{\text{out}}_{p,w-1} \right)
$$
encoded with auxiliary integer variables and the cap-5 ceiling. Hits:
$$
\text{ht}_w \geq \sum_p b^{\text{out}}_{p,w} - \text{ft}_w, \quad \text{ht}_w \geq 0
$$

**Budget evolution:**
$$
\text{bank}_{w} = \text{bank}_{w-1} + \sum_p \text{price}^{\text{sell}}_{p,w} \cdot b^{\text{out}}_{p,w} - \sum_p \text{price}^{\text{buy}}_{p,w} \cdot b^{\text{in}}_{p,w}
$$
$$
\sum_p \text{price}^{\text{buy}}_{p,w} \cdot x_{p,w} + \text{bank}_w = \text{budget}_0 \;+\; \text{accumulated capital gains}
$$

**Chip semantics:**
- Wildcard ($z^{\text{WC}}_w = 1$): suppresses the hit term in GW $w$ (i.e., $\text{ht}_w \cdot (1 - z^{\text{WC}}_w)$ enters the objective; standard big-M).
- Free Hit ($z^{\text{FH}}_w = 1$): squad in GW $w$ may differ freely from $w-1$ and $w+1$ without consuming transfers; revert to $w-1$ squad in $w+1$.
- Bench Boost ($z^{\text{BB}}_w = 1$): all 15 score, i.e., bench multiplier becomes 1.
- Triple Captain ($z^{\text{TC}}_w = 1$): captain multiplier becomes 3.
- Each chip used at most once per season (split into halves for second-WC and second-FH per current FPL rules — verify in Phase 3).
- At most one chip per GW: $z^{\text{WC}}_w + z^{\text{FH}}_w + z^{\text{BB}}_w + z^{\text{TC}}_w \leq 1$.

**Multiplier (for the objective):**
$$
\text{mult}_{p,w} = y_{p,w} + c_{p,w} + 2 \cdot c_{p,w} \cdot z^{\text{TC}}_w + (x_{p,w} - y_{p,w}) \cdot z^{\text{BB}}_w
$$
(captain doubles to 2× normally, 3× with TC; bench scores when BB.) Linearize the bilinear $c_{p,w} \cdot z^{\text{TC}}_w$ and $(x-y) \cdot z^{\text{BB}}_w$ with auxiliary binaries and big-M.

### 5.4 Objective

Deterministic (Phase 3):
$$
\max \;\; \sum_{w \in W} \sum_{p \in P} \big(\text{mult}_{p,w} - \text{EO}_{p,w}\big) \cdot \mathbb{E}[\text{pts}_{p,w}] \;-\; 4 \sum_w \text{ht}_w \cdot (1 - z^{\text{WC}}_w) \;+\; V_T(s_H)
$$

Stochastic (Phase 4): replace the inner expectation with a sample average over scenarios $s \in S$, applied via SAA. First-stage decisions ($x, y, c, v, b^{\text{in}}, b^{\text{out}}$ for the *current* GW) are non-anticipative; subsequent-week decisions are scenario-conditional but typically aggregated (here-and-now vs wait-and-see split). I'll specify the two-stage decomposition in Phase 4's design supplement.

### 5.5 Terminal value $V_T(s_H)$

Spec requires this and forbids fixed-horizon without it. Definition:
$$
V_T(s_H) = \alpha \sum_{p} x_{p,H} \cdot \widehat{\text{pts}}_{p, H+1:H+5} \;+\; \beta \sum_p \text{price}^{\text{sell}}_{p, H} \cdot x_{p,H}
$$

- $\widehat{\text{pts}}_{p, H+1:H+5}$: expected points over the next 5 GWs beyond the horizon, computed with a simplified component-mean forecast (no per-GW lineup detail; just a fixture-difficulty-adjusted xPts/90 × expected starts).
- The squad-value term $\beta \sum_p \text{price}^{\text{sell}} \cdot x_{p,H}$ captures the optionality of holding rising-price assets. $\alpha, \beta$ tuned via backtest.

**Why this matters**: without a terminal value, the optimizer happily strips the team to maximize current-window points, leaving you with a depleted squad at horizon end. Standard rolling-horizon control fix.

---

## 6. Chip-timing sub-problem

Solved as a separate stochastic dynamic program, weekly-resolved.

- **State**: GW index, squad fingerprint (compressed: top-3 captaincy options, average XI xPts, fixture run difficulty), chips remaining.
- **Action**: which chip to play this GW, or none.
- **Transition**: deterministic on chip count, stochastic on points via Layer 1 scenarios.
- **Reward**: chip-marginal points = E[points with chip] − E[points without chip], integrated against EO for rank-utility.
- **Horizon**: full season.

**Why DP and not embed into the rolling MILP**: the rolling MILP only sees 6–8 GWs. Chips have a season-scale optimal-timing structure (e.g., BB during a DGW that's 20+ GWs away). The DP explores the season; the rolling MILP gets the DP's value-of-chip-this-week as a soft constraint / shadow price.

**Coupling**: each GW, the DP outputs $\hat{V}^{\text{chip}}_w$ for each chip. The rolling MILP uses this as a *bonus* added to the objective when the chip variable is set, plus a *penalty* equal to $(\hat{V}^{\text{chip}}_w - \hat{V}^{\text{chip}}_{w'})$ for $w' = \arg\max \hat{V}^{\text{chip}}_{w}$ — a soft "play it the right week" incentive. Re-solve DP weekly as forecasts update.

---

## 7. Evaluation methodology

### 7.1 Splits

- **Walk-forward**: train on seasons $\{S_1, \ldots, S_{k-1}\}$ + GWs $1..(w-1)$ of $S_k$; predict GW $w$ of $S_k$; advance.
- **Held-out dev season**: 2019/20 reserved for research-time iteration. Final reported numbers come from 2020/21–2023/24 walk-forward.
- **Cold-start subset**: report metrics restricted to (a) promoted-team players, (b) new signings, (c) GW1 of each season — these are the hardest cases and dilute headline metrics if not separated.

### 7.2 Calibration

- Reliability diagrams per component, decile-binned, with bootstrap CIs.
- Brier and log loss by feature subgroup (position, price tier, season-half).
- ECE (expected calibration error), reported per-component and per-subgroup.
- **Acceptance gate before Phase 3**: ECE < 0.05 in each subgroup for the minutes model. If not, stop and iterate — the optimizer is downstream of this and cannot fix bad probabilities.

### 7.3 Ablations

For each component:
- Remove each feature group, measure delta on Brier / log loss.
- Remove the entire bookmaker channel, measure delta. (Important: how much of our "alpha" is just market-mirroring?)

For the optimizer:
- Deterministic vs stochastic, points lift in season backtest.
- With vs without terminal value, points lift.
- With vs without chip integration vs heuristic chips.
- With vs without EO term (i.e., E[points] objective vs rank-progress objective). This is the test of whether rank-chasing helps in backtest.

### 7.4 Counterfactuals

- **Oracle prediction + your optimizer**: feed actual realized points back as forecasts; measure ceiling of optimizer.
- **Your prediction + oracle optimizer**: solve MILP with perfect-foresight forecasts disabled, but allow human-tuned heuristics; measure floor of forecast quality.
- Decomposition: total points = oracle ceiling − optimizer gap − prediction gap.

### 7.5 Baselines

- "Rolling mean xPts" point-prediction baseline, fed into the same MILP.
- "Most-popular-XI" template baseline (top-15 by ownership, captain top-owned).
- "Naive transfers": replace lowest-xPts player with highest-xPts available, no horizon.

### 7.6 Leakage audit

A separate test suite, run as a CI gate:
1. Static-import audit (Section 3.4).
2. Synthetic-future-row test (Section 3.4).
3. **Time-shuffle test**: scramble timestamps in a backtest; if performance doesn't degrade catastrophically, there's leakage.
4. **Future-feature search**: for any feature flagged "rolling X", assert the rolling window endpoint ≤ as_of − 1 minute.

---

## 8. Repository structure

```
fpl_bot/
├── pyproject.toml          # uv-managed; deps: polars, lightgbm, pyomo, highspy, sqlalchemy, alembic, httpx, pytest
├── README.md
├── docs/
│   └── design/
│       ├── phase0.md       # this document
│       ├── phase1.md       # to follow per phase
│       └── ...
├── alembic/                # DB migrations (versioned; never hand-edit deployed migrations)
├── src/fpl_bot/
│   ├── ingest/             # one module per source, each exposing fetch_raw_* and parse_raw_*
│   │                       # sources: fpl_api, vaastav, understat, fbref, footballdata, oddsapi, livefpl, fplstatistics
│   ├── db/
│   │   ├── models.py       # SQLAlchemy / SQLModel
│   │   └── pit.py          # the only sanctioned read API
│   ├── features/           # feature builders, all PIT-routed
│   ├── models/             # minutes, goals, assists, cs, bps, cards, saves
│   ├── scenarios/          # joint trajectory sampler
│   ├── optim/
│   │   ├── milp.py
│   │   ├── chip_dp.py
│   │   └── terminal_value.py
│   ├── eval/               # backtest harness, calibration, ablations, counterfactuals
│   └── cli/                # weekly run command — preview-and-confirm
├── tests/
│   ├── unit/
│   ├── integration/
│   └── leakage/            # the four leakage tests, run in CI
├── notebooks/              # research only; gitignored from the pipeline; outputs not consumed downstream
├── data/                   # gitignored; raw + parquet caches
└── configs/                # YAML: horizon, scenarios, λ, α/β, chip rules per season
```

Tooling:
- **uv** for env/lock management.
- **alembic** for migrations.
- **pytest** with `pytest-postgres` fixtures for DB tests.
- **ruff + pyright** for static analysis.
- **pre-commit** to run leakage static check before commit.

**Phase 1 CLI smoke commands** (developer-facing only; not the weekly user CLI):

```bash
fpl-bot ingest <source> --raw-only        # fetch_raw_<source>, write to data/raw/, audit row
fpl-bot ingest <source> --parse-only      # parse_raw_<source> from existing data/raw/
fpl-bot ingest <source>                   # fetch then parse
fpl-bot pit player-status \
  --player-id 1 --as-of 2024-08-16T18:30:00Z   # exercise the PIT API
fpl-bot leakage-check                     # run the four leakage tests (§3.4 / §7.6)
fpl-bot ingest-audit --source livefpl --since 7d   # show recent fetches for compliance review
```

These prove the end-to-end Phase-1 path: `fetch_raw → data/raw/ → parse_raw → tables → PIT query → leakage gate`.

---

## 9. Open questions and risks

### Round-1 questions — resolved

1. **Optimization objective**: ✓ EO-adjusted expected points (Stage A) with $\rho = 1.0$ default, swept over $\{0.7, 1.0, 1.2\}$ in backtest. Stage B threshold formulation deferred to Phase 4 if late-season backtest shows insufficient variance-seeking when behind.
2. **Top-10k EO**: ✓ scrape LiveFPL (primary) with FPLStatistics fallback from day one. Schema `fact_top10k_ownership` and PIT API `top10k_eo_as_of` added.
3. **Chip rules version**: ✓ formulate against 2024/25 ruleset; revise via `bps_rule_set` and chip-config updates when 2025/26 rules confirmed.
4. **BPS simulation**: ✓ full event-level simulation. ~10 additional event models specified in §4.6; supporting schema `fact_player_match_event` and `bps_rule_set` added.
5. **Position-mismatch features**: ✓ manual list in Phase 2.2, learned cluster in Phase 2.5.

### Round-2 questions — resolved

6. **Top-10k EO source**: ✓ LiveFPL primary, FPLStatistics fallback.
7. **Penalty-taker dim**: ✓ derived-with-manual-override. Schema added (`dim_penalty_taker`); manual rows take precedence over derived via `override_set` flag.

### Round-3 adjustments — applied

8. **Two-layer ingestion**: ✓ each source exposes `fetch_raw_*` (writes to `data/raw/`, audit row) and `parse_raw_*` (reads from disk to tables). Parser failures never force a refetch. (§2.3, §8.)
9. **Scraping compliance policy**: ✓ §2.5 added — robots.txt + ToS gate, polite UA, rate budgets, content-hash dedupe, fallback chain, never-evade posture. `ingest_audit` table added (§3.2).
10. **EO table generalization**: ✓ `fact_top10k_ownership` renamed to `fact_eo_snapshot`; `rank_band` and `provenance_url` columns added. PIT API consolidates to `eo_as_of(rank_band, ...)`.
11. **CLI smoke commands**: ✓ added to §8 (developer-facing only; weekly user CLI is Phase 6).

### Round-4 fix — cross-season stable IDs (applied during Phase 1A smoke)

12. **Stable global IDs**: ✓ `dim_player.player_id` and `dim_fixture.fixture_id` rekeyed to FPL `code` (stable across seasons). Two xref tables (`dim_player_season_xref`, `dim_fixture_season_xref`) translate per-season FPL element/fixture `id` to stable `code`. Migration `0002_stable_ids.py` applied; verified with Salah (code=118748, per-season id=381 in 2025/26). Without this, multi-season vaastav backfill would have collided per-season element ids — Salah's element_id varied 191→308 across six seasons.

### Risks I want on the record

- **Free-tier odds limits**: 500 req/mo is fine for weekly EPL, but if we want intraweek line moves to feed price-change models, we'll burn through it fast. Mitigation: cache + delta-detect.
- **LiveFPL scrape fragility**: top-10k EO is now a load-bearing dependency for the rank-utility objective. Layout changes or rate-limits would degrade the optimizer to overall-EO mode. Mitigation: snapshot raw HTML; column-shape validation; FPLStatistics fallback wired from day one.
- **Understat / fbref scraping fragility**: HTML-shape changes break the pipeline. Mitigation: snapshot raw HTML, parse offline, fail loudly.
- **Pre-2017/18 BPS data**: full event simulation requires fbref detailed stats from 2017/18+. Earlier seasons can only run a degraded BPS sim with empirical residuals. Backtest evaluation window 2019/20–2023/24 is unaffected.
- **Promoted-team cold start**: small samples + cross-league prior transfer is genuinely hard. Expect higher error in early-season weeks.
- **Manager changes mid-season**: rotation patterns reset. The minutes model needs a "manager tenure" feature; I'll detail it in Phase 2.1.
- **Penalty-taker volatility**: takers change mid-season more often than people remember. Without low-latency override, the penalty term will lag reality by 1–2 GW.
- **Stochastic MILP scaling**: 100–500 scenarios × 6–8 GWs × ~600 players is large. HiGHS is fast but we may need a candidate-prefilter cutting to ~150 players. Prototype scaling in Phase 3.
- **No clean-sheet markets**: Dixon-Coles inversion is fine but not as good as direct CS markets. If results are weak, candidate for a small paid odds source later.

### What is *not* in this design (deferred / out of scope)

- Multi-team head-to-head leagues — Phase 6 if we have time.
- Cup competitions (FA Cup) modeling — handled only as a congestion feature.
- Manager-fitness models for non-EPL minutes (Champions League minute prediction) — used as a feature but not predicted directly.
- Auto-execution of transfers via reverse-engineered FPL endpoints — explicitly out of scope per user direction.

---

**End of Phase 0 design document. Approved; ready for Phase 1.**
