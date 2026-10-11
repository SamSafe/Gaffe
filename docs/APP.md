# Gaffe browser app

## Start

Run from the repository root, with the project's existing database and training
data configured:

```bash
uv sync --all-extras
uv run alembic upgrade head
uv run gaffe serve
```

The app opens at `http://127.0.0.1:8080`. `--no-open` suppresses browser launch;
`--port 8081` chooses another port. The server is local and single-process.
It has responsive phone layouts; remote hosting and account login are deferred.
An existing recommendation and audit snapshot can populate it immediately.
For a fresh clone, use Settings to select an FPL entry, then refresh its data.
Historical model training data and the CBC solver still need the one-time setup
in [the live runbook](RUNBOOK_live.md).

## Daily workflow

1. **Settings:** enter your team ID, forecast horizon and deadline timezone.
2. **Refresh data:** fetch public players, fixtures, entry, history, squad and
   finalized results. Reconstruct bank, free transfers, chips and purchase prices.
   Only finalized single-fixture results enter the per-match training table.
   Double gameweeks remain aggregate actuals until per-fixture sources arrive.
3. **Generate forecast:** train on completed matches, simulate the upcoming
   horizon, and solve the squad, XI, captain and transfers. It can take several
   minutes. Navigation remains usable and a job banner reports progress.
4. **Transfer planner:** begin with the model plan, edit transfers, optionally
   choose a captain, then compare. It enforces 2/5/5/3 squad positions, max three
   per club, exact half-profit sale prices, bank and four-point transfer hits.
5. **Performance:** inspect net actual points, saved bot scores, provisional
   weeks and missing coverage. Each bot plan starts from the actual team that
   week; this is not a continuously simulated bot season. Chip usage affects
   the comparison. Only finalized pre-deadline plans count.
6. Apply the chosen moves on FPL. Refresh near the next deadline.

The instant planner estimates one week's XI plus captain points after hits.
It does not simulate automatic substitutes or chips. The full recommendation
optimizes across its horizon, so the best immediate gain can differ from its
chosen move. Player dialogs show news and the saved horizon forecast.

Public endpoints expose the last deadline's squad. Pending transfers, an active
Free Hit squad and exact purchase prices can require additional state knowledge.
The worker restores the permanent squad after a Free Hit, excludes temporary
Free Hit purchases, retains audited cost basis and flags unknown prices.
Unknown sale prices prevent a budget-safe recommendation. Never interpret an
unrefreshed public snapshot as authenticated account synchronization.

Bookmaker refresh is optional in Settings and consumes the configured API's
request allowance. The dashboard reports later weeks with market fallback.

## Data and local boundary

- `data/web/preferences.json`: team, horizon and timezone, without credentials.
- `data/web/season_N/team_ID/snapshots/`: complete public snapshots. An atomic
  pointer publishes a new snapshot only after its public requests finish.
- `data/web/jobs/`: durable safe progress records; unfinished jobs are marked
  failed after an app restart.
- `data/live/recommendations/season_N/team_ID/gw_G/`: new per-team forecasts.
  Existing primary recommendations remain intact. Timestamped archives preserve
  each new run and prevent failed solves overwriting a usable recommendation.
- Legacy artifacts are only used when team and season match.

API keys, cookies, access tokens, connection strings and raw worker logs are
never returned to the browser. Mutations require an ephemeral same-origin
token, and the server accepts local hosts. This is a local app, not a remotely
authenticated multi-user service. Job requests select fixed worker operations,
with validated numeric inputs; the API does not accept shell commands or paths.

If a job fails, the dashboard retains existing snapshots/forecasts and reports
a safe error. Check database migrations, historical data and the solver using
the runbook. A public snapshot may be saved even if database ingestion fails.
Only one refresh or forecast job runs at a time; wait before changing settings.

## Architecture and verification

FastAPI serves the API and bundled static HTML, CSS and JavaScript; Uvicorn runs
the local process. No frontend build system, CDN or hosting account is required.
The worker runs in a separate Python process to keep model work off HTTP threads.
FastAPI's documented [static file serving](https://fastapi.tiangolo.com/tutorial/static-files/)
provides the bundled assets. The API tests exercise planning constraints,
partial comparisons, credentials, origin checks, team isolation, refresh-state
reconstruction and overlapping/restarted jobs.

```bash
uv run pytest tests/unit/test_web_app.py -q
uv run ruff check src tests
```

![Desktop dashboard](images/app-desktop.png)

![Mobile dashboard](images/app-mobile.png)
