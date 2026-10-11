# Gaffe app implementation plan

Written before implementation, 11 October 2026.

## Objective

Move routine FPL decisions out of the terminal into a desktop-first browser app
with responsive phone layouts. Reuse the existing Python prediction and
optimization stack and the configured PostgreSQL database. Default to the
configured team, currently entry 718959.

## Stack and delivery

- FastAPI and Uvicorn serve the API and bundled frontend from one process.
- Native HTML, CSS and JavaScript keep installation and packaging simple;
  the app needs no Node build step or external hosting account.
- Bind to localhost by default. Keep API keys and account tokens on the server.
- Add `gaffe serve` and document installation and launch. The app's refresh and
  forecast controls handle subsequent routine usage.

## Screens

1. **Overview:** current team points, rank, bank/free transfers, next deadline,
   next-gameweek forecast, captain and transfer advice, and data freshness.
2. **Squad:** pitch view, starters/bench, player forecasts and availability;
   switch to a readable list on small screens.
3. **Planner:** recommended transfers and a simple editable what-if plan,
   respecting budget, position and club limits; compare with holding the squad.
4. **Performance:** actual team versus saved pre-deadline recommendations,
   gameweek breakdown and clear coverage of incomplete/provisional results.
5. **Settings:** team selection, forecast horizon, timezone and data connection
   status. Do not expose secret values to the browser.

## Data and jobs

- Read current recommendation artifacts and audited public snapshots so the
  first screen works before an expensive model run.
- Prefer fresh team-specific snapshots. Label stale, provisional and missing
  data explicitly rather than substituting invented numbers.
- Queue public-data refresh and recommendation runs in a background worker;
  only one model/data job runs at a time. Expose progress and failures without
  blocking navigation. Save outputs per team and preserve prior forecasts.
- Refresh public FPL team/history/picks and bootstrap/fixtures using existing
  ingestion functions. Train only on finalized results.
- Continue to provide advice and local planning. Real FPL account transfers and
  chips are applied by the manager on FPL; this project has no submission flow.

## Design

Use a quiet editorial layout: dark navigation, light content, a green pitch,
clear typography, compact statistics and restrained team accents. Desktop gets
a persistent sidebar and useful side-by-side panels. Phones get compact
navigation, stacked cards, accessible controls and contained table scrolling.
Include loading, empty and failure states and keyboard-accessible navigation.

## Validation and Git

1. Review and retain existing model/audit changes, including new local
   refinements; test them before committing.
2. Add meaningful API/service tests for real payloads, planning constraints,
   background-job behavior and secret handling.
3. Check the app in a browser at desktop and phone sizes. Verify refresh,
   recommendations, navigation and the planner against the live local service.
4. Run lint, lockfile/install checks and the full existing regression suite.
5. Commit model refinements and app implementation separately, then push the
   existing branch `season26/live-cycle-fixes` to `origin` without force-pushing.

## Deferred scope

Remote multi-user hosting, account authentication, automatic FPL submissions,
native phone-store packaging and new predictive-model claims are outside this
first app. The local responsive web app establishes the daily-use foundation.

## Delivery record

Implemented all five screens, the local launcher, team-specific public snapshots,
isolated background refresh/forecast jobs, legal transfer comparisons and partial
performance reporting. Chromium checks covered desktop (1440px) and phone
(390px) layouts, all navigation, player dialogs, planner comparisons and settings.
The real team 718959 refresh and forecast completed through the app. The package
build includes all frontend assets. Twenty-four app tests cover API boundaries,
state reconstruction, live-history lag, planning and job/archive behavior.
Final regression suite: **407 passed**. Ruff, JavaScript syntax, locked install,
both CLI launchers, and the wheel asset check pass.
Screenshots and usage instructions are in [APP.md](APP.md).
