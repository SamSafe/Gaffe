# Gaffe operations and prospective validation plan

Written before implementation, 11 October 2026.

1. Rerun end-to-end xPts and rolling team benchmarks with chronological training
   validation, appearance-scaled defensive points and correct half-profit sales.
   Record configuration, coverage, predictive metrics, team points and every
   validity failure. Keep historical closing odds, future horizon features and
   previously examined seasons explicit limitations; do not claim untouched
   holdout performance. Use no chips across development seasons to avoid applying
   a later ruleset to earlier years.
2. Start a separate prospective bot ledger from the user's exact current squad,
   bank, purchase prices, transfers and chip inventory. Forecast from the bot's
   own state after initialization. Preserve pre-deadline decisions, apply price
   and chip accounting once, score finalized results with saved bench priority,
   and show the independent comparison separately from old weekly recommendations.
   Missing deadlines remain visible; never fabricate historical choices.
3. Add persistent scheduled work: refresh finalized results periodically, refresh
   odds/news and generate both human advice and bot decisions before deadlines,
   then freeze and settle the bot. Retry failures with backoff, avoid overlapping
   jobs and expose the schedule and last outcomes in Settings. Install the local
   app as a user service so automation survives terminal closure and restarts.
4. Add HTTPS phone access on the local network, paired through a short-lived
   desktop-generated link. Keep the existing localhost app and credentials on
   the server. Provide connection information and revoke access in Settings.
5. Test accounting, freeze/catch-up timing, concurrency, scheduling and access
   control. Exercise real refresh/model jobs, desktop/phone browsers, service
   restart and the complete suite. Commit, push and verify GitHub CI.
