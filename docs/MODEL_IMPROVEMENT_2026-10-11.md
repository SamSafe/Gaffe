# Improving the model against team 718959

Updated 11 October 2026. The immediate priorities are reliable starter minutes,
captain forecasts for the players the team can own, and transfer decisions that
beat keeping the existing squad. More complexity needs evidence before promotion.

## What the losses show

The original primary recommendations scored 157 versus the team's 247 across
GW1, GW2 and GW4. Those are three independent recommendations from the actual
team's state, not a continuously managed bot season. There are no saved primary
recommendations for GW3, GW5 or GW6.

| GW | Your actual score | Your score with normal captaincy and no chip | Original bot score |
|---|---:|---:|---:|
| 1 | 63 | 45 | 27 |
| 2 | 90 | 90 | 68 |
| 4 | 94 | 82 | 62 |
| Total | 247 | 217 | 157 |

Bench Boost and Triple Captain contributed 30 of the observed 90-point lead.
The remaining 60-point difference persists when those two chips are removed,
using the recorded bench order and normal substitutions. This adjustment does
not account for the future value of chips the bot could have saved in a separate
season simulation.

GW2 is the clearest transfer failure: the bot sold Bruno Fernandes (23 points)
and João Pedro (9), bought Watkins (0) and Bentancur (2), and took a four-point
hit. The incoming versus outgoing player total, including that hit, was **-34**
for that week. This is a realized transfer diagnostic, not the change in total
XI points or proof the outcome was knowable. Multiweek value still matters.

Captaincy also contributed: your ordinary captain bonus was 9 versus the bot's
2 in GW2, and 12 versus 7 in GW4 before your Triple Captain extra. The audit's
live João Pedro example shows why minutes deserves priority: a single DNP was
pulling his next-match expectation down to 37.1 minutes, even while provisional
GW6 already showed him playing 71 minutes. Those GW6 results remain outside
training until finalization.

Scoring provenance is in `data/audits/2026-10-11/bot_vs_team.json` and `.md`.

## Implemented now: trustworthy training stops

The previous core evaluators used the outer test season to choose LightGBM's
best iteration. Live trainers instead evaluated stopping against the same rows
used for fitting. Both practices have been replaced in the core pipeline:

1. Take the latest 20% of distinct kickoff groups from the training history
   as an internal chronological validation set. Players in simultaneous
   fixtures stay together; a three-hour gap excludes unfinished preceding
   matches from the fitting partition.
2. Select the number of boosting rounds on that internal holdout.
3. Refit on all available completed training history at the selected round
   count, including the recent matches used for internal validation.
4. Predict on the outer evaluation season or upcoming fixtures. Their outcomes
   never control stopping.

This applies to minutes, goals, assists, bonus-simulator inputs, clean-sheet
evaluation and their ablation/calibration diagnostics. It is a methodological
repair, not a claim of extra FPL points. The disabled price-change predictor
is outside this change and remains disabled.

New prediction and raw-sample cache names include `chronological_v1`, preventing
reuse of artifacts produced by the old stopping policy. Live metadata records
the policy. Optional forward calibration now sources only earlier seasons;
it remains disabled by default. Historical files have been preserved.

The shared helper is `src/fpl_bot/models/validation.py`. Twelve new leakage
checks cover chronological separation, unfinished-result gaps, unsafe input,
restoring recent history during refitting, future-label isolation in the three
core predictors, multiclass boosting-round accounting during refitting, and
avoiding legacy raw-sample caches.

## Component benchmarks after the repair

Each fold trains on seasons 19 through the year preceding its evaluation
season. Lower values are better. All comparisons use the same available corpus.

| Evaluation season | Minutes log loss | With team rotation history | Rolling-minutes baseline |
|---|---:|---:|---:|
| 2024/25 | 0.50518 | 0.50487 | 1.38053 |
| 2025/26 | 0.45891 | 0.45850 | 1.30946 |

Rotation history improves the component metric in both seasons, but the gain
is small: approximately 0.06% and 0.09%. **The baseline minutes feature mode
remains the live default.** The rotation candidate still needs joint-xPts,
captain and team checks under the repaired policy before promotion.

| Target | Evaluation season | Model Poisson deviance | Rolling xG/xA baseline | Reduction |
|---|---|---:|---:|---:|
| Goals | 2024/25 | 0.36877 | 0.50363 | 26.8% |
| Goals | 2025/26 | 0.37175 | 0.52566 | 29.3% |
| Assists | 2024/25 | 0.38826 | 0.47638 | 18.5% |
| Assists | 2025/26 | 0.38804 | 0.48466 | 19.9% |

These reductions compare the current weighted component models with their
simple baselines. They do not measure improvement over the previous stopping
policy and do not imply equivalent improvements in team points.

Full component results are saved in
`data/audits/2026-10-11/training_policy_evaluation.json`. Reproduce them using
`eval.minutes_eval.run_walk_forward_cv` with `feature_mode="baseline"` and
`"rotation"`, and `eval.goals_eval.run_walk_forward_cv` with
`weighted_options=(True,)`, with folds
`[{"train": [19,20,21,22,23], "test": 24},
  {"train": [19,20,21,22,23,24], "test": 25}]`.

Important limits remain: these seasons were examined in previous research;
they are development benchmarks, not newly untouched holdouts. Closing odds
and the existing player-position corpus can still create historical look-ahead.
Complete joint-xPts and team backtests have not been regenerated here.

## Prioritized next changes and requirements

| Priority | Change | Evidence needed before promotion |
|---|---|---|
| 1 | Improve minutes for established starters, rotation and injury returns; learn minutes conditional on a start/appearance as well as the 0/short/60+ buckets; combine recent history with timestamped team news and supported per-GW analyst overrides | Lower minutes log loss/Brier and expected-minutes error for likely starters, premiums and season openers; improved calibration; no future news in training |
| 2 | Improve captain and vice decisions using calibrated points and appearance probabilities | Better forecasts in the high-points range, plus stable captain results across development folds and prospective deadlines; keep the points-maximizing live objective |
| 3 | Compare every transfer plan with holding; include the hit, sale tax, free-transfer value and uncertain minutes over the horizon | Positive modeled gain over holding and consistent realized multiweek gains; validate any extra uncertainty margin before changing its default |
| 4 | Refresh fixture odds and player roles near deadlines; collect anytime-goalscorer odds as a separate candidate | Timestamped coverage and prospective calibration; promote player-prop blending only if it improves goals/points forecasts. Its live weight remains zero |
| 5 | Test whether shorter horizons or discounted distant forecasts improve decisions when later fixture odds are absent | Accuracy and team comparisons across multiple periods; avoid fitting a horizon to just GW7 or the three known losing weeks |
| 6 | Maintain a separate prospective bot squad and decision ledger | Same initial GW7 squad, bank, purchase prices, free transfers and chip inventory as the user; own state thereafter; freeze choices before each deadline |

The existing live objective uses `rho=0`, maximizing expected points without
an ownership penalty. The earlier audit already repaired stale features,
latest-result inclusion, vice selection, defensive points for predicted DNPs,
price accounting and recorded bench order. Those repairs should remain covered
while testing new predictive changes.

For priority 6, record both forecasts and decisions before the deadline,
including code/data versions, model policy, player-level minutes and xPts,
transfers, captain/vice, bench order and chips. After finalization, score actual
points and report transfer gains, captain outcomes, lineup mistakes and forecast
calibration. Keep ordinary lineup/captain points separate from chips and hits
in the diagnostic breakdown. The primary result should still be total net points
over the remaining season. Short winning streaks alone cannot establish an edge.

The user can contribute most through timely, sourced fitness/starting-role
information and final decisions recorded before deadlines. Do not supply later
outcomes as if they were available at the decision time. No extra paid data or
larger model is required to begin these tests.

## Review corrections

A same-day review of these changes found three defects. Each is fixed and
covered by a regression test. Evidence scripts and logs are in
`data/audits/2026-10-11/review/`.

**1. Live rest days were outside the training range.** About 59% of training
rows are unused-squad rows, so `days_since_last_match` measures the club's
schedule: median 7 days, p90 14, and only 1.2% of in-season rows exceed 21.
The live builder measured rest from the last *ingested* match instead. It fed
28 days for GW7 and 70 for GW12, a gap seen in training only at season
openers. `season_match_count` was also frozen while `gameweek` advanced. Both
now follow the club schedule, including GW6 fixtures that are not yet ingested.
With the same trained model, established starters (`start60_rate_5` ≥ 0.8,
165 per GW) changed as follows:

| GW | 7 | 8 | 9 | 10 | 11 | 12 |
|---|---:|---:|---:|---:|---:|---:|
| Mean P(60+ minutes), before | 0.73 | 0.73 | 0.73 | 0.72 | 0.69 | 0.69 |
| Mean P(60+ minutes), after | 0.82 | 0.83 | 0.83 | 0.83 | 0.80 | 0.82 |

The error compressed forecasts toward the middle: nailed starters lost 5–8
expected minutes and recently benched players gained some. Before this change
the builder carried the last row's own rest value, which was stale but within
the training range.

**2. Backtest DefCon used look-ahead.** In backtest mode, the DefCon addition
was applied only to player/GWs present in the DefCon source, which keeps
`minutes > 0` rows. It went only to players who actually appeared. Backtests
now add it to every predicted player/GW and scale it by expected appearances,
as live does. The fold output carries `p_appearance` for this purpose. On the
25/26 fold (train 19–24), with identical predictions in each arm:

| Metric | DefCon off | Leaky (old) | Fixed |
|---|---:|---:|---:|
| DEF squared error | 4.382 | 4.267 | 4.332 |
| MID squared error | 3.421 | 3.350 | 3.395 |
| DEF Spearman | 0.668 | 0.690 | 0.670 |
| MID Spearman | 0.766 | 0.778 | 0.765 |
| MILP points (H=6, ρ=0, chips on) | 2262 | 2353 | 2367 |

DefCon still improves mean accuracy without the leak. However, roughly 60% of
the previously measured accuracy gain, and almost all of its ranking gain,
came from look-ahead. The single-season points ordering is within MILP noise,
so it is not evidence either way. MAE favours "off" in every position, but
MAE rewards the median of right-skewed points, while the solver needs means.

The shipped shrinkage (DEF 0.4, MID 1.0) was tuned under the leak.
Least-squares shrinkage on 25/26 is DEF 0.76 (90% bootstrap 0.59–0.93) and
MID 0.82 (0.66–0.97). On 26/27 GW1–5, out of sample, it is DEF 0.73
(0.23–1.37) and MID 0.37 (0.04–0.73). Both seasons point to more DEF weight and
less MID weight. Moving to those values improves squared error by at most
0.4%, so the constants are unchanged; this is a candidate for the next
reviewed tuning pass, not a measured win.

**3. Corrected purchase prices lapsed after one gameweek.** Overrides were
read for the exact gameweek only. From GW8, the public snapshot's current
price would therefore be treated as the purchase price, which credits the
full profit on a rise. `cost_basis` entries now carry forward until a later
block replaces them. Bank, free transfers and chips still apply only to their
own gameweek.

The recommendation's chip plan now lists unscheduled chips with their expiry.
FH1 was absent because the scheduler places a Free Hit only on a known blank
gameweek.

## Updated GW7 forecast

This supersedes the 50.8-point forecast above. It was regenerated with the
corrections, the same finalized-GW5 data and market snapshot, 1,000
simulation iterations and train seasons 19–26.

**Rogers and Schade → Palmer and Szoboszlai. No hit; the bank is £0.0m.
Keep João Pedro on the bench. Captain Haaland, vice Groß.** The proposed XI
is Raya; De Cuyper, Calafiori, Konsa, Muharemović; Groß, Palmer, Szoboszlai,
Tavernier; Haaland, Wissa. The bench order is Gibbs-White, Ajayi, Tzolakis and
João Pedro.

This is the audit's "fit João Pedro" choice, now reached from the baseline,
so the transfers no longer depend on that fitness assumption. João Pedro's
baseline fell from 39.5 to 29.2 expected minutes: his latest ingested match
is the GW5 DNP, and the old error had pulled him toward the middle. His
provisional GW6 minutes are still outside the data. The expected GW7 total
(55.1) is not comparable with 50.8, because the minutes correction raises
every nailed starter's forecast. Re-run after GW6 is finalized and before the
17 October deadline.

The updated outputs are under `data/live/recommendations/season_26/gw_7/`.
The 50.8-point outputs are preserved under
`data/audits/2026-10-11/before_rest_day_fix/`.

## Validation

Full unit, integration, database and leakage suite: **383 passed** after the
review corrections. Ruff and `git diff --check` pass. The two-season component evaluation completed successfully.
No real-account transfers or chip actions were submitted.

For why separate chronological validation is necessary, see the primary
[LightGBM early-stopping documentation](https://lightgbm.readthedocs.io/en/stable/pythonapi/lightgbm.early_stopping.html)
and [scikit-learn time-series validation documentation](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html).

## Browser app follow-up, 17:37 UTC

The new app completed a real public-data refresh and forecast run for 718959.
The entry summary now reports **385 points** and **62 provisional GW6 points**;
the gameweek history endpoint still reports the older 370 / 47 snapshot. The
dashboard prefers the live entry summary for that unfinished week and labels
it provisional. Bootstrap and fixtures confirm GW6 is unfinished. Training
therefore correctly remains through **GW5**, rather than ingesting partial
labels. The earlier 370-point comparison describes the earlier audit snapshot.

The latest forecast is **54.7 GW7 points**, replacing the earlier 55.1 for
current usage: **Rogers + Schade → Anderson + Palmer**, zero hits, **£0.6m bank**,
**Haaland captain / Groß vice**. Its instant one-week comparison is **+2.5 xPts**
against the best XI from holding the squad (52.2). This rerun is current advice,
not evidence of better model accuracy. Near-equal MILP plans can differ within
the solver's tolerance. GW9–12 still lack bookmaker coverage and use fallback.

The original team-versus-bot lead remains **+90 across GW1, GW2 and GW4**.
The app preserves the primary legacy recommendations and stores new runs under
`data/live/recommendations/season_26/team_718959/gw_7/`, with dated archives.
No real-account moves were submitted. See [the app guide](APP.md) for usage.

## Follow-up, 19 October 2026

**DefCon midfielder weight 1.0 → 0.82.** The weights fitted on 25/26 were
scored forward on 26/27 GW1–6 (fresh fold, train 19–25, finalized GW6), which
that fit never saw. Squared error, lower is better:

| Position | Off | Shipped (DEF 0.4, MID 1.0) | 25/26 fit (DEF 0.76, MID 0.82) | Fit − shipped, 90% GW-block CI |
|---|---:|---:|---:|---|
| DEF | 5.8774 | 5.8432 | 5.8476 | +0.0044 (−0.049, +0.051) |
| MID | 4.5561 | 4.5598 | 4.5517 | −0.0081 (−0.014, −0.003) |

MID improves forward, matching 25/26's direction, so it ships. DEF does not:
0.4 stays. The gain is about 0.2% of MID squared error and six gameweek blocks
make a crude bootstrap, so expect no visible change in points. Script and log:
`data/audits/2026-10-19/defcon_oos.{py,log}`.

**Plan vs rolling the transfer (priority 3).** When a forecast proposes
transfers, the optimizer now solves again with no buys in the first week
(later weeks stay free, so the rolled transfer keeps its value). The
recommendation records both horizon objectives in `solve_meta.hold_comparison`,
and the app and alerts show the gain. Both solves stop within CBC's 1% gap, so a
gain inside that tolerance is labelled as not separating the plans. This is
decision support only and changes no choice. A per-transfer penalty was already
measured and rejected (−41 points across folds; `docs/design/phase6_live_run.md`).

**Measured and not shipped: per-player 60+ split.** The simulator splits the
60+ minutes bucket into 70 or 90 minutes with one rate per position (DEF 0.13,
MID 0.38, FWD 0.45 subbed before 90). A per-player rate, using the player's
last 20 games of 60+ minutes shrunk toward the position rate with 10
pseudo-games, predicts the split much better (log loss, lower is better):

| Test season | Position rate | Player rate |
|---|---:|---:|
| 2024/25 | 0.552 | 0.495 |
| 2025/26 | 0.542 | 0.488 |
| 2026/27 GW1-6 | 0.582 | 0.542 |

Joint xPts accuracy did not follow. With paired simulations (same seeds and
draw streams), overall squared error went 4.1616 → 4.1666 (24/25), 3.6000 →
3.5962 (25/26) and 4.6185 → 4.6187 (26/27 GW1-7), each 90% week-block interval
spanning zero. Defenders were significantly worse in 24/25 (+0.007 to +0.044).
The top-20-per-week slice improved in 24/25 and 25/26 and worsened in 26/27,
none significant. Minutes inside 60+ mostly move goal credit between
teammates, not points, so under the accuracy rule it stays off
(`minutes_split="player"` is a research mode). Scripts and logs:
`data/audits/2026-10-19/sub_rate_{component,xpts}.py`.

**Goalscorer odds are now collected (priority 4).** Every odds refresh also
records anytime-goalscorer prices (~10 of the 500 monthly API credits per
gameweek). Collection had stopped after GW4 once the weekly manual routine
gave way to the scheduler. Their forecast weight stays 0.0. The history is
for testing market-implied availability against actual minutes.

**Shipped: crowd transfers in next-week minutes (priority 1).** Each
gameweek's pre-deadline transfers and ownership act as timestamped team news.
Minutes log loss fell 8-9% overall and 16-17% for regulars after a missed
game; joint xPts squared error fell 1.8% (24/25), 2.7% (25/26) and 2.6%
(26/27 GW1-6, out of sample), every interval excluding zero. Details and live
limits (82% of a week's transfers are visible six hours out) are in
`docs/design/phase9_minutes.md`, step B′.
