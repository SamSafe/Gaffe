# Gaffe

Gaffe helps Fantasy Premier League managers decide whether to spend a free transfer or roll it. It forecasts player points, builds a legal starting XI and bench, and compares transfer and captaincy choices across upcoming gameweeks.

![Gaffe desktop dashboard showing the gameweek overview](docs/images/app-desktop.png)

[App guide](docs/APP.md) · [Mobile screenshot](docs/images/app-mobile.png) · [Model design](docs/design/) · [Historical benchmark](docs/OPERATIONS_BENCHMARK.md)

## What would a transfer buy you?

Suppose you want to replace a midfielder before a difficult run of fixtures. The planner checks what you can afford at the player's actual selling price, whether the move costs points, and how it changes your projected score compared with keeping the squad. A small projected gain is easier to judge when you can see the cost of giving up the free transfer.

The dashboard also shows the recommended XI, bench order, captain and chip timing. After a deadline, the performance view scores saved recommendations against completed gameweeks. You make the final moves in FPL.

## How the forecast works

1. **Estimate playing time and returns.** The models combine minutes, expected goals and assists, set-piece roles and market-implied team strength to estimate each player's outcomes.
2. **Simulate the gameweek.** A Monte Carlo simulation combines those outcomes with clean sheets and bonus points to produce player points distributions.
3. **Choose a squad over several weeks.** A mixed-integer optimizer weighs expected points against the budget, formation, club limits and transfer costs. The chip planner considers when to use the available chips.

The prediction code uses Python, Polars and LightGBM. PostgreSQL holds the historical inputs, and Pyomo builds the optimization problem for CBC. [Prediction design](docs/design/phase2_5_joint_xpts.md) · [Optimizer design](docs/design/phase3_deterministic_milp.md) · [Set-piece observations](configs/set_piece_observations.csv)

## Historical results

The development benchmark trained on earlier seasons and simulated 38 gameweeks per test season with chips disabled.

| Test season | Gaffe points | Buy-and-hold points | Difference |
| --- | ---: | ---: | ---: |
| 2024/25 | 2,508 | 2,104 | +404 |
| 2025/26 | 2,331 | 1,994 | +337 |

Some closing odds and player metadata were available only after the relevant deadlines. Those timing limits can flatter a backtest. A full season of forecasts saved before each deadline is still needed to establish live performance. [Benchmark settings and limitations](docs/OPERATIONS_BENCHMARK.md)

## Run locally

From the repository root, install the project with Python 3.11+ and uv. The CLI help is a quick way to explore the available commands.

```bash
uv sync --all-extras
uv run gaffe --help
```

For the browser app, configure PostgreSQL as described in the app guide, then run the migrations and start the local server.

```bash
uv run alembic upgrade head
uv run gaffe serve
```

The dashboard opens at `http://127.0.0.1:8080`. Generating a forecast also requires the CBC solver and populated historical training data. The screenshots above show the interface without any setup. [App setup](docs/APP.md) · [Forecast runbook](docs/RUNBOOK_live.md)

## Data access

Gaffe is an independent project. FPL's 2026/27 terms address commercial use in clause 28(b) and automated access and extraction in clause 28(d). Clause 28(d) has no stated exception for personal or non-commercial use. Gaffe's live refresh sends automated requests to FPL even when started with a button. Obtain permission covering that use before enabling it. Each other data provider sets its own terms. The set-piece ranking rebuilds offline from the linked match observations. [FPL terms](https://fantasy.premierleague.com/help/terms)

## Development

The maintainer directs the project, makes modelling decisions and reviews the results. AI coding agents wrote much of the implementation, documentation and tests. [Model experiments and rejected changes](docs/MODEL_IMPROVEMENT_2026-10-11.md)
