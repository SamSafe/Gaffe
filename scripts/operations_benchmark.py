"""Corrected end-to-end development benchmark; never silently label it leakage-free."""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import json
from pathlib import Path

import polars as pl

from fpl_bot.eval.milp_backtest import backtest_season
from fpl_bot.eval.xpts_eval import _run_one_fold
from fpl_bot.models.validation import TRAINING_POLICY_VERSION


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", type=int, required=True)
    args = parser.parse_args()
    season = args.season
    train = list(range(19, season))
    output = Path(f"data/audits/2026-10-11/operations/season_{season}")
    output.mkdir(parents=True, exist_ok=True)
    print(f"Generating fresh joint xPts fold for season {season}", flush=True)
    fold = _run_one_fold(train, season, n_iterations=200, seed=42)
    if fold is None:
        raise RuntimeError("No predictions generated")
    metrics, frame, _ = fold
    cache = output / "predictions"
    cache.mkdir(exist_ok=True)
    frame.write_parquet(
        cache
        / f"season_{season}_train_{'_'.join(map(str, train))}_{TRAINING_POLICY_VERSION}.parquet"
    )
    result = backtest_season(
        season,
        train,
        horizon=6,
        rho=0,
        alpha=0.8,
        beta=0,
        enable_chips=False,
        n_iterations=200,
        cache_dir=cache,
    )
    limits = [
        "Historical odds can be closing prices rather than deadline snapshots.",
        "Later horizon feature rows can contain results unknown at the decision deadline.",
        "Player positions/clubs can use later season information.",
        "These seasons were used in earlier research and are development benchmarks.",
    ]
    summary = {
        "generated_at": dt.datetime.now(dt.UTC).isoformat(),
        "test_season": season,
        "train_seasons": train,
        "training_policy": TRAINING_POLICY_VERSION,
        "configuration": {
            "horizon": 6,
            "rho": 0,
            "alpha": 0.8,
            "beta": 0,
            "chips": False,
            "simulations": 200,
            "captain_minutes_attenuation": False,
        },
        "joint_xpts_metrics": dataclasses.asdict(metrics),
        "gameweeks": len(result.gw_records),
        "total_points": result.perf_total_points,
        "template_points": result.perf_template_total_points,
        "delta_template": result.perf_total_points - result.perf_template_total_points,
        "accounting_valid": {
            "feasibility": result.validity_feasibility,
            "budget": result.validity_budget,
            "transfers": result.validity_transfers,
            "chips": result.validity_chips,
        },
        "leakage_free_verified": False,
        "limitations": limits,
        "validity_failures": result.validity_failures,
        "weekly": [
            {
                "gameweek": r.gameweek,
                "points": r.actual_points,
                "bank": r.state_after.bank,
                "hits": r.decisions.hits,
                "captain": r.decisions.captain,
                "squad": sorted(r.decisions.squad),
                "solve_status": r.solve_status,
                "solve_time_seconds": r.solve_time_s,
            }
            for r in result.gw_records
        ],
    }
    (output / "results.json").write_text(json.dumps(summary, indent=2, allow_nan=False))
    pl.DataFrame(summary["weekly"]).write_parquet(output / "weekly.parquet")
    print(json.dumps({k: v for k, v in summary.items() if k != "weekly"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
