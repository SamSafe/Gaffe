"""Phase 4 SAA walk-forward at |S|=25 with CBC + chip-DP.

CBC unlocks SAA at larger scenario counts that HiGHS couldn't handle.
Compare vs Phase 5 v1 (deterministic + chip-DP) baseline.
"""
from __future__ import annotations

import subprocess


FOLDS = [
    {"train": [19, 20], "test": 21},
    {"train": [19, 20, 21], "test": 22},
    {"train": [19, 20, 21, 22], "test": 23},
    {"train": [19, 20, 21, 22, 23], "test": 24},
]


def run_one_fold(test_season: int, train_seasons: list[int]) -> None:
    train_str = ", ".join(str(s) for s in train_seasons)
    code = f"""
from fpl_bot.eval.milp_backtest import backtest_season, format_validity_report, format_performance_report
r = backtest_season(
    test_season={test_season}, train_seasons=[{train_str}],
    horizon=6, rho=1.0, alpha=0.8, beta=0.0,
    enable_chips=True, use_chip_schedule=True,
    n_iterations=200,
    use_saa=True, n_scenarios=25,
)
print(format_validity_report(r))
print()
print(format_performance_report(r))
print(f"_MARKER_pass={{r.all_validity_gates_pass}}_pts={{r.perf_total_points}}_template={{r.perf_template_total_points}}")
"""
    subprocess.run(["uv", "run", "python", "-u", "-c", code], timeout=3600)


def main() -> None:
    for fold in FOLDS:
        print(f"\n{'='*60}\nFOLD test={fold['test']} (SAA |S|=25 + chip-DP + CBC)\n{'='*60}", flush=True)
        run_one_fold(fold["test"], fold["train"])


if __name__ == "__main__":
    main()
