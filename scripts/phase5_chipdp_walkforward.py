"""Phase 5 chip-DP walk-forward across folds 21-24.

Subprocess-isolated (one Python process per fold) to dodge the HiGHS
state-leak issue documented in Phase 3 v1.4. Each fold runs:
- pre-fold chip schedule via the heuristic chip_scheduler
- rolling MILP with chips forced to the schedule
"""
from __future__ import annotations

import re
import subprocess


FOLDS = [
    {"train": [19, 20], "test": 21},
    {"train": [19, 20, 21], "test": 22},
    {"train": [19, 20, 21, 22], "test": 23},
    {"train": [19, 20, 21, 22, 23], "test": 24},
]


def run_one_fold(test_season: int, train_seasons: list[int]) -> tuple[bool, int, int, dict]:
    train_str = ", ".join(str(s) for s in train_seasons)
    code = f"""
from fpl_bot.eval.milp_backtest import backtest_season, format_validity_report, format_performance_report
r = backtest_season(
    test_season={test_season},
    train_seasons=[{train_str}],
    horizon=6, rho=1.0, alpha=0.8, beta=0.0,
    enable_chips=True, use_chip_schedule=True,
    n_iterations=200,
)
print(format_validity_report(r))
print()
print(format_performance_report(r))
print(f"_MARKER_pass={{r.all_validity_gates_pass}}_pts={{r.perf_total_points}}_template={{r.perf_template_total_points}}")
"""
    result = subprocess.run(
        ["uv", "run", "python", "-u", "-c", code],
        capture_output=False,
        timeout=3600,
    )
    if result.returncode != 0:
        print(f"FOLD {test_season} CRASHED rc={result.returncode}", flush=True)
        return False, 0, 0, {}
    return True, 0, 0, {}


def main() -> None:
    for fold in FOLDS:
        print(f"\n{'='*60}\nFOLD test={fold['test']} (Phase 5 chip-DP)\n{'='*60}", flush=True)
        run_one_fold(fold["test"], fold["train"])


if __name__ == "__main__":
    main()
