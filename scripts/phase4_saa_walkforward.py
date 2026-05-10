"""Phase 4 SAA walk-forward — subprocess-isolated, one fold per process.

Each fold uses the cached raw samples from data/cache/xpts_raw_samples/.
Configuration matches the Phase 3 v1.2 picked settings (ρ=1.0, α=0.8, β=0.0,
H=6, chips on, cost-basis sell-tax via cached predictions cache).
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


def run_one_fold(test_season: int, train_seasons: list[int], n_scenarios: int) -> tuple[bool, int, int]:
    train_str = ", ".join(str(s) for s in train_seasons)
    code = f"""
from fpl_bot.eval.milp_backtest import backtest_season, format_validity_report, format_performance_report
r = backtest_season(
    test_season={test_season},
    train_seasons=[{train_str}],
    horizon=6, rho=1.0, alpha=0.8, beta=0.0,
    enable_chips=True, n_iterations=200,
    use_saa=True, n_scenarios={n_scenarios},
)
print(format_validity_report(r))
print()
print(format_performance_report(r))
print(f"_MARKER_pass={{r.all_validity_gates_pass}}_pts={{r.perf_total_points}}_template={{r.perf_template_total_points}}")
"""
    result = subprocess.run(
        ["uv", "run", "python", "-u", "-c", code],
        capture_output=False,
        timeout=7200,
    )
    if result.returncode != 0:
        print(f"FOLD {test_season} CRASHED: rc={result.returncode}", flush=True)
        return False, 0, 0
    return True, 0, 0  # the inline subprocess just streams output; per-fold parse via separate file capture if needed.


def main() -> None:
    # |S|=10 picked from tractability sweep: 9.3s/GW with chips on at fold 21
    # (comparable to Phase 3 deterministic). |S|=25 hits ~90s/GW with chips.
    n_scenarios = 10
    for fold in FOLDS:
        print(f"\n{'='*60}", flush=True)
        print(f"FOLD test={fold['test']} (train={fold['train']}, |S|={n_scenarios})", flush=True)
        print(f"{'='*60}", flush=True)
        run_one_fold(fold["test"], fold["train"], n_scenarios)


if __name__ == "__main__":
    main()
