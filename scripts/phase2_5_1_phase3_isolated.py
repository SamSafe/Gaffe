"""Phase 2.5.1 → Phase 3 walk-forward with each fold in a fresh subprocess.

The HiGHS solver leaks state across folds in a single Python process, leading
to SIGABRT on later folds. Running each fold in its own subprocess isolates
the leak.
"""
from __future__ import annotations

import re
import subprocess
import sys


FOLDS = [
    {"train": [19, 20], "test": 21},
    {"train": [19, 20, 21], "test": 22},
    {"train": [19, 20, 21, 22], "test": 23},
    {"train": [19, 20, 21, 22, 23], "test": 24},
]


def run_one_fold(test_season: int, train_seasons: list[int]) -> tuple[bool, int, int]:
    """Returns (validity_pass, milp_total, template_total)."""
    train_str = ", ".join(str(s) for s in train_seasons)
    code = f"""
from fpl_bot.eval.milp_backtest import backtest_season, format_validity_report, format_performance_report
r = backtest_season(test_season={test_season}, train_seasons=[{train_str}], horizon=6, rho=1.0, alpha=0.8, beta=0.0, enable_chips=True, n_iterations=200)
print(format_validity_report(r))
print()
print(format_performance_report(r))
print(f"_MARKER_pass={{r.all_validity_gates_pass}}_pts={{r.perf_total_points}}_template={{r.perf_template_total_points}}")
"""
    result = subprocess.run(
        ["uv", "run", "python", "-u", "-c", code],
        capture_output=True,
        text=True,
        timeout=2400,
    )
    print(result.stdout, flush=True)
    if result.returncode != 0:
        print(f"FOLD {test_season} CRASHED: rc={result.returncode}", flush=True)
        print(result.stderr, flush=True)
        return False, 0, 0

    m = re.search(
        r"_MARKER_pass=(True|False)_pts=(\d+)_template=(\d+)", result.stdout
    )
    if not m:
        print(f"FOLD {test_season}: marker not found", flush=True)
        return False, 0, 0
    return m.group(1) == "True", int(m.group(2)), int(m.group(3))


def main() -> None:
    summary = []
    for fold in FOLDS:
        print(f"\n{'='*60}", flush=True)
        print(f"FOLD test={fold['test']} (train={fold['train']})", flush=True)
        print(f"{'='*60}", flush=True)
        ok, milp, template = run_one_fold(fold["test"], fold["train"])
        summary.append((fold["test"], ok, milp, template))

    print("\n\n========== Phase 2.5.1 → Phase 3 SUMMARY ==========")
    print("(ρ=1.0, α=0.8, β=0.0, H=6+chips, multinomial-fix predictions, 1% MIP gap)")
    for season, ok, milp, template in summary:
        flag = "✓" if ok else "✗"
        delta = milp - template
        print(
            f"  20{season}/{(season+1)%100:02d}: {flag} validity, "
            f"points={milp}, template={template}, Δ={delta:+d}"
        )


if __name__ == "__main__":
    main()
