"""Phase 5 ρ/α/β re-sweep on fold 24/25 with full Phase 5 setup
(chip-DP, auto-sub, multinomial predictions). Picked α=0.8 came from
Phase 3 v1.2 (before chip-DP + auto-sub); re-tuning may find better.
"""
from __future__ import annotations

import subprocess


def run_one(rho: float, alpha: float, beta: float) -> int:
    """Returns total points. -1 if anything failed."""
    code = f"""
from fpl_bot.eval.milp_backtest import backtest_season
r = backtest_season(
    test_season=24, train_seasons=[19, 20, 21, 22, 23],
    horizon=6, rho={rho}, alpha={alpha}, beta={beta},
    enable_chips=True, use_chip_schedule=True, n_iterations=200,
)
ok = r.all_validity_gates_pass
print(f"_MARKER_pass={{ok}}_pts={{r.perf_total_points}}_template={{r.perf_template_total_points}}")
"""
    result = subprocess.run(
        ["uv", "run", "python", "-u", "-c", code],
        capture_output=True, text=True, timeout=1800,
    )
    if result.returncode != 0:
        print(f"  CRASHED rc={result.returncode}", flush=True)
        return -1
    import re
    m = re.search(r"_MARKER_pass=(True|False)_pts=(\d+)_template=(\d+)", result.stdout)
    if not m:
        return -1
    return int(m.group(2)) if m.group(1) == "True" else -1


def main() -> None:
    results = []
    # 9-cell ρ × α grid at β=0.0 (β was inert at H=4, weak at H=6 in prior sweeps).
    # If a ρ × α winner emerges, optionally sweep β as follow-up.
    for rho in (0.7, 1.0, 1.2):
        for alpha in (0.5, 0.8, 1.0):
            beta = 0.0
            print(f"\n=== ρ={rho}, α={alpha}, β={beta} ===", flush=True)
            pts = run_one(rho, alpha, beta)
            results.append((rho, alpha, beta, pts))
            print(f"  → {pts} pts", flush=True)
    print("\n=== SUMMARY ===")
    results.sort(key=lambda x: x[3], reverse=True)
    for rho, alpha, beta, pts in results:
        flag = "✓" if pts > 0 else "✗"
        print(f"  ρ={rho}, α={alpha}, β={beta}: {flag} {pts}")


if __name__ == "__main__":
    main()
