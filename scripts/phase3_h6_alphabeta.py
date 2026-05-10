"""α/β re-tuning at H=6 + chips on fold 24/25 at picked ρ=1.0.

α/β were inert across the 9-cell grid at H=4 — terminal value had no
discriminative effect. Worth re-checking at H=6 + chips.
"""
from __future__ import annotations

from fpl_bot.eval.milp_backtest import (
    backtest_season,
    format_validity_report,
    format_performance_report,
)


def main() -> None:
    rho = 1.0
    results = []
    for alpha in (0.5, 0.8, 1.0):
        for beta in (0.0, 0.05, 0.1):
            label = f"H=6+chips, ρ={rho}, α={alpha}, β={beta}, fold 24/25"
            print(f"\n=== {label} ===", flush=True)
            r = backtest_season(
                test_season=24,
                train_seasons=[19, 20, 21, 22, 23],
                horizon=6,
                rho=rho,
                alpha=alpha,
                beta=beta,
                enable_chips=True,
                n_iterations=200,
            )
            print(format_validity_report(r))
            print()
            print(format_performance_report(r))
            results.append(((alpha, beta), r.perf_total_points,
                            r.perf_total_points - r.perf_template_total_points,
                            r.all_validity_gates_pass))

    print("\n\n========== α/β GRID @ H=6 + CHIPS, ρ=1.0, fold 24/25 ==========")
    for (a, b), pts, delta, ok in results:
        flag = "✓" if ok else "✗"
        print(f"  α={a}, β={b}: {flag} validity, points={pts}, Δ template={delta:+d}")


if __name__ == "__main__":
    main()
