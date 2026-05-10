"""Phase 3.5 walk-forward: H=6 + chips with cost-basis sell-tax active.

Predictor was shelved (CV gates failed). The cost-basis sell-tax fix ships
alone — restores FPL's actual rule that profits incur 50% tax on sells.
"""
from __future__ import annotations

from fpl_bot.eval.milp_backtest import (
    backtest_season,
    format_validity_report,
    format_performance_report,
)


def main() -> None:
    rho, alpha, beta = 1.0, 0.8, 0.0
    folds = (
        {"train": [19, 20], "test": 21},
        {"train": [19, 20, 21], "test": 22},
        {"train": [19, 20, 21, 22], "test": 23},
        {"train": [19, 20, 21, 22, 23], "test": 24},
    )

    results = []
    for fold in folds:
        label = f"v3.5 (cost-basis sell-tax) fold test={fold['test']}"
        print(f"\n=== {label} ===", flush=True)
        r = backtest_season(
            test_season=fold["test"],
            train_seasons=fold["train"],
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
        results.append((fold["test"], r))

    print("\n\n========== v3.5 WALK-FORWARD SUMMARY ==========")
    print(f"(ρ={rho}, α={alpha}, β={beta}, time_limit=120s, cost-basis sell-tax)")
    for season, r in results:
        flag = "✓" if r.all_validity_gates_pass else "✗"
        delta = r.perf_total_points - r.perf_template_total_points
        print(
            f"  20{season}/{(season+1)%100:02d}: {flag} validity, "
            f"points={r.perf_total_points}, template={r.perf_template_total_points}, "
            f"Δ template={delta:+d}"
        )


if __name__ == "__main__":
    main()
