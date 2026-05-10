"""H=6 + chips walk-forward across folds 21-24 at picked (ρ=1.0, α=0.5, β=0.0).

This is development data (test seasons reuse the picked settings). Not an
unbiased estimate.
"""
from __future__ import annotations

from fpl_bot.eval.milp_backtest import (
    backtest_season,
    format_validity_report,
    format_performance_report,
)


def _run(label: str, **kwargs) -> dict:
    print(f"\n=== {label} ===", flush=True)
    r = backtest_season(**kwargs, n_iterations=200)
    print(format_validity_report(r))
    print()
    print(format_performance_report(r))
    return {
        "label": label,
        "all_validity_pass": r.all_validity_gates_pass,
        "total_points": r.perf_total_points,
        "template_points": r.perf_template_total_points,
        "delta": r.perf_total_points - r.perf_template_total_points,
    }


def main() -> None:
    rho, alpha, beta = 1.0, 0.5, 0.0
    folds = (
        {"train": [19, 20], "test": 21},
        {"train": [19, 20, 21], "test": 22},
        {"train": [19, 20, 21, 22], "test": 23},
        {"train": [19, 20, 21, 22, 23], "test": 24},
    )

    results = []
    for fold in folds:
        out = _run(
            f"WALK-FORWARD H=6+chips fold test={fold['test']}",
            test_season=fold["test"],
            train_seasons=fold["train"],
            horizon=6,
            rho=rho,
            alpha=alpha,
            beta=beta,
            enable_chips=True,
        )
        results.append((fold["test"], out))

    print("\n\n========== H=6 + CHIPS WALK-FORWARD SUMMARY ==========")
    print(f"(ρ={rho}, α={alpha}, β={beta})")
    for season, r in results:
        flag = "✓" if r["all_validity_pass"] else "✗"
        print(
            f"  20{season}/{(season+1)%100:02d}: {flag} validity, "
            f"points={r['total_points']}, template={r['template_points']}, "
            f"Δ template={r['delta']:+d}"
        )


if __name__ == "__main__":
    main()
