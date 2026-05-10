"""Phase 3 polish sweep — ρ on 24/25, then α/β grid at picked ρ, then
full walk-forward 21-24 at picked (ρ, α, β).

Per round-1 review: this is development/tuning data, NOT an unbiased
performance estimate (predictions and tuning live in the same fold for
the sweep portion).
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
        "validity_failures": list(r.validity_failures),
    }


def main() -> None:
    summary: list[dict] = []

    # Phase A: ρ sweep on 24/25 at α=0.8, β=0.05, H=4
    rho_results = []
    for rho in (0.7, 1.0, 1.2):
        out = _run(
            f"rho={rho}, alpha=0.8, beta=0.05, H=4, fold 24/25",
            test_season=24, train_seasons=[19, 20, 21, 22, 23],
            horizon=4, rho=rho, alpha=0.8, beta=0.05,
        )
        rho_results.append((rho, out))
    best_rho = max(rho_results, key=lambda x: x[1]["delta"] if x[1]["all_validity_pass"] else -1e18)[0]
    print(f"\n>>> picked rho = {best_rho}")

    # Phase B: α × β grid at picked ρ, on 24/25
    ab_results = []
    for alpha in (0.5, 0.8, 1.0):
        for beta in (0.0, 0.05, 0.1):
            out = _run(
                f"rho={best_rho}, alpha={alpha}, beta={beta}, H=4, fold 24/25",
                test_season=24, train_seasons=[19, 20, 21, 22, 23],
                horizon=4, rho=best_rho, alpha=alpha, beta=beta,
            )
            ab_results.append(((alpha, beta), out))
    best_ab = max(ab_results, key=lambda x: x[1]["delta"] if x[1]["all_validity_pass"] else -1e18)[0]
    best_alpha, best_beta = best_ab
    print(f"\n>>> picked (alpha, beta) = {best_ab}")

    # Phase C: Full walk-forward at picked (ρ, α, β)
    print(f"\n=== Full walk-forward at rho={best_rho}, alpha={best_alpha}, beta={best_beta} ===")
    wf_results = []
    for fold in (
        {"train": [19, 20], "test": 21},
        {"train": [19, 20, 21], "test": 22},
        {"train": [19, 20, 21, 22], "test": 23},
        {"train": [19, 20, 21, 22, 23], "test": 24},
    ):
        out = _run(
            f"WALK-FORWARD fold test={fold['test']} (rho={best_rho}, alpha={best_alpha}, beta={best_beta})",
            test_season=fold["test"], train_seasons=fold["train"],
            horizon=4, rho=best_rho, alpha=best_alpha, beta=best_beta,
        )
        wf_results.append((fold["test"], out))

    print("\n\n========== SUMMARY ==========")
    print(f"\nρ sweep (fold 24/25, α=0.8, β=0.05):")
    for rho, r in rho_results:
        flag = "✓" if r["all_validity_pass"] else "✗"
        print(f"  ρ={rho}: {flag} validity, points={r['total_points']}, Δ template={r['delta']:+d}")
    print(f"\nα × β grid (fold 24/25, ρ={best_rho}):")
    for (a, b), r in ab_results:
        flag = "✓" if r["all_validity_pass"] else "✗"
        print(f"  α={a}, β={b}: {flag} validity, points={r['total_points']}, Δ template={r['delta']:+d}")
    print(f"\nWalk-forward (ρ={best_rho}, α={best_alpha}, β={best_beta}):")
    for season, r in wf_results:
        flag = "✓" if r["all_validity_pass"] else "✗"
        print(
            f"  20{season}/{(season+1)%100:02d}: {flag} validity, "
            f"points={r['total_points']}, Δ template={r['delta']:+d}"
        )


if __name__ == "__main__":
    main()
