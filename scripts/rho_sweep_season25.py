"""rho sweep on fold 25/26 — the ONLY fold where the EO term is real.

`fact_eo_snapshot` and `fact_player_status.selected_by_percent` exist for
seasons 25 and 26 only. Season 26 is live (2 played GWs), so season 25 is the
sole backtestable fold with a template to be differential against.

This matters for reading older tuning runs: they swept rho on fold 24/25, and
the backtest's EO lookup was unscoped, so it applied TODAY's ownership to a
2024-25 season — a different set of clubs and players. Any rho picked there
was picked against the wrong template. eo.py is now season-scoped.

rho = 0 is included deliberately: it tests whether the EO term earns its
place at all, rather than assuming it does and only tuning its size.
"""
from __future__ import annotations

from fpl_bot.eval.milp_backtest import (
    backtest_season,
    format_performance_report,
    format_validity_report,
)

RHOS = (0.0, 0.7, 1.0, 1.2)


def main() -> None:
    results = []
    for rho in RHOS:
        label = f"fold 25/26, H=6+chips, rho={rho}"
        print(f"\n=== {label} ===", flush=True)
        r = backtest_season(
            test_season=25,
            train_seasons=[19, 20, 21, 22, 23, 24],
            horizon=6,
            rho=rho,
            alpha=0.8,
            beta=0.05,
            enable_chips=True,
            n_iterations=200,
        )
        print(format_validity_report(r), flush=True)
        print(format_performance_report(r), flush=True)
        results.append(
            (
                rho,
                r.perf_total_points,
                r.perf_total_points - r.perf_template_total_points,
                r.all_validity_gates_pass,
            )
        )

    print("\n\n========== RHO SWEEP, fold 25/26, season-scoped EO ==========",
          flush=True)
    for rho, pts, delta, ok in results:
        flag = "PASS" if ok else "FAIL"
        print(f"  rho={rho}: {flag} validity, points={pts}, delta vs template={delta:+d}",
              flush=True)
    best = max(results, key=lambda t: t[1])
    print(f"\n  best by total points: rho={best[0]} ({best[1]} pts)", flush=True)
    print("  NOTE: one fold, one season. Treat a small gap as noise.", flush=True)


if __name__ == "__main__":
    main()
