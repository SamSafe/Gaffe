"""Smoke test: H=6 + chips on fold 24/25 (1 fold only) at picked ρ=1.0,
α=0.5, β=0.0. Goal: verify the weekly-aggregate chip refactor makes H=6 +
chips tractable.
"""
from __future__ import annotations

from fpl_bot.eval.milp_backtest import (
    backtest_season,
    format_validity_report,
    format_performance_report,
)


def main() -> None:
    print("=== H=6 + chips smoke test (fold 24/25, ρ=1.0, α=0.5, β=0.0) ===", flush=True)
    r = backtest_season(
        test_season=24,
        train_seasons=[19, 20, 21, 22, 23],
        horizon=6,
        rho=1.0,
        alpha=0.5,
        beta=0.0,
        enable_chips=True,
        n_iterations=200,
    )
    print(format_validity_report(r))
    print()
    print(format_performance_report(r))


if __name__ == "__main__":
    main()
