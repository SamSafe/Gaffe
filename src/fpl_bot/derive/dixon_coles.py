"""Bookmaker → team xG inversion via independent Poisson (§2.2).

Reads `fact_odds` (1X2 + totals 2.5), strips overround, fits (λ_home, λ_away)
under independent Poisson goal counts, derives clean-sheet probability as
exp(-λ_opponent), writes `fact_market_xg`.

Dixon-Coles low-score correction (τ) is NOT applied in this v1; the pure
Poisson inversion is calibration-anchor enough for the prediction layer.
The τ correction can be added later without API changes.
"""
from __future__ import annotations

import datetime as dt
import math

from scipy import optimize
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert

from fpl_bot.db.models import DimFixture, FactMarketXg, FactOdds
from fpl_bot.db.session import session_scope


def derive_market_xg_for_season(season_id: int) -> dict[str, int]:
    """Run the inverter for every fixture in the season that has odds."""
    counts = {
        "fixtures_with_odds": 0,
        "rows_written": 0,
        "skipped_no_odds": 0,
        "skipped_fit_failed": 0,
    }

    with session_scope() as s:
        fixtures = (
            s.execute(select(DimFixture).where(DimFixture.season_id == season_id))
            .scalars()
            .all()
        )

        for fixture in fixtures:
            odds_rows = s.execute(
                select(
                    FactOdds.bookmaker,
                    FactOdds.market,
                    FactOdds.selection,
                    FactOdds.decimal_odds,
                    FactOdds.event_time,
                ).where(FactOdds.fixture_id == fixture.fixture_id)
            ).all()
            if not odds_rows:
                counts["skipped_no_odds"] += 1
                continue

            agg = _aggregate_implied_probs(odds_rows)
            if agg["p_home"] is None or agg["p_draw"] is None or agg["p_away"] is None:
                counts["skipped_no_odds"] += 1
                continue

            h, d, _a = _strip_overround(agg["p_home"], agg["p_draw"], agg["p_away"])
            o = _strip_two_way(agg.get("p_over"), agg.get("p_under"), default=0.55)

            try:
                lam_h, lam_a = fit_lambdas(h, d, o)
            except RuntimeError:
                counts["skipped_fit_failed"] += 1
                continue

            cs_home = math.exp(-lam_a)
            cs_away = math.exp(-lam_h)
            source_recorded_at = agg["max_event_time"] or dt.datetime.now(dt.UTC)

            for team_id, lam, cs_prob in (
                (fixture.home_team_id, lam_h, cs_home),
                (fixture.away_team_id, lam_a, cs_away),
            ):
                stmt = pg_insert(FactMarketXg).values(
                    fixture_id=fixture.fixture_id,
                    team_id=team_id,
                    source_recorded_at=source_recorded_at,
                    lambda_=round(lam, 4),
                    cs_prob=round(cs_prob, 5),
                )
                stmt = stmt.on_conflict_do_nothing(
                    index_elements=["fixture_id", "team_id", "source_recorded_at"]
                )
                s.execute(stmt)
                counts["rows_written"] += 1

            counts["fixtures_with_odds"] += 1

    return counts


def fit_lambdas(
    p_home_observed: float,
    p_draw_observed: float,
    p_over_2_5_observed: float,
    *,
    max_goals: int = 8,
    init: tuple[float, float] = (1.5, 1.0),
) -> tuple[float, float]:
    """Solve for (λ_home, λ_away) under independent Poisson to match observed
    bookmaker market probabilities (1X2 + totals 2.5). Nelder-Mead minimization
    on squared residuals."""

    def loss(params: list[float]) -> float:
        lh, la = params
        if lh <= 0.05 or la <= 0.05 or lh > 6 or la > 6:
            return 1e6
        ph, pd, _, po = _market_probs(lh, la, max_goals)
        return (
            (ph - p_home_observed) ** 2
            + (pd - p_draw_observed) ** 2
            + (po - p_over_2_5_observed) ** 2
        )

    result = optimize.minimize(
        loss, x0=list(init), method="Nelder-Mead", options={"xatol": 1e-4, "fatol": 1e-7}
    )
    if not result.success and result.fun > 1e-3:
        raise RuntimeError(f"Lambda fit did not converge: {result.message}")
    return float(result.x[0]), float(result.x[1])


def _market_probs(
    lam_h: float, lam_a: float, max_goals: int
) -> tuple[float, float, float, float]:
    """Returns (p_home, p_draw, p_away, p_over_2.5) under independent Poisson."""
    h_pmf = [math.exp(-lam_h) * lam_h**h / math.factorial(h) for h in range(max_goals + 1)]
    a_pmf = [math.exp(-lam_a) * lam_a**a / math.factorial(a) for a in range(max_goals + 1)]
    p_home = p_draw = p_away = p_over = 0.0
    for h in range(max_goals + 1):
        for a in range(max_goals + 1):
            j = h_pmf[h] * a_pmf[a]
            if h > a:
                p_home += j
            elif h == a:
                p_draw += j
            else:
                p_away += j
            if h + a > 2:
                p_over += j
    return p_home, p_draw, p_away, p_over


def _aggregate_implied_probs(odds_rows: list) -> dict:
    """Mean implied probability per (market, selection) across bookmakers."""
    sums: dict[tuple[str, str], list[float]] = {}
    max_event_time: dt.datetime | None = None
    for row in odds_rows:
        odds = float(row.decimal_odds)
        if odds <= 1.0:
            continue
        sums.setdefault((row.market, row.selection), []).append(1.0 / odds)
        if max_event_time is None or row.event_time > max_event_time:
            max_event_time = row.event_time

    def avg(market: str, selection: str) -> float | None:
        vals = sums.get((market, selection))
        return sum(vals) / len(vals) if vals else None

    return {
        "p_home": avg("1X2", "home"),
        "p_draw": avg("1X2", "draw"),
        "p_away": avg("1X2", "away"),
        "p_over": avg("totals_2.5", "over"),
        "p_under": avg("totals_2.5", "under"),
        "max_event_time": max_event_time,
    }


def _strip_overround(p_h: float, p_d: float, p_a: float) -> tuple[float, float, float]:
    total = p_h + p_d + p_a
    if total <= 0:
        return 0.0, 0.0, 0.0
    return p_h / total, p_d / total, p_a / total


def _strip_two_way(p_a: float | None, p_b: float | None, *, default: float) -> float:
    if p_a is None and p_b is None:
        return default
    if p_a is None:
        return 1 - (p_b / 1) if p_b else default  # type: ignore[operator]
    if p_b is None:
        return p_a / 1
    total = p_a + p_b
    return p_a / total if total > 0 else default
