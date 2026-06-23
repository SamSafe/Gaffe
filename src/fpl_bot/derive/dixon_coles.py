"""Bookmaker → team xG inversion via independent Poisson (§2.2).

Selects coherent point-in-time bookmaker snapshots and removes overround per
book before forming a consensus. Production fits (λ_home, λ_away) from 1X2 +
totals 2.5; an opt-in `all` mode also consumes alternate totals and Asian
handicaps for shadow evaluation. Derived clean-sheet probability is
exp(-λ_opponent).

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


def derive_market_xg_for_season(
    season_id: int,
    *,
    as_of: dt.datetime | None = None,
    fit_mode: str = "legacy",
) -> dict[str, int]:
    """Run the inverter using the latest complete markets at ``as_of``.

    ``legacy`` remains the accepted production fitter. ``all`` is retained for
    shadow evaluation after mixed downstream results across historical folds.
    """
    if as_of is not None and as_of.utcoffset() is None:
        raise ValueError("as_of must be timezone-aware")
    if fit_mode not in {"legacy", "all"}:
        raise ValueError("fit_mode must be 'legacy' or 'all'")
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
                    FactOdds.quote_time,
                ).where(FactOdds.fixture_id == fixture.fixture_id)
            ).all()
            if not odds_rows:
                counts["skipped_no_odds"] += 1
                continue

            consensus = _consensus_market_probs(odds_rows, as_of=as_of)
            h2h = consensus["markets"].get("1X2")
            if h2h is None:
                counts["skipped_no_odds"] += 1
                continue

            h = h2h["home"]
            d = h2h["draw"]
            totals_2_5 = consensus["markets"].get("totals_2.5")
            o = totals_2_5["over"] if totals_2_5 is not None else 0.55

            try:
                if fit_mode == "all":
                    lam_h, lam_a, _fit_loss = fit_lambdas_from_markets(
                        consensus["markets"]
                    )
                else:
                    lam_h, lam_a = fit_lambdas(h, d, o)
            except RuntimeError:
                counts["skipped_fit_failed"] += 1
                continue

            cs_home = math.exp(-lam_a)
            cs_away = math.exp(-lam_h)
            source_recorded_at = consensus["max_quote_time"] or dt.datetime.now(dt.UTC)

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
                stmt = stmt.on_conflict_do_update(
                    index_elements=["fixture_id", "team_id", "source_recorded_at"],
                    set_={"lambda": round(lam, 4), "cs_prob": round(cs_prob, 5)},
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


def fit_lambdas_from_markets(
    markets: dict[str, dict[str, float]],
    *,
    max_goals: int = 12,
    init: tuple[float, float] | None = None,
) -> tuple[float, float, float]:
    """Fit team goal rates to all supported consensus markets.

    Supported inputs are 1X2, totals lines, and Asian handicaps encoded from
    the home-team perspective. Quarter lines use exact half-stake settlement
    rather than pretending they are ordinary binary thresholds.

    Returns ``(lambda_home, lambda_away, mean_squared_market_error)``.
    """
    supported = {
        market: probabilities
        for market, probabilities in markets.items()
        if _required_selections(market) is not None
    }
    if "1X2" not in supported:
        raise RuntimeError("A complete 1X2 market is required")

    if init is None:
        h2h = supported["1X2"]
        totals_2_5 = supported.get("totals_2.5")
        init = fit_lambdas(
            h2h["home"],
            h2h["draw"],
            totals_2_5["over"] if totals_2_5 is not None else 0.55,
            max_goals=min(max_goals, 10),
        )

    def loss(params: list[float] | tuple[float, float]) -> float:
        lh, la = float(params[0]), float(params[1])
        if lh <= 0.05 or la <= 0.05 or lh > 6.0 or la > 6.0:
            return 1e6
        model = _model_market_probabilities(lh, la, supported, max_goals=max_goals)
        market_losses: list[float] = []
        for market, observed in supported.items():
            predicted = model.get(market)
            required = _required_selections(market)
            if predicted is None or required is None:
                continue
            market_losses.append(
                sum((predicted[s] - observed[s]) ** 2 for s in required) / len(required)
            )
        return sum(market_losses) / len(market_losses) if market_losses else 1e6

    result = optimize.minimize(
        loss,
        x0=list(init),
        method="Nelder-Mead",
        options={"xatol": 1e-5, "fatol": 1e-10, "maxiter": 2000},
    )
    if not result.success and result.fun > 1e-3:
        raise RuntimeError(f"Multi-market lambda fit did not converge: {result.message}")
    return float(result.x[0]), float(result.x[1]), float(result.fun)


def _split_asian_line(line: float) -> tuple[float, ...]:
    """Expand a quarter line into its two adjacent half/integer lines."""
    doubled = line * 2.0
    if math.isclose(doubled, round(doubled), abs_tol=1e-9):
        return (line,)
    return (line - 0.25, line + 0.25)


def _fair_binary_probability(
    outcomes: list[tuple[float, float]],
    *,
    line: float,
    side: str,
) -> float:
    """Fair implied probability under Asian full/half win-loss settlement.

    ``outcomes`` contains `(result_value, probability)` where totals use total
    goals and handicaps use home goal difference. ``side`` is `high` for over
    or home and `low` for under or away.
    """
    split_lines = _split_asian_line(line)
    win_mass = 0.0
    loss_mass = 0.0
    stake = 1.0 / len(split_lines)
    for result_value, probability in outcomes:
        for component_line in split_lines:
            margin = result_value - component_line
            if side == "low":
                margin = -margin
            if margin > 1e-9:
                win_mass += probability * stake
            elif margin < -1e-9:
                loss_mass += probability * stake
            # A push contributes neither profit nor loss.
    denominator = win_mass + loss_mass
    return win_mass / denominator if denominator > 0.0 else 0.5


def _score_outcomes(
    lam_h: float,
    lam_a: float,
    *,
    max_goals: int,
) -> list[tuple[int, int, float]]:
    h_pmf = [
        math.exp(-lam_h) * lam_h**goals / math.factorial(goals)
        for goals in range(max_goals + 1)
    ]
    a_pmf = [
        math.exp(-lam_a) * lam_a**goals / math.factorial(goals)
        for goals in range(max_goals + 1)
    ]
    outcomes = [
        (home, away, h_pmf[home] * a_pmf[away])
        for home in range(max_goals + 1)
        for away in range(max_goals + 1)
    ]
    total_mass = sum(probability for _, _, probability in outcomes)
    return [
        (home, away, probability / total_mass)
        for home, away, probability in outcomes
    ]


def _model_market_probabilities(
    lam_h: float,
    lam_a: float,
    markets: dict[str, dict[str, float]],
    *,
    max_goals: int = 12,
) -> dict[str, dict[str, float]]:
    """Model-equivalent fair probabilities for the requested market keys."""
    scores = _score_outcomes(lam_h, lam_a, max_goals=max_goals)
    out: dict[str, dict[str, float]] = {}

    if "1X2" in markets:
        home = sum(p for h, a, p in scores if h > a)
        draw = sum(p for h, a, p in scores if h == a)
        away = sum(p for h, a, p in scores if h < a)
        out["1X2"] = {"home": home, "draw": draw, "away": away}

    totals_outcomes = [(float(h + a), p) for h, a, p in scores]
    handicap_outcomes = [(float(h - a), p) for h, a, p in scores]
    for market in markets:
        if market.startswith("totals_"):
            line = float(market.removeprefix("totals_"))
            p_over = _fair_binary_probability(totals_outcomes, line=line, side="high")
            out[market] = {"over": p_over, "under": 1.0 - p_over}
        elif market.startswith("ah_"):
            home_line = float(market.removeprefix("ah_"))
            # Home wins when goal difference + handicap > 0, equivalently
            # goal difference > -handicap.
            threshold = -home_line
            p_home = _fair_binary_probability(
                handicap_outcomes,
                line=threshold,
                side="high",
            )
            out[market] = {"home": p_home, "away": 1.0 - p_home}
    return out


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
    """Backward-compatible view of the coherent, de-vigged consensus."""
    consensus = _consensus_market_probs(odds_rows)
    h2h = consensus["markets"].get("1X2", {})
    totals = consensus["markets"].get("totals_2.5", {})

    return {
        "p_home": h2h.get("home"),
        "p_draw": h2h.get("draw"),
        "p_away": h2h.get("away"),
        "p_over": totals.get("over"),
        "p_under": totals.get("under"),
        "max_event_time": consensus["max_quote_time"],
    }


def _required_selections(market: str) -> tuple[str, ...] | None:
    if market == "1X2":
        return ("home", "draw", "away")
    if market.startswith("totals_"):
        return ("over", "under")
    if market.startswith("ah_"):
        return ("home", "away")
    return None


def _consensus_market_probs(
    odds_rows: list,
    *,
    as_of: dt.datetime | None = None,
) -> dict:
    """Latest complete, per-book de-vigged consensus for each market.

    Selections are first grouped by an exact `(bookmaker, market, quote_time)`
    snapshot. Incomplete snapshots are discarded. The latest eligible complete
    snapshot is selected per bookmaker/market, de-vigged within that book, and
    only then averaged across bookmakers.
    """
    if as_of is not None and as_of.utcoffset() is None:
        raise ValueError("as_of must be timezone-aware")

    snapshots: dict[tuple[str, str, dt.datetime], dict[str, float]] = {}
    for row in odds_rows:
        odds = float(row.decimal_odds)
        if odds <= 1.0 or not math.isfinite(odds):
            continue
        quote_time = getattr(row, "quote_time", None) or row.event_time
        if quote_time.utcoffset() is None:
            quote_time = quote_time.replace(tzinfo=dt.UTC)
        if as_of is not None and quote_time > as_of:
            continue
        key = (str(row.bookmaker), str(row.market), quote_time)
        snapshots.setdefault(key, {})[str(row.selection)] = odds

    latest: dict[tuple[str, str], tuple[dt.datetime, dict[str, float]]] = {}
    for (bookmaker, market, quote_time), selections in snapshots.items():
        required = _required_selections(market)
        if required is None or any(selection not in selections for selection in required):
            continue
        key = (bookmaker, market)
        previous = latest.get(key)
        if previous is None or quote_time > previous[0]:
            latest[key] = (quote_time, selections)

    per_market: dict[str, list[dict[str, float]]] = {}
    max_quote_time: dt.datetime | None = None
    for (_bookmaker, market), (quote_time, selections) in latest.items():
        required = _required_selections(market)
        if required is None:
            continue
        inverse = {selection: 1.0 / selections[selection] for selection in required}
        overround = sum(inverse.values())
        if overround <= 0.0:
            continue
        devigged = {selection: value / overround for selection, value in inverse.items()}
        per_market.setdefault(market, []).append(devigged)
        if max_quote_time is None or quote_time > max_quote_time:
            max_quote_time = quote_time

    consensus: dict[str, dict[str, float]] = {}
    bookmaker_counts: dict[str, int] = {}
    for market, books in per_market.items():
        required = _required_selections(market)
        if required is None:
            continue
        consensus[market] = {
            selection: sum(book[selection] for book in books) / len(books)
            for selection in required
        }
        bookmaker_counts[market] = len(books)

    return {
        "markets": consensus,
        "bookmaker_counts": bookmaker_counts,
        "max_quote_time": max_quote_time,
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
