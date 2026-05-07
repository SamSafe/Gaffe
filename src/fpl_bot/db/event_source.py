"""BPS event-source interface (§4.5–§4.6, Phase 2.4).

Round-5 fix: fbref's detailed events (tackles, interceptions, blocks,
dribbles, big chances, errors, passes-completed) are blocked by Cloudflare
anti-bot. The BPS simulator therefore falls back to empirical-residual
imputation behind this interface, so a future swap to sports-reference
Premium or another event source is a single-class change.

The interface is `simulate_unmodeled_bps(position, minutes_played, rng)`
returning the BPS contribution from events not directly modeled. Both
implementations conform to this contract:
  - EmpiricalResidualEventSource: samples from fitted per-(position, minutes-bucket)
    residual distribution.
  - FbrefEventSource (future): would simulate per-event-type counts from
    rolling rates in fact_player_match_event and apply the BPS rule table.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
import polars as pl

# Bucket midpoints per design review round-1 decision
MINUTES_BUCKET_MIDPOINTS: tuple[int, ...] = (0, 30, 70, 90)


def minutes_to_bucket(minutes: int) -> int:
    """Map actual minutes to bucket midpoint: 0, 30, 70, or 90."""
    if minutes <= 0:
        return 0
    if minutes < 60:
        return 30
    if minutes < 90:
        return 70
    return 90


class EventSource(Protocol):
    """BPS contribution from events not directly modeled (tackles,
    interceptions, blocks, recoveries, dribbles, big chances created/missed,
    errors leading to goal, passes-completed, own goals).

    Per design: implementations MAY also provide direct per-match event
    counts via player_event_history, but the BPS simulator only requires
    simulate_unmodeled_bps. Splitting these into a separate Protocol method
    keeps the empirical-residual fallback cheap (no per-player query).
    """

    def simulate_unmodeled_bps(
        self,
        position: str,
        minutes_played: int,
        rng: np.random.Generator,
    ) -> float: ...

    def fit(self, train_df: pl.DataFrame) -> None: ...


# Bucket boundaries used to map actual minutes → bucket label for residual fitting.
# Same as minutes_to_bucket, but exposed as a labeling helper.
def _minutes_bucket_label(minutes: int) -> str:
    return f"b{minutes_to_bucket(minutes)}"


@dataclass
class EmpiricalResidualEventSource:
    """Phase 2.4 v1 fallback when fbref / a Premium source is unavailable.

    Fits a Gaussian per (position, minutes_bucket) on the residual
        actual_BPS - simulated_BPS_from_known_events
    where simulated_BPS_from_known_events is computed by applying the
    BPS rule table to the actual events visible in fact_player_match
    (goals, assists, CS via team-level goals_conceded, minutes, cards,
    saves, penalty events). The residual captures all un-modeled events
    in aggregate.

    Sampling at simulation time draws from N(mu, sigma) for the matching
    bucket; clamps to [-30, +50] to keep tail outliers from blowing up
    individual MC realizations.
    """

    # mu, sigma keyed on (position, bucket_int). Position ∈ {GKP, DEF, MID, FWD};
    # bucket_int ∈ {0, 30, 70, 90}.
    _residual_stats: dict[tuple[str, int], tuple[float, float]] = field(
        default_factory=dict
    )
    _fitted: bool = False

    # Clamp range for sampled residuals (prevents pathological tails)
    _clip_low: float = -30.0
    _clip_high: float = 50.0

    def fit(self, train_df: pl.DataFrame) -> None:
        """Compute per-(position, bucket) mean + std from a training DataFrame.

        Required columns: position_code, minutes, residual_bps.
        residual_bps = actual_bps - simulated_bps_from_known_events; the
        caller is responsible for computing it (see fit_from_player_match).
        """
        from collections import defaultdict

        groups: dict[tuple[str, int], list[float]] = defaultdict(list)
        for row in train_df.iter_rows(named=True):
            pos = row.get("position_code")
            minutes = row.get("minutes")
            res = row.get("residual_bps")
            if pos is None or minutes is None or res is None:
                continue
            groups[(pos, minutes_to_bucket(int(minutes)))].append(float(res))

        for key, vals in groups.items():
            if len(vals) < 5:
                # too few; fall back to overall mean
                continue
            arr = np.asarray(vals, dtype=np.float64)
            self._residual_stats[key] = (float(arr.mean()), float(arr.std(ddof=1)))

        # Fallback: overall mean/std for any (position, bucket) we never observed
        all_vals = [v for vs in groups.values() for v in vs]
        if all_vals:
            arr = np.asarray(all_vals, dtype=np.float64)
            self._fallback_mu = float(arr.mean())
            self._fallback_sigma = float(arr.std(ddof=1))
        else:
            self._fallback_mu = 0.0
            self._fallback_sigma = 5.0

        self._fitted = True

    def simulate_unmodeled_bps(
        self,
        position: str,
        minutes_played: int,
        rng: np.random.Generator,
    ) -> float:
        if not self._fitted:
            raise RuntimeError(
                "EmpiricalResidualEventSource: must call fit() before sampling"
            )
        if minutes_played <= 0:
            return 0.0  # bench players accumulate no BPS, residual or otherwise
        bucket = minutes_to_bucket(int(minutes_played))
        key = (position, bucket)
        mu, sigma = self._residual_stats.get(
            key, (self._fallback_mu, self._fallback_sigma)
        )
        sample = float(rng.normal(mu, sigma))
        return float(np.clip(sample, self._clip_low, self._clip_high))

    @property
    def n_buckets_fitted(self) -> int:
        return len(self._residual_stats)


class FbrefEventSource:
    """Future implementation: when fbref / sports-reference Premium becomes
    available, populate fact_player_match_event from that source and read
    here. Per-event counts are sampled from rolling rates and scored against
    the BPS rule table directly — same `simulate_unmodeled_bps` contract,
    no upstream/downstream API changes."""

    def fit(self, train_df: pl.DataFrame) -> None:
        raise NotImplementedError(
            "FbrefEventSource: requires sports-reference Premium subscription "
            "or another fbref-equivalent event source. "
            "Use EmpiricalResidualEventSource until then."
        )

    def simulate_unmodeled_bps(
        self,
        position: str,
        minutes_played: int,
        rng: np.random.Generator,
    ) -> float:
        raise NotImplementedError("FbrefEventSource: not yet implemented")
