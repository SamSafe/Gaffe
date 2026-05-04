"""BPS event-source interface (§4.5–§4.6).

Round-5 fix: fbref's detailed events (tackles, interceptions, blocks,
dribbles, big chances, errors, passes-completed) are blocked by Cloudflare
anti-bot. We can't bypass it (§2.5: never evade detection). The BPS
simulator therefore falls back to empirical-residual imputation behind
this interface, so a future swap to sports-reference Premium or another
event source is a single-class change — the prediction/optimization
pipeline does not need rewriting.

Implementation of EmpiricalResidualEventSource is deferred to Phase 2.4
(BPS simulator). This module exists in Phase 1 as the contract.
"""
from __future__ import annotations

import datetime as dt
from typing import Protocol

import polars as pl


class EventSource(Protocol):
    """Per-player per-match detailed event counts.

    Implementations return rows with at least these columns when measured:
        tackles_won, interceptions, blocks, recoveries,
        dribbles_completed, big_chances_created, big_chances_missed,
        passes_completed, errors_leading_to_goal.
    Columns the source cannot measure are NULL; consumers (the BPS
    simulator) must impute them from empirical residuals or skip them.
    """

    def player_event_history(
        self, player_id: int, before: dt.datetime
    ) -> pl.DataFrame: ...


class EmpiricalResidualEventSource:
    """Phase-1 fallback when fbref / a Premium source is unavailable.

    For events we DO measure (cards, saves, goals, assists, CS, conceded),
    pass-through is exact via fact_player_match. For events we do NOT
    measure, the simulator samples per-position empirical means at run
    time and adds calibrated noise so the resulting bonus distribution
    matches historical assignments.

    The contract is honored: this class returns whatever measured columns
    exist (NULLs otherwise), and downstream code is responsible for the
    imputation policy.
    """

    def player_event_history(
        self, player_id: int, before: dt.datetime
    ) -> pl.DataFrame:
        raise NotImplementedError(
            "EmpiricalResidualEventSource: scaffold only; implementation in Phase 2.4"
        )


class FbrefEventSource:
    """Future implementation: when fbref / sports-reference Premium becomes
    available, populate fact_player_match_event from that source and read
    here. Until then, raises NotImplementedError so misconfiguration is
    loud, not silent."""

    def player_event_history(
        self, player_id: int, before: dt.datetime
    ) -> pl.DataFrame:
        raise NotImplementedError(
            "FbrefEventSource: requires sports-reference Premium subscription "
            "or another fbref-equivalent event source. Use EmpiricalResidualEventSource."
        )
