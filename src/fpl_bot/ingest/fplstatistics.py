"""FPLStatistics fallback EO source — DEFERRED.

Status (Phase 1B closure):
  - https://www.fplstatistics.co.uk and https://fplstatistics.co.uk both
    persistently time out from this environment (60s timeout, multiple
    attempts, no response).
  - Could be: site is down, geo-blocked, or rate-limit-aggressive.
  - Per §2.5 we do not work around connectivity issues.

This module is a structural stub. If the site comes back online and offers
clean HTML/JSON, populate fetch_raw_fplstatistics + parse_raw_fplstatistics
following the two-layer pattern. Until then, EO falls back to
fpl_api_approx (see livefpl.py docstring).
"""
from __future__ import annotations


def fetch_raw_fplstatistics(*args, **kwargs):
    raise NotImplementedError(
        "FPLStatistics ingest deferred — site unreachable as of Phase 1B closure. "
        "See module docstring."
    )


def parse_raw_fplstatistics(*args, **kwargs):
    raise NotImplementedError("FPLStatistics ingest deferred — see module docstring.")
