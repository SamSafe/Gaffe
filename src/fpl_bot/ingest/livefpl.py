"""LiveFPL top-10k EO ingest — DEFERRED.

Status (Phase 1B closure):
  - livefpl.net is reachable (HTTP 200, no robots.txt → no path restrictions)
  - The site is a JavaScript SPA; data is loaded via XHR calls to internal
    endpoints, NOT exposed via a documented public API
  - Reverse-engineering those endpoints is brittle and §2.5-adjacent
    (not an anti-bot bypass, but the spirit of "use sanctioned sources" leans
    against it). We do not pursue it without explicit operator decision.

Path forward (when revisited):
  1. Inspect browser network panel for the actual data XHR endpoint(s)
  2. Confirm those endpoints are intended for public consumption
  3. If yes, implement fetch_raw_livefpl reading those JSON endpoints
  4. If no, defer further

Until then, top-10k EO is approximated from the FPL API: overall ownership
× captain-percentage heuristic, written to `fact_eo_snapshot` with
`source = 'fpl_api_approx'`. The optimizer's rank-utility (Phase 4) uses
whichever EO source is highest fidelity at run time.
"""
from __future__ import annotations


def fetch_raw_livefpl(*args, **kwargs):
    raise NotImplementedError(
        "LiveFPL ingest deferred — see module docstring. "
        "Top-10k EO falls back to fpl_api_approx until a clean public endpoint is wired up."
    )


def parse_raw_livefpl(*args, **kwargs):
    raise NotImplementedError("LiveFPL ingest deferred — see module docstring.")
