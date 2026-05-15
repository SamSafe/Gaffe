"""DGW / BGW detection per (season, team, gameweek).

Reads from `dim_fixture` only — pure metadata, no PIT concerns. Used by
the Phase 5 chip-DP scheduler to identify chip-friendly GWs.

Definitions:
- A team has a DGW at GW w if it plays ≥ 2 fixtures in GW w.
- A team has a BGW at GW w if it plays 0 fixtures in GW w (typically due
  to FA Cup / UEFA scheduling pushes).
- "Squad-BGW" / "Squad-DGW" are derived per a specific squad: how many
  squad members have a BGW / DGW at GW w.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from sqlalchemy import select

from fpl_bot.db.models import DimFixture
from fpl_bot.db.session import session_scope


@dataclass(frozen=True)
class SeasonFixtureAnalytics:
    """Per-season precomputed analytics. Reused across chip heuristics."""

    season_id: int
    # team_id → list of GWs where this team plays
    gws_per_team: dict[int, list[int]]
    # (team_id, gw) → fixture count (0 = BGW; 2+ = DGW)
    fixture_count: dict[tuple[int, int], int]
    # All GWs that have at least one fixture in the season
    all_gws: list[int]


def load_fixture_analytics(season_id: int) -> SeasonFixtureAnalytics:
    """Pull per-(team, gw) fixture counts for one season from dim_fixture."""
    with session_scope() as s:
        rows = s.execute(
            select(
                DimFixture.gameweek,
                DimFixture.home_team_id,
                DimFixture.away_team_id,
            ).where(DimFixture.season_id == season_id)
        ).all()

    fixture_count: dict[tuple[int, int], int] = defaultdict(int)
    gws_per_team: dict[int, set[int]] = defaultdict(set)
    all_gws: set[int] = set()
    for r in rows:
        gw = int(r.gameweek)
        all_gws.add(gw)
        for team in (int(r.home_team_id), int(r.away_team_id)):
            fixture_count[(team, gw)] += 1
            gws_per_team[team].add(gw)

    return SeasonFixtureAnalytics(
        season_id=season_id,
        gws_per_team={t: sorted(gs) for t, gs in gws_per_team.items()},
        fixture_count=dict(fixture_count),
        all_gws=sorted(all_gws),
    )


def team_plays_dgw(analytics: SeasonFixtureAnalytics, team_id: int, gw: int) -> bool:
    """True if team_id has ≥ 2 fixtures in gw."""
    return analytics.fixture_count.get((team_id, gw), 0) >= 2


def team_plays_bgw(analytics: SeasonFixtureAnalytics, team_id: int, gw: int) -> bool:
    """True if team_id has 0 fixtures in gw (and the gw exists at all in the season)."""
    if gw not in analytics.all_gws:
        return False
    return analytics.fixture_count.get((team_id, gw), 0) == 0


def squad_blank_count(
    analytics: SeasonFixtureAnalytics,
    squad_team_ids: list[int],
    gw: int,
) -> int:
    """How many squad members have a blank GW at gw.

    `squad_team_ids` is per-squad-slot (not deduped): e.g. a 15-man squad
    with 3 City players gives 3 entries for the City team_id. Each is
    counted as a separate blank — matches FPL's view that "X of my 15
    players have no fixture".
    """
    return sum(
        1 for tid in squad_team_ids if team_plays_bgw(analytics, tid, gw)
    )


def squad_dgw_count(
    analytics: SeasonFixtureAnalytics,
    squad_team_ids: list[int],
    gw: int,
) -> int:
    """How many squad members have a DGW at gw (counted per slot)."""
    return sum(
        1 for tid in squad_team_ids if team_plays_dgw(analytics, tid, gw)
    )


def first_half_gws(analytics: SeasonFixtureAnalytics) -> list[int]:
    """GWs 1-19 inclusive that exist in this season (per FPL convention)."""
    return [gw for gw in analytics.all_gws if 1 <= gw <= 19]


def second_half_gws(analytics: SeasonFixtureAnalytics) -> list[int]:
    """GWs 20-38 inclusive."""
    return [gw for gw in analytics.all_gws if 20 <= gw <= 38]
