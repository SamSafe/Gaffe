"""Tests for DGW/BGW detection used by the Phase 5 chip-DP scheduler."""
from __future__ import annotations

from fpl_bot.optim.fixture_analytics import (
    SeasonFixtureAnalytics,
    first_half_gws,
    second_half_gws,
    squad_blank_count,
    squad_dgw_count,
    team_plays_bgw,
    team_plays_dgw,
)


def _make_analytics(
    season_id: int = 23,
    fixtures: list[tuple[int, int, int]] | None = None,
) -> SeasonFixtureAnalytics:
    """fixtures: list of (gw, home_team_id, away_team_id)."""
    fixtures = fixtures or []
    fixture_count: dict[tuple[int, int], int] = {}
    gws_per_team: dict[int, set[int]] = {}
    all_gws: set[int] = set()
    for gw, h, a in fixtures:
        all_gws.add(gw)
        for t in (h, a):
            fixture_count[(t, gw)] = fixture_count.get((t, gw), 0) + 1
            gws_per_team.setdefault(t, set()).add(gw)
    return SeasonFixtureAnalytics(
        season_id=season_id,
        gws_per_team={t: sorted(gs) for t, gs in gws_per_team.items()},
        fixture_count=fixture_count,
        all_gws=sorted(all_gws),
    )


def test_team_plays_dgw_detects_double():
    a = _make_analytics(fixtures=[
        (5, 1, 2),
        (5, 1, 3),  # team 1 plays twice in GW5 → DGW
        (5, 4, 5),
    ])
    assert team_plays_dgw(a, team_id=1, gw=5)
    assert not team_plays_dgw(a, team_id=2, gw=5)
    assert not team_plays_dgw(a, team_id=4, gw=5)


def test_team_plays_bgw_when_no_fixture_for_team():
    a = _make_analytics(fixtures=[
        (5, 1, 2),
        (5, 3, 4),
        # team 5 has no fixture in GW5 → BGW (but GW5 exists)
    ])
    # team 5 not in fixture_count → bgw=True iff gw in all_gws
    assert team_plays_bgw(a, team_id=5, gw=5)
    assert not team_plays_bgw(a, team_id=1, gw=5)


def test_team_plays_bgw_only_if_gw_exists():
    a = _make_analytics(fixtures=[(5, 1, 2)])
    # GW10 doesn't exist in this season at all
    assert not team_plays_bgw(a, team_id=1, gw=10)


def test_squad_blank_count():
    a = _make_analytics(fixtures=[
        (5, 1, 2),
        # teams 3, 4, 5 have BGW in GW5
    ])
    # Squad with team_ids [1, 1, 1, 2, 3, 3, 4, 5]
    # In GW5: 1 plays, 2 plays, 3,4,5 blank
    # Count: 3 entries from {3, 3, 4} blank + 1 from {5} = need to count 3 (player_id-3 ×2 ≠ counts both), 4 ×1, 5 ×1 = 4 blanks
    squad = [1, 1, 1, 2, 3, 3, 4, 5]
    assert squad_blank_count(a, squad, gw=5) == 4


def test_squad_dgw_count():
    a = _make_analytics(fixtures=[
        (10, 1, 2),
        (10, 1, 3),  # team 1 has DGW
        (10, 4, 5),
    ])
    squad = [1, 1, 2, 3, 4, 5]  # 2 City players, 1 each of teams 2-5
    # In GW10: only team 1 has DGW; 2 squad slots are team 1 → 2 DGW slots
    assert squad_dgw_count(a, squad, gw=10) == 2


def test_first_half_second_half_split():
    a = _make_analytics(fixtures=[(gw, 1, 2) for gw in range(1, 39)])
    fh = first_half_gws(a)
    sh = second_half_gws(a)
    assert fh == list(range(1, 20))
    assert sh == list(range(20, 39))


def test_first_half_excludes_missing_gws():
    a = _make_analytics(fixtures=[(gw, 1, 2) for gw in [1, 2, 5, 19, 20]])
    fh = first_half_gws(a)
    assert fh == [1, 2, 5, 19]
    assert second_half_gws(a) == [20]
