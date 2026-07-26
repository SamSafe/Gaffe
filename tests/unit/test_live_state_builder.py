"""Tests for Phase 6 live state builder + status filter."""
from __future__ import annotations

import datetime as dt
import json

import pytest
from sqlalchemy import delete

from fpl_bot.config import settings
from fpl_bot.db.models import FactPlayerStatus, FactUserTeamSnapshot
from fpl_bot.db.session import session_scope
from fpl_bot.live.state_builder import (
    DROP_STATUSES,
    LiveStatusOverrides,
    load_status_overrides,
    load_user_sell_prices,
    load_user_state,
)
from fpl_bot.optim.state import INITIAL_BUDGET_TENTHS

# Every test here reads/writes the dev Postgres (autouse cleanup + inserts).
pytestmark = pytest.mark.integration


def _purge_test_rows():
    """Clean up between tests; uses test-only season_id=99 and player_ids ≥ 90000."""
    with session_scope() as s:
        s.execute(delete(FactUserTeamSnapshot).where(FactUserTeamSnapshot.season_id == 99))
        s.execute(delete(FactPlayerStatus).where(FactPlayerStatus.season_id == 99))


@pytest.fixture(autouse=True)
def _cleanup():
    _purge_test_rows()
    yield
    _purge_test_rows()


def _insert_user_team(
    *,
    season_id: int = 99,
    gameweek: int = 5,
    team_id: int = 12345,
    players: list[tuple[int, int, int, bool, bool]] | None = None,
    bank: int = 25,
    ft: int = 2,
    chips_used: list[str] | None = None,
    sell_prices: dict[int, int] | None = None,
) -> None:
    """players: list of (player_id, purchase_price, position, is_captain, is_vice)."""
    if players is None:
        players = [(90000 + i, 50 + i, i + 1, i == 0, i == 1) for i in range(15)]
    chips_used_json = json.dumps(chips_used or [])
    now = dt.datetime.now(dt.UTC)
    with session_scope() as s:
        for pid, price, pos, cap, vc in players:
            sell_price = sell_prices.get(pid, price) if sell_prices else price
            s.add(
                FactUserTeamSnapshot(
                    season_id=season_id,
                    gameweek=gameweek,
                    team_id=team_id,
                    player_id=pid,
                    recorded_at=now,
                    purchase_price_tenths=price,
                    selling_price_tenths=sell_price,
                    multiplier=2 if cap else (3 if False else 1),
                    is_captain=cap,
                    is_vice=vc,
                    position=pos,
                    bank_tenths=bank,
                    free_transfers=ft,
                    chips_used_json=chips_used_json,
                )
            )


def _insert_status(
    *,
    player_id: int,
    status_code: str,
    cop: int | None = None,
    season_id: int = 99,
) -> None:
    now = dt.datetime.now(dt.UTC)
    with session_scope() as s:
        s.add(
            FactPlayerStatus(
                player_id=player_id,
                recorded_at=now,
                season_id=season_id,
                position_code="MID",
                team_id=1,
                price_tenths=100,
                status_code=status_code,
                news=None,
                chance_of_playing_next_round=cop,
                selected_by_percent=10.0,
                event_time=now,
            )
        )


def test_load_user_state_basic():
    _insert_user_team(season_id=99, gameweek=5, team_id=12345)
    state = load_user_state(season_id=99, gameweek=5, team_id=12345)
    assert state.season_id == 99
    assert state.gameweek == 5
    assert len(state.squad) == 15
    assert state.bank == 25
    assert state.free_transfers == 2
    assert len(state.cost_basis) == 15


def test_load_user_state_applies_local_override(tmp_path, monkeypatch):
    override_path = tmp_path / "live_state_overrides.yaml"
    override_path.write_text(
        """
99:
  5:
    12345:
      bank_tenths: 61
      free_transfers: 4
      chips_used: [WC1, FH1]
      cost_basis:
        90000: 42
""".strip()
    )
    monkeypatch.setattr(settings, "live_state_overrides_path", override_path)
    _insert_user_team(season_id=99, gameweek=5, team_id=12345)

    state = load_user_state(season_id=99, gameweek=5, team_id=12345)

    assert state.bank == 61
    assert state.free_transfers == 4
    assert state.chips_used == frozenset({"WC1", "FH1"})
    assert state.cost_basis[90000] == 42


def test_load_user_sell_prices_uses_latest_snapshot_values():
    _insert_user_team(
        season_id=99,
        gameweek=5,
        team_id=12345,
        sell_prices={90000: 49, 90001: 53},
    )
    sell_prices = load_user_sell_prices(season_id=99, gameweek=6, team_id=12345)
    assert sell_prices[90000] == 49
    assert sell_prices[90001] == 53


def test_load_user_state_missing_raises():
    with pytest.raises(ValueError, match="No user-team snapshot"):
        load_user_state(season_id=99, gameweek=5, team_id=99999)


def test_gw1_with_no_snapshot_cold_starts():
    """At a season's first deadline there is no prior squad and the public picks
    endpoint 404s, so GW1 must solve from a cold start rather than erroring —
    otherwise the bot cannot produce its first recommendation of the season."""
    state = load_user_state(season_id=99, gameweek=1, team_id=99999)
    assert state.squad == frozenset()
    assert state.bank == INITIAL_BUDGET_TENTHS  # £100.0m
    assert state.gameweek == 1
    assert state.season_id == 99
    assert state.chips_used == frozenset()
    assert state.cost_basis == {}


def test_cold_start_is_restricted_to_gw1():
    """Any later gameweek with no snapshot must stay an error: silently
    substituting an empty £100m squad would discard a real team."""
    for gw in (2, 5, 20, 38):
        with pytest.raises(ValueError, match="Only GW1"):
            load_user_state(season_id=99, gameweek=gw, team_id=99999)


def test_gw1_prefers_a_real_snapshot_over_cold_start():
    """With a cookie the GW1 squad IS observable pre-deadline; if one was
    ingested, it must win over the cold-start fallback."""
    _insert_user_team(season_id=99, gameweek=1, team_id=12345, bank=7)
    state = load_user_state(season_id=99, gameweek=1, team_id=12345)
    assert state.squad != frozenset()
    assert state.bank == 7


def test_gw1_cold_start_still_honours_local_overrides(tmp_path, monkeypatch):
    """A manager who has picked a squad without a cookie can still describe it
    in configs/live_state_overrides.yaml."""
    path = tmp_path / "overrides.yaml"
    path.write_text(
        json.dumps({99: {1: {99999: {"bank_tenths": 3, "squad": [90000, 90001]}}}})
    )
    monkeypatch.setattr(settings, "live_state_overrides_path", path)
    state = load_user_state(season_id=99, gameweek=1, team_id=99999)
    assert state.squad == frozenset({90000, 90001})
    assert state.bank == 3


def test_chips_used_parsed_to_slot_codes_first_half():
    # 3xc played in GW5 → TC1 (first-half slot).
    _insert_user_team(season_id=99, gameweek=5, team_id=12345, chips_used=["3xc"])
    state = load_user_state(season_id=99, gameweek=5, team_id=12345)
    assert "TC1" in state.chips_used
    assert "TC2" not in state.chips_used


def test_chips_used_parsed_to_slot_codes_second_half():
    # bboost played in GW25 → BB2 (second-half slot).
    _insert_user_team(season_id=99, gameweek=25, team_id=12345, chips_used=["bboost"])
    state = load_user_state(season_id=99, gameweek=25, team_id=12345)
    assert "BB2" in state.chips_used
    assert "BB1" not in state.chips_used


def test_chips_used_accumulates_across_snapshots():
    # WC played in GW8, FH played in GW18 (both first-half), TC played in GW25.
    _insert_user_team(season_id=99, gameweek=8, team_id=12345, chips_used=["wildcard"])
    _insert_user_team(season_id=99, gameweek=18, team_id=12345, chips_used=["freehit"])
    _insert_user_team(season_id=99, gameweek=25, team_id=12345, chips_used=["3xc"])
    state = load_user_state(season_id=99, gameweek=30, team_id=12345)
    assert state.chips_used == frozenset({"WC1", "FH1", "TC2"})


def test_status_drops_injured_suspended():
    _insert_status(player_id=90001, status_code="i")
    _insert_status(player_id=90002, status_code="s")
    _insert_status(player_id=90003, status_code="a")
    overrides = load_status_overrides(candidate_player_ids=[90001, 90002, 90003])
    assert 90001 in overrides.excluded_player_ids
    assert 90002 in overrides.excluded_player_ids
    assert 90003 not in overrides.excluded_player_ids
    assert overrides.xpts_attenuator == {}


def test_status_doubtful_attenuates_with_cop():
    _insert_status(player_id=90004, status_code="d", cop=50)
    _insert_status(player_id=90005, status_code="d", cop=75)
    _insert_status(player_id=90006, status_code="d", cop=None)  # default 75
    overrides = load_status_overrides(candidate_player_ids=[90004, 90005, 90006])
    assert overrides.xpts_attenuator[90004] == 0.5
    assert overrides.xpts_attenuator[90005] == 0.75
    assert overrides.xpts_attenuator[90006] == 0.75
    assert overrides.excluded_player_ids == frozenset()


def test_status_available_passes_through():
    _insert_status(player_id=90007, status_code="a")
    overrides = load_status_overrides(candidate_player_ids=[90007])
    assert 90007 not in overrides.excluded_player_ids
    assert 90007 not in overrides.xpts_attenuator


def test_drop_statuses_constant():
    """Sanity: the documented drop set matches the actual constant."""
    assert frozenset({"i", "n", "s", "u"}) == DROP_STATUSES


def test_empty_candidate_list_returns_empty_overrides():
    overrides = load_status_overrides(candidate_player_ids=[])
    assert overrides == LiveStatusOverrides(
        excluded_player_ids=frozenset(),
        xpts_attenuator={},
    )
