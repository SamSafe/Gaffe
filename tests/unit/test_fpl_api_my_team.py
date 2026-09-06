"""Tests for FPL my-team parsing state fidelity."""
from __future__ import annotations

import datetime as dt
import json

import pytest
from sqlalchemy import delete

from fpl_bot.db.models import (
    DimFixture,
    DimFixtureSeasonXref,
    DimPlayerSeasonXref,
    FactPlayerMatch,
    FactPlayerStatus,
    FactUserTeamSnapshot,
)
from fpl_bot.db.session import session_scope
from fpl_bot.ingest.fpl_api import (
    parse_my_team,
    parse_raw_fpl_event_live_actuals,
    parse_raw_fpl_event_live_matches,
)
from fpl_bot.ingest.vaastav import _ingest_fixtures

# Reads/writes the dev Postgres (autouse cleanup + xref inserts).
pytestmark = pytest.mark.integration


def _purge_test_rows() -> None:
    with session_scope() as s:
        s.execute(delete(FactUserTeamSnapshot).where(FactUserTeamSnapshot.season_id == 99))
        s.execute(delete(FactPlayerStatus).where(FactPlayerStatus.season_id == 99))
        s.execute(
            delete(FactPlayerMatch).where(FactPlayerMatch.fixture_id == 99001)
        )
        s.execute(delete(DimPlayerSeasonXref).where(DimPlayerSeasonXref.season_id == 99))
        s.execute(
            delete(DimFixtureSeasonXref).where(
                DimFixtureSeasonXref.season_id == 99
            )
        )
        s.execute(delete(DimFixture).where(DimFixture.season_id == 99))


@pytest.fixture(autouse=True)
def _cleanup():
    _purge_test_rows()
    yield
    _purge_test_rows()


def _insert_xrefs() -> None:
    with session_scope() as s:
        s.add_all(
            [
                DimPlayerSeasonXref(
                    season_id=99,
                    fpl_element_id=1,
                    player_id=91001,
                ),
                DimPlayerSeasonXref(
                    season_id=99,
                    fpl_element_id=2,
                    player_id=91002,
                ),
            ]
        )


def _insert_status_prices() -> None:
    now = dt.datetime.now(dt.UTC)
    with session_scope() as s:
        for player_id, price in [(91001, 60), (91002, 75)]:
            s.add(
                FactPlayerStatus(
                    player_id=player_id,
                    recorded_at=now,
                    season_id=99,
                    position_code="MID",
                    team_id=1,
                    price_tenths=price,
                    status_code="a",
                    news=None,
                    chance_of_playing_next_round=None,
                    selected_by_percent=10.0,
                    event_time=now,
                )
            )


def test_parse_authenticated_my_team_uses_exact_prices_bank_and_ft(tmp_path) -> None:
    _insert_xrefs()
    _insert_status_prices()
    raw_path = tmp_path / "my_team.json"
    raw_path.write_text(
        json.dumps(
            {
                "active_chip": "freehit",
                "transfers": {"bank": 23, "free_transfers": 2},
                "picks": [
                    {
                        "element": 1,
                        "purchase_price": 55,
                        "selling_price": 57,
                        "multiplier": 2,
                        "is_captain": True,
                        "is_vice_captain": False,
                        "position": 1,
                    },
                    {
                        "element": 2,
                        "multiplier": 1,
                        "is_captain": False,
                        "is_vice_captain": True,
                        "position": 2,
                    },
                ],
            }
        )
    )

    inserted = parse_my_team(raw_path, season_id=99, gameweek=18, team_id=12345)

    assert inserted == 2
    with session_scope() as s:
        rows = (
            s.query(FactUserTeamSnapshot)
            .filter(FactUserTeamSnapshot.season_id == 99)
            .order_by(FactUserTeamSnapshot.player_id)
            .all()
        )

    assert [(r.player_id, r.purchase_price_tenths, r.selling_price_tenths) for r in rows] == [
        (91001, 55, 57),
        (91002, 75, 75),
    ]
    assert {r.bank_tenths for r in rows} == {23}
    assert {r.free_transfers for r in rows} == {2}
    assert json.loads(rows[0].chips_used_json) == ["freehit"]


def test_parse_event_live_actuals_maps_elements_to_stable_players(tmp_path) -> None:
    _insert_xrefs()
    raw_path = tmp_path / "event_1_live.json"
    raw_path.write_text(
        json.dumps(
            {
                "elements": [
                    {"id": 1, "stats": {"total_points": 8, "minutes": 90}},
                    {"id": 2, "stats": {"total_points": 1, "minutes": 12}},
                    {"id": 999, "stats": {"total_points": 2, "minutes": 45}},
                ]
            }
        )
    )

    points, minutes, counts = parse_raw_fpl_event_live_actuals(
        raw_path, season_id=99
    )

    assert points == {91001: 8, 91002: 1}
    assert minutes == {91001: 90, 91002: 12}
    assert counts == {
        "elements_seen": 3,
        "players_resolved": 2,
        "players_unresolved": 1,
    }


def test_parse_finalized_single_gw_event_live_into_match_rows(tmp_path) -> None:
    _insert_xrefs()
    _insert_status_prices()
    with session_scope() as s:
        s.add(
            DimFixture(
                fixture_id=99001,
                season_id=99,
                gameweek=1,
                kickoff_utc=dt.datetime(2026, 8, 21, 18, 0, tzinfo=dt.UTC),
                home_team_id=1,
                away_team_id=2,
                finished=True,
            )
        )
        s.add(
            DimFixtureSeasonXref(
                season_id=99,
                fpl_fixture_id=1,
                fixture_id=99001,
            )
        )
    raw_path = tmp_path / "event_1_live.json"
    raw_path.write_text(
        json.dumps(
            {
                "elements": [
                    {
                        "id": 1,
                        "stats": {
                            "total_points": 8,
                            "minutes": 90,
                            "goals_scored": 1,
                            "assists": 0,
                            "clean_sheets": 1,
                            "goals_conceded": 0,
                            "saves": 0,
                            "yellow_cards": 0,
                            "red_cards": 0,
                            "bonus": 2,
                            "bps": 31,
                            "defensive_contribution": 4,
                            "tackles": 1,
                            "recoveries": 2,
                            "clearances_blocks_interceptions": 1,
                        },
                        "explain": [{"fixture": 1, "stats": []}],
                    },
                    {
                        "id": 2,
                        "stats": {"total_points": 0, "minutes": 0},
                        "explain": [],
                    },
                ]
            }
        )
    )

    counts = parse_raw_fpl_event_live_matches(
        raw_path, season_id=99, gameweek=1
    )

    assert counts["fact_player_match"] == 2
    with session_scope() as s:
        rows = (
            s.query(FactPlayerMatch)
            .filter(FactPlayerMatch.fixture_id == 99001)
            .order_by(FactPlayerMatch.player_id)
            .all()
        )
    assert [(r.player_id, r.minutes, r.total_points) for r in rows] == [
        (91001, 90, 8),
        (91002, 0, 0),
    ]
    assert rows[0].goals == 1
    assert rows[0].defensive_contribution == 4


def test_stale_vaastav_fixture_cannot_revert_finished_state(tmp_path) -> None:
    with session_scope() as s:
        s.add(
            DimFixture(
                fixture_id=99001,
                season_id=99,
                gameweek=1,
                kickoff_utc=dt.datetime(2026, 8, 21, 18, 0, tzinfo=dt.UTC),
                home_team_id=1,
                away_team_id=2,
                finished=True,
            )
        )
    (tmp_path / "fixtures.csv").write_text(
        "id,code,event,kickoff_time,team_h,team_a,finished\n"
        "1,99001,1,2026-08-21T18:00:00Z,1,2,False\n"
    )

    _ingest_fixtures(tmp_path, season_id=99)

    with session_scope() as s:
        fixture = s.get(DimFixture, 99001)
        assert fixture is not None
        assert fixture.finished is True
