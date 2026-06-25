"""Tests for FPL my-team parsing state fidelity."""
from __future__ import annotations

import datetime as dt
import json

import pytest
from sqlalchemy import delete

from fpl_bot.db.models import (
    DimPlayerSeasonXref,
    FactPlayerStatus,
    FactUserTeamSnapshot,
)
from fpl_bot.db.session import session_scope
from fpl_bot.ingest.fpl_api import parse_my_team

# Reads/writes the dev Postgres (autouse cleanup + xref inserts).
pytestmark = pytest.mark.integration


def _purge_test_rows() -> None:
    with session_scope() as s:
        s.execute(delete(FactUserTeamSnapshot).where(FactUserTeamSnapshot.season_id == 99))
        s.execute(delete(FactPlayerStatus).where(FactPlayerStatus.season_id == 99))
        s.execute(delete(DimPlayerSeasonXref).where(DimPlayerSeasonXref.season_id == 99))


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
