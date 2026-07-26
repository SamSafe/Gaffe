"""Guards for the season-rollover failure class.

Two rollover mistakes are silent rather than loud, which is what makes them
worth a test:

  1. A promoted club missing from an ingest name map. Its odds simply never
     link to a fixture, and since bookmaker odds are the bot's main edge over
     buy-and-hold, that quietly removes the edge for 10% of the league.
  2. A stale `current_season_id`, which writes the new season's data into the
     previous season's rows.

`CURRENT_SEASON_CLUBS` is deliberately hardcoded: it must be updated at each
season rollover, and this test failing every August is the intended reminder.
"""
from __future__ import annotations

import pytest

from fpl_bot.config import settings
from fpl_bot.features.manual_overrides import set_piece_takers_raw
from fpl_bot.ingest.footballdata import FD_TO_FPL_SHORT
from fpl_bot.ingest.oddsapi import OA_TO_FPL_SHORT
from fpl_bot.ingest.understat import (
    FPL_FULL_NAME_ALTERNATES,
    US_TO_FPL_FULL_NAME,
    _fpl_full_name_candidates,
)

CURRENT_SEASON_ID = 26

# FPL `dim_team.full_name` → `short_name` for the 2026-27 Premier League.
# Promoted for 26/27: COV, HUL, IPS.
CURRENT_SEASON_CLUBS: dict[str, str] = {
    "Arsenal": "ARS",
    "Aston Villa": "AVL",
    "Bournemouth": "BOU",
    "Brentford": "BRE",
    "Brighton": "BHA",
    "Chelsea": "CHE",
    "Coventry City": "COV",
    "Crystal Palace": "CRY",
    "Everton": "EVE",
    "Fulham": "FUL",
    "Hull City": "HUL",
    "Ipswich Town": "IPS",
    "Leeds": "LEE",
    "Liverpool": "LIV",
    "Man City": "MCI",
    "Man Utd": "MUN",
    "Newcastle": "NEW",
    "Nott'm Forest": "NFO",
    "Spurs": "TOT",
    "Sunderland": "SUN",
}


def test_current_season_id_matches_the_clubs_listed_here() -> None:
    assert settings.current_season_id == CURRENT_SEASON_ID, (
        "config.current_season_id and this test's club list disagree — one of "
        "them was not updated at the season rollover."
    )


def test_train_seasons_are_all_before_the_current_season() -> None:
    """The current season may only be appended once it has ~5+ played GWs, and
    never a future one — see docs/RUNBOOK_live.md."""
    assert settings.train_season_ids, "train_seasons is empty"
    assert max(settings.train_season_ids) <= settings.current_season_id


@pytest.mark.parametrize(
    ("source", "name_map"),
    [("oddsapi", OA_TO_FPL_SHORT), ("footballdata", FD_TO_FPL_SHORT)],
)
def test_every_current_club_is_mapped_to_a_short_name(
    source: str, name_map: dict[str, str]
) -> None:
    mapped = set(name_map.values())
    missing = sorted(s for s in CURRENT_SEASON_CLUBS.values() if s not in mapped)
    assert not missing, (
        f"{source} name map has no entry producing {missing} — those clubs' "
        "odds would be dropped silently. Add their external names to the map."
    )


def test_every_current_club_is_reachable_in_the_understat_map() -> None:
    reachable = set(US_TO_FPL_FULL_NAME.values())
    missing = sorted(f for f in CURRENT_SEASON_CLUBS if f not in reachable)
    assert not missing, f"understat map does not resolve to FPL full names {missing}"


def test_renamed_clubs_keep_their_historical_full_name_as_an_alternate() -> None:
    """FPL renamed IPS between 24/25 and 26/27; the fixture lookup spans all
    seasons, so both spellings must resolve."""
    candidates = _fpl_full_name_candidates("Ipswich")
    assert candidates[0] == "Ipswich Town"
    assert "Ipswich" in candidates

    for primary, alternates in FPL_FULL_NAME_ALTERNATES.items():
        assert primary in US_TO_FPL_FULL_NAME.values(), (
            f"{primary} has alternates but is not a mapped FPL full name"
        )
        assert alternates, f"{primary} has an empty alternates tuple"


def test_unmapped_understat_team_yields_no_candidates() -> None:
    assert _fpl_full_name_candidates("Madeup FC") == ()
    assert _fpl_full_name_candidates(None) == ()


def test_set_piece_block_covers_every_current_club() -> None:
    block = set_piece_takers_raw().get(CURRENT_SEASON_ID, {})
    missing = sorted(f for f in CURRENT_SEASON_CLUBS if f not in block)
    assert not missing, (
        f"configs/set_piece_takers.yaml season {CURRENT_SEASON_ID} is missing "
        f"{missing}; regenerate with scripts/gen_set_piece_takers.py"
    )
    for team, info in block.items():
        assert info.get("penalty"), f"{team} has no primary penalty taker"
