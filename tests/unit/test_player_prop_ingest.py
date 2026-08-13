"""Unit tests for the Phase 8 player-prop ingest, consensus, and blending.

The payload fixtures below mirror the real Odds API response shape, verified
against the live `/events/{id}/odds` endpoint on 2026-07-26 (player in
`description`, "Yes" in `name`, `bookmakers: []` when nothing is priced).

The name-resolution tests carry the most weight: a wrong player_id would
silently move a striker's goal rate onto a teammate, which is worse than
having no prop at all. So the resolver must return None on ambiguity rather
than guess, and these tests pin that.
"""
from __future__ import annotations

import datetime as dt
import json

import polars as pl
import pytest

from fpl_bot.derive.player_props import (
    attach_market_goal_rates,
    consensus_anytime_probs,
    devig_anytime_prob,
    market_implied_per90_rate,
)
from fpl_bot.ingest.oddsapi_props import (
    NameIndex,
    PlayerRow,
    _parse_prop_outcome,
    normalize_name,
    resolve_player_name,
)

ARS, COV = 1, 7


def _index() -> NameIndex:
    return NameIndex(
        [
            PlayerRow(1, "Saka", "Bukayo", "Saka", ARS),
            PlayerRow(2, "Ødegaard", "Martin", "Ødegaard", ARS),
            PlayerRow(3, "Rice", "Declan", "Rice", ARS),
            PlayerRow(4, "Wright", "Haji", "Wright", COV),
            # Same surname as a player at the other club in the fixture.
            PlayerRow(5, "Wright", "Kyle", "Wright", ARS),
            # Same surname AND same club: unresolvable without a first name.
            PlayerRow(6, "Torp", "Jacob", "Torp", COV),
            PlayerRow(7, "Torp", "Jesper", "Torp", COV),
            # A player at a club not in the fixture.
            PlayerRow(8, "Isak", "Alexander", "Isak", 14),
        ]
    )


# ─── Name normalization + resolution ───────────────────────────────────────


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Bukayo Saka", "bukayo saka"),
        ("Martin Ødegaard", "martin odegaard"),
        ("M.Salah", "m salah"),
        ("  Declan   Rice ", "declan rice"),
        ("Robert Lewandowski", "robert lewandowski"),
        ("Nicolò Barella", "nicolo barella"),
    ],
)
def test_normalize_name_folds_accents_and_punctuation(raw: str, expected: str) -> None:
    assert normalize_name(raw) == expected


def test_resolves_full_name() -> None:
    assert resolve_player_name("Bukayo Saka", _index(), {ARS, COV}) == 1


def test_resolves_accented_name_written_plainly() -> None:
    assert resolve_player_name("Martin Odegaard", _index(), {ARS, COV}) == 2


def test_resolves_surname_only_when_unambiguous_in_the_fixture() -> None:
    assert resolve_player_name("Rice", _index(), {ARS, COV}) == 3


def test_surname_collision_across_the_two_clubs_needs_a_first_name() -> None:
    idx = _index()
    # "Wright" exists at both clubs in this fixture → refuse to guess.
    assert resolve_player_name("Wright", idx, {ARS, COV}) is None
    # With a first name it resolves to the right club's player.
    assert resolve_player_name("Haji Wright", idx, {ARS, COV}) == 4


def test_team_scoping_disambiguates_a_surname_collision() -> None:
    idx = _index()
    # Restricting to Coventry alone makes the bare surname unambiguous — this
    # is why candidates are scoped to the fixture's clubs before matching.
    assert resolve_player_name("Wright", idx, {COV}) == 4


def test_identical_names_at_the_same_club_are_never_guessed() -> None:
    assert resolve_player_name("Torp", _index(), {ARS, COV}) is None


def test_resolves_short_form_against_a_multi_token_surname() -> None:
    """FPL stores "Martinelli Silva"; books write "Gabriel Martinelli". Verified
    against the real 26/27 Arsenal squad, where this was the one miss before the
    token-subset rule existed."""
    idx = NameIndex(
        [
            PlayerRow(444145, "Martinelli", "Gabriel", "Martinelli Silva", ARS),
            PlayerRow(226597, "Gabriel", "Gabriel", "dos Santos Magalhães", ARS),
        ]
    )
    assert resolve_player_name("Gabriel Martinelli", idx, {ARS}) == 444145
    assert resolve_player_name("Gabriel Magalhaes", idx, {ARS}) == 226597
    # A bare "Gabriel" is web_name-exact for Magalhães, which is the same
    # convention the bookmakers use. Pinned because it is a precedence choice.
    assert resolve_player_name("Gabriel", idx, {ARS}) == 226597


def test_token_subset_still_refuses_an_ambiguous_short_form() -> None:
    """Neither player owns the bare surname as a web_name, so nothing exact
    matches and the token rule sees two candidates — it must decline."""
    idx = NameIndex(
        [
            PlayerRow(10, "Silva.B", "Bernardo", "Veiga Silva", ARS),
            PlayerRow(11, "Silva.T", "Thiago", "Emiliano Silva", ARS),
        ]
    )
    assert resolve_player_name("Silva", idx, {ARS}) is None
    # Adding the first name makes it unique again.
    assert resolve_player_name("Thiago Silva", idx, {ARS}) == 11


def test_player_outside_the_fixture_is_not_matched() -> None:
    assert resolve_player_name("Alexander Isak", _index(), {ARS, COV}) is None


def test_unknown_and_empty_names_resolve_to_none() -> None:
    idx = _index()
    assert resolve_player_name("Nobody At All", idx, {ARS, COV}) is None
    assert resolve_player_name("", idx, {ARS, COV}) is None


# ─── Outcome parsing ───────────────────────────────────────────────────────


def test_parses_player_in_description_with_yes_name() -> None:
    assert _parse_prop_outcome(
        {"name": "Yes", "description": "Bukayo Saka", "price": 2.1}
    ) == ("Bukayo Saka", 2.1)


def test_parses_legacy_shape_with_player_in_name() -> None:
    assert _parse_prop_outcome({"name": "Bukayo Saka", "price": 2.1}) == (
        "Bukayo Saka",
        2.1,
    )


def test_rejects_the_no_side_and_unusable_prices() -> None:
    assert _parse_prop_outcome({"name": "No", "description": "Saka", "price": 1.5}) is None
    assert _parse_prop_outcome({"name": "No", "price": 1.5}) is None
    assert _parse_prop_outcome({"name": "Yes", "description": "Saka"}) is None
    assert (
        _parse_prop_outcome({"name": "Yes", "description": "Saka", "price": 1.0}) is None
    )
    assert (
        _parse_prop_outcome({"name": "Yes", "description": "Saka", "price": "n/a"})
        is None
    )


# ─── Consensus ─────────────────────────────────────────────────────────────


class _Row:
    def __init__(
        self, fixture_id: int, player_id: int, bookmaker: str, quote_time: dt.datetime,
        decimal_odds: float,
    ) -> None:
        self.fixture_id = fixture_id
        self.player_id = player_id
        self.bookmaker = bookmaker
        self.quote_time = quote_time
        self.decimal_odds = decimal_odds


T0 = dt.datetime(2026, 8, 21, 12, 0, tzinfo=dt.UTC)
T1 = dt.datetime(2026, 8, 21, 16, 0, tzinfo=dt.UTC)


def test_consensus_uses_each_books_latest_quote() -> None:
    rows = [
        _Row(100, 1, "OA_a", T0, 5.0),
        _Row(100, 1, "OA_a", T1, 2.0),  # same book, later → supersedes
    ]
    out = consensus_anytime_probs(rows)
    assert out[(100, 1)].n_books == 1
    assert out[(100, 1)].p_anytime == pytest.approx(devig_anytime_prob(2.0))
    assert out[(100, 1)].latest_quote_time == T1


def test_consensus_pools_across_books_with_the_median() -> None:
    rows = [
        _Row(100, 1, "OA_a", T0, 2.0),
        _Row(100, 1, "OA_b", T0, 3.0),
        _Row(100, 1, "OA_c", T0, 50.0),  # one wild line
    ]
    out = consensus_anytime_probs(rows)
    assert out[(100, 1)].n_books == 3
    # Median resists the outlier: it is the 3.0 book's de-vigged prob.
    assert out[(100, 1)].p_anytime == pytest.approx(devig_anytime_prob(3.0))


def test_consensus_respects_as_of_for_point_in_time() -> None:
    rows = [_Row(100, 1, "OA_a", T0, 5.0), _Row(100, 1, "OA_a", T1, 2.0)]
    out = consensus_anytime_probs(rows, as_of=T0)
    # The post-deadline quote must not leak backwards.
    assert out[(100, 1)].p_anytime == pytest.approx(devig_anytime_prob(5.0))


def test_consensus_rejects_naive_as_of() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        consensus_anytime_probs([], as_of=dt.datetime(2026, 8, 21, 12, 0))


def test_consensus_drops_degenerate_prices() -> None:
    assert consensus_anytime_probs([_Row(100, 1, "OA_a", T0, 1.0)]) == {}


# ─── Blending into predictions ─────────────────────────────────────────────


def _predictions() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "player_id": [1, 2],
            "fixture_id": [100, 100],
            "lambda_goals_per_90": [0.40, 0.30],
            "p_minutes_zero": [0.1, 0.5],
            "p_minutes_short": [0.1, 0.3],
            "p_minutes_full": [0.8, 0.2],
        }
    )


def _consensus_for_player_one() -> dict:
    return consensus_anytime_probs([_Row(100, 1, "OA_a", T0, 2.5)])


def test_market_weight_zero_leaves_rates_untouched_but_still_logs() -> None:
    out = attach_market_goal_rates(
        _predictions(), _consensus_for_player_one(), market_weight=0.0
    )
    # The blend is inert...
    assert out["lambda_goals_per_90"].to_list() == pytest.approx([0.40, 0.30])
    # ...but the market columns are populated, which is what makes the live
    # shadow comparison possible before the weight is ever raised.
    priced = out.filter(pl.col("player_id") == 1)
    assert priced["market_goal_rate_per_90"][0] > 0
    assert priced["market_n_books"][0] == 1


def test_full_market_weight_replaces_the_rate() -> None:
    out = attach_market_goal_rates(
        _predictions(), _consensus_for_player_one(), market_weight=1.0
    )
    row = out.filter(pl.col("player_id") == 1)
    assert row["lambda_goals_per_90"][0] == pytest.approx(
        row["market_goal_rate_per_90"][0]
    )
    # The model rate is preserved alongside it for comparison.
    assert row["lambda_goals_per_90_model"][0] == pytest.approx(0.40)


def test_players_without_a_prop_keep_the_model_rate() -> None:
    out = attach_market_goal_rates(
        _predictions(), _consensus_for_player_one(), market_weight=1.0
    )
    unpriced = out.filter(pl.col("player_id") == 2)
    assert unpriced["lambda_goals_per_90"][0] == pytest.approx(0.30)
    assert unpriced["market_goal_rate_per_90"][0] is None
    assert unpriced["market_n_books"][0] == 0


def test_market_rate_expression_matches_scalar_function() -> None:
    """The frame path is a vectorized copy of `market_implied_per90_rate`; if
    they diverge, the unit-tested minutes guard stops protecting production."""
    out = attach_market_goal_rates(
        _predictions(), _consensus_for_player_one(), market_weight=0.0
    )
    row = out.filter(pl.col("player_id") == 1)
    e_minutes_fraction = (0.1 * 30.0 + 0.8 * 75.0) / 90.0
    expected = market_implied_per90_rate(
        row["market_p_anytime"][0], e_minutes_fraction
    )
    assert row["market_goal_rate_per_90"][0] == pytest.approx(expected)


def test_no_props_still_yields_the_expected_columns() -> None:
    out = attach_market_goal_rates(_predictions(), {}, market_weight=0.5)
    for col in ("market_p_anytime", "market_goal_rate_per_90", "market_n_books"):
        assert col in out.columns
    assert out["lambda_goals_per_90"].to_list() == pytest.approx([0.40, 0.30])


def test_empty_predictions_frame_is_passed_through() -> None:
    assert attach_market_goal_rates(
        pl.DataFrame(), _consensus_for_player_one(), market_weight=1.0
    ).is_empty()


# ─── Shadow log ────────────────────────────────────────────────────────────


def test_shadow_log_writes_both_rates_and_expected_minutes(tmp_path) -> None:
    """The validation instrument itself. This had no test initially and shipped
    a column-projection bug that only surfaced in a full pipeline run."""
    from fpl_bot.eval.xpts_eval import _write_prop_shadow_log

    pm = attach_market_goal_rates(
        _predictions(), _consensus_for_player_one(), market_weight=0.0
    )
    out = tmp_path / "shadow.parquet"
    _write_prop_shadow_log(pm, out)

    log = pl.read_parquet(out)
    # Only the priced player is logged.
    assert log.height == 1
    assert log["player_id"][0] == 1
    for col in (
        "lambda_goals_per_90_model",
        "market_goal_rate_per_90",
        "market_p_anytime",
        "market_n_books",
        "e_minutes",
    ):
        assert col in log.columns
    assert log["e_minutes"][0] == pytest.approx(0.1 * 30.0 + 0.8 * 75.0)


def test_shadow_log_is_skipped_when_nothing_is_priced(tmp_path) -> None:
    from fpl_bot.eval.xpts_eval import _write_prop_shadow_log

    out = tmp_path / "shadow.parquet"
    _write_prop_shadow_log(
        attach_market_goal_rates(_predictions(), {}, market_weight=0.0), out
    )
    assert not out.exists()


# ─── Payload shape (documents the real API response) ───────────────────────


def test_unpriced_event_payload_is_a_db_free_no_op(tmp_path) -> None:
    """A month before kickoff the API returns `bookmakers: []`. That must be a
    quiet no-op that never opens a transaction — it is the normal early-week
    state, and this test would fail with a connection error if it did."""
    from fpl_bot.ingest import oddsapi_props

    payload = [
        {
            "id": "abc",
            "sport_key": "soccer_epl",
            "commence_time": "2026-08-21T19:00:00Z",
            "home_team": "Arsenal",
            "away_team": "Coventry City",
            "bookmakers": [],
        }
    ]
    raw = tmp_path / "soccer_epl_player_props.json"
    raw.write_text(json.dumps(payload))

    counts = oddsapi_props.parse_raw_player_props(raw, season_id=26)
    assert counts["fact_player_odds"] == 0
    assert counts["events_in_payload"] == 1
    assert counts["events_with_prices"] == 0


def test_priced_payload_resolves_players_and_prices(monkeypatch, tmp_path) -> None:
    """End-to-end parse of a realistic priced payload, with the DB layer faked.

    Pins the two things that must line up for a real pull to work: the fixture
    is found from (kickoff date, home, away), and bookmaker spellings resolve
    to player_ids — including one deliberately unresolvable name, which must be
    counted and reported rather than attached to the wrong player.
    """
    from fpl_bot.ingest import oddsapi_props

    kickoff = dt.datetime(2026, 8, 21, 19, 0, tzinfo=dt.UTC)
    payload = [
        {
            "id": "abc",
            "commence_time": "2026-08-21T19:00:00Z",
            "home_team": "Arsenal",
            "away_team": "Coventry City",
            "bookmakers": [
                {
                    "key": "draftkings",
                    "last_update": "2026-08-21T15:00:00Z",
                    "markets": [
                        {
                            "key": "player_goal_scorer_anytime",
                            "last_update": "2026-08-21T15:00:00Z",
                            "outcomes": [
                                {"name": "Yes", "description": "Bukayo Saka", "price": 2.1},
                                {"name": "No", "description": "Bukayo Saka", "price": 1.7},
                                {"name": "Yes", "description": "Martin Odegaard", "price": 4.5},
                                # Ambiguous: two Torps at Coventry.
                                {"name": "Yes", "description": "Torp", "price": 9.0},
                            ],
                        }
                    ],
                }
            ],
        }
    ]
    raw = tmp_path / "soccer_epl_player_props.json"
    raw.write_text(json.dumps(payload))

    monkeypatch.setattr(
        oddsapi_props,
        "_build_lookups",
        lambda season_id: (
            _index(),
            {"ARS": ARS, "COV": COV},
            {(kickoff.date(), ARS, COV): 5001},
        ),
    )
    inserted: list[dict] = []
    monkeypatch.setattr(
        oddsapi_props,
        "session_scope",
        lambda: _FakeSession(inserted),
    )

    counts = oddsapi_props.parse_raw_player_props(raw, season_id=26)

    assert counts["fact_player_odds"] == 2  # Saka + Ødegaard; "No" side dropped
    assert counts["skipped_unresolved_player"] == 1  # the ambiguous "Torp"
    assert counts["distinct_unresolved_names"] == 1
    assert json.loads(
        raw.with_suffix(".unresolved.json").read_text()
    ) == ["Torp"]

    by_player = {row["player_id"]: row for row in inserted}
    assert set(by_player) == {1, 2}
    assert by_player[1]["decimal_odds"] == 2.1
    assert by_player[1]["fixture_id"] == 5001
    assert by_player[1]["bookmaker"] == "OA_draftkings"
    assert by_player[1]["market"] == "anytime_goalscorer"
    assert by_player[1]["source_player_name"] == "Bukayo Saka"
    # quote_time comes from the market's own last_update, not our fetch time,
    # so repeated pulls of an unchanged line collapse to one snapshot.
    assert by_player[1]["quote_time"] == dt.datetime(
        2026, 8, 21, 15, 0, tzinfo=dt.UTC
    )


class _FakeSession:
    """Captures the values of each INSERT without needing PostgreSQL."""

    def __init__(self, sink: list[dict]) -> None:
        self._sink = sink

    def __enter__(self) -> _FakeSession:
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def execute(self, stmt) -> None:
        self._sink.append(dict(stmt.compile().params))


def test_normalizes_letters_nfkd_leaves_alone() -> None:
    """Turkish dotless 'ı' is a distinct letter, not an accented 'i', so NFKD
    never folds it. A real GW1 payload missed Kadıoğlu because of this."""
    assert normalize_name("Ferdi Kadıoğlu") == normalize_name("Ferdi Kadioglu")
    assert normalize_name("Straße") == "strasse"
    assert normalize_name("Æthelred") == "aethelred"


def test_resolves_when_the_book_prints_extra_given_names() -> None:
    """Books often carry fuller legal names than FPL stores. All four of these
    appeared unresolved in the first real priced pull (2026-08-13)."""
    idx = NameIndex(
        [
            PlayerRow(535301, "Baleba", "Carlos", "Baleba", ARS),
            PlayerRow(575476, "Murillo", "Murillo", "Costa dos Santos", ARS),
            PlayerRow(514254, "Gomez", "Diego", "Gómez Amarilla", ARS),
            # Same surname, different club — the safety case.
            PlayerRow(171287, "Gomez", "Joe", "Gomez", COV),
        ]
    )
    assert resolve_player_name("Carlos Baleba Noom Quomah", idx, {ARS}) == 535301
    assert resolve_player_name("Murillo Santiago Costa dos Santos", idx, {ARS}) == 575476
    assert resolve_player_name("Diego Alexander Gomez Amarilla", idx, {ARS}) == 514254
    # Joe Gomez must NOT be swallowed by the longer Diego name: {joe, gomez}
    # is not a subset of it.
    assert resolve_player_name("Diego Alexander Gomez Amarilla", idx, {ARS, COV}) == 514254
    assert resolve_player_name("Joe Gomez", idx, {COV}) == 171287


def test_extra_given_names_still_refuse_an_ambiguous_match() -> None:
    """Two teammates whose FPL names are token-identical: the longer bookmaker
    name contains both, so the rule must decline rather than pick one."""
    idx = NameIndex(
        [
            PlayerRow(1, "Santos", "Carlos", "Santos", ARS),
            PlayerRow(2, "Santos", "Carlos", "Santos", ARS),
        ]
    )
    assert resolve_player_name("Carlos Santos Filho", idx, {ARS}) is None
