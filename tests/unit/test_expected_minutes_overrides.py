"""Tests for the manual expected-minutes override channel (Phase 9 step C).

This is the human-judgement input for team news the model structurally cannot
learn — FPL publishes no historical availability data, so those fields are
untrainable (docs/design/phase9_minutes.md). Minutes multiply every attacking
term, so this is the highest-leverage manual input in the project.
"""
from __future__ import annotations

import polars as pl
import pytest

from fpl_bot.features.manual_overrides import resolve_expected_minutes
from fpl_bot.models.minutes import (
    apply_expected_minutes_overrides,
    minutes_to_bucket_probs,
)

NAMES = {"Haaland": 223094, "Wirtz": 494595, "M.Salah": 118748}


def _preds() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "player_id": [223094, 494595],
            "gameweek": [1, 1],
            "p_minutes_zero": [0.8, 0.2],
            "p_minutes_short": [0.1, 0.3],
            "p_minutes_full": [0.1, 0.5],
        }
    )


def _e_minutes(row: dict) -> float:
    return 30.0 * row["p_minutes_short"] + 75.0 * row["p_minutes_full"]


# ─── Bucket conversion ─────────────────────────────────────────────────────


@pytest.mark.parametrize("target", [0.0, 5.0, 15.0, 30.0, 45.0, 60.0, 75.0])
def test_bucket_probs_preserve_expected_minutes(target: float) -> None:
    p_zero, p_short, p_full = minutes_to_bucket_probs(target)
    assert p_zero + p_short + p_full == pytest.approx(1.0)
    assert min(p_zero, p_short, p_full) >= 0.0
    assert 30.0 * p_short + 75.0 * p_full == pytest.approx(target)


def test_bucket_probs_endpoints_and_saturation() -> None:
    assert minutes_to_bucket_probs(0.0) == (1.0, 0.0, 0.0)
    # Above the 60+ midpoint there is no mass left to shift.
    assert minutes_to_bucket_probs(90.0) == (0.0, 0.0, 1.0)


def test_bucket_probs_rejects_nonsense() -> None:
    with pytest.raises(ValueError):
        minutes_to_bucket_probs(-1.0)
    with pytest.raises(ValueError):
        minutes_to_bucket_probs(float("nan"))


# ─── Application ───────────────────────────────────────────────────────────


def test_override_replaces_the_distribution_and_hits_the_target() -> None:
    out = apply_expected_minutes_overrides(_preds(), {(223094, 1): 90.0})
    row = out.filter(pl.col("player_id") == 223094).to_dicts()[0]
    assert _e_minutes(row) == pytest.approx(75.0)  # saturated "certain to start"
    assert row["p_minutes_zero"] == pytest.approx(0.0)


def test_override_can_express_a_cameo_not_just_absence() -> None:
    """The pre-existing news attenuator can only shift mass toward 'did not
    play'. Rotation intel needs the full distribution."""
    out = apply_expected_minutes_overrides(_preds(), {(494595, 1): 30.0})
    row = out.filter(pl.col("player_id") == 494595).to_dicts()[0]
    assert _e_minutes(row) == pytest.approx(30.0)


def test_unlisted_players_are_untouched() -> None:
    out = apply_expected_minutes_overrides(_preds(), {(223094, 1): 0.0})
    before = _preds().filter(pl.col("player_id") == 494595).to_dicts()[0]
    after = out.filter(pl.col("player_id") == 494595).to_dicts()[0]
    assert _e_minutes(after) == pytest.approx(_e_minutes(before))


def test_no_overrides_is_a_passthrough() -> None:
    base = _preds()
    assert apply_expected_minutes_overrides(base, None).equals(base)
    assert apply_expected_minutes_overrides(base, {}).equals(base)


def test_missing_columns_raise() -> None:
    with pytest.raises(ValueError, match="missing required columns"):
        apply_expected_minutes_overrides(
            pl.DataFrame({"player_id": [1], "gameweek": [1]}), {(1, 1): 90.0}
        )


# ─── Config resolution ─────────────────────────────────────────────────────


def test_resolves_web_names_and_numeric_ids(monkeypatch) -> None:
    import fpl_bot.features.manual_overrides as mo

    monkeypatch.setattr(
        mo, "expected_minutes_overrides_raw",
        lambda: {26: {1: {"Haaland": 90, 494595: 30}}},
    )
    out = resolve_expected_minutes(26, [1], NAMES)
    assert out == {(223094, 1): 90.0, (494595, 1): 30.0}


def test_unresolvable_name_raises_rather_than_being_ignored(monkeypatch) -> None:
    """A typo must never leave the operator believing the bot was told
    something it wasn't — that is worse than having no override at all."""
    import fpl_bot.features.manual_overrides as mo

    monkeypatch.setattr(
        mo, "expected_minutes_overrides_raw",
        lambda: {26: {1: {"Haalandd": 90}}},
    )
    with pytest.raises(ValueError, match="could not resolve"):
        resolve_expected_minutes(26, [1], NAMES)


def test_out_of_range_minutes_raise(monkeypatch) -> None:
    import fpl_bot.features.manual_overrides as mo

    monkeypatch.setattr(
        mo, "expected_minutes_overrides_raw", lambda: {26: {1: {"Haaland": 120}}}
    )
    with pytest.raises(ValueError, match="between 0 and 90"):
        resolve_expected_minutes(26, [1], NAMES)


def test_only_requested_gameweeks_are_returned(monkeypatch) -> None:
    import fpl_bot.features.manual_overrides as mo

    monkeypatch.setattr(
        mo, "expected_minutes_overrides_raw",
        lambda: {26: {1: {"Haaland": 90}, 2: {"Wirtz": 0}}},
    )
    assert resolve_expected_minutes(26, [1], NAMES) == {(223094, 1): 90.0}


def test_shipped_config_is_empty_and_parses() -> None:
    """The committed file must be an inert placeholder, not someone's leftover
    GW overrides silently steering every future run."""
    from fpl_bot.features.manual_overrides import expected_minutes_overrides_raw

    assert expected_minutes_overrides_raw() == {}
