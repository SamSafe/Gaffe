"""Guard train-only stopping and inclusion of recent history in the refit."""
from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import numpy as np
import polars as pl
import pytest

from fpl_bot.models.validation import (
    TRAINING_POLICY_VERSION,
    chronological_validation_split,
    train_with_chronological_validation,
)

pytestmark = pytest.mark.leakage


def _history() -> pl.DataFrame:
    start = datetime(2024, 1, 1, tzinfo=UTC)
    return pl.DataFrame([
        {"fixture_id": gw, "player_id": player, "kickoff_utc": start + timedelta(days=gw),
         "season_id": 24, "gameweek": gw, "min_last_1": 90, "minutes_bucket": 2,
         "goals": gw % 2, "assists": 0, "minutes_factor": 1.0}
        for gw in range(1, 11) for player in (1, 2)
    ])


def test_split_is_chronological_and_keeps_fixture_players_together():
    history = _history().sample(fraction=1, shuffle=True, seed=42)
    fit, validation = chronological_validation_split(history)
    assert fit["kickoff_utc"].max() < validation["kickoff_utc"].min()
    assert set(fit["fixture_id"]).isdisjoint(validation["fixture_id"])
    assert set(validation["fixture_id"]) == {9, 10}
    assert fit.height + validation.height == history.height


def test_split_purges_results_not_finished_when_validation_starts():
    kickoff = datetime(2024, 1, 10, 15, tzinfo=UTC)
    history = pl.DataFrame({"kickoff_utc": [
        kickoff - timedelta(days=1), kickoff - timedelta(hours=2), kickoff,
    ]})
    fit, validation = chronological_validation_split(history)
    assert fit.height == 1
    assert validation.height == 1
    assert fit["kickoff_utc"].max() < validation["kickoff_utc"].min() - timedelta(hours=3)


@pytest.mark.parametrize("history", [
    pl.DataFrame({"value": [1, 2]}),
    pl.DataFrame({"kickoff_utc": ["2024-01-01", "2024-01-02"]}),
    pl.DataFrame({"kickoff_utc": [datetime(2024, 1, 1), None]}),
    pl.DataFrame({"kickoff_utc": [datetime(2024, 1, 1)]}),
])
def test_unsafe_history_fails_instead_of_using_test_or_training_for_stopping(history):
    with pytest.raises(ValueError):
        chronological_validation_split(history)


def test_refit_keeps_latest_history_without_an_early_stopping_set():
    history = _history()
    calls = []

    def trainer(rows, *, valid, num_boost_round=1000):
        calls.append((rows, valid, num_boost_round))
        return SimpleNamespace(num_iterations=7)

    train_with_chronological_validation(trainer, history, num_boost_round=100)
    selected_train, selected_valid, _ = calls[0]
    assert selected_train["kickoff_utc"].max() < selected_valid["kickoff_utc"].min()
    refit, refit_valid, rounds = calls[1]
    assert refit.equals(history)
    assert refit_valid is None
    assert rounds == 7


def test_multiclass_refit_reports_rounds_instead_of_three_trees_per_round():
    from fpl_bot.models.minutes import train_minutes_model

    history = _history().with_columns((pl.col("fixture_id") % 3).alias("minutes_bucket"))
    model = train_with_chronological_validation(
        train_minutes_model, history, feature_subset=["min_last_1"],
        num_boost_round=10, early_stopping_rounds=3,
    )
    assert model.num_iterations == model.booster.current_iteration()
    assert model.booster.num_trees() == 3 * model.num_iterations
    assert np.allclose(model.predict_proba(history).sum(axis=1), 1)


@pytest.mark.parametrize("target", ["minutes", "goals", "assists"])
def test_outer_labels_cannot_change_stopping_or_predictions(monkeypatch, target):
    from fpl_bot.eval import bps_eval

    history = _history()
    future = history.with_columns(
        pl.lit(25, dtype=pl.Int64).alias("season_id"),
        (pl.col("fixture_id") + 100).alias("fixture_id"),
        (pl.col("kickoff_utc") + timedelta(days=366)).alias("kickoff_utc"),
    )
    all_rows = pl.concat([history, future])
    calls = []

    class Model:
        num_iterations = 5

        def predict_proba(self, rows):
            return np.tile([0.1, 0.1, 0.8], (len(rows), 1))

        def predict_per_90(self, rows):
            return np.full(len(rows), 0.5)

    def trainer(rows, *, valid=None, **kwargs):
        calls.append((rows, valid, kwargs))
        assert set(rows["season_id"]) == {24}
        if valid is not None:
            assert set(valid["season_id"]) == {24}
        return Model()

    builder = "build_minutes_feature_table" if target == "minutes" else "build_goals_feature_table"
    trainer_name = "train_minutes_model" if target == "minutes" else "train_per_90_model"
    monkeypatch.setattr(bps_eval, builder, lambda **kwargs: all_rows)
    monkeypatch.setattr(bps_eval, trainer_name, trainer)

    def predict():
        if target == "minutes":
            return bps_eval._train_minutes_predictor([24], 25)
        return bps_eval._train_goals_or_assists_predictor([24], 25, target=target)

    before = predict()
    all_rows = all_rows.with_columns([
        pl.when(pl.col("season_id") == 25).then(999).otherwise(pl.col(col)).alias(col)
        for col in ("minutes_bucket", "goals", "assists")
    ])
    after = predict()
    assert before.equals(after)
    assert calls[0][0].equals(calls[2][0])
    assert calls[0][1].equals(calls[2][1])


def test_old_raw_sample_cache_cannot_be_loaded_by_new_policy(tmp_path):
    from fpl_bot.optim.scenarios import cache_path_for_fold

    legacy = tmp_path / "season_25_train_19_20_n200.parquet"
    legacy.touch()
    current = cache_path_for_fold(25, [19, 20], cache_dir=tmp_path)
    assert TRAINING_POLICY_VERSION in current.name
    assert current != legacy
    assert not current.exists()
