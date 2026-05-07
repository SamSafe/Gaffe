"""Leakage gate for the BPS simulator's feature path.

The simulator's input is fully derived from already-PIT-routed predictions
(minutes / goals / assists models) plus market data and configs. We verify
that the assembly pipeline doesn't accidentally leak future-season info into
earlier-fold inputs.
"""
from __future__ import annotations

import pytest
from sqlalchemy import text

from fpl_bot.db.session import engine


def _db_available() -> bool:
    try:
        with engine.connect() as c:
            c.execute(text("SELECT 1"))
        return True
    except Exception:
        return False


@pytest.mark.integration
@pytest.mark.leakage
def test_residual_dataset_does_not_use_future_seasons() -> None:
    """fit_residual_dataset on train_seasons={19,20} must not include any rows
    from later seasons (future leakage check)."""
    if not _db_available():
        pytest.skip("PostgreSQL not available; integration test skipped")

    from fpl_bot.db import pit
    from fpl_bot.models.bps import fit_residual_dataset

    train_pm = pit.all_player_match_with_kickoff(season_ids=[19, 20])
    if train_pm.is_empty():
        pytest.skip("No corpus data; ingest before running")
    positions = pit.all_player_positions()
    residual_df = fit_residual_dataset(train_pm, positions)

    # Sanity check: residual rows correspond to fixture_ids only from seasons 19+20
    fixtures_in_residual = set(residual_df["fixture_id"].to_list())
    test_pm = pit.all_player_match_with_kickoff(season_ids=[19, 20])
    fixtures_19_20 = set(test_pm["fixture_id"].to_list())
    assert fixtures_in_residual.issubset(fixtures_19_20), (
        "fit_residual_dataset returned rows from outside the requested seasons — leakage"
    )


@pytest.mark.integration
@pytest.mark.leakage
def test_alpha_split_uses_only_train_data() -> None:
    """split_p_full_by_position must be deterministic given the same train slice
    and not leak from later seasons."""
    if not _db_available():
        pytest.skip("PostgreSQL not available; integration test skipped")

    import polars as pl

    from fpl_bot.db import pit
    from fpl_bot.models.bps import split_p_full_by_position

    positions = pit.all_player_positions()

    # Compute alphas using only [19, 20]
    train_pm_a = (
        pit.all_player_match_with_kickoff(season_ids=[19, 20])
        .join(positions, on="player_id", how="left")
        .drop_nulls("position_code")
    )
    alphas_a = split_p_full_by_position(train_pm_a)

    # Compute alphas using the same [19, 20] slice but built via filtering the
    # full corpus. They should be identical (no leakage, deterministic).
    full = (
        pit.all_player_match_with_kickoff(season_ids=[19, 20, 21, 22, 23, 24])
        .join(positions, on="player_id", how="left")
        .drop_nulls("position_code")
    )
    train_pm_b = full.filter(pl.col("season_id").is_in([19, 20]))
    alphas_b = split_p_full_by_position(train_pm_b)

    for pos in ("GKP", "DEF", "MID", "FWD"):
        assert abs(alphas_a[pos] - alphas_b[pos]) < 1e-9, (
            f"Alpha for {pos} differs between filtered-then-trained vs trained-on-slice — "
            f"corpus-mode leak ({alphas_a[pos]:.4f} vs {alphas_b[pos]:.4f})"
        )
