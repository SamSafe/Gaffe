"""The rho-EO term must measure alignment with the RIGHT season's template."""
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
def test_eo_lookups_are_scoped_to_the_requested_season():
    """Unscoped lookups mix every season's snapshot into one template.

    `latest_top10k_eo()` took max(event_time) per player with no season
    filter, so a backtest of season 25 was scored against a template built
    partly from season 26 — different clubs, different players, and a
    player set larger than either season really had. Measured locally:
    unscoped returned 1020 players against 839 for season 25 and 653 for
    season 26.
    """
    if not _db_available():
        pytest.skip("PostgreSQL not available; integration test skipped")

    from fpl_bot.optim.eo import latest_top10k_eo, overall_eo_snapshot

    s25 = latest_top10k_eo(25)
    s26 = latest_top10k_eo(26)
    if not s25 or not s26:
        pytest.skip("EO snapshots for seasons 25 and 26 not ingested locally")

    unscoped = latest_top10k_eo()
    assert len(unscoped) > len(s25), "unscoped lookup should span more than one season"
    assert len(unscoped) > len(s26)

    # A scoped lookup must not exceed that season's registered player count.
    with engine.connect() as c:
        n25 = c.execute(
            text(
                "SELECT count(*) FROM dim_player_season_xref WHERE season_id = 25"
            )
        ).scalar_one()
    assert len(s25) <= n25

    # The overall band must scope too, or the blend re-imports the wrong season.
    o25 = overall_eo_snapshot(25)
    o26 = overall_eo_snapshot(26)
    assert not o25.is_empty() and not o26.is_empty()
    assert o25.height != o26.height or set(o25["player_id"]) != set(o26["player_id"])


@pytest.mark.integration
def test_backtest_passes_its_test_season_to_the_eo_lookup():
    """Guard the wiring, not just the helper."""
    import inspect

    from fpl_bot.eval import milp_backtest

    src = inspect.getsource(milp_backtest)
    assert "eo_for_candidates(candidates, season_id=test_season)" in src, (
        "the backtest must scope EO to the season under test"
    )
