"""Feature builder for the goals + assists per-90 models (Phase 2.2).

Conditional on starting (`minutes ≥ 1`). PIT-routed: imports only
`fpl_bot.db.pit` and `fpl_bot.features.manual_overrides`.

Output: per-(player, fixture) feature matrix with both `goals` and `assists`
labels and an `offset_minutes` column for Poisson offsetting.
"""
from __future__ import annotations

import polars as pl

from fpl_bot.db import pit
from fpl_bot.features import manual_overrides

POSITION_CODES = ("GKP", "DEF", "MID", "FWD")


def build_feature_table(season_ids: list[int] | None = None) -> pl.DataFrame:
    """Build the goals/assists feature matrix.

    PIT correctness: rolling features use `shift(1).over(player_id)` to
    exclude the current row. Joins to understat / market_xg / manual configs
    are static-table joins (no leakage by construction). Position is
    static-metadata via current snapshot (Phase 2.1 pattern).
    """
    pm = pit.all_player_match_with_kickoff(season_ids=season_ids)
    if pm.is_empty():
        return pm

    # Conditional on playing (filter substitutes out only if minutes==0)
    pm = pm.filter(pl.col("minutes") >= 1)
    pm = pm.with_columns(pl.col("kickoff_utc").dt.date().alias("match_date"))

    # Player's team and opponent for the fixture
    pm = pm.with_columns(
        pl.when(pl.col("was_home"))
        .then(pl.col("home_team_id"))
        .otherwise(pl.col("away_team_id"))
        .alias("player_team_id"),
        pl.when(pl.col("was_home"))
        .then(pl.col("away_team_id"))
        .otherwise(pl.col("home_team_id"))
        .alias("opponent_team_id"),
    )

    # ── Understat per-match join (by player_id + match_date) ──────────────────
    us = pit.understat_player_match_history()
    if not us.is_empty():
        # Only keep rows with resolved player_id and match_date
        us = us.filter(pl.col("player_id").is_not_null()).select(
            "player_id",
            "match_date",
            "shots",
            "xg",
            "xa",
            "key_passes",
            "npg",
            "npxg",
            "xg_chain",
            "xg_buildup",
            pl.col("minutes_us").alias("us_minutes"),
        )
        # Aggregate duplicates (player can appear in multiple competitions on the
        # same date; we keep the first) — extremely rare in practice
        us = us.unique(subset=["player_id", "match_date"], keep="first")
        pm = pm.join(us, on=["player_id", "match_date"], how="left")
    else:
        pm = pm.with_columns(
            [pl.lit(None).alias(c) for c in ("xg", "xa", "shots", "key_passes", "npg", "npxg", "xg_chain", "xg_buildup")]
        )

    # ── Sort by player+time before any rolling computation ────────────────────
    pm = pm.sort(by=["player_id", "kickoff_utc"])

    # Rolling per-90 windows (over previous starts, since the feature table is
    # already filtered to minutes≥1)
    minutes_factor = pl.col("minutes") / 90.0
    for stat in ("xg", "xa", "npxg", "shots", "key_passes"):
        prev = (pl.col(stat) / minutes_factor).shift(1).over("player_id")
        pm = pm.with_columns(prev.alias(f"{stat}_per_90_prev"))
        for w in (3, 5, 10):
            pm = pm.with_columns(
                pl.col(f"{stat}_per_90_prev")
                .rolling_mean(window_size=w, min_samples=1)
                .over("player_id")
                .alias(f"{stat}_per_90_last_{w}")
            )
    pm = pm.with_columns(
        ((pl.col("xg_chain") / minutes_factor).shift(1).over("player_id"))
        .rolling_mean(window_size=5, min_samples=1)
        .over("player_id")
        .alias("xg_chain_per_90_last_5"),
        ((pl.col("xg_buildup") / minutes_factor).shift(1).over("player_id"))
        .rolling_mean(window_size=5, min_samples=1)
        .over("player_id")
        .alias("xg_buildup_per_90_last_5"),
    )

    # ── Market xG join (player's team and opponent for this fixture) ──────────
    market = pit.market_xg_for_fixtures()
    if not market.is_empty():
        team_market = market.select(
            pl.col("fixture_id"),
            pl.col("team_id").alias("player_team_id"),
            pl.col("lambda_market_xg").alias("team_lambda_market_xg"),
        )
        opp_market = market.select(
            pl.col("fixture_id"),
            pl.col("team_id").alias("opponent_team_id"),
            pl.col("lambda_market_xg").alias("opponent_lambda_market_xg"),
        )
        pm = pm.join(team_market, on=["fixture_id", "player_team_id"], how="left")
        pm = pm.join(opp_market, on=["fixture_id", "opponent_team_id"], how="left")
    else:
        pm = pm.with_columns(
            pl.lit(None).alias("team_lambda_market_xg"),
            pl.lit(None).alias("opponent_lambda_market_xg"),
        )

    # ── Manual override joins ─────────────────────────────────────────────────
    pk_takers_df = _resolved_pk_takers(season_ids)
    if not pk_takers_df.is_empty():
        pm = pm.join(pk_takers_df, on=["season_id", "player_team_id"], how="left")
        pm = pm.with_columns(
            (pl.col("player_id") == pl.col("pk_taker_player_id"))
            .fill_null(False)
            .cast(pl.Int8)
            .alias("is_penalty_taker")
        )
        pm = pm.drop("pk_taker_player_id")
    else:
        pm = pm.with_columns(pl.lit(0, dtype=pl.Int8).alias("is_penalty_taker"))

    role_mismatch_pids = _resolved_role_mismatch_player_ids()
    pm = pm.with_columns(
        pl.col("player_id")
        .is_in(list(role_mismatch_pids))
        .cast(pl.Int8)
        .alias("role_mismatch")
    )

    # Position one-hot from current snapshot
    positions = pit.all_player_positions()
    if not positions.is_empty():
        pm = pm.join(positions, on="player_id", how="left")
    else:
        pm = pm.with_columns(pl.lit(None).alias("position_code"))
    for code in POSITION_CODES:
        pm = pm.with_columns(
            (pl.col("position_code") == code).cast(pl.Int8).alias(f"pos_{code}")
        )

    # ── Days into season ─────────────────────────────────────────────────────
    pm = pm.with_columns(
        pl.col("kickoff_utc").min().over("season_id").alias("season_start_utc")
    )
    pm = pm.with_columns(
        (
            (
                pl.col("kickoff_utc").cast(pl.Datetime)
                - pl.col("season_start_utc").cast(pl.Datetime)
            ).dt.total_seconds()
            / 86400.0
        ).alias("days_into_season")
    )

    # Cleanup
    drops = [
        "season_start_utc",
        "position_code",
        "us_minutes",
        "xg_per_90_prev",
        "xa_per_90_prev",
        "npxg_per_90_prev",
        "shots_per_90_prev",
        "key_passes_per_90_prev",
    ]
    pm = pm.drop([c for c in drops if c in pm.columns])

    # Cast numeric features to float for LightGBM
    for col in (
        "team_lambda_market_xg",
        "opponent_lambda_market_xg",
    ):
        if col in pm.columns:
            pm = pm.with_columns(pl.col(col).cast(pl.Float64))

    # Offset for Poisson (90-minute equivalent)
    pm = pm.with_columns(
        (pl.col("minutes").cast(pl.Float64) / 90.0).alias("minutes_factor")
    )

    return pm


FEATURE_COLUMNS: list[str] = [
    "xg_per_90_last_3",
    "xg_per_90_last_5",
    "xg_per_90_last_10",
    "npxg_per_90_last_3",
    "npxg_per_90_last_5",
    "npxg_per_90_last_10",
    "xa_per_90_last_3",
    "xa_per_90_last_5",
    "xa_per_90_last_10",
    "shots_per_90_last_3",
    "shots_per_90_last_5",
    "shots_per_90_last_10",
    "key_passes_per_90_last_3",
    "key_passes_per_90_last_5",
    "key_passes_per_90_last_10",
    "xg_chain_per_90_last_5",
    "xg_buildup_per_90_last_5",
    "team_lambda_market_xg",
    "opponent_lambda_market_xg",
    "was_home",
    "is_penalty_taker",
    "role_mismatch",
    "days_into_season",
    "gameweek",
    "pos_GKP",
    "pos_DEF",
    "pos_MID",
    "pos_FWD",
]


def build_prediction_feature_table(
    test_season: int,
    upcoming_fixture_ids: list[int],
) -> pl.DataFrame:
    """Build goals/assists feature rows for UPCOMING fixtures (Phase 6 v2).

    Each row matches the schema of `build_feature_table` but is for a fixture
    that hasn't been played yet (`minutes` is null, labels null). Rolling
    features come from the player's most recent played match (computed via
    the existing `build_feature_table` then propagated forward).

    Strategy:
    1. Build the historical feature table for `test_season` (played fixtures).
    2. Per-player, take their LATEST row's rolling features.
    3. For each upcoming (fixture, eligible player) pair, emit a row with:
       - rolling features from step 2
       - fixture-specific fields (was_home, team_lambda_market_xg, etc.)
         resolved from dim_fixture + market_xg + position snapshot
       - labels (goals, assists) set to None
    """
    if not upcoming_fixture_ids:
        return pl.DataFrame()

    # 1. Load historical played-fixture feature table for the season
    history = build_feature_table(season_ids=[test_season])

    # 2. Per-player latest row (max kickoff_utc)
    if history.is_empty():
        per_player_rolling = pl.DataFrame()
    else:
        per_player_rolling = (
            history.sort(["player_id", "kickoff_utc"])
            .group_by("player_id", maintain_order=True)
            .last()
        )

    # 3. Pull upcoming fixture metadata + eligible players (PIT-routed)
    fixtures_df = pit.upcoming_fixtures(upcoming_fixture_ids)
    players_df = pit.season_player_status_snapshot(test_season).rename(
        {"team_id": "current_team_id"}
    )

    if fixtures_df.is_empty() or players_df.is_empty():
        return pl.DataFrame()

    # 4. Cross join: each player × each fixture, filtered to player's team
    cross = fixtures_df.join(players_df, how="cross")
    cross = cross.filter(
        (pl.col("current_team_id") == pl.col("home_team_id"))
        | (pl.col("current_team_id") == pl.col("away_team_id"))
    )
    cross = cross.with_columns(
        (pl.col("current_team_id") == pl.col("home_team_id")).alias("was_home"),
        pl.when(pl.col("current_team_id") == pl.col("home_team_id"))
        .then(pl.col("home_team_id"))
        .otherwise(pl.col("away_team_id"))
        .alias("player_team_id"),
        pl.when(pl.col("current_team_id") == pl.col("home_team_id"))
        .then(pl.col("away_team_id"))
        .otherwise(pl.col("home_team_id"))
        .alias("opponent_team_id"),
    )

    # 5. Join market xG
    market = pit.market_xg_for_fixtures()
    if not market.is_empty():
        team_market = market.select(
            pl.col("fixture_id"),
            pl.col("team_id").alias("player_team_id"),
            pl.col("lambda_market_xg").alias("team_lambda_market_xg"),
        )
        opp_market = market.select(
            pl.col("fixture_id"),
            pl.col("team_id").alias("opponent_team_id"),
            pl.col("lambda_market_xg").alias("opponent_lambda_market_xg"),
        )
        cross = cross.join(team_market, on=["fixture_id", "player_team_id"], how="left")
        cross = cross.join(opp_market, on=["fixture_id", "opponent_team_id"], how="left")
    else:
        cross = cross.with_columns(
            pl.lit(None, dtype=pl.Float64).alias("team_lambda_market_xg"),
            pl.lit(None, dtype=pl.Float64).alias("opponent_lambda_market_xg"),
        )
    for c in ("team_lambda_market_xg", "opponent_lambda_market_xg"):
        cross = cross.with_columns(pl.col(c).cast(pl.Float64))

    # 6. Join rolling features from each player's last historical row.
    #    Take ONLY the rolling-feature columns; bring in via player_id.
    rolling_cols = [c for c in FEATURE_COLUMNS if "per_90" in c]
    if not per_player_rolling.is_empty():
        keep = ["player_id"] + [c for c in rolling_cols if c in per_player_rolling.columns]
        rolling = per_player_rolling.select(keep)
        cross = cross.join(rolling, on="player_id", how="left")
    else:
        for c in rolling_cols:
            cross = cross.with_columns(pl.lit(0.0).alias(c))

    # 7. PK taker flags + role mismatch
    pk_takers_df = _resolved_pk_takers([test_season])
    if not pk_takers_df.is_empty():
        cross = cross.join(pk_takers_df, on=["season_id", "player_team_id"], how="left")
        cross = cross.with_columns(
            (pl.col("player_id") == pl.col("pk_taker_player_id"))
            .fill_null(False)
            .cast(pl.Int8)
            .alias("is_penalty_taker")
        ).drop("pk_taker_player_id")
    else:
        cross = cross.with_columns(pl.lit(0, dtype=pl.Int8).alias("is_penalty_taker"))

    role_mismatch_pids = _resolved_role_mismatch_player_ids()
    cross = cross.with_columns(
        pl.col("player_id")
        .is_in(list(role_mismatch_pids))
        .cast(pl.Int8)
        .alias("role_mismatch")
    )

    # 8. Position one-hot
    for code in POSITION_CODES:
        cross = cross.with_columns(
            (pl.col("position_code") == code).cast(pl.Int8).alias(f"pos_{code}")
        )

    # 9. days_into_season + minutes_factor (default 1.0 = full 90 for prediction)
    season_start = pit.season_start_kickoff(test_season)
    if season_start is not None:
        cross = cross.with_columns(
            (
                (pl.col("kickoff_utc").cast(pl.Datetime) - pl.lit(season_start).cast(pl.Datetime))
                .dt.total_seconds()
                / 86400.0
            ).alias("days_into_season")
        )
    else:
        cross = cross.with_columns(pl.lit(0.0).alias("days_into_season"))

    cross = cross.with_columns(
        pl.lit(1.0).alias("minutes_factor"),  # default-assume full-90 for prediction
        pl.lit(None, dtype=pl.Int16).alias("goals"),
        pl.lit(None, dtype=pl.Int16).alias("assists"),
        pl.col("was_home").cast(pl.Int8),
    )

    # Fill any missing rolling features with 0 (e.g., promoted-team players
    # with no prior PL match)
    for c in rolling_cols:
        if c in cross.columns:
            cross = cross.with_columns(pl.col(c).fill_null(0.0))
        else:
            cross = cross.with_columns(pl.lit(0.0).alias(c))

    return cross


def _resolved_pk_takers(season_ids: list[int] | None) -> pl.DataFrame:
    """Materializes a (season_id, player_team_id, pk_taker_player_id) table from
    the manual config + dim_team / dim_player joins."""
    raw = manual_overrides.set_piece_takers_raw()
    web_to_pid = pit.web_name_to_player_id()
    seasons = season_ids or list(raw.keys())
    full_to_team = pit.team_id_by_full_name(seasons)

    rows: list[dict] = []
    for season_id in seasons:
        teams = raw.get(season_id, {})
        for team_full, info in teams.items():
            taker_web = info.get("penalty") if isinstance(info, dict) else None
            if not taker_web:
                continue
            taker_pid = web_to_pid.get(taker_web)
            team_id = full_to_team.get((season_id, team_full))
            if taker_pid is None or team_id is None:
                continue
            rows.append(
                {
                    "season_id": season_id,
                    "player_team_id": team_id,
                    "pk_taker_player_id": taker_pid,
                }
            )
    if not rows:
        return pl.DataFrame()
    return pl.DataFrame(rows)


def _resolved_role_mismatch_player_ids() -> set[int]:
    """Set of stable player_ids whose actual role differs from FPL classification."""
    overrides = manual_overrides.position_role_overrides()
    web_to_pid = pit.web_name_to_player_id()
    return {pid for name in overrides if (pid := web_to_pid.get(name)) is not None}
