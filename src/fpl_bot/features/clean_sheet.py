"""Feature builder for the clean-sheet model (Phase 2.3).

Team-level binary classification (one row per (team_id, fixture_id)). Uses
the Dixon-Coles–derived market CS probability as a strong prior; downstream
LightGBM model carries the *residual* via init_score = logit(market).

PIT-routed: imports only `fpl_bot.db.pit`. Rolling features use polars
`shift(1).over(team_id)` to exclude the current row before the rolling
window. Team_id one-hot encoding is fold-internal (see `make_team_id_encoder`)
to prevent future-season team identities leaking into earlier folds.
"""
from __future__ import annotations

import polars as pl

from fpl_bot.db import pit


def build_feature_table(season_ids: list[int] | None = None) -> pl.DataFrame:
    """Build the per-(team, fixture) feature matrix.

    Aggregates the player-level `pit.all_player_match_with_kickoff` to one
    row per (team_id, fixture_id) by deduping team-replicated fields
    (clean_sheet, goals_conceded, was_home). Joins market λ + cs_prob from
    `pit.market_xg_for_fixtures`. Computes team-level rolling features.
    """
    pm = pit.all_player_match_with_kickoff(season_ids=season_ids)
    if pm.is_empty():
        return pm

    # NB: both `clean_sheet` and `goals_conceded` are per-PLAYER stats keyed
    # to time-on-pitch:
    #   - clean_sheet = 1 only for ≥60-min appearances on a team that kept CS
    #   - goals_conceded = goals against the team WHILE the player was on
    # So a sub who came on at minute 85 has goals_conceded=0 even if the
    # team conceded 3 earlier. To recover the team total, aggregate with max:
    # the starter who played the full match has the true team-level value.
    pm = pm.filter(pl.col("goals_conceded").is_not_null())

    pm = pm.with_columns(
        pl.when(pl.col("was_home"))
        .then(pl.col("home_team_id"))
        .otherwise(pl.col("away_team_id"))
        .alias("team_id"),
        pl.when(pl.col("was_home"))
        .then(pl.col("away_team_id"))
        .otherwise(pl.col("home_team_id"))
        .alias("opponent_team_id"),
    )

    # ── Aggregate to (team, fixture) level ────────────────────────────────────
    # goals_conceded / was_home are team-level, replicated to every teammate.
    # team_goals_for is the sum of teammate goals. team-level clean_sheet is
    # derived from goals_conceded post-aggregation.
    tf = pm.group_by(["team_id", "fixture_id"]).agg(
        pl.first("season_id").alias("season_id"),
        pl.first("gameweek").alias("gameweek"),
        pl.first("kickoff_utc").alias("kickoff_utc"),
        pl.first("was_home").alias("was_home"),
        pl.first("opponent_team_id").alias("opponent_team_id"),
        pl.col("goals_conceded").max().alias("goals_conceded"),
        pl.col("goals").sum().alias("team_goals_for"),
    )
    tf = tf.with_columns(
        (pl.col("goals_conceded") == 0).cast(pl.Int8).alias("clean_sheet")
    )

    # Sort by team + time before any rolling computation (PIT discipline)
    tf = tf.sort(by=["team_id", "kickoff_utc"])

    tf = tf.with_columns(
        pl.col("clean_sheet").cast(pl.Float32).shift(1).over("team_id").alias("cs_prev"),
        pl.col("goals_conceded").cast(pl.Float32).shift(1).over("team_id").alias("gc_prev"),
        pl.col("team_goals_for").cast(pl.Float32).shift(1).over("team_id").alias("gf_prev"),
    )
    for w in (3, 5, 10):
        tf = tf.with_columns(
            pl.col("cs_prev")
            .rolling_mean(window_size=w, min_samples=1)
            .over("team_id")
            .alias(f"team_cs_rate_last_{w}"),
            pl.col("gc_prev")
            .rolling_mean(window_size=w, min_samples=1)
            .over("team_id")
            .alias(f"team_goals_conceded_last_{w}"),
        )
    tf = tf.with_columns(
        pl.col("gf_prev")
        .rolling_mean(window_size=5, min_samples=1)
        .over("team_id")
        .alias("team_goals_for_last_5"),
    )

    # Days since last match (team-level)
    tf = tf.with_columns(
        (
            (
                pl.col("kickoff_utc").cast(pl.Datetime)
                - pl.col("kickoff_utc").shift(1).cast(pl.Datetime).over("team_id")
            ).dt.total_seconds()
            / 86400.0
        ).alias("days_since_last_match")
    )

    # Days into season
    tf = tf.with_columns(
        pl.col("kickoff_utc").min().over("season_id").alias("season_start_utc")
    )
    tf = tf.with_columns(
        (
            (
                pl.col("kickoff_utc").cast(pl.Datetime)
                - pl.col("season_start_utc").cast(pl.Datetime)
            ).dt.total_seconds()
            / 86400.0
        ).alias("days_into_season")
    )

    # ── Market xG join (for team and opponent) ────────────────────────────────
    market = pit.market_xg_for_fixtures()
    if not market.is_empty():
        team_market = market.select(
            pl.col("fixture_id"),
            pl.col("team_id"),
            pl.col("lambda_market_xg").alias("team_lambda_market_xg"),
            pl.col("cs_prob_market").alias("market_cs_prob"),
        )
        opp_market = market.select(
            pl.col("fixture_id"),
            pl.col("team_id").alias("opponent_team_id"),
            pl.col("lambda_market_xg").alias("opponent_lambda_market_xg"),
        )
        tf = tf.join(team_market, on=["fixture_id", "team_id"], how="left")
        tf = tf.join(opp_market, on=["fixture_id", "opponent_team_id"], how="left")
    else:
        tf = tf.with_columns(
            pl.lit(None).cast(pl.Float64).alias("team_lambda_market_xg"),
            pl.lit(None).cast(pl.Float64).alias("opponent_lambda_market_xg"),
            pl.lit(None).cast(pl.Float64).alias("market_cs_prob"),
        )

    # logit(market_cs_prob) for init_score (clip for numerical stability per
    # plan §"Edge cases" — avoids ±inf when market is 0 or 1)
    eps = 1e-4
    tf = tf.with_columns(
        pl.col("market_cs_prob").clip(eps, 1 - eps).alias("_clipped"),
    )
    tf = tf.with_columns(
        (pl.col("_clipped") / (1 - pl.col("_clipped"))).log().alias("logit_market_cs_prob"),
    )

    tf = tf.drop(
        ["cs_prev", "gc_prev", "gf_prev", "season_start_utc", "_clipped"]
    )

    tf = tf.with_columns(pl.col("was_home").cast(pl.Int8))

    return tf


FEATURE_COLUMNS: list[str] = [
    "team_lambda_market_xg",
    "opponent_lambda_market_xg",
    "team_cs_rate_last_3",
    "team_cs_rate_last_5",
    "team_cs_rate_last_10",
    "team_goals_conceded_last_3",
    "team_goals_conceded_last_5",
    "team_goals_conceded_last_10",
    "team_goals_for_last_5",
    "was_home",
    "days_since_last_match",
    "days_into_season",
    "gameweek",
]

LABEL_COLUMN = "clean_sheet"
INIT_SCORE_COLUMN = "logit_market_cs_prob"
MARKET_PROB_COLUMN = "market_cs_prob"


class TeamIdEncoder:
    """Fold-internal one-hot encoder for team_id.

    Fitted on the set of team_ids seen in TRAIN seasons only. Applied to
    train and test alike. Test rows whose team_id wasn't in train (e.g.,
    promoted teams making their first appearance in the test fold) get
    an all-zero one-hot vector — the gradient-boosted model interprets
    this as "unknown team" and falls back on rolling/market features.

    Critical: this encoder must NEVER be fitted on the full corpus across
    folds; doing so leaks future-season team identities into earlier folds.
    """

    def __init__(self, seen_team_ids: set[int]) -> None:
        self.seen_team_ids = sorted(seen_team_ids)
        self.column_names: list[str] = [f"team_id_{tid}" for tid in self.seen_team_ids]

    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        if not self.seen_team_ids:
            return df
        new_cols = [
            (pl.col("team_id") == tid).cast(pl.Int8).alias(f"team_id_{tid}")
            for tid in self.seen_team_ids
        ]
        return df.with_columns(new_cols)


def make_team_id_encoder(seen_team_ids: set[int]) -> TeamIdEncoder:
    return TeamIdEncoder(seen_team_ids)


def feature_names_with_team_id(encoder: TeamIdEncoder) -> list[str]:
    """Concatenates the static feature columns with the encoder's team_id columns."""
    return [*FEATURE_COLUMNS, *encoder.column_names]
