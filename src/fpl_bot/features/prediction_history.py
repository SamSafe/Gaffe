"""Advance shifted historical features through the last completed match."""

import polars as pl


def append_prediction_history_rows(history: pl.DataFrame) -> pl.DataFrame:
    """Append an unlabelled row per player, without inventing any outcome.

    Historical features exclude their own match via shift(1). A future row
    is therefore needed to include the most recent completed result.
    These rows are used only to extract live rolling features, never training.
    """
    if history.is_empty():
        return history
    latest = history.sort(["player_id", "kickoff_utc"]).group_by("player_id").last()
    metadata = {
        "player_id", "fixture_id", "season_id", "gameweek", "kickoff_utc",
        "home_team_id", "away_team_id", "was_home", "player_team_id",
        "opponent_team_id", "match_date",
    }
    latest = latest.with_columns(
        pl.lit(-1).cast(history.schema["fixture_id"]).alias("fixture_id"),
        (pl.col("kickoff_utc") + pl.duration(microseconds=1)).alias("kickoff_utc"),
        *[
            pl.lit(None).cast(dtype).alias(column)
            for column, dtype in history.schema.items()
            if column not in metadata
        ],
    )
    return pl.concat([history, latest.select(history.columns)], how="vertical")
