"""Initial schema — all tables from docs/design/phase0.md §3.2.

Raw DDL via op.execute() to match the approved design verbatim.
Note: bps_rule_set.position_filter uses sentinel 'ALL' rather than NULL
because PostgreSQL does not allow NULLs in primary key components.

Revision ID: 0001
Revises:
Create Date: 2026-04-30
"""
from __future__ import annotations

from typing import Sequence

from alembic import op

revision: str = "0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


_DDL = [
    # ─── Dimension tables ─────────────────────────────────────────────────
    """
    CREATE TABLE dim_team (
        team_id     SMALLINT NOT NULL,
        season_id   SMALLINT NOT NULL,
        short_name  TEXT NOT NULL,
        full_name   TEXT NOT NULL,
        promoted    BOOLEAN NOT NULL DEFAULT FALSE,
        PRIMARY KEY (team_id, season_id)
    )
    """,
    """
    CREATE TABLE dim_player (
        player_id     INT PRIMARY KEY,
        web_name      TEXT NOT NULL,
        first_name    TEXT,
        last_name     TEXT,
        understat_id  INT,
        fbref_id      TEXT
    )
    """,
    """
    CREATE TABLE dim_fixture (
        fixture_id    INT PRIMARY KEY,
        season_id     SMALLINT NOT NULL,
        gameweek      SMALLINT NOT NULL,
        kickoff_utc   TIMESTAMPTZ NOT NULL,
        home_team_id  SMALLINT NOT NULL,
        away_team_id  SMALLINT NOT NULL,
        finished      BOOLEAN NOT NULL DEFAULT FALSE
    )
    """,
    """
    CREATE TABLE dim_penalty_taker (
        team_id        SMALLINT NOT NULL,
        rank_in_order  SMALLINT NOT NULL,
        valid_from     TIMESTAMPTZ NOT NULL,
        source         TEXT NOT NULL,
        player_id      INT NOT NULL,
        override_set   BOOLEAN NOT NULL DEFAULT FALSE,
        valid_to       TIMESTAMPTZ,
        recorded_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
        PRIMARY KEY (team_id, rank_in_order, valid_from, source)
    )
    """,
    # ─── Fact tables (append-only, bitemporal-lite) ───────────────────────
    """
    CREATE TABLE fact_player_status (
        player_id                     INT NOT NULL,
        recorded_at                   TIMESTAMPTZ NOT NULL DEFAULT now(),
        season_id                     SMALLINT NOT NULL,
        position_code                 CHAR(3) NOT NULL,
        team_id                       SMALLINT NOT NULL,
        price_tenths                  SMALLINT NOT NULL,
        status_code                   CHAR(1) NOT NULL,
        news                          TEXT,
        chance_of_playing_next_round  SMALLINT,
        selected_by_percent           NUMERIC(5,2),
        event_time                    TIMESTAMPTZ NOT NULL,
        PRIMARY KEY (player_id, recorded_at)
    )
    """,
    "CREATE INDEX ix_player_status_pit ON fact_player_status (player_id, recorded_at DESC)",
    """
    CREATE TABLE fact_match_result (
        fixture_id   INT NOT NULL,
        recorded_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
        home_score   SMALLINT,
        away_score   SMALLINT,
        finished     BOOLEAN NOT NULL DEFAULT FALSE,
        PRIMARY KEY (fixture_id, recorded_at)
    )
    """,
    """
    CREATE TABLE fact_player_match (
        player_id      INT NOT NULL,
        fixture_id     INT NOT NULL,
        recorded_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
        minutes        SMALLINT,
        goals          SMALLINT,
        assists        SMALLINT,
        clean_sheet    BOOLEAN,
        goals_conceded SMALLINT,
        saves          SMALLINT,
        yellow_cards   SMALLINT,
        red_cards      SMALLINT,
        bonus          SMALLINT,
        bps            SMALLINT,
        total_points   SMALLINT,
        PRIMARY KEY (player_id, fixture_id, recorded_at)
    )
    """,
    """
    CREATE TABLE fact_understat_shot (
        shot_id          TEXT PRIMARY KEY,
        fixture_id       INT,
        player_id        INT,
        minute           SMALLINT,
        xg               NUMERIC(5,4),
        xa_for_assister  NUMERIC(5,4),
        result           TEXT,
        situation        TEXT,
        shot_type        TEXT,
        body_part        TEXT,
        is_set_piece     BOOLEAN,
        recorded_at      TIMESTAMPTZ NOT NULL DEFAULT now()
    )
    """,
    """
    CREATE TABLE fact_odds (
        fixture_id    INT NOT NULL,
        bookmaker     TEXT NOT NULL,
        market        TEXT NOT NULL,
        selection     TEXT NOT NULL,
        event_time    TIMESTAMPTZ NOT NULL,
        decimal_odds  NUMERIC(8,4) NOT NULL,
        recorded_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
        PRIMARY KEY (fixture_id, bookmaker, market, selection, event_time)
    )
    """,
    """
    CREATE TABLE fact_market_xg (
        fixture_id          INT NOT NULL,
        team_id             SMALLINT NOT NULL,
        source_recorded_at  TIMESTAMPTZ NOT NULL,
        "lambda"            NUMERIC(6,4) NOT NULL,
        cs_prob             NUMERIC(6,5) NOT NULL,
        recorded_at         TIMESTAMPTZ NOT NULL DEFAULT now(),
        PRIMARY KEY (fixture_id, team_id, source_recorded_at)
    )
    """,
    """
    CREATE TABLE fact_eo_snapshot (
        player_id            INT NOT NULL,
        season_id            SMALLINT NOT NULL,
        gameweek             SMALLINT NOT NULL,
        rank_band            TEXT NOT NULL,
        source               TEXT NOT NULL,
        event_time           TIMESTAMPTZ NOT NULL,
        ownership_pct        NUMERIC(6,3) NOT NULL,
        captaincy_pct        NUMERIC(6,3) NOT NULL,
        effective_ownership  NUMERIC(6,3) NOT NULL,
        provenance_url       TEXT,
        recorded_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
        PRIMARY KEY (player_id, season_id, gameweek, rank_band, source, event_time)
    )
    """,
    "CREATE INDEX ix_eo_snapshot_pit ON fact_eo_snapshot (player_id, season_id, gameweek, rank_band, recorded_at DESC)",
    """
    CREATE TABLE fact_player_match_event (
        player_id              INT NOT NULL,
        fixture_id             INT NOT NULL,
        recorded_at            TIMESTAMPTZ NOT NULL DEFAULT now(),
        passes_completed       SMALLINT,
        tackles_won            SMALLINT,
        interceptions          SMALLINT,
        blocks                 SMALLINT,
        recoveries             SMALLINT,
        dribbles_completed     SMALLINT,
        big_chances_created    SMALLINT,
        big_chances_missed     SMALLINT,
        penalties_won          SMALLINT,
        penalties_conceded     SMALLINT,
        penalties_saved        SMALLINT,
        penalties_missed       SMALLINT,
        errors_leading_to_goal SMALLINT,
        own_goals              SMALLINT,
        offsides               SMALLINT,
        was_fouled             SMALLINT,
        fouls_committed        SMALLINT,
        PRIMARY KEY (player_id, fixture_id, recorded_at)
    )
    """,
    """
    CREATE TABLE bps_rule_set (
        season_id       SMALLINT NOT NULL,
        event_code      TEXT NOT NULL,
        position_filter CHAR(3) NOT NULL DEFAULT 'ALL',  -- sentinel: 'ALL' = applies to all positions
        bps_value       SMALLINT NOT NULL,
        cap_per_match   SMALLINT,
        notes           TEXT,
        PRIMARY KEY (season_id, event_code, position_filter)
    )
    """,
    """
    CREATE TABLE ingest_audit (
        audit_id       BIGSERIAL PRIMARY KEY,
        source         TEXT NOT NULL,
        url            TEXT NOT NULL,
        request_ts     TIMESTAMPTZ NOT NULL,
        response_code  SMALLINT,
        byte_size      INT,
        content_hash   TEXT,
        raw_path       TEXT,
        parse_status   TEXT,
        parse_error    TEXT,
        user_agent     TEXT,
        recorded_at    TIMESTAMPTZ NOT NULL DEFAULT now()
    )
    """,
    "CREATE INDEX ix_ingest_audit_source_ts ON ingest_audit (source, request_ts DESC)",
]


_DROP = [
    "DROP TABLE IF EXISTS ingest_audit",
    "DROP TABLE IF EXISTS bps_rule_set",
    "DROP TABLE IF EXISTS fact_player_match_event",
    "DROP TABLE IF EXISTS fact_eo_snapshot",
    "DROP TABLE IF EXISTS fact_market_xg",
    "DROP TABLE IF EXISTS fact_odds",
    "DROP TABLE IF EXISTS fact_understat_shot",
    "DROP TABLE IF EXISTS fact_player_match",
    "DROP TABLE IF EXISTS fact_match_result",
    "DROP TABLE IF EXISTS fact_player_status",
    "DROP TABLE IF EXISTS dim_penalty_taker",
    "DROP TABLE IF EXISTS dim_fixture",
    "DROP TABLE IF EXISTS dim_player",
    "DROP TABLE IF EXISTS dim_team",
]


def upgrade() -> None:
    for stmt in _DDL:
        op.execute(stmt)


def downgrade() -> None:
    for stmt in _DROP:
        op.execute(stmt)
