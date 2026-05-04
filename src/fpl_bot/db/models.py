"""SQLAlchemy 2.0 typed declarative models — schema per docs/design/phase0.md §3.2.

Append-only / bitemporal-lite: every fact carries `recorded_at`; PIT queries
filter `recorded_at <= as_of` and pick the latest row per natural key.
"""
from __future__ import annotations

import datetime as dt

from sqlalchemy import (
    CHAR,
    BigInteger,
    Boolean,
    Date,
    DateTime,
    Index,
    Integer,
    Numeric,
    SmallInteger,
    Text,
    func,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


# ─── Dimension tables ──────────────────────────────────────────────────────


class DimTeam(Base):
    __tablename__ = "dim_team"
    team_id: Mapped[int] = mapped_column(SmallInteger, primary_key=True)
    season_id: Mapped[int] = mapped_column(SmallInteger, primary_key=True)
    short_name: Mapped[str] = mapped_column(Text, nullable=False)
    full_name: Mapped[str] = mapped_column(Text, nullable=False)
    promoted: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)


class DimPlayer(Base):
    """player_id is the FPL `code` (stable across seasons), NOT the per-season element id."""

    __tablename__ = "dim_player"
    player_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    web_name: Mapped[str] = mapped_column(Text, nullable=False)
    first_name: Mapped[str | None] = mapped_column(Text)
    last_name: Mapped[str | None] = mapped_column(Text)
    understat_id: Mapped[int | None] = mapped_column(Integer)
    fbref_id: Mapped[str | None] = mapped_column(Text)


class DimFixture(Base):
    """fixture_id is the FPL fixture `code` (stable across seasons)."""

    __tablename__ = "dim_fixture"
    fixture_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    season_id: Mapped[int] = mapped_column(SmallInteger, nullable=False)
    gameweek: Mapped[int] = mapped_column(SmallInteger, nullable=False)
    kickoff_utc: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    home_team_id: Mapped[int] = mapped_column(SmallInteger, nullable=False)
    away_team_id: Mapped[int] = mapped_column(SmallInteger, nullable=False)
    finished: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)


class DimPlayerSeasonXref(Base):
    """Translates per-season FPL element_id → stable player_id (= FPL code)."""

    __tablename__ = "dim_player_season_xref"
    season_id: Mapped[int] = mapped_column(SmallInteger, primary_key=True)
    fpl_element_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    player_id: Mapped[int] = mapped_column(Integer, nullable=False)


class DimFixtureSeasonXref(Base):
    """Translates per-season FPL fixture id → stable fixture_id (= FPL code)."""

    __tablename__ = "dim_fixture_season_xref"
    season_id: Mapped[int] = mapped_column(SmallInteger, primary_key=True)
    fpl_fixture_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    fixture_id: Mapped[int] = mapped_column(Integer, nullable=False)


class DimPenaltyTaker(Base):
    """Derived-with-manual-override; manual rows take precedence (§4.6)."""

    __tablename__ = "dim_penalty_taker"
    team_id: Mapped[int] = mapped_column(SmallInteger, primary_key=True)
    rank_in_order: Mapped[int] = mapped_column(SmallInteger, primary_key=True)
    valid_from: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), primary_key=True)
    source: Mapped[str] = mapped_column(Text, primary_key=True)  # 'derived' | 'manual'
    player_id: Mapped[int] = mapped_column(Integer, nullable=False)
    override_set: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    valid_to: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True))
    recorded_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )


# ─── Fact tables (append-only) ─────────────────────────────────────────────


class FactPlayerStatus(Base):
    __tablename__ = "fact_player_status"
    player_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    recorded_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), primary_key=True, server_default=func.now()
    )
    season_id: Mapped[int] = mapped_column(SmallInteger, nullable=False)
    position_code: Mapped[str] = mapped_column(CHAR(3), nullable=False)
    team_id: Mapped[int] = mapped_column(SmallInteger, nullable=False)
    price_tenths: Mapped[int] = mapped_column(SmallInteger, nullable=False)
    status_code: Mapped[str] = mapped_column(CHAR(1), nullable=False)
    news: Mapped[str | None] = mapped_column(Text)
    chance_of_playing_next_round: Mapped[int | None] = mapped_column(SmallInteger)
    selected_by_percent: Mapped[float | None] = mapped_column(Numeric(5, 2))
    event_time: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    __table_args__ = (
        Index("ix_player_status_pit", "player_id", "recorded_at"),
    )


class FactMatchResult(Base):
    __tablename__ = "fact_match_result"
    fixture_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    recorded_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), primary_key=True, server_default=func.now()
    )
    home_score: Mapped[int | None] = mapped_column(SmallInteger)
    away_score: Mapped[int | None] = mapped_column(SmallInteger)
    finished: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)


class FactPlayerMatch(Base):
    __tablename__ = "fact_player_match"
    player_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    fixture_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    recorded_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), primary_key=True, server_default=func.now()
    )
    minutes: Mapped[int | None] = mapped_column(SmallInteger)
    goals: Mapped[int | None] = mapped_column(SmallInteger)
    assists: Mapped[int | None] = mapped_column(SmallInteger)
    clean_sheet: Mapped[bool | None] = mapped_column(Boolean)
    goals_conceded: Mapped[int | None] = mapped_column(SmallInteger)
    saves: Mapped[int | None] = mapped_column(SmallInteger)
    yellow_cards: Mapped[int | None] = mapped_column(SmallInteger)
    red_cards: Mapped[int | None] = mapped_column(SmallInteger)
    bonus: Mapped[int | None] = mapped_column(SmallInteger)
    bps: Mapped[int | None] = mapped_column(SmallInteger)
    total_points: Mapped[int | None] = mapped_column(SmallInteger)


class FactUnderstatPlayerMatch(Base):
    """Per-match Understat aggregates (round-5 fix; vaastav-mirror sourced).

    Replaces the original fact_understat_shot. We can't scrape understat.com
    directly (robots.txt Disallow), and vaastav's redistributed mirror is
    per-match aggregate, not per-shot. fixture_id and player_id are populated
    at ingest time via name/date matching; both nullable until resolved.
    """

    __tablename__ = "fact_understat_player_match"
    understat_player_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    match_date: Mapped[dt.date] = mapped_column(Date, primary_key=True)
    recorded_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), primary_key=True, server_default=func.now()
    )
    understat_match_id: Mapped[int | None] = mapped_column(Integer)
    h_team: Mapped[str | None] = mapped_column(Text)
    a_team: Mapped[str | None] = mapped_column(Text)
    fixture_id: Mapped[int | None] = mapped_column(Integer)
    player_id: Mapped[int | None] = mapped_column(Integer)
    position: Mapped[str | None] = mapped_column(Text)
    minutes: Mapped[int | None] = mapped_column(SmallInteger)
    goals: Mapped[int | None] = mapped_column(SmallInteger)
    shots: Mapped[int | None] = mapped_column(SmallInteger)
    xg: Mapped[float | None] = mapped_column(Numeric(7, 5))
    xa: Mapped[float | None] = mapped_column(Numeric(7, 5))
    key_passes: Mapped[int | None] = mapped_column(SmallInteger)
    npg: Mapped[int | None] = mapped_column(SmallInteger)
    npxg: Mapped[float | None] = mapped_column(Numeric(7, 5))
    xg_chain: Mapped[float | None] = mapped_column(Numeric(7, 5))
    xg_buildup: Mapped[float | None] = mapped_column(Numeric(7, 5))
    assists: Mapped[int | None] = mapped_column(SmallInteger)


class FactOdds(Base):
    __tablename__ = "fact_odds"
    fixture_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    bookmaker: Mapped[str] = mapped_column(Text, primary_key=True)
    market: Mapped[str] = mapped_column(Text, primary_key=True)
    selection: Mapped[str] = mapped_column(Text, primary_key=True)
    event_time: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), primary_key=True)
    decimal_odds: Mapped[float] = mapped_column(Numeric(8, 4), nullable=False)
    recorded_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )


class FactMarketXg(Base):
    """Bookmaker-implied team xG via Dixon-Coles inversion (§2.2)."""

    __tablename__ = "fact_market_xg"
    fixture_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    team_id: Mapped[int] = mapped_column(SmallInteger, primary_key=True)
    source_recorded_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), primary_key=True
    )
    lambda_: Mapped[float] = mapped_column("lambda", Numeric(6, 4), nullable=False)
    cs_prob: Mapped[float] = mapped_column(Numeric(6, 5), nullable=False)
    recorded_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )


class FactEoSnapshot(Base):
    """Effective ownership per rank band, snapshotted before kickoff (§1.2, §2.5)."""

    __tablename__ = "fact_eo_snapshot"
    player_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    season_id: Mapped[int] = mapped_column(SmallInteger, primary_key=True)
    gameweek: Mapped[int] = mapped_column(SmallInteger, primary_key=True)
    rank_band: Mapped[str] = mapped_column(Text, primary_key=True)  # top10k|top100k|overall
    source: Mapped[str] = mapped_column(Text, primary_key=True)  # livefpl|fplstatistics|fpl_api_approx
    event_time: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), primary_key=True)
    ownership_pct: Mapped[float] = mapped_column(Numeric(6, 3), nullable=False)
    captaincy_pct: Mapped[float] = mapped_column(Numeric(6, 3), nullable=False)
    effective_ownership: Mapped[float] = mapped_column(Numeric(6, 3), nullable=False)
    provenance_url: Mapped[str | None] = mapped_column(Text)
    recorded_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    __table_args__ = (
        Index(
            "ix_eo_snapshot_pit",
            "player_id",
            "season_id",
            "gameweek",
            "rank_band",
            "recorded_at",
        ),
    )


class FactPlayerMatchEvent(Base):
    """Detailed per-player per-match event counts from fbref (§4.5–§4.6)."""

    __tablename__ = "fact_player_match_event"
    player_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    fixture_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    recorded_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), primary_key=True, server_default=func.now()
    )
    passes_completed: Mapped[int | None] = mapped_column(SmallInteger)
    tackles_won: Mapped[int | None] = mapped_column(SmallInteger)
    interceptions: Mapped[int | None] = mapped_column(SmallInteger)
    blocks: Mapped[int | None] = mapped_column(SmallInteger)
    recoveries: Mapped[int | None] = mapped_column(SmallInteger)
    dribbles_completed: Mapped[int | None] = mapped_column(SmallInteger)
    big_chances_created: Mapped[int | None] = mapped_column(SmallInteger)
    big_chances_missed: Mapped[int | None] = mapped_column(SmallInteger)
    penalties_won: Mapped[int | None] = mapped_column(SmallInteger)
    penalties_conceded: Mapped[int | None] = mapped_column(SmallInteger)
    penalties_saved: Mapped[int | None] = mapped_column(SmallInteger)
    penalties_missed: Mapped[int | None] = mapped_column(SmallInteger)
    errors_leading_to_goal: Mapped[int | None] = mapped_column(SmallInteger)
    own_goals: Mapped[int | None] = mapped_column(SmallInteger)
    offsides: Mapped[int | None] = mapped_column(SmallInteger)
    was_fouled: Mapped[int | None] = mapped_column(SmallInteger)
    fouls_committed: Mapped[int | None] = mapped_column(SmallInteger)


class BpsRuleSet(Base):
    """Versioned BPS rules. position_filter='ALL' applies to all positions (sentinel)."""

    __tablename__ = "bps_rule_set"
    season_id: Mapped[int] = mapped_column(SmallInteger, primary_key=True)
    event_code: Mapped[str] = mapped_column(Text, primary_key=True)
    position_filter: Mapped[str] = mapped_column(CHAR(3), primary_key=True, default="ALL")
    bps_value: Mapped[int] = mapped_column(SmallInteger, nullable=False)
    cap_per_match: Mapped[int | None] = mapped_column(SmallInteger)
    notes: Mapped[str | None] = mapped_column(Text)


class IngestAudit(Base):
    """Compliance audit log — every fetch from any source (§2.5)."""

    __tablename__ = "ingest_audit"
    audit_id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    source: Mapped[str] = mapped_column(Text, nullable=False)
    url: Mapped[str] = mapped_column(Text, nullable=False)
    request_ts: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    response_code: Mapped[int | None] = mapped_column(SmallInteger)
    byte_size: Mapped[int | None] = mapped_column(Integer)
    content_hash: Mapped[str | None] = mapped_column(Text)
    raw_path: Mapped[str | None] = mapped_column(Text)
    parse_status: Mapped[str | None] = mapped_column(Text)
    parse_error: Mapped[str | None] = mapped_column(Text)
    user_agent: Mapped[str | None] = mapped_column(Text)
    recorded_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    __table_args__ = (
        Index("ix_ingest_audit_source_ts", "source", "request_ts"),
    )
