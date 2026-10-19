"""Validated browser inputs; credentials never belong in these payloads."""

import datetime as dt
from typing import Literal
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import BaseModel, ConfigDict, Field, field_validator


class Preferences(BaseModel):
    model_config = ConfigDict(extra="forbid")
    team_id: int | None = Field(default=None, ge=1, le=100_000_000)
    horizon: int = Field(default=6, ge=1, le=8)
    timezone: str = "America/Edmonton"
    automation_enabled: bool = False
    automation_odds: bool = True
    phone_enabled: bool = False
    # Local "HH:MM" hours when the computer is usually asleep; None disables the shift.
    quiet_start: str | None = None
    quiet_end: str | None = None
    # Private ntfy.sh topic for phone alerts; None sends nothing.
    notify_topic: str | None = Field(default=None, pattern=r"^[A-Za-z0-9_-]{12,64}$")

    @field_validator("quiet_start", "quiet_end")
    @classmethod
    def valid_clock(cls, value: str | None) -> str | None:
        if value in (None, ""):
            return None
        try:
            dt.time.fromisoformat(value)
        except ValueError as exc:
            raise ValueError("Use a 24-hour time, such as 23:00") from exc
        return value[:5]

    @field_validator("notify_topic", mode="before")
    @classmethod
    def blank_topic(cls, value):
        return value or None

    @field_validator("timezone")
    @classmethod
    def valid_timezone(cls, value: str) -> str:
        try:
            ZoneInfo(value)
        except (ZoneInfoNotFoundError, ValueError) as exc:
            raise ValueError("Use a valid timezone, such as America/Edmonton") from exc
        return value


class Transfer(BaseModel):
    model_config = ConfigDict(extra="forbid")
    out: int = Field(gt=0)
    incoming: int = Field(gt=0)


class PlanRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    transfers: list[Transfer] = Field(default_factory=list, max_length=15)
    captain: int | None = Field(default=None, gt=0)


class JobRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: Literal["refresh", "recommend", "bot", "cycle", "freeze"]
    refresh_odds: bool = False
    freeze_bot: bool = False
