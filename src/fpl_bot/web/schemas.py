"""Validated browser inputs; credentials never belong in these payloads."""

from typing import Literal
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import BaseModel, ConfigDict, Field, field_validator


class Preferences(BaseModel):
    model_config = ConfigDict(extra="forbid")
    team_id: int | None = Field(default=None, ge=1, le=100_000_000)
    horizon: int = Field(default=6, ge=1, le=8)
    timezone: str = "America/Edmonton"

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
    kind: Literal["refresh", "recommend"]
    refresh_odds: bool = False
