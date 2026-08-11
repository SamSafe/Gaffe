"""Pure tests for current FPL bearer-token authentication."""

from __future__ import annotations

import base64
import datetime as dt
import json

import pytest

from fpl_bot.ingest.fpl_api import _authenticated_fpl_headers


def _jwt_with_expiry(expiry: dt.datetime) -> str:
    def encode(value: dict[str, object]) -> str:
        raw = json.dumps(value, separators=(",", ":")).encode()
        return base64.urlsafe_b64encode(raw).decode().rstrip("=")

    return f"{encode({'alg': 'none'})}.{encode({'exp': expiry.timestamp()})}.signature"


def test_explicit_access_token_builds_api_authorization_header() -> None:
    headers = _authenticated_fpl_headers(
        access_token="Bearer current-token",
        cookie_header="other=value",
    )

    assert headers == {
        "X-API-Authorization": "Bearer current-token",
        "Cookie": "other=value",
    }


def test_access_token_cookie_is_a_backward_compatible_fallback() -> None:
    headers = _authenticated_fpl_headers(
        access_token=None,
        cookie_header="other=value; access_token=current-token",
    )

    assert headers["X-API-Authorization"] == "Bearer current-token"


def test_expired_jwt_gets_an_actionable_error_before_the_request() -> None:
    now = dt.datetime(2026, 8, 11, tzinfo=dt.UTC)
    token = _jwt_with_expiry(now - dt.timedelta(minutes=1))

    with pytest.raises(RuntimeError, match="expired"):
        _authenticated_fpl_headers(
            access_token=token,
            cookie_header=None,
            now=now,
        )


def test_missing_auth_gets_an_actionable_error() -> None:
    with pytest.raises(RuntimeError, match="FPL_BOT_FPL_ACCESS_TOKEN"):
        _authenticated_fpl_headers(access_token=None, cookie_header=None)
