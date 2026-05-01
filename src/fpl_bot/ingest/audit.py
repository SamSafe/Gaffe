"""Ingest audit log helper — every fetch writes a row to `ingest_audit` (§2.5)."""
from __future__ import annotations

import datetime as dt
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field

from sqlalchemy import select

from fpl_bot.db.models import IngestAudit
from fpl_bot.db.session import session_scope


@dataclass
class AuditContext:
    source: str
    url: str
    request_ts: dt.datetime
    response_code: int | None = None
    byte_size: int | None = None
    content_hash: str | None = None
    raw_path: str | None = None
    parse_status: str | None = None
    parse_error: str | None = None
    user_agent: str | None = None
    extra: dict[str, str] = field(default_factory=dict)


@contextmanager
def audit_fetch(source: str, url: str, user_agent: str | None = None) -> Iterator[AuditContext]:
    """Wrap an HTTP fetch; always writes one audit row, success or failure."""
    ctx = AuditContext(
        source=source,
        url=url,
        request_ts=dt.datetime.now(dt.UTC),
        user_agent=user_agent,
    )
    try:
        yield ctx
        if ctx.parse_status is None:
            ctx.parse_status = "ok"
    except Exception as e:
        ctx.parse_status = "failed"
        ctx.parse_error = repr(e)
        raise
    finally:
        _write_audit_row(ctx)


def _write_audit_row(ctx: AuditContext) -> None:
    with session_scope() as s:
        s.add(
            IngestAudit(
                source=ctx.source,
                url=ctx.url,
                request_ts=ctx.request_ts,
                response_code=ctx.response_code,
                byte_size=ctx.byte_size,
                content_hash=ctx.content_hash,
                raw_path=ctx.raw_path,
                parse_status=ctx.parse_status,
                parse_error=ctx.parse_error,
                user_agent=ctx.user_agent,
            )
        )


def recent_audit(source: str | None = None, limit: int = 50) -> list[dict]:
    with session_scope() as s:
        stmt = select(IngestAudit).order_by(IngestAudit.request_ts.desc()).limit(limit)
        if source:
            stmt = stmt.where(IngestAudit.source == source)
        rows = s.execute(stmt).scalars().all()
    return [
        {
            "audit_id": r.audit_id,
            "source": r.source,
            "url": r.url,
            "request_ts": r.request_ts,
            "response_code": r.response_code,
            "byte_size": r.byte_size,
            "parse_status": r.parse_status,
            "parse_error": r.parse_error,
            "raw_path": r.raw_path,
        }
        for r in rows
    ]
