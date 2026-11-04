"""Source submission and reads for bot keys: PUT/GET/DELETE /api/v1/external/sources.

Scopes: `sources:write` submits (and cannot read back), `context:read` lists and reads, `context:write` deletes.
Every route refuses when `context_enabled` is off (CONTEXT_DISABLED) and in shared workspaces
(PERSONAL_WORKSPACE). Bodies are JSON envelopes (`{"version": 1, ...}`) or raw TXT, Markdown, WebVTT or SRT
chosen by Content-Type; raw bodies take title, occurred_at, timezone and external_revision as query parameters.
"""

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, Query, Request
from pydantic import AwareDatetime
from starlette.concurrency import run_in_threadpool

from ... import bot_access
from ...db import session_scope
from ...domain import DomainError
from .. import imports, sources
from .. import settings as context_settings
from ..schema import SOURCE_STATUSES


def body_limit() -> int:
    """Largest accepted request body in bytes: the character limit with room for JSON structure."""
    return context_settings.current().max_source_chars * 2 + 100_000


async def _identity(request: Request):
    # Same as external_routes.external_identity; kept local so this module does not import external_routes.
    identity = await run_in_threadpool(bot_access.authenticate, request)
    with bot_access.bind(identity):
        yield identity


Bot = Annotated[str, Depends(_identity)]
router = APIRouter(tags=["Meeting context"])


def authorize(db, required: str, *, write: bool = False):
    if not context_settings.enabled():
        raise DomainError("CONTEXT_DISABLED", "Meeting context is not turned on.", 403)
    bot = bot_access.authorize(db, required=required, write=write)
    if bot_access.shared(db, bot.owner_id):
        raise DomainError("PERSONAL_WORKSPACE", "Meeting context works only in your personal workspace.", 403)
    return bot


async def read_body(request: Request, cap: int) -> bytes:
    chunks, size = [], 0
    async for chunk in request.stream():
        size += len(chunk)
        if size > cap:
            raise DomainError("SOURCE_TOO_LARGE", f"The request body is over {cap} bytes.", 413)
        chunks.append(chunk)
    return b"".join(chunks)


@router.put("/sources/{connector}/{external_id}")
async def submit_source(
    connector: str, external_id: str, request: Request, identity: Bot,
    title: str = Query("", max_length=500), occurred_at: AwareDatetime | None = None,
    timezone: str | None = Query(None, max_length=64), external_revision: str | None = Query(None, max_length=200),
):
    """Submit a meeting under a client label (`recorder`, `zoom`, `granola`, ...): recorded as connector `api`
    with the label kept. Idempotent by label and external ID; changed content creates a revision."""
    def check():
        with session_scope() as db:
            authorize(db, "sources:write")

    await run_in_threadpool(check)
    key = sources.api_external_id(connector, external_id)
    raw = await read_body(request, body_limit())
    text = imports.decode(raw)
    fmt = imports.detect(request.headers.get("content-type"), data=text)
    envelope = imports.parse(text, fmt, connector="api", external_id=key, title=title, occurred_at=occurred_at,
                             timezone=timezone, external_revision=external_revision)

    def save():
        with session_scope() as db:
            bot = authorize(db, "sources:write", write=True)
            result = sources.receive(db, bot.owner_id, envelope)
            return {"id": result.source_id, "connector": "api", "label": connector.strip().lower(),
                    "external_id": external_id, "revision": result.revision, "revision_id": result.revision_id,
                    "ingestion_status": result.change, "processing_status": result.status,
                    "job_id": result.job_id}

    return await run_in_threadpool(save)


@router.get("/sources")
def list_sources(identity: Bot, status: str | None = Query(None, max_length=20),
                 limit: int = Query(50, ge=1, le=100), offset: int = Query(0, ge=0, le=100_000)):
    if status is not None and status not in SOURCE_STATUSES:
        raise DomainError("INVALID_ARGUMENT", "Unknown source status.")
    with session_scope() as db:
        bot = authorize(db, "context:read")
        return sources.list_sources(db, bot.owner_id, status=status, limit=limit, offset=offset)


@router.get("/sources/{source_id}")
def get_source(source_id: UUID, identity: Bot):
    with session_scope() as db:
        bot = authorize(db, "context:read")
        return sources.get_source(db, bot.owner_id, str(source_id))


@router.delete("/sources/{source_id}")
def delete_source(source_id: UUID, identity: Bot):
    with session_scope() as db:
        bot = authorize(db, "context:write", write=True)
        return sources.delete_source(db, bot.owner_id, str(source_id))
