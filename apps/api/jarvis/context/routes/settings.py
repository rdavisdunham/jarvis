"""Sources settings for the signed-in owner (browser session, not bot keys): /api/v1/context/settings.

Mounted through `routes.owner_router`, never the external router, because `/api/v1/external/*` responses are
re-authenticated as bot keys. Every route refuses when `context_enabled` is off (CONTEXT_DISABLED) and in shared
workspaces (PERSONAL_WORKSPACE). The live sources without any account are file import (connector `upload`)
and the submission endpoint (`PUT /api/v1/external/sources/{connector}/{external_id}`, `sources:write`). The
Granola REST adapter is optional: its routes refuse unless `context_granola_enabled` is set, the key is kept
only as ciphertext (`granola.seal_credential`) and is never returned or logged.

Owner preferences (`PATCH /settings/preferences`) are an allow-list over `ContextOwnerState.settings`:
`synthesize_briefs`, `semantic_supersede`, `lint_budget_usd` and `review_destination`, written under the owner-state
row lock. Other keys in that JSON (`engine`, lint progress, the remembered Linear team) are never exposed or changed
here. Like `routes/owner.py`, the route runs under `service.as_owner(user)` and `service.authorize(write=True)`, and
`auth.authenticate` requires the CSRF header; bot keys cannot reach the session router.
"""

import hashlib
from datetime import UTC, datetime
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, Query, Request
from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, StrictBool, model_validator
from sqlalchemy import func, select
from starlette.concurrency import run_in_threadpool

from ... import bot_access
from ...auth import Identity, authenticate
from ...config import get_settings
from ...db import session_scope
from ...domain import DomainError, advisory, enqueue_job
from ...models import Job
from .. import connectors, imports, service, sources
from .. import settings as context_settings
from ..connectors import granola
from ..lint import DEFAULT_BUDGET_USD, RECONCILE_RESERVE_USD
from ..models import ContextConnection, ContextOwnerState, ContextSource, ContextSourceRevision
from ..proposals import DESTINATIONS
from .admin import lock_state
from .sources import body_limit, read_body

User = Annotated[Identity, Depends(authenticate)]
router = APIRouter(prefix="/settings", tags=["Meeting context settings"])

SYNC_JOB = "context_granola_sync"
BACKFILL_JOB = "context_granola_backfill"
SUBMISSION_LABELS = ("recorder", "zoom", "granola", "other")
LINT_BUDGET_MAX_USD = 5.0  # 100 reconcile calls a week at the lint's per-call reserve
PREFERENCE_DEFAULTS = {"synthesize_briefs": False, "semantic_supersede": False, "lint_budget_usd": DEFAULT_BUDGET_USD,
                       "review_destination": "eridani"}
ENTITLEMENT = ("Granola refused this key. Granola's API needs a Business or Enterprise workspace; on other plans, "
               "bring meetings in with file import or the submission endpoint instead.")


def owner_of(db, user: Identity, *, write: bool = False) -> str:
    if not context_settings.enabled():
        raise DomainError("CONTEXT_DISABLED", "Meeting context is not turned on.", 403)
    if bot_access.shared(db, user.owner_id):
        raise DomainError("PERSONAL_WORKSPACE", "Meeting context works only in your personal workspace.", 403)
    if write and user.role != "owner":
        raise DomainError("NOT_AUTHORIZED", "Only the workspace owner can change meeting sources.", 403)
    return user.owner_id


def require_granola() -> None:
    if not context_settings.current().granola_enabled:
        raise DomainError("GRANOLA_DISABLED", "The Granola connection is not turned on for this server.", 409)


def granola_client(api_key: str) -> granola.GranolaClient:
    """A client for one-off checks: no retries, so a bad key or an outage answers at once. Tests replace it."""
    return granola.GranolaClient(api_key, max_retries=0)


def iso(value: datetime | None) -> str | None:
    return value.isoformat() if value else None


# ---------------------------------------------------------------- reads


def deliveries(db, owner: str) -> list[dict]:
    """Per connector label: how many sources it delivered and when content last arrived."""
    label = func.coalesce(ContextSourceRevision.payload["metadata"][connectors.LABEL_KEY].astext,
                          ContextSource.connector)
    rows = db.execute(
        select(ContextSource.connector, label, func.count(func.distinct(ContextSource.id)),
               func.max(ContextSourceRevision.received_at))
        .join(ContextSourceRevision, ContextSourceRevision.source_id == ContextSource.id)
        .where(ContextSource.owner_id == owner, ContextSource.status != "deleted")
        .group_by(ContextSource.connector, label)
        .order_by(func.max(ContextSourceRevision.received_at).desc())).all()
    return [{"connector": connector, "label": name, "sources": count, "last_delivery_at": iso(last)}
            for connector, name, count, last in rows]


def public_connection(connection: ContextConnection) -> dict:
    return {"id": connection.id, "connector": connection.connector, "account": connection.external_account,
            "has_key": bool(connection.credential_ciphertext), "enabled": connection.enabled,
            "health": connection.health, "last_error": connection.last_error,
            "last_success_at": iso(connection.last_success_at), "created_at": iso(connection.created_at)}


def granola_connections(db, owner: str) -> list[ContextConnection]:
    return list(db.scalars(select(ContextConnection).where(
        ContextConnection.owner_id == owner, ContextConnection.connector == "granola",
        ContextConnection.enabled.is_(True)).order_by(ContextConnection.created_at)))


def limits() -> dict:
    return {"max_source_chars": context_settings.current().max_source_chars, "max_segments": sources.MAX_SEGMENTS,
            "max_body_bytes": body_limit(), "formats": list(imports.FORMATS),
            "extensions": sorted(imports.EXTENSIONS)}


def submission() -> dict:
    base = get_settings().origin.rstrip("/")
    return {"method": "PUT", "url": base + "/api/v1/external/sources/{connector}/{external_id}",
            "scope": "sources:write", "labels": list(SUBMISSION_LABELS),
            "bridge": f"ERIDANI_SOURCES_KEY=<key> python scripts/bridge_submit.py ./exports --base {base} "
                      "--connector recorder"}


@router.get("")
def overview(user: User):
    """Adapters, deliveries per connector label, submission endpoint, limits and (when enabled) Granola."""
    granola_on = context_settings.current().granola_enabled
    with session_scope() as db:
        owner = owner_of(db, user)
        connections = [public_connection(c) for c in granola_connections(db, owner)] if granola_on else []
        delivered = deliveries(db, owner)
        current = preferences_of(db.get(ContextOwnerState, owner))
    live = {"upload": True, "api": True, "granola": bool(granola_on and connections), "zoom": False}
    return {
        "preferences": current,
        "preference_limits": {"lint_budget_usd": {"min": 0.0, "max": LINT_BUDGET_MAX_USD, "default": DEFAULT_BUDGET_USD,
                                                  "per_call": RECONCILE_RESERVE_USD},
                              "review_destination": list(DESTINATIONS)},
        "adapters": [{"key": key, "description": adapter.description, "live": live[key]}
                     for key, adapter in connectors.ADAPTERS.items()],
        "deliveries": delivered,
        "submission": submission(),
        "limits": limits(),
        "granola": {"enabled": granola_on, "connections": connections},
    }


# ---------------------------------------------------------------- owner preferences


def preferences_of(state: ContextOwnerState | None) -> dict:
    """The allow-listed owner settings with their effective values (what the pipeline reads when a key is absent)."""
    stored = (state.settings or {}) if state else {}
    values = {key: stored.get(key, default) for key, default in PREFERENCE_DEFAULTS.items()}
    values["synthesize_briefs"] = bool(values["synthesize_briefs"])
    values["semantic_supersede"] = bool(values["semantic_supersede"])
    values["lint_budget_usd"] = float(values["lint_budget_usd"])
    if values["review_destination"] not in DESTINATIONS:
        values["review_destination"] = "eridani"
    return values


class Preferences(BaseModel):
    """Changed fields only; anything else in the body is refused."""
    model_config = ConfigDict(extra="forbid")
    synthesize_briefs: StrictBool | None = None
    semantic_supersede: StrictBool | None = None
    lint_budget_usd: float | None = Field(None, ge=0, le=LINT_BUDGET_MAX_USD, allow_inf_nan=False)
    review_destination: Literal["eridani", "linear"] | None = None

    @model_validator(mode="after")
    def one_change(self):
        if not self.model_fields_set or any(getattr(self, name) is None for name in self.model_fields_set):
            raise ValueError("Send at least one setting, each with a value.")
        return self


@router.patch("/preferences")
def update_preferences(body: Preferences, user: User):
    with service.as_owner(user), session_scope() as db:
        caller = service.authorize(db, write=True)
        if not caller.session:
            raise DomainError("NOT_AUTHORIZED", "Only you can change meeting memory settings, from Settings.", 403)
        state = lock_state(db, caller.owner)
        values = {**(state.settings or {}), **body.model_dump(include=body.model_fields_set)}
        if values != (state.settings or {}):
            state.settings, state.updated_at = values, datetime.now(UTC)
        db.flush()
        return {"preferences": preferences_of(state)}


# ---------------------------------------------------------------- file import


@router.post("/import")
async def import_file(
    request: Request, user: User, filename: str = Query("", max_length=255), title: str = Query("", max_length=500),
    occurred_at: AwareDatetime | None = None, timezone: str | None = Query(None, max_length=64),
    external_id: str | None = Query(None, max_length=400),
):
    """One uploaded file (raw body; the format comes from Content-Type, else the file name) received as connector
    `upload`. Without `external_id` the file's bytes name it, so importing the same file twice creates nothing."""
    def check():
        with session_scope() as db:
            owner_of(db, user, write=True)

    await run_in_threadpool(check)
    raw = await read_body(request, body_limit())
    if not raw.strip():
        raise imports.bad("The file is empty.")
    text = imports.decode(raw)
    fmt = imports.detect(request.headers.get("content-type"), filename=filename or None, data=text)
    external_id = external_id or "sha256-" + hashlib.sha256(raw).hexdigest()[:32]
    sources.check_label("upload", external_id)
    stem = filename.rsplit("/", 1)[-1].rsplit("\\", 1)[-1].rsplit(".", 1)[0]
    envelope = imports.parse(text, fmt, connector="upload", external_id=external_id, title=title,
                             occurred_at=occurred_at, timezone=timezone)
    if not envelope.title and stem:  # a Markdown heading or JSON title wins over the file name
        envelope = envelope.model_copy(update={"title": stem[:500]})

    def save():
        with session_scope() as db:
            owner = owner_of(db, user, write=True)
            result = connectors.receive(db, owner, "upload", envelope)
            return {"id": result.source_id, "external_id": external_id, "title": envelope.title, "format": fmt,
                    "revision": result.revision, "ingestion_status": result.change,
                    "processing_status": result.status}

    return await run_in_threadpool(save)


# ---------------------------------------------------------------- Granola (optional)


class GranolaKey(BaseModel):
    model_config = ConfigDict(extra="forbid")
    api_key: str = Field(min_length=10, max_length=1000)
    account: str = Field(min_length=3, max_length=320, pattern=r"^[^@\s]+@[^@\s]+$")


class Backfill(BaseModel):
    model_config = ConfigDict(extra="forbid")
    days: int = Field(90, ge=1, le=365)


def verify(api_key: str, account: str) -> dict:
    """One page of notes: proves the key works and the workspace is entitled. Upstream errors are explained."""
    try:
        with granola_client(api_key) as client:
            page = client.list_notes()
    except granola.GranolaAuthFailed as error:
        raise DomainError("GRANOLA_KEY_REJECTED", "Granola did not accept this API key. Check it was copied whole "
                          "and has not been revoked.", 400) from error
    except granola.GranolaError as error:
        if error.retryable:
            raise DomainError("GRANOLA_UNAVAILABLE", "Granola did not answer just now. Try again in a few minutes.",
                              503) from error
        if error.code == "forbidden":
            raise DomainError("GRANOLA_NOT_ENTITLED", ENTITLEMENT, 400) from error
        raise DomainError("GRANOLA_REJECTED", f"Granola refused the check ({error.code}).", 400) from error
    owned = sum(1 for note in page.notes
                if str((note.get("owner") or {}).get("email") or "").strip().lower() == account)
    return {"listed": len(page.notes), "owned": owned}


def owned_connection(db, owner: str, connection_id: str) -> ContextConnection:
    connection = db.get(ContextConnection, connection_id)
    if connection is None or connection.owner_id != owner or connection.connector != "granola" or not connection.enabled:
        raise DomainError("NOT_FOUND", "That Granola connection was not found.", 404)
    return connection


@router.post("/granola")
async def granola_connect(body: GranolaKey, user: User):
    """Verify the key with one cheap list call, then store it sealed. One Granola account per owner; connecting the
    same account again replaces its key."""
    account, api_key = body.account.strip().lower(), body.api_key.strip()

    def check():
        require_granola()
        with session_scope() as db:
            owner = owner_of(db, user, write=True)
            other = next((c for c in granola_connections(db, owner) if c.external_account != account), None)
            if other is not None:
                raise DomainError("GRANOLA_CONNECTED", f"Granola is already connected as {other.external_account}. "
                                  "Remove that connection first.", 409)

    await run_in_threadpool(check)
    checked = await run_in_threadpool(verify, api_key, account)

    def save():
        with session_scope() as db:
            owner = owner_of(db, user, write=True)
            advisory(db, f"context-connection:{owner}:granola")
            when = datetime.now(UTC)
            connection = db.scalar(select(ContextConnection).where(
                ContextConnection.owner_id == owner, ContextConnection.connector == "granola",
                ContextConnection.external_account == account))
            if connection is None:
                connection = ContextConnection(owner_id=owner, connector="granola", external_account=account,
                                               created_at=when)
                db.add(connection)
            connection.credential_ciphertext = granola.seal_credential(api_key)
            connection.enabled, connection.health, connection.last_error = True, "ok", None
            connection.updated_at = when
            db.flush()
            return {**public_connection(connection), "verification": checked}

    return await run_in_threadpool(save)


@router.post("/granola/{connection_id}/test")
async def granola_test(connection_id: str, user: User):
    """Re-check the stored key and record the result as the connection's health."""
    def load():
        require_granola()
        with session_scope() as db:
            connection = owned_connection(db, owner_of(db, user, write=True), connection_id)
            key = granola.open_credential(connection)
            if not key:
                raise DomainError("GRANOLA_NO_KEY", "This connection has no API key. Connect Granola again.", 409)
            return key, connection.external_account

    api_key, account = await run_in_threadpool(load)
    try:
        checked, health, error = await run_in_threadpool(verify, api_key, account), "ok", None
    except DomainError as failure:
        checked, error = None, failure.message
        health = {"GRANOLA_KEY_REJECTED": "auth_failed", "GRANOLA_UNAVAILABLE": "degraded",
                  "GRANOLA_NOT_ENTITLED": "not_entitled"}.get(failure.code, "rejected")

    def record():
        with session_scope() as db:
            connection = owned_connection(db, owner_of(db, user, write=True), connection_id)
            connection.health, connection.last_error = health, error
            connection.updated_at = datetime.now(UTC)
            return {**public_connection(connection), "ok": health == "ok", "verification": checked}

    return await run_in_threadpool(record)


@router.delete("/granola/{connection_id}")
def granola_remove(connection_id: str, user: User):
    """Forget the key and stop polling. Imported meetings stay: they are Eridani's own copy."""
    with session_scope() as db:
        owner = owner_of(db, user, write=True)
        advisory(db, f"context-connection:{owner}:granola")
        connection = owned_connection(db, owner, connection_id)
        connection.credential_ciphertext, connection.enabled = None, False
        connection.health, connection.last_error = "removed", None
        connection.updated_at = datetime.now(UTC)
        return {"id": connection.id, "removed": True}


def queue(db, owner: str, connection: ContextConnection, kind: str, payload: dict) -> dict:
    if not connection.credential_ciphertext and not context_settings.current().granola_api_key:
        raise DomainError("GRANOLA_NO_KEY", "This connection has no API key. Connect Granola again.", 409)
    advisory(db, f"context-connection:{owner}:granola")
    pending = db.scalar(select(Job).where(
        Job.owner_id == owner, Job.kind == kind, Job.status.in_(["queued", "running"]),
        Job.payload["connection_id"].astext == connection.id).order_by(Job.created_at.desc()))
    if pending is not None and pending.payload == payload:
        return {"job_id": pending.id, "status": pending.status, "already_queued": True}
    return {"job_id": enqueue_job(db, owner, kind, payload).id, "status": "queued", "already_queued": False}


@router.post("/granola/{connection_id}/sync")
def granola_sync_now(connection_id: str, user: User):
    require_granola()
    with session_scope() as db:
        owner = owner_of(db, user, write=True)
        connection = owned_connection(db, owner, connection_id)
        return queue(db, owner, connection, SYNC_JOB, {"connection_id": connection.id})


@router.post("/granola/{connection_id}/backfill")
def granola_backfill(connection_id: str, body: Backfill, user: User):
    require_granola()
    with session_scope() as db:
        owner = owner_of(db, user, write=True)
        connection = owned_connection(db, owner, connection_id)
        return queue(db, owner, connection, BACKFILL_JOB, {"connection_id": connection.id, "days": body.days})
