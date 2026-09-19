"""External credentials never grant browser, personal-memory or integration access."""

import secrets
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import timedelta

from fastapi.encoders import jsonable_encoder
from sqlalchemy import select

from .auth import digest
from .db import session_scope
from .domain import DomainError, advisory
from .models import BotCredential, UserAccount, now

principal = ContextVar("eridani_bot_credential", default=None)
SCOPES = {
    "schema:read": "Read type, field and relationship definitions",
    "schema:write": "Preview and apply workspace schema changes (owner only)",
    "records:read": "Read custom records, including their configured work and content fields",
    "records:write": "Create and edit custom records and their relationships",
    "tasks:read": "Read tasks",
    "tasks:write": "Create and edit tasks",
    "organization:read": "Read spaces, areas, goals, projects and assignees",
    "organization:write": "Create and edit organization",
    "notes:read": "Read notes",
    "notes:write": "Create and edit notes",
    "work:run": "Ask Eri to work using this key's permitted records and tools",
}
COMMAND_SCOPES = {
    **{f"structure.{op}": "schema:write" for op in ("preview", "apply", "restore")},
    **{f"record.{op}": "records:write" for op in ("create", "update", "link")},
    **{f"task.{op}": "tasks:write" for op in ("create", "update", "complete", "reopen")},
    **{
        f"{kind}.{op}": "organization:write"
        for kind in ("space", "area", "goal", "project", "actor")
        for op in ("create", "update")
    },
    **{f"note.{op}": "notes:write" for op in ("create", "update", "append", "replace")},
}
READ_SCOPES = {
    "structure_schema": "schema:read",
    "record_list": "records:read",
    "record_search": "records:read",
    "record_get": "records:read",
    "task_list": "tasks:read",
    "task_get": "tasks:read",
    "task_resolve": "tasks:read",
    "organization_list": "organization:read",
    "project_list": "organization:read",
    "note_search": "notes:read",
    "note_read": "notes:read",
    "note_lists": "notes:read", "note_list_items": "notes:read",
    "time_resolve": None,
}


def current_id():
    from .access import execution

    grant = execution.get()
    return principal.get() or (grant or {}).get("credential_id")


@contextmanager
def bind(credential_id):
    token = principal.set(credential_id)
    try:
        yield
    finally:
        principal.reset(token)


def normalize(scopes):
    values = set(scopes)
    if not values or values - SCOPES.keys():
        raise DomainError("INVALID_ARGUMENT", "Select valid bot permissions.")
    values |= {s.replace(":write", ":read") for s in values if s.endswith(":write")}
    if "work:run" in values and not any(s.endswith(":read") for s in values):
        raise DomainError("INVALID_ARGUMENT", "Queued work needs at least one record read permission.")
    return sorted(values)


def authorize(db, owner=None, required=None, *, write=False, credential_id=None):
    from .access import role

    identity = credential_id or current_id()
    row = db.get(BotCredential, identity, populate_existing=True) if identity else None
    if not row or (owner is not None and row.owner_id != owner):
        raise DomainError("NOT_AUTHORIZED", "A valid Eridani bot key is required.", 401)
    permission = role(db, row.owner_id, row.account_id, lock=write)
    if write:
        advisory(db, "bot:" + row.id)
        db.refresh(row)
    if row.revoked_at or row.expires_at <= now() or not db.get(UserAccount, row.account_id):
        raise DomainError("ACCESS_REVOKED", "This bot key expired or was revoked.", 401)
    if required and required not in row.scopes:
        raise DomainError("INSUFFICIENT_SCOPE", f"This bot key needs {required} permission.", 403)
    if write and permission == "viewer":
        raise DomainError("READ_ONLY", "This workspace currently grants view access only.", 403)
    return row


def authenticate(request, *, touch=True):
    header = request.headers.get("authorization", "")
    scheme, _, token = header.partition(" ")
    if scheme.lower() != "bearer" or not token.startswith("eri_bot_") or len(token) > 150:
        raise DomainError("NOT_AUTHORIZED", "Use Authorization: Bearer with an Eridani bot key.", 401)
    with session_scope() as db:
        row = db.scalar(select(BotCredential).where(BotCredential.token_hash == digest(token)))
        if not row:
            raise DomainError("NOT_AUTHORIZED", "That bot key is not valid.", 401)
        row = authorize(db, credential_id=row.id)
        if not touch:
            return row.id
        advisory(db, "bot-rate:" + row.id)
        db.refresh(row)
        if now() - row.rate_window >= timedelta(minutes=1):
            row.rate_window, row.rate_count = now(), 0
        if row.rate_count >= 120:
            raise DomainError("RATE_LIMITED", "This bot reached 120 requests per minute. Retry shortly.", 429)
        row.rate_count += 1
        row.last_used_at = now()
        return row.id


def public(row):
    return jsonable_encoder(
        {
            key: getattr(row, key)
            for key in (
                "id",
                "name",
                "prefix",
                "scopes",
                "created_at",
                "expires_at",
                "revoked_at",
                "last_used_at",
            )
        }
    )


def create(db, owner, account, name, scopes, days):
    from .access import role

    permission = role(db, owner, account, lock=True)
    scopes = normalize(scopes)
    if permission == "viewer" and any(s.endswith(":write") or s == "work:run" for s in scopes):
        raise DomainError("READ_ONLY", "Viewers can create read-only keys.", 403)
    if "schema:write" in scopes and permission != "owner":
        raise DomainError("OWNER_REQUIRED", "Only the workspace owner can grant structure editing.", 403)
    # Secrets are returned once; neither the database nor receipts store the token.
    token = "eri_bot_" + secrets.token_urlsafe(32)
    row = BotCredential(
        owner_id=owner,
        account_id=account,
        name=name,
        token_hash=digest(token),
        prefix=token[:16],
        scopes=scopes,
        expires_at=now() + timedelta(days=days),
    )
    db.add(row)
    db.flush()
    return {"credential": public(row), "token": token}


def check_command(db, owner, tool, arguments):
    scope = COMMAND_SCOPES.get(tool)
    if scope is None:
        raise DomainError("INSUFFICIENT_SCOPE", "That action is not available to external bots.", 403)
    row = authorize(db, owner, scope, write=True)
    # Prevent implicit organizational creation through convenient name-based task inputs.
    if (
        tool.startswith("task.")
        and (arguments.get("project") or arguments.get("assignee") not in (None, "", "owner"))
        and "organization:write" not in row.scopes
    ):
        raise DomainError(
            "INSUFFICIENT_SCOPE", "Use existing project/assignee IDs, or grant organization:write.", 403
        )
    return row


def tool_allowed(name, scopes):
    if name in READ_SCOPES:
        required = READ_SCOPES[name]
        return required is None or required in scopes
    return COMMAND_SCOPES.get(name.replace("_", ".", 1)) in scopes


def check_tool(db, owner, name):
    row = authorize(db, owner)
    if not tool_allowed(name, row.scopes):
        raise DomainError("INSUFFICIENT_SCOPE", "This tool is outside the bot's granted permissions.", 403)
    return row
