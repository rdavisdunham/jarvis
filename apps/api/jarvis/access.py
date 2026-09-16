"""Membership is checked at request, command, tool and streaming boundaries.

A shared space/project has its own owner namespace. Existing owner-scoped queries
therefore cannot mix private records with shared records. Assignment never grants
membership. Personal integration credentials and learned memory are not shared.
"""

from contextvars import ContextVar

from sqlalchemy import select

from .models import AuthSession, SharedWorkspace, WorkspaceMember, now

principal = ContextVar("eridani_session_hash", default=None)
execution = ContextVar("eridani_work_principal", default=None)


def actor(db, owner):
    grant = execution.get()
    if grant:
        authorize_execution(db, owner)
        return grant["account_id"]
    session = db.get(AuthSession, principal.get()) if principal.get() else None
    return session.owner_id if session else owner


def authorize_execution(db, owner, *, write=False):
    from .domain import DomainError, advisory
    from .models import AgentWork, UserAccount

    grant = execution.get()
    if not grant:
        return None
    if write:
        advisory(db, "work:" + grant["id"])
    row = db.get(AgentWork, grant["id"], populate_existing=True)
    if (not row or row.owner_id != owner or row.account_id != grant["account_id"]
            or not db.get(UserAccount, row.account_id)):
        raise DomainError("ACCESS_REVOKED", "This request no longer has account access.", 403)
    if row.cancel_requested:
        raise DomainError("WORK_CANCELLED", "This request was cancelled. Saved actions remain.", 409)
    if row.expires_at <= now():
        raise DomainError("WORK_EXPIRED", "This request expired before completing.", 409)
    if row.revision != grant["revision"]:
        raise DomainError("WORK_CHANGED", "This request has a newer correction. Reload its instructions.", 409)
    permission = role(db, owner, row.account_id, lock=write)
    if write and permission == "viewer":
        raise DomainError("READ_ONLY", "You have view access to this workspace.", 403)
    return row.account_id, permission

PERSONAL_TOOLS = ("memory.", "settings.", "calendar.", "linear.")
PERSONAL_PATHS = (
    "/api/v1/memory",
    "/api/v1/sources",
    "/api/v1/integrations/",
    "/api/v1/export",
    "/api/v1/jobs",
    "/api/v1/push",
    "/api/v1/auth/google/",
)


def role(db, namespace, account, *, lock=False):
    from .domain import DomainError, advisory

    if lock:
        advisory(db, f"access:{namespace}")
    workspace = db.get(SharedWorkspace, namespace)
    if not workspace:
        if namespace != account:
            raise DomainError("NOT_AUTHORIZED", "This workspace is not accessible.", 403)
        return "owner"
    member = db.get(WorkspaceMember, (namespace, account), populate_existing=True, with_for_update=lock)
    if not member or not member.active:
        raise DomainError("ACCESS_REVOKED", "Workspace access ended. Return to your personal workspace.", 403)
    return member.role


def identity(db, session, *, personal=False):
    from .domain import DomainError

    if not session or session.expires_at <= now():
        raise DomainError("NOT_AUTHORIZED", "Sign in to Eridani.", 401)
    namespace = session.owner_id if personal else session.workspace_id or session.owner_id
    return namespace, role(db, namespace, session.owner_id)


def check_device(owner, device):
    from .db import session_scope
    from .domain import DomainError

    with session_scope() as db:
        if execution.get():
            authorize_execution(db, owner)
            if device and device != execution.get()["device_id"]:
                raise DomainError("ACCESS_REVOKED", "This request belongs to another device.", 403)
            return None
        shared=db.get(SharedWorkspace,owner)
        session_hash=principal.get()
        if not shared and not session_hash:
            return None  # Trusted internal/private jobs have no interactive session.
        session=db.get(AuthSession,session_hash) if session_hash else db.scalar(
            select(AuthSession).where(AuthSession.device_id==device))
        namespace,permission=identity(db,session)
        if namespace!=owner or (device and session.device_id!=device):
            raise DomainError("ACCESS_REVOKED","The active workspace changed. Start a fresh conversation.",403)
        return session.token_hash,permission


def request_access(request, namespace, permission, account):
    from .domain import DomainError

    if namespace == account:
        return
    path = request.url.path
    if path.startswith(PERSONAL_PATHS):
        raise DomainError(
            "PERSONAL_WORKSPACE", "Use your personal workspace for memory and connected accounts.", 403
        )
    reads = {
        "/api/v1/chat",
        "/api/v1/conversations",
        "/api/v1/ui/sync",
        "/api/v1/task-views",
        "/api/v1/task-views/remove",
    }
    if (
        permission == "viewer"
        and request.method not in {"GET", "HEAD", "OPTIONS"}
        and path not in reads
        and not path.startswith(("/api/v1/voice", "/api/v1/work"))
    ):
        raise DomainError("READ_ONLY", "You have view access to this workspace.", 403)


def command_access(db, owner, tool, arguments):
    from .domain import DomainError, advisory

    grant = authorize_execution(db, owner, write=True)
    workspace = db.get(SharedWorkspace, owner)
    if not workspace:
        return
    advisory(db, f"access:{owner}")
    if grant:
        from types import SimpleNamespace
        account, permission = grant
        session = SimpleNamespace(owner_id=account)
        namespace = owner
    else:
        session = db.get(AuthSession, principal.get(), populate_existing=True) if principal.get() else None
        namespace, permission = identity(db, session)
    if namespace != owner:
        raise DomainError("ACCESS_REVOKED", "The active workspace changed.", 403)
    if permission == "viewer":
        raise DomainError("READ_ONLY", "You have view access to this workspace.", 403)
    if tool.startswith(PERSONAL_TOOLS) or arguments.get("google_calendar_id"):
        raise DomainError(
            "PERSONAL_WORKSPACE",
            "Connected accounts and personal memory stay in your personal workspace.",
            403,
        )
    if (
        tool in {"task.create", "note.create"}
        and not arguments.get("space_id")
        and not arguments.get("project_id")
    ):
        arguments["project_id" if workspace.kind == "project" else "space_id"] = workspace.root_id
    if (
        tool == "task.create"
        and not arguments.get("assignee_id")
        and arguments.get("assignee", "owner") == "owner"
    ):
        member = db.get(WorkspaceMember, (owner, session.owner_id))
        if member.actor_id:
            arguments["assignee_id"] = member.actor_id
            arguments.pop("assignee", None)


def tool_access(db, owner, name):
    from .domain import DomainError

    if not db.get(SharedWorkspace, owner):
        return
    if name.startswith(("memory_", "linear_", "settings_")) or name in {
        "calendar_event_read",
        "calendar_connection",
        "calendar_sync",
        "calendar_create",
        "calendar_update",
        "calendar_delete",
        "calendar_select",
        "calendar_write_status",
    }:
        raise DomainError(
            "PERSONAL_WORKSPACE", "Use the personal workspace for private memory and integrations.", 403
        )


def shared_preferences(db, owner, values):
    workspace = db.get(SharedWorkspace, owner)
    if not workspace:
        return values
    # Conversations and memories are private, even when the records being discussed are shared.
    return {
        **values,
        "preferred_name": "workspace member",
        "history_enabled": False,
        "memory_learning": False,
        "deep_sleep_enabled": False,
        "shared_workspace": workspace.name,
    }


def assert_current(owner, device):
    check_device(owner, device)


def person_preferences(db, owner, device, values):
    if not db.get(SharedWorkspace, owner):
        return values
    grant = execution.get()
    session = db.scalar(select(AuthSession).where(AuthSession.device_id == device)) if not grant else None
    account = grant["account_id"] if grant else session.owner_id if session else None
    if not account:
        return values
    from .domain import preferences

    personal = preferences(db, account)
    return {
        **values,
        "shared_role":role(db,owner,account),
        **{
            key: personal[key]
            for key in ("preferred_name", "agent_provider", "timezone", "default_reminder_hour")
        },
    }
