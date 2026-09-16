"""Invited accounts and explicitly shared space/project workspaces."""

from datetime import timedelta
from typing import Annotated, Literal
from uuid import NAMESPACE_URL, uuid5

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import func, select

from . import access
from .auth import Identity, authenticate_personal, digest
from .config import get_settings
from .db import session_scope
from .domain import DomainError, advisory, emit, preferences
from .models import (
    Actor,
    AuthSession,
    GoogleIdentity,
    OwnerSettings,
    Project,
    SharedWorkspace,
    Space,
    UserAccount,
    WorkspaceInvite,
    WorkspaceMember,
    now,
    uid,
)

router = APIRouter(prefix="/api/v1/accounts")
User = Annotated[Identity, Depends(authenticate_personal)]


def ensure_account(db, account, name=None):
    row = db.get(UserAccount, account)
    if not row:
        row = UserAccount(
            id=account,
            name=(
                name or (get_settings().owner_name if account == get_settings().owner_id else "New member")
            )[:100],
        )
        db.add(row)
        db.flush()
        if account != get_settings().owner_id and not db.get(OwnerSettings, account):
            db.add(OwnerSettings(owner_id=account, values={"preferred_name": row.name}))
            db.flush()
    return row


def admin(db, workspace, account):
    if access.role(db, workspace, account, lock=True) != "owner":
        raise DomainError("NOT_AUTHORIZED", "Only the workspace owner can manage sharing.", 403)


def invite_public(db, row):
    workspace = db.get(SharedWorkspace, row.workspace_id) if row.workspace_id else None
    return {
        "id": row.id,
        "workspace_id": row.workspace_id,
        "workspace": workspace.name if workspace else "Personal account",
        "email": row.email,
        "role": row.role,
        "status": row.status,
        "expires_at": row.expires_at.isoformat(),
    }


class Body(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class WorkspaceCreate(Body):
    id: str = Field(default_factory=uid, min_length=36, max_length=36)
    name: str = Field(min_length=1, max_length=200)
    kind: Literal["space", "project"] = "space"


class Switch(Body):
    workspace_id: str | None = None


class Invite(Body):
    workspace_id: str | None = None
    email: str = Field(min_length=3, max_length=320, pattern=r"^[^\s@]+@[^\s@]+\.[^\s@]+$")
    role: Literal["viewer", "editor"] = "editor"


class InviteAction(Body):
    invite_id: str = Field(max_length=36)


class MemberChange(Body):
    workspace_id: str = Field(max_length=36)
    account_id: str = Field(max_length=100)
    role: Literal["viewer", "editor"] = "editor"
    active: bool = True
    expected_revision: int = Field(ge=1)


@router.get("")
def overview(request: Request, user: User):
    with session_scope() as db:
        ensure_account(db, user.owner_id)
        session = db.get(AuthSession, digest(request.cookies.get("jarvis_session", "")))
        workspaces = []
        for workspace, member in db.execute(
            select(SharedWorkspace, WorkspaceMember)
            .join(WorkspaceMember, WorkspaceMember.workspace_id == SharedWorkspace.id)
            .where(WorkspaceMember.account_id == user.owner_id, WorkspaceMember.active.is_(True))
        ):
            workspaces.append(
                {"id": workspace.id, "name": workspace.name, "kind": workspace.kind, "role": member.role}
            )
        account = db.get(GoogleIdentity, user.owner_id)
        incoming = (
            []
            if not account
            else [
                invite_public(db, r)
                for r in db.scalars(
                    select(WorkspaceInvite).where(
                        WorkspaceInvite.email == account.email.casefold(),
                        WorkspaceInvite.status == "pending",
                        WorkspaceInvite.expires_at > now(),
                    )
                )
            ]
        )
        return {
            "account_id": user.owner_id,
            "name": preferences(db, user.owner_id)["preferred_name"],
            "email": account.email if account else None,
            "active_workspace_id": session.workspace_id,
            "workspaces": workspaces,
            "invitations": incoming,
            "can_invite_accounts": user.owner_id == get_settings().owner_id,
            "outgoing": [
                invite_public(db, r)
                for r in db.scalars(
                    select(WorkspaceInvite).where(
                        WorkspaceInvite.inviter_id == user.owner_id,
                        WorkspaceInvite.workspace_id.is_(None),
                        WorkspaceInvite.status == "pending",
                        WorkspaceInvite.expires_at > now(),
                    )
                )
            ],
        }


@router.post("/workspaces")
def create_workspace(body: WorkspaceCreate, user: User):
    with session_scope() as db:
        ensure_account(db, user.owner_id)
        advisory(db, f"new-workspace:{user.owner_id}")
        workspace_id = str(uuid5(NAMESPACE_URL, "eridani/shared/" + user.owner_id + "/" + body.id))
        if db.get(UserAccount, workspace_id):
            raise DomainError("REVISION_CONFLICT", "Workspace identity is unavailable.", 409)
        prior = db.get(SharedWorkspace, workspace_id)
        if prior:
            if prior.creator_id == user.owner_id and prior.name == body.name and prior.kind == body.kind:
                return {"id": prior.id, "name": prior.name, "kind": prior.kind}
            raise DomainError("REVISION_CONFLICT", "Workspace request identity is already used.", 409)
        if (
            db.scalar(
                select(func.count())
                .select_from(SharedWorkspace)
                .where(SharedWorkspace.creator_id == user.owner_id)
            )
            >= 50
        ):
            raise DomainError("LIMIT_EXCEEDED", "This account already owns 50 shared workspaces.")
        workspace = SharedWorkspace(id=workspace_id, creator_id=user.owner_id, name=body.name, kind=body.kind)
        db.add(workspace)
        db.flush()
        db.add(
            OwnerSettings(
                owner_id=workspace.id,
                values={
                    "productivity_initialized": True,
                    "history_enabled": False,
                    "memory_learning": False,
                    "deep_sleep_enabled": False,
                    "timezone": preferences(db, user.owner_id)["timezone"],
                },
            )
        )
        root = Space(owner_id=workspace.id, name=body.name)
        actor = Actor(
            owner_id=workspace.id, name=preferences(db, user.owner_id)["preferred_name"], kind="person"
        )
        if actor.name == "Eri":
            actor.name = "Eri (person)"
        db.add_all([root, actor, Actor(owner_id=workspace.id, name="Eri", kind="agent")])
        db.flush()
        workspace.root_id = root.id
        if body.kind == "project":
            project = Project(owner_id=workspace.id, name=body.name, space_id=root.id)
            db.add(project)
            db.flush()
            workspace.root_id = project.id
        db.add(
            WorkspaceMember(
                workspace_id=workspace.id, account_id=user.owner_id, role="owner", actor_id=actor.id
            )
        )
        return {"id": workspace.id, "name": workspace.name, "kind": workspace.kind}


@router.post("/switch")
def switch_workspace(body: Switch, request: Request, user: User):
    with session_scope() as db:
        session = db.get(AuthSession, digest(request.cookies["jarvis_session"]))
        for namespace in sorted({x for x in (session.workspace_id, body.workspace_id) if x}):
            advisory(db, f"access:{namespace}")
        if body.workspace_id:
            access.role(db, body.workspace_id, user.owner_id, lock=True)
            if not db.get(SharedWorkspace, body.workspace_id):
                raise DomainError("NOT_FOUND", "Choose an accessible shared workspace.", 404)
        session = db.get(AuthSession, digest(request.cookies["jarvis_session"]), with_for_update=True)
        session.workspace_id = body.workspace_id
        # Isolate conversations/UI state across switches, including old browser tabs.
        session.device_id = uid()
        return {"switched": True}


@router.post("/invitations")
def invite(body: Invite, user: User):
    with session_scope() as db:
        ensure_account(db, user.owner_id)
        if body.workspace_id:
            admin(db, body.workspace_id, user.owner_id)
        elif user.owner_id != get_settings().owner_id:
            raise DomainError("NOT_AUTHORIZED", "Only the server owner can invite a standalone account.", 403)
        email = body.email.casefold()
        advisory(db, f"invite:{body.workspace_id}:{email}")
        prior = db.scalar(
            select(WorkspaceInvite).where(
                WorkspaceInvite.workspace_id == body.workspace_id,
                WorkspaceInvite.email == email,
                WorkspaceInvite.status == "pending",
                WorkspaceInvite.expires_at > now(),
            )
        )
        if prior:
            if prior.role != body.role:
                raise DomainError(
                    "REVISION_CONFLICT",
                    "Revoke the pending invitation before choosing a different role.",
                    409,
                )
            return invite_public(db, prior)
        row = WorkspaceInvite(
            inviter_id=user.owner_id,
            workspace_id=body.workspace_id,
            email=email,
            role=body.role,
            expires_at=now() + timedelta(days=7),
        )
        db.add(row)
        db.flush()
        return invite_public(db, row)


@router.get("/invitations/{invite_id}")
def invitation_detail(invite_id: str, user: User):
    with session_scope() as db:
        row = db.get(WorkspaceInvite, invite_id)
        account = db.get(GoogleIdentity, user.owner_id)
        if not row or not account or row.email != account.email.casefold():
            raise DomainError("NOT_AUTHORIZED", "This link is unavailable for this Google account. Use the invited account or ask its sender for a new link.", 403)
        result = invite_public(db, row)
        if row.status == "pending" and row.expires_at <= now():
            result["status"] = "expired"
        return result


@router.post("/accept")
def accept(body: InviteAction, user: User):
    with session_scope() as db:
        row = db.get(WorkspaceInvite, body.invite_id, with_for_update=True)
        account = db.get(GoogleIdentity, user.owner_id)
        if not row or not account or row.email != account.email.casefold():
            raise DomainError(
                "NOT_AUTHORIZED", "Sign in with the Google account invited to this workspace.", 403
            )
        if row.status == "accepted" and row.accepted_by == user.owner_id:
            return {"accepted": True, "workspace_id": row.workspace_id}
        if row.status != "pending" or row.expires_at <= now():
            raise DomainError("INVITE_EXPIRED", "This invitation expired or was revoked.", 409)
        if row.workspace_id:
            access.role(db, row.workspace_id, row.inviter_id, lock=True)
            member = db.get(WorkspaceMember, (row.workspace_id, user.owner_id))
            if member is None:
                name = preferences(db, user.owner_id)["preferred_name"]
                if db.scalar(select(Actor).where(Actor.owner_id == row.workspace_id, Actor.name == name)):
                    name = name[:60] + " · " + user.owner_id[:8]
                actor = Actor(owner_id=row.workspace_id, name=name, kind="person")
                db.add(actor)
                db.flush()
                member = WorkspaceMember(
                    workspace_id=row.workspace_id, account_id=user.owner_id, role=row.role, actor_id=actor.id
                )
                db.add(member)
            else:
                member.active = True
                if member.role != "owner":
                    member.role = row.role
                member.revision += 1
                if member.actor_id:
                    db.get(Actor, member.actor_id).archived = False
            emit(db, row.workspace_id, "membership.changed", row.workspace_id)
        row.status, row.accepted_by = "accepted", user.owner_id
        return {"accepted": True, "workspace_id": row.workspace_id}


@router.get("/members/{workspace_id}")
def members(workspace_id: str, user: User):
    with session_scope() as db:
        permission = access.role(db, workspace_id, user.owner_id)
        rows = [
            {
                "account_id": r.account_id,
                "name": preferences(db, r.account_id)["preferred_name"],
                "role": r.role,
                "active": r.active,
                "revision": r.revision,
                "actor_id": r.actor_id,
            }
            for r in db.scalars(select(WorkspaceMember).where(WorkspaceMember.workspace_id == workspace_id))
        ]
        invites = (
            []
            if permission != "owner"
            else [
                invite_public(db, r)
                for r in db.scalars(
                    select(WorkspaceInvite).where(
                        WorkspaceInvite.workspace_id == workspace_id,
                        WorkspaceInvite.status == "pending",
                        WorkspaceInvite.expires_at > now(),
                    )
                )
            ]
        )
        return {"members": rows, "invitations": invites, "role": permission}


@router.post("/members")
def change_member(body: MemberChange, user: User):
    with session_scope() as db:
        admin(db, body.workspace_id, user.owner_id)
        row = db.get(WorkspaceMember, (body.workspace_id, body.account_id), with_for_update=True)
        if not row or row.role == "owner":
            raise DomainError("INVALID_ARGUMENT", "The workspace owner cannot be removed or demoted here.")
        if row.revision != body.expected_revision:
            raise DomainError("REVISION_CONFLICT", "Membership changed. Reload before trying again.", 409)
        row.role, row.active = body.role, body.active
        row.revision += 1
        if row.actor_id:
            db.get(Actor, row.actor_id).archived = not body.active
        # Personal push destinations are never added to a shared namespace.
        emit(db, body.workspace_id, "membership.changed", body.workspace_id)
        return {"updated": True, "revision": row.revision}


@router.post("/revoke-invitation")
def revoke_invitation(body: InviteAction, user: User):
    with session_scope() as db:
        row = db.get(WorkspaceInvite, body.invite_id, with_for_update=True)
        if not row:
            raise DomainError("NOT_FOUND", "Invitation not found.", 404)
        if row.workspace_id:
            admin(db, row.workspace_id, user.owner_id)
        elif row.inviter_id != user.owner_id:
            raise DomainError("NOT_AUTHORIZED", "Invitation is not yours.", 403)
        row.status = "revoked"
        return {"revoked": True}
