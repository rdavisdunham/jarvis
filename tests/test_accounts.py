from datetime import timedelta
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from jarvis.api import app
from jarvis.auth import new_session
from jarvis.config import get_settings
from jarvis.db import session_scope
from jarvis.domain import DomainError, execute, preferences
from jarvis.models import (
    GoogleIdentity,
    Memory,
    SharedWorkspace,
    WorkspaceInvite,
    WorkspaceMember,
    now,
)
from jarvis.tools import call_tool
from sqlalchemy import select


def client_for(account, email=None):
    token, csrf = new_session(account, "google")
    with session_scope() as db:
        if not db.get(GoogleIdentity, account):
            db.add(
                GoogleIdentity(
                    owner_id=account, subject="subject-" + account, email=email or account + "@example.test"
                )
            )
    c = TestClient(app)
    c.cookies.set("jarvis_session", token)
    c.headers["X-CSRF-Token"] = csrf
    return c


def post(c, path, body, status=200):
    r = c.post("/api/v1" + path, json=body)
    assert r.status_code == status, r.text
    return r.json()


def command(c, tool, **args):
    return post(c, "/commands", {"command_id": str(uuid4()), "tool": tool, "arguments": args})["data"]


def shared(owner, guest, role="editor"):
    w = post(owner, "/accounts/workspaces", {"name": "Team launch", "kind": "project"})
    invite = post(
        owner, "/accounts/invitations", {"workspace_id": w["id"], "email": "guest@example.test", "role": role}
    )
    post(guest, "/accounts/accept", {"invite_id": invite["id"]})
    for c in (owner, guest):
        post(c, "/accounts/switch", {"workspace_id": w["id"]})
    return w


def test_private_accounts_do_not_share_records_or_profile(client):
    other = client_for("guest")
    private = command(client, "task.create", title="Private owner work")
    note = command(client, "note.create", title="Private owner note", content="Keep private")
    assert other.get("/api/v1/tasks").json()["items"] == []
    assert other.get("/api/v1/tasks/" + private["id"]).status_code == 404
    assert other.get("/api/v1/notes/" + note["id"]).status_code == 404
    assert other.get("/api/v1/notes?q=private").json()["items"] == []
    command(other, "settings.update", preferred_name="Guest")
    assert other.get("/api/v1/bootstrap").json()["name"] == "Guest"
    assert client.get("/api/v1/bootstrap").json()["name"] != "Guest"
    with session_scope() as db:
        assert preferences(db, "guest")["preferred_name"] == "Guest"


@pytest.mark.asyncio
async def test_shared_editor_viewer_revocation_api_tools_receipts_and_notes(client):
    guest = client_for("guest")
    private = command(client, "task.create", title="Secret personal")
    w = shared(client, guest)
    task = command(client, "task.create", title="Shared launch", notes="Team notes", due_date="2030-01-12")
    note = command(guest, "note.create", title="Shared notes", content="Searchable team evidence")
    assert guest.get("/api/v1/tasks/" + task["id"]).status_code == 200
    assert guest.get("/api/v1/tasks/" + private["id"]).status_code == 404
    assert client.get("/api/v1/notes/" + note["id"]).status_code == 200
    assert guest.get("/api/v1/notes?q=evidence").json()["items"][0]["id"] == note["id"]
    cal = guest.get("/api/v1/calendar?start=2030-01-01&end=2030-02-01")
    assert task["id"] in str(cal.json()) and private["id"] not in str(cal.json())
    for path in ("/memory", "/integrations/google", "/integrations/linear", "/export"):
        assert guest.get("/api/v1" + path).status_code == 403
    convo = post(guest, "/conversations", {})
    assert convo["private"] and not convo["learning"]
    # Account-private saved views remain distinct even in the same shared workspace.
    post(
        guest, "/task-views", {"id": str(uuid4()), "name": "Guest view", "state": {}, "expected_revision": 0}
    )
    assert client.get("/api/v1/task-views").json()["items"] == []
    boot = guest.get("/api/v1/bootstrap").json()
    result = await call_tool(
        w["id"], str(uuid4()), 0, "task_get", {"task_id": task["id"]}, device=boot["device_id"]
    )
    assert task["title"] in str(result)
    member = next(
        m
        for m in client.get("/api/v1/accounts/members/" + w["id"]).json()["members"]
        if m["account_id"] == "guest"
    )
    post(
        client,
        "/accounts/members",
        {
            "workspace_id": w["id"],
            "account_id": "guest",
            "role": "viewer",
            "expected_revision": member["revision"],
            "active": True,
        },
    )
    post(
        guest,
        "/commands",
        {
            "command_id": str(uuid4()),
            "tool": "task.update",
            "arguments": {"task_id": task["id"], "expected_revision": 1, "title": "Forbidden"},
        },
        403,
    )
    with pytest.raises(DomainError, match="view access"):
        await call_tool(
            w["id"],
            str(uuid4()),
            0,
            "task_update",
            {"task_id": task["id"], "expected_revision": 1, "title": "Forbidden"},
            device=boot["device_id"],
        )
    post(
        client,
        "/accounts/members",
        {
            "workspace_id": w["id"],
            "account_id": "guest",
            "role": "viewer",
            "expected_revision": member["revision"] + 1,
            "active": False,
        },
    )
    assert guest.get("/api/v1/tasks").status_code == 403
    assert guest.get("/api/v1/notes/" + note["id"]).status_code == 403
    with pytest.raises(DomainError, match="access ended"):
        await call_tool(
            w["id"], str(uuid4()), 0, "task_get", {"task_id": task["id"]}, device=boot["device_id"]
        )
    # Revoked users can still use their own private account, but cannot switch back.
    post(guest, "/accounts/switch", {"workspace_id": None})
    assert guest.get("/api/v1/tasks").json()["items"] == []
    post(guest, "/accounts/switch", {"workspace_id": w["id"]}, 403)


def test_invites_bound_to_verified_email_expire_revoke_and_do_not_grant_by_assignment(client):
    guest = client_for("guest")
    wrong = client_for("wrong")
    w = post(client, "/accounts/workspaces", {"name": "Shared space", "kind": "space"})
    invite = post(
        client,
        "/accounts/invitations",
        {"workspace_id": w["id"], "email": "GUEST@example.test", "role": "viewer"},
    )
    post(wrong, "/accounts/accept", {"invite_id": invite["id"]}, 403)
    with session_scope() as db:
        db.get(WorkspaceInvite, invite["id"]).expires_at = now() - timedelta(seconds=1)
    post(guest, "/accounts/accept", {"invite_id": invite["id"]}, 409)
    invite = post(
        client,
        "/accounts/invitations",
        {"workspace_id": w["id"], "email": "guest@example.test", "role": "viewer"},
    )
    post(client, "/accounts/revoke-invitation", {"invite_id": invite["id"]})
    post(guest, "/accounts/accept", {"invite_id": invite["id"]}, 409)
    post(client, "/accounts/switch", {"workspace_id": w["id"]})
    command(client, "actor.create", name="guest", kind="person")
    command(client, "task.create", title="Assigned but not shared", assignee="guest")
    post(guest, "/accounts/switch", {"workspace_id": w["id"]}, 403)


def test_old_device_requests_rejected_after_workspace_switch(client):
    before = client.get("/api/v1/bootstrap").json()
    w = post(client, "/accounts/workspaces", {"name": "Separate context", "kind": "space"})
    post(client, "/accounts/switch", {"workspace_id": w["id"]})
    r = client.get("/api/v1/tasks", headers={"X-Device-Id": before["device_id"]})
    assert r.status_code == 409 and r.json()["error"]["code"] == "WORKSPACE_CHANGED"


@pytest.mark.asyncio
async def test_revocation_during_tool_retrieval_discards_result(client, monkeypatch):
    from jarvis import tools

    guest = client_for("guest")
    w = shared(client, guest)
    boot = guest.get("/api/v1/bootstrap").json()

    async def delayed(*args, **kwargs):
        with session_scope() as db:
            db.get(WorkspaceMember, (w["id"], "guest")).active = False
        return {"confidential": "Never returned after revocation"}

    monkeypatch.setattr(tools, "_call_tool", delayed)
    with pytest.raises(DomainError, match="access ended"):
        await tools.call_tool(w["id"], "turn", 0, "note_search", {}, device=boot["device_id"])


def test_shared_command_service_requires_membership_even_without_http(client):
    w = post(client, "/accounts/workspaces", {"name": "Boundary", "kind": "space"})
    with session_scope() as db, pytest.raises(DomainError):
        execute(db, w["id"], str(uuid4()), "task.create", {"title": "No actor context"})


def test_notifications_and_memory_stay_in_their_account(client):
    from jarvis.models import Notification

    guest = client_for("guest")
    owner = get_settings().owner_id
    with session_scope() as db:
        db.add(Notification(owner_id=owner, title="Private alert", body="Only owner", scheduled_at=now()))
        execute(db, owner, str(uuid4()), "memory.capture", {"content": "Personal secret"})
    assert guest.get("/api/v1/notifications").json()["items"] == []
    with session_scope() as db:
        assert list(db.scalars(select(Memory).where(Memory.owner_id == "guest"))) == []


def test_shared_search_indexes_do_not_include_private_vectors_and_recheck_revocation(client,monkeypatch):
    from jarvis.models import Job, NoteEmbedding
    from jarvis.notes import index_note, search_notes
    guest=client_for("guest")
    private=command(client,"note.create",title="Private vectors",content="Personal camping plan")
    monkeypatch.setattr("jarvis.notes.embeddings",lambda *a,**k:[[1.0,0.0]])
    with session_scope() as db:
        job=db.scalar(select(Job.id).where(Job.kind=="embed_note"))
    index_note(job)
    w=shared(client,guest)
    shared_note=command(client,"note.create",title="Team vectors",content="Team camping plan")
    with session_scope() as db:
        job=db.scalar(select(Job.id).where(Job.owner_id==w["id"],Job.kind=="embed_note"))
    index_note(job)
    result=search_notes(w["id"],"trip")
    assert [n["id"] for n in result["items"]]==[shared_note["id"]]
    assert private["id"] not in str(result)
    with session_scope() as db:
        assert len(list(db.scalars(select(NoteEmbedding))))==2
    def revoked(*a,**k):
        with session_scope() as db:db.get(WorkspaceMember,(w["id"],"guest")).active=False
        return [[1.0,0.0]]
    monkeypatch.setattr("jarvis.notes.embeddings",revoked)
    response=guest.get("/api/v1/notes/search?q=trip")
    assert response.status_code==403 and "Team vectors" not in response.text


@pytest.mark.asyncio
async def test_live_watch_closes_after_membership_revocation(client,monkeypatch):
    from jarvis.live_voice import LiveController
    guest=client_for("guest")
    w=shared(client,guest)
    boot=guest.get("/api/v1/bootstrap").json()
    conv=post(guest,"/conversations",{})
    voice=LiveController(w["id"],boot["device_id"],conv["id"],None,boot["preferences"])
    stopped=[]
    async def close(*a,**k):
        stopped.append(True)
        voice.closed=True
    monkeypatch.setattr(voice,"close",close)
    with session_scope() as db:db.get(WorkspaceMember,(w["id"],"guest")).active=False
    await voice.watch()
    assert stopped and "access ended" in voice.error


def test_member_cannot_manage_roles_or_bypass_csrf_or_claim_private_namespace(client):
    guest=client_for("guest")
    w=shared(client,guest)
    post(guest,"/accounts/invitations",{"workspace_id":w["id"],"email":"third@example.test","role":"editor"},403)
    post(guest,"/accounts/members",{"workspace_id":w["id"],"account_id":"guest","role":"editor","expected_revision":1},403)
    guest.headers.pop("X-CSRF-Token")
    post(guest,"/accounts/switch",{"workspace_id":None},403)
    # Client retry IDs are transformed into workspace IDs in the caller's namespace.
    personal_id=str(uuid4())
    client_for(personal_id)
    result=post(client,"/accounts/workspaces",{"id":personal_id,"name":"Cannot claim another account","kind":"space"})
    assert result["id"]!=personal_id
    assert post(client,"/accounts/workspaces",{"id":personal_id,"name":"Cannot claim another account","kind":"space"})["id"]==result["id"]


def test_shared_default_assignee_is_requesting_member_and_project_is_root(client):
    guest=client_for("guest")
    w=shared(client,guest)
    task=command(guest,"task.create",title="Guest-owned assignment")
    with session_scope() as db:
        member=db.get(WorkspaceMember,(w["id"],"guest"))
        workspace=db.get(SharedWorkspace,w["id"])
        assert task["assignee_id"]==member.actor_id
        assert task["project_id"]==workspace.root_id


def test_housekeeping_applies_each_persons_history_policy(client):
    from jarvis.domain import capture_source
    from jarvis.models import Source
    from jarvis.worker import housekeeping
    other=client_for("guest")
    command(other,"settings.update",history_days=1)
    with session_scope() as db:
        row=capture_source(db,"guest","Old history","test-old")
        row.created_at=now()-timedelta(days=2)
        source_id=row.id
    housekeeping()
    with session_scope() as db:assert db.get(Source,source_id).deleted_at is not None

def test_shared_commands_preserve_requesting_account_attribution(client):
    from jarvis.models import Command
    guest=client_for("guest")
    w=shared(client,guest)
    command(guest,"task.create",title="Attributed change")
    with session_scope() as db:
        saved=db.scalar(select(Command).where(Command.owner_id==w["id"]))
        assert saved.account_id=="guest"


@pytest.mark.asyncio
async def test_private_interactive_tools_stop_after_session_revocation(client):
    from jarvis.access import principal
    from jarvis.auth import digest
    from jarvis.models import AuthSession
    guest=client_for("guest")
    boot=guest.get("/api/v1/bootstrap").json()
    session_hash=digest(guest.cookies.get("jarvis_session"))
    token=principal.set(session_hash)
    try:
        with session_scope() as db:db.delete(db.get(AuthSession,session_hash))
        with pytest.raises(DomainError,match="Sign in"):
            await call_tool("guest",str(uuid4()),0,"task_create",{"title":"Must not save"},device=boot["device_id"])
    finally:principal.reset(token)
