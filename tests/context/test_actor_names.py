"""Actor display names may repeat; name lookups resolve only a single active match."""

from uuid import uuid4

from fastapi.testclient import TestClient
from jarvis.api import app
from jarvis.auth import new_session
from jarvis.db import session_scope
from jarvis.models import Actor, GoogleIdentity, WorkspaceMember
from sqlalchemy import select


def send(client, tool, **args):
    return client.post("/api/v1/commands", json={"command_id": str(uuid4()), "tool": tool, "arguments": args})


def cmd(client, tool, **args):
    r = send(client, tool, **args)
    assert r.status_code == 200, r.text
    return r.json()["data"]


def error(client, tool, **args):
    r = send(client, tool, **args)
    assert r.status_code >= 400, r.text
    return r.status_code, r.json()["error"]


def test_duplicate_name_needs_allow_duplicate_and_then_coexists(client):
    first = cmd(client, "actor.create", name="Alex")
    status, err = error(client, "actor.create", name="Alex")
    assert (status, err["code"]) == (409, "DUPLICATE_ACTOR_NAME")
    second = cmd(client, "actor.create", name="Alex", allow_duplicate=True)
    assert second["id"] != first["id"]
    with session_scope() as db:
        owner = db.get(Actor, first["id"]).owner_id
        rows = list(db.scalars(select(Actor).where(Actor.owner_id == owner, Actor.name == "Alex")))
    assert {r.id for r in rows} == {first["id"], second["id"]}
    # Case is significant, as the former owner-plus-name constraint was.
    cmd(client, "actor.create", name="alex")


def test_rename_and_restore_follow_the_same_rule(client):
    alex = cmd(client, "actor.create", name="Alex")
    sam = cmd(client, "actor.create", name="Sam")
    status, err = error(client, "actor.update", actor_id=sam["id"], expected_revision=1, name="Alex")
    assert (status, err["code"]) == (409, "DUPLICATE_ACTOR_NAME")
    renamed = cmd(client, "actor.update", actor_id=sam["id"], expected_revision=1, name="Alex", allow_duplicate=True)
    assert renamed["name"] == "Alex"
    # Unrelated edits to an already-duplicated actor do not re-check the name.
    archived = cmd(client, "actor.update", actor_id=renamed["id"], expected_revision=2, archived=True)
    assert archived["archived"] is True
    status, err = error(client, "actor.update", actor_id=sam["id"], expected_revision=3, archived=False)
    assert err["code"] == "DUPLICATE_ACTOR_NAME"
    cmd(client, "actor.update", actor_id=alex["id"], expected_revision=1, archived=True)
    assert cmd(client, "actor.update", actor_id=sam["id"], expected_revision=3, archived=False)["archived"] is False


def test_unique_name_assignment_resolves_and_unknown_name_creates(client):
    sam = cmd(client, "actor.create", name="Sam")
    task = cmd(client, "task.create", title="Unique", assignee="Sam")
    assert task["assignee_id"] == sam["id"]
    fresh = cmd(client, "task.create", title="New person", assignee="Robin")
    assert fresh["assignee_id"] and fresh["assignee_id"] != sam["id"]
    again = cmd(client, "task.create", title="Same new person", assignee="Robin")
    assert again["assignee_id"] == fresh["assignee_id"]
    default = cmd(client, "task.create", title="Default owner")
    assert default["assignee"] == "owner" and default["assignee_id"]


def test_ambiguous_name_fails_loudly_and_ids_still_work(client):
    first = cmd(client, "actor.create", name="Alex")
    second = cmd(client, "actor.create", name="Alex", allow_duplicate=True)
    status, err = error(client, "task.create", title="Which Alex?", assignee="Alex")
    assert (status, err["code"]) == (409, "AMBIGUOUS_ACTOR")
    assert {c["id"] for c in err["data"]["candidates"]} == {first["id"], second["id"]}
    task = cmd(client, "task.create", title="This Alex", assignee_id=second["id"])
    assert task["assignee_id"] == second["id"] and task["assignee"] == "Alex"
    # Re-sending the unchanged name keeps the task's actor instead of becoming ambiguous.
    kept = cmd(client, "task.update", task_id=task["id"], expected_revision=task["revision"], assignee="Alex")
    assert kept["assignee_id"] == second["id"]


def test_archived_actors_do_not_count(client):
    old = cmd(client, "actor.create", name="Alex")
    cmd(client, "actor.update", actor_id=old["id"], expected_revision=1, archived=True)
    # Only an archived match: keep refusing rather than silently recreating the person.
    status, err = error(client, "task.create", title="Archived only", assignee="Alex")
    assert (status, err["code"]) == (400, "INVALID_ARGUMENT")
    current = cmd(client, "actor.create", name="Alex")
    task = cmd(client, "task.create", title="Active Alex", assignee="Alex")
    assert task["assignee_id"] == current["id"]


def test_workspace_invite_acceptance_produces_a_usable_actor(client):
    token, csrf = new_session("guest", "google")
    with session_scope() as db:
        db.add(GoogleIdentity(owner_id="guest", subject="subject-guest", email="guest@example.test"))
    guest = TestClient(app)
    guest.cookies.set("jarvis_session", token)
    guest.headers["X-CSRF-Token"] = csrf

    def post(c, path, body):
        r = c.post("/api/v1" + path, json=body)
        assert r.status_code == 200, r.text
        return r.json()

    w = post(client, "/accounts/workspaces", {"name": "Names team", "kind": "space"})
    invite = post(client, "/accounts/invitations", {"workspace_id": w["id"], "email": "guest@example.test", "role": "editor"})
    post(guest, "/accounts/accept", {"invite_id": invite["id"]})
    for c in (client, guest):
        post(c, "/accounts/switch", {"workspace_id": w["id"]})
    with session_scope() as db:
        member = db.get(WorkspaceMember, (w["id"], "guest"))
        actor = db.get(Actor, member.actor_id)
        names = [a.name for a in db.scalars(select(Actor).where(Actor.owner_id == w["id"], Actor.archived.is_(False)))]
        actor_id, actor_name = actor.id, actor.name
    # Both accounts share the default preferred name, so the member's actor is distinguishable.
    assert names.count(actor_name) == 1
    assert cmd(guest, "task.create", title="Default to me")["assignee_id"] == actor_id
    assert cmd(client, "task.create", title="By name", assignee=actor_name)["assignee_id"] == actor_id
