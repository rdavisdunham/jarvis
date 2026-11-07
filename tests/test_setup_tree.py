"""Set up with Dani: preview and apply a proposed organization tree (docs/SETUP_WITH_DANI_PLAN.md, PR 5)."""
from uuid import uuid4

import pytest
from cryptography.fernet import Fernet
from sqlalchemy import select

from jarvis.config import get_settings
from jarvis.db import session_scope
from jarvis.models import ActionChange, Area, Goal, GoalProjectLink, Project, SharedWorkspace, Space


@pytest.fixture(autouse=True)
def journal_key(monkeypatch):
    monkeypatch.setenv("JARVIS_INTEGRATION_ENCRYPTION_KEY", Fernet.generate_key().decode())
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def command(client, tool, arguments, status=200):
    response = client.post("/api/v1/commands", json={"command_id": str(uuid4()), "tool": tool, "arguments": arguments})
    assert response.status_code == status, response.text
    return response.json()


def tree():
    return [{"id": "s1", "kind": "space", "title": "Work", "description": "Teaching and coaching.", "children": [
        {"id": "a1", "kind": "area", "title": "Teaching", "children": [
            {"id": "g1", "kind": "goal", "title": "Finish the chemistry unit", "children": [
                {"id": "p1", "kind": "project", "title": "Lab safety module"}]},
            {"id": "p2", "kind": "project", "title": "Parent conferences"}]},
        {"id": "a2", "kind": "area", "title": "Coaching"}]}]


def preview(client, nodes=None):
    return command(client, "setup.tree_preview", {"tree": nodes or tree()})["data"]


def apply(client, proposal, status=200):
    return command(client, "setup.tree_apply", {"proposal_id": proposal["id"], "preview_hash": proposal["preview_hash"]}, status)


def test_preview_changes_nothing_and_apply_creates_the_tree(client):
    client.post("/api/v1/setup/start", json={})
    proposal = preview(client)
    assert proposal["impact"]["create"] == {"space": 1, "area": 2, "goal": 1, "project": 2}
    assert client.get("/api/v1/setup").json()["status"] == "reviewing"
    assert client.get("/api/v1/setup").json()["tree_proposal_id"] == proposal["id"]
    with session_scope() as db:
        assert not db.scalars(select(Space)).all()
    result = apply(client, proposal)["data"]
    assert result["created_count"] == 6
    with session_scope() as db:
        space = db.scalar(select(Space).where(Space.name == "Work"))
        teaching = db.scalar(select(Area).where(Area.name == "Teaching"))
        goal = db.scalar(select(Goal).where(Goal.name == "Finish the chemistry unit"))
        lab = db.scalar(select(Project).where(Project.name == "Lab safety module"))
        assert teaching.space_id == space.id and goal.area_id == teaching.id
        assert lab.area_id == teaching.id and lab.space_id == space.id
        assert db.get(GoalProjectLink, (goal.id, lab.id))
    # The applied tree completes onboarding through the alternative gate.
    done = command(client, "onboarding.save", {"expected_revision": 0, "status": "completed"})
    assert done["data"]["status"] == "completed"
    apply(client, proposal, 409)


def test_hash_mismatch_and_stale_records_need_a_new_preview(client):
    command(client, "space.create", {"name": "Work"})
    proposal = preview(client)
    # The existing space is matched by name and kept, not duplicated.
    assert proposal["tree"][0]["action"] == "keep" and proposal["tree"][0]["existing_id"]
    assert proposal["impact"]["create"]["space"] == 0
    wrong = command(client, "setup.tree_apply", {"proposal_id": proposal["id"], "preview_hash": "0" * 64}, 409)
    assert wrong["error"]["code"] == "PREVIEW_REQUIRED"
    with session_scope() as db:
        space = db.scalar(select(Space).where(Space.name == "Work"))
        space_id, revision = space.id, space.revision
    command(client, "space.update", {"space_id": space_id, "expected_revision": revision, "description": "Changed"})
    stale = apply(client, proposal, 409)
    assert stale["error"]["code"] == "PREVIEW_REQUIRED"
    fresh = preview(client)
    apply(client, fresh)


def test_rename_and_archive_only_when_ticked(client):
    space = command(client, "space.create", {"name": "Job"})["data"]
    old = command(client, "area.create", {"name": "Old club", "space_id": space["id"]})["data"]
    nodes = [{"id": "s1", "kind": "space", "title": "Work", "existing_id": space["id"], "action": "rename", "children": [
        {"id": "a1", "kind": "area", "title": "Old club", "existing_id": old["id"], "action": "archive"}]}]
    unticked = preview(client, nodes)
    assert unticked["impact"]["rename"] == 1 and unticked["impact"]["archive"] == 0
    apply(client, unticked)
    with session_scope() as db:
        assert db.get(Space, space["id"]).name == "Work" and not db.get(Area, old["id"]).archived
    nodes[0]["action"] = "keep"
    nodes[0]["children"][0]["archive_confirmed"] = True
    ticked = preview(client, nodes)
    assert ticked["impact"]["archive"] == 1
    apply(client, ticked)
    with session_scope() as db:
        assert db.get(Area, old["id"]).archived


def test_invalid_trees_are_refused(client):
    bad = [{"id": "a1", "kind": "area", "title": "Loose area"}]
    assert command(client, "setup.tree_preview", {"tree": bad}, 400)["error"]["code"] == "INVALID_ARGUMENT"
    foreign = [{"id": "s1", "kind": "space", "title": "X", "existing_id": str(uuid4()), "action": "keep"}]
    assert command(client, "setup.tree_preview", {"tree": foreign}, 404)["error"]["code"] == "NOT_FOUND"
    command(client, "project.create", {"name": "Taken"})
    clash = [{"id": "s1", "kind": "space", "title": "S", "children": [
        {"id": "p1", "kind": "project", "title": "Fresh"}, {"id": "p2", "kind": "project", "title": "fresh"}]}]
    proposal = preview(client, clash)
    assert proposal["impact"]["blocking_count"] >= 1
    assert apply(client, proposal, 400)["error"]["code"] == "INVALID_ARGUMENT"
    too_many = [{"id": f"s{i}", "kind": "space", "title": f"Space {i}"} for i in range(81)]
    assert command(client, "setup.tree_preview", {"tree": too_many}, 400)


def test_revert_undoes_the_whole_tree(client):
    proposal = preview(client)
    applied = apply(client, proposal)
    with session_scope() as db:
        change = db.scalars(select(ActionChange).where(ActionChange.command_id == applied["command_id"])).first()
        change_id = change.id
    reverted = client.post("/api/v1/work/actions/" + change_id + "/revert", json={"command_id": str(uuid4())})
    assert reverted.status_code == 200, reverted.text
    with session_scope() as db:
        assert all(row.archived for model in (Space, Area, Goal, Project) for row in db.scalars(select(model)))
        assert all(c.reverted_by for c in db.scalars(select(ActionChange).where(ActionChange.command_id == applied["command_id"])))


def test_revert_is_refused_once_other_work_uses_the_tree(client):
    proposal = preview(client)
    applied = apply(client, proposal)
    with session_scope() as db:
        project = db.scalar(select(Project).where(Project.name == "Parent conferences"))
        project_id = project.id
        change_id = db.scalars(select(ActionChange).where(ActionChange.command_id == applied["command_id"])).first().id
    command(client, "task.create", {"title": "Email parents", "project_id": project_id})
    refused = client.post("/api/v1/work/actions/" + change_id + "/revert", json={"command_id": str(uuid4())})
    assert refused.status_code == 409


def test_tree_commands_are_personal_only(client):
    from jarvis import setup_tree
    from jarvis.access import PERSONAL_TOOLS
    from jarvis.domain import DomainError
    assert "setup.tree_apply".startswith(PERSONAL_TOOLS)
    with session_scope() as db:
        db.add(SharedWorkspace(id="davin-shared", name="Shared", kind="project", creator_id="davin"))
    with session_scope() as db:
        with pytest.raises(DomainError) as exc:
            setup_tree.preview(db, "davin-shared", setup_tree.TreePreview(tree=tree()).tree, "c1")
        assert exc.value.code == "PERSONAL_WORKSPACE"
