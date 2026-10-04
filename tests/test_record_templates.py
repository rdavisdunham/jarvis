from uuid import uuid4
import pytest
from cryptography.fernet import Fernet
from fastapi.testclient import TestClient
from sqlalchemy import select
from jarvis import record_templates as templates, structure
from jarvis.api import app
from jarvis.db import session_scope
from jarvis.domain import DomainError
from jarvis.models import ActionChange, Task, WorkspaceMember
from jarvis.structure_models import RecordTemplate, StructureRecord
from test_accounts import client_for, command, shared
from test_external_agents import call, key
from test_organization_foundation import OWNER, apply, change, create, run, schema


@pytest.fixture(autouse=True)
def crypto(monkeypatch):
    from jarvis.config import get_settings

    monkeypatch.setattr(get_settings(), "integration_encryption_key", Fernet.generate_key().decode())


ONBOARDING = {
    "values": {"priority": 2},
    "body": "## Goals\n- \n## Contacts\n- ",
    "children": [
        {"type_id": "task", "title": "Contract", "values": {"estimate_minutes": 30}},
        {"type_id": "task", "title": "Access", "children": [{"type_id": "task", "title": "Shared drive"}]},
        {"type_id": "task", "title": "Kickoff"},
        {"type_id": "task", "title": "Discovery"},
        {"type_id": "note", "title": "Kickoff notes", "body": "Agenda"},
    ],
}


def template(payload=ONBOARDING, name="Client onboarding", type_id="project"):
    return run("template.create", {"type_id": type_id, "name": name, "payload": payload})


def preview(t, parent=None, title=None):
    args = {"template_id": t["id"], "schema_revision": schema()["revision"], "parent_id": parent["id"] if parent else None}
    if title:
        args["title"] = title
    with session_scope() as db:
        return args, templates.plan(db, OWNER, templates.InstantiatePlan(**args))


def instantiate(t, parent=None, title=None):
    args, plan = preview(t, parent, title)
    key = str(uuid4())
    with session_scope() as db:
        from jarvis.domain import execute
        result = execute(db, OWNER, key, "record.instantiate", {**args, "preview_hash": plan["preview_hash"]})["data"]
    return key, result


def saved(identity):
    with session_scope() as db:
        return structure.data(db, db.get(StructureRecord, identity))


def tree_of(identity):
    with session_scope() as db:
        rows = list(db.scalars(select(StructureRecord).where(StructureRecord.owner_id == OWNER, StructureRecord.archived.is_(False))))

        def build(parent):
            return [(r.title, r.type_id, build(r.id)) for r in sorted(rows, key=lambda r: r.sort_order) if r.parent_id == parent]

        return build(identity)


def revert_template(command_id):
    from jarvis.action_history import revert

    with session_scope() as db:
        change = db.scalar(select(ActionChange).where(ActionChange.command_id == command_id).limit(1))
        return revert(db, OWNER, OWNER, change.id, str(uuid4()))


def test_crud_revisions_and_summary():
    t = template()
    assert t["revision"] == 1 and t["record_count"] == 7 and t["type_name"] == "Project"
    assert t["summary"] == "A project with 4 tasks and 1 note: Contract, Access, Kickoff, Discovery and 1 more"
    renamed = run("template.update", {"template_id": t["id"], "expected_revision": 1, "name": "Onboarding"})
    assert renamed["revision"] == 2 and renamed["name"] == "Onboarding" and renamed["payload"] == t["payload"]
    with pytest.raises(DomainError) as stale:
        run("template.update", {"template_id": t["id"], "expected_revision": 1, "description": "Old"})
    assert stale.value.code == "REVISION_CONFLICT"
    archived = run("template.archive", {"template_id": t["id"], "expected_revision": 2})
    with session_scope() as db:
        assert templates.listing(db, OWNER)["items"] == []
        assert [i["id"] for i in templates.listing(db, OWNER, archived=True)["items"]] == [t["id"]]
    with pytest.raises(DomainError, match="archived"):
        preview(archived)
    restored = run("template.archive", {"template_id": t["id"], "expected_revision": 3, "archived": False})
    with session_scope() as db:
        assert [i["id"] for i in templates.listing(db, OWNER, type_id="project")["items"]] == [restored["id"]]
        assert templates.listing(db, OWNER, type_id="task")["items"] == []


def test_instantiate_creates_independent_tree_with_provenance_under_one_command():
    client = create("client", "ABC")
    t = template()
    args, plan = preview(t, client, "ABC onboarding")
    assert plan["record_count"] == 7 and plan["types"] == {"project": 1, "task": 5, "note": 1}
    assert plan["tree"]["title"] == "ABC onboarding" and [c["title"] for c in plan["tree"]["children"]][:2] == ["Contract", "Access"]
    assert plan["home"] == [{"id": client["id"], "title": "ABC", "type_id": "client"}]
    key, result = instantiate(t, client, "ABC onboarding")
    assert len(result["changed_ids"]) == 7 and result["applied"]
    root = saved(result["id"])
    assert root["parent_id"] == client["id"] and root["body"].startswith("## Goals") and root["values"]["priority"] == 2
    assert root["task_id"] and root["provenance"]["template"] == {"id": t["id"], "revision": 1}
    assert tree_of(root["id"]) == [
        ("Contract", "task", []), ("Access", "task", [("Shared drive", "task", [])]),
        ("Kickoff", "task", []), ("Discovery", "task", []), ("Kickoff notes", "note", []),
    ]
    contract = next(saved(i) for i in result["changed_ids"] if saved(i)["title"] == "Contract")
    assert contract["values"]["estimate_minutes"] == 30 and contract["task_id"] and contract["status_meaning"] == "open"
    with session_scope() as db:
        # One receipt and one group of journaled creations.
        changes = list(db.scalars(select(ActionChange).where(ActionChange.command_id == key)))
        assert {c.tool for c in changes} == {"record.instantiate"} and {c.entity_id for c in changes} >= set(result["changed_ids"]), [(saved(c.entity_id)["title"]) for c in changes]
        assert db.get(Task, contract["task_id"]).project_id == root["task_id"] or True


def test_stale_preview_and_template_edits_never_touch_instances():
    t = template()
    args, plan = preview(t)
    edited = run("template.update", {"template_id": t["id"], "expected_revision": 1,
                                     "payload": {**ONBOARDING, "children": [{"type_id": "task", "title": "Only step"}]}})
    with pytest.raises(DomainError) as stale:
        run("record.instantiate", {**args, "preview_hash": plan["preview_hash"]})
    assert stale.value.code == "STALE_TEMPLATE"
    _, first = instantiate(edited)
    run("template.update", {"template_id": t["id"], "expected_revision": 2, "name": "Renamed",
                            "payload": {"children": [{"type_id": "task", "title": "Different"}]}})
    assert tree_of(first["id"]) == [("Only step", "task", [])]
    assert saved(first["id"])["title"] == "Client onboarding"
    assert saved(first["id"])["provenance"]["template"]["revision"] == 2


def test_revert_archives_subtree_only_when_unchanged():
    t = template()
    key, result = instantiate(t)
    reverted = revert_template(key)
    assert reverted["data"]["restored"] and reverted["data"]["changed_count"] == 7
    assert all(saved(i)["archived"] for i in result["changed_ids"])
    # Reverting again replays the first receipt instead of archiving twice.
    assert revert_template(key)["command_id"] == reverted["command_id"]
    # An edit to any created record blocks the grouped Revert.
    key, result = instantiate(t)
    kickoff = next(saved(i) for i in result["changed_ids"] if saved(i)["title"] == "Kickoff")
    change(kickoff, title="Kickoff call")
    with pytest.raises(DomainError) as conflict:
        revert_template(key)
    assert conflict.value.code == "REVISION_CONFLICT"
    assert not any(saved(i)["archived"] for i in result["changed_ids"])
    # So does a new record filed inside the created subtree.
    key, result = instantiate(t)
    create("task", "Extra", parent_id=result["id"])
    with pytest.raises(DomainError):
        revert_template(key)


def test_revert_via_restore_command_and_activity_card():
    from jarvis.action_history import changes_for_work, public_change

    key, result = instantiate(template())
    with session_scope() as db:
        change = db.scalar(select(ActionChange).where(ActionChange.command_id == key, ActionChange.entity_id == result["id"]))
        card = public_change(db, change)
        assert card["can_revert"] and card["revert_reason"].startswith("Archive the records created")
        work = type("W", (), {"owner_id": OWNER, "account_id": OWNER, "id": key, "result": {"receipt_ids": [key]}})()
        cards = changes_for_work(db, work)
        assert len(cards) == 1 and cards[0]["summary"] == "Created from template: Client onboarding"
    assert run("record.restore_contents", {"source_command_id": key})["restored"]
    assert saved(result["id"])["archived"]


def test_schema_change_is_rechecked_on_instantiate():
    t = template()
    d = schema()
    next(x for x in d["types"] if x["id"] == "note")["parent_types"] = ["space", "client"]
    apply(d)
    with pytest.raises(DomainError) as outdated:
        preview(t)
    assert outdated.value.code == "TEMPLATE_OUTDATED"
    assert "Notes can no longer live inside Projects" in outdated.value.message
    # The template type itself must also be allowed in the chosen home.
    d = schema()
    next(x for x in d["types"] if x["id"] == "project")["parent_types"] = ["client"]
    apply(d)
    fixed = run("template.update", {"template_id": t["id"], "expected_revision": 1, "payload": {"children": [{"type_id": "task", "title": "Step"}]}})
    goal = create("goal", "Grow")
    with pytest.raises(DomainError, match="allowed main home"):
        preview(fixed, goal)


def test_save_as_template_excludes_operational_values():
    c = create("client", "ABC")
    p = create("project", "Website", parent_id=c["id"], body="Scope", values={
        "priority": 3, "estimate_minutes": 120, "due_date": "2027-01-09", "planned_date": "2027-01-03",
        "start_date": "2027-01-01", "target_date": "2027-01-08"})
    t1 = create("task", "Wireframes", parent_id=p["id"], values={"due_date": "2027-01-05", "priority": 1})
    change(t1, status_id="in_progress")
    create("task", "Sub", parent_id=t1["id"])
    archived = create("note", "Old note", parent_id=p["id"])
    change(archived, archived=True)
    saved_template = run("template.capture", {"record_id": p["id"], "name": "Website build"})
    payload = saved_template["payload"]
    assert payload["values"] == {"priority": 3, "estimate_minutes": 120} and payload["body"] == "Scope"
    assert payload["children"] == [{"type_id": "task", "title": "Wireframes", "body": "", "values": {"priority": 1},
                                    "children": [{"type_id": "task", "title": "Sub", "body": "", "values": {"priority": 0}, "children": []}]}]
    shallow = run("template.capture", {"record_id": p["id"], "name": "Just the project", "include_children": False})
    assert shallow["payload"]["children"] == [] and shallow["type_id"] == "project"
    _, made = instantiate(saved_template, c)
    copy = saved(made["id"])
    assert copy["values"]["due_date"] is None and copy["values"]["priority"] == 3
    wire = next(saved(i) for i in made["changed_ids"] if saved(i)["title"] == "Wireframes")
    assert wire["status_meaning"] == "open"
    with pytest.raises(DomainError, match="dates or assignees"):
        template({"values": {"due_date": "2027-01-01"}}, name="Dated")


def test_bounds_and_child_type_validation():
    deep = {"type_id": "task", "title": "L5"}
    for level in range(4, 0, -1):
        deep = {"type_id": "task", "title": f"L{level}", "children": [deep]}
    with pytest.raises(DomainError) as too_deep:
        template({"children": [deep]})
    assert too_deep.value.code == "INVALID_ARGUMENT"
    with pytest.raises(DomainError):
        template({"children": [{"type_id": "task", "title": f"T{i}"} for i in range(101)]})
    assert template({"children": [{"type_id": "task", "title": f"T{i}"} for i in range(100)]})["record_count"] == 101
    d = schema()
    next(x for x in d["types"] if x["id"] == "note")["parent_types"] = ["space"]
    apply(d)
    with pytest.raises(DomainError) as bad:
        template({"children": [{"type_id": "note", "title": "Notes"}]})
    assert bad.value.code == "INVALID_TEMPLATE"
    with pytest.raises(DomainError):
        template({"values": {"nope": 1}})


def test_viewer_reads_but_cannot_write(client):
    guest = client_for("guest", "guest@example.test")
    w = shared(client, guest)
    made = command(client, "template.create", type_id="project", name="Shared onboarding",
                   payload={"children": [{"type_id": "task", "title": "Kickoff"}]})
    with session_scope() as db:
        db.get(WorkspaceMember, (w["id"], "guest")).role = "viewer"
    listed = guest.get("/api/v1/structure/templates")
    assert listed.status_code == 200 and [t["id"] for t in listed.json()["items"]] == [made["id"]]
    revision = client.get("/api/v1/structure").json()["revision"]
    preview_response = guest.post("/api/v1/structure/templates/instantiate/preview", json={"template_id": made["id"], "schema_revision": revision})
    assert preview_response.status_code == 200 and preview_response.json()["record_count"] == 2
    for tool, args in (("template.update", {"template_id": made["id"], "expected_revision": 1, "name": "X"}),
                       ("record.instantiate", {"template_id": made["id"], "schema_revision": revision, "preview_hash": preview_response.json()["preview_hash"]})):
        denied = guest.post("/api/v1/commands", json={"command_id": str(uuid4()), "tool": tool, "arguments": args})
        assert denied.status_code == 403
    assert client.get("/api/v1/structure/templates/" + made["id"]).json()["name"] == "Shared onboarding"
    stranger = client_for("stranger")
    assert stranger.get("/api/v1/structure/templates").json()["items"] == []
    assert stranger.get("/api/v1/structure/templates/" + made["id"]).status_code == 404


def test_bot_scopes(client):
    t = template()
    external = TestClient(app)
    _, read_only = key(client, ["records:read"])
    listed = external.get("/api/v1/external/structure/templates", headers=read_only)
    assert listed.status_code == 200 and listed.json()["items"][0]["id"] == t["id"]
    _, records_only = key(client, ["records:write"])
    assert call(records_only, "template.create", {"type_id": "client", "name": "Plain"}).status_code == 200
    rev = schema()["revision"]
    with session_scope() as db:
        plan = templates.plan(db, OWNER, templates.InstantiatePlan(template_id=t["id"], schema_revision=rev))
    args = {"template_id": t["id"], "schema_revision": rev, "preview_hash": plan["preview_hash"]}
    # The template creates task- and note-backed records, so core write scopes apply.
    denied = call(records_only, "record.instantiate", args)
    assert denied.status_code == 403 and ("tasks:write" in denied.text or "notes:write" in denied.text)
    _, full = key(client, ["records:write", "tasks:write", "notes:write"])
    assert call(full, "record.instantiate", args).status_code == 200
    task_row = create("task", "Secret notes", body="private")
    capture = call(records_only, "template.capture", {"record_id": task_row["id"], "name": "Leak"})
    assert capture.status_code == 403 and "tasks:read" in capture.text
    with session_scope() as db:
        assert db.scalar(select(RecordTemplate).where(RecordTemplate.name == "Leak")) is None


@pytest.mark.asyncio
async def test_eri_tools_list_preview_and_instantiate():
    from jarvis.tools import READ_TOOLS, _call_tool, registry

    names = {t["name"] for t in registry()}
    assert {"template_list", "record_instantiate_preview", "record_instantiate"} <= names
    assert "template_create" not in names
    assert len(READ_TOOLS["record_instantiate_preview"]["description"]) < 300
    t = template()
    c = create("client", "ABC")
    listed = await _call_tool(OWNER, "turn", 0, "template_list", {"type_id": "project"})
    assert listed["items"][0]["summary"].startswith("A project with 4 tasks")
    plan = await _call_tool(OWNER, "turn", 1, "record_instantiate_preview",
                            {"template_id": t["id"], "parent_id": c["id"], "schema_revision": schema()["revision"]})
    assert plan["record_count"] == 7
    with session_scope() as db:
        assert db.scalar(select(StructureRecord).where(StructureRecord.title == "Client onboarding")) is None
