"""Audit regressions: null policy, structure/legacy hierarchy agreement, compact task reads."""

from uuid import uuid4

import pytest
from jarvis import models, structure
from jarvis.db import session_scope
from jarvis.domain import COMMANDS, NOT_NULL, DomainError, execute
from jarvis.structure_models import StructureRecord
from jarvis.tools import registry

OWNER = "davin"
REJECTS_NULL = ("cannot be cleared", "cannot be null", "cannot be empty", "check the action details")


def run(tool, arguments, key=None):
    with session_scope() as db:
        return execute(db, OWNER, key or str(uuid4()), tool, arguments)["data"]


def revision(model, identity):
    with session_scope() as db:
        return db.get(model, identity).revision


def nullable(schema):
    return [k for k, v in schema.get("properties", {}).items() if any(o.get("type") == "null" for o in v.get("anyOf", []))]


def accepts_null(tool, arguments):
    try:
        run(tool, arguments)
    except DomainError as exc:
        assert not any(text in exc.message.lower() for text in REJECTS_NULL), (tool, arguments, exc.message)


def test_registry_never_advertises_null_for_uncleared_fields():
    tools = {t["name"]: t["parameters"] for t in registry()}
    for command, fields in NOT_NULL.items():
        name = command.replace(".", "_")
        if name in tools:
            assert not set(nullable(tools[name])) & fields, name
    for name in ("task_batch", "task_selection_update"):
        for definition in tools[name]["$defs"].values():
            assert not set(nullable(definition)) & NOT_NULL["task.update"], name


def test_every_nullable_update_property_is_accepted_as_null():
    tools = {t["name"]: t["parameters"] for t in registry()}
    space = run("space.create", {"name": "Work"})
    area = run("area.create", {"name": "Ops", "space_id": space["id"]})
    goal = run("goal.create", {"name": "Grow"})
    project = run("project.create", {"name": "Launch"})
    actor = run("actor.create", {"name": "Sam"})
    task = run("task.create", {"title": "Ship", "notes": "Details", "priority": 2})
    schedule = run("schedule.create", {"title": "Ping", "when": "2030-01-01T09:00", "timezone": "UTC"})
    with session_scope() as db:
        schema_revision = structure.ensure(db, OWNER).revision
    client = run("record.create", {"type_id": "client", "title": "ABC", "body": "Notes", "schema_revision": schema_revision})
    targets = {
        "space.update": (models.Space, space["id"], {}),
        "area.update": (models.Area, area["id"], {}),
        "goal.update": (models.Goal, goal["id"], {}),
        "project.update": (models.Project, project["id"], {}),
        "actor.update": (models.Actor, actor["id"], {}),
        "task.update": (models.Task, task["id"], {}),
        "schedule.update": (models.Schedule, schedule["id"], {}),
        "record.update": (StructureRecord, client["id"], {"schema_revision": schema_revision}),
    }
    checked = 0
    for command, (model, identity, extra) in targets.items():
        assert command in COMMANDS
        for key in nullable(tools[command.replace(".", "_")]):
            accepts_null(command, {command.split(".")[0] + "_id": identity, "expected_revision": revision(model, identity), **extra, key: None})
            checked += 1
    for key in nullable(tools["task_batch"]["$defs"]["TaskUpdate"]):
        accepts_null("task.batch", {"items": [{"task_id": task["id"], "expected_revision": revision(models.Task, task["id"]), key: None}]})
    assert checked >= 30
    with session_scope() as db:
        saved = db.get(models.Task, task["id"])
        assert (saved.notes, saved.priority) == ("", 0)
        assert db.get(StructureRecord, client["id"]).body == ""


def test_task_update_rejects_null_title_with_clear_message():
    task = run("task.create", {"title": "Keep"})
    with pytest.raises(DomainError, match="cannot be cleared"):
        run("task.update", {"task_id": task["id"], "expected_revision": task["revision"], "title": None})


def schema_data():
    with session_scope() as db:
        return structure.schema_data(db, OWNER)


def record_for(task_id):
    with session_scope() as db:
        from sqlalchemy import select
        row = db.scalar(select(StructureRecord).where(StructureRecord.owner_id == OWNER, StructureRecord.task_id == task_id))
        return structure.data(db, row)


def test_batch_and_selection_edits_reach_structure_records():
    tasks = [run("task.create", {"title": f"Item {i}"}) for i in range(2)]
    schema_data()
    before = record_for(tasks[0]["id"])["revision"]
    run("task.batch", {"items": [{"task_id": t["id"], "expected_revision": t["revision"], "title": "Renamed " + t["title"]} for t in tasks]})
    assert record_for(tasks[0]["id"])["title"] == "Renamed Item 0"
    with session_scope() as db:
        assert db.get(StructureRecord, record_for(tasks[0]["id"])["id"]).title == "Renamed Item 0"
    assert record_for(tasks[0]["id"])["revision"] > before
    from jarvis.task_tools import list_tasks
    with session_scope() as db:
        selection = list_tasks(db, OWNER, {"query": "Renamed"})["selection_id"]
    run("task.selection_update", {"selection_id": selection, "changes": {"notes": "Shared"}})
    with session_scope() as db:
        assert db.get(StructureRecord, record_for(tasks[1]["id"])["id"]).body == "Shared"


def test_project_moves_and_record_homes_stay_in_sync():
    alpha = run("project.create", {"name": "Alpha"})
    beta = run("project.create", {"name": "Beta"})
    task = run("task.create", {"title": "Review", "project_id": alpha["id"]})
    d = schema_data()
    assert record_for(task["id"])["parent_id"] == alpha["id"]
    moved = run("task.update", {"task_id": task["id"], "expected_revision": task["revision"], "project_id": beta["id"]})
    record = record_for(task["id"])
    assert record["parent_id"] == beta["id"]
    run("record.update", {"record_id": record["id"], "expected_revision": record["revision"], "schema_revision": d["revision"], "parent_id": alpha["id"]})
    with session_scope() as db:
        assert db.get(models.Task, task["id"]).project_id == alpha["id"]
    # A custom home clears stale native project membership; external projects stay connector-owned.
    client = run("record.create", {"type_id": "client", "title": "ABC", "schema_revision": d["revision"]})
    record = record_for(task["id"])
    run("record.update", {"record_id": record["id"], "expected_revision": record["revision"], "schema_revision": d["revision"], "parent_id": client["id"]})
    with session_scope() as db:
        assert db.get(models.Task, task["id"]).project_id is None
    assert moved["project_id"] == beta["id"]


def test_task_status_missing_from_custom_workflow_keeps_record_editable():
    d = schema_data()
    t = next(t for t in d["types"] if t["id"] == "task")
    t["statuses"] = [s for s in t["statuses"] if s["meaning"] != "waiting"]
    proposal = run("structure.preview", {"expected_revision": d["revision"], "definition": {k: d[k] for k in ("types", "relationships")}})
    run("structure.apply", {"proposal_id": proposal["id"], "expected_revision": proposal["schema_revision"]})
    task = run("task.create", {"title": "Blocked"})
    run("task.update", {"task_id": task["id"], "expected_revision": task["revision"], "status": "waiting"})
    record = record_for(task["id"])
    assert record["status_id"] == "in_progress"
    with session_scope() as db:
        assert db.get(StructureRecord, record["id"]).status_id == "in_progress"
    schema = schema_data()["revision"]
    run("record.update", {"record_id": record["id"], "expected_revision": record["revision"], "schema_revision": schema, "title": "Still blocked"})
    with session_scope() as db:
        saved = db.get(models.Task, task["id"])
        assert (saved.title, saved.status) == ("Still blocked", "waiting")
    record = record_for(task["id"])
    run("record.update", {"record_id": record["id"], "expected_revision": record["revision"], "schema_revision": schema, "status_id": "open"})
    with session_scope() as db:
        assert db.get(models.Task, task["id"]).status == "open"


def test_reverting_added_bound_fields_restores_task_defaults(monkeypatch):
    from cryptography.fernet import Fernet
    from jarvis.action_history import inverse, revert
    from jarvis.config import get_settings
    from jarvis.models import ActionChange
    from sqlalchemy import select

    monkeypatch.setattr(get_settings(), "integration_encryption_key", Fernet.generate_key().decode())
    d = schema_data()
    item = run("record.create", {"type_id": "task", "title": "Bound", "schema_revision": d["revision"]})
    edit = "edit-bound"
    run("record.update", {"record_id": item["id"], "expected_revision": item["revision"], "schema_revision": d["revision"], "values": {"priority": 2, "assignee": "Sam", "estimate_minutes": 30}}, key=edit)
    with session_scope() as db:
        change = db.scalar(select(ActionChange).where(ActionChange.command_id == edit, ActionChange.entity_kind == "record"))
        command, reason = inverse(db, change)
        assert command, reason
        revert(db, OWNER, OWNER, change.id, str(uuid4()))
        task = db.get(models.Task, item["task_id"])
        assert (task.priority, task.assignee, task.estimate_minutes) == (0, "owner", None)


@pytest.mark.asyncio
async def test_task_reads_expose_flexible_home_for_duplicate_titles(client):
    from jarvis.tools import call_tool

    d = schema_data()
    acme = run("record.create", {"type_id": "client", "title": "Acme", "schema_revision": d["revision"]})
    beacon = run("record.create", {"type_id": "project", "title": "Beacon Dispatch", "parent_id": acme["id"], "schema_revision": d["revision"]})
    target = run("record.create", {"type_id": "task", "title": "Review proposal", "body": "x" * 5000, "parent_id": beacon["id"], "schema_revision": d["revision"]})
    other = run("task.create", {"title": "Review proposal"})
    with session_scope() as db:
        assert db.get(models.Task, target["task_id"]).project_id is None
    found = await call_tool(OWNER, "turn", 0, "task_resolve", {"scope": "search", "query": "review proposal"})
    assert found["ambiguous"] and len(found["tasks"]) == 2
    homes = {t["id"]: t["home"] for t in found["tasks"]}
    assert homes[target["task_id"]] == [{"id": acme["id"], "title": "Acme", "type_id": "client"}, {"id": beacon["id"], "title": "Beacon Dispatch", "type_id": "project"}]
    assert homes[other["id"]] == []
    assert all("notes" not in t and len(t["notes_preview"]) <= 200 for t in found["tasks"])
    narrowed = await call_tool(OWNER, "turn", 1, "task_resolve", {"scope": "search", "query": "review proposal", "home_id": beacon["id"]})
    assert [t["id"] for t in narrowed["tasks"]] == [target["task_id"]] and not narrowed["ambiguous"]
    listed = await call_tool(OWNER, "turn", 2, "task_list", {"home_id": acme["id"]})
    # Projects are actionable now and belong to the same subtree.
    assert {t["id"] for t in listed["tasks"]} == {target["task_id"], beacon["task_id"]}
    assert next(t for t in listed["tasks"] if t["id"] == target["task_id"])["record_id"] == target["id"]
    detail = await call_tool(OWNER, "turn", 3, "task_get", {"task_id": target["task_id"]})
    assert detail["home"][-1]["id"] == beacon["id"] and len(detail["notes"]) == 5000


@pytest.mark.asyncio
async def test_record_list_is_compact_bounded_and_reconciles_with_event(client):
    from jarvis.tools import call_tool, registry
    from sqlalchemy import func, select

    schema = next(t for t in registry() if t["name"] == "record_list")["parameters"]["properties"]
    assert schema["limit"]["maximum"] == 50
    for i in range(30):
        run("task.create", {"title": f"Bulk {i}", "notes": "long body"})
    listed = await call_tool(OWNER, "turn", 0, "record_list", {"capability": "work"})
    assert len(listed["items"]) == 25 and listed["has_more"]
    assert "body" not in listed["items"][0] and "home" in listed["items"][0]
    full = await call_tool(OWNER, "turn", 1, "record_list", {"capability": "work", "limit": 1, "detail": "full"})
    assert full["items"][0]["body"] == "long body"
    with session_scope() as db:
        assert not structure.core_drift(db, OWNER)
        events = db.scalar(select(func.count()).select_from(models.Event))
        task = db.scalar(select(models.Task).where(models.Task.title == "Bulk 0"))
        task.title, task.revision = "Synced elsewhere", task.revision + 1
    with session_scope() as db:
        assert structure.core_drift(db, OWNER)
    with session_scope() as db:
        structure.records(db, OWNER, query="Synced")
        assert db.scalar(select(func.count()).select_from(models.Event)) > events
        record = db.scalar(select(StructureRecord).where(StructureRecord.task_id == task.id))
        assert record.title == "Synced elsewhere"
        changed = db.scalar(select(models.Event).where(models.Event.kind == "record.changed", models.Event.entity_id == record.id).order_by(models.Event.id.desc()))
        assert changed.revision == record.revision
        assert not structure.core_drift(db, OWNER)


def test_agent_ui_record_edits_are_not_human_routing_evidence():
    from jarvis.action_history import request_of
    from jarvis.structure_models import RoutingObservation
    from sqlalchemy import select

    d = schema_data()
    home = run("record.create", {"type_id": "client", "title": "ABC", "schema_revision": d["revision"]})
    agent = run("record.create", {"type_id": "task", "title": "Agent filed", "schema_revision": d["revision"]})
    owner = run("record.create", {"type_id": "task", "title": "Owner filed", "schema_revision": d["revision"]})
    tagged = "ui-agent:work-1:4:" + str(uuid4())
    run("record.update", {"record_id": agent["id"], "expected_revision": agent["revision"], "schema_revision": d["revision"], "parent_id": home["id"]}, key=tagged)
    run("record.update", {"record_id": owner["id"], "expected_revision": owner["revision"], "schema_revision": d["revision"], "parent_id": home["id"]})
    with session_scope() as db:
        observed = {o.record_id for o in db.scalars(select(RoutingObservation).where(RoutingObservation.origin == "human"))}
    assert owner["id"] in observed and agent["id"] not in observed
    assert request_of(tagged) == "work-1"


def test_rule_routed_record_resave_is_not_evidence_but_departure_is():
    from jarvis.structure_models import RoutingObservation
    from sqlalchemy import select

    d = schema_data()
    home = run("record.create", {"type_id": "client", "title": "ABC", "schema_revision": d["revision"]})
    other = run("record.create", {"type_id": "client", "title": "XYZ", "schema_revision": d["revision"]})
    run("routing.create", {"phrase": "ABC CSR", "type_id": "task", "parent_id": home["id"], "reason": "client"})
    task = run("record.create", {"type_id": "task", "title": "ABC CSR docs", "schema_revision": d["revision"]})
    assert task["parent_id"] == home["id"]
    task = run("record.update", {"record_id": task["id"], "expected_revision": task["revision"], "schema_revision": d["revision"], "parent_id": home["id"]})
    with session_scope() as db:
        assert not list(db.scalars(select(RoutingObservation).where(RoutingObservation.record_id == task["id"])))
    run("record.update", {"record_id": task["id"], "expected_revision": task["revision"], "schema_revision": d["revision"], "parent_id": other["id"]})
    with session_scope() as db:
        assert list(db.scalars(select(RoutingObservation).where(RoutingObservation.record_id == task["id"])))


def test_old_journal_offsets_compare_as_instants():
    from datetime import datetime, timezone, timedelta
    from jarvis.action_history import instants
    from jarvis.domain import encode

    chicago = datetime(2030, 3, 9, 9, 0, tzinfo=timezone(timedelta(hours=-6)))
    assert encode({"next_run_at": chicago}) == {"next_run_at": "2030-03-09T15:00:00+00:00"}
    old = {"anchor_at": "2030-03-09T09:00:00-06:00", "values": {"at": ["2030-03-09T20:30:00+05:30"]}, "title": "2030"}
    assert instants(old) == {"anchor_at": "2030-03-09T15:00:00+00:00", "values": {"at": ["2030-03-09T15:00:00+00:00"]}, "title": "2030"}


@pytest.mark.asyncio
async def test_null_optional_home_filter_means_no_filter():
    from jarvis.tools import call_tool
    row = run("task.create", {"title": "Optional home"})
    found = await call_tool(OWNER,"null-filter",0,"task_resolve",{"scope":"search","query":"Optional home","home_id":None})
    assert [t["id"] for t in found["tasks"]] == [row["id"]]
