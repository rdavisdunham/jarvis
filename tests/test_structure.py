import copy
import pytest
from sqlalchemy import select
from jarvis.db import session_scope
from jarvis.domain import DomainError, execute
from jarvis.models import Task, Note, Job
from jarvis import structure
from jarvis.structure_models import StructureRecord, FieldUnderstanding

OWNER = "davin"


def run(tool, arguments, key=None, owner=OWNER):
    from uuid import uuid4

    with session_scope() as db:
        return execute(db, owner, key or str(uuid4()), tool, arguments)["data"]


def definition():
    with session_scope() as db:
        return copy.deepcopy(structure.schema_data(db, OWNER))


def create(type_id="client", title="ABC", **kwargs):
    revision = definition()["revision"]
    return run("record.create", {"type_id": type_id, "title": title, "schema_revision": revision, **kwargs})


def propose(data, **kwargs):
    return run(
        "structure.preview",
        {
            "expected_revision": data["revision"],
            "definition": {k: data[k] for k in ("types", "relationships")},
            **kwargs,
        },
    )


def apply(p):
    return run("structure.apply", {"proposal_id": p["id"], "expected_revision": p["schema_revision"]})


def test_default_schema_requires_descriptions_and_task_behaviors():
    d = definition()
    assert all(t["description"] for t in d["types"])
    assert "work" in next(t for t in d["types"] if t["id"] == "task")["capabilities"]
    d["types"][0]["description"] = "  "
    with pytest.raises(DomainError, match="action details"):
        propose(d)


def test_preview_requires_new_request_and_rejects_stale_record_snapshot():
    d = definition()
    d["types"][0]["name"] = "Context"
    p = run(
        "structure.preview",
        {"expected_revision": 1, "definition": {k: d[k] for k in ("types", "relationships")}},
        key="work:1",
    )
    with pytest.raises(DomainError) as e:
        run("structure.apply", {"proposal_id": p["id"], "expected_revision": 1}, key="work:2")
    assert e.value.code == "SCHEMA_CONFIRMATION_REQUIRED"
    create()
    with pytest.raises(DomainError) as e:
        apply(p)
    assert e.value.code == "STALE_PROPOSAL"


def test_custom_type_and_status_work_without_changing_task_engine():
    d = definition()
    t = copy.deepcopy(next(t for t in d["types"] if t["id"] == "task"))
    t.update(
        id="assignment", name="Assignment", plural="Assignments", description="Coursework with a deadline."
    )
    t["statuses"] = [
        {"id": "ready", "name": "To study", "meaning": "open"},
        {"id": "submitted", "name": "Submitted", "meaning": "completed"},
    ]
    d["types"].append(t)
    apply(propose(d))
    row = create(
        "assignment",
        "Lab report",
        values={"due_date": "2027-01-01", "due_time": "09:00", "due_timezone": "America/Chicago"},
    )
    assert row["status_id"] == "ready" and row["task_id"]
    result = run(
        "record.update",
        {
            "record_id": row["id"],
            "expected_revision": row["revision"],
            "schema_revision": 2,
            "status_id": "submitted",
        },
    )
    assert result["status_meaning"] == "completed"
    with session_scope() as db:
        assert db.get(Task, row["task_id"]).completed_at is not None


def test_parent_cycle_and_cross_owner_links_rejected():
    a = create()
    b = create("project", "Transcript Intelligence", parent_id=a["id"])
    with pytest.raises(DomainError) as e:
        run(
            "record.update",
            {"record_id": a["id"], "expected_revision": 1, "schema_revision": 1, "parent_id": b["id"]},
        )
    assert e.value.code == "HIERARCHY_CYCLE"
    with pytest.raises(DomainError):
        run(
            "record.create",
            {"type_id": "task", "title": "Secret", "schema_revision": 1, "parent_id": a["id"]},
            owner="other",
        )


def test_explicit_clear_overrides_inheritance():
    d = definition()
    task = next(t for t in d["types"] if t["id"] == "task")
    task["fields"].append(
        {
            "id": "client",
            "name": "Client",
            "description": "The customer organization this work is for.",
            "kind": "relation",
            "target_types": ["client"],
            "inherit": True,
        }
    )
    apply(propose(d))
    client = create()
    project = create("project", "Andi", parent_id=client["id"])
    task = create("task", "Central docs", parent_id=project["id"])
    assert task["values"]["client"] == client["id"]
    assert task["inherited"]["client"] == client["id"]
    changed = run(
        "record.update",
        {"record_id": task["id"], "expected_revision": 1, "schema_revision": 2, "values": {"client": None}},
    )
    assert changed["values"]["client"] is None and "client" not in changed["inherited"]


def test_relation_cardinality_and_extra_link_does_not_move_home():
    d = definition()
    d["relationships"].append(
        {
            "id": "client",
            "name": "Client",
            "description": "The customer of this initiative.",
            "source_types": ["project"],
            "target_types": ["client"],
            "cardinality": "many_to_one",
        }
    )
    apply(propose(d))
    a = create()
    b = create(title="Other client")
    p = create("project", "Andi", parent_id=a["id"])
    linked = run(
        "record.link",
        {
            "source_id": p["id"],
            "target_id": a["id"],
            "relationship_id": "client",
            "expected_revision": 1,
            "schema_revision": 2,
        },
    )
    assert linked["parent_id"] == a["id"]
    with pytest.raises(DomainError) as e:
        run(
            "record.link",
            {
                "source_id": p["id"],
                "target_id": b["id"],
                "relationship_id": "client",
                "expected_revision": 2,
                "schema_revision": 2,
            },
        )
    assert e.value.code == "RELATIONSHIP_CARDINALITY"


def test_source_task_and_note_edits_are_not_overwritten_by_record_property_edit():
    task = create("task", "Original")
    with session_scope() as db:
        core = db.get(Task, task["task_id"])
        execute(
            db,
            OWNER,
            "external-task-edit",
            "task.update",
            {"task_id": core.id, "expected_revision": core.revision, "title": "Latest"},
        )
    result = run(
        "record.update",
        {"record_id": task["id"], "expected_revision": 2, "schema_revision": 1, "values": {"priority": 2}},
    )
    assert result["title"] == "Latest" and result["values"]["priority"] == 2
    note = create("note", "Notes", body="Keep all this content")
    with session_scope() as db:
        assert db.get(Note, note["note_id"]).content == "Keep all this content"


def test_schema_change_invalidates_understanding_and_keeps_old_task_data():
    row = create("task", "Keep me")
    d = definition()
    next(t for t in d["types"] if t["id"] == "client")["description"] = (
        "Clients that I do projects for at Brainforge."
    )
    apply(propose(d))
    with session_scope() as db:
        state = db.get(FieldUnderstanding, (OWNER, "type:client"))
        assert state.status == "assessing"
        assert db.get(Task, row["task_id"]).title == "Keep me"
        assert db.scalar(select(Job.id).where(Job.kind == "assess_field"))


def test_retries_do_not_create_duplicate_records():
    a = run("record.create", {"type_id": "task", "title": "Once", "schema_revision": 1}, key="once")
    b = run("record.create", {"type_id": "task", "title": "Once", "schema_revision": 1}, key="once")
    assert a["id"] == b["id"] and a["task_id"] == b["task_id"]
    with session_scope() as db:
        assert len(list(db.scalars(select(StructureRecord)))) == 1


def test_value_validation_and_field_identity_are_enforced():
    with pytest.raises(DomainError):
        create("task", "Bad", values={"priority": True})
    with pytest.raises(DomainError):
        create("task", "Bad", values={"made_up": "x"})
    with pytest.raises(DomainError):
        create("task", "Bad", values={"due_date": "not a date"})


def test_api_schema_and_record_discovery(client):
    response = client.get("/api/v1/structure")
    assert response.status_code == 200 and response.json()["revision"] == 1
    row = create()
    response = client.get("/api/v1/structure/records/" + row["id"])
    assert response.status_code == 200 and response.json()["title"] == "ABC"
    assert client.get("/api/v1/structure/records?query=ABC").json()["items"][0]["id"] == row["id"]


def test_custom_record_receipt_revert_is_guarded(monkeypatch):
    from cryptography.fernet import Fernet
    from jarvis.config import get_settings
    from jarvis.models import ActionChange
    from jarvis.action_history import inverse, revert
    from uuid import uuid4

    monkeypatch.setattr(get_settings(), "integration_encryption_key", Fernet.generate_key().decode())
    item = create("task", "Created through generic work")
    with session_scope() as db:
        changes = list(
            db.scalars(
                select(ActionChange).where(
                    ActionChange.entity_kind == "record", ActionChange.entity_id == item["id"]
                )
            )
        )
        assert len(changes) == 1
        command, reason = inverse(db, changes[0])
        assert command, reason
        result = revert(db, OWNER, OWNER, changes[0].id, str(uuid4()))
        assert result["data"]["archived"]
        assert db.get(Task, item["task_id"]).archived


def test_active_relationship_cardinality_change_blocks_preview():
    a, b, c = create(title="A"), create(title="B"), create(title="C")
    run(
        "record.link",
        {
            "source_id": a["id"],
            "target_id": b["id"],
            "relationship_id": "related",
            "expected_revision": 1,
            "schema_revision": 1,
        },
    )
    run(
        "record.link",
        {
            "source_id": a["id"],
            "target_id": c["id"],
            "relationship_id": "related",
            "expected_revision": 2,
            "schema_revision": 1,
        },
    )
    d = definition()
    next(r for r in d["relationships"] if r["id"] == "related")["cardinality"] = "one_to_one"
    assert propose(d)["impact"]["blocking_count"] > 0


def test_operational_binding_type_invalid_is_clean_api_error(client):
    d = definition()
    next(t for t in d["types"] if t["id"] == "task")["fields"][0]["kind"] = "number"
    from uuid import uuid4

    response = client.post(
        "/api/v1/commands",
        json={
            "command_id": str(uuid4()),
            "tool": "structure.preview",
            "arguments": {
                "expected_revision": 1,
                "definition": {k: d[k] for k in ("types", "relationships")},
            },
        },
    )
    assert response.status_code == 400, response.text


def test_structural_undo_is_new_preview():
    d = definition()
    d["types"][0]["name"] = "Context"
    p = propose(d)
    apply(p)
    restored = run("structure.restore", {"proposal_id": p["id"], "expected_revision": 2})
    assert definition()["types"][0]["name"] == "Context"
    assert restored["requires_confirmation"] and restored["definition"]["types"][0]["name"] == "Space"
    apply(restored)
    assert definition()["types"][0]["name"] == "Space"


def test_imported_core_tasks_appear_after_schema_exists_without_duplicates():
    definition()
    with session_scope() as db:
        core = Task(owner_id=OWNER, title="Imported later")
        foreign = Task(owner_id="other", title="Not mine")
        db.add_all([core, foreign])
        db.flush()
        identity = core.id
    with session_scope() as db:
        first = structure.records(db, OWNER)
        second = structure.records(db, OWNER)
        assert [r["title"] for r in first["items"]] == ["Imported later"]
        assert len(second["items"]) == 1 and second["items"][0]["task_id"] == identity
    with session_scope() as db:
        task = db.get(Task, identity)
        task.archived = True
        task.revision += 1
    with session_scope() as db:
        assert not structure.records(db, OWNER)["items"]
        assert structure.records(db, OWNER, archived=True)["items"][0]["archived"]


def test_integration_edit_invalidates_an_open_generic_record():
    from datetime import date

    item = create("task", "Old open card")
    with session_scope() as db:
        core = db.get(Task, item["task_id"])
        core.due_date = date(2027, 2, 1)
        core.revision += 1
    with pytest.raises(DomainError) as exc:
        run(
            "record.update",
            {
                "record_id": item["id"],
                "expected_revision": item["revision"],
                "schema_revision": 1,
                "values": {"due_date": "2027-01-01"},
            },
        )
    assert exc.value.status == 409
    with session_scope() as db:
        assert db.get(Task, item["task_id"]).due_date == date(2027, 2, 1)


def test_record_deadline_undo_clears_new_binding_and_rejects_later_core_edit(monkeypatch):
    from cryptography.fernet import Fernet
    from jarvis.config import get_settings
    from jarvis.models import ActionChange
    from jarvis.action_history import inverse, revert, public_change
    from datetime import date
    from uuid import uuid4

    monkeypatch.setattr(get_settings(), "integration_encryption_key", Fernet.generate_key().decode())
    item = create("task", "Deadline undo")
    changed = run(
        "record.update",
        {
            "record_id": item["id"],
            "expected_revision": 1,
            "schema_revision": 1,
            "values": {"due_date": "2027-01-01"},
        },
        key="deadline-change",
    )
    with session_scope() as db:
        receipt = db.scalar(select(ActionChange).where(ActionChange.command_id == "deadline-change"))
        inverse_command, reason = inverse(db, receipt)
        assert inverse_command, reason
        assert inverse_command[1]["values"] == {"due_date": None}
        assert public_change(db, receipt)["fields"]["due_date"]["after"] == "2027-01-01"
        assert "values" not in public_change(db, receipt)["fields"]
        revert(db, OWNER, OWNER, receipt.id, str(uuid4()))
        assert db.get(Task, item["task_id"]).due_date is None
    with session_scope() as db:
        current = structure.data(db, db.get(StructureRecord, item["id"]))
    run(
        "record.update",
        {
            "record_id": item["id"],
            "expected_revision": current["revision"],
            "schema_revision": 1,
            "values": {"due_date": "2027-02-01"},
        },
        key="later-deadline",
    )
    with session_scope() as db:
        core = db.get(Task, item["task_id"])
        core.due_date = date(2027, 3, 1)
        core.revision += 1
        db.flush()
        receipt = db.scalar(select(ActionChange).where(ActionChange.command_id == "later-deadline"))
        assert inverse(db, receipt)[0] is None


def test_relationship_receipt_reverts_link_and_unlink(monkeypatch):
    from cryptography.fernet import Fernet
    from jarvis.config import get_settings
    from jarvis.models import ActionChange
    from jarvis.action_history import public_change, revert
    from jarvis.structure_models import StructureLink
    from uuid import uuid4

    monkeypatch.setattr(get_settings(), "integration_encryption_key", Fernet.generate_key().decode())
    a, b = create(title="A"), create(title="B")
    run(
        "record.link",
        {
            "source_id": a["id"],
            "target_id": b["id"],
            "relationship_id": "related",
            "expected_revision": 1,
            "schema_revision": 1,
        },
        key="link-action",
    )
    with session_scope() as db:
        receipts = list(db.scalars(select(ActionChange).where(ActionChange.command_id == "link-action")))
        assert len(receipts) == 1 and public_change(db, receipts[0])["summary"] == "Linked: A → B"
        revert(db, OWNER, OWNER, receipts[0].id, "undo-link")
        assert not list(db.scalars(select(StructureLink)))
        removal = db.scalar(select(ActionChange).where(ActionChange.command_id == "undo-link"))
        assert public_change(db, removal)["can_revert"]
        revert(db, OWNER, OWNER, removal.id, str(uuid4()))
        assert len(list(db.scalars(select(StructureLink)))) == 1


def test_custom_workflow_can_start_in_backlog_without_an_open_status():
    d = definition()
    t = next(t for t in d["types"] if t["id"] == "task")
    t["statuses"] = [
        {"id": "later", "name": "Later", "meaning": "backlog"},
        {"id": "finished", "name": "Finished", "meaning": "completed"},
    ]
    apply(propose(d))
    row = create("task", "A later item")
    assert row["status_id"] == "later" and row["status_meaning"] == "backlog"
    with pytest.raises(DomainError):
        run(
            "record.update",
            {
                "record_id": row["id"],
                "expected_revision": row["revision"],
                "schema_revision": 2,
                "status_id": None,
            },
        )


def test_custom_field_patch_tool_contract_is_bounded():
    from jarvis.ui_contracts import UI_TOOLS
    from jsonschema import validate, ValidationError

    schema = UI_TOOLS["ui_editor"]["parameters"]
    validate(
        {"operation": "patch", "changes": {"values": {"department": "ops", "tags": ["a"], "flag": True}}},
        schema,
    )
    with pytest.raises(ValidationError):
        validate({"operation": "patch", "changes": {"values": {"nested": {"too": "deep"}}}}, schema)
