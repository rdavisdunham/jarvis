import json
from datetime import date, datetime
from uuid import uuid4

import pytest
from dateutil import tz
from jarvis import agent_models, task_tools
from jarvis.agent_instructions import backend_instructions, live_instructions
from jarvis.db import session_scope
from jarvis.domain import DomainError, execute, preferences
from jarvis.models import Task
from jarvis.remote_status import retry_metadata
from jarvis.time_tools import resolve_time
from jarvis.tool_catalog import CORE, GROUPS, ToolSession, strict_compatible
from jarvis.tools import call_tool, registry
from sqlalchemy import select


def command(tool, args, key=None):
    with session_scope() as db:
        return execute(db, "davin", key or str(uuid4()), tool, args)["data"]


def listing(args=None, owner="davin"):
    with session_scope() as db:
        return task_tools.list_tasks(db, owner, args or {})


def seed(count=3):
    with session_scope() as db:
        rows = [
            Task(owner_id="davin", title=f"Matching {i:04d}", tags=["work"], due_date=date(2030, 1, 14))
            for i in range(count)
        ]
        db.add_all(rows)
        db.flush()
        return [r.id for r in rows]


def test_catalog_is_complete_and_discovery_is_local():
    catalog = registry()
    session = ToolSession(catalog)
    names = {t["name"] for t in catalog}
    assert len(names) == len(catalog)
    assert set(session.names) == set(CORE) and len(CORE) < 20
    assert {"note_search", "note_read", "organization_list", "ui_show", "calendar_list"} <= set(session.names)
    assert set().union(*(set(group[1]) for group in GROUPS.values())) | set(CORE) == names
    assert all(len(group[1]) < 10 for group in GROUPS.values())
    old = list(session.names)
    result = session.load({"groups": ["notes", "notes"]})
    assert "note_tasks" in session.names and len(session.names) == len(set(session.names))
    assert session.names[: len(old)] == old
    assert ToolSession(catalog).names == old
    assert "parameters" not in json.dumps(result)
    with pytest.raises(DomainError):
        session.load({"groups": ["shell"]})


def test_patch_omission_and_strict_read_schemas():
    catalog = {t["name"]: t for t in registry()}
    assert strict_compatible(catalog["tools_load"]["parameters"])
    assert strict_compatible(catalog["task_get"]["parameters"])
    assert not strict_compatible(catalog["task_update"]["parameters"])
    schema = catalog["task_update"]["parameters"]
    assert "title" in schema["properties"]
    assert "due_date" not in schema["required"]
    assert "default" not in schema["properties"]["due_date"]


def test_global_prompts_are_small_and_live_has_no_schema_mechanics():
    with session_scope() as db:
        prefs = preferences(db, "davin")
    backend = backend_instructions(prefs)
    live = live_instructions(prefs)
    assert len(backend.encode()) < 7500 and len(live.encode()) < 6500
    assert "expected_revision" not in live and "tools_load" not in live
    assert "Delegation policy:" in live and "untrusted" in live
    assert "untrusted" in backend and "tools_load" in backend
    assert "openai" not in agent_models.catalog()


def test_exact_filters_selection_and_authoritative_counts():
    ids = seed(125)
    with session_scope() as db:
        db.add_all(
            [
                Task(owner_id="other", title="Private", tags=["work"], due_date=date(2030, 1, 14)),
                Task(owner_id="davin", title="Blocked", tags=["work", "blocked"], due_date=date(2030, 1, 14)),
                Task(
                    owner_id="davin",
                    title="In progress",
                    status="in_progress",
                    tags=["work"],
                    due_date=date(2030, 1, 14),
                ),
                Task(owner_id="davin", title="Late", tags=["work"], due_date=date(2030, 1, 15)),
            ]
        )
    preview = listing(
        {
            "tags_all": ["work"],
            "tags_none": ["blocked"],
            "status": "open",
            "due_from": "2030-01-14",
            "due_through": "2030-01-14",
            "limit": 10,
        }
    )
    assert preview["match_count"] == 125 and preview["returned_count"] == 10
    args = {"selection_id": preview["selection_id"], "changes": {"priority": 2}}
    result = command("task.selection_update", args, "selection-once")
    assert result["requested_count"] == result["applied_count"] == 125
    assert result["unchanged_count"] == 0 and set(result["applied_ids"]) == set(ids)
    assert command("task.selection_update", args, "selection-once") == result
    page = listing({"selection_id": preview["selection_id"], "offset": 10, "limit": 100})
    assert len(page["tasks"]) == 100 and all(r["revision"] == 1 for r in page["tasks"])
    with session_scope() as db:
        assert sum(t.priority == 2 for t in db.scalars(select(Task))) == 125


def test_selection_conflict_is_atomic_and_owner_scoped():
    ids = seed()
    preview = listing()
    command("task.update", {"task_id": ids[-1], "expected_revision": 1, "priority": 1})
    with pytest.raises(DomainError, match="changed on another device"):
        command(
            "task.selection_update", {"selection_id": preview["selection_id"], "changes": {"priority": 2}}
        )
    with session_scope() as db:
        assert db.get(Task, ids[0]).priority == 0
    with pytest.raises(DomainError) as e:
        listing({"selection_id": preview["selection_id"]}, owner="other")
    assert e.value.code == "SELECTION_EXPIRED"
    task_tools.snapshots[preview["selection_id"]]["expires"] = 0
    with pytest.raises(DomainError):
        listing({"selection_id": preview["selection_id"]})


def test_empty_oversized_and_unchanged_selection():
    empty = listing()
    result = command(
        "task.selection_update", {"selection_id": empty["selection_id"], "changes": {"priority": 1}}
    )
    assert result["requested_count"] == result["applied_count"] == 0
    ids = seed(1001)
    big = listing()
    assert big["truncated"] and not big["selection_complete"] and big["selection_id"] is None
    result = command("task.batch", {"items": [{"task_id": ids[0], "expected_revision": 1, "priority": 0}]})
    assert result["unchanged_count"] == 1 and result["applied_count"] == 0
    assert result["tasks"][0]["revision"] == 1
    with pytest.raises(DomainError, match="once"):
        command("task.batch", {"items": [{"task_id": ids[0], "expected_revision": 1}] * 2})


@pytest.mark.parametrize(
    "args", [{"limit": 0}, {"due_from": "2030-02-30"}, {"status": "invented"}, {"unknown": True}]
)
async def test_read_validation_is_uniform(args):
    with pytest.raises(DomainError) as e:
        await call_tool("davin", str(uuid4()), 0, "task_list", args)
    assert e.value.code == "INVALID_ARGUMENT"


async def test_malformed_id_recovery_distinguishes_valid_missing_record():
    with pytest.raises(DomainError) as e:
        await call_tool("davin", str(uuid4()), 0, "note_read", {"note_id": "short"})
    assert e.value.code == "MALFORMED_ID"
    assert "does not mean" in str(e.value)
    with pytest.raises(DomainError) as e:
        await call_tool("davin", str(uuid4()), 0, "note_read", {"note_id": str(uuid4())})
    assert e.value.code == "NOT_FOUND"


@pytest.mark.parametrize(
    "local,zone",
    [
        ("2030-03-10T02:30", "America/Chicago"),
        ("2030-10-06T02:15", "Australia/Lord_Howe"),
        ("2011-12-30T12:00", "Pacific/Apia"),
    ],
)
def test_gap_alternatives_are_actually_valid(local, zone):
    result = resolve_time(local, zone)
    assert result["status"] == "nonexistent"
    assert len(result["alternatives"]) == 2
    for value in result["alternatives"]:
        assert tz.datetime_exists(datetime.fromisoformat(value["local"]).replace(tzinfo=tz.gettz(zone)))


def test_fold_offsets_and_date_only():
    result = resolve_time("2030-11-03T01:30", "America/Chicago")
    assert result["status"] == "ambiguous"
    assert {r["offset"] for r in result["candidates"]} == {"-05:00", "-06:00"}
    assert resolve_time("2030-11-03T01:30-06:00", "America/Chicago")["valid"]
    assert resolve_time("2030-03-10T02:30-06:00", "America/Chicago")["status"] == "offset_mismatch"
    with pytest.raises(DomainError):
        resolve_time("2030-01-14", "America/Chicago")


def test_graph_receipt_contains_current_peer_revisions_and_diff():
    p1 = command("project.create", {"name": "First"})
    p2 = command("project.create", {"name": "Second"})
    goal = command("goal.create", {"name": "Outcome", "project_ids": [p1["id"]]})
    result = command(
        "goal.update",
        {"goal_id": goal["id"], "expected_revision": goal["revision"], "project_ids": [p2["id"]]},
    )
    changes = result["relationship_changes"]
    assert changes["added_ids"] == [p2["id"]] and changes["removed_ids"] == [p1["id"]]
    peers = {r["id"]: r for r in changes["peers"]}
    assert peers[p1["id"]]["goal_ids"] == [] and peers[p1["id"]]["revision"] == 3
    assert peers[p2["id"]]["goal_ids"] == [goal["id"]] and peers[p2["id"]]["revision"] == 2


def test_note_task_receipt_proves_provenance_and_deduplication():
    note = command("note.create", {"title": "Decisions", "content": "I will call the dentist."})
    args = {
        "note_id": note["id"],
        "expected_revision": 1,
        "items": [{"title": "Call dentist", "evidence": "I will call the dentist."}],
    }
    result = command("note.tasks", args)
    assert result["created_count"] == 1 and result["existing_count"] == 0
    task = result["tasks"][0]
    assert task["source_note_id"] == note["id"] and task["source_note_revision"] == 1
    repeat = command("note.tasks", args)
    assert repeat["created_count"] == 0 and repeat["existing_count"] == 1
    assert repeat["tasks"][0]["id"] == task["id"] and repeat["tasks"][0]["evidence"] == task["evidence"]


@pytest.mark.parametrize(
    "status,active",
    [("queued", True), ("retrying", True), ("succeeded", False), ("unconfirmed", False), ("failed", False)],
)
def test_remote_pending_is_not_a_followup_promise(status, active):
    result = retry_metadata(status, {"attempts": 3})
    assert result["retry_active"] == active
    assert bool(result["poll_after_seconds"]) == active
    assert result["completion_notification_scheduled"] is False
    assert result["assistant_followup_scheduled"] is False


def test_strict_subset_never_sends_unsupported_array_constraints():
    catalog = {t["name"]: t for t in registry()}
    assert not strict_compatible(catalog["ui_select"]["parameters"])
    assert catalog["ui_select"]["parameters"]["properties"]["task_ids"]["uniqueItems"]
    # Property names are not schema keywords.
    assert strict_compatible(
        {
            "type": "object",
            "properties": {"uniqueItems": {"type": "boolean"}},
            "required": ["uniqueItems"],
            "additionalProperties": False,
        }
    )


async def test_ui_selection_uniqueness_remains_server_enforced():
    identity = str(uuid4())
    with pytest.raises(DomainError) as e:
        await call_tool("davin", str(uuid4()), 0, "ui_select", {"task_ids": [identity, identity]})
    assert e.value.code == "INVALID_ARGUMENT"


def test_navigation_schema_identifies_notes_and_record_opening():
    tools = {t["name"]: t for t in registry()}
    description = tools["ui_show"]["parameters"]["properties"]["entity_id"]["description"]
    assert "note for notes" in description and "Required to show a specific record" in description
    assert "Reading does not display" in tools["note_read"]["description"]
