import hashlib
from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest
from cryptography.fernet import Fernet
from jarvis import planner
from jarvis.config import get_settings
from jarvis.db import session_scope
from jarvis.domain import DomainError, execute
from jarvis.google_auth import seal, unseal
from jarvis.models import Command, Note, NoteGoalLink, PlanningEntry, Task
from jarvis.planner_schema import PlanRequest
from sqlalchemy import func, select


@pytest.fixture(autouse=True)
def planner_encryption(monkeypatch):
    monkeypatch.setattr(get_settings(), "integration_encryption_key", Fernet.generate_key().decode())


def run(tool, owner="davin", **args):
    with session_scope() as db:
        return execute(db, owner, str(uuid4()), tool, args)["data"]


def request(tasks, **extra):
    return {"start": "2026-10-08T09:00:00-05:00", "end": "2026-10-08T17:00:00-05:00",
            "timezone": "America/Chicago", "scope": "local_only", "tasks": tasks, **extra}


def task(title, minutes, **constraints):
    row = run("task.create", title=title, due_date="2026-10-20", notes="Keep this")
    return {"task_id": row["id"], "expected_revision": row["revision"], "minutes": minutes, **constraints}


def test_proves_optimal_schedule_with_release_deadline_and_dependencies():
    a = task("Review", 60, not_before="2026-10-08T13:00:00-05:00")
    b = task("Send", 45, after_task_ids=[a["task_id"]])
    c = task("Call", 30, not_after="2026-10-08T10:00:00-05:00")
    d = task("Build", 75, not_before="2026-10-08T10:30:00-05:00")
    run("planning.create", title="Busy", start="2026-10-08T14:00:00-05:00",
        end="2026-10-08T14:30:00-05:00", timezone="America/Chicago")
    result = planner.propose("davin", request([a, b, c, d]))
    assert result["status"] == "ready" and not result["saved"]
    assert result["finish"] == "2026-10-08T15:15:00-05:00"
    assert result["optimality"] == "proven_earliest_finish"
    saved = run("planning.commit", plan_token=result["plan_token"])
    assert saved["saved_count"] == 4 and not saved["google_published"]
    # A new client command still replays the plan receipt.
    assert run("planning.commit", plan_token=result["plan_token"]) == saved
    with session_scope() as db:
        assert db.scalar(select(func.count(PlanningEntry.id))) == 5
        assert all(t.notes == "Keep this" and t.due_date.isoformat() == "2026-10-20"
                   for t in db.scalars(select(Task)))


@pytest.mark.parametrize("failure", ["busy", "revision", "unknown", "owner", "expired"])
def test_commit_rechecks_and_never_partially_saves(failure, monkeypatch):
    a, b = task("A", 30), task("B", 30)
    proposed = planner.propose("davin", request([a, b]))
    token = proposed["plan_token"]
    before = 0
    if failure == "busy":
        run("planning.create", title="New meeting", start=proposed["blocks"][1]["start"],
            end=proposed["blocks"][1]["end"], timezone="America/Chicago")
        before = 1
    elif failure == "revision":
        run("task.update", task_id=b["task_id"], expected_revision=1, title="Changed B")
    elif failure == "unknown":
        monkeypatch.setattr(planner, "read_availability", lambda *a: {"status": "unavailable", "reason": "Offline"})
    elif failure == "expired":
        with session_scope() as db:
            stored = db.get(Command, ("davin", "planning-proposal:" + token))
            payload = unseal(stored.result["data"]["sealed_proposal"])
            payload["expires_at"] = (datetime.now(UTC) - timedelta(seconds=1)).isoformat()
            sealed = seal(payload)
            stored.result = {**stored.result, "data": {"sealed_proposal": sealed}}
            stored.request_hash = hashlib.sha256(sealed.encode()).hexdigest()
    with pytest.raises(DomainError):
        run("planning.commit", owner="other" if failure == "owner" else "davin", plan_token=token)
    with session_scope() as db:
        assert db.scalar(select(func.count(PlanningEntry.id))) == before


def test_infeasible_is_not_unknown_or_permission_to_write():
    a = task("Deadline", 90, not_after="2026-10-08T10:00:00-05:00")
    result = planner.propose("davin", request([a]))
    assert result["status"] == "infeasible" and not result.get("plan_token")
    unavailable = planner.propose("davin", request([a], scope="selected_calendars"))
    assert unavailable["status"] == "unavailable" and unavailable["blocks"] == []
    with session_scope() as db:
        assert db.scalar(select(func.count(PlanningEntry.id))) == 0


def test_dependency_cycle_and_duplicate_tasks_rejected():
    a, b = task("A", 10), task("B", 10)
    with pytest.raises(DomainError, match="once"):
        planner.constraints(PlanRequest.model_validate(request([a, a])))
    a["after_task_ids"], b["after_task_ids"] = [b["task_id"]], [a["task_id"]]
    with pytest.raises(DomainError, match="cycle"):
        planner.constraints(PlanRequest.model_validate(request([a, b])))


def test_note_reference_error_identifies_field_and_keeps_original_state():
    goal = run("goal.create", name="Real outcome")
    note = run("note.create", title="Decision", content="Original", goal_ids=[goal["id"]])
    bad = str(uuid4())
    with pytest.raises(DomainError) as error:
        run("note.update", note_id=note["id"], expected_revision=1, content="Changed", goal_ids=[bad])
    assert error.value.code == "INVALID_REFERENCE"
    assert error.value.data["field"] == "goal_ids"
    assert error.value.data["reference_type"] == "goal"
    assert error.value.data["invalid_id"] == bad
    with session_scope() as db:
        assert db.get(Note, note["id"]).content == "Original"
        assert list(db.scalars(select(NoteGoalLink.goal_id))) == [goal["id"]]
    updated = run("note.update", note_id=note["id"], expected_revision=1, content="Only this changes")
    assert updated["content"] == "Only this changes"
    assert [g["id"] for g in updated["goals"]] == [goal["id"]]


def test_foreign_note_reference_same_error_as_missing():
    foreign = run("goal.create", owner="someone-else", name="Private")
    note = run("note.create", title="Local")
    with pytest.raises(DomainError) as error:
        run("note.update", note_id=note["id"], expected_revision=1, goal_ids=[foreign["id"]])
    assert error.value.code == "INVALID_REFERENCE"
    assert "Private" not in error.value.message


def test_short_plan_reference_survives_restart_and_committed_receipt_outlives_proposal():
    proposed = planner.propose("davin", request([task("Compact reference", 30)]))
    ref = proposed["plan_token"]
    assert str(__import__("uuid").UUID(ref)) == ref
    assert len(ref) == 36
    # No in-memory registry: reconstructing the module still resolves the proposal.
    import importlib
    importlib.reload(planner)
    saved = run("planning.commit", plan_token=ref)
    with session_scope() as db:
        stored = db.get(Command, ("davin", "planning-proposal:" + ref))
        stored.created_at = datetime.now(UTC) - timedelta(hours=1)
    planner.propose("davin", request([task("Cleanup trigger", 15)]))
    with session_scope() as db:
        assert db.get(Command, ("davin", "planning-proposal:" + ref)) is None
    assert run("planning.commit", plan_token=ref) == saved


def test_missing_and_corrupted_proposal_never_save_blocks():
    proposed = planner.propose("davin", request([task("Integrity", 30)]))
    with session_scope() as db:
        row = db.get(Command, ("davin", "planning-proposal:" + proposed["plan_token"]))
        row.result = {**row.result, "data": {"sealed_proposal": "corrupted"}}
    for ref in [str(uuid4()), proposed["plan_token"]]:
        with pytest.raises(DomainError) as exc:
            run("planning.commit", plan_token=ref)
        assert exc.value.code == "PLAN_INVALID"
    with session_scope() as db:
        assert db.scalar(select(func.count(PlanningEntry.id))) == 0
