import pytest
from jarvis.db import session_scope
from jarvis.domain import DomainError
from jarvis.models import Job, Task
from sqlalchemy import func, select
from test_linear import imported, linear, run  # noqa: F401 - pytest fixture


def plan(**values):
    task = run("task.create", title="Ship the agent link")
    return task, run("task.update", task_id=task["id"], expected_revision=task["revision"], **values)


def test_new_columns_serialize_with_defaults():
    task = run("task.create", title="Plain task")
    assert (task["steps"], task["success_criteria"], task["verify"], task["turn_cap"], task["plan_revision"]) == ([], [], None, 15, 0)
    assert task["prd_note_id"] is None and task["execution_target_id"] is None


def test_task_update_carries_plan_fields_with_stable_ids():
    note = run("note.create", title="PRD", content="Build the link.")
    child = run("task.create", title="Child")
    task, updated = plan(
        steps=[{"title": "Migrations"}, {"id": "tests", "title": "Tests", "task_id": child["id"]}],
        success_criteria=[{"text": "CI is green", "kind": "verify", "verify_id": "ci"}, {"text": "Docs updated"}],
        verify={"target_id": "jarvis", "verify_id": "ci"}, turn_cap=20, prd_note_id=note["id"],
    )
    assert updated["plan_revision"] == 1 and updated["turn_cap"] == 20 and updated["prd_note_id"] == note["id"]
    first, second = updated["steps"]
    assert first["id"].startswith("s_") and second["id"] == "tests" and second["task_id"] == child["id"]
    assert first["status"] == "todo" and first["by"] == "user" and first["revision"] == 1
    assert [c["id"][:2] for c in updated["success_criteria"]] == ["c_", "c_"]
    assert updated["verify"] == {"target_id": "jarvis", "verify_id": "ci"}
    # Editing one step keeps the other step's stamp; an unchanged plan does not advance the revision.
    again = run("task.update", task_id=task["id"], expected_revision=updated["revision"],
                steps=[first, {**second, "status": "done"}])
    assert again["steps"][0] == first and again["steps"][1]["status"] == "done" and again["steps"][1]["revision"] == 2
    assert again["plan_revision"] == 2
    same = run("task.update", task_id=task["id"], expected_revision=again["revision"], turn_cap=20, title="Renamed")
    assert same["plan_revision"] == 2


@pytest.mark.parametrize("values, message", [
    ({"steps": [{"id": "a", "title": "One"}, {"id": "a", "title": "Two"}]}, "own id"),
    ({"success_criteria": [{"text": "Tests pass", "kind": "verify"}]}, "verify_id"),
    ({"steps": None}, "cannot be cleared"),
    ({"prd_note_id": "00000000-0000-0000-0000-000000000000"}, ""),
])
def test_plan_fields_are_validated(values, message):
    with pytest.raises(DomainError, match=message):
        plan(**values)


def test_plan_edits_never_queue_a_linear_write(linear):  # noqa: F811
    tid, rev = imported()
    with session_scope() as db:
        before = db.scalar(select(func.count()).select_from(Job).where(Job.kind.like("linear%")))
    result = run("task.update", task_id=tid, expected_revision=rev, steps=[{"title": "Step"}],
                 success_criteria=[{"text": "Works"}], turn_cap=5)
    assert result["external"].get("sync_state") != "pending"
    with session_scope() as db:
        assert db.scalar(select(func.count()).select_from(Job).where(Job.kind.like("linear%"))) == before
        assert db.get(Task, tid).plan_revision == 1
    assert linear.writes == []
