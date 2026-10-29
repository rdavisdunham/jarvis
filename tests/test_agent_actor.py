"""The completion gate: every actor at every entrance while a task has an unsettled run (agent_actor.py)."""

import pytest
from jarvis import agent_actor
from jarvis import linear_sync as sync
from jarvis.access import execution
from jarvis.agent_actor import Actor, acting
from jarvis.bot_access import bind
from jarvis.db import session_scope
from jarvis.domain import DomainError
from jarvis.models import Task
from jarvis.session_models import AgentRun
from sqlalchemy import select
from test_agent_runs import attached, host, quiet, start  # noqa: F401 - pytest fixtures
from test_linear import imported, linear, run  # noqa: F401 - pytest fixture
from test_organization_foundation import change, create

AUTHORITIES = ["human", "source_sync"]
REFUSED = ["eri_work", "run_session", "bot"]


def complete_update(record):
    task = current(record)
    return run("task.update", task_id=task.id, expected_revision=task.revision, status="completed")


def complete_command(record):
    task = current(record)
    return run("task.complete", task_id=task.id, expected_revision=task.revision)


def complete_batch(record):
    task = current(record)
    return run("task.batch", items=[{"task_id": task.id, "expected_revision": task.revision, "status": "completed"}])


def complete_record(record):
    with session_scope() as db:
        from jarvis.structure_models import StructureRecord

        revision = db.get(StructureRecord, record["id"]).revision
    return change({**record, "revision": revision}, status_id="completed")


def complete_schedule(record):
    """Stage F, M3: a one-time reminder on the task completes it through schedule.complete."""
    from datetime import timedelta

    from jarvis.models import now

    with acting(Actor("human")):
        sched = run("schedule.create", kind="reminder", title="ping", task_id=record["task_id"],
                    when=(now() + timedelta(days=1)).isoformat(), timezone="UTC")
    return run("schedule.complete", schedule_id=sched["id"], expected_revision=sched["revision"])


def complete_notification(record):
    """Stage F, M3: a delivered deadline notice completes its task through notification.complete."""
    from jarvis.models import Notification, now

    with session_scope() as db:
        notice = Notification(owner_id="davin", dedup_key="deadline:" + record["task_id"], category="deadline",
                              title="Due", body="", task_id=record["task_id"], scheduled_at=now(), eligible_at=now())
        db.add(notice)
        db.flush()
        notice_id = notice.id
    return run("notification.complete", notification_id=notice_id)


ENTRANCES = {"task.update": complete_update, "task.complete": complete_command, "task.batch": complete_batch,
             "record.update": complete_record, "schedule.complete": complete_schedule,
             "notification.complete": complete_notification}


def current(record):
    with session_scope() as db:
        task = db.get(Task, record["task_id"])
        db.expunge(task)
        return task


def running(host, title="Gated task"):  # noqa: F811
    record = create("task", title)
    run_id = start(host, record["task_id"])["id"]
    attached(host, run_id, "sess-" + title.replace(" ", "-"))
    return record, run_id


def run_row(run_id):
    with session_scope() as db:
        row = db.get(AgentRun, run_id)
        db.expunge(row)
        return row


@pytest.mark.parametrize("entrance", ENTRANCES)
@pytest.mark.parametrize("kind", AUTHORITIES + REFUSED)
def test_completion_gate_matrix(host, kind, entrance):  # noqa: F811
    record, run_id = running(host)
    with acting(Actor(kind)):
        if kind in REFUSED:
            with pytest.raises(DomainError) as refused:
                ENTRANCES[entrance](record)
            assert refused.value.code == "RUN_OWNS_COMPLETION" and refused.value.status == 409
            assert refused.value.data["run_id"] == run_id
        else:
            ENTRANCES[entrance](record)
    task, row = current(record), run_row(run_id)
    if kind in REFUSED:
        assert task.status != "completed" and row.state == "executing"
    else:
        # The completion is recorded at once; the run is asked to stop, never asserted stopped (F03).
        assert task.status == "completed"
        assert (row.state, row.stop_reason, row.stop_final) == ("stopping", "task_completed_externally", "stopped")
        assert row.sync_state["completion"]["actor"]["kind"] == kind
        stopping = [e for e in row.history if e.get("to") == "stopping"]
        assert stopping[-1]["actor"]["kind"] == kind


def test_linear_import_is_source_sync_whatever_the_ambient_actor(host, linear):  # noqa: F811
    tid, _ = imported()
    run_id = start(host, tid)["id"]
    attached(host, run_id)
    linear.rows["remote"]["state"] = {"id": "done", "name": "Done", "type": "completed"}
    linear.rows["remote"]["updatedAt"] = "2099-01-01T00:00:00+00:00"
    with acting(Actor("run_session")):
        with session_scope() as db:
            job = sync.queue_sync(db, "davin", force=True)
        sync.process_sync(job)
    with session_scope() as db:
        assert db.get(Task, tid).status == "completed"
        row = db.get(AgentRun, run_id)
        assert (row.state, row.stop_reason) == ("stopping", "task_completed_externally")
        assert row.sync_state["completion"]["actor"]["kind"] == "source_sync"


@pytest.mark.parametrize("field, value", [("success_criteria", [{"text": "Docs updated"}]), ("turn_cap", 3),
                                          ("verify", {"target_id": "jarvis", "verify_id": "ci"})])
def test_gate_fields_are_reserved_during_a_run(host, field, value):  # noqa: F811
    record, run_id = running(host)
    task = current(record)
    for kind in REFUSED:
        with acting(Actor(kind)), pytest.raises(DomainError) as refused:
            run("task.update", task_id=task.id, expected_revision=task.revision, **{field: value})
        assert refused.value.code == "RUN_OWNS_COMPLETION"
    run("task.update", task_id=task.id, expected_revision=task.revision, **{field: value})  # human
    assert run_row(run_id).state == "executing"  # A human editing gate fields does not stop the run.
    with acting(Actor("run_session")):  # Ordinary fields stay open to agents.
        task = current(record)
        run("task.update", task_id=task.id, expected_revision=task.revision, notes="Progress notes")


def test_without_an_unsettled_run_agents_may_complete(host):  # noqa: F811
    record = create("task", "Free task")
    with acting(Actor("run_session")):
        complete_command(record)
    assert current(record).status == "completed"


def test_request_context_decides_the_default_actor():
    assert agent_actor.current().kind == "human"
    token = execution.set({"id": "w", "credential_id": None})
    try:
        assert agent_actor.current().kind == "eri_work"
    finally:
        execution.reset(token)
    with bind("cred-1"):
        actor = agent_actor.current()
        assert (actor.kind, actor.credential_id, actor.verified) == ("bot", "cred-1", False)
        with acting(Actor("run_session", run_id="r1")):
            assert agent_actor.current().kind == "run_session"


def test_run_sessions_complete_through_a_run_never(host):  # noqa: F811
    """A settled run no longer gates; an unsettled one blocks even a verified run session."""
    record, run_id = running(host)
    with session_scope() as db:
        row = db.get(AgentRun, run_id)
        actor = Actor("run_session", run_id=row.id, generation=row.generation)
    with acting(actor), pytest.raises(DomainError):
        complete_update(record)
    with session_scope() as db:
        assert db.scalar(select(AgentRun.state).where(AgentRun.id == run_id)) == "executing"
