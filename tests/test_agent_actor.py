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


# ---------------------------------------------------------------- the PRD note (agent_actor.note_gate)


def note_row(note_id):
    from jarvis.models import Note

    with session_scope() as db:
        row = db.get(Note, note_id)
        db.expunge(row)
        return row


def prd_running(host, title="PRD task"):  # noqa: F811
    """A task whose PRD note is frozen with a live run: (note_id, run_id)."""
    note = run("note.create", title="PRD: " + title, content="Plan v1", tags=["prd"])
    record = create("task", title)
    task = current(record)
    run("task.update", task_id=task.id, expected_revision=task.revision, prd_note_id=note["id"])
    run_id = start(host, record["task_id"])["id"]
    attached(host, run_id, "sess-prd-" + title.replace(" ", "-"))
    return note["id"], run_id


def edit_update(note):
    return run("note.update", note_id=note.id, expected_revision=note.revision, content="Rewritten plan")


def edit_append(note):
    return run("note.append", note_id=note.id, expected_revision=note.revision, text="\nAnother step")


def edit_replace(note):
    return run("note.replace", note_id=note.id, expected_revision=note.revision, old_text="v1", new_text="v2")


def edit_archive(note):
    return run("note.update", note_id=note.id, expected_revision=note.revision, archived=True)


def edit_organize(note):
    return run("note.organize", note_id=note.id, expected_revision=note.revision)


NOTE_EDITS = {"note.update": edit_update, "note.append": edit_append, "note.replace": edit_replace,
              "archive": edit_archive, "note.organize": edit_organize}


@pytest.mark.parametrize("edit", NOTE_EDITS)
@pytest.mark.parametrize("kind", REFUSED + ["run_judge", "source_sync"])
def test_a_live_prd_note_is_read_only_to_agents(host, kind, edit):  # noqa: F811
    note_id, run_id = prd_running(host)
    before = note_row(note_id)
    with acting(Actor(kind)), pytest.raises(DomainError) as refused:
        NOTE_EDITS[edit](before)
    assert (refused.value.code, refused.value.status) == ("RUN_OWNS_PRD", 409)
    assert refused.value.data["run_id"] == run_id and run_id in refused.value.message
    after = note_row(note_id)
    assert (after.content, after.revision, after.archived) == ("Plan v1", before.revision, False)


def test_a_human_may_edit_a_live_prd_and_is_told_the_plan_is_frozen(host):  # noqa: F811
    from jarvis.models import Note
    from jarvis.notes import note_data
    from jarvis.session_models import AgentRunPlan

    note_id, run_id = prd_running(host)
    with acting(Actor("human")):
        edited = edit_append(note_row(note_id))
    assert edited["content"] == "Plan v1\nAnother step"
    assert run_id in edited["warning"] and "not the plan" in edited["warning"]
    assert edited["frozen_run_id"] == run_id
    assert run_row(run_id).state == "executing"  # A human PRD edit does not touch the run.
    with session_scope() as db:
        assert note_data(db, db.get(Note, note_id))["frozen_run_id"] == run_id
        assert db.scalar(select(AgentRunPlan.prd_text).where(AgentRunPlan.run_id == run_id)) == "Plan v1"


def test_a_settled_run_releases_its_prd(host):  # noqa: F811
    note_id, run_id = prd_running(host)
    with session_scope() as db:
        db.get(AgentRun, run_id).state = "stopped"
    with acting(Actor("bot")):
        edited = edit_update(note_row(note_id))
    assert edited["content"] == "Rewritten plan" and edited["frozen_run_id"] is None and "warning" not in edited


def test_notes_that_are_not_a_live_prd_stay_open_to_agents(host):  # noqa: F811
    prd_running(host)
    other = run("note.create", title="Scratch", content="Plan v1")
    with acting(Actor("bot")):
        edited = edit_replace(note_row(other["id"]))
    assert edited["content"] == "Plan v2" and edited["frozen_run_id"] is None and "warning" not in edited


def test_only_its_own_runs_plan_write_may_rewrite_the_prd(host):  # noqa: F811
    note_id, run_id = prd_running(host)
    with acting(Actor("run_session", run_id="another-run", capability="plan_write")), \
            pytest.raises(DomainError) as other:
        edit_update(note_row(note_id))
    assert other.value.code == "RUN_OWNS_PRD"
    with acting(Actor("run_session", run_id=run_id, capability="plan_write")):  # submit_plan's writer
        assert edit_update(note_row(note_id))["content"] == "Rewritten plan"


def test_note_tasks_still_links_to_dos_from_a_live_prd(host):  # noqa: F811
    note_id, _ = prd_running(host)
    note = note_row(note_id)
    with acting(Actor("bot")):
        linked = run("note.tasks", note_id=note.id, expected_revision=note.revision,
                     items=[{"title": "Do v1", "evidence": "Plan v1"}])
    assert linked["created_count"] == 1 and note_row(note_id).content == "Plan v1"
