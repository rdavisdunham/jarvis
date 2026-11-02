"""Elapsed archive policy, lifecycle recovery, daily queries and worker authority."""
from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest
from sqlalchemy import select, text

from jarvis.db import session_scope
from jarvis.domain import execute
from jarvis.models import Event, OwnerSettings, Task, now
from jarvis.task_archive import GRACE, _cursors, sweep_owner
from test_agent_runs import host  # noqa: F401 - real isolated host fixture


@pytest.fixture(autouse=True)
def reset_cursor():
    _cursors.clear()


def command(tool, owner="davin", **arguments):
    with session_scope() as db:
        return execute(db, owner, str(uuid4()), tool, arguments)["data"]


def finished(status="completed", **values):
    row = command("task.create", title="Finished", **values)
    return command("task.update", task_id=row["id"], expected_revision=row["revision"], status=status)


def age(task_id, instant):
    with session_scope() as db:
        row = db.get(Task, task_id)
        row.finished_at, row.archive_eligible_at = instant, instant + GRACE
        if row.status == "completed":
            row.completed_at = instant


@pytest.mark.parametrize("status", ["completed", "cancelled"])
def test_elapsed_boundary_events_and_idempotency(status):
    row = finished(status)
    instant = datetime(2026, 11, 1, 6, tzinfo=UTC)  # DST changes inside these elapsed hours.
    age(row["id"], instant)
    with session_scope() as db:
        assert sweep_owner(db, "davin", clock=instant + GRACE - timedelta(seconds=1)) == 0
        assert not db.get(Task, row["id"]).archived
    with session_scope() as db:
        assert sweep_owner(db, "other", clock=instant + GRACE) == 0
        assert sweep_owner(db, "davin", clock=instant + GRACE) == 1
        task = db.get(Task, row["id"])
        assert task.status == status and task.finished_at == instant
        assert task.archive_reason == "automatic" and task.archived_at
        from jarvis.structure_models import StructureRecord
        record = db.scalar(select(StructureRecord).where(StructureRecord.task_id == task.id))
        assert record.archived
        assert db.scalar(select(Event).where(Event.owner_id == "davin", Event.kind == "record.changed"))
    with session_scope() as db:
        assert sweep_owner(db, "davin", clock=instant + GRACE * 3) == 0


def test_ordinary_edits_same_status_restore_and_reopen_clocks():
    row = finished("cancelled")
    first = row["finished_at"]
    edited = command("task.update", task_id=row["id"], expected_revision=row["revision"], title="Edited", status="cancelled")
    assert edited["finished_at"] == first and edited["archive_eligible_at"] == row["archive_eligible_at"]
    archived = command("task.update", task_id=row["id"], expected_revision=edited["revision"], archived=True)
    assert archived["archive_reason"] == "manual"
    restored = command("task.update", task_id=row["id"], expected_revision=archived["revision"], archived=False)
    assert restored["finished_at"] == first and restored["archive_eligible_at"] > row["archive_eligible_at"]
    assert restored["archive_reason"] is None
    reopened = command("task.reopen", task_id=row["id"], expected_revision=restored["revision"])
    assert reopened["finished_at"] is None and reopened["archive_eligible_at"] is None and not reopened["archived"]
    again = command("task.complete", task_id=row["id"], expected_revision=reopened["revision"])
    assert again["finished_at"] > first


def test_legacy_cancel_gets_grace_without_false_finish_date():
    row = finished("cancelled")
    instant = now()
    with session_scope() as db:
        # Simulate an old writer whose cancellation has no reliable transition timestamp.
        db.execute(text("UPDATE tasks SET finished_at=NULL, archive_eligible_at=NULL WHERE id=:id"), {"id": row["id"]})
    with session_scope() as db:
        assert sweep_owner(db, "davin", clock=instant) == 0
        task = db.get(Task, row["id"])
        assert task.finished_at is None and task.archive_eligible_at == instant + GRACE
    with session_scope() as db:
        assert sweep_owner(db, "davin", clock=instant + GRACE) == 1


def test_unfinished_descendants_templates_policy_and_rotation():
    parent = finished()
    child = command("task.create", title="Unfinished", parent_task_id=parent["id"])
    other = finished()
    old = now() - timedelta(days=3)
    for row in (parent, other):
        age(row["id"], old)
    # Force the blocked parent ahead of the eligible task in the bounded scan.
    with session_scope() as db:
        first = min(parent["id"], other["id"])
        if first != parent["id"]:
            # Keyset has no dependency on age/order of creation; seed before parent for a deterministic first page.
            _cursors["davin"] = other["id"]
        assert sweep_owner(db, "davin", limit=1) == 0
        assert not db.get(Task, parent["id"]).archived
    with session_scope() as db:
        # One wrap at most before the unblocked task is reached.
        archived = sweep_owner(db, "davin", limit=1)
    if not archived:
        with session_scope() as db:
            archived = sweep_owner(db, "davin", limit=1)
    assert archived == 1
    with session_scope() as db:
        settings = OwnerSettings(owner_id="davin", values={"auto_archive_finished_tasks": False})
        db.add(settings)
    fresh = finished("cancelled")
    age(fresh["id"], old)
    with session_scope() as db:
        assert sweep_owner(db, "davin") == 0
        assert not db.get(Task, fresh["id"]).archived


def test_daily_query_zone_pagination_archives_and_unknown_cancel(client):
    instant = datetime(2026, 11, 2, 5, 30, tzinfo=UTC)  # Nov 1 in Chicago, Nov 2 UTC.
    rows = [finished(status) for status in ("completed", "cancelled", "completed")]
    for row in rows:
        age(row["id"], instant)
    with session_scope() as db:
        assert sweep_owner(db, "davin", clock=instant + GRACE) == 3
    # Manual filing is excluded from the daily checklist.
    last = command("task.update", task_id=rows[2]["id"], expected_revision=rows[2]["revision"] + 1, archived=False)
    command("task.update", task_id=last["id"], expected_revision=last["revision"], archived=True)
    first = client.get("/api/v1/tasks", params={"finished_on": "2026-11-01", "timezone": "America/Chicago", "limit": 1}).json()
    second = client.get("/api/v1/tasks", params={"finished_on": "2026-11-01", "timezone": "America/Chicago", "limit": 1, "before": first["next_cursor"]}).json()
    assert len(first["items"]) == len(second["items"]) == 1
    assert {first["items"][0]["id"], second["items"][0]["id"]} == {r["id"] for r in rows[:2]}
    assert client.get("/api/v1/tasks", params={"finished_on": "2026-11-01", "timezone": "UTC"}).json()["items"] == []
    assert client.get("/api/v1/tasks", params={"finished_on": "2026-11-01", "timezone": "bad/zone"}).status_code == 400
    assert client.get("/api/v1/tasks", params={"finished_on": "2026-11-01", "include_archived": True}).status_code == 400
    assert client.get("/api/v1/tasks", params={"finished_on": "bad"}).status_code == 422


@pytest.mark.parametrize("state", ["working", "awaiting_approval", "needs_you", "completed"])
def test_unsettled_run_and_terminal_pending_input_are_skipped(state, host):  # noqa: F811
    from jarvis.session_models import AgentQuestion, AgentRun, AgentSession
    row = finished()
    age(row["id"], now() - timedelta(days=2))
    with session_scope() as db:
        run = AgentRun(owner_id="davin", task_id=row["id"], host_id=host.id, state=state)
        db.add(run)
        db.flush()
        if state == "completed":
            session = AgentSession(owner_id="davin", tool="codex", session_id="synthetic", host_id=host.id)
            db.add(session)
            db.flush()
            db.add(AgentQuestion(owner_id="davin", run_id=run.id, session_pk=session.id, host_id=host.id, question_id="q", provider_request_id="p", status="open"))
    with session_scope() as db:
        assert sweep_owner(db, "davin") == 0
        assert not db.get(Task, row["id"]).archived


def test_shared_owner_policy_authority_and_internal_sweep(client):
    from test_accounts import client_for, shared
    from jarvis.models import WorkspaceMember
    owner = client
    guest = client_for("guest")
    workspace = shared(owner, guest)
    namespace = workspace["id"]
    def policy(c, arguments):
        return c.post("/api/v1/commands", json={"command_id": str(uuid4()), "tool": "settings.update", "arguments": arguments})
    assert policy(owner, {"auto_archive_finished_tasks": False}).status_code == 200
    assert policy(guest, {"auto_archive_finished_tasks": True}).status_code == 403
    assert policy(owner, {"auto_archive_finished_tasks": True, "quiet_enabled": False}).status_code == 403
    assert policy(owner, {"auto_archive_finished_tasks": True}).status_code == 200
    # Worker has no request principal: the validated shared policy applies in that namespace.
    with session_scope() as db:
        task = Task(owner_id=namespace, title="Shared finished", status="completed", finished_at=now() - GRACE * 2, archive_eligible_at=now() - GRACE)
        db.add(task)
        db.flush()
        tid = task.id
    with session_scope() as db:
        assert sweep_owner(db, namespace) == 1
        assert db.get(Task, tid).archive_reason == "automatic"
        assert not list(db.scalars(select(Task).where(Task.owner_id == "davin")))
    with session_scope() as db:
        db.get(WorkspaceMember, (namespace, "guest")).role = "viewer"
    assert policy(guest, {"auto_archive_finished_tasks": False}).status_code == 403


def test_organizational_types_survive_but_done_item_parent_archives():
    from test_organization_foundation import apply, change, create, schema
    project = create("project", "Completed project")
    change(project, status_id="completed")
    definition = schema()
    task_type = next(t for t in definition["types"] if t["id"] == "task")
    task_type["opens_as"] = "container"
    apply(definition)
    container = create("task", "Explicit container")
    change(container, status_id="completed")
    for task_id in (project["task_id"], container["task_id"]):
        age(task_id, now() - GRACE * 2)
    with session_scope() as db:
        assert sweep_owner(db, "davin") == 0
        assert not db.get(Task, project["task_id"]).archived
        assert not db.get(Task, container["task_id"]).archived
    definition = schema()
    next(t for t in definition["types"] if t["id"] == "task")["opens_as"] = "item"
    apply(definition)
    child = command("task.create", title="Done child", parent_task_id=container["task_id"])
    command("task.complete", task_id=child["id"], expected_revision=child["revision"])
    with session_scope() as db:
        assert sweep_owner(db, "davin") == 1
        assert db.get(Task, container["task_id"]).archived
        assert not db.get(Task, project["task_id"]).archived


def test_quick_checklist_history_survives_automatic_item_archive():
    from jarvis.quick_lists import read
    row = command("quicklist.create", title="Trip", items=[{"title": "Pack"}])
    item = row["items"][0]
    command("task.complete", task_id=item["id"], expected_revision=item["revision"])
    age(item["id"], now() - GRACE * 2)
    with session_scope() as db:
        assert sweep_owner(db, "davin") == 1
        checklist = read(db, "davin", row["id"])
        assert checklist["done"] == checklist["total"] == 1
        assert checklist["quick_done"] == checklist["quick_total"] == 1
        assert checklist["items"][0]["archived"]


def test_reopened_task_and_competing_sweeps_are_safe():
    from concurrent.futures import ThreadPoolExecutor
    row = finished()
    age(row["id"], now() - GRACE * 2)
    def sweep_once():
        with session_scope() as db:
            return sweep_owner(db, "davin")
    with ThreadPoolExecutor(max_workers=2) as pool:
        assert sorted(pool.map(lambda _: sweep_once(), range(2))) == [0, 1]
    with session_scope() as db:
        rev = db.get(Task, row["id"]).revision
    reopened = command("task.reopen", task_id=row["id"], expected_revision=rev)
    assert not reopened["archived"]
    with session_scope() as db:
        assert sweep_owner(db, "davin", clock=now() + GRACE * 3) == 0


def test_legacy_cancel_uses_recorded_transition_and_excludes_templates(monkeypatch):
    from cryptography.fernet import Fernet
    from jarvis.config import get_settings
    from jarvis.models import ActionChange
    from jarvis.work_crypto import seal
    row = finished("cancelled")
    instant = now() - GRACE * 2
    monkeypatch.setenv("JARVIS_INTEGRATION_ENCRYPTION_KEY", Fernet.generate_key().decode())
    get_settings.cache_clear()
    try:
        with session_scope() as db:
            db.execute(text("UPDATE tasks SET finished_at=NULL, archive_eligible_at=NULL WHERE id=:id"), {"id": row["id"]})
            db.add(ActionChange(owner_id="davin", account_id="davin", command_id="old-transition", tool="task.update", entity_kind="task", entity_id=row["id"], before_ciphertext=seal({"status": "open"}), after_ciphertext=seal({"status": "cancelled"}), created_at=instant))
            template = Task(owner_id="davin", title="Recurring template", status="completed", is_template=True, finished_at=instant, archive_eligible_at=instant + GRACE)
            db.add(template)
            db.flush()
            template_id = template.id
        with session_scope() as db:
            assert sweep_owner(db, "davin") == 1
            assert db.get(Task, row["id"]).finished_at == instant
            assert not db.get(Task, template_id).archived
    finally:
        get_settings.cache_clear()


@pytest.mark.parametrize("failure", ["candidate", "mutation"])
def test_bad_candidate_rolls_back_its_changes_without_blocking_later_work(monkeypatch, failure):
    from jarvis import domain, task_archive
    first, second = finished(), finished("cancelled")
    for row in (first, second):
        age(row["id"], now() - GRACE * 2)
    broken_id = min(first["id"], second["id"])
    good_id = max(first["id"], second["id"])
    if failure == "candidate":
        original = task_archive.organizational_container
        def fail_candidate(db, task):
            if task.id == broken_id:
                raise ValueError("synthetic malformed candidate")
            return original(db, task)
        monkeypatch.setattr(task_archive, "organizational_container", fail_candidate)
    else:
        original = domain.mutate
        def fail_mutation(db, owner, tool, args, command_id):
            result = original(db, owner, tool, args, command_id)
            if getattr(args, "task_id", None) == broken_id:
                db.flush()  # Prove even already-flushed task/alert/events roll back atomically.
                raise ValueError("synthetic post-mutation failure")
            return result
        monkeypatch.setattr(domain, "mutate", fail_mutation)
    with session_scope() as db:
        assert sweep_owner(db, "davin") == 1
    with session_scope() as db:
        assert db.get(Task, good_id).archived
        broken = db.get(Task, broken_id)
        assert not broken.archived and broken.archive_reason is None and broken.archived_at is None
        assert not db.info.get("automatic_task_archive")
    # A short (<200) page with the failing first item never prevents healthy later work.
    later = finished()
    age(later["id"], now() - GRACE * 2)
    with session_scope() as db:
        assert sweep_owner(db, "davin") == 1
        assert db.get(Task, later["id"]).archived


def test_owner_failure_keeps_other_owners_committed(monkeypatch):
    from jarvis import task_archive
    original = task_archive.sweep_owner
    instant = now() - GRACE * 2
    with session_scope() as db:
        for owner in ("a-broken", "b-good"):
            db.add(Task(id=owner, owner_id=owner, title=owner, status="completed", finished_at=instant, archive_eligible_at=instant + GRACE))
    def fail_owner(db, owner):
        if owner == "a-broken":
            db.get(Task, owner).archived = True
            db.flush()
            raise ValueError("synthetic owner failure after write")
        return original(db, owner)
    monkeypatch.setattr(task_archive, "sweep_owner", fail_owner)
    with session_scope() as db:
        assert task_archive.sweep(db) == 1
    with session_scope() as db:
        assert not db.get(Task, "a-broken").archived
        assert db.get(Task, "b-good").archived
