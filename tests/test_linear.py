import copy
from uuid import uuid4

import pytest
from cryptography.fernet import Fernet
from jarvis import linear_commands as commands
from jarvis import linear_sync as sync
from jarvis.config import get_settings
from jarvis.db import session_scope
from jarvis.domain import DomainError, execute
from jarvis.linear_client import LinearClient, LinearFailure
from jarvis.models import Job, LinearConnection, LinearIssue, Task
from sqlalchemy import func, select


def run(tool, **args):
    with session_scope() as db:
        return execute(db, "davin", str(uuid4()), tool, args)["data"]


def issue(id="remote"):
    return {
        "id": id,
        "identifier": "ERI-1",
        "url": "https://linear.app/test/issue/ERI-1",
        "title": "Shared task",
        "description": "Notes",
        "dueDate": "2026-09-20",
        "priority": 2,
        "updatedAt": "2026-09-12T10:00:00Z",
        "archivedAt": None,
        "team": {"id": "team", "name": "Team", "key": "ERI"},
        "state": {"id": "todo", "name": "Todo", "type": "unstarted"},
        "assignee": {"id": "me", "name": "Owner"},
        "project": None,
        "parent": None,
        "labels": {"nodes": [], "pageInfo": {"hasNextPage": False}},
    }


class Fake:
    def __init__(self):
        self.rows = {"remote": issue()}
        self.writes = []
        self.lost = False
        self.fail_sync = False
        self.filters = []

    def close(self):
        pass

    def identity(self):
        return {
            "viewer": {"id": "me", "name": "Owner"},
            "organization": {"id": "ws", "name": "Workspace"},
            "teams": [{"id": "team", "name": "Team"}],
        }

    def directory(self, teams):
        return {
            "states": [
                {"id": "todo", "name": "Todo", "type": "unstarted", "team": {"id": "team"}},
                {"id": "done", "name": "Done", "type": "completed", "team": {"id": "team"}},
            ],
            "users": [{"id": "me", "name": "Owner", "active": True}],
        }

    def issues(self, filter):
        self.filters.append(filter)
        if self.fail_sync:
            raise LinearFailure("unavailable")
        return copy.deepcopy(list(self.rows.values()))

    def issue(self, id):
        return copy.deepcopy(self.rows.get(id))

    def query(self, query, variables):
        create = "issueCreate" in query
        data = variables["input"]
        id = data["id"] if create else variables["id"]
        self.writes.append((id, copy.deepcopy(data)))
        remote = copy.deepcopy(self.rows.get(id) or issue(id))
        for key, value in data.items():
            if key in {"id", "teamId"}:
                continue
            if key in {"stateId", "assigneeId", "projectId"}:
                field = {"stateId": "state", "assigneeId": "assignee", "projectId": "project"}[key]
                remote[field] = (
                    (
                        {
                            "id": value,
                            "name": "Done" if value == "done" else "Todo",
                            "type": "completed" if value == "done" else "unstarted",
                        }
                        if field == "state"
                        else {"id": value, "name": "Owner"}
                    )
                    if value
                    else None
                )
            else:
                remote[key] = value
        remote["updatedAt"] = "2026-09-12T11:00:00Z"
        self.rows[id] = remote
        if self.lost:
            self.lost = False
            raise LinearFailure("unavailable", retry=True, unknown=True)
        return {"issueCreate" if create else "issueUpdate": {"success": True, "issue": copy.deepcopy(remote)}}


@pytest.fixture
def linear(monkeypatch):
    monkeypatch.setattr(get_settings(), "integration_encryption_key", Fernet.generate_key().decode())
    fake = Fake()
    monkeypatch.setattr(sync, "LinearClient", lambda *_: fake)
    monkeypatch.setattr(commands, "LinearClient", lambda *_: fake)
    conn = sync.connect("davin", "fixture-key-never-returned")
    selection = run("linear.select", expected_revision=conn["revision"], team_ids=["team"])
    sync.process_sync(selection["job_id"])
    return fake


def imported():
    with session_scope() as db:
        link = db.scalar(select(LinearIssue))
        return link.task_id, db.get(Task, link.task_id).revision


def test_import_is_idempotent_and_credentials_are_not_exposed(linear, client):
    tid, rev = imported()
    with session_scope() as db:
        jid = sync.queue_sync(db, "davin", force=True)
    sync.process_sync(jid)
    with session_scope() as db:
        assert db.scalar(select(func.count(Task.id))) == 1
        assert db.get(Task, tid).revision == rev
        assert db.get(LinearConnection, "davin").credentials != "fixture-key-never-returned"
    result = client.get("/api/v1/integrations/linear")
    assert "fixture-key" not in result.text and "credentials" not in result.text


def test_lost_create_response_reconciles_original_uuid(linear):
    task = run("task.create", title="New issue")
    queued = run("linear.publish", task_id=task["id"], expected_revision=1, team_id="team")
    linear.lost = True
    with pytest.raises(RuntimeError):
        sync.process_write(queued["job_id"])
    sync.process_write(queued["job_id"])
    assert len(linear.writes) == 1
    with session_scope() as db:
        assert db.get(Job, queued["job_id"]).status == "succeeded"
        assert db.get(Task, task["id"]).external["sync_state"] == "synced"


def test_task_edit_waits_for_remote_and_preserves_local_annotations(linear):
    tid, rev = imported()
    result = run(
        "task.update",
        task_id=tid,
        expected_revision=rev,
        title="Edited",
        tags=["local"],
        due_time="14:00",
        due_timezone="America/Chicago",
    )
    assert result["external"]["sync_state"] == "pending"
    jid = result["external"]["job_id"]
    with pytest.raises(DomainError, match="Wait"):
        run("task.update", task_id=tid, expected_revision=result["revision"], title="Another edit")
    sync.process_write(jid)
    assert linear.writes[0][1] == {"title": "Edited"}
    with session_scope() as db:
        task = db.get(Task, tid)
        assert task.tags == ["local"] and task.due_time == "14:00"


def test_concurrent_remote_change_requires_review(linear):
    tid, rev = imported()
    result = run("task.update", task_id=tid, expected_revision=rev, title="Local choice")
    linear.rows["remote"]["title"] = "Remote choice"
    linear.rows["remote"]["updatedAt"] = "2026-09-12T10:30:00Z"
    sync.process_write(result["external"]["job_id"])
    assert linear.writes == []
    comparison = commands.comparison("davin", tid)
    assert comparison["local"]["title"] == "Local choice"
    saved = run(
        "linear.resolve",
        task_id=tid,
        expected_revision=comparison["local"]["revision"],
        choice="linear",
        edit_token=comparison["edit_token"],
    )
    assert saved["title"] == "Remote choice"
    with pytest.raises(DomainError):
        run(
            "linear.resolve",
            task_id=tid,
            expected_revision=saved["revision"],
            choice="eridani",
            edit_token=comparison["edit_token"],
        )


def test_missing_remote_preserves_task_and_failed_sync_preserves_snapshot(linear):
    tid, _ = imported()
    linear.rows = {}
    with session_scope() as db:
        db.get(LinearConnection, "davin").full_sync_at = None
        jid = sync.queue_sync(db, "davin", force=True)
    sync.process_sync(jid)
    with session_scope() as db:
        task = db.get(Task, tid)
        assert task.title == "Shared task" and not task.archived
        assert task.external["sync_state"] == "unavailable"
    linear.fail_sync = True
    with session_scope() as db:
        jid = sync.queue_sync(db, "davin", force=True)
    sync.process_sync(jid)
    with session_scope() as db:
        assert db.get(Task, tid).title == "Shared task"


def test_disconnect_cancels_unattempted_write_and_rejects_other_owner(linear, client):
    tid, rev = imported()
    saved = run("task.update", task_id=tid, expected_revision=rev, title="Queued")
    assert client.post("/api/v1/integrations/linear/disconnect").status_code == 200
    sync.process_write(saved["external"]["job_id"])
    assert linear.writes == []
    with session_scope() as db:
        assert db.get(Job, saved["external"]["job_id"]).status == "cancelled"
        with pytest.raises(DomainError):
            execute(
                db,
                "someone-else",
                str(uuid4()),
                "task.update",
                {"task_id": tid, "expected_revision": saved["revision"], "title": "wrong owner"},
            )


def test_pagination_rejects_broken_cursor_and_partial_graphql_data():
    client = LinearClient("fixture")
    client.query = lambda *a: {
        "issues": {"nodes": [issue()], "pageInfo": {"hasNextPage": True, "endCursor": "same"}}
    }
    with pytest.raises(LinearFailure, match="range_too_large"):
        client.pages("query", "issues")
    client.close()


def test_keep_owner_change_rechecks_then_writes(linear):
    tid, rev = imported()
    saved = run(
        "task.update", task_id=tid, expected_revision=rev, title="Keep my title", assignee="unassigned"
    )
    linear.rows["remote"]["title"] = "Their title"
    linear.rows["remote"]["updatedAt"] = "2026-09-12T10:30:00Z"
    sync.process_write(saved["external"]["job_id"])
    compare = commands.comparison("davin", tid)
    assert compare["pending_change"]["title"] == "Keep my title"
    queued = run(
        "linear.resolve",
        task_id=tid,
        expected_revision=compare["local"]["revision"],
        choice="eridani",
        edit_token=compare["edit_token"],
    )
    sync.process_write(queued["job_id"])
    assert linear.rows["remote"]["title"] == "Keep my title"
    assert linear.rows["remote"]["assignee"] is None


def test_local_archive_survives_remote_sync(linear):
    tid, rev = imported()
    run("task.update", task_id=tid, expected_revision=rev, archived=True)
    with session_scope() as db:
        jid = sync.queue_sync(db, "davin", force=True)
    sync.process_sync(jid)
    with session_scope() as db:
        assert db.get(Task, tid).archived
    assert linear.writes == []


def test_completing_alert_queues_linked_linear_completion(linear):
    tid, _ = imported()
    alert = run(
        "schedule.create",
        title="Finish report",
        task_id=tid,
        when="2030-01-01T10:00",
        timezone="America/Chicago",
    )
    run("schedule.complete", schedule_id=alert["id"], expected_revision=1)
    with session_scope() as db:
        task = db.get(Task, tid)
        assert task.status == "completed"
        jid = task.external["job_id"]
    sync.process_write(jid)
    assert linear.rows["remote"]["state"]["type"] == "completed"


def test_graphql_partial_or_malformed_response_is_never_success():
    import httpx

    client = LinearClient("fixture")
    client.http.close()
    for body in [{"data": {"issueUpdate": {"success": True}}, "errors": [{"message": "rejected"}]}, []]:
        client.http = httpx.Client(
            transport=httpx.MockTransport(lambda request, body=body: httpx.Response(200, json=body)),
            base_url="https://api.linear.app",
        )
        with pytest.raises(LinearFailure):
            client.query("mutation Example { issueUpdate { success } }")
        client.close()
