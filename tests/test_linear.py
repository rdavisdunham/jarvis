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
        self.comments = []
        self.comment_filters = []
        self.fail_comments = False

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
                {"id": "backlog", "name": "Backlog", "type": "backlog", "team": {"id": "team"}},
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

    def comment_list(self, filter):
        # Linear's root comments query: by issue ids, or changed since a time on in-scope issues.
        self.comment_filters.append(filter)
        if self.fail_comments:
            raise LinearFailure("unavailable")
        ids = filter["issue"].get("id", {}).get("in")
        if ids is not None:
            return copy.deepcopy([c for c in self.comments if c["issueId"] in ids])
        since = filter["updatedAt"]["gte"]
        return copy.deepcopy([c for c in self.comments if c["updatedAt"] >= since])

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
                            "name": {"done": "Done", "backlog": "Backlog"}.get(value, "Todo"),
                            "type": {"done": "completed", "backlog": "backlog"}.get(value, "unstarted"),
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

def test_backlog_round_trips_and_imports_as_backlog(linear):
    tid, rev = imported()
    saved = run("task.update", task_id=tid, expected_revision=rev, status="backlog")
    sync.process_write(saved["external"]["job_id"])
    assert linear.writes[-1][1] == {"stateId": "backlog"}
    with session_scope() as db:
        task = db.get(Task, tid)
        assert task.status == "backlog" and task.completed_at is None
    linear.rows["another"] = issue("another")
    linear.rows["another"]["state"] = {"id": "backlog", "name": "Backlog", "type": "backlog"}
    with session_scope() as db:
        jid = sync.queue_sync(db, "davin", force=True)
    sync.process_sync(jid)
    with session_scope() as db:
        link = db.scalar(select(LinearIssue).where(LinearIssue.remote_id == "another"))
        assert db.get(Task, link.task_id).status == "backlog"


def test_publish_backlog_chooses_matching_linear_state(linear):
    task = run("task.create", title="Future work", status="backlog")
    queued = run("linear.publish", task_id=task["id"], expected_revision=1, team_id="team")
    sync.process_write(queued["job_id"])
    assert linear.writes[-1][1]["stateId"] == "backlog"
    with session_scope() as db:
        assert db.get(Task, task["id"]).status == "backlog"


def resync(full=False):
    with session_scope() as db:
        if full:
            db.get(LinearConnection, "davin").full_sync_at = None
        jid = sync.queue_sync(db, "davin", force=True)
    sync.process_sync(jid)


def linear_state(fake, type, name, at):
    fake.rows["remote"]["state"] = {"id": type, "name": name, "type": type}
    fake.rows["remote"]["updatedAt"] = at


@pytest.mark.parametrize("type, name", [("canceled", "Canceled"), ("duplicate", "Duplicate")])
def test_linear_cancel_records_outcome_with_archive_grace_and_reopen_restores(linear, type, name):
    tid, _ = imported()
    linear_state(linear, type, name, "2026-09-13T10:00:00Z")
    resync()
    with session_scope() as db:
        task = db.get(Task, tid)
        assert (task.status, task.archived) == ("cancelled", False)
        assert task.finished_at is not None and task.archive_eligible_at > task.finished_at
        assert task.external["state"] == name and not task.external["local_archived"]
    linear_state(linear, "unstarted", "Todo", "2026-09-14T10:00:00Z")
    resync()
    with session_scope() as db:
        task = db.get(Task, tid)
        assert (task.status, task.archived) == ("open", False)
    assert linear.writes == []


def test_reopened_linear_issue_stays_archived_when_archived_locally(linear):
    tid, rev = imported()
    run("task.update", task_id=tid, expected_revision=rev, archived=True)
    linear_state(linear, "canceled", "Canceled", "2026-09-13T10:00:00Z")
    resync()
    linear_state(linear, "unstarted", "Todo", "2026-09-14T10:00:00Z")
    resync()
    with session_scope() as db:
        task = db.get(Task, tid)
        assert (task.status, task.archived) == ("open", True)


def test_cancel_with_unassignment_finishes_task_before_it_leaves_scope(linear):
    """The reported case: cancelled and unassigned in one go, so the issue dropped out of 'only mine'."""
    tid, _ = imported()
    with session_scope() as db:
        db.get(LinearConnection, "davin").only_mine = True
    linear_state(linear, "canceled", "Canceled", "2026-09-13T10:00:00Z")
    linear.rows["remote"]["assignee"] = None
    resync()
    with session_scope() as db:
        task = db.get(Task, tid)
        assert (task.status, task.archived) == ("cancelled", False)
        assert task.external["sync_state"] == "out_of_scope" and task.external["state"] == "Canceled"


def test_unassigned_open_issue_only_leaves_scope(linear):
    tid, _ = imported()
    with session_scope() as db:
        db.get(LinearConnection, "davin").only_mine = True
    linear.rows["remote"]["assignee"] = None
    linear.rows["remote"]["updatedAt"] = "2026-09-13T10:00:00Z"
    resync()
    with session_scope() as db:
        task = db.get(Task, tid)
        assert (task.status, task.archived, task.external["sync_state"]) == ("open", False, "out_of_scope")


def test_full_sync_repairs_tasks_already_stuck_open_or_unarchived(linear):
    tid, _ = imported()
    linear.rows["second"] = {**issue("second"), "identifier": "ERI-2"}
    resync()
    with session_scope() as db:
        conn = db.get(LinearConnection, "davin")
        conn.only_mine = True
        links = {link.remote_id: link for link in db.scalars(select(LinearIssue))}
        # Left behind by the old code: one link out of scope with a stale open snapshot, and one task the sync
        # cancelled without archiving.
        links["remote"].sync_state = "out_of_scope"
        sync.external_state(db, links["remote"])
        second = db.get(Task, links["second"].task_id)
        second.status, second.archived = "cancelled", False
        links["second"].snapshot = {**links["second"].snapshot,
                                    "state": {"id": "canceled", "name": "Canceled", "type": "canceled"}}
        second_id = second.id
    linear_state(linear, "canceled", "Canceled", "2026-09-13T10:00:00Z")
    linear.rows["remote"]["assignee"] = None
    linear.rows["second"]["state"] = {"id": "canceled", "name": "Canceled", "type": "canceled"}
    # The issues query with the assignee filter no longer returns the unassigned issue; the full sync re-reads
    # the known out-of-scope link by id.
    original = linear.issues
    def issues(filter):
        rows = original(filter)
        if "id" in filter:
            return [r for r in rows if r["id"] in filter["id"]["in"]]
        return [r for r in rows if "assignee" not in filter or (r.get("assignee") or {}).get("id") == "me"]

    linear.issues = issues
    resync(full=True)
    with session_scope() as db:
        stuck, second = db.get(Task, tid), db.get(Task, second_id)
        assert (stuck.status, stuck.archived, stuck.external["sync_state"]) == ("cancelled", False, "out_of_scope")
        assert (second.status, second.archived) == ("cancelled", False)


def test_held_linear_cancel_is_shown_not_applied(linear):
    from jarvis.sources import task_source

    tid, rev = imported()
    run("task.update", task_id=tid, expected_revision=rev, title="Local edit")  # a write waits on Linear
    linear_state(linear, "canceled", "Canceled", "2026-09-13T10:00:00Z")
    resync()
    with session_scope() as db:
        task = db.get(Task, tid)
        assert (task.status, task.archived) == ("open", False)
        assert task.external["sync_state"] == "pending"
        assert task.external["held_remote_state"] == {"name": "Canceled", "type": "canceled", "status": "cancelled"}
        assert task_source(db, task)["held"]["status"] == "cancelled"


def out_of_scope_task(linear):
    tid, _ = imported()
    with session_scope() as db:
        db.get(LinearConnection, "davin").only_mine = True
    linear.rows["remote"]["assignee"] = None
    linear.rows["remote"]["updatedAt"] = "2026-09-13T10:00:00Z"
    resync()
    with session_scope() as db:
        task = db.get(Task, tid)
        assert task.external["sync_state"] == "out_of_scope"
        return tid, task.revision


def test_finishing_an_out_of_scope_task_here_is_local(linear):
    """The reported case: cancelling an out-of-scope task in Eridani was refused as a conflict."""
    tid, rev = out_of_scope_task(linear)
    saved = run("task.update", task_id=tid, expected_revision=rev, status="cancelled")
    assert saved["status"] == "cancelled"
    assert linear.writes == []
    with pytest.raises(DomainError) as refused:
        run("task.update", task_id=tid, expected_revision=saved["revision"], title="Shared edit")
    assert refused.value.code == "LINEAR_SCOPE"


def test_resolve_without_a_review_token_is_stale_not_google(linear):
    tid, rev = imported()
    run("task.update", task_id=tid, expected_revision=rev, title="Local choice")
    with session_scope() as db:
        link = db.scalar(select(LinearIssue))
        db.get(Job, link.pending_job_id).status = "failed"
        link.sync_state = "conflict"
        rev = db.get(Task, tid).revision
    for token in (None, "not-a-token"):
        with pytest.raises(DomainError) as refused:
            run("linear.resolve", task_id=tid, expected_revision=rev, choice="linear", edit_token=token)
        assert refused.value.code == "LINEAR_STALE" and "Google" not in refused.value.message


def test_undecryptable_linear_key_asks_to_reconnect_linear(linear):
    with session_scope() as db:
        conn = db.get(LinearConnection, "davin")
        conn.credentials = "garbage"
        with pytest.raises(DomainError) as refused:
            sync.linear_key(conn)
    assert refused.value.code == "LINEAR_RECONNECT" and refused.value.message == "Reconnect Linear in Settings."


def test_automatic_linear_archive_survives_reads_but_clears_on_remote_reopen(linear):
    from datetime import timedelta
    from jarvis.task_archive import sweep_owner
    from jarvis.models import now

    tid, _ = imported()
    linear_state(linear, "canceled", "Canceled", "2026-09-13T10:00:00Z")
    resync()
    with session_scope() as db:
        task = db.get(Task, tid)
        task.finished_at = now() - timedelta(hours=25)
        task.archive_eligible_at = task.finished_at + timedelta(hours=24)
    with session_scope() as db:
        assert sweep_owner(db, "davin") == 1
    resync()
    with session_scope() as db:
        task = db.get(Task, tid)
        assert task.archived and task.archive_reason == "automatic"
        assert not task.external["local_archived"]
    linear_state(linear, "unstarted", "Todo", "2026-09-14T10:00:00Z")
    resync()
    with session_scope() as db:
        task = db.get(Task, tid)
        assert task.status == "open" and not task.archived
        assert task.finished_at is None and task.archive_eligible_at is None
    assert linear.writes == []


def test_local_reopen_clears_manual_linear_archive(linear):
    tid, rev = imported()
    archived = run("task.update", task_id=tid, expected_revision=rev, archived=True)
    # Reopen is explicit recovery even for an already-open manually archived task.
    reopened = run("task.reopen", task_id=tid, expected_revision=archived["revision"])
    assert not reopened["archived"] and not reopened["external"]["local_archived"]


def test_recent_changes_preserve_queued_and_succeeded_stored_data(linear, client, monkeypatch):
    from datetime import datetime

    tid, rev = imported()
    edited = run("task.update", task_id=tid, expected_revision=rev, title="Requested Δ title")
    jid = edited["external"]["job_id"]
    with session_scope() as db:
        job = db.get(Job, jid)
        payload = copy.deepcopy(job.payload)
        created_at = job.created_at
    queued = client.get("/api/v1/integrations/linear").json()["recent_changes"][0]
    assert queued == {
        "job_id": jid,
        "status": "queued",
        "message": "",
        "payload": payload,
        "result": None,
        "created_at": created_at.isoformat(),
        "finished_at": None,
        "task_id": tid,
        "issue_identifier": "ERI-1",
        "issue_title": "Requested Δ title",
    }
    assert datetime.fromisoformat(queued["created_at"]).utcoffset() is not None
    sync.process_write(jid)
    with session_scope() as db:
        job = db.get(Job, jid)
        outcome, finished_at = copy.deepcopy(job.result), job.finished_at
        # Changing the live snapshot must not rewrite the stored historical identity.
        link = db.scalar(select(LinearIssue))
        link.snapshot = {**link.snapshot, "identifier": "ERI-NEW", "title": "Later title"}

    def unexpected_client(*_):
        pytest.fail("Reading stored logs must not contact Linear")

    monkeypatch.setattr(sync, "LinearClient", unexpected_client)
    saved = client.get("/api/v1/integrations/linear")
    change = saved.json()["recent_changes"][0]
    assert change["status"] == "succeeded"
    assert change["message"] == outcome["message"]
    assert change["result"] == outcome
    assert change["payload"] == payload
    assert change["finished_at"] == finished_at.isoformat()
    assert datetime.fromisoformat(change["finished_at"]).utcoffset() is not None
    assert change["issue_identifier"] == "ERI-1"
    assert change["issue_title"] == "Requested Δ title"
    for secret in ("fixture-key", "credentials", "Authorization", "api_key"):
        assert secret not in saved.text


def test_recent_changes_new_publish_uses_link_snapshot_and_local_title(linear, client):
    task = run("task.create", title="New local task")
    queued = run("linear.publish", task_id=task["id"], expected_revision=1, team_id="team")
    sync.process_write(queued["job_id"])
    with session_scope() as db:
        job = db.get(Job, queued["job_id"])
        assert not job.payload["base"].get("identifier")
        assert not job.payload["base"].get("title")
        job.payload = {**job.payload, "base": {}}
    change = client.get("/api/v1/integrations/linear").json()["recent_changes"][0]
    assert change["task_id"] == task["id"]
    assert change["issue_identifier"] == "ERI-1"
    assert change["issue_title"] == "New local task"
    with session_scope() as db:
        job = db.get(Job, queued["job_id"])
        job.payload = {**job.payload, "patch": {}}
        link = db.scalar(select(LinearIssue).where(LinearIssue.task_id == task["id"]))
        link.snapshot = {}
    change = client.get("/api/v1/integrations/linear").json()["recent_changes"][0]
    assert change["issue_title"] == "New local task"


def test_recent_changes_keep_missing_history_and_validate_context_owners(linear, client):
    tid, _ = imported()
    nested = {"notes": "Line one\nLine two — 雪 <script>alert(1)</script>", "values": [None, {}, [True, 3]]}
    with session_scope() as db:
        foreign = Task(owner_id="someone-else", title="Private title")
        db.add(foreign)
        db.flush()
        foreign_link = LinearIssue(
            owner_id="someone-else",
            workspace_id="other",
            remote_id="private",
            task_id=foreign.id,
            snapshot={"identifier": "PRIVATE-1", "title": "Private title"},
        )
        db.add(foreign_link)
        db.flush()
        link = db.scalar(select(LinearIssue).where(LinearIssue.task_id == tid))
        removed_id = link.id
        db.delete(link)
        cases = [
            (
                "unlinked",
                {"link_id": removed_id, "base": {"identifier": "OLD-1", "title": "Historical"}},
                None,
            ),
            ("missing", {"link_id": "gone", "base": {}, "metadata": nested}, None),
            ("foreign", {"link_id": foreign_link.id}, {"task_id": foreign.id}),
            ("fallback", {"link_id": "gone"}, {"task_id": tid, "nested": nested}),
            ("empty", {}, {}),
        ]
        for name, payload, result in cases:
            db.add(
                Job(
                    id=name,
                    owner_id="davin",
                    kind="linear_write",
                    status="failed",
                    payload=payload,
                    result=result,
                )
            )
    changes = {j["job_id"]: j for j in client.get("/api/v1/integrations/linear").json()["recent_changes"]}
    assert len(changes) == 5
    assert changes["unlinked"]["task_id"] is None
    assert changes["unlinked"]["issue_identifier"] == "OLD-1"
    assert changes["unlinked"]["issue_title"] == "Historical"
    assert changes["missing"]["payload"]["metadata"] == nested
    for name in ("missing", "foreign", "empty"):
        assert changes[name]["task_id"] is None
        assert changes[name]["issue_identifier"] is None
        assert changes[name]["issue_title"] is None
    assert changes["fallback"]["task_id"] == tid
    assert changes["fallback"]["issue_title"] == "Shared task"
    assert changes["fallback"]["result"]["nested"] == nested
    assert changes["empty"]["result"] == {}
    assert changes["missing"]["result"] is None
    assert all(j["finished_at"] is None for j in changes.values())


def test_recent_changes_newest_twenty_outgoing_only_and_batched(linear, monkeypatch):
    from datetime import UTC, datetime, timedelta

    from jarvis.db import engine
    from sqlalchemy import event

    tid, _ = imported()
    with session_scope() as db:
        link = db.scalar(select(LinearIssue).where(LinearIssue.task_id == tid))
        start = datetime(2026, 10, 1, tzinfo=UTC)
        for i in range(24):
            db.add(
                Job(
                    id=f"write-{i}",
                    owner_id="davin",
                    kind="linear_write",
                    created_at=start + timedelta(seconds=i),
                    payload={"link_id": link.id},
                    status=["queued", "failed", "succeeded", "unconfirmed"][i % 4],
                    result=None,
                )
            )
        for name, owner, kind in [
            ("foreign", "other", "linear_write"),
            ("incoming", "davin", "linear_sync"),
            ("comment", "davin", "linear_comment"),
        ]:
            db.add(Job(id=name, owner_id=owner, kind=kind, created_at=start + timedelta(days=1), payload={}))
    statements = []

    def record(_conn, _cursor, statement, *_args):
        statements.append(statement)

    def unexpected_client(*_):
        pytest.fail("Summary must not fetch Linear")

    monkeypatch.setattr(sync, "LinearClient", unexpected_client)
    event.listen(engine(), "before_cursor_execute", record)
    try:
        with session_scope() as db:
            changes = sync.summary(db, "davin")["recent_changes"]
    finally:
        event.remove(engine(), "before_cursor_execute", record)
    assert [j["job_id"] for j in changes] == [f"write-{i}" for i in range(23, 3, -1)]
    assert {j["status"] for j in changes} == {"queued", "failed", "succeeded", "unconfirmed"}
    assert all(j["task_id"] == tid for j in changes)
    # One connection, jobs, links and tasks query regardless of number of rows.
    assert len(statements) == 4


def test_changing_the_team_scope_after_a_sync_saves(linear):
    # A synced connection has a last_sync_at; linear.select returns the summary, which the command receipt stores as
    # JSON. A raw datetime there failed the whole save ("Selection was not saved") on every change after a sync.
    with session_scope() as db:
        conn = db.get(LinearConnection, "davin")
        assert conn.last_sync_at is not None
        revision = conn.revision
    saved = run("linear.select", expected_revision=revision, team_ids=["team"], only_mine=False)
    assert saved["only_mine"] is False and saved["team_ids"] == ["team"]
    assert isinstance(saved["last_sync_at"], str)

def comment(id, issue_id="remote", body="Looks good", created="2026-09-12T10:05:00Z", **extra):
    return {"id": id, "issueId": issue_id, "parentId": None, "body": body, "url": f"https://linear.app/c/{id}",
            "createdAt": created, "updatedAt": created, "editedAt": None, "resolvedAt": None,
            "user": {"id": "u", "name": "Teammate"}, "externalUser": None, "botActor": None, **extra}


def sync_now(full=False):
    with session_scope() as db:
        if full:
            db.get(LinearConnection, "davin").full_sync_at = None
        jid = sync.queue_sync(db, "davin", force=True)
    sync.process_sync(jid)


def stored_comments(remote_id="remote"):
    with session_scope() as db:
        return db.scalar(select(LinearIssue).where(LinearIssue.remote_id == remote_id)).comments


def test_comments_copy_onto_the_task_and_follow_edits_and_deletions(linear, client):
    linear.comments = [comment("c2", created="2026-09-12T11:00:00Z", body="Second", parentId="c1"),
                       comment("c1", body="First")]
    sync_now(full=True)
    rows = stored_comments()
    assert [c["id"] for c in rows] == ["c1", "c2"]  # oldest first
    assert rows[0] == {"id": "c1", "issue_id": "remote", "parent_id": None, "author": "Teammate", "body": "First",
                       "url": "https://linear.app/c/c1", "created_at": "2026-09-12T10:05:00Z",
                       "updated_at": "2026-09-12T10:05:00Z", "edited_at": None, "resolved_at": None}
    assert rows[1]["parent_id"] == "c1"
    # The issues this sync read have every comment re-read: an edit lands and a deleted comment goes.
    linear.comments = [comment("c1", body="First, edited", editedAt="2026-09-12T12:00:00Z")]
    sync_now()
    assert [(c["id"], c["body"]) for c in stored_comments()] == [("c1", "First, edited")]
    task_id, _ = imported()
    shown = client.get(f"/api/v1/integrations/linear/tasks/{task_id}/comments").json()
    assert shown["identifier"] == "ERI-1" and shown["url"] == "https://linear.app/test/issue/ERI-1"
    assert [c["body"] for c in shown["comments"]] == ["First, edited"] and shown["synced_at"]


def test_comments_on_issues_this_sync_did_not_read_merge_by_change_time(linear, monkeypatch):
    other = {**issue("other"), "identifier": "ERI-2", "title": "Other"}
    linear.rows["other"] = other
    linear.comments = [comment("o1", "other", body="Old")]
    sync_now(full=True)
    assert [c["body"] for c in stored_comments("other")] == ["Old"]
    # Between full syncs only changed issues come back; a new comment on an unchanged issue still arrives.
    monkeypatch.setattr(linear, "issues", lambda _filter: [copy.deepcopy(linear.rows["remote"])])
    linear.comments.append(comment("o2", "other", body="New", created="2099-01-01T00:00:00Z"))
    sync_now()
    assert [c["body"] for c in stored_comments("other")] == ["Old", "New"]
    scoped = linear.comment_filters[-1]
    assert scoped["issue"] == {"team": {"id": {"in": ["team"]}}, "assignee": {"id": {"eq": "me"}}}


def test_a_comment_read_failure_keeps_stored_comments_and_the_issue_sync(linear):
    linear.comments = [comment("c1")]
    sync_now(full=True)
    linear.fail_comments = True
    linear.rows["remote"] = {**linear.rows["remote"], "title": "Renamed", "updatedAt": "2026-09-12T12:00:00Z"}
    sync_now()
    task_id, _ = imported()
    with session_scope() as db:
        assert db.get(Task, task_id).title == "Renamed"
        assert db.get(LinearConnection, "davin").status == "ready"
    assert [c["id"] for c in stored_comments()] == ["c1"]


def test_comments_route_is_owner_scoped_and_needs_a_linear_task(linear, client):
    with session_scope() as db:
        local = Task(owner_id="davin", title="Local only")
        db.add(local)
        db.flush()
        local_id = local.id
    assert client.get(f"/api/v1/integrations/linear/tasks/{local_id}/comments").status_code == 404
    assert client.get(f"/api/v1/integrations/linear/tasks/{uuid4()}/comments").status_code == 404


def test_a_linear_closure_is_marked_so_today_does_not_count_it(linear):
    tid, _ = imported()
    linear.rows["remote"]["state"] = {"id": "done", "name": "Done", "type": "completed"}
    linear.rows["remote"]["updatedAt"] = "2026-09-12T12:00:00Z"
    with session_scope() as db:
        jid = sync.queue_sync(db, "davin", force=True)
    sync.process_sync(jid)
    with session_scope() as db:
        task = db.get(Task, tid)
        assert task.status == "completed" and task.external["finished_via"] == "linear"
        revision = task.revision
    # A status change made in Eridani clears it, and survives the next sync read.
    reopened = run("task.update", task_id=tid, expected_revision=revision, status="open")
    assert reopened["external"]["finished_via"] is None
    sync.process_write(reopened["external"]["job_id"])
    with session_scope() as db:
        revision = db.get(Task, tid).revision
    done = run("task.update", task_id=tid, expected_revision=revision, status="completed")
    sync.process_write(done["external"]["job_id"])
    with session_scope() as db:
        jid = sync.queue_sync(db, "davin", force=True)
    sync.process_sync(jid)
    with session_scope() as db:
        task = db.get(Task, tid)
        assert task.status == "completed" and not task.external.get("finished_via")
