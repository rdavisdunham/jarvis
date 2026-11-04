"""Approval and publication: exactly one task per approval under crashes, retries and duplicate delivery; changed
payloads need a new approval; agents cannot approve; Linear publication reuses the issue mapping and reconciles an
uncertain remote write before retrying. Linear is a fake client; no network."""

import copy

import pytest
from cryptography.fernet import Fernet
from jarvis import bot_access
from jarvis import linear_commands as commands
from jarvis import linear_sync as sync
from jarvis.config import get_settings
from jarvis.context import models as m
from jarvis.context import pipeline, proposals, publication
from jarvis.db import session_scope
from jarvis.domain import DomainError, execute
from jarvis.linear_client import LinearFailure
from jarvis.models import Job, LinearIssue, Task
from sqlalchemy import func, select

from .test_pipeline import drain, use_doubles, vectors  # noqa: F401
from .test_proposals import LINES, by_title, meeting, process, rows

OWNER = "davin"


@pytest.fixture(autouse=True)
def doubles(vectors):  # noqa: F811
    use_doubles()


def tasks() -> list[Task]:
    with session_scope() as db:
        return list(db.scalars(select(Task).where(Task.owner_id == OWNER, Task.title.not_like("owner%"))
                               .order_by(Task.created_at)))


def approve(proposal, command_id="approve1", **kwargs):
    with session_scope() as db:
        return publication.approve(db, OWNER, proposal["id"], proposal["revision"], command_id=command_id,
                                   principal=OWNER, **kwargs)


def pub(publication_id):
    with session_scope() as db:
        return publication.view(db.get(m.ContextPublication, publication_id))


def publish_jobs():
    with session_scope() as db:
        return [j.id for j in db.scalars(select(Job).where(Job.kind == "context_publish").order_by(Job.created_at))]


def hold_publishing(status):
    with session_scope() as db:
        for job in db.scalars(select(Job).where(Job.kind == "context_publish")):
            job.status = status


def own_proposal():
    got = process(meeting())
    return got, by_title(rows(got.source_id), "circulate")


# ---------------------------------------------------------------- Eridani destination


def test_approval_publishes_exactly_one_task_with_meeting_provenance():
    got, mine = own_proposal()
    first = approve(mine)
    assert first["status"] == "queued" and first["destination"] == "eridani"
    assert approve(mine) == first  # the same command replays
    drain()
    [task] = tasks()
    assert task.title == mine["title"] and str(task.due_date) == "2026-11-06"
    assert "From meeting: Pilot sync (2026-11-03)" in task.notes and "> I'll circulate" in task.notes
    assert task.assignee_id == mine["responsible"]
    done = pub(first["id"])
    assert done["status"] == "published" and done["task_id"] == task.id
    item = by_title(rows(got.source_id), "circulate")
    assert (item["disposition"], item["task_id"], item["revision"]) == ("published", task.id, mine["revision"] + 1)


def test_crash_during_publish_and_duplicate_delivery_write_one_task(monkeypatch):
    _, mine = own_proposal()
    approve(mine)
    original = publication._published
    crashes = []

    def crash_once(*args, **kwargs):
        if not crashes:
            crashes.append(1)
            raise RuntimeError("worker died mid-publish")
        return original(*args, **kwargs)
    monkeypatch.setattr(publication, "_published", crash_once)
    [job_id] = publish_jobs()
    with pytest.raises(RuntimeError):
        pipeline.perform(job_id)
    with session_scope() as db:
        assert db.get(Job, job_id).status == "retrying"
        assert db.scalar(select(m.ContextPublication.status)) == "retrying"
    assert tasks() == []  # the crashed transaction left nothing behind
    pipeline.perform(job_id)
    pipeline.perform(job_id)  # replayed delivery of a finished job
    with session_scope() as db:  # a duplicate outbox delivery as a second job for the same publication
        from jarvis.domain import enqueue_job

        enqueue_job(db, OWNER, "context_publish", {"publication_id": db.scalar(select(m.ContextPublication.id))})
    drain()
    assert len(tasks()) == 1


def test_lost_commit_replays_stored_command_results(monkeypatch):
    """The task commit landed but the publication row did not move on (crash between): the retry replays."""
    _, mine = own_proposal()
    approved = approve(mine)
    with session_scope() as db:
        execute(db, OWNER, f"ctxpub:{approved['id']}:task", "task.create", publication.task_payload(
            db, OWNER, db.get(m.ContextProposal, mine["id"])))
    drain()
    assert len(tasks()) == 1 and pub(approved["id"])["status"] == "published"


def test_changed_payload_needs_a_new_approval_and_approved_work_is_never_modified():
    got, mine = own_proposal()
    approved = approve(mine, edits=publication.Edits(title="Circulate the pilot notes"))
    with pytest.raises(DomainError) as stale:  # the approved revision is spent
        approve(mine, command_id="approve2", edits=publication.Edits(title="Something else"))
    assert stale.value.code == "REVISION_CONFLICT"
    hold_publishing("held")
    moved = [("Davin", "owner", "Thanks for joining. I'll circulate the notes by 2026-11-13."), *LINES[1:]]
    process(meeting(moved, rev="r2"))  # re-extraction while the approved publication waits
    held = by_title(rows(got.source_id), "circulate")
    assert held["disposition"] == "approved" and held["due_date"] == "2026-11-06"
    assert pub(approved["id"])["payload"]["title"] == "Circulate the pilot notes"
    hold_publishing("queued")
    drain()
    [task] = tasks()
    assert task.title == "Circulate the pilot notes" and str(task.due_date) == "2026-11-06"
    with session_scope() as db:  # now published: the change becomes a separate, unchecked update
        proposals.regenerate(db, OWNER, got.source_id)
    follow = next(p for p in rows(got.source_id) if p["follows"] == mine["id"])
    assert follow["kind"] == "update" and not follow["preselected"]
    assert follow["before"]["due_date"] == "2026-11-06" and follow["after"] == {"due_date": "2026-11-13"}
    with pytest.raises(DomainError):  # updates are approved separately
        approve(follow, command_id="approve3")
    applied = approve(follow, command_id="approve4", updates=True)
    drain()
    with session_scope() as db:
        assert str(db.get(Task, task.id).due_date) == "2026-11-13"
    assert pub(applied["id"])["status"] == "published" and len(tasks()) == 1


def test_update_against_a_task_edited_since_is_a_conflict_not_an_overwrite():
    got, mine = own_proposal()
    approve(mine)
    drain()
    [task] = tasks()
    moved = [("Davin", "owner", "Thanks for joining. I'll circulate the notes by 2026-11-13."), *LINES[1:]]
    process(meeting(moved, rev="r2"))
    follow = next(p for p in rows(got.source_id) if p["follows"] == mine["id"])
    with session_scope() as db:
        execute(db, OWNER, "user-edit", "task.update", {"task_id": task.id, "expected_revision": task.revision,
                                                        "title": "Edited by me"})
    applied = approve(follow, command_id="approve2", updates=True)
    drain()
    assert pub(applied["id"])["status"] == "conflict"
    with session_scope() as db:
        current = db.get(Task, task.id)
        assert (current.title, str(current.due_date)) == ("Edited by me", "2026-11-06")


def test_link_approval_records_the_task_without_writing_it():
    got = process(meeting())
    with session_scope() as db:
        existing = execute(db, OWNER, "t1", "task.create", {"title": "Draft the rollout checklist"})["data"]
        proposals.regenerate(db, OWNER, got.source_id)
    link = by_title(rows(got.source_id), "checklist")
    assert link["kind"] == "link"
    done = approve(link)
    assert done["status"] == "published" and done["task_id"] == existing["id"]
    assert publish_jobs() == []
    with session_scope() as db:
        assert db.get(Task, existing["id"]).revision == existing["revision"]


def test_agents_cannot_approve():
    _, mine = own_proposal()
    for command_id in ("ui-agent:abc", "mcp:approve"):
        with pytest.raises(DomainError) as refused:
            approve(mine, command_id=command_id)
        assert refused.value.code == "HUMAN_REQUIRED"
    token = bot_access.principal.set("bot-key-id")
    try:
        with pytest.raises(DomainError) as refused:
            approve(mine, command_id="plainlooking")
        assert refused.value.code == "HUMAN_REQUIRED"
    finally:
        bot_access.principal.reset(token)
    with session_scope() as db:
        assert db.scalar(select(func.count()).select_from(m.ContextPublication)) == 0


def test_paused_processing_still_publishes_approved_work():
    _, mine = own_proposal()
    approve(mine)
    with session_scope() as db:
        pipeline.pause(db, OWNER)
    drain()
    assert len(tasks()) == 1


# ---------------------------------------------------------------- Linear destination


def remote(id, **values):
    return {"id": id, "identifier": "ERI-9", "url": "https://linear.app/x/issue/ERI-9", "title": "t",
            "description": "", "dueDate": None, "priority": 0, "updatedAt": "2026-11-04T10:00:00Z",
            "archivedAt": None, "team": {"id": "team", "name": "Team", "key": "ERI"},
            "state": {"id": "todo", "name": "Todo", "type": "unstarted"}, "assignee": None, "project": None,
            "parent": None, "labels": {"nodes": [], "pageInfo": {"hasNextPage": False}}, **values}


class FakeLinear:
    def __init__(self):
        self.rows, self.writes, self.lost = {}, [], False

    def close(self):
        pass

    def identity(self):
        return {"viewer": {"id": "me", "name": "Owner"}, "organization": {"id": "ws", "name": "Workspace"},
                "teams": [{"id": "team", "name": "Team"}]}

    def directory(self, teams):
        return {"states": [{"id": "todo", "name": "Todo", "type": "unstarted", "team": {"id": "team"}}],
                "users": [{"id": "me", "name": "Owner", "active": True}, {"id": "u-priya", "name": "Priya"}]}

    def issues(self, filter):
        return copy.deepcopy(list(self.rows.values()))

    def issue(self, id):
        return copy.deepcopy(self.rows.get(id))

    def query(self, query, variables):
        data = variables["input"]
        self.writes.append(copy.deepcopy(data))
        row = remote(data["id"], title=data["title"], description=data.get("description") or "",
                     dueDate=data.get("dueDate"), priority=data.get("priority", 0),
                     assignee={"id": data["assigneeId"], "name": "Owner"} if data.get("assigneeId") else None)
        self.rows[data["id"]] = row
        if self.lost:
            self.lost = False
            raise LinearFailure("timeout", retry=True, unknown=True)
        return {"issueCreate": {"success": True, "issue": copy.deepcopy(row)}}


@pytest.fixture
def linear(monkeypatch):
    monkeypatch.setattr(get_settings(), "integration_encryption_key", Fernet.generate_key().decode())
    fake = FakeLinear()
    monkeypatch.setattr(sync, "LinearClient", lambda *_: fake)
    monkeypatch.setattr(commands, "LinearClient", lambda *_: fake)
    conn = sync.connect(OWNER, "fixture-key")
    with session_scope() as db:
        selected = execute(db, OWNER, "select", "linear.select", {"expected_revision": conn["revision"],
                                                                  "team_ids": ["team"]})["data"]
    sync.process_sync(selected["job_id"])
    return fake


def linear_job(publication_id):
    with session_scope() as db:
        return db.get(m.ContextPublication, publication_id).outcome["linear_job_id"]


def test_linear_publication_reuses_the_mapping_so_sync_never_duplicates(linear):
    _, mine = own_proposal()
    approved = approve(mine, destination="linear")
    assert approved["payload"]["linear"] == {"team_id": "team", "assignee_id": "me"}  # never by name
    drain()
    queued = pub(approved["id"])
    assert queued["status"] == "in_progress" and queued["remote_issue_id"]
    sync.process_write(linear_job(approved["id"]))
    assert len(linear.writes) == 1 and linear.writes[0]["assigneeId"] == "me"
    with session_scope() as db:
        assert proposals.destination(db, OWNER) == "linear"  # the explicit choice is remembered
    with session_scope() as db:  # status read back from the linear_write job
        row = db.get(m.ContextPublication, approved["id"])
        publication.refresh(db, OWNER, row)
        done = publication.view(row)
    assert done["status"] == "published"
    with session_scope() as db:
        job = sync.queue_sync(db, OWNER, force=True)
    sync.process_sync(job)
    assert len(tasks()) == 1
    with session_scope() as db:
        [issue] = db.scalars(select(LinearIssue))
        assert (issue.task_id, issue.remote_id) == (done["task_id"], done["remote_issue_id"])


def test_remote_timeout_reconciles_under_the_same_issue_before_retrying(linear):
    _, mine = own_proposal()
    approved = approve(mine, destination="linear")
    drain()
    linear.lost = True
    job_id = linear_job(approved["id"])
    with pytest.raises(RuntimeError):
        sync.process_write(job_id)  # Linear saved it but the response was lost
    with session_scope() as db:
        row = db.get(m.ContextPublication, approved["id"])
        assert publication.refresh(db, OWNER, row).status == "retrying"
    sync.process_write(job_id)  # the retry reads the issue by its UUID first and writes nothing
    assert len(linear.writes) == 1
    with session_scope() as db:
        row = db.get(m.ContextPublication, approved["id"])
        assert publication.refresh(db, OWNER, row).status == "published"
    assert len(tasks()) == 1


def test_unconfirmed_linear_publication_retries_with_the_original_identity(linear, monkeypatch):
    _, mine = own_proposal()
    approved = approve(mine, destination="linear")
    drain()
    job_id = linear_job(approved["id"])
    original = linear.query

    def lost_for_good(query, variables):
        original(query, variables)
        raise LinearFailure("timeout", retry=False, unknown=True)
    monkeypatch.setattr(linear, "query", lost_for_good)
    sync.process_write(job_id)
    with session_scope() as db:
        row = db.get(m.ContextPublication, approved["id"])
        assert publication.refresh(db, OWNER, row).status == "unconfirmed"
    monkeypatch.setattr(linear, "query", original)
    with session_scope() as db:
        again = publication.retry(db, OWNER, approved["id"], command_id="retry1")
    assert again["status"] == "in_progress"
    sync.process_write(linear_job(approved["id"]))
    assert len(linear.writes) == 1  # reconciled by reading the issue: no second create
    with session_scope() as db:
        row = db.get(m.ContextPublication, approved["id"])
        assert publication.refresh(db, OWNER, row).status == "published"
    assert len(tasks()) == 1


def test_linear_refusal_leaves_no_local_task(linear):
    _, mine = own_proposal()
    approved = approve(mine, destination="linear")
    with session_scope() as db:
        from jarvis.models import LinearConnection

        db.get(LinearConnection, OWNER).enabled = False  # disconnected between approval and publishing
    drain()
    assert pub(approved["id"])["status"] == "failed" and tasks() == []
