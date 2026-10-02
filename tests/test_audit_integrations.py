from datetime import date, datetime, timedelta
from uuid import uuid4
from zoneinfo import ZoneInfo

import pytest
from google.auth.exceptions import RefreshError
from jarvis import google_auth, google_calendar, notices, planning
from jarvis import linear_sync as sync
from jarvis.config import get_settings
from jarvis.db import session_scope
from jarvis.domain import DomainError, execute
from jarvis.google_calendar import SyncFailure, queue_sync
from jarvis.google_schema import CalendarRead
from jarvis.google_writes import process_write, read_event
from jarvis.models import (
    GoogleIdentity,
    Job,
    LinearConnection,
    LinearIssue,
    Notification,
    OwnerSettings,
    PlanningEntry,
    Task,
    now,
)
from jarvis.workspace import calendar
from sqlalchemy import func, select
from test_google_writes import existing, fields
from test_google_writes import provider as google_provider
from test_linear import imported, issue
from test_linear import linear as linear_fixture

provider = google_provider
linear = linear_fixture


def run(tool, **args):
    with session_scope() as db:
        return execute(db, "davin", str(uuid4()), tool, args)["data"]


def generation():
    with session_scope() as db:
        return db.get(GoogleIdentity, "davin").generation


def test_timed_out_sync_is_fenced_without_cancelling_queued_writes(provider, monkeypatch):
    receipt = run("calendar.create", **fields(), calendar_id=provider.source)
    before = generation()
    with session_scope() as db:
        stale = google_calendar.enqueue_job(db, "davin", "google_sync", {"generation": before})
        stale_id = stale.id

    class Late:
        def request(self, path, params=None, **_):
            # While Google answers, the attempt times out and a newer sync supersedes it.
            with session_scope() as db:
                db.get(Job, stale_id).created_at = now() - timedelta(minutes=20)
                assert queue_sync(db, "davin", force=True) != stale_id
            return {"items": []}

        def close(self):
            pass

    monkeypatch.setattr(google_calendar, "CalendarClient", lambda *_: Late())
    google_calendar.process(stale_id)
    with session_scope() as db:
        job = db.get(Job, stale_id)
        assert job.status == "failed" and job.result == {"error": "sync_timed_out"}
        assert db.get(GoogleIdentity, "davin").last_sync_at is None
    assert generation() == before
    monkeypatch.setattr(google_calendar, "CalendarClient", lambda *_: provider)
    process_write(receipt["job_id"])
    with session_scope() as db:
        assert db.get(Job, receipt["job_id"]).status == "succeeded"
    assert len(provider.events) == 1


def test_retryable_refresh_failure_is_temporary(monkeypatch):
    monkeypatch.setattr(get_settings(), "external_services_enabled", True)

    class Token:
        def __init__(self, *_, **__):
            pass

        def refresh(self, _):
            raise RefreshError("backend error", retryable=True)

    monkeypatch.setattr(google_calendar, "Credentials", Token)
    with pytest.raises(SyncFailure) as error:
        google_calendar.CalendarClient({"refresh_token": "fixture"})
    assert error.value.code == "unavailable" and error.value.status == 0
    Token.refresh = lambda *_: (_ for _ in ()).throw(RefreshError("invalid_grant"))
    with pytest.raises(SyncFailure) as error:
        google_calendar.CalendarClient({"refresh_token": "fixture"})
    assert error.value.code == "reconnect"


def test_write_path_unauthorized_marks_account_for_reconnect(provider, monkeypatch):
    receipt = run("calendar.create", **fields(), calendar_id=provider.source)

    def rejected(*_, **__):
        raise SyncFailure("reconnect", 401)

    monkeypatch.setattr(provider, "request", rejected)
    process_write(receipt["job_id"])
    with session_scope() as db:
        assert db.get(Job, receipt["job_id"]).status == "failed"
        account = db.get(GoogleIdentity, "davin")
        assert account.status == "needs_reconnect" and account.error == "reconnect"


def test_failed_publish_can_be_edited_retried_and_deleted(provider):
    provider.role = "reader"
    entry = run("planning.create", **fields(), google_calendar_id=provider.source)
    process_write(entry["google_job_id"])
    with session_scope() as db:
        assert db.get(Job, entry["google_job_id"]).status == "failed"
        planning.reconcile(db, "davin", provider.source, [], full=True)
        row = db.get(PlanningEntry, entry["id"])
        identity, revision = row.google_event_id, row.revision
        assert row.google_state != "missing"
    provider.role = "owner"
    updated = run("planning.update", entry_id=entry["id"], expected_revision=revision, **fields(title="Retry"))
    with session_scope() as db:
        job = db.get(Job, updated["google_job_id"])
        assert job.payload["operation"] == "create" and job.payload["provider_event"] == identity
    process_write(updated["google_job_id"])
    process_write(updated["google_job_id"])
    with session_scope() as db:
        assert db.get(PlanningEntry, entry["id"]).google_state == "synced"
    assert list(provider.events) == [identity] and provider.events[identity]["summary"] == "Retry"

    provider.role = "reader"
    other = run("planning.create", **fields(), google_calendar_id=provider.source)
    process_write(other["google_job_id"])
    count = write_count()
    deleted = run("planning.delete", entry_id=other["id"], expected_revision=other["revision"])
    assert deleted["status"] == "cancelled" and deleted["google_calendar_id"] is None
    assert write_count() == count


def write_count():
    with session_scope() as db:
        return db.scalar(select(func.count(Job.id)).where(Job.kind == "google_write"))


def test_unconfirmed_copy_explains_how_to_continue(provider):
    entry = run("planning.create", **fields(), google_calendar_id=provider.source)
    with session_scope() as db:
        db.get(Job, entry["google_job_id"]).status = "unconfirmed"
    with pytest.raises(DomainError, match="unlink the copy"):
        run("planning.delete", entry_id=entry["id"], expected_revision=entry["revision"])


def test_one_bad_planning_row_does_not_break_the_calendar(provider):
    good = run("planning.create", **fields(title="Good"))
    bad = run("planning.create", **fields(title="Bad"))
    with session_scope() as db:
        row = db.get(PlanningEntry, bad["id"])
        row.fields = {**row.fields, "start": "not a time"}
    with session_scope() as db:
        warnings = []
        items = planning.project(db, "davin", date(2026, 9, 18), date(2026, 9, 19), "America/Chicago", warnings)
        assert [i["entity_id"] for i in items] == [good["id"]] and warnings
        view = calendar(db, "davin", date(2026, 9, 18), date(2026, 9, 19), "America/Chicago")
        assert [i["title"] for i in view["items"]] == ["Good"]


def test_use_google_stores_local_time_in_the_event_zone(provider):
    # Google returns dateTime in the calendar's offset (New York) for a Chicago event.
    eid = existing(
        provider,
        start={"dateTime": "2026-09-18T11:30:00-04:00", "timeZone": "America/Chicago"},
        end={"dateTime": "2026-09-18T12:30:00-04:00", "timeZone": "America/Chicago"},
    )
    token = google_auth.unseal(read_event("davin", CalendarRead(event_id=eid))["edit_token"])
    assert token["event_fields"]["start"] == "2026-09-18T10:30:00"
    assert token["event_fields"]["end"] == "2026-09-18T11:30:00"
    planning.fields(type("Args", (), token["event_fields"])())

    entry = run("planning.create", **fields(), google_calendar_id=provider.source)
    process_write(entry["google_job_id"])
    with session_scope() as db:
        remote_id = db.get(PlanningEntry, entry["id"]).google_event_id
    provider.events[remote_id].update(
        etag='"changed"',
        start={"dateTime": "2026-09-18T12:00:00-04:00", "timeZone": "America/Chicago"},
        end={"dateTime": "2026-09-18T13:00:00-04:00", "timeZone": "America/Chicago"},
    )
    comparison = planning.comparison("davin", entry["id"])
    resolved = run(
        "planning.resolve",
        entry_id=entry["id"],
        expected_revision=entry["revision"],
        choice="google",
        edit_token=comparison["google"]["edit_token"],
    )
    with session_scope() as db:
        row = db.get(PlanningEntry, resolved["id"])
        assert row.fields["start"] == "2026-09-18T11:00:00" and row.google_state == "synced"
        assert planning.project(db, "davin", date(2026, 9, 18), date(2026, 9, 19), "America/Chicago")


def morning_hour_now():
    local = now().astimezone(ZoneInfo("America/Chicago"))
    with session_scope() as db:
        db.merge(
            OwnerSettings(
                owner_id="davin",
                values={"morning_summary": True, "morning_hour": local.hour, "timezone": "America/Chicago",
                        "quiet_enabled": False},
            )
        )


def test_morning_summary_delivers_taskless_overnight_reminders():
    morning_hour_now()
    with session_scope() as db:
        db.add(
            Notification(
                owner_id="davin",
                dedup_key="reminder:" + str(uuid4()),
                category="reminder",
                title="Call the bank",
                body="Reminder",
                scheduled_at=now() - timedelta(hours=2),
                eligible_at=now() - timedelta(hours=2),
                target={"view": "notifications"},
            )
        )
    instant = now()
    with session_scope() as db:
        notices.morning(db, "davin", instant)
    with session_scope() as db:
        summary = db.scalar(select(Notification).where(Notification.category == "morning"))
        held = db.scalar(select(Notification).where(Notification.category == "reminder"))
        assert held.read_at is not None
        assert notices.eligible(db, summary, instant) and "1 overnight updates" in summary.body


def test_deadline_and_reminder_seconds_apart_are_one_alert():
    with session_scope() as db:
        task = Task(owner_id="davin", title="Pay rent", due_date=date(2030, 1, 5), due_time="09:00",
                    due_timezone="America/Chicago")
        db.add(task)
        db.flush()
        due = datetime(2030, 1, 5, 9, tzinfo=ZoneInfo("America/Chicago"))
        db.add(
            Notification(
                owner_id="davin",
                dedup_key="reminder:" + task.id,
                category="reminder",
                title="Pay rent",
                body="Reminder",
                task_id=task.id,
                scheduled_at=due + timedelta(seconds=30),
                eligible_at=due + timedelta(seconds=30),
                target={"view": "tasks"},
            )
        )
        task_id = task.id
    with session_scope() as db:
        notices.sync_deadlines(db, "davin")
        deadline = db.scalar(select(Notification).where(Notification.dedup_key == "deadline:" + task_id))
        assert deadline.dismissed_at is not None


def test_older_linear_read_never_rolls_back_newer_snapshot(linear):
    tid, _ = imported()
    with session_scope() as db:
        link = db.scalar(select(LinearIssue))
        conn = db.get(LinearConnection, "davin")
        link.snapshot = {**link.snapshot, "updatedAt": "2026-09-12T12:00:00Z", "title": "Newer"}
        db.get(Task, tid).title = "Newer"
        sync.apply_remote(db, conn, link, {**issue(), "title": "Older"})
        assert db.get(Task, tid).title == "Newer" and link.snapshot["title"] == "Newer"


def test_only_mine_full_sync_marks_reassigned_issue_out_of_scope(linear):
    tid, _ = imported()

    def issues(filter):
        rows = list(linear.rows.values())
        if "assignee" in filter:
            rows = [r for r in rows if (r.get("assignee") or {}).get("id") == filter["assignee"]["id"]["eq"]]
        if "id" in filter:
            rows = [r for r in rows if r["id"] in filter["id"]["in"]]
        return [dict(r) for r in rows]

    linear.issues = issues
    linear.rows["remote"]["assignee"] = {"id": "someone", "name": "Someone"}
    with session_scope() as db:
        conn = db.get(LinearConnection, "davin")
        conn.only_mine, conn.full_sync_at = True, None
        jid = sync.queue_sync(db, "davin", force=True)
    sync.process_sync(jid)
    with session_scope() as db:
        assert db.get(Task, tid).external["sync_state"] == "out_of_scope"
