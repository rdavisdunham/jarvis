# ruff: noqa: DTZ001 -- Google all-day recurrence rules intentionally use floating dates.
from datetime import date, datetime
from uuid import uuid4
from zoneinfo import ZoneInfo

import pytest
from dateutil.rrule import rrulestr
from jarvis import planning
from jarvis.calendar_details import normalized_recurrence
from jarvis.db import session_scope
from jarvis.domain import DomainError, execute
from jarvis.google_writes import process_write
from jarvis.models import GoogleCalendarEvent, Job, PlanningEntry, Task
from jarvis.workspace import calendar
from sqlalchemy import func, select
from test_google_writes import fields
from test_google_writes import provider as google_provider

provider = google_provider


def run(tool, **args):
    with session_scope() as db:
        return execute(db, "davin", str(uuid4()), tool, args)["data"]


def test_local_block_is_not_a_deadline_and_cancellation_keeps_task():
    task = run("task.create", title="Write report", due_date="2026-09-20")
    block = run("planning.create", **fields(), kind="block", task_id=task["id"])
    with session_scope() as db:
        assert db.get(Task, task["id"]).due_date == date(2026, 9, 20)
        assert db.scalar(select(func.count(Job.id))) == 0
        items = planning.project(db, "davin", date(2026, 9, 18), date(2026, 9, 19), "America/Chicago")
        assert len(items) == 1 and items[0]["task_id"] == task["id"]
    run("planning.delete", entry_id=block["id"], expected_revision=1)
    with session_scope() as db:
        assert db.get(Task, task["id"]).status == "open"
        assert planning.project(db, "davin", date(2026, 9, 18), date(2026, 9, 19), "America/Chicago") == []


def test_publication_retry_no_double_render_and_conflict(provider):
    entry = run("planning.create", **fields(), google_calendar_id=provider.source)
    provider.fail = "after_create"
    with pytest.raises(RuntimeError):
        process_write(entry["google_job_id"])
    process_write(entry["google_job_id"])
    with session_scope() as db:
        local = db.get(PlanningEntry, entry["id"])
        assert local.google_state == "synced"
        assert len(provider.events) == 1
        items = calendar(db, "davin", date(2026, 9, 18), date(2026, 9, 19), "America/Chicago")["items"]
        assert [e["kind"] for e in items] == ["event"]
        remote_id = local.google_event_id
        remote = {**local.google_snapshot, "etag": "changed", "summary": "Remote change"}
        planning.reconcile(db, "davin", provider.source, [remote])
    with pytest.raises(DomainError, match="Resolve"):
        run("planning.update", entry_id=entry["id"], expected_revision=1, **fields(title="Local change"))
    run("planning.unlink", entry_id=entry["id"], expected_revision=1)
    with session_scope() as db:
        assert db.get(PlanningEntry, entry["id"]).fields["title"] == "Planning"
        assert db.scalar(select(GoogleCalendarEvent).where(GoogleCalendarEvent.provider_id == remote_id))


def test_remote_deletion_never_deletes_local_work(provider):
    entry = run("planning.create", **fields(), google_calendar_id=provider.source)
    process_write(entry["google_job_id"])
    with session_scope() as db:
        planning.reconcile(db, "davin", provider.source, [], full=True)
        assert db.get(PlanningEntry, entry["id"]).google_state == "missing"
        assert db.get(PlanningEntry, entry["id"]).status == "active"


def test_owner_bound_and_busy_intervals():
    entry = run("planning.create", **fields())
    with session_scope() as db, pytest.raises(DomainError):
        execute(
            db, "other", str(uuid4()), "planning.delete", {"entry_id": entry["id"], "expected_revision": 1}
        )
    intervals = planning.busy_intervals(
        "davin",
        datetime.fromisoformat("2026-09-18T08:00-05:00"),
        datetime.fromisoformat("2026-09-18T12:00-05:00"),
    )
    assert len(intervals) == 1 and intervals[0][0].hour == 9
    assert not planning.busy_intervals(
        "other",
        datetime.fromisoformat("2026-09-18T08:00-05:00"),
        datetime.fromisoformat("2026-09-18T12:00-05:00"),
    )


@pytest.mark.parametrize(
    "all_day,until", [(True, "19691231T000000Z"), (True, "20170809T000000Z"), (False, "20221212")]
)
def test_old_finished_google_recurrences_do_not_poison_current_range(all_day, until):
    rules = normalized_recurrence(["RRULE:FREQ=WEEKLY;UNTIL=" + until], all_day, "America/Chicago")
    start = (
        datetime(2020, 1, 1, tzinfo=None)
        if all_day
        else datetime(2020, 1, 1, tzinfo=ZoneInfo("America/Chicago"))
    )
    rule = rrulestr(rules, dtstart=start)
    after = (
        datetime(2026, 9, 1, tzinfo=None)
        if all_day
        else datetime(2026, 9, 1, tzinfo=ZoneInfo("America/Chicago"))
    )
    assert rule.after(after) is None


def test_availability_works_locally_without_google():
    from jarvis.google_calendar import availability

    run("planning.create", **fields())
    result = availability("davin", "2026-09-18T08:00-05:00", "2026-09-18T12:00-05:00")
    assert result["source"] == "eridani_only"
    assert len(result["free"]) == 2
    assert result["free"][1]["start"].startswith("2026-09-18T10:00")


def test_rich_event_details_are_cached_without_unsafe_links():
    from jarvis.google_calendar import clean_event

    event = clean_event(
        {
            "id": "details",
            "description": "Read these notes",
            "hangoutLink": "javascript:alert(1)",
            "conferenceData": {
                "entryPoints": [
                    {"entryPointType": "video", "uri": "https://meet.google.com/safe", "pin": "not-cached"}
                ]
            },
            "attendees": [{"email": "guest@example.test", "responseStatus": "accepted"}],
            "attachments": [
                {"title": "Notes", "fileUrl": "https://docs.google.com/notes"},
                {"title": "Bad", "fileUrl": "javascript:alert(1)"},
            ],
        }
    )
    assert event["description"] == "Read these notes"
    assert event["meeting_url"] == "https://meet.google.com/safe"
    assert event["attendees"][0]["responseStatus"] == "accepted"
    assert len(event["attachments"]) == 1
    assert "not-cached" not in str(event)


def test_local_all_day_conversion_publishes_and_round_trips(provider):
    entry = run("planning.create", **fields(all_day=True, start="2026-09-18", end="2026-09-20"),
                google_calendar_id=provider.source)
    process_write(entry["google_job_id"])
    changed = run("planning.update", entry_id=entry["id"], expected_revision=1, **fields(all_day=False))
    process_write(changed["google_job_id"])
    with session_scope() as db:
        local = db.get(PlanningEntry, entry["id"])
        assert local.google_state == "synced" and not local.fields["all_day"]
        assert set(provider.events[local.google_event_id]["start"]) == {"dateTime", "timeZone"}
        revision = local.revision
    changed = run("planning.update", entry_id=entry["id"], expected_revision=revision,
                  **fields(all_day=True, start="2026-09-21", end="2026-09-24"))
    process_write(changed["google_job_id"])
    with session_scope() as db:
        local = db.get(PlanningEntry, entry["id"])
        assert local.google_state == "synced" and local.fields["all_day"]
        assert provider.events[local.google_event_id]["end"] == {"date": "2026-09-24"}
