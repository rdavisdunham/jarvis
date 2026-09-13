from datetime import UTC, datetime
from uuid import uuid4

import pytest
from jarvis.db import session_scope
from jarvis.domain import DomainError, deliver_occurrence, scan_schedules
from jarvis.models import Job, Occurrence, Project, Task
from jarvis.tools import call_tool
from sqlalchemy import func, select


def command(client, tool, **arguments):
    response = client.post(
        "/api/v1/commands", json={"command_id": str(uuid4()), "tool": tool, "arguments": arguments}
    )
    assert response.status_code == 200, response.text
    return response.json()["data"]


def test_projects_metadata_and_parent_cycles(client):
    project = command(client, "project.create", name="Home", description="Household")
    parent = command(client, "task.create", title="Garden", project_id=project["id"])
    child = command(
        client,
        "task.create",
        title="Plant herbs",
        parent_task_id=parent["id"],
        project_id=project["id"],
        tags=["garden", "garden"],
        assignee="Eri",
        work_type="Planning",
    )
    assert child["tags"] == ["garden"] and child["project"] == "Home"
    renamed = command(client, "project.update", project_id=project["id"], expected_revision=1, name="House")
    updated = client.get("/api/v1/tasks/" + child["id"]).json()
    assert updated["project"] == "House" and updated["revision"] == 2
    response = client.post(
        "/api/v1/commands",
        json={
            "command_id": str(uuid4()),
            "tool": "task.update",
            "arguments": {"task_id": parent["id"], "expected_revision": 2, "parent_task_id": child["id"]},
        },
    )
    assert response.status_code == 400
    command(client, "project.update", project_id=renamed["id"], expected_revision=2, archived=True)
    assert len(client.get("/api/v1/tasks").json()["items"]) == 2
    assert client.get("/api/v1/export").json()["projects"][0]["archived"]
    # Assignment never starts an execution job.
    with session_scope() as db:
        assert db.scalar(select(func.count()).select_from(Job)) == 0


def test_calendar_dates_timezones_recurrence_and_read_only(client):
    command(client, "task.create", title="Floating date", due_date="2030-03-10")
    timed = command(
        client,
        "task.create",
        title="Late Chicago",
        due_date="2030-03-10",
        due_time="23:30",
        due_timezone="America/Chicago",
    )
    reminder = command(
        client,
        "schedule.create",
        title="Morning",
        when="2030-03-09T09:00",
        timezone="America/Chicago",
        recurrence="FREQ=DAILY",
    )
    response = client.get("/api/v1/calendar?start=2030-03-09&end=2030-03-13&timezone=America/New_York")
    assert response.status_code == 200
    events = response.json()["items"]
    assert next(e for e in events if e["title"] == "Floating date")["date"] == "2030-03-10"
    assert next(e for e in events if e["entity_id"] == timed["id"])["date"] == "2030-03-11"
    repeats = [e for e in events if e["entity_id"] == reminder["id"]]
    assert len(repeats) == 4
    assert repeats[0]["at"].endswith("15:00:00+00:00") and repeats[1]["at"].endswith("14:00:00+00:00")
    with session_scope() as db:
        assert db.scalar(select(func.count()).select_from(Occurrence)) == 0
        assert db.scalar(select(func.count()).select_from(Job)) == 0
    assert client.get("/api/v1/calendar?start=2030-01-01&end=2031-01-01").status_code == 400


def test_schedule_edit_stale_revision_and_occurrence_completion(client):
    task = command(client, "task.create", title="Linked")
    schedule = command(
        client,
        "schedule.create",
        title="Reminder",
        when="2030-04-01T09:00",
        timezone="America/Chicago",
        task_id=task["id"],
    )
    schedule = command(
        client,
        "schedule.update",
        schedule_id=schedule["id"],
        expected_revision=1,
        title="Renamed",
        when="2030-04-02T10:00",
    )
    assert schedule["task_id"] == task["id"]
    stale = client.post(
        "/api/v1/commands",
        json={
            "command_id": str(uuid4()),
            "tool": "schedule.update",
            "arguments": {"schedule_id": schedule["id"], "expected_revision": 1, "title": "stale"},
        },
    )
    assert stale.status_code == 409
    command(client, "schedule.complete", schedule_id=schedule["id"], expected_revision=2)
    assert client.get("/api/v1/tasks/" + task["id"]).json()["status"] == "completed"
    # Completing early clears the future alert; it does not invent a delivered occurrence.
    assert client.get("/api/v1/calendar?start=2030-04-01&end=2030-04-03").json()["items"] == []


def test_recurring_completion_keeps_series_and_calendar_history(client):
    routine = command(
        client,
        "schedule.create",
        title="Daily stretch",
        when="2030-04-01T09:00",
        timezone="America/Chicago",
        recurrence="FREQ=DAILY",
    )
    with session_scope() as db:
        scan_schedules(db, datetime(2030, 4, 1, 14, 1, tzinfo=UTC))
    with session_scope() as db:
        job = db.scalar(select(Job).where(Job.kind == "reminder"))
        deliver_occurrence(db, job)
    notice = client.get("/api/v1/notifications").json()["items"][0]
    command(client, "notification.complete", notification_id=notice["id"])
    assert client.get("/api/v1/schedules/" + routine["id"]).json()["status"] == "active"
    events = client.get("/api/v1/calendar?start=2030-04-01&end=2030-04-04").json()["items"]
    assert len(events) == 3 and events[0]["status"] == "completed" and events[1]["projected"]


def test_foreign_projects_and_parents_cannot_be_linked(client):
    with session_scope() as db:
        project = Project(owner_id="other", name="Other")
        task = Task(owner_id="other", title="Other")
        db.add_all([project, task])
        db.flush()
        pid, tid = project.id, task.id
    for args in ({"project_id": pid}, {"parent_task_id": tid}):
        response = client.post(
            "/api/v1/commands",
            json={
                "command_id": str(uuid4()),
                "tool": "task.create",
                "arguments": {"title": "Not allowed", **args},
            },
        )
        assert response.status_code == 404
    assert client.get("/api/v1/projects").json()["items"] == []


async def test_eri_project_and_calendar_tools_share_api_contract():
    project = await call_tool("davin", "project-turn", 0, "project_create", {"name": "Tools"})
    task = await call_tool(
        "davin",
        "task-turn",
        0,
        "task_create",
        {"title": "From Eri", "project_id": project["data"]["id"], "due_date": "2030-09-01"},
    )
    result = await call_tool(
        "davin", "calendar-turn", 0, "calendar_list", {"start": "2030-09-01", "end": "2030-09-02"}
    )
    assert result["items"][0]["entity_id"] == task["data"]["id"]
    with pytest.raises(DomainError):
        await call_tool(
            "other",
            "calendar-turn",
            0,
            "task_update",
            {"task_id": task["data"]["id"], "expected_revision": 1, "title": "No"},
        )


def test_completed_task_hides_linked_future_reminders(client):
    task = command(client, "task.create", title="Done", due_date="2030-05-01")
    command(
        client,
        "schedule.create",
        title="Suppressed nudge",
        when="2030-05-01T09:00",
        timezone="America/Chicago",
        task_id=task["id"],
    )
    command(client, "task.complete", task_id=task["id"], expected_revision=1)
    entries = client.get("/api/v1/calendar?start=2030-05-01&end=2030-05-02").json()["items"]
    assert len(entries) == 1 and entries[0]["kind"] == "task"


def test_metadata_edit_does_not_drop_queued_reminder(client):
    reminder = command(
        client,
        "schedule.create",
        title="Original",
        when="2030-05-01T09:00",
        timezone="America/Chicago",
        recurrence="FREQ=DAILY",
    )
    with session_scope() as db:
        scan_schedules(db, datetime(2030, 5, 1, 14, 1, tzinfo=UTC))
    command(client, "schedule.update", schedule_id=reminder["id"], expected_revision=1, title="Updated")
    with session_scope() as db:
        job = db.scalar(select(Job).where(Job.kind == "reminder"))
        result = deliver_occurrence(db, job)
        assert result["notification_id"]
    assert client.get("/api/v1/notifications").json()["items"][0]["title"] == "Updated"
