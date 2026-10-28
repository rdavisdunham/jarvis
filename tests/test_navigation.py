from datetime import timedelta
from jarvis.db import session_scope
from jarvis.domain import execute
from jarvis.models import now
from jarvis.navigation import find
from test_google_writes import provider as google_provider, existing

provider = google_provider


def test_kind_hint_does_not_hide_unique_calendar_event(provider):
    day = now().date()
    eid = existing(provider, summary="Weekly check-in",
                   start={"date":day.isoformat()},end={"date":(day+timedelta(days=1)).isoformat()})
    with session_scope() as db:
        result = find(db,"davin","show me the weekly check in task",day.isoformat(),(day+timedelta(days=1)).isoformat())
    assert result["selected"]["id"] == eid
    assert result["selected"]["open"]["tool"]=="ui_calendar"
    assert result["selected"]["open"]["arguments"]["open_details"] is True


def test_same_name_across_kinds_clarifies_instead_of_guessing(provider):
    day = now().date()
    existing(provider,summary="Weekly check-in",start={"date":day.isoformat()},end={"date":(day+timedelta(days=1)).isoformat()})
    with session_scope() as db:
        execute(db,"davin","capture","task.create",{"title":"Weekly check-in"})
        result=find(db,"davin","weekly check in task",day.isoformat(),(day+timedelta(days=1)).isoformat())
    assert result["selected"] is None
    assert result["needs_clarification"] and result["match_count"]==2


def test_incomplete_calendar_does_not_assert_unique_match(monkeypatch):
    from jarvis import workspace
    monkeypatch.setattr(workspace,"calendar",lambda *args:{"items":[],"truncated":True,"warnings":[]})
    with session_scope() as db:
        execute(db,"davin","capture","task.create",{"title":"Weekly check-in"})
        result=find(db,"davin","weekly check in")
    assert result["incomplete"] and result["selected"] is None


def test_navigation_uses_current_backing_title_and_skips_archived(monkeypatch):
    from jarvis import workspace
    from jarvis.models import Task
    from sqlalchemy import select
    monkeypatch.setattr(workspace,"calendar",lambda *args:{"items":[],"truncated":False,"warnings":[]})
    with session_scope() as db:
        execute(db,"davin","capture","task.create",{"title":"Old title"})
        task=db.scalar(select(Task).where(Task.owner_id=="davin",Task.title=="Old title"))
        task.title="Packing Houston"
        db.flush()
        result=find(db,"davin","show packing Houston task")
        assert result["selected"]["title"]=="Packing Houston"
        task.archived=True
        db.flush()
        assert find(db,"davin","packing Houston")["match_count"]==0


async def test_multiday_event_is_one_candidate_and_its_exact_entry_can_open(provider):
    from jarvis.tools import call_tool
    from jarvis.workspace import calendar

    day = now().date()
    eid = existing(provider, summary="Houston trip",
                   start={"date": day.isoformat()},
                   end={"date": (day + timedelta(days=3)).isoformat()})
    with session_scope() as db:
        result = find(db, "davin", "Houston trip", day.isoformat(), (day + timedelta(days=5)).isoformat())
        projected = calendar(db, "davin", day, day + timedelta(days=5), "America/Chicago")
    assert len([item for item in projected["items"] if item["entity_id"] == eid]) == 3
    assert result["match_count"] == 1
    selected = result["selected"]
    assert selected["id"] == eid
    opened = await call_tool("davin", "navigation", 0, selected["open"]["tool"], selected["open"]["arguments"])
    assert opened["ui_action"]["calendar_entry_id"] == selected["open"]["arguments"]["calendar_entry_id"]


def test_same_day_series_occurrences_remain_distinct(provider):
    day = now().date()
    eid = existing(provider, summary="Daily check-in",
                   start={"dateTime": day.isoformat() + "T09:00:00-05:00", "timeZone": "America/Chicago"},
                   end={"dateTime": day.isoformat() + "T10:00:00-05:00", "timeZone": "America/Chicago"},
                   recurrence=["RRULE:FREQ=DAILY;BYHOUR=9,17;COUNT=2"])
    with session_scope() as db:
        result = find(db, "davin", "Daily check in", day.isoformat(), (day + timedelta(days=1)).isoformat())
    assert result["selected"] is None and result["needs_clarification"]
    assert result["match_count"] == 2
    assert {item["id"] for item in result["matches"]} == {eid}
    assert len({item["occurrence_start"] for item in result["matches"]}) == 2
    assert len({item["open"]["arguments"]["calendar_entry_id"] for item in result["matches"]}) == 2


def test_multiday_local_events_deduplicate_without_collapsing_schedule_occurrences(monkeypatch):
    from jarvis import workspace

    items = [
        {"kind": "event", "entity_id": "trip", "title": "Packing trip", "id": "trip:day1", "date": "2026-10-27"},
        {"kind": "event", "entity_id": "trip", "title": "Packing trip", "id": "trip:day2", "date": "2026-10-28"},
        *[{"kind": "reminder", "entity_id": "schedule", "title": "Packing reminder",
           "id": "schedule:" + hour, "date": "2026-10-27", "at": "2026-10-27T" + hour + ":00:00+00:00"}
          for hour in ("09", "17")],
    ]
    monkeypatch.setattr(workspace, "calendar", lambda *args: {"items": items, "truncated": False, "warnings": []})
    with session_scope() as db:
        result = find(db, "davin", "Packing")
    assert result["match_count"] == 3
    assert result["needs_clarification"]
    assert len({item["open"]["arguments"]["calendar_entry_id"] for item in result["matches"]}) == 3
