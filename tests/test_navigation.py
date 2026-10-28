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
