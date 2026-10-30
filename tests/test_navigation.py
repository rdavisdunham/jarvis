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


def test_kind_mismatch_is_named_but_still_selected(provider):
    day = now().date()
    eid = existing(provider, summary="Weekly check-in",
                   start={"date": day.isoformat()}, end={"date": (day + timedelta(days=1)).isoformat()})
    with session_scope() as db:
        result = find(db, "davin", "show me the weekly check in task")
    assert result["requested_kind"] == "task"
    selected = result["selected"]
    assert selected["id"] == eid and not result["needs_clarification"]
    assert selected["kind_name"] == "calendar event" and selected["kind_matches_request"] is False
    assert "never correct their wording" in result["instruction"]
    with session_scope() as db:
        assert find(db, "davin", "open the weekly check in meeting")["selected"]["kind_matches_request"] is True


async def test_task_lookup_with_no_task_offers_the_one_other_kind_match(provider):
    from jarvis.tools import call_tool

    day = now().date()
    eid = existing(provider, summary="Weekly check-in",
                   start={"date": day.isoformat()}, end={"date": (day + timedelta(days=1)).isoformat()})
    result = await call_tool("davin", "resolve", 0, "task_resolve", {"scope": "search", "query": "weekly check in"})
    assert result["tasks"] == []
    other = result["other_kinds"]
    assert other["selected"]["id"] == eid
    assert other["selected"]["open"] == {"tool": "ui_calendar", "arguments": {
        **other["selected"]["open"]["arguments"], "entity_id": eid, "open_details": True}}
    # A real task match keeps the ordinary task answer, without a cross-kind detour.
    with session_scope() as db:
        execute(db, "davin", "capture-weekly", "task.create", {"title": "Weekly check-in"})
    result = await call_tool("davin", "resolve", 1, "task_resolve", {"scope": "search", "query": "weekly check in"})
    assert len(result["tasks"]) == 1 and "other_kinds" not in result


def test_a_selected_match_makes_its_opener_callable_without_discovery():
    from jarvis.navigation import opener
    from jarvis.tool_catalog import ToolSession
    from jarvis.tools import registry

    for policy in ("baseline", "discovery-v1", "reads-v1", "lean-v1"):
        session = ToolSession(registry(), policy)
        assert "task_resolve" in session.names and "ui_calendar" not in session.names
        found = {"selected": {"open": {"tool": "ui_calendar", "arguments": {}}}}
        assert opener(found) == "ui_calendar"
        assert session.follow(found) == "ui_calendar" and "ui_calendar" in session.names
        assert session.follow(found) is None  # Idempotent.
        assert session.follow({"selected": None, "matches": []}) is None
        assert session.follow({"tasks": [], "other_kinds": {"selected": {"open": {"tool": "ui_records"}}}}) == "ui_records"


def test_typed_and_live_instructions_treat_the_kind_as_a_hint():
    from jarvis.agent_instructions import live_instructions
    from jarvis.navigation import INSTRUCTION
    from jarvis.tool_catalog import DESCRIPTIONS

    # Typed chat: global policy stays small (tool mechanics live in the definitions), so the rule rides on
    # the lookup tools' descriptions and on every navigation result.
    assert "is a hint, not a filter" in INSTRUCTION and "never authorizes an edit" in INSTRUCTION
    assert "hint, not a filter" in DESCRIPTIONS["ui_show"] and "was a hint" in DESCRIPTIONS["task_resolve"]
    live = live_instructions({"timezone": "America/Chicago", "default_reminder_hour": 9})
    assert "Delegate show/open requests as said" in live and "a 'task' may be a calendar event" in live
    assert "other_kinds" in DESCRIPTIONS["task_resolve"]
    assert "use navigation_find" in DESCRIPTIONS["ui_show"]


def test_kind_mismatch_eval_contract_passes():
    """evals/app site_controls.26 runs here against the test database, not only in the eval harness."""
    from scripts.app_eval.contracts import CASES, Harness

    CASES["site_controls.26"](Harness({"refs": {}, "owner": "davin"}), "site_controls.26")
