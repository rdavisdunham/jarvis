"""Read-only cross-kind lookup for opening a detail card, never authorizing an edit."""

from datetime import date, timedelta
from sqlalchemy import select
from .domain import DomainError, preferences, zone
from .models import Task, Note, now
from .structure_models import StructureRecord
from .text_normalize import canon

HINTS = set("show open find me my the a an this that please task tasks event events meeting meetings note notes client clients project projects record records reminder reminders".split())


def find(db, owner, query, start=None, end=None):
    from .access import authorize_execution
    from .workspace import calendar
    authorize_execution(db, owner)
    words = set(canon(query).split()) - HINTS
    if not words:
        raise DomainError("INVALID_ARGUMENT", "Include the item's name to find its details.")
    timezone = preferences(db, owner)["timezone"]
    begin = date.fromisoformat(start) if start else now().astimezone(zone(timezone)).date()
    until = date.fromisoformat(end) if end else begin + timedelta(days=31)
    matches = []
    def add(kind, identity, title, tool, arguments, **extra):
        if words <= set(canon(title).split()):
            matches.append({"kind": kind, "id": identity, "title": title,
                            "open": {"tool": tool, "arguments": arguments}, **extra})
    linked = set()
    core = {kind: {row.id:row for row in db.scalars(select(model).where(model.owner_id == owner))}
            for model,kind in ((Task,"task"), (Note,"note"))}
    for row in db.scalars(select(StructureRecord).where(StructureRecord.owner_id == owner, StructureRecord.archived.is_(False))):
        item = core["task"].get(row.task_id) or core["note"].get(row.note_id) or row
        if row.task_id: linked.add(("task", row.task_id))
        if row.note_id: linked.add(("note", row.note_id))
        if not item.archived and not getattr(item,"is_template",False):
            add("record", row.id, item.title, "ui_records", {"record_id": row.id})
    for kind,rows in core.items():
        for row in rows.values():
            if not row.archived and (kind,row.id) not in linked and not getattr(row, "is_template", False):
                add(kind,row.id,row.title,"ui_form",{"form":kind,"entity_id":row.id})
    schedule = calendar(db, owner, begin, until, timezone)
    seen_calendar = set()
    for item in schedule["items"]:
        if item["kind"] == "task":
            continue
        # A multi-day event has one projection per visible day, not several choices.
        # Recurring Google events and schedules still need one choice per occurrence.
        occurrence = item.get("occurrence_start")
        if item["kind"] in {"routine", "reminder"}:
            occurrence = item.get("at")
        identity = (item["kind"], item["entity_id"], occurrence)
        if identity in seen_calendar:
            continue
        seen_calendar.add(identity)
        add(item["kind"], item["entity_id"], item["title"], "ui_calendar",
            {"date": item["date"], "entity_id": item["entity_id"],
             "calendar_entry_id": item["id"], "open_details": True},
            date=item["date"], occurrence_start=occurrence)
    incomplete = bool(schedule.get("truncated") or schedule.get("warnings")) or len(matches)>100
    selected = matches[0] if len(matches)==1 and not incomplete else None
    return {"matches":matches[:100], "match_count":len(matches), "selected":selected,
            "needs_clarification":bool(matches) and selected is None, "incomplete":incomplete,
            "calendar_range":{"start":begin.isoformat(),"end":until.isoformat()},
            "instruction":"Open selected using its exact open command and wait for device acknowledgement. Otherwise clarify the candidates/date range. This result is navigation evidence only; read current state and revision before any mutation."}
