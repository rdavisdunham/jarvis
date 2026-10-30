"""Read-only cross-kind lookup for opening a detail card, never authorizing an edit."""

from datetime import date, timedelta
from sqlalchemy import select
from .domain import DomainError, preferences, zone
from .models import Task, Note, now
from .structure_models import StructureRecord
from .text_normalize import canon

HINTS = set("show open find me my the a an this that please task tasks event events meeting meetings note notes client clients project projects record records reminder reminders todo appointment appointments".split())
# The record kind a person names is a hint, never a filter: "the weekly check-in task" may be an event.
KIND_WORDS = {
    "task": "task", "tasks": "task", "todo": "task",
    "event": "event", "events": "event", "meeting": "event", "meetings": "event", "appointment": "event",
    "appointments": "event",
    "note": "note", "notes": "note",
    "client": "record", "clients": "record", "project": "record", "projects": "record",
    "record": "record", "records": "record",
    "reminder": "reminder", "reminders": "reminder",
}
# What each match kind is called when the answer has to mention it.
KIND_NAMES = {"task": "task", "note": "note", "reminder": "reminder", "routine": "recurring task"}
INSTRUCTION = (
    "Navigation evidence only. If selected is set, open it at once with its exact open command (load that tool "
    "if needed) and wait for the device acknowledgement. The kind the owner named is a hint, not a filter: when "
    "selected.kind differs from requested_kind, still open it and answer about it; never say nothing was found and "
    "never correct their wording. Mention the kind only if it matters, e.g. \"It's a calendar event; want me to "
    "make a task for it?\". With several matches or an incomplete range, ask which one. A name match never "
    "authorizes an edit: read current state and revision before any mutation."
)


def requested_kind(query):
    """The record kind the request names, if any (first kind word wins)."""
    return next((KIND_WORDS[w] for w in canon(query).split() if w in KIND_WORDS), None)


def describe(match, asked):
    """How to name a match in the answer, and whether it is the kind the owner said."""
    kind = match["kind"]
    if kind == "record":
        name = (match.get("backs") or match.get("record_type") or "record").replace("_", " ")
        kinds = {"record", KIND_WORDS.get(name.split()[-1]), match.get("backs")}
    elif kind in KIND_NAMES:
        name, kinds = KIND_NAMES[kind], {kind, "task" if kind == "routine" else None}
    else:  # Local appointments/blocks and Google events are all calendar entries.
        name, kinds = "calendar event", {"event"}
    return {"kind_name": name, "kind_matches_request": asked is None or asked in kinds}


def opener(result):
    """The tool a navigation result's single selected match opens with, so a runner can make it callable."""
    if not isinstance(result, dict):
        return None
    selected = result.get("selected") or (result.get("other_kinds") or {}).get("selected")
    return ((selected or {}).get("open") or {}).get("tool")


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
            add("record", row.id, item.title, "ui_records", {"record_id": row.id},
                record_type=row.type_id, backs="task" if row.task_id else "note" if row.note_id else None)
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
    asked = requested_kind(query)
    if selected:
        selected = {**selected, **describe(selected, asked)}
    return {"matches":matches[:100], "match_count":len(matches), "selected":selected,
            "requested_kind": asked,
            "needs_clarification":bool(matches) and selected is None, "incomplete":incomplete,
            "calendar_range":{"start":begin.isoformat(),"end":until.isoformat()},
            "instruction": INSTRUCTION}
