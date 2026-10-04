"""Review cadence: a type behavior that periodically resurfaces its records for review.

The type setting ``review = {enabled, every}`` lives in the structure definition, so turning
it on or off goes through the normal structure preview → apply. Per-record state lives on
``structure_records``:

- ``next_review_at``: when the record is due. Enabling a type seeds it for that type's active
  records, spread over the first interval by a deterministic per-record jitter, so a large
  type never comes due all at once. New records get ``created + interval``.
- ``last_reviewed_at``: the last "Mark reviewed".
- ``review_paused``: "Stop reviewing this record".
- ``review_queued_at``: when the daily queue surfaced the due review in Questions. A record
  stays in the inbox while it was queued after its last review.

Disabling a type stops new items and hides open ones without deleting any of this history.
"""

from datetime import UTC, timedelta
from hashlib import sha256
from zoneinfo import ZoneInfo

from dateutil.relativedelta import relativedelta
from sqlalchemy import func, select

from .domain import DomainError, advisory, emit, owned, preferences
from .models import Note, ReviewDelivery, SharedWorkspace, Task, now
from .structure_models import StructureRecord, StructureSchema

INTERVALS = {
    "1w": relativedelta(weeks=1), "2w": relativedelta(weeks=2), "1m": relativedelta(months=1),
    "3m": relativedelta(months=3), "6m": relativedelta(months=6), "1y": relativedelta(years=1),
}
LABELS = {"1w": "every week", "2w": "every 2 weeks", "1m": "every month", "3m": "every 3 months",
          "6m": "every 6 months", "1y": "every year"}
# At most this many due reviews are added to an account's Questions inbox per local day.
BATCH = 20
SNOOZE = {"day": timedelta(days=1), "week": timedelta(days=7)}
STATE = ("last_reviewed_at", "next_review_at", "review_paused", "review_queued_at")


def cadence(t):
    """The interval key when a type reviews its records, else None."""
    review = (t or {}).get("review") or {}
    return review.get("every", "1m") if review.get("enabled") and not (t or {}).get("archived") else None


def reviewed_types(definition):
    return {t["id"]: every for t in definition["types"] if (every := cadence(t))}


def after(instant, every):
    return instant + INTERVALS[every]


def jitter(identity):
    """A stable fraction in [0, 1) per record, so seeding is spread and reproducible."""
    return int(sha256(identity.encode()).hexdigest()[:8], 16) / 0x100000000


def seeded(identity, every, instant):
    """First due date when a type starts reviewing: within the first interval, never today."""
    span = after(instant, every) - instant - timedelta(days=1)
    return instant + timedelta(days=1) + span * jitter(identity)


def is_due(row, every, instant=None):
    instant = instant or now()
    return bool(every and not row.archived and not row.review_paused
                and row.next_review_at and row.next_review_at <= instant)


def visible(owner):
    """Records that can be reviewed: active and not a recurring-task template."""
    hidden = select(Task.id).where(Task.owner_id == owner, (Task.archived.is_(True)) | (Task.is_template.is_(True)))
    return (StructureRecord.owner_id == owner, StructureRecord.archived.is_(False),
            (StructureRecord.task_id.is_(None)) | (StructureRecord.task_id.notin_(hidden)))


def payload(row, schema):
    """Review fields added to every record payload."""
    t = next((t for t in schema.definition["types"] if t["id"] == row.type_id), None)
    every = cadence(t)
    return {"review_every": every, "review_due": is_due(row, every)}


# ---- Lifecycle -------------------------------------------------------------------------------

def schedule_new(row, t):
    """A new record of a reviewed type is first due one interval after it was created."""
    every = cadence(t)
    if every and not row.next_review_at:
        row.next_review_at = after(row.created_at or now(), every)


def reseed(db, owner, type_id, every, instant):
    for row in db.scalars(select(StructureRecord).where(
            StructureRecord.type_id == type_id, StructureRecord.review_paused.is_(False), *visible(owner))):
        # A record reviewed recently keeps its natural next date; the rest are spread out.
        natural = row.last_reviewed_at and after(row.last_reviewed_at, every)
        row.next_review_at = natural if natural and natural > instant else seeded(row.id, every, instant)
        row.review_queued_at = None


def apply_schema(db, owner, before, definition, instant=None):
    """Called by structure.apply: (re)seed types whose cadence was turned on or changed."""
    instant = instant or now()
    old = reviewed_types(before)
    for type_id, every in reviewed_types(definition).items():
        if old.get(type_id) != every:
            reseed(db, owner, type_id, every, instant)


def schedule_missing(db, owner, types):
    """Records that reached a reviewed type outside record.create get created + interval."""
    for row in db.scalars(select(StructureRecord).where(
            StructureRecord.type_id.in_(list(types)), StructureRecord.next_review_at.is_(None),
            StructureRecord.review_paused.is_(False), *visible(owner)).limit(500)):
        row.next_review_at = after(row.created_at, types[row.type_id])


# ---- Reads ------------------------------------------------------------------------------------

def _titles(db, rows):
    tasks = {t.id: t.title for t in db.scalars(select(Task).where(Task.id.in_([r.task_id for r in rows if r.task_id])))}
    notes = {n.id: n.title for n in db.scalars(select(Note).where(Note.id.in_([r.note_id for r in rows if r.note_id])))}
    return lambda r: tasks.get(r.task_id) or notes.get(r.note_id) or r.title


def item(db, row, types, every, title, instant):
    from .structure import parent_chain
    return {
        "id": row.id, "title": title, "type_id": row.type_id, "type_name": types[row.type_id]["name"],
        "task_id": row.task_id, "note_id": row.note_id, "revision": row.revision,
        "every": every, "every_label": LABELS[every],
        "last_reviewed_at": row.last_reviewed_at.isoformat() if row.last_reviewed_at else None,
        "next_review_at": row.next_review_at.isoformat() if row.next_review_at else None,
        "due": is_due(row, every, instant),
        "home": [{"id": p.id, "title": p.title, "type_id": p.type_id} for p in reversed(parent_chain(db, row.owner_id, row.parent_id, self_id=row.id))],
    }


def listing(db, owner, *, within_days=0, type_id=None, limit=25, offset=0):
    """Records due for review (or due within ``within_days``), oldest due first."""
    from .structure import ensure
    schema = ensure(db, owner)
    types = {t["id"]: t for t in schema.definition["types"]}
    enabled = reviewed_types(schema.definition)
    if type_id:
        enabled = {k: v for k, v in enabled.items() if k == type_id}
    instant = now()
    if not enabled:
        return {"items": [], "total": 0, "due_count": 0, "next_offset": None, "reviewed_types": []}
    base = (StructureRecord.type_id.in_(list(enabled)), StructureRecord.review_paused.is_(False), *visible(owner))
    count = lambda *extra: db.scalar(select(func.count()).select_from(StructureRecord).where(*base, *extra))  # noqa: E731
    upcoming = StructureRecord.next_review_at <= instant + timedelta(days=within_days)
    total = count(upcoming)
    rows = list(db.scalars(select(StructureRecord).where(*base, upcoming)
                           .order_by(StructureRecord.next_review_at, StructureRecord.id).offset(offset).limit(limit)))
    title = _titles(db, rows)
    return {
        "items": [item(db, r, types, enabled[r.type_id], title(r), instant) for r in rows],
        "total": total,
        "due_count": count(StructureRecord.next_review_at <= instant) if within_days else total,
        "next_offset": offset + limit if offset + limit < total else None,
        "reviewed_types": [{"id": k, "name": types[k]["name"], "every": v} for k, v in enabled.items()],
    }


def inbox_items(db, owner):
    """Queued reviews for the personal Questions inbox: pending when due, deferred when snoozed."""
    from .structure import ensure
    schema = ensure(db, owner)
    enabled = reviewed_types(schema.definition)
    if not enabled:
        return []
    types = {t["id"]: t for t in schema.definition["types"]}
    instant = now()
    rows = list(db.scalars(select(StructureRecord).where(
        StructureRecord.type_id.in_(list(enabled)), StructureRecord.review_paused.is_(False),
        StructureRecord.review_queued_at.is_not(None), StructureRecord.next_review_at.is_not(None),
        (StructureRecord.last_reviewed_at.is_(None)) | (StructureRecord.review_queued_at > StructureRecord.last_reviewed_at),
        *visible(owner)).order_by(StructureRecord.next_review_at, StructureRecord.id).limit(200)))
    title = _titles(db, rows)
    result = []
    for row in rows:
        data = item(db, row, types, enabled[row.type_id], title(row), instant)
        snoozed = row.next_review_at > instant
        result.append(dict(
            key="review:" + row.id, kind="record_review", revision=row.revision,
            status="deferred" if snoozed else "pending",
            question=data["title"], source_id=row.id, record=data,
            deferred_until=row.next_review_at if snoozed else None, created_at=row.review_queued_at,
            answer_tool="record.mark_reviewed", answer_args={"record_id": row.id},
        ))
    return result


# ---- Queue ------------------------------------------------------------------------------------

def queue_owner(db, owner, instant=None):
    """Surface up to BATCH due reviews in Questions once per local day, outside quiet hours."""
    from .notices import quiet_until
    from .structure import ensure

    instant = instant or now()
    if db.get(SharedWorkspace, owner):
        return 0  # Questions belong to the personal workspace.
    schema = ensure(db, owner)
    types = reviewed_types(schema.definition)
    if not types:
        return 0
    advisory(db, "record-review-queue:" + owner)
    prefs = preferences(db, owner)
    local = instant.astimezone(ZoneInfo(prefs["timezone"]))
    day_start = local.replace(hour=0, minute=0, second=0, microsecond=0).astimezone(UTC)
    if db.scalar(select(StructureRecord.id).where(
            StructureRecord.owner_id == owner, StructureRecord.review_queued_at >= day_start).limit(1)):
        return 0
    if quiet_until(prefs, instant) > instant:
        return 0
    # Never change the inbox under an invitation that is being delivered.
    if db.scalar(select(ReviewDelivery.id).where(
            ReviewDelivery.owner_id == owner, ReviewDelivery.state.in_(["reserved", "forwarded"]),
            ReviewDelivery.expires_at > instant).limit(1)):
        return 0
    schedule_missing(db, owner, types)
    db.flush()
    rows = list(db.scalars(select(StructureRecord).where(
        StructureRecord.type_id.in_(list(types)), StructureRecord.review_paused.is_(False),
        StructureRecord.next_review_at <= instant,
        (StructureRecord.review_queued_at.is_(None))
        | (StructureRecord.last_reviewed_at.is_not(None) & (StructureRecord.review_queued_at <= StructureRecord.last_reviewed_at)),
        *visible(owner)).order_by(StructureRecord.next_review_at, StructureRecord.id).limit(BATCH)))
    for row in rows:
        row.review_queued_at = instant
    if rows:
        emit(db, owner, "review.changed", owner)
    return len(rows)


def queue_due_record_reviews(db, instant=None):
    owners = db.scalars(select(StructureSchema.owner_id).where(
        StructureSchema.definition.contains({"types": [{"review": {"enabled": True}}]})))
    return sum(queue_owner(db, owner, instant) for owner in list(owners))


# ---- Commands ---------------------------------------------------------------------------------

def mutate(db, owner, tool, args, command_id):
    from .structure import data, ensure, record_type

    schema = ensure(db, owner)
    row = owned(db, StructureRecord, args.record_id, owner, lock=True)
    t = record_type(schema, row.type_id, archived=True)
    every = cadence(t)
    instant = now()
    action = "reviewed" if tool == "record.mark_reviewed" else args.action
    if action in {"reviewed", "snooze"} and not every:
        raise DomainError("REVIEW_OFF", f"{t['name']} records have no review cadence. Turn it on in the type's behaviors first.")
    if action in {"reviewed", "snooze"} and row.archived:
        raise DomainError("ARCHIVED_RECORD", "Restore this record before reviewing it.")
    if action == "reviewed":
        row.last_reviewed_at = instant
        row.next_review_at = after(instant, every)
    elif action == "snooze":
        # The same defer semantics as other Questions: the item stays, deferred until then.
        row.next_review_at = instant + SNOOZE[args.until]
    elif action == "pause":
        row.review_paused = True
        row.review_queued_at = None
    elif action == "resume":
        row.review_paused = False
        if every and (not row.next_review_at or row.next_review_at <= instant):
            row.next_review_at = after(instant, every)
    else:
        for key in STATE:
            setattr(row, key, getattr(args.state, key))
    row.revision += 1
    db.flush()
    emit(db, owner, "record.changed", row.id, row.revision)
    return data(db, row, schema)


def defer(db, owner, record_id, revision, delta):
    """review.defer for a queued record review: snooze it like any other question."""
    from .domain import check_revision
    row = owned(db, StructureRecord, record_id, owner, lock=True)
    check_revision(row, revision)
    row.next_review_at = now() + delta
    row.revision += 1
    emit(db, owner, "record.changed", row.id, row.revision)
    return row
