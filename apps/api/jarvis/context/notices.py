"""Meeting-ready notices, the consolidated backfill notice and the 17:00 context digest (plan_final.md section 8).

Everything here writes ordinary `Notification` rows; `jarvis.notices` (`eligible`, `plan`) and the worker's
delivery loop push them under the owner's quiet hours, priority threshold and device rules. `scheduler.tick`
calls `sync` from the worker's housekeeping.

- `context_ready`: one notice per meeting review, dedup key `context:ready:{source_id}:{revision}`. A meeting is
  ready when its source is `ready` (or `awaiting_input`: the source budget stopped it and the owner must act)
  and no `context_*` job for it is still pending. A later source revision notifies again only when it brings
  material new pending work: a pending proposal that no earlier notice for the meeting listed at its current
  revision (a new proposal, a materially revised one, or an update proposal correcting approved work). The
  target lists the notified `{proposal_id: revision}` so the next revision can be compared.
- `context_backfill`: historical meetings (ingested more than `HISTORICAL` after they ended, as a 90-day
  backfill or a history import does) and meetings that became ready long ago never get their own notice; they
  are folded into one consolidated notice per owner-local day, extended while it is still undelivered.
- `context_digest`: during the 17:00 hour in the owner's timezone, one push (dedup key
  `context:digest:{owner}:{local date}`) when anything is pending: unreviewed meetings, proposals the owner set
  to Later, and context Questions the owner deferred whose deferral has elapsed. Only items that arrived by
  17:00 count (a meeting when its ready notice was written, a Later proposal when it was deferred or its
  `until`, a Question when its deferral ends); later arrivals wait for the next day. Entries are keyed by stable
  identity (source, proposal task link or fingerprint, question key), so proposal updates, publication retries
  and two meetings about the same work count once. They resolve when the meeting is reviewed, the proposal is
  published or dismissed, or the Question is answered.

`current` re-checks a notice just before it is pushed (from `jarvis.notices.eligible`): a ready notice whose
meeting was deleted or whose listed proposals were all decided is dropped, and a digest is recounted, so a
push held by quiet hours never reports work the owner already did.
"""

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from sqlalchemy import select

from ..domain import preferences
from ..models import Job, Notification, SharedWorkspace, now
from .models import ContextDecision, ContextProposal, ContextSource, ContextSourceRevision

READY, BACKFILL, DIGEST = "context_ready", "context_backfill", "context_digest"
CATEGORIES = (READY, BACKFILL, DIGEST)
DIGEST_HOUR = 17
READY_STATUSES = ("ready", "awaiting_input")
PENDING_JOBS = ("queued", "dispatched", "running", "retrying")
OPEN = "pending"
LATER = "later"
HISTORICAL = timedelta(days=2)  # ingested this long after the meeting ended: historical
RECENT = timedelta(days=1)  # ready for longer than this before it is noticed: consolidated, not its own push
SCAN = timedelta(days=14)  # ready sources considered for a notice
MEMORY = timedelta(days=45)  # earlier ready and backfill notices consulted


def _plural(count: int, noun: str) -> str:
    return f"{count} {noun}{'' if count == 1 else 's'}"


# ---------------------------------------------------------------- review state


def unreviewed(db, owner: str) -> set[str]:
    """Meetings that still wait for review: a live source with a pending proposal."""
    return set(db.scalars(
        select(ContextProposal.source_id).join(ContextSource, ContextSource.id == ContextProposal.source_id)
        .where(ContextProposal.owner_id == owner, ContextProposal.disposition == OPEN,
               ContextSource.deleted_at.is_(None)).distinct()))


def _pending(db, owner: str, source_id: str) -> dict[str, int]:
    return dict(db.execute(select(ContextProposal.id, ContextProposal.revision).where(
        ContextProposal.owner_id == owner, ContextProposal.source_id == source_id,
        ContextProposal.disposition == OPEN)).all())


def _busy(db, owner: str) -> set[str]:
    """Sources with a context job still queued or running."""
    return {s for s in db.scalars(select(Job.payload["source_id"].astext).where(
        Job.owner_id == owner, Job.kind.like("context\\_%"), Job.status.in_(PENDING_JOBS))) if s}


def _earlier(db, owner: str, at: datetime) -> list[Notification]:
    return list(db.scalars(select(Notification).where(
        Notification.owner_id == owner, Notification.category.in_((READY, BACKFILL)),
        Notification.created_at >= at - MEMORY).order_by(Notification.created_at, Notification.id)))


def _sources_of(n: Notification) -> list[str]:
    target = n.target or {}
    return list(target.get("source_ids") or ([target["source_id"]] if target.get("source_id") else []))


def ready_at(db, owner: str, at: datetime) -> dict[str, datetime]:
    """When each meeting was last announced as ready (its review's arrival time for the digest)."""
    seen: dict[str, datetime] = {}
    for n in _earlier(db, owner, at):
        for source_id in _sources_of(n):
            seen[source_id] = max(seen.get(source_id, n.created_at), n.created_at)
    return seen


# ---------------------------------------------------------------- meeting-ready and backfill


def _historical(source: ContextSource, revision) -> bool:
    ended = (revision.ended_at or revision.occurred_at) if revision else None
    return ended is not None and source.created_at - ended > HISTORICAL


def announce(db, owner: str, at: datetime | None = None) -> dict:
    """Write ready notices for meetings whose review became ready, and fold historical ones into the day's
    consolidated notice. Idempotent: retries and repeated ticks find the earlier rows."""
    at = at or now()
    out = {"ready": 0, "backfill": 0}
    if db.get(SharedWorkspace, owner):
        return out
    earlier = _earlier(db, owner, at)
    listed: dict[str, int] = {}  # proposal -> highest revision any notice listed
    covered: set[tuple[str, int]] = set()  # (source, source revision) some notice announced
    announced: set[str] = set()
    for n in earlier:
        target = n.target or {}
        for source_id in _sources_of(n):
            announced.add(source_id)
            covered.add((source_id, int((target.get("revisions") or {}).get(source_id, 0))))
        for proposal_id, revision in (target.get("proposals") or {}).items():
            listed[proposal_id] = max(listed.get(proposal_id, 0), int(revision))
    busy = _busy(db, owner)
    batch = []
    rows = db.execute(
        select(ContextSource, ContextSourceRevision)
        .outerjoin(ContextSourceRevision, (ContextSourceRevision.source_id == ContextSource.id)
                   & (ContextSourceRevision.revision == ContextSource.current_revision))
        .where(ContextSource.owner_id == owner, ContextSource.status.in_(READY_STATUSES),
               ContextSource.deleted_at.is_(None), ContextSource.updated_at >= at - SCAN)
        .order_by(ContextSource.updated_at, ContextSource.id)).all()
    for source, revision in rows:
        if source.id in busy:
            continue
        key = f"context:ready:{source.id}:{source.current_revision}"
        if (source.id, source.current_revision) in covered or \
                db.scalar(select(Notification.id).where(Notification.dedup_key == key)):
            continue
        pending = _pending(db, owner, source.id)
        if source.id in announced:
            if all(listed.get(pid) == rev for pid, rev in pending.items()):
                continue  # A revision without material new pending work stays quiet.
        elif _historical(source, revision) or source.updated_at < at - RECENT:
            batch.append((source, pending))
            continue
        count = len(pending)
        db.add(Notification(
            owner_id=owner, dedup_key=key, category=READY, title="Meeting ready for review",
            body=(source.title or "Meeting") + (" · " + _plural(count, "proposal") if count else ""),
            importance="normal", scheduled_at=at, eligible_at=at, created_at=at,
            target={"view": "meetings", "source_id": source.id, "revisions": {source.id: source.current_revision},
                    "proposals": pending}))
        out["ready"] += 1
    if batch:
        out["backfill"] = _consolidate(db, owner, batch, at)
    db.flush()
    return out


def _consolidate(db, owner: str, batch: list[tuple[ContextSource, dict[str, int]]], at: datetime) -> int:
    from ..notices import state

    local = at.astimezone(ZoneInfo(preferences(db, owner)["timezone"]))
    key = f"context:backfill:{owner}:{local.date().isoformat()}"
    row = db.scalar(select(Notification).where(Notification.dedup_key == key))
    if row is not None and state(row, at) not in ("pending", "scheduled"):
        return 0  # Already seen today: the rest wait for tomorrow's notice.
    if row is None:
        row = Notification(owner_id=owner, dedup_key=key, category=BACKFILL, title="Earlier meetings imported",
                           importance="normal", scheduled_at=at, eligible_at=at, created_at=at, target={})
        db.add(row)
    target = row.target or {}
    ids = list(dict.fromkeys(target.get("source_ids", []) + [s.id for s, _ in batch]))
    row.target = {"view": "meetings", "source_ids": ids,
                  "revisions": {**target.get("revisions", {}), **{s.id: s.current_revision for s, _ in batch}},
                  "proposals": {**target.get("proposals", {}), **{k: v for _, p in batch for k, v in p.items()}}}
    row.body = _plural(len(ids), "meeting") + " ready for review"
    return len(batch)


# ---------------------------------------------------------------- digest


def _later_until(db, owner: str, ids: list[str]) -> dict[str, datetime]:
    """The `until` of the latest Later decision on each proposal, when one was given."""
    until: dict[str, datetime] = {}
    d = ContextDecision
    for target_id, effective_at in db.execute(
            select(d.target_id, d.effective_at).where(d.owner_id == owner, d.target_type == "proposal",
                                                      d.operation == "later", d.target_id.in_(ids))
            .order_by(d.seq)):
        if effective_at:
            until[target_id] = effective_at
        else:
            until.pop(target_id, None)
    return until


def items(db, owner: str, cutoff: datetime) -> dict:
    """Pending digest entries that arrived by `cutoff`."""
    from .decisions import question_items

    arrived = ready_at(db, owner, cutoff + RECENT)
    sources = sorted(s for s in unreviewed(db, owner) if s in arrived and arrived[s] <= cutoff)
    later = list(db.scalars(select(ContextProposal).where(
        ContextProposal.owner_id == owner, ContextProposal.disposition == LATER)
        .order_by(ContextProposal.updated_at, ContextProposal.id)))
    until = _later_until(db, owner, [p.id for p in later])
    proposals, work = [], set()
    for p in later:
        if max(p.updated_at, until.get(p.id, p.updated_at)) > cutoff:
            continue
        identity = ("task", p.task_id) if p.task_id else ("work", p.fingerprint)
        if identity in work:
            continue
        work.add(identity)
        proposals.append(p.id)
    questions = sorted({q["key"] for q in question_items(db, owner)
                        if q.get("deferred_until") and q["deferred_until"] <= cutoff})
    return {"source_ids": sources, "proposal_ids": proposals, "question_keys": questions}


def _text(found: dict) -> tuple[str, str]:
    parts = [_plural(len(found["source_ids"]), "meeting") + " to review" if found["source_ids"] else "",
             _plural(len(found["proposal_ids"]), "deferred proposal") if found["proposal_ids"] else "",
             _plural(len(found["question_keys"]), "deferred question") if found["question_keys"] else ""]
    return "Meeting follow-ups", " · ".join(p for p in parts if p)


def cutoff(db, owner: str, at: datetime) -> datetime | None:
    """Today's 17:00 in the owner's timezone while `at` is in that hour, else None."""
    local = at.astimezone(ZoneInfo(preferences(db, owner)["timezone"]))
    if local.hour != DIGEST_HOUR:
        return None
    return local.replace(minute=0, second=0, microsecond=0)


def digest(db, owner: str, at: datetime | None = None) -> Notification | None:
    """The owner's 17:00 context digest, once per local day and only when something is pending."""
    at = at or now()
    if db.get(SharedWorkspace, owner) or (start := cutoff(db, owner, at)) is None:
        return None
    key = f"context:digest:{owner}:{start.date().isoformat()}"
    if db.scalar(select(Notification.id).where(Notification.dedup_key == key)):
        return None
    found = items(db, owner, start)
    if not any(found.values()):
        return None
    title, body = _text(found)
    row = Notification(owner_id=owner, dedup_key=key, category=DIGEST, title=title, body=body,
                       importance="normal", scheduled_at=at, eligible_at=at, created_at=at,
                       target={"view": "meetings", "cutoff": start.isoformat(), **found})
    db.add(row)
    db.flush()
    return row


def sync(db, owner: str, at: datetime | None = None) -> dict:
    at = at or now()
    out = announce(db, owner, at)
    out["digest"] = 1 if digest(db, owner, at) else 0
    return out


# ---------------------------------------------------------------- delivery check


def current(db, n: Notification, at: datetime | None = None) -> bool:
    """Whether a context notice is still worth pushing now; refreshes a digest's text."""
    at = at or now()
    target = n.target or {}
    if n.category == READY:
        source = db.get(ContextSource, target.get("source_id"))
        if source is None or source.deleted_at is not None:
            return False
        listed = list(target.get("proposals") or {})
        return not listed or bool(db.scalar(select(ContextProposal.id).where(
            ContextProposal.id.in_(listed), ContextProposal.disposition == OPEN).limit(1)))
    if n.category == BACKFILL:
        return bool(db.scalar(select(ContextSource.id).where(
            ContextSource.id.in_(target.get("source_ids") or []), ContextSource.deleted_at.is_(None)).limit(1)))
    if n.category == DIGEST:
        start = datetime.fromisoformat(target["cutoff"]) if target.get("cutoff") else n.scheduled_at
        found = items(db, n.owner_id, start)
        found = {k: [v for v in found[k] if v in set(target.get(k) or [])] for k in found}
        if not any(found.values()):
            return False
        n.title, n.body = _text(found)
        return True
    return True
