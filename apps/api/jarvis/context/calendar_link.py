"""Associate a source with a Google Calendar event and seed attendee identity candidates.

An envelope `calendar_event_id` equal to a `GoogleCalendarEvent.provider_id` of the owner's calendars is an
exact match. Otherwise the best event by title similarity and start-time proximity within `WINDOW` becomes a
provisional (`inferred`) association with its score. The association is a `ContextEntityLink` from
`("source", source_id)` to `("calendar_event", <event row id>)`; a human decision (confirmed or rejected link)
is never overwritten, and a rejected event is never proposed again. Calendar tables are read with SQL here,
without foreign keys (module boundary, plan section 2).

Identity seeding (`seed_identities`) is a processing step for the `context_ingest` job, not part of receipt:
`identity.seed_candidates` may create actors through the command path, which must not depend on the scopes of
the bot key that delivered the transcript. It passes the current revision's participants and, only for an exact
or confirmed calendar link, the event's attendees.
"""

import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from difflib import SequenceMatcher
from typing import Literal

from sqlalchemy import select, text

from ..calendar_details import details
from ..models import now
from .models import ContextEntityLink, ContextSource, ContextSourceRevision
from .schema import Participant, SourceEnvelope

try:  # Package B4 implements this; until it lands, seeding is skipped.
    from .identity import seed_candidates
except ImportError:
    seed_candidates = None

WINDOW = timedelta(hours=2)
MIN_SCORE = 0.6
MIN_TITLE = 0.5
ROLE = "calendar_event"

_EVENTS_NEAR = text("""
    SELECT e.id, e.provider_id, e.payload FROM google_calendar_events e
    JOIN google_calendars c ON c.id = e.calendar_id
    WHERE c.owner_id = :owner
      AND e.payload -> 'start' ->> 'dateTime' IS NOT NULL
      AND (e.payload -> 'start' ->> 'dateTime')::timestamptz BETWEEN :lo AND :hi
      AND coalesce(e.payload ->> 'status', 'confirmed') <> 'cancelled'
    LIMIT 200
""")
_EVENT_EXACT = text("""
    SELECT e.id, e.provider_id, e.payload FROM google_calendar_events e
    JOIN google_calendars c ON c.id = e.calendar_id
    WHERE c.owner_id = :owner AND e.provider_id = :provider_id
    LIMIT 1
""")
_EVENT_BY_ID = text("""
    SELECT e.id, e.provider_id, e.payload FROM google_calendar_events e
    JOIN google_calendars c ON c.id = e.calendar_id
    WHERE c.owner_id = :owner AND e.id = :event_id
""")


@dataclass(frozen=True)
class CalendarMatch:
    event_id: str
    provider_id: str
    status: Literal["exact", "inferred"]
    score: float
    attendees: list[Participant] = field(default_factory=list)


def _words(value: str) -> str:
    return " ".join(re.findall(r"\w+", (value or "").lower()))


def _start(payload: dict) -> datetime | None:
    try:
        return datetime.fromisoformat(payload["start"]["dateTime"])
    except (KeyError, TypeError, ValueError):
        return None


def title_similarity(a: str, b: str) -> float:
    """The larger of character similarity and word overlap (shared words over the shorter title's words)."""
    a, b = _words(a), _words(b)
    if not a or not b:
        return 0.0
    left, right = set(a.split()), set(b.split())
    return max(SequenceMatcher(None, a, b).ratio(), len(left & right) / min(len(left), len(right)))


def score(title: str, occurred_at: datetime, event_title: str, event_start: datetime) -> float:
    """0.6 title similarity plus 0.4 start proximity within WINDOW. Titles below MIN_TITLE score 0, so time
    alone never links."""
    similarity = title_similarity(title, event_title)
    if similarity < MIN_TITLE:
        return 0.0
    closeness = max(0.0, 1 - abs((event_start - occurred_at).total_seconds()) / WINDOW.total_seconds())
    return round(0.6 * similarity + 0.4 * closeness, 4)


def attendees(payload: dict) -> list[Participant]:
    people = []
    for guest in details(payload)["attendees"]:
        name, email = guest.get("displayName"), guest.get("email")
        if name or email:
            people.append(Participant(name=(name or None) and name[:200], email=(email or None) and email[:320],
                                      role="organizer" if guest.get("organizer") else "attendee"))
    return people


def find_event(db, owner: str, envelope: SourceEnvelope, rejected: set[str] = frozenset()) -> CalendarMatch | None:
    if envelope.calendar_event_id:
        row = db.execute(_EVENT_EXACT, {"owner": owner, "provider_id": envelope.calendar_event_id}).first()
        if row and row.id not in rejected:
            return CalendarMatch(row.id, row.provider_id, "exact", 1.0, attendees(row.payload))
    rows = db.execute(_EVENTS_NEAR, {"owner": owner, "lo": envelope.occurred_at - WINDOW,
                                     "hi": envelope.occurred_at + WINDOW}).all()
    scored = [(score(envelope.title, envelope.occurred_at, row.payload.get("summary") or "", start), row)
              for row in rows if row.id not in rejected and (start := _start(row.payload))]
    if not scored:
        return None
    best, row = max(scored, key=lambda pair: pair[0])
    return CalendarMatch(row.id, row.provider_id, "inferred", best) if best >= MIN_SCORE else None


def _links(db, owner: str, source_id: str) -> list[ContextEntityLink]:
    return list(db.scalars(select(ContextEntityLink).where(
        ContextEntityLink.owner_id == owner, ContextEntityLink.subject_type == "source",
        ContextEntityLink.subject_id == source_id, ContextEntityLink.role == ROLE)))


def link_source(db, owner: str, source_id: str, revision_id: str, envelope: SourceEnvelope) -> CalendarMatch | None:
    """Record (or refresh) the source's calendar association. Human decisions (confirmed or rejected links) stay;
    a rejected event is not proposed again; an earlier automatic association is replaced."""
    links = _links(db, owner, source_id)
    decided = [link for link in links if link.decision_id or link.status in ("confirmed", "rejected")]
    if any(link.status == "confirmed" for link in decided):
        return None
    match = find_event(db, owner, envelope, {link.target_id for link in decided if link.status == "rejected"})
    for link in links:
        if link not in decided and (match is None or link.target_id != match.event_id):
            db.delete(link)
    if match:
        link = next((x for x in links if x not in decided and x.target_id == match.event_id), None)
        if link is None:
            link = ContextEntityLink(owner_id=owner, subject_type="source", subject_id=source_id,
                                     target_type="calendar_event", target_id=match.event_id, role=ROLE)
            db.add(link)
        link.status, link.score, link.mention = match.status, match.score, envelope.title[:500]
        link.evidence = [{"revision_id": revision_id, "provider_id": match.provider_id,
                          "basis": "calendar_event_id" if match.status == "exact" else "title_and_time"}]
        link.updated_at = now()
    db.flush()
    return match


def seed_identities(db, owner: str, source_id: str):
    """`context_ingest` step: seed actor identity candidates for the source's current revision. Returns
    `seed_candidates`' result, or None when the identity module is absent or the source is gone."""
    source = db.get(ContextSource, source_id)
    if seed_candidates is None or source is None or source.owner_id != owner or source.status == "deleted":
        return None
    revision = db.scalar(select(ContextSourceRevision).where(ContextSourceRevision.source_id == source_id,
                                                             ContextSourceRevision.revision == source.current_revision))
    participants = [Participant.model_validate(p) for p in (revision.participants if revision else [])]
    event = next((link for link in _links(db, owner, source_id) if link.status in ("exact", "confirmed")), None)
    row = event and db.execute(_EVENT_BY_ID, {"owner": owner, "event_id": event.target_id}).first()
    return seed_candidates(db, owner, source_id, participants, attendees(row.payload) if row else [])
