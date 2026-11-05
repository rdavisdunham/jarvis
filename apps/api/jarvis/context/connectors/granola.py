"""Optional Granola REST adapter: an in-app poller for Business or Enterprise workspaces with an API key.

Off unless both `context_enabled` and `context_granola_enabled` are set. Facts it relies on (plan section 4, verified against docs.granola.ai):
5 requests per second; `GET /v1/notes` pages of up to 30 with a date-granular `updated_after`; `GET
/v1/notes/{id}` for metadata; `GET /v1/notes/{id}/transcript` pages of up to 100 items with an opaque cursor
(never `?include=transcript`, which can return 413 TRANSCRIPT_TOO_LARGE).

Synchronization: scan from the day before the last successful scan date (the filter's day boundary and
timezone are unverified), page fully, skip notes whose `updated_at` was already delivered, deliver each note
to `sources.receive` independently, and advance the watermark only when the whole scan succeeded. 401 marks the
connection `auth_failed`; 429, 5xx and timeouts mark it `degraded`; neither advances the watermark. Nothing is
ever deleted here: a 403 or 404 for one note only parks it for retry. Notes without a transcript yet are parked
the same way. Only notes whose `owner.email` is the connection's account are imported.

Speakers: "me" lines are the note owner's and "them" lines keep Granola's name or diarization label. A group call
(more than two people) heard through one audio source with no names is not separated: its lines get no speaker
and no attribution rather than all going to the note owner (`speakers_separated`).

Secrets and transcript text are never logged.
"""

import copy
import email.utils
import logging
import re
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from typing import Any

import httpx
from sqlalchemy import select

from .. import settings as context_settings
from ..models import ContextConnection, ContextCursor
from ..schema import Participant, Segment, SourceEnvelope, content_hash

log = logging.getLogger(__name__)

BASE_URL = "https://public-api.granola.ai"
RATE_PER_SECOND = 5.0
NOTES_PAGE_SIZE = 30
TRANSCRIPT_PAGE_SIZE = 100
MAX_RETRIES = 4
MAX_BACKOFF_S = 60.0
TIMEOUT_S = 30.0
NOTE_ID = re.compile(r"^not_[a-z0-9]{14}$", re.IGNORECASE)
CURSOR_NAME = "notes"
BACKFILL_CURSOR_NAME = "backfill"
PENDING_MAX_AGE = timedelta(days=7)
PENDING_MAX_ATTEMPTS = 20


class GranolaError(Exception):
    """An upstream failure. `retryable` failures pause the scan; others concern one request only."""

    def __init__(self, code: str, status: int | None = None, retryable: bool = False):
        self.code, self.status, self.retryable = code, status, retryable
        super().__init__(f"{code} ({status})" if status else code)


class GranolaAuthFailed(GranolaError):
    def __init__(self, status: int = 401):
        super().__init__("auth_failed", status)


class TokenBucket:
    """`rate` tokens per second, bursting to `capacity`. Clock and sleep are injectable for tests."""

    def __init__(self, rate: float = RATE_PER_SECOND, capacity: float | None = None,
                 clock: Callable[[], float] = time.monotonic, sleep: Callable[[float], None] = time.sleep):
        self.rate, self.capacity = rate, capacity if capacity is not None else rate
        self.clock, self.sleep = clock, sleep
        self.tokens, self.updated = self.capacity, clock()

    def acquire(self) -> None:
        while True:
            current = self.clock()
            self.tokens = min(self.capacity, self.tokens + (current - self.updated) * self.rate)
            self.updated = current
            if self.tokens >= 1 - 1e-9:  # tolerance: float refill must not leave a sub-ulp wait that never ends
                self.tokens = max(0.0, self.tokens - 1)
                return
            self.sleep((1 - self.tokens) / self.rate)


def retry_after(response: httpx.Response, attempt: int) -> float:
    """Seconds to wait before retrying: `Retry-After` (seconds or HTTP date) when present, else exponential."""
    header = response.headers.get("Retry-After")
    if header:
        try:
            return min(MAX_BACKOFF_S, max(0.0, float(header)))
        except ValueError:
            try:
                when = email.utils.parsedate_to_datetime(header)
                return min(MAX_BACKOFF_S, max(0.0, (when - datetime.now(UTC)).total_seconds()))
            except (TypeError, ValueError):
                pass
    return min(MAX_BACKOFF_S, 2.0 ** attempt)


@dataclass(frozen=True)
class NotesPage:
    notes: list[dict]
    cursor: str | None
    has_more: bool


class GranolaClient:
    def __init__(self, api_key: str, *, base_url: str = BASE_URL, transport: httpx.BaseTransport | None = None,
                 limiter: TokenBucket | None = None, sleep: Callable[[float], None] = time.sleep,
                 max_retries: int = MAX_RETRIES):
        if transport is None:
            from ...config import require_external_services
            require_external_services()
        self.http = httpx.Client(base_url=base_url, transport=transport, timeout=TIMEOUT_S,
                                 headers={"Authorization": f"Bearer {api_key}", "Accept": "application/json"})
        self.limiter = limiter or TokenBucket(sleep=sleep)
        self.sleep, self.max_retries = sleep, max_retries

    def close(self) -> None:
        self.http.close()

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self.close()

    def _get(self, path: str, params: dict[str, Any] | None = None) -> dict:
        for attempt in range(self.max_retries + 1):
            self.limiter.acquire()
            try:
                response = self.http.get(path, params={k: v for k, v in (params or {}).items() if v is not None})
            except httpx.TimeoutException:
                failure, wait = GranolaError("timeout", retryable=True), min(MAX_BACKOFF_S, 2.0 ** attempt)
            except httpx.TransportError:
                failure, wait = GranolaError("unreachable", retryable=True), min(MAX_BACKOFF_S, 2.0 ** attempt)
            else:
                status = response.status_code
                if status == 401:
                    raise GranolaAuthFailed()
                if status == 429 or status >= 500:
                    failure = GranolaError("rate_limited" if status == 429 else "upstream", status, retryable=True)
                    wait = retry_after(response, attempt)
                elif status >= 400:
                    # 403 and 404 are ambiguous (permissions, workspace scope): never deletion evidence.
                    raise GranolaError({403: "forbidden", 404: "not_found", 413: "too_large"}.get(status, "rejected"),
                                       status)
                else:
                    try:
                        body = response.json()
                    except ValueError:
                        body = None
                    if not isinstance(body, dict):
                        raise GranolaError("invalid_response", status, retryable=True)
                    return body
            log.warning("granola request attempt %d failed: %s", attempt + 1, failure.code)
            if attempt < self.max_retries:
                self.sleep(wait)
        raise failure

    def list_notes(self, updated_after: date | None = None, cursor: str | None = None, *,
                   created_after: date | None = None) -> NotesPage:
        body = self._get("/v1/notes", {
            "updated_after": updated_after.isoformat() if updated_after else None,
            "created_after": created_after.isoformat() if created_after else None,
            "page_size": NOTES_PAGE_SIZE, "cursor": cursor})
        notes = body.get("notes")
        if not isinstance(notes, list):
            raise GranolaError("invalid_response", retryable=True)
        return NotesPage([n for n in notes if isinstance(n, dict)], body.get("cursor"), bool(body.get("hasMore")))

    def iter_notes(self, updated_after: date | None = None, *, created_after: date | None = None) -> Iterator[dict]:
        """Every listed note across all pages."""
        cursor = None
        while True:
            page = self.list_notes(updated_after, cursor, created_after=created_after)
            yield from page.notes
            if not page.has_more or not page.cursor or page.cursor == cursor:
                return
            cursor = page.cursor

    @staticmethod
    def _note_path(note_id: str) -> str:
        if not NOTE_ID.match(note_id):
            raise GranolaError("invalid_id")
        return f"/v1/notes/{note_id}"

    def get_note(self, note_id: str) -> dict:
        return self._get(self._note_path(note_id))

    def get_transcript(self, note_id: str) -> list[dict]:
        path = self._note_path(note_id) + "/transcript"
        items: list[dict] = []
        cursor = None
        while True:
            body = self._get(path, {"page_size": TRANSCRIPT_PAGE_SIZE, "cursor": cursor})
            page = body.get("transcript")
            if not isinstance(page, list):
                raise GranolaError("invalid_response", retryable=True)
            items.extend(i for i in page if isinstance(i, dict))
            next_cursor = body.get("cursor")
            if not body.get("hasMore") or not next_cursor or next_cursor == cursor:
                return items
            cursor = next_cursor


# ---------------------------------------------------------------- mapping


def _time(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def _text(value: Any) -> str | None:
    return value.strip() or None if isinstance(value, str) else None


def owner_label(note: dict) -> str:
    owner = note.get("owner") or {}
    return _text(owner.get("name")) or _text(owner.get("email")) or "Note owner"


def _speaker(item: dict) -> dict:
    return item.get("speaker") if isinstance(item.get("speaker"), dict) else {}


def _emails(note: dict) -> set[str]:
    """Everyone the note knows was in the meeting: owner, attendees, invitees and organiser, by email."""
    event = note.get("calendar_event") or {}
    people = [note.get("owner") or {}, *(note.get("attendees") or []), *(event.get("invitees") or [])]
    found = {(_text(p.get("email")) or "").lower() for p in people if isinstance(p, dict)}
    found.add((_text(event.get("organiser")) or "").lower())
    return found - {""}


def speakers_separated(note: dict, transcript: list[dict]) -> bool:
    """False for a group call (more than two people) heard through one audio source with no names or diarization:
    a colleague's single-device recording, where every line comes out as "me" whoever spoke."""
    speakers = [_speaker(i) for i in transcript if isinstance(i.get("text"), str) and i["text"].strip()]
    if len(_emails(note)) <= 2 or not speakers:
        return True
    if any(_text(s.get("name")) or _text(s.get("diarization_label")) for s in speakers):
        return True
    return len({(s.get("source"), s.get("attribution")) for s in speakers}) > 1


def _segments(note: dict, transcript: list[dict]) -> list[Segment]:
    me = owner_label(note)
    separated = speakers_separated(note, transcript)
    timed = [t for t in (_time(i.get("start_time")) for i in transcript) if t]
    base = timed[0] if timed else None
    segments = []
    for item in transcript:
        text = item.get("text")
        if not isinstance(text, str) or not text.strip():
            continue
        speaker = _speaker(item)
        attribution = speaker.get("attribution")
        if not separated:
            # One audio source in a group call: who spoke is unknown, not the note owner.
            label, attributed = None, None
        elif attribution == "me":
            # "me" is the note owner (the person who took the note), never the Eridani importer.
            label, attributed = me, "owner"
        else:
            # "them" or unknown: only what Granola says; an unnamed speaker stays unnamed.
            label = _text(speaker.get("name")) or _text(speaker.get("diarization_label"))
            attributed = "other" if attribution == "them" else None
        start, end = _time(item.get("start_time")), _time(item.get("end_time"))
        start_ms = max(0, int((start - base).total_seconds() * 1000)) if start and base else None
        end_ms = max(0, int((end - base).total_seconds() * 1000)) if end and base else None
        if start_ms is not None and end_ms is not None and end_ms < start_ms:
            end_ms = start_ms
        segments.append(Segment(seq=len(segments), speaker_label=label and label[:200], attribution=attributed,
                                text=text, start_ms=start_ms, end_ms=end_ms))
    return segments


def _participants(note: dict, segments: list[Segment]) -> list[Participant]:
    owner = note.get("owner") or {}
    event = note.get("calendar_event") or {}
    organiser = (_text(event.get("organiser")) or "").lower()
    people: dict[str, dict] = {}

    def add(email: Any, name: Any = None) -> None:
        address = _text(email)
        if not address:
            return
        entry = people.setdefault(address.lower(), {"email": address, "name": None})
        entry["name"] = entry["name"] or _text(name)

    add(owner.get("email"), owner.get("name"))
    for person in note.get("attendees") or []:
        if isinstance(person, dict):
            add(person.get("email"), person.get("name"))
    for person in event.get("invitees") or []:
        if isinstance(person, dict):
            add(person.get("email"))
    if organiser:
        add(organiser)
    owner_email = (_text(owner.get("email")) or "").lower()
    me = owner_label(note)
    result = [Participant(name=p["name"] and p["name"][:200], email=p["email"][:320],
                          role="organizer" if key == organiser else "attendee",
                          speaker_label=me if key == owner_email and any(s.attribution == "owner" for s in segments)
                          else None)
              for key, p in people.items()]
    named = {p.name for p in result if p.name} | {me}
    for label in dict.fromkeys(s.speaker_label for s in segments if s.speaker_label and s.attribution != "owner"):
        if label not in named:
            result.append(Participant(role="speaker", speaker_label=label))
    return result


def to_envelope(note: dict, transcript: list[dict]) -> SourceEnvelope:
    """Map a Granola note and its full transcript to an envelope. Deterministic: same input, same envelope.
    Raises ValueError when the note has no transcript text yet."""
    segments = _segments(note, transcript)
    if not segments:
        raise ValueError("note has no transcript text")
    event = note.get("calendar_event") or {}
    owner = note.get("owner") or {}
    starts = [t for t in (_time(i.get("start_time")) for i in transcript) if t]
    ends = [t for t in (_time(i.get("end_time")) for i in transcript) if t]
    scheduled_start, scheduled_end = _time(event.get("scheduled_start_time")), _time(event.get("scheduled_end_time"))
    occurred = min(starts) if starts else scheduled_start or _time(note.get("created_at"))
    if occurred is None:
        raise ValueError("note has no usable time")
    ended = max(ends) if ends else scheduled_end if not starts else None
    private_notes = _text(note.get("private_notes_markdown"))
    metadata = {k: v for k, v in {
        "web_url": note.get("web_url"),
        "created_at": note.get("created_at"),
        "scheduled_start_time": event.get("scheduled_start_time"),
        "scheduled_end_time": event.get("scheduled_end_time"),
        "event_title": event.get("event_title"),
        "folders": [f.get("name") for f in note.get("folder_membership") or [] if isinstance(f, dict)] or None,
    }.items() if v}
    if not speakers_separated(note, transcript):
        metadata["speakers_separated"] = False
    return SourceEnvelope(
        connector="granola", external_id=str(note["id"]),
        title=(_text(note.get("title")) or _text(event.get("event_title")) or "")[:500],
        occurred_at=occurred, ended_at=ended if ended and ended >= occurred else None,
        participants=_participants(note, segments), segments=segments, private_notes=private_notes,
        summary=_text(note.get("summary_markdown")) or _text(note.get("summary_text")),
        external_revision=_text(note.get("updated_at")),
        content_hash=content_hash([s.text for s in segments] + ([private_notes] if private_notes else [])),
        calendar_event_id=_text(event.get("calendar_event_id")),
        upstream_owner_email=_text(owner.get("email")), upstream_owner_name=_text(owner.get("name")),
        metadata={"granola": metadata})


# ---------------------------------------------------------------- credentials


def seal_credential(api_key: str) -> str:
    """Ciphertext for `ContextConnection.credential_ciphertext` (Fernet, integration_encryption_key)."""
    from ...work_crypto import seal
    return seal({"api_key": api_key})


def open_credential(connection: ContextConnection) -> str | None:
    """The connection's own API key, or None. There is deliberately no server-wide fallback: a key reads every note
    its Granola account can see, so it must belong to the owner who connected it. Raises DomainError when the
    ciphertext cannot be opened."""
    if connection.credential_ciphertext:
        from ...work_crypto import unseal
        key = unseal(connection.credential_ciphertext).get("api_key")
        if isinstance(key, str) and key:
            return key
    return None


# ---------------------------------------------------------------- poller


@dataclass
class SyncReport:
    status: str = "ok"  # ok, disabled, auth_failed, degraded, unconfigured
    listed: int = 0
    delivered: int = 0
    unchanged: int = 0
    foreign: int = 0  # notes owned by someone else: never imported
    deleted_upstream: int = 0  # visible only to workspace-wide keys; the Eridani copy is kept
    pending: list[str] = field(default_factory=list)  # parked for retry (no transcript yet, 403/404, receive error)
    abandoned: list[str] = field(default_factory=list)
    watermark: str | None = None
    advanced: bool = False
    error: str | None = None


def _cursor(db, connection: ContextConnection, name: str) -> ContextCursor:
    row = db.scalar(select(ContextCursor).where(ContextCursor.connection_id == connection.id,
                                                ContextCursor.name == name))
    if row is None:
        row = ContextCursor(owner_id=connection.owner_id, connection_id=connection.id, name=name, value={})
        db.add(row)
        db.flush()
    return row


def _state(cursor: ContextCursor) -> dict:
    """A deep copy, so assigning the result back is a change SQLAlchemy persists (JSONB is not mutation-tracked)."""
    state = copy.deepcopy(cursor.value or {})
    state.setdefault("seen", {})
    state.setdefault("pending", {})
    return state


def _deliver(db, owner: str, connection: ContextConnection, envelope: SourceEnvelope):
    from .. import sources
    with db.begin_nested():
        return sources.receive(db, owner, envelope, connection_id=connection.id)


def _set_health(connection: ContextConnection, health: str, error: str | None, when: datetime) -> None:
    connection.health, connection.last_error, connection.updated_at = health, error, when
    if health == "ok":
        connection.last_success_at = when


def _scan(db, owner: str, connection: ContextConnection, client: GranolaClient, state: dict, report: SyncReport,
          when: datetime, *, updated_after: date | None = None, created_after: date | None = None) -> None:
    """Deliver every changed note the listing returns, then retry parked notes. Upstream pause conditions raise."""
    account = connection.external_account.strip().lower()
    seen: dict = state.setdefault("seen", {})
    pending: dict = state.setdefault("pending", {})
    tried: set[str] = set()

    def fetch_and_deliver(note_id: str, listed_updated_at: str | None) -> None:
        tried.add(note_id)
        try:
            note = client.get_note(note_id)
            owner_email = (_text((note.get("owner") or {}).get("email")) or "").lower()
            if owner_email != account:
                report.foreign += 1
                pending.pop(note_id, None)
                return
            envelope = to_envelope(note, client.get_transcript(note_id))
        except GranolaError as error:
            if error.retryable or isinstance(error, GranolaAuthFailed):
                raise
            reason = error.code
        except ValueError:
            reason = "no_transcript"
        else:
            try:
                _deliver(db, owner, connection, envelope)
            except Exception as error:  # noqa: BLE001 - one bad note never blocks the others
                log.warning("granola note %s not received: %s", note_id, type(error).__name__)
                reason = "receive_failed"
            else:
                seen[note_id] = {"updated_at": envelope.external_revision or listed_updated_at,
                                 "hash": envelope.content_hash}
                pending.pop(note_id, None)
                report.delivered += 1
                return
        entry = pending.setdefault(note_id, {"since": when.isoformat(), "attempts": 0})
        entry.update(attempts=entry["attempts"] + 1, reason=reason)

    for summary in client.iter_notes(updated_after, created_after=created_after):
        report.listed += 1
        note_id = summary.get("id")
        if not isinstance(note_id, str):
            continue
        if (_text((summary.get("owner") or {}).get("email")) or "").lower() != account:
            report.foreign += 1
            continue
        if summary.get("deleted_at"):
            report.deleted_upstream += 1
            continue
        updated_at = summary.get("updated_at")
        if note_id not in pending and (seen.get(note_id) or {}).get("updated_at") == updated_at:
            report.unchanged += 1
            continue
        fetch_and_deliver(note_id, updated_at)
    for note_id in list(pending):
        entry = pending[note_id]
        since = _time(entry.get("since")) or when
        if when - since > PENDING_MAX_AGE or entry.get("attempts", 0) >= PENDING_MAX_ATTEMPTS:
            report.abandoned.append(note_id)
            del pending[note_id]
        elif note_id not in tried:
            fetch_and_deliver(note_id, None)
    report.pending = sorted(pending)


def _client_for(connection: ContextConnection, report: SyncReport, when: datetime,
                transport: httpx.BaseTransport | None, sleep: Callable[[float], None]) -> GranolaClient | None:
    from ...domain import DomainError
    try:
        api_key = open_credential(connection)
    except DomainError:
        report.status, report.error = "auth_failed", "credential could not be opened"
        _set_health(connection, "auth_failed", report.error, when)
        return None
    if not api_key:
        report.status, report.error = "unconfigured", "no API key"
        _set_health(connection, "unconfigured", report.error, when)
        return None
    return GranolaClient(api_key, transport=transport, sleep=sleep)


def _check(owner: str, connection: ContextConnection) -> str | None:
    if connection.owner_id != owner or connection.connector != "granola":
        raise ValueError("connection does not belong to this owner's Granola connector")
    configured = context_settings.current()
    if not (configured.enabled and configured.granola_enabled and connection.enabled):
        return "disabled"
    return None


def _pause(connection: ContextConnection, report: SyncReport, error: GranolaError, when: datetime) -> None:
    report.status = "auth_failed" if isinstance(error, GranolaAuthFailed) else "degraded"
    report.error = str(error)
    _set_health(connection, report.status, report.error, when)


def sync(db, owner: str, connection: ContextConnection, *, now: datetime | None = None,
         transport: httpx.BaseTransport | None = None, sleep: Callable[[float], None] = time.sleep) -> SyncReport:
    """One incremental scan. The caller owns the transaction; source rows persist per note through savepoints."""
    report = SyncReport()
    if disabled := _check(owner, connection):
        report.status = disabled
        return report
    when = now or datetime.now(UTC)
    client = _client_for(connection, report, when, transport, sleep)
    if client is None:
        return report
    cursor = _cursor(db, connection, CURSOR_NAME)
    state = _state(cursor)
    scan_from = date.fromisoformat(state["scan_from"]) if state.get("scan_from") else when.date()
    # The filter is date-granular with an unverified day boundary: start a day early and dedupe locally.
    updated_after = scan_from - timedelta(days=1)
    try:
        with client:
            _scan(db, owner, connection, client, state, report, when, updated_after=updated_after)
    except GranolaError as error:
        _pause(connection, report, error, when)
    else:
        state["scan_from"] = when.date().isoformat()
        horizon = (updated_after - timedelta(days=2)).isoformat()
        state["seen"] = {k: v for k, v in state["seen"].items() if (v.get("updated_at") or "") >= horizon}
        report.advanced = True
        _set_health(connection, "ok", None, when)
    report.watermark = state.get("scan_from")
    cursor.value, cursor.updated_at = state, when
    db.flush()
    return report


def backfill(db, owner: str, connection: ContextConnection, days: int = 90, *, now: datetime | None = None,
             transport: httpx.BaseTransport | None = None, sleep: Callable[[float], None] = time.sleep) -> SyncReport:
    """Import notes created in the last `days`. Callers run it below new-meeting work in priority and send one
    consolidated notice from the report. It shares the incremental scan's dedupe state, so rerunning an
    interrupted backfill skips what was delivered; completion is recorded so a finished one is not repeated."""
    report = SyncReport()
    if disabled := _check(owner, connection):
        report.status = disabled
        return report
    when = now or datetime.now(UTC)
    progress = _cursor(db, connection, BACKFILL_CURSOR_NAME)
    if (progress.value or {}).get("completed_at") and progress.value.get("days", 0) >= days:
        report.status = "complete"
        return report
    client = _client_for(connection, report, when, transport, sleep)
    if client is None:
        return report
    notes = _cursor(db, connection, CURSOR_NAME)
    state = _state(notes)
    created_after = (progress.value or {}).get("created_after") or (when.date() - timedelta(days=days)).isoformat()
    try:
        with client:
            _scan(db, owner, connection, client, state, report, when,
                  created_after=date.fromisoformat(created_after))
    except GranolaError as error:
        _pause(connection, report, error, when)
        progress.value = {"created_after": created_after, "days": days}
    else:
        progress.value = {"created_after": created_after, "days": days, "completed_at": when.isoformat()}
        report.advanced = True
        _set_health(connection, "ok", None, when)
    progress.updated_at = notes.updated_at = when
    notes.value = state
    db.flush()
    return report
