"""Source lifecycle: receipt, revisions, segments and chunks, reads, deletion and the purge job body.

`receive` is the one ingestion entry point for every connector. Identity is (owner, connector, external_id).
The registry (`connectors.prepare`) keeps known connectors (`granola`, `zoom`, ...) as themselves, so a bridge
delivery and the in-app poller dedupe to one source; any other label becomes connector `api` with
`metadata["connector_label"]`, and its stored external ID is namespaced by that label (`recorder:123`) so labels
never collide. A revision's text is its segment texts joined by newlines, so `content_hash(texts)` is the
sha256 of that text and segment offsets index it. Receipt never calls a model. Transcript text is never logged.
"""

import math
import re
from dataclasses import dataclass
from typing import Literal

from sqlalchemy import func, select, update

from ..domain import DomainError, advisory, emit, enqueue_job
from ..models import now
from ..redaction import redact
from . import calendar_link
from . import settings as context_settings
from .connectors import LABEL_KEY
from .imports import text_segments
from .models import (
    ContextChunk,
    ContextConnection,
    ContextEntityLink,
    ContextEvidence,
    ContextRun,
    ContextSegment,
    ContextSource,
    ContextSourceRevision,
)
from .schema import Chunk, Segment, SourceEnvelope, content_hash

try:  # Package B5 implements this; until it lands, purge skips dependent-assertion propagation.
    from .evidence import propagate_source_deletion
except ImportError:
    propagate_source_deletion = None

MAX_SEGMENTS = 20_000
CHUNK_TOKENS = 5_000
CHARS_PER_TOKEN = 4
INGEST_JOB = "context_ingest"
DELETE_JOB = "context_delete"
CHANGED = "context.changed"  # views.changes: (source ID, current revision) on new revisions, refreshes, deletion
LABEL = re.compile(r"^[a-z0-9][a-z0-9_-]{0,39}$")
Change = Literal["created", "content", "metadata", "none", "deleted"]


@dataclass(frozen=True)
class ReceiveResult:
    source_id: str
    revision: int
    revision_id: str
    status: str
    change: Change
    job_id: str | None = None

    @property
    def created(self) -> bool:
        return self.change == "created"

    @property
    def revised(self) -> bool:
        return self.change in ("created", "content")

    @property
    def unchanged(self) -> bool:
        return self.change in ("none", "deleted")


def check_label(label: str, external_id: str) -> None:
    """Refuse connector labels and external IDs that cannot be stored."""
    if not LABEL.match(label.strip().lower()):
        raise DomainError("INVALID_ARGUMENT", "Connector labels are 1-40 lowercase letters, digits, - or _.")
    if not external_id or len(external_id) > 400 or any(c in external_id for c in "\x00\r\n"):
        raise DomainError("INVALID_ARGUMENT", "External IDs are 1-400 characters on one line.")


def _label(connector: str, metadata: dict) -> str | None:
    label = metadata.get(LABEL_KEY) if connector == "api" else None
    return label if isinstance(label, str) and label else None


def storage_id(envelope: SourceEnvelope) -> str:
    """The stored external ID: the upstream one, namespaced by the client label for `api` fallback labels."""
    label = _label(envelope.connector, envelope.metadata)
    stored = f"{label}:{envelope.external_id}" if label else envelope.external_id
    if len(stored) > 500:
        raise DomainError("INVALID_ARGUMENT", "The external ID is too long.")
    return stored


def split_external_id(source: ContextSource, revision: ContextSourceRevision | None) -> tuple[str, str]:
    """(label, upstream external ID) as the client sent them."""
    label = _label(source.connector, ((revision.payload if revision else None) or {}).get("metadata") or {})
    if label and source.external_id.startswith(label + ":"):
        return label, source.external_id[len(label) + 1:]
    return source.connector, source.external_id


# ---------------------------------------------------------------- building a revision


def _check_limits(envelope: SourceEnvelope) -> None:
    limit = context_settings.current().max_source_chars
    texts = [s.text for s in envelope.segments or []] + [envelope.text or "", envelope.private_notes or ""]
    size = sum(map(len, texts))
    if size > limit:
        raise DomainError("SOURCE_TOO_LARGE", f"This source has {size} characters; the limit is {limit}.", 413,
                          data={"max_source_chars": limit, "max_segments": MAX_SEGMENTS})
    if envelope.segments is not None and len(envelope.segments) > MAX_SEGMENTS:
        raise DomainError("SOURCE_TOO_LARGE", f"This source has more than {MAX_SEGMENTS} segments.", 413,
                          data={"max_source_chars": limit, "max_segments": MAX_SEGMENTS})


def _paragraphs(text: str) -> list[str]:
    return [p.strip() for p in re.split(r"\n\s*\n", text.replace("\r\n", "\n")) if p.strip()]


def build_segments(envelope: SourceEnvelope) -> list[Segment]:
    """Transcript segments (renumbered from 0, redacted) then private notes as owner `private_note` segments,
    with offsets into the revision text."""
    transcript = envelope.segments if envelope.segments is not None else text_segments(envelope.text)
    if len(transcript) > MAX_SEGMENTS:
        raise DomainError("SOURCE_TOO_LARGE", f"This source has more than {MAX_SEGMENTS} segments.", 413)
    drafts = [(redact(s.text), {"kind": "transcript", "speaker_label": s.speaker_label, "attribution": s.attribution,
                                "start_ms": s.start_ms, "end_ms": s.end_ms}) for s in transcript]
    drafts += [(redact(p), {"kind": "private_note", "attribution": "owner"})
               for p in _paragraphs(envelope.private_notes or "")]
    segments, offset = [], 0
    for seq, (text, fields) in enumerate(drafts):
        segments.append(Segment(seq=seq, text=text, char_start=offset, char_end=offset + len(text), **fields))
        offset += len(text) + 1  # the newline joining segment texts
    return segments


def _tokens(text: str) -> int:
    return math.ceil(len(text) / CHARS_PER_TOKEN)


def _line(segment: Segment) -> str:
    return f"{segment.speaker_label}: {segment.text}" if segment.speaker_label else segment.text


def _chunk_segments(segments: list[Segment], budget: int = CHUNK_TOKENS) -> list[Chunk]:
    """Greedy consecutive packing to about `budget` tokens. A chunk never mixes transcript and private notes and
    never splits a segment unless that segment alone exceeds the budget. Package B3 replaces this."""
    chunks: list[Chunk] = []
    group: list[Segment] = []

    def flush():
        if group:
            text = "\n".join(map(_line, group))
            chunks.append(Chunk(seq=len(chunks), segment_start=group[0].seq, segment_end=group[-1].seq,
                                token_count=_tokens(text), text=text))
            group.clear()

    for segment in segments:
        size = _tokens(_line(segment))
        if size > budget:
            flush()
            line, step = _line(segment), budget * CHARS_PER_TOKEN
            for start in range(0, len(line), step):
                piece = line[start:start + step]
                chunks.append(Chunk(seq=len(chunks), segment_start=segment.seq, segment_end=segment.seq,
                                    token_count=_tokens(piece), text=piece))
            continue
        if group and (group[-1].kind != segment.kind or _tokens("\n".join(map(_line, [*group, segment]))) > budget):
            flush()
        group.append(segment)
    flush()
    return chunks


# ---------------------------------------------------------------- receipt


def _display(envelope: SourceEnvelope) -> dict:
    """Revision fields that may change without new content: refreshed in place, never re-extracted."""
    return {
        "occurred_at": envelope.occurred_at,
        "ended_at": envelope.ended_at,
        "timezone": envelope.timezone,
        "calendar_event_id": envelope.calendar_event_id,
        "external_revision": envelope.external_revision,
        "participants": [p.model_dump(mode="json") for p in envelope.participants],
        "summary": redact(envelope.summary) if envelope.summary else None,
    }


def _source_display(envelope: SourceEnvelope) -> dict:
    return {"title": redact(envelope.title), "upstream_owner_email": envelope.upstream_owner_email,
            "upstream_owner_name": envelope.upstream_owner_name}


def _payload(envelope: SourceEnvelope) -> dict:
    return {"metadata": envelope.metadata, "envelope_content_hash": envelope.content_hash}


def _differs(row, values: dict) -> bool:
    return any(getattr(row, key) != value for key, value in values.items())


def _find(db, owner, connector, external_id):
    return db.scalar(select(ContextSource).where(ContextSource.owner_id == owner,
                                                 ContextSource.connector == connector,
                                                 ContextSource.external_id == external_id))


def _current(db, source):
    return db.scalar(select(ContextSourceRevision).where(ContextSourceRevision.source_id == source.id,
                                                         ContextSourceRevision.revision == source.current_revision))


def receive(db, owner: str, envelope: SourceEnvelope, connection_id: str | None = None) -> ReceiveResult:
    """Persist one delivery idempotently. Same content: nothing new (metadata refreshed in place if it changed).
    New content: an immutable revision with segments and chunks, plus one `context_ingest` job. A deleted
    source ignores redelivery of any content it already held (tombstone); different content restores it."""
    _check_limits(envelope)
    if connection_id is not None:
        connection = db.get(ContextConnection, connection_id)
        if connection is None or connection.owner_id != owner:
            raise DomainError("NOT_FOUND", "That connector connection was not found.", 404)
    segments = build_segments(envelope)
    digest = content_hash(s.text for s in segments)
    external_id = storage_id(envelope)
    advisory(db, f"context-source:{owner}:{envelope.connector}:{external_id}")
    source = _find(db, owner, envelope.connector, external_id)
    stamp = now()
    if source is not None:
        current = _current(db, source)
        if source.status == "deleted":
            held = set(db.scalars(select(ContextSourceRevision.content_hash)
                                  .where(ContextSourceRevision.source_id == source.id)))
            if digest in held:
                return ReceiveResult(source.id, source.current_revision, current.id, "deleted", "deleted")
            source.deleted_at = None
        elif current is not None and current.content_hash == digest:
            display, source_display = _display(envelope), _source_display(envelope)
            payload = {**current.payload, **_payload(envelope)}
            if not (_differs(current, display) or _differs(source, source_display) or payload != current.payload):
                return ReceiveResult(source.id, source.current_revision, current.id, source.status, "none")
            for key, value in display.items():
                setattr(current, key, value)
            for key, value in source_display.items():
                setattr(source, key, value)
            current.payload = {**payload, "metadata_updated_at": stamp.isoformat()}
            source.updated_at = stamp
            if connection_id:
                source.connection_id = connection_id
            db.flush()
            calendar_link.link_source(db, owner, source.id, current.id, envelope)
            emit(db, owner, CHANGED, source.id, source.current_revision)
            return ReceiveResult(source.id, source.current_revision, current.id, source.status, "metadata")
    created = source is None
    if created:
        source = ContextSource(owner_id=owner, connector=envelope.connector, external_id=external_id,
                               current_revision=0, created_at=stamp)
        db.add(source)
    source.connection_id = connection_id or source.connection_id
    for key, value in _source_display(envelope).items():
        setattr(source, key, value)
    source.current_revision += 1
    db.flush()
    revision = ContextSourceRevision(owner_id=owner, source_id=source.id, revision=source.current_revision,
                                     content_hash=digest, change_kind="content", payload=_payload(envelope),
                                     received_at=stamp, created_at=stamp, **_display(envelope))
    db.add(revision)
    db.flush()
    db.add_all(ContextSegment(owner_id=owner, revision_id=revision.id, seq=s.seq, kind=s.kind,
                              speaker_label=s.speaker_label, speaker_attribution=s.attribution, text=s.text,
                              start_ms=s.start_ms, end_ms=s.end_ms, char_start=s.char_start, char_end=s.char_end)
               for s in segments)
    db.add_all(ContextChunk(owner_id=owner, revision_id=revision.id, seq=c.seq, segment_start=c.segment_start,
                            segment_end=c.segment_end, token_count=c.token_count, text=c.text)
               for c in _chunk_segments(segments))
    job = enqueue_job(db, owner, INGEST_JOB, {"source_id": source.id, "revision_id": revision.id})
    source.status, source.updated_at = "queued", stamp
    db.flush()
    calendar_link.link_source(db, owner, source.id, revision.id, envelope)
    emit(db, owner, CHANGED, source.id, source.current_revision)
    return ReceiveResult(source.id, revision.revision, revision.id, source.status,
                         "created" if created else "content", job.id)


# ---------------------------------------------------------------- reads


def owned(db, owner: str, source_id: str, *, include_deleted: bool = False) -> ContextSource:
    source = db.get(ContextSource, str(source_id))
    if source is None or source.owner_id != owner or (source.status == "deleted" and not include_deleted):
        raise DomainError("NOT_FOUND", "That source was not found.", 404)
    return source


def _latest_runs(db, source_ids: list[str], *conditions) -> dict[str, ContextRun]:
    """Each source's newest run (matching `conditions`)."""
    if not source_ids:
        return {}
    ranked = (select(ContextRun.id, func.row_number().over(partition_by=ContextRun.source_id,
                                                           order_by=ContextRun.created_at.desc()).label("n"))
              .where(ContextRun.source_id.in_(source_ids), *conditions).subquery())
    rows = db.scalars(select(ContextRun).join(ranked, ranked.c.id == ContextRun.id).where(ranked.c.n == 1))
    return {run.source_id: run for run in rows}


def _last_errors(db, source_ids: list[str]) -> dict[str, ContextRun]:
    return _latest_runs(db, source_ids, ContextRun.error.is_not(None))


def public(source: ContextSource, revision: ContextSourceRevision | None, run: ContextRun | None,
           failure: ContextRun | None, segment_count: int) -> dict:
    label, external_id = split_external_id(source, revision)
    return {
        "id": source.id,
        "connector": source.connector,
        "label": label,
        "external_id": external_id,
        "title": source.title,
        "kind": source.kind,
        "status": source.status,
        "availability": source.availability,
        "revision": source.current_revision,
        "occurred_at": revision.occurred_at.isoformat() if revision and revision.occurred_at else None,
        "calendar_event_id": revision.calendar_event_id if revision else None,
        "segment_count": segment_count,
        "processing": None if run is None else {
            "stage": run.stage, "status": run.status, "coverage": run.coverage, "attempts": run.attempts,
            "updated_at": run.updated_at.isoformat()},
        "last_error": None if failure is None else {
            "stage": failure.stage, "status": failure.status, "message": failure.error,
            "retryable": failure.status not in ("failed", "cancelled"), "at": failure.updated_at.isoformat()},
        "created_at": source.created_at.isoformat(),
        "updated_at": source.updated_at.isoformat(),
    }


def list_sources(db, owner: str, *, status: str | None = None, limit: int = 50, offset: int = 0) -> dict:
    """Sources newest first, with status, coverage and last error; never segment text. Deleted ones are hidden."""
    query = select(ContextSource).where(ContextSource.owner_id == owner, ContextSource.status != "deleted")
    if status:
        query = query.where(ContextSource.status == status)
    total = db.scalar(select(func.count()).select_from(query.subquery()))
    sources = list(db.scalars(query.order_by(ContextSource.updated_at.desc(), ContextSource.id)
                              .limit(limit).offset(offset)))
    ids = [s.id for s in sources]
    revisions = {r.source_id: r for r in db.scalars(
        select(ContextSourceRevision).join(ContextSource, (ContextSource.id == ContextSourceRevision.source_id)
                                           & (ContextSource.current_revision == ContextSourceRevision.revision))
        .where(ContextSource.id.in_(ids)))} if ids else {}
    counts = dict(db.execute(select(ContextSegment.revision_id, func.count()).where(
        ContextSegment.revision_id.in_([r.id for r in revisions.values()])).group_by(ContextSegment.revision_id)
    ).all()) if revisions else {}
    runs, failures = _latest_runs(db, ids), _last_errors(db, ids)
    items = [public(s, revisions.get(s.id), runs.get(s.id), failures.get(s.id),
                    counts.get(revisions[s.id].id, 0) if s.id in revisions else 0) for s in sources]
    return {"items": items, "total": total, "limit": limit, "offset": offset}


def get_source(db, owner: str, source_id: str) -> dict:
    """One source with its revisions, participants, display summary and current segments."""
    source = owned(db, owner, source_id)
    revisions = list(db.scalars(select(ContextSourceRevision).where(ContextSourceRevision.source_id == source.id)
                                .order_by(ContextSourceRevision.revision)))
    current = next((r for r in revisions if r.revision == source.current_revision), None)
    segments = list(db.scalars(select(ContextSegment).where(ContextSegment.revision_id == current.id)
                               .order_by(ContextSegment.seq))) if current else []
    run, failure = _latest_runs(db, [source.id]).get(source.id), _last_errors(db, [source.id]).get(source.id)
    return {
        **public(source, current, run, failure, len(segments)),
        "ended_at": current.ended_at.isoformat() if current and current.ended_at else None,
        "timezone": current.timezone if current else None,
        "participants": current.participants if current else [],
        "summary": current.summary if current else None,  # upstream summary: display only, never evidence
        "upstream_owner": {"email": source.upstream_owner_email, "name": source.upstream_owner_name},
        "revisions": [{"revision": r.revision, "id": r.id, "content_hash": r.content_hash,
                       "external_revision": r.external_revision, "change_kind": r.change_kind,
                       "received_at": r.received_at.isoformat()} for r in revisions],
        "segments": [{"id": s.id, "seq": s.seq, "kind": s.kind, "speaker_label": s.speaker_label,
                      "attribution": s.speaker_attribution, "speaker_actor_id": s.speaker_actor_id,
                      "speaker_link_status": s.speaker_link_status, "text": s.text, "start_ms": s.start_ms,
                      "end_ms": s.end_ms, "char_start": s.char_start, "char_end": s.char_end} for s in segments],
    }


# ---------------------------------------------------------------- deletion


def delete_source(db, owner: str, source_id: str) -> dict:
    """Exclude the source from every read now and queue the content purge. Idempotent."""
    source = owned(db, owner, source_id, include_deleted=True)
    advisory(db, f"context-source:{owner}:{source.connector}:{source.external_id}")
    db.refresh(source)
    job_id = None
    if source.status != "deleted":
        stamp = now()
        source.status, source.deleted_at, source.updated_at = "deleted", stamp, stamp
        job_id = enqueue_job(db, owner, DELETE_JOB, {"source_id": source.id,
                                                     "through_revision": source.current_revision}).id
        db.flush()
        emit(db, owner, CHANGED, source.id, source.current_revision)
    return {"id": source.id, "status": source.status, "deleted_at": source.deleted_at.isoformat(), "job_id": job_id}


def purge_source(db, owner: str, source_id: str, through_revision: int | None = None) -> dict:
    """The `context_delete` job body: blank content of the source's revisions up to `through_revision`
    (default: all) in segments, chunks, embeddings, evidence quotes, links and run results; mark briefs that
    depend on it stale. Keeps a non-content tombstone: connector, external ID and revision content hashes."""
    source = db.get(ContextSource, str(source_id))
    if source is None or source.owner_id != owner:
        return {"purged": False}
    advisory(db, f"context-source:{owner}:{source.connector}:{source.external_id}")
    db.refresh(source)
    query = select(ContextSourceRevision.id).where(ContextSourceRevision.source_id == source.id)
    if through_revision is not None:
        query = query.where(ContextSourceRevision.revision <= through_revision)
    revision_ids = list(db.scalars(query))
    if source.status == "deleted":
        source.title, source.upstream_owner_email, source.upstream_owner_name = "", None, None
    if revision_ids:
        db.execute(update(ContextSourceRevision).where(ContextSourceRevision.id.in_(revision_ids)).values(
            payload={"purged": True}, participants=[], summary=None, calendar_event_id=None))
        segments = db.execute(update(ContextSegment).where(ContextSegment.revision_id.in_(revision_ids)).values(
            text="", speaker_label=None, speaker_provenance={})).rowcount
        chunks = db.execute(update(ContextChunk).where(ContextChunk.revision_id.in_(revision_ids)).values(
            text="", embedding=None, embedding_model=None, embedded_at=None, updated_at=now())).rowcount
        evidence = db.execute(update(ContextEvidence).where(ContextEvidence.revision_id.in_(revision_ids)).values(
            quote="")).rowcount
        db.execute(update(ContextRun).where(ContextRun.revision_id.in_(revision_ids)).values(stage_results={}))
    else:
        segments = chunks = evidence = 0
    db.execute(update(ContextEntityLink).where(ContextEntityLink.owner_id == owner,
                                               ContextEntityLink.subject_type == "source",
                                               ContextEntityLink.subject_id == source.id).values(mention=""))
    from . import apply, briefs

    supported = list(db.scalars(select(ContextEvidence.assertion_id).where(
        ContextEvidence.owner_id == owner, ContextEvidence.source_id == source.id)))
    before = apply.versions(db, owner, supported)
    if propagate_source_deletion is not None:
        propagate_source_deletion(db, owner, source.id)
    apply.emit_assertions(db, owner, supported, before)  # retracted (or rechained) by the deletion
    stale = briefs.mark_stale(db, owner, source_id=source.id)
    db.flush()
    return {"purged": True, "revisions": len(revision_ids), "segments": segments, "chunks": chunks,
            "evidence": evidence, "stale_briefs": stale}
