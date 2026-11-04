"""Context platform tables (plan_final.md sections 3-6). Migrations 0048-0050 mirror this module exactly.

Module boundary: Eridani entities are referenced as `(entity_type, entity_id)` strings (actors, tasks, projects,
calendar events), never foreign keys; foreign keys only join context tables. Enumerated values are plain strings
whose value sets live in `context/schema.py` as Literals. Embeddings are pgvector `vector(512)` projections with
the producing model recorded beside them (`context/settings.py`); exact cosine search, no ANN index yet.
"""

from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    DDL,
    BigInteger,
    Boolean,
    Computed,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    Text,
    UniqueConstraint,
    event,
)
from sqlalchemy import text as sql
from sqlalchemy.dialects.postgresql import JSONB, TSVECTOR
from sqlalchemy.orm import Mapped, mapped_column

from ..models import Base, now, uid

# Any `create_all` on a fresh database (tests, eval and validation scripts) needs pgvector before the context tables;
# migrations create it in 0048. Production and CI images ship the extension.
event.listen(Base.metadata, "before_create", DDL("CREATE EXTENSION IF NOT EXISTS vector"))

DIMENSIONS = 512
# Identifier namespaces that may name several actors (plan section 3, People); every other namespace is strong.
AMBIGUOUS_NAMESPACES = ("alias", "calendar_name")


def _tsv(column):
    return mapped_column(TSVECTOR, Computed(f"to_tsvector('english', {column})", persisted=True))


def _gin(table):
    return Index(f"ix_{table}_tsv", "tsv", postgresql_using="gin")


# Stage 1: sources (migration 0048_context_sources)


class ContextConnection(Base):
    """A connector account. Credentials are Fernet ciphertext (integration_encryption_key), never plaintext."""
    __tablename__ = "context_connections"
    __table_args__ = (UniqueConstraint("owner_id", "connector", "external_account", name="uq_context_connections_account"),)
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=uid)
    owner_id: Mapped[str] = mapped_column(String(100), index=True)
    connector: Mapped[str] = mapped_column(String(20))
    external_account: Mapped[str] = mapped_column(String(320))
    credential_ciphertext: Mapped[str | None] = mapped_column(Text)
    checkpoint: Mapped[dict] = mapped_column(JSONB, default=dict, server_default="{}")
    health: Mapped[str] = mapped_column(String(20), default="unknown", server_default="unknown")
    last_error: Mapped[str | None] = mapped_column(Text)
    last_success_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    enabled: Mapped[bool] = mapped_column(Boolean, default=True, server_default="true")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)


class ContextSource(Base):
    __tablename__ = "context_sources"
    __table_args__ = (UniqueConstraint("owner_id", "connector", "external_id", name="uq_context_sources_external"),)
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=uid)
    owner_id: Mapped[str] = mapped_column(String(100), index=True)
    connection_id: Mapped[str | None] = mapped_column(ForeignKey("context_connections.id"), index=True)
    connector: Mapped[str] = mapped_column(String(20))
    external_id: Mapped[str] = mapped_column(String(500))
    kind: Mapped[str] = mapped_column(String(30), default="meeting", server_default="meeting")
    title: Mapped[str] = mapped_column(String(500), default="", server_default="")
    upstream_owner_email: Mapped[str | None] = mapped_column(String(320))
    upstream_owner_name: Mapped[str | None] = mapped_column(String(200))
    current_revision: Mapped[int] = mapped_column(Integer, default=0, server_default="0")
    status: Mapped[str] = mapped_column(String(20), default="received", server_default="received")
    availability: Mapped[str] = mapped_column(String(20), default="available", server_default="available")
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)


class ContextSourceRevision(Base):
    """Immutable. `change_kind` is content (re-extract) or metadata (display refresh only)."""
    __tablename__ = "context_source_revisions"
    __table_args__ = (UniqueConstraint("source_id", "revision", name="uq_context_source_revisions_revision"),)
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=uid)
    owner_id: Mapped[str] = mapped_column(String(100), index=True)
    source_id: Mapped[str] = mapped_column(ForeignKey("context_sources.id"))
    revision: Mapped[int] = mapped_column(Integer)
    content_hash: Mapped[str] = mapped_column(String(64))
    external_revision: Mapped[str | None] = mapped_column(String(200))
    change_kind: Mapped[str] = mapped_column(String(20), default="content", server_default="content")
    payload: Mapped[dict] = mapped_column(JSONB, default=dict, server_default="{}")
    occurred_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    ended_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    timezone: Mapped[str | None] = mapped_column(String(64))
    calendar_event_id: Mapped[str | None] = mapped_column(String(300))
    upstream_updated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    participants: Mapped[list] = mapped_column(JSONB, default=list, server_default="[]")
    summary: Mapped[str | None] = mapped_column(Text)  # Upstream summary: display only, never evidence.
    received_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)


class ContextSegment(Base):
    """The citation unit: exact text with offsets into the revision. Lexical search only; not embedded."""
    __tablename__ = "context_segments"
    __table_args__ = (UniqueConstraint("revision_id", "seq", name="uq_context_segments_seq"), _gin("context_segments"))
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=uid)
    owner_id: Mapped[str] = mapped_column(String(100), index=True)
    revision_id: Mapped[str] = mapped_column(ForeignKey("context_source_revisions.id"))
    seq: Mapped[int] = mapped_column(Integer)
    kind: Mapped[str] = mapped_column(String(20), default="transcript", server_default="transcript")
    speaker_label: Mapped[str | None] = mapped_column(String(200))
    speaker_attribution: Mapped[str | None] = mapped_column(String(10))  # owner | other, when the connector knows
    speaker_actor_id: Mapped[str | None] = mapped_column(String(36))
    speaker_link_status: Mapped[str] = mapped_column(String(20), default="unresolved", server_default="unresolved")
    speaker_provenance: Mapped[dict] = mapped_column(JSONB, default=dict, server_default="{}")
    text: Mapped[str] = mapped_column(Text)
    start_ms: Mapped[int | None] = mapped_column(Integer)
    end_ms: Mapped[int | None] = mapped_column(Integer)
    char_start: Mapped[int] = mapped_column(Integer)
    char_end: Mapped[int] = mapped_column(Integer)
    tsv: Mapped[str] = _tsv("text")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)


class ContextChunk(Base):
    """Extraction and embedding unit over an inclusive segment `seq` range."""
    __tablename__ = "context_chunks"
    __table_args__ = (UniqueConstraint("revision_id", "seq", name="uq_context_chunks_seq"), _gin("context_chunks"))
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=uid)
    owner_id: Mapped[str] = mapped_column(String(100), index=True)
    revision_id: Mapped[str] = mapped_column(ForeignKey("context_source_revisions.id"))
    seq: Mapped[int] = mapped_column(Integer)
    segment_start: Mapped[int] = mapped_column(Integer)
    segment_end: Mapped[int] = mapped_column(Integer)
    token_count: Mapped[int] = mapped_column(Integer, default=0, server_default="0")
    text: Mapped[str] = mapped_column(Text)
    embedding: Mapped[list[float] | None] = mapped_column(Vector(DIMENSIONS))
    embedding_model: Mapped[str | None] = mapped_column(String(100))
    embedded_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    tsv: Mapped[str] = _tsv("text")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)


class ContextRun(Base):
    """One processing stage attempt. `stage_results` persists completed work so a retry skips model calls;
    `epoch` is the owner's processing epoch at start (ContextOwnerState.epoch); a stale epoch never applies."""
    __tablename__ = "context_runs"
    __table_args__ = (Index("ix_context_runs_owner_stage_status", "owner_id", "stage", "status"),)
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=uid)
    owner_id: Mapped[str] = mapped_column(String(100), index=True)
    stage: Mapped[str] = mapped_column(String(20))
    status: Mapped[str] = mapped_column(String(20), default="queued", server_default="queued")
    job_id: Mapped[str | None] = mapped_column(String(36))
    source_id: Mapped[str | None] = mapped_column(ForeignKey("context_sources.id"), index=True)
    revision_id: Mapped[str | None] = mapped_column(ForeignKey("context_source_revisions.id"))
    entity_type: Mapped[str | None] = mapped_column(String(50))
    entity_id: Mapped[str | None] = mapped_column(String(100))
    engine: Mapped[str] = mapped_column(String(60), default="native", server_default="native")
    model: Mapped[str | None] = mapped_column(String(100))
    prompt_version: Mapped[str | None] = mapped_column(String(60))
    config_version: Mapped[str | None] = mapped_column(String(60))
    decision_watermark: Mapped[int] = mapped_column(BigInteger, default=0, server_default="0")
    epoch: Mapped[int] = mapped_column(Integer, default=0, server_default="0")
    coverage: Mapped[dict] = mapped_column(JSONB, default=dict, server_default="{}")
    stage_results: Mapped[dict] = mapped_column(JSONB, default=dict, server_default="{}")
    attempts: Mapped[int] = mapped_column(Integer, default=0, server_default="0")
    input_tokens: Mapped[int] = mapped_column(Integer, default=0, server_default="0")
    output_tokens: Mapped[int] = mapped_column(Integer, default=0, server_default="0")
    cost_usd: Mapped[float] = mapped_column(Numeric(12, 6), default=0, server_default="0")
    error: Mapped[str | None] = mapped_column(Text)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)


class ContextCursor(Base):
    """A connector's resumable position (for example Granola's last successful scan date)."""
    __tablename__ = "context_cursors"
    __table_args__ = (UniqueConstraint("connection_id", "name", name="uq_context_cursors_name"),)
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=uid)
    owner_id: Mapped[str] = mapped_column(String(100), index=True)
    connection_id: Mapped[str] = mapped_column(ForeignKey("context_connections.id"))
    name: Mapped[str] = mapped_column(String(100))
    value: Mapped[dict] = mapped_column(JSONB, default=dict, server_default="{}")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)


# Stage 2: interpretation (migration 0049_context_assertions)


class ContextDecision(Base):
    """Append-only human, agent or system decision. Undo is a new row with `undoes_id`. `seq` is monotonic per
    owner, allocated from ContextOwnerState.decision_seq under that row's lock; runs record the highest seq they
    saw as `decision_watermark`."""
    __tablename__ = "context_decisions"
    __table_args__ = (
        UniqueConstraint("owner_id", "seq", name="uq_context_decisions_seq"),
        Index("ix_context_decisions_target", "owner_id", "target_type", "target_id"),
    )
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=uid)
    owner_id: Mapped[str] = mapped_column(String(100), index=True)
    seq: Mapped[int] = mapped_column(BigInteger)
    target_type: Mapped[str] = mapped_column(String(50))
    target_id: Mapped[str] = mapped_column(String(100))
    expected_revision: Mapped[int | None] = mapped_column(Integer)
    operation: Mapped[str] = mapped_column(String(20))
    reason: Mapped[str] = mapped_column(Text, default="", server_default="")
    negative: Mapped[bool] = mapped_column(Boolean, default=False, server_default="false")
    origin: Mapped[str] = mapped_column(String(20))
    principal: Mapped[str] = mapped_column(String(100))
    command_id: Mapped[str | None] = mapped_column(String(100), index=True)
    before: Mapped[dict] = mapped_column(JSONB, default=dict, server_default="{}")
    after: Mapped[dict] = mapped_column(JSONB, default=dict, server_default="{}")
    effective_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    recorded_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)
    undoes_id: Mapped[str | None] = mapped_column(ForeignKey("context_decisions.id"))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)


class ContextAssertion(Base):
    """Current head of an assertion; history lives in ContextAssertionVersion. Effective time is the half-open
    interval [valid_from, valid_to). `confidence` is diagnostic only. Commitment fields are null otherwise."""
    __tablename__ = "context_assertions"
    __table_args__ = (
        Index("ix_context_assertions_subject", "owner_id", "subject_type", "subject_id"),
        Index("ix_context_assertions_lifecycle", "owner_id", "lifecycle"),
        _gin("context_assertions"),
    )
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=uid)
    owner_id: Mapped[str] = mapped_column(String(100), index=True)
    kind: Mapped[str] = mapped_column(String(20))
    statement: Mapped[str] = mapped_column(Text)
    subject_type: Mapped[str] = mapped_column(String(50))
    subject_id: Mapped[str] = mapped_column(String(100))
    object_type: Mapped[str | None] = mapped_column(String(50))
    object_id: Mapped[str | None] = mapped_column(String(100))
    predicate: Mapped[str] = mapped_column(String(100))
    key: Mapped[str | None] = mapped_column(String(300))
    modality: Mapped[str] = mapped_column(String(20), default="asserted", server_default="asserted")
    explicit: Mapped[bool] = mapped_column(Boolean, default=True, server_default="true")
    valid_from: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    valid_to: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    valid_precision: Mapped[str] = mapped_column(String(20), default="unknown", server_default="unknown")
    lifecycle: Mapped[str] = mapped_column(String(20), default="current", server_default="current")
    current_version: Mapped[int] = mapped_column(Integer, default=1, server_default="1")
    supersedes_id: Mapped[str | None] = mapped_column(ForeignKey("context_assertions.id"))
    run_id: Mapped[str | None] = mapped_column(ForeignKey("context_runs.id"))
    confidence: Mapped[float | None] = mapped_column(Float)
    responsible: Mapped[list] = mapped_column(JSONB, default=list, server_default="[]")
    recipient: Mapped[str | None] = mapped_column(String(100))
    due_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    due_precision: Mapped[str | None] = mapped_column(String(20))
    completion_evidence: Mapped[list] = mapped_column(JSONB, default=list, server_default="[]")
    embedding: Mapped[list[float] | None] = mapped_column(Vector(DIMENSIONS))
    embedding_model: Mapped[str | None] = mapped_column(String(100))
    embedded_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    tsv: Mapped[str] = _tsv("statement")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)


class ContextAssertionVersion(Base):
    """Immutable transaction history: [recorded_from, recorded_to) is when Eridani believed this version."""
    __tablename__ = "context_assertion_versions"
    __table_args__ = (UniqueConstraint("assertion_id", "version", name="uq_context_assertion_versions_version"),)
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=uid)
    owner_id: Mapped[str] = mapped_column(String(100), index=True)
    assertion_id: Mapped[str] = mapped_column(ForeignKey("context_assertions.id"))
    version: Mapped[int] = mapped_column(Integer)
    statement: Mapped[str] = mapped_column(Text)
    lifecycle: Mapped[str] = mapped_column(String(20))
    valid_from: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    valid_to: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    valid_precision: Mapped[str] = mapped_column(String(20), default="unknown", server_default="unknown")
    recorded_from: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)
    recorded_to: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    decision_id: Mapped[str | None] = mapped_column(ForeignKey("context_decisions.id"))
    run_id: Mapped[str | None] = mapped_column(ForeignKey("context_runs.id"))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)


class ContextEvidence(Base):
    """A span supporting or contradicting one assertion version. `segment_id` is null only for derived evidence
    whose `derivation` names its dependencies."""
    __tablename__ = "context_evidence"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=uid)
    owner_id: Mapped[str] = mapped_column(String(100), index=True)
    assertion_id: Mapped[str] = mapped_column(ForeignKey("context_assertions.id"), index=True)
    assertion_version: Mapped[int] = mapped_column(Integer)
    source_id: Mapped[str] = mapped_column(ForeignKey("context_sources.id"), index=True)
    revision_id: Mapped[str] = mapped_column(ForeignKey("context_source_revisions.id"))
    segment_id: Mapped[str | None] = mapped_column(ForeignKey("context_segments.id"), index=True)
    quote: Mapped[str] = mapped_column(Text, default="", server_default="")
    char_start: Mapped[int | None] = mapped_column(Integer)
    char_end: Mapped[int | None] = mapped_column(Integer)
    role: Mapped[str] = mapped_column(String(20), default="support", server_default="support")
    origin: Mapped[str] = mapped_column(String(20), default="independent", server_default="independent")
    derivation: Mapped[dict] = mapped_column(JSONB, default=dict, server_default="{}")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)


class ContextEntityLink(Base):
    """A mention resolved (or not) to an Eridani entity. The subject is the linking record, usually
    ("assertion", id) or ("segment", id); the target is null while unresolved. Rejections persist."""
    __tablename__ = "context_entity_links"
    __table_args__ = (
        Index("ix_context_entity_links_subject", "owner_id", "subject_type", "subject_id"),
        Index("ix_context_entity_links_target", "owner_id", "target_type", "target_id"),
    )
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=uid)
    owner_id: Mapped[str] = mapped_column(String(100), index=True)
    assertion_id: Mapped[str | None] = mapped_column(ForeignKey("context_assertions.id"), index=True)
    subject_type: Mapped[str] = mapped_column(String(50))
    subject_id: Mapped[str] = mapped_column(String(100))
    target_type: Mapped[str | None] = mapped_column(String(50))
    target_id: Mapped[str | None] = mapped_column(String(100))
    mention: Mapped[str] = mapped_column(String(500), default="", server_default="")
    role: Mapped[str] = mapped_column(String(50))
    status: Mapped[str] = mapped_column(String(20), default="unresolved", server_default="unresolved")
    score: Mapped[float | None] = mapped_column(Float)
    margin: Mapped[float | None] = mapped_column(Float)
    evidence: Mapped[list] = mapped_column(JSONB, default=list, server_default="[]")
    revision: Mapped[int] = mapped_column(Integer, default=1, server_default="1")
    decision_id: Mapped[str | None] = mapped_column(ForeignKey("context_decisions.id"))
    run_id: Mapped[str | None] = mapped_column(ForeignKey("context_runs.id"))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)


class ContextActorIdentity(Base):
    """A namespaced identifier for an Eridani actor. A value in a strong namespace (email, linear_user,
    granola_speaker scoped to its source, organization) names one actor; aliases and calendar display names may
    name several. `source_scope` qualifies source-local identifiers; null means global (NULLS NOT DISTINCT)."""
    __tablename__ = "context_actor_identities"
    __table_args__ = (
        Index("uq_context_actor_identities_actor", "owner_id", "namespace", "normalized", "source_scope", "actor_id",
              unique=True, postgresql_nulls_not_distinct=True),
        Index("uq_context_actor_identities_strong", "owner_id", "namespace", "normalized", "source_scope",
              unique=True, postgresql_nulls_not_distinct=True,
              postgresql_where=sql("namespace NOT IN ('alias', 'calendar_name')")),
    )
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=uid)
    owner_id: Mapped[str] = mapped_column(String(100), index=True)
    actor_id: Mapped[str] = mapped_column(String(36), index=True)
    namespace: Mapped[str] = mapped_column(String(50))
    value: Mapped[str] = mapped_column(String(500))
    normalized: Mapped[str] = mapped_column(String(500))
    source_scope: Mapped[str | None] = mapped_column(String(100))
    status: Mapped[str] = mapped_column(String(20), default="inferred", server_default="inferred")
    provenance: Mapped[dict] = mapped_column(JSONB, default=dict, server_default="{}")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)


class ContextActorMerge(Base):
    """A reviewed merge; `snapshot` holds the loser's prior assignments so unmerge can restore them."""
    __tablename__ = "context_actor_merges"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=uid)
    owner_id: Mapped[str] = mapped_column(String(100), index=True)
    winner_actor_id: Mapped[str] = mapped_column(String(36))
    loser_actor_id: Mapped[str] = mapped_column(String(36))
    snapshot: Mapped[dict] = mapped_column(JSONB, default=dict, server_default="{}")
    decision_id: Mapped[str | None] = mapped_column(ForeignKey("context_decisions.id"))
    undone_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)


# Stage 3: review and briefs (migration 0050_context_review)


class ContextProposal(Base):
    """A persistent task proposal. `fingerprint` dedupes regeneration per source (lint proposals have no source;
    the constraint treats nulls as equal)."""
    __tablename__ = "context_proposals"
    __table_args__ = (
        UniqueConstraint("owner_id", "source_id", "fingerprint", name="uq_context_proposals_fingerprint",
                         postgresql_nulls_not_distinct=True),
        Index("ix_context_proposals_disposition", "owner_id", "disposition"),
    )
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=uid)
    owner_id: Mapped[str] = mapped_column(String(100), index=True)
    source_id: Mapped[str | None] = mapped_column(ForeignKey("context_sources.id"))
    revision_id: Mapped[str | None] = mapped_column(ForeignKey("context_source_revisions.id"))
    kind: Mapped[str] = mapped_column(String(20))
    fingerprint: Mapped[str] = mapped_column(String(64))
    title: Mapped[str] = mapped_column(String(500))
    payload: Mapped[dict] = mapped_column(JSONB, default=dict, server_default="{}")
    destination: Mapped[str] = mapped_column(String(20), default="eridani", server_default="eridani")
    responsible: Mapped[str | None] = mapped_column(String(36))
    due_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    assertion_ids: Mapped[list] = mapped_column(JSONB, default=list, server_default="[]")
    evidence: Mapped[list] = mapped_column(JSONB, default=list, server_default="[]")
    revision: Mapped[int] = mapped_column(Integer, default=1, server_default="1")
    disposition: Mapped[str] = mapped_column(String(20), default="pending", server_default="pending")
    preselected: Mapped[bool] = mapped_column(Boolean, default=False, server_default="false")
    task_id: Mapped[str | None] = mapped_column(String(36))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)


class ContextPublication(Base):
    """An approved proposal revision sent to its destination; `operation_id` makes publication idempotent."""
    __tablename__ = "context_publications"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=uid)
    owner_id: Mapped[str] = mapped_column(String(100), index=True)
    proposal_id: Mapped[str] = mapped_column(ForeignKey("context_proposals.id"), index=True)
    proposal_revision: Mapped[int] = mapped_column(Integer)
    payload: Mapped[dict] = mapped_column(JSONB, default=dict, server_default="{}")
    destination: Mapped[str] = mapped_column(String(20))
    principal: Mapped[str] = mapped_column(String(100))
    command_id: Mapped[str | None] = mapped_column(String(100))
    operation_id: Mapped[str] = mapped_column(String(100), unique=True)
    task_id: Mapped[str | None] = mapped_column(String(36))
    remote_issue_id: Mapped[str | None] = mapped_column(String(100))
    status: Mapped[str] = mapped_column(String(20), default="pending", server_default="pending")
    outcome: Mapped[dict] = mapped_column(JSONB, default=dict, server_default="{}")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)


class ContextBrief(Base):
    """A versioned entity brief. Identical dependency fingerprints never create a new version."""
    __tablename__ = "context_briefs"
    __table_args__ = (UniqueConstraint("owner_id", "entity_type", "entity_id", "version", name="uq_context_briefs_version"),)
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=uid)
    owner_id: Mapped[str] = mapped_column(String(100), index=True)
    entity_type: Mapped[str] = mapped_column(String(50))
    entity_id: Mapped[str] = mapped_column(String(100))
    sections: Mapped[list] = mapped_column(JSONB, default=list, server_default="[]")
    dependencies: Mapped[dict] = mapped_column(JSONB, default=dict, server_default="{}")
    watermarks: Mapped[dict] = mapped_column(JSONB, default=dict, server_default="{}")
    fingerprint: Mapped[str] = mapped_column(String(64))
    version: Mapped[int] = mapped_column(Integer, default=1, server_default="1")
    stale: Mapped[bool] = mapped_column(Boolean, default=False, server_default="false")
    run_id: Mapped[str | None] = mapped_column(ForeignKey("context_runs.id"))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)


class ContextOwnerState(Base):
    """Per-owner processing control. Bumping `epoch` invalidates in-flight runs started under an older one;
    `decision_seq` is the last ContextDecision.seq handed out (increment it under SELECT ... FOR UPDATE)."""
    __tablename__ = "context_owner_state"
    owner_id: Mapped[str] = mapped_column(String(100), primary_key=True)
    epoch: Mapped[int] = mapped_column(Integer, default=0, server_default="0")
    decision_seq: Mapped[int] = mapped_column(BigInteger, default=0, server_default="0")
    paused: Mapped[bool] = mapped_column(Boolean, default=False, server_default="false")
    paused_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    settings: Mapped[dict] = mapped_column(JSONB, default=dict, server_default="{}")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)
