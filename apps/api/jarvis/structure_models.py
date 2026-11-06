"""User-defined organization; typed task/note capabilities retain their services."""

from datetime import datetime
from sqlalchemy import Boolean, DateTime, ForeignKey, Float, Index, Integer, String, Text, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column
from .models import Base, now, uid


class StructureSchema(Base):
    __tablename__ = "structure_schemas"
    owner_id: Mapped[str] = mapped_column(String(100), primary_key=True)
    revision: Mapped[int] = mapped_column(Integer, default=1)
    definition: Mapped[dict] = mapped_column(JSONB, default=dict)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)


class StructureProposal(Base):
    __tablename__ = "structure_proposals"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=uid)
    owner_id: Mapped[str] = mapped_column(String(100), index=True)
    account_id: Mapped[str] = mapped_column(String(100))
    schema_revision: Mapped[int] = mapped_column(Integer)
    definition: Mapped[dict] = mapped_column(JSONB)
    impact: Mapped[dict] = mapped_column(JSONB)
    applied_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


class StructureRecord(Base):
    __tablename__ = "structure_records"
    __table_args__ = (Index("ix_structure_records_owner_id_next_review_at", "owner_id", "next_review_at"),)
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=uid)
    owner_id: Mapped[str] = mapped_column(String(100), index=True)
    type_id: Mapped[str] = mapped_column(String(80), index=True)
    title: Mapped[str] = mapped_column(String(500))
    body: Mapped[str] = mapped_column(Text, default="")
    sort_order: Mapped[float] = mapped_column(Float, default=0, server_default="0")
    local_notes: Mapped[str] = mapped_column(Text, default="", server_default="")
    values: Mapped[dict] = mapped_column(JSONB, default=dict)
    status_id: Mapped[str | None] = mapped_column(String(80))
    parent_id: Mapped[str | None] = mapped_column(ForeignKey("structure_records.id"), index=True)
    task_id: Mapped[str | None] = mapped_column(ForeignKey("tasks.id"), unique=True)
    note_id: Mapped[str | None] = mapped_column(ForeignKey("notes.id"), unique=True)
    legacy_kind: Mapped[str | None] = mapped_column(String(30))
    archived: Mapped[bool] = mapped_column(Boolean, default=False)
    revision: Mapped[int] = mapped_column(Integer, default=1)
    schema_revision: Mapped[int] = mapped_column(Integer, default=1)
    provenance: Mapped[dict] = mapped_column(JSONB, default=dict)
    # Review cadence state (see record_reviews). review_queued_at marks when the daily queue
    # surfaced a due review in Questions; it is history, never a second due date.
    last_reviewed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    next_review_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    review_queued_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    review_paused: Mapped[bool] = mapped_column(Boolean, default=False, server_default="false")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)


class StructureLink(Base):
    __tablename__ = "structure_links"
    __table_args__ = (UniqueConstraint("owner_id", "relationship_id", "source_id", "target_id"),)
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=uid)
    owner_id: Mapped[str] = mapped_column(String(100), index=True)
    relationship_id: Mapped[str] = mapped_column(String(80))
    source_id: Mapped[str] = mapped_column(ForeignKey("structure_records.id"), index=True)
    target_id: Mapped[str] = mapped_column(ForeignKey("structure_records.id"), index=True)


class RecordTemplate(Base):
    """A reusable starting structure for one type. Instances are independent copies."""
    __tablename__ = "record_templates"
    __table_args__ = (Index("ix_record_templates_owner_id_type_id", "owner_id", "type_id"),)
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=uid)
    owner_id: Mapped[str] = mapped_column(String(100))
    type_id: Mapped[str] = mapped_column(String(80))
    name: Mapped[str] = mapped_column(String(120))
    description: Mapped[str] = mapped_column(Text, default="")
    payload: Mapped[dict] = mapped_column(JSONB, default=dict)
    revision: Mapped[int] = mapped_column(Integer, default=1)
    archived: Mapped[bool] = mapped_column(Boolean, default=False)
    created_by: Mapped[str] = mapped_column(String(100))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)


class FieldUnderstanding(Base):
    deferred_until: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    __tablename__ = "field_understandings"
    owner_id: Mapped[str] = mapped_column(String(100), primary_key=True)
    definition_id: Mapped[str] = mapped_column(String(180), primary_key=True)
    fingerprint: Mapped[str] = mapped_column(String(64))
    status: Mapped[str] = mapped_column(String(30), default="assessing")
    understanding: Mapped[dict] = mapped_column(JSONB, default=dict)
    questions: Mapped[list] = mapped_column(JSONB, default=list)
    answers: Mapped[list] = mapped_column(JSONB, default=list)
    revision: Mapped[int] = mapped_column(Integer, default=1)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)


class RoutingObservation(Base):
    __tablename__ = "routing_observations"
    __table_args__ = (UniqueConstraint("owner_id", "source_key"),)
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=uid)
    owner_id: Mapped[str] = mapped_column(String(100), index=True)
    record_id: Mapped[str] = mapped_column(ForeignKey("structure_records.id"), index=True)
    source_key: Mapped[str] = mapped_column(String(160))
    record_revision: Mapped[int] = mapped_column(Integer)
    schema_revision: Mapped[int] = mapped_column(Integer)
    origin: Mapped[str] = mapped_column(String(30))
    evidence: Mapped[dict] = mapped_column(JSONB)
    suppressed: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)


class RoutingPattern(Base):
    __tablename__ = "routing_patterns"
    __table_args__ = (UniqueConstraint("owner_id", "fingerprint"),)
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=uid)
    owner_id: Mapped[str] = mapped_column(String(100), index=True)
    fingerprint: Mapped[str] = mapped_column(String(64))
    condition: Mapped[dict] = mapped_column(JSONB)
    assignment: Mapped[dict] = mapped_column(JSONB)
    evidence_ids: Mapped[list] = mapped_column(JSONB, default=list)
    reason: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(30), default="candidate")
    schema_revision: Mapped[int] = mapped_column(Integer)
    revision: Mapped[int] = mapped_column(Integer, default=1)
    origin: Mapped[str] = mapped_column(String(30), default="learned")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)


class RoutingReview(Base):
    __tablename__ = "routing_reviews"
    __table_args__ = (UniqueConstraint("owner_id", "period"),)
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=uid)
    owner_id: Mapped[str] = mapped_column(String(100), index=True)
    period: Mapped[str] = mapped_column(String(80))
    status: Mapped[str] = mapped_column(String(30), default="queued")
    summary: Mapped[dict] = mapped_column(JSONB, default=dict)
    questions: Mapped[list] = mapped_column(JSONB, default=list)
    answers: Mapped[list] = mapped_column(JSONB, default=list)
    revision: Mapped[int] = mapped_column(Integer, default=1)
    offered_on: Mapped[str | None] = mapped_column(String(10))
    deferred_until: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class QuickSortSession(Base):
    """Bounded filing interview; the ordinary queue executes its steps."""
    __tablename__ = "quick_sort_sessions"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=uid)
    owner_id: Mapped[str] = mapped_column(String(100), index=True)
    account_id: Mapped[str] = mapped_column(String(100), index=True)
    conversation_id: Mapped[str] = mapped_column(String(36))
    status: Mapped[str] = mapped_column(String(30), default="active")
    revision: Mapped[int] = mapped_column(Integer, default=1)
    state: Mapped[dict] = mapped_column(JSONB, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)


class RecordOverview(Base):
    """One version of a client or project's living overview (record_overviews.py). Versions are never edited."""
    __tablename__ = "record_overviews"
    __table_args__ = (UniqueConstraint("record_id", "version"),
                      Index("ix_record_overviews_owner_record", "owner_id", "record_id"))
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=uid)
    owner_id: Mapped[str] = mapped_column(String(100))
    record_id: Mapped[str] = mapped_column(ForeignKey("structure_records.id", ondelete="CASCADE"))
    version: Mapped[int] = mapped_column(Integer)
    sections: Mapped[dict] = mapped_column(JSONB, default=dict)
    seed: Mapped[str] = mapped_column(Text, default="")  # The owner's description this version was built on.
    mode: Mapped[str] = mapped_column(String(20))  # rebuild | update
    origin: Mapped[str] = mapped_column(String(20), default="scheduled")  # scheduled | manual
    stats: Mapped[dict] = mapped_column(JSONB, default=dict)
    model: Mapped[str | None] = mapped_column(String(80))
    prompt_version: Mapped[str | None] = mapped_column(String(40))
    cost: Mapped[float | None] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)


class RecordOverviewState(Base):
    """What the latest overview was built from (digests), and when the record is next looked at."""
    __tablename__ = "record_overview_states"
    record_id: Mapped[str] = mapped_column(ForeignKey("structure_records.id", ondelete="CASCADE"), primary_key=True)
    owner_id: Mapped[str] = mapped_column(String(100), index=True)
    seen: Mapped[dict] = mapped_column(JSONB, default=dict)
    checked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    next_check_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), index=True)
    last_error: Mapped[dict | None] = mapped_column(JSONB)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)


class PersonBucketLink(Base):
    """A person's connection to a client or project (people_links.py): non-exclusive, with evidence counts.
    Inferred links follow the evidence; the owner can confirm one or reject it, and that decision is kept."""
    __tablename__ = "person_bucket_links"
    __table_args__ = (UniqueConstraint("actor_id", "record_id"),
                      Index("ix_person_bucket_links_owner_record", "owner_id", "record_id"))
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=uid)
    owner_id: Mapped[str] = mapped_column(String(100), index=True)
    actor_id: Mapped[str] = mapped_column(ForeignKey("actors.id", ondelete="CASCADE"), index=True)
    record_id: Mapped[str] = mapped_column(ForeignKey("structure_records.id", ondelete="CASCADE"))
    tasks: Mapped[int] = mapped_column(Integer, default=0)
    meeting_facts: Mapped[int] = mapped_column(Integer, default=0)
    evidence: Mapped[dict] = mapped_column(JSONB, default=dict)
    status: Mapped[str] = mapped_column(String(20), default="inferred")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)
