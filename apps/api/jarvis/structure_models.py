"""User-defined organization; typed task/note capabilities retain their services."""

from datetime import datetime
from sqlalchemy import Boolean, DateTime, ForeignKey, Integer, String, Text, UniqueConstraint
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
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=uid)
    owner_id: Mapped[str] = mapped_column(String(100), index=True)
    type_id: Mapped[str] = mapped_column(String(80), index=True)
    title: Mapped[str] = mapped_column(String(500))
    body: Mapped[str] = mapped_column(Text, default="")
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


class FieldUnderstanding(Base):
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
