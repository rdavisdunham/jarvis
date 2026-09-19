"""Saved note filters and source-backed extraction; never personal memory."""

from datetime import datetime

from sqlalchemy import Boolean, DateTime, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from .models import Base, now, uid


class NoteList(Base):
    __tablename__ = "note_lists"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=uid)
    owner_id: Mapped[str] = mapped_column(String(100), index=True)
    name: Mapped[str] = mapped_column(String(80))
    description: Mapped[str] = mapped_column(Text)
    filters: Mapped[dict] = mapped_column(JSONB, default=dict)
    automatic: Mapped[bool] = mapped_column(Boolean, default=True)
    extract_entries: Mapped[bool] = mapped_column(Boolean, default=True)
    archived: Mapped[bool] = mapped_column(Boolean, default=False)
    revision: Mapped[int] = mapped_column(Integer, default=1)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)


class NoteOrganization(Base):
    __tablename__ = "note_organizations"
    note_id: Mapped[str] = mapped_column(ForeignKey("notes.id"), primary_key=True)
    owner_id: Mapped[str] = mapped_column(String(100), index=True)
    tags_locked: Mapped[bool] = mapped_column(Boolean, default=False)
    generated: Mapped[bool] = mapped_column(Boolean, default=False)
    status: Mapped[str] = mapped_column(String(30), default="idle")
    fingerprint: Mapped[str] = mapped_column(String(64), default="")
    result: Mapped[dict] = mapped_column(JSONB, default=dict)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)


class NoteEntrySource(Base):
    __tablename__ = "note_entry_sources"
    __table_args__ = (UniqueConstraint("owner_id", "source_id", "entry_key"),)
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=uid)
    owner_id: Mapped[str] = mapped_column(String(100), index=True)
    source_id: Mapped[str] = mapped_column(ForeignKey("notes.id"), index=True)
    entry_id: Mapped[str] = mapped_column(ForeignKey("notes.id"), index=True)
    entry_key: Mapped[str] = mapped_column(String(64))
    evidence: Mapped[str] = mapped_column(Text)
    source_revision: Mapped[int] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)
