"""Derived search documents and separate, account-scoped vocabulary evidence."""

from datetime import datetime
from sqlalchemy import Boolean, DateTime, Integer, String, Text, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column
from .models import Base, now, uid


class SearchDocument(Base):
    __tablename__ = "search_documents"
    __table_args__ = (UniqueConstraint("owner_id", "target_key"),)
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=uid)
    owner_id: Mapped[str] = mapped_column(String(100), index=True)
    target_key: Mapped[str] = mapped_column(String(300))
    fingerprint: Mapped[str] = mapped_column(String(64))
    content: Mapped[str] = mapped_column(Text)
    embedding_model: Mapped[str] = mapped_column(String(100))
    vectors: Mapped[list] = mapped_column(JSONB, default=list)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)


class SearchIndexState(Base):
    __tablename__ = "search_index_states"
    owner_id: Mapped[str] = mapped_column(String(100), primary_key=True)
    generation: Mapped[int] = mapped_column(Integer, default=1)
    indexed_generation: Mapped[int] = mapped_column(Integer, default=0)
    job_id: Mapped[str | None] = mapped_column(String(36))
    status: Mapped[str] = mapped_column(String(30), default="queued")
    document_count: Mapped[int] = mapped_column(Integer, default=0)
    error: Mapped[str | None] = mapped_column(String(200))
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)


class SearchPreference(Base):
    __tablename__ = "search_preferences"
    owner_id: Mapped[str] = mapped_column(String(100), primary_key=True)
    account_id: Mapped[str] = mapped_column(String(100), primary_key=True)
    learning: Mapped[bool] = mapped_column(Boolean, default=True)


class SearchAlias(Base):
    __tablename__ = "search_aliases"
    __table_args__ = (UniqueConstraint("owner_id", "account_id", "phrase", "target_key"),)
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=uid)
    owner_id: Mapped[str] = mapped_column(String(100), index=True)
    account_id: Mapped[str] = mapped_column(String(100), index=True)
    phrase: Mapped[str] = mapped_column(String(200))
    target_key: Mapped[str] = mapped_column(String(300))
    target_fingerprint: Mapped[str] = mapped_column(String(64))
    status: Mapped[str] = mapped_column(String(30), default="provisional")
    revision: Mapped[int] = mapped_column(Integer, default=1)
    review_fingerprint: Mapped[str | None] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)


class SearchSession(Base):
    __tablename__ = "search_sessions"
    __table_args__ = (UniqueConstraint("owner_id", "account_id", "request_key"),)
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=uid)
    owner_id: Mapped[str] = mapped_column(String(100), index=True)
    account_id: Mapped[str] = mapped_column(String(100), index=True)
    conversation_id: Mapped[str | None] = mapped_column(String(36), index=True)
    work_id: Mapped[str | None] = mapped_column(String(36), index=True)
    request_key: Mapped[str] = mapped_column(String(150))
    query: Mapped[str] = mapped_column(Text)
    candidates: Mapped[list] = mapped_column(JSONB, default=list)
    result_ids: Mapped[list] = mapped_column(JSONB, default=list)
    selected_key: Mapped[str | None] = mapped_column(String(300))
    phrase: Mapped[str | None] = mapped_column(String(200))
    selected_records: Mapped[list] = mapped_column(JSONB, default=list)
    alias_id: Mapped[str | None] = mapped_column(String(36))
    outcome: Mapped[str] = mapped_column(String(30), default="unknown")
    signal: Mapped[str | None] = mapped_column(String(100))
    presented_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    accepted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    suppressed: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)
