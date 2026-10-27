"""External agent sessions (Claude Code, Codex, desktop apps) and the rules that file them."""

from datetime import datetime
from sqlalchemy import Boolean, DateTime, ForeignKey, Index, Integer, String, Text, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column
from .models import Base, now, uid


class AgentSession(Base):
    """A ledger entry: metadata, a short excerpt and a summary. Full transcripts stay on the user's machine."""
    __tablename__ = "agent_sessions"
    __table_args__ = (
        UniqueConstraint("owner_id", "tool", "session_id", name="uq_agent_session"),
        Index("ix_agent_sessions_owner_id_last_active_at", "owner_id", "last_active_at"),
    )
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=uid)
    owner_id: Mapped[str] = mapped_column(String(100), index=True)
    tool: Mapped[str] = mapped_column(String(30))
    session_id: Mapped[str] = mapped_column(String(100))
    machine: Mapped[str] = mapped_column(String(100), default="")
    environment: Mapped[str] = mapped_column(String(20), default="")
    cwd: Mapped[str] = mapped_column(Text, default="")
    repo: Mapped[str] = mapped_column(String(300), default="")
    branch: Mapped[str] = mapped_column(String(300), default="")
    title: Mapped[str] = mapped_column(String(500), default="")
    first_prompt: Mapped[str] = mapped_column(Text, default="")
    last_prompt: Mapped[str] = mapped_column(Text, default="")
    excerpt: Mapped[str] = mapped_column(Text, default="")
    summary: Mapped[str] = mapped_column(Text, default="")
    pr_urls: Mapped[list] = mapped_column(JSONB, default=list)
    issue_refs: Mapped[list] = mapped_column(JSONB, default=list)
    resume: Mapped[dict] = mapped_column(JSONB, default=dict)
    turn_count: Mapped[int] = mapped_column(Integer, default=0)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    last_active_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    # Explicit and manual filing is never replaced by automatic rebinding.
    binding_source: Mapped[str] = mapped_column(String(20), default="")
    record_id: Mapped[str | None] = mapped_column(ForeignKey("structure_records.id"), index=True)
    task_id: Mapped[str | None] = mapped_column(ForeignKey("tasks.id"), index=True)
    fingerprint: Mapped[str] = mapped_column(String(64), default="")
    summary_fingerprint: Mapped[str] = mapped_column(String(64), default="")
    embedding: Mapped[list | None] = mapped_column(JSONB)
    embedding_model: Mapped[str | None] = mapped_column(String(100))
    archived: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)


class AgentSessionRule(Base):
    """A folder or repo prefix that files matching sessions under one record (a client or project)."""
    __tablename__ = "agent_session_rules"
    __table_args__ = (UniqueConstraint("owner_id", "path_prefix", name="uq_agent_session_rule"),)
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=uid)
    owner_id: Mapped[str] = mapped_column(String(100), index=True)
    path_prefix: Mapped[str] = mapped_column(String(500))
    record_id: Mapped[str] = mapped_column(ForeignKey("structure_records.id"))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)
