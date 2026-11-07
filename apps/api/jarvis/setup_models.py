"""Set up with Dani state (docs/SETUP_WITH_DANI_PLAN.md). The transcript is the setup Conversation."""

from datetime import datetime

from sqlalchemy import DateTime, Float, ForeignKey, Integer, String
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from .models import Base, now, uid


class SetupSession(Base):
    """One row per owner, rewritten for each run of the interview."""
    __tablename__ = "setup_sessions"
    owner_id: Mapped[str] = mapped_column(String(100), primary_key=True)
    conversation_id: Mapped[str] = mapped_column(ForeignKey("conversations.id"), index=True)
    mode: Mapped[str] = mapped_column(String(20), default="typed")
    status: Mapped[str] = mapped_column(String(20), default="interviewing")
    focus: Mapped[str] = mapped_column(String(20), default="both")
    notes: Mapped[dict] = mapped_column(JSONB, default=dict)
    tree_proposal_id: Mapped[str | None] = mapped_column(String(36))
    memory_candidates: Mapped[list] = mapped_column(JSONB, default=list)
    confirmed_memory_ids: Mapped[list] = mapped_column(JSONB, default=list)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    model: Mapped[str | None] = mapped_column(String(100))
    cost: Mapped[float] = mapped_column(Float, default=0)
    revision: Mapped[int] = mapped_column(Integer, default=1)


class OrganizationProposal(Base):
    """A proposed tree of spaces, areas, goals and projects. Mirrors StructureProposal: preview, then a
    separate confirmed apply against the same preview_hash."""
    __tablename__ = "organization_proposals"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=uid)
    owner_id: Mapped[str] = mapped_column(String(100), index=True)
    account_id: Mapped[str] = mapped_column(String(100))
    conversation_id: Mapped[str | None] = mapped_column(String(36))
    tree: Mapped[list] = mapped_column(JSONB, default=list)
    impact: Mapped[dict] = mapped_column(JSONB, default=dict)
    preview_hash: Mapped[str] = mapped_column(String(64))
    applied_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    command_id: Mapped[str | None] = mapped_column(String(100))
