from datetime import UTC, datetime
from uuid import uuid4

from sqlalchemy import (
    Boolean,
    Date,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


def now():
    return datetime.now(UTC)


def uid():
    return str(uuid4())


class Base(DeclarativeBase):
    pass


class AuthSession(Base):
    __tablename__ = "auth_sessions"
    token_hash: Mapped[str] = mapped_column(String(64), primary_key=True)
    owner_id: Mapped[str] = mapped_column(String(100), index=True)
    device_id: Mapped[str] = mapped_column(String(36))
    csrf: Mapped[str] = mapped_column(String(64))
    workspace_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    auth_method: Mapped[str] = mapped_column(String(20), default="pairing")
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


class OwnerSettings(Base):
    __tablename__ = "owner_settings"
    owner_id: Mapped[str] = mapped_column(String(100), primary_key=True)
    values: Mapped[dict] = mapped_column(JSONB, default=dict)


class Space(Base):
    __tablename__ = "spaces"
    __table_args__ = (UniqueConstraint("owner_id", "name", name="uq_space_owner_name"),)
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=uid)
    owner_id: Mapped[str] = mapped_column(String(100), index=True)
    name: Mapped[str] = mapped_column(String(200))
    description: Mapped[str] = mapped_column(Text, default="")
    archived: Mapped[bool] = mapped_column(Boolean, default=False)
    revision: Mapped[int] = mapped_column(Integer, default=1)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)


class Area(Base):
    __tablename__ = "areas"
    __table_args__ = (UniqueConstraint("space_id", "name", name="uq_area_space_name"),)
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=uid)
    owner_id: Mapped[str] = mapped_column(String(100), index=True)
    space_id: Mapped[str] = mapped_column(ForeignKey("spaces.id"), index=True)
    name: Mapped[str] = mapped_column(String(200))
    description: Mapped[str] = mapped_column(Text, default="")
    archived: Mapped[bool] = mapped_column(Boolean, default=False)
    revision: Mapped[int] = mapped_column(Integer, default=1)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)


class Goal(Base):
    __tablename__ = "goals"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=uid)
    owner_id: Mapped[str] = mapped_column(String(100), index=True)
    space_id: Mapped[str | None] = mapped_column(ForeignKey("spaces.id"), index=True)
    area_id: Mapped[str | None] = mapped_column(ForeignKey("areas.id"), index=True)
    parent_goal_id: Mapped[str | None] = mapped_column(ForeignKey("goals.id"), index=True)
    name: Mapped[str] = mapped_column(String(200))
    description: Mapped[str] = mapped_column(Text, default="")
    success_criteria: Mapped[str] = mapped_column(Text, default="")
    status: Mapped[str] = mapped_column(String(30), default="planned")
    horizon: Mapped[str] = mapped_column(String(20), default="unspecified")
    target_date: Mapped[datetime | None] = mapped_column(Date)
    metric_unit: Mapped[str] = mapped_column(String(80), default="")
    metric_baseline: Mapped[float] = mapped_column(Numeric, default=0)
    metric_current: Mapped[float | None] = mapped_column(Numeric)
    metric_target: Mapped[float | None] = mapped_column(Numeric)
    archived: Mapped[bool] = mapped_column(Boolean, default=False)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    revision: Mapped[int] = mapped_column(Integer, default=1)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)


class Actor(Base):
    __tablename__ = "actors"
    __table_args__ = (UniqueConstraint("owner_id", "name", name="uq_actor_owner_name"),)
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=uid)
    owner_id: Mapped[str] = mapped_column(String(100), index=True)
    name: Mapped[str] = mapped_column(String(100))
    kind: Mapped[str] = mapped_column(String(20), default="person")
    archived: Mapped[bool] = mapped_column(Boolean, default=False)
    revision: Mapped[int] = mapped_column(Integer, default=1)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)


class Project(Base):
    __tablename__ = "projects"
    __table_args__ = (UniqueConstraint("owner_id", "name", name="uq_project_owner_name"),)
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=uid)
    owner_id: Mapped[str] = mapped_column(String(100), index=True)
    space_id: Mapped[str | None] = mapped_column(ForeignKey("spaces.id"), index=True)
    area_id: Mapped[str | None] = mapped_column(ForeignKey("areas.id"), index=True)
    name: Mapped[str] = mapped_column(String(200))
    description: Mapped[str] = mapped_column(Text, default="")
    status: Mapped[str] = mapped_column(String(30), default="planned")
    success_criteria: Mapped[str] = mapped_column(Text, default="")
    start_date: Mapped[datetime | None] = mapped_column(Date)
    target_date: Mapped[datetime | None] = mapped_column(Date)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)
    archived: Mapped[bool] = mapped_column(Boolean, default=False)
    revision: Mapped[int] = mapped_column(Integer, default=1)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)


class Task(Base):
    __tablename__ = "tasks"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=uid)
    owner_id: Mapped[str] = mapped_column(String(100), index=True)
    space_id: Mapped[str | None] = mapped_column(ForeignKey("spaces.id"), index=True)
    area_id: Mapped[str | None] = mapped_column(ForeignKey("areas.id"), index=True)
    deadline_alert: Mapped[str] = mapped_column(String(20), default="default", server_default="default")
    alert_urgent: Mapped[bool] = mapped_column(Boolean, default=False, server_default="false")
    title: Mapped[str] = mapped_column(String(500))
    notes: Mapped[str] = mapped_column(Text, default="")
    status: Mapped[str] = mapped_column(String(30), default="open")
    priority: Mapped[int] = mapped_column(Integer, default=0)
    project: Mapped[str | None] = mapped_column(String(200))
    project_id: Mapped[str | None] = mapped_column(ForeignKey("projects.id"), index=True)
    parent_task_id: Mapped[str | None] = mapped_column(ForeignKey("tasks.id"), index=True)
    assignee: Mapped[str] = mapped_column(String(100), default="owner")
    work_type: Mapped[str] = mapped_column(String(80), default="")
    tags: Mapped[list] = mapped_column(JSONB, default=list)
    planned_date: Mapped[datetime | None] = mapped_column(Date, index=True)
    estimate_minutes: Mapped[int | None] = mapped_column(Integer)
    assignee_id: Mapped[str | None] = mapped_column(ForeignKey("actors.id"), index=True)
    due_date: Mapped[datetime | None] = mapped_column(Date)
    due_time: Mapped[str | None] = mapped_column(String(14))
    due_timezone: Mapped[str | None] = mapped_column(String(100))
    revision: Mapped[int] = mapped_column(Integer, default=1)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    archived: Mapped[bool] = mapped_column(Boolean, default=False)
    occurrence_id: Mapped[str | None] = mapped_column(String(36), unique=True)
    is_template: Mapped[bool] = mapped_column(Boolean, default=False)
    external: Mapped[dict] = mapped_column(JSONB, default=dict)


class Command(Base):
    __tablename__ = "commands"
    account_id: Mapped[str | None] = mapped_column(String(100))
    owner_id: Mapped[str] = mapped_column(String(100), primary_key=True)
    id: Mapped[str] = mapped_column(String(100), primary_key=True)
    request_hash: Mapped[str] = mapped_column(String(64))
    result: Mapped[dict] = mapped_column(JSONB)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)


class Event(Base):
    __tablename__ = "events"
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    owner_id: Mapped[str] = mapped_column(String(100), index=True)
    kind: Mapped[str] = mapped_column(String(60))
    entity_id: Mapped[str] = mapped_column(String(100))
    revision: Mapped[int | None] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)


class Schedule(Base):
    __tablename__ = "schedules"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=uid)
    owner_id: Mapped[str] = mapped_column(String(100), index=True)
    title: Mapped[str] = mapped_column(String(500))
    task_id: Mapped[str | None] = mapped_column(ForeignKey("tasks.id"))
    timezone: Mapped[str] = mapped_column(String(100))
    recurrence: Mapped[str | None] = mapped_column(String(250))
    anchor_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    next_run_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), index=True)
    kind: Mapped[str] = mapped_column(String(30), default="reminder")
    status: Mapped[str] = mapped_column(String(20), default="active")
    revision: Mapped[int] = mapped_column(Integer, default=1)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    project_id: Mapped[str | None] = mapped_column(ForeignKey("projects.id"), index=True)
    original_words: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)


class Occurrence(Base):
    __tablename__ = "schedule_occurrences"
    __table_args__ = (UniqueConstraint("schedule_id", "revision", "scheduled_at"),)
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=uid)
    schedule_id: Mapped[str] = mapped_column(ForeignKey("schedules.id"))
    revision: Mapped[int] = mapped_column(Integer)
    scheduled_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    status: Mapped[str] = mapped_column(String(30), default="pending")


class Job(Base):
    __tablename__ = "jobs"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=uid)
    owner_id: Mapped[str] = mapped_column(String(100), index=True)
    kind: Mapped[str] = mapped_column(String(50))
    payload: Mapped[dict] = mapped_column(JSONB)
    status: Mapped[str] = mapped_column(String(30), default="queued")
    result: Mapped[dict | None] = mapped_column(JSONB)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class Outbox(Base):
    __tablename__ = "workflow_outbox"
    job_id: Mapped[str] = mapped_column(ForeignKey("jobs.id"), primary_key=True)
    submitted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class Notification(Base):
    __tablename__ = "notifications"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=uid)
    category: Mapped[str] = mapped_column(String(40), default="reminder", server_default="reminder")
    dedup_key: Mapped[str | None] = mapped_column(String(200), unique=True)
    importance: Mapped[str] = mapped_column(String(20), default="normal", server_default="normal")
    target: Mapped[dict] = mapped_column(JSONB, default=dict, server_default="{}")
    eligible_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    generation: Mapped[int] = mapped_column(Integer, default=1, server_default="1")
    owner_id: Mapped[str] = mapped_column(String(100), index=True)
    occurrence_id: Mapped[str | None] = mapped_column(ForeignKey("schedule_occurrences.id"), unique=True)
    title: Mapped[str] = mapped_column(String(500))
    body: Mapped[str] = mapped_column(Text, default="")
    task_id: Mapped[str | None] = mapped_column(ForeignKey("tasks.id"))
    scheduled_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)
    read_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    dismissed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class PushSubscription(Base):
    __tablename__ = "push_subscriptions"
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    owner_id: Mapped[str] = mapped_column(String(100))
    device_id: Mapped[str] = mapped_column(String(36))
    subscription: Mapped[dict] = mapped_column(JSONB)
    active: Mapped[bool] = mapped_column(Boolean, default=True)


class Delivery(Base):
    __tablename__ = "delivery_attempts"
    __table_args__ = (UniqueConstraint("notification_id", "subscription_id", "generation", name="uq_delivery_generation"),)
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=uid)
    notification_id: Mapped[str] = mapped_column(ForeignKey("notifications.id"))
    subscription_id: Mapped[str] = mapped_column(ForeignKey("push_subscriptions.id"))
    generation: Mapped[int] = mapped_column(Integer, default=1, server_default="1")
    status: Mapped[str] = mapped_column(String(30), default="pending")
    attempts: Mapped[int] = mapped_column(Integer, default=0)
    next_attempt_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)
    lease_until: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    error_code: Mapped[str | None] = mapped_column(String(100))


class Conversation(Base):
    __tablename__ = "conversations"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=uid)
    owner_id: Mapped[str] = mapped_column(String(100), index=True)
    device_id: Mapped[str] = mapped_column(String(36))
    private: Mapped[bool] = mapped_column(Boolean, default=False)
    learning: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)


class Source(Base):
    __tablename__ = "sources"
    __table_args__ = (UniqueConstraint("owner_id", "native_id"),)
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=uid)
    owner_id: Mapped[str] = mapped_column(String(100), index=True)
    conversation_id: Mapped[str | None] = mapped_column(ForeignKey("conversations.id"))
    native_id: Mapped[str] = mapped_column(String(200))
    kind: Mapped[str] = mapped_column(String(50))
    role: Mapped[str] = mapped_column(String(30))
    content: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    explicit: Mapped[bool] = mapped_column(Boolean, default=False)
    memory_version: Mapped[int] = mapped_column(Integer, default=0)


class Memory(Base):
    __tablename__ = "memory_assertions"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=uid)
    owner_id: Mapped[str] = mapped_column(String(100), index=True)
    source_id: Mapped[str] = mapped_column(ForeignKey("sources.id"))
    content: Mapped[str] = mapped_column(Text)
    attribution: Mapped[str] = mapped_column(String(40), default="owner_statement")
    tags: Mapped[list] = mapped_column(JSONB, default=list)
    evidence: Mapped[str] = mapped_column(Text, default="")
    fact_key: Mapped[str] = mapped_column(String(200), default="")
    fingerprint: Mapped[str] = mapped_column(String(64), default="", index=True)
    embedding: Mapped[list | None] = mapped_column(JSONB)
    embedding_model: Mapped[str | None] = mapped_column(String(100))
    revision: Mapped[int] = mapped_column(Integer, default=1)
    suppressed: Mapped[bool] = mapped_column(Boolean, default=False)
    supersedes_id: Mapped[str | None] = mapped_column(ForeignKey("memory_assertions.id"))
    merged_into_id: Mapped[str | None] = mapped_column(ForeignKey("memory_assertions.id"))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)


class MemoryReview(Base):
    __tablename__ = "memory_reviews"
    __table_args__ = (UniqueConstraint("owner_id", "pair_key", name="uq_memory_review_pair"),)
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=uid)
    owner_id: Mapped[str] = mapped_column(String(100), index=True)
    pair_key: Mapped[str] = mapped_column(String(64))
    memory_ids: Mapped[list] = mapped_column(JSONB)
    memory_revisions: Mapped[list] = mapped_column(JSONB)
    kind: Mapped[str] = mapped_column(String(30), default="spelling")
    status: Mapped[str] = mapped_column(String(30), default="pending")
    revision: Mapped[int] = mapped_column(Integer, default=1)
    result_memory_id: Mapped[str | None] = mapped_column(ForeignKey("memory_assertions.id"))
    last_offered_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    deferred_until: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)
    resolved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class BudgetReservation(Base):
    __tablename__ = "budget_reservations"
    id: Mapped[str] = mapped_column(String(100), primary_key=True)
    owner_id: Mapped[str] = mapped_column(String(100), index=True)
    amount: Mapped[float] = mapped_column(Numeric(12, 6))
    actual: Mapped[float | None] = mapped_column(Numeric(12, 6))
    model: Mapped[str] = mapped_column(String(100))
    state: Mapped[str] = mapped_column(String(30), default="reserved")
    last_activity_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)
    closed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    settlement: Mapped[dict] = mapped_column(JSONB, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)


class Usage(Base):
    __tablename__ = "usage_events"
    request_id: Mapped[str] = mapped_column(String(150), primary_key=True)
    owner_id: Mapped[str] = mapped_column(String(100))
    reservation_id: Mapped[str] = mapped_column(ForeignKey("budget_reservations.id"))
    model: Mapped[str] = mapped_column(String(100))
    amount: Mapped[float] = mapped_column(Numeric(12, 6))
    tokens: Mapped[dict] = mapped_column(JSONB)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)


class WorkerHealth(Base):
    __tablename__ = "worker_health"
    id: Mapped[str] = mapped_column(String(30), primary_key=True)
    last_scan_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


Index("sources_search", Source.owner_id, Source.created_at)


class Note(Base):
    __tablename__ = "notes"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=uid)
    owner_id: Mapped[str] = mapped_column(String(100), index=True)
    space_id: Mapped[str | None] = mapped_column(ForeignKey("spaces.id"), index=True)
    area_id: Mapped[str | None] = mapped_column(ForeignKey("areas.id"), index=True)
    title: Mapped[str] = mapped_column(String(200))
    content: Mapped[str] = mapped_column(Text, default="")
    tags: Mapped[list] = mapped_column(JSONB, default=list)
    project_id: Mapped[str | None] = mapped_column(ForeignKey("projects.id"), index=True)
    conversation_id: Mapped[str | None] = mapped_column(ForeignKey("conversations.id"))
    archived: Mapped[bool] = mapped_column(Boolean, default=False)
    revision: Mapped[int] = mapped_column(Integer, default=1)
    index_state: Mapped[str] = mapped_column(String(30), default="queued")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)


class NoteEmbedding(Base):
    __tablename__ = "note_embeddings"
    note_id: Mapped[str] = mapped_column(ForeignKey("notes.id"), primary_key=True)
    position: Mapped[int] = mapped_column(Integer, primary_key=True)
    revision: Mapped[int] = mapped_column(Integer)
    embedding: Mapped[list] = mapped_column(JSONB)
    model: Mapped[str] = mapped_column(String(100))


class NoteTaskLink(Base):
    __tablename__ = "note_task_links"
    __table_args__ = (UniqueConstraint("note_id", "fingerprint", name="uq_note_extraction"),)
    note_id: Mapped[str] = mapped_column(ForeignKey("notes.id"), primary_key=True)
    task_id: Mapped[str] = mapped_column(ForeignKey("tasks.id"), primary_key=True)
    linked: Mapped[bool] = mapped_column(Boolean, default=True)
    evidence: Mapped[str] = mapped_column(Text, default="")
    note_revision: Mapped[int | None] = mapped_column(Integer)
    fingerprint: Mapped[str | None] = mapped_column(String(64))


class TaskReference(Base):
    __tablename__ = "task_references"
    conversation_id: Mapped[str] = mapped_column(ForeignKey("conversations.id"), primary_key=True)
    task_id: Mapped[str] = mapped_column(ForeignKey("tasks.id"), primary_key=True)
    owner_id: Mapped[str] = mapped_column(String(100), index=True)
    touched_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)


class GoogleIdentity(Base):
    __tablename__ = "google_identities"
    owner_id: Mapped[str] = mapped_column(String(100), primary_key=True)
    subject: Mapped[str] = mapped_column(String(255), unique=True)
    email: Mapped[str] = mapped_column(String(320))
    credentials: Mapped[str | None] = mapped_column(Text)
    calendar_enabled: Mapped[bool] = mapped_column(Boolean, default=False)
    calendar_write_enabled: Mapped[bool] = mapped_column(Boolean, default=False)
    generation: Mapped[int] = mapped_column(Integer, default=1)
    status: Mapped[str] = mapped_column(String(30), default="not_connected")
    last_sync_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    next_sync_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    error: Mapped[str] = mapped_column(String(100), default="")
    linked_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)


class GoogleOAuthAttempt(Base):
    __tablename__ = "google_oauth_attempts"
    state_hash: Mapped[str] = mapped_column(String(64), primary_key=True)
    browser_hash: Mapped[str] = mapped_column(String(64))
    purpose: Mapped[str] = mapped_column(String(20))
    return_to: Mapped[str | None] = mapped_column(String(1000))
    account_subject: Mapped[str | None] = mapped_column(String(255))
    account_generation: Mapped[int | None] = mapped_column(Integer)
    session_hash: Mapped[str | None] = mapped_column(String(64))
    nonce: Mapped[str] = mapped_column(String(100))
    verifier: Mapped[str] = mapped_column(Text)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


class GoogleCalendar(Base):
    __tablename__ = "google_calendars"
    __table_args__ = (UniqueConstraint("owner_id", "provider_id", name="uq_google_calendar"),)
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=uid)
    owner_id: Mapped[str] = mapped_column(ForeignKey("google_identities.owner_id"), index=True)
    provider_id: Mapped[str] = mapped_column(String(1024))
    access_role: Mapped[str] = mapped_column(String(40), default="reader")
    details_version: Mapped[int] = mapped_column(Integer, default=1)
    title: Mapped[str] = mapped_column(String(500))
    timezone: Mapped[str] = mapped_column(String(100), default="UTC")
    selected: Mapped[bool] = mapped_column(Boolean, default=False)
    available: Mapped[bool] = mapped_column(Boolean, default=True)
    primary: Mapped[bool] = mapped_column(Boolean, default=False)
    revision: Mapped[int] = mapped_column(Integer, default=1)
    sync_token: Mapped[str | None] = mapped_column(Text)
    last_sync_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class GoogleCalendarEvent(Base):
    __tablename__ = "google_calendar_events"
    __table_args__ = (UniqueConstraint("calendar_id", "provider_id", name="uq_google_event"),)
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=uid)
    calendar_id: Mapped[str] = mapped_column(ForeignKey("google_calendars.id"), index=True)
    provider_id: Mapped[str] = mapped_column(String(1024))
    payload: Mapped[dict] = mapped_column(JSONB)


class PlanningEntry(Base):
    __tablename__ = "planning_entries"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=uid)
    owner_id: Mapped[str] = mapped_column(String(100), index=True)
    kind: Mapped[str] = mapped_column(String(20), default="event")
    task_id: Mapped[str | None] = mapped_column(ForeignKey("tasks.id"), index=True)
    fields: Mapped[dict] = mapped_column(JSONB)
    status: Mapped[str] = mapped_column(String(20), default="active")
    revision: Mapped[int] = mapped_column(Integer, default=1)
    google_calendar_id: Mapped[str | None] = mapped_column(String(36))
    google_event_id: Mapped[str | None] = mapped_column(String(1024))
    google_snapshot: Mapped[dict | None] = mapped_column(JSONB)
    google_job_id: Mapped[str | None] = mapped_column(String(36))
    google_state: Mapped[str] = mapped_column(String(30), default="local")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)


class LinearConnection(Base):
    __tablename__ = "linear_connections"
    owner_id: Mapped[str] = mapped_column(String(100), primary_key=True)
    credentials: Mapped[str | None] = mapped_column(Text)
    workspace_id: Mapped[str] = mapped_column(String(100))
    workspace_name: Mapped[str] = mapped_column(String(250))
    viewer_id: Mapped[str] = mapped_column(String(100))
    enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    generation: Mapped[int] = mapped_column(Integer, default=1)
    revision: Mapped[int] = mapped_column(Integer, default=1)
    team_ids: Mapped[list] = mapped_column(JSONB, default=list)
    only_mine: Mapped[bool] = mapped_column(Boolean, default=True)
    directory: Mapped[dict] = mapped_column(JSONB, default=dict)
    last_sync_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    full_sync_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    next_sync_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    status: Mapped[str] = mapped_column(String(30), default="ready")
    error: Mapped[str] = mapped_column(String(200), default="")


class LinearIssue(Base):
    __tablename__ = "linear_issues"
    __table_args__ = (UniqueConstraint("owner_id", "workspace_id", "remote_id", name="uq_linear_issue"),)
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=uid)
    owner_id: Mapped[str] = mapped_column(String(100), index=True)
    workspace_id: Mapped[str] = mapped_column(String(100))
    remote_id: Mapped[str] = mapped_column(String(100))
    task_id: Mapped[str] = mapped_column(ForeignKey("tasks.id"), unique=True)
    snapshot: Mapped[dict] = mapped_column(JSONB, default=dict)
    pending_job_id: Mapped[str | None] = mapped_column(String(36))
    sync_state: Mapped[str] = mapped_column(String(30), default="synced")
    latest_remote: Mapped[dict | None] = mapped_column(JSONB)


class GoalProjectLink(Base):
    __tablename__ = "goal_project_links"
    goal_id: Mapped[str] = mapped_column(ForeignKey("goals.id"), primary_key=True)
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id"), primary_key=True)


class NoteGoalLink(Base):
    __tablename__ = "note_goal_links"
    note_id: Mapped[str] = mapped_column(ForeignKey("notes.id"), primary_key=True)
    goal_id: Mapped[str] = mapped_column(ForeignKey("goals.id"), primary_key=True)


class NoteProjectLink(Base):
    __tablename__ = "note_project_links"
    note_id: Mapped[str] = mapped_column(ForeignKey("notes.id"), primary_key=True)
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id"), primary_key=True)


class NoteNoteLink(Base):
    __tablename__ = "note_note_links"
    note_id: Mapped[str] = mapped_column(ForeignKey("notes.id"), primary_key=True)
    related_note_id: Mapped[str] = mapped_column(ForeignKey("notes.id"), primary_key=True)

class UserAccount(Base):
    __tablename__ = "user_accounts"
    id: Mapped[str] = mapped_column(String(100), primary_key=True)
    name: Mapped[str] = mapped_column(String(100))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)


class SharedWorkspace(Base):
    __tablename__ = "shared_workspaces"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=uid)
    name: Mapped[str] = mapped_column(String(200))
    kind: Mapped[str] = mapped_column(String(20))
    creator_id: Mapped[str] = mapped_column(ForeignKey("user_accounts.id"))
    root_id: Mapped[str | None] = mapped_column(String(36))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)


class WorkspaceMember(Base):
    __tablename__ = "workspace_members"
    workspace_id: Mapped[str] = mapped_column(ForeignKey("shared_workspaces.id"), primary_key=True)
    account_id: Mapped[str] = mapped_column(ForeignKey("user_accounts.id"), primary_key=True)
    role: Mapped[str] = mapped_column(String(20))
    active: Mapped[bool] = mapped_column(Boolean, default=True)
    actor_id: Mapped[str | None] = mapped_column(ForeignKey("actors.id"))
    revision: Mapped[int] = mapped_column(Integer, default=1)


class WorkspaceInvite(Base):
    __tablename__ = "workspace_invites"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=uid)
    workspace_id: Mapped[str | None] = mapped_column(ForeignKey("shared_workspaces.id"))
    inviter_id: Mapped[str] = mapped_column(ForeignKey("user_accounts.id"))
    email: Mapped[str] = mapped_column(String(320), index=True)
    role: Mapped[str] = mapped_column(String(20))
    status: Mapped[str] = mapped_column(String(20), default="pending")
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    accepted_by: Mapped[str | None] = mapped_column(ForeignKey("user_accounts.id"))


class AgentWork(Base):
    """Actor-scoped durable requests; encrypted inputs/checkpoints never enter DBOS results."""
    __tablename__ = "agent_work"
    id: Mapped[str] = mapped_column(ForeignKey("jobs.id"), primary_key=True)
    owner_id: Mapped[str] = mapped_column(String(100), index=True)
    account_id: Mapped[str] = mapped_column(String(100), index=True)
    device_id: Mapped[str] = mapped_column(String(36))
    conversation_id: Mapped[str] = mapped_column(ForeignKey("conversations.id"), index=True)
    parent_id: Mapped[str | None] = mapped_column(String(36), index=True)
    voice_session_id: Mapped[str | None] = mapped_column(String(36), index=True)
    credential_id: Mapped[str | None] = mapped_column(String(36), index=True)
    input_hash: Mapped[str] = mapped_column(String(64))
    input_ciphertext: Mapped[str | None] = mapped_column(Text)
    checkpoint_ciphertext: Mapped[str | None] = mapped_column(Text)
    revision: Mapped[int] = mapped_column(Integer, default=1)
    cancel_requested: Mapped[bool] = mapped_column(Boolean, default=False)
    dependencies: Mapped[list] = mapped_column(JSONB, default=list)
    resources: Mapped[list] = mapped_column(JSONB, default=list)
    result: Mapped[dict] = mapped_column(JSONB, default=dict)
    transient: Mapped[bool] = mapped_column(Boolean, default=False)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)
    seen_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class VoiceInbox(Base):
    __tablename__ = "voice_inboxes"
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    owner_id: Mapped[str] = mapped_column(String(100), index=True)
    account_id: Mapped[str] = mapped_column(String(100))
    device_id: Mapped[str] = mapped_column(String(36))
    conversation_id: Mapped[str] = mapped_column(ForeignKey("conversations.id"))
    content_ciphertext: Mapped[str] = mapped_column(Text)
    cursor: Mapped[int] = mapped_column(Integer, default=0)
    revision: Mapped[int] = mapped_column(Integer, default=0)
    last_input_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)
    end_requested: Mapped[bool] = mapped_column(Boolean, default=False)
    closed: Mapped[bool] = mapped_column(Boolean, default=False)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)


class DeviceBridge(Base):
    __tablename__ = "device_bridges"
    owner_id: Mapped[str] = mapped_column(String(100), primary_key=True)
    device_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    account_id: Mapped[str] = mapped_column(String(100))
    context_ciphertext: Mapped[str] = mapped_column(Text)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)


class DeviceAction(Base):
    __tablename__ = "device_actions"
    owner_id: Mapped[str] = mapped_column(String(100), primary_key=True)
    id: Mapped[str] = mapped_column(String(100), primary_key=True)
    device_id: Mapped[str] = mapped_column(String(36), index=True)
    account_id: Mapped[str] = mapped_column(String(100))
    action_ciphertext: Mapped[str] = mapped_column(Text)
    result_ciphertext: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)
    sent_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)


class ActionChange(Base):
    __tablename__ = "action_changes"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=uid)
    owner_id: Mapped[str] = mapped_column(String(100), index=True)
    account_id: Mapped[str] = mapped_column(String(100), index=True)
    command_id: Mapped[str] = mapped_column(String(100), index=True)
    tool: Mapped[str] = mapped_column(String(70))
    entity_kind: Mapped[str] = mapped_column(String(40))
    entity_id: Mapped[str] = mapped_column(String(100), index=True)
    before_ciphertext: Mapped[str | None] = mapped_column(Text)
    after_ciphertext: Mapped[str | None] = mapped_column(Text)
    reverted_by: Mapped[str | None] = mapped_column(String(100))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)


class BotCredential(Base):
    """One revocable, hashed credential bound to its creator and workspace."""
    __tablename__ = "bot_credentials"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=uid)
    owner_id: Mapped[str] = mapped_column(String(100), index=True)
    account_id: Mapped[str] = mapped_column(ForeignKey("user_accounts.id"), index=True)
    name: Mapped[str] = mapped_column(String(80))
    token_hash: Mapped[str] = mapped_column(String(64), unique=True)
    prefix: Mapped[str] = mapped_column(String(20))
    scopes: Mapped[list] = mapped_column(JSONB)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    last_used_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    rate_window: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)
    rate_count: Mapped[int] = mapped_column(Integer, default=0)

# Register configurable planner tables with the shared metadata.
from .structure_models import (StructureSchema, StructureProposal, StructureRecord, StructureLink, FieldUnderstanding, RoutingObservation, RoutingPattern, RoutingReview)  # noqa: E402,F401

from .search_models import SearchDocument, SearchIndexState, SearchPreference, SearchAlias, SearchSession  # noqa: E402,F401

from .note_list_models import NoteList, NoteOrganization, NoteEntrySource  # noqa: E402,F401
