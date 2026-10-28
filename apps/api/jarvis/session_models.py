"""External agent sessions (Claude Code, Codex, desktop apps) and the rules that file them."""

from datetime import datetime
from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Index, Integer, String, Text, UniqueConstraint, text
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
    # Host-reported terminal state over the agent link (agent_link.py); empty for collector-only rows.
    host_id: Mapped[str | None] = mapped_column(ForeignKey("agent_hosts.id"), index=True)
    surf: Mapped[str] = mapped_column(String(100), default="", server_default="")
    incarnation: Mapped[int] = mapped_column(Integer, default=0, server_default="0")
    state: Mapped[str] = mapped_column(String(20), default="", server_default="")
    state_since: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    state_detail: Mapped[dict] = mapped_column(JSONB, default=dict, server_default="{}")
    last_turn_seq: Mapped[int] = mapped_column(Integer, default=0, server_default="0")
    # The Eridani run this session executes (agent_runs, 0028); set on the observed start.
    run_id: Mapped[str | None] = mapped_column(ForeignKey("agent_runs.id"), index=True)
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


class AgentHost(Base):
    """One remotea server linked to a workspace. The lease fences superseded connections."""
    __tablename__ = "agent_hosts"
    __table_args__ = (UniqueConstraint("owner_id", "server_id", name="uq_agent_host"),)
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=uid)
    owner_id: Mapped[str] = mapped_column(String(100), index=True)
    account_id: Mapped[str] = mapped_column(ForeignKey("user_accounts.id"), index=True)
    credential_id: Mapped[str] = mapped_column(ForeignKey("bot_credentials.id"), index=True)
    server_id: Mapped[str] = mapped_column(String(100))
    name: Mapped[str] = mapped_column(String(100), default="")
    machine: Mapped[str] = mapped_column(String(100), default="")
    boot_id: Mapped[str] = mapped_column(String(100), default="")
    proto: Mapped[int] = mapped_column(Integer, default=1)
    conn_epoch: Mapped[int] = mapped_column(Integer, default=0)
    lease_until: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    caps: Mapped[dict] = mapped_column(JSONB, default=dict)
    last_seen_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)


class AgentCommand(Base):
    """Outbox/journal of commands to a host; written in the same transaction as the domain change."""
    __tablename__ = "agent_commands"
    __table_args__ = (
        UniqueConstraint("owner_id", "op_id", name="uq_agent_command_op"),
        Index("ix_agent_commands_host_id_status", "host_id", "status"),
    )
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=uid)
    owner_id: Mapped[str] = mapped_column(String(100), index=True)
    host_id: Mapped[str] = mapped_column(ForeignKey("agent_hosts.id"))
    op_id: Mapped[str] = mapped_column(String(100))
    kind: Mapped[str] = mapped_column(String(40))
    run_id: Mapped[str | None] = mapped_column(ForeignKey("agent_runs.id"), index=True)
    generation: Mapped[int] = mapped_column(Integer, default=0)
    payload: Mapped[dict] = mapped_column(JSONB, default=dict)
    digest: Mapped[str] = mapped_column(String(64))
    status: Mapped[str] = mapped_column(String(20), default="queued")
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    outcome: Mapped[dict | None] = mapped_column(JSONB)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)


class AgentTurn(Base):
    """An agent's final response for one root provider turn (bounded; never the transcript)."""
    __tablename__ = "agent_turns"
    __table_args__ = (
        UniqueConstraint("session_pk", "provider_turn_id", name="uq_agent_turn"),
        Index("ix_agent_turns_session_pk_seq", "session_pk", "seq"),
    )
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=uid)
    owner_id: Mapped[str] = mapped_column(String(100), index=True)
    session_pk: Mapped[str] = mapped_column(ForeignKey("agent_sessions.id"))
    host_id: Mapped[str | None] = mapped_column(ForeignKey("agent_hosts.id"), index=True)
    provider_session_id: Mapped[str] = mapped_column(String(100))
    provider_turn_id: Mapped[str] = mapped_column(String(200))
    surf: Mapped[str] = mapped_column(String(100), default="")
    incarnation: Mapped[int] = mapped_column(Integer, default=0)
    run_id: Mapped[str | None] = mapped_column(ForeignKey("agent_runs.id"), index=True)
    generation: Mapped[int] = mapped_column(Integer, default=0)
    seq: Mapped[int] = mapped_column(Integer, default=0)
    role: Mapped[str] = mapped_column(String(20), default="assistant")
    text: Mapped[str] = mapped_column(Text, default="")
    prompt: Mapped[str] = mapped_column(Text, default="")
    truncated: Mapped[bool] = mapped_column(Boolean, default=False)
    source: Mapped[str] = mapped_column(String(30), default="hook")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)


class AgentQuestion(Base):
    """A structured question an agent asked (stage C answers it); kept here so the schema lands once."""
    __tablename__ = "agent_questions"
    __table_args__ = (UniqueConstraint("session_pk", "provider_request_id", name="uq_agent_question"),)
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=uid)
    owner_id: Mapped[str] = mapped_column(String(100), index=True)
    session_pk: Mapped[str] = mapped_column(ForeignKey("agent_sessions.id"))
    host_id: Mapped[str] = mapped_column(ForeignKey("agent_hosts.id"), index=True)
    run_id: Mapped[str | None] = mapped_column(ForeignKey("agent_runs.id"), index=True)
    generation: Mapped[int] = mapped_column(Integer, default=0)
    question_id: Mapped[str] = mapped_column(String(100))
    provider_request_id: Mapped[str] = mapped_column(String(200))
    group_id: Mapped[str] = mapped_column(String(100), default="")
    kind: Mapped[str] = mapped_column(String(30), default="")
    items: Mapped[list] = mapped_column(JSONB, default=list)
    status: Mapped[str] = mapped_column(String(20), default="open")
    answers: Mapped[list | None] = mapped_column(JSONB)
    answered_via: Mapped[str] = mapped_column(String(30), default="")
    # Stage C: where it was asked, how long the host holds it, and what became of the answer.
    surf: Mapped[str] = mapped_column(String(100), default="", server_default="")
    incarnation: Mapped[int] = mapped_column(Integer, default=0, server_default="0")
    provider_turn_id: Mapped[str] = mapped_column(String(200), default="", server_default="")
    hold_until: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    remote_answerable: Mapped[bool] = mapped_column(Boolean, default=True, server_default="true")
    answer_op_id: Mapped[str | None] = mapped_column(String(100))
    detail: Mapped[str] = mapped_column(String(300), default="", server_default="")
    resolved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)


class AgentParity(Base):
    """Latest shadow comparison per session between a new collector's report and the stored row."""
    __tablename__ = "agent_parity"
    __table_args__ = (UniqueConstraint("owner_id", "tool", "session_id", name="uq_agent_parity"),)
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=uid)
    owner_id: Mapped[str] = mapped_column(String(100), index=True)
    host_id: Mapped[str | None] = mapped_column(ForeignKey("agent_hosts.id"), index=True)
    tool: Mapped[str] = mapped_column(String(30))
    session_id: Mapped[str] = mapped_column(String(100))
    matched: Mapped[bool] = mapped_column(Boolean, default=False)
    missing: Mapped[bool] = mapped_column(Boolean, default=False)
    diff: Mapped[dict] = mapped_column(JSONB, default=dict)
    reports: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)


TERMINAL_RUN_STATES = ("completed", "stopped", "capped", "failed")


class ExecutionTarget(Base):
    """Where a run executes: a remotea execution target (repo + worktree root) on one host.

    `record_id` saves a one-time selection for a record in a task's chain; tasks can also name
    a target directly (tasks.execution_target_id).
    """
    __tablename__ = "execution_targets"
    __table_args__ = (
        Index("uq_execution_target_record", "owner_id", "record_id", unique=True,
              postgresql_where=text("record_id IS NOT NULL")),
    )
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=uid)
    owner_id: Mapped[str] = mapped_column(String(100), index=True)
    host_id: Mapped[str | None] = mapped_column(ForeignKey("agent_hosts.id"), index=True)
    # No foreign key: structure_records → tasks → execution_targets would form a cycle; owned() checks it.
    record_id: Mapped[str | None] = mapped_column(String(36), index=True)
    host_target: Mapped[str] = mapped_column(String(64))
    name: Mapped[str] = mapped_column(String(200), default="")
    repo: Mapped[str] = mapped_column(Text, default="")
    worktree_root: Mapped[str] = mapped_column(Text, default="")
    launch: Mapped[dict] = mapped_column(JSONB, default=dict)
    verify: Mapped[list] = mapped_column(JSONB, default=list)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)


class AgentRun(Base):
    """One run of an agent on a task: a versioned state machine (run_coordinator.py)."""
    __tablename__ = "agent_runs"
    __table_args__ = (
        # One unsettled run per task; stopping/interviewing/planning count as unsettled.
        Index("uq_agent_run_active_task", "task_id", unique=True,
              postgresql_where=text("state NOT IN ('completed', 'stopped', 'capped', 'failed')")),
        Index("ix_agent_runs_owner_id_state", "owner_id", "state"),
    )
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=uid)
    owner_id: Mapped[str] = mapped_column(String(100), index=True)
    account_id: Mapped[str | None] = mapped_column(ForeignKey("user_accounts.id"))
    task_id: Mapped[str] = mapped_column(ForeignKey("tasks.id"), index=True)
    host_id: Mapped[str] = mapped_column(ForeignKey("agent_hosts.id"), index=True)
    # No foreign key: agent_sessions.run_id points back here, and a cycle would block deletes.
    session_pk: Mapped[str | None] = mapped_column(String(36), index=True)
    execution_target_id: Mapped[str | None] = mapped_column(ForeignKey("execution_targets.id"), index=True)
    generation: Mapped[int] = mapped_column(Integer, default=1)
    attempt: Mapped[int] = mapped_column(Integer, default=1)
    state: Mapped[str] = mapped_column(String(30), default="queued")
    state_revision: Mapped[int] = mapped_column(Integer, default=1)
    state_reason: Mapped[str] = mapped_column(String(300), default="")
    state_since: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)
    plan_revision: Mapped[int] = mapped_column(Integer, default=0)
    worktree: Mapped[str] = mapped_column(Text, default="")
    branch: Mapped[str] = mapped_column(String(200), default="")
    base_branch: Mapped[str] = mapped_column(String(200), default="")
    tool: Mapped[str] = mapped_column(String(30), default="claude")
    model: Mapped[str] = mapped_column(String(80), default="opus")
    effective_permission_mode: Mapped[str] = mapped_column(String(40), default="")
    surf: Mapped[str] = mapped_column(String(120), default="")
    incarnation: Mapped[int] = mapped_column(Integer, default=0)
    turn_count: Mapped[int] = mapped_column(Integer, default=0)
    turn_cap: Mapped[int] = mapped_column(Integer, default=15)
    wall_clock_limit: Mapped[int] = mapped_column(Integer, default=6 * 3600)  # seconds
    no_progress_limit: Mapped[int] = mapped_column(Integer, default=45 * 60)  # seconds
    last_progress_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    stop_requested_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    stop_mode: Mapped[str] = mapped_column(String(20), default="")
    stop_reason: Mapped[str] = mapped_column(String(60), default="")
    # The terminal state a confirmed stop settles in: stopped, or completed after an approval.
    stop_final: Mapped[str] = mapped_column(String(20), default="stopped")
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    completion_policy: Mapped[str] = mapped_column(String(20), default="approve")
    sync_state: Mapped[dict] = mapped_column(JSONB, default=dict)
    run_token_hash: Mapped[str | None] = mapped_column(String(64), unique=True)
    run_token_expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    # Every transition with its actor, newest last (bounded).
    history: Mapped[list] = mapped_column(JSONB, default=list)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)


class AgentRunPlan(Base):
    """A frozen plan revision a run executes (and, from stage D, is judged against)."""
    __tablename__ = "agent_run_plans"
    __table_args__ = (UniqueConstraint("run_id", "revision", name="uq_agent_run_plan"),)
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=uid)
    owner_id: Mapped[str] = mapped_column(String(100), index=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("agent_runs.id"), index=True)
    revision: Mapped[int] = mapped_column(Integer)
    task_plan_revision: Mapped[int] = mapped_column(Integer, default=0)
    title: Mapped[str] = mapped_column(String(500), default="")
    prd_text: Mapped[str] = mapped_column(Text, default="")
    steps: Mapped[list] = mapped_column(JSONB, default=list)
    criteria: Mapped[list] = mapped_column(JSONB, default=list)
    verify: Mapped[dict | None] = mapped_column(JSONB)
    frozen_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)


class AgentAttempt(Base):
    """One verification or judge attempt bound to (run, generation, turn, plan revision). Logic is stage D."""
    __tablename__ = "agent_attempts"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=uid)
    owner_id: Mapped[str] = mapped_column(String(100), index=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("agent_runs.id"), index=True)
    generation: Mapped[int] = mapped_column(Integer, default=0)
    kind: Mapped[str] = mapped_column(String(20))
    turn_seq: Mapped[int] = mapped_column(Integer, default=0)
    plan_revision: Mapped[int] = mapped_column(Integer, default=0)
    workspace_fingerprint: Mapped[str] = mapped_column(String(200), default="")
    status: Mapped[str] = mapped_column(String(20), default="queued")
    result: Mapped[dict] = mapped_column(JSONB, default=dict)
    model: Mapped[str] = mapped_column(String(100), default="")
    prompt_version: Mapped[str] = mapped_column(String(40), default="")
    cost: Mapped[float | None] = mapped_column(Float)
    latency_ms: Mapped[int | None] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)


class AgentInterview(Base):
    """The interview phase's question form (stage D)."""
    __tablename__ = "agent_interviews"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=uid)
    owner_id: Mapped[str] = mapped_column(String(100), index=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("agent_runs.id"), index=True)
    questions: Mapped[list] = mapped_column(JSONB, default=list)
    answers: Mapped[dict] = mapped_column(JSONB, default=dict)
    status: Mapped[str] = mapped_column(String(20), default="open")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)


class AutolinkRejection(Base):
    """An Undo of an automatic session link; the dream sequence (stage E) never re-proposes it."""
    __tablename__ = "autolink_rejections"
    __table_args__ = (UniqueConstraint("session_pk", "task_id", name="uq_autolink_rejection"),)
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=uid)
    owner_id: Mapped[str] = mapped_column(String(100), index=True)
    session_pk: Mapped[str] = mapped_column(ForeignKey("agent_sessions.id"), index=True)
    task_id: Mapped[str] = mapped_column(ForeignKey("tasks.id"), index=True)
    evidence_revision: Mapped[str] = mapped_column(String(64), default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)
