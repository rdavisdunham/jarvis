"""The agent link: one outbound WebSocket from each remotea server (docs/AGENT_CONTROL.md).

The host reports sessions, final responses per turn, terminal state, questions and reconcile
snapshots, and runs collector parity. Commands to the host (spawn, input, answer, stop) are written
to agent_commands by the run coordinator and drained here, resent until acknowledged.

The http `security` middleware does not run for websocket scopes, so the handshake repeats the
maintenance, Origin and size guards itself. Every inbound message runs in its own short
transaction that re-checks the key (a privileged write) and the connection's lease epoch.
"""

import asyncio
import contextlib
import hashlib
import json
import logging
import time
import uuid
from datetime import UTC, datetime, timedelta
from typing import Annotated, ClassVar, Literal

from fastapi import APIRouter, WebSocket
from fastapi.responses import JSONResponse
from pydantic import (
    AwareDatetime,
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    ValidationError,
    model_validator,
)
from sqlalchemy import func, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import object_session
from starlette.concurrency import run_in_threadpool

from . import agent_retention as retention
from . import agent_sessions as sessions
from . import bot_access
from . import run_coordinator as coordinator
from .config import get_settings
from .db import session_scope
from .domain import DomainError, advisory, emit, encode
from .models import OwnerSettings, now
from .redaction import redact
from .session_models import AgentCommand, AgentHost, AgentParity, AgentParityDay, AgentSession, AgentTurn

logger = logging.getLogger(__name__)
router = APIRouter()
PATH = "/api/v1/external/agents/link"
SUBPROTOCOL = "eridani.v1"
PROTO = 1
MAX_FRAME = 262_144  # Matches uvicorn --ws-max-size in deploy.py.
MAX_ASSEMBLED = 1_048_576
MAX_PARTS = 64
MAX_PENDING_CHUNKS = 4
PARTIAL_SECONDS = 60  # Partial chunked frames and snapshots expire (stage F, L3).
HOSTS_PER_CREDENTIAL = 3  # Distinct server_ids one link key may register (stage F, L3).
FRAMES_PER_MINUTE = 1200
BYTES_PER_MINUTE = 16 * 1_048_576
HELLO_SECONDS = 15
TICK_SECONDS = 30  # Re-authorize, renew the lease and drain commands.
LEASE = timedelta(seconds=90)
REPLAY_HORIZON_DAYS = 7
TURN_TEXT_BYTES = 65_536
SERVER_CAPS = ["chunk", "chunking", "session.report", "session.report.parity", "turn", "status", "exit",
               "sessions.snapshot", "command.outcome", "runs.snapshot", "question", "context", "worktrees.report",
               "spawn", "input", "answer", "stop", "verify", "verify.result",
               # Stage F: context.response.scope is "run" (full run context) or "candidates" (D6).
               "context.scope"]
# Command kind -> host capability it needs.
COMMANDS: dict[str, str] = {"spawn": "spawn", "input": "input", "answer": "answer", "stop": "stop", "verify": "verify"}
RESEND = timedelta(seconds=60)  # A sent command without an ack goes out again (same op_id and digest).
# Refusals the host should resend later (ack retry:true); any other refusal is permanent.
RETRYABLE = {"RATE_LIMITED", "UNKNOWN_SURFACE", "STALE_EPOCH"}
LIVE_STATES = ("working", "needs_you", "idle")
STATES = (*LIVE_STATES, "restoring", "lost", "exited")
HARNESS_TOOLS = {"claude": "claude_code", "codex": "codex_cli"}


class Closed(Exception):
    def __init__(self, code, reason, bye=None):
        self.code, self.reason, self.bye = code, reason, bye
        super().__init__(reason)


def wire_time(value):
    """Timestamps on the wire are epoch milliseconds; ISO-8601 with a zone is also accepted."""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return datetime.fromtimestamp(value / 1000, UTC)
    return value


WireTime = Annotated[AwareDatetime, BeforeValidator(wire_time)]


def epoch_ms(value):
    return int(value.timestamp() * 1000)


class Frame(BaseModel):
    # Newer hosts may add fields; capability negotiation decides what either side uses.
    model_config = ConfigDict(extra="ignore", str_strip_whitespace=True)
    # Host free text redacted again on ingestion with the shared secret patterns (stage F, L6).
    REDACT: ClassVar[tuple[str, ...]] = ()

    @model_validator(mode="after")
    def redacted(self):
        for name in self.REDACT:
            value = getattr(self, name)
            if isinstance(value, str) and value:
                setattr(self, name, redact(value))
            elif isinstance(value, dict):
                setattr(self, name, {k: redact(v) if isinstance(v, str) else v for k, v in value.items()})
        return self


class Hello(Frame):
    server_id: str = Field(pattern=r"^[A-Za-z0-9._:-]{1,100}$")
    boot_id: str = Field(default="", max_length=100)
    hostname: str = Field(default="", max_length=100)
    machine: str = Field(default="", max_length=100)
    proto: int = Field(default=PROTO, ge=1, le=100)
    version: str | None = Field(default=None, max_length=40)
    caps: list[str] = Field(default_factory=list, max_length=100)
    harnesses: dict[str, dict] = Field(default_factory=dict, max_length=10)
    # Optional: execution targets configured on the host, offered in the page's target picker.
    targets: list[dict] = Field(default_factory=list, max_length=100)


class Report(Frame):
    tool: sessions.Tool
    session_id: str = Field(max_length=100)
    report: sessions.SessionInput
    surf: str = Field(default="", max_length=100)
    incarnation: int = Field(default=0, ge=0)
    parity: bool = False


class Turn(Frame):
    REDACT: ClassVar[tuple[str, ...]] = ("text", "prompt")
    tool: str = Field(max_length=30)
    surf: str = Field(default="", max_length=100)
    incarnation: int = Field(default=0, ge=0)
    run_id: str | None = Field(default=None, max_length=36)
    generation: int = Field(default=0, ge=0)
    provider_session_id: str = Field(max_length=100)
    provider_turn_id: str = Field(min_length=1, max_length=200)
    root: bool = True
    seq: int = Field(default=0, ge=0)
    text: str = Field(default="", max_length=MAX_ASSEMBLED)
    truncated: bool = False
    missing: bool = False
    prompt: str = Field(default="", max_length=MAX_ASSEMBLED)
    source: Literal["hook", "rollout"] = "hook"
    # A background subagent's hand-back, or a root Stop while background work still ran: not a turn of its own.
    synthetic: bool = False
    interim: bool = False


class Surface(Frame):
    """Names a terminal surface, and optionally the provider session it runs (tool/provider + session id)."""
    surf: str = Field(default="", max_length=100)
    incarnation: int = Field(default=0, ge=0)
    tool: str | None = Field(default=None, max_length=30)
    session_id: str | None = Field(default=None, max_length=100)
    provider: str | None = Field(default=None, max_length=30)
    provider_session_id: str | None = Field(default=None, max_length=100)

    @model_validator(mode="after")
    def identity(self):
        self.tool = self.tool or self.provider
        self.session_id = self.session_id or self.provider_session_id
        if not (self.tool and self.session_id):
            self.tool = self.session_id = None
        return self


class Status(Surface):
    REDACT: ClassVar[tuple[str, ...]] = ("reason",)
    state: Literal[STATES]
    since: WireTime
    reason: str = Field(default="", max_length=300)
    effective_permission_mode: str = Field(default="", max_length=40)


class SnapshotSurface(Status):
    # The host reports terminal state live|lost|exited and the agent's own state separately.
    state: Literal[(*STATES, "live")]
    since: WireTime | None = None
    agent_state: str | None = Field(default=None, max_length=20)
    run_id: str | None = Field(default=None, max_length=36)
    generation: int = Field(default=0, ge=0)
    stop_requested: bool = False
    questions: list[str] = Field(default_factory=list, max_length=100)

    @model_validator(mode="after")
    def merged(self):
        if self.state == "live":
            self.state = self.agent_state if self.agent_state in LIVE_STATES else "idle"
        return self


class Exit(Surface):
    REDACT: ClassVar[tuple[str, ...]] = ("reason",)
    since: WireTime | None = None
    exited_at: WireTime | None = None
    code: int | None = None
    # Stage F remotea: why a run surface ended without a resume (a lost run is reported as exit, code -1).
    reason: str = Field(default="", max_length=300)


class Snapshot(Frame):
    surfaces: list[SnapshotSurface] = Field(default_factory=list, max_length=1000)
    complete: bool = True
    at: WireTime | None = None
    journal: dict = Field(default_factory=dict)


class QuestionItem(Frame):
    id: str = Field(min_length=1, max_length=64)
    text: str = Field(min_length=1, max_length=4000)
    header: str | None = Field(default=None, max_length=200)
    choices: list[dict] | None = Field(default=None, max_length=32)
    multi: bool = False

    REDACT: ClassVar[tuple[str, ...]] = ("text", "header")

    @model_validator(mode="after")
    def clean_choices(self):
        if self.choices is not None:
            self.choices = [{"id": str(c.get("id"))[:64], "label": redact(str(c.get("label")))[:300],
                             **({"description": redact(str(c["description"]))[:1000]} if c.get("description") else {})}
                            for c in self.choices if isinstance(c, dict) and c.get("id") and c.get("label")]
        return self


class Question(Frame):
    question_id: str = Field(min_length=1, max_length=200)
    surf: str = Field(default="", max_length=120)
    incarnation: int = Field(default=0, ge=0)
    run_id: str | None = Field(default=None, max_length=36)
    generation: int = Field(default=0, ge=0)
    provider: str = Field(default="claude", max_length=30)
    provider_session_id: str = Field(min_length=1, max_length=100)
    provider_request_id: str = Field(min_length=1, max_length=200)
    provider_turn_id: str | None = Field(default=None, max_length=200)
    kind: str = Field(default="ask", max_length=30)
    group_id: str = Field(default="", max_length=200)
    items: list[QuestionItem] = Field(min_length=1, max_length=16)
    hold_until: WireTime | None = None


class QuestionResolved(Frame):
    REDACT: ClassVar[tuple[str, ...]] = ("detail", "answer_echo")
    question_id: str = Field(min_length=1, max_length=200)
    generation: int = Field(default=0, ge=0)
    outcome: Literal["accepted", "timed_out", "failed", "canceled", "answered_locally"]
    still_open: bool = False
    detail: str = Field(default="", max_length=300)
    answer_echo: dict[str, str] | None = None


class VerifyResult(Frame):
    REDACT: ClassVar[tuple[str, ...]] = ("tail", "error")
    attempt_id: str = Field(min_length=1, max_length=200)
    run_id: str = Field(max_length=36)
    generation: int = Field(ge=0)
    turn_seq: int = Field(ge=0)
    plan_revision: int = Field(ge=0)
    verify_id: str = Field(default="", max_length=64)
    workspace_fingerprint: str = Field(default="", max_length=300)
    exit_code: int | None = None
    tail: str = Field(default="", max_length=MAX_ASSEMBLED)
    truncated: bool = False
    duration_ms: int = Field(default=0, ge=0)
    timed_out: bool = False
    canceled: bool = False
    error: str | None = Field(default=None, max_length=2000)
    # Stage D (additive on the host): a checkout that changed around the run is a non-pass.
    stale_workspace: bool = False
    initiator: str | None = Field(default=None, max_length=20)
    adopted: str | None = Field(default=None, max_length=200)


class ContextRequest(Frame):
    req_id: str = Field(min_length=1, max_length=200)
    surf: str = Field(default="", max_length=120)
    incarnation: int = Field(default=0, ge=0)
    provider: str = Field(default="claude", max_length=30)
    provider_session_id: str = Field(default="", max_length=100)
    source: str = Field(default="startup", max_length=30)
    cwd: str = Field(default="", max_length=2000)
    run_id: str | None = Field(default=None, max_length=36)
    generation: int = Field(default=0, ge=0)
    task_id: str | None = Field(default=None, max_length=36)


class WorktreesReport(Frame):
    worktrees: list[dict] = Field(default_factory=list, max_length=500)


class Ack(Frame):
    acked_id: str = Field(max_length=100)
    op_id: str | None = Field(default=None, max_length=100)
    ok: bool = True
    error: str | dict | None = None
    retry: bool = False
    outcome: dict | None = None


class Outcome(Frame):
    REDACT: ClassVar[tuple[str, ...]] = ("detail",)
    command_id: str = Field(max_length=100)
    state: Literal["executed", "failed", "rejected", "uncertain"]
    detail: str = Field(default="", max_length=300)
    kind: str | None = Field(default=None, max_length=40)
    result: dict | None = None


DIGEST_EXCLUDED = {"v", "id", "conn_epoch", "at", "op_id", "digest"}


def digest(frame):
    """Operation digest shared with the host: sha256 of canonical JSON of the frame minus its envelope.

    Keys sorted at every level, no whitespace, non-ASCII kept. `t` is included. Outbound commands
    are stamped with it once, when queued; an inbound frame whose digest does not match is refused.
    """
    body = {k: v for k, v in frame.items() if k not in DIGEST_EXCLUDED}
    canonical = json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode()).hexdigest()


def enabled(db, owner):
    row = db.get(OwnerSettings, owner)
    return bool((row.values if row else {}).get("remotea_enabled", False))


def gate(db, owner):
    if not enabled(db, owner):
        raise DomainError("REMOTEA_DISABLED", "Turn on remotea in Settings first.", 403)


def tool_name(value):
    name = HARNESS_TOOLS.get(value, value)
    if name not in sessions.TOOLS:
        raise DomainError("INVALID_ARGUMENT", "Unknown agent tool.")
    return name


def clip_bytes(text, limit):
    data = text.encode()
    if len(data) <= limit:
        return text, False
    return data[:limit].decode(errors="ignore"), True


# ---------------------------------------------------------------- ingestion (one transaction each)


def report(db, bot, host_id, frame):
    """session.report: the same upsert the collector's HTTP PUT runs, attached to the host."""
    result = sessions.upsert(db, bot.owner_id, frame.tool, frame.session_id, frame.report,
                             reveal="records:read" in bot.scopes)
    row = sessions.find(db, bot.owner_id, frame.tool, frame.session_id)
    if host_id and row.host_id not in (None, host_id):
        host_session(db, bot.owner_id, host_id, frame.tool, frame.session_id)  # Refuses: another host's session.
    if host_id:
        row.host_id = host_id
        if frame.surf and frame.incarnation >= row.incarnation:
            row.surf, row.incarnation = frame.surf, frame.incarnation
    emit(db, bot.owner_id, "session.changed", row.id)
    return {"session": result}


def turn(db, bot, host_id, frame):
    """A root turn's final response, stored once per provider turn id however often it is resent."""
    # A host cannot bind a turn to a run Eridani never assigned it.
    run = coordinator.bound(db, host_id, frame.run_id) if frame.run_id else None
    if not frame.root:
        return {"ignored": "subagent"}
    owner = bot.owner_id
    if run is not None and not run.session_pk:
        with contextlib.suppress(DomainError):  # An older generation's turn is kept but does not attach.
            coordinator.attach(db, host_id, run.id, frame.generation, frame.tool, frame.provider_session_id,
                               surf=frame.surf, incarnation=frame.incarnation)
    row = host_session(db, owner, host_id, frame.tool, frame.provider_session_id)
    db.flush()
    row.host_id = host_id or row.host_id
    if frame.surf and frame.incarnation >= row.incarnation:
        row.surf, row.incarnation = frame.surf, frame.incarnation
    text, cut = clip_bytes(frame.text, TURN_TEXT_BYTES)
    prompt, _ = clip_bytes(frame.prompt, 4000)
    if not retention.admit(db, owner, text, prompt):
        text, prompt, cut = "", "", True  # Over today's ingestion quota: the turn counts, its text is dropped.
    inserted = db.scalar(
        insert(AgentTurn)
        .values(id=str(uuid.uuid4()), owner_id=owner, session_pk=row.id, host_id=host_id,
                provider_session_id=frame.provider_session_id, provider_turn_id=frame.provider_turn_id,
                surf=frame.surf, incarnation=frame.incarnation, run_id=run.id if run else None,
                generation=frame.generation,
                seq=frame.seq, role="assistant", text=text, prompt=prompt, truncated=frame.truncated or cut,
                source=frame.source, created_at=now())
        .on_conflict_do_nothing(constraint="uq_agent_turn")
        .returning(AgentTurn.id)
    )
    if inserted is None:
        return {"duplicate": True}
    row.last_turn_seq = max(row.last_turn_seq or 0, frame.seq)
    row.updated_at = now()
    if run is not None and run.session_pk == row.id:
        coordinator.record_turn(db, run, frame)
    emit(db, owner, "agent.changed", row.id)
    return {"turn_id": inserted, "truncated": frame.truncated or cut, "missing": frame.missing}


def host_session(db, owner, host_id, tool, session_id):
    """Claim a provider session for this host. A session another host reported, or one bound to a run
    Eridani assigned another host, is refused: naming its id must not move it (stage F, M2)."""
    row = sessions.claim(db, owner, tool_name(tool), session_id)
    if host_id:
        run = coordinator.run_for_session(db, row, lock=False)
        if (row.host_id and row.host_id != host_id) or (run is not None and run.host_id != host_id):
            raise DomainError("RUN_NOT_BOUND", "That session belongs to another host.", 409)
    return row


def surface_session(db, owner, host_id, frame):
    if frame.tool and frame.session_id:
        return host_session(db, owner, host_id, frame.tool, frame.session_id)
    if not frame.surf:
        raise DomainError("INVALID_ARGUMENT", "Name the surface or the session.")
    return db.scalar(
        select(AgentSession)
        .where(AgentSession.owner_id == owner, AgentSession.host_id == host_id, AgentSession.surf == frame.surf)
        .order_by(AgentSession.incarnation.desc(), AgentSession.updated_at.desc())
        .limit(1)
    )


def apply_state(db, owner, host_id, row, state, since, incarnation, surf, detail):
    """Order by surface incarnation, then the host's own clock; late or replayed states never win."""
    if incarnation < (row.incarnation or 0):
        return False
    if incarnation == (row.incarnation or 0) and row.state_since and since < row.state_since:
        return False
    row.state, row.state_since, row.state_detail = state, since, detail
    row.incarnation, row.host_id = incarnation, host_id or row.host_id
    row.surf = surf or row.surf
    row.updated_at = now()
    emit(db, owner, "session.changed", row.id)
    return True


def status(db, bot, host_id, frame):
    row = surface_session(db, bot.owner_id, host_id, frame)
    if row is None:
        raise DomainError("UNKNOWN_SURFACE", "No session is reported for that surface yet.", 404)
    db.flush()
    detail = {k: v for k, v in (("reason", frame.reason), ("permission_mode", frame.effective_permission_mode)) if v}
    applied = apply_state(db, bot.owner_id, host_id, row, frame.state, frame.since, frame.incarnation, frame.surf, detail)
    if applied:
        attach_by_surface(db, host_id, row, frame)
        coordinator.on_status(db, row, frame, host_id=host_id)
    return {"session_id": row.id, "applied": applied}


def attach_by_surface(db, host_id, row, frame):
    """A status naming the provider session on a run's spawned surface confirms that run's start."""
    if row.run_id or not (frame.surf and frame.tool and frame.session_id):
        return
    from .session_models import AgentRun

    run = db.scalar(select(AgentRun).where(AgentRun.host_id == host_id, AgentRun.surf == frame.surf,
                                           AgentRun.session_pk.is_(None), AgentRun.state == "spawning"))
    if run is not None:
        coordinator.attach(db, host_id, run.id, run.generation, frame.tool, frame.session_id,
                           surf=frame.surf, incarnation=frame.incarnation)


def exited(db, bot, host_id, frame):
    row = surface_session(db, bot.owner_id, host_id, frame)
    if row is None:
        raise DomainError("UNKNOWN_SURFACE", "No session is reported for that surface yet.", 404)
    db.flush()
    detail = {k: v for k, v in (("exit_code", frame.code), ("reason", frame.reason)) if v not in (None, "")}
    applied = apply_state(db, bot.owner_id, host_id, row, "exited", frame.exited_at or frame.since or now(),
                          frame.incarnation, frame.surf, detail)
    if applied:
        said = "; ".join(filter(None, (f"exit code {frame.code}" if frame.code is not None else "", frame.reason)))
        coordinator.on_exit(db, row, detail=said, host_id=host_id)
    return {"session_id": row.id, "applied": applied}


def snapshot(db, bot, host_id, frame):
    """Reconcile after welcome. A complete snapshot ends sessions the host no longer has."""
    at = frame.at or now()
    seen, applied = set(), 0
    for surface in frame.surfaces:
        try:
            row = surface_session(db, bot.owner_id, host_id, surface)
        except DomainError as exc:
            if exc.code != "RUN_NOT_BOUND":
                raise
            continue  # Another host's session: never re-homed by this host's snapshot.
        if row is None:
            continue
        db.flush()
        seen.add(row.id)
        detail = {k: v for k, v in (("reason", surface.reason), ("permission_mode", surface.effective_permission_mode)) if v}
        applied += apply_state(db, bot.owner_id, host_id, row, surface.state, surface.since or at, surface.incarnation,
                               surface.surf, detail)
    ended = 0
    if frame.complete:
        stale = db.scalars(
            select(AgentSession).where(
                AgentSession.owner_id == bot.owner_id, AgentSession.host_id == host_id,
                AgentSession.state.not_in(("", "exited")), AgentSession.id.not_in(seen or {""}),
            )
        )
        for row in stale:
            if row.state_since is None or row.state_since < at:
                ended += apply_state(db, bot.owner_id, host_id, row, "exited", at, row.incarnation, "",
                                     {"reason": "absent_from_snapshot"})
    runs = coordinator.reconcile(db, bot.owner_id, host_id, frame.surfaces, frame.journal)
    return {"applied": applied, "ended": ended, "runs": runs}


def question(db, bot, host_id, frame):
    row, duplicate = coordinator.record_question(db, bot.owner_id, host_id, frame)
    return {"question": row.id, **({"duplicate": True} if duplicate else {})}


def question_resolved(db, bot, host_id, frame):
    row = coordinator.question_resolved(db, host_id, frame)
    return {"question": row.id, "status": row.status}


def context_request(db, bot, host_id, frame):
    return coordinator.context(db, bot.owner_id, host_id, frame)


def worktrees(db, bot, host_id, frame):
    """The host's managed worktrees with unmerged/unpushed checks; the page lists the unmerged ones."""
    host = db.get(AgentHost, host_id)
    host.caps = {**(host.caps or {}), "worktrees": {"at": now().isoformat(), "worktrees": frame.worktrees}}
    emit(db, bot.owner_id, "agent.changed", host_id)
    coordinator.check_integrations(db, bot.owner_id, host_id)  # A sub-task branch may have been merged.
    return {"stored": len(frame.worktrees)}


def verify_result(db, bot, host_id, frame):
    return coordinator.verify_result(db, host_id, frame)


PARITY_FIELDS = ("machine", "environment", "cwd", "repo", "branch", "title", "first_prompt", "last_prompt",
                 "excerpt", "pr_urls", "turn_count", "started_at", "last_active_at", "resume")


def comparable(value):
    return encode(value)


def stable_resume(value):
    """The part of `resume` that names the session; `live` and `hint` describe its current terminal and drift."""
    return {k: v for k in ("cmd", "url") if (v := (value or {}).get(k))}


def clip(value):
    text = json.dumps(comparable(value))
    return text if len(text) <= 300 else text[:297] + "..."


def parity(db, bot, host_id, tool, session_id, body):
    """Compare a shadow collector's report with what the current collector stored; never writes the session."""
    owner = bot.owner_id
    if not sessions.SESSION_ID.match(session_id or ""):
        raise DomainError("INVALID_ARGUMENT", "Use the session's own ID.")
    row = sessions.find(db, owner, tool, session_id)
    values = body.model_dump()
    values["resume"] = {k: v for k, v in values["resume"].items() if v}
    values["cwd"] = sessions.normalize_path(values["cwd"])
    diff = {}
    if row is not None:
        for field in PARITY_FIELDS:
            stored, reported = getattr(row, field), values[field]
            if field == "resume":
                stored, reported = stable_resume(stored), stable_resume(reported)
            if comparable(stored) != comparable(reported):
                diff[field] = {"stored": clip(stored), "reported": clip(reported)}
    advisory(db, f"agent-parity:{owner}:{tool}:{session_id}")
    entry = db.scalar(select(AgentParity).where(
        AgentParity.owner_id == owner, AgentParity.tool == tool, AgentParity.session_id == session_id))
    if entry is None:
        entry = AgentParity(owner_id=owner, tool=tool, session_id=session_id, reports=0)
        db.add(entry)
    entry.host_id = host_id or entry.host_id
    entry.missing, entry.matched, entry.diff = row is None, row is not None and not diff, diff
    entry.reports, entry.updated_at = (entry.reports or 0) + 1, now()
    count_day(db, owner, "missing" if entry.missing else "matched" if entry.matched else "mismatched")
    db.flush()
    return {"matched": entry.matched, "missing": entry.missing, "fields": sorted(diff)}


# Cutover readiness (stage E): this many consecutive clean days, ending today or yesterday.
CLEAN_DAYS = 7


def owner_today(db, owner, clock=None):
    from .domain import preferences, zone

    return (clock or now()).astimezone(zone(preferences(db, owner)["timezone"])).date()


def count_day(db, owner, outcome):
    """One parity report on the owner's local day; insert-or-increment so concurrent reports never collide."""
    day = owner_today(db, owner)
    values = {"reports": 1, "matched": 0, "mismatched": 0, "missing": 0, outcome: 1}
    table = AgentParityDay.__table__
    statement = insert(table).values(id=str(uuid.uuid4()), owner_id=owner, day=day, updated_at=now(), **values)
    db.execute(statement.on_conflict_do_update(
        constraint="uq_agent_parity_day",
        set_={k: table.c[k] + statement.excluded[k] for k in values} | {"updated_at": statement.excluded.updated_at}))


def clean_day(row):
    """A day counts when the new collector reported and no session differed from the current collector.

    `missing` (a session only the new collector saw) does not count against it: the new collector seeing
    more is not a regression, and the old collector often reports the same session minutes later."""
    return row is not None and row.reports > 0 and row.mismatched == 0


def readiness(db, owner, clock=None):
    """{ready, streak, days[...]}: ready after CLEAN_DAYS consecutive clean days ending today or yesterday
    (today may still be in progress)."""
    today = owner_today(db, owner, clock)
    rows = {r.day: r for r in db.scalars(select(AgentParityDay).where(
        AgentParityDay.owner_id == owner, AgentParityDay.day >= today - timedelta(days=CLEAN_DAYS + 1)))}
    start = today if clean_day(rows.get(today)) else today - timedelta(days=1)
    streak = 0
    while clean_day(rows.get(start - timedelta(days=streak))) and streak < CLEAN_DAYS:
        streak += 1
    days = []
    for back in range(CLEAN_DAYS):
        day = today - timedelta(days=back)
        row = rows.get(day)
        counts = {k: getattr(row, k) for k in ("reports", "matched", "mismatched", "missing")} if row else {"reports": 0}
        days.append({"day": day.isoformat(), **counts, "clean": clean_day(row)})
    return {"ready": streak >= CLEAN_DAYS, "streak": streak, "needed": CLEAN_DAYS, "days": days}


def parity_summary(db, owner):
    rows = list(db.scalars(select(AgentParity).where(AgentParity.owner_id == owner)))
    fields = {}
    for row in rows:
        for field in row.diff:
            fields[field] = fields.get(field, 0) + 1
    problems = sorted((r for r in rows if not r.matched), key=lambda r: r.updated_at, reverse=True)[:20]
    matched = sum(r.matched for r in rows)
    missing = sum(r.missing for r in rows)
    return encode({
        "total": len(rows),
        "matched": matched,
        "missing": missing,
        "mismatched": len(rows) - matched - missing,
        "clean": bool(rows) and matched == len(rows),
        "fields": dict(sorted(fields.items(), key=lambda kv: -kv[1])),
        "first_report_at": min((r.created_at for r in rows), default=None),
        "last_report_at": max((r.updated_at for r in rows), default=None),
        "recent": [{"tool": r.tool, "session_id": r.session_id, "missing": r.missing, "diff": r.diff,
                    "updated_at": r.updated_at} for r in problems],
        "cutover": readiness(db, owner),
    })


# ---------------------------------------------------------------- hosts, leases and commands


def connected(host, clock=None):
    return bool(host.lease_until and host.lease_until > (clock or now()))


def public_host(host):
    db = object_session(host)
    return encode({
        "id": host.id, "name": host.name, "machine": host.machine, "connected": connected(host),
        # Over the owner's daily ingestion quota (stage F, L7): turns arrive without their text until tomorrow.
        "status": "degraded" if db is not None and retention.degraded(db, host.owner_id) else "ok",
        "last_seen_at": host.last_seen_at, "conn_epoch": host.conn_epoch, "caps": host.caps.get("caps", []),
        "harnesses": host.caps.get("harnesses", {}), "proto": host.proto,
    })


def admit(credential_id, hello):
    """Register the host and take its lease; the new epoch fences every older connection."""
    with session_scope() as db:
        bot = bot_access.authorize(db, required="agents:control", credential_id=credential_id)
        gate(db, bot.owner_id)
        advisory(db, f"agent-host:{bot.owner_id}:{hello.server_id}")
        host = db.scalar(select(AgentHost).where(AgentHost.owner_id == bot.owner_id, AgentHost.server_id == hello.server_id))
        if host is None:
            hosts = db.scalar(select(func.count(AgentHost.id)).where(AgentHost.credential_id == bot.id))
            if hosts >= HOSTS_PER_CREDENTIAL:
                raise DomainError("HOST_LIMIT", f"This link key already serves {HOSTS_PER_CREDENTIAL} remotea servers.", 403)
            host = AgentHost(owner_id=bot.owner_id, account_id=bot.account_id, server_id=hello.server_id, conn_epoch=0)
            db.add(host)
        elif host.credential_id != bot.id and not successor(db, host.credential_id, bot):
            # A host stays bound to the key it first linked with, or to that key's rotation (stage F, info I2).
            raise DomainError("HOST_BOUND", "That remotea server is linked with another key.", 403)
        host.credential_id, host.account_id = bot.id, bot.account_id
        host.name, host.machine, host.boot_id, host.proto = hello.hostname, hello.machine, hello.boot_id, hello.proto
        host.caps = {**{k: v for k, v in (host.caps or {}).items() if k == "worktrees"},
                     "caps": hello.caps, "harnesses": hello.harnesses, "targets": hello.targets}
        host.conn_epoch = (host.conn_epoch or 0) + 1
        host.last_seen_at, host.lease_until = now(), now() + LEASE
        db.flush()
        emit(db, bot.owner_id, "agent.changed", host.id)
        return {"owner": bot.owner_id, "host_id": host.id, "conn_epoch": host.conn_epoch}


def successor(db, previous_id, bot):
    """The new key is a rotation of the host's previous one: same account and name (the remotea link key)."""
    from .models import BotCredential

    previous = db.get(BotCredential, previous_id) if previous_id else None
    return previous is None or (previous.account_id == bot.account_id and previous.name == bot.name
                                and previous.owner_id == bot.owner_id)


def holding(db, credential_id, host_id, epoch):
    """Re-check the key (a privileged write) and the lease before any effect of this connection."""
    from .models import BotCredential

    key = db.get(BotCredential, credential_id)
    if key is not None and not enabled(db, key.owner_id):
        # Disabling also revokes the key; the host still hears that remotea was turned off.
        raise Closed(4403, "disabled", {"reason": "disabled"})
    bot = bot_access.authorize(db, required="agents:control", credential_id=credential_id)
    host = db.get(AgentHost, host_id, with_for_update=True)
    if host is None or host.owner_id != bot.owner_id or host.conn_epoch != epoch:
        raise Closed(4001, "superseded", {"reason": "superseded"})
    host.last_seen_at, host.lease_until = now(), now() + LEASE
    return bot


def probe(credential_id, hello):
    """`remotea eridani connect` checks the key with boot_id probe-*; that must not take the live lease."""
    with session_scope() as db:
        bot = bot_access.authorize(db, required="agents:control", credential_id=credential_id)
        gate(db, bot.owner_id)
        host = db.scalar(select(AgentHost).where(AgentHost.owner_id == bot.owner_id, AgentHost.server_id == hello.server_id))
        return {"host_id": host.id if host else None, "conn_epoch": host.conn_epoch if host else 0}


def refused_access(exc):
    """After the upgrade: 4401 when the key no longer authenticates, 4403 when it lacks permission."""
    if exc.status == 401:
        return Closed(4401, "auth failed", {"reason": "access_revoked"})
    reason = {"REMOTEA_DISABLED": "disabled", "HOST_LIMIT": "host_limit", "HOST_BOUND": "host_bound"}
    return Closed(4403, "forbidden", {"reason": reason.get(exc.code, "access_revoked")})


def wire(frame):
    """Protocol timestamps (at, server_time, expires_at) travel as epoch ms; an `outcome` keeps the API's own encoding."""
    return {k: epoch_ms(v) if isinstance(v, datetime) else v for k, v in frame.items()}


def release(host_id, epoch):
    with session_scope() as db:
        host = db.get(AgentHost, host_id, with_for_update=True)
        if host and host.conn_epoch == epoch:
            host.lease_until, host.last_seen_at = now(), now()
            emit(db, host.owner_id, "agent.changed", host.id)


def drain(db, host, epoch):
    """Queued commands the host advertised, each sent under this connection's envelope; unacked ones resent.

    The payload (op_id, digest, expiry, fences) is fixed when the command is queued (F06); only the
    envelope changes. Expiry is the coordinator's sweep; here an expired command is simply not sent.
    """
    clock = now()
    kinds = [k for k, cap in COMMANDS.items() if cap in host.caps.get("caps", [])]
    if not kinds:
        return []
    rows = list(db.scalars(
        select(AgentCommand)
        .where(AgentCommand.host_id == host.id, AgentCommand.kind.in_(kinds), AgentCommand.expires_at > clock,
               (AgentCommand.status == "queued") | ((AgentCommand.status == "sent") & (AgentCommand.updated_at <= clock - RESEND)))
        .order_by(AgentCommand.created_at).limit(20).with_for_update(skip_locked=True)
    ))
    frames = []
    for row in rows:
        row.status, row.updated_at = "sent", clock
        frames.append({**row.payload, "op_id": row.op_id, "digest": row.digest})
    return frames


def command_for(db, host_id, op_id=None, envelope=None):
    row = db.scalar(select(AgentCommand).where(AgentCommand.host_id == host_id, AgentCommand.op_id == op_id)) \
        if op_id else None
    if row is None and envelope:
        row = db.get(AgentCommand, envelope)
    return row if row is not None and row.host_id == host_id else None


def outcome(db, host_id, frame):
    """command.outcome is separate from the ack: what the host did with a command."""
    row = command_for(db, host_id, frame.command_id, frame.command_id)
    if row is not None:
        coordinator.on_outcome(db, row, frame.state, frame.detail, frame.result)
    return {"recorded": row is not None}


def acked(db, host_id, frame):
    """The host journaled the command (accepted), or answered a duplicate with its recorded outcome."""
    row = command_for(db, host_id, frame.op_id, frame.acked_id)
    if row is None or row.status not in ("queued", "sent"):
        return
    if not frame.ok:
        if frame.retry:
            row.status, row.updated_at = "queued", now()
        else:
            error = frame.error if isinstance(frame.error, str) else (frame.error or {}).get("code", "refused")
            coordinator.on_outcome(db, row, "rejected", f"refused by the host: {error}"[:300])
        return
    state = (frame.outcome or {}).get("state", "accepted")
    if state in ("executed", "failed", "rejected", "uncertain"):
        coordinator.on_outcome(db, row, state, str((frame.outcome or {}).get("detail") or "")[:300],
                               (frame.outcome or {}).get("result"))
        return
    row.status, row.updated_at = "acked", now()
    coordinator.redact_spawn(row)
    facts = {k: v for k, v in (frame.outcome or {}).items() if k in ("surf", "incarnation")}
    row.outcome = {"state": "accepted", **facts}
    if row.kind == "spawn" and row.run_id and facts.get("surf"):
        from .session_models import AgentRun

        run = db.get(AgentRun, row.run_id, with_for_update=True)
        if run is not None and not run.surf:
            run.surf, run.incarnation = str(facts["surf"])[:120], int(facts.get("incarnation") or 0)


# ---------------------------------------------------------------- the socket

LINKS = {}
NO_ACK = object()


def wake(host_id):
    """Called after a commit that queued a command: the host's link drains now instead of at the next tick."""
    link = LINKS.get(host_id)
    if link is not None and link.loop is not None:
        link.loop.call_soon_threadsafe(link.wakeup.set)


class Link:
    def __init__(self, websocket, credential_id):
        self.loop = asyncio.get_running_loop()
        self.wakeup = asyncio.Event()
        self.websocket, self.credential_id = websocket, credential_id
        self.owner = self.host_id = None
        self.epoch = 0
        self.window = (time.monotonic(), 0, 0)
        self.strikes = 0
        self.chunks = {}
        self.snapshots = {}
        self.closed = False
        self.caps = []
        # Queued effects wait until this connection's sessions.snapshot is reconciled (C08).
        self.reconciled = False

    async def send(self, kind, **values):
        frame = encode(wire({"v": PROTO, "id": str(uuid.uuid4()), "conn_epoch": self.epoch, "t": kind, "at": now(), **values}))
        text = json.dumps(frame)
        if len(text.encode()) <= MAX_FRAME:
            await self.websocket.send_text(text)
            return
        # Large payloads go out in parts the host reassembles by the logical frame id.
        size = MAX_FRAME // 2
        parts = [text[i:i + size] for i in range(0, len(text), size)]
        for i, data in enumerate(parts):
            await self.websocket.send_text(json.dumps({
                "v": PROTO, "id": str(uuid.uuid4()), "conn_epoch": self.epoch, "t": "chunk", "at": epoch_ms(now()),
                "part": {"of": frame["id"], "i": i, "n": len(parts)}, "data": data}))

    async def close(self, code, reason, bye=None):
        if self.closed:
            return
        self.closed = True
        with contextlib.suppress(Exception):  # The peer may already be gone.
            if bye is not None:
                await self.send("bye", **bye)
            await self.websocket.close(code, reason)

    def metered(self, size):
        start, frames, total = self.window
        clock = time.monotonic()
        if clock - start >= 60:
            start, frames, total = clock, 0, 0
        frames, total = frames + 1, total + size
        self.window = (start, frames, total)
        return frames <= FRAMES_PER_MINUTE and total <= BYTES_PER_MINUTE

    def assemble(self, frame):
        part = frame.get("part") or {}
        logical, i, n, data = part.get("of"), part.get("i"), part.get("n"), frame.get("data")
        if not (isinstance(logical, str) and 0 < len(logical) <= 100 and isinstance(i, int) and isinstance(n, int)
                and 0 <= i < n <= MAX_PARTS and isinstance(data, str)):
            raise DomainError("INVALID_ARGUMENT", "Malformed chunk.")
        clock = time.monotonic()
        for key in [k for k, v in self.chunks.items() if clock - v["at"] > PARTIAL_SECONDS]:
            self.chunks.pop(key)
        if logical not in self.chunks and len(self.chunks) >= MAX_PENDING_CHUNKS:
            raise DomainError("RATE_LIMITED", "Too many partial frames.", 429)
        entry = self.chunks.setdefault(logical, {"n": n, "parts": {}, "size": 0, "at": clock})
        if entry["n"] != n:
            raise DomainError("INVALID_ARGUMENT", "Chunk count changed.")
        if i not in entry["parts"]:
            entry["size"] += len(data.encode())
        entry["parts"][i] = data
        if entry["size"] > MAX_ASSEMBLED:
            self.chunks.pop(logical)
            raise DomainError("INVALID_ARGUMENT", "Frame too large.", 413)
        if len(entry["parts"]) < n:
            return None
        self.chunks.pop(logical)
        return json.loads("".join(entry["parts"][k] for k in range(n)))

    def snapshot_parts(self, frame):
        """The host splits a large sessions.snapshot into frames sharing part.of; reconcile once all arrive."""
        part = frame.get("part")
        if not part:
            return frame
        logical, i, n = part.get("of"), part.get("i"), part.get("n")
        if not (isinstance(logical, str) and 0 < len(logical) <= 100 and isinstance(i, int) and isinstance(n, int)
                and 0 < n <= MAX_PARTS and 1 <= i <= n):
            # remotea numbers snapshot parts 1..n; 0..n used to be accepted, so n parts could arrive with one missing.
            raise DomainError("INVALID_ARGUMENT", "Malformed snapshot part.")
        clock = time.monotonic()
        for key in [k for k, v in self.snapshots.items() if clock - v["at"] > PARTIAL_SECONDS]:
            self.snapshots.pop(key)  # A partial snapshot that never completed (stage F, L3).
        if logical not in self.snapshots and len(self.snapshots) >= MAX_PENDING_CHUNKS:
            raise DomainError("RATE_LIMITED", "Too many partial snapshots.", 429)
        entry = self.snapshots.setdefault(logical, {"n": n, "parts": {}, "sizes": {}, "first": frame, "at": clock})
        surfaces = frame.get("surfaces") or []
        if not isinstance(surfaces, list) or entry["n"] != n:
            raise DomainError("INVALID_ARGUMENT", "Malformed snapshot part.")
        entry["parts"][i] = surfaces
        entry["sizes"][i] = len(json.dumps(surfaces))
        if sum(len(p) for p in entry["parts"].values()) > 1000 or sum(entry["sizes"].values()) > MAX_ASSEMBLED:
            self.snapshots.pop(logical)
            raise DomainError("INVALID_ARGUMENT", "Snapshot too large.", 413)
        if len(entry["parts"]) < n:
            return None
        self.snapshots.pop(logical)
        return {**entry["first"], "surfaces": [s for k in sorted(entry["parts"]) for s in entry["parts"][k]]}

    def effect(self, handler, frame):
        def run():
            with session_scope() as db:
                bot = holding(db, self.credential_id, self.host_id, self.epoch)
                return handler(db, bot, self.host_id, frame)
        return run_in_threadpool(run)

    async def dispatch(self, frame):
        kind = frame.get("t")
        if kind == "session.report":
            body = Report.model_validate(frame)
            if body.parity:
                return await self.effect(lambda db, bot, host, f: parity(db, bot, host, f.tool, f.session_id, f.report), body)
            return await self.effect(report, body)
        if kind == "turn":
            return await self.effect(turn, Turn.model_validate(frame))
        if kind == "status":
            return await self.effect(status, Status.model_validate(frame))
        if kind == "exit":
            return await self.effect(exited, Exit.model_validate(frame))
        if kind == "sessions.snapshot":
            frame = self.snapshot_parts(frame)
            if frame is None:
                return {"pending": True}
            result = await self.effect(snapshot, Snapshot.model_validate(frame))
            await self.send("runs.snapshot", runs=result.pop("runs"), questions=[])
            self.reconciled = True
            self.wakeup.set()  # Resend what is still authorized, now that both sides agree.
            return result
        if kind == "question":
            return await self.effect(question, Question.model_validate(frame))
        if kind == "question.resolved":
            return await self.effect(question_resolved, QuestionResolved.model_validate(frame))
        if kind == "context.request":
            # Ephemeral: answered with context.response by req_id, never acked or replayed.
            body = ContextRequest.model_validate(frame)
            try:
                reply = await self.effect(context_request, body)
            except DomainError as exc:
                reply = {"text": f"Eridani could not load this session's context ({exc.code}).", "scope": "candidates"}
            await self.send("context.response", req_id=body.req_id, **reply)
            return NO_ACK
        if kind == "verify.result":
            return await self.effect(verify_result, VerifyResult.model_validate(frame))
        if kind == "worktrees.report":
            return await self.effect(worktrees, WorktreesReport.model_validate(frame))
        if kind == "ack":
            body = Ack.model_validate(frame)
            await self.effect(lambda db, bot, host, f: acked(db, host, f), body)
            return NO_ACK
        if kind == "command.outcome":
            return await self.effect(lambda db, bot, host, f: outcome(db, host, f), Outcome.model_validate(frame))
        if kind == "bye":
            raise Closed(1000, str(frame.get("reason") or "bye")[:60])
        raise DomainError("UNSUPPORTED", f"This server does not handle {str(kind)[:40]!r} frames.")

    async def handle(self, text):
        if len(text.encode()) > MAX_FRAME:
            raise Closed(1009, "frame too large")
        if not self.metered(len(text.encode())):
            self.strikes += 1
            if self.strikes > 3:
                raise Closed(1008, "rate limited", {"reason": "rate_limited"})
            return {"ok": False, "retry": True, "error": {"code": "RATE_LIMITED", "message": "Slow down."}}
        try:
            frame = json.loads(text)
        except ValueError:
            raise Closed(4400, "invalid json") from None
        if not isinstance(frame, dict) or frame.get("v") != PROTO or not isinstance(frame.get("id"), str):
            raise Closed(4400, "invalid envelope")
        identity = frame["id"][:100]
        echo = {"op_id": frame["op_id"][:100]} if isinstance(frame.get("op_id"), str) else {}
        if frame.get("conn_epoch") not in (None, self.epoch):
            return {"acked_id": identity, **echo, "ok": False, "retry": True,
                    "error": {"code": "STALE_EPOCH", "message": "Superseded connection."}}
        try:
            if frame.get("t") == "chunk":
                frame = self.assemble(frame)
                if frame is None:
                    return None
                if not isinstance(frame, dict) or frame.get("t") == "chunk" or not isinstance(frame.get("id"), str):
                    raise DomainError("INVALID_ARGUMENT", "Malformed chunked frame.")
                identity = frame["id"][:100]
            if isinstance(frame.get("op_id"), str) and not isinstance(frame.get("digest"), str) and "digest" in self.caps:
                # A host that advertises `digest` stamps every op_id frame; a bare one is refused (stage F).
                return {"acked_id": identity, **echo, "ok": False,
                        "error": {"code": "DIGEST_REQUIRED", "message": "Frames with an op_id carry its digest."}}
            if isinstance(frame.get("digest"), str) and isinstance(frame.get("op_id"), str) \
                    and frame["digest"] != digest(frame):
                logger.warning("agent link: digest mismatch on %s op_id=%s host=%s", str(frame.get("t"))[:40],
                               frame["op_id"][:100], self.host_id)
                return {"acked_id": identity, **echo, "ok": False,
                        "error": {"code": "DIGEST_MISMATCH", "message": "The frame does not match its digest."}}
            outcome = await self.dispatch(frame)
        except ValidationError as exc:
            errors = [{"loc": [str(p) for p in e["loc"]], "msg": e["msg"]} for e in exc.errors()[:5]]
            return {"acked_id": identity, **echo, "ok": False,
                    "error": {"code": "INVALID_ARGUMENT", "message": "Invalid frame.", "data": errors}}
        except DomainError as exc:
            if exc.status == 401 or exc.code in {"ACCESS_REVOKED", "INSUFFICIENT_SCOPE", "READ_ONLY", "REMOTEA_DISABLED",
                                                  "PERSONAL_WORKSPACE"}:
                raise refused_access(exc) from None
            retry = {"retry": True} if exc.code in RETRYABLE else {}
            return {"acked_id": identity, **echo, "ok": False, **retry, "error": {"code": exc.code, "message": exc.message}}
        if outcome is NO_ACK:
            return None
        return {"acked_id": identity, **echo, "ok": True, **({"outcome": outcome} if outcome is not None else {})}

    async def tick(self):
        def run():
            with session_scope() as db:
                holding(db, self.credential_id, self.host_id, self.epoch)
                host = db.get(AgentHost, self.host_id)
                if not self.reconciled and "sessions.snapshot" in host.caps.get("caps", []):
                    return []
                return drain(db, host, self.epoch)
        for frame in await run_in_threadpool(run):
            kind = frame.pop("t")
            await self.send(kind, **frame)


async def refuse(websocket, status, code, message):
    """Before the upgrade completes: an HTTP status when the server supports it, else policy close."""
    if "websocket.http.response" in websocket.scope.get("extensions", {}):
        await websocket.send_denial_response(JSONResponse({"error": {"code": code, "message": message}}, status_code=status))
    else:
        await websocket.close(1008, code)


async def handshake(websocket):
    settings = get_settings()
    if settings.maintenance_mode:
        return await refuse(websocket, 503, "MAINTENANCE", "Eridani is being updated. Please try again shortly.")
    # A browser always sends Origin; the remotea server never does.
    if websocket.headers.get("origin") or websocket.headers.get("sec-fetch-site") == "cross-site":
        return await refuse(websocket, 403, "NOT_AUTHORIZED", "This origin is not allowed.")
    if SUBPROTOCOL not in websocket.scope.get("subprotocols", []):
        return await refuse(websocket, 400, "INVALID_ARGUMENT", f"Use the {SUBPROTOCOL} subprotocol.")
    try:
        credential_id = await run_in_threadpool(bot_access.authenticate, websocket)

        def check():
            with session_scope() as db:
                bot = bot_access.authorize(db, required="agents:control", credential_id=credential_id)
                gate(db, bot.owner_id)
        await run_in_threadpool(check)
    except DomainError as exc:
        return await refuse(websocket, exc.status, exc.code, exc.message)
    return credential_id


@router.websocket(PATH)
async def agent_link(websocket: WebSocket):
    credential_id = await handshake(websocket)
    if credential_id is None:
        return
    await websocket.accept(subprotocol=SUBPROTOCOL)
    link = Link(websocket, credential_id)
    try:
        try:
            text = await asyncio.wait_for(websocket.receive_text(), HELLO_SECONDS)
            frame = json.loads(text) if len(text.encode()) <= MAX_FRAME else None
            if not isinstance(frame, dict) or frame.get("v") != PROTO or frame.get("t") != "hello":
                raise ValueError
            hello = Hello.model_validate(frame)
        except (TimeoutError, ValueError, ValidationError):
            raise Closed(4400, "expected hello") from None
        try:
            if hello.boot_id.startswith("probe-"):
                checked = await run_in_threadpool(probe, credential_id, hello)
                await link.send("welcome", acked_id=frame["id"][:100] if isinstance(frame.get("id"), str) else "",
                                **checked, server_time=now(), replay_horizon=REPLAY_HORIZON_DAYS * 86400, probe=True,
                                proto=PROTO, caps=SERVER_CAPS, max_frame=MAX_FRAME)
                await asyncio.wait_for(websocket.receive(), HELLO_SECONDS)
                raise Closed(1000, "probe")
            admitted = await run_in_threadpool(admit, credential_id, hello)
        except DomainError as exc:
            raise refused_access(exc) from None
        except TimeoutError:
            raise Closed(1000, "probe") from None
        link.owner, link.host_id, link.epoch = admitted["owner"], admitted["host_id"], admitted["conn_epoch"]
        link.caps = list(hello.caps)
        previous = LINKS.get(link.host_id)
        LINKS[link.host_id] = link
        if previous is not None:
            await previous.close(4001, "superseded", {"reason": "superseded"})
        await link.send("welcome", acked_id=str(frame.get("id", ""))[:100], host_id=link.host_id, conn_epoch=link.epoch,
                        server_time=now(), replay_horizon=REPLAY_HORIZON_DAYS * 86400, lease_seconds=int(LEASE.total_seconds()),
                        proto=PROTO, caps=SERVER_CAPS, max_frame=MAX_FRAME)
        receiving = waking = None
        while not link.closed:
            receiving = receiving or asyncio.ensure_future(websocket.receive())
            waking = waking or asyncio.ensure_future(link.wakeup.wait())
            done, _ = await asyncio.wait({receiving, waking}, timeout=TICK_SECONDS, return_when=asyncio.FIRST_COMPLETED)
            if receiving not in done:
                if waking in done:
                    link.wakeup.clear()
                    waking = None
                await link.tick()
                continue
            message, receiving = receiving.result(), None
            if message["type"] == "websocket.disconnect":
                break
            if message.get("text") is None:
                raise Closed(1003, "text frames only")
            reply = await link.handle(message["text"])
            if reply is not None:
                await link.send("ack", **reply)
        for pending in (receiving, waking):
            if pending is not None:
                pending.cancel()
    except Closed as closed:
        await link.close(closed.code, closed.reason, closed.bye)
    except Exception:  # noqa: BLE001 - a peer disconnect mid-send surfaces as transport errors
        await link.close(1011, "internal error")
    finally:
        if link.host_id and LINKS.get(link.host_id) is link:
            LINKS.pop(link.host_id, None)
        if link.host_id:
            await run_in_threadpool(release, link.host_id, link.epoch)


async def shutdown():
    """Lifespan shutdown: tell every host to reconnect after the restart instead of waiting out the lease."""
    await asyncio.gather(*(link.close(1012, "restart", {"reason": "restart"}) for link in list(LINKS.values())),
                         return_exceptions=True)
    LINKS.clear()
