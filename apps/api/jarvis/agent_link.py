"""The agent link: one outbound WebSocket from each remotea server (docs/AGENT_CONTROL.md).

Stage B is read-only: the host reports sessions, final responses per turn, terminal state and
reconcile snapshots, and runs collector parity. Commands to the host (spawn, input, answer, stop)
arrive in stage C through the agent_commands drain loop sketched here.

The http `security` middleware does not run for websocket scopes, so the handshake repeats the
maintenance, Origin and size guards itself. Every inbound message runs in its own short
transaction that re-checks the key (a privileged write) and the connection's lease epoch.
"""

import asyncio
import contextlib
import hashlib
import json
import time
import uuid
from datetime import UTC, datetime, timedelta
from typing import Annotated, Literal

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
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from starlette.concurrency import run_in_threadpool

from . import agent_sessions as sessions
from . import bot_access
from .config import get_settings
from .db import session_scope
from .domain import DomainError, advisory, emit, encode
from .models import OwnerSettings, now
from .session_models import AgentCommand, AgentHost, AgentParity, AgentSession, AgentTurn

router = APIRouter()
PATH = "/api/v1/external/agents/link"
SUBPROTOCOL = "eridani.v1"
PROTO = 1
MAX_FRAME = 262_144  # Matches uvicorn --ws-max-size in deploy.py.
MAX_ASSEMBLED = 1_048_576
MAX_PARTS = 64
MAX_PENDING_CHUNKS = 4
FRAMES_PER_MINUTE = 1200
BYTES_PER_MINUTE = 16 * 1_048_576
HELLO_SECONDS = 15
TICK_SECONDS = 30  # Re-authorize, renew the lease and drain commands.
LEASE = timedelta(seconds=90)
REPLAY_HORIZON_DAYS = 7
TURN_TEXT_BYTES = 65_536
SERVER_CAPS = ["chunk", "chunking", "session.report", "session.report.parity", "turn", "status", "exit",
               "sessions.snapshot", "command.outcome"]
# Command kind -> host capability it needs. Stage C adds spawn/input/answer/stop/verify.
COMMANDS: dict[str, str] = {}
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


class Hello(Frame):
    server_id: str = Field(pattern=r"^[A-Za-z0-9._:-]{1,100}$")
    boot_id: str = Field(default="", max_length=100)
    hostname: str = Field(default="", max_length=100)
    machine: str = Field(default="", max_length=100)
    proto: int = Field(default=PROTO, ge=1, le=100)
    version: str | None = Field(default=None, max_length=40)
    caps: list[str] = Field(default_factory=list, max_length=100)
    harnesses: dict[str, dict] = Field(default_factory=dict, max_length=10)


class Report(Frame):
    tool: sessions.Tool
    session_id: str = Field(max_length=100)
    report: sessions.SessionInput
    surf: str = Field(default="", max_length=100)
    incarnation: int = Field(default=0, ge=0)
    parity: bool = False


class Turn(Frame):
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
    state: Literal[STATES]
    since: WireTime
    reason: str = Field(default="", max_length=300)
    effective_permission_mode: str = Field(default="", max_length=40)


class SnapshotSurface(Status):
    # The host reports terminal state live|lost|exited and the agent's own state separately.
    state: Literal[(*STATES, "live")]
    agent_state: str | None = Field(default=None, max_length=20)

    @model_validator(mode="after")
    def merged(self):
        if self.state == "live":
            self.state = self.agent_state if self.agent_state in LIVE_STATES else "idle"
        return self


class Exit(Surface):
    since: WireTime | None = None
    exited_at: WireTime | None = None
    code: int | None = None


class Snapshot(Frame):
    surfaces: list[SnapshotSurface] = Field(default_factory=list, max_length=1000)
    complete: bool = True
    at: WireTime | None = None


class Ack(Frame):
    acked_id: str = Field(max_length=100)
    ok: bool = True
    outcome: dict | None = None


class Outcome(Frame):
    command_id: str = Field(max_length=100)
    state: Literal["executed", "failed", "rejected", "uncertain"]
    detail: str = Field(default="", max_length=300)


DIGEST_EXCLUDED = {"v", "id", "conn_epoch", "at", "op_id", "digest"}


def digest(frame):
    """Operation digest shared with the host: sha256 of canonical JSON of the frame minus its envelope.

    Keys sorted at every level, no whitespace, non-ASCII kept. `t` is included. Stage C stamps
    outbound commands with it; stage B accepts inbound digests without enforcing them.
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
    if host_id:
        row.host_id = host_id
        if frame.surf and frame.incarnation >= row.incarnation:
            row.surf, row.incarnation = frame.surf, frame.incarnation
    emit(db, bot.owner_id, "session.changed", row.id)
    return {"session": result}


def turn(db, bot, host_id, frame):
    """A root turn's final response, stored once per provider turn id however often it is resent."""
    if frame.run_id:
        # No runs exist before stage C; a host cannot bind a turn to a run Eridani never assigned it.
        raise DomainError("RUN_NOT_BOUND", "That run is not associated with this host.", 409)
    if not frame.root:
        return {"ignored": "subagent"}
    owner = bot.owner_id
    row = sessions.claim(db, owner, tool_name(frame.tool), frame.provider_session_id)
    db.flush()
    row.host_id = host_id or row.host_id
    if frame.surf and frame.incarnation >= row.incarnation:
        row.surf, row.incarnation = frame.surf, frame.incarnation
    text, cut = clip_bytes(frame.text, TURN_TEXT_BYTES)
    prompt, _ = clip_bytes(frame.prompt, 4000)
    inserted = db.scalar(
        insert(AgentTurn)
        .values(id=str(uuid.uuid4()), owner_id=owner, session_pk=row.id, host_id=host_id,
                provider_session_id=frame.provider_session_id, provider_turn_id=frame.provider_turn_id,
                surf=frame.surf, incarnation=frame.incarnation, run_id=None, generation=frame.generation,
                seq=frame.seq, role="assistant", text=text, prompt=prompt, truncated=frame.truncated or cut,
                source=frame.source, created_at=now())
        .on_conflict_do_nothing(constraint="uq_agent_turn")
        .returning(AgentTurn.id)
    )
    if inserted is None:
        return {"duplicate": True}
    row.last_turn_seq = max(row.last_turn_seq or 0, frame.seq)
    row.updated_at = now()
    emit(db, owner, "agent.changed", row.id)
    return {"turn_id": inserted, "truncated": frame.truncated or cut, "missing": frame.missing}


def surface_session(db, owner, host_id, frame):
    if frame.tool and frame.session_id:
        return sessions.claim(db, owner, tool_name(frame.tool), frame.session_id)
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
    return {"session_id": row.id, "applied": applied}


def exited(db, bot, host_id, frame):
    row = surface_session(db, bot.owner_id, host_id, frame)
    if row is None:
        raise DomainError("UNKNOWN_SURFACE", "No session is reported for that surface yet.", 404)
    db.flush()
    detail = {"exit_code": frame.code} if frame.code is not None else {}
    applied = apply_state(db, bot.owner_id, host_id, row, "exited", frame.exited_at or frame.since or now(),
                          frame.incarnation, frame.surf, detail)
    return {"session_id": row.id, "applied": applied}


def snapshot(db, bot, host_id, frame):
    """Reconcile after welcome. A complete snapshot ends sessions the host no longer has."""
    at = frame.at or now()
    seen, applied = set(), 0
    for surface in frame.surfaces:
        row = surface_session(db, bot.owner_id, host_id, surface)
        if row is None:
            continue
        db.flush()
        seen.add(row.id)
        detail = {k: v for k, v in (("reason", surface.reason), ("permission_mode", surface.effective_permission_mode)) if v}
        applied += apply_state(db, bot.owner_id, host_id, row, surface.state, surface.since, surface.incarnation, surface.surf, detail)
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
    # Stage C fills the desired run state; until then the host has nothing to continue.
    return {"applied": applied, "ended": ended, "runs": []}


PARITY_FIELDS = ("machine", "environment", "cwd", "repo", "branch", "title", "first_prompt", "last_prompt",
                 "excerpt", "pr_urls", "turn_count", "started_at", "last_active_at", "resume")


def comparable(value):
    return encode(value)


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
            stored, reported = comparable(getattr(row, field)), comparable(values[field])
            if stored != reported:
                diff[field] = {"stored": clip(getattr(row, field)), "reported": clip(values[field])}
    advisory(db, f"agent-parity:{owner}:{tool}:{session_id}")
    entry = db.scalar(select(AgentParity).where(
        AgentParity.owner_id == owner, AgentParity.tool == tool, AgentParity.session_id == session_id))
    if entry is None:
        entry = AgentParity(owner_id=owner, tool=tool, session_id=session_id, reports=0)
        db.add(entry)
    entry.host_id = host_id or entry.host_id
    entry.missing, entry.matched, entry.diff = row is None, row is not None and not diff, diff
    entry.reports, entry.updated_at = (entry.reports or 0) + 1, now()
    db.flush()
    return {"matched": entry.matched, "missing": entry.missing, "fields": sorted(diff)}


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
    })


# ---------------------------------------------------------------- hosts, leases and commands


def connected(host, clock=None):
    return bool(host.lease_until and host.lease_until > (clock or now()))


def public_host(host):
    return encode({
        "id": host.id, "name": host.name, "machine": host.machine, "connected": connected(host),
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
            host = AgentHost(owner_id=bot.owner_id, account_id=bot.account_id, server_id=hello.server_id, conn_epoch=0)
            db.add(host)
        host.credential_id, host.account_id = bot.id, bot.account_id
        host.name, host.machine, host.boot_id, host.proto = hello.hostname, hello.machine, hello.boot_id, hello.proto
        host.caps = {"caps": hello.caps, "harnesses": hello.harnesses}
        host.conn_epoch = (host.conn_epoch or 0) + 1
        host.last_seen_at, host.lease_until = now(), now() + LEASE
        db.flush()
        emit(db, bot.owner_id, "agent.changed", host.id)
        return {"owner": bot.owner_id, "host_id": host.id, "conn_epoch": host.conn_epoch}


def holding(db, credential_id, host_id, epoch):
    """Re-check the key (a privileged write) and the lease before any effect of this connection."""
    bot = bot_access.authorize(db, required="agents:control", credential_id=credential_id)
    if not enabled(db, bot.owner_id):
        raise Closed(4403, "disabled", {"reason": "disabled"})
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
    return Closed(4403, "forbidden", {"reason": "disabled" if exc.code == "REMOTEA_DISABLED" else "access_revoked"})


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
    """Stage C: queued commands the host advertised, sent under this connection's envelope."""
    clock = now()
    for row in db.scalars(select(AgentCommand).where(
            AgentCommand.host_id == host.id, AgentCommand.status.in_(("queued", "sent")), AgentCommand.expires_at <= clock)):
        row.status, row.updated_at = "expired", clock
    kinds = [k for k, cap in COMMANDS.items() if cap in host.caps.get("caps", [])]
    if not kinds:
        return []
    rows = list(db.scalars(
        select(AgentCommand)
        .where(AgentCommand.host_id == host.id, AgentCommand.status == "queued", AgentCommand.kind.in_(kinds))
        .order_by(AgentCommand.created_at).limit(20).with_for_update(skip_locked=True)
    ))
    frames = []
    for row in rows:
        row.status, row.updated_at = "sent", clock
        frames.append({"t": row.kind, "id": row.id, "command_id": row.id, "op_id": row.op_id, "digest": row.digest,
                       "run_id": row.run_id, "generation": row.generation,
                       "expires_at": row.expires_at, **row.payload})
    return frames


def outcome(db, host_id, frame):
    """command.outcome is separate from the ack: what the host did with a command."""
    row = db.get(AgentCommand, frame.command_id) or db.scalar(
        select(AgentCommand).where(AgentCommand.host_id == host_id, AgentCommand.op_id == frame.command_id))
    if row and row.host_id == host_id:
        row.status, row.updated_at = frame.state, now()
        row.outcome = {"state": frame.state, "detail": frame.detail}
    return {"recorded": bool(row and row.host_id == host_id)}


def acked(db, host_id, frame):
    row = db.get(AgentCommand, frame.acked_id)
    if row and row.host_id == host_id and row.status == "sent":
        row.status, row.updated_at = ("acked" if frame.ok else "rejected"), now()
        row.outcome = frame.outcome if not frame.ok else row.outcome


# ---------------------------------------------------------------- the socket

LINKS = {}


class Link:
    def __init__(self, websocket, credential_id):
        self.websocket, self.credential_id = websocket, credential_id
        self.owner = self.host_id = None
        self.epoch = 0
        self.window = (time.monotonic(), 0, 0)
        self.strikes = 0
        self.chunks = {}
        self.snapshots = {}
        self.closed = False

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
        if logical not in self.chunks and len(self.chunks) >= MAX_PENDING_CHUNKS:
            raise DomainError("RATE_LIMITED", "Too many partial frames.", 429)
        entry = self.chunks.setdefault(logical, {"n": n, "parts": {}, "size": 0})
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
                and 0 < n <= MAX_PARTS and 0 <= i <= n):
            raise DomainError("INVALID_ARGUMENT", "Malformed snapshot part.")
        if logical not in self.snapshots and len(self.snapshots) >= MAX_PENDING_CHUNKS:
            raise DomainError("RATE_LIMITED", "Too many partial snapshots.", 429)
        entry = self.snapshots.setdefault(logical, {"n": n, "parts": {}, "first": frame})
        surfaces = frame.get("surfaces") or []
        if not isinstance(surfaces, list) or entry["n"] != n:
            raise DomainError("INVALID_ARGUMENT", "Malformed snapshot part.")
        entry["parts"][i] = surfaces
        if sum(len(p) for p in entry["parts"].values()) > 1000:
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
            return result
        if kind == "ack":
            body = Ack.model_validate(frame)
            await self.effect(lambda db, bot, host, f: acked(db, host, f), body)
            return None
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
            outcome = await self.dispatch(frame)
        except ValidationError as exc:
            errors = [{"loc": [str(p) for p in e["loc"]], "msg": e["msg"]} for e in exc.errors()[:5]]
            return {"acked_id": identity, **echo, "ok": False,
                    "error": {"code": "INVALID_ARGUMENT", "message": "Invalid frame.", "data": errors}}
        except DomainError as exc:
            if exc.status == 401 or exc.code in {"ACCESS_REVOKED", "INSUFFICIENT_SCOPE", "READ_ONLY", "REMOTEA_DISABLED"}:
                raise refused_access(exc) from None
            retry = {"retry": True} if exc.code in RETRYABLE else {}
            return {"acked_id": identity, **echo, "ok": False, **retry, "error": {"code": exc.code, "message": exc.message}}
        if frame.get("t") == "ack":
            return None
        return {"acked_id": identity, **echo, "ok": True, **({"outcome": outcome} if outcome is not None else {})}

    async def tick(self):
        def run():
            with session_scope() as db:
                holding(db, self.credential_id, self.host_id, self.epoch)
                return drain(db, db.get(AgentHost, self.host_id), self.epoch)
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
        previous = LINKS.get(link.host_id)
        LINKS[link.host_id] = link
        if previous is not None:
            await previous.close(4001, "superseded", {"reason": "superseded"})
        await link.send("welcome", acked_id=str(frame.get("id", ""))[:100], host_id=link.host_id, conn_epoch=link.epoch,
                        server_time=now(), replay_horizon=REPLAY_HORIZON_DAYS * 86400, lease_seconds=int(LEASE.total_seconds()),
                        proto=PROTO, caps=SERVER_CAPS, max_frame=MAX_FRAME)
        receiving = None
        while not link.closed:
            receiving = receiving or asyncio.ensure_future(websocket.receive())
            done, _ = await asyncio.wait({receiving}, timeout=TICK_SECONDS)
            if not done:
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
        if receiving is not None:
            receiving.cancel()
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
