"""Browser routes for the remotea page and its Settings toggle (off by default).

Enable, disable and status work in either state; every other /api/v1/remotea route answers
403 REMOTEA_DISABLED until the owner turns remotea on.
"""

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import or_, select
from sqlalchemy.dialects.postgresql import distinct_on

from . import agent_control as control
from . import agent_link, agent_plan, bot_access
from . import run_coordinator as coordinator
from . import agent_sessions as sessions
from .auth import Identity, authenticate
from .config import get_settings
from .db import session_scope
from .domain import DomainError, advisory, emit, encode, owned
from .models import BotCredential, OwnerSettings, now
from .session_models import AgentHost, AgentSession, AgentTurn

router = APIRouter(prefix="/api/v1/remotea")
User = Annotated[Identity, Depends(authenticate)]
KEY_NAME = "remotea"
KEY_SCOPES = ["agents:control", "agents:session", "sessions:read", "sessions:write", "tasks:read", "records:read"]
KEY_DAYS = 365
RECENT_EXIT_HOURS = 24


class EnableInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    rotate: bool = False


class AutoCompleteInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    enabled: bool


def link_url():
    origin = get_settings().origin.rstrip("/")
    return origin.replace("https://", "wss://", 1).replace("http://", "ws://", 1) + agent_link.PATH


def active_key(db, user):
    return db.scalar(
        select(BotCredential)
        .where(BotCredential.owner_id == user.owner_id, BotCredential.account_id == user.account_id,
               BotCredential.name == KEY_NAME, BotCredential.revoked_at.is_(None), BotCredential.expires_at > now())
        .order_by(BotCredential.created_at.desc())
        .limit(1)
    )


def state(db, user):
    hosts = db.scalars(select(AgentHost).where(AgentHost.owner_id == user.owner_id).order_by(AgentHost.created_at))
    key = active_key(db, user)
    return {
        "enabled": agent_link.enabled(db, user.owner_id),
        "auto_complete": coordinator.auto_complete(db, user.owner_id),
        "quiet_bypass": coordinator.quiet_bypass(db, user.owner_id),
        "key": bot_access.public(key) if key else None,
        "hosts": [agent_link.public_host(h) for h in hosts],
        "link_url": link_url(),
        "connect_command": "remotea eridani connect --url " + link_url(),
    }


def set_enabled(db, owner, value, key="remotea_enabled"):
    advisory(db, f"budget:{owner}")  # The same lock settings.update takes for owner preferences.
    row = db.get(OwnerSettings, owner)
    if row:
        row.values = {**row.values, key: value}
    else:
        db.add(OwnerSettings(owner_id=owner, values={key: value}))
    db.flush()
    emit(db, owner, "settings.changed", owner)


@router.get("/status")
def status(user: User):
    with session_scope() as db:
        owner_only(db, user)
        return state(db, user)


@router.post("/enable")
def enable(body: EnableInput, user: User):
    """Turn remotea on and mint the link key. The token appears in this response only."""
    with session_scope() as db:
        owner_only(db, user)
        key = active_key(db, user)
        minted = None
        if key is None or body.rotate:
            minted = bot_access.create(db, user.owner_id, user.account_id, KEY_NAME, KEY_SCOPES, KEY_DAYS)
            if key is not None:
                advisory(db, "bot:" + key.id)
                key.revoked_at = now()
        set_enabled(db, user.owner_id, True)
        return {**state(db, user), "token": minted["token"] if minted else None}


@router.post("/disable")
def disable(user: User):
    """Stop admitting the link and revoke its key; connected hosts are closed on their next re-authorization."""
    with session_scope() as db:
        owner_only(db, user)
        set_enabled(db, user.owner_id, False)
        # The link key dies with the switch (stage F, L1): turning remotea on again mints a new one.
        key = active_key(db, user)
        if key is not None:
            advisory(db, "bot:" + key.id)
            key.revoked_at = now()
        return state(db, user)


@router.post("/auto-complete")
def set_auto_complete(body: AutoCompleteInput, user: User):
    """Rollout flag (decision 6, default off): a met verdict completes the task without a tap."""
    with session_scope() as db:
        gated(db, user)
        set_enabled(db, user.owner_id, body.enabled, "remotea_auto_complete")
        return state(db, user)


@router.post("/quiet-bypass")
def set_quiet_bypass(body: AutoCompleteInput, user: User):
    """Stage F, L8 (default off): an agent's question is urgent, so it pushes through quiet hours."""
    with session_scope() as db:
        gated(db, user)
        set_enabled(db, user.owner_id, body.enabled, "agent_quiet_bypass")
        return state(db, user)


def gated(db, user):
    owner_only(db, user)
    agent_link.gate(db, user.owner_id)


def owner_only(db, user):
    """remotea is personal-workspace only (stage F, D1): PERSONAL_PATHS refuses shared namespaces before
    this runs; the role check is the backstop so no member role ever drives an agent."""
    from .access import role

    if bot_access.shared(db, user.owner_id) or role(db, user.owner_id, user.account_id, lock=True) != "owner":
        raise DomainError("PERSONAL_WORKSPACE", "remotea works only in your personal workspace.", 403)


def last_turns(db, ids):
    if not ids:
        return {}
    latest = {}
    for row in db.scalars(  # The agent's last response; a typed prompt (role "user") is not a response.
        select(AgentTurn).where(AgentTurn.session_pk.in_(ids), AgentTurn.role == "assistant")
        .order_by(AgentTurn.session_pk, AgentTurn.seq.desc(), AgentTurn.created_at.desc())
        .ext(distinct_on(AgentTurn.session_pk))
    ):
        latest[row.session_pk] = row
    return latest


def public_turn(row, limit=None):
    text = row.text if limit is None else row.text[:limit]
    return encode({"id": row.id, "seq": row.seq, "role": row.role, "text": text, "length": len(row.text),
                   "clipped": len(text) < len(row.text), "truncated": row.truncated, "generation": row.generation,
                   "source": row.source, "at": row.created_at})


@router.get("/sessions")
def live_sessions(user: User, limit: int = Query(50, ge=1, le=100)):
    """Sessions the host reports, newest state first: live ones plus those that ended in the last day."""
    from datetime import timedelta

    with session_scope() as db:
        gated(db, user)
        rows = list(db.scalars(
            select(AgentSession)
            .where(AgentSession.owner_id == user.owner_id, AgentSession.archived.is_(False), AgentSession.state != "",
                   or_(AgentSession.state != "exited", AgentSession.state_since >= now() - timedelta(hours=RECENT_EXIT_HOURS)))
            .order_by(AgentSession.state_since.desc().nulls_last())
            .limit(limit)
        ))
        hosts = {h.id: h for h in db.scalars(select(AgentHost).where(AgentHost.owner_id == user.owner_id))}
        turns = last_turns(db, [r.id for r in rows])
        titles = sessions.record_titles(db, user.owner_id, {r.record_id for r in rows})
        items = []
        for row in rows:
            host = hosts.get(row.host_id)
            items.append({
                **sessions.public(db, row, titles=titles),
                "host": {"id": host.id, "name": host.name, "connected": agent_link.connected(host)} if host else None,
                "last_turn": public_turn(turns[row.id], 2000) if row.id in turns else None,
            })
        return encode({"items": items, "server_time": now()})


@router.get("/sessions/{identity}/turns")
def session_turns(identity: UUID, user: User, limit: int = Query(20, ge=1, le=100)):
    with session_scope() as db:
        gated(db, user)
        row = owned(db, AgentSession, str(identity), user.owner_id)
        turns = db.scalars(select(AgentTurn).where(AgentTurn.session_pk == row.id)
                           .order_by(AgentTurn.seq.desc(), AgentTurn.created_at.desc()).limit(limit))
        return {"items": [public_turn(t) for t in turns]}


@router.get("/parity")
def parity(user: User):
    with session_scope() as db:
        gated(db, user)
        return agent_link.parity_summary(db, user.owner_id)


# ---------------------------------------------------------------- runs (stage C)


def actor_for(user):
    from .agent_actor import Actor

    return Actor("human")


def writable(db, user):
    gated(db, user)


@router.get("/runs")
def runs(user: User, task_id: UUID | None = None, active: bool = False, limit: int = Query(50, ge=1, le=100)):
    # Inspection and stop stay available after remotea is turned off, for runs that have not settled.
    with session_scope() as db:
        owner_only(db, user)
        return {"items": control.list_runs(db, user.owner_id, task_id=str(task_id) if task_id else None,
                                           active=active, limit=limit)}


@router.post("/runs")
def start(body: control.StartInput, user: User):
    with session_scope() as db:
        writable(db, user)
        return control.start_run(db, user.owner_id, user.account_id, body, actor=actor_for(user))


@router.get("/runs/{identity}")
def run_detail(identity: UUID, user: User, limit: int = Query(control.DETAIL_PAGE, ge=1, le=200),
               offset: int = Query(0, ge=0, le=100_000)):
    with session_scope() as db:
        # Like the list, detail stays readable after remotea is turned off; only the owner reads it.
        owner_only(db, user)
        run = coordinator.load(db, user.owner_id, str(identity), lock=False)
        return control.public_run(db, run, detail=True, limit=limit, offset=offset)


@router.post("/runs/{identity}/message")
def message(identity: UUID, body: control.MessageInput, user: User):
    with session_scope() as db:
        writable(db, user)
        return control.send_message(db, user.owner_id, str(identity), body, actor=actor_for(user))


@router.post("/runs/{identity}/stop")
def stop(identity: UUID, body: control.StopInput, user: User):
    # Stop stays available while remotea is being turned off: recovery controls outlive admission.
    with session_scope() as db:
        owner_only(db, user)
        return control.stop_run(db, user.owner_id, str(identity), body, actor=actor_for(user))


@router.post("/runs/{identity}/approve")
def approve(identity: UUID, body: control.ApproveInput, user: User):
    with session_scope() as db:
        writable(db, user)
        return control.approve(db, user.owner_id, str(identity), body, actor=actor_for(user))


@router.post("/runs/{identity}/replan")
def replan(identity: UUID, body: control.ReplanInput, user: User):
    with session_scope() as db:
        writable(db, user)
        return control.replan(db, user.owner_id, str(identity), body, actor=actor_for(user))


@router.post("/runs/{identity}/verify")
def verify(identity: UUID, user: User):
    with session_scope() as db:
        writable(db, user)
        return control.verify_now(db, user.owner_id, str(identity), actor=actor_for(user))


@router.post("/runs/{identity}/plan-review")
def plan_review(identity: UUID, body: agent_plan.PlanReviewInput, user: User):
    with session_scope() as db:
        writable(db, user)
        return agent_plan.review_plan(db, user.owner_id, str(identity), body, actor=actor_for(user))


@router.post("/interviews/{identity}/answers")
def interview_answers(identity: UUID, body: agent_plan.InterviewAnswers, user: User):
    with session_scope() as db:
        writable(db, user)
        return agent_plan.answer_interview(db, user.owner_id, str(identity), body, actor=actor_for(user))


@router.post("/questions/{identity}/answer")
def answer(identity: UUID, body: control.AnswerInput, user: User):
    with session_scope() as db:
        writable(db, user)
        return control.answer_question(db, user.owner_id, str(identity), body, actor=actor_for(user))


@router.get("/targets")
def targets(user: User):
    from .session_models import ExecutionTarget

    with session_scope() as db:
        gated(db, user)
        rows = db.scalars(select(ExecutionTarget).where(ExecutionTarget.owner_id == user.owner_id)
                          .order_by(ExecutionTarget.created_at))
        return {"items": [control.public_target(t) for t in rows], "advertised": control.advertised(db, user.owner_id)}


@router.post("/targets")
def save_target(body: control.TargetInput, user: User):
    with session_scope() as db:
        writable(db, user)
        return control.save_target(db, user.owner_id, body)


class MigrationSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")
    host_id: str = Field(max_length=36)
    host_target: str = Field(pattern=r"^[A-Za-z0-9._-]{1,64}$")
    # The folder and file pattern for this repository's migrations; empty uses the host's config (eridani.json).
    dir: str | None = Field(default=None, max_length=200)
    pattern: str | None = Field(default=None, max_length=200)
    # The operator's chain head; the higher of it and the host's report wins. null clears it.
    override: int | None = Field(default=None, ge=0, le=99_999)


@router.get("/migrations")
def migrations(user: User):
    from . import migration_numbers

    with session_scope() as db:
        gated(db, user)
        return {"items": migration_numbers.listing(db, user.owner_id)}


@router.post("/migrations")
def set_migration_settings(body: MigrationSettings, user: User):
    from . import migration_numbers

    with session_scope() as db:
        writable(db, user)
        return migration_numbers.configure(db, user.owner_id, body.host_id, body.host_target, dir=body.dir,
                                           pattern=body.pattern, override=body.override)


@router.get("/unmerged")
def unmerged(user: User):
    with session_scope() as db:
        gated(db, user)
        return control.unmerged(db, user.owner_id)


# ---------------------------------------------------------------- Eri's pending control (stage F, M5)


@router.get("/pending")
def pending(user: User):
    """control_start / control_message Eri proposed, waiting for the owner's tap."""
    with session_scope() as db:
        gated(db, user)
        return {"items": control.pending_list(db, user.owner_id)}


class ConfirmInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    # The card's target picker: where a start that answered NEEDS_TARGET should run (owner-checked).
    execution_target_id: str | None = Field(default=None, max_length=36)


@router.post("/pending/{identity}/confirm")
def confirm_pending(identity: UUID, user: User, body: ConfirmInput | None = None):
    with session_scope() as db:
        writable(db, user)
        return control.decide(db, user.owner_id, str(identity), confirm=True, via="app", actor=actor_for(user),
                              execution_target_id=body.execution_target_id if body else None)


@router.post("/pending/{identity}/decline")
def decline_pending(identity: UUID, user: User):
    with session_scope() as db:
        writable(db, user)
        return control.decide(db, user.owner_id, str(identity), confirm=False, via="app", actor=actor_for(user))
