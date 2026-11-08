"""Browser routes for the remotea page and its Settings toggle (off by default).

Enable, disable and status work in either state; every other /api/v1/remotea route answers
403 REMOTEA_DISABLED until the owner turns remotea on.
"""

import base64
import json
import logging
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, Path, Query
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import or_, select
from sqlalchemy.dialects.postgresql import distinct_on

from . import agent_control as control
from . import agent_link, agent_plan, bot_access, run_merge, swarm
from . import agent_receipts as receipts
from . import agent_retention as retention
from . import agent_sessions as sessions
from . import run_coordinator as coordinator
from .auth import Identity, authenticate
from .config import get_settings
from .db import session_scope
from .domain import DomainError, advisory, emit, encode, owned
from .models import BotCredential, OwnerSettings, now
from .session_models import AgentHost, AgentRun, AgentSession, AgentTurn

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/remotea")
User = Annotated[Identity, Depends(authenticate)]
KEY_NAME = "remotea"
KEY_SCOPES = ["agents:control", "agents:session", "sessions:read", "sessions:write", "tasks:read", "records:read"]
KEY_DAYS = 365
# The bot key a setup code carries for the coding agents a run spawns (remotea saves it as eridani-mcp-token and
# hands it to each spawn). Its name marks it: a new setup code revokes every active key with this name. Exactly
# what remotea's eridani-mcp.ts tools need: task_plan, task_step, task_interview and submit_plan (agents:session;
# task_step and submit_plan edit the task and its PRD note: tasks:write, notes:write) and session_link
# (sessions:write). bot_access.normalize adds the matching :read scopes. Never agents:control: it cannot open the
# link, and the link key never gets tasks:write.
MCP_KEY_NAME = "remotea MCP"
MCP_KEY_SCOPES = ["agents:session", "tasks:write", "notes:write", "sessions:write"]
SETUP_CODE_PREFIX = "eridani-setup:1:"
SETUP_COMMAND = "remotea eridani setup"
RECENT_EXIT_HOURS = 24


class EnableInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    rotate: bool = False


class SetupCodeInput(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    # Optional: the machine name remotea reports as (the wizard asks for it otherwise).
    machine: str | None = Field(default=None, max_length=100, pattern=r"^[^\x00-\x1f]*$")


class AutoCompleteInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    enabled: bool


def link_url():
    origin = get_settings().origin.rstrip("/")
    return origin.replace("https://", "wss://", 1).replace("http://", "ws://", 1) + agent_link.PATH


def active_keys(db, user, name=KEY_NAME):
    return list(db.scalars(
        select(BotCredential)
        .where(BotCredential.owner_id == user.owner_id, BotCredential.account_id == user.account_id,
               BotCredential.name == name, BotCredential.revoked_at.is_(None), BotCredential.expires_at > now())
        .order_by(BotCredential.created_at.desc())
    ))


def active_key(db, user, name=KEY_NAME):
    keys = active_keys(db, user, name)
    return keys[0] if keys else None


def revoke(db, rows):
    for row in rows:
        advisory(db, "bot:" + row.id)
        row.revoked_at = row.revoked_at or now()
    return [row.id for row in rows]


def setup_code(link_key, mcp_key=None, machine=None):
    """`eridani-setup:1:` + unpadded base64url(JSON {url, link_key, mcp_key?, machine?}), what remotea's
    `remotea eridani setup` accepts (remotea eridani-setup.ts parseSetupCode). `url` is Eridani's public origin,
    which the wizard turns into the link URL and the MCP URL. A code without mcp_key keeps the host's current one."""
    payload = {"url": get_settings().origin.rstrip("/"), "link_key": link_key,
               **({"mcp_key": mcp_key} if mcp_key else {}), **({"machine": machine} if machine else {})}
    raw = json.dumps(payload, separators=(",", ":")).encode()
    return SETUP_CODE_PREFIX + base64.urlsafe_b64encode(raw).rstrip(b"=").decode()


def mint(db, user, *, mcp=True, machine=None):
    """Mint a new link key and, with `mcp`, a new MCP key, revoking the ones they replace; return the link token
    (`token`, as before) and the setup code carrying both. Both appear in this response only: bot_credentials
    keeps their hash, and the log line names credential ids, never a token or a code."""
    tokens, revoked = {}, []
    for kind, name, scopes in (("link", KEY_NAME, KEY_SCOPES), ("mcp", MCP_KEY_NAME, MCP_KEY_SCOPES)):
        if kind == "mcp" and not mcp:
            continue
        old = active_keys(db, user, name)
        tokens[kind] = bot_access.create(db, user.owner_id, user.account_id, name, scopes, KEY_DAYS)
        revoked += revoke(db, old)
    logger.info("remotea keys minted for owner %s: %s; revoked: %s", user.owner_id,
                ", ".join(f"{kind} {row['credential']['id']}" for kind, row in tokens.items()),
                ", ".join(revoked) or "none")
    return {"token": tokens["link"]["token"],
            "setup_code": setup_code(tokens["link"]["token"], tokens["mcp"]["token"] if mcp else None, machine)}


def state(db, user):
    hosts = db.scalars(select(AgentHost).where(AgentHost.owner_id == user.owner_id).order_by(AgentHost.created_at))
    key = active_key(db, user)
    return {
        "enabled": agent_link.enabled(db, user.owner_id),
        "auto_complete": coordinator.auto_complete(db, user.owner_id),
        "auto_merge": run_merge.auto_default(db, user.owner_id),
        "quiet_bypass": coordinator.quiet_bypass(db, user.owner_id),
        "retention_days": retention.window(db, user.owner_id).days,
        "key": bot_access.public(key) if key else None,
        # The MCP key the last setup code carried (its prefix and expiry; never the token).
        "mcp_key": bot_access.public(mcp) if (mcp := active_key(db, user, MCP_KEY_NAME)) else None,
        "setup_command": SETUP_COMMAND,
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
    """Turn remotea on. The first time (no active link key) it mints the link key and the MCP key and returns both
    in a setup code; `rotate` (New link key) replaces only the link key, and its code carries no MCP key, so the
    host keeps the one it has. Enabling again with a live key mints nothing. Tokens and the code appear in this
    response only."""
    with session_scope() as db:
        owner_only(db, user)
        fresh = active_key(db, user) is None
        minted = mint(db, user, mcp=fresh) if fresh or body.rotate else {"token": None, "setup_code": None}
        set_enabled(db, user.owner_id, True)
        return {**state(db, user), **minted}


@router.post("/setup-code")
def new_setup_code(user: User, body: SetupCodeInput | None = None):
    """New setup code: replace the link key and the MCP key (revoking the earlier ones) and return the one-time code
    for `remotea eridani setup`."""
    with session_scope() as db:
        gated(db, user)
        return {**state(db, user), **mint(db, user, machine=(body.machine if body else None) or None)}


@router.get("/setup")
def setup(user: User):
    """The guided setup page: hosts, then targets with their readiness and fixes (agent_control.setup_report)."""
    with session_scope() as db:
        gated(db, user)
        return encode(control.setup_report(db, user.owner_id))


@router.post("/targets/{key}/probe")
def probe_target(user: User, key: Annotated[str, Path(min_length=1, max_length=200)]):
    """One target's readiness facts and fixes; read from the host's last hello while remotea has no probe command
    (the response says so). `key` is a saved target id or an advertised `host:<host_id>:<host_target>`."""
    with session_scope() as db:
        gated(db, user)
        return encode(control.target_probe(db, user.owner_id, key))


@router.post("/disable")
def disable(user: User):
    """Stop admitting the link and revoke its key; connected hosts are closed on their next re-authorization."""
    with session_scope() as db:
        owner_only(db, user)
        set_enabled(db, user.owner_id, False)
        # The link key dies with the switch (stage F, L1), and so does the MCP key a setup code carried: turning
        # remotea on again mints both in a new setup code.
        revoked = revoke(db, [*active_keys(db, user), *active_keys(db, user, MCP_KEY_NAME)])
        if revoked:
            logger.info("remotea keys revoked for owner %s: %s", user.owner_id, ", ".join(revoked))
        return state(db, user)


@router.post("/auto-complete")
def set_auto_complete(body: AutoCompleteInput, user: User):
    """Rollout flag (decision 6, default off): a met verdict completes the task without a tap."""
    with session_scope() as db:
        gated(db, user)
        set_enabled(db, user.owner_id, body.enabled, "remotea_auto_complete")
        return state(db, user)


@router.post("/auto-merge")
def set_auto_merge(body: AutoCompleteInput, user: User):
    """Default for new runs (off): a run with a real check (a verify entry or CI) merges its pull request by itself."""
    with session_scope() as db:
        gated(db, user)
        set_enabled(db, user.owner_id, body.enabled, "remotea_auto_merge")
        return state(db, user)


@router.post("/quiet-bypass")
def set_quiet_bypass(body: AutoCompleteInput, user: User):
    """Stage F, L8 (default off): an agent's question is urgent, so it pushes through quiet hours."""
    with session_scope() as db:
        gated(db, user)
        set_enabled(db, user.owner_id, body.enabled, "agent_quiet_bypass")
        return state(db, user)


class RetentionInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    days: int = Field(ge=retention.MIN_DAYS, le=retention.MAX_DAYS)


@router.post("/retention")
def set_retention(body: RetentionInput, user: User):
    """How long a settled run's turns, commands, attempts, questions and interviews are kept before they are
    deleted and the run's remaining copies of content are redacted (default 90 days). Works while remotea is
    off, so the owner can shorten it after turning the feature off."""
    with session_scope() as db:
        owner_only(db, user)
        set_enabled(db, user.owner_id, body.days, retention.SETTING)
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
        runs = session_runs(db, rows)
        items = []
        for row in rows:
            host = hosts.get(row.host_id)
            run = runs.get(row.id)
            items.append({
                **sessions.public(db, row, titles=titles),
                "host": {"id": host.id, "name": host.name, "connected": agent_link.connected(host)} if host else None,
                "last_turn": public_turn(turns[row.id], 2000) if row.id in turns else None,
                **session_origin(row, run),
            })
        return encode({"items": items, "server_time": now()})


def session_runs(db, rows):
    """{session pk: the run that owns it}: the session's own run_id, else a run attached to it (session_pk)."""
    ids = [r.id for r in rows]
    if not ids:
        return {}
    run_ids = {r.run_id for r in rows if r.run_id}
    found = list(db.scalars(select(AgentRun).where(or_(AgentRun.id.in_(run_ids or {""}), AgentRun.session_pk.in_(ids)))))
    by_id = {r.id: r for r in found}
    out = {r.session_pk: r for r in found if r.session_pk}
    for row in rows:
        if row.run_id in by_id:
            out[row.id] = by_id[row.run_id]
    return out


def session_origin(row, run):
    """Where a session came from: "run" (an Eridani run started it), "remotea" (opened in remotea by hand) or "other"
    (another harness's chat the collector found, never in a remotea tab)."""
    if run is not None:
        task = run.task_id
        return {"origin": "run", "run": {"id": run.id, "state": run.state, "origin": run.origin or "task",
                                         "task_id": task, "dismissed": run.dismissed_at is not None}}
    return {"origin": "remotea" if row.surf or row.state else "other", "run": None}


@router.get("/other-chats")
def other_chats(user: User, days: int = Query(14, ge=1, le=90), limit: int = Query(30, ge=1, le=100)):
    """Chats from other harnesses (never in a remotea tab), most recently active first, within `days`."""
    from datetime import timedelta

    with session_scope() as db:
        gated(db, user)
        rows = list(db.scalars(
            select(AgentSession)
            .where(AgentSession.owner_id == user.owner_id, AgentSession.archived.is_(False),
                   AgentSession.surf == "", AgentSession.state == "", AgentSession.run_id.is_(None),
                   AgentSession.last_active_at >= now() - timedelta(days=days))
            .order_by(AgentSession.last_active_at.desc().nulls_last())
            .limit(limit)
        ))
        titles = sessions.record_titles(db, user.owner_id, {r.record_id for r in rows})
        return encode({"items": [{**sessions.public(db, r, titles=titles), "origin": "other"} for r in rows]})


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
def runs(user: User, task_id: UUID | None = None, active: bool = False, limit: int = Query(50, ge=1, le=100),
         dismissed: bool = True):
    # Inspection and stop stay available after remotea is turned off, for runs that have not settled.
    # dismissed=false (the remotea page) leaves out runs the owner dismissed there.
    with session_scope() as db:
        owner_only(db, user)
        return {"items": control.list_runs(db, user.owner_id, task_id=str(task_id) if task_id else None,
                                           active=active, limit=limit, dismissed=dismissed)}


@router.post("/sessions/new")
def new_session(body: control.SessionInput, user: User):
    with session_scope() as db:
        writable(db, user)
        return receipts.once(db, user.owner_id, body.execution_target_id, "session", body, lambda: control.start_session(
            db, user.owner_id, user.account_id, body, actor=actor_for(user)))


@router.post("/runs/{identity}/dismiss")
def dismiss(identity: UUID, body: control.DismissInput, user: User):
    with session_scope() as db:
        owner_only(db, user)
        return receipts.once(db, user.owner_id, str(identity), "dismiss", body, lambda: control.dismiss_run(
            db, user.owner_id, str(identity), body, actor=actor_for(user)))


@router.post("/runs")
def start(body: control.StartInput, user: User):
    with session_scope() as db:
        writable(db, user)
        return receipts.once(db, user.owner_id, body.task_id, "start", body, lambda: control.start_run(
            db, user.owner_id, user.account_id, body, actor=actor_for(user)))


@router.get("/runs/preflight")
def preflight(task_id: UUID, user: User, harness: str = Query("claude", max_length=20), execution_target_id: UUID | None = None,
              reviewer_harness: str | None = Query(None, max_length=20), worker_harness: str | None = Query(None, max_length=20)):
    """What Run would do on this task: where it runs (or the picker's choices), the run already on it, whether the
    agent interviews first, which harnesses the server can start, and (`roles`) whether the host can spawn hyperplan
    roles; `reviewer_harness` / `worker_harness` add their availability problems. Read-only; the Run dialog shows it."""
    with session_scope() as db:
        gated(db, user)
        return control.preflight(db, user.owner_id, str(task_id), harness, str(execution_target_id) if execution_target_id else None,
                                 reviewer_harness=reviewer_harness, worker_harness=worker_harness)


@router.get("/runs/{identity}")
def run_detail(identity: UUID, user: User, limit: int = Query(control.DETAIL_PAGE, ge=1, le=200),
               offset: int = Query(0, ge=0, le=100_000)):
    with session_scope() as db:
        # Like the list, detail stays readable after remotea is turned off; only the owner reads it.
        owner_only(db, user)
        run = coordinator.load(db, user.owner_id, str(identity), lock=False)
        return control.public_run(db, run, detail=True, limit=limit, offset=offset)


@router.get("/runs/{identity}/turns/{turn_id}")
def run_turn(identity: UUID, turn_id: UUID, user: User):
    """One turn's whole text: detail clips each turn at agent_control.TURN_CLIP (`clipped`), and the drawer's
    "Full response" fetches the rest here. Gated like detail (owner only, readable after remotea is turned off)."""
    from . import agent_retention as retention

    with session_scope() as db:
        owner_only(db, user)
        run = coordinator.load(db, user.owner_id, str(identity), lock=False)
        turn = db.scalar(control.run_turns(run).where(AgentTurn.id == str(turn_id)))
        if turn is None:
            if retention.run_data_expired(run):
                raise DomainError("TURN_EXPIRED", "This response was deleted: a finished run's turns are kept "
                                  f"{retention.RUN_DATA.days} days.", 404)
            raise DomainError("NOT_FOUND", "That response is not part of this run.", 404)
        return {"id": turn.id, "seq": turn.seq, "text": turn.text, "truncated": turn.truncated}


@router.post("/runs/{identity}/message")
def message(identity: UUID, body: control.MessageInput, user: User):
    with session_scope() as db:
        writable(db, user)
        return receipts.once(db, user.owner_id, str(identity), "message", body, lambda: control.send_message(
            db, user.owner_id, str(identity), body, actor=actor_for(user)))


@router.post("/runs/{identity}/stop")
def stop(identity: UUID, body: control.StopInput, user: User):
    # Stop stays available while remotea is being turned off: recovery controls outlive admission.
    with session_scope() as db:
        owner_only(db, user)
        return receipts.once(db, user.owner_id, str(identity), "stop", body, lambda: control.stop_run(
            db, user.owner_id, str(identity), body, actor=actor_for(user)))


@router.post("/runs/{identity}/focus")
def focus(identity: UUID, user: User):
    """Open the run's terminal in remotea: ask the host's desktop app(s) to bring the run's tab to the front
    (`terminal.focus`). Waits a few seconds for the host's answer. `{sent: true}` when an app was told, otherwise
    `{sent: false, reason}`: `no_desktop` (no desktop app connected to the host), `host_offline`, `no_surface` (the
    run has no terminal, or it is gone), `host_too_old` (the host does not offer it), `no_answer` (the host did
    not answer in time) or `failed`. The page then shows where the terminal is. Changes nothing on the run and
    stores no receipt: a repeat only raises the tab again."""
    import time

    with session_scope() as db:
        writable(db, user)
        op_id, reason = control.focus_terminal(db, user.owner_id, str(identity))
    if op_id is None:
        return {"sent": False, "reason": reason}
    deadline = time.monotonic() + control.FOCUS_WAIT_SECONDS
    while True:
        with session_scope() as db:
            result = control.focus_result(db, user.owner_id, op_id)
        if result is not None:
            return result
        if time.monotonic() >= deadline:
            with session_scope() as db:
                control.give_up_focus(db, user.owner_id, op_id)
                result = control.focus_result(db, user.owner_id, op_id)
            return result if result is not None and result.get("sent") else {"sent": False, "reason": "no_answer"}
        time.sleep(control.FOCUS_POLL_SECONDS)


@router.post("/runs/{identity}/approve")
def approve(identity: UUID, body: control.ApproveInput, user: User):
    with session_scope() as db:
        writable(db, user)
        return receipts.once(db, user.owner_id, str(identity), "approve", body, lambda: control.approve(
            db, user.owner_id, str(identity), body, actor=actor_for(user)))


@router.post("/runs/{identity}/merge")
def merge(identity: UUID, body: control.MergeInput, user: User):
    with session_scope() as db:
        writable(db, user)
        return receipts.once(db, user.owner_id, str(identity), "merge", body, lambda: run_merge.merge(
            db, user.owner_id, str(identity), body, actor=actor_for(user)))


@router.post("/runs/{identity}/update-branch")
def update_branch(identity: UUID, body: control.MergeInput, user: User):
    with session_scope() as db:
        writable(db, user)
        return receipts.once(db, user.owner_id, str(identity), "update_branch", body, lambda: run_merge.update_branch(
            db, user.owner_id, str(identity), body, actor=actor_for(user)))


@router.post("/runs/{identity}/auto-merge")
def run_auto_merge(identity: UUID, body: control.AutoMergeInput, user: User):
    with session_scope() as db:
        writable(db, user)
        return run_merge.set_auto(db, user.owner_id, str(identity), body, actor=actor_for(user))


@router.post("/runs/{identity}/ci/refresh")
def ci_refresh(identity: UUID, user: User):
    with session_scope() as db:
        writable(db, user)
        return run_merge.refresh(db, user.owner_id, str(identity))


@router.post("/runs/{identity}/replan")
def replan(identity: UUID, body: control.ReplanInput, user: User):
    with session_scope() as db:
        writable(db, user)
        return receipts.once(db, user.owner_id, str(identity), "replan", body, lambda: control.replan(
            db, user.owner_id, str(identity), body, actor=actor_for(user)))


@router.post("/runs/{identity}/verify")
def verify(identity: UUID, user: User, body: control.VerifyNowInput | None = None):
    with session_scope() as db:
        writable(db, user)
        return control.verify_now(db, user.owner_id, str(identity), body, actor=actor_for(user))


@router.post("/runs/{identity}/verify-config")
def verify_config(identity: UUID, body: agent_plan.SetVerifyInput, user: User):
    """Set the plan's verify command mid-run (a new plan revision; the next turn is verified)."""
    with session_scope() as db:
        writable(db, user)
        return receipts.once(db, user.owner_id, str(identity), "verify_config", body, lambda: agent_plan.set_verify(
            db, user.owner_id, str(identity), body, actor=actor_for(user)))


@router.post("/runs/{identity}/criteria")
def criteria(identity: UUID, body: agent_plan.SetCriteriaInput, user: User):
    """Edit criteria on a parked run (needs_you, awaiting_approval): a human edit frozen as the next plan revision."""
    with session_scope() as db:
        writable(db, user)
        return receipts.once(db, user.owner_id, str(identity), "criteria", body, lambda: agent_plan.set_criteria(
            db, user.owner_id, str(identity), body, actor=actor_for(user)))


@router.post("/runs/{identity}/judge")
def judge_again(identity: UUID, body: control.JudgeAgainInput, user: User):
    """Judge again: verify and judge the last turn against the plan as frozen now."""
    with session_scope() as db:
        writable(db, user)
        return receipts.once(db, user.owner_id, str(identity), "judge", body, lambda: control.judge_again(
            db, user.owner_id, str(identity), body, actor=actor_for(user)))


@router.post("/runs/{identity}/freestyle")
def freestyle(identity: UUID, body: control.FreestyleInput, user: User):
    """Talk to the agent directly: turns are recorded but not verified, judged or capped while this is on."""
    with session_scope() as db:
        writable(db, user)
        return receipts.once(db, user.owner_id, str(identity), "freestyle", body, lambda: control.set_freestyle(
            db, user.owner_id, str(identity), body, actor=actor_for(user)))


@router.post("/runs/{identity}/plan-review")
def plan_review(identity: UUID, body: agent_plan.PlanReviewInput, user: User):
    with session_scope() as db:
        writable(db, user)
        return receipts.once(db, user.owner_id, str(identity), "plan_review", body, lambda: agent_plan.review_plan(
            db, user.owner_id, str(identity), body, actor=actor_for(user)))


@router.post("/runs/{identity}/redact")
def redact_run(identity: UUID, user: User):
    """Delete one settled run's content now (agent_retention.redact_now); the skeleton stays. Like stop, it
    works after remotea is turned off; only the owner may call it."""
    with session_scope() as db:
        owner_only(db, user)
        counts = retention.redact_now(db, user.owner_id, str(identity), actor=actor_for(user).record())
        run = coordinator.load(db, user.owner_id, str(identity), lock=False)
        return {"redacted": counts, "run": control.public_run(db, run, detail=True)}


@router.post("/interviews/{identity}/answers")
def interview_answers(identity: UUID, body: agent_plan.InterviewAnswers, user: User):
    with session_scope() as db:
        writable(db, user)
        return receipts.once(db, user.owner_id, str(identity), "interview", body, lambda: agent_plan.answer_interview(
            db, user.owner_id, str(identity), body, actor=actor_for(user)))


@router.post("/questions/{identity}/answer")
def answer(identity: UUID, body: control.AnswerInput, user: User):
    with session_scope() as db:
        writable(db, user)
        return receipts.once(db, user.owner_id, str(identity), "answer", body, lambda: control.answer_question(
            db, user.owner_id, str(identity), body, actor=actor_for(user)))


@router.get("/targets")
def targets(user: User):
    from .session_models import ExecutionTarget

    with session_scope() as db:
        gated(db, user)
        rows = db.scalars(select(ExecutionTarget).where(ExecutionTarget.owner_id == user.owner_id)
                          .order_by(ExecutionTarget.created_at))
        adverts = control.advertised(db, user.owner_id)
        index = control.advert_index(adverts)
        return {"items": [control.public_target(t, index) for t in rows], "advertised": adverts}


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


@router.get("/worktrees")
def worktrees(user: User):
    """Worktree maintenance: every managed worktree per host, its bound run, and cleanup commands for the rows that
    lose nothing (agent_control.worktree_maintenance). Read-only: Eridani never removes a worktree."""
    with session_scope() as db:
        gated(db, user)
        return control.worktree_maintenance(db, user.owner_id)


@router.post("/hosts/{identity}/rescan")
def rescan(identity: UUID, user: User):
    """Ask one host for a fresh worktrees.report now (the `worktrees.scan` command, run-less)."""
    with session_scope() as db:
        writable(db, user)
        return control.rescan_worktrees(db, user.owner_id, str(identity))


# ---------------------------------------------------------------- Dani's pending control (stage F, M5)


@router.get("/pending")
def pending(user: User):
    """control_start / control_message Dani proposed, waiting for the owner's tap."""
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


# ---------------------------------------------------------------- swarm mode (swarm.py)


@router.get("/swarms")
def swarms(user: User, active: bool = False, limit: int = Query(20, ge=1, le=50)):
    # Readable after remotea is turned off, like runs.
    with session_scope() as db:
        owner_only(db, user)
        return {"items": swarm.list_swarms(db, user.owner_id, active=active, limit=limit)}


@router.post("/swarms")
def create_swarm(body: swarm.CreateInput, user: User):
    """Play on a project: plan its open tasks into batches. Starts nothing; an open plan for the project comes back
    as it is unless `fresh`."""
    return swarm.create(session_scope, user.owner_id, user.account_id, body, check=lambda db: writable(db, user))


@router.get("/swarms/{identity}")
def swarm_detail(identity: UUID, user: User):
    with session_scope() as db:
        owner_only(db, user)
        return swarm.public(db, swarm.load(db, user.owner_id, str(identity), lock=False))


@router.post("/swarms/{identity}/edit")
def edit_swarm(identity: UUID, body: swarm.EditInput, user: User):
    with session_scope() as db:
        writable(db, user)
        return swarm.edit(db, user.owner_id, str(identity), body)


@router.post("/swarms/{identity}/start")
def start_swarm(identity: UUID, body: swarm.StartInput, user: User):
    with session_scope() as db:
        writable(db, user)
        return swarm.start(db, user.owner_id, str(identity), body, actor=actor_for(user))


@router.post("/swarms/{identity}/cancel")
def cancel_swarm(identity: UUID, body: swarm.CancelInput, user: User):
    # Like stop, cancel stays available after remotea is turned off: it only keeps later batches from starting.
    with session_scope() as db:
        owner_only(db, user)
        return swarm.cancel(db, user.owner_id, str(identity), body)


@router.get("/actions/{action_id}")
def action_receipt(action_id: UUID, user: User):
    with session_scope() as db:
        owner_only(db, user)
        return receipts.lookup(db, user.owner_id, str(action_id))


@router.get("/decisions")
def decisions(user: User, run_id: UUID | None = None):
    from . import agent_decisions
    from .session_models import AgentRun
    with session_scope() as db:
        owner_only(db, user)
        if run_id:
            runs = [coordinator.load(db, user.owner_id, str(run_id), lock=False)]
        else:
            runs = db.scalars(select(AgentRun).where(AgentRun.owner_id == user.owner_id,
                                                     AgentRun.state.not_in(coordinator.TERMINAL)))
        result = [d for run in runs for d in agent_decisions.for_run(db, run)]
        if not run_id:
            from .session_models import AgentPendingControl, AgentQuestion
            for cls, kind, statuses in ((AgentPendingControl, "control", ("pending",)),
                                         (AgentQuestion, "question", ("open", "submitted"))):
                result.extend(agent_decisions.present(db, row, kind) for row in db.scalars(select(cls).where(
                    cls.owner_id == user.owner_id, cls.run_id.is_(None), cls.status.in_(statuses))))
        return {"decisions": result}


from .agent_decisions import PauseInput, RespondInput


@router.post("/decisions/{identity}/respond")
def respond_decision(identity: str, body: RespondInput, user: User):
    from . import agent_decisions
    with session_scope() as db:
        writable(db, user)
        return agent_decisions.respond(db, user.owner_id, identity, body, actor=actor_for(user))


@router.post("/runs/{identity}/retry-planning")
def retry_planning(identity: UUID, body: PauseInput, user: User):
    from . import agent_planning_recovery
    with session_scope() as db:
        writable(db, user)
        def apply():
            run = coordinator.load(db, user.owner_id, str(identity))
            coordinator.check_revision(run, body.expected_revision)
            agent_planning_recovery.retry(db, run, actor=actor_for(user))
            return control.public_run(db, run, detail=True)
        return receipts.once(db, user.owner_id, str(identity), "retry_planning", body, apply)


@router.post("/runs/{identity}/pause")
def pause_run(identity: UUID, body: PauseInput, user: User):
    from . import agent_decisions
    with session_scope() as db:
        writable(db, user)
        return receipts.once(db, user.owner_id, str(identity), "pause", body, lambda: agent_decisions.set_paused(
            db, user.owner_id, str(identity), body, True, actor=actor_for(user)))


@router.post("/runs/{identity}/resume")
def resume_run(identity: UUID, body: PauseInput, user: User):
    from . import agent_decisions
    with session_scope() as db:
        writable(db, user)
        return receipts.once(db, user.owner_id, str(identity), "resume", body, lambda: agent_decisions.set_paused(
            db, user.owner_id, str(identity), body, False, actor=actor_for(user)))


from .agent_automation import ResumeInput as AutomationResumeInput


@router.post("/runs/{identity}/automation/resume")
def resume_automation(identity: UUID, body: AutomationResumeInput, user: User):
    from . import agent_automation
    with session_scope() as db:
        writable(db, user)
        return receipts.once(db, user.owner_id, str(identity), "automation_resume", body,
            lambda: agent_automation.resume(db, user.owner_id, str(identity), body, actor=actor_for(user)))


@router.post("/runs/{identity}/cancel")
def cancel_run(identity: UUID, body: control.StopInput, user: User):
    with session_scope() as db:
        writable(db, user)
        return receipts.once(db, user.owner_id, str(identity), "cancel", body, lambda: control.stop_run(
            db, user.owner_id, str(identity), body, actor=actor_for(user)))


class RolePreferences(BaseModel):
    model_config = ConfigDict(extra="forbid")
    planner: control.RoleInput | None = None
    reviewer: control.RoleInput | None = None
    worker: control.RoleInput | None = None


@router.get("/role-preferences")
def role_preferences(user: User):
    with session_scope() as db:
        owner_only(db, user)
        row = db.get(OwnerSettings, user.owner_id)
        return (row.values if row else {}).get("remotea_roles", {"planner": None, "reviewer": None, "worker": None})


@router.put("/role-preferences")
def save_role_preferences(body: RolePreferences, user: User):
    with session_scope() as db:
        writable(db, user)
        value = body.model_dump(mode="json")
        set_enabled(db, user.owner_id, value, "remotea_roles")
        return value


@router.get("/model-catalog")
def model_catalog(user: User):
    with session_scope() as db:
        owner_only(db, user)
        hosts = list(db.scalars(select(AgentHost).where(AgentHost.owner_id == user.owner_id)))
        return {"models": [{**model, "host_id": host.id} for host in hosts
                           for model in (host.caps or {}).get("model_catalog", []) if isinstance(model, dict)],
                "legacy_hosts": [host.id for host in hosts if "role.options" not in ((host.caps or {}).get("caps") or [])]}


class DraftInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    value: dict | list | str | None
    expected_revision: int | None = Field(default=None, ge=0)


@router.get("/drafts/{key}")
def draft(key: str, user: User):
    with session_scope() as db:
        owner_only(db, user)
        row = db.get(OwnerSettings, user.owner_id)
        return ((row.values if row else {}).get("remotea_drafts") or {}).get(key, {"value": None, "revision": 0})


@router.put("/drafts/{key}")
def save_draft(key: str, body: DraftInput, user: User):
    from .domain import DomainError
    if len(json.dumps(body.value).encode()) > 131072:
        raise DomainError("INVALID_ARGUMENT", "Draft is too large.")
    if len(key) > 200:
        raise DomainError("INVALID_ARGUMENT", "Draft key is too long.")
    with session_scope() as db:
        writable(db, user)
        advisory(db, f"budget:{user.owner_id}")
        owner_only(db, user)
        row = db.get(OwnerSettings, user.owner_id)
        drafts = dict((row.values if row else {}).get("remotea_drafts") or {})
        revision = drafts.get(key, {}).get("revision", 0)
        if body.expected_revision is not None and body.expected_revision != revision:
            raise DomainError("REVISION_CONFLICT", "The draft was changed on another device.", 409)
        drafts[key] = {"value": body.value, "revision": revision + 1}
        if len(drafts) > 500:
            raise DomainError("INVALID_ARGUMENT", "Clear an old draft before saving another.")
        set_enabled(db, user.owner_id, drafts, "remotea_drafts")
        return drafts[key]
