"""Browser routes for the remotea page and its Settings toggle (off by default).

Enable, disable and status work in either state; every other /api/v1/remotea route answers
403 REMOTEA_DISABLED until the owner turns remotea on.
"""

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel, ConfigDict
from sqlalchemy import or_, select
from sqlalchemy.dialects.postgresql import distinct_on

from . import agent_link, bot_access
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
        "key": bot_access.public(key) if key else None,
        "hosts": [agent_link.public_host(h) for h in hosts],
        "link_url": link_url(),
        "connect_command": "remotea eridani connect --url " + link_url(),
    }


def set_enabled(db, owner, value):
    advisory(db, f"budget:{owner}")  # The same lock settings.update takes for owner preferences.
    row = db.get(OwnerSettings, owner)
    if row:
        row.values = {**row.values, "remotea_enabled": value}
    else:
        db.add(OwnerSettings(owner_id=owner, values={"remotea_enabled": value}))
    db.flush()
    emit(db, owner, "settings.changed", owner)


@router.get("/status")
def status(user: User):
    with session_scope() as db:
        return state(db, user)


@router.post("/enable")
def enable(body: EnableInput, user: User):
    """Turn remotea on and mint the link key. The token appears in this response only."""
    with session_scope() as db:
        key = active_key(db, user)
        minted = None
        if key is None or body.rotate:
            minted = bot_access.create(db, user.owner_id, user.account_id, KEY_NAME, KEY_SCOPES, KEY_DAYS)
            if key is not None:
                advisory(db, "bot:" + key.id)
                key.revoked_at = now()
        else:
            from .access import role

            if role(db, user.owner_id, user.account_id, lock=True) == "viewer":
                raise DomainError("READ_ONLY", "Viewers cannot turn on remotea.", 403)
        set_enabled(db, user.owner_id, True)
        return {**state(db, user), "token": minted["token"] if minted else None}


@router.post("/disable")
def disable(user: User):
    """Stop admitting the link; connected hosts are closed on their next re-authorization."""
    from .access import role

    with session_scope() as db:
        if role(db, user.owner_id, user.account_id, lock=True) == "viewer":
            raise DomainError("READ_ONLY", "Viewers cannot turn off remotea.", 403)
        set_enabled(db, user.owner_id, False)
        return state(db, user)


def gated(db, user):
    agent_link.gate(db, user.owner_id)


def last_turns(db, ids):
    if not ids:
        return {}
    latest = {}
    for row in db.scalars(
        select(AgentTurn).where(AgentTurn.session_pk.in_(ids)).order_by(AgentTurn.session_pk, AgentTurn.seq.desc(), AgentTurn.created_at.desc())
        .ext(distinct_on(AgentTurn.session_pk))
    ):
        latest[row.session_pk] = row
    return latest


def public_turn(row, limit=None):
    text = row.text if limit is None else row.text[:limit]
    return encode({"id": row.id, "seq": row.seq, "text": text, "length": len(row.text), "clipped": len(text) < len(row.text),
                   "truncated": row.truncated, "generation": row.generation, "source": row.source, "at": row.created_at})


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
