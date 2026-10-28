"""Browser access to the agent session ledger: find sessions, file them and manage folder rules."""

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, Query
from starlette.concurrency import run_in_threadpool

from . import agent_sessions as sessions
from .auth import Identity, authenticate
from .db import session_scope
from .domain import owned
from .models import now
from .session_models import AgentSession

router = APIRouter(prefix="/api/v1/sessions")
User = Annotated[Identity, Depends(authenticate)]


class ArchiveInput(sessions.Input):
    archived: bool


@router.get("")
async def listing(user: User, query: Annotated[sessions.Listing, Query()]):
    return await run_in_threadpool(sessions.search, user.owner_id, query)


@router.get("/rules")
def rules(user: User):
    with session_scope() as db:
        return sessions.rules(db, user.owner_id)


@router.post("/rules")
def save_rule(body: sessions.RuleInput, user: User):
    with session_scope() as db:
        return sessions.save_rule(db, user.owner_id, body)


@router.delete("/rules/{identity}")
def delete_rule(identity: UUID, user: User):
    with session_scope() as db:
        return sessions.delete_rule(db, user.owner_id, identity)


@router.post("/{identity}/link")
def link(identity: UUID, body: sessions.LinkInput, user: User):
    with session_scope() as db:
        row = owned(db, AgentSession, str(identity), user.owner_id)
        return sessions.link(db, user.owner_id, row.tool, row.session_id, body)


@router.post("/{identity}/autolink/undo")
def undo_autolink(identity: UUID, user: User):
    """Undo a dream-sequence link: tombstones (session, task) so no later cycle re-links it."""
    from .agent_autolink import undo

    with session_scope() as db:
        return undo(db, user.owner_id, str(identity))


@router.post("/{identity}/archive")
def archive(identity: UUID, body: ArchiveInput, user: User):
    with session_scope() as db:
        row = owned(db, AgentSession, str(identity), user.owner_id)
        row.archived, row.updated_at = body.archived, now()
        return sessions.public(db, row)
