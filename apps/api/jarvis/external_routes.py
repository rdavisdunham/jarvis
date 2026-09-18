"""Browser-managed bot keys and a Bearer-only public planner API."""

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, Query, Request
from pydantic import Field
from sqlalchemy import select
from starlette.concurrency import run_in_threadpool

from . import bot_access
from . import external_service as service
from .auth import Identity, authenticate
from .config import get_settings
from .db import session_scope
from .domain import DomainError, advisory
from .models import BotCredential, now

router = APIRouter()
User = Annotated[Identity, Depends(authenticate)]


class KeyInput(service.Input):
    name: str = Field(min_length=1, max_length=80)
    scopes: list[str] = Field(min_length=1, max_length=11)
    expires_in_days: int = Field(default=90, ge=1, le=365)


def managed_key(db, user, identity):
    from .access import role

    permission = role(db, user.owner_id, user.account_id, lock=True)
    row = db.get(BotCredential, str(identity))
    if (
        not row
        or row.owner_id != user.owner_id
        or (row.account_id != user.account_id and permission != "owner")
    ):
        raise DomainError("NOT_FOUND", "That bot key is not available.", 404)
    return row


@router.get("/api/v1/bot-keys")
def keys(user: User):
    with session_scope() as db:
        query = select(BotCredential).where(BotCredential.owner_id == user.owner_id)
        if user.role != "owner":
            query = query.where(BotCredential.account_id == user.account_id)
        return {
            "items": [
                bot_access.public(row) for row in db.scalars(query.order_by(BotCredential.created_at.desc()))
            ],
            "scopes": bot_access.SCOPES,
            "api_url": get_settings().origin.rstrip("/") + "/api/v1/external",
            "mcp_url": get_settings().origin.rstrip("/") + "/api/v1/external/mcp/",
        }


@router.post("/api/v1/bot-keys", status_code=201)
def create_key(body: KeyInput, user: User):
    with session_scope() as db:
        return bot_access.create(
            db, user.owner_id, user.account_id, body.name, body.scopes, body.expires_in_days
        )


@router.post("/api/v1/bot-keys/{identity}/revoke")
def revoke_key(identity: UUID, user: User):
    with session_scope() as db:
        row = managed_key(db, user, identity)
        advisory(db, "bot:" + row.id)
        row.revoked_at = row.revoked_at or now()
        return bot_access.public(row)


async def external_identity(request: Request):
    identity = await run_in_threadpool(bot_access.authenticate, request)
    with bot_access.bind(identity):
        yield identity


Bot = Annotated[str, Depends(external_identity)]
external = APIRouter(prefix="/api/v1/external", tags=["External agents"])


@external.get("/capabilities")
def capabilities(identity: Bot):
    from .external_mcp import definitions

    with session_scope() as db:
        bot = bot_access.authorize(db)
        return {
            "version": "1",
            "name": bot.name,
            "workspace_id": bot.owner_id,
            "scopes": bot.scopes,
            "tools": definitions(bot.scopes),
            "rate_limit_per_minute": 120,
            "authentication": "Bearer bot key",
        }


from .search_schema import SearchQuery

@external.post("/search")
async def semantic_search(body: SearchQuery, identity: Bot):
    with session_scope() as db:
        bot=bot_access.authorize(db,required="records:read")
        owner,account=bot.owner_id,bot.account_id
    from .search_service import search
    return await run_in_threadpool(search,owner,account,body,track=False)


@external.get("/records/{kind}")
def records(kind: service.Kind, identity: Bot, query: Annotated[service.Search, Query()]):
    with session_scope() as db:
        return service.search(db, bot_access.authorize(db), kind, query)


@external.get("/records/{kind}/{record_id}")
def record(kind: service.Kind, record_id: UUID, identity: Bot):
    with session_scope() as db:
        return service.get_record(db, bot_access.authorize(db), kind, record_id)


@external.post("/commands")
def command(body: service.Mutation, identity: Bot):
    with session_scope() as db:
        bot = bot_access.authorize(db)
        # Recheck scope even for idempotent replays.
        bot_access.authorize(db, bot.owner_id, bot_access.COMMAND_SCOPES.get(body.tool))
        if body.tool not in bot_access.COMMAND_SCOPES:
            raise DomainError("INSUFFICIENT_SCOPE", "That action is not available to external bots.", 403)
        return service.direct(db, bot, body.request_id, body.tool, body.arguments)


@external.get("/changes")
def changes(
    identity: Bot, after: int = Query(default=0, ge=0), limit: int = Query(default=100, ge=1, le=200)
):
    with session_scope() as db:
        return service.changes(db, bot_access.authorize(db), after, limit)


@external.post("/requests", status_code=202)
def submit(body: service.RequestInput, identity: Bot):
    with session_scope() as db:
        return service.submit(db, bot_access.authorize(db), body)


@external.get("/requests/{request_id}")
def request_status(request_id: UUID, identity: Bot):
    with session_scope() as db:
        bot = bot_access.authorize(db)
        return service.public_work(db, service.get_work(db, bot, request_id))


@external.post("/requests/{request_id}/reply")
def reply(request_id: UUID, body: service.ReplyInput, identity: Bot):
    with session_scope() as db:
        return service.reply(db, bot_access.authorize(db), request_id, body)


@external.post("/requests/{request_id}/cancel")
def cancel(request_id: UUID, identity: Bot):
    from .agent_work import cancel as cancel_work

    with session_scope() as db:
        advisory(db, "work:" + str(request_id))
        bot = bot_access.authorize(db, required="work:run", write=True)
        return cancel_work(db, service.get_work(db, bot, request_id))


class RevertInput(service.Input):
    request_id: UUID


@external.post("/actions/{action_id}/revert")
def revert(action_id: UUID, body: RevertInput, identity: Bot):
    with session_scope() as db:
        return service.direct(
            db, bot_access.authorize(db), body.request_id, "action.revert", {"action_id": str(action_id)}
        )





@external.get("/structure")
def external_structure(identity: Bot):
    from .structure import schema_data
    with session_scope() as db:
        key=bot_access.authorize(db,required="schema:read")
        return schema_data(db,key.owner_id)

@external.get("/structure/records")
def external_records(identity: Bot,type_id: str | None=None,capability: str | None=None,query: str="",limit: int=Query(100,ge=1,le=200),offset: int=Query(0,ge=0)):
    from .structure import records
    with session_scope() as db:
        key=bot_access.authorize(db,required="records:read")
        return records(db,key.owner_id,type_id=type_id,capability=capability,query=query,limit=limit,offset=offset)

@external.get("/structure/records/{record_id}")
def external_record(record_id: UUID,identity: Bot):
    from .structure import data,ensure
    from .structure_models import StructureRecord
    from .domain import owned
    with session_scope() as db:
        key=bot_access.authorize(db,required="records:read");ensure(db,key.owner_id)
        return data(db,owned(db,StructureRecord,str(record_id),key.owner_id))

router.include_router(external)
