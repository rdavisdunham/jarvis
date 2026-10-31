"""Browser-managed bot keys and a Bearer-only public planner API."""

from typing import Annotated, Literal
from uuid import UUID

from fastapi import APIRouter, Depends, Query, Request
from pydantic import Field
from sqlalchemy import select
from starlette.concurrency import run_in_threadpool

from . import agent_sessions as sessions
from . import bot_access
from . import external_service as service
from .auth import Identity, authenticate
from .config import get_settings
from .db import session_scope
from .domain import DomainError, advisory, emit
from .models import BotCredential, now

router = APIRouter()
User = Annotated[Identity, Depends(authenticate)]


class KeyInput(service.Input):
    name: str = Field(min_length=1, max_length=80)
    scopes: list[str] = Field(min_length=1, max_length=len(bot_access.SCOPES))
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
        owner,account,scopes=bot.owner_id,bot.account_id,list(bot.scopes)
    from .search_service import search
    result = await run_in_threadpool(search,owner,account,body,track=False,withhold=service.withheld(scopes))

    def scrubbed():
        with session_scope() as db:
            return service.custom_scrub(db, owner, result, scopes)

    return await run_in_threadpool(scrubbed)


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
    with session_scope() as db:
        return service.cancel_work(db, bot_access.authorize(db), request_id)


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
def external_records(identity: Bot,type_id: str | None=None,capability: str | None=None,query: str="",limit: int=Query(25,ge=1,le=50),offset: int=Query(0,ge=0)):
    with session_scope() as db:
        key=bot_access.authorize(db,required="records:read")
        arguments={k:v for k,v in dict(type_id=type_id,capability=capability,query=query,limit=limit,offset=offset).items() if v is not None}
        return service.custom_read(db,key,"record_list",arguments)

@external.get("/structure/atlas")
def external_atlas(identity: Bot, limit: int = Query(1000, ge=1, le=5000)):
    from .atlas import skeleton
    with session_scope() as db:
        key=bot_access.authorize(db,required="records:read")
        return skeleton(db,key.owner_id,limit=limit,withhold=service.withheld(key.scopes))

@external.get("/structure/templates")
def external_templates(identity: Bot, type_id: str | None = None):
    with session_scope() as db:
        key=bot_access.authorize(db,required="records:read")
        return service.custom_read(db,key,"template_list",{"type_id":type_id} if type_id else {})

@external.get("/structure/reviews")
def external_reviews(identity: Bot, within_days: int = Query(0, ge=0, le=366), type_id: str | None = None,
                     limit: int = Query(25, ge=1, le=100), offset: int = Query(0, ge=0, le=100000)):
    with session_scope() as db:
        key=bot_access.authorize(db,required="records:read")
        args={"within_days":within_days,"limit":limit,"offset":offset,**({"type_id":type_id} if type_id else {})}
        return service.custom_read(db,key,"reviews_due",args)

@external.get("/structure/records/{record_id}")
def external_record(record_id: UUID,identity: Bot):
    with session_scope() as db:
        key=bot_access.authorize(db,required="records:read")
        return service.custom_read(db,key,"record_get",{"record_id":str(record_id)})

@external.put("/sessions/{tool}/{session_id}")
def session_report(tool: sessions.Tool, session_id: str, body: sessions.SessionInput, identity: Bot):
    with session_scope() as db:
        bot = bot_access.authorize(db, required="sessions:write", write=True)
        result = sessions.upsert(db, bot.owner_id, tool, session_id, body, reveal="records:read" in bot.scopes)
        emit(db, bot.owner_id, "session.changed", result["id"])
        return result


class ParityInput(service.Input):
    tool: sessions.Tool
    session_id: str = Field(min_length=1, max_length=100)
    report: sessions.SessionInput


@external.post("/agents/parity")
def agent_parity(body: ParityInput, identity: Bot):
    """Shadow collector check: compare a report with the stored session without changing it."""
    from . import agent_link

    with session_scope() as db:
        bot = bot_access.authorize(db, required="agents:control")
        agent_link.gate(db, bot.owner_id)
        host = db.scalar(
            select(agent_link.AgentHost.id)
            .where(agent_link.AgentHost.credential_id == bot.id)
            .order_by(agent_link.AgentHost.last_seen_at.desc().nulls_last())
            .limit(1)
        )
        return agent_link.parity(db, bot, host, body.tool, body.session_id, body.report)


@external.get("/agents/runs")
def agent_runs(identity: Bot, task_id: UUID | None = None, active: bool = False, limit: int = Query(50, ge=1, le=100)):
    """Runs started from tasks, newest first (scope agents:read): the same read model as the remotea page."""
    from . import agent_control as control

    with session_scope() as db:
        bot = bot_access.authorize(db, required="agents:read")
        return {"items": control.list_runs(db, bot.owner_id, task_id=str(task_id) if task_id else None,
                                           active=active, limit=limit)}


@external.get("/agents/runs/{run_id}")
def agent_run(run_id: UUID, identity: Bot, limit: int = Query(100, ge=1, le=200), offset: int = Query(0, ge=0, le=100_000)):
    """One run with its timeline: frozen plan, history, turns, commands, questions, interviews and verify/judge
    attempts (scope agents:read). Paged like the browser route: newest `limit` turns/commands after `offset`."""
    from . import agent_control as control
    from . import run_coordinator as coordinator

    with session_scope() as db:
        bot = bot_access.authorize(db, required="agents:read")
        run = coordinator.load(db, bot.owner_id, str(run_id), lock=False)
        return control.public_run(db, run, detail=True, limit=limit, offset=offset)


@external.get("/bug-reports")
def bug_report_list(identity: Bot, status: Literal["open", "fixed", "dismissed"] | None = None, task_id: UUID | None = None,
                    limit: int = Query(50, ge=1, le=200), offset: int = Query(0, ge=0, le=100_000)):
    """Reports from the web app's Report a bug button, newest first (scope tasks:read; each one is also a task)."""
    from . import bug_reports

    with session_scope() as db:
        bot = bot_access.authorize(db, required="tasks:read")
        return bug_reports.listing(db, bot.owner_id, status=status, task_id=str(task_id) if task_id else None,
                                   limit=limit, offset=offset)


@external.get("/bug-reports/{report_id}")
def bug_report_get(report_id: UUID, identity: Bot):
    """One report: description, expected, the scrubbed context, a readable summary and links (scope tasks:read)."""
    from . import bug_reports

    with session_scope() as db:
        return bug_reports.get(db, bot_access.authorize(db, required="tasks:read").owner_id, report_id)


@external.get("/bug-reports/{report_id}/screenshot")
def bug_report_screenshot(report_id: UUID, identity: Bot):
    """The report's screenshot image (scope tasks:read); 404 when it has none."""
    from . import bug_reports

    with session_scope() as db:
        return bug_reports.screenshot(db, bot_access.authorize(db, required="tasks:read").owner_id, report_id)


@external.post("/agents/session/plan")
def agent_task_plan(body: dict, identity: Bot):
    """HTTP twin of the task_plan tool (scope agents:session)."""
    from . import agent_plan
    from .external_mcp import session_arguments

    with session_scope() as db:
        bot = bot_access.authorize(db, required="agents:session")
        return agent_plan.task_plan(db, bot, session_arguments(agent_plan.PlanQuery, body))


@external.post("/agents/session/step")
def agent_task_step(body: dict, identity: Bot):
    """HTTP twin of the task_step tool."""
    from . import agent_plan
    from .external_mcp import session_arguments

    with session_scope() as db:
        bot = bot_access.authorize(db, required="agents:session")
        return agent_plan.task_step(db, bot, session_arguments(agent_plan.StepInput, body))


@external.post("/agents/session/interview")
def agent_task_interview(body: dict, identity: Bot):
    """HTTP twin of the task_interview tool."""
    from . import agent_plan
    from .external_mcp import session_arguments

    with session_scope() as db:
        bot = bot_access.authorize(db, required="agents:session")
        return agent_plan.task_interview(db, bot, session_arguments(agent_plan.InterviewInput, body))


@external.post("/agents/session/submit_plan")
def agent_submit_plan(body: dict, identity: Bot):
    """HTTP twin of the submit_plan tool."""
    from . import agent_plan
    from .external_mcp import session_arguments

    with session_scope() as db:
        bot = bot_access.authorize(db, required="agents:session")
        return agent_plan.submit_plan(db, bot, session_arguments(agent_plan.SubmitPlanInput, body))


@external.post("/sessions/{tool}/{session_id}/link")
def session_link(tool: sessions.Tool, session_id: str, body: sessions.LinkInput, identity: Bot):
    with session_scope() as db:
        bot = bot_access.authorize(db, required="sessions:write", write=True)
        return sessions.link(
            db, bot.owner_id, tool, session_id, body, source="explicit", reveal="records:read" in bot.scopes
        )


@external.get("/sessions/rules")
def session_rules(identity: Bot):
    with session_scope() as db:
        bot = bot_access.authorize(db, required="sessions:read")
        return sessions.rules(db, bot.owner_id, reveal="records:read" in bot.scopes)


@external.post("/sessions/rules")
def session_rule_save(body: sessions.RuleInput, identity: Bot):
    with session_scope() as db:
        bot = bot_access.authorize(db, required="sessions:write", write=True)
        return sessions.save_rule(db, bot.owner_id, body, reveal="records:read" in bot.scopes)


@external.delete("/sessions/rules/{rule_id}")
def session_rule_delete(rule_id: UUID, identity: Bot):
    with session_scope() as db:
        bot = bot_access.authorize(db, required="sessions:write", write=True)
        return sessions.delete_rule(db, bot.owner_id, rule_id, reveal="records:read" in bot.scopes)


@external.get("/sessions")
async def session_search(identity: Bot, query: Annotated[sessions.Listing, Query()]):
    with session_scope() as db:
        bot = bot_access.authorize(db, required="sessions:read")
        owner, reveal = bot.owner_id, "records:read" in bot.scopes
    return await run_in_threadpool(sessions.search, owner, query, reveal=reveal)


router.include_router(external)
