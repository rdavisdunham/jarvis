import asyncio
import csv
import hashlib
import io
import json
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from datetime import date, timedelta
from typing import Annotated
from uuid import UUID

from fastapi import Depends, FastAPI, Query, Request
from fastapi.encoders import jsonable_encoder
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import func, select, text

from . import agent_models, budget
from .auth import Identity, authenticate, digest, sign_in
from .config import get_settings
from .conversation import chat
from .db import session_scope
from .domain import DomainError, execute, owned, preferences, serial
from .google_routes import router as google_router
from .memory_service import semantic_search
from .models import (
    AuthSession,
    Command,
    Conversation,
    Event,
    Job,
    Memory,
    MemoryReview,
    Note,
    NoteTaskLink,
    Notification,
    Occurrence,
    Project,
    PushSubscription,
    Schedule,
    Source,
    Task,
    WorkerHealth,
    now,
)
from .ui_control import UISync, sync
from .voice_options import available_options
from .worker import valid_push_endpoint


@asynccontextmanager
async def lifespan(app):
    yield
    from .voice import controllers

    await asyncio.gather(*(c.close() for c in list(controllers.values())), return_exceptions=True)


app = FastAPI(title="Jarvis", version="1.0.0", lifespan=lifespan)
app.include_router(google_router)
from .saved_views import router as saved_views_router

app.include_router(saved_views_router)
from .accounts import router as accounts_router

app.include_router(accounts_router)
from .integration_routes import router as integration_router

app.include_router(integration_router)
User = Annotated[Identity, Depends(authenticate)]
login_attempts = defaultdict(deque)


@app.exception_handler(DomainError)
async def domain_error(request, exc):
    return JSONResponse(
        status_code=exc.status,
        content=jsonable_encoder({"error": {"code": exc.code, "message": exc.message, "data": exc.data}}),
    )


@app.middleware("http")
async def security(request, call_next):
    settings = get_settings()
    origin = request.headers.get("origin")
    if request.url.path.startswith("/api/") and request.method not in {"GET", "HEAD", "OPTIONS"}:
        if origin and origin.rstrip("/") != settings.origin.rstrip("/"):
            return JSONResponse(
                {"error": {"code": "NOT_AUTHORIZED", "message": "This origin is not allowed."}},
                status_code=403,
            )
        if request.headers.get("sec-fetch-site") == "cross-site":
            return JSONResponse(
                {"error": {"code": "NOT_AUTHORIZED", "message": "Cross-site request rejected."}},
                status_code=403,
            )
    try:
        length = int(request.headers.get("content-length", "0") or "0")
    except ValueError:
        return JSONResponse(
            {"error": {"code": "INVALID_ARGUMENT", "message": "Invalid request length."}}, status_code=400
        )
    if length < 0 or length > 100_000:
        return JSONResponse(
            {"error": {"code": "INVALID_ARGUMENT", "message": "Request is too large."}}, status_code=413
        )
    from .access import principal
    token = principal.set(digest(request.cookies.get("jarvis_session","")) or None)
    try:
        response = await call_next(request)
        # A long retrieval cannot return shared data after access was revoked.
        if request.url.path.startswith("/api/v1/") and not request.url.path.startswith(("/api/v1/accounts","/api/v1/auth","/api/v1/integrations/google/unlink")) and response.status_code < 400:
            try:
                authenticate(request)
            except DomainError as exc:
                return JSONResponse({"error":{"code":exc.code,"message":exc.message}},status_code=exc.status)
    finally:
        principal.reset(token)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Permissions-Policy"] = "microphone=(self), camera=(), geolocation=()"
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; media-src 'self' blob:; connect-src 'self' https://api.openai.com wss://api.openai.com; frame-ancestors 'none'; base-uri 'self'; object-src 'none'"
    )
    if request.url.path.startswith("/api/"):
        response.headers["Cache-Control"] = "no-store"
    return response


class Input(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Login(Input):
    token: str = Field(min_length=1, max_length=256)


class Mutation(Input):
    command_id: str = Field(min_length=1, max_length=100)
    tool: str
    arguments: dict


class ConversationInput(Input):
    private: bool = False


class ChatInput(Input):
    turn_id: UUID
    conversation_id: UUID
    message: str = Field(min_length=1, max_length=12000)
    focus: str | None = None


class PushInput(Input):
    endpoint: str = Field(max_length=4096)
    keys: dict[str, str]
    expirationTime: float | None = None


@app.get("/health/live")
def live():
    return {"status": "alive"}


@app.get("/health/ready")
def ready():
    try:
        with session_scope() as db:
            db.execute(text("SELECT 1"))
        return {"status": "ready"}
    except Exception:  # noqa: BLE001 - isolate provider/process failures without exposing personal data
        return JSONResponse({"status": "unavailable"}, status_code=503)


@app.post("/api/v1/auth/login")
def login(body: Login, request: Request):
    key = request.client.host if request.client else "unknown"
    attempts = login_attempts[key]
    clock = time.monotonic()
    while attempts and attempts[0] < clock - 60:
        attempts.popleft()
    if len(attempts) >= 10:
        raise DomainError("RATE_LIMITED", "Wait a minute before trying another pairing code.", 429)
    attempts.append(clock)
    token, csrf = sign_in(body.token)
    attempts.clear()
    response = JSONResponse({"csrf": csrf})
    response.set_cookie(
        "jarvis_session",
        token,
        httponly=True,
        secure=get_settings().secure_cookie,
        samesite="strict",
        max_age=get_settings().session_hours * 3600,
    )
    return response


@app.post("/api/v1/auth/logout")
def logout(request: Request, user: User):
    with session_scope() as db:
        row = db.get(AuthSession, digest(request.cookies.get("jarvis_session", "")))
        if row:
            db.delete(row)
    response = JSONResponse({"signed_out": True})
    response.delete_cookie("jarvis_session")
    return response


@app.get("/api/v1/bootstrap")
def bootstrap(user: User):
    settings = get_settings()
    with session_scope() as db:
        from .access import person_preferences
        prefs = person_preferences(db,user.owner_id,user.device_id,preferences(db, user.owner_id))
        health = db.get(WorkerHealth, "worker")
        worker_healthy = bool(health and now() - health.last_scan_at < timedelta(seconds=30))
        backup_health = db.get(WorkerHealth, "backup")
        agent = agent_models.selected(prefs)
        return {
            "name": preferences(db,user.account_id or user.owner_id)["preferred_name"],
            "account_id":user.account_id or user.owner_id,
            "workspace": {"id":user.owner_id if user.account_id and user.account_id!=user.owner_id else None,
                          "name":prefs.get("shared_workspace","Personal"),"role":user.role},
            "agent_model": agent.model,
            "agent_profile": agent.profile_id,
            "agent_reasoning": agent.reasoning_effort,
            "agent_provider": agent.provider,
            "agent_options": [m.public() for m in agent_models.catalog().values()],
            "csrf": user.csrf,
            "device_id": user.device_id,
            "preferences": prefs,
            "budget": budget.summary(db, user.owner_id),
            "capabilities": {
                "voice": bool(settings.openai_api_key),
                "chat": agent.available,
                "push": bool(settings.vapid_public_key) and not prefs.get("shared_workspace"),
                "worker": worker_healthy,
            },
            "voice_options": available_options(),
            "last_backup_at": backup_health.last_scan_at.isoformat() if backup_health else None,
            "vapid_public_key": settings.vapid_public_key if not prefs.get("shared_workspace") else None,
            "event_cursor": db.scalar(select(func.max(Event.id)).where(Event.owner_id == user.owner_id)) or 0,
        }


@app.post("/api/v1/commands")
def command(body: Mutation, user: User):
    with session_scope() as db:
        result = execute(db, user.owner_id, body.command_id, body.tool, body.arguments)
    return result


@app.get("/api/v1/commands/{command_id}")
def command_status(command_id: str, user: User):
    with session_scope() as db:
        row = db.get(Command, (user.owner_id, command_id))
        if not row:
            raise DomainError("NOT_FOUND", "No accepted command with that ID.", 404)
        return row.result


@app.get("/api/v1/organization")
def organization(user: User):
    from .productivity import defaults, snapshot

    with session_scope() as db:
        defaults(db, user.owner_id)
        return snapshot(db, user.owner_id)


@app.get("/api/v1/projects")
def projects(user: User):
    from .productivity import data

    with session_scope() as db:
        return {
            "items": [
                data(db, p)
                for p in db.scalars(
                    select(Project).where(Project.owner_id == user.owner_id).order_by(Project.name)
                )
            ]
        }


@app.get("/api/v1/calendar")
def calendar_items(user: User, start: date, end: date, timezone: str | None = None):
    from .workspace import calendar

    with session_scope() as db:
        return calendar(db, user.owner_id, start, end, timezone or preferences(db, user.owner_id)["timezone"])


@app.get("/api/v1/notes")
def notes(
    user: User,
    q: str = Query(default="", max_length=300),
    space_id: str | None = None,
    area_id: str | None = None,
    goal_id: str | None = None,
    project_id: str | None = None,
    task_id: str | None = None,
    archived: bool = False,
    limit: int = Query(default=50, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
):
    from .notes import list_notes

    with session_scope() as db:
        return list_notes(
            db, user.owner_id, q, project_id, task_id, archived, limit, offset, space_id, area_id, goal_id
        )


@app.get("/api/v1/notes/search")
async def notes_search(
    user: User,
    q: str = Query(min_length=1, max_length=500),
    space_id: str | None = None,
    area_id: str | None = None,
    goal_id: str | None = None,
    project_id: str | None = None,
    task_id: str | None = None,
):
    from .notes import search_notes

    return await asyncio.to_thread(
        search_notes, user.owner_id, q, project_id, task_id, space_id, area_id, goal_id
    )


@app.get("/api/v1/notes/{note_id}")
def note_detail(note_id: str, user: User):
    from .notes import note_data

    with session_scope() as db:
        return note_data(db, owned(db, Note, note_id, user.owner_id))


@app.post("/api/v1/notes/{note_id}/suggest-tasks")
async def note_suggestions(note_id: str, user: User):
    from .notes import suggest_tasks

    return await asyncio.to_thread(suggest_tasks, user.owner_id, note_id)


@app.get("/api/v1/budget/holds")
def budget_holds(user: User):
    with session_scope() as db:
        return {"items": budget.holds(db, user.owner_id)}


@app.get("/api/v1/tasks")
def tasks(
    user: User, q: str = "", before: str | None = None, limit: int = 100, include_archived: bool = False
):
    limit = max(1, min(limit, 200))
    with session_scope() as db:
        query = select(Task).where(Task.owner_id == user.owner_id)
        if not include_archived:
            query = query.where(Task.archived.is_(False))
        if q:
            query = query.where(Task.title.ilike("%" + q[:200] + "%"))
        if before:
            query = query.where(Task.id < before)
        rows = list(db.scalars(query.order_by(Task.id.desc()).limit(max(1, min(limit, 200)) + 1)))
        return {
            "items": [serial(t) for t in rows[:limit]],
            "next_cursor": rows[limit - 1].id if len(rows) > limit else None,
        }


@app.get("/api/v1/tasks/{task_id}")
def task(task_id: str, user: User):
    with session_scope() as db:
        return serial(owned(db, Task, task_id, user.owner_id))


@app.get("/api/v1/schedules")
def schedules(user: User, before: str | None = None, limit: int = 200):
    limit = max(1, min(limit, 200))
    with session_scope() as db:
        q = select(Schedule).where(Schedule.owner_id == user.owner_id)
        if before:
            q = q.where(Schedule.id < before)
        rows = list(db.scalars(q.order_by(Schedule.id.desc()).limit(limit + 1)))
        return {
            "items": [serial(r) for r in rows[:limit]],
            "next_cursor": rows[limit - 1].id if len(rows) > limit else None,
        }


@app.get("/api/v1/schedules/{schedule_id}")
def schedule_record(schedule_id: str, user: User):
    with session_scope() as db:
        return serial(owned(db, Schedule, schedule_id, user.owner_id))


@app.get("/api/v1/notifications")
def notifications(user: User):
    with session_scope() as db:
        return {
            "items": [
                {
                    **serial(n),
                    "schedule_id": db.get(Occurrence, n.occurrence_id).schedule_id
                    if n.occurrence_id
                    else None,
                }
                for n in db.scalars(
                    select(Notification)
                    .where(Notification.owner_id == user.owner_id, Notification.dismissed_at.is_(None))
                    .order_by(Notification.created_at.desc())
                    .limit(100)
                )
            ]
        }


@app.post("/api/v1/push")
def push(body: PushInput, user: User):
    if not valid_push_endpoint(body.endpoint) or set(body.keys) != {"p256dh", "auth"}:
        raise DomainError("INVALID_ARGUMENT", "That push subscription is not supported.")
    if any(not value or len(value) > 500 for value in body.keys.values()):
        raise DomainError("INVALID_ARGUMENT", "Invalid subscription key.")
    sid = hashlib.sha256(body.endpoint.encode()).hexdigest()
    with session_scope() as db:
        row = db.get(PushSubscription, sid)
        if row and row.owner_id != user.owner_id:
            raise DomainError("NOT_AUTHORIZED", "Subscription is already registered.", 403)
        if row:
            row.subscription, row.active, row.device_id = body.model_dump(), True, user.device_id
        else:
            db.add(
                PushSubscription(
                    id=sid, owner_id=user.owner_id, device_id=user.device_id, subscription=body.model_dump()
                )
            )
    return {"registered": True}


@app.post("/api/v1/conversations")
def create_conversation(body: ConversationInput, user: User):
    with session_scope() as db:
        prefs = preferences(db, user.owner_id)
        row = Conversation(
            owner_id=user.owner_id,
            device_id=user.device_id,
            private=body.private or not prefs["history_enabled"],
            learning=prefs["memory_learning"],
        )
        db.add(row)
        db.flush()
        return serial(row)


@app.get("/api/v1/conversations/{conversation_id}")
def conversation(conversation_id: str, user: User):
    with session_scope() as db:
        row = owned(db, Conversation, conversation_id, user.owner_id)
        if row.device_id != user.device_id:
            raise DomainError("NOT_AUTHORIZED", "This conversation belongs to another device.", 403)
        sources = list(
            db.scalars(
                select(Source)
                .where(Source.conversation_id == row.id, Source.deleted_at.is_(None))
                .order_by(Source.created_at.desc())
                .limit(200)
            )
        )
        return {**serial(row), "messages": [serial(s) for s in reversed(sources)]}


@app.post("/api/v1/ui/sync")
async def ui_sync(body: UISync, user: User):
    return sync(user.owner_id, user.device_id, body)


@app.post("/api/v1/chat")
async def chat_endpoint(body: ChatInput, user: User):
    return await chat(
        user.owner_id, user.device_id, str(body.turn_id), str(body.conversation_id), body.message, body.focus
    )


@app.get("/api/v1/memory")
async def memories(user: User, q: str = ""):
    items = await semantic_search(user.owner_id, q[:500], 100 if not q else 20)
    with session_scope() as db:
        pending = db.scalar(
            select(func.count())
            .select_from(Job)
            .where(
                Job.owner_id == user.owner_id,
                Job.kind.in_(["extract_memory", "embed_memory"]),
                Job.status.in_(["queued", "running"]),
            )
        )
        retrying = db.scalar(
            select(func.count())
            .select_from(Job)
            .where(
                Job.owner_id == user.owner_id,
                Job.kind.in_(["extract_memory", "embed_memory"]),
                Job.status.in_(["retrying", "failed"]),
            )
        )
        deferred = db.scalar(
            select(func.count())
            .select_from(Job)
            .where(
                Job.owner_id == user.owner_id,
                Job.kind.in_(["extract_memory", "embed_memory"]),
                Job.status == "deferred_budget",
            )
        )
        enabled = preferences(db, user.owner_id)["memory_learning"]
        from .memory_review import pending_reviews, review_data, status

        reviews = [review_data(db, r) for r in pending_reviews(db, user.owner_id)]
        maintenance = status(db, user.owner_id)
    return {
        "items": items,
        "reviews": reviews,
        "maintenance": maintenance,
        "learning": {"enabled": enabled, "pending": pending, "retrying": retrying, "deferred": deferred},
    }


@app.post("/api/v1/memory/review")
def run_memory_review(user: User):
    from .memory_review import queue_review

    with session_scope() as db:
        job = queue_review(db, user.owner_id, manual=True)
        if not job:
            raise DomainError("INVALID_ARGUMENT", "Enable learning and weekly memory review first.")
        return {"job_id": job.id, "status": job.status}


@app.post("/api/v1/memory/retry")
def retry_memory(user: User):
    from .domain import enqueue_job
    from .memory_learning import VERSION, eligible

    count = 0
    with session_scope() as db:
        rows = list(
            db.scalars(
                select(Job)
                .where(
                    Job.owner_id == user.owner_id,
                    Job.kind.in_(["extract_memory", "embed_memory"]),
                    Job.status == "failed",
                )
                .with_for_update()
            )
        )
        for job in rows:
            if job.kind == "extract_memory":
                source = db.get(Source, job.payload["source_id"])
                if not eligible(db, source, user.owner_id):
                    continue
            job.status = "cancelled"
            enqueue_job(
                db,
                user.owner_id,
                job.kind,
                {k: v for k, v in job.payload.items() if k != "attempts"} | {"version": VERSION},
            )
            count += 1
    return {"queued": count}


@app.get("/api/v1/sources/{source_id}")
def source(source_id: str, user: User):
    with session_scope() as db:
        row = owned(db, Source, source_id, user.owner_id)
        if row.deleted_at:
            raise DomainError("NOT_FOUND", "This source has been deleted.", 404)
        return serial(row)


@app.get("/api/v1/jobs/{job_id}")
def job(job_id: str, user: User):
    with session_scope() as db:
        return serial(owned(db, Job, job_id, user.owner_id))


@app.get("/api/v1/events")
async def events(request: Request, user: User, after: int = 0):
    try:
        after = max(after, int(request.headers.get("last-event-id", "0")))
    except ValueError:
        pass

    async def stream():
        cursor = after
        ticks = 0
        yield "retry: 3000\n: connected\n\n"
        while not await request.is_disconnected():
            with session_scope() as db:
                session = db.get(AuthSession, digest(request.cookies.get("jarvis_session", "")))
                if not session or session.expires_at <= now():
                    return
                from .access import identity as resolve_identity
                try:
                    namespace,_=resolve_identity(db,session)
                    if namespace!=user.owner_id or session.device_id!=user.device_id:
                        yield 'event: access_revoked\ndata: {"code":"WORKSPACE_CHANGED"}\n\n'
                        return
                except DomainError:
                    yield "event: access_revoked\ndata: {}\n\n"
                    return
                events = list(
                    db.scalars(
                        select(Event)
                        .where(Event.owner_id == user.owner_id, Event.id > cursor)
                        .order_by(Event.id)
                        .limit(200)
                    )
                )
            for event in events:
                cursor = event.id
                yield f"id: {cursor}\ndata: {json.dumps(serial(event))}\n\n"
            ticks += 1
            if ticks % 10 == 0:
                yield ": heartbeat\n\n"
            if ticks % 40 == 0:
                yield "event: refresh\ndata: {}\n\n"
            await asyncio.sleep(0.5)

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"X-Accel-Buffering": "no", "Cache-Control": "no-cache, no-transform"},
    )


@app.get("/api/v1/export")
def export(user: User, format: str = "json"):
    from .models import Actor, Area, Goal, GoalProjectLink, NoteGoalLink, NoteNoteLink, NoteProjectLink, Space

    with session_scope() as db:
        data = {
            model.__tablename__: [
                serial(r) for r in db.scalars(select(model).where(model.owner_id == user.owner_id))
            ]
            for model in (
                Task,
                Project,
                Schedule,
                Notification,
                Memory,
                Source,
                MemoryReview,
                Note,
                Space,
                Area,
                Goal,
                Actor,
            )
        }
        data["note_task_links"] = [
            serial(link)
            for link in db.scalars(
                select(NoteTaskLink)
                .join(Note, Note.id == NoteTaskLink.note_id)
                .where(Note.owner_id == user.owner_id)
            )
        ]
        for link, model, field in (
            (GoalProjectLink, Goal, "goal_id"),
            (NoteGoalLink, Note, "note_id"),
            (NoteProjectLink, Note, "note_id"),
            (NoteNoteLink, Note, "note_id"),
        ):
            data[link.__tablename__] = [
                serial(r)
                for r in db.scalars(
                    select(link)
                    .join(model, model.id == getattr(link, field))
                    .where(model.owner_id == user.owner_id)
                )
            ]
    if format == "csv":
        buffer = io.StringIO()
        writer = csv.DictWriter(
            buffer,
            fieldnames=[
                "id",
                "title",
                "status",
                "space_id",
                "area_id",
                "planned_date",
                "estimate_minutes",
                "assignee_id",
                "due_date",
                "due_time",
                "due_timezone",
                "project",
                "project_id",
                "parent_task_id",
                "assignee",
                "work_type",
                "tags",
                "priority",
                "notes",
            ],
        )
        writer.writeheader()
        for task in data["tasks"]:
            # Spreadsheet programs must treat user text as text, never executable formulas.
            cells = {k: task.get(k) for k in writer.fieldnames}
            writer.writerow(
                {
                    k: "'" + v if isinstance(v, str) and v.startswith(("=", "+", "-", "@", "\t", "\r")) else v
                    for k, v in cells.items()
                }
            )
        return Response(
            buffer.getvalue(),
            media_type="text/csv",
            headers={"Content-Disposition": 'attachment; filename="jarvis-tasks.csv"'},
        )
    return JSONResponse(
        {"version": 1, "exported_at": now().isoformat(), **data},
        headers={"Content-Disposition": 'attachment; filename="jarvis-export.json"'},
    )


# Routes are included before the SPA fallback so API errors never become HTML.
from .voice import router as voice_router

app.include_router(voice_router, prefix="/api/v1")


@app.get("/{path:path}")
def frontend(path: str):
    if path.startswith(("api/", "health/")):
        raise DomainError("NOT_FOUND", "Endpoint not found.", 404)
    root = get_settings().web_dist.resolve()
    candidate = (root / path).resolve()
    if candidate.is_relative_to(root) and candidate.is_file():
        return FileResponse(candidate)
    index = root / "index.html"
    if index.is_file():
        return FileResponse(index)
    return JSONResponse({"message": "Build the web app with npm run build in apps/web."}, status_code=503)
