import asyncio
import logging
import secrets
import time
from collections import defaultdict, deque
from typing import Annotated

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse, RedirectResponse
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import select

from .auth import Identity, authenticate, digest
from .config import get_settings
from .db import session_scope
from .domain import DomainError, parse_when
from .google_auth import CALLBACK_PATH, COOKIE, begin, callback_uri, configured, disconnect_calendar, finish
from .google_calendar import availability, connection_status, queue_sync
from .google_schema import CalendarRead
from .google_writes import read_event, write_status
from .models import GoogleOAuthAttempt, Job


class OAuthLogFilter(logging.Filter):
    def filter(self, record):
        if (
            isinstance(record.args, tuple)
            and len(record.args) >= 3
            and isinstance(record.args[2], str)
            and record.args[2].split("?")[0] == CALLBACK_PATH
        ):
            args = list(record.args)
            args[2] = CALLBACK_PATH
            record.args = tuple(args)
        return True


logging.getLogger("uvicorn.access").addFilter(OAuthLogFilter())
oauth_attempts = defaultdict(deque)
router = APIRouter()
User = Annotated[Identity, Depends(authenticate)]


class Start(BaseModel):
    model_config = ConfigDict(extra="forbid")
    purpose: str
    return_to: str | None = Field(default=None, max_length=1000)


class AvailabilityRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    start: str = Field(max_length=64)
    end: str = Field(max_length=64)
    minutes: int = Field(default=30, ge=5, le=480)


@router.get("/api/v1/auth/options")
def options():
    settings = get_settings()
    return {"google": configured(), "pairing": settings.pairing_enabled and bool(settings.owner_token)}


@router.post("/api/v1/auth/google/start")
def start(body: Start, request: Request):
    key = request.client.host if request.client else "unknown"
    attempts, clock = oauth_attempts[key], time.monotonic()
    while attempts and attempts[0] < clock - 60:
        attempts.popleft()
    if len(attempts) >= 10:
        raise DomainError("RATE_LIMITED", "Wait a minute before starting Google sign-in again.", 429)
    attempts.append(clock)
    session_hash = None
    if body.purpose != "login":
        authenticate(request)
        session_hash = digest(request.cookies.get("jarvis_session", ""))
    url, browser = begin(body.purpose, session_hash, body.return_to)
    response = JSONResponse({"url": url})
    response.set_cookie(
        COOKIE,
        browser,
        httponly=True,
        secure=get_settings().secure_cookie,
        samesite="lax",
        max_age=600,
        path=CALLBACK_PATH,
    )
    return response


@router.get(CALLBACK_PATH)
async def callback(request: Request):
    destination, purpose = "/?view=tasks", "login"
    state, browser = request.query_params.get("state"), request.cookies.get(COOKIE)
    if state and browser:
        with session_scope() as db:
            attempt = db.get(GoogleOAuthAttempt, digest(state))
            if attempt and secrets.compare_digest(attempt.browser_hash, digest(browser)):
                destination = attempt.return_to or "/?view=settings"
                purpose = attempt.purpose
    try:
        token, _ = await asyncio.to_thread(
            finish,
            request.query_params.get("state"),
            request.cookies.get(COOKIE),
            request.query_params.get("code"),
            request.query_params.get("error"),
        )
        response = RedirectResponse(destination + (("&" if "?" in destination else "?") + "google=connected" if purpose != "login" else ""), status_code=303)
        response.set_cookie(
            "jarvis_session",
            token,
            httponly=True,
            secure=get_settings().secure_cookie,
            samesite="strict",
            max_age=get_settings().session_hours * 3600,
        )
    except DomainError as error:
        # Fixed codes only; neither Google text nor authorization parameters reach the SPA.
        known = {"GOOGLE_DENIED": "cancelled", "GOOGLE_SCOPE": "permission", "GOOGLE_ACCOUNT": "account"}
        response = RedirectResponse(
            destination + ("&" if "?" in destination else "?") + "google=" + known.get(error.code, "failed"), status_code=303
        )
    response.delete_cookie(COOKIE, path=CALLBACK_PATH)
    return response


@router.get("/api/v1/integrations/google")
def status(user: User):
    with session_scope() as db:
        return {**connection_status(db, user.owner_id), "callback_uri": callback_uri()}


@router.post("/api/v1/integrations/google/sync")
def sync(user: User):
    with session_scope() as db:
        job_id = queue_sync(db, user.owner_id, force=True)
        if not job_id:
            raise DomainError("GOOGLE_RECONNECT", "Connect Google Calendar in Settings first.", 409)
        return {"job_id": job_id, "status": "queued"}


@router.post("/api/v1/integrations/google/disconnect-calendar")
async def disconnect(user: User):
    return await asyncio.to_thread(disconnect_calendar, user.owner_id)


@router.post("/api/v1/calendar/availability")
async def available(body: AvailabilityRequest, user: User):
    return await asyncio.to_thread(availability, user.owner_id, body.start, body.end, body.minutes)


class DayAvailability(BaseModel):
    model_config = ConfigDict(extra="forbid")
    date: str = Field(max_length=10)
    start_time: str = Field(pattern=r"^\d{2}:\d{2}$")
    end_time: str = Field(pattern=r"^\d{2}:\d{2}$")
    timezone: str = Field(max_length=100)
    minutes: int = Field(default=30, ge=5, le=480)


@router.post("/api/v1/calendar/availability/day")
async def day_availability(body: DayAvailability, user: User):
    start = parse_when(body.date + "T" + body.start_time, body.timezone)
    end = parse_when(body.date + "T" + body.end_time, body.timezone)
    return await asyncio.to_thread(
        availability, user.owner_id, start.isoformat(), end.isoformat(), body.minutes
    )


@router.get("/api/v1/calendar/events/{event_id}")
def event(event_id: str, user: User):
    from .google_calendar import event_detail

    with session_scope() as db:
        return event_detail(db, user.owner_id, event_id)


@router.post("/api/v1/integrations/google/unlink")
async def unlink(user: User):
    from .google_auth import unlink_google

    return await asyncio.to_thread(unlink_google, user.owner_id)


@router.post("/api/v1/calendar/event-detail")
async def event_read(body: CalendarRead, user: User):
    return await asyncio.to_thread(read_event, user.owner_id, body)


@router.get("/api/v1/calendar/writes")
def writes(user: User):
    with session_scope() as db:
        rows = db.scalars(
            select(Job)
            .where(Job.owner_id == user.owner_id, Job.kind == "google_write")
            .order_by(Job.created_at.desc())
            .limit(20)
        )
        return {"items": [write_status(db, user.owner_id, row.id) for row in rows]}


@router.get("/api/v1/calendar/writes/{job_id}")
def write(job_id: str, user: User):
    with session_scope() as db:
        return write_status(db, user.owner_id, job_id)
