"""Bug reports from the web app's Report a bug button.

The browser sends a description, what was expected, a screenshot taken before the dialog opened and
a diagnostics `context` (page, build, device, recent errors, requests and navigations). Each report is
stored in `bug_reports` and opens a task tagged `bug-report` in the project "Eridani bug reports", so it
shows in the task list, can be Run with an agent, and is readable with a `tasks:read` bot key
(`/api/v1/external/bug-reports…`). The browser leaves secrets out; `scrub` removes them again here.
"""

import base64
import binascii
import json
import os
import re
import time
from collections import defaultdict, deque
from datetime import timedelta
from typing import Annotated, Literal
from uuid import UUID

from fastapi import APIRouter, Depends, Query, Request
from fastapi.responses import Response
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from sqlalchemy import select, update
from sqlalchemy.orm import defer
from starlette.concurrency import run_in_threadpool

from .auth import Identity, authenticate
from .config import get_settings
from .db import session_scope
from .domain import DomainError, execute, owned
from .models import BugReport, Project, Task, now
from .redaction import MARK, redact

PATH = "/api/v1/bug-reports"
PROJECT = "Eridani bug reports"
TAG = "bug-report"
# A request carries a ~1.5 MB JPEG as base64 (4/3 larger) plus the context; api.py's security middleware
# allows this one route BODY_MAX instead of its 100,000-byte default.
BODY_MAX = 3_000_000
SCREENSHOT_MAX = 2_000_000
CONTEXT_MAX = 200_000
SCREENSHOT_DAYS = 90
STATUSES = ("open", "fixed", "dismissed")
IMAGES = {"image/jpeg": (b"\xff\xd8\xff",), "image/png": (b"\x89PNG\r\n\x1a\n",), "image/webp": (b"RIFF",)}
# Keys whose values are never stored, whatever the browser sent.
SECRET_KEY = re.compile(
    r"authorization|cookie|token|secret|password|passwd|api[_-]?key|csrf|credential|bearer|private[_-]?key|session[_-]?id",
    re.IGNORECASE,
)
URL_KEY = re.compile(r"url|href|path|referrer|location|source|script", re.IGNORECASE)
QUERY_VALUE = re.compile(r"([?&])([^=&#\s]+)=([^&#\s]*)")
# Page parameters that only name a page, a section or a record id (the web app's deep links).
SAFE_PARAMS = {"view", "run", "section", "activity", "task", "record", "note", "tab", "sharing", "google"}
reports_by_owner = defaultdict(deque)


class ReportInput(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    description: str = Field(min_length=1, max_length=10_000)
    expected: str = Field(default="", max_length=5_000)
    context: dict = Field(default_factory=dict)
    # A data URL (`data:image/jpeg;base64,…`); None when Include screenshot was off or capture failed.
    screenshot: str | None = Field(default=None, max_length=BODY_MAX)
    screenshot_error: str | None = Field(default=None, max_length=300)


class StatusInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    status: Literal["open", "fixed", "dismissed"]


def strip_query(text):
    """Keep a URL's path and parameter names; drop every query value except the deep-link ones."""
    return QUERY_VALUE.sub(
        lambda m: m.group(0) if m.group(2) in SAFE_PARAMS and re.fullmatch(r"[\w.:-]{0,80}", m.group(3))
        else f"{m.group(1)}{m.group(2)}=…", text)


def scrub(value, key="", depth=0):
    """Defense in depth for the browser's context: drop secret-named keys, redact secret-looking text,
    drop query values from URLs, and cap size and depth."""
    if depth > 8:
        return None
    if isinstance(value, dict):
        return {
            str(k)[:100]: MARK if SECRET_KEY.search(str(k)) else scrub(v, str(k), depth + 1)
            for k, v in list(value.items())[:200]
        }
    if isinstance(value, list):
        return [scrub(v, key, depth + 1) for v in value[:200]]
    if isinstance(value, str):
        text = redact(value[:4000])
        return strip_query(text) if URL_KEY.search(key) else text
    if isinstance(value, bool) or value is None or isinstance(value, (int, float)):
        return value
    return str(value)[:200]


def decode_screenshot(data_url):
    match = re.fullmatch(r"data:(image/(?:jpeg|png|webp));base64,([A-Za-z0-9+/=\s]+)", data_url or "")
    if not match:
        raise DomainError("INVALID_ARGUMENT", "The screenshot must be a JPEG, PNG or WebP data URL.")
    try:
        raw = base64.b64decode(match.group(2), validate=False)
    except (binascii.Error, ValueError):
        raise DomainError("INVALID_ARGUMENT", "The screenshot could not be read.") from None
    if len(raw) > SCREENSHOT_MAX:
        raise DomainError("INVALID_ARGUMENT", "The screenshot is too large.", 413)
    mime = match.group(1)
    if not raw.startswith(IMAGES[mime]) or (mime == "image/webp" and raw[8:12] != b"WEBP"):
        raise DomainError("INVALID_ARGUMENT", "The screenshot is not the image type it claims.")
    return mime, raw


def server_build():
    return {
        key: value
        for key, value in {
            "commit": (os.getenv("RAILWAY_GIT_COMMIT_SHA") or "")[:40],
            "deployment": (os.getenv("RAILWAY_DEPLOYMENT_ID") or "")[:60],
            "environment": get_settings().deployment_environment,
        }.items()
        if value
    }


def title_of(description):
    line = next((part.strip() for part in description.splitlines() if part.strip()), "Bug report")
    return line if len(line) <= 80 else line[:79].rstrip() + "…"


def links(report_id):
    origin = get_settings().origin.rstrip("/")
    return {
        "report": f"{origin}{PATH}/{report_id}",
        "screenshot": f"{origin}{PATH}/{report_id}/screenshot",
        "external": f"{origin}/api/v1/external/bug-reports/{report_id}",
    }


def bullet_list(items, line):
    return "\n".join("- " + line(item) for item in items if isinstance(item, dict))


def summary(report):
    """A short readable digest of the context for the task's notes."""
    ctx = report.context or {}
    page, build, device = ctx.get("page") or {}, ctx.get("build") or {}, ctx.get("device") or {}
    lines = [
        f"- View: {page.get('view') or '?'}" + (f" (run {page['run_id']})" if page.get("run_id") else "")
        + (f" (task {page['task_id']})" if page.get("task_id") else ""),
        f"- URL: {page.get('url') or '?'}",
        f"- Build: {build.get('version') or '?'} {build.get('commit') or ''} {build.get('built_at') or ''}".rstrip(),
        f"- Server: {(ctx.get('server') or {}).get('commit') or (ctx.get('server') or {}).get('environment') or '?'}",
        f"- Device: {device.get('platform') or '?'}, {device.get('viewport') or '?'}"
        + (", installed app" if device.get("standalone") else "") + ("" if device.get("online", True) else ", offline"),
    ]
    if page.get("dialogs"):
        lines.append("- Open dialogs: " + ", ".join(str(d) for d in page["dialogs"][:5]))
    errors = [e for e in (ctx.get("errors") or []) if isinstance(e, dict)][-5:]
    failing = [
        r for r in (ctx.get("requests") or [])
        if isinstance(r, dict) and (isinstance(r.get("status"), int) and r["status"] >= 400 or r.get("error"))
    ][-5:]
    text = "\n".join(lines)
    if errors:
        text += "\n\nLast errors:\n" + bullet_list(errors, lambda e: f"{e.get('level', 'error')}: {str(e.get('message', ''))[:300]}")
    if failing:
        text += "\n\nFailing requests:\n" + bullet_list(
            failing, lambda r: f"{r.get('method', '?')} {r.get('path', '?')} → {r.get('status') or r.get('error')}"
            + (f" ({r['error']})" if r.get("error") and r.get("status") else "")
            + (f" request {r['request_id']}" if r.get("request_id") else ""),
        )
    if report.screenshot_error:
        text += f"\n\nScreenshot: not captured ({report.screenshot_error})"
    return text


def task_notes(report):
    link = links(report.id)
    parts = [report.description]
    if report.expected:
        parts.append("Expected:\n" + report.expected)
    parts.append("Context:\n" + summary(report))
    parts.append(
        f"Report: {link['report']}"
        + (f"\nScreenshot: {link['screenshot']}" if report.screenshot_bytes else "")
        + f"\nBot API (tasks:read): {link['external']}"
    )
    text = "\n\n".join(parts)
    return text if len(text) <= 20_000 else text[:19_999] + "…"


def throttle(owner):
    attempts, clock = reports_by_owner[owner], time.monotonic()
    while attempts and attempts[0] < clock - 600:
        attempts.popleft()
    if len(attempts) >= 30:
        raise DomainError("RATE_LIMITED", "That is a lot of bug reports. Wait a few minutes and try again.", 429)
    attempts.append(clock)


def create(db, owner, account, body: ReportInput):
    mime, raw = decode_screenshot(body.screenshot) if body.screenshot else (None, None)
    context = scrub(body.context)
    if len(json.dumps(context)) > CONTEXT_MAX:
        raise DomainError("INVALID_ARGUMENT", "The attached details are too large.", 413)
    context["server"] = server_build()
    report = BugReport(
        owner_id=owner, account_id=account, description=redact(body.description), expected=redact(body.expected),
        context=context, screenshot=raw, screenshot_mime=mime, screenshot_bytes=len(raw or b""),
        screenshot_error=redact(body.screenshot_error)[:300] if body.screenshot_error and not raw else None,
    )
    db.add(report)
    db.flush()
    project = db.scalar(select(Project).where(Project.owner_id == owner, Project.name == PROJECT))
    arguments = {"title": title_of(report.description), "notes": task_notes(report), "tags": [TAG]}
    if not (project and project.archived):
        # task.create makes the project the first time (organization.project_changes).
        arguments["project"] = PROJECT
    task = execute(db, owner, f"bug-report:{report.id}", "task.create", arguments)["data"]
    report.task_id = task["id"]
    return public(db, report, detail=True)


def effective_status(report, task):
    """The stored status, except that an open report whose task was completed reads `fixed`, and one
    whose task was cancelled reads `dismissed`."""
    if report.status == "open" and task is not None:
        if task.status == "completed":
            return "fixed"
        if task.status == "cancelled":
            return "dismissed"
    return report.status


def public(db, report, *, detail=False, task=None):
    task = task if task is not None else (db.get(Task, report.task_id) if report.task_id else None)
    if task is not None and task.owner_id != report.owner_id:
        task = None
    item = {
        "id": report.id,
        "title": title_of(report.description),
        "status": effective_status(report, task),
        "created_at": report.created_at.isoformat(),
        "updated_at": report.updated_at.isoformat(),
        "task_id": report.task_id,
        "task_status": task.status if task else None,
        "view": ((report.context or {}).get("page") or {}).get("view"),
        "screenshot": (
            {"mime": report.screenshot_mime, "bytes": report.screenshot_bytes, "url": links(report.id)["screenshot"]}
            if report.screenshot_bytes else None
        ),
        "screenshot_error": report.screenshot_error,
        "links": links(report.id),
    }
    if detail:
        item.update(description=report.description, expected=report.expected, context=report.context or {},
                    summary=summary(report))
    return item


def listing(db, owner, *, status=None, task_id=None, limit=50, offset=0):
    query = select(BugReport).options(defer(BugReport.screenshot)).where(BugReport.owner_id == owner)
    if task_id:
        query = query.where(BugReport.task_id == task_id)
    rows = list(db.scalars(query.order_by(BugReport.created_at.desc(), BugReport.id)))
    tasks = {
        t.id: t for t in db.scalars(select(Task).where(Task.owner_id == owner, Task.id.in_([r.task_id for r in rows if r.task_id])))
    } if rows else {}
    items = [public(db, r, task=tasks.get(r.task_id)) for r in rows]
    if status:
        items = [i for i in items if i["status"] == status]
    return {"items": items[offset:offset + limit], "total": len(items)}


def get(db, owner, report_id):
    return public(db, owned(db, BugReport, str(report_id), owner), detail=True)


def screenshot(db, owner, report_id):
    report = owned(db, BugReport, str(report_id), owner)
    if not report.screenshot:
        raise DomainError("NOT_FOUND", "This report has no screenshot.", 404)
    return Response(
        report.screenshot, media_type=report.screenshot_mime,
        headers={"Content-Disposition": f'inline; filename="bug-report-{report.id}.{report.screenshot_mime.split("/")[1]}"'},
    )


def set_status(db, owner, report_id, status):
    report = owned(db, BugReport, str(report_id), owner, lock=True)
    report.status, report.updated_at = status, now()
    return public(db, report, detail=True)


def delete(db, owner, report_id):
    """Delete the report and its screenshot. The task stays: it is the owner's to archive."""
    report = owned(db, BugReport, str(report_id), owner, lock=True)
    db.delete(report)
    return {"deleted": True, "id": str(report_id), "task_id": report.task_id}


def sweep(db, owner, clock=None):
    """Retention (worker housekeeping): screenshots are kept SCREENSHOT_DAYS; the text and context stay."""
    cutoff = (clock or now()) - timedelta(days=SCREENSHOT_DAYS)
    db.execute(
        update(BugReport)
        .where(BugReport.owner_id == owner, BugReport.created_at < cutoff, BugReport.screenshot.is_not(None))
        .values(screenshot=None, screenshot_bytes=0, screenshot_mime=None,
                screenshot_error=f"removed after {SCREENSHOT_DAYS} days")
    )


def export(db, owner):
    """The account export's copy: every column, the screenshot as base64."""
    return [
        {
            "id": r.id, "account_id": r.account_id, "created_at": r.created_at.isoformat(),
            "updated_at": r.updated_at.isoformat(), "description": r.description, "expected": r.expected,
            "context": r.context, "status": r.status, "task_id": r.task_id, "screenshot_mime": r.screenshot_mime,
            "screenshot_error": r.screenshot_error,
            "screenshot_base64": base64.b64encode(r.screenshot).decode() if r.screenshot else None,
        }
        for r in db.scalars(select(BugReport).where(BugReport.owner_id == owner).order_by(BugReport.created_at))
    ]


async def read_body(request: Request):
    """The body, refused past BODY_MAX even when the client sent no Content-Length."""
    chunks, size = [], 0
    async for chunk in request.stream():
        size += len(chunk)
        if size > BODY_MAX:
            raise DomainError("INVALID_ARGUMENT", "This report is too large. Turn off the screenshot and try again.", 413)
        chunks.append(chunk)
    return b"".join(chunks)


router = APIRouter()
User = Annotated[Identity, Depends(authenticate)]


def owner_only(user):
    # Personal workspace only (access.PERSONAL_PATHS), where the signed-in account is the owner.
    if user.role != "owner":
        raise DomainError("NOT_AUTHORIZED", "Only the owner can file and read bug reports.", 403)


@router.post("/api/v1/bug-reports", status_code=201)
async def report_create(request: Request, user: User):
    owner_only(user)
    try:
        body = ReportInput.model_validate_json(await read_body(request))
    except ValidationError as exc:
        raise DomainError(
            "INVALID_ARGUMENT", "Check the report: it needs a description, and the fields must be valid.", data=exc.errors(include_url=False, include_context=False, include_input=False)
        ) from None
    throttle(user.owner_id)

    def save():
        with session_scope() as db:
            return create(db, user.owner_id, user.account_id, body)

    return await run_in_threadpool(save)


@router.get("/api/v1/bug-reports")
def report_list(user: User, status: Literal["open", "fixed", "dismissed"] | None = None, task_id: UUID | None = None,
                limit: int = Query(50, ge=1, le=200), offset: int = Query(0, ge=0, le=100_000)):
    owner_only(user)
    with session_scope() as db:
        return listing(db, user.owner_id, status=status, task_id=str(task_id) if task_id else None, limit=limit, offset=offset)


@router.get("/api/v1/bug-reports/{report_id}")
def report_get(report_id: UUID, user: User):
    owner_only(user)
    with session_scope() as db:
        return get(db, user.owner_id, report_id)


@router.get("/api/v1/bug-reports/{report_id}/screenshot")
def report_screenshot(report_id: UUID, user: User):
    owner_only(user)
    with session_scope() as db:
        return screenshot(db, user.owner_id, report_id)


@router.post("/api/v1/bug-reports/{report_id}/status")
def report_status(report_id: UUID, body: StatusInput, user: User):
    owner_only(user)
    with session_scope() as db:
        return set_status(db, user.owner_id, report_id, body.status)


@router.delete("/api/v1/bug-reports/{report_id}")
def report_delete(report_id: UUID, user: User):
    owner_only(user)
    with session_scope() as db:
        return delete(db, user.owner_id, report_id)
