"""Synthetic Google transport for isolated browser acceptance. Never use the owner database."""

import asyncio
import copy
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import ClassVar
from urllib.parse import unquote
from uuid import uuid4

from jarvis import google_auth, google_calendar, linear_commands, linear_sync

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from test_linear import Fake

linear_fixture = Fake()
linear_sync.LinearClient = lambda *_: linear_fixture
linear_commands.LinearClient = lambda *_: linear_fixture
from jarvis.api import User, app
from jarvis.db import session_scope
from jarvis.google_writes import process_write
from jarvis.models import Job
from jarvis.tools import call_tool
from sqlalchemy import select


def seed(title, event_id):
    return {
        "id": event_id,
        "etag": '"seed"',
        "summary": title,
        "status": "confirmed",
        "start": {"dateTime": "2026-09-18T09:00:00-05:00", "timeZone": "America/Chicago"},
        "end": {"dateTime": "2026-09-18T10:00:00-05:00", "timeZone": "America/Chicago"},
        "htmlLink": "https://calendar.google.com/calendar/event?eid=fixture",
        "location": "Main office",
        "description": "Review the proposed milestones.",
        "hangoutLink": "https://meet.google.com/fixture",
        "attendees": [
            {"displayName": "Fixture guest", "email": "guest@example.test", "responseStatus": "accepted"}
        ],
        "organizer": {"email": "owner@example.test", "self": True},
        "attachments": [{"title": "Agenda notes", "fileUrl": "https://docs.google.com/document/d/fixture"}],
    }


class GoogleFixture:
    events: ClassVar[dict] = {
        "owner@example.test": {"dentist": seed("Dentist", "dentist")},
        "work@example.test": {"work-meeting": seed("Project review", "work-meeting")},
    }
    version = 0

    def __init__(self, *_):
        pass

    def request(self, path, params=None, body=None, *, method=None, headers=None):
        method = method or ("POST" if body is not None else "GET")
        params = params or {}
        if path == "users/me/calendarList":
            return {
                "items": [
                    {
                        "id": "owner@example.test",
                        "summary": "Personal",
                        "primary": True,
                        "timeZone": "America/Chicago",
                        "accessRole": "owner",
                    },
                    {
                        "id": "work@example.test",
                        "summary": "Work calendar",
                        "timeZone": "America/Chicago",
                        "accessRole": "writer",
                    },
                ]
            }
        if path.startswith("users/me/calendarList/"):
            return {"accessRole": "owner"}
        if path == "freeBusy":
            return {
                "calendars": {
                    item["id"]: {
                        "busy": [{"start": "2026-09-18T09:00:00-05:00", "end": "2026-09-18T10:00:00-05:00"}]
                    }
                    for item in body["items"]
                }
            }
        parts = path.split("/")
        calendar = unquote(parts[1])
        events = self.events[calendar]
        if path.endswith("/instances"):
            parent = events[unquote(parts[-2])]
            original = params["originalStart"]
            child_id = parent["id"] + "instance"
            child = {
                **copy.deepcopy(parent),
                "id": child_id,
                "recurringEventId": parent["id"],
                "originalStartTime": ({"date": original} if len(original) == 10 else {"dateTime": original}),
            }
            child.pop("recurrence", None)
            events.setdefault(child_id, child)
            return {"items": [copy.deepcopy(events[child_id])]}
        if method == "GET":
            if len(parts) == 3:
                return {"items": copy.deepcopy(list(events.values())), "nextSyncToken": "fixture-cursor"}
            eid = unquote(parts[-1])
            if eid not in events:
                raise google_calendar.SyncFailure("missing", 404)
            return copy.deepcopy(events[eid])
        if method == "POST":
            eid = body["id"]
            if eid in events:
                raise google_calendar.SyncFailure("duplicate", 409)
            events[eid] = copy.deepcopy(body)
        else:
            eid = unquote(parts[-1])
            if headers.get("If-Match") != events[eid].get("etag"):
                raise google_calendar.SyncFailure("conflict", 412)
            if method == "DELETE":
                events[eid]["status"] = "cancelled"
                return {}
            events[eid].update(copy.deepcopy(body))
        GoogleFixture.version += 1
        events[eid]["etag"] = '"fixture-' + str(self.version) + '"'
        events[eid]["htmlLink"] = "https://calendar.google.com/calendar/event?eid=fixture"
        events[eid]["status"] = "confirmed"
        return copy.deepcopy(events[eid])

    def close(self):
        pass


google_calendar.CalendarClient = GoogleFixture
google_auth.exchange = lambda purpose, state, verifier, code: {
    "id_token": code,
    "refresh_token": "fixture-refresh",
    "scope": "openid "
    + google_auth.CALENDAR_SCOPE
    + (" " + google_auth.WRITE_SCOPE if purpose == "calendar_write" else ""),
}
google_auth.verify_identity = json.loads
google_auth.httpx.post = lambda *args, **kwargs: type("Result", (), {"status_code": 200})()


@app.middleware("http")
async def fixture_worker(request, call_next):
    response = await call_next(request)
    if request.method == "POST" and request.url.path == "/api/v1/commands":
        with session_scope() as db:
            ids = list(db.scalars(select(Job.id).where(Job.kind == "google_write", Job.status == "queued")))
        for job in ids:
            await asyncio.to_thread(process_write, job)
        with session_scope() as db:
            rows = list(
                db.execute(
                    select(Job.id, Job.kind).where(
                        Job.kind.in_(["linear_sync", "linear_write"]), Job.status == "queued"
                    )
                )
            )
        for jid, kind in rows:
            await asyncio.to_thread(
                linear_sync.process_sync if kind == "linear_sync" else linear_sync.process_write, jid
            )
    return response


@app.post("/api/v1/__test_ui")
async def ui_fixture(body: dict, user: User):
    return await call_tool(
        user.owner_id, str(uuid4()), 0, body["name"], body["arguments"], device=user.device_id
    )


@app.post("/api/v1/__test_google_sync")
def sync_fixture(user: User):
    with session_scope() as db:
        job_id = google_calendar.queue_sync(db, user.owner_id, force=True)
    if job_id:
        google_calendar.process(job_id)
    return {"done": True}


@app.post("/api/v1/__test_google_dense")
def dense_fixture(user: User):
    for i in range(22):
        event = seed("Agenda item " + str(i + 1), "dense" + str(i))
        point = datetime.fromisoformat("2026-09-18T10:00:00-05:00") + timedelta(minutes=i * 20)
        event["start"]["dateTime"] = point.isoformat()
        event["end"]["dateTime"] = (point + timedelta(minutes=15)).isoformat()
        GoogleFixture.events["owner@example.test"][event["id"]] = event
    return sync_fixture(user)


@app.post("/api/v1/__test_linear_change")
def linear_change(user: User):
    linear_fixture.rows["remote"]["title"] = "Linear conflict version"
    linear_fixture.rows["remote"]["updatedAt"] = "2026-09-12T10:30:00Z"
    return {"done": True}
