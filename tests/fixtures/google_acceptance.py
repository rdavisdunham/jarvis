"""Synthetic Google transport for isolated browser acceptance. Never run on the owner database."""

import json
from uuid import uuid4

from jarvis import google_auth, google_calendar
from jarvis.api import User, app
from jarvis.db import session_scope
from jarvis.tools import call_tool


class GoogleFixture:
    def __init__(self, *_):
        pass

    def request(self, path, params=None, body=None):
        if path == "users/me/calendarList":
            return {
                "items": [
                    {
                        "id": "owner@example.test",
                        "summary": "Personal",
                        "primary": True,
                        "timeZone": "America/Chicago",
                    },
                    {"id": "work@example.test", "summary": "Work calendar", "timeZone": "America/Chicago"},
                ]
            }
        if path.endswith("/events"):
            work = "work%40" in path
            return {
                "items": [
                    {
                        "id": "work-meeting" if work else "dentist",
                        "summary": "Project review" if work else "Dentist",
                        "start": {"dateTime": "2026-09-18T09:00:00-05:00", "timeZone": "America/Chicago"},
                        "end": {"dateTime": "2026-09-18T10:00:00-05:00", "timeZone": "America/Chicago"},
                        "htmlLink": "https://calendar.google.com/calendar/event?eid=fixture",
                        "location": "Main office",
                    }
                ],
                "nextSyncToken": "fixture-cursor",
            }
        if path == "freeBusy":
            return {
                "calendars": {
                    item["id"]: {
                        "busy": [
                            {
                                "start": "2026-09-18T09:00:00-05:00",
                                "end": "2026-09-18T10:00:00-05:00",
                            }
                        ]
                    }
                    for item in body["items"]
                }
            }
        raise AssertionError("Unexpected fixture path")

    def close(self):
        pass


google_calendar.CalendarClient = GoogleFixture
google_auth.exchange = lambda purpose, state, verifier, code: {
    "id_token": code,
    "refresh_token": "fixture-refresh",
    "scope": "openid " + google_auth.CALENDAR_SCOPE,
}
google_auth.verify_identity = json.loads
google_auth.httpx.post = lambda *args, **kwargs: type("Result", (), {"status_code": 200})()


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
