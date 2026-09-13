"""Synthetic Google transport over real calendar/domain code for local evals.

Only fixture-owned credentials/events/jobs are used. Production connection,
projection, availability merging, queue creation, and synchronous status polling
remain active; the provider client itself is replaced before it can perform I/O.
"""

import copy
from contextlib import ExitStack, contextmanager
from datetime import datetime
from unittest.mock import patch
from uuid import NAMESPACE_URL, uuid5

from jarvis import google_calendar, google_writes
from jarvis.db import session_scope
from jarvis.domain import DomainError
from jarvis.models import GoogleCalendar, GoogleCalendarEvent, GoogleIdentity, Job


def seed_calendar(fixture, owner, clock, *, coverage_start, coverage_end, busy, unavailable=False):
    calendar_id = fixture.setdefault(
        "remote_calendar", str(uuid5(NAMESPACE_URL, "expert-synthetic-calendar"))
    )
    provider_id = "fixture-calendar@example.invalid"
    state = {
        "calendar_id": calendar_id,
        "provider_id": provider_id,
        "coverage_start": coverage_start,
        "coverage_end": coverage_end,
        "busy": copy.deepcopy(busy),
        "unavailable": unavailable,
        "status": "connected",
        "calendar_count": 1,
    }
    fixture["calendar_fixture"] = state
    fixture["integration_calls"] = []
    with session_scope() as db:
        db.add(
            GoogleIdentity(
                owner_id=owner,
                subject="fixture-account",
                email="rowan@example.invalid",
                credentials="synthetic-no-provider-credential",
                calendar_enabled=True,
                calendar_write_enabled=True,
                status="connected",
                last_sync_at=clock,
                linked_at=clock,
            )
        )
        db.flush()
        db.add(
            GoogleCalendar(
                id=calendar_id,
                owner_id=owner,
                provider_id=provider_id,
                title="Work",
                timezone="America/Chicago",
                selected=True,
                available=True,
                primary=True,
                access_role="owner",
                last_sync_at=clock,
            )
        )
        db.flush()
        for index, block in enumerate(busy):
            provider_event = f"fixture-busy-{index}"
            db.add(
                GoogleCalendarEvent(
                    id=str(uuid5(NAMESPACE_URL, f"{fixture['case']}:{fixture['repeat']}:{provider_event}")),
                    calendar_id=calendar_id,
                    provider_id=provider_event,
                    payload={
                        "id": provider_event,
                        "summary": f"Reserved time {index + 1}",
                        "status": "confirmed",
                        "start": {"dateTime": block["start"], "timeZone": "America/Chicago"},
                        "end": {"dateTime": block["end"], "timeZone": "America/Chicago"},
                        "transparency": "opaque",
                    },
                )
            )
    return state


@contextmanager
def calendar_transport(fixture, owner, tool_name):
    state = fixture.get("calendar_fixture")
    if not state:
        yield
        return
    traces = fixture.setdefault("integration_calls", [])
    original_queue = google_writes.queue_write
    original_status = google_writes.write_status

    class FixtureClient:
        def __init__(self, credentials):
            if credentials != {"fixture": True}:
                raise AssertionError("Only synthetic credentials belong in this transport")

        def request(self, path, params=None, *, body=None, **kwargs):
            trace = {"kind": "provider_fixture", "tool": tool_name, "path": path, "body": copy.deepcopy(body)}
            traces.append(trace)
            if path != "freeBusy" or kwargs or params:
                trace["blocked"] = True
                raise DomainError("EVAL_BLOCKED", "Only synthetic freebusy transport is available.")
            identifiers = [row["id"] for row in (body or {}).get("items", [])]
            begin = datetime.fromisoformat(body["timeMin"])
            until = datetime.fromisoformat(body["timeMax"])
            if identifiers != [state["provider_id"]]:
                raise AssertionError("Availability must consult the seeded selected calendar")
            if (
                state["unavailable"]
                or begin < datetime.fromisoformat(state["coverage_start"])
                or until > datetime.fromisoformat(state["coverage_end"])
            ):
                trace["error"] = "synthetic_unavailable"
                raise google_calendar.SyncFailure("synthetic_unavailable")
            result = {"calendars": {state["provider_id"]: {"busy": copy.deepcopy(state["busy"])}}}
            trace["outcome"] = copy.deepcopy(result)
            return result

        def close(self):
            pass

    def queue(db, requested_owner, operation, arguments):
        if requested_owner != owner or operation != "calendar.create" or fixture["case"] != "pending_remote":
            raise DomainError("EVAL_BLOCKED", "Only the declared synthetic queued create is allowed.")
        result = original_queue(db, requested_owner, operation, arguments)
        fixture["remote_job"] = result["job_id"]
        traces.append({"kind": "queue_receipt", "tool": tool_name, "outcome": copy.deepcopy(result)})
        return result

    def status(db, requested_owner, job_id):
        job = db.get(Job, job_id)
        if not job or requested_owner != owner or job.owner_id != owner or job.kind != "google_write":
            return original_status(db, requested_owner, job_id)
        # Simulate the worker's attempted remote write; no worker or external write runs.
        job.status = "retrying"
        job.result = {"attempts": 1, "message": "Synthetic Google request is awaiting server retry."}
        db.flush()
        result = original_status(db, requested_owner, job_id)
        traces.append(
            {
                "kind": "internal_status_poll" if tool_name == "calendar_create" else "explicit_status_read",
                "tool": tool_name,
                "job_id": job_id,
                "outcome": copy.deepcopy(result),
            }
        )
        return result

    with ExitStack() as stack:
        stack.enter_context(patch.object(google_calendar, "configured", return_value=True))
        stack.enter_context(patch.object(google_writes, "configured", return_value=True))
        stack.enter_context(patch.object(google_calendar, "unseal", return_value={"fixture": True}))
        stack.enter_context(patch.object(google_calendar, "CalendarClient", FixtureClient))
        stack.enter_context(patch.object(google_writes, "queue_write", queue))
        stack.enter_context(patch.object(google_writes, "write_status", status))
        yield
