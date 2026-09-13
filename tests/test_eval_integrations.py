"""Calendar evaluation adapters retain the real application contracts."""

import asyncio
import gzip
import json
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import expert_eval_cases as cases
from jarvis import tools
from jarvis.db import session_scope
from jarvis.models import Job
from sqlalchemy import select


async def invoke(fixture, name, arguments, index=0):
    return await cases.invoke_tool(
        fixture,
        tools.call_tool,
        cases.OWNER,
        "calendar-fidelity",
        index,
        name,
        arguments,
        device=cases.DEVICE,
        conversation_id=fixture["conversation"],
    )


@pytest.mark.asyncio
async def test_calendar_list_and_freebusy_share_connected_calendar_and_busy_intervals():
    fixture = cases.seed_case("constraint_schedule", 1)
    listed = await invoke(
        fixture,
        "calendar_list",
        {
            "start": "2030-01-15",
            "end": "2030-01-16",
            "timezone": "America/Chicago",
        },
    )
    availability = await invoke(
        fixture,
        "calendar_availability",
        {
            "start": "2030-01-15T09:00:00-06:00",
            "end": "2030-01-15T16:00:00-06:00",
            "minutes": 5,
        },
        1,
    )
    assert listed["google"]["configured"] and listed["google"]["linked"]
    assert listed["google"]["status"] == "connected"
    assert listed["google"]["calendars"][0]["id"] == fixture["remote_calendar"]
    assert availability["source"] == "google_freebusy_and_eridani"
    assert availability["calendar_count"] == 1
    begin = datetime.fromisoformat(availability["start"])
    until = datetime.fromisoformat(availability["end"])
    projected = sorted(
        (
            max(begin, datetime.fromisoformat(row["busy_start"])),
            min(until, datetime.fromisoformat(row["end_at"])),
        )
        for row in listed["items"]
        if row["kind"] == "google"
        and row["busy"]
        and datetime.fromisoformat(row["busy_start"]) < until
        and datetime.fromisoformat(row["end_at"]) > begin
    )
    confirmed = sorted(
        (datetime.fromisoformat(row["start"]), datetime.fromisoformat(row["end"]))
        for row in availability["busy"]
    )
    assert projected == confirmed
    assert len(confirmed) == 3


@pytest.mark.asyncio
async def test_connected_unknown_availability_does_not_become_local_only():
    fixture = cases.seed_case("calendar_unknown", 1)
    listed = await invoke(fixture, "calendar_list", {"start": "2030-01-15", "end": "2030-01-16"})
    availability = await invoke(
        fixture,
        "calendar_availability",
        {
            "start": "2030-01-15T10:00:00-06:00",
            "end": "2030-01-15T11:00:00-06:00",
            "minutes": 60,
        },
        1,
    )
    assert listed["google"]["linked"] and listed["google"]["status"] == "connected"
    assert availability["status"] == "unavailable"
    assert availability["free"] == []
    assert availability.get("source") != "eridani_only"


@pytest.mark.asyncio
async def test_freebusy_includes_real_local_blocks_added_during_conversation():
    fixture = cases.seed_case("constraint_schedule", 1)
    cases.command(
        "planning.create",
        title="Work C",
        kind="block",
        task_id=fixture["work"]["C"],
        start="2030-01-15T09:00:00-06:00",
        end="2030-01-15T09:30:00-06:00",
        timezone="America/Chicago",
    )
    result = await invoke(
        fixture,
        "calendar_availability",
        {
            "start": "2030-01-15T09:00:00-06:00",
            "end": "2030-01-15T10:15:00-06:00",
            "minutes": 5,
        },
    )
    first = datetime.fromisoformat(result["free"][0]["start"]).astimezone(ZoneInfo("America/Chicago"))
    assert (first.hour, first.minute) == (9, 30)


@pytest.mark.asyncio
async def test_queued_create_uses_real_internal_polling_and_separate_explicit_read(monkeypatch):
    fixture = cases.seed_case("pending_remote", 1)
    sleeps = []

    async def immediate(delay):
        sleeps.append(delay)

    monkeypatch.setattr(asyncio, "sleep", immediate)
    result = await invoke(
        fixture,
        "calendar_create",
        {
            "calendar_id": fixture["remote_calendar"],
            "title": "Client review",
            "start": "2030-01-15T10:00:00-06:00",
            "end": "2030-01-15T11:00:00-06:00",
            "timezone": "America/Chicago",
        },
    )
    assert result["status"] == "retrying"
    assert result["retry_active"] and result["assistant_followup_scheduled"] is False
    assert sleeps == [0.5] * 12
    internal = [row for row in fixture["integration_calls"] if row["kind"] == "internal_status_poll"]
    assert len(internal) == 12
    assert all(row["job_id"] == result["job_id"] for row in internal)
    explicit = await invoke(fixture, "calendar_write_status", {"job_id": result["job_id"]}, 1)
    assert explicit["status"] == "retrying"
    assert len([row for row in fixture["integration_calls"] if row["kind"] == "explicit_status_read"]) == 1
    assert len([row for row in fixture["tools"] if row["name"] == "calendar_create"]) == 1
    with session_scope() as db:
        assert len(list(db.scalars(select(Job).where(Job.kind == "google_write")))) == 1


@pytest.mark.parametrize(
    "name", ["constraint_schedule", "impossible_schedule", "calendar_unknown", "pending_remote"]
)
def test_calendar_adapter_keeps_original_tracked_seed_hash(name):
    path = Path(__file__).resolve().parents[1] / "docs/evals/expert-fixtures-2026-09-13.json.gz"
    with gzip.open(path, "rt") as file:
        baseline = json.load(file)["fixtures"]
    original = next(row for row in baseline if row["case"] == name and row["repeat"] == 1)
    fixture = cases.seed_case(name, 1)
    assert fixture["fixture_hash"] == original["fixture_hash"]
    assert fixture["prompts"] == original["prompts"]
