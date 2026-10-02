import json
from datetime import timedelta
from uuid import uuid4

import httpx
import pytest
from cryptography.fernet import Fernet
from jarvis import work_runner, worker
from jarvis.config import get_settings
from jarvis.db import engine, session_scope
from jarvis.domain import DomainError, execute
from jarvis.models import (
    AgentWork,
    BudgetReservation,
    Delivery,
    Job,
    Notification,
    Outbox,
    PushSubscription,
    Usage,
    now,
)
from jarvis.structure_models import StructureRecord
from sqlalchemy import select, text


@pytest.fixture(autouse=True)
def settings(monkeypatch):
    monkeypatch.setenv("JARVIS_INTEGRATION_ENCRYPTION_KEY", Fernet.generate_key().decode())
    monkeypatch.setenv("JARVIS_OPENAI_API_KEY", "synthetic")
    monkeypatch.setenv("JARVIS_GEMINI_API_KEY", "synthetic")
    monkeypatch.setenv("JARVIS_EXTERNAL_SERVICES_ENABLED", "true")
    monkeypatch.setenv("JARVIS_COST_TRACKING_ENABLED", "true")
    monkeypatch.setenv("JARVIS_BUDGET_ENFORCEMENT_ENABLED", "false")
    get_settings.cache_clear()

    async def memories(*args):
        return ""

    monkeypatch.setattr(work_runner, "prompt_context", memories)
    yield
    get_settings.cache_clear()


def action(client, message="Add Alpha"):
    conv = client.post("/api/v1/conversations", json={}).json()
    payload = {"turn_id": str(uuid4()), "conversation_id": conv["id"], "message": message}
    result = client.post("/api/v1/work", json=payload)
    assert result.status_code == 200, result.text
    work = result.json()
    with session_scope() as db:
        db.get(Job, work["id"]).kind = "agent_action"
    return work


def response(calls=None, message=None, finish=None):
    choice = {
        "message": {
            "role": "assistant",
            "content": message,
            "tool_calls": [
                {"id": str(uuid4()), "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}
                for name, args in calls or []
            ],
        }
    }
    if finish:
        choice["finish_reason"] = finish
    return {"id": str(uuid4()), "usage": {"prompt_tokens": 10, "completion_tokens": 10}, "choices": [choice]}


def responder(monkeypatch, responses, seen=None):
    async def model(agent, messages, *args, **kwargs):
        assert responses, "Unexpected provider call"
        if seen is not None:
            seen.append([dict(m) for m in messages])
        item = responses.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item() if callable(item) else item

    monkeypatch.setattr(work_runner, "request_model", model)


def status(identity):
    with session_scope() as db:
        return db.get(Job, identity).status


@pytest.mark.asyncio
async def test_revised_finished_request_runs_again_with_its_own_budget_session(client, monkeypatch):
    work = action(client)
    responder(monkeypatch, [response([("task_create", {"title": "Alpha"})]), response(message="Added Alpha.")])
    await work_runner.run(work["id"])
    assert status(work["id"]) == "succeeded"
    revised = client.post(
        "/api/v1/work/" + work["id"] + "/revise", json={"message": "Also add Beta", "expected_revision": 1}
    )
    assert revised.status_code == 200, revised.text
    responder(monkeypatch, [response([("task_create", {"title": "Beta"})]), response(message="Added Beta.")])
    await work_runner.run(work["id"])
    with session_scope() as db:
        assert db.get(Job, work["id"]).status == "succeeded"
        usage = list(db.scalars(select(Usage)))
        assert len(usage) == 4
        assert {u.reservation_id for u in usage} == {work["id"], work["id"] + ":2"}
        assert {r.state for r in db.scalars(select(BudgetReservation))} == {"closed"}
    assert "Beta" in client.get("/api/v1/work/" + work["id"]).json()["message"]


@pytest.mark.asyncio
async def test_concurrent_invocation_of_same_request_exits_without_executing(client, monkeypatch):
    work = action(client)
    responder(monkeypatch, [])  # Any provider call fails the test.
    monkeypatch.setattr(work_runner.asyncio, "sleep", _no_sleep)
    with engine().connect() as holder:
        assert holder.scalar(
            text("SELECT pg_try_advisory_lock(hashtextextended(:k, 0))"), {"k": "work-run:" + work["id"]}
        )
        holder.commit()
        with session_scope() as db:
            db.get(Job, work["id"]).status = "running"
        await work_runner.run(work["id"])
        assert status(work["id"]) == "running"
        with session_scope() as db:
            db.get(Job, work["id"]).status = "dispatched"
        await work_runner.run(work["id"])
        with session_scope() as db:
            job = db.get(Job, work["id"])
            # The losing invocation hands off to a fresh durable dispatch instead of stranding the job.
            assert job.status == "queued" and job.payload["dispatch_revision"] >= 1
            assert db.get(Outbox, work["id"]).submitted_at is None
        holder.execute(text("SELECT pg_advisory_unlock_all()"))
        holder.commit()


async def _no_sleep(*args):
    return None


def test_dispatch_never_flips_running_work_back_to_dispatched(client):
    work = action(client)
    enqueued = []

    class Client:
        def enqueue(self, options, job_id):
            enqueued.append(options["workflow_id"])

    with session_scope() as db:
        db.get(Job, work["id"]).status = "running"
    worker.dispatch_outbox(Client())
    with session_scope() as db:
        assert db.get(Job, work["id"]).status == "running"
        assert db.get(Outbox, work["id"]).submitted_at is not None
    assert not [w for w in enqueued if w.startswith(work["id"])]


@pytest.mark.asyncio
async def test_retired_profile_and_unexpected_error_finish_instead_of_staying_running(client, monkeypatch):
    work = action(client)
    with session_scope() as db:
        job = db.get(Job, work["id"])
        job.payload = {**job.payload, "profile": "retired-profile"}

    async def broken(*args, **kwargs):
        raise RuntimeError("synthetic database failure")

    monkeypatch.setattr(work_runner, "call_tool", broken)
    responder(monkeypatch, [response([("task_create", {"title": "Alpha"})])])
    await work_runner.run(work["id"])
    with session_scope() as db:
        assert db.get(Job, work["id"]).status == "failed"
        assert db.get(BudgetReservation, work["id"]).state == "closed"


def test_reaper_finishes_stale_rows_but_not_live_invocations(client):
    stale, live = action(client), action(client, "Add Beta")
    with session_scope() as db:
        for identity in (stale["id"], live["id"]):
            db.get(Job, identity).status = "running"
            db.get(AgentWork, identity).updated_at = now() - timedelta(hours=2)
    with engine().connect() as holder:
        holder.scalar(text("SELECT pg_advisory_lock(hashtextextended(:k, 0))"), {"k": "work-run:" + live["id"]})
        holder.commit()
        worker.reap_agent_work()
        holder.execute(text("SELECT pg_advisory_unlock_all()"))
        holder.commit()
    assert status(stale["id"]) == "failed"
    assert status(live["id"]) == "running"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("failure", "expected"),
    [
        (httpx.ReadTimeout("synthetic timeout"), "uncertain"),
        (response(message="cut", finish="length"), "closed"),
    ],
)
async def test_only_unknown_provider_outcomes_keep_an_uncertain_hold(client, monkeypatch, failure, expected):
    work = action(client)
    responder(monkeypatch, [failure])
    await work_runner.run(work["id"])
    with session_scope() as db:
        assert db.get(Job, work["id"]).status == "failed"
        assert db.get(BudgetReservation, work["id"]).state == expected


@pytest.mark.asyncio
async def test_round_limit_closes_the_budget_session(client, monkeypatch):
    work = action(client)
    monkeypatch.setattr(get_settings(), "max_model_rounds_per_request", 0)
    responder(monkeypatch, [response([("tools_load", {"groups": ["tasks"]})])])
    await work_runner.run(work["id"])
    with session_scope() as db:
        assert db.get(Job, work["id"]).status == "failed"
        assert db.get(BudgetReservation, work["id"]).state == "closed"


@pytest.mark.asyncio
async def test_offer_after_one_nudge_stays_a_finished_answer(client, monkeypatch):
    # Regression (2026-10-02): "...want me to reschedule any?" was forced into a pending
    # question, which then swallowed the user's next request. One nudge, then trust the reply.
    work = action(client, "Summarize my overdue tasks")
    seen = []
    responder(
        monkeypatch,
        [
            response([("task_list", {})]),
            response(message="You have 3 overdue tasks. Want me to reschedule any?"),
            response(message="You have 3 overdue tasks. Want me to reschedule any of them?"),
        ],
        seen,
    )
    await work_runner.run(work["id"])
    assert seen[-1][-1] == {"role": "system", "content": work_runner.QUESTION_CHECK}
    card = client.get("/api/v1/work/" + work["id"]).json()
    assert card["status"] == "succeeded"
    assert not card.get("clarification")


@pytest.mark.asyncio
async def test_corrective_round_can_answer_without_question(client, monkeypatch):
    work = action(client, "What is on my list?")
    responder(
        monkeypatch,
        [
            response([("task_list", {})]),
            response(message="You have nothing due. Want me to add something?"),
            response(message="You have nothing due."),
        ],
    )
    await work_runner.run(work["id"])
    assert status(work["id"]) == "succeeded"


@pytest.mark.asyncio
async def test_plain_conversation_question_is_not_rechecked(client, monkeypatch):
    work = action(client, "Hello")
    responder(monkeypatch, [response(message="Hi! How can I help?")])
    await work_runner.run(work["id"])
    assert status(work["id"]) == "succeeded"


@pytest.mark.asyncio
async def test_failed_task_edit_recovered_through_backing_record(client, monkeypatch):
    owner = get_settings().owner_id
    with session_scope() as db:
        task = execute(db, owner, str(uuid4()), "task.create", {"title": "Report"})["data"]
        record = db.scalar(select(StructureRecord).where(StructureRecord.task_id == task["id"]))
        if record is None:
            record = StructureRecord(owner_id=owner, type_id="task", title="Report", task_id=task["id"])
            db.add(record)
            db.flush()
        record_id = record.id
    work = action(client, "Rename Report")

    async def tools(owner, turn, index, name, args, **kwargs):
        if name == "task_update":
            raise DomainError("INVALID_ARGUMENT", "synthetic task-path rejection")
        return {"command_id": f"{turn}:{index}", "status": "succeeded"}

    monkeypatch.setattr(work_runner, "call_tool", tools)
    monkeypatch.setattr(work_runner.work_coordination, "reserve", lambda *args: None)
    responder(
        monkeypatch,
        [
            response([("tools_load", {"groups": ["records"]})]),
            response([("task_update", {"task_id": task["id"], "expected_revision": 1, "title": "Q3 report"})]),
            response([("record_update", {"record_id": record_id, "expected_revision": 1, "schema_revision": 1, "title": "Q3 report"})]),
            response(message="Renamed it."),
        ],
    )
    await work_runner.run(work["id"])
    with session_scope() as db:
        row = db.get(AgentWork, work["id"])
        assert db.get(Job, work["id"]).status == "succeeded", row.result
        assert row.result["errors"] == [] and row.result["recovered_errors"] == 1


def test_push_success_on_final_attempt_is_not_recorded_failed(monkeypatch):
    monkeypatch.setattr(get_settings(), "vapid_private_key", "synthetic")
    monkeypatch.setattr(worker, "webpush", lambda **kwargs: None)
    monkeypatch.setattr("jarvis.notices.eligible", lambda db, n: True)
    owner = get_settings().owner_id
    with session_scope() as db:
        note = Notification(owner_id=owner, title="Due", scheduled_at=now(), category="reminder")
        sub = PushSubscription(
            id="sub", owner_id=owner, device_id=str(uuid4()),
            subscription={"endpoint": "https://fcm.googleapis.com/fcm/send/x"},
        )
        db.add_all([note, sub])
        db.flush()
        db.add(Delivery(notification_id=note.id, subscription_id=sub.id, attempts=4, status="retry"))
    worker.send_deliveries()
    with session_scope() as db:
        row = db.scalar(select(Delivery))
        assert row.attempts == 5 and row.status == "submitted"


def test_one_failing_scan_does_not_block_dispatch(client, monkeypatch):
    work = action(client)
    enqueued = []

    class Client:
        def enqueue(self, options, job_id):
            enqueued.append(job_id)

    def broken(db):
        raise RuntimeError("synthetic scan failure")

    monkeypatch.setattr(worker, "scan_schedules", broken)
    monkeypatch.setattr("jarvis.notices.scan", broken)
    worker.supervisor_cycle(Client(), 1)
    assert work["id"] in enqueued
