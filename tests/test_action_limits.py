import asyncio
import json
from unittest.mock import AsyncMock
from uuid import uuid4

import httpx
import pytest
from jarvis import conversation
from jarvis.config import get_settings
from jarvis.db import session_scope
from jarvis.domain import execute
from jarvis.models import BudgetReservation, Conversation, Job, Task
from jarvis.tools import call_tool, registry
from sqlalchemy import select


def task(title):
    with session_scope() as db:
        return execute(db, "davin", str(uuid4()), "task.create", {"title": title})["data"]


def setup(monkeypatch, responses):
    monkeypatch.setattr(get_settings(), "openai_api_key", "synthetic-test-key")
    monkeypatch.setattr(conversation, "prompt_context", AsyncMock(return_value=""))
    sent = []

    class Client:
        def __init__(self, **_):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_):
            pass

        async def post(self, url, *, json, **_):
            sent.append(json)
            msg = responses.pop(0)
            if isinstance(msg, BaseException):
                raise msg
            return httpx.Response(
                200,
                request=httpx.Request("POST", url),
                json={
                    "id": str(uuid4()),
                    "usage": {"prompt_tokens": 100, "completion_tokens": 20},
                    "choices": [{"message": msg}],
                },
            )

    monkeypatch.setattr(conversation.httpx, "AsyncClient", Client)
    with session_scope() as db:
        conv = Conversation(owner_id="davin", device_id="test", private=True, learning=False)
        db.add(conv)
        db.flush()
        return conv.id, sent


def call(name, args):
    return {"id": str(uuid4()), "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}


def response(*calls):
    return {"role": "assistant", "content": None, "tool_calls": list(calls)}


async def test_bulk_edits_survive_more_than_four_calls_and_five_rounds(monkeypatch):
    tasks = [task(str(i)) for i in range(8)]
    responses = [response(call("task_list", {}))]
    responses += [
        response(
            call(
                "task_update", {"task_id": t["id"], "expected_revision": 1, "title": "Updated " + t["title"]}
            )
        )
        for t in tasks
    ]
    responses += [{"role": "assistant", "content": "Updated all eight tasks."}]
    cid, sent = setup(monkeypatch, responses)
    result = await conversation.chat("davin", "test", str(uuid4()), cid, "Update all eight tasks")
    assert result["status"] == "succeeded" and len(result["actions"]) == 8
    assert result["tool_calls"] == 9 and len(sent) == 10
    with session_scope() as db:
        assert all(t.title.startswith("Updated") for t in db.scalars(select(Task)))


async def test_limit_retains_saved_results_and_suppresses_further_mutations(monkeypatch):
    monkeypatch.setattr(get_settings(), "max_tool_calls_per_request", 2)
    calls = [call("task_create", {"title": str(i)}) for i in range(3)]
    cid, sent = setup(
        monkeypatch, [response(*calls), {"role": "assistant", "content": "Saved two; one is unfinished."}]
    )
    result = await conversation.chat("davin", "test", str(uuid4()), cid, "Create three tasks")
    assert result["status"] == "partial" and len(result["actions"]) == 2
    assert result["tool_errors"][0]["error"] == "LIMIT_EXCEEDED"
    assert sent[-1]["tool_choice"] == "none"
    with session_scope() as db:
        assert len(list(db.scalars(select(Task)))) == 2


async def test_spoken_correction_blocks_remaining_calls_from_the_same_old_plan(monkeypatch):
    cid, _ = setup(
        monkeypatch,
        [
            response(call("task_create", {"title": "Stale A"}), call("task_create", {"title": "Stale B"})),
            {"role": "assistant", "content": "I stopped to consider your correction."},
        ],
    )
    guard = iter(["The user changed the request.", None])
    result = await conversation.chat(
        "davin", "test", str(uuid4()), cid, "Create things", tool_guard=lambda: next(guard)
    )
    assert not result["actions"]
    with session_scope() as db:
        assert list(db.scalars(select(Task))) == []


async def test_cancelled_request_finalizes_local_job_without_replaying_actions(monkeypatch):
    cid, _ = setup(monkeypatch, [response(call("task_create", {"title": "Saved"})), asyncio.CancelledError()])
    turn = str(uuid4())
    with pytest.raises(asyncio.CancelledError):
        await conversation.chat("davin", "test", turn, cid, "Save and summarize")
    with session_scope() as db:
        assert len(list(db.scalars(select(Task)))) == 1
        assert db.get(Job, turn).status == "cancelled"
        assert db.get(BudgetReservation, turn).state == "uncertain"


async def test_existing_memory_and_notification_actions_are_in_shared_registry():
    names = {t["name"] for t in registry()}
    assert {
        "memory_correct",
        "memory_forget",
        "notification_snooze",
        "notification_read",
        "notification_dismiss",
    } <= names
    with session_scope() as db:
        memory = execute(db, "davin", "remember", "memory.capture", {"content": "I like tea."})["data"]
    corrected = await call_tool(
        "davin",
        "correct",
        0,
        "memory_correct",
        {"memory_id": memory["id"], "content": "I like coffee."},
        device="test",
    )
    assert corrected["status"] == "succeeded"


async def test_task_list_paginates_instead_of_silently_truncating():
    for i in range(35):
        task(str(i))
    first = await call_tool("davin", "read", 0, "task_list", {})
    second = await call_tool("davin", "read", 1, "task_list", {"offset": first["next_offset"]})
    assert len(first["tasks"]) == 30 and len(second["tasks"]) == 5
    assert second["next_offset"] is None
    assert len({t["id"] for t in first["tasks"] + second["tasks"]}) == 35


async def test_cancel_during_initial_memory_lookup_releases_chat_reservation(monkeypatch):
    cid, sent = setup(monkeypatch, [])
    monkeypatch.setattr(conversation, "prompt_context", AsyncMock(side_effect=asyncio.CancelledError()))
    turn = str(uuid4())
    with pytest.raises(asyncio.CancelledError):
        await conversation.chat("davin", "test", turn, cid, "Look something up")
    with session_scope() as db:
        assert db.get(Job, turn).status == "cancelled"
        assert db.get(BudgetReservation, turn).state == "closed"
    assert not sent
