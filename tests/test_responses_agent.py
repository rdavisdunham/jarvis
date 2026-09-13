import asyncio
import copy
import json
from unittest.mock import AsyncMock
from uuid import uuid4

import httpx
import pytest
from jarvis import agent_models, conversation
from jarvis.config import get_settings
from jarvis.db import session_scope
from jarvis.domain import DomainError, execute, preferences
from jarvis.models import BudgetReservation, Conversation, Job, OwnerSettings, Task, Usage
from sqlalchemy import select


def choose(profile):
    with session_scope() as db:
        return execute(db, "davin", str(uuid4()), "settings.update", {"agent_profile": profile})


def packet(*output, status="completed"):
    return {
        "id": "resp_" + uuid4().hex,
        "status": status,
        "output": list(output),
        "usage": {
            "input_tokens": 100,
            "output_tokens": 30,
            "input_tokens_details": {"cached_tokens": 40, "cache_write_tokens": 20},
            "output_tokens_details": {"reasoning_tokens": 20},
        },
    }


def thought(label):
    return {"type": "reasoning", "id": "rs_" + label, "summary": [], "encrypted_content": "opaque-" + label}


def function(name, args, label):
    return {
        "type": "function_call",
        "id": "fc_" + label,
        "call_id": "call_" + label,
        "name": name,
        "arguments": json.dumps(args),
        "status": "completed",
    }


def answer(text):
    return {
        "type": "message",
        "role": "assistant",
        "id": "msg_" + uuid4().hex,
        "status": "completed",
        "content": [{"type": "output_text", "text": text, "annotations": []}],
    }


def setup(monkeypatch, packets):
    monkeypatch.setattr(get_settings(), "openai_api_key", "synthetic-openai")
    monkeypatch.setattr(conversation, "prompt_context", AsyncMock(return_value="Synthetic memory"))
    choose("luna")
    sent = []

    class Client:
        def __init__(self, **_):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_):
            pass

        async def post(self, url, *, json, headers):
            sent.append({"url": url, "body": copy.deepcopy(json), "headers": headers})
            data = packets.pop(0)
            if isinstance(data, BaseException):
                raise data
            return httpx.Response(200, request=httpx.Request("POST", url), json=data)

    monkeypatch.setattr(conversation.httpx, "AsyncClient", Client)
    with session_scope() as db:
        conv = Conversation(owner_id="davin", device_id="test", private=True, learning=False)
        db.add(conv)
        db.flush()
        return conv.id, sent


async def test_reasoning_parallel_calls_and_continuations_preserve_native_items(monkeypatch):
    first = packet(
        thought("a"),
        function("task_create", {"title": "One"}, "a"),
        function("task_create", {"title": "Two"}, "b"),
    )
    second = packet(thought("b"), function("task_list", {}, "c"))
    cid, sent = setup(monkeypatch, [first, second, packet(thought("c"), answer("Saved both."))])
    turn = str(uuid4())
    result = await conversation.chat("davin", "test", turn, cid, "Create two tasks and verify.")
    assert result["status"] == "succeeded" and len(result["actions"]) == 2
    assert (
        result["model"] == "gpt-5.6-luna" and result["provider"] == "openai" and result["profile"] == "luna"
    )
    assert all(s["url"] == "https://api.openai.com/v1/responses" for s in sent)
    body = sent[0]["body"]
    assert body["reasoning"] == {"effort": "low"}
    assert body["store"] is False and body["max_output_tokens"] == 8192
    assert "messages" not in body and "previous_response_id" not in body
    assert body["tools"][0]["type"] == "function" and "function" not in body["tools"][0]
    tool_schemas = {t["name"]: t for t in body["tools"]}
    assert tool_schemas["task_get"]["strict"] is True
    assert tool_schemas["task_update"]["strict"] is False
    history = sent[-1]["body"]["input"]
    assert history[2:5] == first["output"]
    assert history[7:9] == second["output"]
    assert [i["call_id"] for i in history if i.get("type") == "function_call_output"] == [
        "call_a",
        "call_b",
        "call_c",
    ]
    assert len([i for i in history if i.get("type") == "function_call"]) == 3
    with session_scope() as db:
        assert len(list(db.scalars(select(Task)))) == 2
        job = db.get(Job, turn)
        assert "opaque-" not in json.dumps(job.result) and "rs_" not in json.dumps(job.result)
        assert job.payload["profile"] == "luna"
        assert all(u.model == "gpt-5.6-luna" for u in db.scalars(select(Usage)))
    repeated = await conversation.chat("davin", "test", turn, cid, "Create two tasks and verify.")
    assert repeated["actions"] == result["actions"] and len(sent) == 3


async def test_incomplete_response_does_not_execute_partial_calls(monkeypatch):
    cid, sent = setup(
        monkeypatch,
        [
            packet(
                thought("limit"),
                function("task_create", {"title": "Do not execute"}, "bad"),
                answer("I ran out of room before saving the task."),
                status="incomplete",
            )
        ],
    )
    result = await conversation.chat("davin", "test", str(uuid4()), cid, "Create a task")
    assert result["status"] == "partial" and not result["actions"]
    with session_scope() as db:
        assert list(db.scalars(select(Task))) == []
    assert len(sent) == 1


async def test_reasoning_action_limit_and_disabled_costs(monkeypatch):
    cid, sent = setup(
        monkeypatch,
        [
            packet(
                thought("a"),
                function("task_create", {"title": "One"}, "a"),
                function("task_create", {"title": "Two"}, "b"),
            ),
            packet(answer("One saved, one unfinished.")),
        ],
    )
    monkeypatch.setattr(get_settings(), "max_tool_calls_per_request", 1)
    monkeypatch.setattr(get_settings(), "cost_tracking_enabled", False)
    result = await conversation.chat("davin", "test", str(uuid4()), cid, "Save two")
    assert result["status"] == "partial" and len(result["actions"]) == 1
    assert sent[-1]["body"]["tool_choice"] == "none"
    with session_scope() as db:
        assert not list(db.scalars(select(Usage))) and not list(db.scalars(select(BudgetReservation)))


@pytest.mark.parametrize("failure", [httpx.ReadTimeout("synthetic"), asyncio.CancelledError()])
async def test_failed_continuation_never_repeats_saved_actions(monkeypatch, failure):
    cid, sent = setup(
        monkeypatch,
        [
            packet(thought("a"), function("task_create", {"title": "Saved once"}, "a")),
            failure,
        ],
    )
    turn = str(uuid4())
    if isinstance(failure, asyncio.CancelledError):
        with pytest.raises(asyncio.CancelledError):
            await conversation.chat("davin", "test", turn, cid, "Save task")
    else:
        result = await conversation.chat("davin", "test", turn, cid, "Save task")
        assert result["status"] == "failed"
    with session_scope() as db:
        assert len(list(db.scalars(select(Task)))) == 1
        assert db.get(BudgetReservation, turn).state == "uncertain"
    assert len(sent) == 2


def test_luna_model_profile_and_legacy_preferences(monkeypatch, client):
    monkeypatch.setattr(get_settings(), "openai_api_key", "synthetic-openai")
    with session_scope() as db:
        db.add(
            OwnerSettings(
                owner_id="davin", values={"agent_provider": "openai", "preferred_name": "Keep name"}
            )
        )
    choose("luna")
    boot = client.get("/api/v1/bootstrap").json()
    assert boot["agent_profile"] == "luna" and boot["agent_provider"] == "openai"
    assert boot["agent_reasoning"] == "low"
    assert boot["preferences"]["preferred_name"] == "Keep name"
    assert len({m["id"] for m in boot["agent_options"]}) == len(boot["agent_options"])
    with session_scope() as db:
        # An older client's provider-only edit replaces the model profile too.
        execute(db, "davin", str(uuid4()), "settings.update", {"agent_provider": "openai"})
        assert preferences(db, "davin")["agent_profile"] == "luna"
    monkeypatch.setattr(get_settings(), "openai_api_key", "")
    with pytest.raises(DomainError, match="OPENAI_API_KEY"):
        choose("luna")


def test_luna_cache_and_reasoning_accounting():
    agent = agent_models.catalog()["luna"]
    usage = agent.normalize(packet(answer("Done")))["usage"]
    # 40 ordinary, 40 cached and 20 cache-write input tokens; 30 total output
    # includes 20 reasoning tokens already, so do not count those twice.
    assert agent.usage_cost(usage) == pytest.approx((40 * 0.2 + 40 * 0.02 + 20 * 0.25 + 30 * 1.2) / 1e6)
    assert agent.reserve_cost(100) == pytest.approx((100 * 0.25 + 8192 * 1.2) / 1e6)
