import copy
import json
from datetime import date
from unittest.mock import AsyncMock
from uuid import uuid4

import httpx
import pytest
from jarvis import agent_models, conversation
from jarvis.config import get_settings
from jarvis.db import session_scope
from jarvis.domain import DomainError, execute
from jarvis.models import BudgetReservation, Conversation, Job, Task, Usage
from sqlalchemy import select


@pytest.fixture
def keys(monkeypatch):
    settings = get_settings()
    monkeypatch.setattr(settings, "openai_api_key", "synthetic-openai-key")
    monkeypatch.setattr(settings, "groq_api_key", "")
    monkeypatch.setattr(settings, "gemini_api_key", "synthetic-gemini-key")
    monkeypatch.setattr(conversation, "prompt_context", AsyncMock(return_value="Remembered context"))
    return settings


def select_provider(provider):
    with session_scope() as db:
        return execute(db, "davin", str(uuid4()), "settings.update", {"agent_provider": provider})


def fake_provider(monkeypatch, responses):
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
            response = responses.pop(0)
            if callable(response):
                response = response()
            if isinstance(response, BaseException):
                raise response
            if url.endswith("/responses") and isinstance(response, dict):
                return httpx.Response(
                    200,
                    request=httpx.Request("POST", url),
                    json={
                        "id": "resp_test",
                        "status": "completed",
                        "usage": {"input_tokens": 100, "output_tokens": 40},
                        "output": [
                            {
                                "type": "message",
                                "role": "assistant",
                                "content": [{"type": "output_text", "text": response["message"]["content"]}],
                            }
                        ],
                    },
                )
            return httpx.Response(
                response if isinstance(response, int) else 200,
                request=httpx.Request("POST", url),
                json={"error": "synthetic rejection"}
                if isinstance(response, int)
                else {
                    "id": str(uuid4()),
                    "usage": {
                        "prompt_tokens": 100,
                        "completion_tokens": 40,
                        "completion_tokens_details": {"reasoning_tokens": 25},
                    },
                    "choices": [response],
                },
            )

    monkeypatch.setattr(conversation.httpx, "AsyncClient", Client)
    with session_scope() as db:
        conv = Conversation(owner_id="davin", device_id="test", private=True, learning=False)
        db.add(conv)
        db.flush()
        return conv.id, sent


def tool(name, arguments, signature=None):
    call = {
        "id": str(uuid4()),
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments)},
    }
    if signature:
        call["extra_content"] = {"google": {"thought_signature": signature}}
    return call


def response(*calls, content=None):
    return {
        "message": {"role": "assistant", "content": content, "tool_calls": list(calls)},
        "finish_reason": "tool_calls" if calls else "stop",
    }


def test_selection_is_persisted_and_bootstrap_never_exposes_keys(keys, client):
    boot = client.get("/api/v1/bootstrap").json()
    assert boot["agent_provider"] == "openai"
    assert boot["agent_model"] == "gpt-5.6-luna"
    assert next(m for m in boot["agent_options"] if m["provider"] == "gemini")["available"]
    select_provider("gemini")
    boot = client.get("/api/v1/bootstrap").json()
    assert boot["agent_provider"] == boot["preferences"]["agent_provider"] == "gemini"
    assert boot["agent_model"] == "gemini-3.8-flash"
    assert boot["capabilities"]["voice"] and boot["capabilities"]["chat"]
    assert "synthetic-" not in json.dumps(boot)
    assert "synthetic-" not in repr(agent_models.catalog())
    select_provider("openai")
    assert client.get("/api/v1/bootstrap").json()["agent_provider"] == "openai"


def test_missing_key_rejected_and_unavailable_selection_visible(keys, client, monkeypatch):
    monkeypatch.setattr(keys, "gemini_api_key", "")
    with pytest.raises(DomainError, match="GEMINI_API_KEY"):
        select_provider("gemini")
    assert client.get("/api/v1/bootstrap").json()["agent_provider"] == "openai"
    result = client.post(
        "/api/v1/commands",
        json={
            "command_id": str(uuid4()),
            "tool": "settings.update",
            "arguments": {"agent_provider": "arbitrary-endpoint"},
        },
    )
    assert result.status_code == 400
    monkeypatch.setattr(keys, "gemini_api_key", "synthetic-gemini")
    select_provider("gemini")
    monkeypatch.setattr(keys, "gemini_api_key", "")
    boot = client.get("/api/v1/bootstrap").json()
    assert boot["agent_provider"] == "gemini" and not boot["capabilities"]["chat"]
    # Removing a credential does not silently send private context to a different provider.
    with pytest.raises(DomainError, match="GEMINI_API_KEY"):
        agent_models.selected(boot["preferences"], require_key=True)


def test_legacy_groq_default_and_gemini_only_setup(keys, monkeypatch):
    monkeypatch.setattr(keys, "openai_api_key", "")
    monkeypatch.setattr(keys, "groq_api_key", "synthetic-groq")
    assert agent_models.default_provider() == "groq"
    monkeypatch.setattr(keys, "groq_api_key", "")
    assert agent_models.default_provider() == "gemini"
    model = agent_models.selected({})
    assert model.model == "gemini-3.8-flash"


async def test_gemini_parallel_and_sequential_tools_keep_signatures_and_receipts(keys, monkeypatch):
    select_provider("gemini")
    first = response(
        tool("task_create", {"title": "First"}, "opaque-A"),
        tool("task_create", {"title": "Second"}),
    )
    first["message"]["extra_content"] = {"google": {"thought_signature": "opaque-message"}}
    second = response(tool("task_list", {}, "opaque-B"))
    cid, sent = fake_provider(monkeypatch, [first, second, response(content="Both tasks are saved.")])
    turn = str(uuid4())
    result = await conversation.chat("davin", "test", turn, cid, "Create two tasks and check them.")
    assert result["status"] == "succeeded" and len(result["actions"]) == 2
    assert result["provider"] == "gemini" and result["model"] == "gemini-3.8-flash"
    assert all(s["url"] == agent_models.catalog()["gemini"].endpoint for s in sent)
    assert all(s["headers"]["Authorization"] == "Bearer synthetic-gemini-key" for s in sent)
    assert sent[0]["body"]["reasoning_effort"] == "low"
    assert sent[0]["body"]["max_tokens"] == 8192
    assert "max_completion_tokens" not in sent[0]["body"]
    assert "Remembered context" in sent[0]["body"]["messages"][0]["content"]
    assert len(sent[0]["body"]["tools"]) < 20
    assert {"note_search", "note_read", "ui_show"} <= {
        t["function"]["name"] for t in sent[0]["body"]["tools"]
    }
    history = sent[-1]["body"]["messages"]
    assistant = [m for m in history if m["role"] == "assistant"]
    assert assistant == [first["message"], second["message"]]
    assert [m["role"] for m in history][-5:] == ["assistant", "tool", "tool", "assistant", "tool"]
    with session_scope() as db:
        assert len(list(db.scalars(select(Task)))) == 2
        usage = list(db.scalars(select(Usage)))
        assert len(usage) == 3
        # Reasoning tokens are included in completion_tokens, not counted twice.
        assert all(float(u.amount) == pytest.approx((100 * 0.75 + 40 * 3.75) / 1_000_000) for u in usage)
        job = db.get(Job, turn)
        assert job.payload["model"] == "gemini-3.8-flash"
        assert "opaque-" not in json.dumps(job.result)
    repeated = await conversation.chat("davin", "test", turn, cid, "Create two tasks and check them.")
    assert repeated["actions"] == result["actions"] and len(sent) == 3


async def test_provider_stays_pinned_during_turn_and_changes_on_next_turn(keys, monkeypatch):
    select_provider("gemini")
    cid, sent = fake_provider(
        monkeypatch,
        [
            response(tool("tools_load", {"groups": ["settings"]}, "opaque-load")),
            response(tool("settings_update", {"agent_provider": "openai"}, "opaque-A")),
            response(content="Updated the task agent."),
            response(content="OpenAI handles this request."),
        ],
    )
    await conversation.chat("davin", "test", str(uuid4()), cid, "Switch to the OpenAI task agent.")
    result = await conversation.chat("davin", "test", str(uuid4()), cid, "Hello")
    assert [s["body"]["model"] for s in sent] == ["gemini-3.8-flash"] * 3 + ["gpt-5.6-luna"]
    assert result["provider"] == "openai"
    assert "reasoning_effort" not in sent[-1]["body"]
    assert "max_tokens" not in sent[-1]["body"]
    assert sent[-1]["body"]["max_output_tokens"] == 8192
    assert sent[-1]["body"]["reasoning"] == {"effort": "low"}


@pytest.mark.parametrize("failure", [403, 429, httpx.ReadTimeout("synthetic timeout")])
async def test_failure_after_saved_action_does_not_fallback_or_replay(keys, monkeypatch, failure):
    select_provider("gemini")
    cid, sent = fake_provider(
        monkeypatch,
        [
            response(tool("task_create", {"title": "Saved once"}, "opaque-A")),
            failure,
        ],
    )
    turn = str(uuid4())
    result = await conversation.chat("davin", "test", turn, cid, "Save one task.")
    assert result["status"] == "failed" and len(result["actions"]) == 1
    assert "Gemini 3.8 Flash" in result["message"]
    assert len(sent) == 2 and all("googleapis.com" in s["url"] for s in sent)
    with session_scope() as db:
        assert len(list(db.scalars(select(Task)))) == 1
        assert db.get(BudgetReservation, turn).state == (
            "uncertain" if isinstance(failure, Exception) else "closed"
        )


async def test_gemini_obeys_action_limit_and_disabled_cost_tracking(keys, monkeypatch):
    select_provider("gemini")
    monkeypatch.setattr(keys, "cost_tracking_enabled", False)
    monkeypatch.setattr(keys, "max_tool_calls_per_request", 1)
    cid, sent = fake_provider(
        monkeypatch,
        [
            response(
                tool("task_create", {"title": "One"}, "opaque-A"), tool("task_create", {"title": "Two"})
            ),
            response(content="One saved, one still unfinished."),
        ],
    )
    result = await conversation.chat("davin", "test", str(uuid4()), cid, "Create two tasks.")
    assert result["status"] == "partial" and len(result["actions"]) == 1
    assert sent[-1]["body"]["tool_choice"] == "none"
    with session_scope() as db:
        assert list(db.scalars(select(Usage))) == []
        assert list(db.scalars(select(BudgetReservation))) == []


def test_gemini_pricing_changes_at_announced_date(keys):
    model = agent_models.catalog()["gemini"]
    assert model.rates(date(2026, 12, 31)) == (0.75, 3.75)
    assert model.rates(date(2027, 1, 1)) == (1.5, 7.5)


@pytest.mark.parametrize(
    "reply",
    [
        {
            "message": {"role": "assistant", "content": None, "tool_calls": None},
            "finish_reason": "content_filter",
        },
        None,
    ],
)
async def test_empty_or_malformed_provider_reply_finishes_job(keys, monkeypatch, reply):
    select_provider("gemini")
    cid, _ = fake_provider(monkeypatch, [reply])
    turn = str(uuid4())
    result = await conversation.chat("davin", "test", turn, cid, "Hello")
    assert result["status"] == "failed"
    with session_scope() as db:
        assert db.get(Job, turn).status == "failed"
        assert db.get(BudgetReservation, turn).state == "closed"
