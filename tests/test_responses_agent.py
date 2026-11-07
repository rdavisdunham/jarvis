import asyncio
import copy
import json
from unittest.mock import AsyncMock
from uuid import uuid4

import httpx
import pytest
from cryptography.fernet import Fernet
from jarvis import agent_models, work_intake, work_runner
from jarvis.config import get_settings
from jarvis.db import session_scope
from jarvis.domain import DomainError, execute, preferences
from jarvis.models import BudgetReservation, Conversation, Job, OwnerSettings, Task, Usage
from scripts.app_eval.work_turn import run_turn
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
    monkeypatch.setattr(get_settings(), "integration_encryption_key", Fernet.generate_key().decode())
    monkeypatch.setattr(work_runner, "prompt_context", AsyncMock(return_value="Synthetic memory"))
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

    monkeypatch.setattr(httpx, "AsyncClient", Client)
    with session_scope() as db:
        conv = Conversation(owner_id="davin", device_id="test", private=True, learning=False)
        db.add(conv)
        db.flush()
        return conv.id, sent


async def test_reasoning_parallel_calls_and_continuations_preserve_native_items(client, monkeypatch):
    first = packet(
        thought("a"),
        function("task_create", {"title": "One"}, "a"),
        function("task_create", {"title": "Two"}, "b"),
    )
    second = packet(thought("b"), function("task_list", {}, "c"))
    cid, sent = setup(monkeypatch, [first, second, packet(thought("c"), answer("Saved both."))])
    turn = str(uuid4())
    result = await run_turn("davin", "test", turn, cid, "Create two tasks and verify.")
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
    start = next(i for i, item in enumerate(history) if item.get("role") == "user") + 1
    assert history[start:start + 3] == first["output"]
    assert history[start + 5:start + 7] == second["output"]
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
    repeated = await run_turn("davin", "test", turn, cid, "Create two tasks and verify.")
    assert repeated["actions"] == result["actions"] and len(sent) == 3


async def test_incomplete_response_does_not_execute_partial_calls(client, monkeypatch):
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
    result = await run_turn("davin", "test", str(uuid4()), cid, "Create a task")
    assert result["status"] == "failed" and not result["actions"]
    with session_scope() as db:
        assert list(db.scalars(select(Task))) == []
    assert len(sent) == 1


async def test_reasoning_action_limit_and_disabled_costs(client, monkeypatch):
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
    result = await run_turn("davin", "test", str(uuid4()), cid, "Save two")
    assert result["status"] == "partial" and len(result["actions"]) == 1
    assert sent[-1]["body"]["tool_choice"] == "none"
    with session_scope() as db:
        assert not list(db.scalars(select(Usage))) and not list(db.scalars(select(BudgetReservation)))


@pytest.mark.parametrize("failure", [httpx.ReadTimeout("synthetic"), asyncio.CancelledError()])
async def test_failed_continuation_never_repeats_saved_actions(client, monkeypatch, failure):
    cid, sent = setup(
        monkeypatch,
        [
            packet(thought("a"), function("task_create", {"title": "Saved once"}, "a")),
            failure,
            packet(answer("Saved the task.")),
        ],
    )
    turn = str(uuid4())
    if isinstance(failure, asyncio.CancelledError):
        # A killed process is not a request outcome: the next invocation resumes from the checkpoint.
        with pytest.raises(asyncio.CancelledError):
            await run_turn("davin", "test", turn, cid, "Save task")
        result = await run_turn("davin", "test", turn, cid, "Save task")
        assert result["status"] == "succeeded" and len(sent) == 3
    else:
        result = await run_turn("davin", "test", turn, cid, "Save task")
        assert result["status"] == "partial" and len(sent) == 2
        with session_scope() as db:
            assert db.get(BudgetReservation, turn).state == "uncertain"
    assert len(result["actions"]) == 1
    with session_scope() as db:
        assert len(list(db.scalars(select(Task)))) == 1


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
    assert boot["agent_service_tier"] == "fast"
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


def voice_request(cid, text):
    """Speech captured by a voice session and claimed as one request, as GPT-Live delegates it."""
    with session_scope() as db:
        inbox = work_intake.open_voice(db, str(uuid4()), "davin", "davin", "test", cid)
        work_intake.append_voice(db, inbox.id, str(uuid4()), "user", text, 0, 1000)
        return work_intake.claim_voice(db, inbox).id, inbox.id


async def test_voice_end_is_session_bound_and_stops_the_rest_of_the_tool_batch(client, monkeypatch):
    from jarvis.models import AgentWork, VoiceInbox
    from jarvis.work_crypto import unseal
    cid, sent = setup(monkeypatch, [packet(
        function("task_create", {"title": "Requested final task"}, "save"),
        function("voice_end", {}, "end"),
        function("task_create", {"title": "Must not run after goodbye"}, "late"),
    )])
    request, session = voice_request(cid, "Save the task then end voice.")
    await work_runner.run(request)
    with session_scope() as db:
        assert unseal(db.get(VoiceInbox, session).content_ciphertext)["end_intent"]["work_id"] == request
        work = db.get(AgentWork, request)
        assert work.result["tool_calls"] == 2 and len(work.result["actions"]) == 1
        assert list(db.scalars(select(Task.title))) == ["Requested final task"]
    assert len(sent) == 1
    definition = next(t for t in sent[0]["body"]["tools"] if t["name"] == "voice_end")
    assert definition["parameters"]["properties"] == {}
    assert definition["parameters"]["additionalProperties"] is False


async def test_text_chat_cannot_invent_voice_end_and_voice_tool_rejects_target_ids(client, monkeypatch):
    from jarvis.models import VoiceInbox
    from jarvis.work_crypto import unseal
    cid, sent = setup(monkeypatch, [
        packet(function("voice_end", {}, "not-loaded")),
        packet(answer("No active voice session.")),
    ])
    result = await run_turn("davin", "test", str(uuid4()), cid, "End voice")
    assert result["tool_errors"][0]["error"] == "TOOL_NOT_LOADED"
    assert "voice_end" not in [t["name"] for t in sent[0]["body"]["tools"]]
    cid, _ = setup(monkeypatch, [
        packet(function("voice_end", {"session_id": "another-session"}, "bad")),
        packet(answer("I could not apply that control.")),
    ])
    with session_scope() as db:
        session = work_intake.open_voice(db, str(uuid4()), "davin", "davin", "test", cid).id
    result = await run_turn("davin", "test", str(uuid4()), cid, "End voice", voice_session_id=session)
    assert result["tool_errors"][0]["error"] == "INVALID_ARGUMENT"
    with session_scope() as db:
        assert "end_intent" not in unseal(db.get(VoiceInbox, session).content_ciphertext)


def test_interactive_fast_default_standard_override_and_maintenance_catalog(monkeypatch):
    monkeypatch.setattr(get_settings(), "agent_service_tier", "fast")
    agent = agent_models.selected({"agent_profile": "luna"})
    assert agent.reasoning_effort == "low" and agent.service_tier == "fast"
    assert agent.request([], [])["service_tier"] == "fast"
    assert agent_models.catalog()["luna"].service_tier == "default"
    assert agent_models.selected({"agent_profile": "gemini"}).service_tier == "default"
    monkeypatch.setattr(get_settings(), "agent_service_tier", "default")
    assert agent_models.selected({"agent_profile": "luna"}).request([], [])["service_tier"] == "default"


@pytest.mark.parametrize("served,multiplier,basis", [
    ("priority", 2, "returned_service_tier"),
    ("fast", 2, "returned_service_tier"),
    ("default", 1, "returned_service_tier"),
    (None, 2, "requested_service_tier_estimate"),
    ("unrecognized", 2, "requested_service_tier_estimate"),
])
def test_fast_usage_prices_returned_tier_and_labels_missing_metadata(served, multiplier, basis):
    agent = agent_models.selected({"agent_profile": "luna"}, service_tier="fast")
    original = packet(answer("Done"))
    if served is not None:
        original["service_tier"] = served
    usage = agent.usage_record(agent.normalize(original))
    standard = agent_models.catalog()["luna"].usage_cost(agent.normalize(original)["usage"])
    assert usage["requested_service_tier"] == "fast"
    assert usage["cost_basis"] == basis
    assert usage["served_service_tier"] == (served if basis == "returned_service_tier" else None)
    assert agent.usage_cost(usage) == pytest.approx(standard * multiplier)
    assert agent.reserve_cost(100) == pytest.approx(agent_models.catalog()["luna"].reserve_cost(100) * 2)
    assert agent.reserve_cost(280000) == pytest.approx(agent_models.catalog()["luna"].reserve_cost(280000) * 2)


async def test_inline_fast_downgrade_records_actual_standard_cost(client, monkeypatch):
    reply = packet(answer("Hello."))
    reply["service_tier"] = "default"
    cid, sent = setup(monkeypatch, [reply])
    monkeypatch.setattr(get_settings(), "agent_service_tier", "fast")
    await run_turn("davin", "test", str(uuid4()), cid, "Hello")
    assert sent[0]["body"]["service_tier"] == "fast"
    with session_scope() as db:
        usage = db.scalar(select(Usage))
        assert usage.tokens["requested_service_tier"] == "fast"
        assert usage.tokens["served_service_tier"] == "default"
        assert float(usage.amount) == pytest.approx(agent_models.catalog()["luna"].usage_cost(usage.tokens), abs=0.0000005)


@pytest.mark.parametrize("path", ["responses", "embeddings"])
def test_maintenance_stays_standard_when_interactive_is_fast(monkeypatch, path):
    from jarvis import memory_learning
    monkeypatch.setattr(get_settings(), "openai_api_key", "synthetic")
    monkeypatch.setattr(get_settings(), "agent_service_tier", "fast")
    sent = []
    class Client:
        def __init__(self, **_): pass
        def __enter__(self): return self
        def __exit__(self, *_): pass
        def post(self, url, *, json, headers):
            sent.append(json)
            data = ({"usage": {"total_tokens": 12}, "data": []} if path == "embeddings"
                    else {**packet(answer("Done")), "service_tier": "default"})
            return httpx.Response(200, request=httpx.Request("POST", url), json=data)
    monkeypatch.setattr(memory_learning.httpx, "Client", Client)
    model = "text-embedding-3-small" if path == "embeddings" else "gpt-5.6-luna"
    memory_learning.provider_request("davin", path, {"model": model}, model, .1)
    assert sent[0].get("service_tier") == (None if path == "embeddings" else "default")
    if path == "responses":
        with session_scope() as db:
            usage = db.scalar(select(Usage))
            assert usage.tokens["requested_service_tier"] == "default"
            assert usage.tokens["served_service_tier"] == "default"
            assert float(usage.amount) == pytest.approx(agent_models.catalog()["luna"].usage_cost(usage.tokens), abs=0.0000005)
