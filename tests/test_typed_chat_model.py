"""Typed chat runs on its own model (GPT-6.1 Terra by default); voice, bots and background work keep the task agent."""

from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from jarvis import agent_models, agent_work, work_intake, work_runner
from jarvis.api import app
from jarvis.config import get_settings
from jarvis.db import session_scope
from jarvis.domain import DomainError, execute, preferences
from jarvis.models import AgentWork, Job
from test_agent_work import accept, recovery_settings, responder, response  # noqa: F401


def update(**values):
    with session_scope() as db:
        return execute(db, "davin", str(uuid4()), "settings.update", values)


def profile_of(identity):
    with session_scope() as db:
        payload = db.get(Job, identity).payload
        return payload["profile"], payload["service_tier"]


def voice_work(client, conversation_id, message="Add Beta"):
    """A request delegated from a GPT-Live session in the same conversation."""
    work, _ = accept(client, "Hello")
    with session_scope() as db:
        typed = db.get(AgentWork, work["id"])
        inbox = work_intake.open_voice(db, str(uuid4()), typed.owner_id, typed.account_id, typed.device_id,
                                       conversation_id)
        work_intake.append_voice(db, inbox.id, "e1", "user", message, 0, 1000)
        return work_intake.claim_voice(db, inbox).id


def test_typed_chat_defaults_to_terra_while_voice_and_bots_stay_on_luna(client, monkeypatch):
    monkeypatch.setattr(get_settings(), "agent_service_tier", "fast")
    boot = client.get("/api/v1/bootstrap").json()
    assert (boot["chat_profile"], boot["chat_model"], boot["chat_reasoning"]) == ("terra-6.1", "gpt-6.1-terra", "medium")
    assert boot["agent_profile"] == "luna" and boot["preferences"]["chat_profile"] == "terra-6.1"
    assert boot["capabilities"]["chat"]
    assert "terra-6.1" in {option["id"] for option in boot["agent_options"]}

    typed, payload = accept(client, "Add Alpha")
    # Terra requests Standard; the Fast tier setting applies to Luna only.
    assert profile_of(typed["id"]) == ("terra-6.1", "default")
    assert profile_of(voice_work(client, payload["conversation_id"])) == ("luna", "fast")

    response = client.post("/api/v1/bot-keys", json={"name": "Codex", "scopes": ["tasks:write", "work:run"],
                                                     "expires_in_days": 30})
    assert response.status_code == 201, response.text
    headers = {"Authorization": "Bearer " + response.json()["token"]}
    bot = TestClient(app).post("/api/v1/external/requests", headers=headers,
                               json={"request_id": str(uuid4()), "message": "Add from remote", "thread_id": str(uuid4())})
    assert bot.status_code == 202, bot.text
    assert profile_of(bot.json()["id"]) == ("luna", "fast")


def test_typed_chat_preference_round_trip_is_separate_from_the_task_agent(client, monkeypatch):
    update(chat_profile="luna-none")
    update(agent_profile="gemini")
    with session_scope() as db:
        prefs = preferences(db, "davin")
        assert (prefs["chat_profile"], prefs["agent_profile"]) == ("luna-none", "gemini")
    boot = client.get("/api/v1/bootstrap").json()
    assert (boot["chat_profile"], boot["agent_profile"]) == ("luna-none", "gemini")
    typed, payload = accept(client, "Add Alpha")
    assert profile_of(typed["id"])[0] == "luna-none"
    assert profile_of(voice_work(client, payload["conversation_id"]))[0] == "gemini"
    update(chat_profile="terra-6.1")
    assert client.get("/api/v1/bootstrap").json()["chat_profile"] == "terra-6.1"

    monkeypatch.setattr(get_settings(), "openai_api_key", "")
    with pytest.raises(DomainError, match="OPENAI_API_KEY"):
        update(chat_profile="terra-6.1")
    # Without an OpenAI key and no saved choice, typed chat follows the task agent instead of failing.
    assert agent_models.typed_chat({"agent_profile": "gemini"}).profile_id == "gemini"


@pytest.mark.asyncio
async def test_clarification_answer_and_revision_keep_the_original_model(client, monkeypatch):
    typed, payload = accept(client, "Hello")
    spoken = voice_work(client, payload["conversation_id"], "Schedule the review")
    with session_scope() as db:
        agent_work.finish(db, db.get(AgentWork, spoken), "needs_input", "What time?")
    question = client.get("/api/v1/work/" + spoken).json()["clarification"]
    # A typed reply to a voice-born question is accepted on Terra, then adopts the question's model.
    reply = client.post("/api/v1/work", json={"conversation_id": payload["conversation_id"],
                                              "turn_id": str(uuid4()), "message": "9 a.m."}).json()
    assert profile_of(reply["id"])[0] == "terra-6.1"
    responder(monkeypatch, [response([("work_answer", {"request_id": spoken, "clarification_id": question["id"]})])])
    await work_runner.run(reply["id"])
    with session_scope() as db:
        assert db.get(Job, spoken).status == "continued"
        assert db.get(AgentWork, reply["id"]).result["continuation_root"] == spoken
    assert profile_of(reply["id"])[0] == "luna"

    # A revision keeps the accepted model even after the preference changes.
    update(chat_profile="luna")
    with session_scope() as db:
        agent_work.revise(db, db.get(AgentWork, typed["id"]), "Actually say hi")
    assert profile_of(typed["id"])[0] == "terra-6.1"
    assert work_runner.model_for("terra-6.1").model == "gpt-6.1-terra"


def test_terra_pricing_cache_long_context_and_reservation():
    agent = agent_models.catalog()["terra-6.1"]
    assert (agent.model, agent.api, agent.reasoning_effort, agent.max_output_tokens) == (
        "gpt-6.1-terra", "responses", "medium", 16384)
    assert agent.rates() == (1.00, 5.00) and agent.rates(service_tier="fast") == (2.00, 10.00)
    usage = {"prompt_tokens": 100, "completion_tokens": 30,
             "prompt_tokens_details": {"cached_tokens": 40, "cache_write_tokens": 20}}
    # 40 ordinary, 40 cached (5% of input) and 20 cache-write (1.25x) input tokens.
    assert agent.usage_cost(usage) == pytest.approx((40 * 1.00 + 40 * 0.05 + 20 * 1.25 + 30 * 5.00) / 1e6)
    assert agent.usage_cost({"prompt_tokens": 300_000, "completion_tokens": 1000}) == pytest.approx(
        (300_000 * 2.00 + 1000 * 7.50) / 1e6)
    assert agent.reserve_cost(100) == pytest.approx((100 * 1.25 + 16384 * 5.00) / 1e6)
    record = agent.usage_record({"usage": usage, "service_tier": "default"})
    assert record["cost_basis"] == "returned_service_tier" and record["requested_service_tier"] == "default"
    request = agent.request([{"role": "user", "content": "Hi"}], [])
    assert request["model"] == "gpt-6.1-terra" and request["reasoning"] == {"effort": "medium"}
    assert request["service_tier"] == "default" and request["max_output_tokens"] == 16384
