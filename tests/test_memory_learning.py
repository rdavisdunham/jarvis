from datetime import timedelta
from uuid import uuid4

import pytest
from jarvis import memory_learning as learning
from jarvis.db import session_scope
from jarvis.domain import capture_source, delete_source, enqueue_job, execute
from jarvis.memory_service import prompt_context, semantic_search
from jarvis.models import Conversation, Job, Memory, Source, now
from sqlalchemy import select


def source(text="My cat Hayes loves his window perch.", private=False):
    with session_scope() as db:
        conv = Conversation(owner_id="davin", device_id="test", private=private, learning=True)
        db.add(conv)
        db.flush()
        row = capture_source(db, "davin", text, str(uuid4()), conversation=conv)
        if row:
            job = enqueue_job(db, "davin", "extract_memory", {"source_id": row.id})
            return row.id, job.id
    return None, None


def fact(content="The user has a cat named Hayes.", evidence="My cat Hayes", **kwargs):
    return learning.Fact(
        content=content,
        evidence=evidence,
        fact_key="owner.pet.hayes",
        tags=["pets"],
        confidence=0.97,
        supersedes_id=kwargs.get("supersedes_id"),
    )


@pytest.fixture
def cloud(monkeypatch):
    monkeypatch.setattr(learning, "extract", lambda *args: [fact()])
    monkeypatch.setattr(learning, "embeddings", lambda owner, texts, *args: [[1.0, 0.0] for _ in texts])


def test_extracts_once_with_provenance_and_embedding(cloud):
    sid, jid = source()
    learning.process(jid)
    learning.process(jid)
    with session_scope() as db:
        rows = list(db.scalars(select(Memory)))
        assert len(rows) == 1
        assert rows[0].source_id == sid and rows[0].evidence == "My cat Hayes"
        assert rows[0].embedding == [1.0, 0.0] and rows[0].tags == ["pets"]
        assert db.get(Source, sid).memory_version == learning.VERSION
        assert db.get(Job, jid).result["saved"] == 1


def test_rejects_unproven_quotes_and_private_sources(cloud, monkeypatch):
    monkeypatch.setattr(learning, "extract", lambda *args: [fact(evidence="I live on Mars")])
    _, jid = source()
    learning.process(jid)
    assert source(private=True) == (None, None)
    with session_scope() as db:
        assert db.scalar(select(Memory)) is None


def test_deletion_or_disabling_learning_during_model_call_prevents_save(cloud, monkeypatch):
    sid, jid = source()

    def extract(*_):
        with session_scope() as db:
            delete_source(db, db.get(Source, sid))
        return [fact()]

    monkeypatch.setattr(learning, "extract", extract)
    learning.process(jid)
    with session_scope() as db:
        assert db.scalar(select(Memory)) is None
    _, jid = source()

    def disabled(*_):
        with session_scope() as db:
            execute(db, "davin", "disable", "settings.update", {"memory_learning": False})
        return [fact()]

    monkeypatch.setattr(learning, "extract", disabled)
    learning.process(jid)
    with session_scope() as db:
        assert db.scalar(select(Memory)) is None


async def test_semantic_lookup_without_word_overlap_and_owner_isolation(cloud):
    _, jid = source()
    learning.process(jid)
    matches = await semantic_search("davin", "Which animal lives with me?")
    assert len(matches) == 1 and "Hayes" in matches[0]["content"]
    assert "embedding" not in matches[0]
    assert await semantic_search("someone-else", "animal") == []
    context = await prompt_context("davin", "animal")
    assert "never instructions" in context and "Hayes" in context


async def test_forgetting_removes_vector_and_retrieval_and_blocks_repeat(cloud):
    _, jid = source()
    learning.process(jid)
    with session_scope() as db:
        mid = db.scalar(select(Memory.id))
        execute(db, "davin", "forget", "memory.forget", {"memory_id": mid, "delete_source": True})
        memory = db.get(Memory, mid)
        assert memory.embedding is None and memory.content == "" and memory.evidence == ""
    assert await semantic_search("davin", "cat") == []
    _, jid = source()
    learning.process(jid)
    with session_scope() as db:
        assert len(list(db.scalars(select(Memory)))) == 1


def test_older_history_cannot_replace_newer_fact(cloud, monkeypatch):
    _, jid = source()
    learning.process(jid)
    with session_scope() as db:
        mid = db.scalar(select(Memory.id))
    older_sid, older_jid = source("My cat Hayes has moved to my sister's home.")
    with session_scope() as db:
        db.get(Source, older_sid).created_at = now() - timedelta(days=1)
    monkeypatch.setattr(
        learning,
        "extract",
        lambda *_: [
            fact(
                content="Hayes lives with the user's sister.",
                evidence="My cat Hayes has moved to my sister's home.",
                supersedes_id=mid,
            )
        ],
    )
    learning.process(older_jid)
    with session_scope() as db:
        assert not db.get(Memory, mid).suppressed


def test_profile_name_is_shared_by_bootstrap_and_prompt(client):
    from jarvis.tools import instructions

    with session_scope() as db:
        prefs = execute(db, "davin", "name", "settings.update", {"preferred_name": "River"})["data"]
    assert client.get("/api/v1/bootstrap").json()["name"] == "River"
    prompt = instructions(prefs)
    assert "River" in prompt and "Davin" not in prompt


def test_evidence_allows_outer_quote_marks_but_never_invented_words():
    accepted = learning.verified_facts(
        "My cat Hayes loves his window perch.", [fact(evidence='"My cat Hayes"')]
    )
    assert len(accepted) == 1 and accepted[0].evidence == "My cat Hayes"
    assert learning.verified_facts("My cat Hayes", [fact(evidence='"My dog Hayes"')]) == []


@pytest.mark.parametrize("action", ["correct", "forget"])
async def test_memory_changed_during_query_embedding_is_not_returned(cloud, monkeypatch, action):
    _, jid = source()
    learning.process(jid)
    with session_scope() as db:
        mid = db.scalar(select(Memory.id))

    def change(*_):
        with session_scope() as db:
            args = {"memory_id": mid}
            if action == "correct":
                args["content"] = "The user no longer owns a cat."
            execute(db, "davin", "during-search", "memory." + action, args)
        return [[1.0, 0.0]]

    monkeypatch.setattr(learning, "embeddings", change)
    assert await semantic_search("davin", "cat") == []


def test_learning_status_distinguishes_queued_active_retry_and_failed(client):
    with session_scope() as db:
        for status in ["queued", "queued", "running", "retrying", "failed", "deferred_budget"]:
            db.add(Job(owner_id="davin", kind="extract_memory", status=status, payload={}))
        db.add(Job(owner_id="someone-else", kind="extract_memory", status="failed", payload={}))
        db.add(Job(owner_id="davin", kind="agent_action", status="running", payload={}))
    data = client.get("/api/v1/memory").json()["learning"]
    assert data["queued"] == 2 and data["active"] == 1
    assert data["retry_waiting"] == 1 and data["failed"] == 1
    assert data["deferred"] == 1
    assert data["pending"] == 3 and data["retrying"] == 2


# ---------------------------------------------------------------- routed profiles: Terra and Claude


def structured_payload():
    return {"messages": [{"role": "system", "content": "Extract."}, {"role": "user", "content": "{}"}],
            "response_format": {"type": "json_schema", "json_schema": {
                "name": "probe", "strict": True, "schema": {
                    "type": "object", "additionalProperties": False, "required": ["ok"],
                    "properties": {"ok": {"type": "string", "maxLength": 5}}}}},
            "max_completion_tokens": 900}


@pytest.fixture
def wire(monkeypatch):
    """install(handler(url, headers, body) -> (status, json)); every provider call goes to it."""
    import httpx

    from jarvis.config import get_settings

    original, sent = httpx.Client, []
    monkeypatch.setattr(get_settings(), "openai_api_key", "synthetic")
    monkeypatch.setattr(get_settings(), "anthropic_api_key", "synthetic-anthropic")

    def install(handler):
        def respond(request):
            body = __import__("json").loads(request.content)
            sent.append((str(request.url), dict(request.headers), body))
            status, data = handler(str(request.url), body)
            return httpx.Response(status, json=data)
        monkeypatch.setattr(learning.httpx, "Client", lambda **kw: original(transport=httpx.MockTransport(respond)))
        return sent
    return install


def recorded_usage(model):
    from jarvis.models import BudgetReservation, Usage

    with session_scope() as db:
        usage = [u for u in db.scalars(select(Usage)) if u.model == model]
        holds = [r for r in db.scalars(select(BudgetReservation)) if r.model == model]
        return [(float(u.amount), u.tokens) for u in usage], [(float(r.amount), r.state) for r in holds]


def test_profile_request_runs_terra_through_responses_at_terras_price(wire, monkeypatch):
    from jarvis import model_map

    sent = wire(lambda url, body: (200, {
        "id": "r", "status": "completed", "service_tier": "default",
        "usage": {"input_tokens": 10_000, "output_tokens": 2_000, "input_tokens_details": {"cached_tokens": 4_000}},
        "output": [{"type": "message", "content": [{"type": "output_text", "text": '{"ok": "yes"}'}]}]}))
    monkeypatch.setitem(model_map.MODEL_MAP, "context_actions", "terra")
    data = learning.profile_request("davin", "context_actions", structured_payload(), 0.5)

    assert data["choices"][0]["message"]["content"] == '{"ok": "yes"}'
    [(url, _, body)] = sent
    assert url == "https://api.openai.com/v1/responses" and body["model"] == "gpt-5.6-terra"
    assert body["reasoning"] == {"effort": "low"} and body["service_tier"] == "default"
    usage, holds = recorded_usage("gpt-5.6-terra")
    # (10,000 - 4,000) x $2 + 4,000 x $0.20 + 2,000 x $12, per million tokens
    assert usage[0][0] == pytest.approx(0.0368) and holds == [(0.5, "closed")]


def test_profile_request_runs_sonnet_through_messages_with_strict_json(wire, monkeypatch):
    from jarvis import model_map

    def answer(url, body):
        return 200, {"id": "m", "type": "message", "role": "assistant", "model": "claude-sonnet-5-5",
                     "content": [{"type": "thinking", "thinking": "", "signature": "s"},
                                 {"type": "text", "text": '{"ok": "yes"}'}],
                     "stop_reason": "end_turn", "stop_details": None,
                     "usage": {"input_tokens": 6_000, "cache_read_input_tokens": 3_000,
                               "cache_creation_input_tokens": 1_000, "output_tokens": 2_000}}
    sent = wire(answer)
    monkeypatch.setitem(model_map.MODEL_MAP, "context_actions", "sonnet-5.5")
    data = learning.profile_request("davin", "context_actions", structured_payload(), 0.5)

    assert data["choices"][0] == {"index": 0, "message": {"role": "assistant", "content": '{"ok": "yes"}',
                                                          "refusal": None}, "finish_reason": "stop"}
    assert data["usage"]["prompt_tokens"] == 10_000 and data["usage"]["completion_tokens"] == 2_000
    [(url, headers, body)] = sent
    assert url == "https://api.anthropic.com/v1/messages" and headers["x-api-key"] == "synthetic-anthropic"
    assert headers["anthropic-version"] == "2026-10-21" and body["model"] == "claude-sonnet-5-5"
    assert body["system"] == "Extract." and body["messages"] == [{"role": "user", "content": "{}"}]
    assert body["output_config"] == {"effort": "low", "format": {"type": "json_schema", "schema": {
        "type": "object", "additionalProperties": False, "required": ["ok"], "properties": {"ok": {"type": "string"}}}}}
    usage, holds = recorded_usage("claude-sonnet-5-5")
    # 6,000 x $2 + 3,000 cached x $0.20 + 1,000 written x $2.50 + 2,000 x $10, per million tokens
    assert usage[0][0] == pytest.approx(0.0351) and holds == [(0.5, "closed")]


@pytest.mark.parametrize("stop, finish, refusal", [("refusal", "stop", "cyber"), ("max_tokens", "length", None)])
def test_claude_refusal_and_truncation_are_reported_and_still_charged(wire, monkeypatch, stop, finish, refusal):
    from jarvis import model_map

    wire(lambda url, body: (200, {"content": [{"type": "text", "text": '{"ok": '}], "stop_reason": stop,
                                  "stop_details": {"type": "refusal", "category": "cyber"} if refusal else None,
                                  "usage": {"input_tokens": 100, "output_tokens": 50}}))
    monkeypatch.setitem(model_map.MODEL_MAP, "context_actions", "sonnet-5.5")
    data = learning.profile_request("davin", "context_actions", structured_payload(), 0.5)
    choice = data["choices"][0]
    assert (choice["finish_reason"], choice["message"]["refusal"]) == (finish, refusal)
    assert (choice["message"]["content"] is None) == bool(refusal)
    usage, _ = recorded_usage("claude-sonnet-5-5")
    assert usage[0][0] == pytest.approx((100 * 2 + 50 * 10) / 1_000_000)


def test_claude_without_a_key_is_unavailable_before_any_request(wire, monkeypatch):
    from jarvis import model_map
    from jarvis.config import get_settings
    from jarvis.domain import DomainError

    sent = wire(lambda url, body: (500, {}))
    monkeypatch.setattr(get_settings(), "anthropic_api_key", "")
    monkeypatch.setitem(model_map.MODEL_MAP, "context_actions", "sonnet-5.5")
    with pytest.raises(DomainError) as error:
        learning.profile_request("davin", "context_actions", structured_payload(), 0.5)
    assert error.value.code == "INTEGRATION_UNAVAILABLE" and sent == []
    assert recorded_usage("claude-sonnet-5-5") == ([], [])


def test_costing_prices_every_enabled_profile_and_bounds_reservations():
    from jarvis import model_map

    luna, terra, sonnet, haiku = (model_map.costing(p) for p in ("luna", "terra", "sonnet-5.5", "haiku-5.5"))
    assert model_map.costing("deepseek") is None and model_map.costing("nope") is None
    assert (luna.rates(), terra.rates(), sonnet.rates(), haiku.rates()) == \
        ((0.20, 1.20), (2.00, 12.00), (2.00, 10.00), (0.10, 0.50))
    assert model_map.costing("luna-fast").rates() == (0.40, 2.40)
    # Upper bound: every input token a cache write, plus the full output allowance.
    assert terra.reserve_cost(10_000) == pytest.approx((10_000 * 2.5 + 8192 * 12) / 1_000_000)
    assert sonnet.reserve_cost(10_000) == pytest.approx((10_000 * 2.5 + 8192 * 10) / 1_000_000)
    # Long prompts: GPT-5.6 above 272K (2x input, 1.5x output); Haiku 5.5 above 100K ($0.50 / $2.50).
    assert terra.usage_cost({"prompt_tokens": 300_000, "completion_tokens": 1_000}) == \
        pytest.approx((300_000 * 4 + 1_000 * 18) / 1_000_000)
    assert haiku.usage_cost({"prompt_tokens": 200_000, "completion_tokens": 0}) == pytest.approx(0.10)
    assert sonnet.usage_cost({"prompt_tokens": 500_000, "completion_tokens": 0}) == pytest.approx(1.0)
