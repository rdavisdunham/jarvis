"""Langfuse tracing of model calls (llm_trace.py): off without keys, never in the way, and labeled to sort by."""

import json
from uuid import uuid4

import httpx
import pytest
from jarvis import llm_trace, memory_learning
from jarvis.config import get_settings
from jarvis.cost_features import feature

OWNER = "davin"


@pytest.fixture
def spans(monkeypatch):
    """A real Langfuse client whose spans land in memory instead of being sent."""
    from langfuse import Langfuse
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    exporter = InMemorySpanExporter()
    # A fresh key each time: the SDK shares one span pipeline per public key within a process.
    lf = Langfuse(public_key="pk-lf-" + uuid4().hex, secret_key="sk-lf-test", base_url="http://127.0.0.1:9",
                  environment="local", tracer_provider=TracerProvider(), span_exporter=exporter)
    monkeypatch.setattr(llm_trace, "client", lambda: lf)

    def finished():
        lf.flush()
        return [(s, dict(s.attributes)) for s in exporter.get_finished_spans()]
    yield finished
    lf.shutdown()


def test_off_without_keys_and_a_no_op_call_still_runs():
    settings = get_settings()
    assert not (settings.langfuse_public_key and settings.langfuse_secret_key)
    assert llm_trace.client() is None and not llm_trace.enabled()
    with llm_trace.call(model="m", input="x") as traced:
        traced.done(output="y", usage={"prompt_tokens": 1}, cost=0.1)
    llm_trace.begin(model="m").end()


def test_a_jobs_calls_share_one_trace_labeled_by_job_feature_and_owner(spans):
    @feature("placement_by_meaning")
    def job():
        with llm_trace.unit("place_by_meaning", seed="job:1", owner=OWNER, metadata={"job_id": "1"}):
            with llm_trace.call(model="text-embedding-3-small", kind="embedding", input=["a"]) as traced:
                traced.done(output={"vectors": 1}, usage={"prompt_tokens": 3, "total_tokens": 3}, cost=0.001)
            with llm_trace.call(model="gpt-5.6-luna", input=[{"role": "user", "content": "x"}]) as traced:
                traced.done(output={"choice": "c1"}, cost=0.002, usage={
                    "prompt_tokens": 10, "completion_tokens": 2, "prompt_tokens_details": {"cached_tokens": 4},
                    "completion_tokens_details": {"reasoning_tokens": 1}})
    job()
    got = spans()
    assert len(got) == 2 and len({s.context.trace_id for s, _ in got}) == 1
    (_, embedding), (_, generation) = got
    assert embedding["langfuse.observation.type"] == "embedding"
    assert generation["langfuse.observation.type"] == "generation"
    assert generation["langfuse.trace.name"] == "place_by_meaning"
    assert set(generation["langfuse.trace.tags"]) == {"background", "placement_by_meaning"}
    assert generation["user.id"] == OWNER and generation["langfuse.environment"] == "local"
    assert json.loads(generation["langfuse.observation.usage_details"]) == {
        "input": 10, "output": 2, "input_cached_tokens": 4, "output_reasoning_tokens": 1}
    assert json.loads(generation["langfuse.observation.cost_details"]) == {"total": 0.002}
    assert json.loads(generation["langfuse.observation.output"]) == {"choice": "c1"}


def test_a_call_outside_any_unit_is_its_own_trace_named_by_feature(spans):
    @feature("semantic_search")
    def search():
        for _ in range(2):
            with llm_trace.call(model="m", kind="embedding") as traced:
                traced.done(output={"vectors": 1})
    search()
    got = spans()
    assert len({s.context.trace_id for s, _ in got}) == 2
    assert all(a["langfuse.trace.name"] == "semantic_search" for _, a in got)
    assert all(set(a["langfuse.trace.tags"]) == {"request", "semantic_search"} for _, a in got)


def test_a_failed_call_is_an_error_and_still_raises(spans):
    with pytest.raises(RuntimeError), llm_trace.call(model="m"):
        raise RuntimeError("provider down")
    traced = llm_trace.begin(model="m")
    traced.end()  # Never answered.
    (_, raised), (_, silent) = spans()
    assert raised["langfuse.observation.level"] == "ERROR" and "provider down" in raised[
        "langfuse.observation.status_message"]
    assert silent["langfuse.observation.level"] == "ERROR"


def test_a_broken_langfuse_never_breaks_the_call(monkeypatch):
    class Broken:
        def start_as_current_observation(self, **_):
            raise RuntimeError("langfuse down")
    monkeypatch.setattr(llm_trace, "client", lambda: Broken())
    with llm_trace.call(model="m") as traced:
        traced.done(output="fine")


def test_background_calls_are_traced_with_their_answer_and_recorded_cost(spans, monkeypatch):
    original = httpx.Client
    monkeypatch.setattr(get_settings(), "openai_api_key", "synthetic")
    transport = httpx.MockTransport(lambda request: httpx.Response(200, json={
        "id": "resp_1", "status": "completed", "service_tier": "default", "usage": {"input_tokens": 1000, "output_tokens": 100},
        "output": [{"type": "message", "content": [{"type": "output_text", "text": "{\"choice\": \"c2\"}"}]}]}))
    monkeypatch.setattr(memory_learning.httpx, "Client", lambda **kw: original(transport=transport))

    @feature("record_overviews")
    def overview():
        return memory_learning.provider_request(OWNER, "responses", {
            "input": [{"role": "user", "content": "ABC"}], "reasoning": {"effort": "low"},
            "text": {"format": {"type": "json_schema", "name": "overview"}}}, "gpt-5.6-luna", .08)
    overview()
    [(_, call)] = spans()
    assert call["langfuse.trace.name"] == "record_overviews"
    assert json.loads(call["langfuse.observation.output"]) == {"choice": "c2"}
    assert json.loads(call["langfuse.observation.model.parameters"])["schema"] == "overview"
    assert json.loads(call["langfuse.observation.cost_details"])["total"] > 0


def test_usage_shapes_map_to_langfuse_keys():
    assert llm_trace.usage_details({"input_tokens": 5, "output_tokens": 6, "input_token_details": {
        "audio_tokens": 2, "cached_tokens": 1}, "output_token_details": {"audio_tokens": 3}}) == {
        "input": 5, "output": 6, "input_cached_tokens": 1, "input_audio_tokens": 2, "output_audio_tokens": 3}
    assert llm_trace.usage_details({"seconds": 15.4}) == {"seconds": 15}
    assert llm_trace.usage_details({"prompt_tokens": -1, "completion_tokens": "x"}) == {}
    assert llm_trace.usage_details(None) == {}


def test_assistant_tool_calls_sit_in_the_request_trace_as_tools(spans):
    with llm_trace.unit("assistant", seed="work:9", surface="assistant", owner=OWNER, session="conv-1"):
        with llm_trace.call(feature="assistant", model="gpt-5.6-luna", input=[]) as traced:
            traced.done(output={"tool_calls": [{"name": "record_overview"}]})
        with llm_trace.step("record_overview", input={"record_id": "r1"}) as traced:
            traced.done(output={"title": "ABC"})
    (model, generation), (tool, step) = spans()
    assert model.context.trace_id == tool.context.trace_id
    assert step["langfuse.observation.type"] == "tool" and tool.name == "record_overview"
    assert "langfuse.observation.model.name" not in step
    assert generation["session.id"] == "conv-1" and set(generation["langfuse.trace.tags"]) == {"assistant"}
