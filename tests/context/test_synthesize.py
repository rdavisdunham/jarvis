"""NativeSynthesizer with the provider mocked at the HTTP transport: grounded claims, usage under the
context_synthesis feature, the budget cap and failures."""

import json

import httpx
import pytest
from jarvis import memory_learning
from jarvis.config import get_settings
from jarvis.context.engine import Synthesizer
from jarvis.context.native.synthesize import NativeSynthesizer, SynthesisFailed, ground
from jarvis.context.schema import (
    AssertionView,
    BriefClaim,
    BriefRequest,
    BriefSection,
    EntityRef,
    ScopePredicate,
    SynthesizeRequest,
    TaskView,
)
from jarvis.db import session_scope
from jarvis.models import Usage
from sqlalchemy import select

from .conftest import MEETING_AT

OWNER = "davin"


@pytest.fixture
def provider(monkeypatch):
    """install(handler): route provider_request's HTTP calls to handler(body) -> (status, json)."""
    original = httpx.Client
    monkeypatch.setattr(get_settings(), "openai_api_key", "synthetic")
    sent = []

    def install(handler):
        def respond(request):
            body = json.loads(request.content)
            sent.append(body)
            status, data = handler(body)
            return httpx.Response(status, json=data)
        transport = httpx.MockTransport(respond)
        monkeypatch.setattr(memory_learning.httpx, "Client", lambda **kw: original(transport=transport))
        return sent
    return install


def answer(sections):
    text = json.dumps({"sections": sections})
    return lambda body: (200, {"id": "resp_syn", "status": "completed", "service_tier": "default",
                               "usage": {"input_tokens": 1500, "output_tokens": 400},
                               "output": [{"type": "message", "content": [{"type": "output_text", "text": text}]}]})


@pytest.fixture
def request_(history):
    pilot = EntityRef(entity_type="project", entity_id="proj_pilot", mention="Clinic pilot")
    current = [a for a in history if a.id != "asr_1"]
    superseded = history[0].model_copy(update={"lifecycle": "superseded"})
    retracted = AssertionView.model_validate(
        history[2].model_dump() | {"id": "asr_gone", "lifecycle": "retracted"})
    return SynthesizeRequest(
        brief=BriefRequest(scope=ScopePredicate(owner_id=OWNER, entities=[pilot]), entity=pilot,
                           sections=["current_state_and_why", "open_loops", "recent_changes", "unknowns"]),
        assertions=current, history=[superseded, retracted],
        tasks=[TaskView(id="task_budget", revision=2, title="Revised budget", status="completed",
                        due_at=MEETING_AT, assertion_ids=["asr_2"])])


def test_native_synthesizer_satisfies_the_protocol():
    assert isinstance(NativeSynthesizer(), Synthesizer)


def test_claims_are_grounded_and_usage_is_recorded_under_context_synthesis(provider, request_):
    sent = provider(answer([
        {"name": "current_state_and_why", "gaps": [], "claims": [
            {"text": "The pilot launches on Postgres with pgvector.", "assertion_ids": ["asr_4"], "task_ids": []},
            {"text": "The pilot has 41 clinics.", "assertion_ids": ["asr_404"], "task_ids": []},
            {"text": "Uncited.", "assertion_ids": [], "task_ids": []}]},
        {"name": "open_loops", "gaps": [], "claims": [
            {"text": "Priya owes the revised budget by 2026-11-20; its task is done.",
             "assertion_ids": ["asr_2"], "task_ids": ["task_budget"]},
            {"text": "Old due date.", "assertion_ids": ["asr_1"], "task_ids": []}]},
        {"name": "recent_changes", "gaps": [], "claims": [
            {"text": "The budget moved from 11-13 to 11-20.", "assertion_ids": ["asr_1", "asr_2"], "task_ids": []},
            {"text": "Withdrawn item.", "assertion_ids": ["asr_gone"], "task_ids": []}]},
        {"name": "who_is_who", "gaps": [], "claims": [
            {"text": "Not requested.", "assertion_ids": ["asr_2"], "task_ids": []}]},
    ]))
    sections = NativeSynthesizer().synthesize(request_)

    assert [s.name for s in sections] == ["current_state_and_why", "open_loops", "recent_changes", "unknowns"]
    state, loops, changes, unknowns = sections
    assert [c.assertion_ids for c in state.claims] == [["asr_4"]]
    assert state.gaps == ["2 statement(s) were left out because they did not cite surviving evidence."]
    assert [(c.assertion_ids, c.task_ids) for c in loops.claims] == [(["asr_2"], ["task_budget"])]  # history only in changes
    assert [c.assertion_ids for c in changes.claims] == [["asr_1", "asr_2"]]  # retracted history is not citable
    assert unknowns.claims == [] and unknowns.gaps == []

    prompt = json.loads(sent[0]["input"][1]["content"])
    assert [a["id"] for a in prompt["ASSERTIONS"]] == ["asr_2", "asr_3", "asr_4"]
    assert prompt["SUBJECT"]["name"] == "Clinic pilot" and "untrusted DATA" in sent[0]["input"][0]["content"]
    assert sent[0]["text"]["format"]["name"] == "context_brief"
    with session_scope() as db:
        usage = db.scalars(select(Usage)).all()
        assert len(usage) == 1 and usage[0].tokens["feature"] == "context_synthesis"


def test_budget_cap_refuses_before_calling(provider, request_):
    sent = provider(answer([]))
    with pytest.raises(SynthesisFailed, match="source_budget"):
        NativeSynthesizer(budget_usd=0.0).synthesize(request_)
    assert sent == []


@pytest.mark.parametrize("handler, reason", [
    (lambda body: (500, {"error": "down"}), "HTTPStatusError"),
    (lambda body: (200, {"id": "r", "status": "completed", "service_tier": "default",
                         "usage": {"input_tokens": 10, "output_tokens": 5},
                         "output": [{"type": "message", "content": [{"type": "output_text", "text": "{not json"}]}]}),
     "invalid output"),
])
def test_failures_raise_synthesis_failed(provider, request_, handler, reason):
    provider(handler)
    with pytest.raises(SynthesisFailed, match=reason):
        NativeSynthesizer().synthesize(request_)


def test_empty_requests_make_no_call_and_ground_fills_every_requested_section(provider, request_):
    sent = provider(answer([]))
    empty = request_.model_copy(update={"assertions": [], "tasks": []})
    assert [s.claims for s in NativeSynthesizer().synthesize(empty)] == [[], [], [], []]
    assert sent == []
    merged = ground([BriefSection(name="unknowns", claims=[BriefClaim(text="a", assertion_ids=["asr_3"])]),
                     BriefSection(name="unknowns", gaps=["Who signs off?", "Who signs off?"])], request_)
    assert merged[3].claims[0].text == "a" and merged[3].gaps == ["Who signs off?"]
