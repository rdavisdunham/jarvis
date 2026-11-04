"""NativeExtractor end to end with the provider mocked at the HTTP transport (no network): recorded responses
for the short meeting, a scripted model for the long one, budget exhaustion, failures and cost accounting."""

import json
from datetime import date
from decimal import Decimal
from pathlib import Path

import httpx
import pytest
from jarvis import memory_learning
from jarvis.config import get_settings
from jarvis.context.engine import Extractor
from jarvis.context.native import extract as native
from jarvis.context.native.extract import NativeExtractor
from jarvis.context.native.prompts import PROMPT_VERSION
from jarvis.context.schema import EntityCandidate, EntityRef, ExtractLimits, check_extraction
from jarvis.db import session_scope
from jarvis.domain import DomainError
from jarvis.models import BudgetReservation, Usage
from sqlalchemy import select

from .fixtures.extraction import long_meeting

RECORDED = json.loads((Path(__file__).parent / "fixtures/extraction/pilot_meeting.json").read_text())


@pytest.fixture
def provider(monkeypatch):
    """install(handler): route provider_request's HTTP calls to handler(body) -> (status, json). Returns the
    list of request bodies sent."""
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


def recorded(body):
    text = json.dumps(RECORDED[body["text"]["format"]["name"]])
    return 200, {"id": "resp_rec", "status": "completed", "service_tier": "default",
                 "usage": {"input_tokens": 2000, "output_tokens": 600},
                 "output": [{"type": "message", "content": [{"type": "output_text", "text": text}]}]}


@pytest.fixture
def pilot_request(extract_request):
    return extract_request.model_copy(update={"candidates": [
        EntityCandidate(ref=EntityRef(entity_type="project", entity_id="proj_pilot"), name="Clinic pilot"),
        EntityCandidate(ref=EntityRef(entity_type="actor", entity_id="act_priya"), name="Priya")]})


def test_native_extractor_satisfies_the_protocol():
    assert isinstance(NativeExtractor(), Extractor)


def test_recorded_meeting_yields_validated_assertions_actions_and_usage(provider, pilot_request):
    sent = provider(recorded)
    result = NativeExtractor().extract(pilot_request)

    assert check_extraction(pilot_request, result) == []
    assert [(a.kind, a.statement) for a in result.assertions] == [
        ("fact", "The pilot covers 40 clinics in the first wave."),
        ("decision", "The pilot launches on Postgres with pgvector."),
        ("commitment", "Priya sends the revised budget by 2026-11-20."),
        ("commitment", "Marco drafts the rollout checklist.")]
    fact, decision, priya, marco = result.assertions
    assert fact.subject == EntityRef(entity_type="project", entity_id="proj_pilot", mention="the pilot")
    assert decision.evidence[0].quote == "we launch the pilot on Postgres with pgvector."  # normalized, then exact
    assert priya.responsible == [EntityRef(entity_type="actor", entity_id="act_priya", mention="Priya")]
    assert (priya.due.at.date(), priya.due.expression) == (date(2026, 11, 20), "by 2026-11-20")
    assert marco.responsible == [EntityRef(entity_type="actor", mention="Marco")]  # unknown id stays unresolved
    assert [(x.kind, x.assertion_indexes) for x in result.actions] == [("create", [2]), ("create", [3])]
    assert {(link.mention, link.candidate.entity_id) for link in result.links} == {
        ("the pilot", "proj_pilot"), ("Priya", "act_priya")}
    assert result.coverage == {0: "no_content", 1: "extracted", 2: "extracted", 3: "extracted", 4: "extracted"}

    diagnostics = result.diagnostics
    assert diagnostics["rejection_counts"] == {"attribution_mismatch": 1, "duplicate": 1, "no_promise_language": 1,
                                               "not_entailed": 1, "ungrounded_number": 1}
    assert diagnostics["partial"] is False and diagnostics["prompt_version"] == PROMPT_VERSION
    assert diagnostics["derivations"]["2"][0]["verified"] is True

    assert [b["text"]["format"]["name"] for b in sent] == ["context_detect", "context_enrich", "context_verify"]
    assert all(b["text"]["format"]["strict"] and b["model"] == memory_learning.EXTRACTION_MODEL for b in sent)
    assert "untrusted DATA" in sent[0]["input"][0]["content"]
    assert "[3] Priya: Sounds right." in json.loads(sent[0]["input"][1]["content"])["lines"]
    assert json.loads(sent[1]["input"][1]["content"])["entity_candidates"][0]["id"] == "project:proj_pilot"
    with session_scope() as db:
        usage = db.scalars(select(Usage)).all()
        assert len(usage) == 3 and {u.tokens["feature"] for u in usage} == {"context_extraction"}
        assert sum(u.amount for u in usage) == Decimal(str(diagnostics["cost_usd"]))
        assert {r.state for r in db.scalars(select(BudgetReservation))} == {"closed"}


def test_long_meeting_full_coverage_and_cost_per_meeting(provider):
    request = long_meeting.request()
    model = long_meeting.ScriptedModel()
    provider(model)
    result = NativeExtractor().extract(request)

    assert check_extraction(request, result) == []
    assert set(result.coverage) == {s.seq for s in request.segments}
    assert set(result.coverage.values()) == {"extracted", "no_content"}
    assert result.diagnostics["partial"] is False and result.diagnostics["uncovered_segments"] == []
    commitments, decisions = long_meeting.expected_counts()
    kinds = [a.kind for a in result.assertions]
    assert (kinds.count("commitment"), kinds.count("decision")) == (commitments, decisions)
    assert {a.due.at.date() for a in result.assertions if a.kind == "commitment"} == {date(2026, 11, 6)}
    assert any(a.evidence[0].segment_seq == long_meeting.LONG_AT for a in result.assertions)
    unknown = [a for a in result.assertions if a.responsible and a.responsible[0].mention == "Unknown speaker"]
    assert unknown and all(request.segments[a.evidence[0].segment_seq].speaker_label is None for a in unknown)
    assert result.diagnostics["rejection_counts"] == {}

    chunks = result.diagnostics["chunks"]
    cost = result.diagnostics["cost_usd"]
    assert len(model.calls) == sum(c["calls"] for c in chunks) and cost < 0.25
    assert {c["calls"] for c in chunks} == {1, 3}  # chunks with no candidates stop after detection
    print(f"\nlong fixture: {len(request.segments)} segments, {len(chunks)} chunks, {len(model.calls)} calls, "
          f"{result.diagnostics['usage']['prompt_tokens']} prompt + "
          f"{result.diagnostics['usage']['completion_tokens']} completion tokens, ${cost:.4f}")


def test_source_budget_exhaustion_marks_partial_and_lists_uncovered_segments(provider):
    request = long_meeting.request()
    model = long_meeting.ScriptedModel()
    provider(model)
    result = NativeExtractor(budget_usd=0.03).extract(request)

    assert check_extraction(request, result) == []
    statuses = [c["status"] for c in result.diagnostics["chunks"]]
    assert statuses[0] == "ok" and statuses[-1] == "skipped"
    assert statuses == sorted(statuses, key=lambda s: s == "skipped")  # once skipped, the rest are skipped
    skipped = sorted(seq for seq, status in result.coverage.items() if status == "skipped")
    assert result.diagnostics["partial"] is True and result.diagnostics["uncovered_segments"] == skipped
    assert set(result.diagnostics["skipped"].values()) == {"source_budget"}
    assert result.diagnostics["cost_usd"] <= 0.03
    assert all(e.segment_seq not in skipped for a in result.assertions for e in a.evidence)


def test_owner_budget_refusal_skips_everything(monkeypatch, extract_request):
    def refuse(*args, **kwargs):
        raise DomainError("BUDGET_EXCEEDED", "budget", 402)
    monkeypatch.setattr(native, "call_model", refuse)
    result = NativeExtractor().extract(extract_request)
    assert check_extraction(extract_request, result) == []
    assert set(result.coverage.values()) == {"skipped"} and result.diagnostics["partial"] is True
    assert set(result.diagnostics["skipped"].values()) == {"owner_budget"}


def test_failed_chunk_is_marked_and_later_chunks_still_run(provider):
    request = long_meeting.request()
    provider(long_meeting.ScriptedModel(fail={"context_detect": {2}}))
    result = NativeExtractor().extract(request)

    assert check_extraction(request, result) == []
    chunks = result.diagnostics["chunks"]
    assert [c["status"] for c in chunks].count("failed") == 1 and chunks[1]["status"] == "failed"
    assert chunks[1]["reason"] == "context_detect: HTTPStatusError" and chunks[2]["status"] == "ok"
    failed = {seq for seq, status in result.coverage.items() if status == "failed"}
    assert failed and result.diagnostics["uncovered_segments"] == sorted(failed)
    with session_scope() as db:  # the failed call's outcome is unknown: its reservation stays held
        assert "uncertain" in {r.state for r in db.scalars(select(BudgetReservation))}


def test_unusable_model_output_fails_the_chunk(provider, extract_request):
    provider(lambda body: (200, {"id": "r", "status": "completed", "usage": {"input_tokens": 10, "output_tokens": 5},
                                 "output": [{"type": "message", "content": [
                                     {"type": "output_text", "text": "not json"}]}]}))
    result = NativeExtractor().extract(extract_request)
    assert set(result.coverage.values()) == {"failed"}
    assert result.diagnostics["chunks"][0]["reason"] == "context_detect: invalid output"


def test_request_limits_cap_assertions_and_actions(provider, pilot_request):
    provider(recorded)
    limited = pilot_request.model_copy(update={"limits": ExtractLimits(max_assertions=3, max_actions=0)})
    result = NativeExtractor().extract(limited)
    assert check_extraction(limited, result) == []
    assert len(result.assertions) == 3 and result.actions == []
    assert result.diagnostics["rejection_counts"]["limit"] == 2  # Marco's assertion; Priya's action
