"""The per-meeting model cost computation in evals/context/bench.py (package F2): the native extractor runs on a
synthetic meeting behind a scripted model (no network), finds every planted item, and the tallied tokens price
exactly as the extractor's own accounting."""

import pytest
from jarvis.context import sources
from jarvis.context.native import extract
from jarvis.context.schema import ExtractRequest
from jarvis.model_map import estimate_tokens

from evals.context.bench import model_cost
from evals.context.corpus import meeting


def test_model_cost_prices_the_native_extractor_passes_for_a_meeting(monkeypatch):
    def offline(*args, **kwargs):
        raise AssertionError("model_cost must not reach the provider")

    original = extract.call_model
    monkeypatch.setattr(extract.memory_learning, "extraction_request", offline)
    envelope = meeting(20)
    segments = sources.build_segments(envelope)
    planted = sum(s.text.count("I'll ") + s.text.startswith("Decision:") for s in segments)
    request = ExtractRequest(owner_id="o", source_revision_id="r", occurred_at=envelope.occurred_at,
                             timezone=envelope.timezone, title=envelope.title, participants=envelope.participants,
                             segments=segments)
    cost = model_cost(request, ["first chunk text", "a statement"])
    assert extract.call_model is original  # restored
    assert cost["assertions"] == planted and cost["partial"] is False
    passes = cost["passes"]
    assert set(passes) == {"context_detect", "context_enrich", "context_verify"}
    assert passes["context_detect"]["calls"] == cost["chunks"] >= 1
    assert all(p["prompt"] > 0 and p["completion"] > 1000 for p in passes.values())
    assert cost["extract_usd"] == pytest.approx(cost["extractor_cost_usd"], abs=1e-5)
    assert 0 < cost["extract_usd"] < 1.0
    assert cost["embeddings"]["tokens"] == estimate_tokens("first chunk text") + estimate_tokens("a statement")
    assert cost["brief"]["prompt"] > 0 and cost["reconcile"]["usd"] > 0
