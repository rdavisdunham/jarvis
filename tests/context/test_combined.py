"""Combined extraction mode: one whole-meeting call returns the assertions and the action items, one verify call
covers both, and every deterministic check is the split mode's. Recorded answers in each provider's wire shape
(OpenAI Responses for Luna, Anthropic Messages for Sonnet 5.5); every model call is mocked at the HTTP transport."""
# ruff: noqa: F811  (the imported `wire` fixture is a test argument)

import json
from decimal import Decimal

import pytest
from jarvis import model_map
from jarvis.config import Settings, get_settings
from jarvis.context.native import actions, extract
from jarvis.context.native.extract import NativeExtractor, extraction_mode
from jarvis.context.native.prompts import (
    _ACTION_VERDICT_RULES,
    _ACTIONS_BODY,
    _VERIFY_RULES,
    ACTIONS_PROMPT,
    ACTIONS_VERIFY_PROMPT,
    COMBINED_PROMPT,
    COMBINED_PROMPT_VERSION,
    COMBINED_VERIFY_PROMPT,
    DETECT_PROMPT,
    VERIFY_PROMPT,
)
from jarvis.context.schema import check_extraction
from jarvis.db import session_scope
from jarvis.models import BudgetReservation, Usage
from sqlalchemy import select

from .test_actions import (
    ANSWERS,
    FIXTURE,
    USAGE,
    meeting_request,
    messages_reply,
    system_of,
    use,
    user_of,
    wire,  # noqa: F401  (the recorded-answer transport fixture)
)


def mention(entity_type: str, text: str, candidate: str | None = None) -> dict:
    return {"entity_type": entity_type, "mention": text, "candidate_id": candidate}


def assertion(kind: str, statement: str, *quotes: tuple[str, str], modality: str = "asserted",
              subject: dict | None = None, valid_from: dict | None = None) -> dict:
    return {"kind": kind, "statement": statement, "subject": subject, "object": None, "key": None,
            "modality": modality, "inferred": False, "responsible": [], "responsibility": "none", "recipient": None,
            "valid_from": valid_from, "valid_to": None,
            "evidence": [{"line": line, "quote": quote} for line, quote in quotes], "confidence": 0.8}


ASSERTIONS = [
    # kept: a fact with its subject linked to a listed candidate
    assertion("fact", "The Q4 rubric scores Responsiveness as a yes or no.",
              ("1", "The Q4 rubric still scores Responsiveness as a yes or no.")),
    # kept: a decision in its positive form, negation carried by modality
    assertion("decision", "The team redoes the January batch.",
              ("12", "we're not going to redo the January batch, that's settled."), modality="negated"),
    # kept: a preference cited from a private note and a line, linked to Rowan
    assertion("preference", "Rowan Vale wants to assign specific calls to specific reviewers by name.",
              ("4", "I want to be able to hand specific calls to specific reviewers, like assign them by name."),
              ("N1", "Rowan - reviewer assignment: pick calls per reviewer"),
              subject=mention("actor", "Rowan Vale", "actor:act_rowan")),
    # refused by the verifier: not entailed
    assertion("fact", "The old export column names confused the whole company.",
              ("7", "the export is still showing the old column names."),
              ("8", "that confused the team last week.")),
    # refused deterministically: a number the quotes don't contain
    assertion("fact", "The pilot starts with 10 reviewers.", ("6", "Maybe six to start.")),
    # refused deterministically: a line that does not exist
    assertion("fact", "The vendor contract is with legal.", ("99", "the vendor contract")),
    # refused deterministically: a quote not in its line
    assertion("risk", "The importer may break.", ("11", "the importer will break")),
    # refused deterministically: an asserted reading of a hypothetical
    assertion("fact", "Jules rebuilds the importer from scratch.",
              ("11", "If we were to move everything to the new vendor, I'd rebuild the importer from scratch.")),
]
VERDICTS = [
    {"item": 0, "entailed": True, "modality": "asserted", "attribution_ok": True, "reason": ""},
    {"item": 1, "entailed": True, "modality": "negated", "attribution_ok": True, "reason": ""},
    {"item": 2, "entailed": True, "modality": "asserted", "attribution_ok": True, "reason": ""},
    {"item": 3, "entailed": False, "modality": "asserted", "attribution_ok": True,
     "reason": "the quote says the team, not the company"},
]
COMBINED = {"actions": FIXTURE["context_actions"]["items"], "assertions": ASSERTIONS}
COMBINED_VERDICTS = {"action_verdicts": FIXTURE["context_actions_verify"]["verdicts"],
                     "assertion_verdicts": VERDICTS}
# The split mode's expected list (test_actions): the same checks keep the same items.
EXPECTED_ACTIONS = [
    ("create", "Update the Q4 rubric to a 1-5 scale, fix the N/A rows and re-import it", ["act_me"], False),
    ("create", "Build reviewer assignment for specific calls", ["act_me"], True),
    ("create", "Pull fifty calls for the pilot batch", ["act_rowan"], True),
    ("create", "Tell Rowan when Q4 is live in the tool", ["act_me"], True),
    ("create", "Rename the old column names in the export", ["act_me"], False)]
SPLIT_ACTION_REJECTIONS = ["done_in_meeting", "duplicate", "hypothetical", "negated", "not_a_follow_up",
                           "owner_rejected", "quote_not_found", "ungrounded_number"]


@pytest.fixture
def combined(monkeypatch):
    monkeypatch.setattr(extract, "EXTRACTION_MODE", "combined")
    monkeypatch.setitem(ANSWERS, COMBINED_PROMPT, COMBINED)
    monkeypatch.setitem(ANSWERS, COMBINED_VERIFY_PROMPT, COMBINED_VERDICTS)


@pytest.mark.parametrize("profile", ["luna", "sonnet-5.5"])
def test_one_call_returns_both_lists_and_one_verify_checks_them(wire, monkeypatch, combined, profile):
    sent = wire()
    use(monkeypatch, profile)
    request = meeting_request()
    result = NativeExtractor().extract(request)

    assert check_extraction(request, result) == []
    assert [system_of(body) for _, _, body in sent] == [COMBINED_PROMPT, COMBINED_VERIFY_PROMPT]
    got = [(a.kind, a.title, [r.entity_id for r in a.responsible], bool(a.assertion_indexes)) for a in result.actions]
    assert got == EXPECTED_ACTIONS
    assert [a.statement for a in result.assertions] == [
        "The Q4 rubric scores Responsiveness as a yes or no.",
        "The team redoes the January batch.",
        "Rowan Vale wants to assign specific calls to specific reviewers by name.",
        "Jules will build reviewer assignment for specific calls.",
        "Rowan will pull fifty calls for the pilot batch by Friday.",
        "Jules Park will tell Rowan when Q4 is live in the tool."]
    decision, preference = result.assertions[1], result.assertions[2]
    assert decision.modality == "negated"
    assert {e.segment_seq for e in preference.evidence} == {4, 14}  # N1 is segment 14
    assert preference.subject.entity_id == "act_rowan"
    assert ("Rowan Vale", "act_rowan") in {(link.mention, link.candidate.entity_id) for link in result.links}

    d = result.diagnostics
    assert d["extraction_mode"] == "combined" and d["partial"] is False
    assert set(result.coverage.values()) <= {"extracted", "no_content"}
    assert result.coverage[1] == "extracted" and result.coverage[0] == "no_content"
    assert sorted(r["code"] for r in d["actions"]["rejected"]) == SPLIT_ACTION_REJECTIONS
    assert sorted(r["code"] for r in d["rejected"]) == sorted(SPLIT_ACTION_REJECTIONS + [
        "hypothetical_as_asserted", "not_entailed", "quote_not_found", "ungrounded_number", "unknown_segment"])
    assert all(c["calls"] == 0 for c in d["chunks"])  # no per-chunk passes

    # What the model saw: the action pass's payload plus the entity candidates; then both lists for review.
    user = user_of(sent[0][2])
    assert user["recorded_by"] == "Jules Park" and user["private_notes"].startswith("Typed by Jules Park")
    assert {c["id"] for c in user["entity_candidates"]} == {"actor:act_me", "actor:act_rowan"}
    review = user_of(sent[1][2])
    assert len(review["actions"]) == 7 and len(review["assertions"]) == 4
    assert review["assertions"][2]["evidence"][1] == {
        "line": "N1", "speaker": "Jules Park", "quote": "Rowan - reviewer assignment: pick calls per reviewer"}
    assert "[1] Rowan Vale: The Q4 rubric" in review["context"]
    assert "[N2] Private note: Rubric Q4" in review["context"]

    # Both calls route and price as the context_actions job.
    spec, agent = model_map.PROFILES[profile], model_map.costing(profile)
    cached = 1000 if spec.api == "anthropic_messages" else 0
    one = agent.usage_cost({"prompt_tokens": USAGE[0], "completion_tokens": USAGE[1],
                            "prompt_tokens_details": {"cached_tokens": cached, "cache_write_tokens": 0}})
    assert d["model"] == spec.model and d["cost_usd"] == pytest.approx(2 * one, abs=1e-6)
    assert d["combined"] == {
        "prompt_version": COMBINED_PROMPT_VERSION, "profile": profile, "model": spec.model, "status": "ok",
        "calls": 2, "cost_usd": pytest.approx(2 * one, abs=1e-6), "latency_ms": d["actions"]["latency_ms"],
        "assertions": 3, "action_items": 5}
    assert (d["actions"]["calls"], d["actions"]["mode"], d["actions"]["profile"]) == (2, "whole", profile)
    body = sent[0][2]
    if spec.api == "anthropic_messages":
        assert body["max_tokens"] == 40000 and body["system"] == COMBINED_PROMPT
        schema = json.dumps(body["output_config"]["format"]["schema"])
        assert "maxLength" not in schema and "maxItems" not in schema
    else:
        assert body["text"]["format"]["name"] == "context_combined" and body["text"]["format"]["strict"] is True
    with session_scope() as db:
        rows = [u for u in db.scalars(select(Usage)) if u.model == spec.model]
        assert len(rows) == 2
        for row in rows:
            assert row.amount == pytest.approx(Decimal(str(agent.usage_cost(row.tokens))))
        assert {r.state for r in db.scalars(select(BudgetReservation))} == {"closed"}


def test_split_mode_stays_the_default_flow(wire, monkeypatch):
    monkeypatch.setitem(ANSWERS, COMBINED_PROMPT, COMBINED)
    sent = wire()
    result = NativeExtractor().extract(meeting_request())
    systems = [system_of(body) for _, _, body in sent]
    assert systems[:2] == [ACTIONS_PROMPT, ACTIONS_VERIFY_PROMPT] and DETECT_PROMPT in systems
    assert COMBINED_PROMPT not in systems
    assert result.diagnostics["extraction_mode"] == "split" and "combined" not in result.diagnostics
    assert [a.title for a in result.actions] == [t for _, t, _, _ in EXPECTED_ACTIONS]


def test_the_mode_comes_from_the_lab_override_then_the_setting(monkeypatch):
    assert extraction_mode() == "split"
    monkeypatch.setattr(get_settings(), "context_extraction_mode", "combined")
    assert extraction_mode() == "combined"
    monkeypatch.setattr(extract, "EXTRACTION_MODE", "split")
    assert extraction_mode() == "split"
    monkeypatch.setattr(extract, "EXTRACTION_MODE", "both")
    with pytest.raises(ValueError, match="Unknown extraction mode"):
        extraction_mode()
    monkeypatch.setenv("JARVIS_CONTEXT_EXTRACTION_MODE", "combined")
    assert Settings(_env_file=None).context_extraction_mode == "combined"
    monkeypatch.setenv("JARVIS_CONTEXT_EXTRACTION_MODE", "chunked")
    with pytest.raises(ValueError):
        Settings(_env_file=None)


def test_a_failed_or_unaffordable_combined_call_leaves_the_source_partial(wire, monkeypatch, combined):
    use(monkeypatch, "sonnet-5.5")
    request = meeting_request()
    wire(lambda body: (200, messages_reply(body, "max_tokens")))
    result = NativeExtractor().extract(request)
    assert result.actions == [] and result.assertions == [] and set(result.coverage.values()) == {"failed"}
    assert result.diagnostics["actions"]["reason"] == "context_combined: truncated"
    assert result.diagnostics["combined"]["status"] == "failed" and result.diagnostics["partial"] is True
    assert {c["status"] for c in result.diagnostics["chunks"]} == {"failed"}

    sent = wire()
    sent.clear()  # the fixture keeps one list of everything sent
    result = NativeExtractor(budget_usd=0.01).extract(request)
    assert sent == [] and set(result.coverage.values()) == {"skipped"}
    assert result.diagnostics["actions"]["reason"] == "source_budget"


def test_a_failed_verify_call_keeps_nothing(wire, monkeypatch, combined):
    def answer(body):
        if body["system"] == COMBINED_VERIFY_PROMPT:
            return 200, {**messages_reply(body), "content": [{"type": "text", "text": "{}"}]}
        return 200, messages_reply(body)
    use(monkeypatch, "sonnet-5.5")
    wire(answer)
    result = NativeExtractor().extract(meeting_request())
    assert result.actions == [] and result.assertions == []
    assert result.diagnostics["actions"]["reason"] == "context_combined_verify: invalid output"
    assert result.diagnostics["combined"]["calls"] == 2


def test_long_meetings_use_the_action_pass_chunks_and_one_verify(wire, monkeypatch, combined):
    sent = wire()
    monkeypatch.setitem(ANSWERS, COMBINED_VERIFY_PROMPT, {"action_verdicts": [], "assertion_verdicts": []})
    monkeypatch.setattr(actions, "WHOLE_MEETING_TOKENS", 100)
    monkeypatch.setattr(actions, "CHUNK_TOKENS", 150)
    request = meeting_request()
    result = NativeExtractor().extract(request)

    calls = [user_of(body) for _, _, body in sent if system_of(body) == COMBINED_PROMPT]
    verifies = [body for _, _, body in sent if system_of(body) == COMBINED_VERIFY_PROMPT]
    assert len(calls) > 1 and len(verifies) == 1 and "context_before" in calls[1]
    assert all("entity_candidates" in c for c in calls)
    d = result.diagnostics
    assert d["actions"]["mode"] == "chunked" and d["combined"]["calls"] == len(calls) + 1
    titles = [a.title for a in result.actions]
    assert len(titles) == len(set(titles)) and "Pull fifty calls for the pilot batch" in titles
    # Assertions the verifier skipped are refused as unverified, exactly as in split mode.
    assert result.assertions == [a for a in result.assertions if a.kind == "commitment"]
    assert d["rejection_counts"]["unverified"] >= 3
    assert d["rejection_counts"]["evidence_outside_chunk"] >= 1
    assert check_extraction(request, result) == []


def test_combined_prompts_reuse_the_split_rules():
    assert ACTIONS_PROMPT.endswith(_ACTIONS_BODY) and _ACTIONS_BODY in COMBINED_PROMPT
    assert VERIFY_PROMPT.endswith(_VERIFY_RULES) and ACTIONS_VERIFY_PROMPT.endswith(_ACTION_VERDICT_RULES)
    assert _VERIFY_RULES in COMBINED_VERIFY_PROMPT and _ACTION_VERDICT_RULES in COMBINED_VERIFY_PROMPT
    assert "ENTITY_CANDIDATES" in COMBINED_PROMPT and '"N1"' in COMBINED_PROMPT
