"""The whole-meeting action pass on a synthetic meeting that reproduces the first real meeting's misses, with
recorded answers in each provider's wire shape (OpenAI Responses for Luna and Terra, Anthropic Messages for
Sonnet 5.5). Every model call is mocked at the HTTP transport; no network."""

import json
from datetime import date
from decimal import Decimal
from pathlib import Path

import httpx
import pytest
from jarvis import memory_learning, model_map
from jarvis.config import get_settings
from jarvis.context.native import actions
from jarvis.context.native.extract import NativeExtractor
from jarvis.context.native.prompts import ACTIONS_PROMPT, ACTIONS_VERIFY_PROMPT, PROMPT_VERSION
from jarvis.context.native.validate import SourceContext, ungrounded_numbers
from jarvis.context.schema import (
    Decision,
    EntityCandidate,
    EntityRef,
    ExtractRequest,
    Participant,
    Segment,
    check_extraction,
)
from jarvis.db import session_scope
from jarvis.models import BudgetReservation, Usage
from sqlalchemy import select

FIXTURE = json.loads((Path(__file__).parent / "fixtures/extraction/actions_meeting.json").read_text())
ANSWERS = {ACTIONS_PROMPT: FIXTURE["context_actions"], ACTIONS_VERIFY_PROMPT: FIXTURE["context_actions_verify"]}
USAGE = (6000, 1500)  # input, output tokens per recorded answer


def meeting_request(recorder: str = "Jules Park", **update) -> ExtractRequest:
    segments = [Segment(**{**s, "attribution": ("owner" if s.get("speaker_label") == recorder else "other")
                           if s["kind"] == "transcript" else "owner"}) for s in FIXTURE["segments"]]
    return ExtractRequest(owner_id="davin", source_revision_id="srcrev_actions", occurred_at=FIXTURE["occurred_at"],
                          timezone=FIXTURE["timezone"], title=FIXTURE["title"],
                          participants=[Participant(**p) for p in FIXTURE["participants"]], segments=segments,
                          candidates=[EntityCandidate(**c) for c in FIXTURE["candidates"]], **update)


def system_of(body: dict) -> str:
    return body.get("system") or body["input"][0]["content"]


def user_of(body: dict) -> dict:
    return json.loads(body["messages"][0]["content"] if "system" in body else body["input"][1]["content"])


def answer_for(system: str) -> dict:
    """The recorded answer for a pass, by its system prompt; the chunk passes find nothing."""
    if system in ANSWERS:
        return ANSWERS[system]
    return {"candidates": []}


def responses_reply(body: dict) -> dict:
    text = json.dumps(answer_for(body["input"][0]["content"]))
    return {"id": "resp_rec", "object": "response", "status": "completed", "service_tier": "default",
            "model": body["model"], "usage": {"input_tokens": USAGE[0], "output_tokens": USAGE[1],
                                              "input_tokens_details": {"cached_tokens": 0}},
            "output": [{"type": "reasoning", "summary": []},
                       {"type": "message", "content": [{"type": "output_text", "text": text}]}]}


def messages_reply(body: dict, stop: str = "end_turn") -> dict:
    text = json.dumps(answer_for(body["system"]))
    return {"id": "msg_rec", "type": "message", "role": "assistant", "model": body["model"],
            "content": [{"type": "thinking", "thinking": "", "signature": "sig"}, {"type": "text", "text": text}],
            "stop_reason": stop, "stop_details": None,
            "usage": {"input_tokens": USAGE[0] - 1000, "cache_read_input_tokens": 1000,
                      "cache_creation_input_tokens": 0, "output_tokens": USAGE[1]}}


@pytest.fixture
def wire(monkeypatch):
    """install(anthropic=handler): every provider call is answered from the recorded answers; returns the list of
    (url, headers, body) sent."""
    original = httpx.Client
    settings = get_settings()
    monkeypatch.setattr(settings, "openai_api_key", "synthetic")
    monkeypatch.setattr(settings, "anthropic_api_key", "synthetic-anthropic")
    sent = []

    def install(anthropic=lambda body: (200, messages_reply(body))):
        def respond(request):
            body = json.loads(request.content)
            sent.append((str(request.url), dict(request.headers), body))
            status, data = anthropic(body) if "anthropic" in str(request.url) else (200, responses_reply(body))
            return httpx.Response(status, json=data)
        transport = httpx.MockTransport(respond)
        monkeypatch.setattr(memory_learning.httpx, "Client", lambda **kw: original(transport=transport))
        return sent
    return install


def use(monkeypatch, profile: str) -> None:
    monkeypatch.setitem(model_map.MODEL_MAP, "context_actions", profile)


@pytest.mark.parametrize("profile", ["luna", "terra", "sonnet-5.5"])
def test_synthetic_meeting_recovers_the_missed_patterns(wire, monkeypatch, profile):
    sent = wire()
    use(monkeypatch, profile)
    request = meeting_request()
    result = NativeExtractor().extract(request)

    assert check_extraction(request, result) == []
    me, rowan = "act_me", "act_rowan"
    got = [(a.kind, a.title, [r.entity_id for r in a.responsible], bool(a.assertion_indexes)) for a in result.actions]
    assert got == [
        # implied intent ("I wanna fix those") folded with the rubric change and the private note
        ("create", "Update the Q4 rubric to a 1-5 scale, fix the N/A rows and re-import it", [me], False),
        # a request at line 4, accepted at line 9
        ("create", "Build reviewer assignment for specific calls", [me], True),
        # someone else's commitment, with its due date
        ("create", "Pull fifty calls for the pilot batch", [rowan], True),
        # an identifier (Q4) and a range (1 to 5) grounded in the cited line
        ("create", "Tell Rowan when Q4 is live in the tool", [me], True),
        # work implied by the discussion
        ("create", "Rename the old column names in the export", [me], False)]
    rubric, assignment, fifty = result.actions[:3]
    assert {e.segment_seq for e in rubric.evidence} == {2, 3, 15}  # N2 is segment 15
    assert {e.segment_seq for e in assignment.evidence} == {4, 9, 14}
    assert (fifty.due.at.date(), fifty.due.expression) == (date(2026, 11, 6), "by Friday")
    assert [a.statement for a in result.assertions] == [
        "Jules will build reviewer assignment for specific calls.",
        "Rowan will pull fifty calls for the pilot batch by Friday.",
        "Jules Park will tell Rowan when Q4 is live in the tool."]
    assert result.assertions[0].recipient == EntityRef(entity_type="actor", entity_id=rowan, mention="Rowan Vale")
    assert {(link.mention, link.candidate.entity_id) for link in result.links} >= {("Jules", me), ("Rowan", rowan)}

    diagnostics = result.diagnostics["actions"]
    spec = model_map.PROFILES[profile]
    assert (diagnostics["profile"], diagnostics["model"], diagnostics["prompt_version"]) == \
        (profile, spec.model, PROMPT_VERSION)
    assert (diagnostics["status"], diagnostics["mode"], diagnostics["calls"]) == ("ok", "whole", 2)
    assert [i["owner_basis"] for i in diagnostics["items"]] == [
        "self_assigned_intent", "accepted_request", "stated_promise", "stated_promise", "implied_by_discussion"]
    assert [i["action_index"] for i in diagnostics["items"]] == [0, 1, 2, 3, 4]
    assert diagnostics["items"][0]["evidence"][-1] == {
        "segment": 15, "line": "N2", "speaker": "Jules Park", "quote": "Rubric Q4 -> 1-5, fix N/A rows, re-import"}
    assert sorted(r["code"] for r in diagnostics["rejected"]) == [
        "done_in_meeting", "duplicate", "hypothetical", "negated", "not_a_follow_up", "owner_rejected",
        "quote_not_found", "ungrounded_number"]
    assert all(set(r) == {"code", "reason", "title", "segments"} for r in diagnostics["rejected"])
    assert result.diagnostics["rejection_counts"]["hypothetical"] == 1

    # What the model saw: whole meeting, private notes flagged as the recorder's, people with the recorder.
    action_calls = [(url, headers, body) for url, headers, body in sent if system_of(body) in ANSWERS]
    assert len(action_calls) == 2
    url, headers, body = action_calls[0]
    user = user_of(body)
    assert user["recorded_by"] == "Jules Park" and user["meeting_local_date"] == "2026-11-03"
    assert user["private_notes"].startswith("Typed by Jules Park during the meeting:\n[N1] Rowan - reviewer")
    assert user["lines"].splitlines()[9] == "[9] Jules Park: Cool, I see it. I can get that in, absolutely."
    assert [p.get("recorded_the_meeting") for p in user["people"]] == [True, False]
    verify = user_of(action_calls[1][2])
    assert len(verify["items"]) == 7 and "[N2] Private note: Rubric Q4" in verify["context"]

    if spec.api == "anthropic_messages":
        assert url == "https://api.anthropic.com/v1/messages"
        assert headers["x-api-key"] == "synthetic-anthropic" and headers["anthropic-version"] == "2026-10-21"
        assert body["model"] == "claude-sonnet-5-5" and body["max_tokens"] == 16000
        assert body["output_config"]["effort"] == "low" and body["output_config"]["format"]["type"] == "json_schema"
        schema = json.dumps(body["output_config"]["format"]["schema"])
        assert "maxLength" not in schema and "minimum" not in schema and "maxItems" not in schema
        assert body["output_config"]["format"]["schema"]["additionalProperties"] is False
        assert "temperature" not in body and body["system"] == ACTIONS_PROMPT
    else:
        assert body["model"] == spec.model and body["reasoning"] == {"effort": "low"}
        assert body["text"]["format"]["strict"] is True and body["text"]["format"]["name"] == "context_actions"

    # Each call is reserved and recorded at the routed model's price (the chunk passes stay on Luna).
    agent = model_map.costing(profile)
    cached = 1000 if spec.api == "anthropic_messages" else 0
    one = agent.usage_cost({"prompt_tokens": USAGE[0], "completion_tokens": USAGE[1],
                            "prompt_tokens_details": {"cached_tokens": cached, "cache_write_tokens": 0}})
    assert diagnostics["cost_usd"] == pytest.approx(2 * one, abs=1e-6)
    with session_scope() as db:
        rows = [u for u in db.scalars(select(Usage)) if u.model == spec.model]
        assert len(rows) == (len(sent) if profile == "luna" else 2)
        for row in rows:
            assert row.amount == pytest.approx(Decimal(str(agent.usage_cost(row.tokens))))
        assert {r.state for r in db.scalars(select(BudgetReservation))} == {"closed"}


def test_attended_meeting_resolves_mine_by_identity_not_by_recorder(wire, monkeypatch):
    """Rowan recorded the meeting; the user is Jules (the owner actor's alias). Private notes are Rowan's."""
    sent = wire()
    request = meeting_request(recorder="Rowan Vale")
    result = NativeExtractor().extract(request)

    owners = {a.title: [r.entity_id for r in a.responsible] for a in result.actions}
    assert owners["Pull fifty calls for the pilot batch"] == ["act_rowan"]
    assert owners["Build reviewer assignment for specific calls"] == ["act_me"]
    user = json.loads(sent[0][2]["input"][1]["content"])
    assert user["recorded_by"] == "Rowan Vale" and user["private_notes"].startswith("Typed by Rowan Vale")
    rubric = result.diagnostics["actions"]["items"][0]
    assert rubric["evidence"][-1]["speaker"] == "Rowan Vale"  # a private note is the recorder's


def test_refusal_truncation_and_missing_key_fail_the_pass_and_leave_the_source_partial(wire, monkeypatch):
    use(monkeypatch, "sonnet-5.5")
    request = meeting_request()
    for stop, reason in (("refusal", "context_actions: refusal"), ("max_tokens", "context_actions: truncated")):
        wire(lambda body, stop=stop: (200, {**messages_reply(body, stop), "stop_details":
                                            {"type": "refusal", "category": "cyber"} if stop == "refusal" else None}))
        result = NativeExtractor().extract(request)
        assert result.actions == [] and set(result.coverage.values()) == {"failed"}
        assert result.diagnostics["actions"]["status"] == "failed"
        assert result.diagnostics["actions"]["reason"] == reason
        assert result.diagnostics["partial"] is True
        assert set(result.diagnostics["failed"].values()) == {f"actions: {reason}"}
    monkeypatch.setattr(get_settings(), "anthropic_api_key", "")
    result = NativeExtractor().extract(request)
    assert result.diagnostics["actions"]["reason"] == "context_actions: INTEGRATION_UNAVAILABLE"
    assert result.diagnostics["actions"]["cost_usd"] == 0  # refused before any request was sent


def test_unpriced_profile_is_refused_before_any_call(wire, monkeypatch):
    sent = wire()
    use(monkeypatch, "deepseek")
    result = NativeExtractor().extract(meeting_request())
    assert result.diagnostics["actions"]["reason"] == "context_actions: model profile deepseek has no price"
    assert not any(system_of(body) == ACTIONS_PROMPT for _, _, body in sent)


def test_long_meetings_fall_back_to_chunks_and_merge(wire, monkeypatch):
    sent = wire()
    monkeypatch.setitem(ANSWERS, ACTIONS_VERIFY_PROMPT, {"verdicts": []})  # nothing refused by the verifier
    monkeypatch.setattr(actions, "WHOLE_MEETING_TOKENS", 100)
    monkeypatch.setattr(actions, "CHUNK_TOKENS", 150)
    request = meeting_request()
    result = NativeExtractor().extract(request)

    diagnostics = result.diagnostics["actions"]
    calls = [json.loads(body["input"][1]["content"]) for _, _, body in sent
             if body["input"][0]["content"] == ACTIONS_PROMPT]
    assert diagnostics["mode"] == "chunked" and len(calls) > 1 and diagnostics["calls"] == len(calls) + 1
    assert all(c["private_notes"].startswith("Typed by Jules Park") for c in calls)
    assert "context_before" in calls[1]
    # Each chunk answered with the whole list: items citing other chunks' lines are refused, the rest merge.
    titles = [a.title for a in result.actions]
    assert len(titles) == len(set(titles)) and "Pull fifty calls for the pilot batch" in titles
    assert diagnostics["rejected"] and check_extraction(request, result) == []


def test_route_prefers_the_lab_override_then_the_setting(monkeypatch):
    assert actions.route().profile == "luna"
    monkeypatch.setattr(get_settings(), "context_actions_model", "terra")
    assert actions.route().profile == "terra"
    use(monkeypatch, "sonnet-5.5")
    assert actions.route().profile == "sonnet-5.5"
    assert model_map.JOBS["context_actions"].cost_feature == "context_extraction"


def test_owner_names_resolve_to_one_actor_or_stay_a_mention():
    request = meeting_request()
    meeting = actions.Meeting.of(request, SourceContext.from_request(request))
    assert actions.resolve("Jules", meeting).entity_id == "act_me"
    assert actions.resolve("jules park", meeting).entity_id == "act_me"
    assert actions.resolve("Rowan", meeting).entity_id == "act_rowan"
    assert actions.resolve("Imani", meeting) == EntityRef(entity_type="actor", mention="Imani")
    assert actions.resolve("we", meeting).entity_id is None and actions.resolve(None, meeting) is None
    assert (meeting.seq("N1"), meeting.seq("[N2]"), meeting.seq("3"), meeting.seq("N9")) == (14, 15, 3, None)


def test_identifiers_and_ranges_count_when_a_cited_quote_has_them():
    assert ungrounded_numbers("Update the C2 rubric to a 1-5 scale", "C2 should go one to five", []) == set()
    assert ungrounded_numbers("Update the C2 rubric to a 1 to 5 scale", "move C2 to 1-5", []) == set()
    assert ungrounded_numbers("Update the C3 rubric", "C2 should go one to five", []) == {Decimal(3)}


def test_a_completed_pass_round_trips_for_resumable_callers(wire, monkeypatch):
    wire()
    request = meeting_request()
    ctx = SourceContext.from_request(request)
    meeting = actions.Meeting.of(request, ctx)
    out = actions.ActionRun()
    NativeExtractor()._actions("davin", meeting, out, extract_spend(), 30)
    key = actions.cache_key(meeting)
    saved = json.loads(json.dumps(out.dump(key)))
    restored = actions.ActionRun()
    restored.restore(saved, meeting)
    assert saved["key"] == key and restored.cost == out.cost > 0 and restored.calls == out.calls == 2
    assert restored.diagnostics(meeting) == out.diagnostics(meeting)
    assert [d.assertion for d in restored.drafts] == [d.assertion for d in out.drafts]
    # anything the pass reads changes the key: the lines, the prompt's model, the candidates
    assert actions.cache_key(actions.Meeting.of(request.model_copy(update={"title": "Other"}), ctx)) != key
    # a dismissal since the save rejects that item again; decisions alone don't change the key
    title = out.drafts[0].title
    dismissed = request.model_copy(update={"decisions": [Decision(
        id="dec-1", seq=1, operation="dismiss", origin="human", target_type="proposal", target_id="p-1",
        statement=title, recorded_at=request.occurred_at)]})
    later = actions.Meeting.of(dismissed, SourceContext.from_request(dismissed))
    assert actions.cache_key(later) == key
    again = actions.ActionRun()
    again.restore(saved, later)
    assert [d.title for d in again.drafts] == [d.title for d in out.drafts[1:]]
    assert again.rejected[-1]["code"] == "prior_decision"
    use(monkeypatch, "terra")
    assert actions.cache_key(meeting) != key


def extract_spend():
    from jarvis.context.native.extract import Spend
    return Spend(cap=5.0)
