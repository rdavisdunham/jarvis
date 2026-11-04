"""The context application service: authorization before assembly and again before return, compact cited
results, history only on request, task scopes on linked work, and agent feedback that is only ever a suggestion.

Fixture text is data. No model or network call runs: the query embedding is a fixed vector."""

from contextlib import contextmanager
from datetime import timedelta
from types import SimpleNamespace
from uuid import uuid4

import pytest
from jarvis import bot_access
from jarvis.config import get_settings
from jarvis.context import decisions, service, temporal
from jarvis.context.models import (
    ContextAssertion,
    ContextDecision,
    ContextEntityLink,
    ContextSource,
)
from jarvis.context.native import retrieve as native_retrieve
from jarvis.context.schema import EntityRef, Hit, ReconcileOp
from jarvis.db import session_scope
from jarvis.domain import DomainError
from jarvis.models import Actor, BotCredential, Task, now
from sqlalchemy import select

from .test_temporal import World, days

OWNER = "davin"
DIM = 512


def vec(*axes):
    v = [0.0] * DIM
    for axis in axes:
        v[axis] = 1.0
    return v


@pytest.fixture(autouse=True)
def enabled(monkeypatch):
    monkeypatch.setenv("JARVIS_CONTEXT_ENABLED", "true")
    get_settings.cache_clear()
    monkeypatch.setattr(native_retrieve, "query_vector", lambda owner, text: vec(0))
    yield
    get_settings.cache_clear()


def mint(client, *scopes):
    response = client.post("/api/v1/bot-keys", json={"name": "Claude Code", "scopes": list(scopes),
                                                     "expires_in_days": 30})
    assert response.status_code == 201, response.text
    body = response.json()
    return body["credential"]["id"], {"Authorization": "Bearer " + body["token"]}


@contextmanager
def bot(credential_id):
    with bot_access.bind(credential_id):
        yield


def embedded(db, row, vector=None):
    row.embedding, row.embedding_model = vector or vec(0), native_retrieve.memory_learning.EMBEDDING_MODEL
    db.flush()
    return row


def link(db, row, actor_id):
    db.add(ContextEntityLink(owner_id=OWNER, assertion_id=row.id, subject_type="assertion", subject_id=row.id,
                             target_type="actor", target_id=actor_id, mention="Priya", role="participant",
                             status="exact"))
    db.flush()


def seed():
    """Decisions with Priya about the pilot: MySQL (superseded) then Postgres, with a rationale, plus an open
    question and Priya's budget commitment. Returns the IDs."""
    with session_scope() as db:
        priya = Actor(owner_id=OWNER, name="Priya Shah")
        db.add(priya)
        db.flush()
        w = World(db, OWNER)
        pilot = ("project", "proj-pilot")
        old = embedded(db, w.assertion("We decided to launch the pilot on MySQL.", days(-20), kind="decision",
                                       key=None, subject=pilot))
        new = embedded(db, w.assertion("We decided to launch the pilot on Postgres with pgvector.", days(-5),
                                       kind="decision", key=None, subject=pilot))
        why = embedded(db, w.assertion("Postgres keeps embeddings next to the data, which is why we decided on it.",
                                       days(-5), kind="rationale", key=None, subject=pilot))
        question = embedded(db, w.assertion("Who signs off the pilot budget?", days(-5), kind="open_question",
                                            key=None, subject=pilot))
        budget = w.assertion("Priya sends the revised budget by 2026-11-20.", days(-5), kind="commitment",
                             key=None, subject=pilot)
        for row in (old, new, why, question):
            link(db, row, priya.id)
        temporal.apply_ops(db, OWNER, [ReconcileOp(op="supersede", targets=[old.id, new.id])],
                           semantic_supersede=True)
        assert (old.lifecycle, new.lifecycle) == ("superseded", "current")
        return SimpleNamespace(priya=priya.id, old=old.id, new=new.id, why=why.id, question=question.id,
                               budget=budget.id, pilot=pilot)


def source_id(assertion_id):
    from jarvis.context.models import ContextEvidence

    with session_scope() as db:
        return db.scalar(select(ContextEvidence.source_id).where(ContextEvidence.assertion_id == assertion_id))


def priya(ids):
    return EntityRef(entity_type="actor", entity_id=ids.priya)


# ---------------------------------------------------------------- search


def test_search_returns_current_cited_context_and_history_only_on_request(client):
    ids = seed()
    credential, _ = mint(client, "context:read")
    with bot(credential):
        found = service.search(service.SearchInput(query="what did we decide with Priya and why",
                                                   entity=priya(ids)))
        history = service.search(service.SearchInput(query="what did we decide", entity=priya(ids),
                                                     include_history=True))
    by_id = {r["id"]: r for r in found["results"]}
    assert set(by_id) == {ids.new, ids.why, ids.question}  # superseded MySQL excluded
    top = by_id[ids.new]
    assert (top["status"], top["basis"], top["kind"], top["revision"]) == ("current", "stated", "decision", 1)
    assert top["citation"]["quote"] == "We decided to launch the pilot on Postgres with pgvector."
    assert top["citation"]["source_id"] and top["citation"]["segment"] == 0 and "superseded_by" not in top
    assert found["semantic"] == "available" and found["names"][f"actor:{ids.priya}"] == "Priya Shah"
    assert "include_history" in found["next"]
    past = {r["id"]: r for r in history["results"]}
    assert past[ids.old]["status"] == "historical" and past[ids.old]["superseded_by"] == ids.new
    assert past[ids.new]["status"] == "current"


def test_valid_at_shows_what_was_in_effect_then(client):
    ids = seed()
    credential, _ = mint(client, "context:read")
    with bot(credential):
        then = service.search(service.SearchInput(entity=priya(ids), kinds=["decision"], valid_at=days(-10)))
    [item] = then["results"]
    assert (item["id"], item["status"], item["superseded_by"]) == (ids.old, "current", ids.new)
    assert then["scope"]["valid_at"] == days(-10).isoformat()


def test_known_at_reads_the_versions_recorded_then(client):
    ids = seed()
    credential, _ = mint(client, "context:read")
    with bot(credential):
        ranked = service.search(service.SearchInput(query="pgvector", known_at=now()))
        browsed = service.search(service.SearchInput(entity=priya(ids), kinds=["decision"], known_at=now()))
    assert ids.new in {r["id"] for r in ranked["results"]} and ids.old not in {r["id"] for r in ranked["results"]}
    assert [r["id"] for r in browsed["results"]] == [ids.new] and browsed["results"][0]["revision"] == 1


def test_search_without_query_browses_an_entity_newest_first_by_kind(client):
    ids = seed()
    credential, _ = mint(client, "context:read")
    with bot(credential):
        listed = service.search(service.SearchInput(entity=priya(ids), kinds=["decision", "rationale"]))
    assert [r["id"] for r in listed["results"]] in ([ids.new, ids.why], [ids.why, ids.new])
    assert listed["semantic"] == "not_used" and all("score" not in r for r in listed["results"])


def test_embedding_outage_answers_lexically_and_says_so(client, monkeypatch):
    ids = seed()
    credential, _ = mint(client, "context:read")

    def down(owner, text):
        raise RuntimeError("provider down")

    monkeypatch.setattr(native_retrieve, "query_vector", down)
    with bot(credential):
        result = service.search(service.SearchInput(query="pgvector"))
    assert [r["id"] for r in result["results"]] == [ids.new]
    assert result["semantic"] == "unavailable" and any("keyword" in g for g in result["gaps"])


def test_results_are_rechecked_before_return(client, monkeypatch):
    """A hit for context that was retracted after ranking is withheld and counted, never returned."""
    ids = seed()
    credential, _ = mint(client, "context:read")
    real = native_retrieve.NativeRetriever.retrieve

    def racing(self, query):
        hits = real(self, query)
        with session_scope() as db:  # a correction commits between ranking and return
            row = db.get(ContextAssertion, ids.new)
            decisions.wrong(db, OWNER, row.id, row.current_version, command_id=None, principal=OWNER)
        return hits

    monkeypatch.setattr(native_retrieve.NativeRetriever, "retrieve", racing)
    with bot(credential):
        result = service.search(service.SearchInput(query="pgvector"))
    assert ids.new not in {r["id"] for r in result["results"]}
    assert result["gaps"][0].startswith("1 result withheld")


def test_a_key_revoked_during_retrieval_gets_nothing(client, monkeypatch):
    seed()
    credential, _ = mint(client, "context:read")
    real = native_retrieve.NativeRetriever.retrieve

    def revoking(self, query):
        hits = real(self, query)
        with session_scope() as db:
            db.get(BotCredential, credential).revoked_at = now()
        return hits

    monkeypatch.setattr(native_retrieve.NativeRetriever, "retrieve", revoking)
    with bot(credential), pytest.raises(DomainError) as refused:
        service.search(service.SearchInput(query="pgvector"))
    assert refused.value.code == "ACCESS_REVOKED"


def test_raw_passages_are_marked_source_text_and_deleted_sources_vanish(client):
    ids = seed()
    credential, _ = mint(client, "context:read")
    with bot(credential):
        raw = service.search(service.SearchInput(query="pgvector", include_raw=True))
    kinds = {(r["ref"], r["basis"]) for r in raw["results"]}
    assert ("segment", "source_text") in kinds and ("assertion", "stated") in kinds
    segment = next(r for r in raw["results"] if r["ref"] == "segment")
    assert segment["supports"] == {"status": "current", "superseded_by": None}
    with session_scope() as db:
        gone = db.get(ContextSource, source_id(ids.new))
        gone.status, gone.deleted_at = "deleted", now()
    with bot(credential):
        after = service.search(service.SearchInput(query="pgvector", include_raw=True))
    assert ids.new not in {r["id"] for r in after["results"]}
    assert source_id(ids.new) not in {r["citation"]["source_id"] for r in after["results"]}


def test_searches_refuse_without_the_flag_scope_or_personal_workspace(client, monkeypatch):
    credential, _ = mint(client, "tasks:read")
    with bot(credential), pytest.raises(DomainError) as scope:
        service.search(service.SearchInput(query="x"))
    assert scope.value.code == "INSUFFICIENT_SCOPE"
    reader, _ = mint(client, "context:read")
    monkeypatch.setenv("JARVIS_CONTEXT_ENABLED", "false")
    get_settings.cache_clear()
    with bot(reader), pytest.raises(DomainError) as off:
        service.search(service.SearchInput(query="x"))
    assert off.value.code == "CONTEXT_DISABLED"
    with pytest.raises(DomainError) as nobody:
        service.search(service.SearchInput(query="x"))
    assert nobody.value.code in ("CONTEXT_DISABLED", "NOT_AUTHORIZED")


# ---------------------------------------------------------------- evidence and source


def test_evidence_lists_spans_and_hides_ended_context_until_history_is_asked(client):
    ids = seed()
    credential, _ = mint(client, "context:read")
    with bot(credential):
        current = service.evidence_for(service.EvidenceInput(assertion_id=ids.new))
        ended = service.evidence_for(service.EvidenceInput(assertion_id=ids.old))
        history = service.evidence_for(service.EvidenceInput(assertion_id=ids.old, include_history=True))
        with pytest.raises(DomainError) as missing:
            service.evidence_for(service.EvidenceInput(assertion_id=str(uuid4())))
    assert missing.value.code == "NOT_FOUND"
    span = current["evidence"][0]
    assert (span["role"], span["origin"], span["segment"], span["current_revision"]) == (
        "support", "independent", 0, True)
    assert span["quote"] == "We decided to launch the pilot on Postgres with pgvector."
    assert current["support"] == "supported" and current["corroborating_sources"] == 1
    assert "versions" not in current
    assert ended["status"] == "historical" and ended["evidence"] == [] and ended["superseded_by"] == ids.new
    assert "include_history" in ended["gaps"][0]
    assert history["evidence"] and [v["lifecycle"] for v in history["versions"]] == ["current", "superseded"]


def test_evidence_from_a_deleted_source_is_withheld(client):
    ids = seed()
    credential, _ = mint(client, "context:read")
    with session_scope() as db:
        gone = db.get(ContextSource, source_id(ids.why))
        gone.status, gone.deleted_at = "deleted", now()
    with bot(credential):
        result = service.evidence_for(service.EvidenceInput(assertion_id=ids.why))
    assert result["evidence"] == [] and result["citation"] is None
    assert result["gaps"][0].startswith("1 evidence span withheld")


def test_source_drill_down_pages_and_marks_superseded_passages(client):
    ids = seed()
    credential, _ = mint(client, "context:read")
    with bot(credential):
        page = service.source(service.SourceInput(source_id=source_id(ids.old)))
        with pytest.raises(DomainError) as bad:
            service.source(service.SourceInput(source_id=source_id(ids.old), segment_start=3, segment_end=1))
        with pytest.raises(DomainError) as other:
            service.source(service.SourceInput(source_id=str(uuid4())))
    assert (bad.value.code, other.value.code) == ("INVALID_ARGUMENT", "NOT_FOUND")
    assert page["source"]["revision"] == 1 and page["next_segment"] is None
    [segment] = page["segments"]
    assert segment["text"] == "We decided to launch the pilot on MySQL."
    assert segment["supports"] == [{"assertion_id": ids.old, "kind": "decision", "status": "historical",
                                    "superseded_by": ids.new}]
    with session_scope() as db:
        gone = db.get(ContextSource, source_id(ids.old))
        gone.status, gone.deleted_at = "deleted", now()
    with bot(credential), pytest.raises(DomainError) as deleted:
        service.source(service.SourceInput(source_id=source_id(ids.old)))
    assert deleted.value.code == "NOT_FOUND"


# ---------------------------------------------------------------- briefs and views (C3 shapes, stubbed)


def task_for(title="Send budget"):
    with session_scope() as db:
        task = Task(owner_id=OWNER, title=title)
        db.add(task)
        db.flush()
        return task.id


def stub_views(monkeypatch, ids, task_id):
    loop = {"assertion_id": ids.budget, "revision": 1, "statement": "Priya sends the revised budget by 2026-11-20.",
            "lifecycle": "current", "inferred": False, "bucket": "waiting_for", "waiting_for": True,
            "responsible": [{"actor_id": ids.priya, "name": "Priya Shah"}], "recipient": None, "due_at": None,
            "due_precision": None, "status": "open", "task_state": "open",
            "tasks": [{"id": task_id, "title": "Send budget", "status": "todo"}],
            "subject": {"entity_type": "project", "entity_id": "proj-pilot"}, "valid_from": None,
            "recorded_at": None, "evidence": [{"quote": "x"}]}
    stale = {**loop, "assertion_id": ids.old, "statement": "We decided to launch the pilot on MySQL."}
    brief = {"brief_id": "b1", "version": 2, "built_at": "2026-11-04T00:00:00+00:00", "stale": False,
             "basis": "stored", "notice": None, "entity": {"entity_type": "actor", "entity_id": ids.priya},
             "next_meeting": None,
             "sections": [
                 {"name": "current_state_and_why", "gaps": [], "claims": [
                     {"text": "Launch on Postgres with pgvector.", "assertion_ids": [ids.new, ids.why],
                      "task_ids": []},
                     {"text": "Launch on MySQL.", "assertion_ids": [ids.old], "task_ids": []},
                     {"text": "Who signs off?", "assertion_ids": [ids.question, "gone-assertion"],
                      "task_ids": []}]},
                 {"name": "open_loops", "gaps": [], "claims": [
                     {"text": "Priya owes the budget (task open).", "assertion_ids": [ids.budget],
                      "task_ids": [task_id]}]},
                 {"name": "unknowns", "gaps": ["No date for sign-off."], "claims": []}]}
    calls = []
    fake_briefs = SimpleNamespace(get_brief=lambda db, owner, t, i, **kw: calls.append((owner, t, i)) or brief)
    fake_views = SimpleNamespace(
        open_loops=lambda db, owner, **kw: calls.append(kw) or {"items": [loop, stale], "gaps": []},
        meeting_prep=lambda db, owner, **kw: {
            "event": {"event_id": "e1", "title": "Pilot sync"}, "entity": None, "open_loops": [loop],
            "decisions": [{"assertion_id": ids.new, "revision": 1, "kind": "decision", "statement": "Postgres",
                           "lifecycle": "current", "valid_from": None, "evidence": [{"quote": "x"}]},
                          {"assertion_id": ids.old, "revision": 2, "kind": "decision", "statement": "MySQL",
                           "lifecycle": "superseded", "valid_from": None, "evidence": []}],
            "open_questions": [], "previous_meetings": [{"source_id": source_id(ids.old), "title": "Kick-off"},
                                                        {"source_id": source_id(ids.new), "title": "Review"}],
            "gaps": ["No earlier meeting."]},
        changes=lambda db, owner, after=None, limit=100: {
            "items": [{"cursor": 5, "id": ids.why, "type": "assertion", "deleted": False, "statement": "why",
                       "subject": {}},
                      {"cursor": 6, "id": ids.new, "type": "assertion", "deleted": False, "statement": "new",
                       "subject": {}}],
            "current_cursor": 6, "next_cursor": 6, "has_more": False})
    monkeypatch.setattr(service, "briefs", fake_briefs)
    monkeypatch.setattr(service, "views", fake_views)
    return calls


def test_brief_keeps_only_claims_whose_citations_survive_and_needs_tasks_read_for_task_claims(client,
                                                                                              monkeypatch):
    ids = seed()
    task_id = task_for()
    stub_views(monkeypatch, ids, task_id)
    reader, _ = mint(client, "context:read")
    with bot(reader):
        result = service.brief(service.BriefInput(entity=priya(ids)))
    sections = {s["name"]: s for s in result["sections"]}
    [claim] = sections["current_state_and_why"]["claims"]
    assert claim["assertion_ids"] == [ids.new, ids.why] and len(claim["citations"]) == 2
    assert sections["open_loops"]["claims"] == [] and sections["unknowns"]["gaps"] == ["No date for sign-off."]
    assert result["gaps"][0].startswith("1 statement withheld")
    assert result["gaps"][1].startswith("1 statement about superseded") and "tasks:read" in result["gaps"][2]
    assert "history" not in result and "include_history" in result["next"]
    both, _ = mint(client, "context:read", "tasks:read")
    with bot(both):
        full = service.brief(service.BriefInput(entity=priya(ids), include_history=True,
                                                sections=["open_loops", "current_state_and_why"]))
    assert [s["name"] for s in full["sections"]] == ["current_state_and_why", "open_loops"]
    assert [c["text"] for c in full["sections"][0]["claims"]] == ["Launch on Postgres with pgvector.",
                                                                  "Launch on MySQL."]
    full["sections"] = full["sections"][1:]
    assert full["sections"][0]["claims"][0]["task_ids"] == [task_id]
    assert [h["id"] for h in full["history"]] == [ids.old] and full["history"][0]["superseded_by"] == ids.new


def test_query_brief_groups_current_cited_statements(client):
    ids = seed()
    credential, _ = mint(client, "context:read")
    with bot(credential):
        result = service.brief(service.BriefInput(query="pilot decided budget"))
    sections = {s["name"]: s for s in result["sections"]}
    cited = {i for c in sections["current_state_and_why"]["claims"] for i in c["assertion_ids"]}
    assert ids.new in cited and ids.old not in cited
    assert result["basis"] == "query"


def test_open_loops_and_meeting_prep_recheck_items_and_scrub_task_state(client, monkeypatch):
    ids = seed()
    task_id = task_for()
    calls = stub_views(monkeypatch, ids, task_id)
    reader, _ = mint(client, "context:read")
    with bot(reader):
        loops = service.open_loops(service.OpenLoopsInput(mine=True, waiting_for=False))
        prep = service.meeting_prep(service.MeetingPrepInput(entity=priya(ids)))
    assert calls[-1]["owner_actor"] == "me" and calls[-1]["waiting_for"] is False
    [item] = loops["items"]
    assert item["assertion_id"] == ids.budget and item["task_state"] == "withheld" and "tasks" not in item
    assert item["citation"]["quote"] and loops["gaps"][0].startswith("1 item withheld")
    assert any("tasks:read" in g for g in loops["gaps"])
    assert [d["assertion_id"] for d in prep["decisions"]] == [ids.new] and "evidence" not in prep["decisions"][0]
    assert prep["open_loops"][0]["task_state"] == "withheld"
    with session_scope() as db:
        gone = db.get(ContextSource, source_id(ids.old))
        gone.status, gone.deleted_at = "deleted", now()
    with bot(reader):
        prep = service.meeting_prep(service.MeetingPrepInput(entity=priya(ids)))
    assert [m["title"] for m in prep["previous_meetings"]] == ["Review"]
    both, _ = mint(client, "context:read", "tasks:read")
    with bot(both):
        [item] = service.open_loops(service.OpenLoopsInput())["items"]
    assert item["tasks"][0]["id"] == task_id and item["task_state"] == "open"


def test_changes_tombstone_assertions_whose_evidence_is_gone(client, monkeypatch):
    ids = seed()
    stub_views(monkeypatch, ids, task_for())
    with session_scope() as db:
        gone = db.get(ContextSource, source_id(ids.why))
        gone.status, gone.deleted_at = "deleted", now()
    reader, _ = mint(client, "context:read")
    with bot(reader):
        result = service.changes(service.ChangesInput(after=4))
    by_id = {i["id"]: i for i in result["items"]}
    assert by_id[ids.why]["deleted"] and by_id[ids.why]["statement"] is None
    assert by_id[ids.new]["statement"] == "new" and result["next_cursor"] == 6


def test_views_report_unavailable_until_c3_merges(client, monkeypatch):
    monkeypatch.setattr(service, "views", None)
    reader, _ = mint(client, "context:read")
    with bot(reader), pytest.raises(DomainError) as missing:
        service.open_loops(service.OpenLoopsInput())
    assert missing.value.code == "UNAVAILABLE"


# ---------------------------------------------------------------- feedback and Questions


def suggestion(ids, **values):
    return service.FeedbackInput(**{"request_id": uuid4(), "target_type": "assertion", "target_id": ids.new,
                                    "expected_revision": 1, "suggestion": "wrong",
                                    "explanation": "The 11-04 notes say the pilot stays on MySQL.", **values})


def test_agent_feedback_is_an_agent_suggestion_that_changes_nothing(client):
    ids = seed()
    writer, _ = mint(client, "context:write")
    args = suggestion(ids)
    with bot(writer):
        receipt = service.feedback(args)
        replay = service.feedback(args)
        with pytest.raises(DomainError) as reused:
            service.feedback(args.model_copy(update={"explanation": "Different"}))
        with pytest.raises(DomainError) as stale:
            service.feedback(suggestion(ids, expected_revision=7))
    assert (reused.value.code, stale.value.code) == ("IDEMPOTENCY_CONFLICT", "REVISION_CONFLICT")
    assert replay == receipt
    assert (receipt["status"], receipt["origin"], receipt["applied"]) == ("pending_review", "agent", False)
    with session_scope() as db:
        row = db.get(ContextDecision, receipt["feedback_id"])
        assert (row.origin, row.operation, row.target_type, row.negative) == ("agent", "wrong", "feedback", False)
        assert row.principal == "bot:" + writer and row.target_id == "assertion:" + ids.new
        assert db.get(ContextAssertion, ids.new).lifecycle == "current"  # unchanged
        assert temporal.negated_by(db, OWNER, db.get(ContextAssertion, ids.new)) is None
        [item] = service.feedback_items(db, OWNER)
        assert decisions.question_items(db, OWNER) == []  # B5's own items are untouched
        assert [q["key"] for q in service.question_items(db, OWNER)] == [item["key"]]
    assert item["key"] == receipt["question_key"] and item["kind"] == "context"
    assert item["answer_tool"] == "context.resolve" and item["answer_args"]["actions"] == ["accept", "dismiss"]
    assert item["evidence"]["origin"] == "agent" and item["evidence"]["agent"] == "Claude Code"
    assert "is wrong" in item["question"] and item["candidates"][0]["assertion_id"] == ids.new


def test_reading_keys_cannot_suggest_and_proposals_are_not_targets(client):
    ids = seed()
    reader, _ = mint(client, "context:read")
    with bot(reader), pytest.raises(DomainError) as refused:
        service.feedback(suggestion(ids))
    assert refused.value.code == "INSUFFICIENT_SCOPE"
    with pytest.raises(ValueError):
        suggestion(ids, target_type="proposal")
    with pytest.raises(ValueError):
        suggestion(ids, suggestion="changed")  # needs the new statement
    with pytest.raises(ValueError):
        suggestion(ids, suggestion="reject_link")  # a link suggestion on an assertion


def test_the_owner_accepts_dismisses_or_defers_agent_suggestions(client):
    ids = seed()
    writer, _ = mint(client, "context:write")
    with bot(writer):
        wrong = service.feedback(suggestion(ids))
        changed = service.feedback(suggestion(ids, target_id=ids.why, suggestion="changed",
                                              statement="Postgres was chosen for cost.",
                                              effective_at=days(-1)))
        confirm = service.feedback(suggestion(ids, target_id=ids.question, suggestion="confirm"))
    with session_scope() as db:
        items = {q["key"]: q for q in service.feedback_items(db, OWNER)}
        deferred = service.defer_question(db, OWNER, confirm["question_key"], 1, timedelta(days=1),
                                          command_id="cmd-defer")
        assert deferred["status"] == "deferred"
        assert {q["key"]: q["status"] for q in service.feedback_items(db, OWNER)}[confirm["question_key"]] == \
            "deferred"
        with pytest.raises(DomainError) as stale:
            service.resolve(db, OWNER, service.ResolveInput(question_key=confirm["question_key"],
                                                            expected_revision=1, action="dismiss"),
                            command_id="cmd-1", principal=OWNER)
        assert stale.value.code == "STALE_REVIEW"
        done = service.resolve(db, OWNER, service.ResolveInput(
            question_key=wrong["question_key"], expected_revision=items[wrong["question_key"]]["revision"],
            action="accept"), command_id="cmd-2", principal=OWNER)
        applied = db.get(ContextDecision, done["decision_ids"][0])
        assert (applied.operation, applied.origin, applied.target_id) == ("wrong", "human", ids.new)
        assert db.get(ContextAssertion, ids.new).lifecycle == "retracted"
        service.resolve(db, OWNER, service.ResolveInput(question_key=changed["question_key"], expected_revision=1,
                                                        action="dismiss"), command_id="cmd-3", principal=OWNER)
        assert db.get(ContextAssertion, ids.why).lifecycle == "current"
        assert [q["key"] for q in service.feedback_items(db, OWNER)] == [confirm["question_key"]]
        with pytest.raises(DomainError):
            service.resolve(db, OWNER, service.ResolveInput(question_key=confirm["question_key"],
                                                            expected_revision=2, action="keep"),
                            command_id="cmd-4", principal=OWNER)


def test_a_suggestion_on_changed_context_is_no_longer_asked(client):
    ids = seed()
    writer, _ = mint(client, "context:write")
    with bot(writer):
        service.feedback(suggestion(ids))
    with session_scope() as db:
        row = db.get(ContextAssertion, ids.new)
        decisions.confirm(db, OWNER, row.id, 1, command_id="cmd-x", principal=OWNER)
        assert len(service.feedback_items(db, OWNER)) == 1  # confirm on current changes no revision
        decisions.wrong(db, OWNER, row.id, row.current_version, command_id="cmd-y", principal=OWNER)
        assert service.feedback_items(db, OWNER) == []


def test_link_suggestions_name_the_entity_and_accepting_confirms_the_link(client):
    ids = seed()
    with session_scope() as db:
        other = Actor(owner_id=OWNER, name="Priya Raman")
        db.add(other)
        db.flush()
        row = db.scalar(select(ContextEntityLink).where(ContextEntityLink.assertion_id == ids.new))
        link_id, other_id = row.id, other.id
    writer, _ = mint(client, "context:write")
    with bot(writer):
        receipt = service.feedback(service.FeedbackInput(
            request_id=uuid4(), target_type="link", target_id=link_id, expected_revision=1,
            suggestion="confirm_link", explanation="Priya Raman ran that meeting.",
            entity=EntityRef(entity_type="actor", entity_id=other_id)))
    with session_scope() as db:
        [item] = service.feedback_items(db, OWNER)
        assert "Priya Raman" in item["question"] and item["candidates"] == []
        service.resolve(db, OWNER, service.ResolveInput(question_key=receipt["question_key"], expected_revision=1,
                                                        action="accept"), command_id="cmd-l", principal=OWNER)
        row = db.get(ContextEntityLink, link_id)
        assert (row.status, row.target_id) == ("confirmed", other_id)


def test_hits_for_other_owners_never_leak(client, monkeypatch):
    """Even if an engine returned another owner's assertion, the before-return check drops it."""
    seed()
    with session_scope() as db:
        stranger = World(db, "someone-else").assertion("Their secret plan.", days(-1), kind="decision", key=None,
                                                       subject=("project", "p-x"))
        stranger_id = stranger.id
    monkeypatch.setattr(native_retrieve.NativeRetriever, "retrieve", lambda self, query: [
        Hit(ref_type="assertion", id=stranger_id, revision=1, score=1.0, lifecycle="current")])
    credential, _ = mint(client, "context:read")
    with bot(credential):
        result = service.search(service.SearchInput(query="secret plan"))
    assert result["results"] == [] and result["gaps"][0].startswith("1 result withheld")
