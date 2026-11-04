"""Meeting proposals: stable identity, dedupe, matching, the recommendation table, dismissal persistence and
material revisions. Extraction runs through the deterministic double or a canned `ExtractionResult`; no model
or network call."""

import pytest
from jarvis.context import apply, decisions, pipeline, proposals
from jarvis.context import models as m
from jarvis.context.schema import (
    CandidateAssertion,
    EntityRef,
    EvidenceRef,
    ExtractionResult,
    Participant,
    ProposedAction,
    Segment,
    SourceEnvelope,
    TimeBound,
)
from jarvis.db import session_scope
from jarvis.domain import enqueue_job, execute
from jarvis.models import Job, LinearIssue, Task
from sqlalchemy import func, select

from .conftest import MEETING_AT, at
from .test_pipeline import drain, jobs, receive, use_doubles, vectors  # noqa: F401

OWNER = "davin"
LINES = [
    ("Davin", "owner", "Thanks for joining. I'll circulate the notes by 2026-11-06."),
    ("Priya", "other", "I'll send the revised budget by 2026-11-20."),
    ("Davin", "owner", "Action: Marco to draft the rollout checklist."),
]


def meeting(lines=LINES, rev="r1", external_id="m-1"):
    return SourceEnvelope(connector="api", external_id=external_id, title="Pilot sync", occurred_at=MEETING_AT,
                          timezone="UTC", external_revision=rev,
                          segments=[Segment(seq=i, speaker_label=w, attribution=a, text=t)
                                    for i, (w, a, t) in enumerate(lines)])


def process(envelope):
    """Receive and run the pipeline, then the proposals stage for the revision if nothing queued it."""
    got = receive(envelope)
    drain()
    with session_scope() as db:
        queued = db.scalar(select(func.count()).select_from(Job).where(
            Job.kind == "context_proposals", Job.payload["revision_id"].astext == got.revision_id))
        if not queued:
            enqueue_job(db, OWNER, "context_proposals", {"source_id": got.source_id, "revision_id": got.revision_id})
    drain()
    return got


def rows(source_id=None):
    with session_scope() as db:
        q = select(m.ContextProposal).order_by(m.ContextProposal.created_at, m.ContextProposal.id)
        if source_id:
            q = q.where(m.ContextProposal.source_id == source_id)
        return [{"id": p.id, "kind": p.kind, "title": p.title, "revision": p.revision, "disposition": p.disposition,
                 "preselected": p.preselected, "responsible": p.responsible, "fingerprint": p.fingerprint,
                 "evidence": p.evidence, "assertion_ids": p.assertion_ids, "task_id": p.task_id, **p.payload}
                for p in db.scalars(q)]


def by_title(items, word):
    return next(p for p in items if word in p["title"])


def canned(monkeypatch, build):
    """Replace extraction with `build(request) -> ExtractionResult`."""
    monkeypatch.setattr(pipeline, "run_extractor", lambda engine, request, *a, **k: build(request))


def ref(segment: Segment, quote: str) -> EvidenceRef:
    start = segment.text.index(quote)
    return EvidenceRef(segment_seq=segment.seq, quote=quote, char_start=start, char_end=start + len(quote))


def commitment(segment, quote, statement, who="Davin", **values):
    return CandidateAssertion(kind="commitment", statement=statement, evidence=[ref(segment, quote)],
                              responsible=[EntityRef(entity_type="actor", mention=who)], **values)


@pytest.fixture(autouse=True)
def doubles(vectors):  # noqa: F811
    use_doubles()


def create_task(title, **values):
    with session_scope() as db:
        return execute(db, OWNER, f"t-{len(title)}-{title[:20]}", "task.create", {"title": title, **values})["data"]


# ---------------------------------------------------------------- recommendation table


def test_only_own_clear_unmatched_action_is_preselected():
    got = process(meeting())
    items = rows(got.source_id)
    assert len(items) == 3 and {p["kind"] for p in items} == {"create"}
    mine, priya, marco = by_title(items, "circulate"), by_title(items, "budget"), by_title(items, "checklist")
    assert mine["preselected"] and mine["due_date"] == "2026-11-06" and mine["disposition"] == "pending"
    assert "You said" in mine["reason"]
    assert not priya["preselected"] and priya["reason"].startswith("Waiting on Priya")
    assert priya["due_date"] == "2026-11-20"
    assert not marco["preselected"] and marco["responsible_mention"] == "Marco"
    assert all(p["evidence"] and p["evidence"][0]["quote"] for p in items)


def test_retry_and_duplicate_delivery_change_nothing():
    got = process(meeting())
    before = rows(got.source_id)
    with session_scope() as db:
        for _ in range(2):  # a duplicate outbox delivery and a later retry of the same stage
            enqueue_job(db, OWNER, "context_proposals", {"source_id": got.source_id, "revision_id": got.revision_id})
    drain()
    with session_scope() as db:
        assert proposals.generate(db, OWNER, got.source_id, got.revision_id)["created"] == 0
    after = rows(got.source_id)
    assert [(p["id"], p["revision"], p["fingerprint"]) for p in after] == \
        [(p["id"], p["revision"], p["fingerprint"]) for p in before]


def test_context_only_candidates_make_no_proposal(monkeypatch):
    lines = [("Davin", "owner", "If we win the bid I'd hire two people."),
             ("Davin", "owner", "I already sent the contract."),
             ("Priya", "other", "Should we move the launch?")]
    segs = meeting(lines).segments

    def build(request):
        return ExtractionResult(assertions=[
            commitment(segs[0], "I'd hire two people", "Davin hires two people if the bid wins.",
                       modality="hypothetical"),
            commitment(segs[1], "I already sent the contract", "Davin sends the contract."),
            commitment(segs[2], "Should we move the launch?", "Move the launch.", who="we", modality="question"),
        ], coverage={s.seq: "extracted" for s in request.segments})
    canned(monkeypatch, build)
    got = process(meeting(lines))
    with session_scope() as db:
        done = db.scalar(select(m.ContextAssertion).where(m.ContextAssertion.statement == "Davin sends the contract."))
        done.completion_evidence = [{"quote": "I already sent the contract"}]
        db.execute(m.ContextProposal.__table__.delete())
        proposals.generate(db, OWNER, got.source_id, got.revision_id)
    assert rows(got.source_id) == []


def test_suggestion_is_separate_unchecked_and_never_a_promise(monkeypatch):
    lines = [("Priya", "other", "It might help to share the deck with legal.")]
    seg = meeting(lines).segments[0]

    def build(request):
        return ExtractionResult(actions=[ProposedAction(kind="suggestion", title="Share the deck with legal",
                                                        evidence=[ref(seg, "share the deck with legal")])],
                                coverage={0: "extracted"})
    canned(monkeypatch, build)
    got = process(meeting(lines))
    [item] = rows(got.source_id)
    assert item["kind"] == "suggestion" and not item["preselected"] and item["assertion_ids"] == []
    with session_scope() as db:
        assert db.scalar(select(func.count()).select_from(m.ContextAssertion)
                         .where(m.ContextAssertion.kind == "commitment")) == 0


def test_unclear_ownership_is_unchecked_with_explanation(monkeypatch):
    lines = [("Davin", "owner", "We need to book the venue.")]
    seg = meeting(lines).segments[0]
    canned(monkeypatch, lambda request: ExtractionResult(assertions=[
        commitment(seg, "We need to book the venue.", "Book the venue.", who="we")], coverage={0: "extracted"}))
    got = process(meeting(lines))
    [item] = rows(got.source_id)
    assert item["kind"] == "create" and not item["preselected"] and item["collective"]
    assert item["reason"] == "Who owns this is unclear."


def test_owner_basis_from_the_action_pass_decides_checked_and_reasons(monkeypatch):
    lines = [("Davin", "owner", "I'll circulate the notes."),
             ("Davin", "owner", "The export still shows the old column names."),
             ("Priya", "other", "I'll send the revised budget."),
             ("Priya", "other", "Someone has to book the venue.")]
    segs = meeting(lines).segments

    def build(request):
        me = next(c.ref.entity_id for c in request.candidates if c.name == "owner")
        mine = [EntityRef(entity_type="actor", entity_id=me, mention="Davin")]
        return ExtractionResult(
            assertions=[commitment(segs[0], "I'll circulate the notes.", "Davin will circulate the notes")],
            actions=[ProposedAction(kind="create", title="Circulate the notes", responsible=mine,
                                    evidence=[ref(segs[0], "I'll circulate the notes.")], assertion_indexes=[0]),
                     ProposedAction(kind="create", title="Rename the export columns", responsible=mine,
                                    evidence=[ref(segs[1], "the old column names")]),
                     ProposedAction(kind="create", title="Send the revised budget",
                                    responsible=[EntityRef(entity_type="actor", mention="Priya")],
                                    evidence=[ref(segs[2], "I'll send the revised budget.")]),
                     ProposedAction(kind="create", title="Book the venue",
                                    evidence=[ref(segs[3], "Someone has to book the venue.")]),
                     ProposedAction(kind="create", title="Recheck the scoring weights", responsible=mine,
                                    evidence=[ref(segs[1], "The export still shows")])],
            coverage={s.seq: "extracted" for s in request.segments},
            diagnostics={"actions": {"items": [
                {"action_index": 0, "owner_basis": "stated_promise", "verified": True},
                {"action_index": 1, "owner_basis": "implied_by_discussion", "verified": True},
                {"action_index": 2, "owner_basis": "stated_promise", "verified": True},
                {"action_index": 3, "owner_basis": "implied_by_discussion", "verified": True},
                # the checker skipped it: the user's own, but not checked
                {"action_index": 4, "owner_basis": "self_assigned_intent", "verified": None}]}})
    canned(monkeypatch, build)
    got = process(meeting(lines))
    items = rows(got.source_id)
    notes, export = by_title(items, "Circulate"), by_title(items, "export")
    budget, venue = by_title(items, "budget"), by_title(items, "venue")
    assert notes["preselected"] and notes["reason"].startswith("You said you would do this")
    assert not export["preselected"] and export["reason"].startswith("Implied by the discussion")
    assert not budget["preselected"] and budget["reason"].startswith("Waiting on Priya")
    assert not venue["preselected"] and venue["reason"] == "Who owns this is unclear."
    scoring = by_title(items, "scoring")
    assert not scoring["preselected"] and scoring["reason"] == "The checker did not confirm this one. Check it if it is yours."


def test_attended_meeting_keeps_the_recorders_work_off_the_users_list(monkeypatch):
    """Uttam recorded the meeting, so his lines carry the owner attribution, but the user is not Uttam: the
    actor the action pass resolved by name wins over the recorder's speaker link."""
    lines = [("Uttam Rao", "owner", "I'll draft the statement of work."),
             ("Davin", "other", "Great, thanks.")]
    envelope = meeting(lines).model_copy(update={"participants": [
        Participant(name="Uttam Rao", email="uttam@example.com", speaker_label="Uttam Rao")]})
    segs = envelope.segments

    def build(request):
        uttam = next(c.ref.entity_id for c in request.candidates if c.name == "Uttam Rao")
        who = [EntityRef(entity_type="actor", entity_id=uttam, mention="Uttam Rao")]
        quote = [ref(segs[0], "I'll draft the statement of work.")]
        return ExtractionResult(
            assertions=[CandidateAssertion(kind="commitment", statement="Uttam Rao will draft the statement of work",
                                           evidence=quote, responsible=who)],
            actions=[ProposedAction(kind="create", title="Draft the statement of work", responsible=who,
                                    evidence=quote, assertion_indexes=[0])],
            coverage={s.seq: "extracted" for s in request.segments},
            diagnostics={"actions": {"items": [{"action_index": 0, "owner_basis": "stated_promise"}]}})
    canned(monkeypatch, build)
    got = process(envelope)
    [item] = rows(got.source_id)
    assert not item["preselected"] and item["reason"].startswith("Waiting on Uttam Rao")


# ---------------------------------------------------------------- identity and dedupe


def test_two_actions_from_one_quote_stay_two_and_overlap_duplicates_merge(monkeypatch):
    lines = [("Davin", "owner", "I'll send the budget and draft the checklist."),
             ("Davin", "owner", "Again: I'll send the budget.")]
    segs = meeting(lines).segments
    quote = "I'll send the budget and draft the checklist."
    me = [EntityRef(entity_type="actor", mention="Davin")]

    def build(request):
        both = [ref(segs[0], quote)]
        return ExtractionResult(actions=[
            ProposedAction(kind="create", title="Send the budget", evidence=both, responsible=me),
            ProposedAction(kind="create", title="Draft the checklist", evidence=both, responsible=me),
            # The same action again, from an overlapping chunk with other evidence.
            ProposedAction(kind="create", title="send the budget", evidence=[ref(segs[1], "I'll send the budget.")],
                           responsible=me),
        ], coverage={0: "extracted", 1: "extracted"})
    canned(monkeypatch, build)
    got = process(meeting(lines))
    items = rows(got.source_id)
    assert sorted(p["title"] for p in items) == ["Draft the checklist", "Send the budget"]
    budget = by_title(items, "budget")
    assert {e["quote"] for e in budget["evidence"]} == {quote, "I'll send the budget."}


def test_revised_quote_keeps_the_proposal_and_material_change_revises_it():
    got = process(meeting())
    first = by_title(rows(got.source_id), "circulate")
    revised = [("Davin", "owner", "Thanks for joining. I'll circulate the meeting notes by 2026-11-09."), *LINES[1:]]
    process(meeting(revised, rev="r2"))
    second = by_title(rows(got.source_id), "circulate")
    assert second["id"] == first["id"] and second["fingerprint"] == first["fingerprint"]
    assert second["revision"] == first["revision"] + 1 and second["due_date"] == "2026-11-09"
    assert "2026-11-06" in second["explanation"] and "2026-11-09" in second["explanation"]
    assert "meeting notes" in second["evidence"][0]["quote"]
    others = [p["revision"] for p in rows(got.source_id) if p["id"] != first["id"]]
    assert others == [1, 1]  # unchanged work keeps its revision across the new source revision


def test_dropped_action_is_superseded():
    got = process(meeting())
    process(meeting(LINES[:2], rev="r2"))
    marco = by_title(rows(got.source_id), "checklist")
    assert marco["disposition"] == "superseded" and marco["revision"] == 2


def test_dropped_action_comes_back_as_a_new_revision():
    got = process(meeting())
    process(meeting(LINES[:2], rev="r2"))
    process(meeting(LINES, rev="r3"))
    marco = by_title(rows(got.source_id), "checklist")
    assert marco["disposition"] == "pending" and marco["revision"] == 3


def test_dismissal_persists_until_evidence_changes_materially():
    got = process(meeting())
    mine = by_title(rows(got.source_id), "circulate")
    with session_scope() as db:
        proposals.dismiss(db, OWNER, mine["id"], mine["revision"], command_id="dismiss1", principal=OWNER)
    reworded = [("Davin", "owner", "Thanks for joining. I'll circulate all the notes by 2026-11-06."), *LINES[1:]]
    process(meeting(reworded, rev="r2"))
    kept = by_title(rows(got.source_id), "circulate")
    assert kept["disposition"] == "dismissed" and kept["revision"] == mine["revision"] + 1  # the dismissal only
    moved = [("Davin", "owner", "Thanks for joining. I'll circulate all the notes by 2026-11-13."), *LINES[1:]]
    process(meeting(moved, rev="r3"))
    back = by_title(rows(got.source_id), "circulate")
    assert back["disposition"] == "pending" and back["revision"] == kept["revision"] + 1
    assert "2026-11-13" in back["explanation"]


# ---------------------------------------------------------------- matching existing work


def test_candidate_task_match_is_an_unchecked_link_that_never_mutates():
    task = create_task("Circulate the notes")
    got = process(meeting())
    mine = by_title(rows(got.source_id), "circulate")
    assert mine["kind"] == "link" and mine["match"] == "candidate" and not mine["preselected"]
    assert mine["target"]["task_id"] == task["id"]
    with session_scope() as db:
        assert db.get(Task, task["id"]).revision == task["revision"]


def test_stable_target_with_changed_due_date_is_an_update_with_before_and_after(monkeypatch):
    task = create_task("Revised budget", due_date="2026-11-13")
    lines = [("Davin", "owner", "Priya moved the revised budget to 2026-11-20.")]
    seg = meeting(lines).segments[0]

    def build(request):
        return ExtractionResult(
            assertions=[commitment(seg, "Priya moved the revised budget to 2026-11-20.",
                                   "Priya sends the revised budget by 2026-11-20.", who="Priya",
                                   due=TimeBound(at=at(20), precision="day"))],
            actions=[ProposedAction(kind="update", title="Send the revised budget", assertion_indexes=[0],
                                    target=EntityRef(entity_type="task", entity_id=task["id"]),
                                    due=TimeBound(at=at(20), precision="day"),
                                    evidence=[ref(seg, "moved the revised budget to 2026-11-20")])],
            coverage={0: "extracted"})
    canned(monkeypatch, build)
    got = process(meeting(lines))
    [item] = rows(got.source_id)
    assert item["kind"] == "update" and not item["preselected"] and item["match"] == "stable"
    assert item["before"]["due_date"] == "2026-11-13" and item["after"] == {"due_date": "2026-11-20"}
    assert item["before"]["revision"] == task["revision"]


def test_synced_linear_issue_is_matched_through_its_task(monkeypatch):
    with session_scope() as db:
        task = Task(owner_id=OWNER, title="Draft the rollout checklist")
        db.add(task)
        db.flush()
        issue = LinearIssue(owner_id=OWNER, workspace_id="ws", remote_id="lin-1", task_id=task.id, snapshot={})
        db.add(issue)
        db.flush()
        task_id = task.id
    lines = [("Davin", "owner", "Marco owns ERI-1, the rollout checklist.")]
    seg = meeting(lines).segments[0]
    canned(monkeypatch, lambda request: ExtractionResult(
        actions=[ProposedAction(kind="link", title="Draft the rollout checklist",
                                target=EntityRef(entity_type="linear_issue", entity_id="lin-1"),
                                evidence=[ref(seg, "Marco owns ERI-1, the rollout checklist.")])],
        coverage={0: "extracted"}))
    got = process(meeting(lines))
    [item] = rows(got.source_id)
    assert item["kind"] == "link" and item["target"]["task_id"] == task_id and item["work"] == f"task:{task_id}"


def test_wrong_supersedes_open_proposals_resting_on_it():
    got = process(meeting())
    mine = by_title(rows(got.source_id), "circulate")
    with session_scope() as db:
        assertion = db.get(m.ContextAssertion, mine["assertion_ids"][0])
        decisions.wrong(db, OWNER, assertion.id, assertion.current_version, command_id="wrong1", principal=OWNER)
        assert proposals.invalidate(db, OWNER, assertion.id, "marked wrong") == [mine["id"]]
    assert by_title(rows(got.source_id), "circulate")["disposition"] == "superseded"
    process(meeting(LINES, rev="r2"))  # re-extraction does not revive a wrong interpretation
    assert by_title(rows(got.source_id), "circulate")["disposition"] == "superseded"


def test_paused_owner_refuses_generation():
    got = receive(meeting())
    drain()
    with session_scope() as db:
        db.execute(m.ContextProposal.__table__.delete())
        pipeline.pause(db, OWNER)
        job = enqueue_job(db, OWNER, "context_proposals", {"source_id": got.source_id, "revision_id": got.revision_id})
    pipeline.perform(job.id)
    with session_scope() as db:
        assert db.scalar(select(m.ContextRun.status).where(m.ContextRun.job_id == job.id)) == "stale"
    assert rows(got.source_id) == []


def test_lint_proposals_are_left_alone():
    got = process(meeting())
    with session_scope() as db:
        db.add(m.ContextProposal(owner_id=OWNER, source_id=None, kind="suggestion", fingerprint="lint-x", title="Lint",
                                 payload={"origin": "lint", "check": "x", "key": "k"}))
        db.flush()
        apply.lock(db, OWNER)
        proposals.generate(db, OWNER, got.source_id, got.revision_id)
        lint = db.scalar(select(m.ContextProposal).where(m.ContextProposal.fingerprint == "lint-x"))
        assert (lint.revision, lint.disposition) == (1, "pending")


def test_action_tokens_ignore_dates_names_and_stopwords():
    assert proposals.action_tokens("I'll send Priya the revised budget by Friday 2026-11-20", {"priya"}) == \
        ["budget", "revised", "send"]
    assert proposals.jaccard(["a", "b"], ["b", "c"]) == pytest.approx(1 / 3)


# ---------------------------------------------------------------- the stage


def test_reconcile_queues_proposals_once_and_an_empty_payload_is_a_no_op():
    got = receive(meeting())
    drain()
    queued = [j for j in jobs("context_proposals") if j[2].get("revision_id") == got.revision_id]
    assert len(queued) == 1 and queued[0][1] == "succeeded" and queued[0][2]["extract_run_id"]
    assert len(rows(got.source_id)) == 3
    receive(meeting())  # duplicate delivery of the same revision
    drain()
    assert len([j for j in jobs("context_proposals") if j[2].get("revision_id") == got.revision_id]) == 1
    with session_scope() as db:
        enqueue_job(db, OWNER, "context_proposals", {})
    drain()
    status, _, result = jobs("context_proposals")[-1][1:]
    assert (status, result) == ("succeeded", {"skipped": "nothing to do"})
