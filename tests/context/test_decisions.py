"""Decisions: seq allocation, derived origin, Wrong vs Changed, links, proposals, undo, briefs, Questions."""

from uuid import uuid4

import pytest
from jarvis import bot_access, review_questions
from jarvis.context import decisions, temporal
from jarvis.context.models import (
    ContextBrief,
    ContextDecision,
    ContextEntityLink,
    ContextOwnerState,
    ContextProposal,
)
from jarvis.db import session_scope
from jarvis.domain import DomainError, execute
from sqlalchemy import select

from .test_temporal import OWNER, World, days

HUMAN = {"command_id": "cmd-human", "principal": OWNER}


def contested_pair(w):
    a = w.assertion("Due 2026-11-20.", days(-3))
    b = w.assertion("Due 2026-11-27.", days(-3))
    w.reconcile([a, b])
    assert (a.lifecycle, b.lifecycle) == ("contested", "contested")
    return a, b


def test_decision_seq_is_allocated_per_owner_under_lock():
    with session_scope() as db:
        assert [decisions.next_decision_seq(db, OWNER) for _ in range(3)] == [1, 2, 3]
        assert decisions.next_decision_seq(db, "other") == 1
        assert db.get(ContextOwnerState, OWNER).decision_seq == 3
    with session_scope() as db:
        state = db.get(ContextOwnerState, "third") or ContextOwnerState(owner_id="third", decision_seq=41)
        db.add(state)
        db.flush()
        assert decisions.next_decision_seq(db, "third") == 42
        assert decisions.watermark(db, "third") == 0


def test_origin_is_derived_and_agents_are_never_human():
    assert decisions.origin_of("plain-browser-id") == "human"
    assert decisions.origin_of("plain-browser-id", requested="agent") == "agent"
    assert decisions.origin_of("ui-agent:abc") == "agent"
    assert decisions.origin_of("mcp:abc", requested="human") == "agent"
    assert decisions.origin_of(None) == "system"
    with bot_access.bind("credential-1"):
        assert decisions.origin_of("plain-browser-id") == "agent"
        assert decisions.origin_of(None) == "agent"
    with session_scope() as db:
        row = World(db).assertion("Due 2026-11-20.", days(-3))
        with bot_access.bind("credential-1"):
            made = decisions.confirm(db, OWNER, row.id, row.current_version, command_id="plain-id",
                                     principal="credential-1")
        assert made.origin == "agent" and made.principal == "credential-1"


def test_wrong_retracts_records_a_negative_and_stales_briefs_and_undo_restores():
    with session_scope() as db:
        w = World(db)
        old = w.assertion("Due 2026-11-13.", days(-9))
        new = w.assertion("Due 2026-11-20.", days(-2))
        w.reconcile([old, new])
        brief = ContextBrief(owner_id=OWNER, entity_type="task", entity_id="task-1", fingerprint="f" * 64,
                             dependencies={"assertions": {new.id: 1}})
        untouched = ContextBrief(owner_id=OWNER, entity_type="task", entity_id="task-2", fingerprint="g" * 64,
                                 dependencies={"assertions": ["elsewhere"]})
        db.add_all([brief, untouched])
        db.flush()
        made = decisions.wrong(db, OWNER, new.id, new.current_version, reason="misheard", **HUMAN)
        assert (made.origin, made.negative, made.operation, made.seq) == ("human", True, "wrong", 1)
        assert new.lifecycle == "retracted" and (old.lifecycle, old.valid_to) == ("current", None)
        assert brief.stale and not untouched.stale
        assert w.assertion("Due 2026-11-20.", days(-1)).lifecycle == "retracted"
        with pytest.raises(DomainError) as stale:
            decisions.wrong(db, OWNER, old.id, old.current_version - 1, **HUMAN)
        assert stale.value.status == 409
        undo = decisions.undo(db, OWNER, made.id, **HUMAN)
        assert undo.undoes_id == made.id and undo.operation == "undo"
        assert new.lifecycle == "current" and old.lifecycle == "superseded"
        assert w.assertion("Due 2026-11-20.", days(-1)).lifecycle != "retracted"
        with pytest.raises(DomainError):
            decisions.undo(db, OWNER, made.id, **HUMAN)


def test_wrong_and_changed_have_different_histories():
    with session_scope() as db:
        w = World(db)
        wrong = w.assertion("Launch on MySQL.", days(-5), kind="decision", key=None, subject=("project", "p1"))
        changed = w.assertion("Launch on Postgres.", days(-5), kind="decision", key=None, subject=("project", "p2"))
        decisions.wrong(db, OWNER, wrong.id, 1, **HUMAN)
        made = decisions.changed(db, OWNER, changed.id, 1, statement="Launch on Postgres 17.", **HUMAN)
        successor = db.get(temporal.ContextAssertion, made.after["successor_id"])
        assert [v.lifecycle for v in temporal.history(db, OWNER, wrong.id)] == ["current", "retracted"]
        assert [v.lifecycle for v in temporal.history(db, OWNER, changed.id)] == ["current", "superseded"]
        assert successor.supersedes_id == changed.id and successor.lifecycle == "current"
        assert temporal.history(db, OWNER, successor.id)[0].decision_id == made.id
        assert changed.valid_to == successor.valid_from
        decisions.undo(db, OWNER, made.id, **HUMAN)
        assert (successor.lifecycle, changed.lifecycle, changed.valid_to) == ("retracted", "current", None)
        with pytest.raises(DomainError):
            decisions.changed(db, OWNER, wrong.id, wrong.current_version, statement="x", **HUMAN)


def test_undo_refuses_when_a_later_decision_depends_on_it():
    with session_scope() as db:
        w = World(db)
        row = w.assertion("Launch on MySQL.", days(-5), kind="decision", key=None, subject=("project", "p1"))
        first = decisions.confirm(db, OWNER, row.id, 1, **HUMAN)
        decisions.wrong(db, OWNER, row.id, row.current_version, **HUMAN)
        with pytest.raises(DomainError):
            decisions.undo(db, OWNER, first.id, **HUMAN)


def test_links_confirm_reject_and_undo():
    with session_scope() as db:
        w = World(db)
        row = w.assertion("Due 2026-11-20.", days(-3))
        link = ContextEntityLink(owner_id=OWNER, assertion_id=row.id, subject_type="assertion", subject_id=row.id,
                                 mention="Priya", role="responsible", status="inferred", target_type="actor",
                                 target_id="actor-a")
        db.add(link)
        db.flush()
        confirmed = decisions.confirm_link(db, OWNER, link.id, 1, **HUMAN)
        assert (link.status, link.revision, link.decision_id) == ("confirmed", 2, confirmed.id)
        rejected = decisions.reject_link(db, OWNER, link.id, 2, **HUMAN)
        assert link.status == "rejected" and rejected.negative
        decisions.undo(db, OWNER, rejected.id, **HUMAN)
        assert (link.status, link.target_id, link.revision) == ("confirmed", "actor-a", 4)
        pair = decisions.no_merge(db, OWNER, "actor-b", "actor-a", **HUMAN)
        assert pair.target_id == "actor-a|actor-b" and pair.negative
        merged = decisions.merge(db, OWNER, "actor-a", "actor-b", **HUMAN)
        assert merged.after == {"winner_id": "actor-a", "loser_id": "actor-b"}
        with pytest.raises(DomainError):
            decisions.no_merge(db, OWNER, "actor-a", "actor-a", **HUMAN)


def test_proposal_decisions_are_revision_bound_and_undoable():
    with session_scope() as db:
        proposal = ContextProposal(owner_id=OWNER, kind="create", fingerprint="p" * 64, title="Send budget")
        db.add(proposal)
        db.flush()
        with pytest.raises(DomainError):
            decisions.approve(db, OWNER, proposal.id, 2, **HUMAN)
        later = decisions.later(db, OWNER, proposal.id, 1, until=days(2), **HUMAN)
        assert (proposal.disposition, proposal.revision, later.effective_at) == ("later", 2, days(2))
        approved = decisions.approve(db, OWNER, proposal.id, 2, **HUMAN)
        assert proposal.disposition == "approved"
        decisions.undo(db, OWNER, approved.id, **HUMAN)
        assert (proposal.disposition, proposal.revision) == ("later", 4)
        dismissed = decisions.dismiss(db, OWNER, proposal.id, 4, **HUMAN)
        proposal.disposition = "published"
        with pytest.raises(DomainError):
            decisions.undo(db, OWNER, dismissed.id, **HUMAN)


def test_contested_items_reach_the_questions_inbox_and_resolve():
    with session_scope() as db:
        a, b = contested_pair(World(db))
        page = review_questions.listing(db, OWNER)
        item = next(i for i in page["items"] if i["kind"] == "context")
        assert item["status"] == "pending" and item["agent_tool"] == "context_resolve"
        assert review_questions.listing(db, OWNER, category="organization")["items"] == []
        assert len(review_questions.listing(db, OWNER, category="context")["items"]) == 1
        key, revision = item["key"], item["revision"]
    with session_scope() as db:
        result = execute(db, OWNER, str(uuid4()), "review.defer",
                         {"question_key": key, "expected_revision": revision, "until": "week"})
        assert result["data"]["status"] == "deferred"
        deferred = review_questions.find(db, OWNER, key, result["data"]["revision"])
        assert deferred["status"] == "deferred"
        later = db.scalar(select(ContextDecision).where(ContextDecision.operation == "later"))
        assert later.target_type == "question" and later.origin == "human"  # a browser command ID with no bot credential
    with session_scope() as db:
        with pytest.raises(DomainError):
            decisions.resolve_question(db, OWNER, question_key=key, expected_revision=revision, action="keep",
                                       assertion_id=a.id, **HUMAN)
        current = review_questions.find(db, OWNER, key, deferred["revision"])
        with pytest.raises(DomainError):
            decisions.resolve_question(db, OWNER, question_key=key, expected_revision=current["revision"],
                                       action="both", **HUMAN)
        decisions.resolve_question(db, OWNER, question_key=key, expected_revision=current["revision"],
                                   action="keep", assertion_id=b.id, **HUMAN)
        a, b = db.get(temporal.ContextAssertion, a.id), db.get(temporal.ContextAssertion, b.id)
        assert (a.lifecycle, b.lifecycle) == ("retracted", "current")
        assert not [i for i in review_questions.listing(db, OWNER)["items"] if i["kind"] == "context"]


def test_unresolved_responsible_links_ask_who_and_low_stakes_contests_do_not():
    with session_scope() as db:
        w = World(db)
        row = w.assertion("Send the deck.", days(-1), key=None, subject=("project", "p1"))
        link = ContextEntityLink(owner_id=OWNER, assertion_id=row.id, subject_type="assertion", subject_id=row.id,
                                 mention="Sam", role="responsible", status="unresolved")
        db.add(link)
        facts = [w.assertion(t, days(-1), kind="fact", key=None, subject=("project", "p1")) for t in ("A.", "B.")]
        for fact in facts:
            temporal.record_version(db, fact, lifecycle="contested")
        db.flush()
        items = decisions.question_items(db, OWNER)
        assert len(items) == 1 and items[0]["question"].startswith("Who is “Sam”")
        decisions.resolve_question(db, OWNER, question_key=items[0]["key"], expected_revision=items[0]["revision"],
                                   action="link", entity_id="actor-sam", **HUMAN)
        assert (link.status, link.target_id) == ("confirmed", "actor-sam")
        assert decisions.question_items(db, OWNER) == []
