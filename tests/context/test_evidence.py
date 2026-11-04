"""Evidence roles, independence and the survival rule under source deletion."""

import pytest
from jarvis.context import decisions, evidence, temporal
from jarvis.context.models import ContextBrief, ContextEvidence
from jarvis.context.schema import EvidenceRef
from jarvis.db import session_scope
from jarvis.models import now
from sqlalchemy import select

from .test_temporal import OWNER, World, days

TEXT = "Launch on Postgres."


def quote(text=TEXT, **kw):
    return [EvidenceRef(segment_seq=0, quote=text, char_start=0, char_end=len(text), **kw)]


def supported_twice(w):
    """One decision supported by two independent sources; returns (assertion, first source, second source)."""
    first = w.source(TEXT)
    row = w.assertion(TEXT, days(-3), kind="decision", key=None, subject=("project", "p1"), source=first)
    second = w.source(TEXT)
    evidence.attach(w.db, OWNER, row, quote(), revision_id=second[1].id)
    return row, first[0], second[0]


def test_deleting_one_of_two_sources_keeps_the_assertion_and_the_last_retracts_it():
    with session_scope() as db:
        w = World(db)
        row, first, second = supported_twice(w)
        brief = ContextBrief(owner_id=OWNER, entity_type="project", entity_id="p1", fingerprint="f" * 64,
                             dependencies={"assertions": [row.id]})
        db.add(brief)
        db.flush()
        assert evidence.corroboration(db, OWNER, row.id) == 2
        first.deleted_at = now()
        assert evidence.propagate_source_deletion(db, OWNER, first.id) == {
            "retracted": [], "kept": [row.id], "revalidate": []}
        assert row.lifecycle == "current" and brief.stale is True
        scrubbed = db.scalars(select(ContextEvidence).where(ContextEvidence.source_id == first.id)).all()
        assert scrubbed and all(e.quote == "" and e.segment_id is None for e in scrubbed)
        assert evidence.corroboration(db, OWNER, row.id) == 1
        assert evidence.propagate_source_deletion(db, OWNER, first.id)["kept"] == [row.id]  # idempotent
        result = evidence.propagate_source_deletion(db, OWNER, second.id)
        assert result["retracted"] == [row.id] and row.lifecycle == "retracted"
        assert evidence.propagate_source_deletion(db, OWNER, second.id)["retracted"] == []


def test_contradiction_and_derived_evidence_never_corroborate():
    with session_scope() as db:
        w = World(db)
        source = w.source(TEXT)
        row = w.assertion(TEXT, days(-3), kind="decision", key=None, subject=("project", "p1"), source=source)
        other, other_rev = w.source(TEXT)
        evidence.attach(db, OWNER, row, quote(origin="derived"), revision_id=other_rev.id,
                        derivation={"from": "computation"})
        evidence.attach(db, OWNER, row, quote(role="contradiction"), revision_id=other_rev.id)
        assert evidence.corroboration(db, OWNER, row.id) == 1
        assert evidence.propagate_source_deletion(db, OWNER, source[0].id)["retracted"] == [row.id]
        kept = db.scalars(select(ContextEvidence).where(ContextEvidence.source_id == other.id)).all()
        assert {e.role for e in kept} == {"support", "contradiction"}  # retained as provenance


def test_conjunctive_support_needs_revalidation_when_partly_deleted():
    with session_scope() as db:
        w = World(db)
        a = w.source(TEXT)
        row = w.assertion(TEXT, days(-3), kind="decision", key=None, subject=("project", "p1"), source=a)
        db.execute(ContextEvidence.__table__.update().where(ContextEvidence.assertion_id == row.id)
                   .values(derivation={"all_of": "g1"}))
        b = w.source(TEXT)
        evidence.attach(db, OWNER, row, quote(), revision_id=b[1].id, derivation={"all_of": "g1"})
        assert evidence.support_status(db, OWNER, row) == "supported"
        assert evidence.propagate_source_deletion(db, OWNER, a[0].id)["revalidate"] == [row.id]
        assert row.lifecycle == "current"
        assert evidence.propagate_source_deletion(db, OWNER, b[0].id)["retracted"] == [row.id]


def test_deleting_the_current_due_dates_only_source_reopens_the_previous_value():
    with session_scope() as db:
        w = World(db)
        old = w.assertion("Due 2026-11-13.", days(-9))
        new_source = w.source("Due 2026-11-20.")
        new = w.assertion("Due 2026-11-20.", days(-2), source=new_source)
        w.reconcile([old, new])
        assert old.lifecycle == "superseded"
        evidence.propagate_source_deletion(db, OWNER, new_source[0].id)
        assert (new.lifecycle, old.lifecycle, old.valid_to) == ("retracted", "current", None)


def test_human_stated_successor_survives_source_deletion():
    with session_scope() as db:
        w = World(db)
        source = w.source("Due 2026-11-13.")
        old = w.assertion("Due 2026-11-13.", days(-9), source=source)
        made = decisions.changed(db, OWNER, old.id, old.current_version, statement="Due 2026-11-27.",
                                 effective_at=days(-1), command_id="cmd-human", principal=OWNER)
        successor = db.get(temporal.ContextAssertion, made.after["successor_id"])
        result = evidence.propagate_source_deletion(db, OWNER, source[0].id)
        assert result["retracted"] == [old.id] and successor.id in result["kept"]
        assert successor.lifecycle == "current"


def test_agent_stated_successor_does_not_survive_on_derived_evidence():
    with session_scope() as db:
        w = World(db)
        source = w.source("Due 2026-11-13.")
        old = w.assertion("Due 2026-11-13.", days(-9), source=source)
        made = decisions.changed(db, OWNER, old.id, old.current_version, statement="Due 2026-11-27.",
                                 effective_at=days(-1), command_id="ui-agent:cmd", principal="bot")
        assert made.origin == "agent"
        successor = db.get(temporal.ContextAssertion, made.after["successor_id"])
        assert evidence.propagate_source_deletion(db, OWNER, source[0].id)["retracted"] == sorted([old.id,
                                                                                                    successor.id])


def test_attach_requires_exact_quotes_and_stores_repeats_once():
    with session_scope() as db:
        w = World(db)
        source = w.source(TEXT)
        row = w.assertion(TEXT, days(-3), kind="decision", key=None, subject=("project", "p1"), source=source)
        assert evidence.attach(db, OWNER, row, quote(), revision_id=source[1].id) == []
        with pytest.raises(ValueError):
            evidence.attach(db, OWNER, row, quote("Launch on MySQL."), revision_id=source[1].id)
