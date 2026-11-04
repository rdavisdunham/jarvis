"""Briefs: deterministic sections with citations, fingerprints that never re-version unchanged inputs, stale
marking in the correcting transaction, stale reads that never leak removed content, grounded synthesis."""

import pytest
from jarvis.context import briefs, decisions, evidence, temporal
from jarvis.context.models import ContextBrief, ContextEntityLink, ContextEvidence, ContextSource
from jarvis.context.native.synthesize import SynthesisFailed
from jarvis.context.schema import (
    BRIEF_SECTIONS,
    BriefClaim,
    BriefSection,
    CandidateAssertion,
    EntityRef,
    EvidenceRef,
    TimeBound,
)
from jarvis.db import session_scope
from jarvis.domain import DomainError
from jarvis.models import Actor, Project, Task, now
from sqlalchemy import func, select

from .test_temporal import OWNER, World, days

HUMAN = {"command_id": "cmd-human", "principal": OWNER}


class People(World):
    """A World with actors, a project and helpers for commitments with resolved people."""

    def __init__(self, db, owner=OWNER):
        super().__init__(db, owner)
        self.me = self.actor("owner")
        self.priya = self.actor("Priya")
        self.marco = self.actor("Marco")
        project = Project(owner_id=owner, name="Clinic pilot")
        db.add(project)
        db.flush()
        self.project = project.id

    def actor(self, name):
        row = Actor(owner_id=self.owner, name=name)
        self.db.add(row)
        self.db.flush()
        return row.id

    def task(self, title, status="open"):
        row = Task(owner_id=self.owner, title=title, status=status)
        self.db.add(row)
        self.db.flush()
        return row

    def say(self, statement, *, kind="fact", subject=None, valid_from=None, responsible=(), recipient=None,
            due=None, key=None, modality="asserted", source=None):
        """Admit one assertion supported by its own one-segment source (or `source`)."""
        valid_from = valid_from or days(-3)
        source = source or self.source(statement, valid_from)
        people = [EntityRef(entity_type="actor", entity_id=p) if p not in ("we", "Somebody")
                  else EntityRef(entity_type="actor", mention=p) for p in responsible]
        candidate = CandidateAssertion(
            kind=kind, statement=statement, key=key, modality=modality,
            subject=EntityRef(entity_type=subject[0], entity_id=subject[1]) if subject
            else EntityRef(entity_type="project", entity_id=self.project),
            valid_from=TimeBound(at=valid_from, precision="day"), responsible=people,
            recipient=EntityRef(entity_type="actor", entity_id=recipient) if recipient else None,
            due=TimeBound(at=due, precision="day") if due else None,
            evidence=[EvidenceRef(segment_seq=0, quote=statement, char_start=0, char_end=len(statement))])
        row = temporal.create(self.db, self.owner, candidate, occurred_at=valid_from,
                              fallback_subject=("source", source[0].id))
        evidence.attach(self.db, self.owner, row, candidate.evidence, revision_id=source[1].id)
        return row

    def link_task(self, row, task):
        self.db.add(ContextEntityLink(owner_id=self.owner, assertion_id=row.id, subject_type="assertion",
                                      subject_id=row.id, target_type="task", target_id=task.id, role="task",
                                      status="confirmed"))
        self.db.flush()


def pilot(w):
    """A small project history: a decision with its rationale, a fact, two commitments, an open question."""
    return {
        "decision": w.say("Launch the pilot on Postgres with pgvector.", kind="decision"),
        "why": w.say("Postgres keeps operations in one database.", kind="rationale"),
        "fact": w.say("The pilot covers 40 clinics.", valid_from=days(-30)),
        "budget": w.say("Priya sends the revised budget.", kind="commitment", responsible=[w.priya],
                        due=days(10)),
        "checklist": w.say("I draft the rollout checklist.", kind="commitment", responsible=[w.me]),
        "question": w.say("Which region goes first?", kind="open_question"),
    }


def cited(sections):
    return {i for s in sections for c in s["claims"] for i in c["assertion_ids"]}


def test_ten_rebuilds_without_new_evidence_make_no_new_version():
    with session_scope() as db:
        w = People(db)
        pilot(w)
        first = briefs.build_brief(db, OWNER, "project", w.project)
        for _ in range(10):
            assert briefs.build_brief(db, OWNER, "project", w.project).id == first.id
        assert briefs.get_brief(db, OWNER, "project", w.project)["basis"] == "stored"
        assert db.scalar(select(func.count()).select_from(ContextBrief)) == 1
        w.say("The pilot adds a second region.")
        second = briefs.build_brief(db, OWNER, "project", w.project)
        assert (second.version, second.id != first.id) == (2, True)
        assert set(second.dependencies["assertions"]) > set(first.dependencies["assertions"])


def test_deterministic_sections_cite_current_supported_assertions():
    with session_scope() as db:
        w = People(db)
        made = pilot(w)
        served = briefs.get_brief(db, OWNER, "project", w.project)
        assert served["basis"] == "rebuilt" and served["version"] == 1 and not served["stale"]
        sections = {s["name"]: s for s in served["sections"]}
        assert list(sections) == list(BRIEF_SECTIONS)
        assert all(c["assertion_ids"] or c["task_ids"] for s in served["sections"] for c in s["claims"])
        state = [c["text"] for c in sections["current_state_and_why"]["claims"]]
        assert state[:2] == ["Launch the pilot on Postgres with pgvector.",
                             "Why: Postgres keeps operations in one database."]
        loops = [c["text"] for c in sections["open_loops"]["claims"]]
        assert any(t.startswith("Waiting for Priya: Priya sends the revised budget. (due") for t in loops)
        assert "You owe: I draft the rollout checklist." in loops
        assert sections["unknowns"]["claims"][0]["text"] == "Open question: Which region goes first?"
        assert {made["checklist"].id} <= cited([sections["unknowns"]])  # no due date agreed
        assert sections["next_meeting_prep"]["gaps"] and not sections["next_meeting_prep"]["claims"]
        assert made["fact"].id not in cited([sections["recent_changes"]])  # effective 30 days ago


def test_correction_marks_the_brief_stale_in_the_same_transaction_and_stale_reads_drop_it():
    with session_scope() as db:
        w = People(db)
        made = pilot(w)
        built = briefs.build_brief(db, OWNER, "project", w.project)
        decisions.wrong(db, OWNER, made["decision"].id, made["decision"].current_version, **HUMAN)
        assert built.stale is True  # before commit: same transaction as the correction
        view = briefs.get_brief(db, OWNER, "project", w.project, rebuild_if_stale=False)
        assert (view["stale"], view["basis"], view["version"]) == (True, "current_view", 1)
        assert view["notice"] and made["decision"].id not in cited(view["sections"])
        assert made["decision"].id in built.dependencies["assertions"]  # the stored version still holds it
        rebuilt = briefs.get_brief(db, OWNER, "project", w.project)
        assert (rebuilt["version"], rebuilt["stale"]) == (2, False)
        assert made["decision"].id not in cited(rebuilt["sections"])


def test_source_deletion_stales_and_rebuild_forgets_the_deleted_statement():
    with session_scope() as db:
        w = People(db)
        made = pilot(w)
        built = briefs.build_brief(db, OWNER, "project", w.project)
        gone = db.scalar(select(ContextEvidence.source_id).where(ContextEvidence.assertion_id == made["fact"].id))
        assert gone in built.dependencies["sources"]
        assert briefs.mark_stale(db, OWNER, source_id=gone) == 1 and built.stale
        built.stale = False
        db.get(ContextSource, gone).deleted_at = now()
        evidence.propagate_source_deletion(db, OWNER, gone)
        assert built.stale and made["fact"].lifecycle == "retracted"
        assert briefs.get_brief(db, OWNER, "project", w.project, rebuild_if_stale=False)["stale"]
        rebuilt = briefs.get_brief(db, OWNER, "project", w.project)
        assert rebuilt["version"] == 2 and made["fact"].id not in cited(rebuilt["sections"])
        assert all("40 clinics" not in c["text"] for s in rebuilt["sections"] for c in s["claims"])


def test_task_changes_rebuild_and_completed_task_is_covered_work_not_a_kept_promise():
    with session_scope() as db:
        w = People(db)
        made = pilot(w)
        task = w.task("Rollout checklist")
        w.link_task(made["checklist"], task)
        first = briefs.build_brief(db, OWNER, "project", w.project)
        assert first.dependencies["tasks"] == {task.id: 1}
        assert briefs.mark_stale(db, OWNER, task_ids=[task.id]) == 1
        task.status, task.revision = "completed", 2
        db.flush()
        served = briefs.get_brief(db, OWNER, "project", w.project)
        assert served["version"] == 2
        loops = {s["name"]: s for s in served["sections"]}["open_loops"]["claims"]
        claim = next(c for c in loops if made["checklist"].id in c["assertion_ids"])
        assert claim["task_ids"] == [task.id]
        assert "is completed: that covers the tracked work, not necessarily the whole promise" in claim["text"]


def test_recent_changes_show_a_superseded_due_date_with_both_citations():
    with session_scope() as db:
        w = People(db)
        task = ("task", w.task("Budget").id)
        old = w.say("Budget due 2026-11-13.", kind="commitment", subject=task, key="due_date",
                    responsible=[w.priya], valid_from=days(-20))
        new = w.say("Budget due 2026-11-20.", kind="commitment", subject=task, key="due_date",
                    responsible=[w.priya], valid_from=days(-2))
        w.reconcile([old, new])
        assert old.lifecycle == "superseded"
        sections = {s["name"]: s for s in briefs.get_brief(db, OWNER, *task)["sections"]}
        change = sections["recent_changes"]["claims"][0]
        assert change["assertion_ids"] == [old.id, new.id] and change["text"].startswith("Changed (")


class Canned:
    def __init__(self, sections=None, error=None):
        self.sections, self.error, self.calls = sections, error, 0

    def synthesize(self, request):
        self.calls += 1
        if self.error:
            raise self.error
        return self.sections(request)


def test_synthesis_is_grounded_upgrades_once_and_failures_keep_deterministic_sections():
    with session_scope() as db:
        w = People(db)
        made = pilot(w)
        deterministic = briefs.build_brief(db, OWNER, "project", w.project)
        retracted = w.say("An old idea.", kind="decision")
        decisions.wrong(db, OWNER, retracted.id, retracted.current_version, **HUMAN)

        def sections(request):
            assert {a.id for a in request.assertions} == {r.id for r in made.values()}
            assert "next_meeting_prep" not in request.brief.sections
            return [BriefSection(name="current_state_and_why", claims=[
                BriefClaim(text="The pilot runs on Postgres because it keeps one database.",
                           assertion_ids=[made["decision"].id, made["why"].id]),
                BriefClaim(text="Invented.", assertion_ids=["not-an-assertion"]),
                BriefClaim(text="Revived.", assertion_ids=[retracted.id])])]

        model = Canned(sections)
        upgraded = briefs.build_brief(db, OWNER, "project", w.project, synthesizer=model)
        assert (upgraded.version, upgraded.watermarks["engine"]) == (2, "synthesized")
        state = upgraded.sections[0]
        assert [c["text"] for c in state["claims"]] == ["The pilot runs on Postgres because it keeps one database."]
        assert any("did not cite surviving evidence" in g for g in state["gaps"])
        assert upgraded.sections[2] == deterministic.sections[2]  # next meeting prep stays deterministic
        assert briefs.build_brief(db, OWNER, "project", w.project, synthesizer=model).id == upgraded.id
        assert briefs.build_brief(db, OWNER, "project", w.project).id == upgraded.id
        assert model.calls == 1

        w.say("A new fact.")
        failing = Canned(error=SynthesisFailed("source_budget"))
        fallback = briefs.build_brief(db, OWNER, "project", w.project, synthesizer=failing)
        assert (fallback.version, fallback.watermarks["engine"]) == (3, "deterministic")
        assert fallback.watermarks["synthesis_error"] == "source_budget"


def test_unknown_entities_are_refused_and_actor_briefs_include_their_commitments():
    with session_scope() as db:
        w = People(db)
        made = pilot(w)
        with pytest.raises(DomainError) as missing:
            briefs.get_brief(db, OWNER, "project", "nope")
        assert missing.value.status == 404
        with pytest.raises(DomainError):
            briefs.get_brief(db, OWNER, "meeting", "x")
        served = briefs.get_brief(db, OWNER, "actor", w.priya)
        assert made["budget"].id in cited(served["sections"])
        who = {s["name"]: s for s in served["sections"]}["who_is_who"]["claims"]
        assert who[0]["text"] == "Priya: responsible for 1 open item(s)"
