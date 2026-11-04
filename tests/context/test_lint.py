"""Weekly lint: the dream slot and its gates, contradictions routed to Questions only when consequential,
overdue untracked work, recurring unresolved mentions, duplicate identities, incremental brief rebuilds,
idempotence over ten unchanged runs, durable dismissals, budget limits and resumption after interruption."""

from datetime import UTC, datetime

import pytest
from jarvis.context import briefs, decisions, evidence, lint, temporal
from jarvis.context import settings as context_settings
from jarvis.context.models import (
    ContextAssertion,
    ContextBrief,
    ContextEntityLink,
    ContextOwnerState,
    ContextProposal,
    ContextRun,
)
from jarvis.context.native.reconcile import NativeReconciler
from jarvis.context.schema import (
    BriefClaim,
    BriefSection,
    CandidateAssertion,
    EntityRef,
    EvidenceRef,
    TimeBound,
)
from jarvis.db import session_scope
from jarvis.domain import enqueue_job
from jarvis.models import Job, OwnerSettings, Project
from sqlalchemy import func, select

from .test_briefs import HUMAN, People
from .test_temporal import OWNER, days


@pytest.fixture(autouse=True)
def enabled(monkeypatch):
    monkeypatch.setattr(context_settings, "enabled", lambda: True)


class Check:
    """A model double for the reconciler: Postgres and MySQL launch decisions contradict each other."""

    def __init__(self, fail_after: int | None = None):
        self.calls, self.fail_after = [], fail_after

    def __call__(self, owner, items):
        if self.fail_after is not None and len(self.calls) >= self.fail_after:
            raise RuntimeError("model timeout")
        self.calls.append(items)
        pairs = []
        for item in items:
            for candidate in item["candidates"]:
                if {"Postgres" in item["statement"], "Postgres" in candidate["statement"]} == {True, False} \
                        and "MySQL" in item["statement"] + candidate["statement"]:
                    pairs.append({"item_id": item["item_id"], "candidate_id": candidate["candidate_id"],
                                  "relation": "contradiction"})
        return pairs


def engine(check):
    return NativeReconciler(check)


def mentioned(w, statement, name):
    """A commitment by someone named but not matched to a person, in its own meeting."""
    source = w.source(statement, days(-3))
    candidate = CandidateAssertion(
        kind="commitment", statement=statement, subject=EntityRef(entity_type="project", entity_id=w.project),
        valid_from=TimeBound(at=days(-3), precision="day"), responsible=[EntityRef(entity_type="actor", mention=name)],
        evidence=[EvidenceRef(segment_seq=0, quote=statement, char_start=0, char_end=len(statement))])
    row = temporal.create(w.db, OWNER, candidate, occurred_at=days(-3), fallback_subject=("source", source[0].id))
    evidence.attach(w.db, OWNER, row, candidate.evidence, revision_id=source[1].id)
    return row


def world(db):
    """A project with contradictory launch decisions, an overdue untracked commitment of the owner's, a person
    named in two meetings but not matched, two actors called Alex, and a stale project brief."""
    w = People(db)
    made = {
        "postgres": w.say("Launch the pilot on Postgres.", kind="decision"),
        "mysql": w.say("Launch the pilot on MySQL.", kind="decision"),
        "overdue": w.say("I send the signed contract.", kind="commitment", responsible=[w.me], due=days(-2)),
        "tracked": w.say("I book the venue.", kind="commitment", responsible=[w.me], due=days(-2)),
        "sam1": mentioned(w, "Sam drafts the agenda.", "Sam"),
        "sam2": mentioned(w, "Sam reviews the budget.", "Sam"),
    }
    w.link_task(made["tracked"], w.task("Book the venue"))
    for row in (made["sam1"], made["sam2"]):
        db.add(ContextEntityLink(owner_id=OWNER, assertion_id=row.id, subject_type="assertion", subject_id=row.id,
                                 mention="Sam", role="responsible", status="unresolved"))
    w.alex1, w.alex2 = w.actor("Alex"), w.actor("Alex")
    brief = briefs.build_brief(db, OWNER, "project", w.project)
    brief.stale = True
    db.flush()
    return w, made


def counts(db):
    return {"proposals": db.scalar(select(func.count()).select_from(ContextProposal)),
            "briefs": db.scalar(select(func.count()).select_from(ContextBrief)),
            "questions": sorted((q["key"], q["revision"]) for q in decisions.question_items(db, OWNER))}


def lint_proposals(db, check):
    return {p.payload["key"]: p for p in db.scalars(select(ContextProposal).where(
        ContextProposal.payload["check"].astext == check))}


def state(db):
    row = db.get(ContextOwnerState, OWNER)
    db.refresh(row)
    return row.settings["lint"]


# ---------------------------------------------------------------- the slot


def test_due_uses_the_owner_local_dream_slot_across_dst_and_the_memory_review_gates():
    with session_scope() as db:
        db.add(OwnerSettings(owner_id=OWNER, values={"timezone": "America/New_York"}))
        db.flush()
        assert lint.due(db, OWNER, datetime(2026, 11, 2, 12, tzinfo=UTC)) is False  # no context yet
        People(db).source("A meeting.")
        # DST ended on Sunday 2026-11-01 at 02:00 local: the 03:00 slot is 08:00 UTC that day, 07:00 the week before.
        assert lint.slot(db, OWNER, datetime(2026, 11, 1, 7, 59, tzinfo=UTC)) == "2026-10-25T07:00:00+00:00"
        assert lint.slot(db, OWNER, datetime(2026, 11, 1, 8, tzinfo=UTC)) == "2026-11-01T08:00:00+00:00"
        monday = datetime(2026, 11, 2, 12, tzinfo=UTC)
        assert lint.due(db, OWNER, monday) is True
        lint.run(db, OWNER, now=monday, reconciler=engine(Check()))
        assert lint.due(db, OWNER, monday) is False
        assert lint.due(db, OWNER, datetime(2026, 11, 8, 7, 59, tzinfo=UTC)) is False  # Sunday 02:59 local
        assert lint.due(db, OWNER, datetime(2026, 11, 8, 8, tzinfo=UTC)) is True
        settings = db.get(OwnerSettings, OWNER)
        settings.values = {**settings.values, "deep_sleep_enabled": False}
        db.flush()
        assert lint.due(db, OWNER, datetime(2026, 11, 8, 8, tzinfo=UTC)) is False
        settings.values = {**settings.values, "deep_sleep_enabled": True}
        db.get(ContextOwnerState, OWNER).paused = True
        db.flush()
        assert lint.due(db, OWNER, datetime(2026, 11, 8, 8, tzinfo=UTC)) is False
        assert lint.run(db, OWNER) == {"status": "paused"}


def test_due_is_off_while_the_context_platform_is_disabled(monkeypatch):
    monkeypatch.setattr(context_settings, "enabled", lambda: False)
    with session_scope() as db:
        People(db).source("A meeting.")
        assert lint.due(db, OWNER) is False


# ---------------------------------------------------------------- one pass


def test_a_pass_reviews_contradictions_overdue_work_mentions_duplicates_and_stale_briefs():
    with session_scope() as db:
        w, made = world(db)
        check = Check()
        result = lint.run(db, OWNER, reconciler=engine(check))
        assert result["status"] == "succeeded" and result["pending"] == 0 and result["contested"] == 1
        assert len(check.calls) == 1
        db.refresh(made["postgres"])
        assert made["postgres"].lifecycle == "contested"
        questions = decisions.question_items(db, OWNER)
        contested = [q for q in questions if q["evidence"].get("predicate") == "decision"]
        assert len(contested) == 1 and {c["assertion_id"] for c in contested[0]["candidates"]} == \
            {made["postgres"].id, made["mysql"].id}

        overdue = lint_proposals(db, "overdue_untracked")
        assert list(overdue) == [made["overdue"].id]  # the tracked one has a task
        task = overdue[made["overdue"].id]
        assert (task.kind, task.source_id, task.title, task.responsible, task.assertion_ids) == \
            ("create", None, "I send the signed contract.", w.me, [made["overdue"].id])
        assert task.evidence and task.evidence[0]["quote"] == "I send the signed contract."

        mentions = lint_proposals(db, "recurring_mention")
        assert list(mentions) == ["sam"] and mentions["sam"].payload["name"] == "Sam"
        assert mentions["sam"].payload["action"] == "actor.create" and len(mentions["sam"].payload["source_ids"]) == 2
        merges = lint_proposals(db, "duplicate_identity")
        assert list(merges) == [":".join(sorted((w.alex1, w.alex2)))]
        assert merges[next(iter(merges))].title == "Same person? Alex and Alex"

        brief = briefs.latest(db, OWNER, "project", w.project)
        assert (brief.version, brief.stale, result["briefs_rebuilt"]) == (2, False, 1)
        unknowns = next(s for s in brief.sections if s["name"] == "unknowns")
        assert any(c["text"].startswith("Contested: ") for c in unknowns["claims"])
        run = db.scalar(select(ContextRun).where(ContextRun.stage == "lint"))
        assert (run.status, run.stage_results["last"]["contested"]) == ("succeeded", 1)
        assert state(db)["completed"] == result["period"] and state(db)["pending"] == []


def test_ten_runs_without_new_evidence_create_nothing_and_skip_the_model():
    with session_scope() as db:
        world(db)
        check = Check()
        lint.run(db, OWNER, reconciler=engine(check))
        before = counts(db)
        for _ in range(10):
            result = lint.run(db, OWNER, reconciler=engine(check))
            assert (result["status"], result.get("proposals_created", 0), result.get("briefs_rebuilt", 0)) == \
                ("succeeded", 0, 0)
        assert counts(db) == before and len(check.calls) == 1


def test_the_job_takes_one_model_call_per_transaction_and_ten_jobs_change_nothing():
    synth = Synth()
    with session_scope() as db:
        w, _ = world(db)
        project = w.project
        briefs.build_brief(db, OWNER, "project", project, synthesizer=synth)
    check = Check()
    first = lint.perform(OWNER, reconciler=engine(check), synthesizer=synth)
    assert (first["status"], first["contested"], len(check.calls), synth.calls) == ("succeeded", 1, 1, 2)
    with session_scope() as db:
        assert briefs.latest(db, OWNER, "project", project).watermarks["engine"] == "synthesized"
        before = counts(db)
    for _ in range(10):
        result = lint.perform(OWNER, reconciler=engine(check), synthesizer=synth)
        assert result["status"] == "succeeded" and not result.get("proposals_created")
    with session_scope() as db:
        assert counts(db) == before
    assert (len(check.calls), synth.calls) == (1, 2)


def test_contradictions_that_are_not_consequential_stay_out_of_questions():
    with session_scope() as db:
        w = People(db)
        first = w.say("Postgres keeps operations simple.", kind="rationale")
        w.say("MySQL keeps operations simple.", kind="rationale")
        lint.run(db, OWNER, reconciler=engine(Check()))
        db.refresh(first)
        assert first.lifecycle == "contested" and decisions.question_items(db, OWNER) == []


def test_an_answered_question_is_never_asked_again():
    with session_scope() as db:
        w, made = world(db)
        check = Check()
        lint.run(db, OWNER, reconciler=engine(check))
        question = next(q for q in decisions.question_items(db, OWNER) if q["evidence"].get("predicate") == "decision")
        decisions.resolve_question(db, OWNER, question_key=question["key"], expected_revision=question["revision"],
                                   action="both", **HUMAN)
        w.say("Launch the pilot on MySQL with replicas.", kind="decision")  # new evidence about the subject
        lint.run(db, OWNER, reconciler=engine(check))
        assert len(check.calls) == 2  # the neighbourhood changed, so it was reviewed again
        for row in (made["postgres"], made["mysql"]):
            db.refresh(row)
            assert row.lifecycle == "current"
        assert not [q for q in decisions.question_items(db, OWNER) if q["evidence"].get("predicate") == "decision"]


# ---------------------------------------------------------------- lint proposals


def test_dismissed_findings_stay_dismissed_and_resolved_findings_are_superseded():
    with session_scope() as db:
        w, made = world(db)
        lint.run(db, OWNER, reconciler=engine(Check()))
        mention = lint_proposals(db, "recurring_mention")["sam"]
        decisions.dismiss(db, OWNER, mention.id, mention.revision, **HUMAN)
        decisions.no_merge(db, OWNER, w.alex1, w.alex2, **HUMAN)
        w.link_task(made["overdue"], w.task("Send the signed contract"))
        result = lint.run(db, OWNER, reconciler=engine(Check()))
        assert result.get("proposals_created", 0) == 0 and result["proposals_superseded"] == 2
        assert lint_proposals(db, "recurring_mention")["sam"].disposition == "dismissed"
        assert {p.disposition for p in lint_proposals(db, "duplicate_identity").values()} == {"superseded"}
        assert lint_proposals(db, "overdue_untracked")[made["overdue"].id].disposition == "superseded"
        db.add(ContextEntityLink(owner_id=OWNER, assertion_id=made["sam1"].id, subject_type="assertion",
                                 subject_id=made["sam1"].id, mention="Sam", role="recipient", status="unresolved"))
        lint.run(db, OWNER, reconciler=engine(Check()))
        assert db.scalar(select(func.count()).select_from(ContextProposal)) == 3


# ---------------------------------------------------------------- briefs


class Synth:
    def __init__(self):
        self.calls, self.last_cost = 0, 0.0

    def synthesize(self, request):
        self.calls += 1
        self.last_cost = 0.02
        ids = [a.id for a in request.assertions][:1]
        return [BriefSection(name=name, claims=[BriefClaim(text="Synthesized.", assertion_ids=ids)] if ids else [],
                             gaps=[] if ids else ["Nothing."]) for name in request.brief.sections]


def test_a_synthesized_brief_is_resynthesized_only_when_its_inputs_change():
    with session_scope() as db:
        w = People(db)
        w.say("The pilot covers 40 clinics.")
        synth = Synth()
        briefs.build_brief(db, OWNER, "project", w.project, synthesizer=synth)
        lint.run(db, OWNER, reconciler=engine(Check()), synthesizer=synth)
        assert synth.calls == 1
        w.say("The pilot adds a second region.")
        result = lint.run(db, OWNER, reconciler=engine(Check()), synthesizer=synth)
        latest = briefs.latest(db, OWNER, "project", w.project)
        assert (synth.calls, latest.version, latest.watermarks["engine"]) == (2, 2, "synthesized")
        assert result["spent_usd"] == pytest.approx(0.02)


# ---------------------------------------------------------------- budget and resumption


def test_a_spent_budget_leaves_the_rest_pending_for_the_next_run():
    with session_scope() as db:
        w, _ = world(db)
        check = Check()
        result = lint.run(db, OWNER, reconciler=engine(check), budget_usd=0)
        assert (result["status"], result["budget_exhausted"]) == ("partial", 1) and result["pending"] >= 1
        assert check.calls == [] and lint_proposals(db, "overdue_untracked")  # global checks need no model
        assert lint.due(db, OWNER) is False  # this week's slot is done; the rest waits for the next run
        assert ["project", w.project] in state(db)["pending"]
        result = lint.run(db, OWNER, reconciler=engine(check))
        assert (result["status"], result["pending"], len(check.calls)) == ("succeeded", 0, 1)
        assert state(db)["high"] is None and state(db)["cursor"]


def test_an_interrupted_job_resumes_where_it_stopped():
    with session_scope() as db:
        w = People(db)
        project = Project(owner_id=OWNER, name="Second pilot")
        db.add(project)
        db.flush()
        projects = {w.project, project.id}
        for subject in projects:
            w.say("Launch the pilot on Postgres.", kind="decision", subject=("project", subject))
            w.say("Launch the pilot on MySQL.", kind="decision", subject=("project", subject))
    crashing = Check(fail_after=1)
    with pytest.raises(RuntimeError):
        lint.perform(OWNER, reconciler=engine(crashing))
    with session_scope() as db:
        left = {i for t, i in state(db)["pending"] if t == "project"}
        assert len(left) == 1 and len(crashing.calls) == 1  # the first neighbourhood's unit was committed
        assert lint.due(db, OWNER) is True  # the slot is not done until the pass finishes or the budget stops it
        failed = db.scalar(select(ContextRun).where(ContextRun.stage == "lint"))
        assert (failed.status, failed.error) == ("failed", "RuntimeError")
    check = Check()
    result = lint.perform(OWNER, reconciler=engine(check))
    assert (result["status"], result["pending"], len(check.calls)) == ("succeeded", 0, 1)
    assert result["contested"] == 1  # only the neighbourhood the crash left pending
    assert {i["item_id"] for i in check.calls[0]} <= _ids_about(left.pop())
    with session_scope() as db:
        assert lint.due(db, OWNER) is False
        run = db.scalar(select(ContextRun).where(ContextRun.stage == "lint"))
        assert run.status == "succeeded" and db.scalar(select(func.count()).select_from(ContextRun)) == 1


def _ids_about(project_id):
    with session_scope() as db:
        return set(db.scalars(select(ContextAssertion.id).where(ContextAssertion.subject_id == project_id)))


def test_a_pass_stops_when_the_owner_is_paused_or_the_epoch_moves(monkeypatch):
    with session_scope() as db:
        world(db)
        db.add(ContextOwnerState(owner_id=OWNER))
        db.flush()
        real = lint._neighbourhood

        def bump(db, job, entity, reconciled):
            db.get(ContextOwnerState, OWNER).epoch += 1
            return real(db, job, entity, reconciled)

        monkeypatch.setattr(lint, "_neighbourhood", bump)
        result = lint.run(db, OWNER, reconciler=engine(Check()))
        assert result["status"] == "stale" and result["neighbourhoods"] == 1 and result["pending"] >= 1
        assert lint.due(db, OWNER) is True


# ---------------------------------------------------------------- job dispatch


def test_the_context_lint_job_runs_through_the_pipeline(monkeypatch):
    from jarvis.context import pipeline

    seen = []
    monkeypatch.setattr(lint, "perform", lambda owner, budget_usd=None: seen.append((owner, budget_usd)) or
                        {"status": "succeeded"})
    with session_scope() as db:
        job = enqueue_job(db, OWNER, "context_lint", {"period": "2026-11-01T08:00:00+00:00", "budget_usd": 0.2})
        job_id = job.id
    pipeline.perform(job_id)
    with session_scope() as db:
        job = db.get(Job, job_id)
        assert (job.status, job.result, seen) == ("succeeded", {"status": "succeeded"}, [(OWNER, 0.2)])
