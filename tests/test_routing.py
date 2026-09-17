from datetime import UTC, datetime
import pytest
from sqlalchemy import select
from jarvis import routing, structure
from jarvis.db import session_scope
from jarvis.domain import DomainError
from jarvis.models import Job
from jarvis.structure_models import (
    RoutingPattern,
    RoutingObservation,
    RoutingReview,
    FieldUnderstanding,
)
from test_structure import create, run, definition, propose, apply, OWNER


def test_explicit_rule_routes_new_work_but_preserves_explicit_clear():
    parent = create(title="ABC")
    rule = run(
        "routing.create",
        {
            "phrase": "ABC CSR",
            "type_id": "task",
            "parent_id": parent["id"],
            "reason": "ABC is my work client",
        },
    )
    task = create("task", "Update ABC CSR docs")
    assert task["parent_id"] == parent["id"]
    assert create("task", "Update ABC CSR docs", parent_id=None)["parent_id"] is None
    with session_scope() as db:
        assert not list(
            db.scalars(select(RoutingObservation).where(RoutingObservation.record_id == task["id"]))
        )
    run(
        "record.update",
        {
            "record_id": task["id"],
            "expected_revision": task["revision"],
            "schema_revision": 1,
            "parent_id": None,
        },
    )
    with session_scope() as db:
        assert db.get(RoutingPattern, rule["id"]).status == "conflict"


def test_routing_cannot_change_due_dates_responsibility_or_foreign_homes():
    for values in ({"due_date": "2027-01-01"}, {"assignee": "agent"}):
        with pytest.raises(DomainError):
            run(
                "routing.create",
                {"phrase": "Andi", "type_id": "task", "values": values, "reason": "not authorized"},
            )
    foreign = run(
        "record.create", {"type_id": "client", "title": "Foreign", "schema_revision": 1}, owner="other"
    )
    with pytest.raises(DomainError):
        run(
            "routing.create",
            {
                "phrase": "ABC",
                "type_id": "task",
                "parent_id": foreign["id"],
                "reason": "Must not cross owners",
            },
        )
    with session_scope() as db:
        schema = structure.ensure(db, OWNER)
        with pytest.raises(DomainError):
            routing.validate_assignment(db, OWNER, schema, "task", {"permission": "owner"})


def test_100_phrase_boundary_cases():
    # This matcher regression suite is not the human-reviewed quality gate.
    for i in range(100):
        phrase = f"Client {i}"
        assert routing.matches(f"Update CLIENT {i}: central docs", phrase)
        assert not routing.matches(f"Update Client {i}x docs", phrase)
        assert not routing.matches(f"Update Client x{i} docs", phrase)


def test_conflicting_rules_leave_home_unassigned():
    a, b = create(title="A"), create(title="B")
    for p in (a, b):
        run(
            "routing.create", {"phrase": "ABC", "type_id": "task", "parent_id": p["id"], "reason": "explicit"}
        )
    task = create("task", "ABC docs")
    assert task["parent_id"] is None


def test_learning_toggle_separate_from_routing_and_personal_memory():
    p = create()
    run("routing.create", {"phrase": "ABC", "type_id": "task", "parent_id": p["id"], "reason": "explicit"})
    run("settings.update", {"routing_learning": False, "memory_learning": False})
    assert create("task", "ABC docs")["parent_id"] == p["id"]
    run("settings.update", {"routing_mode": "suggest"})
    task = create("task", "ABC other")
    assert task["parent_id"] is None
    assert task["provenance"]["routing"]["assignment"]["parent_id"] == p["id"]


def test_weekly_schedule_and_queue_are_idempotent():
    definition()
    with session_scope() as db:
        prefs = routing.preferences(db, OWNER)
        slot = routing.period(prefs, datetime(2026, 9, 14, 9, tzinfo=UTC))
        assert slot == "2026-09-14T08:00:00+00:00"
        first = routing.queue(db, OWNER)
        second = routing.queue(db, OWNER)
        assert first.id == second.id


def test_field_assessment_stale_fingerprint_cannot_overwrite(monkeypatch):
    d = definition()
    d["types"][0]["description"] = "My own context boundaries"
    apply(propose(d))
    with session_scope() as db:
        job = db.scalar(select(Job).where(Job.kind == "assess_field"))
        identity = job.id
        state = db.get(FieldUnderstanding, (OWNER, job.payload["definition_id"]))
        state.fingerprint = "new"
    monkeypatch.setattr(routing, "infer", lambda *a: pytest.fail("stale job called model"))
    routing.assess(identity)
    with session_scope() as db:
        assert db.get(Job, identity).status == "cancelled"


def test_existing_reorganization_needs_preview_and_preserves_newer_edits():
    p = create()
    t = create("task", "ABC docs")
    rule = run(
        "routing.create", {"phrase": "ABC", "type_id": "task", "parent_id": p["id"], "reason": "explicit"}
    )
    preview = run("routing.preview", {"pattern_id": rule["id"]})
    run(
        "record.update",
        {
            "record_id": t["id"],
            "expected_revision": t["revision"],
            "schema_revision": 1,
            "title": "ABC changed",
        },
    )
    with pytest.raises(DomainError):
        run("routing.apply", {"preview_id": preview["id"]})


def test_three_training_examples_do_not_bypass_heldout_gate(monkeypatch):
    parent = create()
    # Ensure enough independent training records regardless of stable split.
    for i in range(16):
        create("task", f"ABC docs {i}", parent_id=parent["id"])
    job = run("routing.run", {})
    monkeypatch.setattr(
        routing,
        "infer",
        lambda *a: routing.Candidates(
            candidates=[
                routing.Candidate(
                    phrase="ABC",
                    type_id="task",
                    parent_id=parent["id"],
                    values=[],
                    reason="confirmed examples",
                )
            ]
        ),
    )
    routing.process(job["job_id"])
    with session_scope() as db:
        rule = db.scalar(select(RoutingPattern))
        assert rule.status == "candidate"
        review = db.scalar(select(RoutingReview))
        question = review.questions[0]
    run(
        "routing.answer",
        {
            "review_id": review.id,
            "expected_revision": review.revision,
            "question_id": question["id"],
            "action": "accept",
        },
    )
    assert create("task", "ABC final")["parent_id"] == parent["id"]


def test_no_change_review_avoids_another_model_call(monkeypatch):
    parent = create()
    for i in range(12):
        create("task", f"ABC docs {i}", parent_id=parent["id"])
    monkeypatch.setattr(routing, "infer", lambda *a: routing.Candidates(candidates=[]))
    first = run("routing.run", {})
    routing.process(first["job_id"])
    monkeypatch.setattr(routing, "infer", lambda *a: pytest.fail("unchanged evidence called provider again"))
    second = run("routing.run", {})
    routing.process(second["job_id"])


def test_assessment_outage_is_bounded_and_leaves_manual_use(monkeypatch):
    d = definition()
    d["types"][0]["description"] = "Our departments and responsibilities"
    apply(propose(d))
    with session_scope() as db:
        identity = db.scalar(select(Job.id).where(Job.kind == "assess_field"))

    def fail(*args):
        raise RuntimeError("synthetic outage")

    monkeypatch.setattr(routing, "infer", fail)
    for _ in range(2):
        with pytest.raises(RuntimeError):
            routing.perform(identity, "assess_field")
    routing.perform(identity, "assess_field")
    with session_scope() as db:
        assert db.get(Job, identity).status == "failed"
        assert db.get(FieldUnderstanding, (OWNER, "type:space")).status == "needs_input"
    assert create("space", "Manual space")["title"] == "Manual space"


def test_backend_core_task_creation_uses_same_routing_rules():
    p = create()
    run("routing.create", {"phrase": "ABC", "type_id": "task", "parent_id": p["id"], "reason": "explicit"})
    task = run("task.create", {"title": "ABC CSR update"}, key="backend-request:1")
    with session_scope() as db:
        from jarvis.structure_models import StructureRecord

        record = db.get(StructureRecord, task["record_id"])
        assert record.parent_id == p["id"]
        assert not list(
            db.scalars(select(RoutingObservation).where(RoutingObservation.record_id == record.id))
        )


def test_weekly_learning_supports_classification_without_forcing_a_home(monkeypatch):
    import json

    d = definition()
    task = next(t for t in d["types"] if t["id"] == "task")
    task["fields"].append(
        {
            "id": "department",
            "name": "Department",
            "description": "The department responsible for this work.",
            "kind": "select",
            "options": [{"id": "operations", "name": "Operations"}],
        }
    )
    apply(propose(d))
    for i in range(16):
        create("task", f"ABC docs {i}", values={"department": "operations"})
    with session_scope() as db:
        db.get(FieldUnderstanding, (OWNER, "field:task:department")).status = "ready"
    monkeypatch.setattr(
        routing,
        "infer",
        lambda *a: routing.Candidates(
            candidates=[
                routing.Candidate(
                    phrase="ABC",
                    type_id="task",
                    parent_id=None,
                    values=[
                        routing.CandidateValue(field_id="department", value_json=json.dumps("operations"))
                    ],
                    reason="Human assignments",
                )
            ]
        ),
    )
    job = run("routing.run", {})
    routing.process(job["job_id"])
    with session_scope() as db:
        review = db.scalar(select(RoutingReview))
        rule = db.scalar(select(RoutingPattern))
        assert rule.assignment == {"values": {"department": "operations"}}
        assert "Department: Operations" in review.questions[0]["question"]
        assert rule.status == "candidate"
        review_id, revision, question = review.id, review.revision, review.questions[0]["id"]
    assert "department" not in create("task", "ABC before confirmation")["values"]
    run(
        "routing.answer",
        {"review_id": review_id, "expected_revision": revision, "question_id": question, "action": "accept"},
    )
    assert create("task", "ABC after confirmation")["values"]["department"] == "operations"


def test_only_rules_using_changed_definitions_need_review():
    parent = create()
    rule = run(
        "routing.create",
        {"phrase": "ABC", "type_id": "task", "parent_id": parent["id"], "reason": "explicit"},
    )
    d = definition()
    next(t for t in d["types"] if t["id"] == "note")["description"] = "My handwritten and typed notes"
    apply(propose(d))
    assert create("task", "ABC unchanged rule")["parent_id"] == parent["id"]
    d = definition()
    next(t for t in d["types"] if t["id"] == "client")["description"] = "A different meaning for client"
    apply(propose(d))
    with session_scope() as db:
        assert db.get(RoutingPattern, rule["id"]).status == "needs_review"
    assert create("task", "ABC changed definition")["parent_id"] is None


def test_field_clarification_gets_a_review_offer_and_answer_resolves_it(monkeypatch):
    d = definition()
    next(t for t in d["types"] if t["id"] == "client")["description"] = "The clients I work with"
    apply(propose(d))
    with session_scope() as db:
        identity = db.scalar(select(Job.id).where(Job.kind == "assess_field"))
    monkeypatch.setattr(
        routing,
        "infer",
        lambda *a: routing.Assessment(
            meaning="Client classification", question="Does a client mean a company or an individual?"
        ),
    )
    routing.assess(identity)
    with session_scope() as db:
        state = db.get(FieldUnderstanding, (OWNER, "type:client"))
        revision = state.revision
        review = db.scalar(select(RoutingReview).where(RoutingReview.status == "pending"))
        assert (
            review.summary["definition_id"] == "type:client" and routing.offer_context(db, OWNER) == review.id
        )
        routing.mark_offer(db, OWNER, review.id)
        assert not routing.offer_context(db, OWNER)
    run(
        "routing.understand",
        {"definition_id": "type:client", "expected_revision": revision, "answer": "A company"},
    )
    with session_scope() as db:
        assert not list(db.scalars(select(RoutingReview).where(RoutingReview.status == "pending")))
