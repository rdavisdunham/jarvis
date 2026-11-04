"""Contract tests for jarvis.context.schema, engine and doubles.

The JSON Schema snapshot (snapshots/contracts.json) fails on any change to the published contracts or
enumerations. After an intended change, regenerate it and commit the diff:

    CONTEXT_UPDATE_SNAPSHOTS=1 uv run --no-sync python -m pytest -q tests/context/test_contracts.py
"""

import json
import os
from datetime import datetime
from pathlib import Path

import pytest
from jarvis.context import schema
from jarvis.context.doubles import (
    DoubleExtractor,
    DoubleReconciler,
    DoubleRetriever,
    DoubleSynthesizer,
    Indexed,
    canned_file,
)
from jarvis.context.engine import Extractor, Reconciler, Retriever, Synthesizer
from jarvis.context.schema import (
    BRIEF_SECTIONS,
    BriefRequest,
    CandidateAssertion,
    EntityRef,
    EvidenceRef,
    ExtractionResult,
    Hit,
    LinkProposal,
    ReconcileOp,
    ReconcileRequest,
    RetrieveQuery,
    RunMeta,
    ScopePredicate,
    SourceEnvelope,
    SynthesizeRequest,
    TaskView,
    check_extraction,
)
from pydantic import ValidationError

from .conftest import MEETING_AT, at

SNAPSHOT = Path(__file__).parent / "snapshots" / "contracts.json"
ENUMS = {name: list(value) for name, value in vars(schema).items()
         if name.isupper() and isinstance(value, tuple) and all(isinstance(v, str) for v in value)}


def roundtrip(value):
    assert type(value).model_validate_json(value.model_dump_json()) == value


def test_published_contracts_match_the_snapshot():
    current = {"enums": ENUMS, "schemas": schema.json_schemas()}
    if os.environ.get("CONTEXT_UPDATE_SNAPSHOTS"):
        SNAPSHOT.write_text(json.dumps(current, indent=2, sort_keys=True) + "\n")
    assert json.loads(SNAPSHOT.read_text()) == current, "contracts changed; see this module's docstring"


def test_enumerations_shared_with_the_tables():
    assert schema.ASSERTION_KINDS == ("fact", "preference", "decision", "rationale", "risk", "open_question",
                                      "procedure", "relationship", "commitment")
    assert schema.SOURCE_STATUSES[-2:] == ("awaiting_input", "deleted")
    assert "undo" in schema.DECISION_OPERATIONS and "stale" in schema.RUN_STATUSES


def test_models_roundtrip_through_json(envelope, extract_request, history, owner_id):
    result = DoubleExtractor().extract(extract_request)
    scope = ScopePredicate(owner_id=owner_id, entities=[EntityRef(entity_type="project", entity_id="p1")],
                           valid_at=MEETING_AT, kinds=["decision"])
    brief = BriefRequest(scope=scope, entity=EntityRef(entity_type="project", entity_id="p1"))
    values = [envelope, extract_request, result, *history, scope, brief,
              RetrieveQuery(text="budget", scope=scope),
              Hit(ref_type="assertion", id="asr_1", revision=1, score=0.5, kind="fact", valid_from=MEETING_AT),
              ReconcileOp(op="supersede", targets=["asr_1", "asr_2"], target_revisions={"asr_1": 1, "asr_2": 1}),
              RunMeta(stage="extract", engine="double", config_version="1", input_revisions={"source_revision": "x"}),
              *DoubleSynthesizer().synthesize(SynthesizeRequest(brief=brief, assertions=history))]
    for value in values:
        roundtrip(value)
    assert set(result.coverage) == {0, 1, 2, 3, 4}


def quote(text="We ship.", seq=0):
    return EvidenceRef(segment_seq=seq, quote=text, char_start=0, char_end=len(text))


@pytest.mark.parametrize("build", [
    lambda: CandidateAssertion(kind="fact", statement="No evidence.", evidence=[]),
    lambda: CandidateAssertion(kind="fact", statement="Only contradiction.",
                               evidence=[quote().model_copy(update={"role": "contradiction"})]),
    lambda: CandidateAssertion(kind="commitment", statement="Someone ships.", evidence=[quote()]),
    lambda: CandidateAssertion(kind="fact", statement="Facts have no due date.", evidence=[quote()],
                               due={"at": MEETING_AT}),
    lambda: CandidateAssertion(kind="fact", statement="Backwards.", evidence=[quote()],
                               valid_from={"at": at(5)}, valid_to={"at": at(4)}),
    lambda: CandidateAssertion(kind="fact", statement="Naive time.", evidence=[quote()],
                               valid_from={"at": datetime(2026, 11, 3)}),
    lambda: EvidenceRef(segment_seq=0, quote="abc", char_start=0, char_end=5),
    lambda: EntityRef(entity_type="actor"),
    lambda: LinkProposal(mention="Sam", candidate=EntityRef(entity_type="actor", entity_id="a1"), status="confirmed"),
    lambda: LinkProposal(mention="Sam", candidate=EntityRef(entity_type="actor", mention="Sam"), status="inferred"),
    lambda: ReconcileOp(op="supersede", targets=["asr_1"]),
    lambda: ScopePredicate(owner_id="davin", entities=[EntityRef(entity_type="actor", mention="Sam")]),
])
def test_validators_reject_malformed_contracts(build):
    with pytest.raises(ValidationError):
        build()


@pytest.mark.parametrize("change", [
    {"text": "also text"}, {"external_revision": None}, {"timezone": "Mars/Base"},
    {"ended_at": at(1)}, {"connector": "email"},
])
def test_envelope_rejects_bad_shapes(envelope, change):
    with pytest.raises(ValidationError):
        SourceEnvelope.model_validate({**envelope.model_dump(), **change})


def test_envelope_takes_text_and_keeps_private_notes_out_of_segments(envelope):
    plain = SourceEnvelope.model_validate({**envelope.model_dump(), "segments": None, "text": "Hello."})
    assert plain.text == "Hello."
    segments = [s.model_dump() for s in envelope.segments]
    segments[0]["kind"] = "private_note"
    with pytest.raises(ValidationError):
        SourceEnvelope.model_validate({**envelope.model_dump(), "segments": segments})


def test_doubles_satisfy_the_protocols():
    assert isinstance(DoubleExtractor(), Extractor)
    assert isinstance(DoubleReconciler(), Reconciler)
    assert isinstance(DoubleRetriever(), Retriever)
    assert isinstance(DoubleSynthesizer(), Synthesizer)
    assert not isinstance(object(), Extractor)


def test_rule_based_extraction_quotes_exactly(extract_request):
    result = DoubleExtractor().extract(extract_request)
    assert check_extraction(extract_request, result) == []
    assert [(a.kind, a.statement) for a in result.assertions] == [
        ("decision", "we launch the pilot on Postgres with pgvector."),
        ("commitment", "I'll send the revised budget by 2026-11-20."),
        ("commitment", "Marco to draft the rollout checklist."),
    ]
    budget, checklist = result.assertions[1:]
    assert [r.mention for r in budget.responsible + checklist.responsible] == ["Priya", "Marco"]
    assert budget.due.expression == "2026-11-20" and budget.due.precision == "day"
    assert budget.due.at.date().isoformat() == "2026-11-20"
    assert budget.valid_from is None and budget.effective_from(MEETING_AT).at == MEETING_AT
    assert [a.assertion_indexes for a in result.actions] == [[1], [2]]
    assert result.coverage == {0: "no_content", 1: "no_content", 2: "extracted", 3: "extracted", 4: "extracted"}


def test_check_extraction_flags_inexact_quotes_and_missing_coverage(extract_request):
    bad = ExtractionResult(assertions=[CandidateAssertion(kind="fact", statement="x", evidence=[quote("Nope.", 1)])],
                           coverage={0: "extracted", 9: "extracted"})
    problems = check_extraction(extract_request, bad)
    assert "segment 1 has no coverage disposition" in problems
    assert "coverage names unknown segment 9" in problems
    assert any(p.startswith("evidence quote is not exact in segment 1") for p in problems)


def test_canned_extraction_is_keyed_by_content_hash(extract_request, tmp_path):
    canned = ExtractionResult(diagnostics={"mode": "canned"}, coverage={s.seq: "skipped" for s in extract_request.segments})
    assert DoubleExtractor({extract_request.content_hash: canned}).extract(extract_request) == canned
    canned_file(tmp_path, extract_request, canned)
    assert DoubleExtractor.from_directory(tmp_path).extract(extract_request) == canned
    assert extract_request.content_hash == schema.content_hash(s.text for s in extract_request.segments)


def test_doubles_are_deterministic(extract_request, history, owner_id):
    extract = [DoubleExtractor().extract(extract_request) for _ in range(2)]
    reconcile_request = ReconcileRequest(owner_id=owner_id, changed=history[1:2], related=history)
    reconcile = [DoubleReconciler().reconcile(reconcile_request) for _ in range(2)]
    query = RetrieveQuery(text="pilot budget", scope=ScopePredicate(owner_id=owner_id))
    retrieve = [DoubleRetriever(corpus(history, owner_id)).retrieve(query) for _ in range(2)]
    brief = BriefRequest(scope=ScopePredicate(owner_id=owner_id), query="pilot")
    synth = [DoubleSynthesizer().synthesize(SynthesizeRequest(brief=brief, assertions=history)) for _ in range(2)]
    for first, second in (extract, reconcile, retrieve, synth):
        assert first == second


def test_reconciler_supersedes_keyed_values_in_effective_order(history, owner_id):
    ops = DoubleReconciler().reconcile(ReconcileRequest(owner_id=owner_id, changed=history[1:], related=history[:1]))
    assert [(op.op, op.targets) for op in ops] == [
        ("supersede", ["asr_1", "asr_2"]), ("no_change", ["asr_3"]), ("no_change", ["asr_4"])]
    assert ops[0].target_revisions == {"asr_1": 1, "asr_2": 1}

    same_time = history[1].model_copy(update={"id": "asr_5", "statement": "Budget by 2026-11-27."})
    duplicate = history[2].model_copy(update={"id": "asr_6"})
    ops = DoubleReconciler().reconcile(ReconcileRequest(owner_id=owner_id, changed=[same_time, duplicate],
                                                        related=history))
    assert ("conflict", ["asr_2", "asr_5"]) in [(op.op, op.targets) for op in ops]
    assert ("equivalent", ["asr_3", "asr_6"]) in [(op.op, op.targets) for op in ops]


def corpus(history, owner_id):
    items = [Indexed(owner_id=owner_id, text=a.statement, recorded_from=a.recorded_at,
                     hit=Hit(ref_type="assertion", id=a.id, revision=a.revision, score=0, kind=a.kind,
                             lifecycle="superseded" if a.id == "asr_1" else "current", valid_from=a.valid_from.at,
                             valid_to=MEETING_AT if a.id == "asr_1" else None, entities=[a.subject]))
             for a in history]
    items.append(Indexed(owner_id="someone-else", text="pilot budget", hit=Hit(ref_type="assertion", id="other",
                                                                                   revision=1, score=0)))
    items.append(Indexed(owner_id=owner_id, text="We talked about the pilot budget at length.",
                         hit=Hit(ref_type="segment", id="seg_9", revision=1, score=0)))
    return items


def test_retriever_honours_the_scope_predicate(history, owner_id):
    retriever = DoubleRetriever(corpus(history, owner_id))

    def ids(**scope):
        include_raw = scope.pop("include_raw", False)
        query = RetrieveQuery(text="pilot budget", scope=ScopePredicate(owner_id=owner_id, **scope),
                              include_raw=include_raw)
        return {hit.id for hit in retriever.retrieve(query)}

    assert ids() == {"asr_2", "asr_3", "asr_4"}
    assert ids(include_superseded=True) == {"asr_1", "asr_2", "asr_3", "asr_4"}
    assert ids(include_superseded=True, valid_at=at(1)) == {"asr_1"}
    assert ids(kinds=["fact"]) == {"asr_3"}
    assert ids(known_at=at(1)) == set()
    assert ids(entities=[EntityRef(entity_type="project", entity_id="other")]) == set()
    assert ids(include_raw=True) == {"asr_2", "asr_3", "asr_4", "seg_9"}
    hits = retriever.retrieve(RetrieveQuery(text="pilot budget", scope=ScopePredicate(owner_id=owner_id), limit=1))
    assert hits[0].rank == {"lexical_rank": 1.0} and hits[0].snippet


def test_synthesizer_fills_the_six_sections_with_references(history, owner_id):
    brief = BriefRequest(scope=ScopePredicate(owner_id=owner_id),
                         entity=EntityRef(entity_type="project", entity_id="proj_pilot"))
    retired = history[0].model_copy(update={"lifecycle": "superseded"})
    task = TaskView(id="task_1", revision=2, title="Send budget", status="open", assertion_ids=["asr_2"])
    sections = DoubleSynthesizer().synthesize(SynthesizeRequest(brief=brief, assertions=history[1:],
                                                                history=[retired], tasks=[task]))
    by_name = {s.name: s for s in sections}
    assert tuple(by_name) == BRIEF_SECTIONS
    assert [c.assertion_ids for c in by_name["current_state_and_why"].claims] == [["asr_3"], ["asr_4"]]
    assert by_name["open_loops"].claims[0].task_ids == ["task_1"]
    assert by_name["recent_changes"].claims[0].assertion_ids == ["asr_1"]
    assert by_name["who_is_who"].gaps and by_name["unknowns"].gaps
