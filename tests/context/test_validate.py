"""Deterministic extraction validators: exact evidence, dates, numbers, modality, attribution and dedupe."""

import json
from datetime import date
from decimal import Decimal
from pathlib import Path

import pytest
from jarvis.context.native.prompts import Enriched, Verdict
from jarvis.context.native.validate import (
    Draft,
    Rejection,
    SourceContext,
    apply_verdicts,
    check_item,
    dedupe,
    expected_date,
    locate,
    numbers,
)
from jarvis.context.schema import Decision, EntityCandidate, EntityRef, ExtractRequest, Segment

FIXTURE = json.loads((Path(__file__).parent / "fixtures/extraction/adversarial.json").read_text())
BASE = {"subject": None, "object": None, "key": None, "modality": "asserted", "inferred": False, "responsible": [],
        "responsibility": "none", "recipient": None, "due": None, "valid_from": None, "valid_to": None,
        "confidence": 0.9, "candidate": 0}


def everything(*_):
    return True


def request(**update) -> ExtractRequest:
    return ExtractRequest(owner_id="davin", source_revision_id="srcrev_adv", occurred_at=FIXTURE["occurred_at"],
                          timezone=FIXTURE["timezone"], segments=[Segment(**s) for s in FIXTURE["segments"]],
                          **update)


def item(**fields) -> Enriched:
    return Enriched.model_validate({**BASE, **fields})


@pytest.mark.parametrize("case", FIXTURE["cases"], ids=[c["name"] for c in FIXTURE["cases"]])
def test_adversarial_and_control_cases(case):
    outcome = check_item(item(**case["item"]), SourceContext.from_request(request()), everything)
    if case["expect"] == "accepted":
        assert isinstance(outcome, Draft), outcome
        assert outcome.assertion is not None
        for ref in outcome.evidence:
            assert ref.matches(FIXTURE["segments"][ref.segment_seq]["text"])
    else:
        assert isinstance(outcome, Rejection)
        assert outcome.code == case["expect"], outcome


def test_relative_date_keeps_expression_and_verified_derivation():
    case = next(c for c in FIXTURE["cases"] if c["name"] == "relative_date_grounded")
    draft = check_item(item(**case["item"]), SourceContext.from_request(request()), everything)
    due = draft.assertion.due
    assert (due.at.date(), due.precision, due.expression) == (date(2026, 11, 17), "day", "in two weeks")
    assert due.at.utcoffset().total_seconds() == -5 * 3600  # local midnight in New York
    assert draft.derivations == [{"field": "due", "expression": "in two weeks", "value": "2026-11-17",
                                  "precision": "day", "derivation": "from the meeting's local date 2026-11-03",
                                  "verified": True}]


def test_quotes_match_after_whitespace_and_curly_quote_normalization_only():
    text = "Sure — I'll send  the  budget\nby Friday."
    start, end = locate(text, "“I’ll send the budget by Friday.”")
    assert text[start:end] == "I'll send  the  budget\nby Friday."
    assert locate(text, "I will send the budget by Friday.") is None  # words are never rewritten
    assert locate("abc abc", "abc", lambda a, b: a > 0) == (4, 7)  # prefers the owned occurrence


def test_evidence_in_neighbour_context_is_rejected():
    owned = item(kind="fact", statement="The pilot covers 40 clinics in the first wave.",
                 evidence=[{"segment_seq": 1, "quote": "The pilot covers 40 clinics in the first wave."}])
    outcome = check_item(owned, SourceContext.from_request(request()), lambda seq, a, b: seq != 1)
    assert isinstance(outcome, Rejection) and outcome.code == "evidence_outside_chunk"


def test_validity_is_never_now_and_unknown_time_stays_unknown():
    ctx = SourceContext.from_request(request())
    fact = item(kind="fact", statement="The pilot covers 40 clinics in the first wave.",
                evidence=[{"segment_seq": 1, "quote": "The pilot covers 40 clinics in the first wave."}],
                valid_from={"expression": "now", "value": "2026-11-03", "precision": "day", "derivation": None})
    draft = check_item(fact, ctx, everything)
    assert draft.assertion.valid_from is None
    assert draft.derivations[0]["dropped"].startswith("validity is never now")
    vague = item(kind="commitment", statement="Priya sends the contract.", responsible=[
        {"entity_type": "actor", "mention": "Priya", "candidate_id": None}],
        evidence=[{"segment_seq": 5, "quote": "I'll send the contract in two weeks."}],
        due={"expression": "at some point", "value": "2026-12-01", "precision": "day", "derivation": "guess"})
    assert check_item(vague, ctx, everything).assertion.due is None  # expression not in the quote


def test_number_parsing_and_date_expressions():
    assert {Decimal(42), Decimal(40000), Decimal(1200), Decimal("1.5E6")} <= numbers(
        "forty-two people, $40k, 1,200 rows, 1.5 million users")
    assert numbers("The clinics are ready.") == set()
    tuesday = date(2026, 11, 3)
    assert expected_date("tomorrow", tuesday) == (date(2026, 11, 4), True)
    assert expected_date("in two weeks", tuesday) == (date(2026, 11, 17), True)
    assert expected_date("in a month", tuesday) == (date(2026, 12, 3), True)
    assert expected_date("by Friday", tuesday) == (date(2026, 11, 6), True)
    assert expected_date("on Tuesday", tuesday) == (date(2026, 11, 10), True)
    assert expected_date("November 20th", tuesday) == (date(2026, 11, 20), False)
    assert expected_date("next Friday", tuesday) == (None, False)  # ambiguous: the model's value stands
    assert expected_date("end of the quarter", tuesday) == (None, False)


def test_candidate_ids_resolve_only_when_listed_and_of_the_same_type():
    ctx = SourceContext.from_request(request(candidates=[
        EntityCandidate(ref=EntityRef(entity_type="actor", entity_id="act_priya"), name="Priya Shah")]))
    draft = check_item(item(kind="commitment", statement="Priya sends the contract by 2026-11-17.",
                            responsible=[{"entity_type": "actor", "mention": "Priya", "candidate_id": "actor:act_priya"}],
                            subject={"entity_type": "project", "mention": "contract", "candidate_id": "actor:act_priya"},
                            due={"expression": "in two weeks", "value": "2026-11-17", "precision": "day",
                                 "derivation": "+14 days"},
                            evidence=[{"segment_seq": 5, "quote": "I'll send the contract in two weeks."}]),
                       ctx, everything)
    assert draft.assertion.responsible == [EntityRef(entity_type="actor", entity_id="act_priya", mention="Priya")]
    assert draft.assertion.subject == EntityRef(entity_type="project", mention="contract")  # type mismatch
    assert [(link.mention, link.candidate.entity_id, link.status) for link in draft.links] == [
        ("Priya", "act_priya", "inferred")]


def test_prior_wrong_decisions_are_not_proposed_again():
    wrong = Decision(id="dec_1", seq=4, operation="wrong", origin="human", target_type="assertion",
                     target_id="asr_9", statement="The pilot covers 40 clinics in the first wave.",
                     recorded_at=FIXTURE["occurred_at"])
    outcome = check_item(item(kind="fact", statement="The pilot covers 40 clinics in the first wave.",
                              evidence=[{"segment_seq": 1, "quote": "The pilot covers 40 clinics"}]),
                         SourceContext.from_request(request(decisions=[wrong])), everything)
    assert isinstance(outcome, Rejection) and outcome.code == "prior_decision"


def _draft(kind, statement, seq, quote, **fields):
    outcome = check_item(item(kind=kind, statement=statement, evidence=[{"segment_seq": seq, "quote": quote}],
                              **fields), SourceContext.from_request(request()), everything)
    assert isinstance(outcome, Draft), outcome
    return outcome


def test_verdicts_reject_unentailed_and_unverified_and_downgrade_modality():
    fact = _draft("fact", "The pilot covers 40 clinics.", 1, "The pilot covers 40 clinics")
    other = _draft("fact", "The pilot has a first wave.", 1, "in the first wave")
    promise = _draft("commitment", "The team circulates the notes after the call.", 8,
                     "We'll circulate the notes after the call.",
                     responsible=[{"entity_type": "actor", "mention": "we", "candidate_id": None}])
    missing = _draft("decision", "The team ships the mobile app this quarter.", 3,
                     "We won't ship the mobile app this quarter.", modality="negated")
    kept, rejected = apply_verdicts([fact, other, promise, missing], [
        Verdict(item=0, entailed=True, modality="proposed", attribution_ok=True, reason="only floated"),
        Verdict(item=1, entailed=False, modality="asserted", attribution_ok=True, reason="unsupported"),
        Verdict(item=2, entailed=True, modality="question", attribution_ok=True, reason="asked")])
    assert [d.statement for d in kept] == ["The pilot covers 40 clinics."]
    assert kept[0].assertion.modality == "proposed" and kept[0].notes
    assert [r.code for r in rejected] == ["not_entailed", "commitment_not_asserted", "unverified"]


def test_dedupe_by_statement_and_overlapping_evidence():
    a = _draft("fact", "The pilot has a first wave.", 1, "The pilot covers 40 clinics in the first wave.")
    b = _draft("fact", "the pilot has a first wave", 1, "in the first wave")
    c = _draft("fact", "The pilot has a first wave.", 0, "Sounds right.")
    kept, rejected = dedupe([a, b, c])
    assert kept == [a, c] and [r.code for r in rejected] == ["duplicate"]
