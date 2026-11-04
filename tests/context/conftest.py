"""Shared context fixtures: an owner, a short three-speaker meeting envelope and a small assertion history.

Fixture text is data. The meeting holds one decision, two commitments (one with an ISO due date), a number
and a date, which the rule-based extractor double recognises.
"""

from datetime import UTC, datetime

import pytest
from jarvis.context.schema import (
    AssertionView,
    EntityRef,
    EvidenceRef,
    ExtractRequest,
    Participant,
    Segment,
    SourceEnvelope,
    TimeBound,
)

MEETING_AT = datetime(2026, 11, 3, 15, 0, tzinfo=UTC)
LINES = [
    ("Davin", "owner", "Thanks for joining. We need to settle the pilot scope today."),
    ("Priya", "other", "The pilot covers 40 clinics in the first wave."),
    ("Marco", "other", "Decision: we launch the pilot on Postgres with pgvector."),
    ("Priya", "other", "Sounds right. I'll send the revised budget by 2026-11-20."),
    ("Davin", "owner", "Action: Marco to draft the rollout checklist."),
]


def at(day: int, month: int = 11) -> datetime:
    return datetime(2026, month, day, tzinfo=UTC)


@pytest.fixture
def owner_id():
    return "davin"


@pytest.fixture
def envelope():
    return SourceEnvelope(
        connector="api", external_id="meeting-001", title="Pilot scope", occurred_at=MEETING_AT,
        ended_at=datetime(2026, 11, 3, 15, 30, tzinfo=UTC), timezone="America/New_York",
        participants=[Participant(name="Davin", email="davin@example.com", role="organizer", speaker_label="Davin"),
                      Participant(name="Priya", email="priya@example.com", speaker_label="Priya"),
                      Participant(name="Marco", speaker_label="Marco")],
        segments=[Segment(seq=i, speaker_label=who, attribution=attribution, text=text,
                          start_ms=i * 15_000, end_ms=i * 15_000 + 12_000)
                  for i, (who, attribution, text) in enumerate(LINES)],
        private_notes="Check the clinic count against the contract.",
        external_revision="rev-1", upstream_owner_email="davin@example.com",
        upstream_owner_name="Davin")


@pytest.fixture
def extract_request(owner_id, envelope):
    return ExtractRequest(owner_id=owner_id, source_revision_id="srcrev_1", occurred_at=envelope.occurred_at,
                          timezone=envelope.timezone, title=envelope.title, participants=envelope.participants,
                          segments=envelope.segments)


def evidence(seq: int, quote: str) -> list[EvidenceRef]:
    return [EvidenceRef(segment_seq=seq, quote=quote, char_start=0, char_end=len(quote),
                        source_revision_id="srcrev_1")]


@pytest.fixture
def history():
    """Assertions about the pilot project: the budget due date moved from 11-13 to 11-20."""
    pilot = EntityRef(entity_type="project", entity_id="proj_pilot")
    priya = [EntityRef(entity_type="actor", mention="Priya")]
    return [
        AssertionView(id="asr_1", revision=1, kind="commitment", statement="Priya sends the budget by 2026-11-13.",
                      subject=pilot, key="due_date", responsible=priya,
                      due=TimeBound(at=at(13), precision="day", expression="2026-11-13"),
                      valid_from=TimeBound(at=at(27, 10), precision="instant"), recorded_at=at(27, 10),
                      evidence=evidence(0, "I'll send the budget by 2026-11-13.")),
        AssertionView(id="asr_2", revision=1, kind="commitment", statement="Priya sends the budget by 2026-11-20.",
                      subject=pilot, key="due_date", responsible=priya,
                      due=TimeBound(at=at(20), precision="day", expression="2026-11-20"),
                      valid_from=TimeBound(at=MEETING_AT), recorded_at=MEETING_AT,
                      evidence=evidence(3, "I'll send the revised budget by 2026-11-20.")),
        AssertionView(id="asr_3", revision=1, kind="fact", statement="The pilot covers 40 clinics in the first wave.",
                      subject=pilot, valid_from=TimeBound(at=MEETING_AT), recorded_at=MEETING_AT,
                      evidence=evidence(1, "The pilot covers 40 clinics in the first wave.")),
        AssertionView(id="asr_4", revision=1, kind="decision", statement="Launch the pilot on Postgres with pgvector.",
                      subject=pilot, valid_from=TimeBound(at=MEETING_AT), recorded_at=MEETING_AT,
                      evidence=evidence(2, "Decision: we launch the pilot on Postgres with pgvector.")),
    ]
