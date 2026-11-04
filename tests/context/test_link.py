"""Native entity linking: exact identifiers, scored bands, durable rejections and speaker buckets."""

import math

import pytest
from jarvis.context import identity
from jarvis.context.identity import DecisionInput
from jarvis.context.native import link
from jarvis.context.native.link import LinkContext, Mention
from jarvis.context.schema import EntityRef, EvidenceRef, LinkProposal, Participant, Segment
from jarvis.db import session_scope
from jarvis.domain import execute
from jarvis.memory_learning import EMBEDDING_MODEL
from jarvis.models import LinearConnection, LinearIssue
from jarvis.search_models import SearchDocument
from jarvis.structure_models import StructureRecord

OWNER = "davin"
HUMAN = DecisionInput(origin="human", principal="davin")


def unit(angle, axis=0):
    """A 512-dim unit vector `angle` radians away from the given axis."""
    v = [0.0] * 512
    v[axis], v[axis + 1] = math.cos(angle), math.sin(angle)
    return v


def record(db, type_id, title, vector=None):
    row = StructureRecord(owner_id=OWNER, type_id=type_id, title=title)
    db.add(row)
    db.flush()
    if vector is not None:
        db.add(SearchDocument(owner_id=OWNER, target_key="record:" + row.id, fingerprint="f", content=title,
                              embedding_model=EMBEDDING_MODEL, vectors=[vector]))
        db.flush()
    return row.id


def actor(db, name, email=None, duplicate=False):
    if email:
        return identity.seed_candidates(db, OWNER, "seed", [Participant(name=name, email=email)], [])[0].actor_id
    return execute(db, OWNER, f"test-actor-{name}-{duplicate}", "actor.create",
                   {"name": name, "allow_duplicate": duplicate})["data"]["id"]


def one(db, mention, context=None):
    [proposal] = link.link_mentions(db, OWNER, [mention], context)
    return proposal


@pytest.mark.parametrize(("scored", "rejected", "expected"), [
    ({"a": 1.0}, frozenset(), ("a", 1.0, 1.0)),
    ({"a": 1.0, "b": 0.9}, frozenset(), None),  # inside the margin
    ({"a": 0.84}, frozenset(), None),  # below the automatic threshold
    ({"a": 1.0, "b": 0.8}, frozenset(), ("a", 1.0, 0.2)),
    ({"a": 1.0}, frozenset({"a"}), None),  # a rejected leader blocks the runner-up too
    ({}, frozenset(), None),
])
def test_bands(scored, rejected, expected):
    assert link.choose(scored, rejected) == expected


def test_lexical_scores():
    assert link.lexical("The  pilot", ["the pilot"]) == 1.0
    assert link.lexical("Acme", ["acme corp"]) == 0.8
    assert link.lexical("Acme rollout", ["acme corp"]) == pytest.approx(0.7 / 3)
    assert link.lexical("", ["acme"]) == 0.0


def test_exact_identifiers_come_first():
    with session_scope() as db:
        priya = actor(db, "Priya", "priya@acme.test")
        task = execute(db, OWNER, "test-task", "task.create", {"title": "Rollout"})["data"]["id"]
        db.add(LinearConnection(owner_id=OWNER, workspace_id="ws-1", workspace_name="Acme", viewer_id="v"))
        issue = LinearIssue(owner_id=OWNER, workspace_id="ws-1", remote_id="r1", task_id=task,
                            snapshot={"identifier": "ENG-42"})
        stale = LinearIssue(owner_id=OWNER, workspace_id="ws-old", remote_id="r2",
                            task_id=execute(db, OWNER, "test-task-2", "task.create", {"title": "Old"})["data"]["id"],
                            snapshot={"identifier": "OLD-1"})
        db.add_all([issue, stale])
        project = record(db, "project", "Pilot")
        db.flush()
        by_email, by_key, other_workspace, by_id = link.link_mentions(db, OWNER, [
            "PRIYA@acme.test", "see ENG-42", "OLD-1", Mention(project, entity_types=("project",))])
        assert (by_email.status, by_email.candidate.entity_id) == ("exact", priya)
        assert (by_key.status, by_key.candidate) == ("exact", EntityRef(entity_type="linear_issue", entity_id=issue.id))
        assert other_workspace.status == "unresolved"
        assert (by_id.status, by_id.candidate.entity_type, by_id.candidate.entity_id) == ("exact", "project", project)


def test_scored_bands_for_records():
    quote = "the Pilot rollout"
    evidence = (EvidenceRef(segment_seq=2, quote=quote, char_start=0, char_end=len(quote)),)
    with session_scope() as db:
        pilot = record(db, "project", "Pilot rollout")
        record(db, "client", "Acme Health")
        record(db, "client", "Acme Logistics")
        clear = one(db, Mention("Pilot rollout", evidence=evidence))
        assert (clear.status, clear.candidate.entity_id, clear.score, clear.margin) == ("inferred", pilot, 1.0, 1.0)
        assert clear.evidence == list(evidence)
        assert isinstance(clear, LinkProposal)
        unclear = one(db, Mention("Acme", consequential=True))
        assert unclear.status == "unresolved" and unclear.candidate.entity_id is None
        assert unclear.candidate.entity_type == "client"
        assert sorted(c.name for c in unclear.candidates) == ["Acme Health", "Acme Logistics"]
        assert unclear.ask
        assert not one(db, Mention("Acme")).ask  # only consequential ambiguity becomes a Question
        nothing = one(db, Mention("Budget review", consequential=True))
        assert (nothing.status, nothing.candidates, nothing.ask) == ("unresolved", [], False)
        # Type restrictions apply: a client-only mention never resolves to the project.
        assert one(db, Mention("Pilot rollout", entity_types=("client",))).status == "unresolved"


def test_semantic_scores_use_search_documents_and_survive_embedding_failure():
    with session_scope() as db:
        clinic = record(db, "project", "Clinic onboarding", unit(0.0))
        record(db, "project", "Warehouse move", unit(1.4))

        def embed(texts):
            return [unit(0.1) for _ in texts]

        def broken(texts):
            raise TimeoutError

        semantic = one(db, "the new clinics", LinkContext(embed=embed))
        assert (semantic.status, semantic.candidate.entity_id) == ("inferred", clinic)
        assert semantic.score == pytest.approx(math.cos(0.1), abs=1e-4)
        assert one(db, "the new clinics").status == "unresolved"
        assert one(db, "the new clinics", LinkContext(embed=broken)).status == "unresolved"


def test_rejected_links_are_never_re_proposed():
    with session_scope() as db:
        pilot = record(db, "project", "Pilot")
        record(db, "project", "Pilot extension")
        assert one(db, "Pilot").candidate.entity_id == pilot
        identity.reject_link(db, OWNER, "pilot", EntityRef(entity_type="project", entity_id=pilot), HUMAN,
                             scope="meeting-1")
        blocked = one(db, "Pilot", LinkContext(scope="meeting-1"))
        # The rejected leader blocks the runner-up from being inferred, and is not offered again.
        assert blocked.status == "unresolved"
        assert pilot not in {c.ref.entity_id for c in blocked.candidates}
        assert one(db, "Pilot", LinkContext(scope="meeting-2")).candidate.entity_id == pilot
        identity.reject_link(db, OWNER, "Pilot", EntityRef(entity_type="project", entity_id=pilot), HUMAN)
        assert one(db, "Pilot", LinkContext(scope="meeting-2")).status == "unresolved"


def test_attendees_disambiguate_namesakes():
    with session_scope() as db:
        first = actor(db, "Alex", "alex@acme.test")
        second = actor(db, "Alex", "alex@beta.test")
        assert one(db, Mention("Alex", entity_types=("actor",))).status == "unresolved"
        attendees = identity.seed_candidates(db, OWNER, "meeting-1", [], [{"email": "alex@beta.test"}])
        inferred = one(db, Mention("Alex", entity_types=("actor",)), LinkContext(attendees=attendees))
        assert (inferred.status, inferred.candidate.entity_id) == ("inferred", second)
        assert {c.ref.entity_id for c in inferred.candidates} == {second}
        assert first != second


# ---------------------------------------------------------------- speaker buckets


def segments(*lines):
    return [Segment(seq=i, speaker_label=label, attribution=attribution, text=text)
            for i, (label, attribution, text) in enumerate(lines)]


def by_label(proposals):
    return {p.mention: (p.status, p.candidate.entity_id) for p in proposals}


def test_speakers_resolve_from_owner_lines_attendees_and_introductions():
    with session_scope() as db:
        me = identity.owner_actor(db, OWNER)
        attendees = identity.seed_candidates(db, OWNER, "meeting-1", [
            Participant(name="Priya Shah", email="priya@acme.test"),
            Participant(name="Marco", email="marco@acme.test", speaker_label="Speaker 2")], [])
        priya, marco = (a.actor_id for a in attendees)
        lines = segments(("Speaker 1", "other", "Hi all, I'm Priya and I run the clinics."),
                         ("Speaker 2", "other", "Thanks."),
                         ("Davin", "owner", "Let's start."),
                         (None, "owner", "Note to self."),
                         ("Speaker 3", "other", "I am sure that works."),
                         ("Priya", "other", "Right."))
        result = link.resolve_speakers(db, OWNER, "meeting-1", lines, attendees)
        assert by_label(result) == {
            "Speaker 1": ("inferred", priya), "Speaker 2": ("exact", marco), "Davin": ("exact", me),
            "me": ("exact", me), "Speaker 3": ("unresolved", None), "Priya": ("inferred", priya)}
        intro = next(p for p in result if p.mention == "Speaker 1").evidence[0]
        assert intro.matches(lines[0].text) and intro.quote == "I'm Priya"


def test_the_same_label_in_two_meetings_stays_separate():
    with session_scope() as db:
        a = identity.seed_candidates(db, OWNER, "meeting-a", [
            Participant(name="Alex", email="alex@acme.test", speaker_label="Speaker 1")], [])
        b = identity.seed_candidates(db, OWNER, "meeting-b", [
            Participant(name="Bea", email="bea@acme.test", speaker_label="Speaker 1")], [])
        lines = segments(("Speaker 1", "other", "Hello."))
        assert by_label(link.resolve_speakers(db, OWNER, "meeting-a", lines, a)) == {
            "Speaker 1": ("exact", a[0].actor_id)}
        assert by_label(link.resolve_speakers(db, OWNER, "meeting-b", lines, b)) == {
            "Speaker 1": ("exact", b[0].actor_id)}
        # A third meeting with no attendee data leaves the label unknown rather than borrowing either.
        assert by_label(link.resolve_speakers(db, OWNER, "meeting-c", lines)) == {"Speaker 1": ("unresolved", None)}


def test_rejected_speaker_links_stay_rejected():
    with session_scope() as db:
        attendees = identity.seed_candidates(db, OWNER, "meeting-1", [
            Participant(name="Marco", email="marco@acme.test", speaker_label="Speaker 2")], [])
        marco = attendees[0].actor_id
        identity.reject_link(db, OWNER, "Speaker 2", EntityRef(entity_type="actor", entity_id=marco), HUMAN,
                             scope="meeting-1")
        lines = segments(("Speaker 2", "other", "This is Marco."))
        assert by_label(link.resolve_speakers(db, OWNER, "meeting-1", lines, attendees)) == {
            "Speaker 2": ("unresolved", None)}
