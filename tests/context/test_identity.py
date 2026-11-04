"""Actor identities: seeding from attendees, exact resolution, durable negatives, merge and unmerge."""

import pytest
from jarvis.context import identity
from jarvis.context import models as m
from jarvis.context.identity import DecisionInput, IdentityCandidate
from jarvis.context.schema import EntityRef, Participant
from jarvis.db import session_scope
from jarvis.domain import DomainError, execute
from jarvis.models import Actor, Command, GoogleIdentity, Task, now
from sqlalchemy import func, select

OWNER = "davin"
HUMAN = DecisionInput(origin="human", principal="davin", reason="reviewed")
AGENT = DecisionInput(origin="agent", principal="bot-1")


def command(db, tool, **arguments):
    command.count = getattr(command, "count", 0) + 1
    return execute(db, OWNER, f"test-{tool}-{command.count}", tool, arguments)["data"]


def actors(db, name=None):
    q = select(Actor).where(Actor.owner_id == OWNER)
    return list(db.scalars(q.where(Actor.name == name) if name else q))


def seed(db, source="src-1", participants=(), attendees=()):
    return identity.seed_candidates(db, OWNER, source, list(participants), list(attendees))


def test_two_people_with_one_name_stay_distinct_by_email():
    with session_scope() as db:
        first = seed(db, attendees=[{"email": "alex@acme.test", "displayName": "Alex"}])
        second = seed(db, "src-2", participants=[Participant(name="Alex", email="Alex@Beta.test")])
        assert first[0].status == second[0].status == "exact"
        assert first[0].created and second[0].created
        assert first[0].actor_id != second[0].actor_id
        assert second[0].duplicates == (first[0].actor_id,)
        assert len(actors(db, "Alex")) == 2
        # Emails are normalized; a repeat seeding returns the same actors and creates nothing.
        again = seed(db, "src-3", attendees=[{"email": "ALEX@acme.test", "displayName": "Alex"},
                                             {"email": "alex@beta.test", "displayName": "Alex"}])
        assert [c.actor_id for c in again] == [first[0].actor_id, second[0].actor_id]
        assert not any(c.created for c in again)
        assert len(actors(db, "Alex")) == 2
        assert identity.resolve_exact(db, OWNER, "email", " alex@BETA.test ") == second[0].actor_id
        # Actors were created by the actor.create command, with allow_duplicate only for the namesake.
        made = db.scalars(select(Command).where(Command.owner_id == OWNER, Command.id.like("context-identity:%")))
        assert sorted(c.result["data"]["name"] for c in made) == ["Alex", "Alex"]


def test_retried_seeding_reuses_the_same_create_command():
    with session_scope() as db:
        made = seed(db, attendees=[{"email": "sam@acme.test", "displayName": "Sam"}])[0]
        db.execute(m.ContextActorIdentity.__table__.delete())  # the identity write is lost; the command is not
        again = seed(db, attendees=[{"email": "sam@acme.test", "displayName": "Sam"}])[0]
        assert again.actor_id == made.actor_id
        assert len(actors(db, "Sam")) == 1


def test_names_alone_are_ambiguous_aliases_and_never_create_actors():
    with session_scope() as db:
        a = seed(db, attendees=[{"email": "alex@acme.test", "displayName": "Alex"}])[0]
        b = seed(db, attendees=[{"email": "alex@beta.test", "displayName": "Alex"}])[0]
        priya = seed(db, participants=[Participant(name="Priya", email="priya@acme.test")])[0]
        before = len(actors(db))
        alex, only, nobody = seed(db, "src-9", participants=[
            Participant(name="Alex"), Participant(name="priya"), Participant(name="Marco", speaker_label="Marco")])
        assert (alex.status, alex.actor_id, set(alex.candidates)) == ("unresolved", None, {a.actor_id, b.actor_id})
        assert (only.status, only.actor_id) == ("inferred", priya.actor_id)
        assert (nobody.status, nobody.actor_id, nobody.candidates) == ("unresolved", None, ())
        assert len(actors(db)) == before
        assert sorted(identity.actors_for(db, OWNER, "calendar_name", "ALEX")) == sorted([a.actor_id, b.actor_id])
        with pytest.raises(ValueError):
            identity.resolve_exact(db, OWNER, "alias", "Alex")


def test_calendar_self_and_me_resolve_to_the_owner_actor():
    with session_scope() as db:
        me = identity.owner_actor(db, OWNER)
        assert db.get(Actor, me).name == "owner"
        assert identity.owner_actor(db, OWNER) == me
        mine, other = seed(db, attendees=[
            {"email": "davin@acme.test", "displayName": "Davin", "self": True, "organizer": True},
            {"email": "room-4@resource.calendar.test", "displayName": "Room 4", "resource": True},
            {"email": "priya@acme.test"}])
        assert (mine.actor_id, mine.is_owner, mine.role, mine.created) == (me, True, "organizer", False)
        assert (other.is_owner, db.get(Actor, other.actor_id).name) == (False, "priya")
        # Later meetings resolve the owner by email without the calendar's self flag.
        assert seed(db, "src-2", participants=[Participant(name="Davin", email="davin@acme.test")])[0].is_owner


def test_the_owners_sign_in_email_is_the_owner_without_a_self_flag():
    with session_scope() as db:
        db.add(GoogleIdentity(owner_id=OWNER, subject="sub-1", email="Davin@Acme.test"))
        db.flush()
        me = identity.owner_actor(db, OWNER)
        mine, = seed(db, participants=[Participant(name="Davin", email="davin@acme.test", speaker_label="Davin")])
        assert (mine.actor_id, mine.is_owner, mine.created) == (me, True, False)
        assert not actors(db, "Davin")
        assert identity.resolve_exact(db, OWNER, "granola_speaker", "Davin", "src-1") == me


def test_diarization_labels_are_scoped_to_their_source():
    with session_scope() as db:
        a, = seed(db, "meeting-a", participants=[Participant(name="Alex", email="alex@acme.test",
                                                              speaker_label="Speaker 1")])
        b, = seed(db, "meeting-b", participants=[Participant(name="Bea", email="bea@acme.test",
                                                              speaker_label="Speaker 1")])
        assert identity.resolve_exact(db, OWNER, "granola_speaker", "speaker 1", "meeting-a") == a.actor_id
        assert identity.resolve_exact(db, OWNER, "granola_speaker", "Speaker 1", "meeting-b") == b.actor_id
        assert identity.resolve_exact(db, OWNER, "granola_speaker", "Speaker 1") is None
        assert identity.resolve_exact(db, OWNER, "granola_speaker", "Speaker 1", "meeting-c") is None


def test_strong_identifiers_name_one_actor():
    with session_scope() as db:
        a = command(db, "actor.create", name="Ann")["id"]
        b = command(db, "actor.create", name="Bo")["id"]
        assert identity.add_identity(db, OWNER, a, "linear_user", "user-1", status="exact")
        assert identity.add_identity(db, OWNER, a, "linear_user", "user-1", status="exact")
        assert not identity.add_identity(db, OWNER, b, "linear_user", "user-1")
        assert identity.add_identity(db, OWNER, b, "alias", "AB") and identity.add_identity(db, OWNER, a, "alias", "ab")
        assert identity.resolve_exact(db, OWNER, "linear_user", "user-1") == a


def test_rejected_links_are_durable_scoped_and_human_only():
    pilot = EntityRef(entity_type="project", entity_id="proj-1")
    with session_scope() as db:
        with pytest.raises(DomainError):
            identity.reject_link(db, OWNER, "Pilot", pilot, AGENT)
        scoped = identity.reject_link(db, OWNER, "the  Pilot", pilot, HUMAN, scope="src-1")
        everywhere = identity.reject_link(db, OWNER, "Alex", EntityRef(entity_type="actor", entity_id="a1"), HUMAN)
        assert (scoped.seq, everywhere.seq) == (1, 2)
        assert (scoped.operation, scoped.target_type, scoped.target_id, scoped.negative) == (
            "reject_link", "link", "project:proj-1", True)
        rejected = identity.rejected_links(db, OWNER)
        assert rejected.blocks("The Pilot", "project", "proj-1", "src-1")
        assert not rejected.blocks("The Pilot", "project", "proj-1", "src-2")
        assert rejected.blocks("alex", "actor", "a1", "anything") and rejected.blocks("alex", "actor", "a1")
        assert not rejected.blocks("alex", "actor", "a2")
        # An undo decision lifts the negative.
        db.add(m.ContextDecision(owner_id=OWNER, seq=3, target_type="decision", target_id=everywhere.id,
                                 operation="undo", origin="human", principal="davin", undoes_id=everywhere.id))
        db.flush()
        assert not identity.rejected_links(db, OWNER).blocks("alex", "actor", "a1")
        assert db.get(m.ContextOwnerState, OWNER).decision_seq == 2


def test_merge_candidates_respect_emails_and_no_merge():
    with session_scope() as db:
        a = seed(db, attendees=[{"email": "alex@acme.test", "displayName": "Alex"}])[0].actor_id
        seed(db, attendees=[{"email": "alex@beta.test", "displayName": "Alex"}])
        manual = command(db, "actor.create", name="Alex", allow_duplicate=True)["id"]
        sam = command(db, "actor.create", name="Sam")["id"]
        identity.add_identity(db, OWNER, sam, "alias", "Alex")
        proposals = {p.actor_ids for p in identity.merge_candidates(db, OWNER)}
        # The two emailed Alexes are distinct people; each pairing with an unemailed namesake is proposed.
        assert len(proposals) == 5 and not any(manual not in pair and sam not in pair for pair in proposals)
        with pytest.raises(DomainError):
            identity.no_merge(db, OWNER, a, manual, AGENT)
        identity.no_merge(db, OWNER, manual, a, HUMAN)
        assert all(set(p.actor_ids) != {a, manual} for p in identity.merge_candidates(db, OWNER))


# ---------------------------------------------------------------- merge and unmerge


@pytest.fixture
def graph():
    """Two records of one person (Alex, Al) with tasks, identities, an entity link and a speaker segment."""
    with session_scope() as db:
        winner = seed(db, "meeting-a", participants=[Participant(name="Alex", email="alex@acme.test",
                                                                 speaker_label="Alex")])[0].actor_id
        loser = seed(db, "meeting-b", participants=[Participant(name="Al", email="al@home.test",
                                                                speaker_label="Al")])[0].actor_id
        identity.add_identity(db, OWNER, loser, "alias", "Alex")  # shared alias: deleted on merge, restored after
        tasks = [command(db, "task.create", title=f"Task {i}", assignee_id=loser)["id"] for i in range(2)]
        other = command(db, "task.create", title="Winner task", assignee_id=winner)["id"]
        connection = m.ContextConnection(owner_id=OWNER, connector="api", external_account="davin")
        db.add(connection)
        db.flush()
        source = m.ContextSource(owner_id=OWNER, connection_id=connection.id, connector="api", external_id="b")
        db.add(source)
        db.flush()
        revision = m.ContextSourceRevision(owner_id=OWNER, source_id=source.id, revision=1, content_hash="h" * 64)
        db.add(revision)
        db.flush()
        segment = m.ContextSegment(owner_id=OWNER, revision_id=revision.id, seq=0, speaker_label="Al", text="Hi.",
                                   char_start=0, char_end=3, speaker_actor_id=loser, speaker_link_status="exact")
        link = m.ContextEntityLink(owner_id=OWNER, subject_type="assertion", subject_id="asr-1", target_type="actor",
                                   target_id=loser, mention="Al", role="responsible", status="inferred")
        db.add_all([segment, link])
        db.flush()
        return {"winner": winner, "loser": loser, "tasks": tasks, "other": other, "segment": segment.id,
                "link": link.id}


def identities(db, actor_id):
    return sorted((i.namespace, i.normalized, i.source_scope) for i in db.scalars(
        select(m.ContextActorIdentity).where(m.ContextActorIdentity.actor_id == actor_id)))


def test_merges_need_a_human_decision(graph):
    with session_scope() as db:
        with pytest.raises(DomainError) as error:
            identity.merge(db, OWNER, graph["winner"], graph["loser"], AGENT)
        assert error.value.code == "NOT_AUTHORIZED"
        with pytest.raises(DomainError):
            identity.merge(db, OWNER, graph["winner"], graph["winner"], HUMAN)


def test_merge_then_unmerge_restores_prior_assignments(graph):
    winner, loser = graph["winner"], graph["loser"]
    with session_scope() as db:
        before = {"winner": identities(db, winner), "loser": identities(db, loser)}
        merged = identity.merge(db, OWNER, winner, loser, HUMAN)
        merge_id = merged.id
        assert {db.get(Task, t).assignee_id for t in graph["tasks"]} == {winner}
        assert db.get(Task, graph["tasks"][0]).assignee == "Alex"
        assert db.get(Actor, loser).archived
        assert identities(db, loser) == []
        assert identity.resolve_exact(db, OWNER, "email", "al@home.test") == winner
        assert db.get(m.ContextEntityLink, graph["link"]).target_id == winner
        assert db.get(m.ContextSegment, graph["segment"]).speaker_actor_id == winner
        decision = db.get(m.ContextDecision, merged.decision_id)
        assert (decision.operation, decision.origin, decision.target_id) == ("merge", "human", loser)
        assert sorted(merged.snapshot["tasks"]) == sorted(graph["tasks"])
        with pytest.raises(DomainError):
            identity.merge(db, OWNER, winner, loser, HUMAN)  # the loser is archived now

    with session_scope() as db:
        # After the merge: a person reassigns one task by hand, and new references are made to the winner.
        changed = db.get(Task, graph["tasks"][1])
        command(db, "task.update", task_id=changed.id, expected_revision=changed.revision, assignee="owner")
        later = {mention: m.ContextEntityLink(owner_id=OWNER, subject_type="assertion", subject_id=f"asr-{mention}",
                                              target_type="actor", target_id=winner, mention=mention, role="subject",
                                              status="inferred", updated_at=now())
                 for mention in ("Alex", "Al", "Alexander")}
        db.add_all(later.values())
        db.flush()
        ids = {mention: link.id for mention, link in later.items()}

    with session_scope() as db:
        with pytest.raises(DomainError):
            identity.unmerge(db, OWNER, merge_id, AGENT)
        identity.unmerge(db, OWNER, merge_id, HUMAN)
        assert not db.get(Actor, loser).archived
        assert identities(db, winner) == before["winner"] and identities(db, loser) == before["loser"]
        assert db.get(Task, graph["tasks"][0]).assignee_id == loser
        assert db.get(Task, graph["tasks"][0]).assignee == "Al"
        assert db.get(Task, graph["tasks"][1]).assignee_id != loser  # changed by a person since: kept
        assert db.get(Task, graph["other"]).assignee_id == winner
        link = db.get(m.ContextEntityLink, graph["link"])
        assert (link.target_id, link.status) == (loser, "inferred")
        segment = db.get(m.ContextSegment, graph["segment"])
        assert (segment.speaker_actor_id, segment.speaker_link_status) == (loser, "exact")
        # Post-merge references: "Alex" names both (ambiguous, unresolved), "Al" only the loser.
        alex, al, other = (db.get(m.ContextEntityLink, ids[k]) for k in ("Alex", "Al", "Alexander"))
        assert (alex.target_id, alex.status) == (None, "unresolved")
        assert (al.target_id, al.status) == (loser, "inferred")
        assert other.target_id is None  # names neither: cannot be attributed, stays unresolved
        merge_row = db.get(m.ContextActorMerge, merge_id)
        assert merge_row.undone_at is not None
        undo = db.scalar(select(m.ContextDecision).where(m.ContextDecision.operation == "unmerge"))
        assert undo.undoes_id == merge_row.decision_id
        with pytest.raises(DomainError):
            identity.unmerge(db, OWNER, merge_id, HUMAN)
        assert db.scalar(select(func.count()).select_from(m.ContextDecision)) == 2


def test_candidates_are_plain_frozen_values():
    candidate = IdentityCandidate(actor_id=None, status="unresolved", name="Marco")
    with pytest.raises(AttributeError):
        candidate.actor_id = "x"
