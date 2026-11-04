"""Calendar association: exact provider-ID match, provisional title-and-time match, decisions respected,
and attendee identity seeding (an ingest step) through identity.seed_candidates (package B4)."""

from datetime import timedelta

import pytest
from jarvis.context import calendar_link, sources
from jarvis.context import models as m
from jarvis.db import session_scope
from jarvis.models import GoogleCalendar, GoogleCalendarEvent, GoogleIdentity
from sqlalchemy import select

from .conftest import MEETING_AT

OWNER = "davin"


def event(db, calendar, provider_id, title, start, attendees=(), status="confirmed"):
    row = GoogleCalendarEvent(calendar_id=calendar.id, provider_id=provider_id, payload={
        "id": provider_id, "summary": title, "status": status, "start": {"dateTime": start.isoformat()},
        "end": {"dateTime": (start + timedelta(minutes=30)).isoformat()},
        "attendees": [dict(a) for a in attendees]})
    db.add(row)
    db.flush()
    return row.id


@pytest.fixture
def calendar():
    with session_scope() as db:
        db.add_all([GoogleIdentity(owner_id=OWNER, subject="s-davin", email="davin@example.com"),
                    GoogleIdentity(owner_id="other", subject="s-other", email="other@example.com")])
        db.flush()
        mine = GoogleCalendar(owner_id=OWNER, provider_id="primary", title="Davin")
        theirs = GoogleCalendar(owner_id="other", provider_id="primary", title="Other")
        db.add_all([mine, theirs])
        db.flush()
        guests = [{"email": "priya@example.com", "displayName": "Priya"},
                  {"email": "davin@example.com", "organizer": True, "self": True, "responseStatus": "accepted"},
                  {"email": "room-1@resource.calendar.google.com", "resource": True}]
        return {
            "exact": event(db, mine, "evt_20261103T150000Z", "Weekly sync", MEETING_AT, guests),
            "similar": event(db, mine, "evt-2", "Pilot scope review", MEETING_AT + timedelta(minutes=10)),
            "far": event(db, mine, "evt-3", "Pilot scope", MEETING_AT + timedelta(hours=5)),
            "cancelled": event(db, mine, "evt-4", "Pilot scope", MEETING_AT, status="cancelled"),
            "foreign": event(db, theirs, "evt_foreign", "Pilot scope", MEETING_AT),
        }


def links(source_id):
    with session_scope() as db:
        return [(x.target_id, x.status, x.score) for x in db.scalars(select(m.ContextEntityLink).where(
            m.ContextEntityLink.subject_type == "source", m.ContextEntityLink.subject_id == source_id))]


def receive(envelope):
    with session_scope() as db:
        return sources.receive(db, OWNER, envelope)


def seed(source_id):
    with session_scope() as db:
        return calendar_link.seed_identities(db, OWNER, source_id)


def test_exact_calendar_event_id_links_exactly_and_seeds_attendees(calendar, envelope):
    result = receive(envelope.model_copy(update={"calendar_event_id": "evt_20261103T150000Z"}))
    assert links(result.source_id) == [(calendar["exact"], "exact", 1.0)]
    with session_scope() as db:  # receipt alone creates no identities
        assert db.scalar(select(m.ContextActorIdentity.id)) is None
    candidates = seed(result.source_id)
    assert {c.email for c in candidates if c.email} >= {"priya@example.com", "davin@example.com"}
    with session_scope() as db:
        emails = set(db.scalars(select(m.ContextActorIdentity.normalized).where(
            m.ContextActorIdentity.owner_id == OWNER, m.ContextActorIdentity.namespace == "email")))
    assert {"priya@example.com", "davin@example.com"} <= emails and not any("resource" in e for e in emails)


def fake_seeding(monkeypatch):
    seeded = []
    monkeypatch.setattr(calendar_link, "seed_candidates",
                        lambda db, owner, source_id, participants, attendees: seeded.append(
                            (owner, source_id, participants, attendees)))
    return seeded


def test_exact_match_passes_calendar_attendees_to_seeding(calendar, envelope, monkeypatch):
    seeded = fake_seeding(monkeypatch)
    result = receive(envelope.model_copy(update={"calendar_event_id": "evt_20261103T150000Z"}))
    assert seeded == []
    seed(result.source_id)
    [(owner, source_id, participants, attendees)] = seeded
    assert (owner, source_id) == (OWNER, result.source_id) and [p.name for p in participants] == ["Davin", "Priya", "Marco"]
    assert attendees == [{"email": "priya@example.com", "displayName": "Priya"},
                         {"email": "davin@example.com", "organizer": True, "self": True},
                         {"email": "room-1@resource.calendar.google.com", "resource": True}]


def test_title_and_time_give_a_provisional_match_without_attendee_seeding(calendar, envelope, monkeypatch):
    seeded = fake_seeding(monkeypatch)
    result = receive(envelope)
    [(target, status, score)] = links(result.source_id)
    assert (target, status) == (calendar["similar"], "inferred") and calendar_link.MIN_SCORE <= score < 1
    seed(result.source_id)
    assert seeded[0][3] == []


def test_unknown_event_id_falls_back_and_unrelated_titles_do_not_link(calendar, envelope):
    unrelated = envelope.model_copy(update={"calendar_event_id": "nope", "title": "Dentist"})
    assert links(receive(unrelated).source_id) == []


def test_rejected_links_are_not_proposed_again_and_confirmed_ones_are_kept(calendar, envelope):
    result = receive(envelope)
    with session_scope() as db:
        link = db.scalar(select(m.ContextEntityLink).where(m.ContextEntityLink.subject_id == result.source_id))
        link.status = "rejected"
    later = receive(envelope.model_copy(update={"summary": "now with a summary"}))
    assert later.change == "metadata" and links(result.source_id) == [(calendar["similar"], "rejected", link.score)]
    with session_scope() as db:
        db.get(m.ContextEntityLink, link.id).status = "confirmed"
    receive(envelope.model_copy(update={"summary": "edited again", "calendar_event_id": "evt_20261103T150000Z"}))
    assert [x[:2] for x in links(result.source_id)] == [(calendar["similar"], "confirmed")]


def test_seeding_is_skipped_when_identity_module_is_absent_or_source_deleted(calendar, envelope, monkeypatch):
    from jarvis.context import sources

    result = receive(envelope)
    with session_scope() as db:
        sources.delete_source(db, OWNER, result.source_id)
    assert seed(result.source_id) is None
    monkeypatch.setattr(calendar_link, "seed_candidates", None)
    assert seed(receive(envelope.model_copy(update={"external_id": "m2"})).source_id) is None


def test_score_weights_title_and_time():
    assert calendar_link.score("Pilot scope", MEETING_AT, "pilot   SCOPE!", MEETING_AT) == 1.0
    assert calendar_link.score("", MEETING_AT, "Pilot", MEETING_AT) == 0
    assert calendar_link.score("Pilot scope", MEETING_AT, "Weekly sync", MEETING_AT) == 0
    assert calendar_link.score("Acme kickoff", MEETING_AT, "Kickoff with ACME", MEETING_AT) == 1.0
    assert calendar_link.score("Pilot", MEETING_AT, "Pilot", MEETING_AT + calendar_link.WINDOW) == 0.6
