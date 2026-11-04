"""Derived views: open loops (waiting-for vs own work, linked task state), meeting prep from the calendar,
person pages and the changes feed."""

from datetime import timedelta

import pytest
from jarvis.context import decisions, evidence, identity, views
from jarvis.context.models import ContextEntityLink, ContextEvidence, ContextSource, ContextSourceRevision
from jarvis.db import session_scope
from jarvis.domain import DomainError, emit
from jarvis.models import GoogleCalendar, GoogleCalendarEvent, GoogleIdentity, now
from sqlalchemy import select

from .test_briefs import HUMAN, People, pilot
from .test_temporal import OWNER


def calendar(db, owner=OWNER):
    db.add(GoogleIdentity(owner_id=owner, subject=f"sub-{owner}", email=f"{owner}@example.com",
                          calendar_enabled=True, status="connected"))
    db.flush()
    row = GoogleCalendar(owner_id=owner, provider_id="primary", title="Work", timezone="UTC", selected=True,
                         available=True)
    db.add(row)
    db.flush()
    return row


def event(db, cal, provider_id, title, start, *, guests=(), hours=1, **extra):
    payload = {"id": provider_id, "summary": title, "status": "confirmed",
               "start": {"dateTime": start.isoformat()}, "end": {"dateTime": (start + timedelta(hours=hours)).isoformat()},
               "attendees": [{"email": email, "displayName": name, **({"self": True} if name == "me" else {})}
                             for name, email in guests], **extra}
    row = GoogleCalendarEvent(calendar_id=cal.id, provider_id=provider_id, payload=payload)
    db.add(row)
    db.flush()
    return row


def soon(hours):
    return now().replace(microsecond=0) + timedelta(hours=hours)


def test_open_loops_separate_waiting_for_from_own_work():
    with session_scope() as db:
        w = People(db)
        made = pilot(w)
        between = w.say("Marco sends Priya the logs.", kind="commitment", responsible=[w.marco], recipient=w.priya)
        shared = w.say("We publish the rollout plan.", kind="commitment", responsible=["we"])
        hypothetical = w.say("Marco would migrate the data.", kind="commitment", responsible=[w.marco],
                             modality="hypothetical")
        loops = views.open_loops(db, OWNER)["items"]
        buckets = {i["assertion_id"]: i["bucket"] for i in loops}
        assert buckets == {made["budget"].id: "waiting_for", made["checklist"].id: "own",
                           between.id: "between_others", shared.id: "collective"}
        assert hypothetical.id not in buckets
        assert loops[0]["assertion_id"] == made["budget"].id  # dated items first
        waiting = views.open_loops(db, OWNER, waiting_for=True)["items"]
        assert [i["assertion_id"] for i in waiting] == [made["budget"].id]
        assert waiting[0]["responsible"][0] | {"is_me": False} == {
            "actor_id": w.priya, "name": "Priya", "mention": None, "resolved": True, "is_me": False,
            "collective": False}
        assert waiting[0]["evidence"][0]["quote"] == "Priya sends the revised budget."
        mine = views.open_loops(db, OWNER, owner_actor="me")["items"]
        assert [i["assertion_id"] for i in mine] == [made["checklist"].id]
        assert [i["assertion_id"] for i in views.open_loops(db, OWNER, owner_actor=w.marco)["items"]] == [between.id]
        assert {i["assertion_id"] for i in views.open_loops(db, OWNER, waiting_for=False)["items"]} == {
            made["checklist"].id, between.id, shared.id}
        scoped = views.open_loops(db, OWNER, entity=("actor", w.priya))["items"]
        assert {i["assertion_id"] for i in scoped} == {made["budget"].id, between.id}


def test_completed_linked_task_is_covered_work_and_the_promise_stays_open():
    with session_scope() as db:
        w = People(db)
        made = pilot(w)
        task = w.task("Send budget", status="completed")
        w.link_task(made["budget"], task)
        loop = views.open_loops(db, OWNER, waiting_for=True)["items"][0]
        assert (loop["status"], loop["task_state"]) == ("open", "covered")
        assert loop["tasks"][0]["id"] == task.id and loop["tasks"][0]["status"] == "completed"
        made["budget"].completion_evidence = [{"quote": "Sent it this morning."}]
        db.flush()
        assert views.open_loops(db, OWNER, waiting_for=True)["items"] == []
        done = views.open_loops(db, OWNER, waiting_for=True, include_completed=True)["items"]
        assert done[0]["status"] == "completion_observed"


def test_commitments_without_surviving_evidence_leave_the_loops():
    with session_scope() as db:
        w = People(db)
        made = pilot(w)
        source_id = db.scalar(select(ContextEvidence.source_id).where(
            ContextEvidence.assertion_id == made["budget"].id))
        db.get(ContextSource, source_id).deleted_at = now()
        result = views.open_loops(db, OWNER)
        assert made["budget"].id not in {i["assertion_id"] for i in result["items"]}
        assert result["gaps"] == ["1 commitment(s) lost their supporting evidence and are not shown."]
        evidence.propagate_source_deletion(db, OWNER, source_id)
        assert views.open_loops(db, OWNER)["gaps"] == []  # retracted now, no longer current


def test_meeting_prep_finds_the_next_event_with_a_person_and_resolves_attendees():
    with session_scope() as db:
        w = People(db)
        made = pilot(w)
        identity.add_identity(db, OWNER, w.priya, "email", "priya@example.com")
        cal = calendar(db)
        event(db, cal, "lunch", "Lunch", soon(1))  # no guests
        nxt = event(db, cal, "sync", "Pilot sync", soon(3) - timedelta(days=7), recurrence=["RRULE:FREQ=DAILY"],
                    guests=[("me", "davin@example.com"), ("Priya", "priya@example.com"),
                            ("Dana", "dana@example.com")])
        source = db.scalar(select(ContextSource).join(ContextEvidence, ContextEvidence.source_id == ContextSource.id)
                           .where(ContextEvidence.assertion_id == made["budget"].id))
        source.title = "Pilot sync"
        revision = db.scalar(select(ContextSourceRevision).where(ContextSourceRevision.source_id == source.id))
        revision.calendar_event_id = "sync_20261101T150000Z"  # an instance of the series
        db.flush()

        prep = views.meeting_prep(db, OWNER, entity=("actor", w.priya))
        assert prep["event"]["event_id"] == nxt.id and prep["event"]["title"] == "Pilot sync"
        assert prep["event"]["start"] == soon(3).isoformat()
        attendees = {a["name"]: (a["actor_id"], a["status"]) for a in prep["event"]["attendees"]}
        assert attendees == {"me": (w.me, "exact"), "Priya": (w.priya, "exact"), "Dana": (None, "unresolved")}
        assert [i["assertion_id"] for i in prep["open_loops"]] == [made["budget"].id]
        assert prep["previous_meetings"] == [{"source_id": source.id, "title": "Pilot sync",
                                              "occurred_at": revision.occurred_at.isoformat(), "match": "series"}]
        assert prep["gaps"] == ["Not matched to a person yet: Dana."]

        project = views.meeting_prep(db, OWNER, entity=("project", w.project))  # series of a source about it
        assert project["event"]["event_id"] == nxt.id
        assert made["decision"].id in {d["assertion_id"] for d in project["decisions"]}
        assert [q["assertion_id"] for q in project["open_questions"]] == [made["question"].id]
        assert views.meeting_prep(db, OWNER)["event"]["event_id"] == nxt.id  # next meeting with guests
        assert views.meeting_prep(db, OWNER, calendar_event_id=nxt.provider_id)["event"]["event_id"] == nxt.id
        with pytest.raises(DomainError):
            views.meeting_prep(db, OWNER, calendar_event_id="other-owners-event")


def test_meeting_prep_without_a_matching_event_says_so():
    with session_scope() as db:
        w = People(db)
        pilot(w)
        prep = views.meeting_prep(db, OWNER, entity=("actor", w.marco))
        assert prep["event"] is None and prep["entity"]["name"] == "Marco"
        assert prep["gaps"] == ["No upcoming calendar event involves this actor in the next 14 days."]


def test_person_page_collects_identities_context_loops_tasks_meetings_and_links():
    with session_scope() as db:
        w = People(db)
        made = pilot(w)
        identity.add_identity(db, OWNER, w.priya, "email", "priya@example.com", status="exact")
        role = w.say("Priya leads finance for the pilot.", kind="relationship", subject=("actor", w.priya))
        owed = w.say("Marco sends Priya the logs.", kind="commitment", responsible=[w.marco], recipient=w.priya)
        task = w.task("Review budget")
        task.assignee_id, task.project_id = w.priya, w.project
        db.add(ContextEntityLink(owner_id=OWNER, assertion_id=made["budget"].id, subject_type="assertion",
                                 subject_id=made["budget"].id, target_type="actor", target_id=w.priya,
                                 mention="Priya", role="responsible", status="inferred"))
        db.flush()
        page = views.person(db, OWNER, w.priya)
        assert page["actor"]["name"] == "Priya" and not page["actor"]["is_me"]
        assert [(i["namespace"], i["value"]) for i in page["identities"]] == [("email", "priya@example.com")]
        assert [c["assertion_id"] for c in page["context"]] == [role.id]
        assert [i["assertion_id"] for i in page["responsible_for"]] == [made["budget"].id]
        assert [i["assertion_id"] for i in page["owed_to_them"]] == [owed.id]
        assert [t["id"] for t in page["tasks"]] == [task.id]
        assert page["projects"] == [{"project_id": w.project, "name": "Clinic pilot", "context_items": 2,
                                     "tasks": 1}]
        assert len(page["meetings"]) == 3 and page["links_to_review"][0]["mention"] == "Priya"
        assert views.person(db, OWNER, w.me)["actor"]["is_me"]
        with pytest.raises(DomainError):
            views.person(db, "someone-else", w.priya)


def test_changes_feed_is_cursor_ordered_with_tombstones():
    with session_scope() as db:
        w = People(db)
        made = pilot(w)
        start = views.changes(db, OWNER)["current_cursor"]
        emit(db, OWNER, "context.changed", made["fact"].id, 1)
        emit(db, OWNER, "task.changed", "unrelated", 1)
        decisions.wrong(db, OWNER, made["decision"].id, made["decision"].current_version, **HUMAN)
        source_id = db.scalar(select(ContextEvidence.source_id).where(ContextEvidence.assertion_id == made["why"].id))
        db.get(ContextSource, source_id).deleted_at = now()
        emit(db, OWNER, "context.changed", source_id, 1)

        page = views.changes(db, OWNER, after=start, limit=2)
        assert page["has_more"] and [i["id"] for i in page["items"]] == [made["fact"].id, made["decision"].id]
        fact, wrong = page["items"]
        assert (fact["type"], fact["deleted"], fact["statement"]) == ("assertion", False, "The pilot covers 40 clinics.")
        assert (wrong["type"], wrong["lifecycle"], wrong["deleted"], wrong["statement"]) == (
            "assertion", "retracted", True, None)
        rest = views.changes(db, OWNER, after=page["next_cursor"])
        assert not rest["has_more"] and rest["items"][0] | {"cursor": 0, "changed_at": None} == {
            "cursor": 0, "changed_at": None, "id": source_id, "event_revision": 1, "type": "source", "revision": 1,
            "status": "received", "deleted": True, "title": None}
        assert views.changes(db, OWNER, after=rest["next_cursor"])["items"] == []
        with pytest.raises(DomainError) as ahead:
            views.changes(db, OWNER, after=rest["current_cursor"] + 1)
        assert ahead.value.code == "CURSOR_INVALID"
        assert views.changes(db, "someone-else")["items"] == []


def test_upcoming_skips_declined_and_ended_events():
    with session_scope() as db:
        cal = calendar(db)
        event(db, cal, "ended", "Ended", soon(-3))
        event(db, cal, "declined", "Declined", soon(1), declined=True)
        later = event(db, cal, "later", "Later", soon(2))
        assert [o.event.id for o in views.upcoming(db, OWNER)] == [later.id]
