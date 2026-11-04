"""Meeting notices: one ready notice per review revision, material-only renotification, one consolidated
backfill notice, and the 17:00 owner-local digest (DST, after-cutoff, empty day, dedup across retries and
proposal updates, resolution before delivery, quiet hours and priority rules through jarvis.notices)."""

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest
from jarvis import notices as core
from jarvis.context import notices, service
from jarvis.context.models import ContextDecision, ContextProposal, ContextSource, ContextSourceRevision
from jarvis.db import session_scope
from jarvis.models import Job, Notification, OwnerSettings
from jarvis.worker import notification_url
from sqlalchemy import select

OWNER = "davin"
ZONE = "America/New_York"
# 2026-11-02 (Monday, EST, UTC-5): 17:00 local is 22:00 UTC.
DAY = datetime(2026, 11, 2, 14, 0, tzinfo=UTC)


def local(year, month, day, hour, minute=0):
    from zoneinfo import ZoneInfo

    return datetime(year, month, day, hour, minute, tzinfo=ZoneInfo(ZONE)).astimezone(UTC)


@pytest.fixture(autouse=True)
def owner():
    with session_scope() as db:
        db.add(OwnerSettings(owner_id=OWNER, values={"timezone": ZONE}))


def meeting(db, title="Pilot scope", *, ended=None, created=None, ready=None, status="ready", revision=1):
    ended = ended or DAY - timedelta(hours=2)
    source = ContextSource(owner_id=OWNER, connector="api", external_id=title, title=title, status=status,
                           current_revision=revision, created_at=created or ended + timedelta(minutes=5),
                           updated_at=ready or DAY)
    db.add(source)
    db.flush()
    for n in range(1, revision + 1):
        db.add(ContextSourceRevision(owner_id=OWNER, source_id=source.id, revision=n, content_hash=f"h{n}",
                                     occurred_at=ended - timedelta(minutes=30), ended_at=ended))
    db.flush()
    return source


def proposal(db, source, title="Send the budget", *, disposition="pending", updated=None, task_id=None,
             fingerprint=None):
    row = ContextProposal(owner_id=OWNER, source_id=source.id if source else None, kind="create",
                          fingerprint=fingerprint or title, title=title, disposition=disposition, task_id=task_id,
                          updated_at=updated or DAY)
    db.add(row)
    db.flush()
    return row


def rows(db, category=None):
    query = select(Notification).where(Notification.owner_id == OWNER).order_by(Notification.created_at)
    return list(db.scalars(query.where(Notification.category == category) if category else query))


# ---------------------------------------------------------------- meeting-ready


def test_ready_notice_once_per_review_with_stable_key_and_deep_link():
    with session_scope() as db:
        source = meeting(db)
        proposal(db, source)
        proposal(db, source, "Draft the checklist")
        assert notices.sync(db, OWNER, DAY)["ready"] == 1
        # Retries and later ticks find the same review.
        assert notices.sync(db, OWNER, DAY + timedelta(minutes=1))["ready"] == 0
        [n] = rows(db, notices.READY)
        assert n.dedup_key == f"context:ready:{source.id}:1"
        assert n.title == "Meeting ready for review" and n.body == "Pilot scope · 2 proposals"
        assert notification_url(n) == f"/?view=meetings&source={source.id}"
        assert core.eligible(db, n, DAY)


def test_ready_waits_for_pending_context_work():
    with session_scope() as db:
        source = meeting(db)
        job = Job(owner_id=OWNER, kind="context_proposals", status="queued", payload={"source_id": source.id})
        db.add(job)
        db.flush()
        assert notices.sync(db, OWNER, DAY)["ready"] == 0
        meeting(db, "Still processing", status="processing")
        job.status = "succeeded"
        db.flush()
        assert notices.sync(db, OWNER, DAY)["ready"] == 1


def test_new_revision_notifies_only_for_material_new_pending_work():
    with session_scope() as db:
        source = meeting(db)
        kept = proposal(db, source)
        notices.sync(db, OWNER, DAY)
        # Revision 2 with the same pending proposal at the same revision: quiet.
        source.current_revision = 2
        db.add(ContextSourceRevision(owner_id=OWNER, source_id=source.id, revision=2, content_hash="h2"))
        db.flush()
        assert notices.sync(db, OWNER, DAY + timedelta(hours=1))["ready"] == 0
        # A materially revised proposal is new pending work.
        kept.revision = 2
        db.flush()
        assert notices.sync(db, OWNER, DAY + timedelta(hours=2))["ready"] == 1
        # Revision 3 adds a new proposal; decided ones never count.
        source.current_revision = 3
        proposal(db, source, "Dismissed one", disposition="dismissed")
        db.flush()
        assert notices.sync(db, OWNER, DAY + timedelta(hours=3))["ready"] == 0
        proposal(db, source, "Book the venue")
        assert notices.sync(db, OWNER, DAY + timedelta(hours=3))["ready"] == 1
        assert [n.dedup_key.rsplit(":", 1)[1] for n in rows(db, notices.READY)] == ["1", "2", "3"]


def test_ready_notice_drops_when_the_review_is_done_before_delivery():
    with session_scope() as db:
        source = meeting(db)
        p = proposal(db, source)
        notices.sync(db, OWNER, DAY)
        [n] = rows(db, notices.READY)
        p.disposition = "approved"
        db.flush()
        assert not core.eligible(db, n, DAY)
        p.disposition = "pending"
        source.deleted_at = DAY
        db.flush()
        assert not core.eligible(db, n, DAY)


def test_backfill_produces_one_consolidated_notice():
    with session_scope() as db:
        old = DAY - timedelta(days=40)
        for i in range(5):
            source = meeting(db, f"Old meeting {i}", ended=old + timedelta(days=i), created=DAY - timedelta(hours=1))
            proposal(db, source, f"Task {i}")
        assert notices.sync(db, OWNER, DAY) == {"ready": 0, "backfill": 5, "digest": 0}
        assert notices.sync(db, OWNER, DAY + timedelta(minutes=1)) == {"ready": 0, "backfill": 0, "digest": 0}
        [n] = rows(db, notices.BACKFILL)
        assert n.body == "5 meetings ready for review" and len(n.target["source_ids"]) == 5
        assert notification_url(n) == "/?view=meetings"
        # More history while the notice is unseen joins it; once delivered, the rest wait for the next day.
        meeting(db, "Old meeting 5", ended=old, created=DAY)
        assert notices.sync(db, OWNER, DAY + timedelta(minutes=2))["backfill"] == 1
        assert n.body == "6 meetings ready for review"
        core.apply(n, "deliver", DAY + timedelta(minutes=3))
        meeting(db, "Old meeting 6", ended=old, created=DAY)
        assert notices.sync(db, OWNER, DAY + timedelta(minutes=4))["backfill"] == 0
        assert notices.sync(db, OWNER, DAY + timedelta(days=1))["backfill"] == 1
        assert len(rows(db, notices.BACKFILL)) == 2
        # No meeting of the backfill gets its own ready notice afterwards.
        assert not rows(db, notices.READY)


def test_meetings_ready_long_ago_are_consolidated_not_pushed_one_by_one():
    with session_scope() as db:
        for i in range(3):
            meeting(db, f"Earlier {i}", ready=DAY - timedelta(days=3))
        assert notices.sync(db, OWNER, DAY)["backfill"] == 3
        assert not rows(db, notices.READY)


# ---------------------------------------------------------------- digest


def test_digest_at_1700_owner_local_across_dst():
    """17:00 New York is 21:00 UTC in summer time and 22:00 UTC in standard time, including transition days."""
    for day, utc_hour in (((2026, 3, 8), 21), ((2026, 7, 1), 21), ((2026, 11, 1), 22), ((2026, 11, 2), 22)):
        with session_scope() as db:
            for model in (Notification, ContextProposal, ContextSourceRevision, ContextSource):
                db.execute(model.__table__.delete())
            start = local(*day, 17)
            assert start.hour == utc_hour
            source = meeting(db, f"Meeting {day}", ready=start - timedelta(hours=3))
            proposal(db, source, f"Work {day}")
            notices.sync(db, OWNER, start - timedelta(hours=3))
            assert notices.digest(db, OWNER, start - timedelta(minutes=1)) is None
            row = notices.digest(db, OWNER, start + timedelta(minutes=5))
            assert row is not None and row.dedup_key == f"context:digest:{OWNER}:{datetime(*day).date()}"
            assert row.target["source_ids"] == [source.id]
            assert notices.digest(db, OWNER, start + timedelta(hours=1)) is None  # 18:05: the hour has passed


def test_digest_items_after_cutoff_wait_for_the_next_day():
    with session_scope() as db:
        source = meeting(db, ready=local(2026, 11, 2, 17, 20))
        proposal(db, source)
        notices.sync(db, OWNER, local(2026, 11, 2, 17, 20))
        assert notices.digest(db, OWNER, local(2026, 11, 2, 17, 30)) is None
        row = notices.digest(db, OWNER, local(2026, 11, 3, 17, 5))
        assert row.target["source_ids"] == [source.id] and row.body == "1 meeting to review"


def test_digest_empty_day_sends_nothing():
    with session_scope() as db:
        source = meeting(db)
        proposal(db, source, disposition="approved")
        proposal(db, source, "Published", disposition="published")
        proposal(db, source, "Dismissed", disposition="dismissed")
        notices.sync(db, OWNER, DAY)
        assert notices.digest(db, OWNER, local(2026, 11, 2, 17, 10)) is None
        assert not rows(db, notices.DIGEST)


def test_digest_dedupes_across_retries_proposal_updates_and_same_work(monkeypatch):
    monkeypatch.setattr(service, "question_items", lambda db, owner: [])
    with session_scope() as db:
        a, b = meeting(db, "Monday sync"), meeting(db, "Tuesday sync")
        later = proposal(db, a, "Send the budget", disposition="later", task_id="task-1")
        proposal(db, b, "Send the budget (again)", disposition="later", task_id="task-1")
        proposal(db, b, "Hire a designer", disposition="later")
        at = local(2026, 11, 2, 17, 10)
        first = notices.digest(db, OWNER, at)
        assert first.body == "2 deferred proposals"
        # A retried tick, and a proposal revised before the cutoff, add nothing.
        later.revision, later.title = 3, "Send the revised budget"
        db.flush()
        assert notices.digest(db, OWNER, at + timedelta(minutes=1)) is None
        assert len(rows(db, notices.DIGEST)) == 1
        assert core.eligible(db, first, at + timedelta(minutes=2)) and first.body == "2 deferred proposals"


def test_later_with_until_and_deferred_questions_follow_their_time(monkeypatch):
    at = local(2026, 11, 2, 17, 10)
    questions = [{"key": "context:q1", "deferred_until": at - timedelta(hours=2)},
                 {"key": "context:q2", "deferred_until": at + timedelta(hours=20)},
                 {"key": "context:q3", "deferred_until": None}]
    monkeypatch.setattr(service, "question_items", lambda db, owner: questions)
    with session_scope() as db:
        tomorrow = proposal(db, None, "Lint follow-up", disposition="later")
        db.add(ContextDecision(owner_id=OWNER, seq=1, target_type="proposal", target_id=tomorrow.id,
                               operation="later", origin="human", principal=OWNER,
                               effective_at=at + timedelta(hours=20)))
        db.flush()
        row = notices.digest(db, OWNER, at)
        assert row.target["question_keys"] == ["context:q1"] and row.target["proposal_ids"] == []
        assert row.body == "1 deferred question"
        assert notices.items(db, OWNER, local(2026, 11, 3, 17))["proposal_ids"] == [tomorrow.id]


def test_digest_resolves_on_review_publication_or_dismissal_before_it_is_pushed(monkeypatch):
    monkeypatch.setattr(service, "question_items", lambda db, owner: [])
    with session_scope() as db:
        source = meeting(db)
        pending = proposal(db, source)
        deferred = proposal(db, source, "Later one", disposition="later")
        notices.sync(db, OWNER, DAY)
        at = local(2026, 11, 2, 17, 5)
        row = notices.digest(db, OWNER, at)
        assert row.body == "1 meeting to review · 1 deferred proposal"
        deferred.disposition = "published"
        db.flush()
        assert core.eligible(db, row, at) and row.body == "1 meeting to review"
        pending.disposition = "dismissed"
        db.flush()
        assert not core.eligible(db, row, at)
        # A meeting ready after the digest was written is not added to it.
        other = meeting(db, "Evening call", ready=at)
        proposal(db, other, "New after the digest")
        notices.sync(db, OWNER, at)
        assert not core.eligible(db, row, at + timedelta(minutes=1))


def test_quiet_hours_hold_the_digest_and_plan_pushes_it_alone():
    with session_scope() as db:
        db.get(OwnerSettings, OWNER).values = {"timezone": ZONE, "quiet_start": "16:00", "quiet_end": "18:00",
                                               "notify_min_priority": "urgent"}
        source = meeting(db)
        proposal(db, source)
        notices.sync(db, OWNER, DAY)
        row = notices.digest(db, OWNER, local(2026, 11, 2, 17, 5))
        assert not core.eligible(db, row, local(2026, 11, 2, 17, 6))
        after = local(2026, 11, 2, 18, 1)
        assert core.eligible(db, row, after)
        [ready] = rows(db, notices.READY)
        # Below the owner's threshold the ready notice waits for the hourly digest; the 17:00 digest goes alone.
        pushes = core.plan(db, [row, ready], lambda n: True, after)
        assert row in pushes and ready not in pushes
        assert ready.bundle_id and db.get(Notification, ready.bundle_id).category == "digest"


def test_notification_url_targets_for_context_categories():
    for target, url in (({"view": "meetings", "source_id": "s1"}, "/?view=meetings&source=s1"),
                        ({"view": "meetings", "source_ids": ["s1", "s2"]}, "/?view=meetings")):
        assert notification_url(SimpleNamespace(task_id=None, target=target)) == url


@pytest.fixture
def context_on(monkeypatch):
    from jarvis.config import get_settings

    monkeypatch.setenv("JARVIS_CONTEXT_ENABLED", "true")
    get_settings.cache_clear()
    yield
    monkeypatch.delenv("JARVIS_CONTEXT_ENABLED")
    get_settings.cache_clear()


def test_deferred_agent_suggestion_reaches_the_digest_once_its_deferral_ends(client, context_on):
    """Unmocked: an agent suggestion (C2 `service.question_items`) deferred with Remind me later."""
    from .test_service import bot, mint, seed, suggestion

    ids = seed()
    writer, _ = mint(client, "context:write")
    with bot(writer):
        receipt = service.feedback(suggestion(ids))
    with session_scope() as db:
        service.defer_question(db, OWNER, receipt["question_key"], 1, timedelta(hours=3), command_id="cmd-defer")
        start = datetime.now(UTC)
        assert notices.items(db, OWNER, start)["question_keys"] == []
        assert notices.items(db, OWNER, start + timedelta(hours=4))["question_keys"] == [receipt["question_key"]]
