"""Smart notifications: the state machine, words snoozes, bundles, digests, dedupe and recurring series."""

import json
from datetime import UTC, datetime, timedelta
from zoneinfo import ZoneInfo

import pytest
from jarvis import notices, worker
from jarvis.config import get_settings
from jarvis.db import session_scope
from jarvis.domain import DomainError, deliver_occurrence, scan_schedules
from jarvis.models import Delivery, Job, Notification, Occurrence, PushSubscription, Schedule, Task, now
from sqlalchemy import select
from test_structure import OWNER, run

ENDPOINT = "https://fcm.googleapis.com/fcm/send/device"


@pytest.fixture
def phone():
    """One registered device, no quiet hours (so the test does not depend on the time it runs)."""
    run("settings.update", {"quiet_enabled": False, "timezone": "America/Chicago"})
    with session_scope() as db:
        db.add(PushSubscription(id="phone", owner_id=OWNER, device_id="d",
                                subscription={"endpoint": ENDPOINT, "keys": {"p256dh": "k", "auth": "a"}}))


@pytest.fixture
def pushes(monkeypatch):
    sent = []
    monkeypatch.setattr(get_settings(), "vapid_private_key", "synthetic")
    monkeypatch.setattr(worker, "webpush", lambda **kw: sent.append(json.loads(kw["data"])))
    return sent


def notice(title="Due", **kw):
    with session_scope() as db:
        row = Notification(owner_id=OWNER, title=title, scheduled_at=now() - timedelta(seconds=5), **kw)
        db.add(row)
        db.flush()
        return row.id


def get(identity):
    with session_scope() as db:
        row = db.get(Notification, identity)
        db.expunge(row)
        return row


def deliveries():
    with session_scope() as db:
        return [(db.get(Notification, d.notification_id).category, d.notification_id) for d in db.scalars(select(Delivery))]


# ------------------------------------------------------------------ state machine


def test_states_stay_distinct_through_every_transition():
    t0 = datetime(2030, 1, 1, 12, tzinfo=UTC)
    n = Notification(owner_id=OWNER, title="x", scheduled_at=t0 + timedelta(minutes=5), generation=1)
    assert notices.state(n, t0) == "scheduled"
    assert notices.state(n, t0 + timedelta(minutes=5)) == "pending"
    t1 = t0 + timedelta(minutes=6)
    assert notices.apply(n, "deliver", t1) == "delivered" and n.delivered_at == t1
    with pytest.raises(DomainError):
        notices.apply(n, "deliver", t1)  # Delivered once per generation.
    assert notices.apply(n, "read", t1) == "read"
    until = t1 + timedelta(hours=1)
    assert notices.apply(n, "snooze", t1, until) == "snoozed"
    # A snooze is a new generation: not read, not delivered, waiting until `until`.
    assert (n.generation, n.read_at, n.delivered_at, n.snoozed_until) == (2, None, None, until)
    assert notices.state(n, until) == "pending"
    assert notices.apply(n, "deliver", until) == "delivered"
    assert notices.apply(n, "complete", until) == "completed"
    with pytest.raises(DomainError, match="already completed"):
        notices.apply(n, "snooze", until, until + timedelta(hours=1))
    # Reading a snoozed notice ends the snooze instead of leaving it to fire.
    m = Notification(owner_id=OWNER, title="y", scheduled_at=t0, generation=1)
    notices.apply(m, "snooze", t0, t0 + timedelta(hours=2))
    assert notices.apply(m, "read", t0) == "read" and m.snoozed_until is None


def test_snooze_in_words_resolves_in_the_owners_zone_and_keeps_the_row():
    run("settings.update", {"timezone": "America/Chicago"})
    row = run("task.create", {"title": "Call the bank"})
    identity = notice(task_id=row["id"])
    result = run("notification.snooze", {"notification_id": identity, "until": "tomorrow morning"})
    assert result["state"] == "snoozed"
    until = datetime.fromisoformat(result["snoozed_until"]).astimezone(ZoneInfo("America/Chicago"))
    tomorrow = now().astimezone(ZoneInfo("America/Chicago")).date() + timedelta(days=1)
    assert (until.date(), until.hour, until.minute) == (tomorrow, 9, 0)
    assert result["eligible_at"] == result["snoozed_until"] and result["generation"] == 2
    with pytest.raises(DomainError, match="couldn't read"):
        run("notification.snooze", {"notification_id": identity, "until": "whenever"})
    with pytest.raises(DomainError, match="30 days"):
        run("notification.snooze", {"notification_id": identity, "until": "in 40 days"})


def test_list_shows_state_and_never_a_bundle(client):
    identity = notice()
    with session_scope() as db:
        db.add(Notification(owner_id=OWNER, title="2 reminders", category="bundle", scheduled_at=now()))
    run("notification.snooze", {"notification_id": identity, "until": "in 2 hours"})
    items = client.get("/api/v1/notifications").json()["items"]
    assert [(i["id"], i["state"]) for i in items] == [(identity, "snoozed")]


# ------------------------------------------------------------------ bundles, priorities, digests


def test_several_due_nudges_become_one_bundled_push(phone, pushes):
    ids = [notice("Pay rent"), notice("Call mum"), notice("Water plants")]
    worker.prepare_deliveries()
    worker.prepare_deliveries()  # Idempotent: no second bundle, no member pushes.
    sent = deliveries()
    assert [c for c, _ in sent] == ["bundle"]
    bundle = sent[0][1]
    assert {get(i).bundle_id for i in ids} == {bundle}
    worker.send_deliveries()
    assert len(pushes) == 1 and pushes[0]["url"] == "/?view=notifications" and pushes[0]["category"] == "bundle"
    # The push delivered every member; none is read, and each still opens its own item in-app.
    assert {notices.state(get(i)) for i in ids} == {"delivered"}
    assert get(bundle).title == "3 reminders"


def test_one_due_nudge_and_urgent_ones_push_alone(phone):
    lone = notice("Only one")
    worker.prepare_deliveries()
    assert deliveries() == [("reminder", lone)]
    urgent = notice("Leave now", importance="urgent")
    a, b = notice("A"), notice("B")
    worker.prepare_deliveries()
    categories = sorted(c for c, _ in deliveries())
    assert categories == ["bundle", "reminder", "reminder"]
    assert get(urgent).bundle_id is None and get(a).bundle_id == get(b).bundle_id


def test_same_task_due_twice_is_one_nudge(phone, pushes):
    task = run("task.create", {"title": "Stretch"})
    first, second = notice("Stretch", task_id=task["id"]), notice("Stretch", task_id=task["id"])
    worker.prepare_deliveries()
    assert len(deliveries()) == 1
    carrier = deliveries()[0][1]
    other = second if carrier == first else first
    assert get(other).bundle_id == carrier
    worker.send_deliveries()
    assert len(pushes) == 1 and notices.state(get(other)) == "delivered"


def test_below_minimum_priority_waits_for_one_digest_per_period(phone, pushes):
    run("settings.update", {"notify_min_priority": "high", "digest_cadence": "hourly"})
    low = [notice("Tidy desk"), notice("Read article", importance="low")]
    high = notice("Board meeting", importance="high")
    worker.prepare_deliveries()
    assert sorted(c for c, _ in deliveries()) == ["digest", "reminder"]
    digest = next(i for c, i in deliveries() if c == "digest")
    assert {get(i).bundle_id for i in low} == {digest} and get(high).bundle_id is None
    late = notice("Later low")
    worker.prepare_deliveries()  # The hour's digest already went: the late one waits for the next.
    assert len(deliveries()) == 2 and get(late).bundle_id is None
    worker.send_deliveries()
    assert {p["category"] for p in pushes} == {"digest", "reminder"}
    assert get(digest).title == "Digest: 2 reminders"


def test_digest_off_keeps_quiet_notices_in_app_only(phone):
    run("settings.update", {"notify_min_priority": "urgent", "digest_cadence": "off"})
    notice("A")
    notice("B")
    worker.prepare_deliveries()
    assert deliveries() == []


def test_digest_slots_follow_cadence_in_the_owners_zone():
    prefs = {"timezone": "America/Chicago", "morning_hour": 8}
    at = datetime(2026, 10, 30, 16, 20, tzinfo=UTC)  # 11:20 CDT
    assert notices.digest_slot({**prefs, "digest_cadence": "hourly"}, at) == "2026-10-30T11"
    assert notices.digest_slot({**prefs, "digest_cadence": "every_3_hours"}, at) == "2026-10-30T09"
    assert notices.digest_slot({**prefs, "digest_cadence": "daily"}, at) == "2026-10-30"
    assert notices.digest_slot({**prefs, "digest_cadence": "daily"}, at.replace(hour=12)) is None  # 07:20
    assert notices.digest_slot({**prefs, "digest_cadence": "off"}, at) is None


def test_reading_or_snoozing_a_member_leaves_the_bundle_to_the_rest(phone, pushes):
    a, b = notice("A"), notice("B")
    worker.prepare_deliveries()
    bundle = get(a).bundle_id
    run("notification.read", {"notification_id": a})
    result = run("notification.snooze", {"notification_id": b, "until": "in 1 hour"})
    assert result["bundle_id"] is None and result["state"] == "snoozed"
    # Nothing it carried is still waiting: the bundle does not go out.
    with session_scope() as db:
        assert not notices.eligible(db, db.get(Notification, bundle))
    worker.send_deliveries()
    assert pushes == []
    with session_scope() as db:  # Spent: its delivery expires instead of waiting for a day.
        assert [d.status for d in db.scalars(select(Delivery))] == ["expired"]


def test_opening_a_bundle_reads_what_it_carried(phone):
    a, b = notice("A"), notice("B")
    worker.prepare_deliveries()
    run("notification.read", {"notification_id": get(a).bundle_id})
    assert {notices.state(get(a)), notices.state(get(b))} == {"read"}


def test_morning_summary_does_not_reclaim_bundle_delivered_reminders(phone, pushes):
    a, b = notice("A"), notice("B")
    worker.prepare_deliveries()
    worker.send_deliveries()
    run("settings.update", {"morning_summary": True})
    with session_scope() as db:
        prefs = notices.preferences(db, OWNER)
        local = now().astimezone(ZoneInfo(prefs["timezone"]))
        notices.morning(db, OWNER, now() + timedelta(hours=(prefs["morning_hour"] - local.hour) % 24))
    assert {notices.state(get(a)), notices.state(get(b))} == {"delivered"}


# ------------------------------------------------------------------ recurring series


def deliver_at(clock):
    with session_scope() as db:
        scan_schedules(db, clock)
    with session_scope() as db:
        for job in db.scalars(select(Job).where(Job.kind == "reminder", Job.status != "succeeded")):
            deliver_occurrence(db, job)
            job.status = "succeeded"
    with session_scope() as db:
        return db.scalars(select(Notification.id).order_by(Notification.created_at.desc())).first()


def test_repeating_alert_on_a_task_snooze_and_complete_keep_the_series():
    task = run("task.create", {"title": "Take vitamins"})
    series = run("schedule.create", {"title": "Take vitamins", "task_id": task["id"], "when": "2030-04-01T09:00",
                                     "timezone": "America/Chicago", "recurrence": "FREQ=DAILY"})
    assert series["kind"] == "reminder"
    first = deliver_at(datetime(2030, 4, 1, 14, 1, tzinfo=UTC))
    run("notification.snooze", {"notification_id": first, "minutes": 7 * 24 * 60})
    second = deliver_at(datetime(2030, 4, 2, 14, 1, tzinfo=UTC))
    assert second != first
    # The snoozed nudge does not fire beside the new one for the same task (no duplicate)...
    old = get(first)
    assert old.dismissed_at and old.target["superseded_by"] == second
    # ...and the snooze never stopped the next occurrence.
    with session_scope() as db:
        row = db.get(Schedule, series["id"])
        assert row.status == "active" and row.next_run_at == datetime(2030, 4, 3, 14, tzinfo=UTC)
    run("notification.complete", {"notification_id": second})
    with session_scope() as db:
        row = db.get(Schedule, series["id"])
        assert row.status == "active" and row.next_run_at == datetime(2030, 4, 3, 14, tzinfo=UTC)
        assert db.get(Task, task["id"]).status == "open"
        assert db.get(Occurrence, db.get(Notification, second).occurrence_id).status == "completed"
    assert notices.state(get(second)) == "completed"
    third = deliver_at(datetime(2030, 4, 3, 14, 1, tzinfo=UTC))
    assert third not in {first, second}
    assert notices.state(get(third), datetime(2030, 4, 3, 14, 1, tzinfo=UTC)) == "pending"


def test_recurring_task_occurrences_are_separate_and_survive_snooze_and_complete():
    series = run("schedule.create", {"title": "Stretch", "when": "2030-04-01T09:00",
                                     "timezone": "America/Chicago", "recurrence": "FREQ=DAILY"})
    assert series["kind"] == "recurring_task"
    first = deliver_at(datetime(2030, 4, 1, 14, 1, tzinfo=UTC))
    run("notification.snooze", {"notification_id": first, "until": "in 3 hours"})
    second = deliver_at(datetime(2030, 4, 2, 14, 1, tzinfo=UTC))
    # Each occurrence is its own task: the snoozed one stays snoozed, the new one is pending.
    assert notices.state(get(first)) == "snoozed"
    assert notices.state(get(second), datetime(2030, 4, 2, 14, 1, tzinfo=UTC)) == "pending"
    assert get(first).task_id != get(second).task_id
    run("notification.complete", {"notification_id": first})
    with session_scope() as db:
        assert db.get(Schedule, series["id"]).status == "active"
        assert db.get(Task, db.get(Notification, first).task_id).status == "completed"
        assert db.get(Task, db.get(Notification, second).task_id).status == "open"


def test_one_time_reminder_complete_still_finishes_task_and_schedule():
    task = run("task.create", {"title": "Renew passport"})
    series = run("schedule.create", {"title": "Renew passport", "task_id": task["id"],
                                     "when": "2030-04-01T09:00", "timezone": "America/Chicago"})
    first = deliver_at(datetime(2030, 4, 1, 14, 1, tzinfo=UTC))
    run("notification.complete", {"notification_id": first})
    with session_scope() as db:
        assert db.get(Task, task["id"]).status == "completed"
        assert db.get(Schedule, series["id"]).status == "completed"


def test_priority_follows_the_task():
    assert notices.task_importance(None) == "normal"
    assert notices.task_importance(Task(priority=3, alert_urgent=False)) == "high"
    assert notices.task_importance(Task(priority=1, alert_urgent=False)) == "low"
    assert notices.task_importance(Task(priority=0, alert_urgent=True)) == "urgent"
