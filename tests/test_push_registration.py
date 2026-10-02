from datetime import timedelta

from jarvis import worker
from jarvis.config import get_settings
from jarvis.db import session_scope
from jarvis.models import Delivery, Notification, PushSubscription, now
from sqlalchemy import select

ENDPOINT = "https://fcm.googleapis.com/fcm/send/device"
KEYS = {"p256dh": "key", "auth": "secret"}


def test_new_device_gets_current_alerts_not_a_backlog(client, monkeypatch):
    monkeypatch.setattr("jarvis.notices.eligible", lambda db, n: True)
    owner = get_settings().owner_id
    with session_scope() as db:
        db.add(Notification(owner_id=owner, title="Old", scheduled_at=now(), created_at=now() - timedelta(hours=3)))
    assert client.post("/api/v1/push", json={"endpoint": ENDPOINT, "keys": KEYS}).status_code == 200
    with session_scope() as db:
        db.add(Notification(owner_id=owner, title="New", scheduled_at=now()))
    worker.prepare_deliveries()
    with session_scope() as db:
        titles = {db.get(Notification, d.notification_id).title for d in db.scalars(select(Delivery))}
        assert titles == {"New"}


def test_reregistering_an_active_device_keeps_its_start(client):
    body = {"endpoint": ENDPOINT, "keys": KEYS}
    client.post("/api/v1/push", json=body)
    with session_scope() as db:
        first = db.scalar(select(PushSubscription)).subscription["registered_at"]
    client.post("/api/v1/push", json=body)
    with session_scope() as db:
        assert db.scalar(select(PushSubscription)).subscription["registered_at"] == first


def test_vapid_mismatch_deactivates_and_payload_omits_bookkeeping(monkeypatch):
    from pywebpush import WebPushException

    sent = []

    class Response:
        status_code = 403

    def reject(**kwargs):
        sent.append(kwargs["subscription_info"])
        raise WebPushException("forbidden", response=Response())

    monkeypatch.setattr(get_settings(), "vapid_private_key", "synthetic")
    monkeypatch.setattr(worker, "webpush", reject)
    monkeypatch.setattr("jarvis.notices.eligible", lambda db, n: True)
    owner = get_settings().owner_id
    with session_scope() as db:
        note = Notification(owner_id=owner, title="Due", scheduled_at=now())
        sub = PushSubscription(
            id="sub", owner_id=owner, device_id="d",
            subscription={"endpoint": ENDPOINT, "keys": KEYS, "registered_at": now().isoformat()},
        )
        db.add_all([note, sub])
        db.flush()
        db.add(Delivery(notification_id=note.id, subscription_id=sub.id))
    worker.send_deliveries()
    assert "registered_at" not in sent[0]
    with session_scope() as db:
        assert db.get(PushSubscription, "sub").active is False
