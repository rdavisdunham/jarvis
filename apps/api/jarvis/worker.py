"""Single supervised DBOS executor with a transactional application outbox."""

import json
import logging
import signal
import time
from datetime import timedelta
from urllib.parse import urlsplit

from dbos import DBOS, DBOSClient, Queue
from pywebpush import WebPushException, webpush
from sqlalchemy import or_, select

from . import budget
from .config import get_settings
from .db import session_scope
from .domain import advisory, delete_source, deliver_occurrence, emit, preferences, scan_schedules
from .models import (
    Conversation,
    Delivery,
    Job,
    Memory,
    Notification,
    Outbox,
    PushSubscription,
    Source,
    WorkerHealth,
    now,
)

logger = logging.getLogger("jarvis.worker")


def valid_push_endpoint(endpoint):
    try:
        url = urlsplit(endpoint)
        port = url.port
    except ValueError:
        return False
    allowed = {"fcm.googleapis.com", "updates.push.services.mozilla.com", "web.push.apple.com"}
    return (
        url.scheme == "https"
        and not url.username
        and not url.password
        and port in {None, 443}
        and (url.hostname in allowed or bool(url.hostname and url.hostname.endswith(".notify.windows.com")))
    )


@DBOS.step(retries_allowed=True, interval_seconds=2, max_attempts=10, backoff_rate=2)
def perform_job(job_id):
    with session_scope() as db:
        job = db.scalar(select(Job).where(Job.id == job_id).with_for_update())
        if not job or job.status in {"succeeded", "cancelled", "expired"}:
            return
        if job.kind == "reminder":
            job.result = deliver_occurrence(db, job)
            job.status, job.finished_at = "succeeded", now()
        elif job.kind == "extract_memory":
            # Start with exact owner-stated preferences. No invented model assertions.
            source = db.get(Source, job.payload["source_id"], with_for_update=True)
            conv = db.get(Conversation, source.conversation_id) if source else None
            if (
                source
                and not source.deleted_at
                and conv
                and conv.learning
                and not conv.private
                and preferences(db, job.owner_id)["memory_learning"]
            ):
                content = source.content.strip()
                markers = (
                    "remember that ",
                    "remember this: ",
                    "my favorite ",
                    "i prefer ",
                    "i live in ",
                    "i work at ",
                )
                if content.lower().startswith(markers):
                    db.add(
                        Memory(
                            owner_id=job.owner_id,
                            source_id=source.id,
                            content=content,
                            attribution="owner_statement",
                        )
                    )
                    emit(db, job.owner_id, "memory.changed", source.id)
            job.status, job.finished_at, job.result = "succeeded", now(), {"processed": True}
        else:
            job.status, job.finished_at, job.result = "failed", now(), {"error": "Unsupported job kind"}


@DBOS.workflow(name="jarvis_job_v1")
def run_job(job_id):
    perform_job(job_id)


def dispatch_outbox(client):
    with session_scope() as db:
        rows = db.scalars(
            select(Outbox).where(Outbox.submitted_at.is_(None)).limit(100).with_for_update(skip_locked=True)
        ).all()
        for row in rows:
            # Enqueue identity is stable across a crash before submitted_at commits.
            client.enqueue(
                {"workflow_name": "jarvis_job_v1", "queue_name": "jarvis", "workflow_id": row.job_id},
                row.job_id,
            )
            row.submitted_at = now()


def prepare_deliveries():
    with session_scope() as db:
        advisory(db, "push:prepare")
        notifications = db.scalars(
            select(Notification).where(
                Notification.created_at > now() - timedelta(days=1),
                Notification.dismissed_at.is_(None),
                Notification.read_at.is_(None),
            )
        ).all()
        for notification in notifications:
            subscriptions = db.scalars(
                select(PushSubscription).where(
                    PushSubscription.owner_id == notification.owner_id, PushSubscription.active.is_(True)
                )
            ).all()
            for subscription in subscriptions:
                if not db.scalar(
                    select(Delivery.id).where(
                        Delivery.notification_id == notification.id,
                        Delivery.subscription_id == subscription.id,
                    )
                ):
                    db.add(Delivery(notification_id=notification.id, subscription_id=subscription.id))


def send_deliveries():
    settings = get_settings()
    if not settings.vapid_private_key:
        return
    for _ in range(20):
        with session_scope() as db:
            row = db.scalar(
                select(Delivery)
                .where(
                    Delivery.status.in_(["pending", "retry"]),
                    Delivery.next_attempt_at <= now(),
                    or_(Delivery.lease_until.is_(None), Delivery.lease_until < now()),
                )
                .order_by(Delivery.next_attempt_at)
                .with_for_update(skip_locked=True)
                .limit(1)
            )
            if not row:
                return
            subscription = db.get(PushSubscription, row.subscription_id)
            notification = db.get(Notification, row.notification_id)
            if (
                not subscription.active
                or notification.read_at
                or notification.dismissed_at
                or notification.created_at < now() - timedelta(days=1)
            ):
                row.status = "expired"
                continue
            row.attempts += 1
            row.lease_until = now() + timedelta(seconds=45)
            delivery_id, sub_id = row.id, subscription.id
            payload = dict(subscription.subscription)
            detailed = preferences(db, subscription.owner_id)["detailed_notifications"]
            message = {
                "id": notification.id,
                "title": notification.title if detailed else "Jarvis reminder",
                "body": notification.body if detailed else "A reminder is waiting in your Inbox.",
                "tag": notification.id,
            }
        status, error, inactive = "submitted", None, False
        try:
            if not valid_push_endpoint(payload.get("endpoint", "")):
                raise ValueError("Unapproved push destination")
            webpush(
                subscription_info=payload,
                data=json.dumps(message),
                vapid_private_key=settings.vapid_private_key,
                vapid_claims={"sub": settings.vapid_subject},
                ttl=3600,
                timeout=15,
            )
        except WebPushException as exc:
            code = exc.response.status_code if exc.response is not None else 0
            status, error, inactive = "retry", f"HTTP_{code}", code in {404, 410}
        except Exception:  # noqa: BLE001 - isolate provider/process failures without exposing personal data
            status, error = "retry", "DELIVERY_UNAVAILABLE"
        with session_scope() as db:
            row = db.get(Delivery, delivery_id)
            row.status = "failed" if inactive or row.attempts >= 5 else status
            row.error_code, row.lease_until = error, None
            row.next_attempt_at = now() + timedelta(seconds=min(3600, 30 * 2**row.attempts))
            if inactive:
                db.get(PushSubscription, sub_id).active = False


def housekeeping():
    settings = get_settings()
    with session_scope() as db:
        prefs = preferences(db, settings.owner_id)
        days = prefs["history_days"]
        if days:
            for source in db.scalars(
                select(Source)
                .where(
                    Source.owner_id == settings.owner_id,
                    Source.explicit.is_(False),
                    Source.deleted_at.is_(None),
                    Source.created_at < now() - timedelta(days=days),
                )
                .limit(500)
            ):
                delete_source(db, source)
        # A lost interactive request is not silently replayed after a process restart.
        for job in db.scalars(
            select(Job).where(
                Job.kind == "chat", Job.status == "running", Job.created_at < now() - timedelta(minutes=10)
            )
        ):
            job.status, job.finished_at = "failed", now()
            job.result = {
                "status": "failed",
                "turn_id": job.id,
                "actions": [],
                "message": "The connection ended before a reply was saved. Check your saved tasks before repeating the request.",
            }
            budget.close(db, job.owner_id, job.id, uncertain=True)


def main():
    settings = get_settings()
    logging.basicConfig(level=logging.INFO)
    DBOS(
        config={
            "name": "jarvis",
            "system_database_url": settings.database_url,
            "application_version": "jarvis-v1",
            "run_admin_server": False,
        }
    )
    Queue("jarvis", concurrency=1, worker_concurrency=1)
    DBOS.launch()
    client = DBOSClient(system_database_url=settings.database_url)
    running = True

    def stop(*_):
        nonlocal running
        running = False

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    iteration = 0
    while running:
        try:
            with session_scope() as db:
                scan_schedules(db)
                health = db.get(WorkerHealth, "worker")
                if health:
                    health.last_scan_at = now()
                else:
                    db.add(WorkerHealth(id="worker", last_scan_at=now()))
            dispatch_outbox(client)
            prepare_deliveries()
            send_deliveries()
            if iteration % 12 == 0:
                housekeeping()
            iteration += 1
        except Exception:
            logger.exception("Worker cycle failed; will retry")
        time.sleep(settings.worker_interval_seconds)
    DBOS.destroy(workflow_completion_timeout_sec=10)


if __name__ == "__main__":
    main()
