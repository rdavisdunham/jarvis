"""Single supervised DBOS executor with a transactional application outbox."""

import asyncio
import json
import logging
from datetime import timedelta
from urllib.parse import urlsplit

from dbos import DBOS, DBOSClient, Queue
from pywebpush import WebPushException, webpush
from sqlalchemy import or_, select, text

from . import budget
from .config import get_settings
from .db import session_scope
from .domain import advisory, delete_source, deliver_occurrence, preferences, scan_schedules
from .models import (
    Delivery,
    Job,
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
        job = db.get(Job, job_id)
        kind = job.kind if job else None
        learning = kind in {"extract_memory", "embed_memory"}
    if kind in {"agent_intake", "agent_action"}:
        from . import work_intake, work_runner
        asyncio.run((work_intake if kind == "agent_intake" else work_runner).run(job_id))
        return
    if kind in {"linear_write", "linear_sync"}:
        from .linear_sync import process_sync, process_write

        return (process_write if kind == "linear_write" else process_sync)(job_id)
    if kind == "google_write":
        from .google_writes import process_write

        return process_write(job_id)
    if kind == "google_sync":
        from .google_calendar import process

        return process(job_id)
    if kind == "embed_note":
        from .notes import index_note

        return index_note(job_id)
    if kind == "review_memory":
        from .memory_review import process

        return process(job_id)
    if kind in {"assess_field", "review_routing"}:
        from .routing import perform
        return perform(job_id,kind)
    if learning:
        from .memory_learning import process

        return process(job_id)
    with session_scope() as db:
        job = db.scalar(select(Job).where(Job.id == job_id).with_for_update())
        if not job or job.status in {"succeeded", "cancelled", "expired"}:
            return
        if job.kind == "reminder":
            job.result = deliver_occurrence(db, job)
            job.status, job.finished_at = "succeeded", now()
        else:
            job.status, job.finished_at, job.result = "failed", now(), {"error": "Unsupported job kind"}


@DBOS.workflow(name="jarvis_job_v1")
def run_job(job_id):
    perform_job(job_id)


def dispatch_outbox(client):
    with session_scope() as db:
        rows = db.scalars(
            select(Outbox).join(Job).where(Outbox.submitted_at.is_(None)).order_by(Job.created_at).limit(1000).with_for_update(skip_locked=True)
        ).all()
        for row in rows:
            job = db.get(Job, row.job_id)
            if job.kind in {"agent_intake", "agent_action"}:
                from .agent_work import eligible
                if not eligible(db, job):
                    if job.status not in {"queued", "running", "dispatched"}:
                        row.submitted_at = now()
                    continue
                job.status = "dispatched"
                db.flush()
            # Enqueue identity is stable across a crash before submitted_at commits.
            client.enqueue(
                {
                    "workflow_name": "jarvis_job_v1",
                    "queue_name": "jarvis-intake" if job.kind == "agent_intake"
                    else "jarvis-agent" if job.kind == "agent_action"
                    else "jarvis-linear"
                    if db.get(Job, row.job_id).kind in {"linear_sync", "linear_write"}
                    else "jarvis-google"
                    if db.get(Job, row.job_id).kind in {"google_sync", "google_write"}
                    else "jarvis-memory"
                    if db.get(Job, row.job_id).kind
                    in {"extract_memory", "embed_memory", "review_memory", "embed_note", "assess_field", "review_routing"}
                    else "jarvis",
                    "workflow_id": row.job_id + (":"+str(job.payload["dispatch_revision"]) if job.payload.get("dispatch_revision") else ""),
                },
                row.job_id,
            )
            row.submitted_at = now()


def notification_url(notification):
    from urllib.parse import urlencode
    target=notification.target or {}
    if notification.task_id:return "/?"+urlencode({"view":"tasks","tab":"all","record":"task:"+notification.task_id,"workspace":"personal"})
    if target.get("view")=="activity":return "/?view=tasks&activity=1"
    if target.get("view")=="tasks":return "/?view=tasks&tab=today"
    return "/?view=notifications"


def prepare_deliveries():
    with session_scope() as db:
        advisory(db, "push:prepare")
        notifications = db.scalars(
            select(Notification).where(
                or_(Notification.created_at > now() - timedelta(days=1), Notification.eligible_at > now()-timedelta(days=1)),
                Notification.dismissed_at.is_(None),
                Notification.completed_at.is_(None),
                Notification.read_at.is_(None),
            )
        ).all()
        from .notices import eligible
        for notification in notifications:
            if not eligible(db,notification):continue
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
                        Delivery.generation == notification.generation,
                    )
                ):
                    db.add(Delivery(notification_id=notification.id, subscription_id=subscription.id, generation=notification.generation))


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
            from .notices import eligible, quiet_until
            if notification.category == "work_result" or not subscription.active or row.generation != notification.generation or notification.read_at or notification.dismissed_at or notification.completed_at:
                row.status="expired"
                continue
            if not eligible(db,notification):
                row.next_attempt_at=max(now()+timedelta(minutes=1),quiet_until(preferences(db,notification.owner_id),now(),notification.importance=="urgent"),notification.eligible_at or now())
                continue
            row.attempts += 1
            row.lease_until = now() + timedelta(seconds=45)
            delivery_id, sub_id = row.id, subscription.id
            payload = dict(subscription.subscription)
            detailed = preferences(db, subscription.owner_id)["detailed_notifications"]
            message = {
                "id": notification.id,
                "title": notification.title if detailed else "Eridani update",
                "body": notification.body if detailed else "An update is waiting in Eridani.",
                "url": notification_url(notification),
                "category": notification.category,
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
    from .models import OwnerSettings

    with session_scope() as db:
        owners = (
            set(db.scalars(select(OwnerSettings.owner_id)))
            | set(db.scalars(select(Source.owner_id).distinct()))
            | {get_settings().owner_id}
        )
        for owner in owners:
            housekeeping_owner(db, owner)
        # A lost interactive request is not silently replayed after a process restart.
        for job in db.scalars(
            select(Job).where(
                Job.kind == "chat", Job.status == "running", Job.created_at < now() - timedelta(minutes=10)
            )
        ):
            reservation = db.get(
                __import__("jarvis.models", fromlist=["BudgetReservation"]).BudgetReservation, job.id
            )
            if reservation and reservation.state == "reserved" and not budget.stale(reservation):
                continue
            job.status, job.finished_at = "failed", now()
            job.result = {
                "status": "failed",
                "turn_id": job.id,
                "actions": [],
                "message": "The connection ended before a reply was saved. Check your saved tasks before repeating the request.",
            }
            budget.close(db, job.owner_id, job.id, uncertain=True)


def housekeeping_owner(db, owner):
    prefs = preferences(db, owner)
    days = prefs["history_days"]
    if days:
        for source in db.scalars(
            select(Source)
            .where(
                Source.owner_id == owner,
                Source.explicit.is_(False),
                Source.deleted_at.is_(None),
                Source.created_at < now() - timedelta(days=days),
            )
            .limit(500)
        ):
            delete_source(db, source)
    budget.expire_abandoned(db, owner)
    # Finished DBOS invocations are never replayed under the same workflow ID.
    # Resume intentionally deferred memory work as a new durable job once funds allow.
    if budget.summary(db, owner)["budget_mode"] not in {"defer_optional", "paused"}:
        from .domain import enqueue_job

        for deferred in db.scalars(
            select(Job)
            .where(
                Job.owner_id == owner,
                Job.status == "deferred_budget",
                Job.kind.in_(["extract_memory", "embed_memory", "embed_note"]),
            )
            .with_for_update()
            .limit(20)
        ):
            resumed = enqueue_job(
                db,
                deferred.owner_id,
                deferred.kind,
                {k: v for k, v in deferred.payload.items() if k != "attempts"},
            )
            deferred.status, deferred.result = "cancelled", {"resumed_as": resumed.id}


def main():
    from .deploy import stop_event, validate_deployment, worker_lease

    settings = get_settings()
    validate_deployment(settings)
    stop = stop_event()
    logging.basicConfig(level=logging.INFO)
    if not settings.worker_enabled:
        logger.info("Worker is paused; no jobs or external effects will run.")
        stop.wait()
        return
    with worker_lease(stop) as lease:
        if lease is not None:
            run_supervisor(stop, lease)


def run_supervisor(stop, lease):
    settings = get_settings()
    logging.basicConfig(level=logging.INFO)
    DBOS(
        config={
            "name": "jarvis",
            "system_database_url": settings.database_url,
            "application_version": "jarvis-v1",
            "run_admin_server": False,
            "sys_db_pool_size": settings.dbos_pool_size,
        }
    )
    Queue("jarvis", concurrency=1, worker_concurrency=1)
    Queue("jarvis-memory", concurrency=2, worker_concurrency=2)
    Queue("jarvis-google", concurrency=1, worker_concurrency=1)
    Queue("jarvis-linear", concurrency=1, worker_concurrency=1)
    Queue("jarvis-intake", concurrency=4, worker_concurrency=4)
    Queue("jarvis-agent", concurrency=settings.agent_parallelism, worker_concurrency=settings.agent_parallelism)
    DBOS.launch()
    client = DBOSClient(
        system_database_url=settings.database_url, system_database_pool_size=settings.dbos_client_pool_size
    )
    lease_pid = lease.scalar(text("SELECT pg_backend_pid()"))
    lease.commit()
    iteration = 0
    try:
        while not stop.is_set():
            # A lost lease must stop the supervisor, not silently reconnect without its lock.
            if lease.invalidated or lease.scalar(text("SELECT pg_backend_pid()")) != lease_pid:
                raise RuntimeError("Worker lease was lost.")
            lease.commit()
            try:
                with session_scope() as db:
                    from .device_bridge import cleanup
                    from .work_intake import flush_voice
                    flush_voice(db)
                    cleanup(db)
                    scan_schedules(db)
                    from .notices import scan
                    scan(db)
                    from .memory_learning import queue_backfill

                    queue_backfill(db)
                    from .memory_review import queue_due_reviews

                    queue_due_reviews(db)
                    from .routing import queue_due
                    queue_due(db)
                    from .google_calendar import queue_sync
                    from .models import GoogleIdentity, LinearConnection

                    for account in db.scalars(select(GoogleIdentity.owner_id)):
                        queue_sync(db, account)
                    from .linear_sync import queue_sync as linear_queue_sync

                    for account in db.scalars(select(LinearConnection.owner_id)):
                        linear_queue_sync(db, account)
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
            stop.wait(settings.worker_interval_seconds)
    finally:
        client.destroy()
        DBOS.destroy(workflow_completion_timeout_sec=10)


if __name__ == "__main__":
    main()
