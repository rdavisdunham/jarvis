import os
import subprocess
import sys
import time
from datetime import timedelta
from pathlib import Path
from uuid import uuid4

from jarvis.db import session_scope
from jarvis.domain import execute, scan_schedules
from jarvis.models import Notification, Outbox, now
from sqlalchemy import func, select


def wait_for(predicate, seconds=30):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.2)
    raise AssertionError("Worker did not reach expected state before deadline")


def notification_count():
    with session_scope() as db:
        return db.scalar(select(func.count(Notification.id)))


def test_accepted_reminder_survives_kill_and_outbox_replay(test_database):
    root = Path(__file__).resolve().parents[1]
    log_path = root / ".runtime" / "test-worker.log"
    env = {
        **os.environ,
        "JARVIS_DATABASE_URL": test_database,
        "JARVIS_WORKER_INTERVAL_SECONDS": "1",
        "JARVIS_VAPID_PRIVATE_KEY": "",
        "JARVIS_OPENAI_API_KEY": "",
        "JARVIS_GROQ_API_KEY": "",
    }

    def start():
        return subprocess.Popen(
            [sys.executable, "-m", "jarvis.worker"], cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT
        )

    with session_scope() as db:
        execute(
            db,
            "davin",
            str(uuid4()),
            "schedule.create",
            {"title": "Survive restart", "when": (now() - timedelta(minutes=1)).isoformat()},
        )
        assert scan_schedules(db) == 1
    with log_path.open("w") as log:
        worker = start()
        try:
            wait_for(lambda: notification_count() == 1)
            worker.kill()
            worker.wait(timeout=10)
            # Simulate a crash after DBOS enqueue but before marking the outbox row submitted.
            with session_scope() as db:
                for row in db.scalars(select(Outbox)):
                    row.submitted_at = None
            worker = start()

            def replayed():
                with session_scope() as db:
                    return db.scalar(select(Outbox)).submitted_at is not None

            wait_for(replayed)
            time.sleep(1)
            assert notification_count() == 1
        finally:
            if worker.poll() is None:
                worker.terminate()
                try:
                    worker.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    worker.kill()
                    worker.wait()
