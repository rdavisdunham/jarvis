"""Recovery only for resources whose campaign ownership can be proven."""

import json
import os
import signal
import time
from pathlib import Path

from sqlalchemy import create_engine, text

from .environment import admin_for, validate_url


def mark_trial(url, directory, job, attempt):
    owner = {"campaign": str(directory.resolve()), "job": job, "attempt": str(attempt.resolve())}
    engine = create_engine(url, connect_args={"connect_timeout": 5})
    try:
        with engine.begin() as db:
            db.execute(
                text(
                    "INSERT INTO eval_harness.metadata (key,value) VALUES ('trial_owner',CAST(:value AS jsonb))"
                ),
                {"value": json.dumps(owner)},
            )
    finally:
        engine.dispose()


def reap(attempt, directory, job):
    config_path = attempt / "input.json"
    if not config_path.exists():
        return
    config = json.loads(config_path.read_text())
    url = config["database_url"]
    parsed = validate_url(url, trial=True)
    expected = {"campaign": str(directory.resolve()), "job": job, "attempt": str(attempt.resolve())}
    if config["directory"] != expected["campaign"] or config["job"]["id"] != job:
        raise ValueError("Orphan ownership mismatch")
    admin = admin_for(url)
    try:
        with admin.connect() as db:
            if not db.scalar(
                text("SELECT 1 FROM pg_database WHERE datname=:name"), {"name": parsed.database}
            ):
                return
        engine = create_engine(url, connect_args={"connect_timeout": 5})
        try:
            with engine.connect() as db:
                actual = db.scalar(text("SELECT value FROM eval_harness.metadata WHERE key='trial_owner'"))
                if actual != expected:
                    raise ValueError("Database ownership marker mismatch; refusing cleanup")
        finally:
            engine.dispose()
        pid_file = attempt / "worker.pid"
        if pid_file.exists():
            pid = int(pid_file.read_text())
            command = Path(f"/proc/{pid}/cmdline")
            if command.exists():
                fields = command.read_bytes().split(b"\0")
                if b"scripts.app_eval.worker" not in fields or str(config_path).encode() not in fields:
                    raise ValueError("Worker PID was reused; refusing to signal it")
                os.killpg(pid, signal.SIGTERM)
                for _ in range(30):
                    if not command.exists():
                        break
                    time.sleep(0.1)
                else:
                    os.killpg(pid, signal.SIGKILL)
        with admin.connect() as db:
            db.exec_driver_sql(f'DROP DATABASE "{parsed.database}" WITH (FORCE)')
    finally:
        admin.dispose()
