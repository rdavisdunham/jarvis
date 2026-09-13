"""Run Google browser acceptance in a disposable database with synthetic provider responses."""

import os
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path
from uuid import uuid4

from cryptography.fernet import Fernet
from jarvis.config import get_settings
from jarvis.db import engine
from sqlalchemy import create_engine, text
from sqlalchemy.engine import make_url

ROOT = Path(__file__).resolve().parents[1]


def main():
    original = get_settings().database_url
    name = "jarvis_google_test_" + uuid4().hex
    admin = create_engine(original, isolation_level="AUTOCOMMIT")
    server = None
    with admin.connect() as db:
        db.exec_driver_sql(f'CREATE DATABASE "{name}"')
    try:
        os.environ.update(
            JARVIS_DATABASE_URL=make_url(original).set(database=name).render_as_string(hide_password=False),
            JARVIS_OWNER_TOKEN="google-fixture",
            JARVIS_OWNER_ID="davin",
            JARVIS_COST_TRACKING_ENABLED="false",
            JARVIS_GOOGLE_CLIENT_ID="fixture.apps.googleusercontent.com",
            JARVIS_GOOGLE_CLIENT_SECRET="fixture-secret",
            JARVIS_INTEGRATION_ENCRYPTION_KEY=Fernet.generate_key().decode(),
        )
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        base = f"http://127.0.0.1:{port}"
        os.environ["JARVIS_ORIGIN"] = base
        get_settings.cache_clear()
        engine.cache_clear()

        def migrate(direction, target):
            subprocess.run([sys.executable, "-m", "alembic", direction, target], cwd=ROOT, check=True)

        migrate("upgrade", "0007_notes_context")
        with engine().begin() as db:
            db.execute(
                text("""INSERT INTO auth_sessions (token_hash, owner_id, device_id, csrf, expires_at)
                VALUES ('preserved', 'davin', 'fixture-device', 'fixture-csrf', now()+interval '1 day')""")
            )
            db.execute(
                text("""INSERT INTO notes (id,owner_id,title,content)
                VALUES ('preserved-note','davin','Existing note','Preserve this through migration')""")
            )
        with engine().begin() as db:
            db.execute(
                text("""
                INSERT INTO schedules (id,owner_id,title,timezone,anchor_at,next_run_at,kind,status,revision,original_words,created_at,recurrence,completed_at)
                VALUES ('migration-once','migration-owner','Legacy completed alert','America/Chicago','2030-01-01T15:00Z',null,'reminder','completed',1,'',now(),null,'2030-01-01T16:00Z'),
                       ('migration-routine','migration-owner','Legacy recurring alert','America/Chicago','2030-01-01T15:00Z','2030-01-02T15:00Z','reminder','active',1,'',now(),'FREQ=DAILY',null)
            """)
            )
            db.execute(
                text("""
                INSERT INTO schedule_occurrences (id,schedule_id,revision,scheduled_at,status)
                VALUES ('migration-occurrence','migration-routine',1,'2030-01-01T15:00Z','completed')
            """)
            )
            db.execute(
                text("""
                INSERT INTO notifications (id,owner_id,occurrence_id,title,body,scheduled_at,created_at,completed_at)
                VALUES ('migration-notice','migration-owner','migration-occurrence','Legacy recurring alert','','2030-01-01T15:00Z',now(),'2030-01-01T16:00Z')
            """)
            )
        migrate("upgrade", "head")
        with engine().connect() as db:
            assert (
                db.execute(text("SELECT count(*) FROM tasks WHERE owner_id='migration-owner'")).scalar() == 3
            )
            assert (
                db.execute(
                    text(
                        "SELECT t.is_template FROM tasks t JOIN schedules s ON s.task_id=t.id WHERE s.id='migration-routine'"
                    )
                ).scalar()
                is True
            )
            assert (
                db.execute(
                    text(
                        "SELECT t.status FROM tasks t JOIN notifications n ON n.task_id=t.id WHERE n.id='migration-notice'"
                    )
                ).scalar()
                == "completed"
            )
            assert (
                db.execute(
                    text(
                        "SELECT t.due_date FROM tasks t JOIN schedules s ON s.task_id=t.id WHERE s.id='migration-once'"
                    )
                ).scalar()
                is None
            )
        with engine().connect() as db:
            assert (
                db.execute(
                    text("SELECT auth_method FROM auth_sessions WHERE token_hash='preserved'")
                ).scalar()
                == "pairing"
            )
        migrate("downgrade", "0007_notes_context")
        migrate("upgrade", "head")
        with engine().connect() as db:
            assert (
                db.execute(text("SELECT content FROM notes WHERE id='preserved-note'")).scalar()
                == "Preserve this through migration"
            )
        with engine().connect() as db:
            assert (
                db.execute(text("SELECT count(*) FROM tasks WHERE owner_id='migration-owner'")).scalar() == 3
            )
            assert (
                db.execute(
                    text(
                        "SELECT t.is_template FROM tasks t JOIN schedules s ON s.task_id=t.id WHERE s.id='migration-routine'"
                    )
                ).scalar()
                is True
            )
        # Preserve an existing read-only Google connection through the additive write upgrade.
        migrate("downgrade", "0008_google_calendar")
        with engine().begin() as db:
            db.execute(
                text("""INSERT INTO google_identities (owner_id,subject,email,credentials,calendar_enabled,status)
                VALUES ('migration-owner','migration-subject','migration@example.test','preserved-encrypted-token',true,'ready')""")
            )
            db.execute(
                text("""INSERT INTO google_calendars (id,owner_id,provider_id,title,selected)
                VALUES ('migration-calendar','migration-owner','fixture-calendar','Preserved calendar',true)""")
            )
        migrate("upgrade", "head")
        with engine().connect() as db:
            assert (
                db.execute(
                    text(
                        "SELECT calendar_write_enabled FROM google_identities WHERE owner_id='migration-owner'"
                    )
                ).scalar()
                is False
            )
            assert (
                db.execute(
                    text("SELECT access_role FROM google_calendars WHERE id='migration-calendar'")
                ).scalar()
                == "reader"
            )
        migrate("downgrade", "0008_google_calendar")
        migrate("upgrade", "head")
        with engine().begin() as db:
            assert (
                db.execute(
                    text("SELECT credentials FROM google_identities WHERE owner_id='migration-owner'")
                ).scalar()
                == "preserved-encrypted-token"
            )
            assert (
                db.execute(
                    text("SELECT selected FROM google_calendars WHERE id='migration-calendar'")
                ).scalar()
                is True
            )
            db.execute(text("DELETE FROM google_calendars WHERE id='migration-calendar'"))
            db.execute(text("DELETE FROM google_identities WHERE owner_id='migration-owner'"))

        from jarvis.google_auth import seal
        from jarvis.models import GoogleIdentity, now

        with __import__("jarvis.db", fromlist=["session_scope"]).session_scope() as db:
            db.add(
                GoogleIdentity(
                    owner_id="davin",
                    subject="fixture-subject",
                    email="owner@example.test",
                    calendar_enabled=True,
                    credentials=seal({"refresh_token": "fixture-refresh"}),
                    status="pending",
                    next_sync_at=now(),
                )
            )
        with tempfile.TemporaryFile() as log:
            server = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "uvicorn",
                    "google_acceptance:app",
                    "--app-dir",
                    str(ROOT / "tests" / "fixtures"),
                    "--host",
                    "127.0.0.1",
                    "--port",
                    str(port),
                    "--no-access-log",
                ],
                cwd=ROOT,
                stdout=log,
                stderr=log,
            )
            for _ in range(100):
                try:
                    urllib.request.urlopen(base + "/health/ready", timeout=1).read()
                    break
                except OSError:
                    if server.poll() is not None:
                        raise RuntimeError("Isolated API failed to start")
                    time.sleep(0.1)
            else:
                raise RuntimeError("Isolated API was not ready")
            subprocess.run(
                ["node", "e2e/google.mjs"],
                cwd=ROOT / "apps" / "web",
                env={**os.environ, "JARVIS_GOOGLE_TEST_URL": base},
                check=True,
            )
        print(
            "Migration round-trip preserved notes/sessions and legacy reminder completion/task links; isolated Google/Linear/planning browser acceptance passed."
        )
    finally:
        if server:
            server.terminate()
            server.wait(timeout=15)
        engine().dispose()
        engine.cache_clear()
        with admin.connect() as db:
            db.exec_driver_sql(f'DROP DATABASE "{name}" WITH (FORCE)')
        admin.dispose()


if __name__ == "__main__":
    main()
