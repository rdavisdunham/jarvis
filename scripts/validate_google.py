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
        migrate("upgrade", "head")
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
            "Google migration round-trip preserved existing notes/sessions; isolated browser acceptance passed."
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
