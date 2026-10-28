"""Check self-RSVP UI using a disposable local database and synthetic Google transport."""

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
from sqlalchemy import create_engine
from sqlalchemy.engine import make_url

ROOT = Path(__file__).resolve().parents[1]


def main():
    original = get_settings().database_url
    supplied = os.environ.get("ERIDANI_EVAL_OWNED_DB")
    if supplied:
        from scripts.app_eval.environment import validate_url
        validate_url(supplied, trial=True)
        if supplied != original:
            raise ValueError("Browser fixture database mismatch")
    name = make_url(supplied).database if supplied else "jarvis_rsvp_test_" + uuid4().hex
    admin = create_engine(original, isolation_level="AUTOCOMMIT")
    server = None
    if not supplied:
        with admin.connect() as db:
            db.exec_driver_sql(f'CREATE DATABASE "{name}"')
    try:
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        base = f"http://127.0.0.1:{port}"
        os.environ.update(
            JARVIS_DATABASE_URL=make_url(original).set(database=name).render_as_string(hide_password=False),
            JARVIS_OWNER_TOKEN="google-fixture",
            JARVIS_OWNER_ID="davin",
            JARVIS_GOOGLE_CLIENT_ID="fixture.apps.googleusercontent.com",
            JARVIS_GOOGLE_CLIENT_SECRET="fixture-secret",
            JARVIS_ORIGIN=base,
            JARVIS_OPENAI_API_KEY="synthetic-openai",
            JARVIS_GEMINI_API_KEY="synthetic-gemini",
            JARVIS_GROQ_API_KEY="synthetic-groq",
            JARVIS_COST_TRACKING_ENABLED="false",
            JARVIS_INTEGRATION_ENCRYPTION_KEY=Fernet.generate_key().decode(),
        )
        get_settings.cache_clear()
        engine.cache_clear()
        subprocess.run([sys.executable, "-m", "alembic", "upgrade", "head"], cwd=ROOT, check=True)
        from jarvis.google_auth import seal
        from jarvis.models import GoogleIdentity, now
        from jarvis.db import session_scope
        with session_scope() as db:
            db.add(GoogleIdentity(owner_id="davin",subject="fixture-subject",email="owner@example.test",
                                  calendar_enabled=True,credentials=seal({"refresh_token":"fixture-refresh"}),
                                  status="pending",next_sync_at=now()))
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
                        log.seek(0)
                        raise RuntimeError(log.read().decode()[-3000:])
                    time.sleep(0.1)
            else:
                raise RuntimeError("Isolated API did not become ready")
            try:
                subprocess.run(["node", "e2e/google-rsvp.mjs"], cwd=ROOT / "apps" / "web",
                               env={**os.environ, "JARVIS_GOOGLE_TEST_URL":base}, check=True)
            except subprocess.CalledProcessError:
                log.seek(0)
                print(log.read().decode()[-5000:], file=sys.stderr)
                raise
    finally:
        if server:
            server.terminate()
            server.wait(timeout=10)
        engine().dispose()
        engine.cache_clear()
        if not supplied:
            with admin.connect() as db:
                db.exec_driver_sql(f'DROP DATABASE "{name}" WITH (FORCE)')
        admin.dispose()


if __name__ == "__main__":
    main()
