"""Real-browser durable runs acceptance in a disposable local database.

Requires an explicit local JARVIS_DATABASE_URL, migrated backend source and built web assets.
No provider credentials or existing application records are used.
"""
import argparse
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
from sqlalchemy import create_engine
from sqlalchemy.engine import make_url

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--app-root", type=Path, default=ROOT)
    args = parser.parse_args()
    source = args.app_root.resolve()
    original = os.environ.get("JARVIS_DATABASE_URL")
    if not original:
        raise ValueError("Set an explicit synthetic local JARVIS_DATABASE_URL; default application settings are refused")
    parsed = make_url(original)
    if parsed.drivername != "postgresql+psycopg" or parsed.host not in {"127.0.0.1", "localhost", "::1"} or parsed.query:
        raise ValueError("Acceptance requires local PostgreSQL without connection overrides")
    supplied = os.environ.get("ERIDANI_EVAL_OWNED_DB")
    if supplied:
        from scripts.app_eval.environment import validate_url
        validate_url(supplied, trial=True)
        if supplied != original:
            raise ValueError("Browser fixture database mismatch")
    name = parsed.database if supplied else "jarvis_durable_test_" + uuid4().hex
    admin = create_engine(parsed.set(database="postgres"), isolation_level="AUTOCOMMIT")
    server = None
    if not supplied:
        with admin.connect() as db:
            db.exec_driver_sql(f'CREATE DATABASE "{name}"')
    try:
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        base = f"http://127.0.0.1:{port}"
        env = {key: value for key, value in os.environ.items() if not key.startswith("JARVIS_")}
        env.update(
            JARVIS_ENV_FILE="", JARVIS_DATABASE_URL=parsed.set(database=name).render_as_string(hide_password=False),
            JARVIS_OWNER_ID="durable-fixture-owner", JARVIS_OWNER_NAME="Rowan", JARVIS_OWNER_TOKEN="durable-fixture",
            JARVIS_ORIGIN=base, JARVIS_WEB_DIST=str(ROOT / "apps/web/dist"), JARVIS_PAIRING_ENABLED="true",
            JARVIS_WORKER_ENABLED="false", JARVIS_EXTERNAL_SERVICES_ENABLED="false", JARVIS_COST_TRACKING_ENABLED="false",
            JARVIS_OPENAI_API_KEY="", JARVIS_GEMINI_API_KEY="", JARVIS_GROQ_API_KEY="",
            OPENAI_API_KEY="", GEMINI_API_KEY="", GROQ_API_KEY="",
            JARVIS_INTEGRATION_ENCRYPTION_KEY=Fernet.generate_key().decode(),
            PYTHONPATH=os.pathsep.join([str(source / "apps/api"), str(ROOT)]),
        )
        subprocess.run([sys.executable, "-m", "alembic", "upgrade", "head"], cwd=source, env=env, check=True)
        with tempfile.TemporaryFile() as log:
            server = subprocess.Popen(
                [sys.executable, "-m", "uvicorn", "durable_runs_acceptance:app", "--app-dir", str(ROOT / "tests/fixtures"),
                 "--host", "127.0.0.1", "--port", str(port), "--no-access-log"], cwd=source, env=env, stdout=log, stderr=log,
            )
            for _ in range(150):
                try:
                    urllib.request.urlopen(base + "/health/ready", timeout=1).read()
                    break
                except OSError:
                    if server.poll() is not None:
                        log.seek(0)
                        raise RuntimeError(log.read().decode()[-5000:])
                    time.sleep(0.1)
            else:
                raise RuntimeError("Isolated durable runs API did not become ready")
            try:
                subprocess.run(["node", "e2e/durable-runs.mjs"], cwd=ROOT / "apps/web",
                               env={**env, "JARVIS_DURABLE_TEST_URL": base}, check=True)
            except subprocess.CalledProcessError:
                log.seek(0)
                print(log.read().decode()[-8000:], file=sys.stderr)
                raise
    finally:
        if server:
            server.terminate()
            server.wait(timeout=10)
        if not supplied:
            with admin.connect() as db:
                db.exec_driver_sql(f'DROP DATABASE "{name}" WITH (FORCE)')
        admin.dispose()


if __name__ == "__main__":
    main()
