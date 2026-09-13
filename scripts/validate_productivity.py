"""Migrate legacy data and exercise the productivity UI in a disposable database."""

import os
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path
from uuid import uuid4

from jarvis.config import get_settings
from jarvis.db import engine
from sqlalchemy import create_engine, text
from sqlalchemy.engine import make_url

ROOT = Path(__file__).resolve().parents[1]


def main():
    original = get_settings().database_url
    name = "jarvis_productivity_test_" + uuid4().hex
    admin = create_engine(original, isolation_level="AUTOCOMMIT")
    server = None
    with admin.connect() as db:
        db.exec_driver_sql(f'CREATE DATABASE "{name}"')
    try:
        os.environ.update(
            JARVIS_DATABASE_URL=make_url(original).set(database=name).render_as_string(hide_password=False),
            JARVIS_OWNER_TOKEN="productivity-fixture",
            JARVIS_OWNER_ID="davin",
            JARVIS_COST_TRACKING_ENABLED="false",
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

        migrate("upgrade", "0010_unified_planning")
        with engine().begin() as db:
            db.execute(
                text("""INSERT INTO tasks(id,owner_id,title,notes,status,priority,assignee,work_type,tags,revision,created_at,updated_at,archived,is_template,external,due_date)
                VALUES ('legacy-task','davin','Preserved task','Preserved context','open',0,'Eri','','[]',1,now(),now(),false,false,'{}','2030-01-15')""")
            )
            db.execute(
                text("""INSERT INTO projects(id,owner_id,name,description,archived,revision,created_at)
                VALUES ('legacy-project','davin','Legacy project','Keep me',false,1,now())""")
            )
            legacy = dict(db.execute(text("SELECT * FROM tasks WHERE id='legacy-task'")).mappings().one())
        migrate("upgrade", "head")
        with engine().connect() as db:
            row = dict(db.execute(text("SELECT * FROM tasks WHERE id='legacy-task'")).mappings().one())
            assert all(row[key] == value for key, value in legacy.items())
            assert row["assignee_id"] and row["space_id"] is None and row["planned_date"] is None
        # The migration also supports an empty additive rollback/re-upgrade.
        migrate("downgrade", "0010_unified_planning")
        migrate("upgrade", "head")
        with tempfile.TemporaryFile() as log:
            server = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "uvicorn",
                    "productivity_acceptance:app",
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
                        raise RuntimeError(log.read().decode()[-5000:])
                    time.sleep(0.1)
            else:
                raise RuntimeError("Isolated API did not become ready")
            subprocess.run(
                ["node", "e2e/productivity.mjs"],
                cwd=ROOT / "apps" / "web",
                env={**os.environ, "JARVIS_PRODUCTIVITY_TEST_URL": base},
                check=True,
            )
        print("Productivity migration, legacy preservation, desktop/mobile browser acceptance passed.")
    finally:
        if server:
            server.terminate()
            server.wait(timeout=10)
        engine().dispose()
        engine.cache_clear()
        with admin.connect() as db:
            db.exec_driver_sql(f'DROP DATABASE "{name}" WITH (FORCE)')
        admin.dispose()


if __name__ == "__main__":
    main()
