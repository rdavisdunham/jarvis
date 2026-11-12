"""Full-stack browser acceptance: a real Eridani, a real remotea server, a scripted agent, driven through the web UI.

Boots, on this machine only and with nothing from the owner's accounts:
- a disposable local database (migrated with alembic);
- the real API with a stand-in for the worker (tests/fixtures/full_stack_acceptance.py);
- a remotea server from a built checkout (REMOTEA_CHECKOUT, default ~/remotea/server) in its own config dir, HOME,
  ports, scratch repository and worktree root, whose `claude` is the scripted stub in tests/protocol_harness.py;
then runs apps/web/e2e/full-stack.mjs in Chromium against it.

Requires an explicit local JARVIS_DATABASE_URL, built web assets (apps/web/dist) and a built remotea checkout.
Not run in CI: the remotea repository is private.
"""
import os
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from uuid import uuid4

import httpx
from cryptography.fernet import Fernet
from sqlalchemy import create_engine
from sqlalchemy.engine import make_url

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
PAIRING = "full-stack-fixture"


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def main():
    from protocol_harness import LINK_PATH, RemoteaHost, missing, until

    original = os.environ.get("JARVIS_DATABASE_URL")
    if not original:
        raise ValueError("Set an explicit synthetic local JARVIS_DATABASE_URL; default application settings are refused")
    parsed = make_url(original)
    if parsed.drivername != "postgresql+psycopg" or parsed.host not in {"127.0.0.1", "localhost", "::1"} or parsed.query:
        raise ValueError("Acceptance requires local PostgreSQL without connection overrides")
    os.environ.setdefault("REMOTEA_CHECKOUT", str(Path.home() / "remotea" / "server"))
    reason = missing()
    if reason:
        raise ValueError(reason)
    if not (ROOT / "apps/web/dist/index.html").exists():
        raise ValueError("Build the web app first: npm --prefix apps/web run build")
    name = "jarvis_full_stack_test_" + uuid4().hex
    admin = create_engine(parsed.set(database="postgres"), isolation_level="AUTOCOMMIT")
    server = host = None
    with admin.connect() as db:
        db.exec_driver_sql(f'CREATE DATABASE "{name}"')
    tmp = tempfile.mkdtemp(prefix="eridani-full-stack-")
    log_path = Path(tmp) / "eridani.log"
    try:
        port = free_port()
        base = f"http://127.0.0.1:{port}"
        env = {key: value for key, value in os.environ.items() if not key.startswith("JARVIS_")}
        env.update(
            JARVIS_ENV_FILE="", JARVIS_DATABASE_URL=parsed.set(database=name).render_as_string(hide_password=False),
            JARVIS_OWNER_ID="full-stack-owner", JARVIS_OWNER_NAME="Rowan", JARVIS_OWNER_TOKEN=PAIRING,
            JARVIS_ORIGIN=base, JARVIS_WEB_DIST=str(ROOT / "apps/web/dist"), JARVIS_PAIRING_ENABLED="true",
            JARVIS_WORKER_ENABLED="false", JARVIS_EXTERNAL_SERVICES_ENABLED="false", JARVIS_COST_TRACKING_ENABLED="false",
            JARVIS_OPENAI_API_KEY="", JARVIS_GEMINI_API_KEY="", JARVIS_GROQ_API_KEY="",
            OPENAI_API_KEY="", GEMINI_API_KEY="", GROQ_API_KEY="",
            JARVIS_INTEGRATION_ENCRYPTION_KEY=Fernet.generate_key().decode(),
            PYTHONPATH=os.pathsep.join([str(ROOT / "apps/api"), str(ROOT)]),
        )
        subprocess.run([sys.executable, "-m", "alembic", "upgrade", "head"], cwd=ROOT, env=env, check=True)
        with open(log_path, "wb") as log:
            server = subprocess.Popen(
                [sys.executable, "-m", "uvicorn", "full_stack_acceptance:app", "--app-dir", str(ROOT / "tests/fixtures"),
                 "--host", "127.0.0.1", "--port", str(port), "--ws-max-size", "262144", "--ws-ping-interval", "20",
                 "--ws-ping-timeout", "30", "--no-access-log"], cwd=ROOT, env=env, stdout=log, stderr=log)

        def ready():
            if server.poll() is not None:
                raise RuntimeError(log_path.read_text(errors="replace")[-5000:])
            try:
                return httpx.get(base + "/health/ready", timeout=1).status_code == 200
            except httpx.HTTPError:
                return False
        until(ready, 40, "the isolated API to become ready")
        with httpx.Client(base_url=base, timeout=15) as http:
            login = http.post("/api/v1/auth/login", json={"token": PAIRING})
            login.raise_for_status()
            http.headers["X-CSRF-Token"] = login.json()["csrf"]
            enabled = http.post("/api/v1/remotea/enable", json={})
            enabled.raise_for_status()
            host = RemoteaHost(os.environ["REMOTEA_CHECKOUT"], tmp, f"ws://127.0.0.1:{port}{LINK_PATH}",
                               enabled.json()["token"])
            host.configure_mcp(enabled.json()["setup_code"], base)
            host.script()
            host.start()
            until(lambda: any(h["connected"] for h in http.get("/api/v1/remotea/status").json()["hosts"]), 45,
                  "the remotea host to connect")
        try:
            subprocess.run(["node", "e2e/full-stack.mjs"], cwd=ROOT / "apps/web", check=True, env={
                **env, "JARVIS_FULL_STACK_URL": base, "JARVIS_FULL_STACK_PAIRING": PAIRING,
                "JARVIS_FULL_STACK_TARGET": host.TARGET, "JARVIS_FULL_STACK_REPO": str(host.repo),
                "JARVIS_FULL_STACK_WORKTREES": str(host.worktrees)})
        except subprocess.CalledProcessError:
            print("--- stub\n" + "\n".join(str(e)[:400] for e in host.stub_entries()[-15:]), file=sys.stderr)
            print("--- remotea-server\n" + host.log.read_text(errors="replace")[-3000:], file=sys.stderr)
            print("--- eridani\n" + log_path.read_text(errors="replace")[-6000:], file=sys.stderr)
            raise
    finally:
        if host is not None:
            host.stop()
        if server:
            server.terminate()
            server.wait(timeout=10)
        with admin.connect() as db:
            db.exec_driver_sql(f'DROP DATABASE "{name}" WITH (FORCE)')
        admin.dispose()
        print(f"logs kept in {tmp}" if os.environ.get("JARVIS_FULL_STACK_KEEP") else "", end="")
        if not os.environ.get("JARVIS_FULL_STACK_KEEP"):
            import shutil

            shutil.rmtree(tmp, ignore_errors=True)
        time.sleep(0)


if __name__ == "__main__":
    main()
