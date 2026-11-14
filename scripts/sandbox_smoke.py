"""Smoke-test a deployed sandbox copy of Eridani with a real agent (docs/CLOUD_MIGRATION.md, "A sandbox copy").

Everything on the Eridani side is the deployed app: the real build, the real worker (readiness pass, auto-start
sweep, judge, CI sweep) and Railway's proxy. This script is only the owner (it signs in and calls the same routes
the web app does) and the machine the agent runs on: a throwaway remotea server with a scratch repository and the
installed Claude Code (tests/protocol_harness.RealAgentHost), linked to the sandbox for the length of the run.

    1. Auto: a labelled task is routed, started, planned, done, verified and judged with nobody acting. The scratch
       repository has no GitHub CI, so the run then stops for approval; the script approves and the task completes.
    2. Brainstorm: a run opens as a conversation, takes one reply, and plans on "Plan it".

It refuses any origin that is not a sandbox, and spends real tokens (about two minutes of Sonnet).

    SANDBOX_ENV=~/.config/eridani-sandbox/secrets.env REMOTEA_CHECKOUT=~/remotea/server \
        python scripts/sandbox_smoke.py [auto] [brainstorm]

SANDBOX_ENV names a private file with ORIGIN and OWNER_TOKEN lines (or set SANDBOX_ORIGIN and SANDBOX_TOKEN).
"""

import json
import os
import shutil
import sys
import tempfile
import time
import uuid
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
from protocol_harness import LINK_PATH, RealAgentHost, access_token, missing, remotea_checkout, tail, until  # noqa: E402

PLAN, WORK, CI_WAIT = 420, 600, 900  # Seconds: a plan; the work, verify and verdict; three CI reads two minutes apart.
SETTLED = ("completed", "stopped", "capped", "failed")


def settings():
    values = {}
    named = os.environ.get("SANDBOX_ENV")
    if named:
        for line in Path(named).expanduser().read_text().splitlines():
            key, _, value = line.partition("=")
            values[key.strip()] = value.strip()
    origin = (os.environ.get("SANDBOX_ORIGIN") or values.get("ORIGIN") or "").rstrip("/")
    token = os.environ.get("SANDBOX_TOKEN") or values.get("OWNER_TOKEN") or ""
    if not origin.startswith("https://") or not token:
        raise SystemExit("Set SANDBOX_ENV (a file with ORIGIN and OWNER_TOKEN) or SANDBOX_ORIGIN and SANDBOX_TOKEN.")
    return origin, token


class Owner:
    """The signed-in owner, over the routes the web app uses."""

    def __init__(self, origin, token):
        self.origin = origin
        self.http = httpx.Client(base_url=origin, timeout=30)
        options = self.http.get("/api/v1/auth/options").json()
        if options.get("google") or not options.get("pairing"):
            # A sandbox signs in with its token and has no Google sign-in; production is the other way round.
            raise SystemExit(f"{origin} does not look like a sandbox (sign-in options {options}); not touching it.")
        login = self.http.post("/api/v1/auth/login", json={"token": token})
        if login.status_code != 200:
            raise SystemExit(f"Sign-in to {origin} failed ({login.status_code}).")
        self.http.headers["X-CSRF-Token"] = login.json()["csrf"]

    def call(self, method, path, body=None):
        response = None
        for attempt in range(6):  # A deploy in progress answers 502/503 for a moment; a network blip drops a request.
            try:
                response = self.http.request(method, "/api/v1" + path, **({} if body is None else {"json": body}))
            except httpx.TransportError:
                # A read is safe to repeat. A write is not, except /commands (its command_id makes the repeat a no-op).
                if attempt == 5 or (method != "GET" and path != "/commands"):
                    raise
                time.sleep(5)
                continue
            if response.status_code not in (502, 503, 504) or attempt == 5:
                break
            time.sleep(5)
        if response.status_code != 200:
            raise AssertionError(f"{method} {path}: {response.status_code} {response.text[:400]}")
        return response.json()

    def get(self, path):
        return self.call("GET", path)

    def post(self, path, body=None):
        return self.call("POST", path, body or {})

    def command(self, tool, **arguments):
        return self.post("/commands", {"command_id": str(uuid.uuid4()), "tool": tool, "arguments": arguments})["data"]

    def task(self, title, **changes):
        made = self.command("task.create", title=title)
        return self.command("task.update", task_id=made["id"], expected_revision=made["revision"], **changes) if changes else made

    def run(self, run_id):
        return self.get(f"/remotea/runs/{run_id}")


def where(run):
    return run["state"], run["state_reason"]


def said(run):
    turns = [t for t in run["turns"] if t.get("role") != "user" and t["text"].strip()]
    return max(turns, key=lambda t: t["seq"])["text"] if turns else ""


def auto(owner, host, target_id):
    owner.call("PUT", "/remotea/auto-start", {"enabled": True, "targets": [target_id], "daily_cap": 50, "host_cap": 2,
                                              "harness": "claude"})
    task = owner.task("Add a greet function", tags=["ai-assignable"], execution_target_id=target_id,
                      completion_policy="code_ready",
                      notes="In greeting.py, add greet(name) that returns 'Hello, <name>!' (so greet('Ada') is "
                            "'Hello, Ada!'). Commit the change.")
    started = time.monotonic()
    # The worker's readiness pass (a real model call) routes it; nothing here decides.
    ready = until(lambda: (r := owner.get(f"/remotea/tasks/{task['id']}/readiness"))["route"] and r, 120,
                  "the worker's readiness pass to route the task", interval=2)
    assert ready["route"] == "auto", f"routed {ready['route']}, not auto: {ready['reasons']} risks {ready['risks']}"
    routed = time.monotonic() - started

    def started_run():
        rows = owner.get(f"/remotea/runs?task_id={task['id']}")
        rows = rows.get("items", rows) if isinstance(rows, dict) else rows
        return rows[0]["id"] if rows else None
    run_id = until(started_run, 180, "auto-start to start a run", interval=2)
    until(lambda: [h for h in owner.run(run_id)["history"] if h.get("event") == "plan_approved"]
          or owner.run(run_id)["state"] in ("needs_you", *SETTLED), PLAN, "the agent's plan to go ahead", interval=3)
    run = owner.run(run_id)
    approved = [h["actor"]["kind"] for h in run["history"] if h.get("event") == "plan_approved"]
    assert approved == ["auto_start"], (where(run), approved)
    planned = time.monotonic() - started
    # The real worker verifies through the host and judges; a met verdict then waits on CI the repository does not have.
    until(lambda: owner.run(run_id)["state"] in ("awaiting_ci", "awaiting_approval", "needs_you", *SETTLED), WORK,
          "the work to be verified and judged", interval=3)
    judged = time.monotonic() - started
    run = until(lambda: (r := owner.run(run_id))["state"] != "awaiting_ci" and r, CI_WAIT,
                "the wait for CI to end (three unreadable reads, two minutes apart)", interval=5)
    files = sorted({f for _, changed in host.commits() for f in changed})
    assert "greeting.py" in files, host.commits()
    assert where(run) in (("needs_you", "ci_unavailable"), ("awaiting_approval", run["state_reason"])), where(run)
    owner.post(f"/remotea/runs/{run_id}/approve", {"expected_revision": run["state_revision"]})
    until(lambda: owner.run(run_id)["state"] == "completed", 120, "the run to complete", interval=2)
    return {"run": run_id, "routed_s": round(routed), "planned_s": round(planned), "judged_s": round(judged),
            "finished_s": round(time.monotonic() - started), "commits": host.commits(),
            "verification": (owner.run(run_id).get("verification") or {}).get("status")}


def brainstorm(owner, host, target_id):
    task = owner.task("Make the greeting friendlier", execution_target_id=target_id, completion_policy="code_ready",
                      notes="greeting.py should greet people in a way that feels warmer.")
    run_id = owner.post("/remotea/runs", {"task_id": task["id"], "harness": "claude", "model": "sonnet", "brainstorm": True})["id"]

    def waiting(turns):
        run = owner.run(run_id)
        return where(run) == ("needs_you", "brainstorm") and run["brainstorm"]["turns"] >= turns and run
    opening = said(until(lambda: waiting(1), PLAN, "the agent's opening message", interval=3))
    assert "?" in opening, opening
    owner.post(f"/remotea/runs/{run_id}/message",
               {"text": "I want it to say exactly 'Hey <name>, good to see you!'. Nothing else changes.", "mode": "queue"})
    second = until(lambda: waiting(2), PLAN, "the agent's second message", interval=3)
    owner.post(f"/remotea/runs/{run_id}/brainstorm/plan", {"expected_revision": second["state_revision"]})
    run = until(lambda: where(r := owner.run(run_id)) == ("needs_you", "plan_approval") and r, PLAN,
                "the plan to arrive for approval", interval=3)
    criteria = [c["text"] for c in run["plan"]["criteria"]]
    assert any("good to see you" in c.lower() for c in criteria), criteria
    assert not host.commits(), "no code changes before the plan is approved"
    owner.post(f"/remotea/runs/{run_id}/stop", {"mode": "kill"})
    return {"run": run_id, "opening": opening[:300], "criteria": criteria}


SCENARIOS = {"auto": auto, "brainstorm": brainstorm}


def main():
    wanted = [a for a in sys.argv[1:] if a in SCENARIOS] or list(SCENARIOS)
    origin, token = settings()
    problem = missing() or access_token()[1] or (None if shutil.which("claude") else "claude is not on PATH")
    if problem:
        raise SystemExit(problem)
    owner = Owner(origin, token)
    tmp = Path(tempfile.mkdtemp(prefix="eridani-sandbox-smoke-"))
    host, results, failed = None, {}, None
    try:
        owner.post("/remotea/enable")
        keys = owner.post("/remotea/setup-code")  # A fresh link key and MCP key for this run's throwaway host.
        host = RealAgentHost(remotea_checkout(), tmp, origin.replace("https://", "wss://") + LINK_PATH, keys["token"])
        host.configure_mcp(keys["setup_code"], origin)
        host.start()
        connected = until(lambda: [h for h in owner.get("/remotea/status")["hosts"] if h["connected"]], 60,
                          "the host to connect to the sandbox", interval=2)
        setup = owner.task("Sandbox smoke: scratch repository")
        saved = owner.post("/remotea/targets", {"host_id": connected[-1]["id"], "host_target": host.TARGET, "repo": str(host.repo),
                                                "worktree_root": str(host.worktrees), "task_id": setup["id"]})
        target_id = saved.get("id") or next(t["id"] for t in owner.get("/remotea/targets")["items"]
                                            if t.get("repo") == str(host.repo))
        for name in wanted:
            try:
                results[name] = SCENARIOS[name](owner, host, target_id)
                print(f"ok: {name}: {json.dumps(results[name])[:900]}", flush=True)
            except BaseException as exc:  # noqa: BLE001 - report every scenario, then fail
                failed = failed or exc
                print(f"FAILED: {name}: {exc}\n--- remotea-server log\n{tail(host.log, 40)}", flush=True)
    finally:
        # Leave nothing that acts by itself: auto-start off, this run's unfinished runs stopped, the host gone.
        try:
            owner.call("PUT", "/remotea/auto-start", {"enabled": False, "targets": [], "daily_cap": 5, "host_cap": 2,
                                                      "harness": "claude"})
            active = owner.get("/remotea/runs?active=true")
            for row in (active.get("items", active) if isinstance(active, dict) else active):
                owner.post(f"/remotea/runs/{row['id']}/stop", {"mode": "kill"})
        except Exception as exc:  # noqa: BLE001 - cleanup must not hide the result
            print(f"cleanup: {exc}")
        if host is not None:
            host.stop()
        shutil.rmtree(tmp, ignore_errors=True)
    if failed:
        raise SystemExit(1)
    print(f"Sandbox smoke: {len(results)} scenario(s) passed against {origin}.")


if __name__ == "__main__":
    main()
