"""Helpers for tests/test_protocol_e2e.py: the real remotea server against the real Eridani app.

Three processes and one relay, all on loopback and ephemeral ports:

- ``EridaniApp``: uvicorn serving ``jarvis.api:app`` against the test database (the one conftest.py made),
  with ``JARVIS_ENV_FILE=""`` so no local ``.env`` is read. The worker is not running: the test process plays
  it (it runs the judge job with a stubbed model), as the real worker does from its own process.
- ``LinkProxy``: a websocket relay between the two that records every frame in both directions, and can
  drop acks, cut the connection, re-send a frame, and rewrite hello/welcome to look like an older peer.
- ``RemoteaHost``: ``node remote-server/dist/remote-server/src/main.js`` from a remotea checkout, with a
  throwaway config dir, HOME, git repository and worktree root. Its one execution target launches
  ``claude`` as a stub script (``STUB``) that drives the real hook helper the way Claude Code does, and
  verifies with a fake command.

Plus the wire-fixture shapes (``shape``/``merge``/``drift``): keys and types of the frames each side sent,
recorded under tests/fixtures/protocol/<caps_version>/.
"""

import asyncio
import contextlib
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures" / "protocol"
LINK_PATH = "/api/v1/external/agents/link"
SUBPROTOCOL = "eridani.v1"
OWNER_TOKEN = "test-owner-token"  # conftest.py's JARVIS_OWNER_TOKEN; a throwaway database only.


# ---------------------------------------------------------------- where remotea is


def remotea_checkout():
    """$REMOTEA_CHECKOUT, else the usual places beside this repository."""
    named = os.environ.get("REMOTEA_CHECKOUT")
    candidates = [Path(named)] if named else [Path.home() / "remotea" / "remotea", ROOT.parent / "remotea",
                                              ROOT.parent / "remotea" / "remotea"]
    for path in candidates:
        if (path / "remote-server" / "package.json").is_file():
            return path.resolve()
    return Path(named) if named else None


def server_main(checkout):
    return checkout / "remote-server" / "dist" / "remote-server" / "src" / "main.js"


def hook_helper(checkout):
    return checkout / "resources" / "cli" / "wmux-hook.js"


def missing():
    """Why the end-to-end test cannot run here, or None."""
    node = shutil.which("node")
    if not node:
        return "node is not on PATH"
    if not shutil.which("git"):
        return "git is not on PATH"
    checkout = remotea_checkout()
    if checkout is None or not (checkout / "remote-server").is_dir():
        return "no remotea checkout (set REMOTEA_CHECKOUT)"
    if not server_main(checkout).is_file():
        return f"{checkout}/remote-server/dist is missing: run `npm --prefix remote-server ci && npm --prefix remote-server run build`"
    if not hook_helper(checkout).is_file():
        return f"{checkout}/resources/cli/wmux-hook.js is missing"
    if not (checkout / "remote-server" / "node_modules" / "node-pty").exists():
        return f"{checkout}/remote-server/node_modules is missing: run `npm --prefix remote-server ci`"
    return None


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def until(check, timeout, what, interval=0.05):
    """Poll `check` until it returns something truthy; fail with `what` after `timeout` seconds."""
    deadline = time.monotonic() + timeout
    while True:
        value = check()
        if value:
            return value
        if time.monotonic() > deadline:
            raise AssertionError(f"timed out after {timeout:g}s waiting for {what}")
        time.sleep(interval)


def tail(path, lines=60):
    try:
        return "\n".join(Path(path).read_text(errors="replace").splitlines()[-lines:])
    except OSError:
        return ""


def stop_process(proc, grace=5.0):
    if proc is None or proc.poll() is not None:
        return
    with contextlib.suppress(ProcessLookupError):
        os.killpg(proc.pid, signal.SIGTERM)
    try:
        proc.wait(grace)
    except subprocess.TimeoutExpired:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(proc.pid, signal.SIGKILL)
        proc.wait(5)


# ---------------------------------------------------------------- Eridani (uvicorn)


class EridaniApp:
    def __init__(self, tmp):
        self.port = free_port()
        self.base = f"http://127.0.0.1:{self.port}"
        self.log = Path(tmp) / "eridani.log"
        self.proc = None
        self.http = None

    def start(self):
        env = {**os.environ, "JARVIS_ENV_FILE": "", "JARVIS_ORIGIN": self.base, "JARVIS_OWNER_TOKEN": OWNER_TOKEN,
               "JARVIS_COST_TRACKING_ENABLED": "false", "JARVIS_WORKER_ENABLED": "false",
               "JARVIS_EXTERNAL_SERVICES_ENABLED": "false"}
        assert env.get("JARVIS_DATABASE_URL"), "conftest.py's test database is not set"
        with open(self.log, "wb") as out:
            # The same websocket limits as deploy.py's production command.
            self.proc = subprocess.Popen(
                [sys.executable, "-m", "uvicorn", "jarvis.api:app", "--host", "127.0.0.1", "--port", str(self.port),
                 "--ws-max-size", "262144", "--ws-ping-interval", "20", "--ws-ping-timeout", "30",
                 "--no-access-log", "--log-level", "warning"],
                cwd=ROOT, env=env, stdout=out, stderr=subprocess.STDOUT, start_new_session=True)

        def ready():
            if self.proc.poll() is not None:
                raise AssertionError(f"uvicorn exited:\n{tail(self.log)}")
            with contextlib.suppress(httpx.HTTPError):
                return httpx.get(self.base + "/health/ready", timeout=1).status_code == 200
        until(ready, 30, "Eridani to answer /health/ready")
        self.http = httpx.Client(base_url=self.base, timeout=15)
        login = self.http.post("/api/v1/auth/login", json={"token": OWNER_TOKEN})
        assert login.status_code == 200, login.text
        self.http.headers["X-CSRF-Token"] = login.json()["csrf"]
        return self

    def get(self, path, **kwargs):
        response = self.http.get(path, **kwargs)
        assert response.status_code == 200, f"GET {path}: {response.status_code} {response.text[:500]}"
        return response.json()

    def post(self, path, body=None):
        response = self.http.post(path, json=body or {})
        assert response.status_code == 200, f"POST {path}: {response.status_code} {response.text[:500]}"
        return response.json()

    def stop(self):
        if self.http is not None:
            self.http.close()
        stop_process(self.proc)


# ---------------------------------------------------------------- the recording relay


OLD_HOST_CAPS_DROPPED = {"prompt", "worktrees.scan", "digest", "context.scope", "verify.auto"}
OLD_HOST_FIELDS_DROPPED = {"caps_version", "build_sha", "build_dirty", "hook_version"}
OLD_ERIDANI_CAPS_DROPPED = {"prompt", "context.scope"}


def older_host(hello):
    """What a caps_version 1 host (before prompt frames, digests and worktrees.scan) says in its hello."""
    out = {k: v for k, v in hello.items() if k not in OLD_HOST_FIELDS_DROPPED}
    out["caps"] = [c for c in hello.get("caps", []) if c not in OLD_HOST_CAPS_DROPPED]
    return out


def older_eridani(welcome):
    """What an Eridani from before the `prompt` frame and context scopes says in its welcome."""
    return {**welcome, "caps": [c for c in welcome.get("caps", []) if c not in OLD_ERIDANI_CAPS_DROPPED]}


class LinkProxy:
    """host <-> proxy <-> Eridani. Records every frame (before any rewrite), in a background event loop."""

    def __init__(self, upstream):
        self.upstream = upstream
        self.records = []
        self.lock = threading.Lock()
        self.connections = 0
        self.drop_acks_for = set()  # Uplink frame types whose acks the host never sees.
        self.old_host = False
        self.old_eridani = False
        self.current = None
        self.port = None
        self.loop = asyncio.new_event_loop()
        self.ready = threading.Event()
        self.thread = threading.Thread(target=self.loop.run_forever, name="link-proxy", daemon=True)

    @property
    def url(self):
        return f"ws://127.0.0.1:{self.port}{LINK_PATH}"

    def start(self):
        self.thread.start()
        asyncio.run_coroutine_threadsafe(self._serve(), self.loop)
        assert self.ready.wait(10), "the link proxy did not start"
        return self

    async def _serve(self):
        from websockets.asyncio.server import serve

        self.server = await serve(self._relay, "127.0.0.1", 0, subprotocols=[SUBPROTOCOL], max_size=None,
                                  ping_interval=None)
        self.port = self.server.sockets[0].getsockname()[1]
        self.ready.set()

    def record(self, **entry):
        with self.lock:
            entry["seq"] = len(self.records)
            self.records.append(entry)

    async def _relay(self, down):
        from websockets.asyncio.client import connect
        from websockets.exceptions import InvalidStatus

        self.connections += 1
        conn = self.connections
        headers = {"Authorization": down.request.headers.get("Authorization", "")}
        try:
            up = await connect(self.upstream, additional_headers=headers, subprotocols=[SUBPROTOCOL], max_size=None,
                               user_agent_header=down.request.headers.get("User-Agent"), ping_interval=None,
                               open_timeout=10)
        except InvalidStatus as exc:
            status = exc.response.status_code
            self.record(conn=conn, dir="refused", frame={"status": status})
            await down.close({401: 4401, 403: 4403}.get(status, 1011), "refused")
            return
        sent_up, sent_down = {}, {}  # Envelope id -> frame type, per direction, to key acks by what they ack.
        self.current = (conn, down, up, sent_down)

        async def host_to_eridani():
            async for text in down:
                frame = json.loads(text)
                sent_up[frame.get("id")] = frame.get("t")
                self.record(conn=conn, dir="host", frame=frame,
                            acks=sent_down.get(frame.get("acked_id")) if frame.get("t") == "ack" else None)
                if frame.get("t") == "hello" and self.old_host:
                    text = json.dumps(older_host(frame))
                await up.send(text)

        async def eridani_to_host():
            async for text in up:
                frame = json.loads(text)
                sent_down[frame.get("id")] = frame.get("t")
                acks = sent_up.get(frame.get("acked_id")) if frame.get("t") == "ack" else None
                dropped = acks in self.drop_acks_for
                self.record(conn=conn, dir="eridani", frame=frame, acks=acks, dropped=dropped)
                if frame.get("t") == "welcome" and self.old_eridani:
                    text = json.dumps(older_eridani(frame))
                if not dropped:
                    await down.send(text)

        tasks = [asyncio.ensure_future(host_to_eridani()), asyncio.ensure_future(eridani_to_host())]
        with contextlib.suppress(Exception):
            await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        for task in tasks:
            task.cancel()
        # Pass a close on to the other side with its code (1005/1006 cannot be sent; they become 1011).
        for closed, other in ((up, down), (down, up)):
            code = closed.close_code
            if code is not None:
                code = code if code not in (1005, 1006) else 1011
                with contextlib.suppress(Exception):
                    await other.close(code, closed.close_reason or "")
        for side in (up, down):
            with contextlib.suppress(Exception):
                await side.close(1001, "proxy closing")
        if self.current and self.current[0] == conn:
            self.current = None

    def _call(self, coroutine, timeout=10):
        return asyncio.run_coroutine_threadsafe(coroutine, self.loop).result(timeout)

    def cut(self):
        """Drop the live connection on both sides, as a network blip does: remotea reconnects on its own."""
        current = self.current
        assert current, "no live link to cut"

        async def cut():
            _, down, up, _ = current
            await asyncio.gather(down.close(1001, "network blip"), up.close(1001, "network blip"),
                                 return_exceptions=True)
        self._call(cut())
        return current[0]

    def send_to_host(self, frame):
        """Send one frame to the host on the live connection (a resend, as Eridani's drain makes it)."""
        current = self.current
        assert current, "no live link"
        current[3][frame.get("id")] = frame.get("t")
        self.record(conn=current[0], dir="eridani", frame=frame, injected=True)
        self._call(current[1].send(json.dumps(frame)))

    def frames(self, direction=None, t=None, conn=None, since=0):
        with self.lock:
            return [r for r in self.records[since:] if (direction is None or r["dir"] == direction)
                    and (t is None or r["frame"].get("t") == t) and (conn is None or r["conn"] == conn)]

    def wait(self, direction, t, pred=lambda f: True, timeout=20, since=0, what=None):
        def found():
            hits = [r for r in self.frames(direction, t, since=since) if pred(r["frame"])]
            return hits[0] if hits else None
        return until(found, timeout, what or f"a {t} frame from {direction}")

    def stop(self):
        async def close():
            if self.current:
                for side in self.current[1:3]:
                    with contextlib.suppress(Exception):
                        await side.close(1001, "proxy stopping")
            self.server.close()
            with contextlib.suppress(Exception):
                await asyncio.wait_for(self.server.wait_closed(), 5)
        with contextlib.suppress(Exception):
            if self.port is not None:
                self._call(close())
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.thread.join(5)


# ---------------------------------------------------------------- the remotea host

# A stand-in for Claude Code, launched by remotea exactly as it launches `claude` for a spawn:
# `<bin> --model <m> --permission-mode auto "$ERIDANI_PROMPT"` through the login shell, in the run's worktree.
# It reports through the real hook helper (SessionStart, then UserPromptSubmit + Stop per prompt), reads
# typed input as Claude does (raw mode; bracketed paste, then Enter), and leaves on Ctrl-C.
STUB = r"""#!__NODE__
'use strict';
const { execFileSync } = require('child_process');
const crypto = require('crypto');
const fs = require('fs');
const HOOK = __HOOK__;
const LOG = __LOG__;
const args = process.argv.slice(2);
if (args[0] === '--version') { process.stdout.write('2.1.999 (Claude Code)\n'); process.exit(0); }
const log = (entry) => fs.appendFileSync(LOG, JSON.stringify({ pid: process.pid, ...entry }) + '\n');
const sid = crypto.randomUUID();
const prompt = !args.includes('--resume') && args.length && !args[args.length - 1].startsWith('-') ? args[args.length - 1] : '';
log({ start: prompt ? args.slice(0, -1) : args, cwd: process.cwd() });
function hook(event, payload) {
  try {
    const out = execFileSync(process.execPath, [HOOK, '--event', event], {
      input: JSON.stringify({ session_id: sid, hook_event_name: event, cwd: process.cwd(), ...payload }),
      timeout: 15000,
    });
    if (out.length) process.stdout.write(`[${event}] ${out.length} bytes of hook output\n`);
  } catch (err) {
    process.stdout.write(`STUB HOOK ${event} FAILED: ${err.message}\n`);
  }
}
let turns = 0;
function turn(prompt) {
  turns += 1;
  const promptId = crypto.randomUUID();
  log({ prompt, turn: turns });
  hook('UserPromptSubmit', { prompt, prompt_id: promptId });
  process.stdout.write(`STUB TURN ${turns}\n`);
  hook('Stop', { prompt_id: promptId, stop_hook_active: false,
                 last_assistant_message: `Stub turn ${turns}: made the change; the tests pass.` });
}
const bye = (code, why) => { log({ exit: code, why }); process.exit(code); };
process.on('SIGINT', () => bye(130, 'SIGINT'));
process.on('SIGHUP', () => bye(129, 'SIGHUP'));
hook('SessionStart', { source: args.includes('--resume') ? 'resume' : 'startup' });
if (prompt) turn(prompt);
if (process.stdin.isTTY) process.stdin.setRawMode(true);
let buffer = '';
process.stdin.setEncoding('utf8');
process.stdin.on('end', () => bye(0, 'stdin closed'));
process.stdin.on('data', (data) => {
  if (data.includes('\x03')) bye(130, 'Ctrl-C');
  buffer += data;
  for (;;) {
    const start = buffer.indexOf('\x1b[200~');
    const end = start < 0 ? -1 : buffer.indexOf('\x1b[201~', start);
    const enter = buffer.indexOf('\r', end < 0 ? 0 : end);
    if (start >= 0 && end >= 0 && enter >= 0) {
      const text = buffer.slice(start + 6, end);
      buffer = buffer.slice(enter + 1);
      process.stdout.write('STUB GOT INPUT\n');
      turn(text);
      continue;
    }
    if (start < 0 && enter >= 0) {
      const text = buffer.slice(0, enter).replace(/\x1b/g, '').trim();
      buffer = buffer.slice(enter + 1);
      if (text) turn(text);
      continue;
    }
    break;
  }
});
"""

VERIFY = """#!/bin/sh
echo "verify $(pwd)" >> '__LOG__'
echo "collected 3 items"
echo "3 passed in 0.01s"
exit 0
"""


def git(cwd, *args):
    env = {**os.environ, "GIT_AUTHOR_NAME": "e2e", "GIT_AUTHOR_EMAIL": "e2e@example.invalid",
           "GIT_COMMITTER_NAME": "e2e", "GIT_COMMITTER_EMAIL": "e2e@example.invalid",
           "GIT_CONFIG_GLOBAL": "/dev/null", "GIT_CONFIG_NOSYSTEM": "1"}
    return subprocess.run(["git", *args], cwd=cwd, env=env, check=True, capture_output=True, text=True).stdout


class RemoteaHost:
    """remotea-server from a checkout's dist, sandboxed: own config dir, HOME, repo and ports."""

    TARGET = "e2e"
    VERIFY_ID = "tests"

    def __init__(self, checkout, tmp, link_url, link_key):
        self.checkout = Path(checkout)
        root = Path(tmp) / "remotea"
        self.config = root / "config"
        self.home = root / "home"
        self.repo = root / "repo"
        self.worktrees = root / "worktrees"
        self.bin = root / "bin"
        self.log = root / "server.log"
        self.stub_log = root / "stub.jsonl"
        self.verify_log = root / "verify.log"
        for path in (self.config, self.home, self.repo, self.worktrees, self.bin):
            path.mkdir(parents=True, exist_ok=True)
        self.config.chmod(0o700)
        node = shutil.which("node")
        self.node = node
        self.stub = self.bin / "claude"
        self.stub.write_text(STUB.replace("__NODE__", node).replace("__HOOK__", json.dumps(str(hook_helper(self.checkout))))
                             .replace("__LOG__", json.dumps(str(self.stub_log))))
        self.stub.chmod(0o755)
        self.verify = self.bin / "fake-verify"
        self.verify.write_text(VERIFY.replace("__LOG__", str(self.verify_log)))
        self.verify.chmod(0o755)
        git(self.repo, "init", "-q", "-b", "main")
        (self.repo / "README.md").write_text("# protocol e2e\n")
        (self.repo / ".gitignore").write_text("*.pyc\n")
        git(self.repo, "add", ".")
        git(self.repo, "commit", "-q", "-m", "init")
        # An origin to fetch from, so a spawn bases its worktree on a fresh origin/main as on Davin's machine.
        git(root, "clone", "-q", "--bare", str(self.repo), str(root / "origin.git"))
        git(self.repo, "remote", "add", "origin", str(root / "origin.git"))
        git(self.repo, "fetch", "-q", "origin")
        self.port, self.hook_port = free_port(), free_port()
        self.write("config.json", {"port": self.port, "hookPort": self.hook_port, "bind": ["127.0.0.1"]})
        self.write("eridani.json", {
            "enabled": True, "url": link_url, "machine": "protocol-e2e", "collector": "off",
            "targets": [{"id": self.TARGET, "repo": str(self.repo), "worktreeRoot": str(self.worktrees), "base": "main",
                         "launch": {"claude": {"bin": str(self.stub)}},
                         "verify": [{"id": self.VERIFY_ID, "argv": [str(self.verify)], "timeout": 60}]}]})
        token = self.config / "eridani-token"
        token.write_text(link_key + "\n")
        token.chmod(0o600)
        self.proc = None

    def write(self, name, data):
        path = self.config / name
        path.write_text(json.dumps(data, indent=2) + "\n")
        path.chmod(0o600)

    def env(self):
        # Nothing from the real home: no ~/.claude, ~/.config/remotea or ~/.local/share/remotea is read or written.
        return {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "HOME": str(self.home), "SHELL": "/bin/bash",
                "LANG": "C.UTF-8", "TERM": "xterm-256color", "USER": os.environ.get("USER", "e2e"),
                "XDG_CONFIG_HOME": str(self.home / ".config"), "XDG_DATA_HOME": str(self.home / ".local" / "share"),
                "XDG_STATE_HOME": str(self.home / ".local" / "state"), "REMOTEA_CONFIG_DIR": str(self.config),
                "REMOTEA_HOOKS_HOME": str(self.home), "REMOTEA_COLLECTOR_HOME": str(self.home)}

    def start(self):
        with open(self.log, "ab") as out:
            self.proc = subprocess.Popen([self.node, str(server_main(self.checkout))], cwd=self.home, env=self.env(),
                                         stdout=out, stderr=subprocess.STDOUT, start_new_session=True)

        def listening():
            if self.proc.poll() is not None:
                raise AssertionError(f"remotea-server exited:\n{tail(self.log)}")
            return "listening on" in self.log.read_text(errors="replace")
        until(listening, 30, "remotea-server to listen")
        return self

    def stub_entries(self, key=None):
        try:
            rows = [json.loads(line) for line in self.stub_log.read_text().splitlines() if line.strip()]
        except FileNotFoundError:
            return []
        return [r for r in rows if key is None or key in r]

    def verify_runs(self):
        try:
            return len(self.verify_log.read_text().splitlines())
        except FileNotFoundError:
            return 0

    def stop(self):
        stop_process(self.proc, grace=8)
        # A SIGKILLed server leaves its terminals' processes behind; the stub leaves on SIGHUP, the rest here.
        with contextlib.suppress(Exception):
            out = subprocess.run(["pgrep", "-f", str(self.stub)], capture_output=True, text=True, check=False).stdout
            for pid in out.split():
                with contextlib.suppress(ProcessLookupError, ValueError):
                    os.kill(int(pid), signal.SIGKILL)


# ---------------------------------------------------------------- wire fixtures: keys and types


def type_name(value):
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    return "object"


def shape(value):
    """{types, keys?: {name: {required, shape}}, items?} for one JSON value."""
    out = {"types": [type_name(value)]}
    if isinstance(value, dict):
        out["keys"] = {k: {"required": True, "shape": shape(v)} for k, v in sorted(value.items())}
    elif isinstance(value, list):
        items = None
        for item in value:
            items = merge(items, shape(item))
        if items is not None:
            out["items"] = items
    return out


def merge(a, b):
    """The shape of a value that is sometimes like `a`, sometimes like `b`. A key is required only if always there."""
    if a is None:
        return b
    if b is None:
        return a
    out = {"types": sorted(set(a["types"]) | set(b["types"]))}
    if "keys" in a or "keys" in b:
        ka, kb = a.get("keys"), b.get("keys")
        keys = {}
        for name in sorted(set(ka or {}) | set(kb or {})):
            ea, eb = (ka or {}).get(name), (kb or {}).get(name)
            # A key missing from an object sample is optional; a non-object sample says nothing about keys.
            required = bool((ea or {}).get("required", ka is None) and (eb or {}).get("required", kb is None))
            keys[name] = {"required": required, "shape": merge((ea or {}).get("shape"), (eb or {}).get("shape"))}
        out["keys"] = keys
    if "items" in a or "items" in b:
        out["items"] = merge(a.get("items"), b.get("items"))
    return out


# Optional keys whose presence depends on the machine (a harness that is not installed, a degraded journal),
# not on the protocol: never reported as new.
ENVIRONMENTAL = {"launch_error", "error", "degraded_reason", "build_dirty", "reason", "migrations"}


def drift(recorded, observed, path="$"):
    """What changed between a recorded shape and this run's: renamed or missing fields, new fields, new types."""
    problems = []
    types_recorded = set(recorded["types"]) - {"null"}
    types_observed = set(observed["types"]) - {"null"}
    if types_recorded and types_observed - types_recorded:
        problems.append(f"{path}: type {sorted(types_observed)} (recorded {sorted(types_recorded)})")
    rk, ok = recorded.get("keys"), observed.get("keys")
    if rk is not None and ok is not None:
        for name, entry in rk.items():
            if entry["required"] and (name not in ok or not ok[name]["required"]):
                problems.append(f"{path}.{name}: missing (recorded as always present)")
            elif name in ok:
                problems += drift(entry["shape"], ok[name]["shape"], f"{path}.{name}")
        for name in ok:
            if name not in rk and name not in ENVIRONMENTAL:
                problems.append(f"{path}.{name}: new field (not in the recorded fixture)")
    if recorded.get("items") is not None and observed.get("items") is not None:
        problems += drift(recorded["items"], observed["items"], f"{path}[]")
    return problems


ENVELOPE_ONLY = {"chunk"}


def fixture_name(record):
    frame = record["frame"]
    kind = frame.get("t")
    if kind == "ack":
        return f"{record['dir']}.ack.{record.get('acks') or 'unknown'}"
    return f"{record['dir']}.{kind}"


def observed_shapes(records):
    shapes = {}
    for record in records:
        if record["dir"] not in ("host", "eridani") or record.get("injected") or record["frame"].get("t") in ENVELOPE_ONLY:
            continue
        name = fixture_name(record)
        shapes[name] = merge(shapes.get(name), shape(record["frame"]))
    return shapes


def check_fixtures(records, caps_version, update=False):
    """Compare this run's frames with tests/fixtures/protocol/<caps_version>/; write the ones not recorded yet.

    Returns (problems, written). With `update`, every observed frame kind is rewritten and nothing is compared.
    """
    folder = FIXTURES / str(caps_version)
    folder.mkdir(parents=True, exist_ok=True)
    problems, written = [], []
    for name, observed in sorted(observed_shapes(records).items()):
        path = folder / f"{name}.json"
        if path.exists() and not update:
            recorded = json.loads(path.read_text())
            problems += [f"{name}: {p}" for p in drift(recorded["shape"], observed)]
            continue
        direction, _, frame = name.partition(".")
        path.write_text(json.dumps({"direction": direction, "frame": frame, "shape": observed}, indent=1,
                                   sort_keys=True) + "\n")
        written.append(name)
    return problems, written
