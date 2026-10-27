#!/usr/bin/env python3
"""Report local AI agent sessions to the Eridani session ledger.

Reads, never writes, the session stores of Claude Code, Claude Desktop, Codex CLI and the
ChatGPT desktop app (Codex), on the WSL side and the Windows side of one machine. Sends
metadata plus a short, redacted excerpt per session; full transcripts stay on this machine.

Standard library only, so a systemd timer can run it with the system python3.

    collector.py --dry-run            # table of the last 7 days, nothing sent
    collector.py                      # send changed sessions (needs a sessions:write bot key)
"""

import argparse
import glob
import hashlib
import json
import os
import re
import shlex
import socket
import sqlite3
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import deque
from datetime import UTC, datetime, timedelta
from pathlib import Path

API_URL = os.environ.get("ERIDANI_API_URL", "https://app.eridani.app/api/v1/external")
CONFIG = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")) / "eridani-collector"
STATE = Path(os.environ.get("XDG_STATE_HOME", Path.home() / ".local/state")) / "eridani-collector/state.json"
SETTLED = timedelta(minutes=10)
LIMITS = {"title": 500, "first_prompt": 4000, "last_prompt": 4000, "excerpt": 8000, "cwd": 1000, "repo": 300, "branch": 300}
SECRETS = re.compile(
    r"(sk-(?:proj-|ant-)?[A-Za-z0-9_-]{20,}|gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,}"
    r"|AKIA[0-9A-Z]{16}|xox[abpr]-[A-Za-z0-9-]{10,}|eri_bot_[A-Za-z0-9_-]{20,}|lin_api_[A-Za-z0-9]{20,}"
    r"|(?i:bearer)\s+[A-Za-z0-9._~+/=-]{20,})"
)
NOISE = ("<local-command", "<system-reminder>", "Caveat: The messages below", "[Request interrupted")
# Voice-delegated ChatGPT desktop prompts arrive wrapped in these tags.
WRAPPERS = re.compile(r"</?(?:realtime_delegation|input)>")


def redact(text):
    return SECRETS.sub("[redacted]", text or "")


def clip(text, limit):
    text = redact(text).strip()
    return text if len(text) <= limit else text[: limit - 1] + "…"


def iso(value):
    """Epoch seconds or milliseconds, or an ISO string, as an aware ISO timestamp."""
    if value in (None, ""):
        return None
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(value / 1000 if value > 1e11 else value, UTC).isoformat()
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).astimezone(UTC).isoformat()
    except ValueError:
        return None


def windows_path(value):
    """Codex Desktop stores \\\\?\\ extended paths; keep one ordinary spelling."""
    path = (value or "").replace("\\", "/")
    if path.startswith("//?/UNC/"):
        path = "//" + path[8:]
    elif path.startswith("//?/"):
        path = path[4:]
    return path


def environment_of(cwd, default):
    path = windows_path(cwd).lower()
    if path.startswith(("//wsl.localhost/", "//wsl$/")):
        return "wsl"
    if re.match(r"^[a-z]:/", path):
        return "windows"
    return default


def local_path(cwd):
    """Where this WSL process can reach a session folder, if anywhere."""
    path = windows_path(cwd)
    unc = re.match(r"^//(?:wsl\.localhost|wsl\$)/[^/]+(/.*)$", path, re.I)
    if unc:
        return unc.group(1)
    return path if path.startswith("/") else None


_repos = {}


def repo_of(cwd, origin=""):
    if origin:
        return re.sub(r"\.git$", "", origin.rstrip("/").split("/")[-1].split(":")[-1])
    path = local_path(cwd)
    if not path or not os.path.isdir(path):
        return ""
    if path not in _repos:
        try:
            top = subprocess.run(
                ["git", "-C", path, "rev-parse", "--show-toplevel"], capture_output=True, text=True, timeout=5
            ).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            top = ""
        _repos[path] = os.path.basename(top)
    return _repos[path]


class Transcript:
    """Keeps what the ledger needs from one session, not the whole conversation."""

    def __init__(self):
        self.first_prompt, self.recent, self.turns, self.reply = "", deque(maxlen=3), 0, ""

    def prompt(self, text):
        text = WRAPPERS.sub("", text or "").strip()
        if not text or text.startswith(NOISE):
            return
        self.first_prompt = self.first_prompt or text
        self.recent.append(text)
        self.turns += 1

    def excerpt(self):
        prompts = "\n".join("- " + clip(p, 1200) for p in self.recent)
        reply = clip(self.reply, 2500)
        return "Recent prompts:\n" + prompts + ("\n\nLast reply:\n" + reply if reply else "")


def claude_text(entry):
    if entry.get("isMeta") or entry.get("isSidechain") or entry.get("promptSource") == "system":
        return ""
    if (entry.get("origin") or {}).get("kind") not in (None, "human"):
        return ""
    content = (entry.get("message") or {}).get("content")
    if isinstance(content, list):
        content = "\n".join(c.get("text", "") for c in content if isinstance(c, dict) and c.get("type") == "text")
    text = re.sub(r"<system-reminder>.*?</system-reminder>", "", content or "", flags=re.S)
    command = re.match(r"\s*<command-name>(.*?)</command-name>", text, re.S)
    if not command:
        return text
    arguments = re.search(r"<command-args>(.*?)</command-args>", text, re.S)
    return (command.group(1) + " " + (arguments.group(1) if arguments else "")).strip()


def claude_reply(entry):
    content = (entry.get("message") or {}).get("content")
    if isinstance(content, list):
        return "\n".join(c.get("text", "") for c in content if isinstance(c, dict) and c.get("type") == "text")
    return content if isinstance(content, str) else ""


def claude_session(path, side):
    """One Claude Code transcript (*.jsonl named by session ID); subagent transcripts are skipped."""
    t = Transcript()
    info = dict(cwd="", branch="", ai_title="", custom_title="", entry="", prs=[], first=None, last=None)
    with open(path, encoding="utf-8", errors="replace") as handle:
        for line in handle:
            try:
                d = json.loads(line)
            except ValueError:
                continue
            kind = d.get("type")
            if d.get("timestamp"):
                info["first"] = info["first"] or d["timestamp"]
                info["last"] = d["timestamp"]
            if kind == "custom-title":
                info["custom_title"] = d.get("customTitle") or info["custom_title"]
            elif kind == "ai-title":
                info["ai_title"] = d.get("aiTitle") or info["ai_title"]
            elif kind == "pr-link" and d.get("prUrl"):
                info["prs"].append(d["prUrl"])
            elif kind in ("user", "assistant") and not d.get("isSidechain"):
                info["cwd"] = d.get("cwd") or info["cwd"]
                info["branch"] = d.get("gitBranch") or info["branch"]
                info["entry"] = info["entry"] or d.get("entrypoint") or ""
                if kind == "user":
                    t.prompt(claude_text(d))
                elif text := claude_reply(d).strip():
                    t.reply = text
    if not t.turns:
        return None
    if side == "windows":
        tool, environment = "claude_desktop", "ssh" if path.parent.name.startswith("ssh-") else environment_of(info["cwd"], "windows")
    else:
        tool, environment = ("claude_desktop" if info["entry"] == "claude-desktop" else "claude_code"), "wsl"
    sid, cwd = path.stem, info["cwd"]
    title = info["custom_title"] or info["ai_title"]
    resume = {"cmd": f"cd {shlex.quote(cwd)} && claude --resume {sid}"} if environment == "wsl" and cwd else {}
    if tool == "claude_desktop":
        resume["hint"] = clip("Claude Desktop → Code → " + (title or "this session"), 200)
    return dict(
        tool=tool,
        session_id=sid,
        environment=environment,
        cwd=cwd,
        repo=repo_of(cwd) if environment == "wsl" else "",
        branch="" if info["branch"] == "HEAD" else info["branch"],
        title=title,
        transcript=t,
        pr_urls=list(dict.fromkeys(info["prs"]))[:20],
        started_at=iso(info["first"]),
        last_active_at=iso(info["last"]),
        resume=resume,
    )


def claude_sources(root, side):
    for path in glob.glob(os.path.join(root, "*", "*.jsonl")):
        stat = os.stat(path)
        yield path, f"{stat.st_mtime_ns}:{stat.st_size}", stat.st_mtime, lambda p=path: claude_session(Path(p), side)


def connect(path):
    return sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=5)


def codex_history(home, thread_id, t):
    """Recent prompts and the last reply from Codex's thread history, newest first."""
    path = os.path.join(home, "thread_history_1.sqlite")
    if not os.path.exists(path):
        return False
    with connect(path) as con:
        t.turns = con.execute(
            "SELECT count(*) FROM thread_items WHERE thread_id=? AND item_type='userMessage'", (thread_id,)
        ).fetchone()[0]
        if not t.turns:
            return False
        rows = con.execute(
            "SELECT item_type, item_json FROM thread_items WHERE thread_id=? AND item_type IN ('userMessage','agentMessage')"
            " ORDER BY rollout_ordinal DESC LIMIT 20",
            (thread_id,),
        ).fetchall()
    for kind, raw in rows:
        item = json.loads(raw)
        if kind == "agentMessage" and not t.reply:
            t.reply = item.get("text") or ""
        elif kind == "userMessage" and len(t.recent) < 3:
            text = "\n".join(c.get("text", "") for c in item.get("content", []) if c.get("type") == "text").strip()
            if text := WRAPPERS.sub("", text).strip():
                t.recent.appendleft(text)
    return True


def codex_rollout(path, t):
    with open(path, encoding="utf-8", errors="replace") as handle:
        for line in handle:
            try:
                d = json.loads(line)
            except ValueError:
                continue
            payload = d.get("payload") if isinstance(d.get("payload"), dict) else {}
            if d.get("type") == "event_msg" and payload.get("type") == "user_message":
                t.prompt(payload.get("message"))
            elif d.get("type") == "event_msg" and payload.get("type") == "agent_message":
                t.reply = payload.get("message") or t.reply


def codex_sources(home, tool):
    """Codex keeps its thread list in state_5.sqlite; reviewer and spawned subagent threads are skipped."""
    path = os.path.join(home, "state_5.sqlite")
    if not os.path.exists(path):
        return
    with connect(path) as con:
        con.row_factory = sqlite3.Row
        rows = [dict(r) for r in con.execute("SELECT * FROM threads WHERE source NOT LIKE '%subagent%'")]
    for row in rows:
        updated = row.get("updated_at_ms") or (row.get("updated_at") or 0) * 1000
        version = str(updated) + (":a" if row.get("archived") else "")
        yield f"{path}#{row['id']}", version, updated / 1000, lambda r=row: codex_session(home, tool, r)


def codex_session(home, tool, row):
    if row.get("archived"):
        return None
    t = Transcript()
    t.first_prompt = WRAPPERS.sub("", row.get("first_user_message") or "").strip()
    if not codex_history(home, row["id"], t):
        rollout = local_path(row.get("rollout_path")) or (
            "/mnt/" + windows_path(row["rollout_path"])[0].lower() + windows_path(row["rollout_path"])[2:]
            if re.match(r"^[A-Za-z]:", windows_path(row.get("rollout_path") or ""))
            else None
        )
        if rollout and os.path.exists(rollout):
            codex_rollout(rollout, t)
        t.first_prompt = WRAPPERS.sub("", row.get("first_user_message") or "").strip() or t.first_prompt
    if not (t.turns or t.first_prompt):
        return None
    cwd = windows_path(row.get("cwd"))
    environment = environment_of(cwd, "wsl" if tool == "codex_cli" else "windows")
    title = row.get("name") or row.get("title") or ""
    resume = (
        {"cmd": f"cd {shlex.quote(cwd)} && codex resume {row['id']}"}
        if tool == "codex_cli"
        else {"hint": clip("ChatGPT desktop → " + (title or "this thread"), 200)}
    )
    return dict(
        tool=tool,
        session_id=row["id"],
        environment=environment,
        cwd=cwd,
        repo=repo_of(cwd, row.get("git_origin_url") or ""),
        branch=row.get("git_branch") or "",
        title=title,
        transcript=t,
        pr_urls=[],
        started_at=iso(row.get("created_at_ms") or row.get("created_at")),
        last_active_at=iso(row.get("updated_at_ms") or row.get("updated_at")),
        resume=resume,
    )


def windows_homes():
    for home in sorted(glob.glob("/mnt/c/Users/*")):
        if os.path.basename(home) in {"Public", "Default", "Default User", "All Users"}:
            continue
        if os.path.isdir(os.path.join(home, ".claude")) or os.path.isdir(os.path.join(home, ".codex")):
            yield home


def sources():
    home = str(Path.home())
    yield from claude_sources(os.path.join(home, ".claude/projects"), "wsl")
    yield from codex_sources(os.path.join(home, ".codex"), "codex_cli")
    for windows in windows_homes():
        yield from claude_sources(os.path.join(windows, ".claude/projects"), "windows")
        yield from codex_sources(os.path.join(windows, ".codex"), "codex_desktop")


def payload(session, machine):
    t = session["transcript"]
    body = {
        "machine": machine[:100],
        "environment": session["environment"],
        "cwd": session["cwd"],
        "repo": session["repo"],
        "branch": session["branch"],
        "title": session["title"],
        "first_prompt": t.first_prompt,
        "last_prompt": t.recent[-1] if t.recent else "",
        "excerpt": t.excerpt(),
        "pr_urls": session["pr_urls"],
        "turn_count": t.turns,
        "started_at": session["started_at"],
        "last_active_at": session["last_active_at"],
        "resume": session["resume"],
    }
    for key, limit in LIMITS.items():
        body[key] = clip(body[key], limit)
    return body


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def settled(last_active):
    try:
        return datetime.now(UTC) - datetime.fromisoformat(last_active) >= SETTLED
    except (TypeError, ValueError):
        return False


def token():
    value = os.environ.get("ERIDANI_BOT_KEY", "").strip()
    path = CONFIG / "token"
    if not value and path.exists():
        value = path.read_text().strip()
    if not value.startswith("eri_bot_"):
        sys.exit(f"Set ERIDANI_BOT_KEY or save a sessions:write bot key to {path} (chmod 600).")
    return value


def send(api, key, tool, session_id, body):
    url = f"{api.rstrip('/')}/sessions/{tool}/{urllib.parse.quote(session_id, safe='')}"
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode(),
        method="PUT",
        headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"},
    )
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                return json.load(response)
        except urllib.error.HTTPError as exc:
            if exc.code == 429 and attempt < 2:
                time.sleep(30)
                continue
            detail = exc.read().decode(errors="replace")[:300]
            if exc.code in (401, 403):
                sys.exit(f"Eridani refused the bot key ({exc.code}): {detail}")
            raise RuntimeError(f"{exc.code} {detail}") from None


def load_state():
    try:
        return json.loads(STATE.read_text())
    except (OSError, ValueError):
        return {"sessions": {}}


def save_state(state):
    STATE.parent.mkdir(parents=True, exist_ok=True)
    temporary = STATE.with_suffix(".tmp")
    temporary.write_text(json.dumps(state))
    temporary.replace(STATE)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true", help="print sessions instead of sending them")
    parser.add_argument("--days", type=float, help="only sessions active in the last N days (dry run: 7)")
    parser.add_argument("--api", default=API_URL)
    parser.add_argument("--rate", type=int, default=90, help="maximum requests per minute (server limit 120)")
    args = parser.parse_args()
    days = args.days if args.days is not None else 7 if args.dry_run else None
    cutoff = time.time() - days * 86400 if days else 0
    machine = os.environ.get("ERIDANI_MACHINE") or socket.gethostname()
    state = load_state()
    known = state.setdefault("sessions", {})
    key = None if args.dry_run else token()
    rows, sent, failed, reported = [], 0, 0, set()
    for source, version, modified, read in sources():
        if modified < cutoff:
            continue
        # Unchanged since the last send, and nothing left to report: skip without parsing.
        previous = known.get(source, {})
        if not args.dry_run and previous.get("version") == version and (previous.get("settled") or not settled(previous.get("last_active"))):
            continue
        try:
            session = read()
        except (OSError, sqlite3.Error, ValueError) as exc:
            print(f"skipped a session store entry: {type(exc).__name__}: {exc}", file=sys.stderr)
            continue
        name = session["tool"] + ":" + session["session_id"] if session else ""
        # Claude Desktop mirrors SSH sessions under ssh-<id> on Windows; the host's copy (read first) wins.
        if not session or name in reported:
            continue
        reported.add(name)
        body = payload(session, machine)
        if args.dry_run:
            rows.append((body["last_active_at"] or "", session["tool"], body["title"] or body["first_prompt"][:60], body["cwd"], body["branch"]))
            continue
        fingerprint, quiet = digest(body), settled(body["last_active_at"])
        if previous.get("fingerprint") == fingerprint and (previous.get("settled") or not quiet):
            previous["version"] = version
            continue
        try:
            send(args.api, key, session["tool"], session["session_id"], body)
            sent += 1
        except (RuntimeError, OSError) as exc:
            failed += 1
            print(f"{name}: {exc}", file=sys.stderr)
            continue
        known[source] = {"version": version, "fingerprint": fingerprint, "settled": quiet, "last_active": body["last_active_at"]}
        save_state(state)
        time.sleep(60 / max(1, args.rate))
    if args.dry_run:
        for last, tool, title, cwd, branch in sorted(rows, reverse=True):
            print(f"{last[:16]:16}  {tool:14}  {redact(title)[:48]:48}  {cwd[-40:]:40}  {branch[:24]}")
        print(f"{len(rows)} sessions" + (f" active in the last {days:g} days" if days else ""))
        return
    save_state(state)
    print(f"sent {sent}, failed {failed}")


if __name__ == "__main__":
    main()
