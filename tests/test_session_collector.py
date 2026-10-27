import json
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "session_collector"))
import collector


def lines(path, *entries):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(e) for e in entries) + "\n")
    return path


def user(text, **extra):
    return {"type": "user", "cwd": "/home/davin/acme", "gitBranch": "davin/eng-12-login", "entrypoint": "cli",
            "timestamp": "2026-10-20T10:00:00Z", "message": {"role": "user", "content": text}, **extra}


def test_claude_transcripts_keep_human_prompts_titles_and_links(tmp_path):
    path = lines(
        tmp_path / "-home-davin-acme" / "abc-123.jsonl",
        user("Fix the login test. My key is sk-proj-" + "x" * 30),
        user("<system-reminder>ignore</system-reminder>", isMeta=True),
        user("Task finished", promptSource="system", origin={"kind": "task-notification"}),
        user([{"type": "tool_result", "tool_use_id": "t", "content": "ok", "is_error": False}]),
        user("<command-name>/review</command-name>\n<command-args>PR 4</command-args>"),
        user("side work", isSidechain=True),
        {"type": "assistant", "timestamp": "2026-10-20T10:05:00Z", "message": {"content": [{"type": "text", "text": "Fixed it."}]}},
        {"type": "ai-title", "aiTitle": "Fix login test", "sessionId": "abc-123"},
        {"type": "custom-title", "customTitle": "ENG-12 login", "sessionId": "abc-123"},
        {"type": "pr-link", "prUrl": "https://github.com/acme/app/pull/4", "prNumber": 4},
    )
    session = collector.claude_session(path, "wsl")
    assert session["tool"] == "claude_code" and session["environment"] == "wsl"
    assert session["title"] == "ENG-12 login" and session["branch"] == "davin/eng-12-login"
    assert session["pr_urls"] == ["https://github.com/acme/app/pull/4"]
    assert session["resume"]["cmd"] == "cd /home/davin/acme && claude --resume abc-123"
    body = collector.payload(session, "pc")
    assert body["turn_count"] == 2 and body["last_prompt"] == "/review PR 4"
    assert "sk-proj" not in body["first_prompt"] and "[redacted]" in body["first_prompt"]
    assert "Fixed it." in body["excerpt"] and "side work" not in body["excerpt"]
    assert body["last_active_at"] == "2026-10-20T10:05:00+00:00"


def test_windows_side_claude_sessions_are_desktop_sessions(tmp_path):
    path = lines(tmp_path / "ssh-1234" / "s.jsonl", user("hello", cwd="/srv/app"))
    session = collector.claude_session(path, "windows")
    assert (session["tool"], session["environment"], session["resume"].get("cmd")) == ("claude_desktop", "ssh", None)
    assert collector.claude_session(lines(tmp_path / "x" / "empty.jsonl", {"type": "ai-title", "aiTitle": "t"}), "wsl") is None


def test_codex_threads_skip_subagents_and_read_recent_history(tmp_path):
    state = sqlite3.connect(tmp_path / "state_5.sqlite")
    state.execute(
        "CREATE TABLE threads (id TEXT, rollout_path TEXT, created_at INTEGER, updated_at INTEGER, source TEXT, cwd TEXT,"
        " title TEXT, name TEXT, archived INTEGER, git_branch TEXT, git_origin_url TEXT, first_user_message TEXT,"
        " created_at_ms INTEGER, updated_at_ms INTEGER)"
    )
    rows = [
        ("t1", "C:\\x.jsonl", 1, 2, "vscode", "\\\\?\\UNC\\wsl.localhost\\Ubuntu\\home\\davin\\acme", "Login fix", None, 0,
         "eng-12-login", "git@github.com:acme/acme-app.git", "Fix login", 1760000000000, 1760000600000),
        ("t2", "", 1, 2, '{"subagent":{"other":"guardian"}}', "C:/x", "Review", None, 0, None, None, "r", 1, 2),
        ("t3", "", 1, 2, "vscode", "\\\\?\\C:\\Users\\davin\\Documents", "Old", None, 1, None, None, "o", 1, 2),
    ]
    state.executemany("INSERT INTO threads VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)", rows)
    state.commit()
    history = sqlite3.connect(tmp_path / "thread_history_1.sqlite")
    history.execute("CREATE TABLE thread_items (thread_id TEXT, rollout_ordinal INTEGER, item_type TEXT, item_json TEXT)")
    for ordinal, kind, item in (
        (1, "userMessage", {"type": "userMessage", "content": [{"type": "text", "text": "Fix login"}]}),
        (2, "agentMessage", {"type": "agentMessage", "text": "Looking."}),
        (3, "userMessage", {"type": "userMessage", "content": [{"type": "text", "text": "<realtime_delegation>\n<input>Now add a test</input>"}]}),
        (4, "agentMessage", {"type": "agentMessage", "text": "Added the test."}),
    ):
        history.execute("INSERT INTO thread_items VALUES ('t1', ?, ?, ?)", (ordinal, kind, json.dumps(item)))
    history.commit()
    found = [read() for _, _, _, read in collector.codex_sources(str(tmp_path), "codex_desktop")]
    session = next(s for s in found if s)
    assert [s for s in found if s] == [session] and session["session_id"] == "t1"
    assert session["cwd"] == "//wsl.localhost/Ubuntu/home/davin/acme" and session["environment"] == "wsl"
    assert session["repo"] == "acme-app" and session["branch"] == "eng-12-login"
    assert session["resume"] == {"hint": "ChatGPT desktop → Login fix"}
    body = collector.payload(session, "pc")
    assert (body["turn_count"], body["first_prompt"], body["last_prompt"]) == (2, "Fix login", "Now add a test")
    assert body["excerpt"].endswith("Last reply:\nAdded the test.")


def status(*sessions, hostname="DavisPC"):
    return {"version": 1, "server": {"hostname": hostname, "running": True}, "groups": [], "sessions": list(sessions)}


def terminal(sid, state, group, provider="claude", activity=1, surface="surf-1"):
    return {"id": surface, "groupName": group, "state": state, "lastActivityAt": activity,
            "agent": {"provider": provider, "sessionId": sid} if sid else None}


def test_remotea_terminals_point_sessions_at_their_live_workspace():
    live, groups = collector.remotea_terminals(status(
        terminal("abc", "lost", "Acme", activity=1, surface="surf-old"),
        terminal("abc", "live", "Acme", activity=2, surface="surf-new"),
        terminal("t9", "exited", "Beta", provider="codex"),
        terminal(None, "live", "Shell"),
        terminal("x1", "live", "Other", provider="aider"),
    ))
    assert live == {"abc": {"provider": "claude", "state": "live", "workspace": "Acme", "terminal": "surf-new", "host": "DavisPC"}}
    assert groups == {"abc": "Acme", "t9": "Beta"}
    terminals = {"installed": True, "live": live, "groups": {**groups, "old": "Gamma"}}
    claude = {"tool": "claude_code", "session_id": "abc", "cwd": "/home/davin/acme", "resume": {"cmd": "cd /home/davin/acme && claude --resume abc"}}
    assert collector.resume_in_remotea(claude, terminals) == {
        "hint": "Live in remotea → Acme", "live": {"state": "live", "workspace": "Acme", "terminal": "surf-new", "host": "DavisPC"}}
    codex = {"tool": "codex_cli", "session_id": "t9", "cwd": "/home/davin/beta app", "resume": {"cmd": "cd x && codex resume t9"}}
    assert collector.resume_in_remotea(codex, terminals)["cmd"] == "remotea open --group Beta --cwd '/home/davin/beta app' -- codex resume t9"
    unknown = {**claude, "session_id": "new"}
    assert collector.resume_in_remotea(unknown, terminals)["cmd"].startswith("remotea open --group Inbox ")
    # A Claude terminal never claims a Codex thread with the same ID; desktop threads keep their app hint.
    assert "live" not in collector.resume_in_remotea({**codex, "session_id": "abc"}, terminals)
    desktop = {"tool": "codex_desktop", "session_id": "d1", "cwd": "C:/Users/davin", "resume": {"hint": "ChatGPT desktop → x"}}
    assert collector.resume_in_remotea(desktop, terminals) == {"hint": "ChatGPT desktop → x"}
    assert collector.resume_in_remotea(claude, {"installed": False, "live": {}, "groups": {}}) == claude["resume"]


def test_remotea_cli_runs_under_an_nvm_node_and_a_failed_read_keeps_the_last_terminals(tmp_path, monkeypatch):
    cli = tmp_path / "remotea"
    cli.write_text("#!/usr/bin/env node\n")
    node = tmp_path / "home/.nvm/versions/node/v22.23.2/bin/node"
    node.parent.mkdir(parents=True)
    node.write_text("")
    (tmp_path / "home/.nvm/versions/node/v9.1.0/bin").mkdir(parents=True)
    monkeypatch.setenv("REMOTEA_BIN", str(cli))
    monkeypatch.setattr(collector.Path, "home", lambda: tmp_path / "home")
    monkeypatch.setattr(collector.shutil, "which", lambda name: None)
    assert collector.remotea_command() == [str(node), str(cli)]
    monkeypatch.setenv("REMOTEA_BIN", str(tmp_path / "missing"))
    state = {"remotea": {"installed": True, "live": {"abc": {}}, "groups": {"abc": "Acme"}}}
    assert collector.remotea(state) == {"installed": False, "live": {}, "groups": {"abc": "Acme"}}
    fake = tmp_path / "fake.py"
    monkeypatch.setattr(collector, "remotea_command", lambda: [sys.executable, str(fake)])
    fake.write_text("print('{\"version\": 2}')")
    state = {"remotea": {"installed": True, "live": {"abc": {"state": "live"}}, "groups": {}}}
    assert collector.remotea(state)["live"] == {"abc": {"state": "live"}}
    fake.write_text("import json; print(json.dumps(" + repr(status(terminal("abc", "exited", "Acme"))) + "))")
    assert collector.remotea(state) == {"installed": True, "live": {}, "groups": {"abc": "Acme"}}


def test_a_terminal_opening_or_closing_resends_an_unchanged_transcript(tmp_path, monkeypatch):
    path = lines(tmp_path / "-home-davin-acme" / "abc.jsonl", user("Fix the login test", timestamp="2026-01-01T00:00:00Z"))
    current = {"value": status()}
    sent = []
    monkeypatch.setattr(collector, "STATE", tmp_path / "state.json")
    monkeypatch.setattr(collector, "sources", lambda: collector.claude_sources(str(tmp_path), "wsl"))
    monkeypatch.setattr(collector, "remotea_command", lambda: ["remotea"])
    monkeypatch.setattr(collector.subprocess, "run", lambda *a, **k: type("R", (), {"returncode": 0, "stdout": json.dumps(current["value"])}))
    monkeypatch.setattr(collector, "send", lambda api, key, tool, sid, body: sent.append(body["resume"]))
    monkeypatch.setattr(collector.time, "sleep", lambda s: None)
    monkeypatch.setenv("ERIDANI_BOT_KEY", "eri_bot_test")
    monkeypatch.setattr(sys, "argv", ["collector.py"])
    collector.main()
    assert sent == [{"cmd": "remotea open --group Inbox --cwd /home/davin/acme -- claude --resume abc"}]
    collector.main()
    assert len(sent) == 1
    current["value"] = status(terminal("abc", "live", "Acme"))
    collector.main()
    assert sent[-1]["live"]["workspace"] == "Acme" and "cmd" not in sent[-1]
    current["value"] = status()
    collector.main()
    assert sent[-1] == {"cmd": "remotea open --group Acme --cwd /home/davin/acme -- claude --resume abc"}
    assert path.exists() and len(sent) == 3


def test_a_live_wal_database_is_read_through_a_snapshot_with_its_newest_writes(tmp_path, monkeypatch):
    path = str(tmp_path / "state_5.sqlite")
    writer = sqlite3.connect(path)
    writer.execute("PRAGMA journal_mode=wal")
    writer.execute("PRAGMA wal_autocheckpoint=0")
    writer.execute("CREATE TABLE threads (id TEXT, source TEXT)")
    writer.execute("INSERT INTO threads VALUES ('only-in-the-wal', 'vscode')")
    writer.commit()
    real = sqlite3.connect

    def across_mnt_c(target, *args, **kwargs):
        # What WSL sees for a Windows app's open WAL database under /mnt/c.
        if target == f"file:{path}?mode=ro":
            raise sqlite3.OperationalError("disk I/O error")
        return real(target, *args, **kwargs)

    monkeypatch.setattr(collector.sqlite3, "connect", across_mnt_c)
    monkeypatch.setattr(collector, "READABLE", {})
    monkeypatch.setattr(collector, "codex_session", lambda home, tool, row: row["id"])
    assert [read() for *_, read in collector.codex_sources(str(tmp_path), "codex_desktop")] == ["only-in-the-wal"]
    assert collector.READABLE[path] != path
    writer.close()
    monkeypatch.setattr(collector, "READABLE", {path: str(tmp_path / "missing" / "copy.sqlite")})
    assert list(collector.codex_sources(str(tmp_path), "codex_desktop")) == []
