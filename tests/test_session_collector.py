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
