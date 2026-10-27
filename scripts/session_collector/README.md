# Session collector

Reports local AI agent sessions to the Eridani session ledger (`/api/v1/external/sessions`), so
"where did I work on X?" has one answer across tools. It runs on the work host (WSL) and reads,
never writes, these stores:

| Tool | Store | Ledger `tool` |
|---|---|---|
| Claude Code (WSL) | `~/.claude/projects/*/*.jsonl` | `claude_code` (`claude_desktop` when started from Claude Desktop's Code tab) |
| Claude Desktop (Windows, SSH) | `/mnt/c/Users/*/.claude/projects` | `claude_desktop` |
| Codex CLI (WSL) | `~/.codex/state_5.sqlite`, `thread_history_1.sqlite` | `codex_cli` |
| ChatGPT desktop / Codex app | `/mnt/c/Users/*/.codex/...` | `codex_desktop` |

Subagent and reviewer threads are skipped. Per session it sends:
- metadata: tool, folder, repo, branch, title, PR links, turn count, times and how to resume
- the first and last prompts
- an excerpt of at most 8,000 characters

Common secret formats are redacted before sending. Transcripts never leave the machine.

Plain chats in the ChatGPT and Claude desktop apps live only on those services, so they are not collected.

## Setup

1. In Eridani, create a bot key under **Settings → Integrations → Connected agents** with `sessions:write`.
   Add `sessions:read` too if the same key will also be used by agents to search.
2. Save it with `install -m 600 /dev/stdin ~/.config/eridani-collector/token` (paste the key, then Ctrl-D).
3. Preview what would be sent: `python3 -I scripts/session_collector/collector.py --dry-run`.
4. Install the timer: `scripts/session_collector/install.sh`. The first run backfills every session, at 90 per minute.

## Filing

The ledger files each session automatically, strongest evidence first:
1. a synced Linear issue key in the branch (`davin/eng-123-fix`)
2. exactly one known issue key in the title or prompts
3. a folder rule (Sessions → Rules in Eridani; `/home/davin/brainforge/acme` → the Acme client)

What it can't file appears under **Unfiled**. A manual or agent `session_link` always wins over automatic filing.

## Letting agents file their own session (optional)

Add Eridani's MCP server (`/api/v1/external/mcp/`) to Claude Code or Codex with a `sessions:write` key. Agents can then call:
- `session_search` to find past sessions
- `session_link` to file the current session

`session_hook.py` is a Claude Code `SessionStart` hook. It tells the agent its own session ID so it can call `session_link`.
