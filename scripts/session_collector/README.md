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

A running app keeps its Codex database in WAL mode, and WSL can't map that database's shared-memory index across `/mnt/c`. In that case the collector reads a private copy of the database and its WAL, so the newest writes are included.
- The copy goes in `/dev/shm` and is deleted when the run ends.
- The large thread history is copied only in runs where a thread changed.

## remotea terminals

When [remotea](https://github.com/rdavisdunham/remotea) is installed, the collector reads `remotea status --json` (version 1). It looks for the CLI in `REMOTEA_BIN`, then on `PATH`, then at `~/.local/bin/remotea`, and runs it with node; when node isn't on `PATH`, it uses the newest nvm one.
- **Live terminals:** a session running in a remotea terminal is reported with `resume.live`, which holds its workspace, terminal id and host. It has no resume command, so the session can't be started twice. Eridani shows it under **Live**. A `lost` terminal (after a server restart) shows as Restoring until a remotea app restores it.
- **Stopped sessions:** other WSL Claude Code and Codex sessions resume with `remotea open --group <last workspace> --cwd <folder> -- claude --resume <id>`, so they reopen as a synced tab. Sessions remotea has never seen use the Inbox workspace.
- **Detecting changes:** a terminal opening or closing triggers a resend, even when the transcript hasn't changed.
- **Failures:** if `remotea status` fails, the last good read is kept so resume commands don't flap.
- **Codex:** Codex terminals are marked live once remotea maps Codex processes to threads.

## Setup

1. In Eridani, create a bot key under **Settings → Integrations → Connected agents** with `sessions:write`.
   Add `sessions:read` too if the same key will also be used by agents to search.
2. Save it with `install -D -m 600 /dev/stdin ~/.config/eridani-collector/token` (paste the key, then Ctrl-D).
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

## Retirement

remotea's own collector replaces this one. Until the cutover it runs in parity mode: Eridani compares its
reports with this collector's and stores only this collector's. Never run both as the store's source at once.
These steps mirror remotea's `docs/remotea/eridani-link.md` § Stage E: collector cutover;
`remotea eridani collector authoritative` prints them too.

**Ready.** Eridani's remotea page shows the parity summary as "ready for cutover": 7 consecutive clean days in your
timezone, ending today or yesterday. A clean day has at least one parity report and no mismatches. Reports for
sessions only remotea saw (missing here) don't count against a day. Readiness is computed by
`agent_link.readiness` and returned as `cutover` from `GET /api/v1/remotea/parity`.

**Steps** (Davin runs these on the work host; nothing runs them automatically):
1. Before. The parity summary shows "ready for cutover", and the remotea release with the cutover is the one
   installed and running.
2. `remotea eridani collector authoritative`. Within 30 s the link reconnects with `collector.authoritative`,
   `remotea status --json` turns `version: 2`, and every source is sent once more, without the parity flag.
   From here on, a forgotten timer of this collector stops learning about remotea terminals (it reads only
   version 1) and logs "remotea status unavailable; keeping the last known terminals". It still sends its other
   reports, so step 3 is required.
3. `systemctl --user disable --now eridani-collector.timer`.
4. `remotea hooks --remove-legacy-collector --dry-run`, then without `--dry-run`. It removes only the
   `SessionStart` handler that runs `session_hook.py`, and keeps the original as
   `~/.claude/settings.json.remotea-backup`. Sessions in remotea terminals still learn their own session id; a
   Claude session started outside remotea no longer does.
5. Watch one cycle. `remotea eridani status` shows the link ready, no refusals, the outbox draining. A new or
   resumed session appears in Eridani with its resume hint. The agents' MCP key (`cli-agents`) still works: it is
   not part of the cutover.
6. One release later, after a clean cutover:
   - remove the units: `rm ~/.config/systemd/user/eridani-collector.{service,timer}`, then
     `systemctl --user daemon-reload`;
   - remove the token directory: `rm -r ~/.config/eridani-collector`;
   - revoke the old collector credential in Eridani under **Settings → Integrations → Connected agents**. Never the
     remotea link key or `cli-agents`.

**Rollback** (any time before step 6): `remotea eridani collector parity`, then
`systemctl --user enable --now eridani-collector.timer`, and put the `SessionStart` hook back by hand if you want
it (`python3 ~/jarvis/scripts/session_collector/session_hook.py`). Both sides kept their fingerprints, so going
back sends nothing new.

The code in this directory stays until a later stage removes it.
