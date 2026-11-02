# Agent control: the remotea link

Eridani starts, watches and steers coding agents (Claude Code and Codex) on Davin's desktop through
**remotea**, the always-on terminal host in WSL. This page is the Eridani side; the host side is remotea's
`docs/remotea/eridani-link.md`. What ships today (2026-10-30):

- **The link.** remotea opens one outbound WebSocket to Eridani (nothing on the desktop listens to the
  internet) and reports every Claude Code and Codex session it runs, with turns, status and questions.
  Eridani's commands are rows in a journal (`agent_commands`, each with a stable `op_id`, digest and expiry)
  that the host drains, acks and answers with an outcome; remotea keeps its own journal and outbox, so
  nothing is lost across restarts. After every reconnect `reconcile()` matches runs to their surfaces,
  recovers lost spawn acks, repeats the desired state (`runs.snapshot`: stops, verify config) and only then
  drains the outbox ([Connection lifecycle](#connection-lifecycle), [Commands](#commands)).
- **Runs from any task.** Every task row has a Run control (task details too) while remotea is on; the Run
  dialog picks the execution target and harness with a read-only preflight. Eri can start or message a run
  too, in two phases: `control.start` and `control.message` only propose, and nothing reaches the host until
  the owner says yes on the card, on the remotea page or in a later message
  ([Two-phase control from Eri](#two-phase-control-from-eri)).
- **Interview and planning.** A task without a PRD (and without human-written criteria and verify) starts
  with an interview: the agent asks up to 12 questions through `task_interview`, the owner answers on one
  form, and the agent writes the PRD and calls `submit_plan`. The **plan floor** means an agent's plan may
  only add: a change to a human criterion, the verify command or the completion policy is a proposal, and
  **plan review** (Accept / Keep per change) happens before any work starts
  ([Interview and planning](#interview-and-planning), [Actors, the plan floor and the completion
  gate](#actors-the-plan-floor-and-the-completion-gate)).
- **The execution loop.** After each execution turn the host runs the plan's verify command in the
  worktree, then the judge (`agent_judge.py`) compares the turn and the recorded evidence with the success
  criteria. `not_met` sends the judge's gap as the next prompt; `met` waits for **Approve** (or for merged
  sub-tasks, or completes the task itself when `auto_complete` is on). **Verify by default**: a plan with a
  test-shaped criterion and no verify command gets the target's verify command picked for it, and the owner
  can set one mid-run. The judge's rules park the run with Davin rather than loop: `verify_missing` (a gap
  that needs a verification the plan cannot record), `proof_request` (asks to paste proof are rewritten to
  checkable actions), `pr_unknown` (the PR record is older than the turn) and `no_progress_gap` (the same gap
  or the same rules three verdicts in a row). The turn cap, the 6-hour wall clock and the 45-minute
  no-progress timer park it too ([The judge](#the-judge-agent_judgepy), [Verify by default, and set
  mid-run](#verify-by-default-and-set-mid-run)).
- **Freestyle** switches the judge, verification, gap prompts and the turn cap off while the owner talks to
  the agent directly; **Approve** completes the task as a human; Message (queue), Steer, Re-plan, Verify
  now, Set verify and Stop / Force stop are on every run ([Freestyle](#freestyle)).
- **Where you see it.** The remotea page (runs, unmerged work, hosts, parity), the **side run panel** (a
  column beside any page, opened from a task's run chip or the topbar Runs panel; a sheet on phones), and
  the run timeline, which names who sent every text ([Messages Eridani sends to the
  agent](#messages-eridani-sends-to-the-agent)).
- **Eri** has the `control_*` tools (start, message, stop, answer, confirm, approve, plan review, status)
  while remotea is on ([Voice commands](#voice-commands-control)). **Other assistants** read runs with a
  bot key carrying `agents:read` (`run_list` / `run_get`, [Reading runs](#reading-runs-agentsread)).

Code: [`agent_link.py`](../apps/api/jarvis/agent_link.py) (socket, ingestion, parity),
[`run_coordinator.py`](../apps/api/jarvis/run_coordinator.py) (run states, commands, verify and judge loop),
[`agent_control.py`](../apps/api/jarvis/agent_control.py) (start, message, stop, Eri's `control.*`),
[`agent_plan.py`](../apps/api/jarvis/agent_plan.py) (interview, `submit_plan`, plan review, verify),
[`agent_judge.py`](../apps/api/jarvis/agent_judge.py), [`agent_prompts.py`](../apps/api/jarvis/agent_prompts.py),
[`remotea_routes.py`](../apps/api/jarvis/remotea_routes.py) (browser routes),
[`task_plan.py`](../apps/api/jarvis/task_plan.py) (task plan fields); web
[`Remotea.tsx`](../apps/web/src/Remotea.tsx), [`RunPanel.tsx`](../apps/web/src/RunPanel.tsx) and
[`remotea-model.ts`](../apps/web/src/remotea-model.ts).

**History.** The feature shipped in stages, and the section headings below keep the stage that introduced
them; they describe current behaviour unless they say otherwise. A was remotea's own spike (launching and
observing the harnesses, no Eridani code); B the read-only link (sessions, turns, collector parity, jarvis
#63); C runs, the actor model and manual control; D interview, planning, verify attempts and the judge loop;
E Linear sync outcomes, Eri's voice commands, the dream auto-link and the collector cutover; F the fixes from
the 2026-10-28 security audit (all merged by 2026-10-28, jarvis #63 to #68). Later work (verify by default,
freestyle, the side run panel, action idempotency, per-target capabilities, judge evidence rules) is
described in place.

## Turning it on

remotea is **off by default** (preference `remotea_enabled`). While it is off, Eridani behaves exactly as
before: no nav entry, no page, no Run controls on tasks, no `control_*` tools for Eri, the link refuses
connections, and every `/api/v1/remotea/*` route except `status`, `enable` and `disable` answers
`403 REMOTEA_DISABLED` (reading runs and Stop stay available).

Three things are needed on the desktop: the **link key**, the **MCP key** for the spawned agents, and an
**execution target with a verify entry**. The first two arrive together in one **setup code**; the
[setup page](#setup-readiness-e-q1) then walks what is still missing.

### 0. The setup code (one click, task bbdb6440)

**Settings → Connections → remotea → Show remotea** (the first time) and **New setup code** mint both keys and
show a one-time code with the instruction: run `remotea eridani setup` in WSL and paste the code when asked
(or pipe it to `remotea eridani setup --code -`, which keeps it out of argv and shell history). The wizard
proves the link key, writes `eridani-token` and `eridani.json`, saves the MCP key to
`~/.config/remotea/eridani-mcp-token`, offers repositories as targets and prints its readiness table (remotea's
`docs/remotea/eridani-link.md` § Setup).

The code is `eridani-setup:1:` followed by the unpadded base64url of the JSON
`{"url": "<Eridani origin>", "link_key": "eri_bot_…", "mcp_key": "eri_bot_…", "machine"?: "<name>"}`
(`remotea_routes.setup_code`; remotea's `parseSetupCode` reads it). `url` is the public origin
(`https://app.eridani.app`); the wizard derives the link URL and the MCP URL from it. `machine` is set only when
`POST /setup-code` is given one. A code without `mcp_key` (from **New link key**) leaves the host's MCP key as it
is.

- **The MCP key** is a bot key named `remotea MCP` (365 days) with exactly what remotea's `eridani-mcp.ts` tools
  need: `agents:session` (`task_plan`, `task_step`, `task_interview`, `submit_plan`), `tasks:write` and
  `notes:write` (`task_step` and `submit_plan` edit the task and its PRD note) and `sessions:write`
  (`session_link`); `bot_access.normalize` adds `tasks:read`, `notes:read` and `sessions:read`. Never
  `agents:control`, so it cannot open the link.
- **Rotation.** The name marks the key: a new setup code revokes every active `remotea MCP` key of the account
  and the previous link key. Turning remotea off revokes both. The keys are shown once, in that response;
  `bot_credentials` keeps only their hash, and the audit line `remotea keys minted for owner …` (and
  `… revoked …` on disable) names credential ids, never a token or the code. Both keys appear in
  Settings → Connections → Connected agents with their creation, last use and revocation.
- `GET /status` adds `mcp_key` (prefix, expiry, last use; never the token) and `setup_command`.

### 1. The link key (remotea ↔ Eridani)

**Settings → Connections → remotea → Show remotea** calls `POST /api/v1/remotea/enable`, which:

1. sets `remotea_enabled`;
2. mints a bot key named `remotea` (365 days) with scopes
   `agents:control agents:session sessions:read sessions:write tasks:read records:read`, and (the first time,
   while no link key is active) the MCP key above;
3. returns the link token **once** (`token`) and the setup code (`setup_code`), plus the connect line for an
   older remotea without `eridani setup`:
   `remotea eridani connect --url wss://app.eridani.app/api/v1/external/agents/link`.

Enabling again reuses the active key (no token or code in the response). **New link key** (`{"rotate": true}`)
mints a replacement link key and revokes the old one; its setup code carries no MCP key. **New setup code**
(`POST /setup-code`) replaces both. Turning remotea off **revokes** the link key and the MCP key (stage F);
connected hosts are told `bye {reason: "disabled"}` on their next message or within 30 seconds, and turning
it on again mints a new one. Viewers cannot enable, disable or hold the key. The link key is for the host
only: it never has `tasks:write`, and remotea refuses to hand it to an agent.

`GET /api/v1/remotea/status` → `{enabled, key, mcp_key, setup_command, hosts[], link_url, connect_command}`; a
host is `connected` while its lease is current, otherwise the page shows when it was last seen (or "never").

Each host also carries what its last `hello` said about its build and health (`agent_link.build_identity`, kept in
`agent_hosts.caps.build`, replaced by every hello; null for what an older remotea did not send): `version`,
`build_sha` (40 hex or `"unknown"`), `build` (its first 7 characters), `build_dirty`, `caps_version`,
`hook_version` and `degraded_reason`. `degraded_reason` is set only while the host's journal or outbox cannot
persist (for example `journal_unavailable (write failed: ENOSPC)`): remotea refuses effectful commands until it
clears. The page header shows the connected host's line ("remotea 0.1.0 · build 76c5f62", with a **dirty** badge
for a build with uncommitted changes) and, while degraded, the warning "Host journal degraded: …"; Settings →
Connections → remotea shows the same version line and warning. `status` (the host-wide `ok`/`degraded`) is a
different thing: the owner's daily ingestion quota. Eri's `control_status` lists a degraded host with
`degraded_reason` and the same warning text.

### 2. The MCP key (the agent inside a run ↔ Eridani)

The agent a run spawns calls Eridani's MCP tools (`task_plan`, `task_interview`, `submit_plan`,
`task_step`, `session_link`) with its run token, over a **separate** bot key. The setup code carries one
(above). To make one by hand instead: Settings → Connections → Connected agents → Add agent with "Run tools for
coding agents started from a task" (adds `agents:session`) plus Task scheduling and Notes at Read & edit, and
Agent sessions at Read & edit for `session_link`. Without `agents:session` the MCP server
offers only the planner tools, the agent cannot call `submit_plan`, and the run ends its planning turn in
`needs_you(no_plan)`. There are two ways to give the key to spawned agents:

- **Host-side file (remotea PR #20, preferred).** Save the key to `~/.config/remotea/eridani-mcp-token`
  (or the path in `mcpKeyFile`, global or per target). remotea then launches Claude with
  `--mcp-config <file>` pointing at `https://<link host>/api/v1/external/mcp/` and Codex with
  `-c mcp_servers.eridani.*` and the key in an environment variable for that launch only; the key never
  reaches argv, logs or the journal. A file holding the link key, or anything that is not an `eri_bot_` key,
  is refused. The spawn outcome reports `mcp {configured, reason?}` and each hello target says whether MCP
  is ready, so a host without the file still spawns, but without the tools.
- **User-scope registration in each harness** (older hosts, or the owner's own sessions):
  `claude mcp add --scope user --transport http eridani https://app.eridani.app/api/v1/external/mcp/ --header "Authorization: Bearer <key>"`
  (the URL works with or without the trailing slash).

A third, read-only key with `agents:read` (Connected agents → "Read runs started from tasks") is what an
assistant outside the run uses to review it.

### 3. An execution target with verification

A run starts only in a target the host lists in `~/.config/remotea/eridani.json` (`targets[]`: `id`,
`repo`, `worktreeRoot`, `base`, `launch` per harness, `verify[]`); see remotea's
`docs/remotea/eridani-link.md` § Execution targets. Verification needs, per target:

- **A `verify` entry whose program resolves on the service PATH.** remotea runs it as fixed argv with no
  shell, under systemd's minimal PATH and a minimal environment (`PATH`, `HOME`, `LANG`, `CI=1`,
  `TERM=dumb`, plus only the entry's own `env`). Use an absolute program (`/home/davin/.local/bin/uv`) or a
  wrapper in the repository (`./scripts/verify.sh`), or set `env.PATH`. hello's `verify_status` says
  `ready: false` with a reason ("its program is not on the service PATH …") when it does not resolve.
- **An isolated test database.** For jarvis the entry (or its wrapper) sets `JARVIS_ENV_FILE=""` and a
  disposable `JARVIS_DATABASE_URL` pointing at a local test Postgres, and runs `uv sync --locked --group
  dev` first: managed worktrees get no dependency provisioning, and the suite must never touch the real
  database.
- **A timeout that covers the suite.** The entry's `timeout` in seconds (`--verify-timeout`); the API
  suite takes 470–670 s, so 600 is too tight (the reliability PRD asks for 1800). Eridani's own deadline for
  a verify attempt (`run_coordinator.verify_deadline`, stored on the attempt as `deadline_at`, which the sweep
  reads) is the entry's timeout **as the host advertises it** plus 120 s of grace, never less than 11 minutes
  (`VERIFY_DEADLINE`); the attempt keeps the advertised value as `timeout_s`. A host advertises it as `timeout` on
  the entry's `verify_status` row or on a `verify` item sent as an object `{id, timeout}`. remotea up to PR #24
  sends `verify` as plain ids and no timeout, so until it does the deadline stays 11 minutes: an attempt still
  running then is `timed_out` (a non-pass) and a later result is discarded as `late`.

Targets are re-read on every use, so editing `eridani.json` needs no restart; the next connect's hello
carries the change. Where bubblewrap works, remotea runs the command sandboxed (remotea PR #24); each result says
whether it was (`isolated`, below).

#### jarvis: `scripts/verify.sh`

The jarvis repository carries the wrapper a verify entry calls, so the entry holds no toolchain paths of its
own. [`scripts/verify.sh`](../scripts/verify.sh) `[--dry-run] [--no-sync] [--only api|web] [pytest args…]`:

1. **PATH.** Keeps the caller's `PATH` first, then adds `~/.local/bin`, `~/.cargo/bin`, nvm's default node
   (only when no `node` is on `PATH` yet) and `/usr/local/bin`, so `uv`, `node` and `npm` resolve under the
   service's minimal `PATH`.
2. **Test database.** Exports `JARVIS_ENV_FILE=""` (never the developer's `.env`). Uses the caller's
   `JARVIS_DATABASE_URL` as given: the pytest session creates and drops its own `jarvis_test_<uuid>` database
   on that server (`tests/conftest.py`), so one standing test server serves concurrent runs. Without it the
   script starts a throwaway `postgres:16-alpine` container on a free loopback port (`-p 127.0.0.1::5432`),
   waits for it over TCP and removes it on exit (an `EXIT` trap; `INT`/`TERM` exit through it).
3. **API.** `uv sync --locked --group dev`, then `.venv/bin/python -m pytest -q --tb=short` from the
   repository root (`tests/`); extra arguments go to pytest (`-x`, `-k agent`, a test file).
4. **Web.** In `apps/web`: `npm ci --no-audit --no-fund` when `node_modules` is missing, `npx tsc -b`,
   `npx vitest run`.

It prints a `==> <section>` header and each command (`+ …`) and stops at the first failure with its exit
code. `--no-sync` (or `JARVIS_VERIFY_SYNC=0` in the entry's `env`) skips `uv sync` and `npm ci`;
`--only api|web` (or `JARVIS_VERIFY_ONLY`) runs one half; `--dry-run` prints the plan without running
anything. Alembic (`upgrade head` / `check`) is not part of it: CI runs it against its own database.

**The entry in `~/.config/remotea/eridani.json`** (Davin, on the desktop; task `3a58cf14`). Where bubblewrap
works (it does on the desktop) a verify entry runs sandboxed: `HOME` is a temporary folder, only the
`PATH` folders, the worktree, its venv's base Python and the entry's `mounts` exist, there is no docker
socket and no internet. So the sandboxed entry names the toolchain in `env.PATH`, points
`JARVIS_DATABASE_URL` at a standing test Postgres reached through `network: "loopback"`, and leaves the
installs to the target's dependency preparation (remotea `prepare` steps, S-Q3) with `JARVIS_VERIFY_SYNC=0`.
Added to the existing `jarvis` target (its `id`, `repo`, `worktreeRoot`, `base` and `launch` stay as they
are; match the node path to the installed nvm version):

```json
"prepare": [
  { "id": "api", "argv": ["/home/davin/.local/bin/uv", "sync", "--locked", "--group", "dev"], "timeout": 900 },
  { "id": "web", "argv": ["/home/davin/.nvm/versions/node/v22.23.2/bin/npm", "ci", "--no-audit", "--no-fund"], "cwd": "apps/web", "timeout": 900,
    "env": { "PATH": "/home/davin/.nvm/versions/node/v22.23.2/bin:/usr/bin:/bin" } }
],
"verify": [{
  "id": "full",
  "argv": ["./scripts/verify.sh"],
  "timeout": 1800,
  "network": "loopback",
  "ports": [55432],
  "env": {
    "PATH": "/home/davin/.local/bin:/home/davin/.nvm/versions/node/v22.23.2/bin:/usr/local/bin:/usr/bin:/bin",
    "JARVIS_DATABASE_URL": "postgresql+psycopg://postgres:verify@127.0.0.1:55432/postgres",
    "JARVIS_VERIFY_SYNC": "0"
  }
}]
```

The standing test server, once: `docker run -d --restart unless-stopped --name jarvis-verify-pg -e
POSTGRES_PASSWORD=verify -p 127.0.0.1:55432:5432 postgres:16-alpine`. Until the host runs a remotea build
with `prepare` (remotea branch `target-templates`), drop `prepare` and `JARVIS_VERIFY_SYNC` and set
`"isolation": "none"` on the entry instead: the script then runs as the server user with the network and
docker, syncs dependencies itself and may start its own throwaway Postgres (drop `JARVIS_DATABASE_URL`,
`network` and `ports` too). `remotea doctor` and hello's `verify_status` say whether `./scripts/verify.sh`
resolves (it must be on the checkout's `main`, since readiness is checked against the repository).

The full suite (API 470–670 s plus the web checks) can outlast Eridani's 11-minute verify deadline above while
the host does not advertise the entry's `timeout`; a second, narrower entry (`["./scripts/verify.sh", "--only", "web"]`, or API arguments such as
`["./scripts/verify.sh", "--only", "api", "-x", "tests/test_agent_control.py"]`) stays inside it.

### Setup readiness (E-Q1)

Settings → Connections → remotea shows a guided readiness view (`RemoteaSetup.tsx` `RemoteaReadiness`, data from
`GET /api/v1/remotea/setup`, `agent_control.setup_report`); the remotea page shows the same view above the runs
while nothing is ready. It walks three steps:

1. **Host connected?** Each host with its last hello (`hello_at`, kept in `agent_hosts.caps.hello_at` by every
   hello), version, build, caps version and `degraded_reason`. Problems: `HOST_OFFLINE` (fix: `systemctl --user
   start remotea`, or `remotea eridani setup` if it was never linked) and `HOST_DEGRADED` (free disk space, then
   restart remotea).
2. **Targets advertised?** One entry per host and `host_target`, from hello and from saved targets. Per target:
   each harness (`enabled`, launcher probe `ready`/`version`/`error`, and `harness_problem`'s reason, the same
   refusal `start_run` and the Run dialog give), each verify entry (`ready`, `reason`, and from a host with
   verify isolation `isolated`/`isolation`/`isolation_reason`), and `mcp`. Facts an older host did not send read
   "unknown (host too old)" and are not problems.
3. **Ready to run?** A target is `ready` when none of its problems `blocks`. Problems and their fixes:

| Code | Blocks | Fix (abridged) |
|---|---|---|
| `HOST_MISSING` | yes | run `remotea eridani setup` with a setup code |
| `HOST_OFFLINE` | yes | start remotea |
| `TARGET_NOT_ADVERTISED` | yes | add the target id to `targets[]` in eridani.json (`--add-repo`) |
| `NO_HARNESS` | yes | add `launch.claude` or `launch.codex` and install it |
| `MCP_NOT_CONFIGURED` | yes | New setup code, then `remotea eridani setup` (saves `eridani-mcp-token`); without it the agent cannot submit its plan |
| `HARNESS_NOT_INSTALLED` | no | "Codex is enabled but not installed": install it on the service's PATH, or remove it from `launch` |
| `TARGET_NOT_READY` | yes | the template check failed (remotea S-Q3): fix what each failed `check` names (repo, worktree_root, base, launch, prepare.<id>, limits.disk); `remotea doctor` shows the same check. `NO_HARNESS` and `HARNESS_NOT_INSTALLED` are not added for such a target (it is advertised with no harnesses because of the template), and each harness's `reason` is the template line. Warn-level template problems other than `verify.*` (covered by `VERIFY_NOT_READY`) are notes ("Template migrations: …"); the target also carries `template` and `prepare` |
| `VERIFY_NOT_READY` | no | not on the service PATH: "use scripts/verify.sh or set env.PATH in eridani.json"; demanded bwrap missing: install bubblewrap or set `isolation: "none"`; bad cwd: fix `cwd` |
| `NO_VERIFY` | no | add a `verify[]` entry |

A verify entry that runs unsandboxed because the host lacks bubblewrap (not by the entry's choice) is a note,
not a problem. **Run a readiness probe** on a target calls `POST /remotea/targets/{key}/probe` (`key`: a saved
target id or an advertised `host:<host_id>:<host_target>`). remotea has no probe command yet, so the probe
re-reads the host's last hello and says so: `{source: "hello", hello_at, note, host, target}`; saving
`eridani.json` makes remotea reconnect with fresh facts within 30 seconds, and `remotea eridani status --probe`
checks on the host. When remotea adds a probe command, this route is where it is asked.

**Host self-check (remotea doctor, S-Q1).** Under each host, Settings → Connections → remotea and the readiness
view show a **Host self-check** block (`RemoteaSetup.tsx` `HostSelfCheck`, `remotea-setup.ts` `selfCheck`): the
counts ("2 failing · 1 warning · 30 ok", or "All 30 checks ok"), the report's age ("checked 5m ago", from the
summary's `at`), and each warn and fail line with its fix and check id, fails first; "More findings than shown"
when the host cut the list. A host that advertises `doctor` but has sent nothing yet reads "Not reported yet"; a
host without the cap reads "unknown (host too old)". The data is the host's `doctor` on `GET /remotea/setup` and
`GET /remotea/status` (the `doctor` frame row under [Frames](#frames)). Self-check findings are shown, not turned
into setup problems: they do not change a host's or target's readiness. Eri's `control_status` adds a
`self_check` line to a host whose summary has a failure or warning.

Each unready host or target also has one spoken `line` ("jarvis on desk: Agents on jarvis get no Eridani tools: …
Fix: …"). Eri's `control_status` returns those lines as `setup` (with `setup_hint`) whenever any exists, so voice
and chat can say why a run cannot start; a `control_start` proposal whose target has no problem that makes
confirming fail (only warnings, or `MCP_UNCONFIGURED`, which a freestyle start lifts) but would stall adds "Also
tell the owner before they confirm: <line>" to `next`, so the owner hears the fix.

## Messages Eridani sends to the agent

Besides the spawn prompt, every text that reaches a running agent is an `input` command
(`mode: queue` waits for the current turn to end, `steer` interrupts it). Eridani writes some of them
itself; each of those goes through `run_coordinator.say()`, which also records a `sent` history entry with a
`kind`. The run timeline titles it "Sent by Eridani · <reason>" (`remotea-model.ts` `SENT_REASON`). The
owner's own texts are "Sent by you" or "Sent by Eri, confirmed by you". Wording lives in
[`agent_prompts.py`](../apps/api/jarvis/agent_prompts.py) (`PROMPT_VERSION = "loop-v2"`); task, answer,
gap and reason text inside it is fenced as data.

| `kind` (timeline) | Trigger | Condition | Wording | What the agent is expected to do |
|---|---|---|---|---|
| (spawn prompt, not an input) | Run start | Every run | `interview()` without a plan; `agent_control.spawn_prompt` with a frozen plan | Interview: read the code, ask with `task_interview`, stop. Planned: execute the plan. |
| `interview_answers` · "your interview answers" | Owner answers the interview form (`agent_plan.answer_interview`) | Run `interviewing` (or `needs_you` on `wall_clock`/`no_progress`), current generation | `planning()`: the answers, the offered verify ids, the plan floor | Write the PRD and criteria, call `submit_plan`, stop; do not start the work. |
| `execute` · "plan frozen, start work" | The planning turn settles with a frozen plan (`settle_planning`), or the owner finishes plan review (`review_plan`) | `plan_pending`, no open question or interview, no proposals left | `execute()`: "Your plan is frozen. Execute it now…" plus the rendered plan; after review it opens with what was accepted and kept | Do the work, tick steps with `task_step`. |
| `gap` · "judge gap" | Judge verdict `not_met` (`on_verdict`) | Turn cap not reached, no `verify_missing`, not `no_progress_gap`, freestyle off | `gap()`: the judge's notes and open criteria, then "Close the gap in the code … Then end your turn … If you are blocked on something only the operator can decide, say so plainly in your final message." | Fix the code, make verify pass, commit/push and update the PR if the plan needs one, tick steps, end the turn; name any blocker in the final message. |
| `replan` · "re-plan" | Owner presses Re-plan (`agent_control.replan`) | Executing, needs_you, awaiting_approval/integration or planning | `replan()`: "Stop the current work at a safe point", the reason, "call `submit_plan` … then stop" (`steer` while executing) | Stop at a safe point, revise the plan, `submit_plan`, stop. |
| `verify_set` · "verify command set" | Owner sets the verify command mid-run (`agent_plan.set_verify`) | Only when the run was parked in `needs_you` (it resumes `executing`) | `verify_set()` | Make the work pass that command, end the turn. |
| `merged` · "pull request merged" | The host reports the run's branch merged (`check_integrations`, `recheck_pr`) | `needs_you(awaiting_merge)`, or `pr_unknown` under `pr_merged` | `merged()` | Update the worktree to the merged result, re-run the checks, end the turn. |
| `integrated` · "sub-task pull requests merged" | All required sub-task branches merged into this run's branch (`check_integrations`) | `awaiting_integration` | `integrated()` | Pull the merged sub-tasks, re-check the combined code, end the turn. |
| `plan_critique` · "review sent to the planner" | Hyperplan: a reviewer's turn ends (`finish_review`), or the owner sends the final plan back (`revise_plan`) | The planner's session; `plan_reviewing` → `planning`, or `needs_you(plan_approval)` → `planning` | `incorporate()` (the review fenced as data; `final` on the last round or after a clean approve), `plan_revise()` (the owner's note) | Answer every blocking/major finding in `review_resolutions`, ask with `task_interview` first if needed, `submit_plan`, stop. |
| `handoff` · "plan handed to the worker" | Hyperplan: execution starts on a run whose worker differs from its planner (`hand_off`) | Recorded on the timeline; the text is the worker's spawn prompt, not an input | `worker()`: the frozen plan and the "decisions already made" digest | Execute the plan as written; ask for a re-plan instead of re-opening settled points. |

The owner's texts are not in the table: **Message** (`queue`) and **Steer** (`steer`, advances the
generation) send exactly what was typed (`agent_control.send_message`, also Eri's confirmed
`control.message`), and an answer to the agent's own question is an `answer` command, not an input.

**There are no time-based nudges.** The turn cap, the wall clock (6 h) and the no-progress timer (45 min)
send nothing to the agent: they move the run to `capped` or `needs_you` and push to the owner. So a
"finish up and note any blockers" message is the **judge gap** prompt: each execution turn that ends
`not_met` queues one (its last lines ask the agent to end its turn and say plainly what blocks it). While
the owner is messaging a run, each of the owner's turns is judged when it ends, so two quick exchanges can
queue two gap prompts; Freestyle stops that.

## Scopes

| Scope | Meaning |
|---|---|
| `agents:control` | Open the link, report sessions/turns/status, run parity. Privileged. |
| `agents:session` | Reserved for in-session agent tools (stage C/D: `task_plan`, `task_step`, `submit_plan`). Privileged. |
| `agents:read` | Read runs started from tasks: `GET /api/v1/external/agents/runs[/{run_id}]` and the `run_list` / `run_get` MCP tools. Read-only, not privileged; the same read model as the remotea page. |

Privileged scopes (`bot_access.PRIVILEGED`) are rejected for viewers at key creation and every use is a
write check: `authorize(required="agents:control")` locks membership and fails with `READ_ONLY` if the
account was demoted. The link re-checks on every inbound message and every 30-second tick, so revoking
the key, removing membership or turning remotea off ends the connection. The link key never has
`tasks:write`. The coding agents' own MCP key is separate: the setup code carries one (`remotea MCP`:
`agents:session tasks:write notes:write sessions:write` and their reads, see [The setup code](#0-the-setup-code-one-click-task-bbdb6440)),
or create it in Settings → Connections →
Connected agents with the "Run tools for coding agents" option (adds `agents:session`) plus task, note and record access, and
register it at user scope in each harness (for Claude Code:
`claude mcp add --scope user --transport http eridani https://app.eridani.app/api/v1/external/mcp/ --header "Authorization: Bearer <key>"`;
the URL works with or without the trailing slash). Without `agents:session` the MCP server offers only the planner tools, the
spawned agent cannot call `submit_plan`, and the run ends its planning turn in `needs_you`.

## Connection lifecycle

`wss://app.eridani.app/api/v1/external/agents/link`, subprotocol `eridani.v1`,
`Authorization: Bearer <link key>`, `User-Agent: remotea-server/<version>`, **no `Origin`**.
uvicorn runs with `--ws-max-size 262144 --ws-ping-interval 20 --ws-ping-timeout 30` (`deploy.py`).

The http `security` middleware does not run for WebSockets, so the handshake repeats its checks.
Refusals happen **before** the upgrade, as an HTTP response with the usual error body:

| Status | Code | Cause |
|---|---|---|
| 503 | `MAINTENANCE` | maintenance mode |
| 403 | `NOT_AUTHORIZED` | any `Origin` header (browsers always send one) or `Sec-Fetch-Site: cross-site` |
| 400 | `INVALID_ARGUMENT` | subprotocol `eridani.v1` not offered |
| 401 | `NOT_AUTHORIZED` / `ACCESS_REVOKED` | missing, unknown, expired or revoked key |
| 403 | `INSUFFICIENT_SCOPE` / `READ_ONLY` / `REMOTEA_DISABLED` | no `agents:control`, viewer, or remotea off |
| 429 | `RATE_LIMITED` | more than 120 handshakes/requests per minute for the key |

After the upgrade, Eridani closes with these codes (kept distinct from handshake statuses):

| Code | Meaning |
|---|---|
| 4001 | superseded by a newer connection for the same host (preceded by `bye {reason:"superseded"}`) |
| 4401 | key no longer authenticates (revoked/expired; `bye {reason:"access_revoked"}`). The host waits 5 minutes before retrying |
| 4403 | missing scope, viewer, or remotea turned off (`bye {reason:"access_revoked"|"disabled"}`). Same 5-minute back-off |
| 1000 | normal close after the host's own `bye`, or after a `connect` probe |
| 4400 | protocol error: first frame not `hello`, malformed envelope or JSON |
| 1003 | binary frame |
| 1008 | rate limit exceeded repeatedly (`bye {reason:"rate_limited"}`) |
| 1009 | frame over 256 KiB |
| 1012 | Eridani restarting (`bye {reason:"restart"}` from lifespan shutdown): reconnect with backoff |

**Host lease.** `hello` registers the host by `(workspace, server_id)` and increments its `conn_epoch`;
`lease_until` is now + 90 s and is renewed by every message and every 30-second tick. The newest epoch
owns the host: an older connection in the same process is closed with 4001 at once, and one on another
process is fenced the next time it touches the database. Disconnect clears the lease only if the epoch
still matches, so a superseded socket never marks the live one offline. Startup never resets hosts.

**Wake (when a queued command is sent).** Sockets live in the web processes (`LINKS` in
`agent_link.py` is per process; uvicorn runs several workers, and Railway may run several replicas),
but commands are also queued by the worker (judge gap prompts, interview planning prompts, verify,
replan). `run_coordinator.command()` therefore does two things inside the queuing transaction:

- `wake_after_commit`: after commit, sets the link's wakeup event if *this* process holds the socket
  (no database round trip).
- `agent_link.notify_host`: `SELECT pg_notify('agent_link', <host_id>)`. Postgres delivers a NOTIFY
  only when the transaction commits, and never on rollback, so a link cannot drain before the command
  row is visible. Duplicate notifies in one transaction are folded into one.

Each web process runs `agent_link.listen()` as a lifespan task: one dedicated autocommit psycopg
connection outside the SQLAlchemy pool (`application_name = eridani-agent-link`; LISTEN needs the direct
connection `docs/CLOUD_MIGRATION.md` already requires, not a transaction pooler) that `LISTEN`s on
`agent_link` and calls `wake(host_id)` for each notify. A notify for a host whose socket is in another
process or replica (or nowhere) finds no local link and is ignored; that process's own listener wakes
it. A quiet listener runs `SELECT 1` every 30 s (`LISTEN_IDLE_SECONDS`); any error reconnects with
backoff (0.5 s doubling to 30 s, `LISTEN_BACKOFF`). Notifies sent while it is disconnected are lost, so
every (re)connect wakes all local links once. Waking is only a hint: the drain itself is
`drain()`'s `FOR UPDATE SKIP LOCKED` query, and the 30-second tick (`TICK_SECONDS`) still drains every
link, so a down listener means commands wait for the tick, not that they are lost.

## Frames

Every frame is JSON text: `{v: 1, id, t, at, conn_epoch?, ...}`. `id` is the transport id (≤100 chars).
**Timestamps on the wire are epoch milliseconds** (`at`, `since`, `exited_at`, `ended_at`, `server_time`,
`expires_at`); Eridani also accepts ISO-8601 with a zone inbound. An `outcome` inside an ack keeps the
HTTP API's ISO encoding, so a `session.report` outcome is byte-identical to the HTTP response.
Effectful frames carry a stable `op_id` and a `digest`: sha256 hex of the canonical JSON (keys sorted at
every level, no whitespace, non-ASCII kept) of the frame minus `v, id, conn_epoch, at, op_id, digest`
(`agent_link.digest`). Since stage C the digest is enforced both ways: an inbound effectful frame whose
digest does not match is logged with its `op_id` and answered `ack {ok:false, error:"DIGEST_MISMATCH"}`
(no retry); every command Eridani sends is stamped, and remotea uses Eridani's digest as-is.
A frame carrying a `conn_epoch` other than the current one is answered `STALE_EPOCH`. Unknown fields are
ignored; unknown `t` values get `ack {ok:false, error:{code:"UNSUPPORTED"}}` (capability negotiation: each
side only uses what the other advertised).

Each inbound frame (except `ack`, `bye` and partial chunks) is answered with
`ack {acked_id, op_id?, ok, retry?, outcome?, error?: {code, message, data?}}`. `acked_id` is the envelope
`id` of the frame being acked; `op_id` is echoed. `ok:false, retry:true` (RATE_LIMITED, UNKNOWN_SURFACE,
STALE_EPOCH) means resend later; `ok:false` without `retry` is a permanent refusal. Each frame runs in its
own short transaction.

| Dir | `t` | Body | Effect / outcome |
|---|---|---|---|
| up | `hello` | `server_id, boot_id, hostname, machine, proto, version?, build_sha?, build_dirty?, caps_version?, hook_version?, degraded_reason?, doctor?, caps[], harnesses{}, targets?: [{id, repo, worktree_root, harnesses, enabled?, launch?, verify, verify_status?, mcp?, template?, prepare?}]` | must be first, within 15 s. `doctor` is the host's latest `remotea doctor` summary (see the `doctor` row): kept in `caps.doctor` with `caps.doctor_at`; a hello without it keeps the previous summary while the host still advertises `doctor` (a restarted server sends none until its first check) and drops it when the host no longer does; a malformed one is dropped, never refused. The build fields and `degraded_reason` (only while the host's journal or outbox is degraded) are kept in `caps.build` and returned on the host (above); a malformed value is dropped, never refused. Each target's `harnesses` lists the harnesses it can run, `verify` its verify ids, and a `caps_version: 2` host adds `enabled` (what its `launch` permits), `launch {claude?, codex?: {ready, version, error?}}` (each launcher's probe), `verify_status [{id, ready, reason?, isolated?, isolation?, isolation_reason?}]` (the isolation fields from a host with verify isolation, remotea S-N4) and `mcp {configured, reason?}` (whether spawned agents get Eridani's MCP server); remotea S-Q3 adds `template {ready, problems?: [{check, level: fail\|warn, reason}]}` (the whole target checked before it is advertised; a target that is not ready comes with `harnesses: []`) and `prepare` (its preparation step ids), normalised by `agent_link.hello_targets` (fails first, at most 20 problems, each `reason` one redacted line ≤ 300 characters, at most 8 step ids; a malformed `template` or `prepare` is dropped from its target, never refused); all are kept in `caps.targets` and returned on advertised and saved targets (see **Target capabilities**). The hello's time is kept as `caps.hello_at` (the setup page's probe reports it). A `boot_id` starting `probe-` (`remotea eridani connect`) only checks the key: the welcome carries `probe: true` and the current epoch, no lease is taken, and the probe closes after its `bye` |
| down | `welcome` | `acked_id, host_id, conn_epoch, server_time, replay_horizon (s), lease_seconds, proto, caps[], max_frame` | `caps` is `agent_link.SERVER_CAPS`: what Eridani accepts, including `prompt` and `doctor` (the host sends those frames only when they are listed) |
| up | `session.report` | `tool, session_id, report: SessionInput, surf?, incarnation?, parity?` | same upsert as `PUT /external/sessions/{tool}/{id}`, attached to the host; outcome `{session}` identical to the HTTP response. With `parity: true`, compares instead (below) |
| up | `turn` | `tool, surf, incarnation, run_id?, generation, provider_session_id, provider_turn_id, root, seq, text, truncated, missing?, prompt?, source: hook\|rollout, ended_at?` | stored once per `(session, provider_turn_id)`: a resend with a new transport id → `{duplicate:true}`; `root:false` → `{ignored:"subagent"}`; text capped at 64 KiB (UTF-8) with `truncated` set; a `run_id` must name a run bound to this host (`RUN_NOT_BOUND` otherwise); a turn of a spawned run attaches the session (below) and counts toward the turn cap. `prompt` (the message typed in the terminal, ≤ 4000 bytes) is kept on the turn only when no `prompt` frame with the same `seq` and text was stored first |
| up | `prompt` | `surf, incarnation, run_id?, generation, tool, provider, provider_session_id, seq, text, truncated, at` | the message typed in the terminal, sent on UserPromptSubmit (as the turn starts, not when it ends); `seq` pairs with the following `turn`'s `seq`; text clipped to 4000 bytes by the host and again here. Stored as a `user` row of `agent_turns` (`provider_turn_id` = `prompt:<seq>`) once per `(session, seq)`: a resend → `{duplicate:true}`; a `run_id` must name a run bound to this host (`RUN_NOT_BOUND`). It never counts toward the turn cap and never moves the run; the host sends it only when Eridani's `welcome` caps include `prompt` |
| up | `status` | `surf, incarnation, state, since, reason?, effective_permission_mode?, provider?+provider_session_id? (or tool?+session_id?)` | sets the session's `state`/`state_since`. Older incarnation, or same incarnation with an earlier `since`, is ignored (`applied:false`). Without a provider (or tool) and session id, the surface maps to the newest session reported with that `surf` on this host (`UNKNOWN_SURFACE` otherwise) |
| up | `exit` | `surf, incarnation, exited_at?, code?, provider?+provider_session_id?` | state `exited` at `exited_at` |
| up | `sessions.snapshot` | `surfaces: [{surf, incarnation, state: live\|lost\|exited, agent_state, since, provider, provider_session_id, …}], part?: {of, i, n}` | a large snapshot arrives as several frames sharing `part.of`; non-final parts are acked `{pending:true}` and reconciliation runs once all `n` parts arrived. `live` takes the surface's `agent_state` (working/needs_you/idle). Applies each surface as `status`; when `complete`, sessions of this host missing from the snapshot become `exited` (`reason: absent_from_snapshot`) unless their state is newer than `at`. Eridani first reconciles runs (below), then replies `runs.snapshot {runs: [{run_id, generation, desired: continue\|stop\|none, stop_mode?}]}` and only then resends unacked commands |
| up | `chunk` | `part: {of, i, n}, data` | parts of one logical frame (n ≤ 64, ≤ 1 MiB total, ≤ 4 in flight); the ack names the logical frame's id. Eridani chunks its own frames above 256 KiB the same way |
| up | `ack` | `acked_id, op_id?, ok, error?, retry?, outcome?` | acknowledges a command; an accepted spawn carries `outcome {state:"accepted", surf, incarnation}` |
| up | `command.outcome` | `command_id (= op_id), kind?, state: executed\|failed\|rejected\|uncertain, detail, result?` | recorded on the command and applied to its run (below) |
| up | `question` | `op_id, digest, surf, incarnation, run_id?, generation, provider, provider_session_id, provider_request_id, provider_turn_id?, items[], hold_until?` | a structured question the agent asked; stored once per `(session, provider_request_id)` |
| up | `question.resolved` | `op_id, digest, question_id\|provider_request_id, status: accepted\|answered_locally\|timed_out\|canceled\|failed, still_open?, remote_answerable?` | the host's word on a question; only `accepted` confirms an answer sent from here |
| up | `context.request` | `req_id, text, run_token?` | ephemeral, no ack: answered with `context.response {req_id, text}` |
| up | `worktrees.report` | `op_id, digest, worktrees: [{target, path, branch, base, base_ref?, ahead, behind, unpushed, uncommitted, untracked, merged, merged_reason?, active, reasons}], heads?: [{target, dir, ref, head}]` | replaces the host's last report; feeds **Unmerged work**, the run header's live "N behind" and the migration counter (0033). `merged` is null with `merged_reason: "no commits of its own yet"` for a branch with no commits of its own; Eridani reads null (and a branch missing from the report) as unknown, never merged. Stored with the time it arrived (`caps.worktrees.at`), which the judge compares with the turn |
| up | `doctor` | `op_id, digest, at, ok, warn, fail, items: [{id, level: warn\|fail, summary, fix?}], truncated?` | `remotea doctor`'s summary (remotea S-Q1): the counts cover every check, `items` only the warn and fail lines (≤ 40, `id` ≤ 120 and `summary`/`fix` ≤ 200 characters). Sent only when Eridani's `welcome` caps include `doctor` (Eridani advertises it) and only when the findings changed. Eridani keeps the newest per host in `agent_hosts.caps.doctor` (`at` as ISO-8601) with `caps.doctor_at` (when it arrived; no migration); an older `at` never replaces a newer one (`{stored:false, reason:"older"}`). Validated like the hello field (`agent_link.doctor_summary`): bad counts or `at` drop the whole summary (`{stored:false, reason:"malformed"}`, still `ok:true` so the host does not resend it); a malformed or oversized item is dropped alone and the summary marked `truncated`; a malformed `fix` is dropped from its item; text is redacted again. Fails first. Shown on the host (`public_host.doctor` with `received_at`, so `GET /remotea/status`), on the setup page's hosts (`GET /remotea/setup`, with `doctor_supported`) and in `control_status` (`hosts[].self_check`: "host desk: 2 setup problems: <first fail>…") |
| down | `runs.snapshot` | `runs: [{run_id, generation, desired, stop_mode?}]` | after each reconnect's reconciliation |
| down | `spawn` / `input` / `answer` / `stop` / `verify` | see **Commands** | outbox commands |
| down | `worktrees.scan` | `branch, turn_seq, tool?, session_id?` (outbox command, only to a host whose `hello.caps` has `worktrees.scan`) | the judge needs a PR record newer than the turn: the host rescans its worktrees and sends `worktrees.report` even if unchanged, re-reads the named session and sends its `session.report` even if its fingerprint is unchanged, then acks (`executed`). An older host never gets it; the judge waits for its own tick instead. **Rescan** on the worktree maintenance view (`POST /remotea/hosts/{id}/rescan`) sends a run-less one: `run_id: "rescan", generation: 0` (the fence is informational for this command), no `tool`/`session_id`, 2 min expiry, one waiting per host |
| down | `terminal.focus` | `surf, run_id, generation, expires_at` (outbox command, only to a host whose `hello.caps` has `terminal.focus`) | "Open terminal in remotea": the host asks its connected desktop app(s) to bring the run's tab to the front. `executed` (`result.desktops`) when at least one app was told; `rejected` `no_desktop` when none is connected (or none understands it), `unknown_surf` when the surface is gone or not the run's. Quiet: no timeline entry or history note, never moves the run, and a queued one is rejected at reconcile (it never survives a reconnect) |
| down | `context.response` | `req_id, text` | |
| up | `bye` | `reason: shutdown\|reconfigured\|probe` | Eridani closes 1000 |
| down | `bye` | `reason: superseded\|access_revoked\|disabled\|rate_limited\|restart` | followed by close |

`state` values: `working`, `needs_you`, `idle` (shown as **Live**), `restoring`, `lost` (shown as
**Restoring**), `exited` (**Ended**). `since` must carry a timezone. Every stored change emits
`session.changed` (report, status, exit, snapshot) or `agent.changed` (turn or prompt stored, host connected or
disconnected) on the change feed. The web client skips both in the workspace reload and forwards them as
`eri-agents-changed` to the remotea page and Sessions drawer.

Per-host bounds: 1,200 frames and 16 MiB per minute; over that, frames are answered `RATE_LIMITED`, and
the fourth time the connection closes with 1008.

**Commands (stage C).** `agent_commands` is the outbox: a domain change writes the command in the same
transaction; each tick, `drain()` sends queued commands (and resends `sent` ones unacked for 60 s) whose
kind the host advertised and whose `expires_at` has not passed. Commands carry a stable `op_id` and
payload `digest`, separate from the envelope id: a resend keeps both and gets a new envelope (F06). See
**Runs** below.

## Testing the protocol end to end

`tests/test_agent_link.py` drives Eridani with a fake host and remotea's `remote-server/tests` drive remotea
with a fake Eridani, so each side can drift from the other while both suites stay green.
`tests/test_protocol_e2e.py` (marker `e2e_protocol`, helpers in `tests/protocol_harness.py`) runs the real two:

```
remotea-server (node, the checkout's remote-server/dist) --ws--> LinkProxy --ws--> uvicorn jarvis.api:app --> test database
```

- **Eridani** is uvicorn on an ephemeral port with `JARVIS_ENV_FILE=""`, against the database `conftest.py`
  made for the session, with production's websocket limits. The worker does not run: the test process plays it
  and runs the judge job with a stubbed model (`agent_judge.process(job, model=…)`); its commands reach the
  link through `pg_notify`, as the worker's do.
- **remotea** runs from `$REMOTEA_CHECKOUT` (default `~/remotea/remotea`, or a `remotea` folder beside this
  repository) with a throwaway config dir, `HOME`, repository (with a bare origin) and worktree root, on
  ephemeral ports. It never reads `~/.config/remotea` or `~/.claude` and never touches `remotea.service`. Its
  one target launches `claude` as a stub script that drives the real hook helper
  (`resources/cli/wmux-hook.js`): SessionStart, then UserPromptSubmit + Stop for the spawn prompt and for each
  input typed into it (bracketed paste, then Enter); Ctrl-C ends it. Verify is a fake command that prints
  `3 passed`. The stub answers `--version`, so the launcher probe passes and the hello advertises `spawn`.
- **LinkProxy** relays the link and records every frame both ways before any rewrite. It can drop acks, cut
  the connection, re-send a frame, and make either side look older.

One run, every step bounded by 30 s (about 10 s in all): connect → hello/welcome/snapshot → the run started
over the browser routes (`POST /remotea/targets`, `POST /remotea/runs`) → `spawn` (accepted ack, executed
outcome) → `context.request`/`context.response` → `prompt` → `turn` → `verify` (Eridani's command adopts the
host's attempt; one run) → judging. Then a **reconnect with duplicates**: the proxy has kept Eridani's acks for
the first `prompt` and `turn` from the host, cuts the link, and the host resends both on the new epoch;
Eridani answers `{duplicate: true}` and stores neither again. The proxy then resends the `spawn` (same op_id and
digest, new envelope, new epoch, as Eridani's drain does): the host acks the outcome it journaled and starts
nothing. The new connection is also an **older peer** both ways: the hello Eridani sees lacks `caps_version`,
build identity and the `prompt`, `worktrees.scan`, `digest`, `context.scope` and `verify.auto` caps; the
welcome the host sees lacks `prompt` and `context.scope`. Then the judge says `not_met` → the gap is typed into
the agent → turn 2 comes **without** a `prompt` frame and carries the typed text on the turn, which Eridani
keeps on the assistant turn → verify → judge `met` → `awaiting_approval` → Approve → `stop` → exit → run
`completed`, task completed. Finally: no `ok:false` ack in either direction, each prompt reached the agent once,
one verify run per turn.

**Wire fixtures.** `tests/fixtures/protocol/<caps_version>/` (the host hello's `caps_version`) holds one file
per frame kind and direction: `host.turn.json`, `eridani.spawn.json`, acks keyed by what they acknowledge
(`host.ack.spawn.json`, `eridani.ack.turn.json`). Each is the **shape** of what that side actually sent in the
run, never values: keys, whether each was always present, and the JSON types, merged over every frame of that
kind. A later run fails when a key recorded as always present is missing (a rename shows as that plus a new
key), when a key appears that was never recorded, or when a value has a new type (`null` matches anything).
Optional keys that depend on the machine rather than the protocol (`launch_error`, `error`, `reason`,
`degraded_reason`, `build_dirty`, `migrations`) are never reported as new. A frame kind with no file yet is
recorded on that run. After an intended protocol change on either side, re-record with `UPDATE_FIXTURES=1` and
review the diff like code: it is the contract change.

Run it locally (Postgres as in `docs/ADDING_FEATURES.md`). Build the checkout first: a stale `dist/` is an
older host, and its fixtures land under another `caps_version`.

```sh
npm --prefix ~/remotea/remotea/remote-server ci          # once
npm --prefix ~/remotea/remotea/remote-server run build   # after every remotea change
export JARVIS_ENV_FILE="" JARVIS_DATABASE_URL=postgresql+psycopg://postgres:test@127.0.0.1:54432/jarvis
REMOTEA_CHECKOUT=~/remotea/remotea uv run --no-sync python -m pytest -q -s -m e2e_protocol tests/test_protocol_e2e.py
UPDATE_FIXTURES=1 REMOTEA_CHECKOUT=… uv run --no-sync python -m pytest -q -s tests/test_protocol_e2e.py   # re-record
PROTOCOL_E2E_FRAMES=/tmp/frames.json …   # also write every frame both ways (test tokens only) for review
```

With `-s` it prints the remotea build it ran against, the frame count and the time. It skips, with the reason,
when `node` or `git` is not on PATH, the checkout is missing, or its `remote-server/dist` or `node_modules` is.
CI has no remotea checkout (the repository is private), so CI skips it; see `docs/CI.md`.

## Collector parity

The TypeScript collector inside remotea must match the Python collector
([`scripts/session_collector`](../scripts/session_collector/README.md)) before it takes over. In
**parity mode** it sends what it would report to

- `POST /api/v1/external/agents/parity` (`agents:control`) with `{tool, session_id, report}`, or
- a `session.report` frame with `parity: true`.

Eridani does **not** write the session. It normalizes the report exactly as the upsert would (folder
spelling, empty `resume` keys) and compares `machine, environment, cwd, repo, branch, title, first_prompt,
last_prompt, excerpt, pr_urls, turn_count, started_at, last_active_at, resume` with the row the Python
collector stored, keeping the latest result per session in `agent_parity`. The response is
`{matched, missing, fields}`. `resume` is compared only on its stable keys (`cmd`, `url`): `live` and `hint`
describe the session's current remotea terminal and change over time, so both sides ignore them.

`GET /api/v1/remotea/parity` (and the page's **Collector parity** panel) summarizes `total, matched,
mismatched, missing, clean`, a count per differing field, first/last report times and the 20 most recent
problems with stored vs reported values (each clipped to 300 characters).

Procedure (plan decision 23): run both collectors; the Python one stays authoritative and keeps its timer.
Investigate every differing field (timestamps, Windows/WSL path coverage, redaction, fingerprint and
checkpoint semantics). Cut over only after the summary has been **clean for 7 days** and the host release
is installed: then disable the Python timer, and remove its code and revoke its credential one release
later. Do not delete the Python collector before that. Stage E counts the days for you and lists the exact
steps: see [Cutover](#cutover-stage-e).

## Browser routes

All under `/api/v1/remotea`, cookie auth:

| Route | Gated | |
|---|---|---|
| `GET /status` | no | enabled, key, mcp_key, setup_command, hosts, connect line |
| `POST /enable` `{rotate?}` | no | enable; `token` and `setup_code` in this response only (both keys the first time, the link key alone on rotate) |
| `POST /disable` | no | revokes the link key and the MCP key |
| `POST /setup-code` `{machine?}` | yes | New setup code: replaces the link key and the MCP key; `token` and `setup_code` in this response only |
| `GET /setup` | yes | the setup page: `{ready, hosts[], targets[], lines[]}` ([Setup readiness](#setup-readiness-e-q1)) |
| `POST /targets/{key}/probe` | yes | one target's readiness (`key`: saved target id or `host:<host_id>:<host_target>`); from the last hello while remotea has no probe command (`source: "hello"`, `note`) |
| `GET /sessions?limit=` | yes | host-reported sessions (live, plus those ended in the last 24 h) with host and the last turn (first 2,000 characters) |
| `GET /sessions/{id}/turns?limit=` | yes | recent final responses in full |
| `GET /parity` | yes | parity summary |

The page refreshes on `agent.changed`/`session.changed` and polls every 10 seconds as a fallback. The
Sessions drawer shows **Live/Restoring/Ended in remotea · since …** from the server's `state_since` when a
host has reported the session; collector-only sessions keep the "Reported live … collector snapshot"
wording.

## Task plan fields (migration 0026)

Tasks gain `steps` (ordered, stable `id`s; each `{id, title, status: todo|doing|done|skipped|disputed,
note, task_id?, by, updated_at, revision}`), `prd_note_id` (a note in the same workspace),
`success_criteria` (`[{id, text, kind: text|verify, verify_id?}]`), `verify` (`{target_id, verify_id}`),
`turn_cap` (default 15), `execution_target_id` (foreign key arrives with `execution_targets` in 0028) and
the server-managed `plan_revision`. They are edited through `task.update` (browser, Eri, bots), normalized
by `task_plan.normalize`: missing ids are assigned, unchanged steps keep their stamp, changed steps record
who changed them, and `plan_revision` advances only when the plan actually changes. Plan fields never
queue a Linear write. A run freezes a plan revision when it starts; agents tick steps with `task_step` (below).

## Runs (stage C, migration 0028)

A **run** is Eridani driving one agent on one task through remotea. Tables: `execution_targets`,
`agent_runs`, `agent_run_plans` (frozen plan revisions), `agent_attempts`, `agent_interviews`,
`autolink_rejections`; `run_id` foreign keys on `agent_sessions`, `agent_commands`, `agent_turns` and
`agent_questions`; `tasks.execution_target_id` gets its foreign key. At most one unsettled run per task
(partial unique index).

### Lifecycle

`run_coordinator.TRANSITIONS` is the only map of legal moves; `transition()` is a compare-and-set on
`state_revision` (`REVISION_CONFLICT` when another writer won, `ILLEGAL_TRANSITION` otherwise) and appends
`{at, from, to, reason, detail, actor}` to `history`. Every change emits `agent.changed`.

```
queued → spawning → executing ⇄ waiting_answer
                       ↓  ↑
                    needs_you ⇄ awaiting_approval
any unsettled → stopping → stopped | completed | failed        executing → capped | failed
```

Stage C enters `spawning`, `executing`, `waiting_answer`, `needs_you`, `awaiting_approval`, `stopping`
and the terminal states; stage D adds `interviewing/planning/verifying/judging/awaiting_integration/
replanning` and `capped` (see "Planning and the loop" below). Rules:

- **Nothing is assumed.** A spawn is confirmed by the provider session showing up (first `turn` or
  `status` naming the run, or a snapshot surface), a stop by an exit, a missing surface or a stop outcome,
  an answer by `question.resolved {status:"accepted"}`. A timeout is never a resolution (F04).
- **Stop is desired state** (F03). `POST …/stop` moves to `stopping`, advances `generation`, records
  `stop_mode`/`stop_reason`/`stop_final` and queues `stop`; `runs.snapshot` repeats the desire on every
  reconnect. A spawn never sent is canceled and the run settles at once. A second stop with `mode: kill` while
  stopping escalates (the page's **Force stop**).
- **Generation fences** (F07): steer and stop advance it; the host drops `queue` input of an older
  generation and steer/stop older than its current one; Eridani ignores turns and questions of older
  generations.
- **needs_you reasons** (`state_reason`, shown with `reason_text`): `exited`, `surface_lost`,
  `host_offline` (lease gone 5 minutes; both clear themselves when the surface reports again),
  `delivery_uncertain`, `spawn_expired` (15 minutes), `wall_clock` (6 h of working time, below), `no_progress`
  (45 min), `turn_cap` (C pauses instead of `capped`), `question_timed_out`; hyperplan adds `plan_approval` and
  `host_busy` (see [Hyperplan and roles](#hyperplan-and-roles-migration-0041)). Pushes: `agent_stuck` on needs_you,
  `agent_question` on a question, `agent_done` on awaiting_approval; each opens
  `/?view=remotea&run=<id>`.
- **The wall clock counts working time, not owner waits** (migration 0043). A run waits on the owner while an
  interview for its current generation is open (from `task_interview` until it is answered or superseded), in
  `waiting_answer` and in any `needs_you` (including `plan_review`, `plan_approval`, `host_busy`); hyperplan's
  `plan_reviewing` is not a wait (the reviewer works). `agent_runs.owner_wait_since` marks the wait in progress and
  `owner_wait_total_s` sums the finished ones; the deadline is `started_at + wall_clock_limit + owner_wait_total_s`
  (`run_coordinator.wall_clock_deadline`), and a run waiting now never trips `wall_clock`. When a wait ends the run
  token's expiry slides forward by the same amount (from the stored expiry, so a token that lapsed during the wait
  works again; while the wait lasts the token's expiry already counts it). Resuming a run parked on `wall_clock`
  (a message, steer, re-plan, interview answer or any other move out of it) adds 2 h to `wall_clock_limit`
  (`WALL_CLOCK_GRANT`, more if the run was further past its deadline) and to the token's expiry, recorded on the
  timeline as `{event: "wall_clock_extended", seconds}`, so it does not bounce straight back.
- **Approve** completes the task as a human (`task.complete`); the run moves to `stopping` with
  `stop_final: completed` and becomes `completed` once the agent is gone.

### Starting a run

`POST /remotea/runs {task_id, harness?, model?, execution_target_id?, interview?, freestyle?, reviewer?, worker?,
rounds?, approve_final?}` (the last four are hyperplan's, see [Hyperplan and roles](#hyperplan-and-roles-migration-0041)):

1. Target resolution: explicit → the task's `execution_target_id` → a target saved for the task's
   record or an ancestor (nearest first) → folder rules whose path is under a target's repo → `409
   NEEDS_TARGET {task_id, candidates, advertised, records: [{id, title, type_id}]}`. The page's picker
   then saves one with `POST /remotea/targets {host_id, host_target, repo, worktree_root, task_id | record_id}`
   and retries. "Remember for" defaults to the nearest record that is not a task (usually the project), so
   sibling tasks reuse the choice; "Only this task" binds just the task (reusing a target already saved
   for a record rather than copying it).
2. Refusals: `RUN_ACTIVE`, `HOST_OFFLINE`, `HARNESS_UNAVAILABLE` (the host lacks `spawn` for the harness, or the
   resolved target does not enable it; the message names the target and `data` is `{harness, host_target}`),
   `HARNESS_DISABLED` (opencode), `MCP_UNCONFIGURED` (the target's hello says `mcp.configured: false`: agents
   there get no Eridani tools and could never call `submit_plan`; refused unless `freestyle: true`),
   `TARGET_NOT_READY` (remotea S-Q3: the target's hello `template.ready` is false; checked before the harness, so
   it is not reported as "does not enable <harness>". The message is "The <target> target is not ready: <check>:
   <reason>; …" from the fail-level problems, and `data` is `{host_target, template}`).
3. The run is created in `spawning`, the plan is frozen as revision 1 (PRD note or task notes, steps,
   criteria), a run token is minted and `spawn` is queued in the same transaction. `freestyle: true` starts the
   run in freestyle (below) and skips the interview: the agent works on the task as written, turns are recorded
   but not verified or judged.

Branch: Linear's `branchName`, else `<owner>/<identifier>-<slug>`, else `task/<id>`; base: the parent
task's run branch, else `main`.

**Target capabilities.** `agent_control.host_harness(host, harness, host_target)` decides whether a harness can
start: the host must list `spawn` in `caps` and set `caps.harnesses.<harness>.spawn`, and when its hello entry for
`host_target` carries `harnesses`, that list must include the harness (remotea refuses a harness missing from the
target's `launch`). A target without `harnesses` (an older remotea) falls back to the host-wide flags alone.
`check_host` passes the resolved target's `host_target`, so `start_run`, Eri's `control_start` proposal and
confirm all refuse the same way. Advertised targets (`GET /remotea/targets` `advertised`, `NEEDS_TARGET`
`data.advertised`) carry `harnesses` (null when the host did not say) and `verify`; a saved target carries the
same two fields when its host advertises it. From a `caps_version: 2` host both also carry `enabled`, `launch`,
`verify_status` and `mcp` (null on an advertised target when the host did not say, absent on a saved one). The
Run dialog's preflight reports `harnesses` for the resolved target only, with `harness_reasons` (why each one
cannot start, naming the target; a harness the target enables whose launcher failed its probe reads "Codex is
enabled on jarvis but not installed: <launch.codex.error>", and a host-wide failure adds
`harnesses.<h>.launch_error`); while no target resolves, it reports the union over the picker's candidates and
advertised targets and sets `harness_note`. Another host's capabilities never make the resolved target look ready.
For the resolved target the preflight also reports `verify_status` (as advertised, with `timeout`, `isolated`,
`isolation` and `isolation_reason` when the host sends them), `verify_ready` (the task's verify id when it names one,
else whether **any** id the target offers is ready, since the default pick prefers a ready one; null when the host
did not say or there are none), `verify_reasons` `{id: reason}`, `mcp` and `template` (remotea S-Q3; null when
the host did not say). The picker (`remotea-model.ts` `targetOptions`) adds "not ready: <check>: <reason>; …" to a
target whose template is not ready, and the setup view's target card a "Template: ready · prepares api, web" /
"Template: not ready (2 failing checks)" line.

`problems` (from `readiness_for`, shared by the preflight, Eri's `control_start` proposal and the remotea card) are
`{code, message, blocking, freestyle_ok?}`. The `check_host` refusals block. `agent_control.target_problems` adds,
from the target's hello: `MCP_UNCONFIGURED` "Agents on this target have no Eridani tools: <reason>" (blocks Start
unless the run starts in freestyle, `freestyle_ok: true`) and `VERIFY_NOT_READY` "This target's verify command is
not ready (<id>: <reason>; …)" (when the task's verify id, or every offered id, is not ready; `blocking: false`: the
run still starts and records the failed verification). Eri tells the owner a blocking problem makes confirming fail,
and a warning only that verification will not pass until the host is fixed. An older Eridani's problems have no
`blocking` and all block.

The dialog disables a harness the preflight rules out and shows the reason, notes "This target's verify command is
not ready (…)" and "Agents on this target have no Eridani tools: <reason>", keeps **Start** disabled while a
blocking problem stands (`remotea-model.ts` `startBlock`) and, when the only blocks are `freestyle_ok`, offers
**Start in freestyle** (sent as `freestyle: true`); only
without a preflight does it fall back to "any connected host can spawn it". When the dialog's target picker saves
a target, the dialog reads the preflight again with that `execution_target_id` and waits for **Start** instead
of starting at once. `GET /runs/{id}` adds `verify_readiness {id: {ready, reason?}}` beside `verify_options`; the
header's verify line says when the chosen (or an offered) command is not ready on the host.

When the preflight shows something the guided setup view fixes (`remotea-model.ts` `setupReason`: a target to
choose, a problem in `SETUP_PROBLEMS` such as `HOST_OFFLINE`, `HARNESS_UNAVAILABLE`, `MCP_UNCONFIGURED`,
`VERIFY_NOT_READY` or `TARGET_NOT_READY`, or the picked agent ruled out for the target), the dialog adds **Open
remotea setup** (its tooltip is the problem): it closes the dialog, opens Settings → Connections and scrolls to the
readiness view ([Setup readiness](#setup-readiness-e-q1), `remotea-setup.ts` `revealSetup`).

### Commands

All commands: `{t, command_id (= op_id), op_id, digest, run_id, generation, expires_at (epoch ms), …}`.

| `t` | Body | TTL |
|---|---|---|
| `spawn` | `attempt, harness, model?, target: {id, repo?, branch, base}, task_id, title, prompt, run_token, env {ERIDANI_*}, group` | 15 min |
| `input` | `surf, incarnation, text, mode: queue\|steer` | 10 min |
| `answer` | `surf, incarnation, question_id, provider_request_id, answers: [{item_id, choice_ids?, text?}]` | 10 min |
| `stop` | `surf, incarnation, mode: interrupt\|kill` | 24 h |
| `verify` | `attempt_id, turn_seq, plan_revision, verify_id` (stage D) | 10 min |
| `worktrees.scan` | `branch, turn_seq, tool?, session_id?` (the judge's PR evidence, below) | 2 min |
| `terminal.focus` | `surf` ("Open terminal in remotea", below) | 30 s |

Commands are rows in `agent_commands` (status `queued`). Whichever process queues one, the host's
socket drains it within about a second of the commit (see "Wake" under Connection lifecycle), or at
the next 30-second tick if no wake arrives.

**Deterministic spawn surface.** remotea names a spawned surface
`surf-eri-<sha256(run_id + "\0" + generation + "\0" + attempt) hex, first 32>` (`run_coordinator.surf_for`).
When a spawn's ack or outcome is lost, reconciliation recognizes the run's surface in `sessions.snapshot`
by that id (or by `run_id`) and records it as accepted instead of spawning again.

Outcomes: an accepted spawn ack or `executed` outcome records `surf`, `incarnation`, `worktree`,
`branch`, `base`, `base_sha`; a failed/rejected spawn fails the run (`spawn_failed` / `spawn_rejected`), except the
two remotea S-Q3 rejections (`run_coordinator.spawn_refusal`): `target_not_ready: <reasons>` (result `{template}`:
the template broke since hello) fails it as **`target_not_ready`** with the failed checks and "fix the target on
the host" as detail, and `target_busy` (result `{limit, running}`: the target is at its `limits.maxRuns`) as
**`target_busy`** with "N of M runs allowed on this target are live. Start again when one finishes, or raise
limits.maxRuns on the target."; `uncertain` input moves to
needs_you `delivery_uncertain` (a `worktrees.scan` or `terminal.focus` outcome never moves the run); a failed answer reopens the question; an executed stop settles a stopping
run.

**Reconnect.** After `sessions.snapshot`, `reconcile()` matches each unsettled run to its surface (by
`run_id` or the spawn surf), recovers lost spawn acks, settles stopping runs whose surface is gone,
moves runs whose surface is missing to needs_you, marks queued/sent commands that no longer apply
`rejected` ("stale after reconcile"), sends `runs.snapshot`, and only then drains the outbox.

### Questions

A `question` stores `items` (`{id, text, header?, choices?: [{id, label, description?}], multi?}`) and
moves an executing run to `waiting_answer`. The page offers choice buttons and free text only when the
host advertised structured answers for the harness (`harnesses[tool].question`) and the question is
`remote_answerable`; otherwise it says to answer in the terminal. `POST /remotea/questions/{id}/answer
{answers}` requires every item (`QUESTION_CLOSED`, `ANSWER_PENDING`, `STALE_GENERATION` refusals),
marks it `submitted` and queues `answer`; the run resumes on `question.resolved`. `timed_out` (host
`hold_until` passed, or the sweep) leaves it answerable only in the terminal.

### Actors and the completion gate

`agent_actor` decides who is acting: an explicitly bound actor (`acting(Actor(...))`), else `eri_work`
inside Eri's execution context, else `bot` for an unbound bot key, else `human`. Authorities are
`human`, `source_sync` (the Linear import always runs as it) and `run_judge` (the coordinator).

While a task has an unsettled run, `task.update`/`task.complete`/`task.batch`/`record.update` refuse
completion and edits of the gate fields (`agent_actor.GATE_FIELDS`: `success_criteria`, `verify`, `turn_cap`,
`completion_policy` and `prd_note_id`, so a `tasks:write` key cannot repoint a live run's PRD) from `eri_work`,
`run_session` and `bot` with `409 RUN_OWNS_COMPLETION {run_id, state}`. A human or source_sync completion is recorded at
once and asks the run to stop (`stop_reason: task_completed_externally`); a human approve ends it
`completed`.

The same unsettled run makes its task's PRD note (`tasks.prd_note_id`) read-only to agents
(`agent_actor.note_gate`). `note.update` (including `archived`), `note.append`, `note.replace`,
`note.file`, `note.organize` and a revert of a note change refuse every actor but `human` (so `bot`,
`eri_work`, `run_session`, `run_judge` and `source_sync`) with `409 RUN_OWNS_PRD {run_id, state,
task_id}`. The one exception is `submit_plan` rewriting its own run's PRD (`run_session` with `plan_write`
and that run's id). `note.tasks` stays open: it links to-dos quoted from the note and never changes it. A
human may still edit; the result carries `warning` saying the edit changes the note, not the frozen
plan the run executes and is judged against (`agent_run_plans.prd_text`). `GET /notes/{id}` and the
external note record carry `frozen_run_id`, and the note editor shows "Frozen with run <id>". The
gate is lifted as soon as the run is terminal.

### Reading runs (`agents:read`)

A bot key with `agents:read` (Settings → Connections → Connected agents → "Read runs started from tasks") reads what the
remotea page shows, so an assistant outside the run can review a run without a copy-paste:

- `GET /api/v1/external/agents/runs?task_id=&active=&limit=` (MCP `run_list`): runs newest first, the
  list shape of `agent_control.public_run` (state, `state_reason`, `reason_text`, task, branch, turns, host).
- `GET /api/v1/external/agents/runs/{run_id}?limit=&offset=` (MCP `run_get`): the detail shape — frozen
  `plan` (PRD, steps, criteria, verify), `history` (every transition with reason and actor), `turns`
  (`{id, seq, role, text, clipped, truncated, generation, at, prompt}`: `role` is `assistant` for the agent's
  final response or `user` for the message typed in the terminal, stored from the host's `prompt` frame; an
  assistant turn's `prompt` (≤ 500 chars here) carries the typed message only when the host sent it with the
  Stop alone, as older hosts do, so a message never appears twice),
  `commands` (spawn, input incl. the operator's and the judge's gap prompts, stop, verify), `questions`,
  `interviews` (questions and answers), and `attempts` (verify and judge: verdict, gap, per-criterion
  record, rules, model, cost). Turns and commands page newest first (default 100 each, `limit` ≤ 200);
  `more {turns, commands, offset, limit, remaining {turns, commands}}` says whether older ones remain and how many.
  A turn's `text` is cut at 4,000 characters for display (`clipped: true`; the browser route
  `GET /remotea/runs/{id}/turns/{turn_id}` returns it whole), which is not the same as `truncated` (the host cut the
  final response, or the daily quota dropped it; the rest was never stored). `retention {expires_at, expired}`: a
  settled run's turns and commands are deleted 90 days after it ended (`agent_retention.py`); `expires_at` is null
  while the run is unsettled. A verify attempt adds `isolated`, `isolation`, `isolation_reason`, `workspace_complete`,
  `workspace_reasons` (remotea PR #24; null from older hosts), `stale_workspace`, `timeout_s` and `deadline_at`.
- **Freshness (E-S6).** Both shapes carry the facts a decision rests on, each with its own time, kept apart from the
  run's state (`state_since`): `host.last_seen_at` (the host's lease, renewed every 30 s while connected),
  `last_turn_at` (the newest turn of the run or its session; null before the first), `verification` (the latest
  verification Eridani matched: `{verify_id, status, at, evidence, turn_seq, isolated, isolation_reason,
  workspace_complete, workspace_reasons, deadline_at}`, null before the first; `evidence` is `fresh` for this
  generation, turn and plan revision on an unchanged workspace, `stale` for an older tuple or a checkout that changed
  around it, `incomplete` when the fingerprint missed part of the workspace, `pending` while it runs) and
  `pending_controls` (commands sent since the run entered its state whose effect is not known: `{kind, status:
  pending|uncertain, mode?, at}`, newest first, at most five; `pending` means the host has not taken it yet).

Reads stay available after remotea is turned off, as on the page. Owner scoping is the run's `owner_id`;
another workspace's run is `404`. Nothing here mutates: approving, messaging and stopping remain
cookie-authenticated browser routes.

### In-session tools (`agents:session`)

`task_plan` and `task_step` (MCP tools, and `POST /api/v1/external/agents/session/plan|step`) take
`run_token` (from `ERIDANI_RUN_TOKEN`, or the plan text) or `tool + session_id`. A valid token binds a
verified `run_session` actor; a bad one is `401 RUN_TOKEN_INVALID`; a session fallback is unverified.
`task_step {step_id, status, note?, expected_revision?}` updates one step with its own revision
(`by: agent:<run>[:unverified]`); it does not advance `plan_revision`. `task_interview` and
`submit_plan` (stage D, below) need the run token. `submit_plan_review` (hyperplan) takes only a reviewer's
token; a reviewer token can call `task_plan` and `submit_plan_review` and nothing else (`REVIEWER_SCOPE`). `session_link` accepts `task_id`. The run token is
HMAC-derived from the run id (`ert_…`), stored as a hash, valid 24 h and refreshed by each
`context.request`.

### Browser routes (runs)

| Route | Gated | |
|---|---|---|
| `GET /runs?task_id=&active=&limit=` | no | runs, newest first |
| `POST /runs` | yes | start (above) |
| `GET /runs/preflight?task_id=&harness=&execution_target_id=` | yes | the Run dialog's read-only look at a start: `target` or `needs_target`+`need`, `problems`, `active_run`, `interview`, `harnesses` (what the task's target can start), `harness_reasons`, `harness_note`, `verify_status`, `verify_ready`, `verify_reasons`, `mcp`, `template`, `task` |
| `GET /runs/{id}?limit=&offset=` | no | detail: plan, history, turns, commands with outcomes, questions, attempts, `verify_readiness`, and `changes` (the review packet's PR links and branch state, under "The review packet"); turns and commands page newest first (`more`, `retention`: see "Reading runs") |
| `GET /runs/{id}/turns/{turn_id}` | no | one turn whole, `{id, seq, text, truncated}`, for the "Full response" of a `clipped` turn. Owner only, like detail. 404 `NOT_FOUND` for an unknown id or a turn of another run or owner; 404 `TURN_EXPIRED` once retention deleted the run's turns |
| `POST /runs/{id}/message {text, mode, expected_revision?, action_id?}` | yes | queue or steer |
| `POST /runs/{id}/stop {mode, expected_revision?, action_id?}` | no (viewers refused) | stop stays available after remotea is turned off; the revision stays optional (emergency path) and the drawer never sends one |
| `POST /runs/{id}/focus` | yes | "Open terminal in remotea": queues `terminal.focus` for the run's current surface and waits up to 6 s for the host's answer. `{sent: true, desktops}`, or `{sent: false, reason, detail?}` with `reason` `no_surface` (no terminal yet, or the host says it is gone), `host_offline`, `host_too_old` (the host does not advertise `terminal.focus`), `no_desktop` (no desktop app connected to the host), `no_answer` (no answer in time; a still-queued request is canceled) or `failed`. The first three are answered without queuing anything. Changes nothing on the run (no revision, no timeline entry) and takes no `action_id`: a repeat only raises the tab again |
| `POST /runs/{id}/approve {expected_revision, action_id?}` | yes | the revision is **required** (422 without it, `REVISION_CONFLICT` when stale): approving completes the task, so it must be the state the approver reviewed |
| `POST /questions/{id}/answer` | yes | |
| `POST /runs/{id}/redact` | no (owner only) | delete a settled run's content now; the skeleton stays (`409 RUN_ACTIVE` before it settles; see "Retention and redaction") |
| `POST /retention {days}` | no (owner only) | the run retention window, `agent_retention_days` (1–3650, default 90) |
| `GET`/`POST /targets` | yes | saved and advertised targets, with per-target `harnesses`, `verify`, `enabled`, `launch`, `verify_status` and `mcp` when the host sent them |
| `GET /unmerged` | yes | the hosts' last `worktrees.report` |
| `GET /worktrees` | yes | worktree maintenance: `{hosts: [{id, name, connected, report_at, rescan, count}], items: [...]}`, every reported row with its bound `run` and `cleanup` ([Worktree maintenance](#worktree-maintenance)) |
| `POST /hosts/{id}/rescan` | yes | queue a run-less `worktrees.scan`: `{op_id, queued, connected}` (`queued: false` returns the scan still waiting); 404 `NOT_FOUND` for another owner's or an unknown host, 409 `HOST_TOO_OLD` when the host lacks `worktrees.scan` |

**Retrying a run action (`action_id`, migration 0035).** `message`, `stop`, `approve`, `replan`, `freestyle`,
`verify-config`, `criteria` and `judge` accept an optional client `action_id` (a UUID). The first request acts and stores a receipt
(`agent_action_receipts`: owner, action id, run, route, a SHA-256 of the route, run and body without
`action_id`, and the response) in the same transaction as the effect (`agent_receipts.once`). A repeat with the
same `action_id` and body returns the stored response and acts no more: one `AgentCommand`, one steer
generation, one stop. The same `action_id` with a different body, route or run is refused with 409
`ACTION_MISMATCH`. A refused action (409 `NOT_STARTED`, `REVISION_CONFLICT`, …) rolls back and stores nothing,
so the same id may act later. Without `action_id` every request acts. Receipts are kept 7 days. The web helper
`postAction(path, body)` in `apps/web/src/api.ts` sets one `action_id` per call and resends the same body after
a lost response (`NETWORK`, up to two retries); other errors are not retried.

The page shows runs grouped by the next human action (below), **Unmerged work**, and a run drawer with the
timeline, the questions card, the review packet and Message / Steer / Stop (**Stopping…**, then **Force stop**) /
Approve.

**Attention-first grouping (E-Q5).** The topbar Runs panel and the remotea page list group runs with
`remotea-model.ts` `attentionGroups(runs, pending, hosts, unmerged)` (pure; `attentionKey` decides one run), in this
order, each with a verb heading and a count, empty groups hidden: **Confirm Eri's proposal** (pending
`control_start`/`control_message`), **Answer a question** (`waiting_answer`, `interviewing`,
`needs_you(question_timed_out)`), **Review the plan** (`needs_you(plan_review)`), **Repair the host or setup**
(`needs_you` for `host_offline`, `surface_lost`, `exited`, `spawn_expired`, `delivery_uncertain`, `verify_infra`, or
any working / needs_you run whose host is offline in `hosts`), **Inspect the failed check** (every other
`needs_you`: judge needs you, no progress, verify missing, turn cap, PR unknown, judge unavailable), **Approve**
(`awaiting_approval`; approving needs no host, so an offline host does not move it), **Resolve unmerged work**
(`awaiting_integration`, `needs_you(awaiting_merge)`, and a settled run whose branch is in the hosts' unmerged report
and not active), then **Working** and **Done** (no action). Within a group the most recently changed run comes first.
The Runs badge (`runsBadge`) counts only the action groups (proposals included); its label reads "Runs, N need you,
M working".

**Open terminal in remotea** (drawer header, any run with a terminal, settled ones included) posts
`/runs/{id}/focus`; the remotea desktop app raises its window and selects the run's tab. When that cannot happen
(`sent: false`) the drawer says why and shows where the terminal is instead (`remotea-model.ts`
`terminalFallback`): the host's name, the worktree path and branch, and commands to copy: `cd <worktree>` and
`remotea screen <surf>` (remotea has no interactive attach command). The other direction: a remotea tab Eridani
started shows a small ↗ button and, in its context menu, **Open in Eridani** and **Copy run link**, which open
`<Eridani>/?view=remotea&run=<id>` (the host records Eridani's web URL on the session; see remotea's
`docs/remotea/eridani-link.md`, "Open in Eridani, and the terminal from Eridani").

The drawer loads the newest page of turns and commands. When `more` says older ones remain, the top of the log
shows **Paged out** with how many remain and **Load earlier**, which fetches the next `offset` and merges it
(`remotea-model.ts` `mergeDetail`: deduped by id, ordered by time; a later poll keeps what was loaded). The scroll
stays on the entry that was first on screen. A turn's labels: **Clipped** (cut for display; **Full response**
fetches the turn route and keeps the text), **Truncated by host** (the rest was never stored), and **Expired**
(retention deleted it; also shown at the top of the log for a run past its retention). Message, Steer, Re-plan,
Freestyle, Set verify, Approve and Stop go through `postAction` (one `action_id`, retried on a lost response);
Verify now uses a plain `post` because its route stores no receipt.

**Freshness in the drawer (E-S6).** Under the place and base lines the header shows one chip per fact
(`remotea-model.ts` `freshnessFacts(run, host, now)`, pure): "host seen 12s ago" / "host silent for 4m", "last turn
2m ago", "<state> for 4m", "verify tests passed 3m ago · fresh" (or `stale: …`, `incomplete workspace`, "running for
15m (past its deadline)"), "workspace fingerprint 3m ago · incomplete (…)", "sandboxed" / "not sandboxed (<reason>)",
and "steer 40s ago: delivery uncertain" / "message sent 2m ago, not taken by remotea yet". A chip that weakens a
decision is red, a reassuring one green. On `awaiting_approval` the footer repeats the facts Approve rests on
(verification, workspace, sandbox, host, unconfirmed controls) above the buttons. The ages tick on a local clock;
nothing extra is fetched. In the timeline a verification card shows its exit code, "isolated" / "not sandboxed
(<reason>)" and "complete workspace" / "workspace incomplete (<reasons>)" as separate labels (`attemptFacts`).

**Freshness on run rows.** Each unsettled row in the Runs panel and the remotea page's run list (`RunsPanel.tsx`
`RunRow`) carries compact chips from `rowFreshness(run, host, now)`, a subset of `freshnessFacts`: the host chip
("host seen 2m ago", red past 90 s or when silent), then only the warnings that change what the row means: "stale"
(verification from an older turn or a changed checkout), "workspace incomplete", "verify late" (past its deadline)
and "delivery uncertain"; each chip's tooltip carries the full fact. One clock in `AttentionList` (30 s) ticks every
row's ages; settled runs show none.

**One copy of each text sent to the agent.** The host reports every prompt the agent receives, Eridani's own
inputs included (a `user` turn, or an older host's turn `prompt`), and each copy is clipped differently (the host's
prompt at 4,000 bytes, the "sent" history entry and an input command at 2,000 characters, a turn's `prompt` at 500
here). The timeline (`timeline`, `sameText`) treats two copies as the same text when they are equal after
whitespace normalisation, or when the shorter one (at least 200 characters, a trailing "…" ignored) is a prefix of
the longer; such a host copy is not shown again as "You, in the terminal".

The drawer (and the side panel, which keys it by run id so draft text, busy state and scroll are per run)
binds every control to the run it has loaded, never to the id it was asked for: a run load is latest-only
(`remotea-model.ts` `latestOnly`/`latestRun`: switching runs aborts the earlier request and a late reply,
or one for another run, is dropped), controls stay disabled until the loaded run is the requested one, and
each URL and body comes from `runAction`. Message, Steer, Re-plan, Verify now, the verify command,
Freestyle, Approve and the plan review send `expected_revision` = the loaded run's `state_revision`; Stop
does not. A `REVISION_CONFLICT` shows "This run changed; reloaded" and reloads the run instead of an error. Task details
show **Run** only while remotea is on, disabled with a reason when no host is connected, and **Open run**
while one is active.

### Worktree maintenance

Task 7be3f225 (Astra S-Q4). The remotea page has a **Worktree maintenance** section below **Unmerged work**
(`WorktreeMaintenance.tsx`, data from `GET /remotea/worktrees`, `agent_control.worktree_maintenance`). Unlike
`/unmerged` it lists every row of each host's last `worktrees.report`, candidates included, with the host, the
target's `repo` (from hello `targets`), the raw report fields (`ahead`, `behind`, `unpushed`, `upstream`, `merged`,
`uncommitted`, `untracked`, `ignored`, `candidate`, `reasons`, `writer` when the host sends it) and the bound
`run` (the writer's run, else the newest run on that host whose worktree or branch matches: `{id, state, settled,
task_id, task_title}`). Each host line shows its report's age and **Rescan**, which sends a run-less
`worktrees.scan` (`POST /remotea/hosts/{id}/rescan`; disabled for a host without the cap).

Rows are classed by `remotea-model.ts` `worktreeClass`, in this order:

| Class | When |
|---|---|
| unknown | the row lacks `uncommitted`, `candidate` or `reasons` (an older host) |
| active | `active`, a `writer`, or a bound run that has not settled |
| reclaimable | the folder is gone (`folder missing` reason) and the row says `writer: null` |
| dirty | uncommitted changes or untracked files |
| unpushed | commits not on origin (`unpushed > 0`, or not merged with no upstream) |
| unmerged | pushed but not merged into its base (an open pull request) |
| reclaimable | the host reports it a `candidate` (nothing would be lost) and `writer: null` |
| merged | merged, but the host keeps it for another reason (ignored files such as `.env`, a lock) |
| unknown | anything else (detached HEAD, a failed check), or a candidate from a host that does not send `writer` |

The section shows a count per class (each a filter) and a table: class, branch with host/target/path, why
(`worktreeLoss`: what removing it would lose), and the task/run (opens the run). **Review removal** opens an
explicit review step (`RemovalReview`, `removalPlan`): only reclaimable rows that have cleanup commands get a
checkbox; every other row is listed under "Not offered" with what it would lose (uncommitted changes, untracked
and ignored files, commits not on origin, a live writer, an unsettled run) and cannot be picked. **Produce
commands** shows the picked rows' commands, grouped per host, with Copy. The commands are remotea's own
(`worktrees.ts` `cleanupCommands`, mirrored in `agent_control.cleanup_commands` and computed server-side only for
a row the host reports as a candidate or folder-missing, with no writer): `git -C <repo> update-ref
refs/remotea/recovery/<branch-dir> <head>` then `git -C <repo> worktree remove <path>` (unforced, so it refuses a
worktree that changed since the report); a gone folder gets `git -C <repo> worktree prune`.

Eridani never removes a worktree itself: remotea advertises no removal command (no `worktrees.remove` cap; it
"never removes a worktree"), so the copyable commands are the only action and there is no removal route. If
remotea adds such a command, `POST /remotea/worktrees/remove` would send it only to a host advertising the cap and
only for rows the server classes reclaimable.

## Planning and the loop (stage D, migration 0029)

0029 adds `agent_runs.turn_seq` (the host's seq of the root turn the run is on) and `plan_pending`,
`agent_attempts.deadline_at/retries/verify_id/initiator`, `agent_run_plans.required_children` and
`agent_interviews.generation/answered_at/answered_via`.

```
spawning → interviewing → planning → executing → verifying → judging → executing (not_met: the gap is the next prompt)
                ↑ needs_you(no_plan)        ↑ replanning ←┘     │      ├→ awaiting_approval → completed (Approve)
                                                                │      ├→ stopping(final completed) (auto_complete)
                                                                │      ├→ awaiting_integration → executing
                                                                │      ├→ needs_you (needs_you, no progress, no verdict)
                                                                │      └→ capped (not_met on the last allowed turn)
```

### Interview and planning

- `POST /remotea/runs` takes `interview: true|false|null`. `null` (default): a task without a PRD note is
  interviewed; one with a PRD is frozen at once (stage C's behaviour, now with `spawn.verify` when the task
  has a verify config). `false` runs on whatever plan the task has.
- An interview run spawns with the interview prompt (`agent_prompts.interview`: the task, its direct
  sub-tasks, "ask with `task_interview` and stop") and no frozen plan (`plan_revision` 0); the observed
  start goes to `interviewing`.
- **`task_interview {run_token, questions[≤12: {id, question, header?, options?: [{label, description?}],
  multi?, free_text?}]}`** (interviewing only, one open interview at a time) stores an `agent_interviews`
  row and pushes `agent_question`. A question without options takes free text.
- **`POST /remotea/interviews/{id}/answers {answers: {question_id: label | [labels] | text}}`** checks
  every question has an allowed answer (`INVALID_ARGUMENT`, `INTERVIEW_CLOSED`, `STALE_GENERATION`),
  sends the answers as the next `input` ("now write the PRD with success criteria and call `submit_plan`,
  then stop") and moves the run to `planning`. The run is `interviewing`, or `needs_you` on `wall_clock` or
  `no_progress` with the interview open for its current generation (the page shows the form there too); answering
  from `wall_clock` grants the 2 h described under the needs_you reasons.
- **`submit_plan {run_token, prd, steps[], success_criteria[≥1: {id?, text, kind: text|verify,
  verify_id?}], verify?: {target_id, verify_id}, migration?: bool, expected_plan_revision}`** (interviewing, planning or
  replanning; else `PLAN_LOCKED`): `expected_plan_revision` is the task's `plan_revision` from
  `task_plan` (`REVISION_CONFLICT`); `verify` must name the run's execution target and a verify id from
  its saved list or the host's advertised one (`VERIFY_UNKNOWN`); a `verify` criterion must name the
  plan's verify id. It creates or replaces the PRD note (`PRD: <task>`, tag `prd`, linked to the task,
  with a "Sub-tasks" section), writes the task's template plan through `task.update` as the run judge
  (gate fields are allowed only to that actor during a run), freezes `agent_run_plans` revision N+1 with
  `required_children`, sets `plan_pending` and moves an interviewing run to `planning`. An open interview
  is marked `superseded`.
- **Execution starts only when the planning turn has settled**: the next root turn of the run's
  generation with `plan_pending` set and no open question or interview moves `planning → executing` and
  sends the execute prompt with the frozen plan. A planning turn that ends with neither a question nor a
  plan goes to `needs_you(no_plan)`. The spawn outcome's `result.mcp {configured, reason?}` (remotea PR #20) is
  kept on the run (`sync_state.mcp`, returned as `mcp` by `GET /runs/{id}`); when it says `configured: false`, the
  `no_plan` detail reads "The agent had no Eridani tools, so it could not call submit_plan: <reason>" (the
  one-line reason text is unchanged). Such a run does not wait for its planning turn to end: when the spawn
  outcome says `configured: false` and the run still needs a plan (`run_coordinator.needs_plan`: spawning with no
  frozen plan, or interviewing/planning/replanning without `plan_pending`, and not freestyle), it goes to
  `needs_you(no_plan)` with that detail at once. A run whose plan was frozen at start keeps executing (it needs no
  `submit_plan`). Freestyle resumes from `needs_you(no_plan)`. Interview and planning turns never count toward the cap.
- **Re-plan** (`POST /remotea/runs/{id}/replan {reason?}`, from executing, needs_you, awaiting_approval,
  awaiting_integration or planning) advances the generation, supersedes in-flight attempts, sends the
  re-plan prompt (`steer` while executing, else `queue`) and goes `replanning → planning`; the next
  `submit_plan` freezes revision N+1 the same way. `runs.snapshot` then sends `verify: null` until the new
  revision is executing.

### Sub-tasks (F02)

Direct child tasks (not archived, not cancelled) seed the plan as linked steps: the agent's order first,
missing children appended, existing step ids and statuses kept; a linked step must name a direct child.
Their ids are frozen as `required_children` (a manual stage C freeze uses the children its linked steps
name). A child with no run of its own is a checklist step inside the parent's run. A child with its own
run counts only when that run is `completed` **and** the host's `worktrees.report` shows its branch
`merged` into the parent branch (`base` = the parent run's branch). A `met` verdict with a child still
pending moves the parent to `awaiting_integration` (push `agent_stuck`, the drawer lists what is
pending). Each `worktrees.report` re-checks; once integrated the parent goes back to `executing` with a
"pull the merged sub-tasks and re-check" prompt, so the combined code is verified and judged again
before completion.

### Verify attempts

After each root execution turn of a run whose frozen plan has a verify config, the coordinator creates an
`agent_attempts(kind=verify)` row bound to `(run, generation, turn_seq, plan_revision)` and queues
`verify {attempt_id, turn_seq, plan_revision, verify_id}` (`turn_seq` is the turn frame's `seq`, so host
attempts carry the same tuple). The run is `verifying` until the result.

- `verify.result` is matched by `attempt_id` and the whole tuple (`tuple mismatch`, `unknown attempt`
  and `late` are acked with `discarded`). Status: `passed` (exit 0), `failed`, `timed_out`, `canceled`,
  `error`, `stale` (`stale_workspace`: the checkout changed around the run). Only `passed` passes.
- **Isolation and fingerprint completeness** (remotea `verify-isolation`, optional on the wire): `verify.result`
  may carry `isolated` (the command really ran inside bubblewrap), `isolation` (`bwrap` | `none`, the entry's
  mode), `isolation_reason` (≤ 300 characters, redacted, only when not isolated), `workspace_complete` (false: the
  fingerprint cannot prove the tree unchanged) and `workspace_reasons` (what it missed). They are kept on the attempt's
  `result` and returned by `public_attempt` (null, or `[]` for the reasons, when an older host did not send them) for
  the review packet. The judge does not read them yet (`workspace_complete: false` is not yet treated as not fresh).
- **Host-initiated** results (`attempt_id: "local:…"`, remotea's `verify.auto`) are adopted when they
  match the tuple of a running Eridani attempt; others are stored with `initiator: host` and never move
  the run. `adopted` on a result names the host attempt remotea reused.
- **Isolation and workspace completeness** (remotea PR #24, optional on the wire): `isolated`, `isolation`
  (`bwrap`|`none`), `isolation_reason`, `workspace_complete` and `workspace_reasons` are kept on the attempt
  (`run_coordinator.workspace_facts`; malformed values are dropped). They do not change the status: an exit 0 on an
  unsandboxed run is still `passed`, and the card shows the sandbox apart from the exit code.
  `workspace_complete: false` means the fingerprint cannot prove the tree it tested is the one the turn left, so
  it is **not fresh for auto-completion** (see auto_complete).
- One deadline per attempt, stored as `deadline_at`: the entry's advertised timeout plus 120 s, never under 11
  minutes, and 11 minutes when the host did not advertise one (`verify_deadline`, see the host setup above) →
  `timed_out`, a non-pass; a result after that is discarded and noted on the attempt as `late`. A refused/failed/uncertain `verify` command is
  `refused`/`uncertain`, also a non-pass (never `delivery_uncertain`). A host without `verify` refuses at
  once. Every non-pass still goes to the judge.
- **Failure category** (`run_coordinator.classify_verify`): every settled non-pass gets `result.category` and a
  one-line `result.action` (also on host attempts kept aside), shown by `public_attempt` and the run page
  ("Verification could not run · <action>" for infra). From the host's `error`, `exit_code`, `timed_out`, the
  tail and the target's hello `verify_status` readiness reason (when it says the entry is not ready): the command
  outcome and timeouts first, then the patterns in table order; an `error` nothing explains is `infra_uncertain`.

  | category | when | action (example) |
  |---|---|---|
  | `infra_uncertain` | `uncertain` command outcome; an `error` nothing else explains | Check remotea, then Verify now |
  | `infra_refused` | `refused` (rejected or failed `verify` command, host without `verify`); a bad `cwd` | Check the host and the verify entry in eridani.json |
  | `infra_timeout` | `timed_out`: the host's limit, or no result by the 11-minute deadline (host offline, command expired) | Raise the entry's timeout, or check the host is online |
  | `infra_missing_binary` | exit 127/126, "command not found", `spawn X ENOENT`, readiness "its program is not on the service PATH" | Install uv or set PATH in the verify entry (eridani.json) |
  | `infra_database` | psycopg/SQLAlchemy `OperationalError`, "could not connect to server", connection refused, unknown database/role | Point the verify entry at a disposable test database |
  | `infra_missing_dependency` | `ModuleNotFoundError`, "No module named", "Cannot find module" | Install the module (uv sync, npm install), or message the agent if it is the project's own |
  | `tests_failed` | any other `failed` | Fix the failing tests |

  `passed`, `canceled` and `stale` get no category (`stale` stays the judge's "could not establish" needs_you).
  An infra category says nothing about the code: the judge parks the run (below) instead of sending a gap.
- **On demand**: `POST /remotea/runs/{id}/verify` uses the same machinery for the current tuple
  (`on_demand`), refuses `VERIFY_RUNNING`/`NO_VERIFY`, and never moves the run.
- A message or steer while verifying/judging supersedes in-flight attempts and returns to `executing`.
- The host learns the verify config from `spawn.verify` and `runs.snapshot[].verify` (`{verify_id,
  plan_revision}`; `null` after a re-plan clears it), and from each `verify` command.

### The judge (`agent_judge.py`)

Job `agent_judge` on queue `jarvis-memory`, cost feature `agent_control`, one per judge attempt.
Evidence: the frozen plan (PRD, steps as they stand, criteria, verify), the root turn's final response
(last 24k characters; `truncated` flagged), the bound verify attempt with `fresh` (same tuple and not
stale), the previous verdict's per-criterion record, and the completion policy with the PR record
(`pr_urls`, `merged`, and `pr_fresh` / `merged_fresh` with `as_of` times: whether the session's collector
report and the host's worktrees report arrived after the turn). The verify evidence carries `category`, `action`
and `infra` (true for an `infra_*` category), and remotea's `workspace_complete` / `workspace_reasons` (whether the
host's fingerprint covered the whole workspace; `null` / `[]` when the host did not say). The prompt (`judge-v4`)
tells the model to say so when a verification ran on an incomplete workspace, and says all of it is
untrusted data and that **the agent's text is a claim**: evidence is the verify attempt, the host's worktree and
PR record and the ticked steps. A gap may only ask for what the loop can check (make the verify command pass,
commit/push and open or update the PR, tick steps), never for pasted test output, logs or diffs. The gap prompt
(`agent_prompts.gap`) repeats that to the agent.

1. **Hard rules first.** No criteria → `needs_you` and the model is not asked. Plans without criteria do
   not enter the loop at all: they keep stage C's manual flow (turn cap → needs_you, Davin approves).
2. **Luna** via `memory_learning.extraction_request` (the only model it accepts; decision 20) with a
   strict schema `{verdict: met|not_met|needs_you, gap, summary, criteria: [{id, status: met|not_met|unclear,
   evidence}], steps: [{id, status: done|not_done|disputed|unclear}]}`.
3. **Rules again on the answer.** **`verify_infra`** first: a bound verification with an infra category (the
   check could not run: missing program, module or database, timeout, refused, uncertain) makes the verdict
   `needs_you` with the category's `action` as the gap, whatever the model said; the prompt tells the model not
   to blame the code for it, and its steps cannot be `disputed`. Then `met` with a required verify that is not fresh and passing: `not_met`
   with the failure tail when it ran and failed, else `needs_you` (missing, stale or canceled: the agent cannot fix that;
   refused, timed out and uncertain are infra categories and stopped at `verify_infra`). A `verify`-kind criterion is met only with a fresh passing
   attempt of its verify id. `met` with any criterion not met → `not_met`. Unknown criterion ids are
   dropped; a missing one is `not_met`. `disputed` steps count only when the verification failed. A `met` verdict
   whose bound verification has `workspace_complete: false` stays `met` but fires **`verify_workspace_incomplete`**:
   the summary and every met `verify`-kind criterion's evidence are prefixed/suffixed with "Verification ran on an
   incomplete workspace (reasons)", so the pass never reads as clean (with `auto_complete` on, the coordinator still
   parks it as `needs_you(verify_incomplete)`). Each rule that fired is recorded in `rules`.
4. **Non-verdicts** (refusal, incomplete, timeout, budget, invalid output, unavailable) are retried twice
   (refusal and budget are not retried), then the attempt is `failed` and the run goes to
   `needs_you(judge_unavailable)`. A judge attempt still open after 15 minutes is a `timeout` non-verdict.
5. The attempt keeps model, `prompt_version` (`judge-v4`), cost, latency, retries, the final verdict and the
   completion evidence it used (`evidence.completion`).

**Commit** (`run_coordinator.on_verdict`) only while the run is `judging` with the attempt's exact
`(generation, turn_seq, plan_revision)`; otherwise the attempt is `stale` and nothing moves. Every
transition records `actor: run_judge`.

- `met` → `awaiting_integration` if a required child is pending; else with **`auto_complete`** on (and
  criteria present) the judge completes the task (`task.complete` as `run_judge`) and the run goes to
  `stopping(judge_met, final completed)` (push `agent_done`); else `awaiting_approval` (push
  `agent_done`). **Approve** (a human) completes it.
- `not_met` → if `turn_count ≥ turn_cap`: `capped` (the last allowed turn was judged first; push
  `agent_stuck`). If the last three verdicts were `not_met` with the same gap (case and spacing ignored)
  **or fired the same hard rules** (the judge rewords a gap it cannot close): `needs_you(no_progress_gap)`.
  Else back to `executing` with the gap prompt as the next `input` (recorded with `source: eridani`).
- **`verify_missing`** (rule): a `not_met` gap that asks for a recorded or independent verification on a plan
  that has **no** verify command becomes `needs_you(verify_missing)` at once. Nothing the agent does in its
  worktree can record a verification (run 75661a23 looped six turns on this); only the operator can set one
  (`POST /runs/{id}/verify-config`, the run header's **Set verify**) or approve on the evidence shown.
- **`proof_request`** (rule): a model `not_met` gap that asks the agent to paste, quote, show or include test output,
  logs, diffs or other proof in its reply (`agent_judge.PROOF_ASK`) is rewritten to the checkable actions
  (`verifiable_actions`: make the plan's verify command pass, tick the open steps with `task_step`, commit/push and
  open or update the PR under `pr_open`/`pr_merged`), keeping the rest of the judge's note without the request.
  With no verify command and either a test-shaped ask or no such action, it is `needs_you(verify_missing)`
  instead. A repeated rewrite counts toward `no_progress_gap` (same rules three times).
- **PR evidence freshness** (rule `pr_unknown`, `pr_open`/`pr_merged` only, after a `met`): a missing PR on a
  session record older than the turn, or `merged` that is false on a report older than the turn, null ("no
  commits of its own yet") or absent, is unknown, not a gap. The verdict is **held** first (`EVIDENCE_WAIT`,
  60 s): the attempt goes back to `queued` with `result.held` (the verdict before the policy), Eridani queues
  `worktrees.scan` if the host advertises it, and the judge job is queued again (`release_held`) when a
  `session.report` for the run's session (`session_reported`, which stamps `sync_state.session_report_at`) or a
  `worktrees.report` arrives, or by the sweep when the wait is over. The released job applies the policy to the
  held verdict with the record as it now is, without asking the model again. Still unknown after the wait →
  `needs_you(pr_unknown)`. A later report that shows the PR (`pr_open`: the met path, so approval or
  auto_complete) or the merge (`pr_merged`: `executing(merged)`) moves the run on; otherwise Davin checks.
- **`verify_infra`** (rule, checked first) → `needs_you(verify_infra)` with the action as `detail` and
  `reason_detail`; no gap prompt is sent, whatever the verdict. `control_status` says "host setup problem:
  <action>". After fixing the host the operator messages the agent (its next turn is verified again), uses
  freestyle, or approves on the evidence shown. A `tests_failed` verification keeps the not_met gap prompt.
- `needs_you` → `needs_you(judge_needs_you)` with the gap; push `agent_stuck`.
- **Steps**: the judge may mark `todo`/`doing` steps `done` (`by: judge:<run>`), may mark an agent's
  `done` step `disputed` when the verification contradicts it, and never moves a `done` step back.

Limits: the wall clock (6 h) now also covers interviewing, planning, verifying and judging; the 45-minute
no-progress timer still applies to executing.

### Verify by default, and set mid-run

A criterion that asks for test evidence (`kind: verify`, or text naming tests, a suite, pytest, vitest, CI, lint,
typecheck, compile or build: `agent_plan.EVIDENCE`) can only be judged from a recorded verification. When a plan is
frozen without a verify command and such a criterion exists, Eridani names the execution target's verify command
itself (`agent_plan.default_verify`: `all`/`full`/`tests`/`test`/`check`/`ci` if offered, else the first id;
stored with `by: default`, which is not a human floor and may be replaced). When the host's `verify_status` says
some ids are ready and others not, only the ready ones (and ids it did not report) are considered
(`pick_verify(offered, ready)`); when none is ready the choice is as before and the preflight lists
`VERIFY_NOT_READY` with each reason. When the target offers several ids
the pick is a plan-review proposal (`proposals.verify` with `current: null` and `options`); declining it leaves the
plan without verification. The planning prompt lists the offered ids and says a test criterion must name one.

`POST /remotea/runs/{id}/verify-config {verify_id, expected_revision?}` (`agent_plan.set_verify`): the operator
adds or swaps the verify command mid-run. The task's verify is set as a human edit, the plan is frozen again as
revision N+1 with nothing else changed (history `verify_set`), and a run parked in `needs_you` goes back to
`executing(verify_set)` with a short prompt so its next turn is verified. `GET /runs/{id}` lists `verify_options`.

### Edit criteria and Judge again (ba56714a)

While a run is parked for the owner (`needs_you` other than `plan_review`, or `awaiting_approval`), the run page offers
**Edit criteria** and **Judge again** under the review packet.

- `POST /remotea/runs/{id}/criteria {criteria: [{id?, text, kind: text|verify}] (1–50), verify_id?, expected_revision,
  action_id?}` (`agent_plan.set_criteria`): a human edit of the task's `success_criteria` (and, with `verify_id`, its
  verify command, checked like `set_verify`: `VERIFY_UNKNOWN`), frozen as plan revision N+1 with nothing else changed
  (history `criteria_set`). A `verify`-kind criterion takes the plan's verify id (refused without one). An unchanged
  agent criterion keeps `by: agent`; an edited or new one is the human's. The run does **not** move (unlike
  `verify-config`, which resumes a needs_you run, so the editor sets the verify command here instead), but its
  `state_revision` advances so an Approve sent against the old criteria is refused with `REVISION_CONFLICT`. Other
  states: `409 RUN_NOT_PARKED`. The revision is required.
- `POST /remotea/runs/{id}/judge {expected_revision, action_id?}` (`agent_control.judge_again`,
  `run_coordinator.judge_again`): verify the run's current turn again when the plan has a verify command
  (`needs_you|awaiting_approval → verifying(judge_again)`), else judge it at once (`→ judging(judge_again)`), against
  the plan as frozen now; the verdict then moves the run as after any turn (`not_met` sends the gap prompt). Refused
  with `RUN_NOT_PARKED`, `FREESTYLE` (turn freestyle off first), `NO_CRITERIA`, `NO_TURN` (no execution turn yet) or
  `VERIFY_RUNNING` (an attempt is still open). The editor's **Save and judge again** sends both, the second with the
  revision the first returned.

The judge's full reason is on the page twice: the header's reason box shows `reason_detail` (now also the judge's
gap for `verify_missing`, up to 4000 characters, instead of the transition's 300-character detail), and the review
packet shows the latest verdict's summary and gap.

### The review packet (E-Q3)

Beside **Approve** (and read-only in `needs_you` and settled runs) the run page shows a review packet built by
`remotea-model.ts` `reviewPacket(run)` from `GET /runs/{id}`:

- **Criteria**: the frozen criteria with the latest verdict per criterion (met, not met, unknown: `unclear` or not
  judged) and its evidence; "judged against an earlier plan" when the verdict's `plan_revision` is older than the run's
  (the criteria were edited since).
- **Changes**: `changes` on the detail (`agent_control.changes_info`): the pull requests the agent's session reported
  (collector `pr_urls`, validated links, not a claim in the turn text), `pr_as_of`, and the host's last worktrees
  report row for the run's branch (`ahead`, `behind`, `unpushed`, `uncommitted`, `untracked`, `merged`,
  `reported_at`), plus the branch and worktree.
- **Tests**: the latest Eridani-bound verify attempt: status, exit code, age, in a sandbox or not, and markers in
  words: "stale: the code changed during the run", "workspace incomplete: <reasons>", "not sandboxed (<reason>)",
  "for an earlier turn or plan". Without a verify command it says no test result can be recorded; a verification with
  an `infra_*` category reads "verification could not run (<action>)", never "tests failed".
- **Open questions** (agent questions open, submitted or timed out, and an open interview) and **the judge's reason**.
- **Decision**: on `awaiting_approval`, "Approving is your decision: it completes the task as you. It is not
  completion established by verification." A settled run says "Approved by you" (`state_reason: approved`) or
  "Completed by verification" (`judge_met`, auto_complete), styled differently, so a human approval is never
  presented as independently established completion.

The same packet as one sentence ("2 of 3 criteria met; tests passed 4 min ago in a sandbox; PR #12 open; 1 open
question") is shown next to Approve, and `control_status` returns it for Eri as `awaiting_approval.review_packet`
(`agent_control.review_packet`: `text`, `criteria`, `criteria_met`/`criteria_total`, `tests`, `pull_requests`,
`open_questions`, `reason`; also `review_packet` on a `needs_you` run).

### Freestyle

`POST /remotea/runs/{id}/freestyle {enabled, expected_revision?}` (`agent_runs.freestyle`, migration 0034): the
operator talks to the agent directly. While on, a root turn is recorded (history `turn` with `freestyle: true`)
but not verified, judged or counted toward the turn cap, no gap prompt is sent, and the 45-minute no-progress
timer is skipped (the wall clock still applies). Turning it on from `needs_you(judge_needs_you | no_progress_gap |
verify_missing | verify_infra | no_progress | turn_cap | awaiting_merge | pr_unknown | no_plan | verify_incomplete)` resumes
`executing(freestyle)`; turning it off makes the next turn judged again. Approve and Stop work as usual. The run header
shows the toggle. A run can also start in freestyle (`POST /remotea/runs {freestyle: true}`, the Run dialog's **Start
in freestyle**): no interview, the plan frozen from the task as written; it is the only way to start on a target
whose agents have no Eridani tools (`MCP_UNCONFIGURED`).

### auto_complete

`POST /remotea/auto-complete {enabled}` (owner preference `remotea_auto_complete`, default **off**;
viewers refused); `GET /remotea/status` returns it. Settings → Connections → remotea shows the toggle with a
warning that tasks then close (in Linear too) without a tap. Keep it off until the outcome eval below is clean on
a live run.

A `met` verdict on a plan with a verify command does not auto-complete when the newest verification of the current
turn says `workspace_complete: false`: the run goes to `needs_you(verify_incomplete)` with the detail
"Verification evidence was incomplete: <workspace_reasons>" and the owner reviews and approves (Approve works from
`needs_you`). With auto_complete off the run goes to `awaiting_approval` as before, and the drawer's approval facts
show the incomplete workspace.

### Outcome eval (`evals/agent_control/`)

`cases.json` is a labelled suite: correct completion, plausible false success (failing tests, no evidence,
instructions in the turn text), missing criteria, missing/timed-out verification, a stale test result, a
source change mid-run, an ambiguous blocker, partial progress, recovery after a failure, a truncated turn,
and two loop cases for the no-progress rule. `stub` is what a plausible (sometimes deliberately fooled)
model answers.

```
uv run --no-sync python -m evals.agent_control.run          # stubbed model, free (CI: evals/test_agent_control.py)
uv run --no-sync python -m evals.agent_control.run --live   # Luna through the extraction helper (paid)
```

It reports `false_completions` (met where the label is not), `stale_completions` (false completions on
stale or changed-source evidence) and `missed_completions`, and exits non-zero on any false or stale
completion.

### Browser routes (stage D)

| Route | Gated | |
|---|---|---|
| `POST /interviews/{id}/answers` | yes | interview form |
| `POST /runs/{id}/replan {reason?, expected_revision?, action_id?}` | yes | |
| `POST /runs/{id}/verify {expected_revision?}` | yes | on-demand verification; the body is optional (a second request while one runs is refused, `VERIFY_RUNNING`) |
| `POST /runs/{id}/verify-config {verify_id, expected_revision?, action_id?}` | yes | set the plan's verify command mid-run (new revision) |
| `POST /runs/{id}/freestyle {enabled, expected_revision?, action_id?}` | yes | judge, verify, cap and gap prompts off while on |
| `POST /runs/{id}/plan-review {decisions, verify, completion_policy, expected_revision}` | yes | the revision is **required** (422 without it, `REVISION_CONFLICT` when stale) |
| `POST /runs/{id}/criteria {criteria, verify_id?, expected_revision, action_id?}` | yes | Edit criteria on a parked run (new revision; the run does not move); the revision is required |
| `POST /runs/{id}/judge {expected_revision, action_id?}` | yes | Judge again: verify and judge the current turn against the plan as frozen now |
| `POST /auto-complete {enabled}` | yes | rollout flag |

The run drawer adds the interview form (one form for all questions), success criteria with the latest
verdict's per-criterion evidence and the last verification, pending sub-task integration, **Approve** as
the primary action on `awaiting_approval`, **Re-plan** (the message box text is the reason) and **Verify
now**; the timeline shows verifications, verdicts and interviews.

## Sync, voice and the dream (stage E, migration 0030)

Migration 0030 adds `completion_policy` to `tasks` and `agent_run_plans`, `binding_evidence` to
`agent_sessions` and the `agent_parity_days` table.

### Linear sync outcomes (`agent_sync.py`, `linear_sync.py`)

A run moves its task's linked Linear issue twice: **In Progress** when the run is observed to start
(`mark_in_progress`) and **Done** when the task completes (Approve, or the judge with `auto_complete`). On
`completed` the coordinator also queues one **completion comment** (job `linear_comment`, queue
`jarvis-linear`). Local state and sync state are separate: a refused or failed Linear write is recorded and
never undoes the start or the completion. Each outcome lives on `agent_runs.sync_state["linear"]`:

```
in_progress | done: {status: queued|deferred|blocked|skipped, job_id?, after?, code?, message?, at}
comment:            {status: queued|retrying|posted|failed|unconfirmed|skipped, job_id, marker,
                     attempted, comment_id?, url?, reconciled?, code?, message?, at}
```

- `skipped`: the task has no Linear link (or Linear is not connected / the link is in another workspace).
- `deferred`: another write for the issue is still in flight; the finish is queued after it (`after`).
- `blocked`: Linear refused the write up front (`code`, `message`); the local completion stands.

**Comment once.** Linear's `commentCreate` has no idempotency key. The body ends with a stable run marker
`eridani-run:<run id>`, and the job saves `attempted: true` before it writes. A retry after an attempted
write first reads the issue's comments for the marker; when found it saves that comment's id
(`reconciled: true`) instead of posting again. Up to 5 attempts; a definite refusal is `failed`, an
attempted write that can be neither confirmed nor found is `unconfirmed`. The body names the task, branch,
pull requests, turn count, the completion policy and the judge's summary.

**Ordering guard.** A stale In Progress can never reopen a completed task:
- completing the task cancels a queued In Progress write that has not been attempted yet;
- if one is already running, the Done write waits for it (`deferred`);
- when any write for the issue settles while the task is finished locally and the issue is not in the
  matching state, Eridani keeps its local status and queues the finish again (`finish_owed`) instead of
  applying the remote state.

**Pull requests** come from the session's collected `pr_urls` (validated `…/pull/N` or
`…/merge_requests/N`, deduplicated, at most 10).

**Completion policy** (`code_ready` default | `pr_open` | `pr_merged`), set on the task (`task.update`,
`submit_plan`) and frozen onto the plan and the run like the other gate fields (`RUN_OWNS_COMPLETION` while
a run holds it). The judge's hard rules apply it after a `met` verdict:
- `pr_open` or `pr_merged` with no pull request on a session record reported after the turn → `not_met`
  (rule `pr_required`), so the agent is told to open one; on an older record → held, then
  `needs_you(pr_unknown)` (see **The judge**);
- `pr_merged` with the PR not merged (the host's worktree report for the run's branch, reported after the turn)
  → `needs_you` (rule `pr_not_merged`, reason `awaiting_merge`); an older report, `merged: null` or no row for
  the branch → held, then `needs_you(pr_unknown)`. The integration check resumes the run (reason
  `merged`, prompt `merged`) once the host reports the branch merged, and the next judgment can complete.

The run drawer shows the policy and each Linear outcome.

### Voice commands (`control.*`)

Eri drives runs through the same command service as the remotea page (`agent_control`); there is no second
control path. The commands go through `domain.mutate` (personal tools, never offered to bots) and are
in `VOICE_MUTATIONS`:

| Tool | Input | |
|---|---|---|
| `control.start` | `task_id` or `task` (name), `harness?`, `model?`, `reviewer? {harness, model?}`, `worker? {harness, model?}`, `rounds?`, `approve_final?` | starts a run as the human (`start_run`); the hyperplan fields travel with the proposal and its summary names them ("… with Codex gpt-6-astra reviewing (2 rounds) and Claude Code sonnet working") |
| `control.message` | target, `text`, `mode: queue\|steer` | |
| `control.stop` | target, `mode: interrupt\|kill` | |
| `control.answer` | target, `question_id` + `answers`, or `interview_id` + `responses` | a run question or the planning interview |
| `control.confirm` | `pending_id?`, `decision: confirm\|decline`, `execution_target_id?` | the owner's yes to a proposed start/message (below); a start that asked where to run takes the chosen target |
| `control.approve` | target? | immediate: approves a run that is `awaiting_approval` (`agent_control.approve`; the server supplies `expected_revision` from the run it checked); any other state is refused with `NOT_AWAITING_APPROVAL` naming the state |
| `control.plan_review` | target?, `decisions: [{id, accept, text?}]`, `verify?`, `completion_policy?`, `action?: approve\|revise`, `note?` | immediate: the owner's decisions on a `plan_review` (`agent_plan.review_plan`; the server supplies `expected_revision`); for a hyperplan run in `needs_you(plan_approval)`, `action` approve or revise (with `note`); otherwise `NO_PLAN_REVIEW` |
| `control_status` (read) | target? | active runs, their open questions/interview, `plan_review` proposals, the judge's `awaiting_approval` summary with the `review_packet` (above), `freshness` per run (one line: "verify tests passed 3 min ago for the current turn on a complete workspace; sandboxed; host seen 5 s ago; last turn 2 min ago; in awaiting approval for 4 min; steer sent 40 s ago, delivery uncertain", `agent_control.freshness_text`), open proposals (`pending`, with target options), finished in the last 24 h, hosts (with `last_seen_at`), and `setup` (one line per unready host or target with its fix, when any) |

A target is `run_id`, `task_id` or `task` (a spoken name). Names resolve through an exact Linear identifier, then
`navigation_find`; none → `NOT_FOUND`; a name matching several tasks → `AMBIGUOUS_TASK` with the candidates when none
of them has an active run. Without a target (`run_for(db, owner, target, tool)`), the command fits every unsettled run,
except `control.approve` (only runs `awaiting_approval`) and `control.plan_review` (only `needs_you(plan_review)` or
`needs_you(plan_approval)`): the
run open on the owner's screen (`UIContext.open_run_id`, which the remotea page sends and `context_prompt` names as
"Open run: <id>") is meant when it is also the newest run the command fits (or none fits), else the only run it fits.

**Recipient confirmation (E-Q6).** When more than one run is plausible (several runs fit, the open run is not the
newest one that fits, or a spoken name matches several tasks that have active runs), `control.message`, `control.stop`,
`control.approve` and `control.plan_review` do not act: `run_for` raises `AMBIGUOUS_RUN` with `data.candidates` and
`control_command` returns `{status: "needs_choice", tool, message, candidates: [{run_id, task, host, branch, state,
label: "task · host · branch · state"}], next}` (nothing proposed, sent or journaled). `control.confirm` without a
`pending_id` while several different proposals of the account wait returns the same shape, each candidate a proposal
(`pending_id`, `summary`; a start names its task, chosen place, the branch it will create and `not started`); a
spoken yes still confirms only the newest (the older ones on the page, or after declining the newer). When exactly one
run is meant, the result carries `recipient` (the same fields) and its `next` begins "Name the task, host and branch
this affects when you tell the owner (<label>)": message proposals, stop, approve, plan review, answers (whose
question or interview id already names one run, so they never need a choice) and confirms. Eri's tool descriptions
(`tool_catalog.RECIPIENT`) say to name the task, host and branch before acting and to ask on `needs_choice`.

Every command passes the remotea gate (`REMOTEA_DISABLED` when off). The `control_*` tools and the `agents` catalog
group are in the model's registry only while remotea is enabled (`tool_policy.remotea_tools`, applied to
chat, voice and work runs); `approve` and `plan_review` act only on a run already waiting for the human, so
they need no second confirmation. External bots never see any of them (no scope maps to `control.*`).
While an account has a proposal waiting, queued work preloads `control_confirm` and `control_status`
(`work_runner.preloaded_controls`) so the owner's yes needs no `tools_load` round.

### Dream auto-link (`agent_autolink.py`)

Job `agent_autolink` (queue `jarvis-memory`) runs from the supervisor cycle at most once per 20 hours per
owner with remotea enabled. It looks at sessions active in the last 30 days that are unfiled or filed only
by a folder rule, have an embedding and are not a run's session.

- **Candidates:** tasks of the records under the session's record (not the home's own task), plus tasks
  named by exact Linear identifiers resolved in the connected workspace. Open tasks and tasks completed in
  the last 30 days; never archived, template or cancelled.
- **Score:** best cosine between the session's embedding and the task record's search vectors, plus 0.25
  when the session names the task's issue. Link when the top score is ≥ **0.62** and leads the runner-up
  by ≥ **0.12**. These constants are **provisional**: tuned on the labelled fixtures in
  `tests/test_agent_autolink.py`, not yet on real sessions.
- An automatic link is `binding_source: "auto"` with `binding_evidence` (a digest of title, first prompt,
  repo, branch, folder, issue keys and summary fingerprint). `resolve()` keeps it through collector
  refreshes and rule rebinds until that evidence changes, then files the session afresh.
- **Undo** (`POST /sessions/{id}/autolink/undo`, the Sessions page's "Filed … by the dream sequence ·
  Undo") or a manual relink writes an `autolink_rejections` tombstone for (session, task). The pair is
  blocked for good; if the rejected task would still rank first, the session stays unfiled rather than
  promoting the runner-up.
- Inferred links never start work: runs start only from a task, with the execution-target rules.

### Cutover (stage E)

`GET /remotea/parity` adds `cutover {ready, streak, needed: 7, days[7]}`, from `agent_parity_days` (one
row per owner and day in the owner's timezone, counting reports, matched, mismatched and missing). A day
is **clean** when it had at least one parity report and no mismatch; sessions only remotea saw (`missing`)
do not count against it. `ready` means 7 consecutive clean days ending today, or yesterday while today
has no reports yet. The page's parity summary then reads "· ready for cutover".

The procedure is the host's (`remotea eridani collector authoritative`, the `status --json` v2
interlock, `remotea hooks --remove-legacy-collector`); Davin runs it. The exact steps and the rollback are
in [`scripts/session_collector/README.md` § Retirement](../scripts/session_collector/README.md#retirement),
which mirrors remotea's `docs/remotea/eridani-link.md` § Stage E: collector cutover.

## Run freshness (migration 0033)

- **Base.** remotea fetches `origin/<base>` before basing or reusing a worktree on the target's own base
  (local wins only when it already contains origin; a failed fetch keeps local and warns the agent). The
  spawn outcome's `base_sha, base_ref, fetch {ok, error?}, behind, ahead` land in `agent_runs.base_facts`;
  `public_run.base` shows them with `behind` taken live from the latest `worktrees.report` row for the
  run's branch (`live`, `as_of`), else the spawn-time count. The drawer prints "based on main @ <sha> ·
  N behind · as of …". An older host sends no `base_sha`: `base` is null and nothing is shown.
- **Reuse.** A reused worktree behind its base gets a merge-first note at the top of the spawn prompt from
  the host, plus the re-point/renumber instruction when the spawn carries `migration {number, dir}`.
- **Migration numbers** (`migration_numbers.py`). A target takes part only when something says where its
  migrations live: the folder and file pattern set in Settings (stored on the counter), else the host's
  `migrations {dir, pattern?}` in remotea `eridani.json`. The host scans and reports the chain head only for
  its own folder (nothing Eridani sends becomes a path on the host), so a Settings folder that differs
  counts no host head and relies on the override. `submit_plan` with `migration: true` reserves max(host-reported head, the
  operator's override, the highest held number) + 1 under the counter's row lock (`migration_counters`
  per host target; `migration_reservations`, unique while held). The number is frozen on the plan,
  shown by `task_plan` (`run.migration`, `migration_note`) and `render_plan`, and sent on later spawns
  of the task. Released when the run ends stopped/failed/capped or a later plan drops the flag, and only
  if no higher number is held on that counter; otherwise the gap stays. A completed run keeps its number,
  and a later run of the same task reuses it. Settings → Connections → remotea lists each such target (head, override,
  last reserved, next) with folder, file pattern and override fields (`GET/POST /remotea/migrations
  {host_id, host_target, dir?, pattern?, override?}`; empty values fall back to the host / clear). A reservation cannot see
  migrations merged after it: the host's merge-first note and `tests/test_release_migrations.py` cover that.
- **Version pairing** (`remotea_release.py`). `hello.version` is stored on the host and returned by
  `public_host` with `update: {latest, url} | null`: set when the host trails the latest GitHub release of
  rdavisdunham/remotea by more than one minor version (or any major). The release is looked up in a
  background thread at most every 6 h (15 min after a failure); a 404 (no release yet) or a failure
  shows no note.

## Run summaries (migration 0036)

Each run card carries one plain-language line on what the agent is doing right now ("Adding the parity test for the
summary job"), written by a small model (`run_summaries.py`).

- **Columns** on `agent_runs`: `summary` (≤ 140 characters, `""` until the first one), `summary_at` (when it was
  written), `activity_seq` (bumped by `run_summaries.touch` whenever something the line reflects lands: every root
  turn of the run's generation in `record_turn`, a question opened or resolved, and every `transition`, which covers
  verdicts and judge failures) and `summary_seq` (the `activity_seq` the last summary job handled).
  `summary_seq < activity_seq` means a newer line is due.
- **Debounce.** `touch` queues one `agent_summary` job (queue `jarvis-memory`, like the judge) unless one for the run
  is pending (queued/dispatched/running/retrying, younger than 10 minutes) or one was queued in the last 30 s. An event
  inside the window only counts; the worker's maintenance scan (`run_summaries.queue_due`, every cycle) queues the
  follow-up once the window has passed. So a run gets at most one summary per ~30 s, and only after something
  happened: never on page load and never per change-feed tick. Nothing is queued while no OpenAI key is configured
  (`activity_seq` still counts, so turning it on catches up).
- **Terminal runs.** The transition into a terminal state counts, so a run gets one closing line ("Stopped after …").
  Once `summary_seq` has caught up the run is skipped, and a run that ended more than an hour ago is never summarized
  again.
- **Evidence** (`build_evidence`, pure and bounded): the task title (200 characters), the run state and its
  `REASONS` text, the frozen plan's current step (the first `doing`, else the first not done; 300 characters) with
  "N of M steps done", the last two root (`assistant`) turns of the run or its session cut to 1,500 characters each
  (head and tail), the latest verify or judge attempt (status, verdict, exit code, gap cut to 400) and the newest open
  question (400). The user message is capped at 6,000 characters.
- **Model.** Luna through `memory_learning.extraction_request` (the only model path the judge accepts), strict schema
  `{summary: string}`, no tools, `max_completion_tokens` 200, prompt version `summary-v1`. The answer is cut to one line
  of at most 140 characters at a word.
- **Prompt injection.** Every free text (task title, step, gap, question, turns) sits inside an `agent_prompts.fence`
  nonce fence with `<<<`/`>>>` runs broken up, so turn text cannot close its fence; the system prompt is fixed and says
  the fenced blocks are untrusted data to describe, never instructions, and that the line must not claim the work is
  done, verified or approved unless the run state says so. The model can only fill one short string, which is shown
  as text and never acted on.
- **Result.** On success the job stores `summary`, `summary_at`, `summary_seq` and emits `agent.changed`, so cards
  update live over the change feed. A non-summary (refusal, invalid or empty output, timeout, budget, provider
  unavailable) keeps the old line but still marks that activity handled, so a broken provider costs at most one call
  per event window instead of a retry loop. A job that finds nothing new (`summary_seq ≥ activity_seq`) is cancelled
  without a call.
- **Cost.** Cost feature `agent_summaries` ("Agent run summaries"): its own line in the usage report (Settings → System) next to the judge's
  `agent_control`. Each call also writes an `agent_attempts` row with `kind: "summary"` (status `done`/`failed`, model,
  `prompt_version`, cost, latency, the run tuple, and `result {seq, summary | non_verdict, detail}`) the way judge
  attempts do. Summary attempts are written settled (never queued or running, no deadline), so the attempt sweep,
  supersede and judge queries never see them; `GET /runs/{id}` leaves them out of `attempts`; retention deletes them
  with the run's other attempts.
- **Surfaces.** `public_run` carries `summary`, `summary_at` and `summary_stale` (a line exists and a newer one is due).
  The run row (`RunsPanel.tsx` `RunRow`, on the remotea page and in the topbar Runs panel) shows the line under the
  state chip with a quiet "updating…" while `summary_stale`; the task-row chip (`RunButton.tsx`) uses it as its
  tooltip. `control_status` gives each run `doing` and `doing_as_of`, so Eri can say "the agent is adding the parity
  test". `agent_question` and `agent_stuck` pushes append "Latest: <summary>" to the body when there is one.

Tests: `tests/test_run_summaries.py`; web `apps/web/src/run-anywhere.test.tsx` (`RunRow`, `RunButton`).

## Swarm mode (migration 0040)

Roadmap task 71f97d98 (Astra's E-Q4). **Play on a project opens a swarm plan, not a run.** In the organization
views a project row's Play (`RunButton` with `project`) opens the swarm dialog (`Swarm.tsx` `SwarmDialog`); the remotea
page lists every open plan under **Swarms** (`SwarmList`). Nothing spawns until Davin presses Start.

**Planning** (`swarm.py`, `POST /remotea/swarms {record_id | task_id, harness?, cap?, fresh?}`). The project's open
work is every descendant record that carries an open task, without descending into a task (its sub-tasks belong to its
own run, F02), at most 40; a task that already has a working run is listed under `excluded`. For each task the planner
gets its title, PRD note (or notes) and step titles (each cut to 1,200 characters, fenced as data), where it would run
(`readiness_for`, read-only: target, repository) and its declared Blocks links. The planning model is the `swarm_plan`
job in `model_map.py` (cost feature `agent_control`, through `memory_learning.profile_request`, so today Luna; the task
names Fable, which needs a priced Anthropic route first). It answers batches of task refs with a one-line reason each,
and up to six files or areas per task it expects to change. The answer is refused, and the **deterministic fallback**
plans instead, when it loses, repeats or invents a task or puts a task in the same batch as (or before) a task that
blocks it; the fallback also runs when no model key is configured or the call fails (`planner {source: "fallback",
note}`). The fallback: a task goes after every task that blocks it and never beside another task on the same
repository (host, target and repo; tasks with no target yet count as one repository), otherwise as early as possible.
A project with an open plan gets that plan back unless `fresh` (which cancels it while nothing of it has started).
`409 NO_TASKS` for a project with no open tasks (the dialog offers **Run the project itself**).

**Editing** (`POST /remotea/swarms/{id}/edit {expected_revision, ops[]}`, all ops in one transaction, `REVISION_CONFLICT`
on a stale revision). Ops: `exclude {task_id}` / `include {task_id, batch_id?}` (leave a task out, bring back one you
left out), `move {task_id, batch_id | "new"}`, `split {batch_id, task_ids}` (a new batch right after it), `merge
{batch_id, into}`, `order {batch_id, index}`, `cap {value 1–10}` (working runs per remotea host at once), `harness
{claude | codex}`, `auto_continue {bool}`, and `retry {task_id}` (a task of the running batch that could not start).
Only batches that have not started change (`409 BATCH_STARTED`); empty batches are dropped. Blocks links are advisory
(as in the organization model), so a move that breaks one is allowed and shown as a warning. `warnings[]`: `ORDER`
(a task not after its blocker), `BLOCKED_OUTSIDE` (an unfinished blocker outside the plan), `SAME_REPO` (two tasks on
one repository in one batch whose predicted areas overlap or are unknown), `NEEDS_TARGET` (choose where it runs first).

**Start** (`POST /remotea/swarms/{id}/start {expected_revision, batch_id?, all?}`, gated like `POST /runs`). Batches run
in order: only the next batch starts (`409 BATCH_ORDER` names it), and not while another is running (`409 SWARM_BUSY`).
`all: true` also turns on continue automatically. Every task goes through `agent_control.start_run`, the normal start
path, inside a savepoint: the same refusals (`NEEDS_TARGET`, `HOST_OFFLINE`, `HARNESS_UNAVAILABLE`,
`MCP_UNCONFIGURED`, …) are kept on the item (`error {code, message}`, shown as **Could not start** with Retry) and do
not stop the others; a task that already has a run adopts it. Each run is a normal run: one worktree and branch per
task, its own interview/plan/verify/judge loop and the same approval gate. Its spawn prompt ends with a **swarm
brief** (`swarm.brief`, passed to `start_run` as `brief`, never from a request body): the batch and the planner's
reason, the tasks running beside it on the same repository ("stay inside your task"), the areas it is expected to
change, and that a migration number comes only from Eridani's reservation (`submit_plan` with `migration: true`,
[Run freshness](#run-freshness-migration-0033)), never from the agent's own count. Parallel runs on one repository are
allowed when the plan put them together; each still gets its own branch (S-S4's one writer per branch holds).

**The per-host cap.** A batch starts at most `cap` working runs per remotea host (counting every unsettled run of the
owner on that host that is not waiting for review or a merge); the rest wait as **Waiting for a free slot**. remotea's
own `limits.maxRuns` still applies per target: a spawn it refuses as `target_busy` fails that run with
`spawn_rejected`, and the swarm queues the task again (at most 3 times).

**Sequencing** (`swarm.advance_one`, run by the worker's sweep next to `run_coordinator.sweep`, and right after a
start or an edit). It fills free slots, then settles the running batch once every task is settled: its run is terminal,
or waits for a human or a merge (`awaiting_approval`, `awaiting_integration`, `needs_you(awaiting_merge)`), or it could
not start. Then the next batch starts only by Davin's Start, or by itself when continue automatically is on and the
integrate step found no overlap. The last batch settling makes the plan `done`. `POST /remotea/swarms/{id}/cancel
{expected_revision}` (works while remotea is off) keeps later batches from starting; started runs keep going and are
stopped from their own drawers.

**Integrate.** For a batch that has started, every repository with two or more of its runs on a branch gets
`integrate.repos[]`: each run's branch, pull requests (the session's `pr_urls`), merged / ahead from the host's
`worktrees.report`, changed `files` when the report lists them (remotea does not yet; null), a `merge_order` (merged
first, then those with a PR, then the fewest commits ahead) and `conflicts` (shared paths, by files when both runs have
them, else by the planner's predicted areas, `basis`). `blocking` (an overlap among unmerged work) holds continue
automatically for Davin's Start.

**Cost before Start.** Each batch carries `cost`: the average Eridani spend per finished run (`agent_attempts.cost`:
judge and summary calls) over the owner's last 30 settled runs times the batch's size, with the average turns and
minutes, or "no finished runs yet". The coding agent's own usage is billed by its harness and not recorded by Eridani.

**Storage.** `agent_swarms` (owner, account, project record, title, `state` draft | running | done | canceled,
`revision`, `cap`, `auto_continue`, `harness`, `batches` JSONB `[{id, reason, state: pending | running | settled,
started_at, settled_at, items: [{task_id, run_id, error, retries}]}]`, `tasks` (the planner's facts per task), `excluded`
and `planner`). Every change emits `agent.changed` with the swarm id.

**Not yet.** Eri has no swarm tool (a start would need the two-phase confirmation of `control_start` for a whole batch);
the planner cannot merge two tasks into one run (it can only put them in series); changed-file overlap uses the
planner's predicted areas until remotea's `worktrees.report` lists changed files; the cost estimate does not cover the
harness's own tokens.

| Route | Gated | |
|---|---|---|
| `GET /swarms?active=&limit=` | no (owner only) | plans, newest first, each in the detail shape |
| `POST /swarms` | yes | plan a project (above); starts nothing |
| `GET /swarms/{id}` | no (owner only) | `{id, record_id, title, state, revision, cap, auto_continue, harness, planner, excluded, warnings, next_batch, batches: [{id, number, reason, state, items: [{task_id, title, repo, target, blocked_by, areas, why, status, error, retries, run}], cost, integrate}]}` |
| `POST /swarms/{id}/edit` | yes | edit ops (above) |
| `POST /swarms/{id}/start` | yes | start the next batch (`all`: and each next one) |
| `POST /swarms/{id}/cancel` | no (owner only) | no later batch starts |

## Hyperplan and roles (migration 0041)

Three roles in one run, each with its own harness and model: the **planner** drafts the plan and talks to the owner
(one session through all planning rounds), the **reviewer** reads the plan in a fresh read-only session every round,
and the **worker** executes the approved plan in a fresh session. A worker "same as the planner" means the planner's
session continues into execution, which is the classic run. The host side (remotea #36, `spawn.roles`) is in
`docs/remotea/eridani-link.md` "Roles and read-only modes".

**Roles on the run.** `agent_runs.roles` (JSONB, null = a classic run) is `{planner: {harness, model}, reviewer:
{harness, model} | null, worker: {harness, model} | null, rounds, approve_final}`; `worker` is null when it equals the
planner. `agent_runs.plan_round` counts the reviews completed. `run.tool` / `run.model` always describe the session
that owns the run now: the planner, then the worker after the handoff, so filters and the header keep working.

**Starting.** `StartInput` (and Eri's `control.start`) gains `reviewer: {harness, model?} | null`, `worker: {harness,
model?} | null`, `rounds` (default 2, 1–10) and `approve_final` (default true); `harness` / `model` are the planner.
A role's `model` defaults to `HARNESSES[harness]` (`opus`, `gpt-5.6-sol`); any id matching `MODEL` is accepted
(`fable`, `gpt-6-astra`). Hyperplan is on when `reviewer` is set; a split worker when `worker` differs from the
planner. Either one:

- refuses `freestyle` (`INVALID_ARGUMENT`); needs the host's hello `caps` to list `spawn.roles` (`409 HOST_NO_ROLES`)
  and each role's harness to be spawnable on the target (`409 HARNESS_UNAVAILABLE` with `data.role`);
- forces a planning start: a task without a PRD interviews as before; one that would skip the interview starts in
  `planning` with `prompts.planning_start` instead of freezing its plan (`sync_state.start_phase`), so nothing runs
  unreviewed;
- spawns the planner with `role: "planner"`, and with `mode: "plan"` (read-only) when a different worker will do the
  work.

`GET /remotea/runs/preflight` takes `reviewer_harness` / `worker_harness` and answers `roles` (true / false: the
resolved target's host advertises `spawn.roles`; null while no target resolves) plus a blocking `HOST_NO_ROLES` or
`HARNESS_UNAVAILABLE` problem with `role` for a role it cannot start.

**States.** `plan_reviewing` (the reviewer works, the planner idles) and `handing_off` (the planner is stopped, the
worker spawning). `PLANNING` includes `plan_reviewing`; `submit_plan` is taken only in `PLAN_SUBMIT` (interviewing,
planning, replanning). `phase()` resumes `handing_off` while a handoff is pending and `plan_reviewing` while a review
is live. Transitions: `planning → plan_reviewing → planning`; `planning → interviewing → planning` (a hyperplan
planner may call `task_interview` while planning; a classic run still gets `PLAN_LOCKED`); `planning →
needs_you(plan_approval) → planning` (revise) `| handing_off` (approve, split worker) `| executing` (approve);
`planning → handing_off → executing`; `needs_you → plan_reviewing | handing_off` (resume). New `state_reason`s:
`plan_approval` and `host_busy` (needs_you), `reviewer_blocked` (reserved), and the working reasons `review`,
`review_done`, `review_failed`, `interview`, `plan_approved`, `plan_revise`, `handoff`.

**The loop** (`settle_planning`). When a planner turn ends with a new plan revision and no open question or
interview: while `plan_round < rounds` and the last review was not a clean approve (verdict `approve` with no
blocking or major finding), a review row is created (round `plan_round + 1`) and a reviewer spawned → `plan_reviewing`.
Otherwise the existing proposals check (`needs_you(plan_review)`; deciding the proposals is also the approval), then
`needs_you(plan_approval)` when `approve_final`, else `begin_execution`. A planner turn that ends without resubmitting
after a `revise` review is `needs_you(no_plan)`; after a clean approve the reviewed revision stands.

**The reviewer** (`run_coordinator.spawn_reviewer`). A spawn with `role: "reviewer"`, `mode: "review"`, `share:
true`, the next `attempt` (a fresh surface every round), the reviewer's harness and model, and a reviewer token
(`mint_review_token`: same `ert_…` shape, derived over the review row, only its hash stored on the review). The
prompt (`prompts.plan_review`) carries the rendered plan, the resolution log of earlier findings with the
planner's decisions, what to check, the severities and the one tool call. Its SessionStart context carries its
own token and the reviewed revision, never the run token. It is never the run's `session_pk`: `attach` records it
on the review, its questions never pause the run, its status and exit never move the run. Its turn end
(`agent_link.turn`, matched by surface, interim and synthetic turns ignored) runs `finish_review`: the
`submit_plan_review` call, or else its turn text as a `revise` summary (`via: turn`), status `done`,
`plan_round += 1`, a `stop {mode: kill, role: reviewer}` for its surface (fenced on the reviewer's own
`incarnation`), and the `plan_critique` prompt to the planner (`incorporate`, `final` on the last round or after a
clean approve) → `planning`.

**`submit_plan_review`** (MCP, `agents:session`, reviewer token only; the planner's token gets `403 REVIEWER_ONLY`):
`{run_token, verdict: approve|revise, findings: [{id, severity: blocking|major|minor, area, issue, suggestion}] (≤ 50,
distinct ids), questions_for_user: [str] (≤ 12), summary}` → `{review_id, round, verdict, counts: {blocking, major,
minor}, next}`. Once per round (`409 REVIEW_SUBMITTED`); outside `plan_reviewing` `409 REVIEW_CLOSED`; after the
round the token is dead (`401 RUN_TOKEN_REQUIRED`). A reviewer token on any other in-session tool than `task_plan`
is `403 REVIEWER_SCOPE`.

**`submit_plan`** gains `review_resolutions: [{finding_id, decision: accepted|rejected|deferred, note}]`, stored on
the last review as `resolutions`. After a `revise` review of the current revision, every blocking and major finding
needs one (`409 REVIEW_UNRESOLVED {round, unresolved}`); an unknown id is `INVALID_ARGUMENT`. The planner decides
which of the reviewer's `questions_for_user` to forward with `task_interview`; the answers come back through
`interview_answers`, which is round-aware ("continue incorporating review N").

**Failures.** A reviewer has `REVIEW_SPAWN_DEADLINE` (15 min) to start and `REVIEW_DEADLINE` (20 min) to finish
(`sweep_reviews`, beside the judge and verify deadlines); a timeout, crash, spawn failure or a reviewer the host lost
at reconnect is `failed` (`via: timeout|failed`), its surface killed, `review_failed` on the timeline, and the loop
moves on as if no review were due (no re-spawn). A spawn the host refuses `target_busy` (at `limits.maxRuns`; a
read-only spawn counts) is sent again with backoff (30 s doubling to 5 min, same attempt and surface); after
`REVIEW_BUSY_LIMIT` (10 min) the run is `needs_you(host_busy)` and the retries continue every 5 minutes until one
starts, which resumes `plan_reviewing`. The worker's spawn follows the same rule from `handing_off`; any other
worker spawn failure fails the run (`spawn_failed` / `spawn_rejected`), and a worker spawn that expires is
`needs_you(spawn_expired)`. Stopping the run skips the live review and kills its surface (a reviewer spawn still
queued is canceled; the run's own spawn, never a reviewer's, decides whether the stop settles at once).

**The final plan.** `needs_you(plan_approval)` shows the plan and the review history. `POST
/remotea/runs/{id}/plan-review {action: "approve" | "revise", note?, expected_revision}` (the existing route; the
proposals path is unchanged and refuses `action` elsewhere): `approve` → `begin_execution`; `revise` → `planning`
with the note as a `plan_critique` prompt. Eri: `control_plan_review` with `action` (and `note`); `control_status`
lists `plan_approval {plan_revision, plan_round, title, criteria, reviews [{round, verdict, counts, summary}],
next}` and `actor`.

**The handoff** (`begin_execution` → `hand_off`, the first time on a split-worker run): `stop {surf, incarnation,
mode: kill, role: planner}` for the planner; `attempt += 1`; `sync_state.handoff = {attempt, surf, at, from_session_pk,
from_surf, planner {harness, model}, worker {harness, model}}`; `session_pk` cleared; `tool` / `model` set to the
worker's; a fresh run token (the planner's no longer matches: revoked); a spawn with `role: "worker"` (writable, the
run's normal token scope, the plan's verify and migration) and the prompt `prompts.worker(plan, digest)`: the frozen
plan plus the "decisions already made" (rejected and deferred findings with the planner's reasons, the owner's
interview answers); a `handoff` timeline entry and a `sent` entry with kind `handoff`; → `handing_off`. `attach`
accepts exactly that attempt's surface as the new `session_pk` (a late report from the retired planner cannot take
the run back, and its SessionStart gets no token), sets `handoff.done` → `executing`. Verify, judge, steer, message
and approve then target the worker; re-plans stay with the worker (`replanning`).

**The link.** Spawns carry `role` / `mode` / `share` as above (the host echoes `role` and `mode`; a harness without a
read-only launch is `rejected mode_unsupported`, which fails the round or the start like any refusal). `runs.snapshot`
gets one entry per surface for a run with roles (`run_coordinator.desired_entries`): the owner's entry with `surf`
(or, before the worker has one, `attempt`) and the verify config, each live reviewer's `{surf, desired: continue}`
without verify, and `{surf, desired: stop, stop_mode: kill}` for every surface the run retired (finished reviewers,
the planner after the handoff), so a stop lost in transit is carried out after a reconnect. A classic run's entry has
neither `surf` nor `attempt`, as before. `verify` is only ever sent while the owner's session is up (execution);
`POST …/verify` is `409 NOT_STARTED` during `plan_reviewing` and `handing_off`, since remotea binds verify to the run's
newest non-reviewer session and a run whose only session is a reviewer has none.

**Reading it.** `public_run` adds:

```
roles:      null | {planner: {harness, model}, reviewer: {harness, model} | null, worker: {harness, model} | null,
                    rounds, approve_final}
plan_round: int
reviews:    [{id, round, plan_revision, harness, model, attempt, status: spawning|reviewing|done|failed|skipped,
              verdict: approve|revise|null, via: tool|turn|timeout|failed|"", findings, questions, summary,
              resolutions, counts: {blocking, major, minor}, started_at, finished_at, created_at}]   ([] for classic)
actor:      {role: agent|planner|reviewer|worker, harness, model, round?, rounds?}   (round/rounds for a reviewer)
handoff:    null | {attempt, at, done?, attached_at?, planner {harness, model}, worker {harness, model}, retry_at?,
                    blocked_since?, retries?}
```

Timeline events (`history`): `review_started {review_id, round, rounds, plan_revision, harness, model}`,
`review_submitted {review_id, round, verdict, blocking, major, minor}`, `review {review_id, round, rounds, verdict,
via, harness, model, blocking, major, minor}`, `review_blocked {review_id, round, retry_at, detail}`,
`review_failed` / `review_skipped {review_id, round, detail?}`, `plan_submitted` (now with `review_round` when it
answered a review), `plan_approved`, `plan_revise`, `handoff {attempt, harness, model, actor}`, `handoff_blocked
{retry_at, detail}`, `attached` (with `role: worker` at the handoff), and `command` / `outcome` entries carry
`role` for role commands.

**Security.** A reviewer token can call `task_plan` and `submit_plan_review` only, and only while its round is live;
reviewers never become `session_pk` and their turns, questions, status and exit never move the run. The planner's
token is revoked at the handoff (the worker gets a fresh one), and the retired planner's SessionStart gets no token.
Read-only is a harness-level mode (Claude `--permission-mode dontAsk` with an allow list, Codex `--sandbox
read-only`): Bash allow rules match command text and are not a security boundary, and the reviewer shares the
writer's worktree, so what it can read is bounded by the same git-ignore hygiene as the planner's. Prompts tell the
read-only roles to ask through `task_interview`, never AskUserQuestion (denied under `dontAsk`).

**In the app** (`RunTaskDialog.tsx`, `Remotea.tsx`, `remotea-model.ts`).

- *Run dialog.* The agent and model choice is the **Planner**. **Worker** (every harness and model, default "Same as
  planner") is always shown; **Hyperplan** reveals **Reviewer** (harness and model), **Rounds** (default 2, 1–10)
  and **Approve final plan before running** (default on; also shown for a separate worker alone, which pauses the
  same way). `startBody` sends `reviewer` + `rounds`, `worker` and `approve_final` only when they make a role run
  (a worker equal to the planner is left out), so a classic start's body is unchanged. The preflight is read with
  `reviewer_harness` / `worker_harness`; a harness the target cannot start is greyed out in every role's list, a
  problem with `role` shows under that role, and `roles: false` (or a `409 HOST_NO_ROLES` at Start) turns the
  worker and Hyperplan off with the reason and starts classic. Freestyle turns them off too. A `409
  HARNESS_UNAVAILABLE` with `data.role` shows under the role. The last planner, worker, reviewer, rounds and
  approval choices are kept in `localStorage` (`eri-run-draft`; never the target or freestyle), and restored only
  where the model is still offered. `HARNESSES` lists Fable (`fable`) for Claude Code and GPT-6 Astra
  (`gpt-6-astra`) for Codex after the defaults; the drift guard compares the default (first) model per harness.
- *Timeline.* The planner's submissions read "Draft N · Fable" until the handoff (a re-plan after it is the
  worker's "Plan submitted"); each review round is one entry, "Review N · GPT-6 Astra: revise — 2 blocking, 3
  minor" / "approved" / "timed out" / "failed" / "reviewing", placed where it ended and unfolding to its findings
  (severity, area, issue, suggestion), the planner's resolution of each and the reviewer's questions; "Final plan
  approved by you" / "sent back", "Handed to Sonnet", the first `review_blocked` / `handoff_blocked` of a round,
  and the `plan_critique` / `handoff` texts as "Sent by Eridani · …". Questions between rounds are the planner's
  `task_interview`, answered in the usual interview form while the run is `interviewing`.
- *Final-plan approval.* In `needs_you(plan_approval)` a card shows the frozen plan (title, plan text, steps,
  criteria) and the review history, with **Approve** and **Revise with note** (the note is required for revise),
  posting `{action, note?, expected_revision}`; a `REVISION_CONFLICT` reloads the run like the other controls. The
  review packet, Edit criteria and Judge again stay off at this stop, and the footer's Approve reads **Complete
  task** there, since it completes the task without running the plan. Run chips say "Approve plan" and the run
  counts under "Review the plan".
- *Who acts.* A role run's drawer header, run rows and the task row chip's tooltip carry an actor chip from
  `actor` and `state`: "Planning · Fable", "Reviewing 1/2 · GPT-6 Astra", "Handing off → Sonnet", "Working ·
  Sonnet". Classic and settled runs show none. `needs_you(host_busy)` says which session waits (the reviewer for
  round N, or the worker) and that Eridani keeps retrying, and counts under "Repair the host or setup".

**Not included.** Swarm starts have no roles yet; a replan during execution stays with the worker (escalating it back
to the planner is a follow-up); `reviewer_blocked` is reserved.

## Security model (stage F, migration 0031)

Fixes from the 2026-10-28 security audit. Regression tests are in `tests/test_agent_security.py`, one
per finding, each the audit's reproduction turned around.

### Personal workspace only

remotea is a single-person feature until the multi-user design on the roadmap
([TODO](TODO.md#explicitly-deferred-not-forgotten)) lands. `/api/v1/remotea` is a personal path
(`access.PERSONAL_PATHS`), and every remotea route, the link and the `control.*` tools refuse with
`PERSONAL_WORKSPACE` when the namespace is a shared workspace or the caller is not its owner
(`remotea_routes.owner_only`, `agent_control.owner_control`). `bot_access` refuses to mint
`agents:control` or `agents:session` keys in a shared namespace, and refuses such a key at use if one
already exists there. Turning remotea off revokes the link key; a live host still gets `bye{disabled}`.

### Hosts and frames

- Session-keyed frames (`turn`, `status`, `exit`, `question`, `session.report`) are host-bound: a frame
  naming a session owned by another host, or by another host's run, is refused with `RUN_NOT_BOUND`
  (`agent_link.host_session`) and never moves that run.
- A `server_id` stays bound to its first credential unless the new key is a rotation of it (same owner,
  account and name); otherwise `bye{host_bound}`. One key registers at most 3 hosts (`bye{host_limit}`).
- A host that advertises the `digest` cap must stamp every frame carrying an `op_id` (`DIGEST_REQUIRED`
  otherwise). `ack` is exempt: it only echoes the op_id of the command it acknowledges. Until 2026-10-30
  Eridani refused those acks too, so a current host's command acks were answered with an error ack and never
  recorded (a spawn's `surf`/`incarnation` came only from the later outcome, and a command whose outcome took
  over 60 s was resent); the end-to-end protocol test found it on its first run.
- Snapshot parts are numbered `1..n` (as remotea sends them; the brief said `i < n`, which would refuse
  every real final part), bounded in bytes, and partial snapshots and chunks expire after 60 s.
- `context.request` answers `scope: "run"` (plan and run token) only for a run bound to this host;
  otherwise `scope: "candidates"` with only the fenced candidate list (≤ 4000 chars).

### Actors, the plan floor and the completion gate

- Actor kinds (`agent_actor.KINDS`): `human` (the app, or the owner's confirmed yes), `eri_work` (the model
  in chat or voice), `run_session` (an agent inside a run, proved by its run token), `run_judge`,
  `source_sync` (Linear) and `bot`. Only `human`, `source_sync` and `run_judge` may complete a run-owned
  task or edit its gate fields. An actor may carry a `capability` (`plan_write` for `submit_plan`: gate
  fields under the plan floor, never completion), recorded with every write.
- Plan floor (`agent_plan.plan_floor`): an agent's plan may only add. Human criteria and a human verify
  are kept; what the agent adds is marked `by: "agent"`; the completion policy never drops. Clients cannot
  send `by` themselves (`Literal["agent"]`).
- Plan review (`agent_plan.plan_proposals`, `review_plan`): the agent may *propose* a change to what a human
  set, never apply it. In `submit_plan` a human criterion sent back with different text (plus `because`), or
  with `drop: true`, a different verify, or a lower completion policy becomes a proposal stored on the frozen
  plan (`agent_run_plans.proposals`). Human criteria the agent simply omits are kept, not proposed for
  removal, so a plan that only adds never pauses. When the planning turn ends with proposals,
  `settle_planning` moves the run to `needs_you` with reason `plan_review` instead of executing; the page
  shows each change (original, proposed, the agent's reason) with Accept / Keep and an editable text.
  `POST /api/v1/remotea/runs/{id}/plan-review {decisions:[{id, accept, text?}], verify, completion_policy,
  expected_revision}` (the revision is required: the decisions are about the proposals the reviewer saw) writes the accepted changes to the task as a human edit, freezes revision N+1 without proposals and
  starts execution; the execute prompt opens with what was accepted and kept.
- `reason_detail` on a run: the full text behind the current state (the judge's notes for a judge-caused
  `needs_you`, the proposal summary for `plan_review`, else the transition's detail); `reason_text` stays
  the one-line version.
- The task's PRD note is read-only to every actor but `human` while the run is unsettled
  (`agent_actor.note_gate`, `409 RUN_OWNS_PRD`; see "Actors and the completion gate"), so the live note
  a human opens matches the frozen plan unless the human changed it.
- Every path that completes a task (`task.update`, `task.complete`, `schedule.complete`,
  `notification.complete`, Linear sync, `auto_complete`) passes `agent_actor.gate`; `finish_task` runs it
  again as a backstop. `auto_complete` never completes a task whose plan has human criteria
  (`human_gate`).
- `task_step` needs a verified run token or a key with `tasks:write`. The in-session tools refuse while
  remotea is off.

### Two-phase control from Eri

`control.start` and `control.message` from Eri only propose: they store an `agent_pending_controls` row
(10-minute expiry) and nothing reaches the host. The owner confirms on the card in chat, on the remotea page
(`GET /remotea/pending`, `POST /remotea/pending/{id}/confirm|decline`) or by saying yes: `control.confirm`
works only from a later request by the same account whose own text is the owner's yes, for the newest
pending item. The yes (`agent_control.is_yes`) is a message that starts with an affirmative (yes, yeah,
yep, yup, sure, ok, okay, confirm, go ahead, go, do it, start it, send it, please do), has at most 12 words
in its first 200 characters, is not a question and contains no but / instead / actually / not / don't /
no / wait / cancel / change: "yes start the agent", "yeah go for it" and the joined voice speech "ok cool
yes" count; "yes but change the model" does not (`CONFIRMATION_REQUIRED`). `control.stop`,
`control.answer`, `control.approve`, `control.plan_review` and `control_status` stay immediate.

A proposed start resolves its target when proposed (read-only): when one resolves, its id goes into the
row's `arguments` and the summary names it ("… in jarvis"), and confirm starts there. When none does, the
row is still created and the result says `needs_target: true` (with `task_id`) and lists `targets`: the
saved candidates by `execution_target_id` and the advertised-but-unsaved ones by the key
`host:<host_id>:<host_target>` (the same options the card's picker builds), so Eri asks "run it in jarvis
or remotea?" and confirms with `control.confirm {execution_target_id}`; the owner's short answer naming the
chosen place counts as their yes. An advertised choice is saved first (`save_target`, for the task's
nearest non-task record, else the task itself, as the picker defaults it), then the run starts there.
`control_status` lists the same open proposals under `pending`. An offline server or a harness it cannot start is checked the same
way and returned as `problems` for Eri to relay; neither blocks the proposal. `GET /remotea/pending` lists
each start with that readiness (`needs_target`, the picker's `need` data, `target`, `problems`), and the
card shows the picker in place of Confirm; picking saves the choice and confirms with
`POST /remotea/pending/{id}/confirm {execution_target_id}` (owner-checked). A confirm refused with
`NEEDS_TARGET`, `HOST_OFFLINE`, `TARGET_NOT_READY` or `HARNESS_UNAVAILABLE` leaves the row pending. Listing the pending items
also marks the expired ones `expired` (a refused confirm rolls back its own writes). Background,
parked-continuation and bot work get no `control_*` tools (`tool_policy.live_work`). Every text sent to an
agent is a `sent` entry on the run's timeline with who sent it; every `control.*` call is in the action
journal, visible but never undoable.

### Prompt injection

Task text, Linear text and candidate titles reach the agent only inside nonce fences
(`agent_prompts.fence`, `<<<LABEL_xxxxxxxxxxxx` … `LABEL_xxxxxxxxxxxx>>>`), with `<<<`/`>>>` broken up
inside. Linear-sourced tasks say so ("written by an external party").

### Run tokens

A run token is HMAC-SHA256 over the run id and a per-spawn random salt, keyed by HKDF of the integration
key (info `eridani-run-token-v1`); only the salt is stored. It expires once, at the wall-clock limit plus
24 h; `context.request` does not slide it. Only owner waits (which the wall clock excludes) and wall-clock grants
move the expiry forward, by the same amount. The spawn command's copy becomes `[redacted]` once the spawn
is acked or attached. Rotation is per spawn, not per generation: remotea passes the token as
`ERIDANI_RUN_TOKEN` at spawn, and a steer delivers no new SessionStart, so a rotated token could not reach
the agent.

### Redaction

`redaction.py` holds the 12 patterns of remotea's `docs/remotea/redaction.md` (the collector loads the
same file). Host free text is redacted on ingestion: turn text and prompt, status and exit reasons,
question text and choices, `question.resolved` details, verify tails and errors, outcome details, and
the session ledger's title, prompts and excerpt.

### Retention and redaction (`agent_retention.py`)

`redaction.py` (above) removes secrets from what is stored. This section is about how long run content is
stored at all, and every place a copy of it lives.

**The window.** A settled run's content is kept for the owner's run retention window: `agent_retention_days`
in `OwnerSettings` (default **90**, 1–3650), set with `POST /api/v1/remotea/retention {days}` (owner-only;
works while remotea is off) and returned as `retention_days` by `GET /remotea/status`. There is no Settings
control for it yet; the route is the way to change it. The window starts at the run's `finished_at`; an
unsettled run is never swept.

**At the window** the worker's housekeeping pass (every 12th supervisor cycle, `sweep`) deletes the run's
child records and then **redacts** the skeleton that outlives them (`redact`, at most 200 runs per pass; a
run is redacted once, marked by a `redacted` history entry with reason `retention`):

- every `history[]` entry's `text` and `detail` become `[redacted after retention]`; `at`, `from`, `to`,
  `event`, `reason`, `revision`, `actor`, `kind`, `mode`, `verdict`, ids and counts stay;
- every frozen plan's `prd_text` becomes the marker and `proposals` (the agent's proposed criterion texts
  and reasons) becomes null; `title`, `steps`, `criteria` (the ids and texts the judge used), `verify`,
  `required_children` and `completion_policy` stay, so the run still says what it was judged against;
- the run's notification bodies and pending-control summaries become the marker (a pending control also
  loses `arguments.text`); the run's action receipts are deleted.

**Deleting one run now.** `POST /api/v1/remotea/runs/{id}/redact` (owner-only, CSRF, works while remotea is
off; `409 RUN_ACTIVE` until the run has settled; another workspace's run is `404`) deletes the run's turns,
questions, interviews, plan reviews and commands, scrubs the copies as above (history reason `owner`), and keeps the
attempts with their verdict, rules, exit code and each criterion's id and status while their `tail`,
`error`, `gap`, `summary`, `detail` and per-criterion `evidence` become the marker. It returns
`{redacted: {counts}, run: <detail>}`. Bot `agents:read` reads (`run_list`, `run_get`) and the page read
the same rows, so after either redaction they show the marker and never the text.

**Every place run content lives.** Each `agent_*` table and the other copies, with what is kept:

| Where | Content | Kept |
|---|---|---|
| `agent_turns` `text`, `prompt` | the agent's final responses and the typed prompts (bounded bytes) | a run's (and the unattributed turns of the session it owns): deleted at the window; a session no run owns: 30 days |
| `agent_commands` `payload`, `outcome` | spawn prompt and run token (`[redacted]` once acked), input `text`, outcome details | payload emptied 7 days after the command settles; the row is deleted at the window |
| `agent_run_plans` `prd_text`, `proposals` | the frozen PRD, the agent's proposals | marker / null at the window; title, steps, criteria, verify, policy kept with the run |
| `agent_runs` `history[]` | `sent` (≤ 2,000 chars of each message to the agent), transition and outcome `detail` (≤ 300 chars) | bounded to 200 entries; `text`/`detail` become the marker at the window, the rest stays |
| `agent_runs` other columns | state, reason codes, branch, worktree path, base facts, Linear outcome codes (`sync_state`) | kept with the run (no free agent text) |
| `agent_attempts` `result` | verify tail (≤ 8,000) and error, the judge's gap, summary and per-criterion evidence | deleted at the window; the redact route keeps the row and scrubs the text |
| `agent_questions` `items`, `answers`, `detail` | the agent's questions, choices and the answers | a run's: deleted at the window; a session no run owns: 30 days once no longer open |
| `agent_interviews` `questions`, `answers` | the interview form and answers | deleted at the window |
| `agent_plan_reviews` `findings`, `questions`, `summary`, `resolutions` | hyperplan review rounds: the reviewer's findings and summary (or its turn text), the planner's answers; `token_hash`/`token_salt` of the reviewer token | deleted at the window and by the redact route; exported without the token columns |
| `agent_pending_controls` `summary`, `arguments` | Eri's proposed start or message | marker at the window; the row is deleted 30 days after it expires (10-minute expiry) |
| `agent_action_receipts` `result` | a whole run response (history included) | 7 days; deleted at the window and by the redact route |
| `notifications` `body` (`target.run_id`) | task title plus a question's text or a reason | marker at the window; the row follows the notification lifecycle |
| `agent_sessions` title, prompts, excerpt, `summary` | the collector's session ledger (redacted on ingestion) | kept: it is the collector's record of a session on your machine, re-reported by the collector; not run content this module owns |
| `agent_session_rules`, `agent_hosts`, `execution_targets`, `autolink_rejections`, `migration_counters`, `migration_reservations` | configuration: folder rules, host caps and worktree reports, repo paths and verify commands, numbers | kept until changed or removed; no run text |
| `agent_swarms` `tasks`, `planner`, `batches` | a swarm plan: task titles, the planner's one-line reasons and predicted areas, run ids, start refusals | kept, finished and canceled plans too (no agent output: titles and the planner's text only); exported |
| `agent_parity`, `agent_parity_days`, `agent_ingest_days` | collector-vs-link diffs, daily counts | 30 days |
| Bot `agents:read` output | `public_run` of the rows above | live view: the marker after redaction, nothing after deletion |
| Action journal (`action_changes`) | an encrypted `control.*` entry: title (proposal summary or task title) and status | follows the action journal's retention |
| Linear comment (`agent_sync.comment_body`) | criteria ids and status, verify id and status, PR links: no free text | lives in Linear |

**Account export.** `GET /api/v1/export` includes `agent_runs` (without `run_token_hash` / `run_token_salt`),
`agent_run_plans`, `agent_attempts`, `agent_questions`, `agent_interviews`, `agent_plan_reviews` (without
`token_hash` / `token_salt`), `agent_sessions` (without the
embedding and fingerprints) and `agent_swarms` as they are now: redacted rows export the marker, deleted rows are absent
(`agent_retention.export_tables`). It leaves out `agent_turns` (up to 64 MiB a day per owner, and the export
is built in memory; the host keeps the transcript and `GET /remotea/runs/{id}` pages them), `agent_commands`
(the host-facing journal: its input texts are the `sent` history entries, and a spawn payload carried a run
token until acked) and the configuration and counter tables in the last rows above.

**Quotas.** Each owner may ingest 64 MiB of turn text and prompts per UTC day (`agent_ingest_days`). Past it
the turn is kept with its text dropped and `truncated: true`, and hosts report `status: "degraded"` until the
next day. `GET /remotea/runs/{id}` pages turns and commands: the newest `limit` (default 100, ≤ 200),
`offset` for older, `more.turns` / `more.commands` when there are more.

Other retention: the payload of a settled command, 7 days; run action receipts, 7 days; `agent_parity`,
`agent_parity_days`, `agent_ingest_days` and expired pending controls, 30 days.

### Notifications and transport

An agent's question is `urgent` (it pushes through quiet hours) only when the owner turns on
"Agent questions break quiet hours" (`agent_quiet_bypass`, `POST /remotea/quiet-bypass`). A run sends at
most 3 pushes per 10 minutes; the remotea page still shows everything. Responses carry
`Strict-Transport-Security: max-age=31536000; includeSubDomains`. `POST /api/v1/chat` is gone; chat goes
through `/api/v1/work`. CI runs on every pull request, whatever its base.

### Not yet

Verify commands run without a sandbox on a host where bubblewrap is missing or broken (remotea PR #24 sandboxes
them elsewhere); each result says so (`isolated: false`, `isolation_reason`) and the drawer shows it.
