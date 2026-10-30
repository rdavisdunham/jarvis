# Agent control: the remotea link

The remotea server (Davin's always-on terminal host in WSL) opens one outbound WebSocket to Eridani and
reports the Claude Code and Codex sessions it runs. Eridani shows them on the **remotea** page. This is
**stage B** of the mission-control plan: the link is read-only. Starting, steering, answering and stopping
agents (stage C) are not implemented yet; the command journal and drain loop exist so they can be added
without another protocol change.

Code: [`apps/api/jarvis/agent_link.py`](../apps/api/jarvis/agent_link.py) (socket, ingestion, parity),
[`remotea_routes.py`](../apps/api/jarvis/remotea_routes.py) (browser routes),
[`task_plan.py`](../apps/api/jarvis/task_plan.py) (task plan fields), web
[`Remotea.tsx`](../apps/web/src/Remotea.tsx) and [`remotea-model.ts`](../apps/web/src/remotea-model.ts).

## Turning it on

remotea is **off by default** (preference `remotea_enabled`). While it is off, Eridani behaves exactly as
before: no nav entry, no page, the link refuses connections, and every `/api/v1/remotea/*` route except
`status`, `enable` and `disable` answers `403 REMOTEA_DISABLED`.

**Settings → Connections → remotea → Show remotea** calls `POST /api/v1/remotea/enable`, which:

1. sets `remotea_enabled`;
2. mints a bot key named `remotea` (365 days) with scopes
   `agents:control agents:session sessions:read sessions:write tasks:read records:read`;
3. returns the token **once**, plus the connect line:
   `remotea eridani connect --url wss://app.eridani.app/api/v1/external/agents/link`.

Enabling again reuses the active key (no token in the response). **New link key** (`{"rotate": true}`)
mints a replacement and revokes the old one. Turning remotea off keeps the key; connected hosts are told
`bye {reason: "disabled"}` on their next message or within 30 seconds. Viewers cannot enable, disable or
hold the key.

`GET /api/v1/remotea/status` → `{enabled, key, hosts[], link_url, connect_command}`; a host is
`connected` while its lease is current, otherwise the page shows when it was last seen (or "never").

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
`tasks:write`. The coding agents' own MCP key is separate: create it in Settings → Connections →
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
| up | `hello` | `server_id, boot_id, hostname, machine, proto, version?, caps[], harnesses{}, targets?: [{id, repo, worktree_root, harnesses, verify}]` | must be first, within 15 s. Each target's `harnesses` lists the harnesses it may run (remotea's `launch` for that target) and `verify` its verify ids; both are kept in `caps.targets` and returned on advertised targets (see **Target capabilities**). A `boot_id` starting `probe-` (`remotea eridani connect`) only checks the key: the welcome carries `probe: true` and the current epoch, no lease is taken, and the probe closes after its `bye` |
| down | `welcome` | `acked_id, host_id, conn_epoch, server_time, replay_horizon (s), lease_seconds, proto, caps[], max_frame` | |
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
| down | `runs.snapshot` | `runs: [{run_id, generation, desired, stop_mode?}]` | after each reconnect's reconciliation |
| down | `spawn` / `input` / `answer` / `stop` / `verify` | see **Commands** | outbox commands |
| down | `worktrees.scan` | `branch, turn_seq, tool?, session_id?` (outbox command, only to a host whose `hello.caps` has `worktrees.scan`) | the judge needs a PR record newer than the turn: the host rescans its worktrees and sends `worktrees.report` even if unchanged, re-reads the named session and sends its `session.report` even if its fingerprint is unchanged, then acks (`executed`). An older host never gets it; the judge waits for its own tick instead |
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
| `GET /status` | no | enabled, key, hosts, connect line |
| `POST /enable` `{rotate?}` | no | enable; token in this response only |
| `POST /disable` | no | |
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
  `delivery_uncertain`, `spawn_expired` (15 minutes), `wall_clock` (6 h), `no_progress` (45 min),
  `turn_cap` (C pauses instead of `capped`), `question_timed_out`. Pushes: `agent_stuck` on needs_you,
  `agent_question` on a question, `agent_done` on awaiting_approval; each opens
  `/?view=remotea&run=<id>`.
- **Approve** completes the task as a human (`task.complete`); the run moves to `stopping` with
  `stop_final: completed` and becomes `completed` once the agent is gone.

### Starting a run

`POST /remotea/runs {task_id, harness?, model?, execution_target_id?}`:

1. Target resolution: explicit → the task's `execution_target_id` → a target saved for the task's
   record or an ancestor (nearest first) → folder rules whose path is under a target's repo → `409
   NEEDS_TARGET {task_id, candidates, advertised, records: [{id, title, type_id}]}`. The page's picker
   then saves one with `POST /remotea/targets {host_id, host_target, repo, worktree_root, task_id | record_id}`
   and retries. "Remember for" defaults to the nearest record that is not a task (usually the project), so
   sibling tasks reuse the choice; "Only this task" binds just the task (reusing a target already saved
   for a record rather than copying it).
2. Refusals: `RUN_ACTIVE`, `HOST_OFFLINE`, `HARNESS_UNAVAILABLE` (the host lacks `spawn` for the harness, or the
   resolved target does not enable it; the message names the target and `data` is `{harness, host_target}`),
   `HARNESS_DISABLED` (opencode).
3. The run is created in `spawning`, the plan is frozen as revision 1 (PRD note or task notes, steps,
   criteria), a run token is minted and `spawn` is queued in the same transaction.

Branch: Linear's `branchName`, else `<owner>/<identifier>-<slug>`, else `task/<id>`; base: the parent
task's run branch, else `main`.

**Target capabilities.** `agent_control.host_harness(host, harness, host_target)` decides whether a harness can
start: the host must list `spawn` in `caps` and set `caps.harnesses.<harness>.spawn`, and when its hello entry for
`host_target` carries `harnesses`, that list must include the harness (remotea refuses a harness missing from the
target's `launch`). A target without `harnesses` (an older remotea) falls back to the host-wide flags alone.
`check_host` passes the resolved target's `host_target`, so `start_run`, Eri's `control_start` proposal and
confirm all refuse the same way. Advertised targets (`GET /remotea/targets` `advertised`, `NEEDS_TARGET`
`data.advertised`) carry `harnesses` (null when the host did not say) and `verify`; a saved target carries the
same two fields when its host advertises it. The Run dialog's preflight reports `harnesses` for the resolved
target only, with `harness_reasons` (why each one cannot start, naming the target); while no target resolves, it
reports the union over the picker's candidates and advertised targets and sets `harness_note`. Another host's
capabilities never make the resolved target look ready. The dialog disables a harness the preflight rules out and
shows the reason; only without a preflight does it fall back to "any connected host can spawn it".

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

Commands are rows in `agent_commands` (status `queued`). Whichever process queues one, the host's
socket drains it within about a second of the commit (see "Wake" under Connection lifecycle), or at
the next 30-second tick if no wake arrives.

**Deterministic spawn surface.** remotea names a spawned surface
`surf-eri-<sha256(run_id + "\0" + generation + "\0" + attempt) hex, first 32>` (`run_coordinator.surf_for`).
When a spawn's ack or outcome is lost, reconciliation recognizes the run's surface in `sessions.snapshot`
by that id (or by `run_id`) and records it as accepted instead of spawning again.

Outcomes: an accepted spawn ack or `executed` outcome records `surf`, `incarnation`, `worktree`,
`branch`, `base`, `base_sha`; a failed/rejected spawn fails the run; `uncertain` input moves to
needs_you `delivery_uncertain` (a `worktrees.scan` outcome never moves the run); a failed answer reopens the question; an executed stop settles a stopping
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
completion and edits of `success_criteria`, `verify` and `turn_cap` from `eri_work`, `run_session` and
`bot` with `409 RUN_OWNS_COMPLETION {run_id, state}`. A human or source_sync completion is recorded at
once and asks the run to stop (`stop_reason: task_completed_externally`); a human approve ends it
`completed`.

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
  record, rules, model, cost). Turns and commands page newest first; `more` says whether older ones remain.

Reads stay available after remotea is turned off, as on the page. Owner scoping is the run's `owner_id`;
another workspace's run is `404`. Nothing here mutates: approving, messaging and stopping remain
cookie-authenticated browser routes.

### In-session tools (`agents:session`)

`task_plan` and `task_step` (MCP tools, and `POST /api/v1/external/agents/session/plan|step`) take
`run_token` (from `ERIDANI_RUN_TOKEN`, or the plan text) or `tool + session_id`. A valid token binds a
verified `run_session` actor; a bad one is `401 RUN_TOKEN_INVALID`; a session fallback is unverified.
`task_step {step_id, status, note?, expected_revision?}` updates one step with its own revision
(`by: agent:<run>[:unverified]`); it does not advance `plan_revision`. `task_interview` and
`submit_plan` (stage D, below) need the run token. `session_link` accepts `task_id`. The run token is
HMAC-derived from the run id (`ert_…`), stored as a hash, valid 24 h and refreshed by each
`context.request`.

### Browser routes (runs)

| Route | Gated | |
|---|---|---|
| `GET /runs?task_id=&active=&limit=` | no | runs, newest first |
| `POST /runs` | yes | start (above) |
| `GET /runs/preflight?task_id=&harness=&execution_target_id=` | yes | the Run dialog's read-only look at a start: `target` or `needs_target`+`need`, `problems`, `active_run`, `interview`, `harnesses` (what the task's target can start), `harness_reasons`, `harness_note`, `task` |
| `GET /runs/{id}` | no | detail: plan, history, turns, commands with outcomes, questions |
| `POST /runs/{id}/message {text, mode, expected_revision?, action_id?}` | yes | queue or steer |
| `POST /runs/{id}/stop {mode, expected_revision?, action_id?}` | no (viewers refused) | stop stays available after remotea is turned off; the revision stays optional (emergency path) and the drawer never sends one |
| `POST /runs/{id}/approve {expected_revision, action_id?}` | yes | the revision is **required** (422 without it, `REVISION_CONFLICT` when stale): approving completes the task, so it must be the state the approver reviewed |
| `POST /questions/{id}/answer` | yes | |
| `GET`/`POST /targets` | yes | saved and advertised targets, with per-target `harnesses` and `verify` when the host sent them |
| `GET /unmerged` | yes | the hosts' last `worktrees.report` |

**Retrying a run action (`action_id`, migration 0035).** `message`, `stop`, `approve`, `replan`, `freestyle`
and `verify-config` accept an optional client `action_id` (a UUID). The first request acts and stores a receipt
(`agent_action_receipts`: owner, action id, run, route, a SHA-256 of the route, run and body without
`action_id`, and the response) in the same transaction as the effect (`agent_receipts.once`). A repeat with the
same `action_id` and body returns the stored response and acts no more: one `AgentCommand`, one steer
generation, one stop. The same `action_id` with a different body, route or run is refused with 409
`ACTION_MISMATCH`. A refused action (409 `NOT_STARTED`, `REVISION_CONFLICT`, …) rolls back and stores nothing,
so the same id may act later. Without `action_id` every request acts. Receipts are kept 7 days. The web helper
`postAction(path, body)` in `apps/web/src/api.ts` sets one `action_id` per call and resends the same body after
a lost response (`NETWORK`, up to two retries); other errors are not retried.

The page shows runs (needs-you first), **Unmerged work**, and a run drawer with the timeline, the
questions card and Message / Steer / Stop (**Stopping…**, then **Force stop**) / Approve.

The drawer (and the side panel, which keys it by run id so draft text, busy state and scroll are per run)
binds every control to the run it has loaded, never to the id it was asked for: a run load is latest-only
(`remotea-model.ts` `latestOnly`/`latestRun`: switching runs aborts the earlier request and a late reply,
or one for another run, is dropped), controls stay disabled until the loaded run is the requested one, and
each URL and body comes from `runAction`. Message, Steer, Re-plan, Verify now, the verify command,
Freestyle, Approve and the plan review send `expected_revision` = the loaded run's `state_revision`; Stop
does not. A `REVISION_CONFLICT` shows "This run changed; reloaded" and reloads the run instead of an error. Task details
show **Run** only while remotea is on, disabled with a reason when no host is connected, and **Open run**
while one is active.

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
  then stop") and moves the run to `planning`.
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
  plan goes to `needs_you(no_plan)`. Interview and planning turns never count toward the cap.
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
- **Host-initiated** results (`attempt_id: "local:…"`, remotea's `verify.auto`) are adopted when they
  match the tuple of a running Eridani attempt; others are stored with `initiator: host` and never move
  the run. `adopted` on a result names the host attempt remotea reused.
- One deadline: 11 minutes (the host's 10 plus a minute of grace) → `timed_out`, a non-pass; a result after
  that is discarded and noted on the attempt as `late`. A refused/failed/uncertain `verify` command is
  `refused`/`uncertain`, also a non-pass (never `delivery_uncertain`). A host without `verify` refuses at
  once. Every non-pass still goes to the judge.
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
report and the host's worktrees report arrived after the turn). The prompt (`judge-v2`) says all of it is
untrusted data and that **the agent's text is a claim**: evidence is the verify attempt, the host's worktree and
PR record and the ticked steps. A gap may only ask for what the loop can check (make the verify command pass,
commit/push and open or update the PR, tick steps), never for pasted test output, logs or diffs. The gap prompt
(`agent_prompts.gap`) repeats that to the agent.

1. **Hard rules first.** No criteria → `needs_you` and the model is not asked. Plans without criteria do
   not enter the loop at all: they keep stage C's manual flow (turn cap → needs_you, Davin approves).
2. **Luna** via `memory_learning.extraction_request` (the only model it accepts; decision 20) with a
   strict schema `{verdict: met|not_met|needs_you, gap, summary, criteria: [{id, status: met|not_met|unclear,
   evidence}], steps: [{id, status: done|not_done|disputed|unclear}]}`.
3. **Rules again on the answer.** `met` with a required verify that is not fresh and passing: `not_met`
   with the failure tail when it ran and failed, else `needs_you` (missing, refused, timed out, stale,
   uncertain: the agent cannot fix that). A `verify`-kind criterion is met only with a fresh passing
   attempt of its verify id. `met` with any criterion not met → `not_met`. Unknown criterion ids are
   dropped; a missing one is `not_met`. `disputed` steps count only when the verification failed. Each
   rule that fired is recorded in `rules`.
4. **Non-verdicts** (refusal, incomplete, timeout, budget, invalid output, unavailable) are retried twice
   (refusal and budget are not retried), then the attempt is `failed` and the run goes to
   `needs_you(judge_unavailable)`. A judge attempt still open after 15 minutes is a `timeout` non-verdict.
5. The attempt keeps model, `prompt_version` (`judge-v2`), cost, latency, retries, the final verdict and the
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
stored with `by: default`, which is not a human floor and may be replaced). When the target offers several ids
the pick is a plan-review proposal (`proposals.verify` with `current: null` and `options`); declining it leaves the
plan without verification. The planning prompt lists the offered ids and says a test criterion must name one.

`POST /remotea/runs/{id}/verify-config {verify_id, expected_revision?}` (`agent_plan.set_verify`): the operator
adds or swaps the verify command mid-run. The task's verify is set as a human edit, the plan is frozen again as
revision N+1 with nothing else changed (history `verify_set`), and a run parked in `needs_you` goes back to
`executing(verify_set)` with a short prompt so its next turn is verified. `GET /runs/{id}` lists `verify_options`.

### Freestyle

`POST /remotea/runs/{id}/freestyle {enabled, expected_revision?}` (`agent_runs.freestyle`, migration 0034): the
operator talks to the agent directly. While on, a root turn is recorded (history `turn` with `freestyle: true`)
but not verified, judged or counted toward the turn cap, no gap prompt is sent, and the 45-minute no-progress
timer is skipped (the wall clock still applies). Turning it on from `needs_you(judge_needs_you | no_progress_gap |
verify_missing | no_progress | turn_cap | awaiting_merge | pr_unknown)` resumes `executing(freestyle)`; turning it off makes
the next turn judged again. Approve and Stop work as usual. The run header shows the toggle.

### auto_complete

`POST /remotea/auto-complete {enabled}` (owner preference `remotea_auto_complete`, default **off**;
viewers refused); `GET /remotea/status` returns it. Settings → Connections → remotea shows the toggle with a
warning that tasks then close (in Linear too) without a tap. Keep it off until the outcome eval below is clean on
a live run.

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
| `control.start` | `task_id` or `task` (name), `harness?` | starts a run as the human (`start_run`) |
| `control.message` | target, `text`, `mode: queue\|steer` | |
| `control.stop` | target, `mode: interrupt\|kill` | |
| `control.answer` | target, `question_id` + `answers`, or `interview_id` + `responses` | a run question or the planning interview |
| `control.confirm` | `pending_id?`, `decision: confirm\|decline`, `execution_target_id?` | the owner's yes to a proposed start/message (below); a start that asked where to run takes the chosen target |
| `control.approve` | target? | immediate: approves a run that is `awaiting_approval` (`agent_control.approve`; the server supplies `expected_revision` from the run it checked); any other state is refused with `NOT_AWAITING_APPROVAL` naming the state |
| `control.plan_review` | target?, `decisions: [{id, accept, text?}]`, `verify?`, `completion_policy?` | immediate: the owner's decisions on a `plan_review` (`agent_plan.review_plan`; the server supplies `expected_revision`); otherwise `NO_PLAN_REVIEW` |
| `control_status` (read) | target? | active runs, their open questions/interview, `plan_review` proposals, the judge's `awaiting_approval` summary, open proposals (`pending`, with target options), finished in the last 24 h, hosts |

A target is `run_id`, `task_id` or `task` (a spoken name); none means the run open on the owner's screen
(`UIContext.open_run_id`, which the remotea page sends and `context_prompt` names as "Open run: <id>"), else
the only active run. Names resolve through an exact Linear identifier, then `navigation_find`; several
matches → `AMBIGUOUS_TASK` (or `AMBIGUOUS_RUN`) with the candidates so Eri asks, none → `NOT_FOUND`. Every
command passes the remotea gate (`REMOTEA_DISABLED` when off). The `control_*` tools and the `agents` catalog
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
- A host that advertises the `digest` cap must stamp every frame carrying an `op_id`.
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
`NEEDS_TARGET`, `HOST_OFFLINE` or `HARNESS_UNAVAILABLE` leaves the row pending. Listing the pending items
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
24 h; `context.request` does not slide it. The spawn command's copy becomes `[redacted]` once the spawn
is acked or attached. Rotation is per spawn, not per generation: remotea passes the token as
`ERIDANI_RUN_TOKEN` at spawn, and a steer delivers no new SessionStart, so a rotated token could not reach
the agent.

### Redaction

`redaction.py` holds the 12 patterns of remotea's `docs/remotea/redaction.md` (the collector loads the
same file). Host free text is redacted on ingestion: turn text and prompt, status and exit reasons,
question text and choices, `question.resolved` details, verify tails and errors, outcome details, and
the session ledger's title, prompts and excerpt.

### Retention and quotas (`agent_retention.py`)

The worker's housekeeping pass (every 12th supervisor cycle) deletes:

| Data | Kept for |
|---|---|
| Turns of sessions no run owns | 30 days |
| A settled run's turns, questions, attempts, interviews and commands (the run, plans and history stay) | 90 days after it finished |
| The payload of a settled command | 7 days |
| `agent_parity`, `agent_parity_days`, `agent_ingest_days`, expired pending controls | 30 days |
| Run action receipts (`agent_action_receipts`) | 7 days |

Each owner may ingest 64 MiB of turn text and prompts per UTC day (`agent_ingest_days`). Past it the turn
is kept with its text dropped and `truncated: true`, and hosts report `status: "degraded"` until the next
day. `GET /remotea/runs/{id}` pages turns and commands: the newest `limit` (default 100, ≤ 200), `offset`
for older, `more.turns` / `more.commands` when there are more.

### Notifications and transport

An agent's question is `urgent` (it pushes through quiet hours) only when the owner turns on
"Agent questions break quiet hours" (`agent_quiet_bypass`, `POST /remotea/quiet-bypass`). A run sends at
most 3 pushes per 10 minutes; the remotea page still shows everything. Responses carry
`Strict-Transport-Security: max-age=31536000; includeSubDomains`. `POST /api/v1/chat` is gone; chat goes
through `/api/v1/work`. CI runs on every pull request, whatever its base.

### Not yet

Verify commands run on the host without a sandbox (bubblewrap / codex sandbox, no network) — roadmap.
