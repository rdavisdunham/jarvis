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

Privileged scopes (`bot_access.PRIVILEGED`) are rejected for viewers at key creation and every use is a
write check: `authorize(required="agents:control")` locks membership and fails with `READ_ONLY` if the
account was demoted. The link re-checks on every inbound message and every 30-second tick, so revoking
the key, removing membership or turning remotea off ends the connection. The link key never has
`tasks:write`; the separate in-session MCP key (`cli-agents`) keeps its existing scopes.

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
| up | `hello` | `server_id, boot_id, hostname, machine, proto, version?, caps[], harnesses{}, targets?: [{id, repo, worktree_root, harnesses, verify}]` | must be first, within 15 s. A `boot_id` starting `probe-` (`remotea eridani connect`) only checks the key: the welcome carries `probe: true` and the current epoch, no lease is taken, and the probe closes after its `bye` |
| down | `welcome` | `acked_id, host_id, conn_epoch, server_time, replay_horizon (s), lease_seconds, proto, caps[], max_frame` | |
| up | `session.report` | `tool, session_id, report: SessionInput, surf?, incarnation?, parity?` | same upsert as `PUT /external/sessions/{tool}/{id}`, attached to the host; outcome `{session}` identical to the HTTP response. With `parity: true`, compares instead (below) |
| up | `turn` | `tool, surf, incarnation, run_id?, generation, provider_session_id, provider_turn_id, root, seq, text, truncated, missing?, prompt?, source: hook\|rollout, ended_at?` | stored once per `(session, provider_turn_id)`: a resend with a new transport id → `{duplicate:true}`; `root:false` → `{ignored:"subagent"}`; text capped at 64 KiB (UTF-8) with `truncated` set; a `run_id` must name a run bound to this host (`RUN_NOT_BOUND` otherwise); a turn of a spawned run attaches the session (below) and counts toward the turn cap |
| up | `status` | `surf, incarnation, state, since, reason?, effective_permission_mode?, provider?+provider_session_id? (or tool?+session_id?)` | sets the session's `state`/`state_since`. Older incarnation, or same incarnation with an earlier `since`, is ignored (`applied:false`). Without a provider (or tool) and session id, the surface maps to the newest session reported with that `surf` on this host (`UNKNOWN_SURFACE` otherwise) |
| up | `exit` | `surf, incarnation, exited_at?, code?, provider?+provider_session_id?` | state `exited` at `exited_at` |
| up | `sessions.snapshot` | `surfaces: [{surf, incarnation, state: live\|lost\|exited, agent_state, since, provider, provider_session_id, …}], part?: {of, i, n}` | a large snapshot arrives as several frames sharing `part.of`; non-final parts are acked `{pending:true}` and reconciliation runs once all `n` parts arrived. `live` takes the surface's `agent_state` (working/needs_you/idle). Applies each surface as `status`; when `complete`, sessions of this host missing from the snapshot become `exited` (`reason: absent_from_snapshot`) unless their state is newer than `at`. Eridani first reconciles runs (below), then replies `runs.snapshot {runs: [{run_id, generation, desired: continue\|stop\|none, stop_mode?}]}` and only then resends unacked commands |
| up | `chunk` | `part: {of, i, n}, data` | parts of one logical frame (n ≤ 64, ≤ 1 MiB total, ≤ 4 in flight); the ack names the logical frame's id. Eridani chunks its own frames above 256 KiB the same way |
| up | `ack` | `acked_id, op_id?, ok, error?, retry?, outcome?` | acknowledges a command; an accepted spawn carries `outcome {state:"accepted", surf, incarnation}` |
| up | `command.outcome` | `command_id (= op_id), kind?, state: executed\|failed\|rejected\|uncertain, detail, result?` | recorded on the command and applied to its run (below) |
| up | `question` | `op_id, digest, surf, incarnation, run_id?, generation, provider, provider_session_id, provider_request_id, provider_turn_id?, items[], hold_until?` | a structured question the agent asked; stored once per `(session, provider_request_id)` |
| up | `question.resolved` | `op_id, digest, question_id\|provider_request_id, status: accepted\|answered_locally\|timed_out\|canceled\|failed, still_open?, remote_answerable?` | the host's word on a question; only `accepted` confirms an answer sent from here |
| up | `context.request` | `req_id, text, run_token?` | ephemeral, no ack: answered with `context.response {req_id, text}` |
| up | `worktrees.report` | `op_id, digest, worktrees: [{target, path, branch, base, ahead, behind, unpushed, uncommitted, untracked, merged, active, reasons}]` | replaces the host's last report; feeds **Unmerged work** |
| down | `runs.snapshot` | `runs: [{run_id, generation, desired, stop_mode?}]` | after each reconnect's reconciliation |
| down | `spawn` / `input` / `answer` / `stop` | see **Commands** | outbox commands |
| down | `context.response` | `req_id, text` | |
| up | `bye` | `reason: shutdown\|reconfigured\|probe` | Eridani closes 1000 |
| down | `bye` | `reason: superseded\|access_revoked\|disabled\|rate_limited\|restart` | followed by close |

`state` values: `working`, `needs_you`, `idle` (shown as **Live**), `restoring`, `lost` (shown as
**Restoring**), `exited` (**Ended**). `since` must carry a timezone. Every stored change emits
`session.changed` (report, status, exit, snapshot) or `agent.changed` (turn stored, host connected or
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
`{matched, missing, fields}`.

`GET /api/v1/remotea/parity` (and the page's **Collector parity** panel) summarizes `total, matched,
mismatched, missing, clean`, a count per differing field, first/last report times and the 20 most recent
problems with stored vs reported values (each clipped to 300 characters).

Procedure (plan decision 23): run both collectors; the Python one stays authoritative and keeps its timer.
Investigate every differing field (timestamps, Windows/WSL path coverage, redaction, fingerprint and
checkpoint semantics). Cut over only after the summary has been **clean for 7 days** and the host release
is installed: then disable the Python timer, and remove its code and revoke its credential one release
later. Do not delete the Python collector before that.

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
and the terminal states; `interviewing/planning/verifying/judging/awaiting_integration/replanning` are in
the map for stage D. Rules:

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
   NEEDS_TARGET {task_id, candidates, advertised, records}`. The page then saves one with
   `POST /remotea/targets {host_id, host_target, repo, worktree_root, task_id | record_id}` and retries.
2. Refusals: `RUN_ACTIVE`, `HOST_OFFLINE`, `HARNESS_UNAVAILABLE` (host lacks `spawn` for the harness),
   `HARNESS_DISABLED` (opencode).
3. The run is created in `spawning`, the plan is frozen as revision 1 (PRD note or task notes, steps,
   criteria), a run token is minted and `spawn` is queued in the same transaction.

Branch: Linear's `branchName`, else `<owner>/<identifier>-<slug>`, else `task/<id>`; base: the parent
task's run branch, else `main`.

### Commands

All commands: `{t, command_id (= op_id), op_id, digest, run_id, generation, expires_at (epoch ms), …}`.

| `t` | Body | TTL |
|---|---|---|
| `spawn` | `attempt, harness, model?, target: {id, repo?, branch, base}, task_id, title, prompt, run_token, env {ERIDANI_*}, group` | 15 min |
| `input` | `surf, incarnation, text, mode: queue\|steer` | 10 min |
| `answer` | `surf, incarnation, question_id, provider_request_id, answers: [{item_id, choice_ids?, text?}]` | 10 min |
| `stop` | `surf, incarnation, mode: interrupt\|kill` | 24 h |

**Deterministic spawn surface.** remotea names a spawned surface
`surf-eri-<sha256(run_id + "\0" + generation + "\0" + attempt) hex, first 32>` (`run_coordinator.surf_for`).
When a spawn's ack or outcome is lost, reconciliation recognizes the run's surface in `sessions.snapshot`
by that id (or by `run_id`) and records it as accepted instead of spawning again.

Outcomes: an accepted spawn ack or `executed` outcome records `surf`, `incarnation`, `worktree`,
`branch`, `base`, `base_sha`; a failed/rejected spawn fails the run; `uncertain` input moves to
needs_you `delivery_uncertain`; a failed answer reopens the question; an executed stop settles a stopping
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

### In-session tools (`agents:session`)

`task_plan` and `task_step` (MCP tools, and `POST /api/v1/external/agents/session/plan|step`) take
`run_token` (from `ERIDANI_RUN_TOKEN`, or the plan text) or `tool + session_id`. A valid token binds a
verified `run_session` actor; a bad one is `401 RUN_TOKEN_INVALID`; a session fallback is unverified.
`task_step {step_id, status, note?, expected_revision?}` updates one step with its own revision
(`by: agent:<run>[:unverified]`); it does not advance `plan_revision`. `submit_plan` and
`task_interview` answer `501 NOT_YET` until stage D. `session_link` accepts `task_id`. The run token is
HMAC-derived from the run id (`ert_…`), stored as a hash, valid 24 h and refreshed by each
`context.request`.

### Browser routes (runs)

| Route | Gated | |
|---|---|---|
| `GET /runs?task_id=&active=&limit=` | no | runs, newest first |
| `POST /runs` | yes | start (above) |
| `GET /runs/{id}` | no | detail: plan, history, turns, commands with outcomes, questions |
| `POST /runs/{id}/message {text, mode}` | yes | queue or steer |
| `POST /runs/{id}/stop {mode}` | no (viewers refused) | stop stays available after remotea is turned off |
| `POST /runs/{id}/approve` | yes | |
| `POST /questions/{id}/answer` | yes | |
| `GET`/`POST /targets` | yes | saved and advertised targets |
| `GET /unmerged` | yes | the hosts' last `worktrees.report` |

The page shows runs (needs-you first), **Unmerged work**, and a run drawer with the timeline, the
questions card and Message / Steer / Stop (**Stopping…**, then **Force stop**) / Approve. Task details
show **Run** only while remotea is on, disabled with a reason when no host is connected, and **Open run**
while one is active.
