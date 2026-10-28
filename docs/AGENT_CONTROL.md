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
(`agent_link.digest`). Stage B accepts inbound digests without enforcing them; stage C stamps commands.
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
| up | `hello` | `server_id, boot_id, hostname, machine, proto, version?, caps[], harnesses{}` | must be first, within 15 s. A `boot_id` starting `probe-` (`remotea eridani connect`) only checks the key: the welcome carries `probe: true` and the current epoch, no lease is taken, and the probe closes after its `bye` |
| down | `welcome` | `acked_id, host_id, conn_epoch, server_time, replay_horizon (s), lease_seconds, proto, caps[], max_frame` | |
| up | `session.report` | `tool, session_id, report: SessionInput, surf?, incarnation?, parity?` | same upsert as `PUT /external/sessions/{tool}/{id}`, attached to the host; outcome `{session}` identical to the HTTP response. With `parity: true`, compares instead (below) |
| up | `turn` | `tool, surf, incarnation, run_id?, generation, provider_session_id, provider_turn_id, root, seq, text, truncated, missing?, prompt?, source: hook\|rollout, ended_at?` | stored once per `(session, provider_turn_id)`: a resend with a new transport id → `{duplicate:true}`; `root:false` → `{ignored:"subagent"}`; text capped at 64 KiB (UTF-8) with `truncated` set; any `run_id` → `RUN_NOT_BOUND` until runs exist (stage C) |
| up | `status` | `surf, incarnation, state, since, reason?, effective_permission_mode?, provider?+provider_session_id? (or tool?+session_id?)` | sets the session's `state`/`state_since`. Older incarnation, or same incarnation with an earlier `since`, is ignored (`applied:false`). Without a provider (or tool) and session id, the surface maps to the newest session reported with that `surf` on this host (`UNKNOWN_SURFACE` otherwise) |
| up | `exit` | `surf, incarnation, exited_at?, code?, provider?+provider_session_id?` | state `exited` at `exited_at` |
| up | `sessions.snapshot` | `surfaces: [{surf, incarnation, state: live\|lost\|exited, agent_state, since, provider, provider_session_id, …}], part?: {of, i, n}` | a large snapshot arrives as several frames sharing `part.of`; non-final parts are acked `{pending:true}` and reconciliation runs once all `n` parts arrived. `live` takes the surface's `agent_state` (working/needs_you/idle). Applies each surface as `status`; when `complete`, sessions of this host missing from the snapshot become `exited` (`reason: absent_from_snapshot`) unless their state is newer than `at`. Eridani first replies `runs.snapshot {runs: [], questions: []}` (empty until stage C) |
| up | `chunk` | `part: {of, i, n}, data` | parts of one logical frame (n ≤ 64, ≤ 1 MiB total, ≤ 4 in flight); the ack names the logical frame's id. Eridani chunks its own frames above 256 KiB the same way |
| up | `ack` | `acked_id, op_id?, ok, outcome?` | acknowledges a command (stage C) |
| up | `command.outcome` | `command_id, state: executed\|failed\|rejected\|uncertain, detail` | recorded on the command (in B the host rejects every command as `unsupported`) |
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
transaction; each tick, `drain()` expires overdue commands and sends queued ones whose kind the host
advertised (`COMMANDS` maps kind → required capability; empty in stage B). Commands carry a stable
`op_id` and payload `digest`, separate from the envelope id.

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
queue a Linear write. Runs (stage C) will freeze a plan revision; the per-step agent tool and completion
gate come with them.
