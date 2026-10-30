# remotea for Brainforge members: multi-user design

Status: **design, not built** (Eridani roadmap task `0ab99d68`, 2026-10-30). Nothing here changes
behaviour until a phase in [Rollout](#8-rollout) is built. Until then remotea stays personal-workspace
only ([Agent control § Personal workspace only](AGENT_CONTROL.md#personal-workspace-only)).

## Summary

1. **A host belongs to one person.** Hosts, link keys, execution targets and the coding agents' MCP key
   stay in the member's **personal** namespace. Shared namespaces never admit a host or hold an
   `agents:control` / `agents:session` key (today's rule in `bot_access` does not change).
2. **A run belongs to its task's namespace and to the member who started it.** A run on a shared-workspace
   task has `owner_id` = the workspace and `account_id` = the member. It can only use that member's hosts.
   Only that member drives it (message, steer, answer, approve, re-plan, verify, freestyle, plan review).
   The workspace owner may also stop it (interrupt only).
3. **Other members see a summary by default.** They see state, plan, verdict and who did what. Turn text,
   prompts, sent text, questions, verify output, worktree paths and host machine names stay owner-only
   unless the owner shares the run. One redaction function serves the page, Eri, bots and pushes.
4. **Every action records the account**, not just the actor kind. A separate append-only audit table keeps
   it beyond the 200-entry `history` cap.
5. **Quotas, budgets and confirmation are per member.** Two-phase confirmation stays with the run owner.
   Judge cost is billed to the member's personal budget. New caps limit active runs, starts per day and
   the turn cap.
6. **Workspace runs need isolated verify.** Automatic verify runs on a workspace run only when the host
   reports it ran isolated (S-N4); otherwise verify is on demand only.

The first deliverable (phase 1) needs **no sharing at all**: a second member runs remotea against tasks in
their own personal workspace on their own desktop. That needs only the single-user assumptions removed
and the isolation tests in § 9.

## 0. Where the code is today

| Concern | Today | Where |
|---|---|---|
| Path gate | `/api/v1/remotea` is in `PERSONAL_PATHS`; a shared namespace gets `PERSONAL_WORKSPACE` before any route runs | `access.py` `PERSONAL_PATHS`, `request_access` |
| Route backstop | `owner_only`: refuses a shared namespace or a role other than `owner`; `gated` = `owner_only` + `agent_link.gate` | `remotea_routes.py` |
| Eri control | `control.` in `PERSONAL_TOOLS`, `control_` refused by `tool_access`; `agent_control.owner_control` re-checks | `access.py`, `agent_control.py` |
| Privileged scopes | `PRIVILEGED = {agents:control, agents:session}`: viewers cannot mint them, every use is a write check, refused at mint **and** at use in a shared namespace | `bot_access.py` `create`, `authorize` |
| `agents:read` | not privileged; can be minted in a shared namespace today (there are simply no runs there) | `bot_access.py`, `external_routes.py` `agent_runs`, `external_mcp.py` `run_list/run_get` |
| Host identity | `agent_hosts` unique `(owner_id, server_id)`, also stores `account_id` and `credential_id`; at most 3 hosts per key (`HOSTS_PER_CREDENTIAL`); a host stays bound to its key or that key's rotation (same account, name, owner: `successor`) | `agent_link.py` `admit`, `successor`, `holding` |
| Link-side run lookups | `bound(db, host_id, run_id)` keys on host only; `reconcile`, `check_integrations` filter `AgentRun.owner_id == owner AND host_id == host_id`, where `owner` is the link key's namespace | `run_coordinator.py` |
| Runs | `agent_runs.owner_id` (namespace) and nullable `account_id`; `coordinator.load(db, owner, run_id)` scopes by namespace only | `session_models.py`, `run_coordinator.py` |
| Targets | `execution_targets` unique `(owner_id, record_id)`; `tasks.execution_target_id`; no account column | `session_models.py`, `agent_control.resolve_target/save_target` |
| Actor | `Actor(kind, credential_id, run_id, generation, verified, reason, capability)`: **no account**. Browser routes build `Actor("human")` (`remotea_routes.actor_for`) | `agent_actor.py` |
| History | `transition()` and `note()` append `{at, from, to, reason, actor}` / `{event, actor, text…}` to `agent_runs.history`, bounded to `HISTORY_LIMIT = 200` (oldest dropped) | `run_coordinator.py` |
| Answers | `agent_questions.answered_via` / `agent_interviews.answered_via` store `actor.kind`, not who | `run_coordinator.answer`, `agent_plan.answer_interview` |
| Receipts | `agent_action_receipts` keyed `(owner_id, action_id)`, no account | `agent_receipts.py`, `session_models.py` |
| Eri journal | `journal_control` writes `action_changes` with `account_id` | `agent_control.py` |
| Change feed | `domain.emit` writes `Event(owner_id, kind, entity_id, revision)`: ids only, no content; `/api/v1/events` streams the caller's namespace | `domain.py`, `api.py` |
| Ingest quota | 64 MiB/day per owner (`agent_retention.admit(db, owner)`); turns are stored under the **link key's** namespace | `agent_retention.py`, `agent_link.turn` |
| Judge cost | `agent_judge.ask(owner=attempt.owner_id)` → the run namespace's `monthly_budget_usd` | `agent_judge.py`, `budget.py` |
| Pushes | `Notification(owner_id=run.owner_id)`; quiet bypass from `run.owner_id` | `run_coordinator.notify` |
| Single-user leftovers | branch prefix from `get_settings().owner_id`; `base_for` takes the parent task's latest run branch from any run; prompt text "Davin approves completion" | `agent_control.branch_for/base_for`, `run_coordinator.py` |
| Verify on host | minimal env (`PATH, HOME, LANG, CI=1, TERM=dumb` + entry env), `realpath` containment, **no sandbox**, network on | remotea `remote-server/src/verify-runner.ts` `verifyEnv` |

Everything below is relative to this table.

## 1. Invariants

- **I1. Host ownership.** `agent_hosts.owner_id == agent_hosts.account_id` (a personal namespace is the
  account id: `access.role` returns `owner` only when `namespace == account`). A host is never admitted
  in a shared namespace.
- **I2. Run ↔ host.** `agent_runs.account_id` is NOT NULL and equals `agent_hosts.account_id` of
  `agent_runs.host_id`. `run_coordinator.command()` asserts it before writing an outbox row.
- **I3. Drive = run owner.** Every effectful run action checks `run.account_id == caller account` and that
  the caller is still an `owner`/`editor` of `run.owner_id` (locked role check, as `owner_only` does now).
  Exceptions: workspace owner `stop {mode: interrupt}`; the coordinator and judge (`run_judge`).
- **I4. Content = run owner.** Free text that came from the host (turns, prompts, questions, verify tails)
  or was typed to the agent is owner-only unless the run is shared (§ 3).
- **I5. Account on every actor.** Every `Actor` built from a request carries `account_id`. An actor
  without one is a coordinator/judge/host actor by construction.
- **I6. Spend and quotas follow the account**, never the workspace.
- **I7. Privileged keys stay personal.** `PRIVILEGED` is still refused at mint and at use in a shared
  namespace. Cross-namespace reach comes from the **run token**, not from the key.

## 2. Per-member hosts, link keys and execution targets

### 2.1 Ownership model

| Object | Scope | Table / column | Change |
|---|---|---|---|
| Link key (`agents:control`) | account, personal ns | `bot_credentials` | none (minted only by `POST /remotea/enable` in the member's personal ns) |
| MCP key (`agents:session` + task/note scopes) | account, personal ns | `bot_credentials` | none at mint; in-session tools act on the run's namespace via the run token (§ 5.2) |
| Host | account | `agent_hosts` | add CHECK `owner_id = account_id` after backfill (I1) |
| Execution target | account × namespace | `execution_targets` | add `account_id` NOT NULL (backfill from `agent_hosts.account_id`); unique index becomes `(owner_id, account_id, record_id)`; `owner_id` stays the namespace of the record/task the choice is remembered for |
| Task's direct target | namespace | `tasks.execution_target_id` | honoured only when `execution_targets.account_id == caller`; otherwise resolution continues with the caller's own saved targets, then `409 NEEDS_TARGET` |
| Run | namespace × account | `agent_runs` | `account_id` NOT NULL (backfill = `owner_id` for personal runs); add `visibility` (`owner` default, `workspace`) |
| Session ledger, turns, prompts | account, personal ns | `agent_sessions`, `agent_turns` | none: link frames already store them under the link key's namespace, so a workspace run's turns live in its owner's personal namespace and join through `run_id` / `session_pk` |
| `remotea_enabled`, `remotea_auto_complete`, `agent_quiet_bypass` | account | `owner_settings` (personal) | none |
| Workspace opt-in | workspace | `owner_settings` of the workspace ns: `agent_runs_enabled` (default **off**), `agent_auto_complete_allowed` (default off), caps (§ 6) | new; set by the workspace owner in Settings → Sharing |

Runs and targets cross namespaces only along the account. Nothing ever joins a workspace record with
**another** member's personal data.

### 2.2 Host admission

`agent_link.admit` is unchanged: `bot_access.authorize(required="agents:control")` already refuses a
shared namespace (`PERSONAL_WORKSPACE`), `successor()` already requires the same account for a rotation,
and `HOST_BOUND` stops a second member's key from taking over a `server_id` in the same namespace. Two
members on one machine run two remotea servers with two `server_id`s. That is fine: each has its own OS
user, so the OS separates their agents.

Change: link-side run queries key on the **host**, not the key's namespace. In `run_coordinator.reconcile`
and `check_integrations`, drop `AgentRun.owner_id == owner` and keep `AgentRun.host_id == host_id`
(`bound()` already works this way). Otherwise a workspace run would never reconcile, because its
`owner_id` is the workspace and the link key's is personal. `record_question`, `verify.result` and
`worktrees.report` follow the same rule. The host's own data (`agent_sessions`, `agent_turns`, parity,
ingest days, worktree reports, migration counters) stays under the key's personal namespace.

`agent_link.holding()` gains one check per message and tick: for each unsettled run bound to the host
whose `owner_id` is a shared namespace, the run's `account_id` must still be an active `owner`/`editor`
there. When that membership ends, the coordinator moves the run to `stopping(member_removed)` and queues
`stop {mode: interrupt}`. The host is still the member's, so it keeps its connection.

### 2.3 Starting a run on a workspace task

`agent_control.start_run(db, owner=<workspace>, account_id=<member>, …)`:

1. `workspace_gate(db, owner)`: `agent_runs_enabled` on the workspace, and the caller's personal
   `remotea_enabled` (replaces `agent_link.gate(db, owner)` for shared namespaces).
2. Role `owner`/`editor` (viewers: `READ_ONLY`).
3. `resolve_target` filters candidates and advertised targets to hosts with `account_id == caller`.
   `save_target` refuses a `host_id` of another account (`NOT_FOUND`, not `FORBIDDEN`, so other members'
   host ids are not confirmed).
4. Quotas (§ 6.2).
5. `branch_for`: prefix from the member's handle, not `get_settings().owner_id`. `base_for`: the parent's
   run branch only if that run's `account_id` is the caller (another member's branch exists only on
   their machine), else `main`.
6. Prompts: text that names "Davin" (`run_coordinator.py` render prompt "Davin approves completion",
   docstrings) becomes the operator's `preferred_name`. Task text and PRD notes written by **another**
   member are fenced with the label "written by another workspace member", the same way
   `agent_prompts.fence` labels Linear text.

### 2.4 Routes: replace `owner_only` with a per-route access table

Remove `/api/v1/remotea` from `access.PERSONAL_PATHS`. Each route declares one access level in
`remotea_routes.ROUTE_ACCESS` (path template → level), enforced by one dependency `remotea_access(level)`:

| Level | Meaning | Routes |
|---|---|---|
| `personal` | personal namespace only (as today) | `GET /status`, `POST /enable`, `/disable`, `/auto-complete`, `/quiet-bypass`, `GET /sessions`, `/sessions/{id}/turns`, `/parity`, `GET/POST /migrations`, `GET /unmerged` |
| `account` | any namespace; acts only on the caller's own hosts/targets | `GET/POST /targets`, `GET /runs/preflight` |
| `member_read` | any active member of the namespace; output through § 3 redaction | `GET /runs`, `GET /runs/{id}` |
| `run_owner` | I3 | `POST /runs` (start: caller becomes owner), `/runs/{id}/message`, `/approve`, `/replan`, `/verify`, `/verify-config`, `/freestyle`, `/plan-review`, `/interviews/{id}/answers`, `/questions/{id}/answer`, `GET /pending`, `POST /pending/{id}/confirm|decline`, new `POST /runs/{id}/share` |
| `run_owner_or_ws_owner` | I3, plus the workspace owner for interrupt | `POST /runs/{id}/stop` (`mode: kill` stays run-owner only) |

`owner_only` stays as the implementation of `personal`. `agent_control.owner_control` becomes
`run_control(db, owner, run, need)`, used by `control_command` for each tool.

## 3. Shared visibility

### 3.1 Levels

- **owner**: the run's `account_id`. Sees everything (as today).
- **member**: any other active member of `run.owner_id`, any role (viewers already read every workspace
  record). Sees the summary.
- **shared**: `agent_runs.visibility = "workspace"`, set by the owner (`POST /runs/{id}/share {shared}`,
  recorded in the audit). Members see the owner view minus the **always-private** fields.

### 3.2 Fields

New module `agent_visibility.py` holds the classification. `public_run(db, run, *, detail, viewer)`
builds the full dict and then calls `agent_visibility.redact(result, level)`. No caller builds its own
reduced shape.

| Field (`public_run`) | member | shared | Why |
|---|---|---|---|
| `id, task, state, state_revision, state_reason, reason_text, state_since, generation, plan_revision, tool, model, branch, base_branch, turn_count, turn_cap, started_at, finished_at, stop_requested_at, stop_mode, turn_seq, plan_pending, freestyle, integration, completion_policy, linear, created_at` | yes | yes | state and progress |
| `base` (sha, ref, behind, as_of) | yes, without `fetch.error` | yes | freshness; fetch errors can name local paths |
| `effective_permission_mode` | yes | yes | a safety signal for reviewers |
| `host` | `{owner_name, connected}` | same | host id, hostname and machine are personal |
| `owner` (new: account display name) | yes | yes | |
| `structured_answers` | no | no | describes the owner's harness |
| `reason_detail` | no (`reason_text` only) | yes | judge notes quote turn text |
| `worktree`, `surf`, `session` | no | **no** | absolute paths and terminal ids on a personal machine |
| `plan` (title, PRD, steps, criteria, verify, required_children) | yes | yes | the PRD is already a workspace note (`submit_plan` writes it into the task's namespace) |
| `plan.proposals` | summary (`ids, kind`) | yes | the agent's reasons quote its work |
| `history` transitions `{at, from, to, reason, actor}` | yes, actor as display name + kind + via | yes | the audit is meant to be seen |
| `history` `sent` entries | `{at, event, actor, kind, mode, length}`, no `text` | yes | what was typed to an agent |
| `history[].detail` | yes (≤ 300 chars, coordinator-written) | yes | |
| `turns` (`text`, `prompt`) | **no** (count only: `turns_total`) | yes | agent output from a personal machine |
| `commands` | `{kind, status, created_at}` | full minus `payload.run_token` (already `[redacted]`) and `payload.env` | input text |
| `questions` | `{id, status, created_at}` | full, not answerable | |
| `interviews` | `{id, status, answered_at}` | full | the owner's answers |
| `attempts` verify | `{kind, status, verify_id, isolated, exit_code, created_at}` | plus `result.tail/error` | tails are agent-controlled output |
| `attempts` judge | `{verdict, summary ≤ 500, gap ≤ 500, criteria[].status}` | plus `criteria[].evidence`, `rules` | evidence quotes turn text |
| `attempts[].cost`, `model`, `latency_ms` | no | **no** | billed to the member (§ 6.3) |
| `verify_options`, `more` | no / paging only | yes | describes the owner's target config |

The drift guard (§ 9.2) fails when `public_run` gains a key that is not in this table.

### 3.3 Where the redaction applies

- **Browser:** `GET /remotea/runs` and `/runs/{id}` call `public_run(viewer=Viewer(account, role))`.
  `list_runs` gains an `owner=me|all` filter.
- **Eri `control_status`** (`agent_control.status`): own runs as `spoken_run` (unchanged). Other members'
  runs as `{run_id, task, owner, state, why, turns}`, with no questions, interview or plan review. Also
  `run_for(target)` resolves "the only active run" **among the caller's runs**, so Eri never aims a
  command at someone else's run.
- **SSE `agent.changed`:** the change feed carries only `(kind, entity_id, revision)` (`domain.emit`), so
  there is nothing to redact. The rule to keep: workspace runs emit to `run.owner_id` with the **run id**.
  Session and turn ids (`agent_link.turn`, `prompt`, `report`) emit only to the host's personal
  namespace, as they do now because `owner = bot.owner_id`. The web client refetches through the
  redacted route.
- **Bot `agents:read`** (`external_routes.agent_runs/agent_run`, `external_mcp` `run_list/run_get`): the
  viewer is the key's `account_id` with its role in the key's namespace. A key minted in a workspace by
  the run's owner sees that run at owner level; a key minted by anyone else sees the summary.
  `agents:read` stays mintable by viewers.
- **Pushes:** `run_coordinator.notify` writes to the **run owner's personal namespace**
  (`Notification(owner_id=run.account_id)`, quiet bypass from the same). Personal push subscriptions are
  not copied into shared workspaces ([Accounts and sharing](ACCOUNTS_AND_SHARING.md)), and other members
  never get a run's question bodies.
- **Linear:** a workspace task has no Linear link (Linear credentials are personal), so
  `sync_state.linear` is `skipped`. Nothing changes.

## 4. Audit trail

### 4.1 What is recorded today

`history[]` holds every transition with `actor.record()` (`kind`, `credential_id`, `run_id`, `generation`,
`verified`, `reason`, `capability`). It also holds `note()` events: `command`, `sent` (with text ≤ 2000),
`verify_requested`, `freestyle`, `plan_submitted`, `plan_reviewed`, `verify_set`, `approved` /
`completed_externally`. A start confirmed from Eri carries `reason: "confirmed:<pending_id>"`
(`agent_control.decide`). `action_changes` journals Eri's `control.*` calls with `account_id`.
`agent_pending_controls` keeps `account_id`, `decided_via` and `decided_at`.

### 4.2 Gaps

1. **No account on browser actors.** `remotea_routes.actor_for` returns `Actor("human")`. In a workspace,
   "human stopped the run" does not say which human.
2. `answered_via` on questions and interviews is `actor.kind`. Who answered is not stored.
3. `agent_action_receipts` have no account.
4. `history` is capped at 200 entries, so a long run loses its start entry.
5. The workspace owner's stop, a member's completion of a run-owned task (`external_completion`), a
   share toggle and a membership-loss stop have no distinct reasons.
6. There is no "via" (app, voice, chat, bot), only the kind.

### 4.3 Changes

- `agent_actor.Actor` gains `account_id: str | None` and `via: str` (`app | voice | chat | bot | host |
  system`), and `record()` includes both. `actor_for(user)` → `Actor("human", account_id=user.account_id,
  via="app")`. `agent_actor.current()` fills the account from the work grant (`eri_work`) or the
  credential (`bot`), and `via` from the work's channel. `run_session` uses the run's account.
- `answered_by` (account) on `agent_questions` and `agent_interviews`; `decided_by` on
  `agent_pending_controls`; `account_id` on `agent_action_receipts`.
- New table **`agent_run_audit`** (append-only): `id, owner_id (namespace), run_id, account_id, actor_kind,
  via, action, at, state_revision, detail (≤ 300 chars, never message text; text length and sha256
  instead)`. It is written by the same helpers that write `history` (`transition`, `note`, `sent`) when
  the actor has an account, in the same transaction. Actions: `start`, `confirm`, `message`, `steer`,
  `answer`, `interview_answer`, `approve`, `stop`, `kill`, `replan`, `verify`, `verify_config`,
  `freestyle`, `plan_review`, `share`, `complete_external`, `member_removed`.
- New `state_reason`s: `stopped_by_workspace_owner`, `member_removed`. Each needs a `REASONS` /
  `REASON_TEXT` entry (the existing drift guard `test_every_needs_you_reason_has_text` covers the
  `needs_you` ones).

### 4.4 Retention

`agent_run_audit` rows are kept **400 days** and are never deleted with the run's children. The 90-day
child purge in `agent_retention.py` deletes turns, commands, questions, attempts and interviews, not the
audit. `history` stays bounded (it is the UI timeline). Receipts stay 7 days.

### 4.5 Where it shows

- Run drawer timeline: actor display name + via on each entry (members and owner).
- Run drawer **Activity** tab: the `agent_run_audit` rows for that run.
- Settings → Sharing (workspace owner): "Agent activity in this workspace", a paged list of audit rows
  across runs, filterable by member.
- `agents:read` `run_get` gains `audit[]` (the same rows, member-visible).

## 5. Key minting and host admission in shared namespaces

### 5.1 Who may mint what

| Scope | Personal ns (owner) | Shared ns: owner | Shared ns: editor | Shared ns: viewer |
|---|---|---|---|---|
| `agents:control` | only via `POST /remotea/enable` | refused `PERSONAL_WORKSPACE` | refused | refused |
| `agents:session` | yes (Settings → Connections → Connected agents) | refused | refused | refused |
| `agents:read` | yes | yes (redacted per § 3) | yes | yes |
| other scopes | unchanged (`bot_access.create`) | unchanged | unchanged | read-only |

So `bot_access.create`/`authorize` do **not** change: `PRIVILEGED & shared → PERSONAL_WORKSPACE` stays.
`test_shared_workspace_members_cannot_drive_agents` keeps passing except for the routes that phase 2
opens deliberately. Its assertions move into the matrix (§ 9.1).

### 5.2 In-session tools across namespaces (the MCP key)

Each member registers **their own** MCP key in their own harness config (`claude mcp add --scope user …`).
A run on a workspace task spawns with `ERIDANI_RUN_TOKEN`. The in-session tools (`agent_plan.task_plan`,
`task_step`, `task_interview`, `submit_plan`, and the `external_routes` `/agents/session/*` twins)
change from `bot.owner_id` scoping to:

1. `coordinator.run_for_token(db, token)`: find the run by token hash alone, then require
   `run.account_id == bot.account_id`.
2. `access.role(db, run.owner_id, bot.account_id, lock=True)` in `owner`/`editor`.
3. `workspace_gate(run.owner_id)`.
4. Act on `run.owner_id` (task, PRD note, events), as the verified `run_session` actor with the run's
   account.

The session-id fallback (no token, `verified: false`) keeps working only in the key's own personal
namespace. A token leaked to another member's key fails step 1.

### 5.3 Host admission

There is still no admission in shared namespaces (§ 2.2). The roadmap question "admit hosts into a
workspace" is answered **no**: a workspace reaches hosts only through a member's run.

## 6. Confirmation and quotas per member

### 6.1 Two-phase control stays per owner

`agent_pending_controls` already records `account_id`, and `control.confirm` already requires "a later
request by the same account". In a workspace:

- `control.start` / `control.message` propose under `(owner_id = workspace, account_id = caller)`. Only
  the same account confirms (`pending_for` adds `account_id == caller`). `GET /remotea/pending` lists only
  the caller's proposals.
- `control.message` targets only the caller's runs (I3). It is refused `NOT_RUN_OWNER` before a proposal
  is stored, so nobody can queue text for someone else to confirm by accident.
- `control.` leaves `access.PERSONAL_TOOLS`, and `control_` leaves the `tool_access` deny list. Both are
  replaced by `run_control` checks. `tool_policy.remotea_tools` registers the tools in a workspace only
  when `workspace_gate` passes for the caller.

### 6.2 Caps

| Cap | Today | Multi-user |
|---|---|---|
| Unsettled runs per task | 1 (`uq_agent_run_active_task`) | unchanged: a second member gets `RUN_ACTIVE` with the owner's name |
| Turn cap | per task, default 15 (`tasks.turn_cap`) | clamped at start to `min(task.turn_cap, member max, workspace max)`; preferences `agent_turn_cap_max` (member, default 30) and workspace `agent_turn_cap_max` (default 30). Another editor raising a task's `turn_cap` cannot raise your spend |
| Active runs per member | none | `agent_max_active_runs` (member, default 3), counted by `account_id` across namespaces → `409 RUN_QUOTA` |
| Starts per member per day | none | `agent_max_starts_per_day` (member, default 20), workspace `agent_starts_per_member_day` (default 10) → `429 RUN_QUOTA` |
| Wall clock / no-progress | 6 h / 45 min per run | unchanged |
| Ingest | 64 MiB/day per owner | unchanged and already per member (turns are stored under the host's personal ns) |
| Pushes | 3 per 10 min per run | unchanged, sent to the owner only |
| Hosts per key | 3 | unchanged |

Caps are checked in `start_run` under the account's advisory lock (`advisory(db, f"agent-quota:{account}")`).
Counts come from `agent_runs`; no counter table is needed.

### 6.3 Budget

The judge (`agent_judge.ask`) reserves against **`run.account_id`**: the member's personal
`monthly_budget_usd` through `budget.reserve` and `extraction_request`. It no longer uses
`attempt.owner_id`, which for a workspace run would be the workspace. A new member preference
`agent_judge_budget_usd` (monthly, default 5) is checked first. Over it, the attempt is a `budget`
non-verdict, which already ends in `needs_you(judge_unavailable)` without retries. `budget.usage_report`
gains a `feature=agent_control` breakdown per run. The harness's own model cost (Claude Code / Codex) is
the member's own subscription. Eridani does not see it and does not meter it.

### 6.4 Completion authority in a workspace

`agent_actor.gate` treats any `human` as an authority. In a workspace, "human" becomes the **run owner or
the workspace owner**. Another editor who completes the task, or edits `success_criteria`, `verify`,
`turn_cap` or `completion_policy` while the run is unsettled, gets `409 RUN_OWNS_COMPLETION` naming the
owner. `auto_complete` on a workspace run needs both the owner's `remotea_auto_complete` and the
workspace's `agent_auto_complete_allowed`, plus an **isolated** passing verify (§ 7). `human_gate` still
applies.

## 7. Verify sandboxing on the host

### 7.1 Why multi-user raises the stakes

Verify runs agent-edited code (`package.json` scripts, `conftest.py`) as the host user, with network.
`verify-runner.ts` now passes a minimal env and checks `realpath`, but it is not a sandbox (audit M8,
remotea M3). In a workspace, the text the agent reads can come from **another member** (task notes, PRD
edits, interview context). That text could steer the agent into writing a test that exfiltrates the
owner's `~/.ssh` once verify runs. This is the same work as Astra's **S-N4** (Eridani task `bee876d3`,
[PRD § S-N4](PRD_MISSION_CONTROL_RELIABILITY.md)). This design adds the Eridani-side policy that consumes
it.

### 7.2 Host (remotea, S-N4)

- `eridani.json` per verify entry: `isolation: "bwrap" | "codex" | "none"` (default `bwrap` when
  `bwrap` is on PATH and unprivileged user namespaces work, which they do on WSL2), plus
  `network: false` (default) and `mounts.ro[]` for dependency caches (`~/.cache/uv`, `~/.npm`).
- **bwrap** argv (fixed, no shell): `--unshare-all --die-with-parent --new-session --clearenv`,
  `--ro-bind /usr /usr` (+ `/bin /lib /lib64 /etc/ssl /etc/resolv.conf` as symlinks or read-only binds),
  `--bind <worktree> <worktree>`, `--tmpfs /tmp`, `--tmpfs $HOME` (a **temporary HOME**), the declared
  read-only cache mounts, `--chdir <worktree>`, the minimal env from `verifyEnv`, and `prlimit` for CPU,
  address space, open files and processes, under the existing timeout. `--unshare-all` includes the
  network namespace, so the network is off.
- **codex**: `codex sandbox linux --full-auto -- <argv>` (Landlock + seccomp, workspace-write, no
  network) where bwrap is missing.
- **none**: allowed only with `trust: "unsandboxed"` set explicitly on the entry.
- Dependency provisioning (`uv sync --locked`, `npm ci`) runs **before** and **outside** the sandbox
  with network, from the base branch's lockfile only (S-N3). Inside, it runs offline.
- `hello` caps: `verify_isolation`. The per-target `verify` entries in `hello.targets[]` gain
  `isolation`. `verify.result` gains `isolated: bool` and `isolation`.

### 7.3 Eridani

- `agent_attempts.result.isolated` is stored and shown beside the result ("passed · isolated" /
  "passed · not isolated").
- **Personal runs:** unchanged behaviour. The badge is informational, and `auto_complete` keeps its
  current rules.
- **Workspace runs:** `coordinator.start_verify` queues the automatic per-turn verify only when the
  target advertises `isolation != none`. Otherwise the run uses on-demand verify only (`POST
  /runs/{id}/verify`, the owner's click), with a warning. The judge's hard rule `verify_not_isolated`
  turns a `met` on a non-isolated pass into `awaiting_approval` (never auto-complete).
- Tests (S-N4's criteria plus Eridani's): a fixture cannot read a sentinel outside the mounts, write
  outside the worktree and `/tmp`, or open a socket (remotea `remote-server` tests). An unisolated result
  never auto-completes a workspace run (`tests/test_agent_judge.py`).

## 8. Rollout

**Phase 0: preparation, no behaviour change** (one PR, one migration)
- `Actor.account_id/via`, `actor_for(user)`, `current()`; `answered_by`, `decided_by`, receipt
  `account_id`; `agent_run_audit` and its retention exemption.
- `agent_runs.account_id` NOT NULL (backfill `owner_id`); `execution_targets.account_id` (backfill from
  host) and the new unique index; CHECK `agent_hosts.owner_id = account_id`.
- `reconcile` / `check_integrations` keyed by host only.
- `agent_visibility.py` with the § 3.2 table, and `public_run(viewer=…)` defaulting to owner level.
- Branch prefix from the account, `base_for` restricted to the caller's runs, "Davin" removed from prompts.
- Drift guards § 9.2.

**Phase 1: one additional member, personal workspaces only** (the first goal)
- The member signs in with their own account (the existing invite flow), turns on remotea in their
  **personal** Settings → Connections, installs remotea on their own desktop, links it with their own
  key and registers their own MCP key. Tasks live in their personal workspace. Nothing is shared.
- Build: per-member caps (§ 6.2, member-level only) and judge billing to `run.account_id` (§ 6.3).
- Prove: the cross-account rows of the matrix (§ 9.1). Account B's link key, MCP key, run token,
  `agents:read` key and browser session cannot see or move anything of A's. Most of this already holds
  through namespace scoping. Phase 1 adds the tests.
- Done when: B runs a task end to end and the matrix is green.

**Phase 2: runs on shared-workspace tasks, owner-only control, summary visibility**
- Prerequisite: S-N4 isolated verify on the member's host, or the workspace runs on-demand verify only
  (§ 7.3).
- Remove `/api/v1/remotea` from `PERSONAL_PATHS`; add `ROUTE_ACCESS` and `remotea_access`; workspace
  opt-in (Settings → Sharing); `start_run` with account-filtered targets; `holding()` membership check;
  redaction on the browser routes and `agents:read`; pushes to the owner's namespace; completion
  authority (§ 6.4); workspace caps.
- Web: `Remotea.tsx` and the run drawer render the member view (no message box or actions on other
  members' runs, "Started by …" header, owner filter). `remotea-model.ts` gains `RunView = "owner" |
  "member" | "shared"`.

**Phase 3: Eri control, sharing and oversight**
- `control.*` in workspaces (§ 6.1), the `control_status` member view, `POST /runs/{id}/share`, the
  workspace owner's interrupt-stop, Settings → Sharing activity list, and `audit[]` on `run_get`.

**Phase 4: optional**
- An account-level "my runs everywhere" list on the personal remotea page (only runs with `account_id ==
  me`, with membership re-checked per row), and workspace auto-complete.

## 9. Tests

### 9.1 Actor × tenant × scope matrix (`tests/test_agent_multi_user.py`)

Modelled on `tests/test_agent_security.py`: one parametrized test per action. Fixtures: accounts A
(run owner), B (editor), V (viewer), W (workspace owner), X (non-member); a workspace with A, B, V, W;
A's host `hA` with link key `kA`; B's host `hB` with link key `kB`; MCP keys `sA`, `sB`; `agents:read`
keys minted in the workspace by A, B and V; a run `r` by A on a workspace task.

| Action on A's workspace run `r` | A | B (editor) | V (viewer) | W (ws owner) | X | `kB` / `hB` frames | `sB` + `r`'s token | `agents:read` (B) |
|---|---|---|---|---|---|---|---|---|
| read list/detail | owner view | summary | summary | summary | 404 | `RUN_NOT_BOUND` | n/a | summary |
| read turns/commands text | yes | no | no | no | 404 | no | n/a | no |
| message / steer | ok | `NOT_RUN_OWNER` | `READ_ONLY` | `NOT_RUN_OWNER` | 404 | n/a | n/a | 403 scope |
| answer / interview | ok | `NOT_RUN_OWNER` | `READ_ONLY` | `NOT_RUN_OWNER` | 404 | `RUN_NOT_BOUND` | n/a | 403 |
| approve / replan / verify / verify-config / freestyle / plan-review | ok | `NOT_RUN_OWNER` | `READ_ONLY` | `NOT_RUN_OWNER` | 404 | n/a | n/a | 403 |
| stop interrupt | ok | `NOT_RUN_OWNER` | `READ_ONLY` | ok, audited | 404 | n/a | n/a | 403 |
| stop kill | ok | `NOT_RUN_OWNER` | `READ_ONLY` | `NOT_RUN_OWNER` | 404 | n/a | n/a | 403 |
| complete task / edit gate fields | ok | `RUN_OWNS_COMPLETION` | `READ_ONLY` | ok, run stops | 404 | n/a | n/a | n/a |
| `submit_plan` / `task_step` | n/a | n/a | n/a | n/a | n/a | n/a | `RUN_TOKEN_INVALID` | n/a |
| start on `hB`'s target | `NOT_FOUND` | (own) ok | `READ_ONLY` | `NOT_FOUND` | 404 | n/a | n/a | n/a |
| `control.message` via Eri | proposal | `NOT_RUN_OWNER` | `READ_ONLY` | `NOT_RUN_OWNER` | 403 | n/a | n/a | n/a |
| confirm A's proposal | ok | `NOT_FOUND` | `NOT_FOUND` | `NOT_FOUND` | 403 | n/a | n/a | n/a |

Also covered:
- Personal-namespace rows (phase 1): every action by B on A's **personal** run → 404, and `kB` frames
  naming `r` → `RUN_NOT_BOUND`.
- The key rows of § 5.1 (mint and use) in both namespaces.
- The membership-loss row: remove A from the workspace → `r` goes `stopping(member_removed)`, and A's
  host stays connected.

### 9.2 Drift guards (`tests/test_drift_guards.py`)

- `test_every_public_run_field_is_classified`: build a fixture run with every child row and call
  `public_run(detail=True)` at owner level. Every key, including nested `attempts[]`, `history[]` and
  `commands[]` keys, is in `agent_visibility.MEMBER | SHARED | OWNER_ONLY`. A member-level result
  contains no `OWNER_ONLY` key.
- `test_every_remotea_route_declares_its_access`: the paths of `remotea_routes.router.routes` equal
  `ROUTE_ACCESS` keys, and every level is one of the five in § 2.4.
- `test_browser_actors_carry_an_account`: no `Actor("human")` without `account_id` in
  `remotea_routes.py`, `agent_control.py` or `agent_plan.py` (source scan, like
  `test_every_needs_you_reason_has_text`).
- `test_link_run_queries_key_by_host`: in `run_coordinator.py` and `agent_link.py`, no
  `select(AgentRun)` filters on both `AgentRun.owner_id == owner` and `AgentRun.host_id`.
- `test_audit_actions_match_the_doc`: `agent_run_audit` action names equal the list in § 4.3 of this
  file (parsed from the doc).
- `test_privileged_scopes_stay_personal`: `bot_access.PRIVILEGED` still includes `agents:control` and
  `agents:session`, and `create` refuses them in a shared namespace. This pins I7 against a later
  "convenience" change.

`test_settings_can_grant_every_bot_scope_but_the_link_key` and `test_every_control_tool_is_gated_on_remotea`
stay as they are.

## 10. Open questions for Davin

1. **Phase 1 first?** Should a second member start with personal-workspace runs only, with no shared
   runs until a shared project needs them? *Recommendation: yes. Phase 1 is mostly tests and two small
   fixes, and it proves the per-account isolation everything else depends on.*
2. **Workspace owner's stop.** Should the workspace owner be able to interrupt-stop a member's run?
   *Recommendation: yes, interrupt only, audited. Kill, message and approve stay with the run owner.*
3. **Who may complete a run-owned workspace task?** *Recommendation: the run owner and the workspace owner.
   Other editors get `RUN_OWNS_COMPLETION`, so they cannot stop a colleague's agent by ticking a box.*
4. **Default visibility.** Summary or full timeline for other members? *Recommendation: summary, with a
   per-run Share toggle. Turn text comes from a personal machine and can contain local paths and
   secrets that slipped past redaction.*
5. **Who pays for the judge?** The member or the workspace? *Recommendation: the member's personal
   budget, with a small `agent_judge_budget_usd` default. Workspaces have no billing owner today.*
6. **Unsandboxed verify on workspace runs.** Allow it with an explicit per-target trust flag, or
   require isolation? *Recommendation: no automatic verify without isolation; on demand only, with a
   warning, and never auto-complete.*
7. **Host names.** Should other members see hostnames and machine names? *Recommendation: no. Show
   "<member>'s computer · connected".*
8. **Edits by other members during a run.** Should edits to the task or PRD note by another member
   pause the run? *Recommendation: no pause. The agent works from the frozen plan, other members' text
   is fenced and labelled, and a later `plan_revision` change only shows "plan changed since start" in
   the drawer.*
9. **Audit retention.** *Recommendation: 400 days for `agent_run_audit`, independent of the 90-day
   content purge.*
10. **Where Brainforge members live.** Separate accounts on app.eridani.app, or their own deployments?
    *Recommendation: accounts on the same deployment (invites already exist). A separate deployment
    per person makes every boundary here moot but forks operations.*
