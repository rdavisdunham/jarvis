# PRD: mission control reliability (remotea server, remotea desktop, Eridani)

Written 2026-10-30 from Astra's integration review (`~/remotea-eridani-analysis-2026-10-29.md`, reviewed remotea at 76c5f62 and Eridani at the deployed main of 2026-10-29) and the open items on the Eridani roadmap project. Every work item below is also an Eridani task in the "Eridani roadmap" project; the task list is the source of truth for status, this document for the reasoning and the order. Runs are started from the tasks (Run on the row); items in one batch touch different files and may run in parallel; batches run in order.

## Goals

1. A control action in Eridani always acts on the run the operator is looking at, with the state they reviewed, exactly once.
2. A Stop always wins over anything queued before it, on both sides of the link.
3. Verification is a recorded, reproducible result under the service identity, not a claim; its infrastructure failures are named as such.
4. The desktop never ends work the operator cannot see without saying so, and never loses an intent it accepted.
5. Persistence failures degrade the link, never the terminal host.

Out of scope here: the product items on the roadmap that are not about mission control (Android plan, notification bundles, pgvector, GPT-Live memory, speed and acceptance tracks, Google/Linear onboarding); they stay on the roadmap unchanged.

## Work items

### A. Run control correctness (Eridani)

#### E-N1+E-N3 — Run panel control correctness: no stale response can change which run is shown, and decisions carry the reviewed revision

Repo: jarvis · Status: open · Run batch: 1 · Origin: Astra review E-N1, E-N3 · Eridani task `80852c32`

Astra's review (E-N1, reproduced): RunDrawer.refresh() in apps/web/src/Remotea.tsx starts a fetch for the current id and installs whatever response arrives; switching runs clears the view but does not cancel or ignore the earlier request, and RunPanel reuses the drawer without a per-run key. With out-of-order responses the title showed Task A while Stop POSTed to run B.

E-N3 (source-confirmed): the backend supports expected_revision on message, stop, approve, replan, verify and plan-review, but the drawer sends approve and verify with empty bodies and message/replan without the reviewed state_revision, so a stale tab can approve or replan after another actor changed the run.

Fix both in the drawer and panel:
- Give each fetch a request token (or AbortController); accept a response only if it matches the current run id and latest request. Key the RunDrawer by run id in RunPanel so draft text, busy state and scroll position are per run.
- Bind every control's URL and body to the loaded run, not the prop id; disable controls until the loaded run id equals the prop id.
- Send expected_revision = run.state_revision on approve, replan, verify, verify-config, freestyle and message/steer. Keep Stop unconditional (emergency path) and say so in a comment. On REVISION_CONFLICT show "This run changed; reloaded" and refresh instead of a generic error.
- Backend: make expected_revision required for approve and plan-review (ApproveInput, PlanReviewInput); keep it optional for stop. Update docs/AGENT_CONTROL.md browser routes.

Success criteria:

- With responses for run A and run B resolved out of order, the panel's title, controls and the URL of the next action all refer to the same run (vitest with a fake fetch that resolves B before A).
- Approve and plan-review without expected_revision are refused with REVISION_CONFLICT or a validation error; approve with a stale revision is refused and the drawer refreshes (pytest in tests/test_agent_security.py or tests/test_agent_runs.py).
- Stop still works without a revision.
- The web test suite (vitest) and type check pass; the backend suites tests/test_agent_runs.py, tests/test_agent_security.py and tests/test_drift_guards.py pass.

#### E-N2 — Browser run actions are idempotent: a retried message or steer never becomes a second prompt

Repo: jarvis · Status: open · Run batch: 1 · Origin: Astra review E-N2 · Eridani task `b6a01258`

Astra's review (E-N2, source-confirmed): the web api helper advises retrying after a lost response, but POST /api/v1/remotea/runs/{id}/message accepts only text and mode; agent_control.send_message creates an AgentCommand whose op_id is a fresh UUID, so a retry queues a second prompt or advances a second steer generation. The generic POST /api/v1/commands already uses a client request_id with transactional dedupe; the run-action routes lack it.

Build:
- MessageInput (and ReplanInput, verify-config, freestyle, approve) accept an optional client action_id (UUID). The web generates it once per click and reuses it on retry (apps/web/src/api.ts post helper or a small wrapper in Remotea.tsx).
- Server: a unique index or lookup on (owner, run, action_id) stored on the command row or a small agent_action_receipts table (migration 0035 if a table is needed; see docs/ADDING_FEATURES.md section 6 for the head). A repeat with the same action_id and the same body returns the original result; the same action_id with a different body is refused (409 ACTION_MISMATCH).
- Document in docs/AGENT_CONTROL.md browser routes.

Success criteria:

- Posting the same message twice with the same action_id creates exactly one AgentCommand and returns the same response; a different body with the same action_id is refused (pytest).
- The web sends an action_id with message, steer and replan and reuses it on the retry path (vitest or a unit test of the helper).
- pytest tests/test_agent_security.py tests/test_agent_runs.py tests/test_drift_guards.py and the web suites pass; alembic upgrade head and alembic check pass if a migration is added.

#### E-N4+S-S1 — Capabilities are checked on the selected execution target, not any host

Repo: jarvis · Status: open · Run batch: 1 · Origin: Astra review E-N4, S-S1 · Eridani task `8b9acbc2`

Astra's review (E-N4, source-confirmed): remotea's hello carries per-target harnesses and verify ids, but agent_control.advertised() drops them from the picker model; host_harness() checks host-wide flags, and preflight marks a harness available when any connected host advertises it. The live host enables Claude only on the jarvis target while server-wide caps advertise both Claude and Codex, so Codex can look startable and fail at the host.

Build (Eridani side; the remotea side is S-S1, a separate task):
- Keep per-target harnesses and verify ids in advertised targets (SavedTarget/AdvertisedTarget shapes in agent_control and apps/web/src/remotea-model.ts).
- host_harness(host, harness, host_target) checks the target's own flags when the host advertises them, falling back to host-wide flags for older hosts.
- preflight(...).harnesses reports the selected target's capabilities (or the resolved target's), and start_run/check_host refuse a harness the target does not enable with HARNESS_UNAVAILABLE naming the target.
- The Run dialog disables a harness the chosen target cannot start and says why; verify_options come from the selected target only (they already do via verify_ids).

Success criteria:

- A host advertising two targets where only one enables codex: preflight for a task resolved to the Claude-only target reports codex false, and start_run with harness=codex is refused with HARNESS_UNAVAILABLE naming the target (pytest).
- Another connected host's capabilities cannot make the selected target look ready (pytest).
- Older hosts without per-target harness flags keep working (existing tests pass).
- tests/test_agent_security.py, tests/test_agent_runs.py, tests/test_drift_guards.py and the web suites pass.

#### E-N5 — Run timeline shows older pages and full clipped turns instead of silently hiding them

Repo: jarvis · Status: open · Run batch: 2 · Origin: Astra review E-N5 · Eridani task `8dc3f534`

Astra's review (E-N5): GET /remotea/runs/{id} returns the newest 100 turns and commands with pagination metadata (more.turns, more.commands, offset, limit) and clips turn text at 4,000 characters with a clipped flag. The drawer fetches only the default page, offers no older-page control, and its "Show all" expands only the text already returned.

Build in apps/web/src/Remotea.tsx and remotea-model.ts:
- A "Load earlier" control at the top of the log that fetches the next older page (offset) and merges it, keeping scroll position.
- For a clipped turn, "Full response" fetches the full text (add GET /remotea/runs/{id}/turns/{turn_id} returning the full text, or reuse /remotea/sessions/{id}/turns with the turn id) and shows it.
- Distinguish in the UI: clipped (display), truncated (the host cut it), paged out (older than this page), and expired by retention (gone), each with its own small label.

Success criteria:

- A run with more than 100 commands shows a Load earlier control that fetches and merges the older page (vitest on the model's merge helper plus a render test).
- A turn longer than 4,000 characters can be expanded to its full text via a dedicated fetch (pytest for the route, vitest for the control).
- The web suites and type check pass; the surface inventory is refreshed if a route is added.

### B. Verification readiness

#### S-N3 — Host: make the configured verification command runnable under the remotea service identity

Repo: host · Status: open · Run batch: not a run · Origin: Astra review S-N3 · Eridani task `3a58cf14`

Astra's review (S-N3, observed): the jarvis target's verify command is `uv run --no-sync python -m pytest tests -q -x`, but `uv` is not on the systemd service's PATH (verify runs argv directly with shell: false), there is no explicit isolated test database (JARVIS_DATABASE_URL) or JARVIS_ENV_FILE="" in the verify environment, the timeout is 600 s while the API suite takes 470–670 s, and managed worktrees get no dependency provisioning (uv sync / npm ci).

Manual, on the desktop (Davin): in ~/.config/remotea/eridani.json give the jarvis target a verify entry that uses an absolute uv path (or a wrapper script in the repo, e.g. scripts/verify.sh, that sets PATH, JARVIS_ENV_FILE="" and a disposable JARVIS_DATABASE_URL pointing at a local test Postgres, runs uv sync --locked --group dev, then pytest, then the web type check and tests), timeout 1800; add a second target for /home/davin/remotea/remotea with a verify entry running the remote-server tests. Restart remotea.service afterwards. Eridani's remotea doctor/readiness (S-Q1, E-S2) will later check this automatically.

#### S-N4 — Isolate automatic verification from the host user's credentials and filesystem

Repo: remotea · Status: backlog · Run batch: not a run · Origin: Astra review S-N4 · Eridani task `bee876d3`

Astra's review (S-N4, design limitation): remote-server/src/verify-runner.ts uses fixed argv and a minimal environment but runs agent-editable code (conftest.py, package scripts) with the server user's filesystem and network access; HOME is retained. Before unattended verification of untrusted changes, run it in a constrained environment (temporary HOME, deliberate mounts, resource limits, explicit network policy), and report "tests passed" separately from "the check was isolated". Until then make automatic verification an explicit per-target trust choice in eridani.json.

Success criteria:

- A test fixture cannot read a sentinel file outside allowed mounts, write outside the workspace and scratch area, or reach a disallowed network destination (remote-server tests with a fake fixture).
- Normal tests still run and receive no host credentials.
- The verify.result frame carries an isolated flag that Eridani shows beside the result.

#### S-S2 — Workspace evidence reports its own completeness

Repo: remotea · Status: backlog · Run batch: not a run · Origin: Astra review S-S2 · Eridani task `ec0e9b0b`

Astra's review (S-S2): verify-runner.ts workspaceState() hashes status/diff stdout without requiring those git commands to succeed, caps untracked-file processing at 2,000 and uses metadata for files over 4 MiB, yet the result reads as an exact tree fingerprint. Return completeness and errors with the fingerprint, and make Eridani treat an incomplete fingerprint as not fresh for auto_complete.

Success criteria:

- A failed git status or a tree over the caps yields a fingerprint marked incomplete with the reason (remote-server tests).
- Eridani's judge treats an incomplete fingerprint as not fresh for auto-completion (pytest in tests/test_agent_judge.py).

### C. remotea server durability and control

#### S-N1 — Cancel and re-fence every delayed terminal write: a newer Stop or steer must win

Repo: remotea · Status: open · Run batch: 3 · Origin: Astra review S-N1 · Eridani task `e45db848`

Astra's review (S-N1, reproduced): in remote-server/src/dispatch.ts a steer advances the generation, sends Escape, waits 800 ms and then checks only that the surface is live before typing; the typing helper schedules Enter 150 ms later and returns without awaiting it. Generation, stop flag and incarnation can change during either delay. In a harness, a steer at generation 2 was pasted and its Enter injected after a stop was requested at generation 3, and the handler still reported executed.

Build: track cancellable pending writes per surface and run; recheck generation, incarnation, stop intent and eligible agent state immediately before every byte injection, including the Enter timer; report `executed` only when the intended submission happened, and an explicit outcome (`interrupted`, with what was and was not written) otherwise.

Success criteria:

- Deterministic tests with fake PTYs and controlled timers: Stop during the Escape delay, Stop during the Enter delay, two overlapping steers, exit/respawn during the delay, disconnect and reconciliation during the delay. No superseded operation writes further bytes, and each reports an honest outcome.
- remote-server typecheck and vitest pass.

#### S-N2 — Contain journal and spool persistence errors without losing the terminal host

Repo: remotea · Status: open · Run batch: 3 · Origin: Astra review S-N2 · Eridani task `9d3eda83`

Astra's review (S-N2): remote-server/src/eridani-journal.ts Journal.record() updates its in-memory record before the append and fsync; eridani-link.ts dispatches `void this.onCommand(...)` and several journal writes sit outside the handler's catch, so a disk-full, permission or sync failure can escape as an unhandled rejection in the process that owns every PTY. A synthetic fdatasyncSync failure made record() throw while get(op) already reported `accepted`.

Build: advance in-memory durable state only after the durable append succeeds; catch persistence failures at the asynchronous boundary; stop admitting effectful commands while durability is unavailable and expose a degraded reason (status file and hello/status to Eridani) while existing sessions stay usable; never ack durability that was not achieved.

Success criteria:

- Injected write, sync, compaction and permission failures: existing PTYs remain available, no unjournaled effect runs, the link reports degraded, and reconnect cannot mistake a memory-only state for a durable one (remote-server tests).
- remote-server typecheck and vitest pass; Eridani shows the host's degraded reason on the remotea page (a small jarvis follow-up if the frame changes).

#### S-N5 — Rotating the desktop token revokes existing connections

Repo: remotea · Status: open · Run batch: 3 · Origin: Astra review S-N5 · Eridani task `5fddb759`

Astra's review (S-N5, reproduced): remote-server/src/server.ts checks the bearer token only when admitting a WebSocket; `remotea token --rotate` (cli.ts) writes a new token but existing sockets keep delivering messages. Build: tag each connection with the credential version it authenticated with; on rotation close (or force re-auth of) sockets with the old version while the PTYs stay alive; document that rotation is revocation.

Success criteria:

- After rotation an old socket loses read and write access promptly and a new socket needs the new token; terminal sessions survive (remote-server tests with a fake hub).
- remote-server typecheck and vitest pass; the README's token section says rotation revokes.

#### S-N6 — One supported Node version across installer, package and collector

Repo: remotea · Status: open · Run batch: 4 · Origin: Astra review S-N6 · Eridani task `d7f09459`

Astra's review (S-N6): remote-server/package.json and scripts/install.sh accept Node 20 while the Codex collector loads node:sqlite (Node 22); CI uses 22. Publish one minimum, enforce it in install.sh and at collector activation with an actionable error, and probe node:sqlite explicitly.

Success criteria:

- install.sh refuses an unsupported Node with a clear message; a supported install opens a synthetic Codex database read-only (remote-server tests or a script check).
- package.json engines, install.sh and CI agree on the version.

#### S-S1 — Advertise capabilities per target from the real launch environment

Repo: remotea · Status: open · Run batch: 4 · Origin: Astra review S-S1 · Eridani task `3852620f`

Astra's review (S-S1): eridani-runtime.ts stageCCaps marks both harnesses spawn-capable even when version probes return null, and target launch permissions are not reflected. Probe through the same executable and environment rules as launch; advertise per-target harnesses and verify ids in hello (the fields already exist) and distinguish protocol support, installed harness, target permission and verification readiness. Pairs with the Eridani task "Capabilities are checked on the selected execution target".

Success criteria:

- hello advertises spawn only for harnesses whose probe succeeded and, per target, only those the target's launch config enables (remote-server tests).
- remote-server typecheck and vitest pass.

#### S-S3 — Upload storage budgets and lifecycle

Repo: remotea · Status: backlog · Run batch: not a run · Origin: Astra review S-S3 · Eridani task `f418b057`

Astra's review (S-S3): uploads are capped at 200 MiB each and failed uploads are cleaned, but successful files stay in /tmp/remotea-drop-* with no expiry. Add age and total-size budgets with active-reference protection, free-space reporting and clear degraded behaviour.

Success criteria:

- Old unreferenced uploads are reclaimed under the budget; referenced ones are kept; the status report shows upload usage (remote-server tests).

#### S-S4 — Worktree writer policy: one writer per branch unless deliberately shared

Repo: remotea · Status: backlog · Run batch: not a run · Origin: Astra review S-S4 · Eridani task `0a13f16f`

Astra's review (S-S4): worktrees.ts serializes creation and reuses an existing branch worktree, but nothing prevents two sessions from editing the same branch. Check active use before assigning another writer; require deliberate sharing or a separate branch; report the conflict to Eridani as a spawn refusal with a reason.

Success criteria:

- A second spawn on a branch with a live writer is refused with a reason unless the spawn says share (remote-server tests); Eridani shows the refusal on the run.

#### S-S5 — Build identity in diagnostics: build SHA, protocol and hook versions, target readiness

Repo: remotea · Status: open · Run batch: 4 · Origin: Astra review S-S5 · Eridani task `4238bc0c`

Astra's review (S-S5): the server reports package version 0.1.0 and a running dist directory does not prove which commit built it. Embed the build SHA at build time, report it with protocol/capability version and hook version in hello, `remotea status` and the diagnostics; Eridani shows it on the host row (Settings → Connections) next to the existing version line.

Success criteria:

- `remotea status --json` and hello include build_sha and protocol version; the Eridani host row shows the SHA (remote-server tests; a small jarvis follow-up for the display).

#### S-S6 — Stabilise the flaky eridani-link outcome test and add fault-oriented cases

Repo: remotea · Status: open · Run batch: 4 · Origin: Astra review S-S6 · Eridani task `d0f57c69`

Astra's review (S-S6) and two subagents today saw `eridani-link.test.ts > stage F > the digest is recomputed…` fail in the full suite and pass alone: it asserts the command.outcome before it is journaled and sent. Wait for the outcome (`await until(() => eri.of('command.outcome').length === 1)`), then add cases for storage failure, delayed input after stop, credential rotation and the service environment (PATH) as the related fixes land.

Success criteria:

- The full remote-server suite passes three times in a row with no flaky failure.
- New fault cases exist for the listed scenarios or are added by their fix tasks.

### D. Desktop safety

#### A-N1 — Desktop: keep registry intents until the server confirms them

Repo: remotea · Status: open · Run batch: 5 · Origin: Astra review A-N1 · Eridani task `6dc98942`

Astra's review (A-N1, reproduced storage behaviour): src/main/remote-host-config.ts PendingIntents.take() deletes a host's queued actions and persists the deletion before returning them; remote-host-client.ts onReady() sends them and sendIntent() registers no acknowledgement that retries or restores. A lost connection or app exit between dequeue and application silently loses close, hide, rename, move or group changes.

Build an outbox with stable intent ids: remove an entry only after acknowledged application (or a proven reconciled state); explicit failed, expired and superseded states; a retried metadata change must not revive an older choice over a newer one.

Success criteria:

- Drop the connection before send, after send, and after server application but before ack, restarting the desktop in each case: the final shared state reflects the latest intent exactly once or the UI reports failure (unit tests with a fake host).
- Root typecheck passes and the touched desktop tests pass.

#### A-N2 — Desktop: hide on this device vs end sessions on this host are distinct, and ending shows what it affects

Repo: remotea · Status: open · Run batch: 5 · Origin: Astra review A-N2 · Eridani task `40f97bde`

Astra's review (A-N2): closing a workspace with remoteGroups closes the server group everywhere, including sessions hidden on this device and Eridani runs; confirmWorkspaceClose defaults to false and the optional dialog does not say other devices and unseen sessions are affected. Provide two commands (Hide on this device; End sessions on this host); for ending, show host, active and hidden session counts and associated Eridani tasks, independent of the local-close preference; say when counts are stale offline.

Success criteria:

- A group with visible and hidden sessions cannot be terminated through the ordinary close gesture; the end dialog lists counts and tasks (renderer tests).
- Quitting the desktop still detaches without killing remote sessions.

#### A-S1 — Desktop: bounded registry requests with timeout and reconcile

Repo: remotea · Status: backlog · Run batch: not a run · Origin: Astra review A-S1 · Eridani task `ee1c4fb5`

Astra's review (A-S1): HostClient.request() resolves on ack or disconnect but has no deadline while the socket stays up, so a lost response leaves an operation pending forever. Add a timeout that yields an `uncertain` state and reconciles against the next snapshot before retrying.

Success criteria:

- A request with no reply times out into uncertain and is reconciled from the next snapshot (unit test).

#### A-S2 — Desktop: show connection, terminal and orchestration state separately

Repo: remotea · Status: backlog · Run batch: not a run · Origin: Astra review A-S2 · Eridani task `2173af2f`

Astra's review (A-S2): one green badge conflates transport connected, terminal process state and Eridani run state. Present them separately with last-known timestamps.

#### A-S3 — Desktop: explain transport security when accepting a host URL

Repo: remotea · Status: backlog · Run batch: not a run · Origin: Astra review A-S3 · Eridani task `020f423a`

Astra's review (A-S3): validateHostUrl() accepts any ws: or wss: URL. Add guidance (Tailscale or wss expected) and a deliberate insecure-network acknowledgement for plain ws: outside loopback/Tailscale.

#### A-S4 — Desktop: a Windows release smoke matrix

Repo: remotea · Status: backlog · Run batch: not a run · Origin: Astra review A-S4 · Eridani task `7c01c592`

Astra's review (A-S4): Linux tests cannot cover OpenSSH path resolution, credential storage, ConPTY, suspend/resume or packaging. Add a short manual or scripted smoke pass per release: laptop sleep, host restart, partial upload, two-device group edits, app upgrade.

#### A-S5 — Desktop: reassess main-window privileges (sandbox, webviews)

Repo: remotea · Status: backlog · Run batch: not a run · Origin: Astra review A-S5 · Eridani task `927d6c3a`

Astra's review (A-S5): Node integration is off and context isolation on, but sandbox: false and webviews remain. Audit whether the preload still needs that and narrow privileges behind IPC where practical.

### E. Eridani readiness, observability and docs

#### E-S1 — Cross-process wake for queued host commands

Repo: jarvis · Status: open · Run batch: 2 · Origin: Astra review E-S1 · Eridani task `95ad853a`

Astra's review (E-S1): agent_link.LINKS and wake() are process-local; a command queued by the worker (judge gap prompts, interview planning prompts) waits for the 30-second fallback tick before the web process's socket sends it. Use Postgres LISTEN/NOTIFY (or the existing SSE change bus) so a queued command wakes the socket owner at once, keeping the durable DB poll as recovery.

Success criteria:

- A command queued from a different process is sent within one second in a two-process test (or a unit test that the notify channel triggers the drain).
- tests/test_agent_link.py and tests/test_agent_runs.py pass.

#### E-S2 — Execution infrastructure failures are a distinct, visible category

Repo: jarvis · Status: open · Run batch: 2 · Origin: Astra review E-S2 · Eridani task `3a3eaf6a`

Astra's review (E-S2): missing executable or database, host offline, expired command, failed tests, stale workspace and an inconclusive judge all need different user actions, but reach the run as similar failures. Carry structured failure reasons from verify.result (missing binary, dependency, database, config) and command outcomes into the run view and control_status, with the action that fixes each; never prompt the agent to change code for a host setup problem. Pairs with the host readiness task.

Success criteria:

- A verify attempt whose detail says the executable or database was missing is shown as a host setup problem with the fix, not as failed tests, and the judge does not send a gap prompt for it (pytest + vitest).
- control_status reads the same category to Eri.

#### E-S3 — Retention and redaction policy is explicit across every copy of run content

Repo: jarvis · Status: backlog · Run batch: not a run · Origin: Astra review E-S3 · Eridani task `18e3a5f3`

Astra's review (E-S3): the retention sweep removes old child records but keeps the run, frozen plan and bounded history, and sent() copies up to 2,000 characters of message text into history. Document and implement redaction/export/deletion across all retained representations.

#### E-S4 — docs/AGENT_CONTROL.md: replace the stale read-only introduction with the shipped behaviour

Repo: jarvis · Status: open · Run batch: 2 · Origin: Astra review E-S4 · Eridani task `bf5ad05d`

Astra's review (E-S4): the introduction still describes a read-only stage-B link and says start, steer and stop are unimplemented, while later sections and the code implement them. Rewrite the introduction to the current architecture, label historical stages, include the exact verification setup and the two credential paths (link key vs in-session MCP key).

Success criteria:

- The introduction describes starting, steering, stopping, verification, the judge and freestyle as implemented, and names the two credential paths; tests/test_drift_guards.py doc guards pass.

#### E-S5 — One real protocol path test: Eridani and the remotea server talking to each other

Repo: both · Status: backlog · Run batch: not a run · Origin: Astra review E-S5 · Eridani task `c9aa8f54`

Astra's review (E-S5): Python and TypeScript each test against fake peers. Add a disposable-database test that runs the real link implementations with a fake harness process through connect → spawn → prompt → turn → verify → judge → stop and reconnect, including duplicates and an older peer version; share versioned wire fixtures.

#### E-S6 — Readiness and freshness are visible everywhere a decision is made

Repo: jarvis · Status: backlog · Run batch: not a run · Origin: Astra review E-S6 · Eridani task `f8d32b61`

Astra's review (E-S6): host lease, transcript, run state, verification and workspace freshness are different facts. Show last-known timestamps and pending/uncertain control outcomes where they affect decisions, including Eri's summaries.

### F. Feature expansion from the review

#### S-Q1 — remotea doctor: one redacted readiness report

Repo: both · Status: backlog · Run batch: not a run · Origin: Astra review S-Q1 · Eridani task `0a0c9ff4`

Astra's review (S-Q1): service and build identity, listener reachability, collector mode and parity age, link backlog, hook installation, target launch readiness, verification dependencies (binary, deps, database), disk space and recovery warnings in one `remotea doctor` command, with Eridani's Settings → Connections showing the same.

#### S-Q2 — Recovery inventory with meaningful choices

Repo: remotea · Status: backlog · Run batch: not a run · Origin: Astra review S-Q2 · Eridani task `2eeb1e23`

Astra's review (S-Q2): list lost and resumable sessions with their task and worktree; offer reattach, resume a new process, or start again; say what cannot be recovered.

#### S-Q3 — Managed target templates validated before they are advertised

Repo: both · Status: backlog · Run batch: not a run · Origin: Astra review S-Q3 · Eridani task `7f5a5c5f`

Astra's review (S-Q3): define repository, harness launch, dependency preparation, verification environment, migrations and resource limits once; validate the whole template before advertising it. Folds into one-click setup.

#### S-Q4 — Worktree maintenance preview

Repo: both · Status: backlog · Run batch: not a run · Origin: Astra review S-Q4 · Eridani task `7be3f225`

Astra's review (S-Q4): show merged, unpushed, dirty, active and reclaimable worktrees with an explicit review step before removal; builds on the unmerged-work report.

#### S-Q5 — Upgrade path that drains orchestration without dropping PTYs

Repo: remotea · Status: backlog · Run batch: not a run · Origin: Astra review S-Q5 · Eridani task `7a9761ad`

Astra's review (S-Q5): separate long-lived PTY ownership from the restartable link and collector so a server upgrade does not end every session. Architectural.

#### A-Q1+E-Q2 — Open task/run in Eridani from a terminal, and open the terminal from a run

Repo: both · Status: backlog · Run batch: not a run · Origin: Astra review A-Q1, E-Q2 · Eridani task `03ff84bf`

Astra's review (A-Q1, E-Q2): the server already carries the Eridani binding; show it in the desktop with an Open in Eridani action, and add Open terminal in remotea on the run panel with a fallback when the desktop is not connected.

#### A-Q2 — Group spawned agents by project or task, not one shared Eridani workspace

Repo: remotea · Status: backlog · Run batch: not a run · Origin: Astra review A-Q2 · Eridani task `0ec167e4`

Astra's review (A-Q2): smaller accidental-close scope and easier navigation; keep stable group identity and device-local placement.

#### A-Q3+A-Q4+A-Q5 — Desktop pending-changes summary, host attention overview and unified shortcut language

Repo: remotea · Status: backlog · Run batch: not a run · Origin: Astra review A-Q3, A-Q4, A-Q5 · Eridani task `1149e7e3`

Astra's review (A-Q3, A-Q4, A-Q5): show rename pending / hide awaiting server / session ended elsewhere / resumed after restart; a host overview of active agents, blocked questions, disconnected sessions, verification failures; consistent attach/hide/close/interrupt/kill/resume wording with the affected host and task.

#### E-Q1 — Guided target setup and readiness page

Repo: jarvis · Status: backlog · Run batch: not a run · Origin: Astra review E-Q1 · Eridani task `0a1274a7`

Astra's review (E-Q1): connect a host, pick a repository, see enabled harnesses and checks, resolve setup errors, run a safe readiness probe before starting work. Folds into one-click setup.

#### E-Q3 — A review packet beside Approve

Repo: jarvis · Status: backlog · Run batch: not a run · Origin: Astra review E-Q3 · Eridani task `68487629`

Astra's review (E-Q3): frozen criteria, changed files or PR links, test evidence with stale/incomplete markers, remaining questions, side by side with Approve; keep human approval distinct from independently established completion.

#### E-Q5+E-Q6 — Attention-first run dashboard and recipient confirmation in Eri

Repo: jarvis · Status: backlog · Run batch: not a run · Origin: Astra review E-Q5, E-Q6 · Eridani task `2b45a858`

Astra's review (E-Q5, E-Q6): group runs by the next human action (answer, review plan, repair host, inspect failed check, approve, resolve unmerged); when several runs are plausible, Eri names the task, host and branch it is about to affect.

## Already-open roadmap items this PRD depends on or absorbs

- `6e753e0f` Judge evidence: never ask the agent for proof it cannot supply (partly done by PR #87's verify_missing rule; the PR-record part remains).
- `ba56714a` Plan floor vs interview answers: show criterion conflicts (the plan-review mechanic exists; conflicts display remains).
- `5a4e1290` Run hardening: PRD note read-only to agents while its task has a live run.
- `3bc2803c` remotea runs: give spawned agents the Eridani MCP tools automatically and show why a run needs you.
- `bbdb6440` One-click setup Eridani ↔ remotea (absorbs S-Q3, E-Q1 and the host readiness checks).
- `ec4d8a54` Run cards: a tiny-model one-line summary of what the agent is doing.
- `7f9da9f3` Question answering and run control for Codex and OpenCode.
- `0ab99d68` remotea for Brainforge members: multi-user design.
- `71f97d98` Swarm mode (Astra's E-Q4 is the same idea; build it on the batches above and the worktree writer policy S-S4).
- `a4d7d060` Explore when Eridani sends finish-up messages to an agent (document the triggers; show them on the timeline).
- `fe21bb75` One task card; `b14920b4`, `ce2e0e8d`, `b8d2e28e` search and filter bugs (product, scheduled alongside).

## Run order

| Batch | Tasks | Repo | Notes |
|---|---|---|---|
| host | S-N3 | desktop config | Davin: verify entry with absolute uv path or scripts/verify.sh, isolated test DB, timeout 1800; add a remotea target; restart remotea.service. Needed before batches 3–5 can be verified. |
| 1 | E-N1+E-N3, E-N2, E-N4+S-S1 (Eridani side) | jarvis | Different files: Remotea.tsx/RunPanel.tsx; agent_control send paths + api.ts; agent_control targets + preflight. |
| 2 | E-N5, E-S1, E-S2, E-S4 | jarvis | After batch 1 merges (E-N5 touches the drawer). |
| 3 | S-N1, S-N2, S-N5 | remotea server | Needs the remotea target on the host. dispatch.ts; journal/link; server.ts/cli.ts. |
| 4 | S-N6, S-S1, S-S5, S-S6 | remotea server | install.sh/package; eridani-runtime caps; build identity; tests. |
| 5 | A-N1, A-N2 | remotea desktop | Electron main/renderer; Windows smoke pass after. |
| later | S-N4, S-S2, S-S3, S-S4, A-S*, E-S3, E-S5, E-S6, F.* | both | Before unattended verification and broader use. |

## Validation

Each task's success criteria name its tests; every PR runs CI (ruff selection, pytest with a throwaway Postgres, alembic upgrade head and check, web tsc and vitest, remote-server typecheck and vitest). After batches 1 and 3: replay Astra's two reproductions (out-of-order run responses; steer during the Escape and Enter delays) as regression tests, then one live run end to end with a verify result recorded.
