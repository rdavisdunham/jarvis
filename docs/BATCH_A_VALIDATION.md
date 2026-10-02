# Batch A implementation and verification

October 2, 2026. Local branch: `codex/batch-a-reliability`.
Prepared for review; these changes are **not deployed**. The PR also includes the
previously requested roadmap consolidation and documentation archive. No production
records or infrastructure were changed by this batch, and no paid model calls were made.

## Implemented

- Closing a Live session no longer guesses new work from action verbs. Unclaimed
  speech becomes an encrypted, 24-hour **Unsent voice draft** in chat, with editable
  text and Send/Discard. Short answers such as “yes”, “9” and “ABC” are retained.
  Standalone farewells are omitted. Explicit Live delegations still enter the queue;
  already accepted requests survive voice closure.
- Drafts retain their original conversation and are scoped to owner/workspace,
  account and device. Send is transactional and idempotent; Send/Discard races have
  one durable outcome. Discard removes the captured text. Expiry, rollback and reload
  are covered. Sending after a lost response returns the original disposition.
- PostgreSQL NOTIFY wakes the existing durable dispatcher only after commit.
  LISTEN uses the singleton worker lease; losing that connection stops dispatch,
  and the normal service restart reacquires ownership and rescans the outbox.
  Startup and periodic scans remain authoritative when hints are missed.
- Maintenance scans and notification delivery run separately from dispatch. No API
  write agent or second task execution path was introduced. Keyset pagination reaches
  eligible work beyond the first 1,000 blocked outbox rows. DBOS workflow identities,
  command receipts, resource ordering and budget/cancellation checks remain in place.
- Content-free timing logs cover queue acceptance/dispatch, runner start, context,
  model, tool discovery/execution, voice settlement, Live append and UI delivery/ack.
  `scripts/summarize_latency.py` summarizes stage event counts, errors, median and p95
  from logs without database access or provider calls. Monotonic clocks measure local
  spans; UTC correlates processes. Stages labeled `precommit` are explicitly provisional;
  an append acknowledgement is not proof of audible speech or browser paint.
- Two exact release-core cases now have acceptance bindings: request-ID mismatch
  (`queue.08`) and unrelated voice work while a question is pending
  (`clarifications.10`). The coverage ratchet increased from 98 to 100 fully bound
  cases out of 1,001. This is test binding coverage, not a claim that 1,001 cases pass.

## Read-only cloud verification

- PR #15 is merged at `c1808e6517140d7acfdcd6609c0aa01dbdba1eea`; its three CI jobs passed.
- Both Railway app services are successful/running at
  `440c9f7e4f370551fc98030581c30491e9b08e15`, which includes that merge.
  `https://app.eridani.app/health/ready` returned `{"status":"ready"}`.
- The active Postgres16 deployment uses
  `ghcr.io/railwayapp-templates/postgres-ssl:16.15`. Direct SQL version/worker-heartbeat
  readback was not possible: no public SQL URL is configured, and Railway SSH has no
  available key. This verifies the deployed image, not `SHOW server_version`.
- PG18's service is gone. Its detached `postgres-volume` still exists, as do the active
  `postgres16-volume` and `Postgres-PITR` bucket. The local whole-project declaration
  removes the nonexistent service, retains both volumes and pins the observed image.
  **No Railway plan/apply was run.**
- Web/worker deployment triggers both have `checkSuites: true` for `main`.
  GitHub ruleset `24354023` requires all three checks with strict up-to-date checks,
  and blocks force-pushes/deletion. It does **not** require PRs.
- After the owner disabled PostgreSQL Serverless, a fresh Railway readback confirmed
  `sleepApplication: false` on all three successful deployments: Postgres16, Eridani_Web
  and Eridani_Worker. This audit did not change cloud settings.

## Local verification

All database tests used disposable databases under **local PostgreSQL 16.15**, with
`JARVIS_ENV_FILE=` so personal `.env` credentials could not be loaded.

- Full backend suite: **797 passed, 1 skipped**; report `artifacts/batch-a/backend.xml`.
- After the final short-answer/timing refinements: **104 affected tests passed**.
- Frontend unit suite: **166 passed, 1 skipped**. TypeScript/Vite production build passed.
- Mobile chat acceptance: reload an unsent draft, edit and send it once, discard another;
  also existing chronological action cards, silent navigation, Edit/Revert and no action
  completion notifications. Synthetic provider availability only; actual API/UI paths.
- All six desktop/mobile browser suites passed: custom planner, shell navigation,
  chat activity, semantic search, note lists and usage settings.
- Python correctness checks, eval catalog validation and coverage ratchet passed.
- Wakeup tests include commit/rollback behavior, startup scan, committed wake with a
  60-second fallback, blocked-page fairness, DBOS enqueue response loss and lost lease.
  Recovery tests include concurrent Send/Discard, rollback, retries, expiry and isolation.

## Still required for Batch A acceptance

- Deploy the reviewed change through CI, then verify matching web/worker revisions.
- Actual phone voice: incomplete capture, clarification, unrelated next request, late
  answer, result announced while another answer arrives, goodbye and reconnect.
  Controller tests do not prove microphone/provider behavior.
- Complete the end-to-end timing baseline with real `/work` and Live sessions (at least
  30 successful observations per path, reporting failures too). Add browser paint and
  first-audible-response observations; current stage logs do not provide these.
- Measure turn-protection and tool-discovery delays before changing them. Voice
  settlement and the 2.5-second conversational quiet gate remain unchanged.
- Verify worker heartbeat and actual SQL server version through a private connection;
  decide whether PR-only rules are desired. Database Serverless is now verified off.
- No claim of a measured speedup, real-device pass, full eval pass or production release.
