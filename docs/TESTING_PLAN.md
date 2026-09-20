# Eridani verification plan — September 20, 2026

This is the current testing order. [TODO.md](TODO.md) remains the work ledger;
[APP_FUNCTIONALITY.md](APP_FUNCTIONALITY.md) maps the implemented features.
Unchecked acceptance items below need recorded evidence, even when a related unit test passes.

## Where we are

The core productivity app is implemented: configurable records and relationships,
task views, notes/lists, hybrid search, learning and dream passes, voice, durable
parallel work with clarification continuations, action history, notifications,
Google/Linear connections, sharing and scoped bot API/MCP access.

The next priority is reliability across those features, rather than another
large expansion. Two known eval failures remain: timestamp-equivalent Revert
conflicts (EVAL-001) and clearing task notes/recovered-error queue status
(EVAL-002). See [the findings](../evals/app/FINDINGS.md).

The catalog contains 1,001 scenarios across 40 feature areas. That is authored
coverage, not 1,001 passed tests. The additional component baseline was 87/88.
The first paid backend sample was 19/20 complete outcomes for each model; one
repeat and 20 cases cannot establish overall model quality. Device, provider
consent and end-to-end recovery have separate acceptance requirements.

Cost tracking is on; budget enforcement remains off in development. Settings
shows rolling seven- and thirty-day feature estimates. Missing historical usage
cannot be recovered. Isolated eval spending is recorded separately.

## Batch 1 — verify the two reported fixes

Implementation and regression coverage:
- [x] Google event PATCH explicitly removes the opposite start/end representation.
  All-day-to-timed, timed-to-all-day and multi-day rescheduling keep valid boundaries.
- [x] Retry after a lost provider response recovers the same event without another
  write; metadata, fresh-edit-token and conflict protections remain covered.
- [x] Locally owned planning events can convert and publish both directions.
- [x] Voice status includes durable queued/running work and pending provider sync,
  including resumed clarifications and simultaneous requests.
- [x] Browser idle shutdown pauses during work, then grants a fresh 30 seconds.
  Questions/failures return to listening; goodbye still ends voice while accepted
  work survives independently. Unrelated voice sessions do not hold this one open.

Real-account acceptance, approximately 15–20 minutes:
- [ ] On a disposable Google calendar event named “Eri calendar test,” ask:
  “Make it all day tomorrow,” then “Move it to the following day,” then
  “Make it run from 9 to 10 a.m. instead.” Inspect both apps after each change.
  Dates, local timezone, duration, description and location must agree.
- [ ] Change it back to all-day, then extend it across three days. The final included
  day should match in both apps; Google's exclusive end must not add an extra day.
- [ ] Repeat with an Eridani-owned event published to Google. Confirm one visible
  event, no duplicate, and the confirmed sync state.
- [ ] With a disposable recurring event, specify one occurrence versus the series
  and confirm only the requested scope changes.
- [ ] Start slow/multiple queued work and remain silent for more than 30 seconds.
  Voice must stay active while work is queued/running/syncing. After results finish,
  it should allow a full 30 seconds for a reply before releasing the microphone.
- [ ] Interrupt with a second independent request, then an edit to the first.
  Both requests must finish once, and the edit must apply to the correct item.
- [ ] Answer a clarification; the original card must resume and finish without an
  orphan “Waiting for you.” A pending question must not keep the microphone on forever.
- [ ] Say “Goodbye, Eri” during queued work; confirm voice closes and the saved work
  finishes. Wake a fresh session with “Eri” and “Hey Eri.”

Pass gate: no invalid event boundaries, duplicate writes, lost accepted work,
premature idle shutdown or stuck microphone. Record actual durations and request
IDs. A dropped network connection is distinct from an idle timeout.

## Batch 2 — resolve known correctness failures and widen model evidence

- [ ] Fix EVAL-001 with UTC and America/Chicago database sessions, including DST
  boundaries, before re-running receipt/Revert cases. Preserve stale-change guards.
- [ ] Fix EVAL-002: explicit empty-note semantics and honest final status after a
  recovered tool error. Verify both backend models through the real queue.
- [ ] Re-run all 88 component checks; any remaining failure stays visible.
- [ ] Re-run the 20 paired backend probes with three independent repetitions where
  the remaining dollar allowance permits. Compare saved state, receipts, unresolved
  errors, latency and tokens; retain cap interruptions and provider failures.
- [ ] Add reviewed live-pipeline adapters/gold labels for memory extraction and
  memory dream, task routing and rule dream, note filing/extraction, and semantic
  retrieval/alias learning. Start with 5–10 high-risk cases per pipeline, then
  expand toward its full 25-case feature set.
- [ ] Include correction, abstention, quoted/negative instructions, duplicates,
  conflicting spellings, stale embeddings, workspace isolation and replay cases.
  The current lexical-only paid probes do not establish embedding retrieval quality.

Use Rowan's versioned corpus and disposable per-trial PostgreSQL clones. Keep
exploratory data separate; promote regression fixtures through explicit review.
Record prompt/model/catalog versions and ground truth, rather than grading an
assistant's “Done” message.

Default to offline checks. For paid runs, aim below $2/model and never exceed the
user's $10/model allowance; subtract prior runs rather than treating each command
as a new allowance. The previous sample spent $0.02352348 on Luna and $0.497667 on
Gemini. Request caps and the durable spend journal remain mandatory. This fix
batch makes no paid model calls.

Pass gate: all deterministic checks pass; permission, duplicate-write and wrong-
target failures are zero. Report model success and recovery rates separately;
do not conceal a failing critical case behind a high average.

## Batch 3 — one focused physical-device and connected-account round

Use the Pixel Fold folded/unfolded plus desktop. Allow about 45–60 minutes:
- [ ] Tasks: create, assign custom fields, drag between statuses, edit an individual
  detail, complete/reopen, then Revert; verify intended inheritance and stale-edit protection.
- [ ] Notes/lists: save movie recommendations, view Movies, inspect source links,
  correct filing and reprocess. The correction must survive without duplicates.
- [ ] Search: use an alias such as “pest-control company,” inspect exact and possible
  misfiled matches, correct a wrong identity and inspect learned aliases.
- [ ] Learning: enter synthetic memory conflicts and categorization examples,
  run each dream pass explicitly in a test workspace, review questions/proposals,
  and confirm accepted corrections without mixing routing rules with personal facts.
- [ ] Navigation: Back closes detail/chat before leaving the page; retain filters,
  keyboard usability, scroll position and unsaved drafts across folded/unfolded widths.
- [ ] Voice: interruption, natural goodbye, explicit goodbye, both wake phrases,
  clarification, long-running work, a brief Wi-Fi/mobile-data gap and reconnect.
- [ ] Notifications: a real locked-phone alert opens the intended task; verify
  snooze, quiet hours and a daily summary. Successful agent actions stay in Activity.
- [ ] Google and Linear: disposable create/edit, provider-side edit, sync conflict,
  reconnect and selected-calendar boundaries; compare both applications.
- [ ] Sharing/bots: separate-account invitation and shared-workspace invitation,
  viewer/editor boundaries, membership removal, scoped API/MCP success/denial and
  credential revocation. Run these with distinct test identities.

Pass gate: complete each workflow without data loss, access leakage, duplicate
effects or forced reload. Log minor visual defects separately from blocking failures.

## Batch 4 — cloud recovery, then the usage pilot

- [ ] Verify access with the host PC off, and one controlled API/worker restart
  while synthetic work is pending. Accepted work must recover exactly once.
- [ ] Recheck current Railway PITR retention/archives and restore a selected timestamp
  into an isolated target. Earlier PITR restore evidence exists; do not restore over production.
- [ ] When R2 credentials are available, activate the prepared job and verify actual
  upload, download, decryption, isolated restore and scheduled execution.
- [ ] Finish automated restore drills and failed/stale-backup alerts; test failures.
- [ ] Check Google production consent/onboarding before inviting a wider audience.
- [ ] After device/operations gates, run the seven-day pilot with at least 50
  successful task/reminder interactions. Capture failures, recovery, response time,
  useful/incorrect learning and daily per-feature usage. Keep partial periods labeled.
- [ ] After a full month, review actual thirty-day feature costs. Separate model
  estimates, eval charges and hosting bills.

Android follows stable backend/client contracts and this acceptance round.
Deferred R2 credentials do not block the current bug fixes or device testing;
keep the independent-backup gap explicit.

## Execution and evidence

Run from Linux/WSL with the locked repository dependencies and the dedicated
localhost eval PostgreSQL instance (see [setup](../evals/app/README.md)):

~~~sh
.venv/bin/python -m scripts.app_eval.runner validate
.venv/bin/python -m scripts.app_eval.runner regressions --scope all
.venv/bin/python -m scripts.app_eval.runner contracts
~~~

The component command currently reports the known EVAL-001 failure; do not relabel
that result as a clean run. CI also validates migrations, frontend production build,
catalog integrity and six synthetic browser suites. Paid/provider/device protocols
are explicitly separate.

For each manual result, record: commit, date, device/browser, test identity/workspace,
steps, expected versus observed state, request/receipt IDs, duration and pass/fail.
Use screenshots when useful; omit credentials and unrelated personal content.
A failure gets a minimal regression case and a linked TODO item.

Current change evidence:
- Calendar and Live focused suites passed (including the reproduced failures).
- Frontend: 136 passed, one intentionally skipped paused-Realtime test; build passed.
- Eval harness: 29 passed, two opt-in integration checks skipped; catalog valid.
- Full offline backend: 694 passed, one intentional skip; Python correctness passed.
- Deployment and CI results are attached to this change's GitHub commit/run.
- No physical microphone or real Google mutation was claimed from mock tests.

Local full-backend evidence: artifacts/app-evals/calendar-voice-fix-20260920/.
Google nested PATCH semantics:
https://developers.google.com/workspace/calendar/api/guides/performance#patch
