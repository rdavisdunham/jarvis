# Eridani verification plan — September 23, 2026

> Archived October 2, 2026. Historical context and evidence; status and
> instructions below describe that period. Use the [active TODO](../../TODO.md)
> and [web v1 PRD](../../ERIDANI_WEB_V1_PRD.md) for current priorities.

This is the current testing order. [TODO.md](TODO_HISTORY.md) remains the work ledger;
[APP_FUNCTIONALITY.md](../../APP_FUNCTIONALITY.md) maps the implemented features.
Unchecked acceptance items below need recorded evidence, even when a related unit test passes.

## On-demand automation update

The current [eval guide](../../../evals/app/README.md) supersedes the older runner counts
and $2 default below. The CLI now selects six types, deduplicates shared suites,
isolates worker processes/databases, supports resume and reports exact evidence
coverage. No schedule was added.

All 1,001 cases are represented in the plan. Ninety-eight have complete acceptance
bindings and 224 have component bindings (these counts overlap). Nine hundred
still need complete individual assertions; three explicitly need physical-device
evidence. The report keeps those gaps in the denominator. There are 245 execution
jobs for the currently available full-catalog run, including 65 paid trials.
This is not 1,001 automated passing tests.

Use a **$10 maximum for the entire campaign**, including Luna, pipelines,
embeddings and judges (judge subcap $1). Gemini is paused. Retain all uncertain
charges across interruptions and subtract previous expenditure before starting a
replacement campaign. App failures are recorded during this batch, not repaired.

## Independent semantic grading — September 23

The balanced half-suite selected 501 cases across all 40 features and executed all
153 available jobs. Selection and raw evidence are saved under
artifacts/app-evals/half-20260923-selection.json and half-20260923-sol-graded/.
Known failures were forced into the sample, so it is a regression sample rather
than an unbiased estimate. Estimated application API usage was $0.055128104.

Use --judge external to defer built-in semantic judgments to independent review.
This reserves zero judge API budget and saves exact criteria and evidence next to
the attempt's state and tool traces. GPT-6 Sol's grader.json and grader.md are
separate from the unmodified automated report. An external review may identify
incomplete evidence even where declared acceptance bindings pass; preserve both
outcomes and explain the disagreement. Codex grader usage is outside the app API
cost ledger. No scheduled runs are configured.

## Langfuse export — September 23

Completed synthetic campaigns can now be exported using
python -m scripts.app_eval.runner langfuse RUN_DIRECTORY. Use --dry-run first,
and --verify for readback without further writes. See [LANGFUSE.md](../../LANGFUSE.md).
This is explicit post-run export; CI, reports and ordinary tests do not send data.
It preserves missing acceptance coverage and separate code/Sol verdicts.
No model calls or cloud judges are triggered by the exporter itself.

## Where we are

The core productivity app is implemented: configurable records and relationships,
task views, notes/lists, hybrid search, learning and dream passes, voice, durable
parallel work with clarification continuations, action history, notifications,
Google/Linear connections, sharing and scoped bot API/MCP access.

The next priority is reliability across those features. The current findings
include timestamp-equivalent Revert conflicts, intermittent note clearing,
clarification continuity, memory contradictions and embedding-failure persistence,
incomplete recommendation extraction, and task resolution missing flexible project
relationships. See [the dated findings](../../../evals/app/FINDINGS.md) for reproduced,
intermittent, and component-only evidence; later single passes do not close them.

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
  recovered tool error. Verify Luna through the real queue; Gemini comparisons are paused.
- [ ] Re-run the expanded command/recovery components; any remaining failure stays visible.
- [ ] Re-run the expanded 41 backend probes with Luna and three independent repetitions where
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

Default to offline checks. Paid runs use Luna only and a shared $10 campaign
ceiling. The twenty-case historical sample cost $0.02352348 on Luna and $0.497667
on Gemini; it does not price the full mixed catalog. Real GPT-Live audio, embedding
retrieval quality and broader pipeline trials need their own measured evidence.
The current harness records paid text/pipeline/judge costs separately from app
usage. See its guide for STOP/resume, uncertainty and connected-resource handling.

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
localhost eval PostgreSQL instance (see [setup](../../../evals/app/README.md)):

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

## September 22 current full-run evidence

The current available full-catalog run is `artifacts/app-evals/full-20260922-localpg-luna/`: 245 jobs, 233 passed, eight failed, four blocked after a retained provider-error retry. Its 1,001 acceptance classifications are 92 passed, six failed, 146 partial, 755 blocked and two component-failed. A separate corrected-oracle trial at `artifacts/app-evals/due-time-oracle-20260922/` passed the offset-aware 16:00 Chicago case that the original raw-string grader falsely failed. See [findings](../../../evals/app/FINDINGS.md) for failures, costs and limits. Eval self-tests pass (64, with two opt-in skips); frontend tests pass (136, one paused-Realtime skip) and its production build succeeds.

Next product work remains EVAL-001 through EVAL-006 plus EVAL-007's missed After Yang extraction. The three connected smoke probes need dedicated test credentials/resources; actual phone, real GPT-Live/audio, and full restore/PITR evidence remain separate gates. No result in this run closes those gates.
