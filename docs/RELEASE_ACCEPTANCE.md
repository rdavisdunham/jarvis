# Web v1 acceptance and pilot

Status: **partial machine-run acceptance; physical checks and pilot deferred**.
October 27 findings are recorded in [the active roadmap](TODO.md#current-finishing-pass--october-27). Automated evidence is in [Batch D verification](BATCH_D_VALIDATION.md).
A browser viewport test does not replace a folded phone, real microphone, locked
notification, real provider sync or a new person's experience.

## Machine-run finishing evidence — October 27

PR #40 merged as 2e3655d after backend, frontend/browser and security CI passed;
Railway reported successful production deployment. The policy remains opt-in.

The finishing branch passed 1,065 backend tests (one skip), followed by 15 focused
checks for the final occurrence-navigation correction. Frontend: 243 passed,
one skip, type-check and production build passed. Offline eval harness: 95 passed,
two skips; catalog/coverage checks passed with 215/1,051 fully bound cases.
All 14 isolated browser flows passed, including synthetic Google self-RSVP and
Revert. Fresh forward migrations and migration/model agreement passed.
These results do not claim a full paid eval run or physical/provider acceptance.

The matched latency trial results, costs and limitations are in
[lean-policy validation](ERI_LEAN_POLICY_VALIDATION.md); usability behavior and
remaining real-phone/Google checks are in
[usability validation](USABILITY_FOLLOWUP_VALIDATION.md).

## Before the pilot

Record a result for each row using the template below. Use dedicated disposable
provider items; retain IDs until both local and remote cleanup are verified.

| Check | Expected result | Status |
| --- | --- | --- |
| Deploy / cloud | Same intended revision on API and worker; current schema; fresh DB heartbeat | Public health and signed-in app pass Oct 27; GitHub reports production 2e3655d success; individual service/config/SQL blocked by Railway auth |
| Voice clarification | Incomplete to-do asks once; partial answer remains one waiting card; complete answer resolves it | Pending |
| Independent / dependent requests | Call Alex + buy milk can run independently; “make that call tomorrow” updates the correct request | Typed task + note + task correction passed Oct 27; voice/restart pending |
| Optional offer / unrelated request | “Anything else?” never captures or drops a separate next request | Pending |
| Voice lifecycle | Eri / Hey Eri, explicit and natural goodbye, 30-second silence, no premature timeout while work runs | Pending |
| Connection recovery | Wi-Fi/cellular change, offline/reconnect, close/reopen; accepted work persists; Send/Discard is explicit | Pending |
| Physical UX | Pixel folded/unfolded and desktop; Back, keyboard, scroll, focus, controls and screen reader | Pending |
| Activity | Compact cards stay in conversation order; Edit/Revert preserves later unrelated edits; stale history clears safely | Typed receipts, detail readback and date Revert pass Oct 27; note-indexing creation-Revert repaired and regression-tested; finishing deployment tracked in GitHub |
| Quick capture | Houston list with Hayes, packing, moped and work sections; correct edits/reorder/promotion without mandatory organization | Pending |
| Notes/search/learning | Film reuse, misfiled results, alias corrections, source-backed memory and separate rule interview | Real dream outcomes inspected Oct 27; quality gaps in TODO; broader flows pending |
| Memory review | No microphone opens by itself; questions offered only in the appropriate conversation | Pending |
| Google / Linear | Local and remote edits agree after readback; source tags, recurrence/all-day boundaries, retry/conflict and local notes correct | Pending |
| Notifications | Locked phone receives correct item; snooze/quiet hours/urgent settings and morning summary; no Eri completion spam | Pending |
| New user | Invited person completes or skips/resumes setup on a phone without developer explanation; own Personal data stays separate | Pending |
| Operations | Pending synthetic work survives cloud restart with home PC off; backup/restore/PITR evidence; Google consent readiness | Pending |

For every check, record:

```text
Date/time and timezone:
App commit / API deployment / worker deployment:
Device, OS, browser, folded/unfolded, connection:
Scenario and exact steps:
Request / receipt / local record / provider record IDs:
Expected:
Observed (including remote readback where relevant):
Evidence path or recording:
Result: pass / fail / blocked / not run
Failure severity and follow-up issue:
Cleanup verified:
```

## Seven-day pilot

Start only when the release gates are accepted. Complete at least seven calendar
days and 50 successful planner interactions; retain failed attempts in the denominator.
A retried action is one user interaction with multiple attempts, not several successes.

Use `docs/evals/2026-10-03-batch-d/pilot.csv` as the empty log template. Do not commit
private conversation content or provider credentials. Record an opaque ID instead.
Track task/quick-list capture and edits, completed/reopened work, notes and search,
calendar/source operations, clarification, Revert and notification handling.

Review daily: lost/duplicate/wrong-record effects, corrections, orphan waiting cards,
voice recovery/comfort, end-to-end visible/spoken latency and per-feature spend.
Report median/p90 only with sample counts; include failures and fallback costs.
Weekly/monthly cost projections must state observed days and interaction volume.

Release sign-off requires no unresolved critical/high access, data-loss or incorrect
mutation failures, accepted device/provider checks, seven days of evidence and at
least 50 successful interactions. Android work can then reuse these verified contracts.
