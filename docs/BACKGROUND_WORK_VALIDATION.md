# Background-work batch validation — September 16, 2026

Released to production September 16, 2026. Implementation commit: `e5d9234`.

## Direct-execution correction — September 16

Final broad check: **559 backend tests passed**, one existing skip; **113 frontend
tests passed**, one intentionally paused Realtime test skipped. Production build and
Ruff passed. An additional focused regression verifies that read-only backend results
still reach Live even when they do not create action cards.

- Removed the exact-quote intake gate after owner screenshots showed natural speech
  being rejected before the task agent ran. Separate requests now execute directly.
- New regressions cover concurrent creation, pending/completed follow-ups, failed
  predecessors, record reservations, unknown scopes, account isolation, legacy
  intake retry, original-input preservation, past/future conversation isolation,
  voice_end, and follow-up-only reversal.
- Initial real-model probes exposed an unnecessary dependency and replay of earlier
  speech. Tightened the follow-up contract and changed earlier conversation to
  reference data; each run has only its own active user instruction.
- Both Luna and Gemini passed synthetic Alex/milk/early-correction flows, timed
  natural speech with repetitions, follow-up reversal, and backend voice_end.
  These checks used real providers with disposable local databases and synthetic
  records. No production tasks or connected-service writes were used.
- Six browser states at desktop/mobile sizes verified outcome-first cards, collapsed
  speech, Edit to the correct record, Revert of the due-date change, optional failure
  dismissal without retry, and chat. No browser exceptions or page-width overflow.
- Local ignored evidence: `.runtime/direct-work/`. Remaining owner microphone and
  production-connected-service acceptance still applies. Earlier failed requests
  are not automatically replayed; Retry uses direct execution while input is retained.

## Automated evidence

- **550 backend tests passed**, one pre-existing skip. Includes account/workspace
  isolation, role revocation, encrypted intake, duplicate delivery, late corrections,
  partial clarification, dependency waits, cancellation, safe revert conflicts,
  OAuth destinations, invitation privacy, and current integration reconciliation.
- **110 frontend tests passed**, one intentionally paused Realtime test skipped.
  Voice tests cover the 30-second timeout, contextual shutdown, continued speech,
  immediate microphone release, repeated close, reconnect and fresh session entry.
  Wake tests preserve the request following “Eri” / “Hey Eri”.
- Real worker-process recovery: killed the process after a task command committed
  but before its checkpoint, restarted DBOS, and verified exactly one task and one
  command receipt with a successful final result.
- Production TypeScript/Vite build and Ruff checks passed. Existing warnings are
  third-party FastAPI test-client deprecations, dependency annotation warnings,
  and the planner bundle size warning. The public entry is separately loaded.
- A disposable database upgraded from `0012_accounts` to `0013_agent_work` while
  preserving an existing task. A fresh migration also started the browser fixture.

## Provider checks

Both configured **GPT-5.6 Luna** and **Gemini 3.8 Flash** passed five synthetic
flows each: create two tasks without dates/projects, rename a specifically named
task, queue another create, cancel that queued request, and route goodbye.
No production records were used or changed. The real provider calls used the
configured keys, a disposable local database, and no memory retrieval or connected
calendar/Linear writes. The synthetic database was removed afterward.

The first Luna run asked an unnecessary clarification when a fresh edit also
referenced a known request. The intake contract now treats that as an ordered new
instruction, without overwriting old input. Both profiles passed the rerun.
This small release check is not a new comprehensive ranking of the models.

## Browser evidence

Thirteen states at 1440×1000 and 390×844: landing, privacy, terms, help, Activity,
inline task Edit, Revert, task list, saved views, mobile month calendar, and chat.
No application JavaScript errors or document-width overflow were recorded.
Edit opened the intended record; Revert archived the unchanged creation; Cancel
stopped the queued request. Personal private-chat controls were absent.

Screenshots/results are ignored local evidence in `.runtime/batch-verify/`.
The earlier full-site audit remains in [recommendations.md](../recommendations.md),
with its 76 screenshots and explicit production-login versus synthetic-app limits.

## Operational boundaries and follow-ups

- Owner reports basic phone voice is working well (September 16). This is
  owner-reported real-device feedback; wake words and the updated goodbye sequence
  were explicitly not tested.
- Desktop audio, phone/desktop wake recognition and updated goodbye, rapid requests,
  late corrections, browser backgrounding and production interruption/reconnect
  still need a real-device pass. Automated media mocks and provider text checks
  do not establish microphone performance.
- Production Google/Linear writes and a second person's Google invitation were
  not exercised in this batch. Backend contracts and synthetic account separation
  were tested. Google consent verification remains an owner-operated process.
- Existing local task/note/project/goal changes support guarded Revert; relationship
  rewrites and external calendar reversals explain that manual record review is
  required. An already-dispatched provider/browser operation may still finish.
- Native Railway PITR remains active. Independent R2 exports are not active: the
  endpoint/access-key/secret-key handoff is still missing. Bucket `eridani-backups`
  exists. Export/download/restore verification remains on the TODO.
- Public assets have an explicit Cloudflare deployment step; Railway app/worker
  deployment continues from GitHub main. No new paid queue or database service.

## Production rollout

- GitHub main: `e5d9234db68a8109b2194440b01a5342b0894638`.
- Railway API deployment `71f8d405-06ad-4f04-9209-1f3a737d118a`: SUCCESS.
- Railway worker deployment `d2417095-97a3-4abe-a068-563ff3f8f10b`: SUCCESS.
  Startup logs confirm the separate `jarvis-intake` and `jarvis-agent` queues.
- Cloudflare public version `8f83601b-9ca3-4615-b28a-c73a85d5905d` published.
- Apex, www, privacy, terms, help and app login all returned HTTPS 200 with no
  browser JavaScript errors. Landing Sign in links to the authenticated app.
- The production schema readiness endpoint returned 200/ready. OpenAPI exposes
  the new work routes; anonymous work access returns 401. The updated Google
  button successfully navigated to `accounts.google.com`.
- No production authentication bypass or private-record mutation was used for
  these public checks. The disposable browser fixture was stopped and removed.
