# Conversational planner release validation

September 13, 2026. This release adds shared task/project projections, typed site
controls, compact organization and verified local scheduling. Schema remains
0011_productivity_graph; no record migration or new service is introduced.

## Completed checks

- Backend: 446 passed, one optional test skipped. The paused Realtime-only
  backend module is excluded. This includes planner optimality/constraints,
  atomic failure, changed availability/revisions, owner isolation, short-reference
  integrity/expiry and replay after proposal cleanup, note preservation/reference
  errors, UI acknowledgements and Live clarification/recovery.
- Frontend: 75 passed, one retained Realtime case skipped. Active voice tests now
  emit Live's required session.started event; the earlier generic fixtures
  incorrectly assumed the old Realtime default. Retained Realtime branches are
  paused; this release makes no Realtime acceptance claim.
- TypeScript and production build pass. Ruff passes for all application/test
  Python and the changed evaluation/validation scripts. A broad scripts check
  also identified two existing lint findings in untouched scripts/smoke_live.py;
  those are outside this release. The build retains its existing large-bundle
  advisory (about 274 KB gzip JavaScript); controller extraction/code splitting
  is recorded as a follow-up.
- Real rendered planner acceptance: 85 authenticated, acknowledged actions using
  CopilotKit's registered handlers and actual domain commands, in an isolated
  PostgreSQL database. Desktop 1440×1000 and mobile 390×844. Covers task/project
  boards/timelines, status changes, exact sparse note saves, draft reads/patches,
  invalid-patch atomicity, dirty-navigation refusal, revision-conflict retention,
  all organization editors, bulk actions, memory correction, device preferences,
  filtering/search reset, quick capture, mobile chat closure and overflow.
- Google/Linear browser acceptance passes: Google consent and return to
  Integrations, details, selected calendars, mobile calendar views, 30-second
  scroll stability, local appointment publication/CRUD, retry after a lost
  response, task work blocks, Linear import/publication/conflict review,
  disconnect and unlink. Provider I/O uses synthetic fixtures; the browser,
  API, domain, durable-job and revision behavior are real.
- Real GPT-Live acceptance uses generated speech with actual OpenAI media and
  backend delegation in a disposable database. Exactly one synthetic task was
  created, both captions appeared, spoken confirmation and incoming audio were
  received, voice remained visible outside the closed mobile chat, quiet timeout
  closed the session, a new chat restarted Live, and all peers closed. No page,
  API or provider errors were recorded. No owner microphone or personal task
  data was used. Deterministic tests separately cover clarification, new speech
  during delegation, interruption, failure recovery and saved receipts.
- Paired backend evaluation: eight fresh cases × three repeats × two models,
  all 48 trials / 66 turns independently reviewed by a Codex subagent. A separate
  six-trial diagnostic verifies the short-reference fix. Original scores,
  failure cases, source snapshots, usage, grader corrections and synthetic-service
  limits remain visible in [RELIABILITY_HELDOUT_RESULTS.md](RELIABILITY_HELDOUT_RESULTS.md).

## Deployed verification

API, worker and PostgreSQL are healthy; the backup service remains running.
HTTPS serves index-C5suLihr.js and index-CG2uIg69.css. All 52 Python application
module hashes match the running API container.

Authenticated production browser checks passed for Today, board/timeline
switching, Live-only options and mobile overflow with no JavaScript errors.
No records were edited by that production browser smoke check.

Pre/post deployment hashes match across fifteen checked canonical/ledger tables,
including all 55 tasks, two projects, two spaces, two actors, eight schedules,
eleven memory assertions and 322 historical reservations. Empty goal/note/block
and note-link tables are unchanged. Cost tracking remains disabled. These
checks do not claim that normal sync/job/heartbeat tables remain byte-identical.

The main held-out source snapshot predates the short-reference fix and final
Notes/OAuth view polish. The separate scheduling diagnostic freezes those changes.
A final Ruff fix removes one extra blank line in note_schema.py; it changes no
runtime behavior. The final active voice-test fixture update likewise changes
tests only. The evaluated snapshots remain immutable.

## Reproduce

From the repository in WSL with the development dependencies and PostgreSQL
available:

```sh
.venv/bin/pytest -q --ignore=tests/test_voice.py
cd apps/web
npm test
npm run build
cd ../..
.venv/bin/python scripts/validate_planner.py
.venv/bin/python scripts/validate_google.py
```

Each browser validation script creates and removes its own database. The Google
and Linear acceptance uses synthetic credentials and provider fixtures.

The explicit real-provider check is:

```sh
.venv/bin/python scripts/validate_live.py
```

It uses the configured OpenAI key and the generated
tests/fixtures/live-create-task.wav phrase. It incurs API usage, routes writes
only to its disposable database, uses no microphone, and removes that database
on exit. It does not run Realtime. Browser evidence is written under .runtime/.

Physical-phone microphone behavior, Android background/wake behavior and recovery
on the owner's actual network remain separate real-use checks. They are not
established by simulated audio or backend evals. The seven-day pilot remains
after the agreed expansion and device/operation checks.
