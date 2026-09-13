# Jarvis upgrade: implementation and operation

Implemented September 10–11, 2026. The owner delegated the design choices and confirmed that the host PC stays awake.

## Open Jarvis

Use **https://davispc.tail957c2.ts.net:9443** while connected to Tailscale. Pair each device with the code in `.runtime/pairing-code`. The code is local and is not committed to Git. Each paired device receives its own revocable, HTTP-only session.

The new deployment is the Docker Compose project **jarvis-next**, separate from the original stack. Its API listens on host loopback port 8765; PostgreSQL uses loopback port 54329. The existing Tailscale service on port 8443 remains available.

OpenAI authentication and model access are verified using `OPENAI_API_KEY` in the ignored `.env`. Text uses `gpt-5.4-mini`; foreground voice uses `gpt-realtime-2.1` with Marin. The API and worker were recreated after the owner supplied the key. After future environment changes, run `scripts/operations.ps1 start` to recreate services. Groq remains the text fallback when an OpenAI key is absent.

## Assistant identity and progress

The assistant is **Eridani**, or **Eri**. Her shared text/voice personality lives in
[personality.py](../apps/api/jarvis/personality.py): polished, warm, witty and
lightly playful, with useful assistance ahead of entertainment. The old
`JARVIS_SYSTEM_PROMPT` environment entry has been removed to prevent conflicting
identities. Jarvis remains the repository/infrastructure name.

[TODO.md](TODO.md) records completed work and the remaining roadmap, including
Google account sign-in already requested by the owner.

## Decisions made

- Python/FastAPI owns the domain API; PostgreSQL owns tasks, reminders, sources, sessions, receipts and usage. A separate DBOS worker performs durable work.
- React/TypeScript/Vite provides a responsive, installable web app at the same origin. Text, voice and buttons use the same validated commands.
- Task deadlines support a date with an optional local due time and IANA timezone. Date-only tasks remain supported; time changes do not create notifications. Reminders have explicit IANA time zones and resolved instants. The default is America/Chicago at 10 AM when only a date is supplied.
- Repeated routines create independent task occurrences. Completing one occurrence does not cancel the series. All reminders are task alerts. Repeated alerts on a single task stop when it is completed; routine templates generate independently completable tasks.
- History and conservative memory learning start enabled. Private sessions do not retain transcripts. Explicitly requested tasks and saved memories still persist. Jarvis does not record raw audio.
- The default voice candidate is GPT-Realtime-2.1. The server controls response creation, silence, waiting, interruption and tool execution. Audible confirmations are requested only after commands commit.
- Internal cost recording and budget enforcement are disabled during development. The owner monitors OpenAI Usage. The optional accounting system retains its historical ledger and reservations; reconcile those before re-enabling it.
- Canonical facts and embeddings live in PostgreSQL and link to retained sources. The legacy Qdrant bridge was retired at the owner's request on September 11; the planned later vector index is pgvector/HNSW.

## What is implemented

Tasks support capture, notes, priorities, projects, dates and optional due times, status changes, completion/reopening, archive, search, Inbox/Today/Week/All views, and JSON/CSV export. Stable command IDs prevent duplicate effects after a retry; revision conflicts expose the current record.

Schedules support one-off reminders, daily/weekly/selected-weekday/monthly recurrence, independent recurring tasks, rescheduling through the command API, cancellation, snooze and a durable notification Inbox. Missed recurring runs coalesce into one late occurrence before continuing from the current time. Invalid spring-forward times are rejected; recurring nonexistent times are skipped. Ambiguous initial times require an explicit offset. Monthly dates such as the 31st skip months without that date.

The worker accepts jobs through a transactional outbox, submits stable DBOS workflow IDs, and uses idempotent domain effects. Reminder delivery is deterministic and continues without a model connection. Web Push is opt-in per device; provider submission is not represented as proof that a phone displayed a notification.

The companion supports real text tool calls, private sessions, retained conversation recovery on the same browser tab, explicit memory capture, source inspection, correction through the command API, and deletion. Turning history off affects existing conversations too. Deleting a source blanks dependent assertions and redacts retained response/command content associated with that source.

Canonical memory search combines cloud query embeddings, Python cosine scoring and lexical/tag matches over PostgreSQL records. There is no active Mem0 or Qdrant path. pgvector/HNSW is deferred until base functionality is hardened.

## Original foundation evidence (before later batches)

- **39 backend tests** cover concurrent duplicate commands, ownership, revision conflicts, atomic rollback, source/privacy behavior, budget continuations, recurrence/DST cases, and voice-controller state rules, including unlimited active-session duration/turns/silence and orphan-session cleanup.
- A worker process was killed and restarted against PostgreSQL. Replaying an outbox submission produced one notification and no duplicate task occurrence.
- **9 frontend tests** verify that capture waits for the provider data channel, a brief WebRTC disconnect can recover, and a persistent disconnect releases the microphone and closes the server session. The TypeScript production build and Ruff checks pass.
- Browser acceptance passed at **1440 × 1000** and **390 × 844**: pairing, task creation/editing, completion/reopening, page reload, reminder creation, and phone navigation/dialog/chat layouts. No page exceptions or horizontal overflow were observed.
- A live Groq request created one synthetic task in **0.77 seconds**. Repeating its turn ID created no duplicate. Reported model usage for that check was approximately **$0.000515**. This is one text test, not a voice latency benchmark.
- After the OpenAI key was supplied, a live GPT-5.4-mini request created exactly one task in **3.4 seconds**; retrying the same turn did not duplicate it. Reminder delivery, private transcript exclusion and memory correction/deletion also passed.
- A live Realtime browser test sent synthetic speech over WebRTC, created exactly one task through the server tools, received nonzero audio energy, and interrupted the spoken confirmation. The provider cleared its audio buffer and the saved task remained accessible. No browser or provider errors were reported in that run. Playback begins only after the data channel opens to avoid losing the start of the synthetic recording.
- A separate live voice test transcribed **“Thanks.”**, selected **SILENT**, and produced zero received audio energy. The final task/interruption and silence checks both passed against the rebuilt Docker deployment.
- After the Eridani update, a live private text identity check introduced Eridani/Eri correctly, and the rebuilt deployment passed the synthetic voice task/audio/interruption check again.
- A separate browser text chat survived page reload. A private API conversation retained no transcript.
- The deployed reminder worker delivered a synthetic reminder into the Inbox. API/worker container restart preserved the task and exactly one notification.
- All **430 files** in the original Qdrant volume matched their copied-file checksums before the isolated Qdrant service started. This was verified before the September 11 retirement of the bridge.
- A populated encrypted PostgreSQL backup restored into a new isolated database: 5 tasks, 2 schedules, 1 notification, 2 sources, 2 assertions and 18 command receipts. The test task ID/title/revision, notification ID and deleted-source tombstones were checked.
- The Windows startup helper completed successfully with Docker running. An actual Windows reboot was not performed.

Synthetic tasks were archived, test schedules cancelled, test notices dismissed and test transcripts deleted after validation. Local diagnostic evidence and screenshots are ignored under `.runtime/`.

## Voice transcripts and recovery

Cloud input transcription uses `gpt-live-transcribe` in the existing Realtime
session. The browser displays provider deltas in a “YOU” bubble while speech
continues, then replaces partial text with the final transcript. Partial text is
provisional. Speaker items are reconciled by ID; internal gate/planning output
does not appear in the conversation. The server retains final transcripts under
the existing history policy and timestamps them at speech start so asynchronous
completion does not invert conversational order. Private transcripts stay visible
in the current browser session but are not written to history.

**Respond now** (formerly Submit) asks Eri to respond to the most recent finished
speech turn when she is waiting or unsure. It is disabled while processing and
after that turn has completed an action or answer; it does not submit the typed
message box or restart voice.

Recovered turns clear stale provider-error text. The browser retries transient
status failures and ignores replies from stopped sessions. The task event stream
flushes immediately, sends heartbeats, and has an explicit browser reconnect loop
with backoff and cursor recovery. Ordinary task polling remains a fallback.
Task-sync failures do not stop WebRTC audio. Historical HTTP/2/502 console errors
cannot establish a voice-provider failure on their own.

The personality was checked to occur once in the shared assembled instructions;
neither ignored environment file contains `JARVIS_SYSTEM_PROMPT`.

Transcription reference: [OpenAI live transcription](https://developers.openai.com/api/docs/guides/realtime-transcription).
The deployed browser test passed visible partial captions before end-of-speech, task creation, spoken confirmation, interruption, and a fresh voice session. It recovered from one injected task-stream 502 and one voice-status failure, with no browser exceptions or provider errors. Cloud caption timing was observed before end-of-speech with synthetic audio;
actual microphone latency and accuracy still vary. Transcription is a separate
provider charge; include it in the cost pilot.

## Backups and recovery

A dedicated container creates an encrypted PostgreSQL snapshot at startup and every 24 hours, retaining 30 days of timestamped snapshots. An earlier encrypted legacy-memory archive remains offline; the backup service no longer mounts Qdrant. Backup completion time appears in Settings.

Current locations:

- Encrypted snapshots: `C:\Users\davin\JarvisBackups`
- Recovery key outside the WSL disk: `C:\Users\davin\.jarvis\recovery.key`
- Local configuration: `.env.upgrade`
- Additional local recovery-key copy: `.runtime/backup-key`

The Windows recovery-key file grants access to the owner and SYSTEM. Snapshots contain personal records and authentication data; encryption keys and provider environment secrets are excluded from the database dump. Historical backups can contain records that were subsequently deleted. They expire with rotation; a restore should be reviewed before switching the application to it.

The backups and key are outside the WSL virtual disk, but on this PC. Protection against loss of the physical Windows disk still requires a copy of the encrypted snapshots and recovery key on another device.

From PowerShell in the repository:

```powershell
.\scripts\operations.ps1 status
.\scripts\operations.ps1 start
.\scripts\operations.ps1 backup
.\scripts\operations.ps1 logs
```

Restore into a new, unused database name beginning with `jarvis_restore_`:

```powershell
docker compose --env-file .env.upgrade -f compose.upgrade.yml run --rm --no-deps backup restore --file /backups/CHOSEN-SNAPSHOT.pgdump.enc --target jarvis_restore_review
```

The restore command refuses an existing database and does not launch workers. Verify restored contents before changing the application's database URL. Never run both the live worker and a restored worker against duplicated schedules.

`scripts/operations.ps1 stop` stops the new services while retaining their volumes. The original Jarvis stack and original Qdrant volume remain available independently.

## Startup and development

A Windows Startup shortcut launches `C:\Users\davin\.jarvis\start-jarvis.ps1` after sign-in. It starts WSL and Docker, waits for Docker readiness, then starts the Compose project. Startup output goes to `C:\Users\davin\.jarvis\startup.log`. Containers use `restart: unless-stopped` once Docker is running. Port 9443 is persisted by Tailscale Serve.

Python dependencies are locked in `uv.lock`; web dependencies are locked in `apps/web/package-lock.json`. The initial Alembic migration contains frozen schema SQL.

In WSL, with PostgreSQL running:

```bash
uv sync
uv run pytest -q
uv run ruff check apps/api tests migrations
cd apps/web
npm ci
npm test
npm run build
node e2e/acceptance.mjs
```

Tests create and remove a uniquely named test database. The browser acceptance script targets the configured owner deployment and creates labeled synthetic records; run it deliberately. Set `JARVIS_E2E_CHAT=1` only to include a paid provider check. `scripts/smoke_live.py` is also an explicit live-provider test.

For deliberate paid Realtime checks, run `powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\voice-fixtures.ps1` from the repository in Windows PowerShell to generate two synthetic WAVs under `.runtime/`. Then run `node e2e/realtime.mjs` from `apps/web` in WSL. Set `JARVIS_VOICE_SCENARIO=silence` to check the final-thanks silence gate. Set `JARVIS_E2E_RECOVERY=1` to inject a task-stream 502 and one voice status failure and verify recovery and a fresh voice session. This test substitutes a generated audio stream and never captures the owner's microphone. Evidence is written under `.runtime/`; archive the labeled task created by the task scenario after review.

For local frontend development use Vite's proxy and configure `JARVIS_ORIGIN` to the browser's development origin. For deployment, keep it equal to the private HTTPS origin so cookies and origin checks agree.

## Remaining release gates

This is the daily-use task/reminder foundation, with an initial source-backed memory layer. It is not a completed implementation of all PRD roadmap releases.

Davin reports a successful real microphone conversation. Gate quality across varied speech, measured interruption latency on the phone, locked-phone Web Push and the seven-day cost pilot remain unverified. Live synthetic browser tests establish provider connectivity and the exercised tool/audio paths, but do not replace those device and quality checks. Voice sessions are foreground-only. Application duration, speech-turn and idle-silence caps have been removed. Browser status polls renew a 30-second orphan-cleanup lease; disconnected clients are cleaned up. Explicit stop, provider failures and budget controls still apply.

Home Assistant, calendar, finance, delegated research, GPU embeddings/reranking and the full memory quality benchmark are later PRD increments. No accounts or devices were invented or connected. Legacy migration has been dropped at the owner's request.

Usage is an application estimate based on reported model calls and configured rates. It is not a provider-enforced billing cap; transcription charges, missing usage reports and pricing changes can differ from the displayed total. The checked price sources are [Groq GPT-OSS 120B](https://console.groq.com/docs/model/openai/gpt-oss-120b), [GPT-Realtime-2.1](https://developers.openai.com/api/docs/models/gpt-realtime-2.1), [Realtime Mini](https://developers.openai.com/api/docs/models/gpt-realtime-2.1-mini), and [GPT-5.4 Mini](https://developers.openai.com/api/docs/models/gpt-5.4-mini).


## GPT-Live and daily-use polish — September 11

Settings → Voice & conversation selects Realtime or GPT-Live and a compatible
voice. The browser remembers choices independently for each provider, and the
server rejects unsupported combinations. Voice changes start with a new session.
The chat header has New chat, a shortcut to voice settings, and privacy controls.
Voice mode keeps both captions visible and adds a bottom glow driven by local
microphone amplitude; measuring amplitude does not run speech recognition.

GPT-Live uses `gpt-live-1` with client delegation. Media connects directly over
WebRTC, while the API attaches a server-side Live WebSocket before the microphone
is enabled. The client waits for completed ICE gathering and `session.started`.
Only the API handles delegations. It reuses the existing `gpt-5.4-mini` chat agent
and the same authorized domain tools; there is no second task database/runtime.
Realtime remains `gpt-realtime-2.1` with its existing response gate and tool flow.

Live transcript fragments are retained in bounded session context, grouped by
speaker/timing only for display and history. Grouping never initiates a tool.
Private sessions keep ephemeral context but do not save transcripts. In normal
history mode, completed groups are saved with speaker attribution; an abrupt
process crash can lose the last unfinished group. Delegation IDs are deduplicated,
work is serialized, and new speech is checked again before a tool commits.
Previously committed results remain saved when a request changes or voice ends.
The broader durable planner/job/artifact system in R4 is still future work.

Live is billed by duration, currently $0.05 per minute, plus backend model usage.
Usage snapshots are cumulative; the WebRTC 15-second initialization credit is
counted toward total duration, not added twice. Budget reservations cover
unreported media and shutdown headroom. Uncertain creation/finalization keeps a
reservation. End voice mutes capture immediately, sends `session.close`, and
keeps the media connection open for the final usage event, with bounded cleanup.
Neither API promises unlimited provider sessions.

`ui_show` is an owner-checked tool for opening supported pages and displaying a
task/reminder by ID. Text and both voice providers send typed UI actions to the
current client. Task targets open their editor; reminder targets scroll into view
and highlight. Model output cannot run arbitrary JavaScript or navigate arbitrary
URLs. CopilotKit's frontend-tool pattern informed this initial tool, but the
CopilotKit package/runtime is not installed. Davin explicitly requested the full
integration path next: searches, filters, forms, settings, shared application
state, and the eventual Android surface. Those are recorded in TODO and app tasks.

Reminders now distinguish delivery, completion, and cancellation. Migration
`0002_reminder_completion` adds completion timestamps without deleting existing
records. The reminder view includes upcoming/due, completed, and cancelled
records. `schedule.complete` records completion of a one-time reminder and
supersedes pending delivery. `notification.complete` completes a delivered
occurrence; completing a recurring occurrence leaves its next run active.
Inbox completion also marks the notice read, so pending push delivery expires.

“Hey, Eri” is opt-in under Settings and uses browser SpeechRecognition when
available. It requests the browser speech service, does not download a local
Whisper/model, pauses when the page is hidden or voice is active, and resumes
afterward. Enablement is scoped to this page session. It is not a system-wide
wake word and cannot launch a closed website; native Android/background support
remains a separate roadmap item. Browser permission, service availability, and
physical microphone behavior still need a device check.

Official references checked for this implementation:
[GPT-Live](https://developers.openai.com/api/docs/guides/live),
[client delegation](https://developers.openai.com/api/docs/guides/live-delegation),
[Live voice choices and lifecycle](https://developers.openai.com/api/docs/guides/live-conversations),
[WebRTC](https://developers.openai.com/api/docs/guides/voice-webrtc?api=live),
[sideband control](https://developers.openai.com/api/docs/guides/voice-server-controls?api=live),
[pricing](https://developers.openai.com/api/docs/pricing),
[Realtime voices](https://developers.openai.com/api/docs/guides/realtime-conversations#voice-options),
[CopilotKit frontend tools](https://docs.copilotkit.ai/reference/hooks/useFrontendTool),
[browser SpeechRecognition](https://developer.mozilla.org/en-US/docs/Web/API/SpeechRecognition).


### Validation for the GPT-Live/polish change

51 backend tests and 14 frontend tests passed, with Ruff, TypeScript/Vite build,
and diff checks clean. Live API acceptance used generated speech and confirmed a
single saved task followed by spoken completion, streaming captions, final usage,
New chat, and a switch to Realtime. Realtime acceptance separately confirmed
audible output before interruption, preservation of the saved task, restart,
and recovery from injected event/status failures. A presentation-only timing race
in the original test was corrected to wait for actual received audio energy before
pressing Stop speaking.

Browser polish acceptance confirmed compatible voice choices, a real chat-agent
call to open/highlight a reminder, completion history, and mobile width. Physical
wake-word/browser permissions, locked-phone behavior, long sessions, and voice
preferences remain owner-device acceptance items.

Opt-in provider/browser checks from `apps/web`:
- `node e2e/live.mjs`: real GPT-Live task/audio/captions/close, then Realtime.
  It keeps a silent audio source running after its synthetic phrase, since GPT-Live
  context injection requires ongoing input frames even while the speaker is silent.
  It archives only its exact synthetic task.
- `JARVIS_E2E_RECOVERY=1 node e2e/realtime.mjs`: Realtime task, audio, interruption,
  restart and injected transport recovery.
- `node e2e/polish.mjs`: voice settings, agent navigation, reminder completion,
  and mobile layout. It creates one labeled future reminder; remove that exact
  fixture after reviewing its evidence.

Ignored evidence is in `.runtime/gpt-live-evidence.json`,
`.runtime/realtime-task-evidence.json`, and `.runtime/polish-evidence.json`.
The original checkpoint remains on main; this follow-up is deployed locally and
has not yet been committed or pushed.

## September 11 — automatic memory and conversational site controls

Checkpoint before edits: `404c230ddd9b7dae48fab4c78d6a5c2e87ea3707`, pushed to main
and verified. New work below is deployed locally, not yet committed.
Encrypted backup `jarvis-20260911T080749Z.pgdump.enc` completed before migration.
Additive migration `0003_memory_learning` preserves task/reminder/source history.

### Memory

`memory_learning.py` uses structured gpt-5.4-mini extraction: up to five useful
owner-stated facts, confidence >= .85, exact source evidence, tags and stable fact
keys. Paired quotation wrappers may be removed from otherwise verbatim evidence.
Facts embedded in requests can be learned; the requested action is not a memory.
Private conversations, opt-outs, deleted sources and assistant claims are excluded.
Eligibility is checked again after cloud calls, before committing facts.

Cloud `text-embedding-3-small` creates 512-dimensional embeddings stored in
PostgreSQL JSONB. There is no local inference or additional vector service.
Personal-scale cosine retrieval combines lexical/tag scoring and a lexical
fallback on provider failure. Larger stores may warrant pgvector indexes.
API reads never return raw vectors.

Fingerprints and same-key semantic matches prevent duplicates. Corrections create
new sources/revisions; supersession requires an eligible older fact and a newer
source. Forgetting clears content, evidence and embeddings and keeps a tombstone.
Entity/alias reconciliation remains a follow-up: phonetic names can produce
different supported spellings.

Cloud calls have budget accounting. Learning uses a separate `jarvis-memory` DBOS
queue (concurrency 2), so backfill cannot block reminders. Saved eligible history
is backfilled incrementally. Pipeline version 4 includes the revised evidence
validation. Failures retry, then appear for manual retry; failed current-version
jobs do not create unlimited replacement jobs. The Memory page shows learning
status, tags, source access, correction, forgetting, semantic search and retry.

Text retrieves before its first model call. Realtime seeds memory on connection
and retrieves from input transcription before planning, preserving interruptions.
GPT-Live receives startup memory and delegates personal-data work to the existing
backend, which retrieves using recent user transcript context. Retrieved JSON
is evidence, never executable instructions. Live transcript groups are persisted
after a quiet interval so learning need not wait for the session to close.

### Site controls, name and voice

CopilotKit 1.71's headless React registry manages the typed frontend handler.
Existing authenticated text/Realtime/Live transports invoke it through a per-device
bridge; no extra model runtime, CopilotKit cloud or conversation database is added.
`/api/v1/ui/sync` exchanges page, selected/visible IDs, search, filters, viewport,
chat/voice state and acknowledgements. Context expires in memory and is never
stored as transcript. UI actions wait up to ten seconds for the requesting device's
displayed/failed acknowledgement; other devices cannot receive or acknowledge them.

Tools cover page/record navigation, chat open/close/auto, task or memory search,
status/project filters and task/reminder entry. A capability map explains the site.
Open editors block conflicting navigation. Mobile navigation closes chat while
voice continues. Handler refs keep viewport and editor state current.

Preferred name is an owner setting (1–80 characters) used by bootstrap, greetings
and shared prompts; the personality file no longer hard-codes an owner name.
Existing profile names remain until edited.

Wake recognition accepts Eri/Eridani and Hey Eri plus existing recognition spellings.
The owner confirmed Hey Eri works. It remains opt-in, foreground-only and resumes
after voice cleanup. The browser grants 15 seconds of quiet after responses.
Speech, task work, captions and actual microphone/output audio activity extend
the window, preventing early transcript delivery from cutting off audible speech.
Unused newly opened sessions also close after 15 seconds. Provider/network/budget
limits remain, without the old total-duration or speech-turn caps.

The glow belongs to the app shell. A dock exposes reopen-chat, interrupt and end
while chat is closed. Live assistant captions reveal small word groups with bounded
catch-up and exact final text. This is display smoothing, not exact audio alignment.
Reduced-motion users get immediate captions.

Realtime close cancels/drains responses before settlement and accepts an already
ended media call. Unsettled generations retain a hold. Three verified test holds
were released; older uncertain reservations remain for evidence-based reconciliation.
The $150 monthly limit was unchanged.

### Validation and remaining work

64 backend and 18 frontend tests pass, plus separate real OpenAI extraction,
embedding and semantic-recall acceptance in a disposable database.
Ruff, production build and diff checks pass; migration and Docker health pass.

`e2e/site-control.mjs` verifies actual agent chat closing, filters/search, mobile
navigation, visible learned facts, name Settings, no browser errors or overflow.
`e2e/live.mjs` verifies one task, speaker captions, audio, global mobile dock/glow,
quiet shutdown, final usage, New chat and switching to Realtime.
`e2e/realtime.mjs` with recovery enabled verifies one task, audible output,
interruption with save retained, restart, captions and injected SSE/poll recovery.
The latest Realtime reservation is confirmed closed.

Live history inspection: 65 eligible user sources processed, 2 visible embedded
memories, no pending extraction jobs. Synthetic voice tasks archived.
Inspected `.runtime/voice-dock-mobile.png` and `.runtime/memory-mobile.png`.
The added standalone wake phrase still needs the owner's physical microphone check.

Accepted boundaries: authored notes and derived memory share sources/retrieval
but retain their roles. Tasks/reminders will share a work-item interface while
scheduling, execution, delivery and completion remain distinct underneath.
Those data-model/UI migrations, richer organization, Calendar, notifications,
scoped MCP/API and Android are the next feature phase.

## Weekly deep sleep and legacy retirement — September 11

The owner requested weekly memory consolidation and clarification of similar
spellings. The first version uses a durable `review_memory` job on the existing
`jarvis-memory` DBOS queue. Each owner gets one Sunday 03:00 slot in their saved
IANA timezone; startup catches up the latest missed slot once. Manual Review now
shares pending work rather than duplicating it. Turning memory learning or weekly
review off prevents scheduled and in-flight review effects.

Exact normalized duplicates retain one active fact and preserve the other sources
through `merged_into_id` links. Similar spellings are candidate questions only.
The current candidate detector compares otherwise matching word sequences with
one differing word, using spelling/phonetic similarity; cross-sentence alias
resolution and broader semantic reconciliation remain roadmap work.

`memory_reviews` holds candidate IDs/revisions and review status, without copying
the source text. A source deletion or correction invalidates pending questions.
The Memory page supports a full corrected fact, separate facts, or deferral for
seven days. `memory_review_list` and `memory_resolve` expose the same workflow to
text, Realtime and Live's delegated backend. Confirmed resolutions preserve
provenance, create explicit owner-statement memories, and queue re-embedding.

Prompt context can offer one optional clarification per owner per 24 hours.
It asks the assistant to finish the user's request first and ask only at a natural
pause. This is an offer to a model, not evidence the question was spoken or heard.
The UI queue remains the durable place to review unresolved questions.

The active Qdrant reader, frontend legacy results, settings, Compose service and
backup mount have been removed. Existing volumes and historical encrypted
archives are retained offline, but are not read by the new app. PostgreSQL with
pgvector/HNSW is the future vector-store direction after base hardening.

OpenAI documents `session.thinking.append` for quiet mid-session context updates.
Updates are injected progressively; acknowledgments do not guarantee the model
has consumed all content or will use it in its next speech. This capability is
recorded in TODO; the app currently retrieves at startup and for delegated work.
References: [context timing](https://developers.openai.com/api/docs/guides/live-conversations),
[append events](https://developers.openai.com/api/docs/guides/live-delegation).

Validation: 76 backend tests pass, including 12 weekly-review tests for duplicate
consolidation, source preservation, spelling ambiguity, ownership, stale edits,
idempotent resolution, deferral, prompt timing, concurrent enqueue, retries and
DST-aware weekly slots. One opt-in paid-provider test is skipped. All 18 frontend
tests and the production build pass. An isolated migration check retained an
existing memory through upgrade/downgrade/upgrade; the mobile browser correction
flow and Review now passed without writing synthetic memories to the live owner.
The screenshot was inspected for readable controls and horizontal overflow.

Before deployment, encrypted snapshot `jarvis-20260911T160818Z.pgdump.enc`
was created. Docker applied migration `0004_memory_review`; API, worker and
PostgreSQL are healthy. The retired `jarvis-next-legacy-memory-1` container was
removed without deleting its volumes. The first durable review scanned 2 facts,
merged none and queued the Hayes/Haze clarification. Next weekly run is
September 13 at 03:00 America/Chicago. A credential-free HTTPS check verified the
`index-B0knHE1o.js` bundle. The optional authenticated tailnet smoke test was
blocked by automatic approval review because pairing-token transfer to that
destination was considered unverified; no token was sent by that test.
Both TODO and the app roadmap record the weekly review, retired bridge, deferred
vector index and quiet Live context follow-up. No owner clarification has been
answered on their behalf.

## Temporary fixed pairing PIN — September 11

At the owner's request, pairing now uses their chosen fixed PIN in the ignored
`JARVIS_OWNER_TOKEN` setting, mirrored to `.runtime/pairing-code`. The existing
setup script preserves configured values, so setup and container restarts do not
replace this PIN. Only the API container needs recreation to reload the setting.
Session tokens, CSRF values and infrastructure secrets still use secure randomness.
Existing paired sessions remain valid; Google OAuth remains on the roadmap.
The PIN value is intentionally not copied into tracked documentation or source.


## September 11 hardening: multi-action work, voice endings and timed deadlines

The previous deployed memory/UI/weekly-review baseline was committed and pushed
to main as b9d1a4cd63216790a473f1b94b7101ee93e205fa before this batch.

- The shared backend now allows 100 tool calls and 30 planning rounds per request,
  with one final text summary call. Realtime uses the same configured allowance.
  Override JARVIS_MAX_TOOL_CALLS_PER_REQUEST (1–1000) and
  JARVIS_MAX_MODEL_ROUNDS_PER_REQUEST (2–100) in deployment configuration, then
  recreate the API. Reads count as calls. These are application controls, not a
  GPT-Live provider restriction. Paid requests still reserve budget.
- Task reads return pagination information. Bulk work preserves command receipts,
  reports partial outcomes and blocks the rest of an old tool batch after a spoken
  correction. Closing Live cancels pending backend work and optional retrieval;
  completed domain effects persist. Initial-retrieval cancellation releases its
  unused chat reservation; an interrupted paid request retains uncertain headroom.
- Existing memory correction/forgetting and notification read/snooze/dismiss
  commands are exposed through the same text/voice tool registry.
- Standalone farewell/thanks phrases end voice and release its microphone.
  Continued speech cancels a pending close; quoted phrases and thanks followed by
  another request do not match. Wake listening resumes through the existing
  enabled wake-word controller. Live transcript grouping is heuristic, so actual
  conversational pause behavior still belongs in the device pilot.
- Migration 0005_task_due_time adds nullable due_time and due_timezone to tasks.
  Tools, task editing, row labels and CSV/JSON export preserve them. Clearing a due
  date clears its time. Nonexistent local times are rejected; a repeated DST hour
  needs an explicit offset. Due times do not create reminders. Calendar/project/
  timeline/day views remain deferred to the task-tracker expansion.
- Semantic retrieval rechecks source visibility, suppression and assertion
  revisions after the cloud query. Weekly review exposes failures/retries and
  last success. Its offer cooldown follows a question containing the candidate
  spellings, rather than context preparation; this does not establish audible
  delivery or full semantic clarification tracking.
- Live can receive small relevant memory updates during a session through
  session.thinking.append. Updates are debounced, deduplicated and cancelled on
  close. UTF-8 byte bounds keep each payload below the documented 500-token limit.
  Acknowledgments establish accepted context, not that the next speech uses all
  of it. See [Managing GPT-Live sessions](https://developers.openai.com/api/docs/guides/live-conversations).
- Budget reporting separates recorded spend, active reservations and uncertain
  holds. It includes a calendar-month pace projection, an 80% warning and a 95%
  deferral of optional memory work. Deferred jobs resume with a new durable job
  when capacity returns. New usage records identify their configured pricing
  assumptions; existing rates remain estimates pending the provider-account audit.

Deployment and recovery: the current API/worker/PostgreSQL health checks pass.
The updated web bundle is index-DCkz93wi.js. Before migration, encrypted backup
jarvis-20260911T191149Z.pgdump.enc was saved. After deployment,
jarvis-20260911T191619Z.pgdump.enc restored into a new isolated database at revision
0005_task_due_time: 46 tasks, 8 schedules, 2 notifications, 182 sources,
7 memory assertions, 1 review and 140 command receipts. Dispatch was disabled,
and the temporary restored database was removed after verification.

A local, in-container HTTP check authenticated with the configured pairing PIN,
verified the new bootstrap/memory/task responses, and removed only its temporary
test session. At 19:16 UTC, recorded estimated spend was $2.551407 and older
uncertain holds totaled $124.267111. Those holds were not released without final
provider evidence. There was 1 visible canonical memory at that snapshot.

Validation: **99 backend tests and 33 frontend tests pass**, with one optional
paid-provider test skipped. Ruff and the production build pass. Checks include
a populated-memory migration upgrade/downgrade/upgrade,
isolated mobile memory review and task-time editing, budget UI checks, and visual
inspection at 390 by 844. New voice lifecycle and Live-context behavior have
automated tests; no new paid-provider voice trial was run in this batch. The
remaining acceptance work is real phone wake/ending/interruption and locked
notifications, an actual Windows reboot, an off-PC backup/key copy, older-hold
reconciliation and the seven-day owner pilot. The current deployment is its
baseline; historic synthetic records are not successful owner interactions.


## Unified workspace, calendar and accounting — September 11, 2026

The owner moved the seven-day pilot after the workspace expansion and real-use
voice/device/operations checks. This release adds the Work list and month calendar
with a selected-day agenda. Existing task/reminder persistence and notification
delivery remain authoritative. The calendar is a read-only projection, including
future recurrence previews; viewing it creates no occurrences or jobs.

Migration 0006_workspace_accounting creates projects, migrates every nonempty old
project label, adds task parent/assignee/work-type/tags metadata, optional schedule
project links and budget activity/settlement fields. Project renames update task
labels and revisions; project archives retain records. Owner-scoped graph changes
are serialized to prevent concurrent parent edits creating cycles. Agent assignee
labels are organization only.

All new domain actions share the existing authenticated, revision-checked,
idempotent command path. Eri gets project_list/create/update, schedule_update,
calendar_list and ui_calendar, plus expanded task fields and calendar filters.
A device must acknowledge calendar navigation/filtering before success is reported.
An open editor declines navigation. Reminder metadata edits preserve pending
deliveries, while timing edits deliberately supersede the previous schedule revision.

The calendar endpoint uses an exclusive end date, at most 62 days per query and
a 2,000-item response cap with an explicit truncation indicator. Date-only tasks
remain on their date; timed tasks are converted into the requested IANA timezone.
Recurring reminders preserve their wall-clock time over DST. Month cells and
agenda share status/project/kind/search filters. A completed linked task suppresses
future reminder previews as the worker suppresses their delivery.

Budget activity is durable. A reservation without activity for three minutes is
shown as uncertain, and housekeeping records its expired lease. Time alone never
proves a provider call was free. Explicit text-provider 4xx rejections except 408
release unused headroom; timeouts, 5xx and responses missing usage retain it.
Cancelled Realtime responses without final usage remain unconfirmed. Duration-based
transcription usage is recorded separately and idempotently at the current
gpt-live-transcribe rate. Source verification:
- [Realtime transcription usage event](https://developers.openai.com/api/reference/resources/realtime/server-events.md#conversation.item.input_audio_transcription.completed)
- [Realtime 2.1 rates](https://developers.openai.com/api/docs/models/gpt-realtime-2.1)
- [Live transcription rates](https://developers.openai.com/api/docs/models/gpt-live-transcribe)
- [GPT-Live rates](https://developers.openai.com/api/docs/models/gpt-live-1)
- [Provider usage and Costs API](https://developers.openai.com/api/reference/resources/admin/subresources/organization/subresources/usage)

The new estimate version is configured-2026-09-11-v2. Local estimates remain
distinct from provider invoices. Older estimates did not include the separate
transcription reports. A read-only Costs API request with the existing project key
returned 403; no credentials or response secrets were logged. Historical holds
cannot be safely settled from that key alone.

Run `.venv/bin/python scripts/reconcile_budget.py` from the WSL project to inspect
holds. To settle a specific reservation only after obtaining provider evidence, use
`--reservation <id> --final-usd <verified amount> --evidence <provider reference> --apply`.
The operator command rejects another owner's record and active requests, retains
all original usage events, adds a signed adjustment in the original period, and
records the evidence. Repeated identical settlement is idempotent. No reconciliation
mutation is exposed to the model.

Validation before deployment: all 116 backend tests and 36 frontend tests passed;
one optional paid-provider test remained skipped. Isolated browser acceptance covered project creation,
metadata, linked reminders, calendar/DST projection, mobile layouts, unconfirmed
usage display and the actual CopilotKit calendar/filter/declined-navigation
acknowledgment path. Existing project/task and budget history survived migration
0005 -> 0006 -> 0005 -> 0006. No fixtures were written into the owner's database.
The encrypted pre-deploy snapshot is jarvis-20260911T201738Z.pgdump.enc.

Physical microphone/phone behavior, Windows reboot, off-PC recovery copies and the
seven-day pilot remain unverified and scheduled after this expansion. No new paid
voice-provider test was run in this batch.


Deployment verified at approximately 20:27 UTC: API, worker and PostgreSQL healthy,
migration 0006_workspace_accounting, all existing project labels linked to one of
two project records. The live calendar read returned five entries without truncation.
Read-only authenticated smoke checks removed their temporary login session.
Budget snapshot: $2.551407 estimated recorded spend, $128.614433 unconfirmed
headroom across 26 older sessions, $0 active headroom, $18.83416 available under
the unchanged $150 limit. The older abandoned active session is now correctly
classified as uncertain; no historical hold was released without evidence.

Post-deploy encrypted backup jarvis-20260911T202655Z.pgdump.enc restored into a new
isolated database at migration 0006. Verified 46 tasks, 2 projects, 8 schedules,
2 notifications, 182 sources, 7 memory assertions, 1 memory review and 140 command
receipts. No worker/notification dispatch started on the restore; the temporary
database was removed after verification. Frontend bundle: index-Bo-rGFUu.js.
The actual Tailnet HTTPS endpoint returned 200 with that bundle.


## Conversational voice sign-off — September 11, 2026

Both media providers share voice-only conduct in personality.py. Eri can offer
"Anything else I can help you with?" or "Will that be all for now?" when the
exchange is complete, sparingly and without an outstanding question or task.
Realtime's existing speech gate interprets confirmations in conversational context.

The browser also tracks the latest direct closing question within the current
voice session. It preserves question polarity, including polite yes/no replies.
Offers apply only to the next utterance and expire after 30 seconds; they are not
restored from old history. Compound/quoted questions and an added task request do
not trigger a contextual ending. Continued speech cancels the existing one-second
closing grace period. Normal cleanup preserves Live final-usage collection and
returns to wake listening once voice is fully closed.

Live passes raw transcript text to the closing tracker separately from animated
captions, so a fast reply can be understood before display animation finishes.
If Live groups the reply with an earlier user display bubble, the tracker uses
only the suffix after the closing offer. Classification is limited to the supported
English closing phrases; the natural-choice prompt and actual microphone timing
remain part of the owner's real-use acceptance.

Budget investigation remains read-only: the September 11 snapshot has 26 older
Realtime reservations from 04:36 to 06:59 UTC, $128.614433 of held headroom,
$2.551407 recorded estimates and $18.83416 available under the $150 limit.
These are internal allowances, not bank/payment-card holds or confirmed provider
charges. New lifecycle fixes do not prove the cost of an older unreported response.

Provider evidence can come from the Jarvis project's Costs dashboard/export or an
organization billing API credential with access to the Costs endpoint. The existing
project key was denied that endpoint. Official Costs results are aggregated by
time bucket and project (and supported billing dimensions), not session ID:
[OpenAI Costs API](https://developers.openai.com/api/reference/resources/admin/subresources/organization/subresources/usage/methods/costs).
The current reconcile_budget.py command requires a session-specific final amount.
A project/date export therefore needs period-scoped reconciliation that accounts
for already-recorded usage and unrelated project activity; it must not be divided
into invented per-session charges. This remains an explicit follow-up pending
billing evidence. No allowance or spending cap was changed in this batch.

Validation: 34 focused backend tests passed (voice lifecycle, Live cleanup,
Realtime settlement and budget reconciliation), and all 71 frontend tests passed.
New cases cover both question polarities, polite replies, continued requests,
fresh-session/expired/stale questions, compound/quoted questions, early Live
captions and multiple exchanges sharing a display bubble. Ruff and the final
Docker TypeScript/Vite production build passed. No paid-provider microphone test
was run; that remains in the owner acceptance list.

Deployed at approximately 21:21 UTC: API/worker/PostgreSQL health checks pass.
The actual Tailnet HTTPS site returns 200 and serves index-D_YBQZJy.js.
The running image contains the shared voice prompt and contextual Realtime gate.
A read-only post-deploy budget check confirmed unchanged historical holds and no
active reservation. No database migration or owner-record mutation was needed.


## Development mode, contextual tasks and linked notes — September 11, 2026

The owner requested that development costs be monitored directly in OpenAI Usage.
Ignored local .env.upgrade now sets JARVIS_COST_TRACKING_ENABLED=false. Settings
displays that state instead of dollar values/holds/limit controls. The shared
budget boundary skips new Usage and BudgetReservation writes, reservation refresh,
settlement, expiration and enforcement, including optional memory deferral. The
operator reconciliation command is unavailable while disabled. Existing historical
ledger rows and saved limits are retained. This switch does not disable task action
receipts, workflow events, transcripts under their existing policy, or learning.

The default for a new installation is still true, explicitly generated by setup.
To restore accounting later, set the flag to true and recreate API/worker containers
after handling old holds and the missing development period from provider evidence.
Usage while disabled cannot be reconstructed from this local ledger. Never present
that gap as zero provider spend. No new credential is needed to use development mode.

Migration 0007_notes_context adds notes, note_task_links, note_embeddings and
task_references. Authored notes contain editable text, tags and optional project,
task and conversation links. Archive hides a note from normal/semantic retrieval;
restore requeues indexing. JSON export and encrypted PostgreSQL backups include
notes, links and extraction evidence. The task details editor shows note backlinks.
A note may reference an existing conversation without copying its transcript.

Saving a note queues an embed_note job on the existing durable memory worker queue.
It chunks text into 1,800-character spans with 200-character overlap, prefixes title
and tags, and calls the existing text-embedding-3-small cloud helper at 512 dimensions.
Chunks are stored in PostgreSQL JSONB with note revision/model; stale replies are
discarded after the provider call. Retry/deferred/failed state is retained, and saving
again retries a failed index. There is no Mem0, Qdrant or local embedding runtime.

Keyword search supports literal text/tags and task/project links with pagination.
Explicit meaning search uses a cloud query embedding and hybrid lexical/cosine
ranking over at most 1,000 recent active notes, returns up to 30 results and reports
truncation. It rechecks current note revision/archive state after the provider await.
Provider errors fall back to keywords. The UI does not request semantic embeddings
on every typed search character; the vector index remains deferred. Eri retrieves
notes on demand through note_search/note_read; notes are not automatically injected
as learned facts into every response.

Find to-dos calls the existing gpt-5.4-mini helper with structured output. It returns
a preview only. Each proposed task needs an exact quote from the current note, and
the revision is checked again before task creation. note.tasks creates selected
items atomically and retains evidence plus the originating note revision. A normalized
evidence fingerprint prevents repeat extraction from duplicating a task even after
unlinking it. Editing the note does not rewrite its previously created tasks.
The UI shows original evidence; no personal Memory assertions are created by this flow.
Eri has the same create/update/search/extract/task-link commands and may create
to-dos when the owner explicitly requests it; a request only to review remains a preview.

task_resolve can read selected tasks, currently visible task IDs, recent same-conversation
references or keyword candidates. It returns fresh revisions and explicit ambiguity
rather than mutating an uncertain target. References record up to 100 task IDs per
conversation; private/history-disabled sessions do not persist them. Recent unqualified
lookup uses the last touched group. Structured selected/search scope remains available
without that stored history. The bulk editor changes status, project, assignee, priority
and date; existing times stay with a new date, and clearing a date clears its time.
task.batch preflights up to 100 revisions under the existing workspace lock; one conflict
rolls back every edit. UI retries after a lost mutation response reuse the command receipt.

The per-device CopilotKit bridge now handles notes, task-group selection, note editor
opening and linked-record navigation. It reports selected_note_id/selected_task_ids and
visible note IDs; an open editor blocks assistant-driven navigation. Full field-filling,
settings controls, archived-note filter parity and project board/timeline remain follow-ups.
No extra agent runtime was added.

Validation: 130 backend tests and 71 frontend tests pass; one optional paid-provider
test remains skipped. TypeScript/Vite production build and Ruff pass for the app,
tests, migrations and changed scripts. The older smoke_live.py script has two unrelated
pre-existing Ruff findings when linting every script. Browser acceptance used an
isolated database and stubbed cloud responses: note authoring, extraction preview,
exact provenance and deduplication, keyword/meaning search, mobile layouts, task
backlinks, bulk changes, actual CopilotKit acknowledgments, disabled cost UI and a
lost-response Save retry. Changing the query after meaning search does not trigger
another semantic request. Existing tasks and old holds survived the 0006 -> 0007 ->
0006 -> 0007 migration round-trip. No fixtures were added to the owner's data.

Pre-deploy encrypted snapshot: jarvis-20260911T215928Z.pgdump.enc.
Deployment and post-deploy restore verification follow.
The large physical-device/voice/operations round and seven-day pilot remain deferred.
No paid-provider voice acceptance was run for this batch.


Deployed at approximately 22:09 UTC: API, worker and PostgreSQL are healthy at
schema 0007_notes_context. Authenticated bootstrap reports tracking_enabled=false,
budget_mode=disabled and no displayed amounts/holds. A content digest of all 322
historical budget reservations and 560 usage events matched before and after the
update and smoke reads. The test login session was removed. The actual Tailnet
HTTPS endpoint returns 200 and serves index-CrzwT8qP.js.

Encrypted backup jarvis-20260911T220845Z.pgdump.enc restored into a newly created
isolated database at schema 0007. Verified 46 tasks, 2 projects, 8 schedules,
2 notifications, 182 sources, 7 memory assertions, 1 memory review and 140 command
receipts, plus all four new notes/context tables (empty in the owner's live data).
No worker was started on the restore; its temporary database was removed.


## Google sign-in and read-only Calendar — September 11

Migration 0008_google_calendar adds google_identities, google_oauth_attempts,
google_calendars and google_calendar_events, plus auth_sessions.auth_method.
Google is optional: pairing works while the client credentials are absent.
[GOOGLE_SETUP.md](GOOGLE_SETUP.md) contains the exact callback and setup steps.

FastAPI owns OAuth authorization-code exchange using google-auth-oauthlib and
ID-token verification using google-auth. Linking starts from a current owner session
with CSRF protection. Login requires the previously linked stable Google subject;
an unpaired visitor cannot claim the account. A one-use state, browser cookie,
PKCE verifier and nonce bind each redirect. Signature, issuer, audience, expiry,
nonce and verified email are checked. Callback access logs omit query parameters.
Google sessions use the existing Secure, HTTP-only session cookies. No JWT or
refresh token is exposed to frontend code.

Identity/email and Calendar consent are separate. Calendar uses calendar.readonly
and an offline refresh token encrypted with Fernet in PostgreSQL. The key is generated
in ignored .env.upgrade, with a local .runtime/integration-key recovery copy. Neither
the key nor client credentials are committed. JSON exports omit integration secrets.
Encrypted database backups retain the encrypted credentials. Disconnect removes
cached events and attempts provider revocation; unlink additionally invalidates Google
sessions. Existing pairing sessions remain usable. Changing credentials or losing the
encryption key requires reconnecting Calendar.

The existing durable DBOS worker polls selected calendars, normally every five
minutes. CalendarList and Events pagination complete before events and sync cursors
commit together. An invalid sync token (HTTP 410) triggers a fresh snapshot. Deleted
events, removed calendars and inaccessible sources are handled. Generation checks
discard work completed after disconnect or selection changes; stale pending jobs
recover after fifteen minutes. Expired refresh grants require reconnection.
Source metadata defaults to the primary calendar only; additional sources are opt-in.

The month/day workspace overlays read-only events on tasks and reminders. Date-only
Google events stay all-day; supported recurrence includes daily/weekly/monthly/yearly
rules, exceptions, moved instances and DST. Projection is bounded to 2,000 entries
per series/output; sub-daily rules are marked incomplete. Initial sync is bounded to
2,500 calendars and 50,000 events per calendar. A collection beyond those limits
requires a later narrower/indexed sync design. A truncated view says it is incomplete.

Availability queries use Google's current freeBusy endpoint for selected calendars,
up to a seven-day window and fifty sources. They merge overlaps and return free
intervals only when every requested calendar succeeds and the connection remains
current. Cached events are labelled with their sync time. Task deadlines/reminders
do not reserve time; a timed deadline inside a cached busy event gets a conflict hint.
This batch cannot create, move or delete Google events.

Eri uses calendar_connection, calendar_sync, calendar_availability, calendar_list and
revisioned calendar.select through the current agent runtime. CopilotKit can navigate
to a selected event and highlight it. The selected event is part of the app context;
open editors block navigation. Google consent remains a user action in Settings.

Synthetic tests cover OAuth state/replay/identity/scope enforcement, encryption,
disconnect races, refresh recovery, pagination, sync reset, atomic failures,
recurrence/DST, free/busy errors and ownership. Browser acceptance uses a disposable
database and synthetic Google responses to exercise redirects, source selection,
availability, event details/highlights, desktop/mobile layouts, disconnect/reconnect
and unlink. Signed-JWT tests run through Google's verifier. These checks do not
prove the owner's real Cloud project or consent grant is configured.

The browser automation helper failed to launch twice while opening Google Console;
no Google client or real grant was created. The two client credential slots in .env
remain empty. Real account linking, first sync and second-device sign-in are open.
The broad physical-device/voice round and seven-day pilot remain deferred.


Validation/deployment: 157 backend tests and 71 frontend tests pass; one optional
paid-provider test remains skipped. TypeScript/Vite, scoped Ruff, migration
round-trip, isolated browser acceptance and visual desktop/mobile review pass.
No real Google requests or paid voice acceptance were run for this batch.

Pre-deploy encrypted snapshot: jarvis-20260911T230811Z.pgdump.enc. Deployed at
approximately 23:12 UTC with healthy API/worker/PostgreSQL, schema 0008_google_calendar
and HTTPS bundle index-uUPdYU9g.js. Live authenticated checks verified working pairing,
the unconfigured Google state, and disabled cost tracking. All 322 historical
reservations and 560 usage events matched the pre-deploy content digest.
The temporary smoke-login session was removed.

Post-deploy encrypted backup jarvis-20260911T231255Z.pgdump.enc restored in an isolated
database: 46 tasks, 2 projects, 8 schedules, 2 notifications, 182 sources, 7 memory
assertions, 1 memory review and 140 receipts. Notes/context and Google tables exist
and are currently empty. The restore started no worker and its temporary database
was removed. This proves schema/data recovery; encrypted real Google credential
recovery still requires a real connected account.


## Mobile Calendar and optional Google editing — September 12

The owner confirmed real read sync and authorized event writes in this batch.
Month, Week and Day views share the existing task/reminder/Google projection;
double-tap or Open day selects a dedicated agenda. Calendar view is persisted per
browser and included in CopilotKit context/navigation. Background polling keeps
existing entries visible until the replacement arrives. The old data-clearing
effect collapsed long agendas every 30 seconds and clamped mobile scroll position.
Assistant highlights now scroll once rather than on every task refresh.

Migration 0009_calendar_writes adds calendar_write_enabled and calendar access_role.
Existing encrypted credentials and selections remain intact; editing defaults off.
A separate calendar_write OAuth purpose requests calendar.events alongside existing
identity/read scopes. Declining the extra grant preserves the current connection.

Website forms and Eri use shared calendar.create/update/delete commands, through the
existing command receipts, transactional outbox and DBOS Google queue. There is no
additional agent runtime. calendar_event_read fetches a current provider event and
returns an expiring encrypted edit token bound to owner, source, generation,
recurrence scope and ETag. Every write rechecks source selection and current Google
ACL. PATCH and DELETE use If-Match, so a stale edit cannot overwrite newer content.

Creation uses a persistent provider event ID; create/update also attach a private
write marker. Retries read Google first to reconcile lost responses. Unknown outcomes
remain unconfirmed, including when Google access is lost after a write started.
Queued writes expire after one hour before a new outbound attempt. Confirmed writes
update the local event cache and invalidate sync snapshots started before that
change. calendar_write_status and Recent calendar changes expose the durable result;
Eri must not claim a queued operation has been saved.

Forms support timed/all-day events, title, location, notes, busy/free and basic
daily/weekly/monthly recurrence on creation. All-day last dates are inclusive in
the UI and exclusive at the provider boundary. Recurring changes explicitly select
one occurrence or the whole series. Guest events, invitations and special Google
event types remain managed in Google; no guest notifications are sent by this UI.
Read-only Calendar remains available without the additional write consent.

Validation: the full backend suite passed 183 tests with one optional provider test
skipped; a subsequently added unknown-outcome regression passed with all 27 write
tests. All 74 frontend tests passed. Production build, scoped Ruff and isolated
migration/browser acceptance passed. Acceptance includes 0008/0009 round-trip
credential/selection preservation, mobile Month/Week/Day, actual 30-second refresh
scroll stability, create/edit/delete, response-loss retry and consent/disconnect.
Provider writes were synthetic and isolated; no fixture events were created in the
owner's real calendars. Cost tracking remains disabled. Real voice/device checks
and the seven-day usage pilot remain deferred.

Deployment verified at approximately 00:26 UTC on September 13 (September 12 local).
Schema 0009_calendar_writes is active; API/worker/PostgreSQL are healthy. The Tailnet
HTTPS page serves index-DpWrne0k.js and includes the editing controls. Google remains
linked with read sync enabled and write consent pending. Identity/credential and
calendar-selection digests match the pre-deploy baseline; all 322 historical cost
reservations and 560 usage events also match, with tracking still disabled.
A real read-only sync was queued to refresh source access roles. No Google write
jobs or synthetic events were created on the owner's account.

Pre-deploy backup jarvis-20260913T002220Z.pgdump.enc restored at schema 0008.
Post-deploy backup jarvis-20260913T002615Z.pgdump.enc restored at schema 0009,
including encrypted credentials, selections/access roles and 2,680 Google events.
The restored snapshot also contains 47 tasks, 2 projects, 8 schedules, 3 notifications,
238 sources, 11 memory assertions, 1 memory review and 146 command receipts.
Neither restore started a worker; both temporary databases were removed.


## Unified planning and Linear (schema 0010)

A Task stores the work and completion state. Schedule stores its alert timing or
routine recurrence; Occurrence and Notification retain delivery/completion history.
Legacy standalone schedules gain tasks without converting alert times into
deadlines. A repeating template creates one task per delivered occurrence.
Completing a task also closes its remaining alerts and delivered notices.

PlanningEntry stores local appointments and task work blocks with an optional
Google publication link, last shared snapshot and durable write job. Google is an
optional copy destination. External edits require review; a missing Google copy
does not delete local work. Calendar projection suppresses the duplicate linked
copy and availability merges local busy intervals with live Google free/busy.
Google cache backfill adds rich details while preserving stable local event IDs.

LinearConnection holds an encrypted personal key, selected teams, member/workspace
identity, cursor and directory. LinearIssue maps the stable provider UUID to one
Task and stores shared/pending/conflicting snapshots. Polling and writes use a
dedicated DBOS queue, the existing transactional outbox and shared domain commands.
Eri and UI task edits use the same write path. No Slack relay or second MCP writer
is introduced. Read [LINEAR_SETUP.md](LINEAR_SETUP.md) for setup, mappings and the
provider's conditional-write limitation.

Langfuse remains deferred. The external bot API/MCP and notification expansion
remain separate future work.


## Productivity graph — September 12, 2026

Schema 0011 adds private spaces, areas, goals, project lifecycle, many-to-many
goal/project relationships, note backlinks and multi-record links, planned task
dates and stable assignee IDs. Goals & projects is the new management page.
Every command is available through the shared Eri registry; filters and navigation
use the existing CopilotKit bridge. See [PRODUCTIVITY_SCHEMA.md](PRODUCTIVITY_SCHEMA.md)
for relationship and completion semantics.

Validated with 213 backend tests (one optional skipped), 76 frontend tests, production
build, Ruff, migration/legacy preservation, desktop/mobile productivity acceptance,
and the existing Google/Linear browser suite. Deployment preserved old data in
16 tables, including all 55 tasks, integrations and the disabled cost ledger.
The HTTPS bundle is index-GZSDsE7r.js. Pre/post encrypted backups restored successfully;
the post-upgrade snapshot is jarvis-20260913T042040Z.pgdump.enc.

Background model selection is discussed in
[BACKGROUND_MODEL_COMPARISON.md](BACKGROUND_MODEL_COMPARISON.md).
The running agent remains gpt-5.4-mini. Comparative provider evaluation and
Langfuse remain future work; no model switch was made.


## September 13: optional Gemini task agent

Added a server-defined backend-provider catalog and persistent owner selection in
Settings. OpenAI stays the configured default; Gemini 3.8 Flash uses Google's
OpenAI-compatible Chat Completions endpoint with low reasoning and an 8,192-token
generation allowance. Each text/GPT-Live delegated turn pins its provider, keeps
Gemini thought signatures across the existing tool loop, and retains durable
receipts without provider fallback after errors. Audio, memory/note extraction
and embeddings keep their existing providers. No schema migration is required;
the choice uses the existing OwnerSettings JSON record. Development cost
accounting remains disabled.

GEMINI_API_KEY is loaded only on the server; .env has a blank row for the owner.
Bootstrap exposes model names and credential availability, never credentials.
See docs/GEMINI_SETUP.md for container recreation and the synthetic real-provider
tool-catalog/continuation check. Real Gemini access remains unverified until the
key is supplied.

The existing dedicated backup service already performs encrypted daily database
backups with 30-day retention. TODO now records that explicitly and separates
off-PC recovery copies, automated restore drills and failure alerts still to do.


Validation and deployment: 225 backend tests (one optional skip), 76 frontend
tests, production build, Ruff, and desktop/mobile Settings acceptance passed.
Empty/filtered/malformed model responses now terminate the local job cleanly.
The rebuilt API/worker are healthy; authenticated bootstrap identifies the default
gpt-5.4-mini route and the Gemini option waiting for its key. HTTPS serves
index-CjYJ0is5.js. Task, project, goal, note, memory and historical cost-ledger
counts match before/after deployment. No real Gemini call has run yet.


## September 13: real Gemini/Luna acceptance

Loaded the owner's Gemini key by recreating API/worker; both are healthy.
The real Gemini tool-catalog/continuation handshake passed. The reusable
scripts/evaluate_task_agents.py runs actual cloud models against a disposable
PostgreSQL fixture with local-only tool authorization and no workers. Final
comparison: Gemini low reasoning 12/12 clean passes; Luna reasoning none 11/12
clean, with a project-ID typo rejected and then corrected. Both produced correct
final records in every case. Responses-enabled Luna reasoning remains a later
test; Chat Completions rejects Luna tools plus reasoning low.

Results, caveats and traces are in docs/BACKGROUND_MODEL_COMPARISON.md and
docs/evals/task-agents-2026-09-13.json. Production remains gpt-5.4-mini, accounting
off; task/memory and historical ledger counts are unchanged. Gemini is available
for owner selection in Settings.
