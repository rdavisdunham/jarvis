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
- Task deadlines remain date-only. Reminders have explicit IANA time zones and resolved instants. The default is America/Chicago at 10 AM when only a date is supplied.
- Repeated routines create independent task occurrences. Completing one occurrence does not cancel the series. Independent recurring reminders also remain independent of task completion.
- History and conservative memory learning start enabled. Private sessions do not retain transcripts. Explicitly requested tasks and saved memories still persist. Jarvis does not record raw audio.
- The default voice candidate is GPT-Realtime-2.1. The server controls response creation, silence, waiting, interruption and tool execution. Audible confirmations are requested only after commands commit.
- The model budget starts at $150/month and is editable in Settings. Each paid continuation reserves capacity before execution; uncertain provider outcomes retain their reservation.
- Preserve legacy memory through a read-only adapter and an isolated Qdrant copy. New assertions belong to Jarvis and link to retained source records.

## What is implemented

Tasks support capture, notes, priorities, projects, dates, status changes, completion/reopening, archive, search, Inbox/Today/Week/All views, and JSON/CSV export. Stable command IDs prevent duplicate effects after a retry; revision conflicts expose the current record.

Schedules support one-off reminders, daily/weekly/selected-weekday/monthly recurrence, independent recurring tasks, rescheduling through the command API, cancellation, snooze and a durable notification Inbox. Missed recurring runs coalesce into one late occurrence before continuing from the current time. Invalid spring-forward times are rejected; recurring nonexistent times are skipped. Ambiguous initial times require an explicit offset. Monthly dates such as the 31st skip months without that date.

The worker accepts jobs through a transactional outbox, submits stable DBOS workflow IDs, and uses idempotent domain effects. Reminder delivery is deterministic and continues without a model connection. Web Push is opt-in per device; provider submission is not represented as proof that a phone displayed a notification.

The companion supports real text tool calls, private sessions, retained conversation recovery on the same browser tab, explicit memory capture, source inspection, correction through the command API, and deletion. Turning history off affects existing conversations too. Deleting a source blanks dependent assertions and redacts retained response/command content associated with that source.

New memory search uses PostgreSQL full-text search. Legacy search uses a bounded lexical bridge over up to 1,000 records. Legacy facts are labeled unverified and have no claimed original transcript. The restored collection currently contains 263 records with 384-dimensional cosine vectors.

## Verified evidence

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
- All **430 files** in the original Qdrant volume matched their copied-file checksums before the isolated Qdrant service started. The restored collection reports healthy.
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

A dedicated container creates an encrypted PostgreSQL snapshot at startup and every 24 hours, retaining 30 days of timestamped snapshots. The original legacy memory archive is encrypted separately. Backup completion time appears in Settings.

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

Home Assistant, calendar, finance, delegated research, GPU embeddings/reranking and the full memory quality benchmark are later PRD increments. No accounts or devices were invented or connected. Legacy correction/deletion overlays and semantic legacy retrieval remain part of that later memory work.

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
