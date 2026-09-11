# Eridani / Jarvis — progress and next steps

Updated September 11, 2026. Eridani (Eri) is the assistant's name.
Jarvis remains the repository and infrastructure project name.

## Current position

The daily-use task/reminder foundation is running in Docker at
https://davispc.tail957c2.ts.net:9443. OpenAI text and Realtime voice are connected.
Davin reports that a real voice conversation works very well. This is the working
foundation, not completion of every integration in the upgrade PRD.

Detailed implementation, recovery instructions, evidence, and limitations:
[JARVIS_IMPLEMENTATION.md](JARVIS_IMPLEMENTATION.md).

## Done

- [x] FastAPI domain API, PostgreSQL storage, and a separate durable DBOS worker.
- [x] Responsive web app with task capture, editing, completion, reminders,
      recurring schedules, notification Inbox, and export.
- [x] Persistent pairing sessions: secure HTTP-only cookie, 30-day expiry.
- [x] OpenAI project key loaded from the ignored .env; live text actions verified.
- [x] Realtime browser voice with server-owned tools, silence gate, interruption,
      and spoken confirmation after a successful save.
- [x] Browser microphone capture waits for provider readiness; brief connection
      drops have a recovery window.
- [x] Private conversation mode, source-backed memory capture/correction/deletion,
      and read-only access to an isolated copy of legacy memory.
- [x] Command retry deduplication, durable reminder recovery, and budget tracking.
- [x] Private Tailscale HTTPS access, Docker startup helper, encrypted backups,
      and a populated backup restore check.
- [x] Owner has tried a real microphone conversation successfully.
- [x] Replace competing Buster/Jarvis personality instructions with Eridani/Eri:
      witty and lightly playful, polished and formal, with assistance as her purpose.
- [x] Share one personality source between text and voice:
      [personality.py](../apps/api/jarvis/personality.py).
      Remove the old JARVIS_SYSTEM_PROMPT entry from environment configuration.
- [x] Remove application voice session duration, speech-turn, and idle-silence caps.
      Keep browser-disconnect cleanup, provider failure handling, explicit End
      voice, and the existing budget controls.

## Latest validation

September 11: 35 backend tests and 3 frontend tests passed, with Ruff and the
production build clean. After the containers were rebuilt, a live private text
check correctly introduced Eridani/Eri. A live synthetic voice request created
exactly one task, received spoken confirmation, and successfully interrupted it
without losing the saved task. No browser/provider errors were reported.
Synthetic task records were archived after testing.

## Next: daily-use polish and account access

- [ ] Add Google OAuth/OpenID Connect sign-in, restricted to Davin's authorized
      Google account. Preserve persistent sessions and device revocation.
      Also saved in Eridani's task list as “Add Google account sign-in to Jarvis.”
- [ ] Test locked-phone Web Push on the actual phone, including permission,
      delivery, opening the notice, and recovery after a connection gap.
- [ ] Validate long conversations on the actual device, varied speech, pauses,
      and interruption timing. Provider/network interruptions remain possible
      even though application session caps are removed.
- [ ] Run the seven-day usage/cost and voice-quality pilot; record issues and
      tune the gate and personality from actual use.
- [ ] Test startup after an actual Windows reboot.
- [ ] Keep a recovery copy of encrypted backups and the recovery key off this PC.

## Later PRD work

- [ ] Improve canonical and legacy memory retrieval: semantic search,
      embeddings/reranking, correction/deletion overlays, and quality benchmarks.
- [ ] Connect Home Assistant with explicit device mappings.
- [ ] Add calendar integration and private-host-compatible synchronization.
- [ ] Add finance and delegated research capabilities in bounded increments.

## Maintenance rule

Record completed work and concrete next steps here after substantive changes.
Keep detailed operational evidence in JARVIS_IMPLEMENTATION.md; the PRD and
upgrade decisions remain the design references. Environment files hold secrets
and deployment settings, not an alternate personality.
