# Jarvis upgrade: decision brief

Implementation update: the owner delegated the choices and confirmed an always-on host. The selected architecture, deployed foundation, evidence and remaining release gates are recorded in [JARVIS_IMPLEMENTATION.md](JARVIS_IMPLEMENTATION.md). The proposal below is preserved as the design baseline.

September 10, 2026. Read this first, then use the [full PRD](/home/davin/jarvis/docs/JARVIS_UPGRADE_PRD.md) for detailed requirements, source citations, data contracts, alternatives, and acceptance tests. This is a proposed design, not an implementation report.

## The recommendation

Build Jarvis around durable personal services, with Realtime as the conversational interface. Start with the exact loop you selected: **Pixel voice → Realtime → task API → PostgreSQL → the same task in the UI**. Make silence and interruption part of that first experiment.

Use a Python/FastAPI API, a separate DBOS worker, PostgreSQL, and a React/TypeScript web app. Keep the domains as separate modules with clear APIs, initially sharing one application package. Preserve the existing Qdrant memory data. Add Home Assistant next, then source-backed memory/context, then finance and richer delegated work.

This retains the useful Python and retrieval ecosystem without carrying the current process-wide conversation loop into the new system. It also avoids requiring a separate container and network service for every domain before the task loop works.

## What the current code tells us

The existing application is a voice prototype: one large React component, Express audio middleware, a shared Python conversation process, and Mem0/Qdrant. Accepted work, session isolation, and recovery need new foundations. There is no application task database or durable job system to extend today.

The most important migration risks are shared audio/text broadcasts, process-local conversation state, non-durable background threads, and legacy memory that lacks complete source provenance. The existing port-9001 transcription path may have a glasses client, so it should remain until that consumer is checked. The [architecture inventory](/home/davin/jarvis/docs/JARVIS_UPGRADE_PRD.md:28) links each finding to the inspected code. Existing uncommitted changes were left untouched.

## Choices worth making first

| Choice | Recommended starting point | Main alternative and when it wins |
|---|---|---|
| Voice transport | Browser WebRTC with server-side session/tool control | Relay audio through the PC if a demonstrated local-processing requirement needs it. |
| Silence | Disable automatic response creation; test a short classifier that chooses silence, waiting, responding, or ending | A transcript-based small/local classifier if the Realtime gate misses the latency or cost target. |
| Backend and durable work | FastAPI + DBOS + PostgreSQL | TypeScript + Graphile Worker if an all-TypeScript core is the stronger maintenance preference. Temporal if distributed workflow operation later warrants it. |
| Memory ownership | Jarvis owns retained sources and versioned assertions; extraction/index engines are replaceable | Hindsight is the primary packaged-engine challenger to benchmark. Mem0 remains the migration baseline and a possible adapter. |
| Retrieval storage | Keep Qdrant initially | pgvector if measured retrieval quality and simpler operation justify consolidation. |
| Phone notifications | Durable Inbox plus opt-in Web Push | A native companion if locked-Pixel delivery tests reveal unacceptable limitations. |

These are recommendations for review, not choices you need to answer before the research is complete.

## Findings that materially affect the design

Silence needs more than semantic VAD. VAD decides whether speech has ended; the response policy decides whether Jarvis should speak. An application-controlled gate gives silence an enforceable meaning, but its extra latency must be measured. This is the highest-priority experiment, not a solved performance claim. [Voice design and supporting documentation](/home/davin/jarvis/docs/JARVIS_UPGRADE_PRD.md:143).

A reminder is not permission to perform its contents. “Remind me to email Josh” creates a notification, not an email. Simple reminder delivery needs no planner call. More complex scheduled work stores intent and selects a model when execution begins; an already-running job records its model/workflow version so recovery stays consistent.

Home Assistant reduces device-specific code, but it does not automatically import every Google Home device. The adapter must distinguish a requested action from observed device state. Calendar push callbacks also cannot directly reach a private tailnet server, so the proposed calendar adapter polls outbound. Browser Web Push is a different mechanism and does not require a public Jarvis callback. [Home integration evidence](/home/davin/jarvis/docs/JARVIS_UPGRADE_PRD.md:561), [calendar and notification design](/home/davin/jarvis/docs/JARVIS_UPGRADE_PRD.md:236).

The 3080 reports 10 GB VRAM. Use it first for local embeddings and reranking, with resource limits and a fallback that leaves tasks/reminders operational. Do not require a local conversational model or custom model training for the first release. Personalize ranking only after collecting enough reliable relevance feedback to evaluate an improvement.

Keep original sources where capture and retention are enabled. Imported legacy facts without transcripts must stay labeled as unverified legacy material. Corrections, deletion, index rebuilds, and trained-model provenance are explicit parts of the memory design.

## Budget and privacy

Treat $150/month as a monitored operating limit, not an assumed bill. The [budget section](/home/davin/jarvis/docs/JARVIS_UPGRADE_PRD.md:596) includes current model candidates and a speech-only cost table; repeated context, the silence classifier, research, retries, and extraction add costs beyond that table. A seven-day metered pilot determines whether the proposed allocation fits your actual use.

Reserve spending before paid work begins, share limits across delegated jobs, and stop accepting new paid work at the configured limit. Local task editing and deterministic reminder delivery should still work. Increasing the budget is an explicit setting change.

The task and memory databases stay local, but cloud voice sends speech and selected context to OpenAI. Proposed settings keep raw audio recording off, expose a private-session mode, and make transcript retention visible. These settings are recommendations for your review; they have not been enabled.

## What to authorize when implementation begins

The smallest useful commitment is W0 plus R0, not the whole roadmap at once. W0 is an estimated 3–5 working days of focused experiments; R0 is an estimated 5–8 days for the integrated task loop. These are planning estimates, not delivery promises.

The evidence to collect is concrete:

1. Pixel audio tests showing comfortable pauses, appropriate silence, accurate requests, and interruption timing.
2. A task surviving a forced server restart, with a repeated command producing no duplicate.
3. A DBOS enqueue/recovery test and a locked-Pixel notification prototype.
4. An actual model-cost trace and verified private HTTPS/session identity through Windows/WSL.

Then build daily-use tasks/reminders, run the seven-day pilot, and proceed through home control, memory/context, and delegated finance/research. The full PRD defines the exit conditions for each release.

Public documentation cannot settle your speech-model performance, device compatibility, provider-account access, bank coverage, backup destination, or tolerance for PC downtime. Those are explicitly assigned checks, not missing architecture. The research and PRD are complete enough to make the initial implementation decisions without pretending those experiments have already happened.
