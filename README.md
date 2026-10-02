# Eridani

Eridani ("Eri") is a durable personal assistant: tasks, reminders, calendar blocks,
notes and lists, source-backed memory, text chat and GPT-Live voice, with Google
Calendar and Linear integrations.

**Production runs on Railway at https://app.eridani.app.** See the
[cloud runbook](docs/CLOUD_MIGRATION.md). The old local production stack
(`compose.upgrade.yml`) is retired; never start it as a live writer against its stale data.

## Architecture

- **Web:** React 19 + TypeScript, built with Vite (`apps/web`). Served by the API on the same origin.
- **API:** Python 3.12, FastAPI/Uvicorn, SQLAlchemy 2, Alembic (`apps/api/jarvis`, `migrations/`).
- **Worker:** DBOS durable worker (same image) for accepted requests, reminders, integration
  sync/writes, extraction, indexing and Web Push.
- **Database:** PostgreSQL 16 (canonical store, including DBOS state). Railway PITR plus
  planned independent encrypted R2 exports ([R2 backups](docs/R2_BACKUPS.md)).
- **Deploy:** `Dockerfile.upgrade`; Railway services start via `python -m jarvis.deploy api|worker`
  with `jarvis.deploy migrate` as pre-deploy ([.railway/railway.ts](.railway/railway.ts) is a
  whole-project plan; read its header before applying anything).

Start with the [audit handoff](docs/CLAUDE_AUDIT_HANDOFF.md) for a full map, then
[implementation notes](docs/JARVIS_IMPLEMENTATION.md), [CI](docs/CI.md),
[app evals](evals/app/README.md) and [TODO](docs/TODO.md).
Assistant personality: [Eridani / Eri](apps/api/jarvis/personality.py).

## Local development and tests

Use a disposable/development database only. Never point local tooling at production
or at a restored production copy with the worker or external services enabled.

```sh
uv sync --locked --group dev
# Backend tests need a throwaway PostgreSQL 16; fixtures create/remove their own databases.
JARVIS_ENV_FILE="" JARVIS_DATABASE_URL=postgresql+psycopg://USER:PASS@127.0.0.1:PORT/postgres \
  uv run --no-sync pytest -q
uv run --no-sync python -m pytest -q evals          # offline eval harness tests
cd apps/web && npm ci && npm test && npm run build   # frontend
npm run dev                                          # Vite on 127.0.0.1, proxies /api to :8765
```

`compose.upgrade.yml` is kept for local development and restore drills and is safe by
default: `docker compose -f compose.upgrade.yml up` starts only PostgreSQL, migrations and
the API, with `JARVIS_WORKER_ENABLED` and `JARVIS_EXTERNAL_SERVICES_ENABLED` forced to
`false` and no automatic restarts. The worker (reminders, push, Google/Linear sync) and the
local backup job sit behind compose profiles and an explicit opt-in:

```sh
ERIDANI_LOCAL_LIVE_WRITER=true docker compose -f compose.upgrade.yml --profile worker up   # own local data only
docker compose -f compose.upgrade.yml --profile backup up backup
```

---

# Legacy prototype (archived)

> **Not part of the current app.** Everything below describes the original J.A.R.V.I.S.
> voice prototype in `backend/`, `client/` and the GPU `docker-compose.yml` (Node.js
> middleware, Mem0/Qdrant memory, Kokoro/Orpheus TTS). It is unmaintained, is not deployed,
> and its memory reader has been retired from Eridani. Kept for reference only.

A real-time voice AI assistant you can actually talk to. It listens, thinks, speaks back, and remembers what you've discussed across conversations.

Built with a React frontend, Node.js middleware, and a Python backend that orchestrates STT, LLM, TTS, and long-term memory. Runs locally via Docker with GPU acceleration.

## Features

- **Voice conversation** — Speak naturally with voice activity detection (VAD); JARVIS listens, responds, and can be interrupted mid-sentence
- **End-of-utterance detection** — LiveKit's turn detector model decides whether you're done speaking or just pausing mid-thought, so the mic stays open when needed
- **Live transcription** — Partial transcripts appear word-by-word as you speak (local STT mode)
- **Dual STT engines** — Groq Whisper API (cloud, no GPU needed) or faster-whisper (local GPU, streaming partials)
- **Dual TTS engines** — Groq Orpheus (cloud, expressive emotion annotations) or Kokoro-82M (local GPU, high quality)
- **Long-term memory** — Remembers facts about you and its own past advice across sessions using Mem0 + Qdrant vector search
- **Token-aware context** — Rolling summarization keeps conversations going without hitting token limits
- **Text input** — Type messages when you don't want to talk
- **Conversational mode** — Auto-listen after each response for hands-free back-and-forth
- **Customizable personality** — Define JARVIS's behavior via system prompt
- **Runtime toggles** — Toggle TTS, conversational mode, and memory (full / read-only / off) from the UI

## Architecture

```
┌─────────────┐  WS + SSE  ┌──────────────────┐  stdin/stdout  ┌─────────────┐
│   React     │◀──────────▶│  Express Server   │◀──────────────▶│   Python    │
│   Client    │            │  (audio-server)   │               │  (JARVIS)   │
└─────────────┘            └──────────────────┘               └──────┬──────┘
      │                            │                                  │
      │ VAD + PCM capture          │ WebSocket relay (PCM → STT)      ├── Groq API (LLM + Whisper + Orpheus TTS)
      │ Live partial transcripts   │ SSE (TTS audio files)            ├── Kokoro-82M (local TTS on GPU)
      │ LLM response streaming     │                                  ├── faster-whisper + LiveKit EOU (local STT)
      │ WebSocket (interrupts)     │                          ┌───────┴───────┐
      │                            │                          │  Mem0 + Qdrant│
      │                     ┌──────┴──────┐                  │  (memory)     │
      └────────────────────▶│   Qdrant    │                  └───────────────┘
                            │  (vectors)  │
                            └─────────────┘
```

**Voice data flow (local STT):** Browser captures PCM via AudioWorklet → WebSocket to Express → relayed to faster-whisper STT server → partial transcripts and EOU scores stream back to browser in real time → on finalize, transcript sent to JARVIS.py via stdin → LLM streams response tokens → Kokoro generates TTS → audio pushed via SSE → browser plays.

**Voice data flow (Groq STT):** Browser VAD captures full utterance → WAV uploaded to Express → JARVIS.py transcribes via Groq Whisper → LLM streams response → TTS generated → audio via SSE.

## How End-of-Utterance Detection Works

When using local STT, JARVIS uses [LiveKit's turn detector model](https://huggingface.co/livekit/turn-detector) (`v0.4.1-intl`) to decide whether you've finished speaking or are just pausing mid-thought.

After each partial transcript, the model scores the probability that your utterance is complete. This score is combined with VAD (voice activity detection):

- **VAD ends + EOU agrees** → transcript finalized immediately, JARVIS responds
- **VAD ends + EOU disagrees** → mic stays open, waiting for you to continue
- **Timeout expires** (`EOU_TIMEOUT`) → finalized anyway so you're never stuck

The model runs on CPU (~25ms per inference) and uses per-language calibrated thresholds — for English the threshold is `0.011`, meaning any score above ~1% confidence signals a complete turn.

## How Memory Works

JARVIS maintains two categories of long-term memory:

- **User facts** — your preferences, goals, experiences, opinions (e.g., "Considering switching to Rust")
- **JARVIS facts** — advice, recommendations, and explanations JARVIS has given (e.g., "Suggested starting with the Rust Book")

Each conversation turn is processed in a background thread by Mem0, which uses an LLM to extract relevant facts and stores them as embeddings in Qdrant. On the next query, relevant memories are retrieved via semantic search and injected into the system prompt.

The two categories use separate extraction prompts (triggered by Mem0's `agent_id` parameter) so user facts and assistant facts don't get mixed up.

## Quick Start

### Prerequisites

- Docker with GPU support (NVIDIA Container Toolkit)
- A [Groq API key](https://console.groq.com)
- A [Hugging Face token](https://huggingface.co/settings/tokens) (for Kokoro or EOU model download, if using local modes)

### Setup

1. Copy `.env.example` to `.env` and fill in your API keys:
   ```bash
   cp .env.example .env
   ```

2. Start everything:
   ```bash
   docker compose up --build
   ```

3. Open **http://localhost:8080** in your browser

### Wiping Memory

If you need to clear all stored memories (e.g., after changing extraction settings):

```bash
docker compose down
docker volume rm jarvis_qdrant_data
docker compose up --build
```

## Environment Variables

### Required

| Variable | Description |
|----------|-------------|
| `GROQ_API_KEY` | Groq API key (LLM, Whisper STT, Orpheus TTS) |
| `HF_TOKEN` | Hugging Face token (Kokoro and/or EOU model download) |

### Speech-to-Text

| Variable | Default | Description |
|----------|---------|-------------|
| `STT_PROVIDER` | `groq` | `groq` (Groq Whisper API) or `local` (streaming faster-whisper on GPU with live partials) |
| `WHISPER_MODEL` | `medium.en` | faster-whisper model size when `STT_PROVIDER=local`: `tiny.en`, `base.en`, `small.en`, `medium.en`, `large-v3` |

### End-of-Utterance Detection

Only applies when `STT_PROVIDER=local`. Uses the [LiveKit turn detector](https://huggingface.co/livekit/turn-detector) ONNX model (~400MB RAM, ~25ms on CPU).

| Variable | Default | Description |
|----------|---------|-------------|
| `EOU_ENABLED` | `false` | Enable EOU detection. When disabled, JARVIS responds as soon as VAD detects silence |
| `EOU_THRESHOLD` | `0.011` | Minimum EOU confidence to treat an utterance as complete. `0.011` is the LiveKit-calibrated threshold for English |
| `EOU_TIMEOUT` | `3.0` | Seconds to wait after VAD silence before force-finalizing, even if EOU says the user isn't done |

### Text-to-Speech

| Variable | Default | Description |
|----------|---------|-------------|
| `TTS_PROVIDER` | `groq` | `groq` (Orpheus cloud) or `kokoro` (local GPU) |
| `TTS_ORPHEUS_VOICE` | `autumn` | Orpheus voice: `autumn`, `tara`, `leah`, `jess`, `leo`, `dan`, `mia`, `zac` |
| `TTS_ORPHEUS_SPEED` | `1.0` | Orpheus speed (`0.25`–`4.0`) |
| `TTS_KOKORO_VOICE` | `af_heart` | Kokoro voice preset ([full list](https://huggingface.co/hexgrad/Kokoro-82M)) |
| `TTS_KOKORO_SPEED` | `1.0` | Kokoro speed (`0.5`–`2.0`) |
| `TTS_KOKORO_LANG` | `a` | `a` = American English, `b` = British English |

### Memory

| Variable | Default | Description |
|----------|---------|-------------|
| `MEMORY_ENABLED` | `true` | Enable/disable memory system entirely |
| `MEMORY_DEVICE` | `cuda` | Embedding device: `cuda` or `cpu` |
| `MEMORY_LLM_MODEL` | `llama-3.3-70b-versatile` | LLM for memory extraction (must follow extraction prompts reliably) |
| `MEMORY_SEARCH_THRESHOLD` | `0.5` | Minimum relevance score for search results (`0.0`–`1.0`) |
| `MEMORY_SEARCH_LIMIT_USER` | `10` | Max user memories returned per query |
| `MEMORY_SEARCH_LIMIT_JARVIS` | `10` | Max JARVIS memories returned per query |

### Personality

The active deployment uses [Eridani / Eri's shared personality](apps/api/jarvis/personality.py)
for both text and voice. Edit that versioned source and rebuild the API/worker;
the retired `JARVIS_SYSTEM_PROMPT` environment setting no longer controls this deployment.

## License

MIT
