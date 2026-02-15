# J.A.R.V.I.S.

A real-time voice AI assistant you can actually talk to. It listens, thinks, speaks back, and remembers what you've discussed across conversations.

Built with a React frontend, Node.js middleware, and a Python backend that orchestrates STT, LLM, TTS, and long-term memory. Runs locally via Docker with GPU acceleration.

## Features

- **Voice conversation** — Speak naturally with voice activity detection (VAD); JARVIS listens, responds, and can be interrupted mid-sentence
- **Dual TTS engines** — Groq Orpheus (cloud, expressive emotion annotations) or Kokoro-82M (local GPU, high quality)
- **Long-term memory** — Remembers facts about you and its own past advice across sessions using Mem0 + Qdrant vector search
- **Token-aware context** — Rolling summarization keeps conversations going without hitting token limits
- **Text input** — Type messages when you don't want to talk
- **Conversational mode** — Auto-listen after each response for hands-free back-and-forth
- **Customizable personality** — Define JARVIS's behavior via system prompt
- **Runtime toggles** — Toggle TTS, conversational mode, and memory (full / read-only / off) from the UI

## Architecture

```
┌─────────────┐  WS + SSE  ┌─────────────────┐  stdin/stdout  ┌─────────────┐
│   React     │◀──────────▶│  Express Server  │◀──────────────▶│   Python    │
│   Client    │            │  (audio-server)  │               │  (JARVIS)   │
└─────────────┘            └─────────────────┘               └──────┬──────┘
      │                           │                                 │
      │ VAD + audio capture       │ ffmpeg (format conversion)      ├── Groq API (LLM + Whisper + Orpheus TTS)
      │ SSE audio playback        │ file watcher (TTS output)       ├── Kokoro-82M (local TTS on GPU)
      │ WebSocket (interrupts)    │                                 ├── Mem0 (memory extraction)
      │                           │                                 └── Qdrant (vector storage)
      │                           │
      │                    ┌──────┴──────┐
      │                    │   Qdrant    │
      │                    │  (vectors)  │
      │                    └─────────────┘
```

**Data flow:** Browser captures audio via VAD → WebSocket sends WAV to Express → Express pipes to Python via stdin → Python transcribes (Groq Whisper or local), queries memory, calls LLM, generates TTS → audio files pushed back via SSE → browser plays them in sequence.

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
- A [Hugging Face token](https://huggingface.co/settings/tokens) (for Kokoro model download, if using local TTS)

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
| `HF_TOKEN` | Hugging Face token (Kokoro model download) |

### Speech-to-Text

| Variable | Default | Description |
|----------|---------|-------------|
| `USE_LOCAL_WHISPER` | `false` | Use local Whisper model instead of Groq API |
| `REACT_APP_STT_MODE` | `api` | `api` (Groq Whisper) or `browser` (Transformers.js, works offline) |

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

| Variable | Default | Description |
|----------|---------|-------------|
| `JARVIS_SYSTEM_PROMPT` | *(friendly AI)* | System prompt defining JARVIS's personality and behavior |

## License

MIT
