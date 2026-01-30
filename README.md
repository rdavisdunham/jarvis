# J.A.R.V.I.S.

A voice-enabled AI assistant with real-time speech-to-text and text-to-speech capabilities. Talk to your AI companion through a modern chat interface with support for both voice and text input.

## Features

- **Voice Input** - Record audio messages that are transcribed using Groq's Whisper API (or local Whisper model)
- **Text Input** - Type messages directly in the chat interface
- **Text-to-Speech** - AI responses are spoken aloud using configurable TTS providers:
  - **Orpheus** (Groq) - Low-latency, expressive voices with emotion annotations
  - **Chatterbox** (DeepInfra) - Consistent voice with custom voice cloning support
  - **CSM** (DeepInfra) - Sesame's conversational speech model
- **Real-time Audio** - SSE-based audio delivery for instant playback as responses are generated
- **Customizable Personality** - Configure the AI's system prompt to your liking
- **TTS Toggle** - Enable/disable voice responses from the UI

## Architecture

```
┌─────────────┐     ┌─────────────────┐     ┌─────────────┐
│   React     │────▶│  Express Server │────▶│   Python    │
│   Client    │◀────│  (audio-server) │◀────│  (JARVIS)   │
└─────────────┘ SSE └─────────────────┘     └─────────────┘
                           │                       │
                           ▼                       ▼
                    ┌─────────────┐         ┌─────────────┐
                    │   ffmpeg    │         │  Groq API   │
                    │  (convert)  │         │ (LLM + TTS) │
                    └─────────────┘         └─────────────┘
```

## Quick Start

1. Copy `.env.example` to `.env` and configure your API keys
2. Run with Docker:
   ```bash
   docker-compose up --build
   ```
3. Open http://localhost:8080 in your browser

## Environment Variables

### Required

| Variable | Description |
|----------|-------------|
| `GROQ_API_KEY` | Your Groq API key for LLM, Whisper, and Orpheus TTS |

### Speech-to-Text

| Variable | Default | Description |
|----------|---------|-------------|
| `USE_LOCAL_WHISPER` | `false` | Set to `true` to use local Whisper model instead of Groq API (requires CUDA) |

### Text-to-Speech

| Variable | Default | Description |
|----------|---------|-------------|
| `TTS_PROVIDER` | `chatterbox` | TTS provider: `orpheus` (Groq), `chatterbox` (DeepInfra), or `csm` (DeepInfra) |
| `TTS_MAX_CHARS` | `1000` | Max characters to synthesize (longer text is truncated at sentence boundary) |

#### Orpheus (Groq) Options

| Variable | Default | Description |
|----------|---------|-------------|
| `TTS_ORPHEUS_VOICE` | `autumn` | Voice selection: `autumn`, `tara`, `leah`, `jess`, `leo`, `dan`, `mia`, `zac` |
| `TTS_ORPHEUS_SPEED` | `1.0` | Speech speed: `0.25` (slow) to `4.0` (very fast) |

#### DeepInfra Options

| Variable | Default | Description |
|----------|---------|-------------|
| `DEEPINFRA_TOKEN` | - | DeepInfra API token (only required for `chatterbox` or `csm` providers) |
| `TTS_VOICE` | - | Custom voice name for Chatterbox (create via DeepInfra's `/v1/voices/add` endpoint) |

### Personality

| Variable | Default | Description |
|----------|---------|-------------|
| `JARVIS_SYSTEM_PROMPT` | (friendly AI) | System prompt that defines JARVIS's personality and behavior |

### Frontend

| Variable | Default | Description |
|----------|---------|-------------|
| `REACT_APP_API_URL` | `http://localhost:3000` | Backend API URL (set at build time) |

## Orpheus Voice Annotations

When using the Orpheus TTS provider, you can include emotion annotations in the AI's responses. These are automatically hidden from the displayed text but affect the voice output:

- `[warmly]` - Warm, friendly tone
- `[laughs]` - Laughter
- `[sighs]` - Sighing
- `[whispers]` - Whispering
- `[excited]` - Excited tone

Example system prompt:
```
You are a friendly assistant. Use emotion annotations like [warmly], [laughs], or [excited] to make your responses more expressive.
```

## License

MIT
