# Try Gemini as Eri's task agent

Gemini 3.8 Flash can handle text chat and the task work delegated by GPT-Live.
The existing GPT-5.4 mini option remains the default when its key is configured.
The selection is saved for the owner, applies across devices on the next request,
and remains fixed throughout an in-progress request.

## Connect

1. Generate a Gemini API key in [Google AI Studio](https://aistudio.google.com/apikey).
2. Paste it into the existing blank `GEMINI_API_KEY=` row in the repository's
   ignored `.env` file. `JARVIS_GEMINI_API_KEY` is also supported.
   Keep the OpenAI key for voice, embeddings and automatic memory/note extraction.
3. Recreate the API and worker containers to load changed environment variables.
   A plain Docker restart does not reload an env file. From a PowerShell terminal
   in the repository:

   ```powershell
   docker compose --env-file .env.upgrade -f compose.upgrade.yml up -d --force-recreate api worker
   ```

4. Refresh Eri, then select **Settings → Task agent → Backend model →
   Gemini 3.8 Flash**. Switch back to GPT-5.4 mini in the same menu.

An option marked **API key needed** is unavailable until its key is loaded.
A loaded key does not prove model access or available quota; the first real check
verifies those. Credential and quota failures identify the provider in chat.
Eri never automatically changes providers after a failed request or repeats its
already committed task actions.

## Verify before using real work

From the WSL repository directory, run:

```bash
uv run python scripts/check_gemini.py
```

This opt-in check sends the full function catalog and synthetic conversation to
Gemini, requests a task-list call, supplies a synthetic result, and verifies the
continuation. It never executes the requested tools or touches a database,
Google Calendar or Linear. It makes two billable model calls.

Then try ordinary text requests and GPT-Live task delegation: find a known task,
create a disposable task, edit its date, complete it, and open its page. Compare
accuracy and latency with the OpenAI option. The real provider handshake and the first six-workflow/two-repeat acceptance
run passed on September 13, 2026. See
[the model comparison](BACKGROUND_MODEL_COMPARISON.md). Real-device voice and
connected-service acceptance remain separate checks.

## Scope and implementation

- Model: `gemini-3.8-flash`, reasoning effort `low`, up to 8,192 generated tokens
  per call to leave room for thinking and an answer.
- Google's OpenAI-compatible Chat Completions endpoint keeps the existing tool
  registry, owner/device authorization, revision checks, receipts and action limits.
  Both message and nested tool-call `extra_content` are returned unchanged during
  continuations, preserving Gemini's opaque thought signatures. These signatures
  stay in the active request's memory; they are not stored as conversation text.
- Settings choose only server-defined providers, endpoints and models. Keys stay
  on the server and never appear in bootstrap responses or browser storage.
- GPT-Live and Realtime remain the audio providers. Realtime executes the existing
  tools directly; GPT-Live delegates task work to the chosen backend model.
  Automatic memory extraction, note extraction and embeddings remain on OpenAI.
- Development cost tracking remains disabled. If re-enabled, Gemini uses its own
  published estimated rates, including the January 2027 pricing change, and its
  generated-token allowance. Completion usage includes thinking tokens.

Automated checks cover routing, missing credentials, persistent selection, signed
parallel/sequential tool calls, preserved receipts, mid-request model changes,
provider rejection/timeouts and disabled accounting. Desktop/mobile Settings
checks run in a disposable database with synthetic credentials.

References:
[Gemini 3.8 Flash](https://ai.google.dev/gemini-api/docs/models/gemini-3.8-flash),
[OpenAI compatibility](https://ai.google.dev/gemini-api/docs/openai),
[thought signatures](https://ai.google.dev/gemini-api/docs/generate-content/thought-signatures),
[pricing](https://ai.google.dev/gemini-api/docs/pricing).
