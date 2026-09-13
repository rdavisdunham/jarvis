# Reasoning-enabled Luna

Select **Settings → Task agent → Backend model → GPT-5.6 Luna · reasoning**.

Luna uses the existing server-side OpenAI API key. No new credential is needed.
The selection applies to text chat and the task work GPT-Live delegates, starting
with the next request. GPT-5.4 mini is retired; its saved OpenAI selection now
resolves to Luna. Gemini remains available as the other tested backend.

The Luna profile uses `gpt-5.6-luna` through `/v1/responses`, with low reasoning
effort and an 8,192-token generation allowance (including reasoning). Reasoning is
enabled even though the UI shows only Eri's answer and saved actions.

## How it fits

The Responses adapter translates the existing conversation/tool format at the
provider boundary. It replays complete native output items, including encrypted
reasoning, followed by tool outputs linked by `call_id`. It uses `store: false`;
native output and reasoning state remain only in the active server request.
They are not written to the transcript, memories or durable job results.

The loop starts with eighteen common read/navigation/task tools and loads other groups on
demand. Responses uses strict mode for compatible read schemas; sparse updates
explicitly remain non-strict to preserve omitted fields versus deliberate nulls.
Server validation remains authoritative. Eri's authorization, revision checks,
action receipts, action limits and no-replay recovery still run in the same loop.
An incomplete model response cannot launch partially generated tool calls.

The model choice is now a stable `agent_profile` separate from `agent_provider`.
Luna uses the OpenAI provider. Older saved profile/provider values of openai
resolve to Luna; no database migration is needed. The chosen profile is pinned throughout each active request.

Voice audio uses GPT-Live. Automatic memory and note extraction now use Luna
Responses with low reasoning and strict structured output. Embeddings remain
text-embedding-3-small. Development cost recording remains disabled. If it is
re-enabled later, the Luna route accounts for its own rates, cached input,
cache-write premiums and output usage including reasoning.

## Earlier Luna rollout evidence (before tool refinement)

- 232 backend tests, one optional skip; 76 frontend tests; production build, Ruff
  and desktop/mobile Settings checks passed.
- Real Luna Responses calls completed 12 synthetic workflows without a tool
  error. The API reported 470 reasoning tokens across those runs, with an average
  complete-workflow time of 7.12 seconds.
- One original grade compared deadline strings and flagged a valid explicit UTC
  offset (`14:00-06:00`, Chicago in January). The actual deadline was correct.
  The grader now compares instants; an additional fresh timed-task/reminder run
  passed. The original raw grade and the follow-up are both retained.
- Gemini's real function-catalog/continuation check also passed after this change.
- Deployed API/worker are healthy. Authenticated bootstrap exposes Luna with low
  reasoning; HTTPS serves `index-Cnils54F.js`. Saved-record counts are unchanged
  and GPT-5.4 mini remains selected.
- Native reasoning, parallel/sequential function calls, call-ID correlation,
  cancellation, timeout recovery, incomplete responses, profile compatibility
  and disabled accounting have automated coverage.

Reproduce the live synthetic test in a disposable database:

```bash
uv run python scripts/evaluate_task_agents.py --models gpt-5.6-luna --repeats 2
```

The default Luna path in that script is now Responses with low reasoning.
`--luna-api chat_completions` retains the earlier reasoning-none baseline.
`--cases timed_task` narrows a follow-up. UI acknowledgements are simulated and
memory context is synthetic; real-device audio and connected-service acceptance
remain separate checks.

[Detailed results](evals/luna-reasoning-2026-09-13.json) preserve the evidence.
[Earlier model comparison](BACKGROUND_MODEL_COMPARISON.md) remains available.

References: [Luna](https://developers.openai.com/api/docs/models/gpt-5.6-luna),
[reasoning state](https://developers.openai.com/api/docs/guides/reasoning),
[function calling](https://developers.openai.com/api/docs/guides/function-calling).
