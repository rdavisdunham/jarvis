# Background model candidates

Checked September 12, 2026 against official model and pricing documentation.
Update September 13: Gemini 3.8 Flash is integrated as an optional task agent in
Settings; GPT-5.4 mini remains the default. See [GEMINI_SETUP.md](GEMINI_SETUP.md).
Real Gemini and Luna checks completed September 13. The Gemini key is loaded
in API/worker; production still defaults to GPT-5.4 mini.

Standard text prices, USD per million tokens, uncached input / output:

- GPT-5.4 mini: $0.75 / $4.50. Current proven integration.
  https://developers.openai.com/api/docs/models/gpt-5.4-mini
- GPT-5.6 Luna: $0.20 / $1.20. OpenAI positions it in the nano-like,
  cost-sensitive tier; a newer name does not establish superior accuracy.
  https://developers.openai.com/api/docs/models/gpt-5.6-luna
- GPT-5.6 Terra: $2 / $12. OpenAI positions it in the balanced, mini-like tier.
  https://developers.openai.com/api/docs/models/gpt-5.6-terra
- Gemini 3.8 Flash: $0.75 / $3.75 through December 31, 2026, then $1.50 / $7.50
  starting January 1, 2027. Google lists output as including thinking tokens.
  https://ai.google.dev/gemini-api/docs/pricing
  https://ai.google.dev/gemini-api/docs/models/gemini-3.8-flash

Illustration: 1,000 calls using 10,000 uncached input and 1,000 billed output
tokens each cost $12 with mini, $3.20 with Luna, $32 with Terra, or $11.25 with
Flash's current introductory price ($22.50 at its announced standard price).
These are equal-token illustrations, not measured Eridani costs. Reasoning,
retries, tool rounds, caching and provider tokenization change actual cost.
Luna/Terra also have higher rates for prompts above 272K input tokens.

Recommendation: evaluate Luna first for bounded extraction, classification,
summaries and routine task operations. Keep 5.4 mini as the working baseline.
Consider Terra for difficult planning/escalation if measured improvements justify
its cost. Flash is now available for evaluation through the shared tool loop. Its
introductory pricing should not be treated as permanent.

An evaluation should measure cost per correct completed workflow, not only token
price: exact record resolution, safe multi-task edits, date/time interpretation,
many-to-many goal/project changes, evidence-grounded note extraction, memory
contradiction handling, tool-result accuracy, latency, retries and failures.
Use synthetic or explicitly approved cases and prohibit writes to real external
calendars/issues during comparative tests. The first live-provider acceptance comparison is recorded below. It is a small
functional check, not a general intelligence ranking or production benchmark.


## First real-provider acceptance — September 13, 2026

Both requested model IDs were returned by their actual APIs. Six synthetic
workflows ran twice per model through Eri's existing conversation, tools and
domain commands in a disposable PostgreSQL database:

- Create a timed task with an earlier reminder on that same task.
- Complete three matching project tasks while preserving unrelated tasks.
- Link a project to two goals while preserving an existing project/goal link.
- Create a note with exact content and multiple project/goal links.
- Ask for clarification between two identically named tasks.
- Use injected memory and open the exact task, ignoring an instruction embedded
  in the synthetic task's imported notes.

Gemini 3.8 Flash, low reasoning: **12/12 clean passes**, correct final records
in all 12. Mean complete-workflow latency **6.77 seconds**, median 6.68 seconds,
range 2.01–12.04 seconds. 39 model calls in total.

GPT-5.6 Luna, reasoning none: **11/12 clean passes**, correct final records in
all 12. Mean complete-workflow latency **5.70 seconds**, median 5.82 seconds,
range 3.16–8.15 seconds. 35 model calls in total. On one note creation it
mistyped a project ID; Eri rejected the invalid write and Luna corrected it.
The app retained a partial status because the turn contained a tool error,
even though the requested note and all links were ultimately correct.

Luna initially returned HTTP 400 with function tools and reasoning low on
Chat Completions. The provider error explicitly requires either reasoning none
on that endpoint or the Responses API for reasoning-enabled tool calls. Plain
Luna text with low reasoning succeeded. This is an endpoint compatibility limit,
not an account-access failure. Reasoning-enabled Luna through Responses remains
untested and is not implemented by the production adapter.

For a normalized price comparison, applying uncached standard rates to the
measured tokens yields about **$0.087 for Luna** and **$0.547 for Gemini** across
the 12 workflows each. Those are not billed totals: actual usage includes cache
hits, Luna cache-write premiums, and diagnostic/preliminary runs not included
here. Production cost tracking remained disabled throughout. The published rates
and their sources are listed above.

Interpretation: Gemini was cleaner in this small trial; Luna was somewhat faster
and has a materially cheaper price schedule. Both are credible candidates for
real-use testing, but twelve small cases cannot establish reliability over long
conversations, unusual task edits, voice timing or connected-service writes.
For the next immediate real-use trial, Gemini is ready in Settings. Keep the
5.4-mini baseline available. Evaluate Luna through Responses if we want to compare
its reasoning-enabled behavior before choosing a long-term default.

## Evidence and limits

[Machine-readable results and synthetic tool traces](evals/task-agents-2026-09-13.json)
include per-case grades, returned model IDs, latency and usage. Reproduce with:

```bash
uv run python scripts/evaluate_task_agents.py --repeats 2
```

The final run used the same six cases, two repetitions, 8,192 generated-token cap,
12 model rounds and 30 tool calls per request for each candidate. Model order
alternated by repetition. Both received the real Eri prompt and full function
catalog, but only local synthetic operations were permitted. Memory was injected
from a synthetic fact; optional memory search used its canonical lexical record.
UI acknowledgements were simulated. Embedding quality, memory extraction, browser
rendering, audio behavior, Google/Linear writes and long conversations were not
evaluated here. No worker ran against the test database, which was removed.

Diagnostic runs are excluded: the incompatible Luna reasoning setting, and a
fixture that initially blocked a legitimate memory_search call. After correcting
that fixture, all cases were rerun for both models. The ID-copying mistake above
occurred in the final corrected run and remains recorded; it was not rerun away.

API/worker are healthy with the Gemini key loaded. The live agent is still
gpt-5.4-mini; no production model switch occurred. Existing task/memory and
historical cost-ledger counts remained unchanged.
