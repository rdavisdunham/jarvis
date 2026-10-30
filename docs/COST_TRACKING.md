# AI cost tracking

Tracking is independent of spending-limit enforcement. Development uses
JARVIS_COST_TRACKING_ENABLED=true and JARVIS_BUDGET_ENFORCEMENT_ENABLED=false.
Old uncertain reservations stay visible but cannot pause Eri in this mode.
Enabling enforcement restores monthly limits and optional-work thresholds.

Settings → System → Model usage shows rolling last-7-day and last-30-day totals
by feature, plus a calendar-month total for budgets. Costs come from reported
tokens or GPT-Live session seconds. Each Usage row stores tokens.feature;
concurrent feature contexts stay independent. Memory queries and extraction,
note organization, field/rule learning and search indexing have separate buckets.
Local deterministic work adds no AI charge. CPU/database/hosting costs are excluded.

Set JARVIS_COST_TRACKING_SINCE to an explicit UTC timestamp when recording resumes
after a gap. Incomplete periods are labeled and monthly projections are suppressed.
No historical charges are invented. Earlier usage without attribution stays in
an unclassified bucket. A future user-facing tracking toggle must update this marker.

Usage is an estimate, not a provider invoice. Credits, taxes, infrastructure and
isolated eval runs are excluded. Missing usage remains uncertain. Known usage
from an incomplete answer is recorded; repeated usage IDs cannot double-charge.

Luna Fast accounting added October 4, 2026. Interactive requests default to Fast;
scheduled learning and unqualified eval campaigns stay Standard. Usage metadata
records requested/served tiers and whether the tier price is estimated. Fast
reservations use premium rates; a returned Standard tier is charged at Standard
rates. Stored amounts retain the database's existing six-decimal precision.

Configured prices (Luna Fast checked October 4; other rates last verified September 19):
- Luna Standard: $0.20/M input, $0.02/M cached input, $0.25/M cache writes, $1.20/M output.
  Fast: $0.40/M input, $0.04/M cached input, $0.50/M cache writes, $2.40/M output.
  [Fast tier behavior](https://developers.openai.com/api/docs/guides/fast-mode).
  Above 272K input, input rates double and output is 1.5 times.
  https://developers.openai.com/api/docs/models/gpt-5.6-luna
- Gemini 3.8 Flash: $0.75/M input and $3.75/M output through 2026, then $1.50/$7.50.
  Estimates conservatively do not deduct Gemini caching.
  https://ai.google.dev/gemini-api/docs/latest-model
- GPT-Live 1: $0.05/minute, billed by the second, plus backend usage.
  https://developers.openai.com/api/docs/models/gpt-live-1
- text-embedding-3-small: $0.02/M tokens, unchanged configured rate.

Paid evals use disposable synthetic databases with app accounting disabled.
Their separate spending.json journal and committed summary keep eval cost out of
normal usage. The September 19 20-case run cost about $0.52 combined. The full
acceptance catalog is not a claim every feature has been paid-evaluated.

Validated through isolated budget tests, concurrent attribution, incomplete and
missing usage, frontend checks and phone/fold/desktop browser coverage.

## Model map

`apps/api/jarvis/model_map.py` names every backend job that calls a text model, the
cost feature its usage is recorded under (the Model usage buckets above), the
production call, and its model profile. Defaults are today's models, so nothing
changed: Luna with low reasoning through `extraction_request` for every structured
job, and the Settings task agent (Luna at the Fast tier by default) for Eri chat.

| job type | cost feature | production call | default |
|---|---|---|---|
| session_summary | session_ledger | agent_sessions.summarize_text | luna |
| routing_review | organization_learning | routing.process (weekly review) | luna |
| field_understanding | field_understanding | routing.assess | luna |
| run_judge | agent_control | agent_judge.ask | luna |
| run_summary | agent_summaries once run_summaries lands (agent_control until then) | run_summaries.ask | luna |
| memory_extraction | memory_learning | memory_learning.process | luna |
| note_tasks | note_tasks | notes.suggest_tasks | luna |
| swarm_plan | agent_control | swarm.ask (Play on a project: which tasks run in parallel) | luna |
| eri_chat | assistant | conversation (tool loop) | Settings choice |

`route(job, prompt_tokens)` is the router hook: `RULES` (empty) are checked in
order by job type and prompt length, first match wins, else `MODEL_MAP`. Production
reaches it through `memory_learning.profile_request(owner, job, payload, ...)` and
`model_map.agent_model(job, prefs)`; `swarm.ask` is the first call site (the task notes name Fable as
the intended planner; it needs an Anthropic route the budget ledger prices before the map can point there). Both refuse any
route that is not today's model, exactly as `extraction_request`'s allowlist does.
`model_map.attributed(job)` records usage under the job's cost feature.

Before a non-Luna profile can serve production traffic it needs: a provider
transport with the same budget reservation as `provider_request` (OpenAI only
today), its rates in `agent_models` so the ledger prices its usage, and its
refusal and structured-output handling mapped to the same non-verdicts. The
request builders and parsers in `model_map` (`build`, `parse`) already cover the
Responses API, the Anthropic Messages API and OpenAI-compatible chat.

A rule for the likely first switch would look like this (do not add it until a real
run supports it):

~~~python
RULES = (Rule("haiku-5.5", jobs=("session_summary", "run_summary"),
              max_prompt_tokens=100_000, name="short-summaries"),)
~~~

The token bound keeps long prompts off Haiku's long-prompt rates (see prices below).

## Model battle royale (roadmap task f9040e0a)

Question: per job type, the cheapest model that stays smart. Method:

- Cases: `evals/models/cases/*.json` (schema in `evals/models/schema.py` and
  `schema.json`), fixed inputs from real Eridani job types: session summaries (6),
  the weekly routing review (5), Eri chat tool selection (10), the run judge (13,
  reused from `evals/agent_control`), run-card summaries (6). Each case has an
  `expect` block and a `stub`, a correct answer in the production response shape;
  the validator checks every stub against the production Pydantic model.
- Requests: the production builders (`agent_sessions.summary_request`,
  `routing.infer_request` with `REVIEW_PROMPT`, `agent_judge.request`, Eri's static
  instructions plus real tool definitions; run summaries use `run_summaries.request`
  when that module is on main and a copy of its summary-v1 prompt until then).
- Graders: deterministic first. Summaries: length, required mentions, forbidden
  text (prompt-injection cases). Routing: `routing.process`'s own filters (home
  exists, phrase matches an example, every match agrees) plus required finds.
  Eri chat: the first tool call. Judge: the final verdict after `agent_judge.decide`
  (hard rules and completion policy), with the model's own verdict reported as
  "model alone". `--grader <profile>` adds a rubric verdict for the text jobs;
  grader spend is reported separately and never counted as job cost.
- Metrics per job type and profile: pass rate, non-verdict rate (refusal, invalid
  or truncated output, transport error), refusal rate, cost per completed job (all
  spend on that job, refusals included, divided by usable answers), p50/p95
  latency, mean tokens.
- Decision: the cheapest profile whose pass rate is within 5 points of today's
  model, whose non-verdict rate is no more than 5 points worse, and, for the judge,
  with no more false completions and a model-alone pass rate within 5 points.
  Fewer than 10 runs per job is flagged as too few to decide.

Reference prices (USD per 1M tokens; the script's dated `PRICES` table):

| model | input | output | notes | checked |
|---|---|---|---|---|
| GPT-5.6 Luna (Standard) | 0.20 | 1.20 | cached input 0.02; above 272K input x2, output x1.5 | 2026-10-28 (task) |
| GPT-5.6 Luna (Fast) | 0.40 | 2.40 | Eri chat's default tier | 2026-10-04 |
| Claude Haiku 5.5 | 0.10 | 0.50 | above 100K-token prompts: the task says doubling ($0.20 / $1.00); Anthropic's model table lists $0.50 / $2.50, which the script uses until one is confirmed | 2026-10-28 (task) |
| Claude Sonnet 5.5 | 2.00 | 10.00 | cached input 0.20; not in the repo's docs, from Anthropic's model table (claude-api reference cached 2026-10-06): verify before deciding | 2026-10-30 |
| DeepSeek, Kimi, MiniMax | placeholder | placeholder | set `MODEL_EVAL_<ID>_MODEL`, `_ENDPOINT`, `_PRICE="in,out"` | none |

### Run it

Free, deterministic, no network (CI runs this through `tests/test_model_eval.py`):

~~~sh
uv run --no-sync python scripts/model_eval.py --stub
~~~

The real comparison (paid). Put `OPENAI_API_KEY` (or the app's
`JARVIS_OPENAI_API_KEY`/.env) and `ANTHROPIC_API_KEY` in the environment, check the
plan, then run:

~~~sh
uv run --no-sync python scripts/model_eval.py --plan --repeats 3 --profiles haiku-5.5,sonnet-5.5 --grader sonnet-5.5
uv run --no-sync python scripts/model_eval.py --repeats 3 --profiles haiku-5.5,sonnet-5.5 --grader sonnet-5.5 --max-usd 5
~~~

The plan prints about 360 job requests, typically about $1.70 for the jobs (Sonnet is
most of it) plus under $1 of Sonnet rubric grading (about 108 short grades). `--max-usd` stops the run
before any request whose worst case would cross it; the report is written either
way (exit 2 when stopped early). Add placeholder providers with their
`MODEL_EVAL_*` variables and keys and list them in `--profiles`.

### Read the results

`evals/models/results/<date>.md` (and the full `<date>.json`): one table per job
type, then the decision table, then every miss with its reason. Read in order:
non-verdicts first (a model that refuses or breaks the schema costs retries and
surfaces as needs_you), then pass rate against today's row, then cost per
completed job. For the judge, any false completion disqualifies a candidate; a
low "model alone" rate means the hard rules are doing the work. A pick on fewer
than 10 runs, or on a job with a gap under 5 points, needs more repeats before
changing `MODEL_MAP`. Stub files are `<date>-stub.*` so they never replace real
numbers.

### Decision table (stubbed until Davin runs it for real)

From `evals/models/results/2026-10-30-stub.md`. The fake provider replays each
recorded answer with simulated misses per profile: this shows the format, not
an outcome. Costs use the real prices and the stub's token estimates.

| job | current | stub pick | pass (current → pick) | $ per job (current → pick) | decision |
|---|---|---|---|---|---|
| session_summary | luna | haiku-5.5 | 100% → 100% | $0.000326 → $0.000118 | pending real run |
| routing_review | luna | haiku-5.5 | 100% → 100% | $0.001319 → $0.000613 | pending real run |
| eri_chat | luna-fast | haiku-5.5 | 100% → 100% | $0.003454 → $0.000807 | pending real run |
| run_judge | luna | haiku-5.5 | 100% → 100% | $0.000503 → $0.000204 | pending real run |
| run_summary | luna | haiku-5.5 | 100% → 100% | $0.000322 → $0.000118 | pending real run |

### Hosted router or two fixed models

A hosted router (OpenRouter-style: one OpenAI-compatible endpoint, per-request
model choice or automatic routing) against two fixed models chosen per job type by
`model_map`:

- Cost: our jobs cost fractions of a cent each, so a router's platform fee is
  small in dollars; the bigger cost is unpredictability. Automatic routing changes
  the served model per request, so cost per completed job (and the budget
  reservation's upper bound) is no longer a property of the job type.
- Latency: one extra network hop. Irrelevant to the background jobs (judge,
  summaries, weekly review); noticeable only on Eri chat, which already makes
  several model rounds per turn.
- Caching: provider prompt caches are per model. Eri chat's stable prefix
  (instructions and tool definitions, about 7K tokens, kept byte-stable so it
  stays in the provider cache) only pays off when consecutive rounds hit the same model; a
  router that switches models forfeits it. Fixed models keep caches warm.
- Failure modes: a router is a single point of failure in front of every model,
  where two direct providers fail independently and can back each other up.
  Structured-output strictness, refusal semantics and usage reporting differ by the
  model it picks, which turns into invalid output and uncertain charges. Eri chat
  replays Responses-native reasoning items (`reasoning.encrypted_content`), which an
  OpenAI-compatible chat endpoint does not carry. Prompts (session excerpts, task
  titles) also pass through one more processor with its own retention terms.

Recommendation: two fixed models through `model_map` (today's Luna plus the
cheapest model that passes per job in the real run), with the rule hook for prompt
length. Reconsider a hosted router only to reach more than two or three providers,
and then only as a fallback path for background jobs, not for Eri chat.
