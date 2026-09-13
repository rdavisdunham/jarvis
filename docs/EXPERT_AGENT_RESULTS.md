# Expert task-agent comparison — September 13, 2026

Gemini completed more of these difficult Eridani workflows and was faster.
Luna was substantially cheaper at published uncached rates. I lean toward
Gemini for the next primary-agent trial, but this suite does not establish a
general winner: the supplementary check of correct state plus factual replies
tied. The production selection remains GPT-5.4 mini.

Open the [interactive report](evals/expert-agent-report-2026-09-13.html) to
filter all 144 trials and inspect conversations, tool calls, results and grades.

## Measured results

- **Gemini 3.8 Flash, low reasoning:** 71/72 workflows completed correctly
  (65 clean, six recovered); 23/24 scenarios passed all three repetitions.
  Mean workflow time 6.65 seconds, median 5.61, 90th percentile 10.72.
- **GPT-5.6 Luna, low reasoning through Responses:** 69/72 workflows completed
  correctly (61 clean, eight recovered); 22/24 scenarios passed all repetitions.
  Mean workflow time 10.17 seconds, median 8.37, 90th percentile 16.06.
- Neither model made an unrequested saved change, including temporary changes
  later undone. All planned trials completed with the requested model IDs;
  there were no skipped trials, infrastructure errors or automatic regrades.
- The supplementary unblinded Codex review found **68/72 trials per model**
  with both correct state and confirmed factual replies. Gemini had one further
  correct-state trial with an uncertain follow-up promise. This model-based
  qualitative review is separate from the deterministic scores.

Workflow time includes the actual conversation/tool loop and its per-tool state
audits, excluding fixture setup and final grading. Mean provider-only time was
6.46 seconds for Gemini and 9.98 for Luna. These are measurements from this run,
not latency guarantees.

The paired completion difference was 2.78 percentage points in Gemini's favor.
A 10,000-sample bootstrap resampling whole scenarios, retaining their three
repetitions, gives a 95% interval of 0 to 8.33 points. It includes zero and
describes these 24 authored scenarios, not all future requests.

## What failed

**Luna:** in bulk selection repeat 2 it updated 22 of 24 eligible tasks; in
repeat 3 it updated 23 of 24. The task list spans 126 records with independent
project, status, assignee, tag and deadline decoys. In evidence extraction
repeat 1 it copied a note UUID incorrectly, received NOT_FOUND, found the correct
note again, but stopped without retrying or creating the requested tasks.
Its explanation that the note had become unavailable was unsupported.

Luna also gave incorrect completed counts in bulk repeats 1 and 2. Repeat 1
saved all 25 eligible records but claimed 26. Repeat 3 accurately reported 23
writes but did not explain that the requested scope was unfinished.

**Gemini:** in evidence extraction repeat 3 it created the two correct tasks
and copied the exact supporting quotes into their text, but used task_create
instead of note_tasks. The required structured source-note links were missing.

Gemini's DST-gap replies in repeats 1 and 2 suggested 2:00 AM Chicago as a valid
alternative even though it is also inside that day's missing hour. The final
clarified 3:30 AM deadlines were correct. One pending-calendar reply promised
monitoring without clearly distinguishing durable server retry from an assistant
follow-up; the review records that claim as uncertain.

Both models passed the injected lost-acknowledgement and concurrency recovery,
prompt-injection, ambiguous-reference, unavailable-capability, infeasible-plan
and unsaved-editor checks. That is evidence for these specific scenarios, not
a general safety guarantee.

## Price interpretation

Applying published uncached rates to the measured usage gives **$0.68 for Luna**
and **$3.66 for Gemini**, across 72 scored trials each. Including failed attempts,
that is approximately $0.0098 versus $0.0515 per successful workflow.

These are normalized rate equivalents, not invoices. Cache discounts, cache-write
premiums and diagnostic calls are excluded from this comparison. Production cost
recording remains disabled.

Luna used 3,239,282 input and 24,084 output tokens, including 6,665 reported
reasoning tokens. Gemini used 4,741,991 input and 27,165 output tokens; separate
reasoning usage was unavailable. Reasoning is not charged a second time.

Rates checked September 13: Luna $0.20 input / $1.20 output per million tokens;
Gemini $0.75 / $3.75 under introductory pricing through December 31, 2026.
Sources: [Luna documentation](https://developers.openai.com/api/docs/models/gpt-5.6-luna)
and [Google pricing](https://ai.google.dev/gemini-api/docs/pricing).

## Tool improvements worth testing next

The [tool-improvement backlog](AGENT_TOOL_IMPROVEMENTS.md) retains seven candidates
with concrete trace references and validation requirements. Start with exact
task filters and stable selections, authoritative batch counts, and clearer
source-note provenance. Then address malformed record references, DST resolution,
remote retry semantics and relationship revisions.

These are hypotheses to test, not proof that the tools caused the mistakes.
Keep this baseline, make narrow changes, and compare both models again with
additional independently authored, held-out scenarios.

## Evidence, verification and limits

The [protocol](EXPERT_AGENT_EVALUATION.md) documents the 24 scenarios, fixed
settings, isolation and grading. Real model APIs used the production tool catalog
and real domain commands in disposable PostgreSQL databases. Google/UI responses
were simulated; memory was seeded and optional note search used a lexical fixture.
Voice, actual Google/Linear writes, automatic memory learning, embedding quality
and long real conversations were not tested.

A separate Codex subagent reviewed every trial and turn under a stated rubric.
It saw model identities, can miss problems, and was neither an independent human
reviewer nor a separate paid judge call. Its interpretation and revision history
are retained without replacing automatic grades.

- [Machine-readable summary](evals/expert-agent-summary-2026-09-13.json)
- [Complete scored traces](evals/expert-agents-2026-09-13.json.gz)
- [All 72 expected-state fixtures](evals/expert-fixtures-2026-09-13.json.gz)
- [Full qualitative review](evals/expert-response-review-2026-09-13.json)
- [Integrity checks and artifact hashes](evals/expert-integrity-2026-09-13.json)
- Diagnostics: [initial infrastructure failure](evals/expert-diagnostic-v1-2026-09-13.json.gz)
  and [corrected eight-trial preflight](evals/expert-diagnostic-v2-2026-09-13.json.gz).
  The first blocked all eight attempts locally because of a DNS argument type
  in the isolation guard; its original safety labels describe that harness
  defect, not model behavior. No provider response or usage was recorded.
  The corrected preflight included a genuine Luna bulk
  omission; diagnostic outcomes and costs are excluded from the scored run.

All 72 paired fixtures and first system prompts matched. Exported fixture hashes
match all 144 trials. Credentials and native reasoning were excluded, and the
disposable databases were removed. The evaluation suites passed 115 tests
(80 scenario/oracle, 22 runner, 13 summary tests); scoped Ruff checks and the
report's desktop/mobile layout and filters passed.

This change adds evaluation code, tests and documentation. It makes no production
tool, model-selection or deployment change.
