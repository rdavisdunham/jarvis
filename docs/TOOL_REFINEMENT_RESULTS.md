# Tool refinement: matched expert regression

The new tool interface is smaller and the seven earlier tool defects are fixed.
The selected twenty expert scenarios ran three times with each backend model:
120 scored trials. Every fixture and question matches the corresponding original
baseline. This is a development regression after several iterations on these
cases, not a held-out benchmark.

[Interactive report and full traces](evals/tool-refinement-report-2026-09-13.html) ·
[Research and implementation](TOOL_DESIGN_RESEARCH.md) ·
[Fixed evaluation protocol](TOOL_REFINEMENT_EVAL_PLAN.md)

## Results

After explicit review of three phrase-grader false negatives, Gemini completed
58/60 workflows and Luna 57/60. The original automatic scores remain 55/60 and
57/60, respectively. Both models made supported final statements in 58/60 trials;
a truthful admission of an unfinished task does not count as task completion.
The combined task-and-supported-reply counts after review are also Gemini 58/60
and Luna 57/60.

On the exact same original sixty-trial subsets, Gemini previously completed
59/60 and Luna 57/60. Their earlier combined task-and-supported-reply counts were
56/60 each. This supports a context-efficiency improvement and several specific
repairs, not a claim that overall task accuracy improved for both models.

The original native outcome breakdown is Luna 49 clean, eight recovered, three
failed; Gemini 48 clean, seven recovered, five failed. Three Gemini failures are
explicitly adjudicated separately: it correctly declined to infer free time in
two unavailable-calendar trials and correctly explained an infeasible schedule
in another. No original check, question, state oracle or score was rewritten.

## Remaining failures

- Gemini twice found a feasible four-block schedule but left an avoidable
  fifteen-minute gap, then claimed the schedule was optimal.
- Luna once used invalid availability parameters repeatedly, fell back to a
  calendar lookup with no connected source, and saved a schedule overlapping the
  fixture's busy interval. The fixture-validation and source-consistency limits
  below matter to interpreting this failure.
- Luna once omitted the requested separate queued-write status lookup and
  attributed our polling recommendation to Google. It correctly described the
  event as queued and did not invent an assistant monitoring job.
- Luna once copied an unchanged goal UUID incorrectly while editing a note.
  The atomic write rejected the nonexistent link and preserved all records.
  Luna reported the failure truthfully but did not correct that link and finish.

Exact bulk edits and note-evidence extraction passed all six trials each.
The retained scope, concurrency, retry, DST, relationship, injection and refused
navigation cases remained covered. No trial changed records outside its allowed
scope; a poor schedule within the authorized calendar writes is still a real
task failure.

Next tool work should add a planning preflight for availability, durations and
dependencies, and field-specific invalid-link errors that help a model repair a
sparse note edit without re-copying every unchanged relationship. Fresh held-out
questions should accompany that work.

## Context, latency and cost

The initial catalog changed from 78 tools to eighteen, with 81 available across
thirteen discoverable groups. Compact initial schema bytes fell from 51,310 to
20,833 (59.4%); static backend instructions fell from 12,789 to 4,532 (64.6%).
These are UTF-8 bytes, not estimated tokens.

Across the matched sixty trials per model:

- Luna input tokens fell from 2,830,047 to 1,617,128 (42.9%). Mean task time was
  8.63 seconds, previously 10.73. The uncached-rate equivalent was $0.3452,
  previously $0.5920.
- Gemini input tokens fell from 4,210,913 to 1,985,813 (52.8%). Mean task time was
  6.51 seconds, previously 7.10. The uncached-rate equivalent was $1.5471,
  previously $3.2530.

Discovery added 51 loader calls for Luna and 25 for Gemini. Provider requests rose
from 200 to 258 for Luna and from 207 to 234 for Gemini. Offered tools ranged from
18 to 34. Deferring schemas trades some extra turns for less context; it does not
make discovery free.

These costs are published uncached-rate equivalents, not invoices or a promise
of equivalent bill savings. Cached-token reporting and cache behavior differ
between providers and runs; Gemini cache fields are absent on many new requests.
Latency was measured in separate before/after runs, so provider variability also
matters. Development attempts and probes are additional usage excluded from the
scored totals.

## Fixture fidelity and audit limits

Every response was reviewed by a Codex subagent with access to model identity and
expected state. That review is unblinded. Raw automatic grades and explicit
review overrides are both retained.

The scored fixture read validator returned a generic invalid-argument message.
Production now returns the particular schema error, such as a minimum value.
A separate six-trial diagnostic tests the corrected validation feedback using the
same three scheduling fixtures and both models; its scores do not replace or
enter the 120-trial result.

The scheduling fixture's availability adapter supplies a known connected
calendar and busy window. Its fallback real calendar-list route sees no real
Google connection. Those paths give inconsistent source context; Luna did not
receive the busy window in the failed repeat after its availability calls were
rejected. The resulting schedule is incorrect, but this is not evidence that it
ignored busy information it had actually seen.

The queued-write adapter returns immediately. Production's create tool also
polls local status briefly before returning. The original oracle requires a
separate status-read call, an implementation-specific condition. Its native
failure remains visible; the unsupported attribution to Google is independently
a factual error.

Google, Linear and external writes are synthetic in this suite. Desktop/mobile
browser smoke checks do not establish microphone, wake-word, audio recovery or
real connected-service behavior.

## Separate validation-feedback diagnostic

With the fixture returning the same detailed validation error as production,
both models completed all three scheduling fixtures and gave supported replies.
Luna initially requested an invalid one-minute minimum in each trial, then
corrected it to five immediately after the precise error and finished the optimal
schedule. Gemini completed all three cleanly without encountering that error;
its improvement cannot be attributed to validation feedback it never saw.

This small rerun supports the usefulness of actionable errors for recovery. It
is not a replacement benchmark or proof of reliable scheduling. The original
120 scores and review entries are byte-identical. The fallback calendar-source
inconsistency remains documented. The diagnostic consumed 114,369 input and
3,225 output tokens for Luna; 204,618 input and 2,856 output for Gemini, including
reasoning where reported. Raw traces, methodology and integrity checks are
archived separately.

## Evidence and reproduction

The original `expert-*` artifacts remain unchanged. New `tool-refinement-*`
artifacts retain the complete final run, sixty paired fixtures, manual audit,
comparison, source integrity, byte measurements, provider schema acceptance and
synthetic extraction checks. The aborted strict-schema run and partial
development run remain separate, including their failures. Recorded usage from
the first interrupted attempt may omit its unfinished in-flight trial.

To reproduce the scored protocol, run the command in the evaluation plan.
To regenerate this report from the retained evidence:

```sh
.venv/bin/python scripts/compare_tool_refinement.py \
  docs/evals/tool-refinement-agents-2026-09-13.json.gz \
  --after-review docs/evals/tool-refinement-response-review-2026-09-13.json
```

The full traces include synthetic prompts, tool arguments, receipts and observed
effects. Credentials, native reasoning content and encrypted continuation state
are excluded.


## Release verification

The exact evaluated application source is deployed: nineteen source hashes match
the running API image, and HTTPS serves index-DCL2bqzL.js. API, worker and
PostgreSQL are healthy; the backup service remains running. Active selection
resolves to Luna through the legacy OpenAI selection compatibility mapping. Gemini remains
selectable; GPT-Live is the only active voice mode. Cost tracking stays disabled.

Final backend validation passed 379 tests with one optional skip; the paused
Realtime test module was excluded. Sixty applicable frontend tests, build, Ruff
and desktop/mobile Settings/report checks passed. The complete report has 120
inspectable trials, working model/finding filters and no horizontal overflow.
