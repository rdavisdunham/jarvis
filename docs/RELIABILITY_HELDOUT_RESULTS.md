# Fresh reliability evaluation — September 13, 2026

The full run exercised eight fresh scenarios three times with each production
route: 48 trials and 66 conversational turns. Gemini completed more of these
workflows with supported replies. The run also exposed a concrete planner
interface defect, which was repaired and checked separately afterward.

The [protocol](RELIABILITY_HELDOUT_EVAL_PLAN.md) fixes questions, fixtures and
state oracles. These scenarios cover sparse notes, reference-error recovery,
constrained planning, unknown availability, mobile controls, dirty editors,
entity discovery and natural infeasibility explanations. They are different
from the earlier twenty-scenario regression, so its scores are not a paired
before/after comparison.

## Full-run scores

- **GPT-5.6 Luna:** native state **19/24**; explicitly adjudicated state
  **20/24**; completed task plus supported replies **20/24**. Native outcomes
  were ten clean successes, nine recovered successes and five failures.
- **Gemini 3.8 Flash:** native and adjudicated state **22/24**; completed task
  plus supported replies **21/24**. Native outcomes were nineteen clean
  successes, three recovered successes and two failures.

The single state override accepts Luna protecting a known dirty editor
proactively. It waited for the owner to save, then opened the requested timeline.
The literal grader required an attempted navigation and failed acknowledgement.
The original check and score remain unchanged in the raw result.

Every turn received a separate, unblinded Codex-subagent factual review. Luna had
21 supported trial-level replies, two unsupported, and one application
boilerplate response excluded from model-authored factual grading. Gemini had
22 supported and two unsupported. Truthful partial completion does not become
task completion. This is a small authored test set, not a universal model ranking
or an independent human evaluation.

## What the failures show

Luna struggled with the original 1,892-character encrypted planner token in all
three scheduling variants. Two corrupted copies recovered after fresh proposals.
The third generation reached the 8,192-token output cap with malformed commit
arguments, and no blocks were saved. The application misleadingly described that
HTTP-200 incomplete generation as a lost connection; this boilerplate was not
attributed to the model.

Luna also opened the correct project in list layout instead of the requested
timeline twice. One reply falsely claimed the timeline was open; the other
accurately described the incomplete result. In another mobile workflow, it used
the preferred name “Morgan” as an assignee filter instead of the saved actor
“owner,” producing empty results while claiming the requested tasks were shown.

Gemini once replaced existing Unicode characters with literal backslash-u text
while appending a line, then claimed preservation. The exact-content oracle
caught this material data-integrity failure. Tags and relationships survived,
and the broader field-scope check passed because editing content was authorized;
that broad check must not be read as proof that content was preserved.

Gemini also requested `calendar_sync` after unavailable calendar verification.
The fixture intentionally does not run the sync worker and blocked the request.
Its no-schedule reply was accurate, but that production recovery path is
**untested**, not an automatic pass or evidence of an unsafe external write.
A different otherwise-correct refusal attributed the application's reconnect
guidance to Google, causing the additional factual-score deduction.

All six stale-note-link variants recovered through the real atomic
`INVALID_REFERENCE` path. The strict note oracle and literal UI state checks
remain intact; no failures were removed or silently regraded.

## Separate planner-reference diagnostic

After the full run, the application replaced the model-visible encrypted payload
with a **36-character owner-bound reference**, retaining the encrypted proposal
internally, expiry, task revisions, fresh availability checks and committed replay.

The same three scheduling fixtures then ran once per provider in a separately
labeled six-trial diagnostic. **Both models passed 3/3 task and factual checks.**
Each copied the short reference exactly and used one proposal and one commit.
Luna had two recoverable minimum-duration argument errors; Gemini completed
cleanly. No token-copy failure or incomplete generation occurred.

Across those three scheduling trials, Luna's measured task time fell from
151.86 to 41.21 seconds; Gemini's was 28.26 before and 28.73 after. This is a
targeted regression observation with three trials per route. These results do
not replace or enter the original 48-trial score.

## Usage and latency

For the full 24-trial run per provider:

- **Luna:** 154 provider requests, 1,153,893 input and 24,288 output tokens;
  17.30-second mean task time, 12.40-second median. Reported output includes
  2,880 reasoning tokens. Reported cached input was 761,513 tokens.
- **Gemini:** 155 requests, 1,596,988 input and 12,250 output tokens;
  11.23-second mean task time, 9.07-second median. Reported cached input was
  104,747 tokens across 23 requests; absent cache fields and separate reasoning
  usage are unavailable, not assumed zero.

Task time includes conversation/tool execution and intermediate state auditing,
excluding fixture setup and final grading. Provider-only totals were 406.21
seconds for Luna and 260.87 for Gemini. These are observed latencies, not service
guarantees or isolated estimates of model thinking time.

The excluded ten-row first attempt and six-row post-fix diagnostic also consumed
usage. Across all three retained attempts, **each provider received 205 requests**:
Luna reported 1,517,784 input / 28,373 output tokens; Gemini reported
2,105,579 input / 15,627 output. These are measured token totals for these
evaluations, not invoices or total account usage. Reasoning is not added twice.

## Integrity and limitations

The first attempt stopped after ten complete rows because the fresh allowlist
omitted the legitimate `calendar_connection` read. The corrected full run
preserved all 24 fixture payloads, questions and oracle inputs. The stopped
attempt remains separate exploratory evidence.

All fourteen full-run and fifteen diagnostic integrity checks passed. Each
disposable database was removed; provider identities matched; no unexpected
egress, hidden retries, credential values or native reasoning payloads were
recorded. Exact tool arguments, results, usage, fixture hashes and source text
are retained. The original 24- and 20-scenario archives are unchanged.

Synthetic Google rows feed the real connection summary, calendar projection,
free/busy merge and local save code. Provider transport is mocked. The repaired
legacy queued-write fixture now runs the real queue command and internal status
polling, recording internal versus explicit reads separately; it is not one of
these eight fresh scenarios.

UI acknowledgements model typed device transitions and real note saves; they do
not render React or test browser permissions. Semantic note retrieval uses a
labeled lexical substitute, memory search is empty, and automatic memory
learning is disabled. Real microphone/voice, live external writes and embedding
quality are outside this evaluation.

The archives freeze 69 source files, including all backend Python modules and
the relevant frontend action/editor/workspace files. After the diagnostic,
Ruff removed one extra blank line in `note_schema.py`; its Python AST is
identical. Browser-only test changes are outside the evaluated runtime snapshot.
The artifact index records that provenance.

## Evidence and reproduction

- [Artifact index, immutable hashes and path mapping](evals/reliability-heldout-artifact-index-2026-09-13.json)
- Full run: [raw traces](evals/reliability-heldout-agents-2026-09-13.json.gz),
  [24 initial fixtures](evals/reliability-heldout-fixtures-2026-09-13.json.gz),
  [factual review and explicit override](evals/reliability-heldout-response-review-2026-09-13.json),
  [integrity and metrics](evals/reliability-heldout-integrity-2026-09-13.json),
  [exact sources](evals/reliability-heldout-frozen-sources-2026-09-13.json.gz).
- Short-reference diagnostic: [raw traces](evals/reliability-plan-reference-diagnostic-2026-09-13.json.gz),
  [six-row factual review](evals/reliability-plan-reference-response-review-2026-09-13.json),
  [integrity and metrics](evals/reliability-plan-reference-integrity-2026-09-13.json),
  [exact sources](evals/reliability-plan-reference-frozen-sources-2026-09-13.json.gz).
- Stopped first attempt: [raw traces](evals/reliability-heldout-aborted-v1-2026-09-13.json.gz),
  [interruption and usage](evals/reliability-heldout-v1-interruption-2026-09-13.json),
  [separate review](evals/reliability-heldout-v1-response-review-2026-09-13.json).
  Its fixture and source copies are linked in the artifact index.

Use the protocol's runner command with a new output path. The follow-up is:

```sh
uv run python scripts/evaluate_expert_agents.py --suite reliability8 \
  --cases schedule_three_blocks --repeats 3 --diagnostic-only \
  --output .runtime/reliability-plan-reference-new.json
```

The fixed questions and original traces remain useful for the next tool work:
safe append/anchored note edits, canonical assignee resolution, complete
record-plus-layout navigation, clearer incomplete-generation errors and
structured recovery-message provenance.
