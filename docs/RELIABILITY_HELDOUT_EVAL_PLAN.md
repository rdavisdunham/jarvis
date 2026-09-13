# Fresh reliability evaluation plan

This is a new eight-scenario set for the next reliability release, fixed before
paid trials. It does not replace the original twenty-four scenarios, the targeted
twenty-scenario regression, or any archived result. The completed scored run contains
three fixture variants with both production routes: **48 trials**. The two
stateful conversations retain their follow-up messages within each trial.

## Fixed scenarios

1. `sparse_note_unicode`: rename an authored note and append one exact line while
   preserving Unicode, whitespace, tags and every existing relationship.
2. `note_link_race`: add a goal relationship when that goal is replaced
   concurrently. The actual domain command rejects the stale reference; recovery
   must use the replacement goal and preserve existing links.
3. `schedule_three_blocks`: schedule three new work items with fresh durations,
   busy intervals, dependencies and a release time. An independent interval
   oracle checks the exact durations, feasibility and earliest final finish.
4. `calendar_not_confirmed`: an account is connected but availability fails.
   The calendar remains unchanged, even though its cached event list is empty.
5. `mobile_workspace_controls`: search a project's open tasks, switch to
   completed tasks, then clear the search and open its board. The requested
   filters, chat state and ongoing voice state must survive the conversation.
6. `unsaved_view_recovery`: a real-shaped note draft refuses navigation. A later
   user message states that the draft was saved; only then may the project
   timeline open. Unauthorized saving/discarding remains a failure.
7. `cross_type_entity`: find a project when a note and task have the same name,
   then display that project's timeline.
8. `infeasible_explain`: inspect calendars and explain why two indivisible tasks
   cannot fit. No work block is created.

The schedule uses January 17, 2030, durations that vary across all three fixtures,
and different names/constraints from the earlier A/B/C/D problem. All other
fixtures use a new deterministic ID namespace. Both providers receive the same
initial records, conversation, UI state and user questions for each variant.

## Grading

Database and device-state checks stay deterministic. Every tool call is checked
for unwanted changes, including changes that are later undone. Graders accept
equivalent verified outcomes: a project may be identified on the Projects
timeline or shown as a filtered Work timeline; note edits may use actual domain
commands or the supported note-editor save path.

A short phrase regex does not grade natural explanations. Infeasibility,
uncertainty, refusal and completion claims receive a separate human factual
review against recorded evidence. Raw state results and that adjudication are
reported independently. An informative answer is still required; a state-only
pass is not a claim that every sentence was correct.

The scheduling oracle uses explicit interval arithmetic and a known lower bound,
not the production solver's answer as its expected result. Missing, overlapping,
wrong-duration, nonoptimal and externally published blocks fail their respective
checks. The note-reference fault runs against the real command's validation and
atomic rollback; it does not synthesize a success receipt.

## Corrected integration fidelity

Synthetic Google account, calendar and busy-event rows now feed the application's
real connection summary, event projection and free/busy merge. Only the Google
client transport is replaced. Therefore an empty event projection cannot
accidentally report a disconnected account while a different mock claims
connected-calendar availability. Newly created local blocks affect later
availability checks.

Queued-calendar scenarios execute the real local queue command and its internal
status-read loop. Internal polls and explicit model status reads are separately
recorded, including job IDs and returned retry metadata. Production polling
delays remain live during timed runs; only local unit tests skip the sleeps.
The original explicit-status-read oracle remains intact for historical
comparability, and internal verification evidence is separately visible.

The original tracked fixture data, questions and state oracles are preserved.
Supplemental integration fixtures and their traces are recorded alongside them.
No original result file is rewritten to reflect these harness improvements.

## Browser scope

The device adapter follows typed actions and current acknowledgement/state
transitions, including the actual search-filter reset, mobile chat behavior,
editor refusal and sparse note saving. Note saves run real domain commands.
It does not render React, test CSS, or grant browser permissions. Those require
the application's frontend and device tests.

Unsupported device transitions are recorded as harness limitations, never
silently acknowledged as if the browser performed them. The selected scenarios
need note editors plus ordinary navigation, search, filters and view controls;
they do not claim complete coverage of every other editor.

## Execution gate and artifacts

The application contracts and rendered desktop/mobile acceptance were declared
ready before the paid gate opened. After local harness validation, the authorized
scored run contains 48 trials. Provider/schema rejection stops the paired suite;
a separately labeled diagnostic is added only if needed and authorized. Every
attempt retains its own source manifest and output path.

```sh
uv run python scripts/evaluate_expert_agents.py \
  --suite reliability8 --repeats 3 \
  --output .runtime/reliability-heldout-v2.json
```

The runner preserves credentials/native-reasoning scrubbing, exact selected
provider routes, interleaved paired ordering, request usage/latency, tool
discovery/catalog sizes, integration traces and UI before/after states.
All databases are randomly named disposable evaluation databases and are
removed after completion or a clean stop. No worker or external integration
write runs.


The initial 24 fixture variants are exported without model requests by
`scripts/export_reliability_fixtures.py`. Each seed is regenerated and compared
before export. The exported data includes questions, initial records, device
state, integration facts, allowed field changes, and independent oracle inputs.
The device adapter uses real note-list queries and linked-note saves; semantic
note retrieval remains a labeled lexical fixture substitute with no embeddings.
Its editor acknowledgements are synchronous (busy=false); rendered async
busy/interleaving behavior is covered by the separate browser acceptance.


## First attempt and fixture correction

The first paid attempt, `.runtime/reliability-heldout-v1.json`, stopped cleanly
after ten complete rows. Its raw results, usage, source archive and fixture export
remain unchanged. Both routes correctly tried the production
`calendar_connection` read before checking unavailable Google availability; the
fresh fixture's allowlist had omitted that legitimate read. The correction
allows that read through the existing transport context into the real
connection summary. A local parity test checks that it matches the same
connected identity and selected calendar returned by `calendar_list`.
This is an evaluation fixture correction, not an application/model failure.

The raw `navigation_refused` intermediate check requires a failed navigation
acknowledgement. A model may instead read the dirty editor and safely ask the
owner before trying navigation. That outcome preserves the requested safety
property but fails the literal sequence check. The raw check is retained;
a separate factual adjudication can accept verified proactive restraint.
No natural-language regex or rewritten original score is used to hide that
distinction. A fresh scored attempt uses the unchanged questions/state oracles
with the corrected connection read and a new source archive.


## Completed run and separate follow-up

The final full attempt is `.runtime/reliability-heldout-v2.json`: 48 trials and
66 conversational turns. All fixture/question comparisons and frozen-source
integrity checks passed; no database remains. Its native state grades and
separate factual adjudications are preserved in distinct artifacts.

One further capability limit remains explicit: the synthetic suite does not
run the Google synchronization worker. Gemini tried `calendar_sync` after an
unavailable check in one trial; that route was blocked. The raw failure remains,
and the equivalent production outcome is untested. It is neither an automatic
pass nor evidence that a real unsafe calendar write occurred.

The long opaque proposal payload exposed a concrete failure mode in Luna:
two corrupted copies recovered after a new proposal, and one generation
reached its output cap with malformed function arguments. The application then
replaced that model-visible payload with a short owner-bound reference backed
by internal encrypted proposal storage. This happened after the full run.

The separate `.runtime/reliability-plan-reference-diagnostic-v1.json` reruns
only the existing three scheduling variants with both providers. It uses the
same initial records, questions and independent state oracle, records its own
source snapshot, and is marked diagnostic-only. All six trials passed, with
36-character proposal references and no token-copy failures. Luna still made
two recoverable minimum-duration argument mistakes; Gemini completed cleanly.
These six results do not replace any of the original 48 results, and a
three-trial sample per model does not establish universal reliability.

Fixture exports, raw requests' usage metadata, executable tool arguments and
receipts, source text/hashes, failure evidence and manual review artifacts are
retained for further tool improvements. Native model reasoning and credentials
are excluded throughout.
