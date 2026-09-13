# Tool improvements informed by the expert evaluation

These are evidence-backed follow-up candidates, not changes to the tool surface.
The expert-suite baseline remains frozen during the comparison. Real model calls
operate on synthetic records; Google/UI behavior uses declared adapters.

The raw evaluation keeps exact arguments, returned records and command receipts,
errors, per-turn output, ordered tool steps and provider usage. This is enough to
reconstruct the observable failure without storing native model reasoning.
See [the protocol](EXPERT_AGENT_EVALUATION.md) for isolation and interpretation.

## 1. Exact task filtering and stable bulk selections

Observed: in the diagnostic, Luna read the full 126-record list but omitted two
of 25 eligible tasks. In final repeat 1 it supplied an unsupported project_id to
task_list, recovered, and saved the correct set, but reported 26 instead of 25.
In final repeat 2 it saved 22 of 24 eligible tasks and claimed 23; in repeat 3
it saved 23 of 24 and reported 23 without explaining the unfinished scope.
Gemini completed all three scored bulk selections correctly.
See [the final results](EXPERT_AGENT_RESULTS.md) and the retained traces.

The current task_list can filter title text, space, area and goal; it cannot
directly express the requested intersection of project, status, assignee,
included/excluded tags and deadline range. Both models have to screen full
records. That increases opportunities for omissions and raises token use.
This is a tool-design hypothesis, not proof that filtering caused a given miss.

Candidate: add validated structured filters, inclusive deadline boundaries,
compact projected fields and an exact match count. Add an owner-scoped selection
preview with stable IDs/revisions and a clear apply result. Pagination must remain
stable if records change during traversal; offset pagination ordered by updated
time needs particular attention.

Validation: compare both models before/after on the same independent decoys,
including more than 100 matches, edits between pages, changed revisions, empty
sets and near-matching names. Check all intended effects and unrelated records.

## 2. Record references and actionable lookup failures

Observed: Luna's evidence_extraction repeat 1 correctly identified the two
commitments but dropped a character while copying the note UUID. note_read
returned NOT_FOUND. A later search returned the correct record; Luna stopped
instead of retrying the current ID, explaining that the note became unavailable.

Candidate: validate UUID shape before a lookup; distinguish an invalid identifier
from a valid-shaped identifier with no accessible match. Give a precise recovery
instruction to refresh and use the returned reference. Consider short opaque,
owner-scoped references if typing long IDs remains a repeated failure source.

Never silently substitute a similarly named record or expose records from another
owner. A malformed identifier does not prove that a record was deleted.

Validation: malformed UUIDs, stale-but-valid IDs, renamed/deleted records,
cross-owner IDs and repeated names. Successful recovery must not introduce
duplicate tasks or mutate the wrong record.

## 3. Receipts that make completed counts unambiguous

Observed: Luna reported 26 matching updates after correctly saving 25 distinct
tasks in final bulk repeat 1, and 23 after saving 22 in repeat 2.

Candidate: return requested_count, applied_count and distinct entity IDs for group
edits, with any unchanged/skipped/conflicting records explicitly separated.
Have the assistant's summary use those receipts. Keep atomic batch semantics;
a rejected batch must not sound partially applied.

Validation: repeated IDs, idempotent retries, no-op updates, several batches,
partial action limits and a conflict in the final item. Compare declared counts
with actual distinct effects and audit events.

## 4. Read-only time validation and safe alternatives

Observed: Gemini correctly identified the nonexistent March 10, 2030 2:30 AM
Chicago time but offered 2:00 AM as an alternative. That time is also inside the
DST gap. The clarified 3:30 AM deadline was ultimately correct.

Candidate: expose read-only local-time resolution that identifies a gap or fold,
returns valid offsets for repeated hours and suggests actual nearest valid local
times. It should not save anything while resolving uncertainty.

Validation: both DST transitions, non-hour transitions, date-only values,
explicit offsets inconsistent with a zone and cross-zone date rollovers.
Check the model's proposed alternatives as well as the final saved instant.

## 5. Explicit remote retry and follow-up semantics

Observed: both models correctly distinguished queued/retrying from confirmed
Google writes. Gemini made more immediate status polls, and one reply offered to
monitor the write without an explicit assistant follow-up being scheduled.

Candidate: expose whether a durable server retry is active, when another lookup
is useful, and whether completion will produce a notification. Separate server
retry from a new assistant promise to check back. The tool should make that
distinction easy to explain.

Validation: queued, retrying, succeeded, failed, cancelled and unknown outcomes;
connection changes; interrupted clients; exactly-once creates. Confirm that
wording about monitoring matches the app's actual delivery behavior.

## 6. Relationship edits and peer revisions

Observed: Luna goal_rewire repeat 2 completed the requested graph with two goal
edits, then attempted redundant project edits using stale revisions. The real
bidirectional relationships were already correct; subsequent conflicts added
rounds and left a partial status despite completed work.

Candidate: return the relationship diff and all affected peer revisions after
an edit. Consider explicit add/remove-link operations instead of requiring full
replacement lists for narrow requests. Preserve optimistic concurrency and
idempotent link semantics.

Validation: multi-goal/project edits from both directions, pre-existing links,
concurrent changes and stale references. Check that the agent stops once the
requested graph exists and never drops unrelated links or changes progress.

## 7. Source-note provenance as an explicit creation contract

Observed: Gemini evidence_extraction repeat 3 identified the correct commitments
and created two tasks with verbatim evidence in their text, but omitted the
structured source-note links that note_tasks would have created.

Candidate: make provenance requirements prominent in the note-derived creation
tool's name/description, or support a shared source_note_id plus exact evidence
contract on task creation. Both entry points should use one implementation and
the same evidence fingerprint/idempotency rules; avoid competing write paths.

Validation: note backlinks, evidence quotes, exact source revisions, repeated
extraction, renamed notes, explicit unrelated one-off tasks and note deletion.
Correct-looking task text must not substitute for the actual relationship.

## How to use this backlog

Classify each finding as a model mistake, a tool-design opportunity, an app bug
or unresolved. Cite model/case/repeat keys, preserve the original trace, and state
what would disprove the proposed explanation.

After choosing a narrow improvement, validate its domain behavior and run a
fresh paired follow-up with both models. Keep the original score and report
improvements in correct outcomes, truthful summaries, calls, tokens and latency.
Do not tune only to these 24 known prompts: add independently authored held-out
cases before calling a tool change an improvement.
