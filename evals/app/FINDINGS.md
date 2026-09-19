# Findings from establishing the eval baseline

## EVAL-001 — equivalent timestamp offsets can block Revert

Status: reproduced; app fix pending. Case: time_deadlines.26.

With PostgreSQL session time zone America/Chicago, a freshly created task’s action receipt can report “This record changed after it was created” even when no user edit happened. Receipt snapshots created before/after database round trips contain equivalent datetime values rendered with different UTC offsets. Equality then treats them as a change. Completion-time string stability also fails in this environment.

Evidence: the original backend run at artifacts/app-evals/20260919T060940Z-b37303c4/ and the dedicated time_deadlines.26 component run. The same existing backend suite passes with the dedicated eval role set to UTC.

Suggested fix: normalize datetime values to a consistent representation before receipt comparison/serialization, while retaining revision and linked-record guards. Add a session-timezone matrix for receipt/revert and completion timestamps. Do not solve it by weakening stale-write protection.

The baseline eval role uses UTC to match Docker CI. The explicit non-UTC scenario remains a failing check; it has not been hidden or marked passed. This evaluation batch does not modify application behavior.

## Harness issues resolved during setup

- Generic projects are organizational records, not necessarily legacy Project rows. The corpus creates task backing records through the current generic record system.
- Cloud code disabled globally prevents tests from reaching their synthetic provider transports. The regression runner uses fake keys and an offline network plugin instead.
- Existing regression runners initially held a connection to the corpus, preventing PostgreSQL template cloning. Their administration connection now uses postgres on the dedicated eval server.
- Component graders now expect exact production validation codes and distinguish API projection fields from canonical persisted fields.

These setup errors are not model-quality failures. Early raw reports are retained locally for provenance; use the final baseline report for the verified run.

## EVAL-002 — unclear note-clearing contract and recovery status

Status: reproduced in paid queued-backend probes; fix pending. Case: task_edit.25.

Both models passed null to task_update.notes when asked to clear task notes. The
tool rejected it with "notes cannot be empty." Luna stopped without changing the
task. Gemini explored alternatives, cleared the generic record body with an empty
string, and verified the task, but the queue still reported partial because an
earlier tool error remained. The strict grader therefore reports 19/20 complete
outcomes for each model; Gemini's final data mutation succeeded.

Clarify empty string versus null semantics in the task tool schema/error message,
and track recovered errors separately from unresolved effects before calculating
the final queue status. Preserve this case as regression evidence. No prompts or
tools were tuned between these model runs.

Paid estimates: Luna $0.02352348 across 62 requests; Gemini $0.497667 across 69.
Both were capped at $2, below the user's $10-per-model maximum. Gemini's original
meter setup rejected its max_tokens parameter locally; those attempts sent no
provider requests. The corrected Gemini run supplies the scored results.
