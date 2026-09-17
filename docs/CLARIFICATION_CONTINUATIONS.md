# Clarification continuations

The previous follow-up link described a dependency, not an answer. A new voice turn
could update the task while leaving the earlier request in `needs_input`.

## Request lifecycle

`work_needs_input` records a stable clarification ID, execution request ID, revision
and question. These are available to the backend in recent work and `work_list`.
Live receives the same question identity as context; it does not own execution.

The existing backend calls `work_answer` before any effects. Under the workspace
queue lock, the server checks account/credential identity, ordering, pending status,
question revision, expiry, cancellation and saved effects. It takes the entire
captured user turn verbatim; the model cannot replace it with a rewritten answer.

The paused execution becomes `continued`; the new execution is linked beneath the
original activity card and dispatched with a new durable invocation ID. Its prompt
contains the original request, prior questions/answers, current answer and actual
saved command receipts. The old tool plan is not replayed. Further questions attach
to the newest execution, and further answers keep the same visible card.

Normal edits to completed work still use `work_followup`. Independent requests
remain separate and may run concurrently. A related pending clarification cannot
be bypassed with a standalone mutation. Concurrent replies cannot both consume one
question: the later attempt must reread current work and apply a fresh correction.
Saved command IDs and revision checks remain the source of truth for actions/revert.

Activity aggregates saved changes across attempts and shows the latest state. The
original request and question/answer history remain expandable. Cancel/Revise work
on the logical request; Live reports a continuation in its current voice session.
No database migration, routing model or provider change is required.

## Clearing activity

`POST /api/v1/work/clear` uses the signed-in account/workspace and an optional UTC
cutoff. It cancels unfinished work and archives matching activity records. It does
not delete tasks, notes, projects, schedules, commands or action receipts. Cleared
cards are omitted from Activity, model recent-work context and Live announcements.
The UI confirms the scope before sending the request. An in-flight model response
cannot perform a command after cancellation; subsequent finalization cannot make
an archived card visible again.

## Validation

- Regression coverage: original saved effects + verbal clarification, multi-round
  answers, idempotent handoff, concurrent/stale answers, wrong account/question,
  cancellation, manual revision, legacy question identities, dependency readiness,
  bypass prevention, clear cutoff/account scope, and clearing during a model call.
- Real Luna and Gemini tests used disposable local databases and synthetic records.
  Both kept Buy milk independent and resumed a pending review deadline at 9 a.m.
  on the original card, without creating another review task.
- Six mobile browser checks: one completed card, expandable question/answer history,
  layout, cancel clear, confirm clear and saved task preservation. No JS errors.
- Final verification: 96 focused backend tests, 121 frontend tests, six mobile
  browser checks, and real-provider scenarios on Luna and Gemini passed. TypeScript
  production build and changed-file lint checks passed.
- Physical microphone/wake-word testing remains a separate acceptance step.
