# Findings from establishing the eval baseline

## October 2 repair reconciliation — historical results preserved

The statuses and traces below describe the September runs. They have not been
rewritten as passes. Source review against `e734f7f` and
[Claude's session summary](../../docs/CLAUDE_SESSION_SUMMARY.md) shows subsequent
repairs; this reconciliation did not rerun paid trials or production acceptance.

- **EVAL-001/002:** UTC instant comparison, nullable-field agreement and recovered
  error handling have implementations and audit regression tests.
- **EVAL-003:** voice clarification ownership changed, then the broad deterministic
  answer binding was removed after a production regression. Still an open real
  voice/continuation acceptance concern; optional offers must not capture new requests.
- **EVAL-004/008:** task tools now expose flexible record/home paths and compact
  results, with minimal legacy/record sync. Repeat model disambiguation trials;
  canonical organization cleanup remains a distinct workstream.
- **EVAL-005/006:** numeric conflict questions and fact persistence with separate
  embedding retry have code repairs and targeted tests.
- **EVAL-007:** a recovery pass and explicit omission reporting exist. Reporting an
  omission is not the same as satisfying the original complete-extraction criterion.

Claude reports successful automated suites; this document does not independently
re-certify those runs. Re-run/regrade the original cases, including stochastic repeats,
and link new evidence before closing acceptance. Sol's six downgraded bindings are
still evidence gaps, not newly proven app bugs. The [completion PRD](../../docs/ERIDANI_WEB_V1_PRD.md)
defines the risk-weighted release core; the rest of the catalog remains available.


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


## September 20 automation campaign

The full available plan ran in `artifacts/app-evals/automation-20260920-run/`.
Harness corrections were verified separately in
`automation-20260920-corrections/` and `automation-20260920-planned-day/`.
The final offline baseline is `automation-20260920-offline-final/`.
Do not merge these different harness fingerprints into a fabricated single clean run.

No application behavior was changed in this evaluation batch.

### EVAL-003 — natural clarification text can leave completed work

Status: reproduced; product follow-up pending. Cases: clarifications.03 and .20.

When explicitly told to ask for an exact time, or for a start time and duration,
the backend asked the right question in prose but returned a successful/quiet
result. The queue marked the item succeeded instead of needs_input. A vague reply
then produced another completed question/card rather than resuming one open
clarification. No calendar write occurred in these trials.

The issue is the durable continuation state, not the wording of the question.
Keep the real multi-turn trials and ensure questions requiring an answer enter
the structured clarification path.

### EVAL-004 — project disambiguation can exhaust useful context

Status: reproduced in one Luna trial; repeat and product follow-up pending.
Case: clarifications.01.

The first request correctly asked which “Review proposal.” The answer named Beacon
Dispatch. The continuation listed tasks through legacy projections with empty
project fields, then searched the generic record hierarchy with a limit of 100.
The result contained the target and many pagination fixtures. Work ended with
“This request needs a narrower scope” without renaming the target.

Inspect target resolution, search payload size and continuation context limits.
This is not evidence of a lost accepted mutation; no rename was saved. The full
trace is retained locally and the assertion stays failed.

### EVAL-005 — numeric contradictions do not produce dream questions

Status: reproduced deterministically. Case: memory_dream.19.

“The user has 2 cats” and “The user has 3 cats” both remain active, but the dream
pass does not propose a clarification. Existing spelling-similarity review handles
Miso/Mizo; it does not cover this numeric contradiction. Source records remain
intact. Broaden the review only in a later product batch.

### EVAL-006 — extracted facts depend on embedding success

Status: reproduced at the extraction component boundary; architecture/acceptance
gap to resolve. Case: memory_capture.24.

With an injected embedding outage, the source conversation is retained, but no
Memory row is committed before the embedding request succeeds. The catalog asks
for a durable extracted fact plus separate embedding retry. This component trial
does not prove that the dispatcher loses the source or cannot retry extraction;
it demonstrates that extraction and embedding persistence are coupled.

Decide whether to persist an unindexed fact separately from the embedding job.
Do not call this permanent loss of the original conversation.

### Harness corrections, not product failures

- The paid worker's blank isolation environment initially shadowed the configured
  OpenAI key. Only explicit paid workers now load the key; offline workers stay blank.
- Python's asynchronous DNS executor does not copy context variables. The outbound
  guard rejected legitimate metered DNS. It now permits only provider DNS during
  an active authorized request; socket/URL guards remain in place.
- A legacy reliability fixture recognized only old test database names. It now
  also accepts the exact validated parent-owned eval clone.
- The worker recovery test inherited worker_enabled=false. Its dedicated trial
  now explicitly enables the synthetic worker.
- Clearing due_time legitimately clears due_timezone. The sparse-edit oracle now
  expects both while preserving the date and other fields.
- “Keep its planned day” now seeds a real planned day and supplies the required
  revision, so the fixture tests preservation rather than a false presupposition.
- memory_capture.07 prohibits attributing Alex's preference to Rowan, not storing
  a correctly attributed Alex fact. The oracle now follows the catalog.
- The recommendation replay fixture uses two explicit items, avoiding the valid
  single-item behavior of filing the original note. Replay preserves both links.
- The semantic judge now uses exact criterion IDs in a strict output schema.
  Invalid/incomplete judgments become needs_review and retain evidence.

All corresponding targeted checks passed after these harness corrections.
EVAL-001 remains reproduced. EVAL-002 did not recur in this single Luna sample;
that does not close the historical failure or establish a product fix.

### Spending

Across the successful campaign, targeted corrections and transport diagnostics:
estimated recorded usage **$0.069323630**, or **$0.708247443 including conservative
uncertain reservations**. Thirty-three interrupted/transport attempts retain their
bounds; the runner does not silently forgive them. The replacement campaign
allowance was reduced to keep the entire batch within the original $10 approval.
There were no Gemini or real GPT-Live calls, and no connected-service credentials
were supplied.


Final offline verification: **177 execution jobs; 173 passed, three expected
product/architecture failures, one optional cloud check blocked.** The reused
backend suites contain **634 passing tests and one optional skip**; all six browser
fixtures passed. The eval self-tests report **63 passes and two opt-in skips**.
Frontend unit checks report **136 passes and one skip**; the production frontend
build succeeds. Resuming the completed planned-day trial reused its evidence with
no additional provider request or cost.

These execution totals are separate from catalog acceptance coverage.

## September 22 full available Luna campaign

The full current 245-job plan ran in `artifacts/app-evals/full-20260922-localpg-luna/` against the marked synthetic corpus. A workspace-local PostgreSQL 16.15 server on 127.0.0.1:54340 replaced Docker Desktop because its engine could not start from a locked stale socket. No product behavior was changed.

After a provider ConnectError was retried through `resume`, 233 jobs passed, eight failed, and four were blocked. The original transport attempt and uncertain reservation remain. Backend suites reported 634 passed and one optional cloud check blocked; all six synthetic browser fixtures passed. Google, Linear, and R2 probes were blocked without dedicated test configuration.

Catalog classifications across 1,001 cases: 92 passed, six failed, 146 partial, 755 blocked, and two component-failed. Complete bindings still cover 98 scenarios; 224 have component evidence, with overlap. Nine hundred need complete assertions and three need physical-device evidence. No real GPT-Live/audio provider adapter ran.

- EVAL-001 `time_deadlines.26` still fails under the Chicago database session: a fresh Revert sees an equivalent timestamp as a later change.
- EVAL-002 `task_edit.25` recurred with Luna: null passed to `task_update.notes` was rejected; work ended partial without clearing the note.
- EVAL-003 `clarifications.03` entered `needs_input` first, but a vague reply left two activity cards and unresolved work. `clarifications.20` completed a prose question without durable clarification and also duplicated/stranded a card.
- EVAL-004 `clarifications.01` passed one project-disambiguation continuation. Its earlier context-limit failure remains evidence of variability.
- EVAL-005 `memory_dream.19` and EVAL-006 `memory_capture.24` still fail at their documented component boundaries.
- **EVAL-007: incomplete recommendation extraction.** In `note_organization.01`, the source said "Sam recommended Arrival and After Yang. Save these movies to watch." Luna saved a source-linked Arrival entry but omitted After Yang. The original note remained intact.

The full run initially flagged `task_capture.05` because its grader demanded raw `16:00`. The saved `16:00-06:00` with `America/Chicago` on January 18 is the requested local time. The grader now accepts equivalent local time and rejects a wrong offset. A targeted corrected-oracle Luna run passed at `artifacts/app-evals/due-time-oracle-20260922/`. The original failure remains in the full report; the two harness fingerprints are separate.

The full campaign recorded $0.073937489 estimated and $0.085541139 including uncertain across 166 requests. The targeted correction used $0.000590311 across two requests. September 22 totals are $0.074527800 estimated and $0.086131450 including uncertain, within the shared $10 cap. No Gemini or connected-service writes occurred.

## September 23 balanced half-suite with independent GPT-6 Sol grading

Selected 501 of 1,001 cases, 12–13 per feature across all 40 features, using a
frozen SHA-256-ranked sample with prior failures and the corrected time oracle
forced into the sample. This is a regression-oriented sample, not an unbiased
estimate of whole-app reliability. Selection:
artifacts/app-evals/half-20260923-selection.json.
Campaign: artifacts/app-evals/half-20260923-sol-graded/.

All 153 scheduled execution jobs finished: 139 passed, five failed, five deferred
to external semantic review, and four blocked. Backend supporting suites reported
634 passed and one optional skip; six synthetic browser fixtures passed.
The harness self-tests reported 65 passed and two optional skips. Raw catalog
classification: 44 passed, four failed, four needs_review, 75 partial,
373 blocked, and one component_failed. Only 52 selected scenarios have complete
bindings; 448 still need complete assertions and one requires device evidence.
A supporting suite pass does not establish a catalog acceptance pass.

GPT-6 Sol reviews the actual saved state, tool results, and exact criteria in
separate grader artifacts. The runner's original reports and hard failures are
retained. The new external-grader mode makes no Luna judge requests; its API
judge sublimit is zero. Application API usage was $0.055128104 estimated across
104 requests, with no unresolved charge reservations, within the $10 cap.
Sol's Codex usage is separate from this application API ledger.

- EVAL-001 reproduced: unchanged create/Revert conflicts on equivalent timestamps.
- EVAL-003 reproduced in clarifications.03: both turns reported success instead
  of durable clarification and left duplicate/stranded activity cards.
- EVAL-005 reproduced: conflicting numeric memories still receive no clarification.
- EVAL-006 reproduced at the extraction component boundary: embedding failure
  prevents committing a Memory row, while the source remains intact.
- EVAL-002 did not reproduce in final state. Luna first sent rejected notes: null,
  then recovered with notes: ""; the intended target notes were cleared.
  Keep the tool/schema affordance issue open rather than claiming a product fix.
- EVAL-007 did not reproduce: Arrival reused its preexisting note unchanged and
  After Yang was created, both linked to the source. One passing sample does not
  resolve the earlier omission.
- clarifications.20 passed the automated queue assertions in this sample.
  Historical inconsistent behavior remains open.

### EVAL-008 — task disambiguation misses flexible project relationships

Status: reproduced, clarifications.01, independently reviewed by GPT-6 Sol.

The user clarified that the intended duplicate “Review proposal” task belonged to
Beacon Dispatch. The synthetic structure_records contain that project and a
task record whose task_id points to the intended task and whose parent_id
points to Beacon Dispatch. The other duplicate lacks that relationship.
The model consulted legacy task/project tools, received empty legacy project
fields, then incorrectly stated that neither task was filed under Beacon Dispatch
and requested another clarification. It did not perform the requested rename.

No task mutated. The automatic message “Edited task field: title” means the
expected final title was not reached; it is not evidence of an unauthorized edit.
An initial suspicion of missing fixture linkage was disproved by inspecting the
flexible-schema relationship. This differs from EVAL-004's historical context
overflow symptom.

Make task resolution/search expose or traverse the current flexible organization
links. Add a focused regression with identical task titles, blank legacy
project_id fields, and a distinguishing flexible project parent. Assert that the
correct task is renamed and the sibling is untouched.

The isolated per-trial databases were removed; the reusable synthetic corpus was
retained and the eval-only PostgreSQL server stopped. No production records,
product behavior, commits, pushes, or deployments were changed.

### Completed independent grading

GPT-6 Sol's final review covers all 501 exact selected case IDs: **42 passed,
four failed, six need review, 75 partial, one component failure, and 373
unassessable**. See [grader.md](../../artifacts/app-evals/half-20260923-sol-graded/grader.md)
and the adjacent grader.json for criterion-level judgments and existing evidence paths.
The original automated report remains unchanged.

All five deferred semantic memory judgments were met (four acceptance-level,
one component-level). Sol conservatively downgraded six raw acceptance passes:
memory_dream.09 lacks retrieval/index/prompt deletion checks; .12/.14/.16 lack
actual scheduler-tick evidence; .25 lacks physical microphone behavior; and
task_lifecycle.09 lacks active-view exclusion and user-facing recovery evidence.
These are incomplete verification, not newly proven app failures. Expand those
checks before using their binding labels as full acceptance proof.
