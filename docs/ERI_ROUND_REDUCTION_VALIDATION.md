# Tool-round reduction verification

October 4, 2026. Implementation of the first two stages of
[the round-reduction plan](ERI_ROUND_REDUCTION_PLAN.md).

## Implementation and rollout

- `discovery-v1` is the default for newly accepted durable requests. Already exposed
  tools are called directly; only absent capabilities need discovery. Note creation
  joins the initial set (20 catalog tools, plus the existing work controls).
  Discovery is idempotent and reports newly loaded/already available names.
- `reads-v1` includes discovery plus guidance to reuse one sufficient fresh lookup
  for a sparse edit. It is implemented and tested, but staged behind
  `JARVIS_AGENT_TOOL_POLICY=reads-v1` until the discovery phone check below.
  Body edits, replacement links, ambiguous targets and missing dependencies still
  require the appropriate full read. Existing expected-revision validation,
  source-write status, browser acknowledgements and receipts are unchanged.
- `baseline` preserves the original catalog/guidance for rollback and comparisons.
  Set `JARVIS_AGENT_TOOL_POLICY` consistently on API and worker. Enqueue captures
  the version; retries/corrections of accepted jobs retain it. Legacy jobs with no
  value use baseline. No database migration or new agent/queue was introduced.
- Resumed read-reuse invocations tell the model old lookups are historical before
  it plans further edits. Planned command replay keeps existing command IDs.
  Revoked/removed tools cannot be restored just by a saved tool-name list.
- Content-free timing now includes every tool attempt (including clarification,
  dependency and voice controls), model-round tool names, validated discovery
  groups, newly loaded/already available counts and definition size. Request
  revision and invocation identity remain attached; arguments/body text are absent.
  The summary distinguishes missing tool tracing from measured zero-call responses.
- The eval campaign supports `--tool-policy baseline|discovery-v1|reads-v1` for
  isolated agent comparisons. Default evals stay baseline/Standard even when the
  application default changes.

The optional deterministic completion shortcut is deferred. The measured targets
are already met, while a successful tool call still does not prove that all clauses
of a free-form request are finished. Keep the ordinary final response and existing
clarification/cancellation/dependency checks.

## Matched paid comparison

All arms used Luna low reasoning, Fast requested and `priority` returned. Four
existing state-checked scenarios ran three times per arm in fresh synthetic local
database clones. Block order rotated baseline/discovery/reads, reads/baseline/
discovery, discovery/reads/baseline. No production records or connected accounts
were used. No paid model judge or Langfuse upload was added.

All **36/36** trials passed. Across twelve trials per arm:

- Baseline: **4.853 s** median runner-to-committed-outcome, **41 model rounds**,
  **$0.038763772**.
- Discovery only: **3.524 s**, **31 rounds**, **$0.031333055**.
- Discovery plus sufficient-read reuse: **3.246 s**, **27 rounds**,
  **$0.024775668**.

Discovery improved this sample's median by about **27%**; both stages improved it
by about **33%**. Spend was about **19%** and **36%** lower respectively.

Observed rounds by scenario:
- Simple task creation: baseline `3/3/3`; discovery `2/2/2`; reads `2/2/2`.
- Title-only note: baseline `3/3/3`; discovery `2/2/3`; reads `2/2/2`.
- Deadline edit: baseline `5/5/4`; discovery `4/4/4`; reads `3/3/3`.
- Ambiguous edit: baseline `3/3/3`; discovery `2/2/2`; reads `2/2/2`.
  All stopped correctly for clarification; these are successful test outcomes,
  not completed record mutations.

Per-scenario medians with both stages: task creation **3.952 → 2.953 s**, note
creation **4.328 → 2.723 s**, deadline edit **7.148 → 4.444 s**, clarification
**4.467 → 3.418 s**. Total input tokens fell from **356,719** to **244,777** despite
the slightly larger initial catalog. Cache-hit proportions differed; retain
first-block/cold costs and do not infer identical cache placement or stable p95.

A second smoke covered nine harder scenarios once per arm: date-only and timed
creation, clearing a deadline while preserving planned work, changing only its
clock time, replacing tags, clearing notes, duplicate-target clarification,
partial exact-time answers and partial calendar-block answers. **27/27 passed**,
including continuation/card identity oracles. This is broader correctness smoke,
not a repeated latency benchmark. It cost **$0.175953799**.

Total paid spend: **$0.270826294**. All provider calls settled with a verified
returned tier; no retries or uncertain charges were observed. Independent ledger
caps bounded the entire turn at **$3**, below the owner's $10 ceiling.

[Sanitized evidence](evals/2026-10-04-rounds/comparison.json) retains every primary
trial, round sequence, token/cache counts, cost, policy and fingerprint, plus the
edge outcomes. Full synthetic artifacts remain at
`artifacts/round-trim-2026-10-04/`. Matching source, fixture and harness fingerprints
were enforced during every block. Subsequent changes only selected discovery as
the default, labelled missing telemetry and documented results.

## Deterministic verification

- Final focused tool/queue/latency/model suite under discovery-v1: **106 passed**.
  Offline eval harness rerun: **95 passed, 2 skipped**.
- Full backend run under reads-v1: **1,010 passed, 1 skipped, 1 failed**. The failure
  was test-environment configuration: synthetic eval settings forced Standard while
  an existing default-profile test expected Fast. No product fix was needed; all
  **18 tests** in that module passed with the actual Fast default.
- Separate worker crash/recovery test: **1 passed**.
- Offline eval harness: **95 passed, 2 skipped**.
- Existing backend coverage includes replay after committed writes, independent
  work and related follow-ups, corrections/cancellation, clarification continuations,
  scoped bot/workspace access, source pending/conflicts, batch atomicity, full note
  preservation, custom records/schema/templates and device acknowledgement.
- Final local checks and required GitHub CI are recorded with the PR. A skipped test
  or scripted tool result is not evidence of live-model parity for that scenario.

## Remaining acceptance

- [ ] CI passes for the final commit.
- [ ] Deploy API and worker together with `discovery-v1`; verify health and the
  accepted policy in a known request's content-free trace.
- [ ] On the real phone, create a task and note with typed chat and GPT-Live,
  then try ambiguity → clarification and add/correct overlapping work. Check that
  voice, cards and visible results still finish naturally.
- [ ] After that first-stage check, enable `reads-v1` consistently on API/worker
  and repeat a sparse date edit, note/body edit, connected-source edit and a
  partial clarification. New jobs use the selected policy; accepted jobs retain
  their version. Roll new jobs back to discovery or baseline if needed.

Production speech, first-useful audio and device rendering are not measured by
these backend trials. Broader source/navigation/schema model behavior and actual
phone latency remain separate acceptance work.
