# Batch D — release verification

Started October 3, 2026 from `ca79dc8f0c1ed09be5776070fa75548289053fff`
(PR #24 / Batch C), on `codex/batch-d-release-verification`.
This report separates automated checks from real-device/provider acceptance.
The seven-day pilot has **not** started.

## Repairs found during verification

- Required questions can appear before the first tool call or have text after the
  question mark. The durable runner now includes those replies in its existing
  one-time model check. The model still decides whether it needs an answer;
  punctuation never creates a waiting card or attaches the next request by itself.
- Pending question data is reiterated immediately before the current turn. Partial
  answers and target choices must use `work_answer` before further questions or
  edits. Unrelated requests remain independent. No new interpretation model or
  deterministic answer classifier was added.
- The optional task `home_id` read filter accepts null, matching its existing
  unfiltered behavior. This prevents a harmless omitted filter from leaving an
  otherwise completed edit marked partial. Record/owner authorization is unchanged.
- The memory-outage grader now follows the implemented split: successful extraction
  keeps a searchable fact; a separate embedding job records a retry and later
  indexes the same fact. It verifies recovery without extracting twice.
- Movie extraction evidence now requires the original unambiguous film ID and
  authored body to survive, and an exact link back to the recommendation source.

The question check can add one model round to a tool-free question, including a
casual offer. Ordinary replies and replies with completed effects are unaffected.
An optional offer remains completed even if the model repeats it after the nudge.
This is a correctness repair, not a claimed latency improvement.

## Coverage and Sol's binding review

The catalog remains 1,051 scenarios across 42 features. **215 are fully bound**,
up from 112, spanning 20 of 42 feature groups. The other groups retain supporting
regressions and/or explicit acceptance gaps; they are not fully qualified. Binding means executable assertions exist; it does not mean every
mode/device/provider has been run. Tests of mocked providers are labeled as such.
The rest remain visible in `evals/app/automation-coverage.json`.

The coverage ratchet is deliberately regenerated in this PR. One previous claim,
`memory_dream.25`, is withdrawn: unchanged VoiceInbox rows cannot prove a physical
microphone did not open. Its component evidence remains; physical acceptance is
pending. The ratchet implementation and its refusal to silently lose bindings
remain unchanged.

| Disputed scenario | Stronger evidence / remaining gate |
| --- | --- |
| memory_dream.09 | Forget removes the source, vector, lexical/semantic retrieval, prompt context and pending review. |
| memory_dream.12 | Actual worker supervisor ticks with review disabled enqueue no review jobs. |
| memory_dream.14 | Worker ticks after several missed weeks enqueue one catch-up, then no duplicate. |
| memory_dream.16 | Worker ticks across Chicago DST create one review per local week; UTC gap is 167 hours. |
| memory_dream.25 | Component only; physical microphone acceptance still required. |
| task_lifecycle.09 | Archive removes the item from active canonical collections; archived lookup and same-ID restoration work. Paid archive selection remains separate evidence. |

New bindings concentrate on account/workspace boundaries, scoped API/MCP keys,
revocation, idempotency, schema/record validation, safe archive/restore, source
annotations and operations. New scheduler tests execute production scan wiring;
unrelated provider/maintenance services are stubbed. There are no production writes.

## Automated results

- Backend: **863 passed, 1 opt-in cloud test skipped** on local PostgreSQL 16.15.
- Eval harness: **82 passed, 2 skipped**.
- Frontend: **175 passed, 1 skipped**; TypeScript/Vite build passed, with the existing
  bundle-size advisory.
- All **eight** desktop/mobile browser fixtures passed in the offline campaign.
- N-1 schema test: populated 0019 → head, repeat upgrade and model agreement;
  an old-column insert/update still works; downgrade → upgrade retains task IDs,
  annotation content and setup/profile JSON. This is SQL-shape compatibility,
  not a simultaneous deployment of two full application versions.
- New B/C access checks cover shared list viewers, source-note viewers/removal,
  foreign records, scoped bots and account setup from a second device.

Campaign summaries, exact versions/fingerprints and cost totals are recorded in
[the machine-readable evidence](evals/2026-10-03-batch-d/verification.json).
Raw synthetic campaign artifacts remain under `artifacts/app-evals/batch-d-*`.
Original September reports and failed baseline trials are not overwritten.

The full offline campaign ran every available offline adapter, not all 1,051
acceptance scenarios. Missing bindings, paid modes and physical/provider checks
remain partial/blocked/not run. The opt-in real memory-provider test is intentionally
skipped offline. A subsequent targeted run corrects the stale outage oracle; it
must be read alongside, rather than substituted into, the original report.

## EVAL-001–008 regrade

The five critical stochastic probes ran three independent trials each. The final
15 executions all passed their saved-state oracles. Early trials reproduced orphan
clarifications and the null-filter error; those failures are preserved. The broader
model core passed **53/53**, and combined evidence covers **215/215 currently bound
scenarios**. It does not certify the other 836 catalog scenarios. Total estimated
paid spend was **$0.268779391**, including failed baselines; uncertain spend was zero.

| Finding | Regrade evidence |
| --- | --- |
| EVAL-001: equivalent offsets / Revert | `time_deadlines.26` passed in the offline campaign. |
| EVAL-002: clearing task notes | `task_edit.25` passed 3/3 final Luna trials. |
| EVAL-003: partial clarification | `clarifications.03` and `.20` passed 3/3 each after repair. Real Live acceptance remains pending. |
| EVAL-004 and EVAL-008: duplicate title/home | `clarifications.01` passed 3/3 final trials, with one continued card and the correct saved task. |
| EVAL-005: conflicting numeric memories | `memory_dream.19` passed; both facts retained and a question offered. |
| EVAL-006: durable fact during embedding outage | Updated `memory_capture.24` oracle passes separate retry and recovery. |
| EVAL-007: incomplete movie extraction | Both film identities passed 3/3; strengthened reuse/source-link checks pass all three saved-trace regrades and the broader live run. |

These are deterministic saved-state regrades and Luna pipeline judgments where
configured. This batch does not claim a new independent Sol/human grading report.
Gemini stays paused. All paid calls share a **$10 total batch allowance** by
subtracting prior campaigns' spend, including uncertainty, before each new run.

## Production readback

Read-only Railway CLI checks on October 3 confirmed:

- Web deployment `a28e12aa-e971-4fee-aa04-3d759a8d49f1` and worker deployment
  `bba07ee4-7f2d-413f-95b1-748f609ddeb3` both run Batch C commit `ca79dc8` successfully.
- PostgreSQL's active image is `ghcr.io/railwayapp-templates/postgres-ssl:16.15`.
  Sleeping is disabled for web, worker and database.
- `https://app.eridani.app/health/ready` returned HTTP 200 with `{"status":"ready"}`.
  `/ready` alone is the SPA fallback and is **not** readiness evidence.
- Worker startup logs show DBOS queues initialized. This does not prove a current
  database heartbeat or recovery during an actual restart.
- GitHub main rules require Backend and migrations, Frontend and browser, and
  Security audit with strict status checks; no force pushes or branch deletion.
  They do not require PR-only changes. No branch policy was changed in this batch.

Direct SQL/heartbeat remains pending: there is no configured public PostgreSQL
URL, and Railway SSH reports no SSH key. No public database exposure or SSH keys
were added. No service was restarted and no production record was modified.

## Remaining release gates

Use [the acceptance checklist and pilot log](RELEASE_ACCEPTANCE.md). Before starting
its seven-day clock:

1. Merge this PR after CI, then verify the new web/worker revisions.
2. Run real Pixel Fold/desktop voice and clarification checks, including an unrelated
   request after an optional offer, network loss and close while work is running.
3. Use dedicated Google/Linear records to verify bidirectional edits, conflicts,
   recurrence/all-day scope, Revert and preservation of Eridani-only notes.
4. Verify locked-phone notifications, new-person onboarding and weekly interviews.
5. Finish direct cloud heartbeat, pending-work restart/home-PC-off, backup/PITR and
   consent readiness checks already listed in TODO.
6. Review remaining model failures, if any, and accept no unresolved critical
   access, wrong-record mutation or data-loss issue.

The release and Android handoff remain gated by those checks and the actual
seven-day/50-successful-interaction pilot.
