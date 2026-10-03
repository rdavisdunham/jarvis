# Eridani — active roadmap

Updated October 3, 2026. This is the single active work checklist.
The [web v1 PRD](ERIDANI_WEB_V1_PRD.md) defines scope and acceptance;
the [latency plan](ERI_LATENCY_PLAN.md) defines the response-speed work.
Start with the [documentation index](README.md) for implementation and operations guides.

**[x]** means the stated work is recorded as implemented/reviewed, not that every
production or device check passed. **[ ]** remains pending. Old release checklists
and the full request history are preserved in the
[archived ledger](archive/2026-10-02/TODO_HISTORY.md); their old “next/current”
headings do not set today's priority.

Finish line: a dependable invite-only web planner on desktop and Pixel Fold,
with clear organization, fast capture, reliable Eri actions and guided setup.
Android follows the release checks and seven-day/50-interaction pilot.

## Completed planning and implementation reconciliation

- [x] Reconcile Claude's audit/UI changes against the earlier backlog and Sol grading;
  produce the web v1 completion PRD. See the [session summary](CLAUDE_SESSION_SUMMARY.md).
- [x] Record code repairs for queue locking/recovery, budgets, Revert/nullable updates,
  memory extraction/conflicts, record homes, spoken-title search and access/sync guards.
  Original model failures remain evidence until rerun; see [eval findings](../evals/app/FINDINGS.md).
- [x] Record Orbit UI and local conversational-control registry implementation.
  Preserve these features and fix specific gaps rather than restart a broad redesign.
- [x] Record CI backend, frontend/browser, security and coverage-ratchet implementation.
  Deployment and branch-rule readback remain separate checks below.
- [x] Audit and revise the latency plan: one durable runner, measured waits, safe
  voice handling and a testable Luna no-reasoning profile. No latency code has shipped
  as part of that audit.
- [x] Separate current documentation from historical plans, audits and release reports.

## Batch A — baseline, voice recovery and operations

Merged as PR #22 on October 2. Production/device acceptance below remains separate from the merge.
See [Batch A verification](BATCH_A_VALIDATION.md) for exact evidence and remaining checks.

- [x] Implement encrypted Send/Discard voice recovery, commit-delivered queue wakeups,
  independent maintenance, outbox pagination and content-free stage timing logs.
- [x] Verify PR15 is included in both running app revisions, readiness, active PG16.15
  image, both Railway CI gates and current GitHub rules. Reconcile local Railway inventory
  while retaining the detached old volume; no infrastructure apply.
- [x] Run backend/frontend regression and mobile chat recovery checks; bind two exact
  release-core evals and increase the coverage baseline to 100/1,001 fully bound cases.
- [x] Owner disabled PostgreSQL Serverless; verify sleeping is off on all three
  deployed services via fresh Railway readback.
- [ ] Finish cloud readback: direct SQL/worker heartbeat. Decide PR-only policy
  (required CI checks exist; a PR rule does not).

- [ ] **A1 — verify the voice repair release.** The pending-question fix is merged
  with CI green; confirm the deployed revision and both services. Test a genuine
  clarification, an unrelated next request, a late answer and a question resolved
  while its result is being announced. No swallowed request or orphan waiting card.
- [ ] **A2 — queue and voice recovery.** Verify independent parallel requests,
  dependent corrections, cancellation during a provider call, revisions/budget holds,
  crash/restart and reconnect. Accepted work continues after voice closes.
  Replace ambiguous close-time action-verb auto-enqueue with a recoverable Send/Discard draft.
- [ ] **Clarification ownership regression.** “Hey Eri, add this to-do” →
  Eri asks once → “Introduction email to Greg, head of marketing at ABC” →
  one task and one completed card, no premature delegation or repeated question.
  Existing fixes/instructions still need real voice acceptance.
- [ ] **A3 — verify cloud/configuration.** Read back actual services and reconcile
  the stale PG18 declaration without applying an unreviewed whole-project plan.
  Confirm PostgreSQL version, worker/DB availability, branch PR requirements and
  both Railway CI gates. Triage dependency majors separately.
- [ ] **A4 — measure and reduce latency.** Follow Steps 0–1 of the
  [audited plan](ERI_LATENCY_PLAN.md): stage timings, actual /work + Live baseline,
  prompt dispatch wakeups and measured result/tool-discovery delays. Preserve
  outbox recovery, speech completeness and conversational turn protection.
  Do not remove the durable queue or promise savings from timer arithmetic.
- [x] Begin binding release-core eval cases and add regression coverage with Batch A.
  Continue toward the release-core target in Batch D; Batch C raises the current count to 112.

## Batch B — organization, sources and daily workflows

Merged as PR #23. Validation in [Batch B verification](BATCH_B_VALIDATION.md).
This batch does not change the backend model default or bypass CI.

- [x] **B1 — understandable Organization (implementation).** Establish canonical reads/writes, then
  present an actual record hierarchy with safe moves and a separate types/fields/
  relationships editor. Opening a client/project should reveal its related work.
  Preserve flexible defaults, inheritance, links, revision checks and touch/keyboard
  alternatives to dragging. Explain “main home” versus additional links.
- [x] **B2 — shared external-source contract (implementation).** Apply to every current/future
  connector and supported record kind, including Google Calendar and Linear:
  literal provider badge, editable color, original link and distinguishable
  account/calendar/team context across views and details.
- [ ] Source-backed edits from inline cards, boards, calendar, bulk, Eri and API/MCP
  automatically update the same remote item where supported (implemented and mock-tested;
  dedicated real-provider acceptance still pending). Completion, deletion
  and Revert obey source semantics; local archive is not remote deletion.
  Show pending/confirmed/error/conflict states and provider-specific fields.
  Clearly distinguish unsupported/read-only and local-only fields.
- [x] Add collapsible **Eridani-only notes**, stored separately from synced
  descriptions. Never send them to any provider; sync/source Revert preserves them.
  “Private” means not externally synced, with existing workspace visibility.
  Mock-provider readback/retry/conflict and annotation preservation are tested.
  Real-provider readback/acceptance remains below.
- [x] **B3 — scheduling and notifications (implementation/offline checks).** Apply editable scheduling windows,
  timezone and task constraints separately from weak work/personal routing hints.
  Complete natural-language snooze, quiet hours, priority and summary controls;
  verify real delivery and no duplicate deadline/reminder alerts. Offline deadline,
  snooze, quiet-hour and summary regressions are covered; locked-phone delivery is pending.
  Agent completions stay in chat/Activity, not notifications.
- [x] **B4 — focused UX reliability (implementation/browser checks).** Preserve drafts during conflicts and offer
  compare/reload/reapply where allowed. Fix crowded controls, week-view/due-chip
  collisions and foldable layouts; preserve Back, scroll, keyboard focus and
  autosave/dirty-editor protection. Verify guarded note edits, canonical assignees
  and observed UI acknowledgement. Refresh Eri's site map after UI changes.
- [ ] **B5 — faster model/tool execution, only if A4 still misses targets.**
  Test an explicit Luna no-reasoning profile alongside low reasoning, with a clear
  Settings choice and actual request-setting verification. Keep the durable runner.
  Trial compact tools/context and composites only when traces justify them;
  expand under the same work root with receipts, dependencies and cumulative budget.
  Keep current defaults until an explicit user choice or measured rollout decision.
- [ ] Owner-flag rollout requires correctness parity, useful latency improvement
  and no average cost increase per successful request, including fallback spend.
  Measure visible/spoken results, not only database completion. Initial warm median
  targets: navigation ≤1.5 s; agenda/local-task confirmation ≤2.5 s. Report p90,
  failures and sample counts separately; these remain unproven targets.

## Batch C — quick capture and first-use setup

Implemented on `codex/batch-c-capture-setup`; see [Batch C verification](BATCH_C_VALIDATION.md).
Implementation/offline checks are distinct from real voice and new-user acceptance.

- [x] **C1 — Quick lists.** Title/checklist capture with optional sections, deadline
  and Today pin; inline completion/reordering, persistence and searchable task IDs.
  One summary in ordinary Tasks/Today views; search can still find individual items.
  No required client/project or schema setup. Alerts are opt-in; expired deadlines
  do not delete or complete items.
- [x] Add Eri tools for capture, reading, editing and reordering. Promote the same
  list to an ordinary parent task/subtasks, optionally in an existing project.
  No copies or lost IDs/history. Creating a new project uses the normal project tools.
- [ ] Real voice Houston packing acceptance: Hayes, personal packing, moped equipment
  and work essentials; Eri follows corrections and suggests supplies without silently
  creating commitments. [Full requirements](archive/2026-10-02/TODO_HISTORY.md#quick-lists--quick-projects--september-27).
- [x] **C2 — guided setup (implementation).** Name, timezone and purpose; editable
  defaults or a custom group with a description/example. Per-account skip/resume,
  separate schema preview/confirmation, existing field clarification tools, text/voice
  setup tools. No sample work is silently created.
- [x] Explain personal-account ownership versus selected shared-workspace membership
  in setup/login/invitation copy.
- [ ] Test setup with a new person on a phone without developer explanation,
  including real conversational clarification and later resume.
- [x] **C3 — organization learning.** Explicitly confirmed rules work immediately;
  show destination, reason, evidence and edit/confirm/pause/forget controls.
  Automated rules retain field-understanding and held-out quality gates. Edits resolve
  matching weekly questions; personal memory remains separate. Silence is not approval.
- [ ] Accept the combined weekly rule interview and field-clarification experience
  in actual conversation; offline regressions do not establish model quality.
- [x] **C4 — production logo.** Install the selected v19 soft-square peeling-note
  concept as vector, monochrome and PWA assets. See [brand assets](BRAND.md).
- [x] Add 25 Quick-capture and 25 setup eval scenarios. Bind 12 exact deterministic
  scenarios; ratchet to **112/1,051** fully bound cases across **42 features**.
  Remaining model/voice/device cases retain unrun acceptance status.

## Batch D — verify, then pilot

- [ ] **D1 — release-core evaluations.** Fully bind about 200 risk-weighted cases
  from the 1,051-case catalog. Fix Sol's six weak bindings; regrade EVAL-001–008 and
  repeat critical stochastic Luna cases at least three times. Preserve old reports.
  Default to offline checks; paid campaigns use the existing **$10 total ceiling**,
  including pipelines/embeddings/judges/retries and uncertain spend. Gemini stays paused.
- [ ] **D2 — migrations and access.** N-1 compatibility, account/workspace
  isolation, viewer/editor boundaries, removed members and scoped/revoked bot keys,
  including source annotations and new capture/onboarding paths.
- [ ] **D3 — real acceptance.** Pixel folded/unfolded plus desktop: rapid requests,
  clarification, natural/explicit goodbye, “Eri”/“Hey Eri,” 30-second quiet timeout
  while work runs, Wi-Fi/cellular loss, reconnect, Back, keyboard and accessibility.
- [ ] Verify receipt/Edit/Revert behavior, stale/failed Activity cleanup, notes/list
  filing and extraction, search aliases/misfiled matches, both learning/review flows,
  locked-phone push, snooze/quiet hours and summary delivery.
- [ ] Test dedicated Google/Linear items: source-side and Eridani-side edits,
  all-day/timed and recurrence scope, remote readback, conflict/retry, connection loss
  and multiple selected calendars. Mock results do not satisfy this gate.
- [ ] Verify cloud API/worker restart with pending synthetic work and home PC off;
  confirm Google consent readiness before broader invitations. Record actual commit,
  device, request/receipt IDs, expected/observed state and failures.
- [ ] Recheck latency/cost gates on actual voice/chat flows after feature changes;
  no wall-clock “600 ms plumbing” test on ordinary shared CI.
- [ ] **D4 — seven-day pilot.** After expansion/device/operations acceptance,
  record at least 50 successful task/reminder interactions, plus failures,
  corrections, voice comfort, latency and per-feature costs. Review a full month's
  costs when available; separate eval/hosting charges and incomplete periods.

Verification entry points: [CI](CI.md), [on-demand eval guide](../evals/app/README.md),
[findings and Sol evidence](../evals/app/FINDINGS.md), [latency checks](ERI_LATENCY_PLAN.md#8-verification-and-implementation-order).
A checked implementation task never replaces connected-provider or physical-device evidence.

## Explicitly deferred, not forgotten

- Independent R2 exports, isolated restore drills, key escrow and stale/failed-backup
  alerts remain deferred under the recorded owner decision. Revisit before broader
  multi-user/public release; preserve current recovery facilities and verification limits.
  [Prepared backup guide](R2_BACKUPS.md).
- pgvector/index scaling after measured need; advanced memory/entity/alias
  reconciliation, linked-fact correction and retrieval/context feedback.
- Task dependencies/blocked reasons and metric history when warranted; Quick-list
  templates; richer note-source lifecycle and opt-in note-to-memory review.
- Broader Linear project/label/cycle controls, an Eri agent inside Linear, native
  recurring appointments, MCP OAuth and signed webhooks.
- Public self-service, broader tenancy/RLS/encryption and sharing permissions;
  Android, background wake-word service and offline-first sync.
- Ambient conversational clarification beyond setup/review; Home Assistant,
  finance and general research/execution agents; further backend-model comparisons.
- Wholesale legacy-table/directory removal or whole-App refactors.
  Langfuse historical eval export is implemented; continuous production tracing
  remains optional and is not required for the latency instrumentation.

## Maintenance

Add new requests to the matching active batch or deferred list. Keep one delivery
order. Record dates/commit/evidence when closing items; archive finished release
narratives instead of appending another competing roadmap. The
[full historical ledger](archive/2026-10-02/TODO_HISTORY.md) preserves detailed user
stories, prior test counts and old decisions; it is not an additional execution queue.
