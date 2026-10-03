# Eridani web v1 — completion PRD

Status: active delivery plan. Batches A–D (#22–25) merged. Section 10 records the October 3 follow-up plan; its changes are not yet implemented.
See [current TODO](TODO.md) and [Batch D verification](BATCH_D_VALIDATION.md) for current evidence.
Prepared October 2, 2026, against local source `e734f7f` on `fix-voice-requests`.
This is a documentation/reconciliation pass, not a new test run or deployment.

## 1. The finish line

Eridani is a personal productivity workspace with an assistant: capture a commitment,
organize it in a system you choose, connect it to useful notes, see it in your day,
and trust that Eri's actions happened exactly as reported.

**Web v1 is finished when Davis can use it daily, and an invited person can set up
their own workspace without a developer explaining the schema.** It must work on
desktop and the Pixel 10 Pro Fold, including folded/unfolded use, real voice,
connected calendars/tasks, and locked-phone reminders.

This is an **invite-only web release**, not a promise of public SaaS readiness.
Existing accounts, shared workspaces and bot access must remain isolated and usable.
Public self-service growth, additional tenancy defense, native Android, home control
and finance are separate milestones.

The main change in direction is consolidation. We have most of the capabilities.
The remaining work is to make their behavior consistent, their organization legible,
and their reliability demonstrable. Do not start another general UI redesign.

## 2. Evidence and how to read status

Sources reviewed:

- [Claude session summary](CLAUDE_SESSION_SUMMARY.md), [audit findings](archive/2026-10-02/CLAUDE_AUDIT_FINDINGS.md),
  [Orbit design system](DESIGN.md), and [CI notes](CI.md).
- [Current backlog](TODO.md), [original PRD](archive/2026-10-02/JARVIS_UPGRADE_PRD.md),
  [upgrade decisions](archive/2026-10-02/JARVIS_UPGRADE_DECISIONS.md), and [background-work PRD](BACKGROUND_WORK_PRD.md).
- Current voice intake, planner availability, model profiles, task detail/structure UI,
  normalization, audit regression tests, and Railway declaration.
- [Eval findings](../evals/app/FINDINGS.md) and
  [Sol's independent grading report](../artifacts/app-evals/half-20260923-sol-graded/grader.md).

Use four distinct states: **implemented**, **automatically verified**, **accepted in
real use**, and **deferred**. A checked implementation task is not device acceptance.
Historical reports retain their original results after a repair.

Claude reports 783 backend passes/one skip under UTC and Chicago DB settings,
82 eval-harness passes/two skips, 166 frontend passes/one skip, and six browser
acceptance suites. Those are reported results, not rerun in this planning pass.
The presence of a regression test is source evidence, not proof it currently passes.
At the original planning snapshot, the coverage baseline had 98 of 1,001 fully bound scenarios and 224 overlapping
component cases. The ratchet prevents loss of coverage; it does not establish full
app acceptance. Sol's historical 501-case review found 42 passes, four failures,
six needing review, 75 partial, 373 unassessable and one component failure. This
regression-weighted sample is not an unbiased quality percentage, and it predates
the repairs. Re-run/regrade relevant cases before closing their acceptance status.

Final GitHub readback during this review: “Fix voice requests being captured by
pending questions” is merged, with all three CI jobs successful (merge c1808e6).
Davis completed the merge separately. The inspected local source remains e734f7f;
verify the merged/deployed revision and both services before calling that repair
live. No production DB inspection or cloud-state mutation was performed here.

The main-branch rules readback contains required status checks, no force pushes,
and no deletion. It did **not** return a required-pull-request rule. The summary's
“nothing can land without a PR” statement is therefore too strong. Keep required
checks, explicitly add/verify a PR requirement if that is the desired policy, and
independently verify Railway's CI gating. Do not infer deployment from merge.

## 3. Reconciliation: what Claude already covered

| Backlog area | Current evidence | Remaining work |
| --- | --- | --- |
| Queue recovery, repeat execution, budget holds | Run lock, reaper, per-revision reservations, catch-all, scan isolation and tests exist | Combined revise/cancel/restart scenarios and actual queue/voice acceptance |
| Premature voice delegation and duplicate questions | Quiet speech retained; explicit Live delegation; single question announcements; follow-up PR removes broad answer capture | Real multi-turn voice, missed delegation at close, unrelated next request, answer arriving during execution |
| EVAL-001/002: Revert and clearing notes | UTC instant comparison, nullable-field contract, recovered-error handling and tests | Repeat the original paid cases; preserve stale-write guards |
| EVAL-004/008: flexible task homes | Compact task resolution exposes record/home paths; minimal two-way legacy sync | Repeat duplicate-title trials; establish one organization authority |
| EVAL-005/006/007: memory and note extraction | Numeric conflict review, facts persist without vectors, retries and omission reporting | Real-model repeats and source/completeness assertions; omission warning is not complete extraction |
| Spoken “test test 123” search | Shared spoken-number normalization plus task/note/search tests | Eri deletion/disambiguation flow against near matches in real use |
| Account and bot authority | Scope checks, session/key revocation, owner-issued invites and logout fixes | Two-account, shared-workspace and scoped-client acceptance |
| Calendar, Linear and push repairs | Sync fencing, stale-read guards, failed-publish recovery, delivery dedupe/resubscription tests | Real provider writes, conflicting edits, rate-limit behavior and locked-phone delivery |
| Compact UI, settings and navigation | Today landing, grouped rows, light/dark theme, dialog primitives, settings rail, reduced polling and local UI registry | Specific foldable/keyboard/accessibility defects, conflict presentation, tool acknowledgement |
| CI/security/eval automation | Three CI jobs, dependency/secret audits and coverage ratchet | Migration compatibility, risk-weighted release suite, honest coverage and deployment-gate readback |
| New product gaps | Quick lists, source colors/private annotations, visual structure editor and onboarding still pending | Deliver batches below |

CopilotKit is no longer the runtime dependency. Keep the conversational-site-control
capability through the local registry; do not reinstall a framework to satisfy an
old backlog wording.

The landing/login pages, calendar writes/multiple calendars, task/reminder workspace,
task boards/timelines, notes/list filing, routing dreams, accounts, API/MCP and Langfuse
historical eval export are already implemented. They need verification or targeted
completion, not rebuilding.

## 4. Changes from the original PRD

- **Hosting:** Railway API + worker + PostgreSQL, not a home-PC-dependent service.
  Keep production/dev/CI on the same intentional PostgreSQL major. Do not adopt
  a major Dependabot upgrade independently of migrations, extension and restore compatibility.
- **Voice:** GPT-Live with durable backend work; Realtime stays disabled. Luna remains
  the active backend/eval baseline. Gemini comparisons stay paused.
- **Storage:** PostgreSQL is canonical. Current memory/search use custom application
  services and embeddings; old Mem0/Qdrant/local-GPU plans are not the live architecture.
- **Product:** flexible types/fields/relationships replace mandatory goals/projects.
  Defaults remain helpful starting points, not restrictions.
- **Ordering:** Home Assistant, finance and native Android no longer define the next
  release. Finish the planner before expanding into new domains.
- **Cost:** continue tracking and budget controls. Report feature/category costs,
  measured usage and uncertainty. Keep the existing $10 campaign ceiling; never
  silently buy a larger test campaign to satisfy repeat counts.
- **Recovery:** Claude records the owner's deferral of R2, restore drills and key
  escrow until broader multi-user/privacy work. Preserve that deferral and existing
  recovery facilities. Their current operation is not freshly verified here.

## 5. Product and architecture decisions

### One work item, multiple presentations

Tasks are actionable commitments. Due dates/times, reminders, recurrence and planned
calendar blocks describe tasks rather than creating duplicate task engines.
A standard calendar event need not be completable. A task deadline is not a reserved
work block; show that distinction in calendars and detail views.

Notes are authored/source content. Personal memory contains derived assertions with
provenance and correction controls. Organization learning is separate evidence and
rules about where work belongs. A note can produce linked tasks/list entries without
being silently replaced by memory.

### Flexible structure with a single authority

Use the existing structure records, stable IDs, fields, descriptions and named links.
A record may have one main home and additional cross-links. A tree displays that main
home hierarchy; it must not pretend all relationships form one tree.

Make structure relationships the authority for organization reads and writes.
Keep task and note domain services for completion, schedules, content, integrations
and revision checks. Legacy project/space adapters can remain temporarily, but must
derive the same answer. Start with parity/backfill and adapters; removing all legacy
tables is **not** a prerequisite for v1 if consistency is proven. Never do a big-bang
storage rewrite merely to simplify the diagram.

### Conversation is not a review queue

Live gathers the minimum missing information. The backend executes an actionable
request; independent requests may run concurrently and related writes remain ordered.
Receipts describe committed effects, not proposed interpretations. Navigation and
small talk do not create action receipts.

A pending question must never automatically capture every later request. Use the
existing explicit clarification/continuation tools and persistent IDs. Repeated answers,
late results and reconnects must converge on one logical request. No separate intake
interpretation model is required.

**Close-time proposal:** retain ambiguous or undelegated speech as one resumable draft
with Send/Discard, rather than treating the presence of an action verb as permission
to enqueue it. Already accepted backend work continues after voice closes. Implement
this as a small, tested recovery path, not a second inbox users must manage.

### Learning starts useful, not overconfident

Keep explicit, user-confirmed rules usable immediately; show learned candidates in
the weekly interview. Agent actions and silence are not equivalent to human-approved
training labels. Uncorrected searches may be weak evidence, not immediate authoritative
aliases. Do not lower automatic-rule quality thresholds just to show more automation.
Expose why a rule is pending, active or paused, and how to correct or forget it.

## 6. Delivery batches and acceptance

Each batch ends with focused regressions, the existing CI gates, a reviewed PR, and
a deploy smoke check when released. This PRD itself does not authorize a deployment.

### Batch A — establish a trustworthy baseline

Implementation/verification began October 2; see [Batch A evidence](BATCH_A_VALIDATION.md).
Recovery and dispatch code is implemented locally; deployment, physical voice checks
and real end-to-end latency sampling remain open.

**A1. Finish the voice repair release.** Verify the merged/deployed revision, then
exercise summary-with-optional-offer → unrelated task request; incomplete request →
detail answer; and answer → original card resolution. Include a follow-up arriving
while the original job is still running. No swallowed request, duplicate mutation,
or orphan waiting card is acceptable.

**A2. Close queue/recovery seams.** Test concurrent unrelated work, ordered same-record
edits, cancellation during a provider call, budget session revision, worker crash and
reconnect. Verify actual persisted effects and final receipt state, not only response text.
Preserve incomplete voice input as the explicit draft described above.

**A3. Reconcile operations documentation/configuration.** Inventory cloud services
read-only. The summary says PG18 was deleted, but the Railway whole-project declaration
still contains it and its volume. Reconcile only from verified inventory and review
the plan for destructive changes; do not apply it blindly. Verify active DB version,
always-on worker/DB expectations, CI gating and the real branch rules. Triage dependency
majors separately rather than bulk merging them.

**A4. Baseline and remove avoidable latency.** Follow Steps 0–1 of the
[audited response-speed plan](ERI_LATENCY_PLAN.md): stage timing through actual
web /work and Live paths, then dispatch wakeups and measured result/tool-discovery
delays. Keep the durable runner, dependency ordering, transcript completeness and
conversational turn protection. Measure useful visible/spoken outcomes, failures
and cost; do not infer total savings by adding overlapping timer constants.

**Exit:** the current release is identified, the highest-risk voice regressions have
reproducible checks, and infrastructure declarations cannot silently recreate or
remove a database. Any unavailable real-device checks remain explicit release gates.

### Batch B — make the existing workspace coherent

**B1. Organization authority and visual Structure view.** First test equivalent reads
and moves through UI, task tools, record tools and external API. Then add a compact
tree for actual records and a clearly separate types/fields/relationships editor.
Show descriptions and inheritance. Drag to reorder/reparent only where valid; offer
Move controls for keyboard/touch. Preview schema changes, reject cycles and invalid
parents, retain links/history, and surface stale-write conflicts. Update Eri's site map
and commands after the UI settles.

**B2. Shared external-source contract and detail cards.** These requirements apply
to **every current and future external connector**, including Google Calendar and
Linear, and to every imported record kind: tasks, events, notes and other supported
records. Connector support is a capability boundary, not an exception to the UX.

Show a literal provider badge such as **Google Calendar** or **Linear**, its icon and
editable source color in lists, boards, timelines, calendar, search results and
detail cards wherever the item appears. Color alone is insufficient. Retain stable
provider/account/container/item identity (including recurring-instance identity
where applicable) and the original-item link when available; show calendar, team or
account context where needed to distinguish sources. Source identity is integration
metadata, not a user-maintained classification tag, and survives local moves or
renames. Native records remain identifiable as Eridani.

Inventory each connector's actual payload/schema and supported operations. Display
available source details in readable groups with editable, read-only and local-only
states. Include provider-specific metadata rather than reducing every item to shared
task fields; do not invent unavailable fields or imply all remote features are
supported. Google Calendar appointments remain events, not completable tasks;
preserve all-day/exclusive-end semantics and explicit occurrence-versus-series scope.
Local hierarchy/custom fields do not become remote projects, labels or calendars
without an explicit mapping.

The default contract for already-linked records is **edit here, update the source**.
All source-backed edits through details, lists/boards, calendar, bulk actions, Eri
or external-agent tools must update the same remote item through the durable connector.
Do not silently create a local fork or require per-edit publishing. Initial account
consent/write access still applies; read-only or unsupported edits are visibly
unavailable with a reason. Native records are not automatically published externally.
An explicitly published/linked native record follows this same contract afterward,
for its mapped fields. A link to an unrelated record is not a sync relationship.

Inbound updates and outbound edits use the same declared field mappings and preserve
local-only data. Keep provider revision/conflict checks, durable idempotent retry and
sync-loop prevention; expose pending, confirmed and failed/conflicted states. Eri and
the UI distinguish saved locally from confirmed at the source. Supported completion,
deletion and Revert use the same connector semantics and existing action safeguards;
do not treat local archive as remote deletion or overwrite a newer remote edit.
Multiple external links must have explicit field ownership rather than silently
broadcasting an edit to every connected service.

Add collapsible **Eridani-only notes** to imported-item details, stored separately
from provider descriptions. Here private means not synced to any provider; existing
account/workspace visibility still applies and must be explained in the UI.
Author-only sharing is a separate future permission feature. Current Linear
Task.notes is a synced description and cannot be relabeled as private. Outbound
payloads never include local annotations; inbound sync and reverting a source edit
preserve them. Apply existing scope rules consistently to read, search and tool access.

Verify the contract for every supported connector and editing surface: remote
readback, incoming changes, duplicate/retried delivery, concurrent edits, disconnected
or read-only accounts, provider-specific record semantics and local-note preservation.
New connectors must meet this contract before being described as supporting two-way
sync. Google Calendar and Linear are the initial acceptance targets, not the limits
of these requirements.

**B3. Scheduling and notification completion.** Intersect planner availability with
editable planning hours, timezone and task constraints. Distinguish scheduling hours
from the weak work/personal routing hint; a personal task is not automatically limited
to work hours. Explicit user overrides are visible. Test overnight windows, DST,
all-day exclusive end dates and genuine infeasibility. Reuse current notifications;
finish natural-language snooze/quiet-time/priority behavior and verify no duplicate
task-deadline/reminder alert. Completion receipts stay in Activity/chat, not notifications.

**B4. Targeted UX reliability.** Shared conflict treatment preserves the draft and
offers compare/reload/reapply when allowed. Resolve launcher/action overlap, due-chip
overlap and cramped week-view collisions. Preserve mobile Back, scroll position,
keyboard focus, dirty-edit handling, bottom sheets, touch movement and light/dark
contrast. Add guarded note append/anchored edits to avoid rewriting a whole note
for a small change. Verify assignee identity resolution and visible-result
acknowledgement before Eri says a filter/view has been applied.

Refactor only the view/data/voice boundaries touched by this work. Splitting all of
App.tsx is maintenance, not a new product milestone.

**B5. Conditional compact execution.** If A4 still misses the latency targets,
test an explicit Luna no-reasoning profile alongside the current low-reasoning
option, then compact tools/context only where measurements justify it. Both
channels retain the existing durable runner. A broader-profile continuation
preserves the same work root, receipts, dependencies and cumulative budget.
Expose a clear Settings choice; default-on requires correctness parity, useful
measured improvement and no average cost increase per successful request.
Provider support is documented; app configuration and quality are still untested.
See the latency plan for rollout and regression gates.

**Exit:** a record has the same home, source, schedule and private annotation semantics
from every entry point; organization and editing are understandable on a phone.

### Batch C — capture quickly and help new people start

**C1. Quick lists.** One-step title + checklist capture, optional sections/deadline,
inline completion/reordering and Today pin. No mandatory client/project/goal setup.
Use a lightweight container within the existing record/task system, not another queue
or a new task engine. Show one list summary by default, retain searchable item IDs,
and let Eri add/check/reorder items conversationally. Optional promotion to a project
must preserve IDs, completion and history. Templates can wait.

Acceptance story: “Get ready for Houston” with Hayes, personal packing, moped supplies
and work essentials; add items while packing, close/reopen, check them off and ask
what remains. No invented commitments or permanent classifications from section names.

**C2. Guided conversational onboarding.** After sign-in, ask preferred name, timezone,
what the person wants to track and how they organize it. Offer a short skippable flow
with voice/text and sensible defaults. Capture required field descriptions and ask
targeted clarifications. Preview a small example and schema before applying it.
Save progress per account, support resume and later editing, and do not replay setup
for returning users. Separate personal-account invitations from workspace membership
and explain who owns/sees data. Reuse the existing landing/login flow.

**C3. Organization learning transparency.** Connect onboarding descriptions and
user corrections to the existing field-understanding/rule-dream mechanisms. Surface
weekly questions with evidence and confirm/edit/dismiss controls. Keep personal memory
and routing rules visibly separate. Basic usefulness cannot depend on collecting
hundreds of examples. Automatic activation beyond confirmed rules remains gated.

**C4. Finish brand assets.** Choose the peeling-note variant, make production SVG/icon,
small-size and monochrome forms, and apply it consistently to login, app and PWA assets.
The generated raster concepts are not already installed. Preserve the established
Orbit UI and user-preferred Eri launcher.

**Exit:** a new invited user can capture, find and complete their first task and Quick
list, explain their chosen structure, and return later without a guided developer session.

### Batch D — qualify the release, then pilot

October 3: [automated verification](BATCH_D_VALIDATION.md) binds and passes 215
scenarios across recorded offline/Luna campaigns. Twenty feature groups have fully
bound cases; remaining feature-specific acceptance gaps, physical/provider checks
and the pilot are explicit in [the acceptance checklist](RELEASE_ACCEPTANCE.md).
This is not a whole-app release sign-off.

**D1. Risk-weighted eval core.** Select approximately 200 scenarios from the existing
1,051-case catalog; retain the rest as exploratory/expansion coverage. Cover every
currently shipped feature, weighted toward authority, mutation/Revert, queue/voice,
data-loss recovery, integrations, search/learning and UX. Fully bind each core criterion
or explicitly identify connected/device evidence. A component test cannot pass a
whole acceptance scenario. Repair Sol's six overbroad bindings.

Run deterministic suites once per unchanged build. For stochastic core paths, use at
least three independent Luna trials against reset synthetic fixtures. Include all
EVAL-001–008 regressions. Plan/reserve cost first, keep the $10 campaign ceiling,
stop safely if exhausted, and report incomplete coverage rather than bypassing it.
Independent grading uses saved traces and writes; Langfuse remains the export/reporting
surface. No scheduled paid evaluations are required.

**D2. Migration and access safety.** Add N-1 schema compatibility coverage for rolling
deployments, enforce expand/contract changes, and test isolation across accounts,
shared-workspace roles, removed members and scoped bots, including new annotations.
Do not assume adding RLS is necessary to prove the current authorization contract;
broader multi-user defense remains a separate project.

**D3. Connected and physical acceptance.** Use dedicated Google/Linear test items for
create/edit/readback/conflict/retry; measure Linear sync call volume and rate handling.
Use the real Pixel folded/unfolded plus desktop: wake phrase, immediate request,
30-second idle behavior while work runs, sign-off, long/paused speech, two requests,
late correction, Wi-Fi/cellular loss, close/reopen, locked-phone push and accessibility.
Exercise cloud worker restart and operation with the home PC off. Browser emulation
does not count as physical acceptance. Confirm Google consent readiness before expanding
invitations; never claim approval from working test-account sign-in.

**D4. Seven-day pilot.** Only after expansion and the checks above, complete at least
50 successful task/reminder interactions. Log attempted/successful interactions,
corrections, blocked work, voice comfort, latency and per-feature costs. Repair material
failures before calling v1 complete.

**Exit:** release scorecard below is signed off with evidence. Native Android planning
then reuses tested identity, command, record-link, clarification, sync and notification
contracts; it is not a parallel rewrite.

## 7. Release scorecard

- Zero unresolved critical/high-severity access, data-loss, duplicate-write or
  stale-overwrite defects. Every known EVAL issue has a repair or an explicit,
  acceptable containment plus a passing relevant regression.
- All deterministic release-core checks pass; stochastic critical mutation/authority
  cases pass all planned repeats. Noncritical quality misses are recorded and assessed,
  not averaged into a misleading whole-app pass rate.
- No lost accepted request, wrong-record mutation or orphan clarification in the
  scripted real-voice acceptance set.
- Physical locked-phone reminder received, snoozed and resolved; provider acceptance
  alone does not count as device delivery. External writes are read back.
- Onboarding, Quick lists and Structure work on folded/unfolded phone and desktop,
  including keyboard-only alternatives and no action hidden by the launcher.
- Seven days and at least 50 successful interactions, comfort average at least 4/5,
  with failed attempts and correction rate also reported.
- Measure API latency and commit-to-visible latency separately from model/voice.
  Retain original warm targets: local task API p95 below 250 ms and connected UI update
  p95 below 500 ms after commit. Measure speech-end and turn-ready-to-useful-feedback
  separately; the original 1.2 s p50 / 2.5 s p95 voice targets remain goals pending
  a GPT-Live baseline, not achieved claims. Document any accepted miss.
- Weekly measured cost by voice, backend, extraction, embeddings and dreams; show a
  monthly projection with workload assumptions and infrastructure separately. The
  historical $150 API budget is a ceiling reference, not a spending target. Do not
  extrapolate the tiny synthetic half-suite cost as a full month of real voice use.

## 8. Deferred scope and decision boundaries

Remain on the backlog, not in this finish-line critical path:

- pgvector/index scaling, driven by measured retrieval latency/data size; do not
  confuse storage optimization with new semantic-search functionality.
- R2, isolated restore drills and key escrow under the recorded owner deferral.
  Revisit before broader multi-user/public release; retain existing PITR and report
  its verification limitations honestly.
- RLS/encryption redesign, public self-service expansion, richer sharing permissions.
- Native Android, background wake-word service, offline-first sync.
- Home Assistant, finance, arbitrary research/execution agents.
- Additional backend comparisons. The Luna no-reasoning trial is now included
  conditionally in B5; implementation/default rollout still requires evidence.
- Proactive clarifications during ordinary conversation outside setup/review.
- MCP OAuth for clients lacking bearer support, signed webhooks, advanced Linear
  project/cycle editing and recurring native appointments.
- Full legacy-directory/table removal and a whole-App refactor once parity is proven.
- Continuous production Langfuse tracing unless diagnostics demonstrate a need.

The original A → B → C → D implementation sequence is complete; physical/provider acceptance remains open. Section 10 and the active TODO describe the next follow-up. Planning defaults can be revised before implementation.

## 9. Backlog maintenance

The active checklist at the top of [TODO.md](TODO.md) owns the delivery order.
Older sections retain history; their repeated unchecked items do not create extra
release requirements. Use this PRD's identifiers in follow-up PRs. A completed batch
records changed behavior, CI evidence, deployment state and remaining manual checks.

Historical eval artifacts and Sol's grading remain immutable. Append repair/rerun
evidence instead of rewriting old failures as passes. New source fingerprints,
trial counts and real-service/device limitations belong in each new report.

## 10. Organization and review-delivery follow-up

October 3 owner direction: provide useful browsing and visual customization.
Organization opens in Browse; Types & fields opens in a tree, retaining the old
editor. The owner also approved a visible questions inbox and reliable dream-question
delivery. The guided structure-design interview stays backlog only. This is the
plan before implementation, not a claim of shipped behavior or new tests.

### 10.1 Two trees with different meanings

The record tree contains actual things: for example Work → ABC → Transcript
Intelligence → Finish central docs. A record has one main home and optional extra
links. Moving it changes home/configured inheritance under existing permissions,
revisions, source mappings and cycle checks.

The type tree contains definitions and their fields. Space → Client → Project →
Task is an illustrative arrangement, with dates/deadlines as distinct field leaves.
It does not mean every project must have a client. Today's permitted-parent schema
is a graph with multiple homes and type-level loops; actual record cycles remain
invalid. Do not render every permission recursively or reduce it to one legal parent.

**Accepted direction:** a flexible visual arrangement
with optional explicit restrictions. Dragging a type changes its diagram position;
editing “Can live inside” changes filing permissions through impact preview. Neither
silently reparents records. Adding a nested type may prefill a proposed permitted
relationship, but cannot silently restrict it to one parent. A diagram never chooses
an arbitrary actual client/project for a new record.

### 10.2 Organization: Browse first, Structure available

- Show actual top-level containers and Unfiled, respecting custom names/types rather
  than forcing Work/Personal or hiding nondefault types.
- Container pages show compact description/context, contained groups, Tasks, Notes
  and Related, with All contents for other types. Group by capabilities, not
  hard-coded IDs; avoid duplicate counts for multi-capability records.
- Distinguish direct children, all descendants and linked records. Related items
  are not automatically contained. Counts/empty states match scope/status/archive filters.
- Add here selects the actual home and valid active types using existing creation
  tools; it does not silently publish native records to external providers.
- Retain actual-record Structure, list/board/timeline layouts, inline detail editing,
  source indicators and accessible Move controls. Remember branch expansion.
- Container titles open container pages; a clear details affordance opens their
  cards. Leaves open details. Keep these actions predictable on touch and keyboard.
- URLs retain container/view/filter context. Back closes details or returns to the
  previous page with scroll restored and no reload. Preserve drafts/dirty guards.
  Folded screens use focused panels rather than cramped desktop columns.

### 10.3 Types & fields: visual first, Advanced retained

- A finite tree/map shows distinct type/field nodes; all custom, unplaced and
  archived definitions remain discoverable. Show self-nesting and other legal homes
  as connections, not infinite branches. Extra relationships are not containment.
- Click a type/field for focused settings; contextual Add type/Add field, required
  descriptions and plain-language behavior/status controls.
- Advanced retains the full existing editor. Both views edit one draft with the
  same preview/apply path; switching must not discard edits or fork the schema.
- Reorder type presentation and fields within a type by drag plus touch/keyboard
  alternatives. Cross-type field transfers need a data/binding migration design;
  do not disguise them as harmless visual moves.
- Renames preserve IDs; archives preserve history. Permission/behavior/field/status
  removal previews affected records and required remediation before apply.

### 10.4 Backend and compatibility

Reuse StructureSchema, StructureRecord and StructureLink, not another task engine
or per-custom-type tables. Add scoped/paginated container, subtree and related reads/
counts if needed. Lazy-load branches with complete breadcrumbs; never label a
truncated flat response as the entire tree.

Store the flexible map's optional presentation metadata apart from
semantic definitions, preferably in existing schema JSON. Validate known IDs,
single presentation nodes, ordering and presentation cycles. Omitted types appear
at root. Existing accounts see every type; starter maps are suggestions, not inferred
rules. Likely API/schema changes, no record rewrite or SQL migration for the map.
Verify older strict validators, mixed-version behavior and rollback before finalizing
storage; no-SQL-change does not mean automatically backward-compatible.

Presentation edits must not invalidate field understanding or retrain rules.
Semantic edits retain existing impact/learning checks. Workspace schema permissions
remain enforced; personal expanded-state/tab preferences are separate. Local filing
does not change remote provider project/calendar mappings. Update Eri's site map and
navigation tools after routes settle; navigation creates no action-history card.

### 10.5 Questions inbox and dream delivery

Adapt existing MemoryReview, RoutingReview and FieldUnderstanding through a personal
Questions API, retaining domain authority. Return kind, stable source ID, revision,
question, evidence and actions. Quiet count; Answer/Review, Later and appropriate
dismiss/distinct choices. Organization shows its relevant subset. These are optional
reviews, not required work clarifications or successful-action notifications.

Separate pending/deferred/resolved/dismissed-or-distinct/stale questions from
reserved/presented/interrupted/expired invitation delivery. Add a small durable
delivery record only if existing data cannot express this, scoped to account,
question revision, conversation/channel, with a short reservation lease. A lost tab
must not suppress offers for a day. Existing personal-learning exclusions for shared
workspaces remain; this inbox does not share personal memories with other members.

Repair code-supported gaps: ordinary Live refresh omits memory questions; offers at
the tail of successful result prose can be hidden by cards or truncated by Live's
420-byte result forwarding; visible/spoken acknowledgement is incomplete. Send an
eligible invitation separately at a natural pause after the user's request. On
explicit review start, pass one question ID/revision to existing backend/resolution
tools. No new interpretation or intake model.

Visible rendering may acknowledge text presentation. Context/provider ACK or a
generated transcript cannot prove completed voice playback. Correlate playback
completion where supported; otherwise retain honest status and a visible card.
Invitation presentation is not question presentation. Interruption never resolves
a question. Coordinate devices, persist defer/answers, reject stale revisions and
refresh after another device answers.

Required task clarification has priority. Unrelated work pauses optional review
rather than becoming its answer. Silence never approves a rule or memory merge.
Ask one at a time with a stopping point. This repairs scheduled-review delivery,
not deferred ambient interviews.

Inspect account jobs/results when authorized access is available. Show actual last
successful runs, pending reviews, merges/rules and failures with provenance.
October 3 proves deployed worker activity, not a successful account dream run or
useful extracted findings; those require persisted account evidence.

### 10.6 Deferred: Chat about it

Future Types & fields button: a guided, resumable Eri workflow using the current
schema, selected type and concrete examples. Clarify descriptions/homes/links; show
before/after and require explicit apply. Support skip/resume and workspace access.
Do not create sample work or permanent facts from hypothetical examples. Reuse the
current runtime/tools; this is a product workflow, not a new agent or Codex skill.
Defer both button and interview, avoiding an inert CTA. Existing field review stays usable.

### 10.7 Delivery order and acceptance

1. Inventory hierarchy consumers and conflicts; design the canonical Home transition
   and reusable field-library compatibility before building the new editing surfaces.
2. Implement core field/home contracts, actionable projects, subtree move/removal and
   same-type subtasks. Preserve Quick lists, recurrence and source mappings.
3. Implement Browse/container pages, visual Types & fields and Advanced over the same
   contracts; then update Eri's site controls and nested timelines.
4. Implement Questions/delivery as a separate focused change, tested independently.
5. Focused regressions and CI, then desktop/Pixel Fold, actual voice/provider acceptance;
   keep existing pilot gates.

Test ABC's projects/tasks/notes and linked goals; custom Course types; standalone
projects; multiple legal homes; self-nesting and cyclic type permissions; empty/
archived containers; more than one page of data; stale previews; viewers and removed
members. Record/source IDs survive view/map changes. Verify Back, deep links, drafts,
folded/unfolded editing and keyboard movement. Hidden/long results cannot swallow
offers. Live-only review, interrupted/reconnected speech and two devices retain
one answerable question. Task answers/unrelated work cannot silently approve rules
or merge memories.

### 10.8 Accepted record/field/move decisions and subtask consolidation

October 3 discussion confirms generic records, modular reusable fields, one primary
Home with optional extra links, move-with/without-contents, and explicit parent
completion. Subtasks remain ordinary records in one authoritative Home hierarchy.
Advisory blocking is accepted. No open product decision blocks implementation;
engineering preflight/migration verification is part of the build. Implementation
remains pending.

**Editable behavior, reusable fields, separate building from use**

- Any type can enable actionable work, including Client. Project should offer the
  actionable-work plus timeline preset: complete/status/assignee/priority/deadline,
  description and nested work. Do not force these behaviors onto every record type.
  Preserve custom schema choices when upgrading defaults.
- Introduce a workspace-scoped reusable field library with stable identities and
  per-type attachments. The library owns meaning, value kind, description and any
  operational binding; attachments own order, visibility and permitted per-type
  configuration. Values remain per record. Workflow statuses can retain type-specific
  choices mapped to shared operational meanings.
- Seed Priority, Due date/time, Do date, Assignee, Status and timeline dates as
  supported defaults; allow new custom fields. Due date is the deadline. Do date
  is when the user intends to work, backed by existing planned_date semantics.
  A Do date does not reserve a calendar slot or change the deadline.
- Create/attach/detach fields in Types & fields, shared between visual and Advanced
  editors. Detail cards edit values only, with an optional Customize this type link
  to the focused builder, not inline schema creation. Show all affected types before
  editing a shared definition. Detaching a field must preserve data/history or
  present an explicit migration; never silently discard values.
- Migrate current inline field definitions deliberately. Do not merge custom fields
  merely because labels match. Canonical operational bindings keep their validated
  types/behavior. Schema version, field-understanding, routing evidence and Revert
  need migration/compatibility coverage; library reuse is not cosmetic presentation.

**Home picker and learning**

- Browse actual records progressively, for example Work → ABC → Transcript
  Intelligence; allow stopping at any permitted home and direct search/jump.
  Save the chosen immediate parent once, deriving the breadcrumb from ancestry.
  Do not separately require Work, Client and Project assignments.
- Extra relationships are collapsed by default: Related to, Supports and explicit
  Blocks/Blocked by where supported. Blocking is directed dependency behavior,
  not merely a renamed generic link; reject self/cyclic blockers. Blocking is
  advisory: show unfinished prerequisites and account for them when Eri recommends
  work, but permit an explicit start/completion after a clear warning. Do not require
  users to complete prerequisites just to record real progress. Starting/completing
  the dependent task does not complete its blockers or remove their links. Avoid
  silent rescheduling/completion; no automatic dependency rescheduling in this batch.
  Dependencies remain distinct from containment and queue dependencies.
- Manual filing supplies independent routing evidence. Rules point to stable home
  IDs and current ancestry; repeated auto-assignment, subtree moves and incidental
  ancestry changes must not count as many new independent training examples.
  Conflicting explicit classifications need review, not silent overwrite.

**Move and delete contents deliberately**

- Move with contents is the default. Change the parent record's home; children keep
  their parent references and follow it. Keep notes, deeper descendants and their
  own values/dates together; retain extra links. Recompute derived context without
  copying operational deadlines, assignees or priority down the tree.
- Move only this item promotes its immediate children to its previous home. Each
  child retains its own subtree. If the old home is root, they become root records;
  if it cannot accept a child, require a valid destination before applying.
- Deleting an actual container offers delete with all descendants or delete only
  this item and promote children to its previous home. Show affected counts/types
  and provider consequences. Related records are not descendants and are never
  cascaded merely because of a link. Retire a type through its separate schema flow.
- Preserve archive versus delete semantics and favor recoverable local removal.
  Native subtree changes must be atomic, revision-checked and reversible, including
  sibling order and child destinations. Revert may not overwrite newer independent
  changes. External deletes follow connector permissions/semantics and durable
  per-item receipts; do not claim remote atomicity or turn local archive into deletion.

**Keep subtasks; use one hierarchy**

- A subtask is an ordinary actionable record whose Home is another actionable
  record. Add subtask creates the usual task preset beneath the current item;
  no Subtask type, schema setup or second parent selector is needed.
- Same-type nesting is supported when permitted (Task inside Task), displayed as
  a self-nesting connection in the type editor rather than another duplicate type.
  Each child retains its own Do/Due dates, assignee, status and links.
- StructureRecord.parent_id is the proposed canonical local containment authority.
  Core task parent/project fields must become compatibility projections/adapters,
  not independent writable sources of truth. Translate legacy API/tool writes to
  the canonical move command. Resolve core task IDs via StructureRecord.task_id;
  do not assume record and task IDs always match.
- Audit and migrate Quick-list membership/order/sections/progress, task parent APIs,
  core/record observers, search, routing, planner and source-sync consumers before
  retiring old columns. Quick lists stay lightweight views over ordinary nested work.
- The legacy parent_task_id also stores recurring-occurrence lineage today.
  Separate origin/template linkage from actual Home; do not turn recurring instances
  into children of a hidden template during migration. Preserve occurrence/schedule
  identity, idempotency and existing completion behavior.
- Linear's imported parent/project relationships need an explicit source adapter.
  Preserve remote identity and provenance, distinguish remote hierarchy from chosen
  local Home, and prevent sync from undoing a user-directed local move unless an
  explicit supported mapping makes it source-owned.
- Inventory existing disagreements, classify relationships by provenance, and
  surface ambiguous cases rather than arbitrarily choosing a winner. Cut over reads
  and writes in stages with rollback verification; remove compatibility storage
  only after all supported entry points use the canonical authority. This data/
  behavior migration is separate from the optional presentation-only type map.

**Nested completion and timelines**

- Children contribute progress; all children done makes a parent ready to complete,
  not automatically completed. Parent completion with unfinished descendants gives
  a clear choice. Parent completion/reopening never silently cascades child statuses
  or hides still-open child commitments.
- Expand parent/child rows on timelines. Distinguish the parent's own planned span
  and deadline from a derived child span; show out-of-range children. Filing moves
  do not shift dates. Whole-schedule shifting requires a separate explicit operation.
- Avoid double-counting container work and descendants in effort/progress summaries.
  Do not count linked items as children or infer completion from dates alone.

Additional acceptance: reuse one field on Client/Project/Task without sharing values;
independent Do/Due; make Client actionable; add a subtask without type creation;
nested assignees/timelines; carry/promote/delete/restore including notes/grandchildren;
invalid destinations and stale Revert; recurrence lineage; Quick-list sections;
Linear refresh after local filing; and old/core API commands reaching the same Home.


### 10.9 Implementation and acceptance evidence — October 3

The foundation, contents backend and Organization UI shipped through PRs #26–#28,
each merged after required CI passed. The final review-delivery batch uses the
existing memory/routing/field question sources and resolution tools. It adds a
personal Questions page (plus an Organization subset), source-revision snoozing,
and a separate invitation table with two-minute leases. Concurrent devices reserve
one invitation; presentation never implies an answer. A text invitation needs a
visible-element acknowledgement. Live records only forwarding/interruption, never
a claim that audio was heard. A failed optional receipt must not stop voice shutdown.

Normal personal facts still enter model context. Optional questions do not enter
that factual bundle and no longer append to task-result prose. Required work
clarifications take priority. Explicit review interest loads the existing backend
review tools; a yes to the invitation does not approve any rule or memory merge.
There is no new interpreter or agent runtime.

Questions shows the latest recorded extraction, embedding, memory-review,
organization-review and field-understanding jobs, including IDs, results and errors.
Production usefulness remains unverified: the October 3 read-only attempt reached
the linked Railway project, but private database access needs a configured SSH key.
Synthetic fixtures establish behavior, not the quality of the owner's learned facts.

Local regression coverage includes schema reuse, container navigation/Back/deep
links, subtree versus item-only moves, direct subtasks, nested timelines, review
source adapters, stale answers, cross-device reservations, long Live results and
interruption. Browser fixtures cover 390/892/1440px. Physical Pixel Fold, real Live
playback and connector-side write acceptance remain explicit release checks.
