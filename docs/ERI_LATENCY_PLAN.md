# Eri response-speed plan

**Status — October 4, 2026:** SPEED1 measurement infrastructure and SPEED2's
selectable Luna no-reasoning profile are implemented and verified locally against
main `de14a22`, including Claude's Organization follow-ups. The first infrastructure
pass was already deployed: [Batch A verification](BATCH_A_VALIDATION.md) records
dispatch wakeups, independent maintenance and outbox pagination. These changes are included in the speed release prepared October 4.
Local verification predates deployment; the PR, CI and deployment readback establish
shipping status. Production/device acceptance below remains separate.
See [SPEED1/2 validation](SPEED_1_2_VALIDATION.md) for tests, the bounded paid smoke
comparison, measurement limits and remaining acceptance checks.

**Accepted local default:** interactive `gpt-5.6-luna` requests use
`reasoning.effort="low"` with `service_tier="fast"`. The owner approved this after
the Standard/Fast and Fast/none tests. Memory/note extraction and dream maintenance
stay explicitly Standard. Premium reservations and returned-tier accounting are
implemented; missing returned-tier metadata is labelled an estimate. Old queued
jobs without a saved tier retain Standard. API/worker can override with
`JARVIS_AGENT_SERVICE_TIER=default`. This is code, not a completed production rollout.

The new local `luna-none` profile explicitly sends
`reasoning.effort="none"` and is selectable in Backend Settings after deployment.
Both use the same model, tools and durable runner. Existing accepted jobs keep
their captured profile. The owner's saved production preference was not queried
or changed. GPT-Live remains the separate voice model.

Production latency still lacks a controlled end-to-end baseline. A read-only sample
of the latest Web and Worker deployment logs (past day, capped at 2,000 filtered
lines per service) returned **zero latency events** on October 4. This does not prove
there were no requests, that logging is broken, or that responses are fast. Verify
collection with a known request before benchmarking. Local evidence is in
`artifacts/speed-2026-10-04/summary.json` (not a checked-in benchmark).
The earlier 5–10-second figures are owner reports, not measured stage timings.

### Already implemented versus still to do

- **Implemented:** PostgreSQL commit notifications wake the dispatcher; fallback
  scans and pagination retain durable recovery. Slow sync/dream maintenance runs
  independently of dispatch. All three Railway services have sleeping disabled.
- **Already reused:** HTTP connections within a runner invocation, owner-scoped
  short-lived query embeddings, and 19 core tools. Basic task capture and simple
  navigation do not inherently require a tool-discovery round.
- **Implemented locally:** content-free events join committed acceptance/outcome,
  runner invocations, provider attempts, token usage and first visible chat result.
  Same-tab typed requests report a visible card/reply approximation; actual target
  content and audible voice completion remain unmeasured. Cost reporting includes
  unsuccessful attempts and flags incomplete evidence.
- **Implemented locally:** explicit Luna low/none options, eval profile propagation,
  and an interleaved four-scenario comparison per profile. Both passed; this sample
  does not establish general quality parity or production performance.
- **Pending:** a production/device baseline, measured improvements to result/UI
  delivery and any reduced-context/tool-round execution profile (SPEED3/4).

Claude's container behavior, Atlas/Blueprint, templates and review cadence are part
of the current regression surface. Their UI work does not establish lower backend
inference latency. Include these workflows when testing speed changes.

**Principle:** make the existing durable path faster before introducing another
execution path. Speed must preserve accepted requests, corrections, permissions,
source sync, receipts and Revert.

## 1. Audit conclusions

| Earlier proposal or assumption | Correction |
| --- | --- |
| Every voice/chat request enters the queue | The current web chat posts to `/work`, and delegated Live requests enqueue work. Ordinary Live conversation stays in Live. Legacy `/chat` still runs inline; it is not the current web send path. |
| Worker wait is at most 5 seconds; other waits add up to 6–13 seconds | Five seconds is the sleep **after** a supervisor cycle. Scans/deliveries, queue occupancy and retries add time. Several voice timers overlap backend work. The total and estimated savings cannot be inferred by summing timer constants. |
| Navigation always needs a tool-loading round | Core already includes `ui_show`, which can open Calendar. Specific view/filter/custom-record operations need other tools. Measure the actual selected tool path. |
| Run simple writes inside the API/Live controller; receipts make them durable | A receipt protects one chosen command. The durable runner also checkpoints the exact tool plan, orders conflicting work, checks cancellation/revisions and records completion. Retain that runner. Live controller shutdown currently cancels its delegation tasks, making inline mutation execution particularly risky. |
| Retry using request ID plus tool index | Safe only when the original planned call and arguments are persisted. Re-running the model can change ordering/arguments, causing conflicts or duplicate effects. |
| Shorten settle whenever a transcript exists; skip silence for the newest result | Partial text does not prove a complete utterance. The silence gate tracks both speakers. Preserve transcript completeness and conversational timing; reduce only measured excess waiting. |
| Hand the original request to a new background job after a fast attempt | Partial work could be repeated. Continue the same work root with saved receipts and remaining work; never race two executors on a timeout. |
| No reasoning guarantees equal/lower cost | It is a testable hypothesis. Extra attempts, fallback, cache behavior and voice duration can offset savings. Compare total cost per successfully completed request. |

The October 2 audit used `229fe4c`; the table above explains why the original
proposal changed. Dispatch waiting described there is historical; the current dispatcher does not
impose that five-second wait on each request. The direction remains: measure, remove demonstrated
waits, reduce unnecessary model work. Start with a reasoning-only comparison in
the existing durable runner. A compact execution profile is conditional later work,
not a prerequisite or a second API runner for mutations.

## 2. Goals and retained preferences

| Request | Initial warm-session median target | Completion milestone |
| --- | --- | --- |
| Open Calendar / show a uniquely identified task | ≤ 1.5 s | Requested content visibly rendered on the requesting device |
| Today's schedule | ≤ 2.5 s | First useful factual answer shown/spoken, grounded in the returned agenda |
| Create one unambiguous local task | ≤ 2.5 s | Saved receipt plus visible/spoken confirmation of that saved task |

These are **targets, not predicted results or release claims**. For voice, measure
from the end of the relevant user utterance; for chat, from Send. Report median,
p90, sample count, failures and fallback rate separately for each channel/scenario.
Warm sessions are already connected and authenticated. Report cold starts, reconnects,
idle-to-active transitions and constrained mobile networks separately. Measure UI
navigation independently from subsequent spoken confirmation.

Retain the preferences recorded in the original plan:

- No new paid infrastructure. The original plan excluded Priority/Fast; the owner
  subsequently authorized a bounded Standard/Fast API comparison on October 4.
  It is recorded in [validation](SPEED_1_2_VALIDATION.md#follow-up-standard-versus-fast-retaining-low-reasoning).
  The owner subsequently approved Luna low + Fast as the interactive default.
  Its local implementation still needs the normal PR/CI rollout and device acceptance.
- Try Luna without reasoning for simple work; retain reasoning for harder work.
- Both typed chat and voice benefit.
- Long operations continue in the background; acknowledgement never claims completion.
- Start small and stop adding optimizations when quality, cost and latency are satisfactory.

No increase in average cost per successful request remains the rollout constraint
for reasoning/context optimizations. The authorized Fast-mode experiment explicitly
compares a premium-priced tier; the owner accepted that premium for interactive
Luna work. Scheduled learning remains Standard. Measure database/worker load too. Do not add another model
solely to classify requests into fast versus slow.

## 3. Current path and evidence

~~~text
Web Send → /work ───────────────────────┐
                                       ├→ durable AgentWork + Job + Outbox commit
Live delegation → settle → claim speech ┘
  → outbox dispatch → DBOS agent queue → work_runner
  → scoped context → model → validated tools → model reply → durable finish
  → web work-change refresh / Live result delivery → visible or audible result
~~~

| Stage | What the code establishes; what still needs measuring |
| --- | --- |
| Speech claim | Live settle has a 0.6 s minimum, then only the **remaining** 0.8 s quiet interval, bounded around 4 s. These are not unconditional sequential delays. No definitive final-turn marker is consumed by the current delta handler. |
| Outbox dispatch | Commit-delivered NOTIFY wakes the dispatcher, separately from maintenance. The default 5 s interval is a fallback rescan, not a mandatory wait for every request. Measure acceptance-to-dispatch and dependency eligibility. |
| DBOS dispatch | Installed DBOS 2.31.1 has a 1.0 s base polling interval with jitter/backoff logic. It is not a guaranteed fixed delay or proof of production configuration. |
| Context | Personal-memory retrieval may embed the query when stored vectors exist. Query embeddings already have owner-scoped, short-lived reuse. Shared-workspace and bot requests omit personal memory. |
| Model/tools | Luna uses Responses with explicit low reasoning. Nineteen core tools plus orchestration tools are offered. Task create/read/update/complete, search and `ui_show` are already core; generic-record/schema/template/review operations can need discovery. Each model round, provider retry and tool round must be measured. HTTP connections already reuse a pool within a runner invocation. |
| Browser control | Browser sync runs about every 700 ms while active / 3 s while idle. Durable acknowledgement polling adds up to another 250 ms per check. A handler acknowledgement is not necessarily a painted screen. |
| Web result delivery | Existing SSE announces database changes, not model text tokens. The server feed polls about every 500 ms; Activity coalesces updates for about 150 ms before refetching. |
| Voice result delivery | The controller checks work around once a second, with a 2.5 s quiet gate based on user **and assistant** transcript activity. Quiet time can overlap model execution. Append acknowledgement is not completed speech. |

Code references: [Live controller](../apps/api/jarvis/live_voice.py),
[speech claim](../apps/api/jarvis/work_intake.py),
[worker](../apps/api/jarvis/worker.py),
[runner](../apps/api/jarvis/work_runner.py),
[tool catalog](../apps/api/jarvis/tool_catalog.py),
[memory retrieval](../apps/api/jarvis/memory_service.py),
[embedding reuse](../apps/api/jarvis/memory_learning.py),
[device bridge](../apps/api/jarvis/device_bridge.py),
[web send/UI sync](../apps/web/src/App.tsx),
[API event feed](../apps/api/jarvis/api.py) and
[Activity refresh](../apps/web/src/Activity.tsx).

## 4. Step 0 — establish a baseline

Extend the existing [timing events](../apps/api/jarvis/latency.py) and
[summary script](../scripts/summarize_latency.py); do not introduce a second tracing
system. Existing fields cover stages, revision/round/tool index and durations,
and now include profile/channel/token/retry breakdown. The updated script joins
requests by revision and durable outcome, while keeping raw stage counts.
Its local implementation is complete; the production/device baseline below is
still an acceptance requirement, not an achieved performance claim.

Join bounded events by request/work root, attempt, revision and voice session where
applicable. Add allowlisted channel, model/profile/reasoning, operation/tool name,
retry count and available token/cache/usage identifiers. Derive successful requests
from durable terminal state and committed receipts, never from a precommit log.
The existing runner `queue_ms` includes time since job creation; it is not an
isolated DBOS queue-delay measurement.

Store timestamps/metrics without transcript, note or task content. Keep instrumentation
separate from encrypted runner checkpoints and completion state; concurrent telemetry
updates must not overwrite work results. Reuse existing usage/Langfuse records where
available rather than send duplicate content or count usage twice.

Measure:

- User speech-end marker / chat Send; delegation receipt; transcript claim;
  durable acceptance; dispatch; runner start; dependency/capacity wait.
- Context substeps, cache hit/miss, each model round, each tool, receipt commit,
  final work state and connector confirmation where relevant.
- UI action received, acknowledgement and first visible render.
- Work event delivered, reply rendered, Live append sent/acknowledged and first
  useful spoken result. If precise audio timing is unavailable, mark it unmeasured
  and use annotated device trials; do not substitute append acknowledgement.

Use monotonic clocks for durations inside one process; correlate cross-process
events with UTC timestamps and IDs, accounting for clock skew. Calculate the actual
critical path rather than summing overlapping spans. Record timeout/failure/censored
runs instead of calculating percentiles only from successes.

Use deterministic models to verify instrumentation, dependencies and timer behavior.
Replay Calendar navigation, named-task navigation, agenda read, local task creation
and source-backed editing through the **actual /work and synthetic Live paths**.
Fake-model tests do not predict inference latency, speech behavior or answer quality.

Start with ten real trials per scenario/channel as a smoke baseline. Before accepting
performance claims, gather at least thirty successful trials per simple scenario/
channel and report all attempted trials; this is an initial engineering sample,
not strong statistical proof. Interleave before/after comparisons where practical,
include warm/cold caches and competing jobs, and retain the existing $10 maximum
for an explicitly authorized paid evaluation campaign. The initial eight-trial comparison is recorded in the validation report; it is not this larger baseline.

**Exit:** measured stage breakdown, correctness outcomes and cost per completed
request. Set tail-latency gates from the baseline before rollout; do not redefine
them afterward to fit the result. Re-estimate implementation effort at this point;
the original half-day/one-day/three-day figures are not validated.

## 5. Step 1 — remove avoidable waiting in the current path

### 5.1 Preserve the shipped dispatch improvements

Transactional PostgreSQL NOTIFY already acts as a **wake hint**, with the committed
outbox as the source of truth. LISTEN is established and committed before an initial
rescan. Connection loss exits the dispatcher so process recovery can reacquire the
lease and rescan. Fallback scanning handles missed hints, timed retries and changed
eligibility. Keyset pagination scans beyond the former 1,000-row boundary.

Maintenance runs separately under the worker's existing supervision. Preserve this
design, concurrency limits, content-free wake payloads and restart tests; it does not
need to be implemented again. NOTIFY is delivered on transaction commit and is not
a replacement durable queue. See PostgreSQL's
[NOTIFY](https://www.postgresql.org/docs/16/sql-notify.html) and
[LISTEN startup ordering](https://www.postgresql.org/docs/16/sql-listen.html).

**Remaining conditional experiment:** if dispatch-to-runner measurements identify
DBOS polling as material, verify the deployed queue setting and trial a 0.2–0.5 s
base interval for the **agent queue only**. Installed DBOS defaults to 1 s and the
application does not explicitly override that constructor setting; this is not a
production database readback. Measure database traffic, pool pressure,
idle-to-first-job latency and saturation. Preserve concurrency limits; faster
polling cannot create capacity. Do not accelerate every queue or introduce an
API-side second worker.

### 5.2 Reduce result-delivery delay without interrupting speech

Wake result handling when work changes, retaining periodic recovery and the current
deduplication, latest-revision, clarification-announcement and session-close checks.
The wake signal triggers a check; it is not permission to announce a stale result.
Prefer a process-level listener/fan-out over one database connection per voice
session. A restarted API, dropped notification or reconnect must still recover
from committed work state through the existing polling fallback.

If browser bridge transport dominates navigation latency, use the existing
authenticated event feed to wake UI-action retrieval, with account/device targeting,
expiry and acknowledgements intact. Merely sending another event through the
500 ms polling feed does not remove that feed's delay. Benchmark the complete path
before deciding whether the server feed also needs a commit notification.

Keep conversational turn protection. A shorter gate needs evidence that the user is
not continuing and Eri is not speaking. Track both speakers and verify actual phone
behavior; “this is the newest request” is insufficient. Keep speech settling
unchanged until late-transcript/paused-speech tests justify a reliable replacement.
A complete-turn signal may be evaluated only after protocol support is verified.

Live instructions already allow brief acknowledgements. Tune only if trials show a
problem; don't add a compulsory spoken filler or another model call to every request.
An acknowledgement describes intended/in-progress work, never a successful save.

### 5.3 Remove unnecessary tool discovery and duplicate work

Measure which navigation actually loads tools. Keep using core `ui_show` for simple
page navigation; trial `ui_workspace`, `ui_records` or `ui_calendar` in an appropriate
initial tool set only when their saved round outweighs extra prompt cost.

Preserve device/workspace checks, pending autosave and unsaved-draft protection,
action expiry and explicit UI acknowledgement. Navigation creates no mutation
receipt/card. Read/search tools may record search feedback/context, so an eventual
“read-only” shortcut must inventory those effects too.

Measure memory and connection setup before changing them. Preserve stable prompt/tool
ordering. Cross-request connection reuse is optional if connection establishment is
material; intra-request pooling already exists.

**Exit:** compare against Step 0. Stop here if the targets and quality gates are met.

## 6. Step 2 — compare reasoning first; compact execution only if needed

### 6.1 Model and execution identity

Official documentation lists `reasoning.effort: "none"` for
[`gpt-5.6-luna`](https://developers.openai.com/api/docs/models/gpt-5.6-luna).
Use this exact model, not a guessed alias or another generation. The local
[model catalog](../apps/api/jarvis/agent_models.py) now exposes `luna` (low) and
`luna-none` (**Luna · no reasoning**). Both explicitly send their effort setting;
`luna` remains the default. Verify project access plus
tool-call quality before rollout. Existing job profiles and usage attribution must
retain their meaning; selecting a new default must not silently change in-flight work.

**First comparison:** same model ID, prompt, context, tools, output allowance and
runner; change only reasoning effort. Expose both options in backend Settings and
the eval runner, with the effective mode visible in timing/cost reports. Do not
implicitly switch dream, routing or other background features as a side effect.
An output-token maximum is a ceiling, not an instruction to generate that many
tokens; lowering it alone is not evidence of reduced latency.

Only after this comparison should prompt/tool compaction or automatic escalation
be evaluated. Both channels still accept durable work. The compact profile changes
model effort, context and offered tools **inside** the existing runner. Keep:

- Principal/account/device/scoped-bot checks and permission rechecks.
- Exact planned calls checkpointed before effects, stable indices and saved receipts.
- `work_followup`, resource reservations and related-request ordering.
- Request revisions, explicit cancellation, continuation and recovery.
- Existing action journal, chat placement, Activity, Edit and Revert semantics.

Do not execute model work in the Live event receive loop. Closing voice must not
cancel accepted work. Keep typed chat on `/work` with persisted replies; token
streaming would need a separate protocol and is not a prerequisite.

### 6.2 Small tool surface, existing contracts

Begin by reusing existing tools instead of creating ten competing wrappers.
Keep orchestration/clarification controls available in the compact profile.
Candidate simple operations: navigation, uniquely resolved record reads, an agenda
read, one local task capture/update/completion and a quick note.

Consider composites only where traces demonstrate avoidable rounds:

| Candidate | Required behavior |
| --- | --- |
| Resolve and open a record | Resolve within the current scope, return ambiguity without acting, then use the normal acknowledged UI path. Never choose the first fuzzy match blindly. |
| Today's agenda | Extract a shared backend query used by tools and the Today UI. Include planned work as well as due/overdue tasks, events and reminders; deduplicate, preserve timezone/all-day semantics and report truncation or stale integration data. Today currently aggregates client-side; there is no existing single shared agenda contract to wrap. |
| Quick task capture | Reuse canonical validation and organization rules. Preserve planned date versus deadline versus reminder distinctions, DST checks and current schema; a speed shortcut must not invent a home or turn a date-only task into an alarm. |

For an update, resolve the actual target, current revision and source linkage before
execution. A `task_update` can trigger Linear sync even if the user never says Linear.
Source-backed writes retain the durable connector path and truthful local/remote
status required by [web v1 PRD Batch B2](ERIDANI_WEB_V1_PRD.md#batch-b--make-the-existing-workspace-coherent).

### 6.3 Scope expansion and partial completion

Use a compact first pass, with a provisional allowance of two model rounds and three
tool calls, **not a correctness cutoff**. Define the broader-runner handoff as an
explicit runner transition, not a tool that blindly enqueues the original request.

The model can request broader capabilities/reasoning; the server enforces scope
before unsupported effects. Missing tools, complex planning, linked external writes,
multi-step changes and uncertain matches must expand scope or ask for clarification.
The first rollout can conservatively exclude such mutations. No extra classifier
model is needed.

Persist a one-way transition to the broader low-reasoning profile under the same
work root, preserving original request, corrections, exact calls, receipts,
dependencies, remaining clauses and the cumulative budget. If some actions already
committed, continue only remaining work. A timeout never starts a competing attempt;
stop or recover the current attempt using the existing lock/checkpoint mechanism.
Do not oscillate between profiles or reset allowances to evade limits.

Keep genuine `work_needs_input` questions and explicit `work_answer` linkage.
PR #15 fixed indiscriminate capture of unrelated replies; it did not make durable
clarification intrinsically wrong. Incomplete requests should normally be clarified
by Live before delegation. Once accepted, questions retain their identity, resolved
state and reload/reconnect behavior. Optional “anything else?” offers must not create
pending work or swallow the next request.

### 6.4 Context and fewer model rounds

Keep a shared policy core: authorization, source-write semantics, date handling,
user-defined organization, question ownership, recent work/receipts and untrusted
content treatment. Remove irrelevant detail only with measured quality parity.

The backend does not automatically inherit Live's full injected memory simply
because both are part of one conversation. Any reusable context bundle must be
explicitly passed, scoped and freshness-checked. Cache query vectors or bounded
context with owner/account and revision/deletion invalidation; never revive deleted
facts or inject personal memory into shared/bot requests. Skip optional memory only
when the operation needs none; requests requiring memory must still retrieve it.

Avoid a final model round solely to rephrase a verified tool result when the runner
can prove the entire bounded request is complete. A receipt-derived confirmation
can go to chat and to Live. Do not finish after the first tool in a multi-clause
request or turn a partial/error result into success.

For Live, preserve the client delegation ID, the 500-token append limit and concise
verified facts. Append acknowledgement measures context delivery, not audible
completion. Client/Responses delegation mode changes require a new Live session;
keep client mode here. See the official
[Live delegation guide](https://developers.openai.com/api/docs/guides/live-delegation).

### 6.5 Accounting and rollout

Keep per-round budget headroom, provider-response usage IDs, retry accounting and
uncertain-spend holds. Reserve/close alone does not record usage. Retain the
`assistant` feature with profile/phase metadata, or deliberately register a new
feature everywhere; don't silently invent an unrecognized `fast_lane` category.

Compare aggregate cost across compact attempts **and** continuations, memory/search
calls and attributable voice use. Track cost per successful completion plus failures,
not merely cheaper individual model responses. Whole-session Live cost may need
separate reporting rather than false per-request precision.

Ship behind an owner/per-user flag with bounded concurrency and an immediate rollback.
First verify the none profile with existing tools, then compact the prompt/tools;
change one variable at a time. Rollback leaves already-accepted jobs recoverable.
Default-on requires quality parity, useful measured latency improvement and no
average cost increase in the representative request mix.

## 7. Conditional later work

| Idea | Evidence required before adding it |
| --- | --- |
| API shortcut for narrowly scoped read/navigation | Significant residual queue overhead after Step 1, plus a proven scope/effect boundary. Mutation execution remains durable. |
| SSE/push delivery of UI actions | Bridge transport dominates visible latency. Account for the existing SSE server poll; retain expiry, acknowledgement and reconnect recovery. |
| Persistent Responses connection / Live Responses delegation | Backend connection overhead dominates and a measured prototype beats client delegation without weakening task ownership or increasing cost. |
| Parallel/cached optional retrieval | Measured context cost, correct scope/freshness, cancellation and accounting; late context must not silently change an already-executed action. |
| Agenda cache | Shared agenda query is materially slow; invalidate on task/calendar changes and show freshness. |
| Deterministic navigation shortcuts | Frequent unambiguous phrases justify a small maintained path with the same navigation guards and acknowledgement. |

Do not add speculative mutations, automatic fuzzy-match writes, a new intake model,
or a second background architecture to meet an unmeasured latency estimate.

## 8. Verification and implementation order

Use existing CI gates and documentation updates before any later rollout.
SPEED1 instrumentation, SPEED2 profile selection and the owner-approved Luna
low + Fast interactive default are implemented locally; production/device
acceptance and a representative model comparison remain open.
SPEED3/4 have not started. No commit, push or deployment was performed in this pass.

### SPEED1 — finish measurement and obtain a useful baseline

**Local implementation verified.** Durable commit markers, per-invocation model
attempts/usage, bounded authenticated browser timing and joined reports are in
place. Browser acceptance proves collection for typed card and plain-text results.
Eval workers save `latency.log` and `latency-summary.json` beside existing evidence.
Langfuse remains an opt-in synthetic artifact export, not live production tracing.
The broader baseline described here is still pending.

Extend the existing spans/summarizer with the joined measurements in Step 0.
Instrument browser receipt/render milestones; keep actual audible timing as an
explicit physical-device check unless the voice API exposes a reliable signal.
First prove a known request produces a complete trace, then baseline the current
low-reasoning profile. Record failures, retries and competing-job load.

Deliverable: a reproducible report of median/tail latency, model/tool round counts,
stage contributions, outcome and cost. Unit/regression tests use deterministic
models and fake clocks; a timing report is not a new flaky CI wall-clock gate.

### SPEED2 — add and compare Luna without reasoning

**Local implementation verified.** Both modes passed task creation, ambiguous
editing, deadline changes and note creation in the four-scenario-per-profile smoke.
No reasoning reported zero reasoning tokens. Median runner-to-durable-outcome time
was 5.43 s (low) versus 4.99 s (none); cache and order effects remain.
Keep low as default. This is API compatibility and narrow correctness evidence,
not representative parity or a production latency result.

Add the selectable profile from section 6.1. Keep low reasoning as default while
testing; interleave matched low/none scenarios against isolated equivalent state
to avoid cache/order effects and duplicate production writes. Use the same known
answers/state assertions, including difficult cases, rather than asking a grader
whether a faster-sounding response seems successful. Reuse the existing eval
scaffolding; cover the real durable runner as well as adapter behavior.

Include simple navigation, a named-record read, today's agenda, local task capture/
editing, a quick note, dependent corrections, ambiguity and external-source edits.
Add generic custom-record creation, template instantiation and review updates so
Claude's new features are covered. Complex cases are quality checks, not subject
to the simple-action latency targets.

Report reasoning/output/input/cache tokens where exposed, tool/model rounds,
retries, failures and total cost per successful request. One short answer cannot
establish a quality or price advantage. Any paid campaign retains the $10 hard
ceiling, should aim well below it, and needs explicit scope; the completed smoke used a smaller allowance.
The larger sample requirement in Step 0 is for performance claims, not an
instruction to launch hundreds of paid trials before a small comparison works.

**Decision:** promote none only for the tested scope if it provides a useful
latency improvement, no critical correctness regression and no average cost
increase. Otherwise retain low reasoning and address the measured waits. Do not
assume automatic routing/fallback exists just because two profiles are selectable.

### SPEED3 — shorten demonstrated delivery waits

Use SPEED1 evidence to choose individual changes: Live work-change wakeups,
UI-action retrieval wakeups, or the agent-only DBOS interval experiment in section
5.1. Introduce and compare them one at a time. Reuse PostgreSQL and the existing
event feed; no Redis or new always-on service is planned.

Keep the existing speech settling and 2.5 s conversational quiet protection at
first. Phone testing, including interrupted/overlapping speech and late transcript
suffixes, must precede any reduction. Reconnect, revision, cancellation and
clarification ownership remain release gates.

### SPEED4 — reduce model work only where traces justify it

The [round-reduction implementation plan](ERI_ROUND_REDUCTION_PLAN.md) is the
concrete October 4 follow-up, with a checked-in synthetic baseline and separate PRs.

If targets remain unmet, prioritize the measured cause:

- Discovery rounds: preload a small relevant tool set, preserving stable ordering
  and measuring the extra prompt cost. Simple `ui_show` and basic task create/edit/complete already
  have core tools, so do not expect a discovery saving there.
- Context time/size: reuse correctly scoped fresh context and shorten irrelevant
  history while retaining policy, referenced records, current schema, recent
  receipts and correction/clarification state.
- Final-reply rounds: derive a concise acknowledgement from verified results only
  when the entire bounded request is provably complete.
- Complex work: consider the same-runner compact-to-low continuation in section
  6.3 only if the measured benefit warrants its extra recovery/quality surface.

Structure edits still require preview/apply. Templates can create many records;
neither templates nor schema edits become a blind quick-write path. Every
continuation retains the work root, receipts, budget and remaining request.

**Stop condition:** once the representative simple actions meet useful measured
targets and correctness/cost gates, leave conditional ideas unimplemented.
Before default-on, run the relevant deterministic regression suites and the
focused live/device comparison; provide an immediate profile/config rollback
that preserves already-accepted jobs.

### Required regression coverage

- Three-request ordering: “add Call Alex”, “add Buy milk”, “make that call tomorrow”;
  independent parallel work proceeds and the dependent correction applies once.
- Crash after tool-plan checkpoint, after commit, and before announcement; restart/
  replay produces no duplicate records, orphan requests or missing cards.
- Scope expansion after a partial save; lost provider response; exhausted budget;
  concurrent retries; no second executor or unaccounted fallback spend.
- Goodbye/voice close during execution; accepted work finishes. No stale announcements
  after cancellation/revision, and no unrelated reply consumed as a clarification.
- Late transcript suffix, self-correction, deliberate pause, duplicate delegation,
  speech overlap and disconnect during claim.
- Ambiguous/duplicate/spoken-number names; dates/DST; custom fields and routing;
  restricted/viewer/bot requests; revoked access and workspace switches.
- Task edits that trigger external sync; pending versus confirmed results, conflicts,
  disconnected accounts and local-only fields.
- UI refusal for unsaved drafts, pending autosave, hidden/offline device, expiry and
  actual content visibility; navigation remains absent from action cards.
- Agenda completeness/deduplication/freshness and memory correction/deletion isolation.

Use fake clocks and deterministic assertions for ordering, wake/recovery behavior
and configured timer bounds in CI. Do not enforce “total minus fake-model time
under 600 ms” as a shared-runner wall-clock test: spans overlap and existing bridge/
feed timers already make that assumption unsound. Keep a controlled performance
benchmark separate from ordinary correctness CI.

Re-run the relevant existing
[agent-work](../tests/test_agent_work.py),
[work-continuation](../tests/test_work_continuation.py),
[Live](../tests/test_live_voice.py) and
[Responses](../tests/test_responses_agent.py) tests, then bind new cases into the
[app eval findings/coverage process](../evals/app/FINDINGS.md). Paid evals and physical
voice/device checks remain separate from deterministic CI. The linked validation report records the checks and small comparison that passed.
The wider device checks, latency targets and representative quality/cost gates
remain pending.

The experiment order also follows OpenAI's
[latency optimization guidance](https://developers.openai.com/api/docs/guides/latency-optimization):
reduce unnecessary requests and generation, and preserve reusable prompt prefixes.
Those general principles do not establish a speedup for Eridani without measurement.

This work fits the latency/voice reliability checks in the
[web v1 completion PRD](ERIDANI_WEB_V1_PRD.md). It does not authorize a separate
deployment or replace that plan's correctness and source-sync requirements.
