# Eri response-speed plan

**Status:** first implementation pass, not deployed. The October 2 audit used
`229fe4c`; [Batch A verification](BATCH_A_VALIDATION.md) records subsequent stage
instrumentation, dispatch wakeups, maintenance separation and tests. Production
latency has not yet been measured. The earlier 5–10-second figures are owner
reports, not a stage-by-stage baseline. The audit conclusions below describe the
pre-change baseline; Step 0's real end-to-end sampling remains open.

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

The original direction—measure, remove waits, reduce unnecessary model work—is
sound. The main architectural change is **a compact profile within the existing
durable runner**, rather than a second API runner for mutations.

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

- No priority service tier or new paid infrastructure for this work.
- Try Luna without reasoning for simple work; retain reasoning for harder work.
- Both typed chat and voice benefit.
- Long operations continue in the background; acknowledgement never claims completion.
- Start small and stop adding optimizations when quality, cost and latency are satisfactory.

No increase in average cost per successful request is the rollout constraint, not a
promise about every call. Measure database/worker load too. Do not add another model
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
| Supervisor dispatch | Default sleep is 5 s after the cycle. Cycle duration and outbox eligibility are part of latency. |
| DBOS dispatch | Installed DBOS 2.31.1 has a 1.0 s base polling interval with jitter/backoff logic. It is not a guaranteed fixed delay or proof of production configuration. |
| Context | Personal-memory retrieval may embed the query when stored vectors exist. Query embeddings already have owner-scoped, short-lived reuse. Shared-workspace and bot requests omit personal memory. |
| Model/tools | Luna currently uses Responses with low reasoning. Core tools plus orchestration tools are offered; discovery and final-reply rounds vary by request. HTTP connections already reuse a pool within a runner invocation. |
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

Add bounded timing events keyed by request ID, attempt/revision, channel, profile
and operation class. Store timestamps/metrics without transcript, note or task
content. Keep instrumentation separate from encrypted runner checkpoints and
completion state; concurrent telemetry updates must not overwrite work results.

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
for an explicitly authorized paid evaluation campaign. This audit runs no paid tests.

**Exit:** measured stage breakdown, correctness outcomes and cost per completed
request. Set tail-latency gates from the baseline before rollout; do not redefine
them afterward to fit the result. Re-estimate implementation effort at this point;
the original half-day/one-day/three-day figures are not validated.

## 5. Step 1 — remove avoidable waiting in the current path

### 5.1 Wake dispatch promptly

Use transactional PostgreSQL NOTIFY as a **wake hint**, with the committed outbox as
the source of truth. LISTEN must be established and committed before an initial
rescan; reconnect and rescan after listener loss. Keep periodic fallback scanning
and wake/rescan when dependencies finish or work becomes eligible again. Timed
retries still need a scheduled wake or bounded poll.

Keep the dispatcher independent of slow supervisor maintenance, under the existing
worker lease and dispatch/concurrency rules. Do not run every maintenance scan on
each notification; housekeeping currently depends on cycle counts. Keep wake payloads
content-free, debounce storms, and inspect whether many ineligible old outbox rows
can delay ready work behind the current 1,000-row scan limit.

NOTIFY is delivered on transaction commit; it is not a replacement durable queue.
See PostgreSQL's [NOTIFY](https://www.postgresql.org/docs/16/sql-notify.html) and
[LISTEN startup ordering](https://www.postgresql.org/docs/16/sql-listen.html).

Experiment with a 0.2–0.5 s **agent queue** base interval after verifying the deployed
DBOS setting. Measure database traffic, pool pressure, idle-to-first-job latency and
queue saturation. Preserve concurrency limits; faster polling cannot create capacity.
Do not accelerate every queue or introduce an API-side second worker.

### 5.2 Reduce result-delivery delay without interrupting speech

Wake result handling when work changes, retaining periodic recovery and the current
deduplication, latest-revision, clarification-announcement and session-close checks.
The wake signal triggers a check; it is not permission to announce a stale result.

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

## 6. Step 2 — compact execution in the same durable runner

### 6.1 Model and execution identity

Official documentation lists `reasoning.effort: "none"` for
[`gpt-5.6-luna`](https://developers.openai.com/api/docs/models/gpt-5.6-luna).
Use this exact model, not a guessed alias or another generation. The current
[model catalog](../apps/api/jarvis/agent_models.py) exposes only Luna low reasoning.
Add a separately testable none profile, explicitly sending `"none"` rather than
omitting the setting, and verify project access plus tool-call quality before rollout.

Both channels still accept durable work. The compact profile changes model effort,
context and offered tools **inside** the existing runner. Keep:

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

1. Instrument the current path and collect baseline evidence.
2. Improve dispatch wakeups and measured delivery/tool-discovery delays.
3. Compare the none profile within the durable runner.
4. Add compact scope/continuation and composites only if still needed.
5. Run representative correctness, cost and device trials before default-on.

Required regression coverage:

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
voice/device checks remain separate from deterministic CI. This plan does not claim
that any new checks, latency targets or cost comparisons have passed.

This work fits the latency/voice reliability checks in the
[web v1 completion PRD](ERIDANI_WEB_V1_PRD.md). It does not authorize a separate
deployment or replace that plan's correctness and source-sync requirements.
