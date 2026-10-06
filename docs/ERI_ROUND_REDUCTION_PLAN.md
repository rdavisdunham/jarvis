# Reducing Eri model and tool rounds

Updated October 6, 2026. The first two stages are implemented and independently
selectable. Discovery is the first-stage default; sufficient-read reuse is staged
behind its policy setting until the phone check. See the
[implementation, measurements and remaining acceptance](ERI_ROUND_REDUCTION_VALIDATION.md).
The optional completion shortcut remains deferred. A fourth, opt-in stage (`lean-v1`, below)
shrinks the per-round prompt and makes its prefix cacheable across requests.

Start by removing unnecessary discovery, then avoid repeated reads when a fresh
result already contains everything needed. Keep the durable runner, human
clarification, revision checks and verified receipts. Do not introduce a request
interpretation model, a second execution queue, or automatic reasoning selection.

## Evidence and limits

The [checked-in round baseline](evals/2026-10-04-speed/round-baseline.json) contains
24 synthetic trials across four scenarios, three trials per scenario per tier.
The [validation report](SPEED_1_2_VALIDATION.md) records the latency and cost
comparison. These observations are backend measurements; they exclude real
phone speech, rendering and production dispatch.

- Simple task capture took two or three model rounds. Some runs called
  `tools_load` before `task_create` even though task creation is in CORE.
- Note capture took three rounds: discovery, creation, then the response. Here
  discovery is expected because `note_create` is not initially exposed.
- A deadline edit took three to five rounds. The sequences included
  resolve → update and resolve → get → update, followed by the response.
- Ambiguous edits correctly stopped for clarification. A short execution is not
  successful if it changes the wrong record or leaves a clarification orphaned.

The logs omit loader group arguments and some built-in work-control calls.
Therefore, redundant discovery and duplicate reads are hypotheses to inspect,
not blanket declarations that those calls can be removed. The baseline preserves
all cases, including necessary clarification and unfavorable timings.

Current relevant code:

- `apps/api/jarvis/tool_catalog.py`: CORE, tool descriptions, group discovery.
- `apps/api/jarvis/agent_instructions.py`: already says common tools are loaded;
  the loader description separately says to load groups before using their tools.
- `apps/api/jarvis/task_tools.py` and `tools.py`: resolver results include current
  IDs/revisions and compact fields; notes are only a preview.
- `apps/api/jarvis/work_runner.py`: checkpoints, tool results, built-in work
  controls, current-revision edits, action receipts and final response generation.
- `scripts/app_eval`: isolated state oracles, provider usage and paired campaigns.

## First PR Remove redundant discovery

1. Record content-free per-round tool names, requested capability groups,
   newly added versus already loaded tool counts and built-in work-control calls.
   Include request revision and invocation ID. Do not log arguments, titles,
   note bodies, personal context or provider secrets.
2. Make loader wording explicitly apply to tools absent from the current list.
   Keep a short, consistent rule: call an exposed tool directly; load a group
   only to obtain missing capabilities. Avoid repeating all capability
   instructions in the system prompt.
3. Return `newly_loaded` and `already_available` information from the loader.
   Repeated discovery remains safe and idempotent; it must never trigger an
   action or punish a legitimate recovery after a resumed checkpoint.
4. Trial `note_create` in the small initial set. This removes a known note-capture
   prerequisite without a classifier. Maintain stable tool order and measure the
   added input tokens/cache effects on task-only and complex requests.
5. Keep schema previews, source sync, organization changes, templates and less
   common tools deferred. Do not expose every tool to every turn.

Expected shape: a straightforward task or note creation usually needs a call
round and a result/answer round. This is a benchmark target, not a hard two-round
cap. Use the full existing loop when validation, references or errors need it.

Acceptance: already available task tools work without an obligatory loader;
missing groups still load; bot permissions still filter the catalog; unavailable
tools remain rejected; a repeated loader call changes no records. Ordinary
task/note capture should lose a discovery round where it was actually redundant.

## Second PR Reuse sufficient fresh reads

1. Define the lookup contract explicitly: exact ID, current revision, source
   ownership and the fields needed for a particular sparse mutation. Name
   truncated previews as previews. Extend compact results only where their
   omission actually forces a detail read.
2. Clarify instructions that a fresh resolver/list/detail result with current
   revision satisfies the read-first requirement for a sparse change that does
   not depend on omitted content. For example, changing only a date need not
   reread full notes when omitted fields are preserved by the server.
3. Keep `task_get`/`record_get` for note edits, relationship replacements,
   missing timing/source information, ambiguous matches or any operation needing
   fields absent from the lookup. A title match alone never authorizes a write.
4. Scope reused evidence to the current account, workspace, record, revision and
   active invocation. Never treat saved browser context, a previous conversation
   or a restored checkpoint as a fresh authorization/read.
5. Retain server-side expected-revision checks. On conflict, refresh and replan;
   never retry the same patch against a new revision blindly.
6. Make saved mutation receipts sufficient to acknowledge the change. Do not
   reread merely to check an atomic local write that already returned its committed
   result. External writes still require their existing sync-status semantics.

Expected shape: an unambiguous sparse edit generally takes resolve → mutate →
answer, with extra reads retained when they are necessary. For tasks with the
same requested change, use the existing exact selection/batch tools rather than
a model round per task; do not invent a second bulk-edit implementation.

Acceptance: omitted fields stay unchanged, explicit null still clears,
same-name tasks still clarify, source-linked writes never claim remote success
early, stale revisions refresh, and edits preserve links, notification policies
and notes.

## Third PR Conditional completion improvements

Keep the ordinary final model response in the first two PRs. A tool success is
not proof that a natural-language request containing several clauses is finished.

Only after the first two measurements, consider a deterministic acknowledgement
for a server-defined, bounded action whose entire scope is known. It must require
all requested effects and device acknowledgements to be confirmed, no pending
clarification/dependency, no unhandled error and no newer request revision.
Record completion through the same checkpoint/receipt path. If scope is uncertain,
continue the existing model loop. Do not add a model just to decide whether to skip
another model call.

Exclude free-form multi-part speech, scheduling/availability reasoning, semantic
search explanations, source writes awaiting confirmation, schema preview/apply,
templates and mixed navigation/mutation requests from this experiment initially.
If scope cannot be established without new interpretation complexity, defer this
PR entirely.

Independent calls can already be returned in one model response. Review prompt
examples for unnecessary sequencing, but parallel tool execution is a separate
change with its own locking and recovery risks; it does not automatically reduce
model rounds. Leave a new concurrency mechanism out of these PRs.

## Verification and rollout

Use Luna low plus Fast for both comparison arms. Standard and none results are
historical context, not controls for this experiment. Reuse the existing isolated
corpus and state oracles; never seed a speed benchmark into the owner's account.

Start with deterministic regressions for each changed contract, then interleave
matched baseline/candidate synthetic trials with equivalent initial state.
Use three repetitions per arm for the targeted smoke and expand only if the
direction is promising. Bind any missing scenario to a real oracle before claiming
coverage. The paid campaign keeps the existing $10 hard cap, targets well below $2
for a focused comparison, and records failures and uncertain spend within that cap.

Required scenarios:

- Plain and dated task capture; note creation and evidence-linked note tasks.
- Sparse date/time edits, explicit clearing, relationship preservation and a
  note edit that truly needs full content.
- Duplicate titles; unresolved partial answers; clarification continuation with
  one surviving action history.
- “Add Call Alex”, “add Buy milk”, “make that call tomorrow”, including a restart,
  correction, cancellation or concurrent edit between lookup and mutation.
- Unavailable groups, repeated discovery and restricted bot/shared-workspace tools.
- Generic custom records, task homes, structure preview/apply and templates.
- Google/Linear edits, remote pending/failure/conflict, navigation acknowledgements,
  and mixed requests such as “create it and show it”.

Report model rounds and tool names, discovery overhead, redundant read candidates,
runner-to-commit time, failures/retries, tokens/cache usage and cost per successful
request including failed-request spend. Do not optimize on successful fast cases
alone. Keep production UI/audio waiting time separate from model execution time.

Promotion gates: no critical wrong-target, duplicate-write, authorization,
clarification, cancellation or false-success regression; existing CI green;
a repeatable round-count reduction in targeted cases; no meaningful latency
regression elsewhere; and no increase in average cost per successful request
without an explicit decision. A small sample supports a trial, not a p95 promise.

Use a configuration-controlled policy version pinned at job acceptance. A rollback
changes new jobs while accepted jobs retain their tool policy and saved receipts.
If any early-ending prototype is promoted later, keep its flag independently
reversible. Both implemented policies can share one reviewed change, but activate them one at a time; verify a known typed and GPT-Live request
on the phone before starting the next behavior change.

## Order and stopping point

- [x] Implement first stage: detailed round evidence, consistent discovery instructions,
  trial note capture in CORE, deterministic and paired checks.
- [x] Implement second stage behind `reads-v1`: field-sufficient lookups/receipts and fewer redundant reads,
  preserving revision and source semantics.
- [x] Third-stage decision: defer it; the measured targets are met. Do not build an early-exit
  interpreter or another queue to save a final answer round.
- [ ] Record production and phone checks beside the existing release acceptance log.

Delivery wakeups from SPEED3 remain a separate measured option. If network,
dispatch or speech delivery dominates, address that wait rather than forcing
these optimizations. Stop when the common workflows are responsive and reliable.

## Fourth stage: `lean-v1`, a smaller and cache-stable prompt (October 6, opt-in)

Production traces from October 5–6 (6 GPT-Live requests, content-free events)
showed where the per-round cost goes:
- Each model round took a median of 1.9 s, with requests needing 2–6 rounds.
- Tool definitions were 34.3 KB of every round (about 70% of the ~12k input tokens).
- **Every first round had zero cached tokens**, because the instructions put the current
  time (to the second) and screen context near the top, so the prompt prefix changed
  with every request. Some second rounds also missed the cache.
- A `work_followup` round preceded the lookups in 2 of 6 requests, even though recent
  work already carried the finished request's saved record IDs.

`lean-v1` is `reads-v1` plus:

1. **Cache-stable layout.** Static instructions (policy, capabilities, tool guidance,
   runner rules, search rules, voice-end policy) come first and are byte-identical
   across requests of the same workspace kind. Per-request DATA (profile and current
   time, focus, screen, memory, receipts, recent searches, recent work) moves to a
   second system message after them. No instruction text changes meaning.
2. **`prompt_cache_key`** derived from the tool names plus static text, so requests
   sharing that prefix route to the same cache.
3. **Smaller always-loaded tools.**
   - `task_batch` and `task_selection_update` move behind the `tasks` group, where
     bulk edits load them.
   - `calendar_event_read` and `calendar_connection` join the initial set, because
     agenda requests were loading `calendar_read` for them.
   - `tools_load` lists each group with one short phrase. Usage rules stay in the
     loaded tools' own definitions.
   - Tool definitions drop from 35.4 KB to 27.7 KB.
4. **Follow-up linking in the same round.** When recent work already shows the earlier
   request succeeded with the needed IDs, `work_followup` is returned alongside the
   lookups. Only writes wait for its result.
5. **Context budgets.** Earlier conversation keeps the newest turns within 6,000
   characters, each at most 1,500. Recent-work items keep IDs, status and saved
   records, but trim request/outcome text to 600/500 characters.

Measured locally (deterministic model, first round): 47.1 KB → 39.8 KB of input
(−15%). Of the lean total, about 35 KB (tools plus static text) is identical across
requests and can stay cached, against roughly the tool block today. Expected effect:
lower time-to-first-token and cost on every first round. This is not yet a
measured speedup.

Rollout: set `JARVIS_AGENT_TOOL_POLICY=lean-v1` on API and worker together (dev
first). Accepted jobs keep their pinned policy; rollback changes new jobs only. Compare
with `--tool-policy reads-v1|lean-v1` in the paid campaign, then a typed and GPT-Live
phone check, before making it the default.

Why not drop injected context in favour of search? Every lookup the model must
request costs a full model round (about 2 s). The small injected DATA (recent
work, memory, screen) prevents those rounds. It is cheap once the static prefix is
cached, and it is now budgeted. Better search would help the lookups that remain,
not replace the injected context.

## Official guidance

OpenAI recommends reducing unnecessary requests, combining work that genuinely
belongs in the same call and measuring the result. It does not justify bypassing
application correctness checks. See [latency optimization](https://developers.openai.com/api/docs/guides/latency-optimization).
Its [function-calling guidance](https://developers.openai.com/api/docs/guides/function-calling)
also supports clear tool descriptions, moving deterministic work into code and
keeping the initial tool set manageable. The PR sequence above is an application-
specific proposal based on our traces, not a claim of guaranteed API savings.
