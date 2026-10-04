# SPEED1/2 local validation — October 4, 2026

Scope: finish and independently review the interrupted speed work against main
`de14a22`. This report records local implementation and checks before release; those checks
alone do not establish that the changes are deployed. The cancelled subagent and completion automation were not restarted.
SPEED3/4 and production settings remain untouched.

## Implemented

- Luna has separate low and no-reasoning profiles using the same `gpt-5.6-luna`
  model, Responses transport, 8,192-token ceiling, prompt, tools and durable runner.
  `luna` (low) stays the default; `luna-none` sends an explicit `none`.
  Settings, shared-account preference lookup and eval CLI/worker propagation retain
  the selected profile. Already accepted jobs retain their captured selection.
- Content-free timing covers acceptance, runner invocations, model attempts/retries,
  token/cache/reasoning usage, tools, committed outcomes and accounting uncertainty.
  Rollbacks cannot report a successful durable save. A fallback logging handler
  makes the events visible when Uvicorn has not configured the root logger.
- The summarizer joins request revisions and invocation IDs, retains censored/failed
  attempts and deduplicates provider response IDs and repeated log lines.
  Cost per durable success includes recorded unsuccessful-request spend; missing
  usage or uncertain outcomes are flagged as incomplete lower bounds.
- Typed chat measures the first observed foreground card or assistant reply for
  requests sent by the same tab. Historical/reloaded cards are excluded. Browser
  timing reports are bounded, deduplicated, rate limited and checked against the
  request's account, device, revision and terminal status.
- Each eval attempt saves `latency.log` and `latency-summary.json` beside existing
  traces and budget evidence. Langfuse export gains explicit effort and recorded
  HTTP-duration metadata; no duplicate tracing framework or automatic export was added.

Review fixed misleading cost-per-success calculations, precommit outcome attribution,
missing profile propagation, restarted-run retry identity and timing of historical
cards. A plain assistant reply now gets measured even when navigation correctly
creates no action card.

## Verification

- Complete backend suite: **988 passed, 1 skipped, 1 failed** on its first run.
  The one failure was worker recovery because the invocation disabled its worker.
  Rerunning with that worker enabled passed. Final focused run: **29 passed**,
  covering that recovery test, transaction/timing guards, report arithmetic and
  eval-profile propagation. The broader suite was not rerun solely to change its
  historical result; this is 989 passing backend tests across the full run and
  the corrected recovery rerun, with one existing skip.
- Earlier focused backend regression: **114 passed**, including work continuation,
  Live handling, Responses and agent selection.
- Frontend: **225 passed, 1 skipped**; production build passed. Existing Rollup
  annotation/chunk-size warnings remain.
- Eval scaffolding: **84 passed, 2 skipped** after the fixture changes.
- Real browser acceptance passed at a phone-sized viewport, with desktop overflow/
  screenshot checks. It covers silent navigation, compact Edit/Revert cards,
  out-of-order completion, anchored history, reload, voice-draft recovery, no work
  completion notifications, accepted render timing and no historical remeasurement.
  This is browser automation, not a physical phone or real microphone test.
- Lint/syntax and diff checks passed. Coverage ratchet retains **215 fully bound**
  cases out of the **1,051-case** catalog; that is not a claim that every
  catalog case was run or fully automated.

The initial new browser assertion watched the wrong endpoint; correcting it to the
actual `/work/{id}/latency` contract made the check pass. Application logs already
showed that the render event itself was accepted.

Local verification logs are under `artifacts/speed-2026-10-04/`. The complete test
run retains `full-backend.log`; the corrected focused run is `final-focused.log`.
Browser and final harness logs are `browser-final.log` and `evals-final.log`.

## Bounded paid comparison

Four deterministic scenario oracles per profile, interleaved low/none then none/low:

- `task_capture.01`: create an unambiguous task.
- `task_edit.19`: ask for clarification on an ambiguous edit without changing a task.
- `task_edit.04`: update a task deadline.
- `notes.02`: create a note.

Both profiles passed **4/4** scenario assertions. Each produced three successful
durable mutations plus one correctly waiting clarification. Both used 14 provider
calls, with no provider retries. None returned zero reasoning tokens; low used
109 reasoning tokens across these requests.

Observed median runner-start to committed outcome, including the clarification:
**5.428 s low, 4.994 s none**. Median summed model time: **5.074 s low, 4.643 s none**.
The deadline edit used five model calls; the other scenarios used three each.
This suggests model-round reduction may be worth a later measured experiment,
but this pass does not change prompts, tools, discovery or final-reply behavior.

Recorded successful comparison spend:
- Low: **$0.003963855**.
- None: **$0.003423134**.
- Combined: **$0.007386989**.

These are four trials per profile, with substantial prompt-cache hits and no
controlled cache flush. Order, model variance, network conditions and cache effects
remain. This does not prove general quality parity or a repeatable speed/cost
advantage. Keep low as default. No reasoning is available as an explicit option
after these code changes are deployed.

The smoke uses the real durable runner against disposable local synthetic databases.
It does not benchmark production dispatch, actual browser/network latency or Live
speech. Do not compare these backend times directly to end-to-end target values.

### Preserved unsuccessful fixture run

The first eight trials all failed the strict oracle on an unexpected
`structure_schemas` mutation. Their requested operations otherwise completed or
correctly asked for clarification. The saved corpus was at migration 0020 and its
definitions predated current lazy schema normalization.

The harness now migrates only each owned disposable clone to 0023/current head and
normalizes existing owner definitions **before** the before/after snapshot. A
second normalization is idempotent. The original corpus remains at 0020 and was
not altered. Schema mutation during model execution remains forbidden; the oracle
was not loosened to force a passing result.

Original failed evidence remains in `low-a`, `none-a`, `none-b` and `low-b`.
The corrected comparison is in the corresponding `*-v2` directories.
`paired-smoke.json`, `paired-smoke-v2.json` and `comparison.json` retain the results.
All trials including fixture-invalid runs cost **$0.032171889**; no uncertain spend
remained. First-run block caps were $0.50 and corrected-run block caps $0.25.
Actual combined spend stayed well below the stated $2 total smoke ceiling.
No production data or external connector writes were used.

## Langfuse status

Read access works for **My Project / Davis's Organization**, project
`cmtz0kofn00mrad0cqoefpl6h`. Existing synthetic eval observations were readable.

The existing integration exports historical synthetic evidence after a run; it is
not live production tracing. Reconstructed child-span placement remains approximate.
Use saved transport duration metadata and local joined events for measured timings.
The new exporter fields passed offline tests; these smoke runs were not newly
uploaded. No transcript collection or cloud evaluator was enabled.

## Remaining acceptance

- Deploy only through a separately approved commit/PR/CI rollout. No production
  benefit is claimed from uncommitted local changes.
- Prove a known request produces correlated Web/Worker production events, then
  gather representative warm/cold and competing-job samples.
- Measure actual requested page visibility and first useful/completed speech on a
  real phone. Chat-card visibility and Live append acknowledgement are not those
  milestones; voice render/audio timing remains unmeasured.
- Expand profile comparisons to navigation, schedule reads, corrections, custom
  records/templates/reviews and source-backed edits before any default promotion.
  Passing deterministic regressions does not establish model parity there.
- Only then choose SPEED3/4 work from measured waits and round counts. Keep speech
  settling, quiet gates, durable queue semantics and automatic profile selection
  unchanged in this batch.

## Follow-up: Standard versus Fast, retaining low reasoning

The owner subsequently authorized a Fast-mode comparison on October 4. This is
separate from the earlier low/none comparison. No production tier or profile was
changed, and no new application Fast toggle was added.

The same four scenarios ran three times per tier, serially against fresh isolated
synthetic clones: Standard/Fast, Fast/Standard, Standard/Fast. All 24 outcomes
passed the existing state oracles: each tier produced nine successful mutations
and three correct clarification requests. Model, low reasoning, tools, output
limit and fixture were unchanged. Natural model variation remains; instructions
use the fixture clock, while runtime timestamps and request IDs are not identical.

Every Standard response reported `default`; every Fast response reported `priority`,
the documented response label for GPT-5.6 Fast. There were no downgrades, retries
or unsettled calls. Actual returned tier was verified rather than inferred from
the requested setting.

Across twelve trials per tier:
- Median runner-start to committed outcome: **5.260 s Standard / 4.272 s Fast**,
  a **0.988 s / 18.8%** reduction.
- Paired savings: median **1.190 s**, mean **1.565 s**; Fast was quicker in all
  twelve pairs. These statistics differ from the difference of cohort medians.
- Observed p90: **8.592 s / 5.699 s**. Twelve samples are insufficient for a stable
  production tail-latency estimate.
- Per-scenario median: task creation **4.617 s / 3.767 s**; note creation
  **4.487 s / 4.128 s**; deadline editing **8.592 s / 5.699 s**; ambiguous-edit
  clarification **5.367 s / 4.302 s**.
- Model calls: **40 Standard / 37 Fast**; reasoning tokens **390 / 399**.
  This measures whole workflow behavior, not a controlled identical-token
  inference benchmark. Some gain coincided with fewer model rounds on Fast.

Total spend for this follow-up: **$0.048293337**, comprising **$0.011322847 Standard**
and **$0.036970490 Fast**. Six independent $0.30 block limits bounded the maximum
at $1.80, below the promised $2 test ceiling. The earlier low/none runs are separate.

Fast's published token rates are twice Standard's for this model. The observed
cost ratio here was higher because cache usage differed: **96.9% cached input
Standard versus 86.1% Fast**. The first Fast block cost $0.022215684 with substantial
cache writes; later Fast blocks cost $0.007533702 and $0.007221104, roughly 1.9 times
their corresponding Standard blocks. All blocks remain in the result. There was
no cache flush or independent control of provider cache placement, so the initial
difference does not establish that tiers use separate caches.

Recommendation: Fast plus low reasoning is a more promising next interactive
option than removing reasoning solely for latency. A later app rollout should be
explicit and account for premium pricing/returned tiers; scheduled background
maintenance can remain Standard. These narrow tests show no observed quality
regression, not general equivalence. Production dispatch, device rendering and
Live speech remain outside these measurements.

### Reproducibility and harness checks

The agent eval campaign accepts `--service-tier default` or `--service-tier fast`
with `--model luna --mode live-model --no-support`. Selection is confined to the
isolated provider transport; it does not change app defaults or project settings.
The fixture explicitly sets Standard on ordinary eval requests so a project-level
Fast setting cannot silently change the control group. Fast selection is restricted
to agent-only campaigns.

Reservations include the 2× premium before network access. Settlement uses the
served tier, preserving downgrade costs and leaving unverified tiers uncertain.
The raw application timing log is preserved; the derived latency report replaces
its Standard model estimates with matching provider-ledger tier costs by response
ID. It therefore does not silently report Fast at Standard prices.

Offline harness verification: **92 passed, 2 skipped**; correctness lint and diff
checks passed. New coverage includes tier propagation, unchanged reasoning/prompt,
premium reservations, actual-tier settlement, downgrade pricing, missing-tier
uncertainty, and report repricing without overwriting raw evidence. Application
code was not changed for this follow-up.

Evidence: `artifacts/speed-2026-10-04/fast-comparison/summary.json` and
`comparison.json`, plus six block directories with immutable inputs, state,
transport traces, latency reports and budget ledgers. No Langfuse upload or paid
LLM judge was used; correctness checks are the existing deterministic state oracles.

Official references checked for this experiment:
- [Fast-mode request and returned-tier behavior](https://developers.openai.com/api/docs/guides/fast-mode)
- [API pricing, including GPT-5.6 Luna](https://developers.openai.com/api/docs/pricing)

## Quick follow-up: Fast with reasoning disabled

At the owner's request, one further four-scenario batch used `luna-none` with
`service_tier=fast`. All four state oracles passed (three completed mutations and
one correct clarification). All fourteen provider responses confirmed `priority`,
all requested effort values were `none`, and reported reasoning tokens were zero.
No retries or unsettled costs occurred.

Median runner-start to committed outcome: **4.860 seconds**, versus **4.272 seconds**
in the earlier twelve-trial Fast/low sample. Per-case times were task creation
**4.652 s**, note creation **4.932 s**, ambiguity clarification **4.788 s**, and
deadline editing **9.525 s**. The deadline edit took five model calls; the other
cases took three.

This quick run did not show an extra speed benefit from disabling reasoning.
It is one run per case, compared with an earlier larger sample rather than a fresh
interleaved control. Cache hits were lower (84,254 of 121,908 input tokens, 69.1%);
provider timing and tool-round variation also remain uncontrolled. Do not conclude
that none is inherently slower. The evidence does not justify reducing reasoning
for speed; retain the recommendation of low reasoning for interactive Fast trials.

Spend: **$0.023133768** under the $0.50 hard cap. Evidence is in
`artifacts/speed-2026-10-04/fast-none-quick/quick-summary.json` with the original
traces, state oracles, returned tiers and ledger alongside it.
The existing harness was used without source changes. Production settings,
application defaults and deployment remain untouched.


## Accepted interactive default

After the Fast/none follow-up, the owner approved **Luna low reasoning + Fast**.
The application now defaults interactive Luna work to explicit `service_tier=fast`,
including durable work delegated from GPT-Live. No new classifier or execution
path was introduced. The separate none profile remains available, without becoming
the default reasoning profile.

- `JARVIS_AGENT_SERVICE_TIER` is typed as `fast` (default) or `default` (Standard).
  Apply the same override to API and worker. Settings show the effective tier.
- Jobs capture the tier with their model profile when accepted. Later preference/
  configuration changes cannot alter it. Jobs accepted before this field existed
  continue on Standard.
- Reservations use Fast rates, including cache-write and long-context adjustments.
  Actual response tiers drive recorded cost: `priority`/`fast` use premium rates;
  `default` uses Standard rates. Returned tier, requested tier and accounting basis
  persist with usage. Missing/unrecognized metadata uses the requested price and
  explicitly marks an estimate, rather than silently reporting verified billing.
- Tier-aware latency cohorts separate Standard and Fast. Downgrades remain in the
  requested Fast cohort with their actual returned tier exposed. Estimated-cost
  cases remain marked in aggregate reports.
- Scheduled extraction, dream/rule/note learning stay explicitly Standard;
  embeddings do not receive an unsupported service-tier argument.
- Eval environment remains Standard unless its manifest selects Fast. The isolated
  metered transport continues to reserve premium spend before sending, reconcile
  the returned tier and avoid double multiplying application prices.

This verification preceded release. No production settings or personal data were
changed during verification; no additional paid evaluation was run for this default.
Next is PR/CI rollout, a known-request production trace and phone/voice acceptance,
then SPEED3/4 chosen from observed delivery waits and unnecessary model rounds.


### Default-change verification

- Full backend suite with worker disabled: **1,001 passed, 1 skipped** in 292.81 s.
  The worker crash/replay test ran separately with its worker enabled: **1 passed**.
  Combined: **1,002 passed, 1 skipped**.
- Eval-harness suite: **92 passed, 2 skipped**, without live provider calls.
- Frontend TypeScript/production build passed. Existing dependency-annotation and
  bundle-size warnings remain.
- Python correctness lint (`E9,F63,F7,F82,F811`) and `git diff --check` passed.
- The initial focused run had two new money assertions below the database's existing
  six-decimal storage precision. Their tolerance now matches half a stored unit;
  the complete rerun passed. Provider-rate unit assertions retain strict precision.
- Regressions cover Fast/default/legacy queue pinning, reported Standard downgrades,
  Fast reservations including long context, absent-tier estimate labels, maintenance
  and embedding isolation, visible bootstrap tier and separate latency cohorts.

No new paid calls were needed; the earlier paid comparison remains the evidence
for the latency tradeoff. Production tier/readback, phone speech and real-world
end-to-end latency remain unverified.

## Release preparation

The owner authorized shipping on October 4. The speed release includes this work,
the [round-reduction plan](ERI_ROUND_REDUCTION_PLAN.md), and its sanitized 24-trial
baseline. Frontend unit verification: **225 passed, 1 skipped**. Catalog validation
and the coverage ratchet passed (215 fully bound cases). CI must pass before merge;
production checks are distinct from these local results. No round-reduction behavior
is enabled in this release.

The first PR CI run caught a test-only import-path dependency in the new latency
summary test. It now loads the script by its repository-relative path, matching
existing script tests, and is checked with console-script pytest without a
repository-root PYTHONPATH. The failing CI run is retained as evidence.

The second PR run passed frontend/browser and security, plus 1,001 backend tests,
but found the new browser-timing test relied on a locally supplied encryption key.
That test now creates its own synthetic key. Its clean-environment rerun excludes
local environment files, inherited provider keys and an inherited encryption key.
