# Eridani app evaluations

Start with [the functionality overview](../../docs/APP_FUNCTIONALITY.md),
[Rowan’s persona](personas/rowan-v1.md), and [execution protocols](protocols.md).

The catalog contains **1,051 scenarios across 42 features**. The on-demand runner
indexes every scenario, selects its available adapters, and reports unmet criteria.
This does **not** mean every scenario is automated. Current bindings cover **112**
complete acceptance scenarios; component evidence is available for **224** cases
(overlapping acceptance coverage). Remaining scenarios retain incomplete or physical
acceptance requirements. Batch C adds 25 Quick-capture and 25 onboarding scenarios,
with 12 exact deterministic bindings. See
[automation-coverage.json](automation-coverage.json) for the per-feature backlog.
Historical reports keep their original scenario counts and outcomes.

## Execution types

- **contracts:** task/record commands, dates, recurrence, schema, permissions,
  notifications, budgets and recovery.
- **agent:** queued conversations, clarifications and action receipts.
- **pipeline:** memories, both dream systems, notes/list organization and retrieval.
- **integration:** Google Calendar, Linear and scoped external-agent API/MCP.
- **browser:** navigation, settings and site controls, using synthetic provider/media fixtures.
- **voice:** current Live/idle/wake/shutdown regressions; actual microphone, OS
  permissions, background behavior and real GPT-Live quality remain separate evidence.

A type identifies a scenario’s primary feature, not its only implementation layer:
task cases can have both deterministic and real queued-Luna variants.

## Run on demand

Run from the repository root in Linux/WSL with locked Python and web dependencies.
No scheduler, new production endpoint, or paid CI job is installed.

~~~sh
docker compose -f compose.eval.yml up -d
.venv/bin/python -m scripts.app_eval.runner validate
.venv/bin/python -m scripts.app_eval.runner coverage --output artifacts/eval-coverage.json

# Inspect execution and cost limits without connecting to a database/provider.
.venv/bin/python -m scripts.app_eval.runner plan --mode offline,live-model,live-service

# Offline components and existing regression/browser suites.
.venv/bin/python -m scripts.app_eval.runner run --mode offline

# Available paid Luna and learning trials. ONE allowance includes all workers,
# embeddings, judge calls and retries. Gemini is excluded.
.venv/bin/python -m scripts.app_eval.runner run --mode offline,live-model --run-paid --max-usd 10

# Narrow a run; comma-separated selectors can be combined.
.venv/bin/python -m scripts.app_eval.runner run --types pipeline --features memory_dream --mode offline
.venv/bin/python -m scripts.app_eval.runner run --cases task_capture.01,clarifications.01 --mode live-model --run-paid --repeats 3 --max-usd 10

# --no-support omits related whole-suite evidence; explicit case bindings still run.
.venv/bin/python -m scripts.app_eval.runner plan --features memory_capture --no-support
~~~

Defaults: four worker processes, at most two simultaneous paid jobs and two browser
jobs, one repetition, a $10 campaign ceiling and a $1 judge sublimit within that
ceiling. A request count limit is also enforced. A supplied lower dollar cap lowers
the judge ceiling too. These are maximum allowances, not cost predictions.

The plan currently expands to 245 reusable jobs: 108 command contracts, 41 queued
agent trials, 43 memory/note pipeline trials, 44 backend test files, six browser
fixtures and three opt-in connected-service smoke probes. Sixty-five jobs use paid
inference. Suite results are reused within a repetition without reusing mutable
application state. There are no real GPT-Live/audio provider calls in this runner yet.

The legacy `models` command delegates to the same campaign ledger and only accepts
Luna. Its old request-count flags remain supported. Historical paired-model reports
are retained as historical evidence, not current defaults.

Exit codes: **0** means all selected acceptance evidence passed; **1** means an
assertion, component/suite, safety or infrastructure failure; **2** means incomplete
coverage or prerequisites without a recorded failure. An unimplemented scenario
cannot silently pass because a related test file passed.

## Isolation and costs

`compose.eval.yml` owns a dedicated PostgreSQL volume, role and localhost port
54340. The marked Rowan corpus persists. Each job receives a fresh, randomly named
clone and its own process; the parent drops only that validated clone. Production
and ordinary development database URLs are refused. Existing pytest/browser
fixtures use the owned clone rather than silently targeting a second database.

Corpus, fixture, grader, dependency and application fingerprints are retained.
Relative user dates use the existing fixed persona clock; real lease/network clocks
remain real. Model tests inspect saved state and cross-account canaries instead
of trusting “Done.” Fault tests explicitly state whether the provider is real or mocked.

Offline workers block outbound provider access and use synthetic credentials.
Paid workers load only the configured OpenAI key from the local environment/.env.
Synchronous and asynchronous HTTP calls share a transactional SQLite allowance;
each call reserves a conservative bound before network contact. Unknown usage keeps
its reservation after process death or timeout. There is no automatic forgiveness
of uncertain charges or resetting of the allowance on resume.

The ledger includes estimated actual cost, uncertain reservations, request counts
and attribution to agent/pipeline/embedding/judge. It is separate from the production
app’s weekly/monthly usage history. The old extrapolation from twenty text probes is
not a measured cost for all 1,001 mixed scenarios.

A per-campaign embedding cache reuses identical text/model/dimension inputs without
reusing mutable retrieval state. Fault and stale-index checks use separate fixtures.
Semantic Luna judging is restricted to supplied criteria and saved evidence, cannot
override a code assertion failure, and can return `needs_review`. Broader judge
calibration and retrieval-quality adapters remain on the coverage backlog.

## Stop, resume and compare

~~~sh
# Replace RUN with the directory printed by the runner.
touch artifacts/app-evals/RUN/STOP

# After correcting a missing prerequisite, remove only this STOP file.
rm artifacts/app-evals/RUN/STOP
.venv/bin/python -m scripts.app_eval.runner resume artifacts/app-evals/RUN
.venv/bin/python -m scripts.app_eval.runner report artifacts/app-evals/RUN

.venv/bin/python -m scripts.app_eval.runner compare artifacts/app-evals/BASELINE artifacts/app-evals/CANDIDATE --output artifacts/comparison.json
~~~

Completed passes **and failures** are preserved; resume retries only incomplete,
blocked or infrastructure outcomes. Prior attempts remain on disk. Resume refuses
source/fixture changes or changed connected-resource configuration. A code change
requires a new campaign: subtract all previous estimated and uncertain charges from
the still-authorized allowance, rather than resetting to $10.

Orphan cleanup requires matching campaign/job/attempt markers in the database.
A worker is signaled only if its PID still belongs to that precise worker command.
Service cleanup journals are retained; unresolved remote cleanup requires inspection
before retrying with live credentials.

Comparisons require the same catalog, corpus, harness/oracles, selections, model,
modes and repeat count. Application revisions may differ. One sample does not
establish a stable model ranking.

## Connected integrations

Simulated Google/Linear/API/MCP and backup regressions run offline by default.
Connected smoke probes require `--mode live-service --live-config PATH`.
Copy [live-services.example.json](live-services.example.json) outside tracked files
and export the named `ERIDANI_EVAL_*` credential variables. Never put credential values
in the configuration file or use production resources.

- Google: a dedicated secondary calendar whose name starts with “Eridani Eval.”
  The probe creates, reads, edits and removes a generated event, with no attendees.
- Linear: the exact configured organization and a dedicated team whose name starts
  with “Eridani Eval.” Cleanup capability is checked before creating an issue.
- R2: a dedicated test bucket and the `eridani-eval/` prefix; an encrypted synthetic
  object is uploaded, downloaded, verified and removed. This is not a full database
  restore/PITR acceptance test. The optional backup runtime dependencies must exist.

Only resource IDs generated and journaled by the probe are deleted. Missing credentials
are reported as blocked. These smoke probes are supporting service evidence, not
proof that all Google/Linear sync or recovery scenarios passed.

## Evidence and extending coverage

Each campaign writes `manifest.json`, `report.json`, `report.html`, `junit.xml`,
`spending.json`, a durable budget database and per-attempt state/tool/provider/browser
evidence. HTML filters by case, feature, type and status. JSON is authoritative.
JUnit includes acceptance outcomes **and** execution failures, so a component
failure cannot disappear behind a skipped acceptance scenario.

Keep generated artifacts local: they contain synthetic conversation and database
snapshots. The artifact directory is gitignored.

To automate another scenario:
1. Add its isolated adapter/oracle to contracts, model_runner, pipelines, a named
   pytest node, or a browser fixture.
2. Add an explicit entry to `bindings.json`, listing exactly which `expected.N`
   and `invariants.N` criteria the assertion checks.
3. Use `component` while only part of the protocol is exercised. Whole related
   suites stay `supporting`; do not relabel them acceptance to inflate coverage.
4. Run validate, the harness tests and a targeted trial. Regenerate coverage:
   `python -m scripts.app_eval.runner coverage --output evals/app/automation-coverage.json`.

Physical evidence can be imported with `runner import-evidence RUN evidence.json`.
It must name case_id, criteria, status, observed_at, observer, device, commit,
fingerprint and attachment paths inside the campaign. Observations must match the
campaign revision and time. Manual evidence cannot erase a failed automated repeat.

CI validates catalog/tool-surface drift, binding targets, coverage plans, budget
concurrency, transport guards and report integrity without provider secrets.
[Findings](FINDINGS.md) and [TODO](../../docs/TODO.md) track failures and unfinished
automation separately.

### External semantic grading

Use `--judge external` on a campaign to save semantic grading requests with raw state evidence instead of calling the built-in Luna judge. The judge API sublimit becomes zero. Automated reports retain `needs_review` until an external review is supplied separately; external review must preserve hard failures and incomplete acceptance coverage. External Codex usage is separate from the application API ledger.

### Langfuse export

The opt-in command exports completed saved evidence without rerunning models:

    python -m scripts.app_eval.runner langfuse --status
    python -m scripts.app_eval.runner langfuse RUN_DIRECTORY --dry-run
    python -m scripts.app_eval.runner langfuse RUN_DIRECTORY
    python -m scripts.app_eval.runner langfuse RUN_DIRECTORY --verify

See [setup, evidence scope, and recovery](../../docs/LANGFUSE.md). All selected
cases remain visible, while code outcomes, external grades, and missing coverage
remain distinct. Local receipts prevent blind replay of immutable observations.
