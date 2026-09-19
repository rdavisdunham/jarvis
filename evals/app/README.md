# Eridani app evaluations

Start with [the functionality overview](../../docs/APP_FUNCTIONALITY.md), [Rowan’s persona](personas/rowan-v1.md), and [execution protocols](protocols.md).

This version contains **1,001 acceptance scenarios across 40 feature areas**: 25 per area, plus a 26th date/time case for an observed database-timezone defect. The catalog covers existing functionality, disabled-feature boundaries, failure recovery, permissions and device behavior. It is not a claim that all 1,001 scenarios are automated or passing.

## What is executable now

- The existing backend regression suite, frontend unit tests and six desktop/mobile browser suites.
- 88 additional database component checks against a populated Rowan corpus. Each gets a fresh PostgreSQL clone and preserves before/after state and tool evidence.
- Twenty paired Luna/Gemini queued-backend probes with state-based graders, alternating model order, repetitions, provider request limits and a stop file.
- Meta-tests for dataset integrity, exposed-surface drift, database boundaries, graders and the production queue adapter.

The remaining catalog entries are detailed acceptance protocols. Device, real OAuth and deployment scenarios need their named environment and evidence. Pipeline model-quality cases need gold-label review and dedicated live adapters before unattended scoring. A component pass does not mark the corresponding complete conversational/browser scenario passed.

## Quick start

Run from the repository root inside Linux/WSL with locked project dependencies installed.

~~~sh
docker compose -f compose.eval.yml up -d
.venv/bin/python -m scripts.app_eval.runner seed
.venv/bin/python -m scripts.app_eval.runner validate
.venv/bin/python -m scripts.app_eval.runner catalog
.venv/bin/python -m scripts.app_eval.runner case routing_dream.04
.venv/bin/python -m scripts.app_eval.runner contracts
.venv/bin/python -m scripts.app_eval.runner regressions --scope all
.venv/bin/python -m pytest -q evals/test_app_eval.py
~~~

The latest component baseline has 87 passes and one reproduced app defect. The contracts command currently returns a nonzero exit for the reproducible EVAL-001 defect. Keep that failure visible. Individual cases can be selected with a comma-separated --cases argument.

The Docker service uses PostgreSQL 16.15, a localhost-only port 54340, the dedicated eridani_eval role, database eridani_eval_corpus and persistent volume eridani-eval-postgres. The password is synthetic-eval-only and must never be used for a public service. No production env file, application worker or external account credentials are mounted.

At authoring time Docker Desktop could not start. An isolated native PostgreSQL 16 instance was provisioned at .runtime/eval-postgres on the same local port. Its role uses UTC, matching Docker CI. Do not start both on the same port. The corpus is already seeded there; the reproducible Compose file is the portable path when Docker is available.

## Baseline versus accumulated data

The marked corpus contains Rowan’s organization, 151 tasks, 26 notes, five memory assertions and four fictional identities. It includes 125 pagination rows, ambiguous task/contact names, correct and incorrect client assignments, negative/quoted recommendations, duplicate film editions and conflicting memory spellings.

The corpus is a **golden baseline**, not a scratch workspace. Trial databases have generated eridani_eval_trial_UUID names and are dropped only by the process that created them. Failures retain JSON evidence; they do not contaminate the next trial.

Keep exploratory synthetic history in a separate database or run artifacts. Promote useful findings into a reviewed fixture/case version. This allows the suite to grow without model A and model B being evaluated on different accumulated histories.

The harness checks the local hostname, dedicated database role, namespace, synthetic marker and corpus hash. It refuses production/remote URLs, arbitrary databases and unmarked occupied databases. A partial seed gets a building marker and requires deliberate recovery; the seed command does not erase it. Export old data before intentionally creating a new baseline. Do not run compose down -v unless you mean to remove the synthetic corpus.

Relative-date truth is January 14, 2030, 09:00 America/Chicago. The live backend adapter freezes the instruction clock and uses absolute dates in its dated probes. It does not freeze runtime leases. Broader relative-date protocols must freeze the time-resolver clock as well.

## Real Luna/Gemini comparisons

No paid inference is needed to build, inspect or validate the dataset.

~~~sh
.venv/bin/python -m scripts.app_eval.runner models \
  --run-paid --models luna,gemini \
  --cases task_capture.01,task_edit.19 \
  --repeats 3 --max-provider-requests 40 --max-usd-per-model 2
~~~

Only the selected provider endpoint/model is allowed; Google, Linear, push and storage calls are blocked. Only provider keys are read from the local env file. The agent uses the actual durable queue, current production prompts, lazy tool discovery and real database commands. Persona oracle-only truth is never injected. Memory retrieval uses the real lexical fallback; this corpus intentionally has no paid embedding vectors. This subset is not a semantic-retrieval quality benchmark.

The request limit counts actual HTTP attempts, including production retries. A separate per-model dollar ceiling defaults to $2 and cannot exceed $10. Each text request reserves a conservative UTF-8 input/output bound, including cache-write and long-context premiums, before network traffic. Reported token usage settles that bound; missing/error responses retain it. The journal is saved as spending.json before each request. Limits apply per invocation: include earlier runs when deciding a new allowance. The default is not renewed permission to spend. Creating a file named STOP in the displayed evidence directory prevents the next provider call. Already issued requests cannot be unspent.

Compare all planned trials, not only successful ones. Keep provider errors and cap interruptions in the denominator, report repeats separately, and preserve clean versus recovered execution. Token usage, latency, request context metrics and prompt hashes are recorded; provider hidden reasoning and credentials are not.

The five end-of-feature cases are a frozen evaluation partition. They are visible in this repository, so this is not a secret or statistically unseen holdout. Do not tune prompts repeatedly on them and then describe the result as generalization.

## Results and continuation

Artifacts are written to artifacts/app-evals/<timestamp>-<id>/. Keep these local; raw state snapshots can become sensitive if you later add non-synthetic data. Generated HTML provides a searchable catalog. JSON evidence remains the authoritative run record.

- acceptance: the entire specified scenario was checked.
- command_contract: typed effects and database invariants were checked.
- queued_backend: natural language went through the current backend queue.
- existing_regressions: a named existing suite ran, without implying a new scenario passed.

Never replace not_run, not_completed or infra_error with passed. Never grade only from “Done” in the assistant’s reply.

Use --scope backend, frontend or browser for existing regressions. Backend provider code is enabled with fake credentials but cloud sockets are blocked by the offline pytest plugin. Browser tests use their existing synthetic provider fixtures. The optional ERIDANI_EVAL_INTEGRATION=1 meta-test runs a scripted provider through a real disposable database and queue.

[Findings](FINDINGS.md) and docs/TODO.md track actual gaps. CI validates the catalog, checks that exposed tools/routes still match the inventory, and runs the no-paid harness self-tests.

The first paid comparison is in [paid-baseline-2026-09-19.json](paid-baseline-2026-09-19.json): Luna $0.02352 and Gemini $0.49767, 19/20 complete queue outcomes each. See FINDINGS.md for failed mutations versus recovered work still marked partial. This is one repeat, not a broad quality ranking.
