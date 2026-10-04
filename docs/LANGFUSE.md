# Langfuse evaluation integration

The first integration exports completed synthetic evaluation campaigns. The
existing runner still owns execution, PostgreSQL isolation, assertions, STOP/resume,
and the shared API budget. Langfuse receives saved evidence and scores afterwards;
exporting never calls a model, opens a database, or enables an evaluator.

## Setup and use

Create project-scoped keys in Langfuse Project Settings → API Keys. Save
LANGFUSE_BASE_URL, LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY in the repository's
ignored .env or environment. Use the project's regional HTTPS origin, without
/api/public. The exporter does not print keys or use redirects/proxy environment
variables. Keys are not passed to campaign worker processes.

From the repository root:

    .venv/bin/python -m scripts.app_eval.runner langfuse --status
    .venv/bin/python -m scripts.app_eval.runner langfuse artifacts/app-evals/RUN --dry-run
    .venv/bin/python -m scripts.app_eval.runner langfuse artifacts/app-evals/RUN
    .venv/bin/python -m scripts.app_eval.runner langfuse artifacts/app-evals/RUN --verify

The dry run requires neither credentials nor network access. The export is
explicit; merely running a test, report, or CI job never sends telemetry.
A completed grader.json is optional. When present, its case IDs, harness/catalog
versions, and original code summary must match the run.

## What appears in Langfuse

- One experiment item for every selected catalog case, including unassessable
  cases. The frozen inputs, expected criteria, coverage gaps, code status, and
  external review remain separate. This is not a percentage of all cases passing.
- One job trace per executed job, referenced by its case items. Shared supporting
  suites are exported once, rather than duplicated hundreds of times.
- Child observations for saved model/tool activity. Business-record changes and
  replies are included; complete DB snapshots, auth tables, worker configuration,
  credentials, and logs are not uploaded.
- Categorical automated_status and external_review_status scores for cases with
  assessment evidence. Blocked/unassessable cases stay visible in the experiment
  and coverage metadata but receive no correctness score.
- Exact criterion judgments and reasons from Sol are retained in the case output.
  Scores are imported API judgments attributed to the recorded grader; exporting
  them does not invoke a Langfuse or OpenAI judge.
- Recorded model cost is attached to model observations once, using the campaign
  ledger. Cache bookkeeping has no model charge. Unsettled calls suppress automatic
  Langfuse cost estimates and carry an explicit unsettled label plus reserved amount;
  zero recorded cost does not mean those calls were free.
  Campaign estimated/uncertain totals are experiment metadata. Codex grader
  usage remains separate from application API costs.

Local artifacts are authoritative and retain complete evidence. Large fields are
bounded with explicit truncation, a hash, and a pointer to local evidence.
Historical job durations are recorded; tool/model placement in the reconstructed
timeline is approximate, clearly labelled, and is not a live distributed trace.
Do not use reconstructed child-span latency as a provider latency measurement;
the saved transport output retains recorded duration_seconds where available.

## Recovery and verification

Exports use the current OTLP/HTTP JSON endpoint with v4 ingestion and the public
Scores API. This small historical-artifact adapter avoids global SDK tracing in
isolated workers. The current SDK remains appropriate for future live tracing.

Langfuse v4 observations are immutable and repeat submissions are not reliably
deduplicated. A local receipt, scoped to project/region and frozen evidence hash,
records each accepted batch. A second identical export skips accepted writes.
The exporter locks the campaign during upload. Keep the langfuse-*.json receipt
with the campaign; deleting it loses this retry protection.

An uncertain request is recorded before network I/O. Resume first checks the
read API; it never blindly re-sends observations whose acknowledgement was lost.
If readback cannot yet prove acceptance, it stops for later verification.
Partially rejected uploads also stop for reconciliation. Credentials/server errors
are surfaced without logging provider response bodies.

A changed report, grading report, or exported evidence produces a separate
experiment revision. Finish grading before exporting when possible. Never mutate
old experiment observations to claim a newer verdict.

Successful verification checks expected observation and score IDs through the
read API. Exit code 2 means ingestion/readback is incomplete; retry --verify
later. Exit code 1 means an error. Existing eval results and exit status are
unchanged by exporter errors.

## Cost and next scope

Exports incur no model calls. They count toward Langfuse's storage/ingestion
allowance. Keep local evidence beyond the cloud plan's access window. No automatic
LLM evaluators, schedules, or production transcript collection are enabled.

The next separate step is optional live tracing across voice handoff, queue,
background agent, tools, memory retrieval, and learning. Propagate request IDs
across workers and redact production content before enabling that telemetry.

Official references:
- https://langfuse.com/integrations/native/opentelemetry/experiments
- https://langfuse.com/integrations/native/opentelemetry/migration-to-v4
- https://langfuse.com/docs/evaluation/evaluation-methods/scores-via-sdk

## Verified first campaign

On September 23, the corrected v2 export of half-20260923-sol-graded was verified
in My Project, Davis's Organization:
https://us.cloud.langfuse.com/project/cmtz0kofn00mrad0cqoefpl6h

Experiment ID: eri-eval-82fb37d8bf2e95a717fcad0b.
It contains all 501 selected cases, 937 observations, and 256 categorical scores
for cases with assessment evidence. The 373 unassessable cases remain visible.
The 104 real provider calls total $0.055128104, matching the original ledger.
A repeated export used only GET requests; all score values matched.
Local verification files are kept beside the campaign.

The earlier importer-v1 experiment, eri-eval-52354b9ff7fcfd85f5835826, was
superseded because embedding-cache bookkeeping was incorrectly treated as paid
model activity. It was deleted with explicit user approval on September 23.
Readback confirmed no old observations, experiment, or scores remain, while the
corrected v2 import retains all 937 observations and 256 verified scores. Local
evidence is preserved, including the deletion audit at
artifacts/app-evals/half-20260923-sol-graded/langfuse-v1-cleanup.json.

## SPEED1/2 measurement additions — October 4, 2026

Project-scoped read access was reverified in My Project / Davis's Organization.
The existing integration works for saved synthetic eval evidence; it does not
currently trace production sessions.

Eval workers now save content-free `latency.log` and joined
`latency-summary.json` in each attempt directory. Reports distinguish committed
outcomes, revisions/invocations, retries, effort, token usage and incomplete cost
evidence. Full trace/DB artifacts continue to use the existing local eval boundary.

The opt-in exporter adds explicit `reasoning_effort` and
`recorded_duration_seconds` metadata when the provider transport captured them.
The latter is measured HTTP duration; reconstructed timeline placement remains
approximate. Do not infer real span latency from reconstructed start/end positions.
Usage is still charged once from the campaign ledger. These additions were tested
offline; no new cloud export or live production tracing was enabled during this pass.

Browser timing covers same-tab typed results observed in the foreground.
It cannot prove a requested Calendar/task page was painted, or that a Live response
was spoken. Those checks remain part of the device baseline.
See [SPEED1/2 validation](SPEED_1_2_VALIDATION.md).
