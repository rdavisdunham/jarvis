# Evaluation baseline — September 19, 2026

The acceptance catalog contains **1,001 scenarios across 40 feature areas**. Each
area has 25 scenarios; date/time has one additional reproduction from this run.

Verified on the synthetic PostgreSQL instance:

- New component checks: **87 passed, 1 failed** (EVAL-001, non-UTC Revert).
- Existing backend regressions: **673 passed, 1 skipped**.
- Existing frontend regressions: **131 passed, 1 skipped**.
- All five existing browser acceptance suites passed.
- Harness self-tests: **22 passed**, including real queue execution with a scripted
  provider, negative grading controls, database cleanup and decryptable cloned receipts.

The broader 1,001-case acceptance set has **not** been fully executed. Twenty
queued-backend probes are ready for capped real Luna/Gemini runs; no paid inference
was run in this batch. Physical microphone, wake-word, push, actual OAuth consent,
R2 and disaster-recovery checks still require their dedicated environments.

[Automation coverage](automation-coverage.json) lists the exact executable cases
per feature. [Machine-readable baseline](baseline-2026-09-19.json) records the
outcomes without embedding raw database dumps.

Local evidence:

- Component state/tool traces: artifacts/app-evals/20260919T062733Z-f25a36fd/
- Backend JUnit/log: artifacts/app-evals/20260919T061558Z-f48c2c2f/
- Frontend log: artifacts/app-evals/20260919T062803Z-df857a09/
- Browser log: artifacts/app-evals/20260919T062804Z-667db37b/
- Searchable catalog and component results: artifacts/app-evals/catalog.html

These files contain synthetic data. No production records were copied or changed.
The database baseline remains at port 54340; repeated seed commands verified its
marker/hash and returned the same 151 tasks, 26 notes and five memory assertions.
