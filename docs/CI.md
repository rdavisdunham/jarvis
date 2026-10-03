# GitHub CI

The `CI` workflow runs on pull requests to `main`, pushes to `main`, and manual
runs. It must be pushed with the configurable planner implementation, including
`scripts/validate_custom_planner.py` and its browser fixture. The workflow is active on GitHub. Current results are in the [Actions tab](https://github.com/rdavisdunham/jarvis/actions/workflows/ci.yml).

Three independent Linux jobs:

- **Backend and migrations:** install the locked Python environment; check
  Python correctness; upgrade an empty PostgreSQL database; check Alembic/model
  agreement; run the complete backend regression suite.
- **Frontend and browser:** install locked Node/Python dependencies; run frontend
  tests; type-check/build; install Chromium; run desktop/mobile planner acceptance
  against a disposable database and the synthetic API fixture.
- **Security audit:** `pip-audit` over the full locked Python environment
  (`uv export --all-groups` with `UV_LOCKED`, hashed, blocking); `npm audit --audit-level=high
  --omit=dev` for `apps/web` (blocking; clean as of 2026-10-02); and a gitleaks secret
  scan over full history (blocking). All actions are pinned to release commit SHAs.

The backend job also enforces an **eval coverage ratchet**: `runner coverage --baseline
evals/app/automation-coverage.json` exits non-zero if the total or any feature's
`fully_bound` count drops below the committed baseline, or (for features whose catalog did
not grow) a case that was bound becomes unbound. After binding new cases, regenerate the
baseline with `python -m scripts.app_eval.runner coverage --output
evals/app/automation-coverage.json` and commit it; lowering the baseline must be an
explicit, reviewed diff.

`.github/dependabot.yml` opens weekly update PRs for GitHub Actions, the uv lockfile and
`apps/web` npm dependencies, and the digest-pinned Docker base images in `Dockerfile.upgrade`. Vite, `@vitejs/*`, TypeScript, Vitest and `@types/*` are grouped into one
`web-toolchain` PR because they only upgrade together. Docker base-image major versions are
ignored, because CI does not build the images and PostgreSQL must match the Railway database;
move those deliberately.

Runtime versions match the production build: Python 3.12, Node 22, uv 0.11.16.
PostgreSQL uses the same 16.15 image as local Compose. Update those pins together.
Actions are pinned to verified release commit hashes, with their versions in
comments. The workflow caches dependencies, cancels superseded runs, has 20-minute
job timeouts, uses a read-only token, and retains failed synthetic reports for
seven days. It does not deploy, use production data, or need provider/API secrets.
Browser checks disable the real worker and external services.

Local equivalents:

```sh
uv sync --locked --group dev
uv run --no-sync ruff check --select E9,F63,F7,F82,F811 apps/api/jarvis tests scripts/validate_custom_planner.py
uv run --no-sync pytest -q --tb=short
uv run --no-sync python -m scripts.app_eval.runner coverage --baseline evals/app/automation-coverage.json
uv export --frozen --all-groups --no-emit-project --format requirements-txt > /tmp/req.txt && uvx pip-audit --disable-pip -r /tmp/req.txt
(cd apps/web && npm audit --audit-level=high --omit=dev)
npm ci --prefix apps/web --no-audit --no-fund
cd apps/web && npm test && npm run build && cd ../..
PYTHONPATH=apps/api JARVIS_WORKER_ENABLED=false JARVIS_EXTERNAL_SERVICES_ENABLED=false uv run --no-sync python scripts/validate_custom_planner.py
```

The migration commands in CI run only against its disposable service database;
never substitute a production connection. Unit fixtures and the browser script
also create and remove their own named test databases.

## Deployment gates (verified October 2, 2026)

GitHub ruleset `24354023` is active on `main`: the three named CI checks are
required with strict up-to-date checks; force pushes and deletion are blocked.
There is **no `pull_request` rule**. Required checks and PR-only changes are different
policies; do not describe the current configuration as PR-only.

Railway deployment triggers for both `Eridani_Web` and `Eridani_Worker` have
`checkSuites: true`, branch `main`. This is separate from GitHub branch rules.
Readback is documented in [Batch A validation](BATCH_A_VALIDATION.md). Do not create
a duplicate ruleset using the historical example below.

Historical configuration example (already applied; for reference only):

```sh
gh api --method POST repos/rdavisdunham/jarvis/rulesets --input - <<'JSON'
{
  "name": "main protection",
  "target": "branch",
  "enforcement": "active",
  "conditions": { "ref_name": { "include": ["~DEFAULT_BRANCH"], "exclude": [] } },
  "rules": [
    { "type": "non_fast_forward" },
    { "type": "deletion" },
    { "type": "required_status_checks",
      "parameters": {
        "strict_required_status_checks_policy": true,
        "required_status_checks": [
          { "context": "Backend and migrations", "integration_id": 15368 },
          { "context": "Frontend and browser", "integration_id": 15368 },
          { "context": "Security audit", "integration_id": 15368 }
        ]
      } }
  ]
}
JSON
```

(`15368` is the GitHub Actions app id, so only Actions can satisfy the checks. Add a
`pull_request` rule as well if direct pushes to `main` should be blocked.)

Read it back:

```sh
gh api repos/rdavisdunham/jarvis/rulesets --jq '.[] | {id, name, enforcement}'
gh api repos/rdavisdunham/jarvis/rules/branches/main --jq '.[] | {type, parameters}'
```

Pricing checked September 17, 2026: this repository is public. Standard
GitHub-hosted runners are free for public repositories. GitHub Free includes
2,000 minutes/month and 500 MB artifact storage for private repositories; paid
larger runners are outside that allowance. See the [GitHub billing reference](https://docs.github.com/en/billing/concepts/product-billing/github-actions).

## Local validation — September 17, 2026

The locked clean environment passed 638 backend tests (one intentional skip),
123 frontend tests (one intentional skip), TypeScript/Vite build, empty-database
upgrade and migration/model agreement, and desktop/mobile browser acceptance.
The backend run excluded local environment files and provider credentials. Tests
that previously depended on local settings now declare synthetic encryption/provider
configuration themselves. Python correctness and whitespace checks also passed.
The build retains its existing large-bundle advisory.

The first GitHub run passed frontend/browser checks and exposed one additional
clean-checkout assumption: the worker recovery test wrote to a pre-existing local
`.runtime` folder. It now uses pytest's temporary directory. Subsequent CI runs
verify that regression together with the full suite.
