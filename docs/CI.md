# GitHub CI

The `CI` workflow runs on pull requests to `main`, pushes to `main`, and manual
runs. It must be pushed with the configurable planner implementation, including
`scripts/validate_custom_planner.py` and its browser fixture. The first GitHub-hosted run starts when this release is pushed; its result is
available in the repository Actions tab.

Two independent Linux jobs:

- **Backend and migrations:** install the locked Python environment; check
  Python correctness; upgrade an empty PostgreSQL database; check Alembic/model
  agreement; run the complete backend regression suite.
- **Frontend and browser:** install locked Node/Python dependencies; run frontend
  tests; type-check/build; install Chromium; run desktop/mobile planner acceptance
  against a disposable database and the synthetic API fixture.

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
npm ci --prefix apps/web --no-audit --no-fund
cd apps/web && npm test && npm run build && cd ../..
PYTHONPATH=apps/api JARVIS_WORKER_ENABLED=false JARVIS_EXTERNAL_SERVICES_ENABLED=false uv run --no-sync python scripts/validate_custom_planner.py
```

The migration commands in CI run only against its disposable service database;
never substitute a production connection. Unit fixtures and the browser script
also create and remove their own named test databases.

After the first GitHub run, `Backend and migrations` and `Frontend and browser`
can be selected as required checks in a main-branch ruleset. That enforcement and
Railway's “wait for CI” behavior are separate repository/deployment settings;
creating this workflow alone does not change either.

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
