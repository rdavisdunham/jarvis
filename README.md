# Eridani

Eridani is a durable personal organizer with Dani as its assistant: tasks, reminders, calendar blocks,
notes and lists, source-backed memory, text chat and GPT-Live voice, with Google
Calendar and Linear integrations.

**Production runs on Railway at https://app.eridani.app.** See the
[cloud runbook](docs/CLOUD_MIGRATION.md). The old local production stack
(`compose.upgrade.yml`) is retired; never start it as a live writer against its stale data.

## Architecture

- **Web:** React 19 + TypeScript, built with Vite (`apps/web`). Served by the API on the same origin.
- **API:** Python 3.12, FastAPI/Uvicorn, SQLAlchemy 2, Alembic (`apps/api/jarvis`, `migrations/`).
- **Worker:** DBOS durable worker (same image) for accepted requests, reminders, integration
  sync/writes, extraction, indexing and Web Push.
- **Database:** PostgreSQL 16 (canonical store, including DBOS state). Railway PITR plus
  planned independent encrypted R2 exports ([R2 backups](docs/R2_BACKUPS.md)).
- **Deploy:** `Dockerfile.upgrade`; Railway services start via `python -m jarvis.deploy api|worker`
  with `jarvis.deploy migrate` as pre-deploy ([.railway/railway.ts](.railway/railway.ts) is a
  whole-project plan; read its header before applying anything).

Start with the [documentation index](docs/README.md), [active TODO](docs/TODO.md)
and [web v1 PRD](docs/ERIDANI_WEB_V1_PRD.md). See [CI](docs/CI.md),
[app evals](evals/app/README.md) and the [latency plan](docs/ERI_LATENCY_PLAN.md)
for verification and response-speed work.
Assistant personality: [Dani](apps/api/jarvis/personality.py).

## Known limits

**The full suite runs in GitHub CI; the local check after each turn is quick.** Each agent run (mission control,
remotea) is checked after every turn by the target's verify entry in `~/.config/remotea/eridani.json`, run by
remotea in that run's worktree: for jarvis the `quick` entry, `./scripts/verify-quick.sh` (ruff, the tests related
to the change, `tsc` and `vitest related`; about 1.5 min). A run whose criteria are met on it then waits for green
GitHub CI on its pull request head (`awaiting_ci`, polled through the host's `gh` every 2 min, `needs_you` after
90 min); red CI goes back to the agent with the failing jobs and log. `./scripts/verify.sh` (the full suite) stays
for an explicit local run, as a second `full` entry. What the code does today for local verification (remotea
`verify-attempts.ts`, `run_coordinator.start_verify`):

- **No queue, no lanes.** remotea runs one verify attempt per agent session at a time (a newer turn cancels it),
  and runs different sessions' attempts in parallel; Eridani does not serialize them either. Runs in the same repo
  therefore verify at the same time and share the machine's CPU and the one standing test Postgres (each pytest
  session creates its own database on it). The quick check keeps that load small; the full suite takes 470–670 s
  alone and has run past 30 min under load when it ran after every turn. A run's verification shows how many other
  runs are verifying on the same host (`verification.alongside`).
- **Ceiling vs deadline.** The hard ceiling is the entry's `timeout` (remotea kills the command; 600 s for the
  `quick` entry, 7200 s for a `full` one).
  Eridani waits for the result until that timeout as remotea advertises it (`timeout_s`) plus 2 min, never less
  than 11 min, and only 11 min when the host advertises no timeout. A result that arrives after Eridani stopped
  waiting is still recorded, and a pass un-parks the run. One notice goes out when a verification passes 30 min.
- **A hung verification holds its run until the ceiling.** It does not block other runs, but it keeps loading the
  machine they verify on.

- **CI needs the host's `gh`.** Eridani has no GitHub credential; CI status comes from the run's host (`ci.status`,
  remotea running `gh` with its own login). A host without that capability parks a quick-check run in
  `needs_you(ci_unavailable)`.

This is the constraint to engineer around for swarm-style parallel fixes: per-run worktrees exist and the per-turn
check is now per-change; to scale further, add a cap on concurrent verifications per host (or a queue with visible
"waiting for run X") and more verification machines (lanes). Details: [agent control](docs/AGENT_CONTROL.md) ("An
execution target with verification", "CI as the final verification"), [background work](docs/BACKGROUND_WORK_PRD.md).

## Local development and tests

Use a disposable/development database only. Never point local tooling at production
or at a restored production copy with the worker or external services enabled.

```sh
uv sync --locked --group dev
# Backend tests need a throwaway PostgreSQL 16; fixtures create/remove their own databases.
JARVIS_ENV_FILE="" JARVIS_DATABASE_URL=postgresql+psycopg://USER:PASS@127.0.0.1:PORT/postgres \
  uv run --no-sync pytest -q
uv run --no-sync python -m pytest -q evals          # offline eval harness tests
cd apps/web && npm ci && npm test && npm run build   # frontend
npm run dev                                          # Vite on 127.0.0.1, proxies /api to :8765
```

`compose.upgrade.yml` is kept for local development and restore drills and is safe by
default: `docker compose -f compose.upgrade.yml up` starts only PostgreSQL, migrations and
the API, with `JARVIS_WORKER_ENABLED` and `JARVIS_EXTERNAL_SERVICES_ENABLED` forced to
`false` and no automatic restarts. The worker (reminders, push, Google/Linear sync) and the
local backup job sit behind compose profiles and an explicit opt-in:

```sh
ERIDANI_LOCAL_LIVE_WRITER=true docker compose -f compose.upgrade.yml --profile worker up   # own local data only
docker compose -f compose.upgrade.yml --profile backup up backup
```

## Historical prototype

The retired local GPU/Mem0/Qdrant prototype is documented in the
[archive](docs/archive/2026-10-02/LEGACY_PROTOTYPE_README.md).
Its startup instructions do not apply to the current Eridani deployment.

## License

MIT
