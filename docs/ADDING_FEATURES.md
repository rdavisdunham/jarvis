# Adding features to Eridani

A checklist for adding something to Eridani so it works with everything already on the site. Many
lists in this codebase are kept by hand in more than one file (the web app and the backend, or two
backend modules). A change that updates one copy works in isolation and breaks somewhere else. Find
the kind of change you are making below, touch every file listed, run the tests named, and run the
verification command at the end of the section.

Verified against `main` at `cb476cc` plus PR #76 (remotea device context), 2026-10-29. Code wins over
this document; when they disagree, fix this document in the same PR.

## Commands used below

```sh
# Backend tests need PostgreSQL. A throwaway container (pick a free port; 54399 is often taken):
docker run -d --rm --name eridani-test-pg -e POSTGRES_PASSWORD=test -e POSTGRES_DB=jarvis \
  -p 127.0.0.1:54399:5432 postgres:16-alpine
export JARVIS_ENV_FILE="" JARVIS_DATABASE_URL=postgresql+psycopg://postgres:test@127.0.0.1:54399/jarvis
uv sync --locked --group dev                      # once per checkout or worktree
uv run --no-sync python -m pytest -q tests/test_drift_guards.py tests/<the files you touched>
docker stop eridani-test-pg                       # --rm deletes it

# Lint exactly as CI does
uv run --no-sync ruff check --select E9,F63,F7,F82,F811 apps/api/jarvis tests scripts/validate_custom_planner.py scripts/validate_google_rsvp.py

# Surface inventory: after adding/removing an HTTP route, a domain command or a READ_TOOLS entry
PYTHONPATH=apps/api uv run --no-sync python -m scripts.app_eval.inventory
uv run --no-sync python -m pytest -q evals/test_app_eval.py -k surface_inventory

# Migrations (against the throwaway database only)
uv run --no-sync alembic upgrade head && uv run --no-sync alembic check

# Web
cd apps/web && npx vitest run && npx tsc -b
```

`JARVIS_ENV_FILE=""` stops `config.py` from loading a local `.env`. The test fixture
(`tests/conftest.py` `test_database`) creates and drops its own `jarvis_test_<hex>` database inside
the server you point at, and builds tables from the models with `Base.metadata.create_all`, not from
migrations. CI (`.github/workflows/ci.yml`) runs: the ruff selection above, the eval catalog
validation and `pytest evals`, `alembic upgrade head` + `alembic check` on an empty database, the
full backend suite, `npm test` + `npm run build` (which is `tsc -b && vite build`), the Playwright
fixtures listed in `scripts/validate_custom_planner.py` (`custom-planner.mjs`, `shell-navigation.mjs`,
`organization*.mjs` and others; the other `apps/web/e2e/*.mjs` scripts are manual), and a security
job (pip-audit, npm audit, gitleaks).

---

## 1. A new page (view) in the web app

The page name is a value of the web `View` type, and the browser reports the current page to the
backend on every `POST /api/v1/ui/sync` (`App.tsx`, `sendChat` awaits it before `POST /work`). The
backend validates it, and Eri's site-control tools emit page names the browser validates in turn.
Every list below must agree, or chat from the new page fails before Eri sees it.

Web (`apps/web/src/`):

- [ ] `types.ts` `View` union.
- [ ] `App.tsx` `nav` (sidebar; `tab: true` also puts it in the phone `tabbar`) or a separate entry
      like `remoteaNav` when it is gated; `titles: Record<View, string>` (the compiler enforces this
      one); the page body (`{view === "…" && …}`); and the `page-search` exclusion list near
      `className="page-search"` if the page has no search.
- [ ] `task-presets.ts` `initialView`: the `?view=` deep-link allowlist. Not type-checked; a missing
      name silently opens Today.
- [ ] `site-actions.ts` `views`: the zod enum for Eri's site controls. It must equal
      `ui_contracts.VIEWS`: Eri may name any page in `ui_show`/`ui_workspace`/`ui_search`, and a view
      missing here throws in `copilot.tsx` (`actionSchema.parse`). `remotea` was missing until the
      fix-remotea-wiring branch.
- [ ] Lazy chunk: large pages go in `viewChunks` with `lazy()` + `<Suspense>` (as `Remotea` does).
- [ ] Extra deep-link parameters (for example `?view=remotea&run=<id>`) are parsed inside the page
      (`remotea-model.ts` `linkedRun`, validated as a UUID). The URL sync block in `App.tsx`
      (`navigationUrl.searchParams…`) sets/deletes only the parameters it knows; add yours there if
      back/forward should preserve it. In-app paths to the same place (a task row's Run, Eri's
      `ui_show`, the Notifications page) go through one helper, as `App.tsx` `openRemoteaRun` does:
      it opens the run in the shell's side panel (`RunPanel.tsx`, state in `remotea-context.tsx`) on
      whatever page is open; `?run=` is kept in the URL by that same sync block.
- [ ] If Eri may open a record on the page, `App.tsx`'s `show` branch (the `action.entity_id` page
      list and an `else if (action.view === "…")` that fetches the record first) and the model map in
      `tools.py` `call_tool` (below).
- [ ] Styling: a page stylesheet imported by the page component (`import "./remotea.css"`), using
      only tokens from `theme.css` (`--paper`, `--surface`, `--ink*`, `--violet*`, `--line*`, …; see
      `docs/DESIGN.md`). No hex colors outside `theme.css`; then dark mode works for free. A new token
      must be defined three times in `theme.css`: `:root`, `:root[data-theme="dark"]`, and
      `@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) }`.
- [ ] Mobile: breakpoints are `ux.tsx` `COMPACT_MAX_WIDTH` 1000 / `PHONE_MAX_WIDTH` 700 /
      `NARROW_MAX_WIDTH` 600 (mirrored in a comment in `style.css`). Check the page at 390 px wide;
      `apps/web/e2e/shell-navigation.mjs` is the CI fixture for shell navigation.
- [ ] PWA: `public/sw.js` caches nothing and only opens URLs starting with `/?`; nothing to change for
      a page. `public/manifest.webmanifest` needs no change.

Backend (`apps/api/jarvis/`):

- [ ] `ui_control.py` `View` literal (validates `UIContext.view` in `UISync`).
- [ ] `ui_contracts.py` `VIEWS` (feeds `ui_workspace` and, via `tools.py`, `ui_search`).
- [ ] `tools.py`: nothing for the page itself. `READ_TOOLS["ui_show"]`'s `view` enum is set from `VIEWS`
      (after `READ_TOOLS.update(UI_TOOLS)`) and `call_tool`'s `ui_show` branch checks that enum. For
      `entity_id` support add the page's model to the `model` map there (`"remotea": AgentRun`) and say
      what the ID is in the `entity_id` description; a gated page checks its gate there (remotea calls
      `agent_link.gate`). `ui_filter`'s `view` enum is a deliberate subset; leave it unless the page shows
      tasks.
- [ ] If the page has its own settings section, see section 8.
- [ ] Personal-only pages: add the API prefix to `access.py` `PERSONAL_PATHS` (as `/api/v1/remotea`).

Tests: `tests/test_drift_guards.py` (`test_web_pages_match_backend_views`,
`test_browser_accepts_the_views_eri_can_send`, `test_ui_show_opens_every_page_in_its_enum`),
`tests/test_ui_control.py` (`test_every_web_view_is_a_valid_device_context`,
`test_ui_show_opens_remotea_and_one_of_your_runs`), `apps/web/src/site-actions.test.ts`,
`apps/web/src/calendar-tasks.test.ts` (`initialView`).

Verify: `pytest -q tests/test_drift_guards.py tests/test_ui_control.py` and
`cd apps/web && npx vitest run && npx tsc -b`; then open the page and send Eri a chat message from it.

## 2. A new HTTP route

- [ ] Put it in a feature module with `router = APIRouter(prefix="/api/v1/…")` (see
      `remotea_routes.py`) and register it in `api.py` with `app.include_router(...)`. Registration
      must come **before** the SPA catch-all `@app.get("/{path:path}")` at the bottom of `api.py`, or
      GETs fall through to `index.html`.
- [ ] Pick the auth style:
  - Browser (cookie): `User = Annotated[Identity, Depends(authenticate)]` (`auth.py`). Non-GET
    requests must carry `X-CSRF-Token` (checked in `auth._authenticate`); the web `api.ts`
    `post()` adds it. `authenticate_personal` resolves the signed-in account's personal namespace
    instead of the active workspace and skips `request_access`.
  - Bot key: put it on the `external` router in `external_routes.py` (prefix `/api/v1/external`,
    dependency `Bot = Depends(external_identity)`), then call
    `bot_access.authorize(db, required="<scope>")` inside, with `write=True` for mutations. Bot
    routes need no CSRF; the `security` middleware still rejects foreign `Origin` and
    `sec-fetch-site: cross-site` and re-authenticates the key after the handler.
  - remotea link: the websocket in `agent_link.py` (`PATH = "/api/v1/external/agents/link"`,
    `agents:control` key). Not a pattern for new routes.
- [ ] Workspace rules (`access.py` `request_access`): a route that must not run in a shared workspace
      goes in `PERSONAL_PATHS`; a POST that a **viewer** may call (read-equivalent) goes in the `reads`
      set, otherwise viewers get `READ_ONLY`. remotea routes additionally call
      `remotea_routes.gated()` (`owner_only` + `agent_link.gate` → `REMOTEA_DISABLED`).
- [ ] Middleware facts (`api.py` `security`): bodies over 100,000 bytes are refused (413);
      `maintenance_mode` returns 503 for `/api/`; every `/api/v1/` response is re-authenticated after
      the handler except the prefixes listed there; `Cache-Control: no-store` on `/api/`.
- [ ] Rate limits are per feature, not global: bot keys 120/min (`bot_access.authenticate`), pairing
      login (`login_attempts`), client latency reports, Google sign-in, queued work (`QUEUE_FULL`),
      link frames in `agent_link.py`. Add one if the route is expensive or unauthenticated.
- [ ] Refresh the surface inventory (`scripts/app_eval/inventory.py`). It reads decorator paths that
      are string literals or module-level string constants (`@router.websocket(PATH)`); anything else
      (an f-string, an imported name) is skipped silently. `route_owner()` maps the path to an eval
      feature (`evals/app/features.json`) and raises `New API route needs a feature owner` for an
      unknown prefix; add a branch there. The owner must be a feature in `features.json` (guard:
      `evals/test_app_eval.py::test_every_surface_owner_is_a_catalog_feature`); a new feature also
      needs 25-50 scenarios in `evals/app/features/<id>.json`, a type in `scripts/app_eval/registry.py`
      `GROUPS`, the catalog counts in `evals/test_app_eval.py` / `evals/test_campaign.py`, and a
      regenerated `evals/app/automation-coverage.json` (`docs/CI.md`). Then regenerate
      `evals/app/surface-inventory.json` and commit it. Guard:
      `evals/test_app_eval.py::test_surface_inventory_matches_current_app`.
- [ ] Docs: there is no checked-in OpenAPI spec (FastAPI's defaults are not overridden). Routes are
      documented by hand: bot routes in `docs/EXTERNAL_AGENTS.md`, remotea routes in the "Browser
      routes" tables of `docs/AGENT_CONTROL.md`.
- [ ] Tests: cookie routes use the `client` fixture (`tests/conftest.py`, logged in with CSRF set).
      Add a negative test for the auth style (no key, wrong scope, viewer, shared workspace).

Verify: refresh the inventory, then `pytest -q evals/test_app_eval.py -k surface_inventory tests/<your test>`.

## 3. A new bot/MCP tool or bot scope

Scopes (`bot_access.py`):

- [ ] `SCOPES` (name → description; `GET /api/v1/bot-keys` returns this dict).
- [ ] `PRIVILEGED` if every use is a write check and viewers/shared workspaces must not hold it
      (today `agents:control`, `agents:session`; `agents:read` is not privileged).
- [ ] `COMMAND_SCOPES` (domain command → scope) for direct commands; `READ_SCOPES` (read tool →
      scope) for read tools. `tool_allowed()` uses both to filter Eri's registry for bot-queued work
      (`external_service.py`, `registry()` filtered by `tool_allowed`).
- [ ] `bot_access.create`: viewers are refused `:write`, `work:run` and `PRIVILEGED`; `schema:write`
      needs the owner; `PRIVILEGED` is refused in shared workspaces. Extend if the new scope needs a
      rule.

MCP and HTTP (`external_mcp.py`, `external_routes.py`):

- [ ] `external_mcp.definitions(scopes)`: add the tool under an `if "<scope>" in scopes:` block.
      Command tools come from `COMMAND_SCOPES` automatically; custom-record reads come from the tuple
      list inside `definitions` (which mirrors `external_service.CUSTOM_READS` and `READ_SCOPES`).
- [ ] `external_mcp.dispatch(name, arguments)`: handle the name and call
      `bot_access.authorize(db, bot.owner_id, "<scope>")` again (with `write=True` for mutations).
      `dispatch` first re-derives `definitions(bot.scopes)` and validates `arguments` against the
      tool's JSON schema.
- [ ] HTTP twin on the `external` router in `external_routes.py` (see `agent_runs` / `agent_run` for
      `run_list` / `run_get`), then refresh the surface inventory (section 2).
- [ ] The MCP app is mounted at `/api/v1/external/mcp` with `streamable_http_path="/"`, so only the
      trailing-slash URL `/api/v1/external/mcp/` works; the no-slash path answers 405. Any doc or UI
      that prints the URL must use the slash (`external_routes.keys` returns `mcp_url` with it).

Web:

- [ ] `apps/web/src/BotSettings.tsx`: the permission groups array (`[["records", …], …]`) or a
      checkbox feeding `permissionScopes(levels, queued, runTools, runRead)`; and `scopeLabel` for a
      non-`group:level` scope. Update `BotSettings.test.tsx`.

Docs and tests:

- [ ] `docs/EXTERNAL_AGENTS.md` (Permissions, API operations, MCP tool names) and the Scopes table in
      `docs/AGENT_CONTROL.md` for `agents:*`.
- [ ] `tests/test_external_agents.py` (HTTP and MCP behavior), `tests/test_agent_security.py`
      (privilege, shared workspace, viewer), `tests/test_agent_voice.py::test_external_bots_never_get_control_tools`.

Guards: `tests/test_drift_guards.py::test_settings_can_grant_every_bot_scope_but_the_link_key`
(every `SCOPES` entry except `agents:control` must be grantable in Settings) and
`::test_external_record_kinds_and_mcp_reads_agree`.

Verify: `pytest -q tests/test_drift_guards.py tests/test_external_agents.py tests/test_agent_security.py`,
`cd apps/web && npx vitest run src/BotSettings.test.tsx`, and
`curl -s -H "Authorization: Bearer $KEY" https://<host>/api/v1/external/capabilities | jq '.tools[].name'`.

## 4. A new Eri tool (chat and voice)

Eri's tool list is `tools.registry()`: every `READ_TOOLS` entry plus every command in
`VOICE_MUTATIONS` (name `a.b` becomes tool `a_b`). Chat (`conversation.py`), the voice controller
(`voice.py`) and queued work (`work_runner.py`) each wrap it in `tool_policy.remotea_tools(...)`;
GPT-Live (`live_voice.py`) delegates to queued work.

- [ ] Read tool: add to `READ_TOOLS` in `tools.py` (description + JSON schema) and implement it in
      `call_tool`. Validate any browser-bound arguments with `jsonschema` the way the `ui_*` branch does.
      Return plain JSON: timestamps as ISO strings (`agent_sessions.stamp`), never `datetime`, `UUID` or model
      objects. HTTP routes and MCP pass results through FastAPI's `jsonable_encoder`, so a stray `datetime`
      only surfaced when Eri called the tool in chat: `json.dumps` of the result raised and the whole request
      finished `failed` ("The task model could not finish this request", `session_search`, 2026-10-29). The
      runners now encode through `tools.tool_json`, but the fix belongs in the tool.
- [ ] Mutation: add the domain command (section 5), then add its name to `VOICE_MUTATIONS` (many are
      added by prefix: `planning.`, `linear.`, `space.`, …, `control.`). Per-field null rules go in
      `domain.NOT_NULL`.
- [ ] Tool group and discovery (`tool_catalog.py`): put the tool name in a `GROUPS` entry (or `CORE`
      if always loaded); a richer description goes in `DESCRIPTIONS`. Guard:
      `tests/test_tool_refinement.py` asserts every registry tool is in a group or `CORE`, and each group
      has fewer than 10 tools.
- [ ] Policy (`tool_policy.py`): `lean-v1` moves tools via `LEAN_DEFERRED` / `LEAN_EXTRA`. Anything
      that drives remotea goes in `CONTROL_TOOLS` so `remotea_tools()` drops it while remotea is off or
      no human is present (`live_work`).
- [ ] Shared workspaces (`access.py`): personal-only tools must match `tool_access`'s underscore
      prefixes (`review_`, `memory_`, `control_`, …) and, for commands, `PERSONAL_TOOLS` (dotted).
- [ ] Bot exposure: a tool reaches bot-queued work only if `bot_access.READ_SCOPES` /
      `COMMAND_SCOPES` maps it; leave it out to keep it Eri-only.
- [ ] Consequential actions (control.start/control.message) are two-phase: `agent_control.propose`
      writes an `AgentPendingControl` row and returns `pending_confirmation`; only `control_confirm`
      after the owner's own yes (or the remotea page's Confirm) executes it. Reuse this for new
      actions that must not run on the model's say-so.
- [ ] Surface inventory: a new name prefix needs an owner in `scripts/app_eval/inventory.py` `PREFIX`
      (or `EXACT`); `owner()` raises `New exposed tool needs a feature owner`. Regenerate the inventory.
- [ ] Evals: add scenarios to the feature in `evals/app/features/` and bindings in
      `evals/app/bindings.json` where automatable (`evals/app/README.md`). The coverage ratchet
      (`runner coverage --baseline evals/app/automation-coverage.json`) only fails if coverage drops.

Tests: `tests/test_agent_voice.py::test_control_tools_are_registered_described_and_grouped`,
`tests/test_tool_refinement.py`, `tests/test_drift_guards.py::test_every_control_tool_is_gated_on_remotea`.

Verify: `pytest -q tests/test_tool_refinement.py tests/test_drift_guards.py` and refresh the inventory.

## 5. A new domain command

- [ ] Pydantic args model and registration in `domain.COMMANDS` (feature modules export a dict merged
      at the bottom of `domain.py`, for example `COMMANDS.update(CONTROL_COMMANDS)`).
- [ ] Route it in `domain.mutate` (prefix dispatch; `control.` → `agent_control.mutate`, `record.` →
      `structure.mutate`, …).
- [ ] Locks in `domain.execute`: commands that touch the owner graph must match the prefix tuple that
      takes `advisory(db, f"workspace:{owner}")`; `memory.*` also takes `memory:{owner}`.
- [ ] Task changes go through the `task.update` branch, which calls, in order, `agent_actor.gate`
      (the run completion gate: while a task has an unsettled run only `human`, `source_sync` or
      `run_judge` may complete it or change `GATE_FIELDS`) and `linear_sync.before_task_update`
      (mapped fields `title`, `notes`, `due_date`, `priority`, `status`, `assignee`, `project_id`
      queue a Linear write; a linked task Linear refuses raises). Do not write `Task` columns directly
      from a new command and skip these.
- [ ] Note changes go through `notes.mutate_note` (or call `agent_actor.note_gate` on the locked note
      first, as `note.file`/`note.organize` do): a task's PRD note is read-only to everyone but a
      human while the task has an unsettled run (`RUN_OWNS_PRD`).
- [ ] Events: `domain.emit(db, owner, "<kind>", entity_id, revision)` (section 7). `emit` also queues
      search indexing for the kinds listed in its set.
- [ ] Receipts and revert: `execute` wraps the mutation in `action_history.journal`, which records an
      `ActionChange` for objects whose model is in `action_history.MODELS`. A new record type needs a
      `MODELS` entry to appear in Activity with Revert; the web `Activity.tsx` `editable` set must also
      know the kind. `journal` skips `quicksort.` and `control.` on purpose.
- [ ] Access: `access.command_access` runs first. Viewers are refused; `PERSONAL_TOOLS` prefixes are
      refused in shared workspaces; bots go through `bot_access.check_command` (`COMMAND_SCOPES`).
- [ ] Exposure: Eri (`VOICE_MUTATIONS`, section 4), bots (`COMMAND_SCOPES`, section 3), the browser
      (`POST /api/v1/commands` with `command_id`), and the surface inventory (a new command appears in
      `commands`; refresh it).

Tests: `tests/test_domain.py`, `tests/test_agent_actor.py` (gate), `tests/test_creation_revert.py`
(revert), `tests/test_linear.py` (Linear interplay).

## 6. A new DB column or table

- [ ] Model in `models.py`, or in `session_models.py` / `structure_models.py` / `note_list_models.py` /
      `search_models.py`, which `models.py` imports at the bottom so Alembic and `create_all` see them.
      A new model module must be added to that import block.
- [ ] Migration `migrations/versions/00NN_<name>.py` with `revision = "00NN_<name>"` and
      `down_revision` = the previous file's `revision` (head is `0035_action_receipts`). Two PRs that both
      add `0033_*` create two heads; `alembic upgrade head` in CI then fails. Rebase and renumber.
- [ ] Rolling deploys: API and worker run `jarvis.deploy migrate` pre-deploy and the previous release
      keeps running briefly (`docs/CLOUD_MIGRATION.md`). New non-null columns need a
      `server_default` so the old writer's inserts still work; destructive changes need their own
      plan. `alembic check` does not compare server defaults (no `compare_server_default` in
      `migrations/env.py`), so it will not catch a missing one.
- [ ] JSONB: `mapped_column(JSONB, default=dict, server_default="{}")` (see `Notification.target`).
      Mutate by reassigning (`row.values = {**row.values, key: v}`), never in place; SQLAlchemy does
      not track in-place JSONB edits.
- [ ] Scoping: `owner_id: String(100), index=True` is the workspace namespace (personal account or
      shared workspace); `account_id` is the person. Read through `domain.owned(db, Model, id, owner)`.
- [ ] Privacy and lifecycle: `api.py` `export` is a hand-written model list; decide whether the new
      table belongs in the account export. Agent tables have retention in `agent_retention.py`.
- [ ] `tests/test_release_migrations.py` holds per-migration upgrade/downgrade tests with real rows;
      add one when the migration transforms data.

Verify: `alembic upgrade head && alembic check` against the throwaway database, then
`pytest -q tests/test_release_migrations.py`.

## 7. A new notification, push category or change-feed event

Notifications (`models.Notification`): `category` is a free string (no enum or check constraint).

- [ ] Create rows with a unique `dedup_key`, `scheduled_at`/`eligible_at`, `importance`
      (`"urgent"` bypasses quiet hours) and a `target` dict. Existing creators: `notices.py`
      (`deadline`, `morning`, `question`, `failure`), `domain.py` (`reminder` occurrences),
      `run_coordinator.notify` (`agent_question`, `agent_done`, `agent_stuck`; capped at
      `PUSHES_PER_RUN` = 3 per 10 minutes per run; `agent_quiet_bypass` makes questions urgent).
- [ ] Delivery rules in `notices.eligible`: `work_result` is never pushed; nothing in shared
      workspaces; quiet hours via `quiet_until`; task-linked notices drop when the task is done.
- [ ] Deep link: `worker.notification_url(notification)` maps `target` to a `/?…` URL (remotea →
      `/?view=remotea&run=<id>`). `public/sw.js` opens only URLs starting with `/?`. The page must parse
      the parameter (section 1). The in-app Notifications page's "Open" (`notice-primary` in `App.tsx`)
      goes through `apps/web/src/notice-target.ts` `noticeDestination`, which mirrors
      `notification_url` in the same order. A new `target.view` needs a branch in both and a row in
      `apps/web/src/notice-targets.json` (notification, push URL, in-app destination); the drift guard
      fails if a `target={"view": …}` in `notices.py` / `run_coordinator.py` has no row.
- [ ] `detailed_notifications` (preference, default off) replaces title/body with a generic text in
      the push payload; write titles that still make sense in-app only.
- [ ] Tests: `tests/test_smart_notifications.py`, `tests/test_push_registration.py`,
      `apps/web/src/push-subscription.test.ts`, `apps/web/src/notice-target.test.ts` and
      `tests/test_drift_guards.py::test_push_and_in_app_notifications_open_the_same_place` (both read
      `notice-targets.json`).

Change feed (SSE `GET /api/v1/events`):

- [ ] Emit with `domain.emit(db, owner, "<thing>.changed", entity_id, revision)` inside the transaction.
- [ ] Web `events.ts`: kinds that only one page cares about go in `LOAD_SKIP` (no workspace reload)
      and/or `REVISION_SKIP` (no dependent view refresh); page-specific kinds dispatch a window event
      (`agent.changed`/`session.changed` → `eri-agents-changed`, `work.changed` → `eri-work-changed`).
      An unknown kind triggers a full reload, which is safe but costly for chatty events. Test:
      `apps/web/src/events.test.ts` (`signalScope`).
- [ ] Bots see only an allowlist (`external_service.changes`: `<kind>.changed` for permitted record
      kinds, plus `record.changed` / `structure.changed`), so a new kind does not leak to bot keys.

## 8. A new setting or feature gate

Two patterns exist; pick deliberately.

- Account preference Eri may change: default in `domain.preferences()`, a field on
  `domain.SettingsUpdate` (validated), written by the `settings.update` command (which Eri has via
  `VOICE_MUTATIONS`). Shared workspaces override some values in `access.shared_preferences`. The web
  reads `boot.preferences` (`types.ts` `Preferences`) and edits through the `settings.update` command
  (for example `PlannerPreferences.tsx` `NotificationPreferences`).
- Owner-only gate Eri must not flip (remotea): a dedicated route writes `OwnerSettings.values`
  (`remotea_routes.set_enabled` for `remotea_enabled`, `remotea_auto_complete`, `agent_quiet_bypass`),
  **not** `SettingsUpdate`. Readers: `agent_link.enabled` / `gate`, `run_coordinator.auto_complete` /
  `quiet_bypass`. `remotea_enabled` also has a default in `preferences()` so the web sees it in
  `boot.preferences`; the other two are read only through their helpers.

Checklist:

- [ ] Every surface that the gate hides: web nav (`remoteaNav` shown only when
      `boot.preferences.remotea_enabled`), page body, the Run controls (`RunButton.tsx` on task rows,
      `Remotea.tsx` `RunTask` in the detail dialogs, the topbar Runs button and `RunPanel.tsx`; all read
      `remotea-context.tsx`, which App feeds from the same preference), backend routes (`gated()`), Eri tools
      (`tool_policy.remotea_tools`), the dream (`agent_autolink.py` checks `remotea_enabled`).
- [ ] Emit `settings.changed` after writing so other tabs reload.
- [ ] Settings page placement: `settings-sections.ts` (8 sections). A new section must also be added
      to `ui_control.UIContext.settings_section`, `ui_contracts.py` `ui_workspace` `settings_section`
      enum and `site-actions.ts` `settings_section` (guard:
      `tests/test_drift_guards.py::test_settings_sections_match_everywhere`), and to the phrase
      routing in `App.tsx` (`target === "settings"` regex chain).

## 9. A new run-loop state or reason (remotea)

- [ ] `run_coordinator.TRANSITIONS`: add the state as a key with its legal targets and add it to the
      target sets of states that can reach it. `STATES` derives from it. Terminal states live in
      `session_models.TERMINAL_RUN_STATES`.
- [ ] `run_coordinator.transition` sends notifications on entering `needs_you`, `awaiting_approval`,
      `capped` and `awaiting_integration`; decide whether the new state notifies.
- [ ] Reason text: `run_coordinator.REASONS` (shown on the run page via `agent_control.REASON_TEXT`,
      in push bodies and in `control_status`). Every `needs_you` reason needs an entry.
- [ ] Web `remotea-model.ts`: `RunState` union, `RUN_LABELS`, `TERMINAL_RUN_STATES`, and the hard-coded
      state lists in `orderRuns` (attention rank), `runControls` (`replan` states) and `timeline` tone.
      `Remotea.tsx` checks `state_reason === "plan_review"` literally.
- [ ] `docs/AGENT_CONTROL.md` lifecycle section.

Tests: `tests/test_agent_runs.py::test_every_listed_transition_is_legal_and_others_are_refused`,
`tests/test_drift_guards.py::test_web_run_states_match_the_coordinator` and
`::test_every_needs_you_reason_has_text`, `apps/web/src/remotea-runs.test.ts`.

## 10. A new background job kind

- [ ] `worker.perform_job`: dispatch on `job.kind`.
- [ ] `worker` dispatcher's `queue_name` chain: a kind not listed lands on the default `jarvis` queue;
      memory-like work belongs in `jarvis-memory`, provider calls in their provider queue.

## 11. Docs and process

| Doc | Mirrors | Update when |
|---|---|---|
| `docs/TODO.md` | The Eridani roadmap task list in the app (project "Eridani roadmap"), which is the source of truth since 2026-10-29 | Add new work in the app first, then mirror it here; close items in both |
| `docs/EXTERNAL_AGENTS.md` | `bot_access.SCOPES`, `external_routes.py`, `external_mcp.definitions`, the MCP URL | Any bot scope, route or MCP tool |
| `docs/AGENT_CONTROL.md` | `agents:*` scopes, `/api/v1/remotea` routes, link frames, run states, the agent migrations | Any remotea change |
| `tests/fixtures/protocol/<caps_version>/` | The link frames each side actually sends (keys and types), recorded by `tests/test_protocol_e2e.py` against a real remotea server | A link frame or field changes on either side: run that test locally (it skips in CI), re-record with `UPDATE_FIXTURES=1`, review the diff |
| `docs/DESIGN.md` | `apps/web/src/theme.css` tokens | New tokens or components |
| `docs/CI.md` | `.github/workflows/ci.yml` | CI steps or local equivalents change |
| `docs/TASK_CARDS.md` | Both task cards' sections and every entry point that opens each | A task-card section or a route into either card changes |
| `docs/APP_FUNCTIONALITY.md` | Feature map, dated September 19 | New user-facing feature |
| `evals/app/README.md`, `evals/app/surface-inventory.json` | Eval catalog and exposed surfaces | New feature, route, command or read tool |

- There is no `CLAUDE.md` or `AGENTS.md` in this repository; coding agents should be pointed at this
  file.
- CI runs on every pull request, including stacked ones; `main` requires the three CI checks with
  strict up-to-date branches (`docs/CI.md`).

---

## How the recent breaks happened

| Break | What was missed | Checklist line | Guard now |
|---|---|---|---|
| Chat from the remotea page failed validation (PR #76) | Web `View` gained `remotea`; `ui_control.View` and `ui_contracts.VIEWS` did not. `POST /ui/sync` 422'd and `sendChat` awaits it | Section 1, backend list | `test_web_pages_match_backend_views` fails on `cb476cc` |
| CI failed after two new `agents:read` routes (PR #73, fixed in #74) | `evals/app/surface-inventory.json` not regenerated | Section 2, surface inventory | Already guarded; the doc now says how to refresh |
| Settings could not grant `agents:session` (PR #72) | New scope in `bot_access.SCOPES` without a `BotSettings.tsx` control | Section 3, Web | `test_settings_can_grant_every_bot_scope_but_the_link_key` fails on the pre-#72 `BotSettings.tsx` |
| `claude mcp add …/mcp` got 405 | `streamable_http_path="/"` serves only `/api/v1/external/mcp/` | Section 3, MCP mount | None; a doc rule. PR #71 tracks accepting the no-slash path |
| Three docs edited by hand | Docs mirror code and the app task list | Section 11 table | None |

## Drift guards

Existing tests that pin a list in two places:

- `tests/test_ui_control.py::test_every_web_view_is_a_valid_device_context`: `ui_control.View` ==
  `ui_contracts.VIEWS` (PR #76).
- `evals/test_app_eval.py::test_surface_inventory_matches_current_app`: routes, commands and read
  tools vs `evals/app/surface-inventory.json`; `::test_every_surface_owner_is_a_catalog_feature`: every
  owner the inventory assigns is a feature in `evals/app/features.json`.
- `tests/test_tool_refinement.py`: registry == union of `tool_catalog.GROUPS` + `CORE`.
- `tests/test_agent_voice.py::test_control_tools_are_registered_described_and_grouped`: control
  commands, `CONTROL_TOOLS`, `DESCRIPTIONS`, `GROUPS["agents"]`.
- `tests/test_agent_runs.py::test_every_listed_transition_is_legal_and_others_are_refused`.
- `apps/web/src/events.test.ts` (`signalScope` skip lists), `BotSettings.test.tsx`
  (`permissionScopes`), `calendar-tasks.test.ts` (`initialView`).
- `apps/web/src/task-card-parity.test.tsx`: RecordCard and TaskDetails both render every section in
  `task-card-sections.ts` `TASK_CARD_SECTIONS` for one task. A section that belongs on both task cards
  goes in `TaskCardSections.tsx` and that constant; see `docs/TASK_CARDS.md`.
- CI `alembic check` (models vs migrations, without server defaults).

Added in `tests/test_drift_guards.py` (backend job; reads web sources as text):

| Test | Asserts |
|---|---|
| `test_web_pages_match_backend_views` | `types.ts` `View` == `ui_control.View` == `ui_contracts.VIEWS` == `task-presets.ts` `initialView` list |
| `test_browser_accepts_the_views_eri_can_send` | `site-actions.ts` `views` == `ui_show`/`ui_search`/`ui_workspace` enums == `VIEWS`; `ui_filter` ⊆ `VIEWS` |
| `test_ui_show_opens_every_page_in_its_enum` | `call_tool`'s `ui_show` branch checks the tool's enum (no copy) and opens every page in it |
| `test_push_and_in_app_notifications_open_the_same_place` | `worker.notification_url` gives each `notice-targets.json` row's URL, the URL and the in-app destination agree, every created `target.view` has a row, and `App.tsx` Open calls `noticeDestination` (`notice-target.test.ts` checks the web side on the same rows) |
| `test_agent_docs_name_real_settings_sections` | every "Settings → X" in `EXTERNAL_AGENTS.md` / `AGENT_CONTROL.md` starts with a `settings-sections.ts` label |
| `test_editor_kinds_match_the_browser` | `site-actions.ts` `editorKinds` == `ui_contracts.EDITOR_KINDS` ⊆ `EditorContext.kind` |
| `test_settings_sections_match_everywhere` | `settings-sections.ts` == `UIContext.settings_section` == `ui_workspace` enum == `site-actions.ts` enum |
| `test_web_run_states_match_the_coordinator` | `remotea-model.ts` `RunState` == `run_coordinator.STATES`; terminal lists equal |
| `test_every_needs_you_reason_has_text` | every `reason=` literal on a `needs_you` transition in `run_coordinator.py` is in `REASONS` |
| `test_every_control_tool_is_gated_on_remotea` | registry tools named `control_*` == `tool_policy.CONTROL_TOOLS` |
| `test_settings_can_grant_every_bot_scope_but_the_link_key` | scopes buildable in `BotSettings.tsx` == `bot_access.SCOPES` minus `agents:control` |
| `test_external_record_kinds_and_mcp_reads_agree` | `external_service.Kind` == `KINDS`; each `CUSTOM_READS` tool appears in `external_mcp.definitions` for its `READ_SCOPES` scope |

Worth adding (lists in two places with no test yet):

- `action_history.MODELS` keys vs `Activity.tsx` `editable`.
- `access.PERSONAL_TOOLS` (dotted) vs `access.tool_access` prefixes (underscored): one list.
- `external_mcp.definitions`' custom-read tuple vs `external_service.CUSTOM_READS` vs
  `bot_access.READ_SCOPES`: build the tuple from `READ_SCOPES`.
- `theme.css`: the two dark blocks define the same tokens as each other and every `:root` color token
  (a vitest reading the file).
- `migrations/versions/`: file prefix == `revision` prefix and `down_revision` is the previous number.
- `api.py` `security` CSP vs `apps/web/public/_headers` CSP (two copies that already differ in
  `media-src` / `connect-src`; check whether `_headers` is still served before syncing).

## Things that surprised me or look like latent drift

- Fixed on the fix-remotea-wiring branch: `remotea` was not a site-control page (`site-actions.ts`
  rejected it, `ui_show` could not open it); the in-app Notifications "Open" ignored
  `target.view == "remotea"`; the remotea link websocket was invisible to the surface inventory;
  `agent_control` had no entry in `evals/app/features.json` (it now has 25 unbound scenarios; the
  outcome eval stays in `evals/agent_control/`); `EXTERNAL_AGENTS.md` said "Settings → Integrations".
- `api.py` `export` omits structure records, record templates and agent session/run tables. I could not
  tell whether that is intentional.
- `bot_access.SCOPES` is served to the web (`GET /api/v1/bot-keys` → `scopes`) but `BotSettings.tsx`
  ignores it and hard-codes the groups; rendering from the server list would remove a mirror.
