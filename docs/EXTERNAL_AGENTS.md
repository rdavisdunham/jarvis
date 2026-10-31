# Connect agents to Eridani

## Create a connection

Open **Settings → Connections → Connected agents** in the intended workspace. Add an agent, choose its permissions and expiration, and copy the key once. It is stored only as a hash; losing it requires a new key. Revoke obsolete keys in the same panel.

Each key is tied to one workspace and the account that created it. Switching the browser workspace does not move the key. A shared-workspace membership downgrade limits writes immediately; leaving or being removed ends access. Workspace owners can revoke keys in that workspace; other members manage their own keys.

- API base: `https://app.eridani.app/api/v1/external`
- MCP endpoint: `https://app.eridani.app/api/v1/external/mcp/`. The form without the trailing slash (`.../mcp`) works the same way.
- Transport: Streamable HTTP, stateless JSON responses.
- Authentication header: `Authorization: Bearer <your bot key>`.
- Limit: 120 HTTP requests per minute per key, including MCP discovery. Retry 429 responses with a delay.

Use a secret store or environment variable for the key, not a prompt, URL, repository or browser storage. Requests need no browser cookie or CSRF token. Bot keys cannot authenticate to browser endpoints.

**Compatibility:** MCP clients must support a configured Bearer header. This release does not implement MCP OAuth discovery, client registration or a consent flow. Clients that only connect through OAuth need that later adapter; Eridani's Google website login is not an MCP authorization server.

## Permissions

`tasks:read/write`, `organization:read/write`, and `notes:read/write` independently control access. Organization includes spaces, areas, goals, projects and assignees. Write includes read. An optional `work:run` permission lets the client send natural-language requests to Eri; model usage is incurred only for those requests. It does not add record permissions.

A bot cannot access learned personal memories, ordinary conversations, account settings, browser controls, Google Calendar or Linear credentials/tools. Linked records are filtered so a project read does not reveal notes without note permission. Task `notes` is the task's own description and is included in task read access.

This is workspace-level access, not per-project or per-field sharing. Put separately restricted records in separate workspaces. Bots cannot delegate permissions, mint keys or change their own grants. Rotate by creating a replacement and revoking the old key; existing queued work remains tied to the original key.

## Direct operations

Direct API and MCP calls use the same domain commands as the site; no interpretation model or review step runs. Writes commit their record, change event, undo journal and Activity item in one transaction. Activity identifies the bot and exposes Edit/Revert where supported.

API operations:

- `GET /capabilities`: current scopes and complete, typed tool schemas.
- `GET /records/{kind}`: search `task`, `note`, `project`, `goal`, `space`, `area`, or `actor`.
- `GET /records/{kind}/{id}`: current details, links and revision.
- `POST /commands`: `{ "request_id": "UUID", "tool": "task.create", "arguments": { "title": "Call Alex" } }`.
- `GET /changes?after=CURSOR&limit=100`: incremental record changes.
- `POST /actions/{action_id}/revert`: `{ "request_id": "UUID" }` reverses this bot's own supported action.
- `GET /bug-reports?status=open|fixed|dismissed&task_id=&limit=&offset=`, `GET /bug-reports/{id}`, `GET /bug-reports/{id}/screenshot` (scope `tasks:read`): reports from the web app's Report a bug button. See "Bug reports" below.

Queries accept `query`, `archived`, `limit` (1–100), `offset`, and applicable exact filters: `status`, `space_id`, `area_id`, `project_id`, `goal_id`, `task_id` (linked notes), `parent_task_id`, `assignee_id`, `work_type`, `due_from`, `due_through`. Dates use `YYYY-MM-DD`. Queries are literal keyword matches, not semantic/vector searches. Pagination returns `next_offset`. Unsupported filters are rejected.

Commands include task create/update/complete/reopen; goal/project/space/area/actor create/update; and note create/update/append/replace. Archive with the corresponding update command. Prefer stable IDs for organization and assignments. Supplying task project/assignee names can create organization entries and therefore requires organization write access. Due dates, times, tags, parents, goal/project links and note links use the existing site schemas returned by `/capabilities`.

MCP exposes `records_list`, `records_get`, `changes_list`, `request_get`, the permitted command names such as `task_create`, and supported undo/queued-work tools. Command arguments are flattened, with an additional required `request_id` UUID.

For an edit, read the record first and include `expected_revision`. Omit unchanged fields. An explicit null clears only fields whose contract allows it. Replacement link lists must preserve links you still want. HTTP 409 means the revision or idempotency request conflicts; fetch the current record and decide the next action instead of retrying with a guessed revision.

Two 409s are not revision conflicts and must not be retried while the run lasts: `RUN_OWNS_COMPLETION` (completing, or editing success criteria, verify or turn cap of, a task with an unsettled agent run) and `RUN_OWNS_PRD` (note update/append/replace, archiving, or reverting a change to a note that is the PRD of a task with an unsettled run). Both carry `data.run_id`; only the owner can make those changes until the run settles. A note record's `frozen_run_id` names the run that froze it, `null` otherwise. See `docs/AGENT_CONTROL.md`, "Actors and the completion gate".

## Retry and outcome rules

Generate one request UUID per intended operation and persist it with the exact arguments. Reuse it unchanged after a timeout or disconnect. The server namespaces it to the bot key; REST and MCP retries share the same receipt. Reusing it for different instructions returns 409. A rejected command saves no partial changes or misleading success card.

The returned `request_id` / Activity `id` is the server's durable work ID. Use that returned ID for `/requests/{id}` or MCP `request_get`. The request UUID supplied by the caller is the idempotency key, not the status URL ID.

Independent operations can run concurrently. Writes still enforce current revisions and short graph transaction locks. Creating the same human idea with two different request UUIDs can create two records: idempotency is based on operation identity, not fuzzy title matching.

Revert is a compensating command, not deletion of history. Creating a record is reversed by archiving when no later changes or links prevent it. Field-level undo preserves unrelated later edits and refuses to overwrite a later change to the same field. Relationship rewrites and external provider changes may require manual edits. A bot can only revert its own actions; its creator can use the site controls even after the key is revoked.

## Bug reports

The web app's **Report a bug** button (bottom-left on every page, Alt+Shift+B) files a report and a task tagged
`bug-report` in the project "Eridani bug reports". A key with `tasks:read` reads both: the task through
`/records/task/{id}` like any task, and the report through the routes below (no separate scope; a report
is part of the task it opened). They are HTTP only; there is no MCP tool.

- `GET /api/v1/external/bug-reports`: newest first, `{items, total}`. Each item: `id`, `title` (the
  description's first line), `status`, `created_at`, `task_id`, `task_status`, `view`, `screenshot`
  (`{mime, bytes, url}` or null), `screenshot_error`, `links`. Filters: `status` (`open`, `fixed`,
  `dismissed`), `task_id`, `limit` (1–200, default 50), `offset`.
- `GET /api/v1/external/bug-reports/{id}`: the same plus `description`, `expected`, `context` (below) and
  `summary`, the readable digest that is also in the task's notes.
- `GET /api/v1/external/bug-reports/{id}/screenshot`: the image bytes (`image/jpeg` from the button;
  PNG and WebP are accepted), 404 when the report has none.

`status` is the stored status (the owner sets it with `POST /api/v1/bug-reports/{id}/status`), except that
an `open` report whose task was completed reads `fixed`, and one whose task was cancelled reads `dismissed`.

`context` holds `page` (view, address with query values removed except `view`, `run`, `section`, `task`,
`record`, `note`, `tab`, `activity`; the open run, task or record id; open dialog names), `build` (web
version, commit, build time), `server` (the API's Railway commit and deployment), `service_worker`,
`device` (user agent, platform, viewport, pixel ratio, installed-app mode, online, theme, locale, time
and time zone), `storage` (localStorage key names; values only when short and not key-like), and three
ring buffers from the page: `errors` (last 50 console errors and warnings, uncaught errors and unhandled
rejections, stacks trimmed), `requests` (last 40 `api.ts` requests: method, path without query values,
status, duration, error code, response request id) and `navigations` (last 20 page or run panel changes).
Never collected: cookies, auth or bot keys, CSRF tokens, request or response bodies, query values, or
storage values that look like keys. The server removes secret-named keys and secret-looking text again
(`bug_reports.scrub`, the patterns in `redaction.py`).

```bash
curl -s -H "Authorization: Bearer $KEY" "$ERIDANI/api/v1/external/bug-reports?status=open" | jq '.items[] | {id, title, task_id}'
curl -s -H "Authorization: Bearer $KEY" "$ERIDANI/api/v1/external/bug-reports/$ID" | jq '{description, expected, summary}'
curl -s -H "Authorization: Bearer $KEY" -o shot.jpg "$ERIDANI/api/v1/external/bug-reports/$ID/screenshot"
```

## Synchronize another system

1. Call `/changes?after=0&limit=1` and save **`current_cursor`** as the baseline before scanning records. The first page can contain historical events; use the current cursor for this initial baseline only.
2. Page through each permitted kind, including a separate `archived=true` scan if your mirror tracks archived records.
3. Poll `/changes?after=BASELINE`, apply all items idempotently by record ID, and persist **`next_cursor`** only after applying the whole page.
4. While `has_more` is true, fetch the next page immediately. Otherwise poll periodically, staying below the key's rate limit.

Events include kind, record ID, change time, revision and the **current** record; multiple events can refer to the same latest state. Archives remain visible. A physical missing record has `deleted: true`. Never use the event revision to overwrite a newer record revision. Cursors are monotonic workspace event IDs; gaps are normal because private or ungranted events are excluded. `current_cursor` is informational during ordinary polling: always advance with `next_cursor` so pagination cannot skip events.

Taking the baseline before the initial scan catches edits that occur during scanning. The event commit-order lock prevents a later committed cursor from hiding an earlier in-flight event for the workspace. Webhook push delivery is deferred; this cursor feed is the initial synchronization interface.

## Ask Eri to perform a request

Grant `work:run` plus the needed record scopes, then:

- `POST /requests` with `{ "request_id": "UUID", "message": "Move the launch task to Friday", "thread_id": "UUID" }`. Returns 202 immediately.
- Poll `GET /requests/{returned-id}` for status, clarification and confirmed action receipts.
- `POST /requests/{id}/reply` with a new `request_id`, `message`, and current `expected_revision` to clarify or correct. The reply is idempotent.
- `POST /requests/{id}/cancel` stops unfinished work. Already saved actions remain.

MCP equivalents are `request_submit`, `request_get`, `request_reply` (uses `work_id` for the original request), and `request_cancel`. Use the same optional `thread_id` for related requests; separate thread IDs keep conversation references separate. Omitting it uses one thread for that key.

The existing durable queue handles interruption, dependency waits and parallel independent requests. Bot requests get only their granted planner tools, with no personal-memory prompt injection. Revocation, expiration and membership changes are checked before tools and at mutation boundaries. A write already committed before revocation remains recorded. Revocation cannot retract a provider request already sent; its response cannot authorize new tool writes.

## HTTP examples

With `ERIDANI_BOT_KEY` supplied securely in the process environment:

```sh
curl --fail-with-body \
  -H "Authorization: Bearer $ERIDANI_BOT_KEY" \
  'https://app.eridani.app/api/v1/external/records/task?status=open&limit=20'
```

Prepare an operation file once and keep it for retries:

```json
{
  "request_id": "6b27f61d-df88-43f6-a2e7-7743d61bb85e",
  "tool": "task.create",
  "arguments": {"title": "Connect my assistant", "due_date": "2026-10-01"}
}
```

Use your own newly generated UUID for a new operation:

```sh
curl --fail-with-body \
  -H "Authorization: Bearer $ERIDANI_BOT_KEY" \
  -H 'Content-Type: application/json' \
  --data-binary @operation.json \
  'https://app.eridani.app/api/v1/external/commands'
```

MCP uses the official Python SDK, pinned in the lockfile. [SDK documentation](https://github.com/modelcontextprotocol/python-sdk) and [transport specification](https://modelcontextprotocol.io/specification/2026-07-28/basic/transports) describe transport/client behavior; Eridani's own schemas and scopes above define the application contract.


## Custom planner structure (September 17 implementation)

New keys can opt into `schema:read`/`schema:write` and
`records:read`/`records:write` in Connected agents. Existing keys are unchanged.
Records mirrored from core Tasks/Notes follow the core scopes: without `tasks:read`
or `notes:read`, record reads, search hits and change-feed entries omit `body` (and a
task's bound field values). Creating or editing a record of a work/content type, or one
backed by a task/note, also requires `tasks:write`/`notes:write`.
Schema writes require workspace ownership and the normal preview/later-confirmed
apply flow. Read the current schema revision before creating or updating records.

HTTP discovery: `GET /api/v1/external/structure` and
`GET /api/v1/external/structure/records` (the latter lists custom records). Writes use the
existing external command endpoint with `structure.preview`, `structure.apply`,
`structure.restore`, `record.create`, `record.update`, or `record.link`.
MCP equivalents include `structure_schema`, `record_list`, `record_get` and the
underscore command names. Stable request IDs retain retry deduplication. The existing change feed includes
custom record/schema events only when the key has the corresponding read scope.

See [CUSTOM_PLANNER_IMPLEMENTATION.md](CUSTOM_PLANNER_IMPLEMENTATION.md) for
capabilities, inheritance, migration and rollout status.

## Agent session ledger

Answers "where did I work on X?" across Claude Code, Codex, ChatGPT desktop and Claude Desktop.
A local collector reports sessions ([`scripts/session_collector`](../scripts/session_collector/README.md)); the remotea server can also report them over the [agent link](AGENT_CONTROL.md).
Eridani keeps metadata, an excerpt of at most 8,000 characters and a summary per session. Transcripts stay on the user's machine.

- `sessions:write`:
  - `PUT /sessions/{tool}/{session_id}`: an idempotent full-state report.
  - `POST /sessions/{tool}/{session_id}/link`: file a session under exactly one of `identifier` (a synced Linear issue key), `record_id`, or `clear`.
  - `POST /sessions/rules` (`path_prefix`, `record_id`) and `DELETE /sessions/rules/{rule_id}`: add, replace or remove a folder rule. Existing sessions are refiled at once.
- `sessions:read`:
  - `GET /sessions?query=&unfiled=&live=&record_id=&tool=`: recent sessions, or ranked by full-text match, issue key, folder and meaning. `live=true` keeps only sessions running in a persistent terminal.
  - A `record_id` covers everything under that record, so a client includes its projects and issues.
  - `GET /sessions/rules`: the folder rules.
- MCP tools: `session_search`, `session_link`, `session_rules` and `session_rule`. Eri has the same `session_search` tool in the `sessions` group.
- `tool` values: `claude_code`, `codex_cli`, `codex_desktop`, `claude_desktop`, `cowork`.

Automatic filing runs in this order, strongest first:
1. a known issue key in the branch
2. exactly one known issue key in the title or prompts
3. the longest matching folder rule (managed in the Sessions drawer or with `session_rule`)

Unknown keys such as `gpt-5` never file a session. Explicit (agent) and manual (browser) filing is never replaced.

Without `records:read`, a key sees whether a session is filed, but not the filed record's title.
`resume` carries how to get back to a session: `cmd`, `url`, `hint`, and `live` when the session runs in a [remotea](https://github.com/rdavisdunham/remotea) terminal.
- `live` has `state` (`live`, or `lost` while an app restores it after a server restart), `workspace`, `terminal` (the remotea session id) and `host`.
- A live session has no `cmd`: resuming it a second time would start a duplicate agent.
- Changing `resume` doesn't trigger a new summary.

A settled session (quiet for 10 minutes) gets one summary and one embedding per content change, via the `summarize_session` job.

### Session ledger visibility and freshness

Session-only bot keys can read filing IDs and folder paths, but linked record titles
require **records:read** in both session and folder-rule responses, including mutation
responses. Browser users keep their authorized workspace labels.

The Sessions panel loads older results with **Load more sessions**. It refreshes the
loaded range every 30 seconds while visible, pauses polling in hidden tabs, and does
not repeatedly run an idle semantic search. **Refresh sessions** is always available.
Terminal labels are **Reported live/restoring**: these are collector snapshots, not
verified current connections. When remotea is turned on and its server reports a session over the
agent link, the drawer shows **Live/Restoring in remotea · since …** from the server instead; see
[Agent control](AGENT_CONTROL.md) for the link protocol, its `agents:control` key and collector parity. Existing data has no trustworthy terminal-observation
timestamp; transcript activity and filing edits must not be treated as heartbeats.
