# Connect agents to Eridani

## Create a connection

Open **Settings → Integrations → Connected agents** in the intended workspace. Add an agent, choose its permissions and expiration, and copy the key once. It is stored only as a hash; losing it requires a new key. Revoke obsolete keys in the same panel.

Each key is tied to one workspace and the account that created it. Switching the browser workspace does not move the key. A shared-workspace membership downgrade limits writes immediately; leaving or being removed ends access. Workspace owners can revoke keys in that workspace; other members manage their own keys.

- API base: `https://app.eridani.app/api/v1/external`
- MCP endpoint: `https://app.eridani.app/api/v1/external/mcp/`
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

Queries accept `query`, `archived`, `limit` (1–100), `offset`, and applicable exact filters: `status`, `space_id`, `area_id`, `project_id`, `goal_id`, `task_id` (linked notes), `parent_task_id`, `assignee_id`, `work_type`, `due_from`, `due_through`. Dates use `YYYY-MM-DD`. Queries are literal keyword matches, not semantic/vector searches. Pagination returns `next_offset`. Unsupported filters are rejected.

Commands include task create/update/complete/reopen; goal/project/space/area/actor create/update; and note create/update/append/replace. Archive with the corresponding update command. Prefer stable IDs for organization and assignments. Supplying task project/assignee names can create organization entries and therefore requires organization write access. Due dates, times, tags, parents, goal/project links and note links use the existing site schemas returned by `/capabilities`.

MCP exposes `records_list`, `records_get`, `changes_list`, `request_get`, the permitted command names such as `task_create`, and supported undo/queued-work tools. Command arguments are flattened, with an additional required `request_id` UUID.

For an edit, read the record first and include `expected_revision`. Omit unchanged fields. An explicit null clears only fields whose contract allows it. Replacement link lists must preserve links you still want. HTTP 409 means the revision or idempotency request conflicts; fetch the current record and decide the next action instead of retrying with a guessed revision.

## Retry and outcome rules

Generate one request UUID per intended operation and persist it with the exact arguments. Reuse it unchanged after a timeout or disconnect. The server namespaces it to the bot key; REST and MCP retries share the same receipt. Reusing it for different instructions returns 409. A rejected command saves no partial changes or misleading success card.

The returned `request_id` / Activity `id` is the server's durable work ID. Use that returned ID for `/requests/{id}` or MCP `request_get`. The request UUID supplied by the caller is the idempotency key, not the status URL ID.

Independent operations can run concurrently. Writes still enforce current revisions and short graph transaction locks. Creating the same human idea with two different request UUIDs can create two records: idempotency is based on operation identity, not fuzzy title matching.

Revert is a compensating command, not deletion of history. Creating a record is reversed by archiving when no later changes or links prevent it. Field-level undo preserves unrelated later edits and refuses to overwrite a later change to the same field. Relationship rewrites and external provider changes may require manual edits. A bot can only revert its own actions; its creator can use the site controls even after the key is revoked.

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
Custom records can include authored note content; grant that scope deliberately.
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
