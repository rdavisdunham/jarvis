# Execution protocols

The JSON cases are acceptance specifications. Each has setup, inputs, steps, expected outcomes and required evidence. A scenario is not passed by running an unrelated regression in the same feature area.

## Shared rules

- Start each independent trial from the versioned Rowan corpus. For a multi-turn scenario, keep the same clone and conversation until it finishes.
- Execute as the specified account/workspace. Rowan, Jules, Sam and Lee are separate identities. Never replace permission checks with an admin client.
- Fixture truth belongs to the grader. The agent receives the normal system prompt, tool definitions, retrieved records and scenario messages. Do not paste persona answer keys into the prompt.
- Dates use January 14, 2030, 09:00 Chicago unless a scenario changes that clock. Freeze user-facing instruction/time-resolution clocks together for relative-date testing. Do not freeze worker leases, retry clocks, OAuth expiration or timeouts accidentally.
- Real provider credentials are opt-in. Google, Linear, push and storage tests use synthetic transports by default. Real consent/device tests require dedicated test identities, not production accounts.
- Preserve the initial failure, retries, recovered outcome, provider errors and partial saved effects. Infrastructure failures and task failures are different; neither counts as a pass.
- Record case ID, app commit, corpus/catalog hashes, exact inputs, account/role, tool arguments/results, before/after records, visible response and fault timing.

## Command or agent

For command-contract testing, submit the exact typed command/API operation and inspect canonical database state, including untouched fields and other accounts. Record rejected operations and confirm zero unintended effects. Use a current revision except when deliberately testing a stale one.

For language testing, send the concrete request through the durable queue. Let the production model discover tools, resolve IDs and perform actions. Do not replace its chosen arguments with ideal answers. Grade saved state independently and check that the final answer agrees. A command-contract pass is only component evidence.

Boundary payloads include an empty title, 500/501-character task title, 30,000/30,001-character note body, invalid time 24:00, unsupported status mystery, a Jules-owned UUID, and an existing command ID with changed arguments. Require the precise rejection contract; unrelated failure is not a successful rejection.

For concurrency, use barriers at the specified boundary: before provider reply, after commit/before acknowledgement, or before revision check. Use separate sessions. For a dependency, verify the second request waits for or follows the actual first outcome; do not simulate success by conveniently serializing independent operations.

## Pipeline

Trigger the actual pipeline entry point and durable job. Separate two measurements:

1. **Transport/validation:** inject a deterministic provider output or fault; inspect what the app accepts, rejects, preserves and retries.
2. **Model quality:** call the configured provider on unchanged source data; grade against independent gold labels.

Personal-memory review preserves sources, merges safe exact duplicates, and asks about Miso/Mizo. Rule review uses human classification observations, field descriptions and current schema. Repeated automatic assignments must not become independent human evidence.

For weekly timing, test before/after the local scheduling boundary, repeat the tick and verify one eligible job. For staleness, change the source/schema between provider request and response. For memory deletion, check retrieval, indexes and prompt context as well as the visible list.

## Provider contract

Use fake OAuth tokens and provider resource IDs. Intercept provider transport with complete fixtures: success, pagination cursor, repeated cursor, empty page, 429, 5xx, timeout before commit, timeout after remote commit, partial GraphQL data, deleted object, scope denial and token revocation.

Assert outgoing method/path/body, stable idempotency/resource mapping, optimistic state, retry behavior and verified final state. Actual Google consent, live Linear linkage, storage credentials and disaster recovery are separate opt-in deployment checks.

## Browser

Use the real API and built frontend against synthetic data. Verify desktop, narrow phone and unfolded foldable widths. Capture screenshots and DOM/accessibility evidence, not just successful clicks.

Navigation: detail versus inline editing, Back/forward, reload restoration, dirty-field protection, profile dropdown and global floating chat button. Boards: drag, keyboard/touch fallback and conflicting moves. Calendar: event/task distinction, full details, date-only versus timed entries, day/week/month and sync without scroll jumps.

Chat: navigation creates no mutation card; action cards remain in chronological position, minimize after completion, and retain Edit/Revert. Completion stays in Activity rather than ordinary notifications. Capture before/after scroll position.

The existing five browser suites are runnable through the harness. Their success is regression evidence, not a blanket pass for every newly authored browser scenario.

## Device or protocol

Record actual voice/wake/shutdown scenarios with device, OS/browser, permission state, foreground/background state, mic route, network condition and timestamps.

- Test Eri and hey Eri, false wakes, background noise, interruption, transcript timing, voice selection and deliberate silence.
- Measure the 30-second timeout from the end of Eri’s response. Verify speech resets the appropriate deadline.
- Test explicit goodbye and a natural anything else? → no exchange. Verify the actual voice_end effect and microphone shutdown, not just a farewell.
- Interrupt a spoken acknowledgement after background work is accepted; work must finish and appear once.
- Test network loss/reconnect with no duplicate task or stale false error banner.
- Realtime is disabled. Verify its disabled boundary; do not reactivate it or pay to test it.

For notifications, use a fake delivery sink first, then dedicated-account device checks for push permission, quiet hours, snooze, bundling and completion. A mocked WebSocket/push response does not prove delivery.

## Recording a manual result

Save an evidence JSON with case_id, layer "acceptance", status, app_commit, corpus_sha256, inputs, assertions, evidence_paths, reviewer and tested_at. Outcomes: passed, failed, safety_failure, infra_error, not_completed, not_run. A pass requires every outcome/invariant checked. Missing device/provider evidence means not_completed.

Keep exploratory sessions separate from the golden corpus. Promote useful failures by removing real personal data, adding a minimal fixture and oracle, and explicitly versioning the baseline. Never quietly change expected outcomes to fit the current model.
