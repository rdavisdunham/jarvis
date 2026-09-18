# Eridani / Jarvis — progress and next steps

Updated September 18, 2026. Eridani (Eri) is the assistant's name.
Jarvis remains the repository and infrastructure project name.

Completed implementation is marked **[x]**. **[ ]** means work or verification is
still pending; automated checks do not mark physical-device acceptance complete.
The newest batches are near the top; earlier sections retain release history.

## Invitation flow cleanup — planned

- [ ] Rename “Standalone personal account” to “Invite to Eridani — separate account.”
- [ ] Separate app-access invitations from shared-workspace invitations. Explain
  that each invited person gets their own private records, and show exactly which
  workspace is shared when one is selected.
- [ ] Hide irrelevant Viewer/Editor roles for separate-account invitations; clarify
  creating the email-bound invitation versus copying a generic sign-in link.
- [ ] Improve first sign-in/acceptance copy and test both invitation paths on phones.

## Semantic search and vocabulary learning — September 18

- [x] Shared hybrid search over tasks, notes and custom records, full descriptions,
  field definitions/options/values, inherited homes and linked names. Eri receives
  structured matches and a separate possible-match lane, including misfiled items.
  Explicit filters and current workspace permissions apply to both lanes.
- [x] Per-account/workspace search aliases, separate from personal memories and
  routing examples. Use or recent conversational continuation can teach provisional
  vocabulary after presentation. Silence and unseen output remain unknown;
  corrections pause disputed mappings. Never relabel records during search.
- [x] Settings → Organization → Search aliases: inspect sources, confirm, correct,
  pause, forget, and disable learning. Weekly review can propose organization rules;
  search acceptance never activates a rule by itself.
- [x] Incremental, chunked cloud embeddings with generation checks, recoverable jobs,
  keyword fallback, complete collection coverage and a production backfill command.
  Use existing PostgreSQL JSON vectors; pgvector infrastructure remains deferred.
- [x] Task-only top search, note meaning search, selected-result Eri navigation,
  authenticated HTTP search and MCP `record_search` under `records:read`.
  No navigation/action cards or success notifications are created by search.
- [x] Migration/model agreement, focused regressions, full backend/frontend suites,
  isolated desktop/mobile browser acceptance and paired Luna/Gemini evaluations.
  See [architecture and verification](SEMANTIC_SEARCH.md).
- [x] Production rollout: API and worker deployed, search enabled on both services,
  and all 67 initial search documents ready with no pending generations/errors.
  GitHub CI and public readiness checks pass.
- [ ] Physical Live acceptance: ask for the pest-control company's tasks, inspect a
  misfiled match, continue, correct the identity, then inspect Search aliases. Verify
  interruption and a reply never heard/seen do not create a positive interaction.
- [ ] Expand the reviewed retrieval benchmark with real failures. Current synthetic
  paired checks are a regression sample, not a broad accuracy claim.

## Conversation action cards — September 17

- [x] Navigation and read-only requests stay ordinary conversation; they do not
  create chat action cards. Saved changes and requests needing attention retain cards.
- [x] Compact chat cards show the outcome and useful changed details, with Edit and
  Revert available directly. Original requests, field differences and clarification
  history are available under Details; full Activity remains available at the top.
- [x] Anchor cards to their originating messages. Completion, parallel requests,
  clarification continuations and reloads preserve their conversation positions;
  subsequent messages continue below them. Completed cards start collapsed.
- [x] Remove routine success notifications, including old success entries and pending
  push deliveries. Keep reminders, deadlines, questions and failures actionable.
- [x] Verify real queue/save/revert behavior in a synthetic browser environment,
  reverse completion order, transcript replies, reloads and existing navigation.
  See [validation](CHAT_ACTIVITY_VALIDATION.md).
- [ ] Physical voice acceptance: navigate with “show me that,” create two tasks,
  answer a clarification and continue talking. Check compact cards stay by their
  original speech and only reminders/questions/errors appear in Notifications.

## Chat, search, foldable settings and phone navigation — September 17

- [x] Persistent bottom-right Eri button on every signed-in page, animated opening,
  accessible open/close controls, reduced-motion support and clear active-voice state.
- [x] Top search consistently finds actionable tasks across collections, including
  completed work, without changing into notification search. Keep page-local search
  available for notes, memory, calendar and organization.
- [x] Focused Settings sections for Profile, Organization, Notifications, Voice,
  Connections, Privacy, Sharing and System; responsive controls for folded phones,
  unfolded/foldable widths and desktop; update Eri's site-control descriptions.
- [x] In-app browser Back/Forward: close details/chat before going to the previous
  page, preserve filters and protect unsaved edits without document reloads.
- [x] Browser acceptance at 390, 600, 768, 820, 1000 and 1440px: no page overflow,
  chat stays within the viewport, Back/Forward stays in the same document, and
  unsaved note/structure drafts survive Back. Reduced-motion opening also passes.
  See [validation](SHELL_UX_VALIDATION.md).
- [ ] Physical Pixel 10 Pro Fold acceptance: folded/unfolded keyboard, scrolling,
  Back gestures and voice remaining active when the conversation is closed.

## Task embeddings and agent search

Implemented in the September 18 semantic-search batch above. Its verification,
remaining phone checks and deferred pgvector upgrade are tracked there and in the
vector-index backlog; this section no longer duplicates unfinished implementation.

## Real-use acceptance checklist — pending

These are owner/device checks, not implementation tasks. Automated and simulated
checks already recorded below do not mark these complete. Use disposable test
records where practical; record device/browser, result and any issue when tested.

- [ ] **Clarification continuity:** make a request that needs more information,
  answer Eri's follow-up, and confirm the original activity card completes with
  the answer and saved action. No duplicate task or stranded Waiting for you card.
- [ ] **Rapid voice requests:** say “Add a task to call Alex,” “Also remind me to
  buy milk,” then “Actually, make that call tomorrow.” Each task appears once;
  only the call gets tomorrow's date. Repeat with the correction arriving both
  before and after the original request finishes.
- [ ] **Action receipts:** verify the card describes what actually changed. Open
  Edit, then try Revert on a separate unchanged test record; the previous value
  returns. If another edit intervenes, Revert must not silently overwrite it.
- [ ] **Voice shutdown and wake:** try “goodbye,” “that's all, thanks,” and “no”
  after Eri asks whether anything else is needed. Voice ends; “Eri” / “Hey Eri”
  starts it again while wake listening is enabled. Check the 30-second idle
  timeout after her response and confirm speech/pending work is not cut off.
- [ ] **Phone recovery:** switch Wi-Fi/cellular, briefly lose connectivity, and
  background then return to the browser. Accepted work survives, no task is
  duplicated, and voice reconnects or offers a working restart.
- [ ] **Profile menu:** on phone and desktop, open Memory and Settings, dismiss
  the menu by tapping outside, and log out/back in. Verify the correct account
  and workspace return; personal memory stays outside shared workspaces.
- [ ] **Activity reset:** finish the previously requested old-history cleanup
  through authenticated access or Clear activity history; confirm old queued/error
  cards disappear while saved tasks and notes remain. See the reset item below.
- [ ] **Locked-phone notifications:** receive a real reminder while the phone is
  locked, open it to the correct record, and check delivery after a network gap.
- [ ] **Calendar and Linear:** verify Google create/edit, event details and selected
  calendars with real accounts; verify a real Linear import/edit syncs correctly
  without duplication. Use disposable events/issues and check both applications.
- [ ] **External bot and sharing:** connect a real API/MCP client with a scoped
  key, verify a permitted action and denied out-of-scope action, then revoke the
  key. Check invitation and membership revocation with a second signed-in account.
- [ ] **Mobile usability/accessibility:** check the physical phone's keyboard,
  scrolling, detail cards and chat controls; finish screen-reader/contrast checks.
- [ ] **Cloud recovery:** confirm normal use with the host PC off and queued-work
  recovery across a controlled cloud API/worker restart. After R2 credentials are
  supplied, verify upload/download, isolated restore and the scheduled backup;
  test failed/stale-backup alerts once implemented. Existing Railway PITR remains.

## Before native Android — recommended remaining order

This is the current recommendation, not a claim these features have shipped.
The detailed sections below retain implementation history and acceptance evidence.

1. **Configurable planner acceptance:** the agreed organization/routing batch is
   deployed September 17. Validate Structure, custom collections, workflows,
   inheritance, learned rules and weekly interviews in real use.
   See [implementation and test notes](CUSTOM_PLANNER_IMPLEMENTATION.md).
2. **Notification acceptance:** deadline alerts, quiet hours, explicit urgency,
   daily summary, actionable questions/errors and durable snooze are implemented
   in this batch. Verify actual locked-phone delivery and real-account behavior.
3. **Operational and connected-account acceptance:** complete the checklist above,
   activate the separate R2 recovery copy when credentials are available, automate
   restore drills/backup alerts, and finish Google production consent verification
   before broader onboarding. Calendar writes, multi-user access and API/MCP are
   implemented features to validate, not features to rebuild.
4. **Android readiness and usage pilot:** document/test how a native client reuses
   sign-in, record links, commands, activity/clarifications and notifications;
   specify retry/conflict behavior for intermittent connectivity. After expansions
   and device/operations checks, run the seven-day pilot with at least 50 successful
   task/reminder interactions and fix material failures before native development.

Vector indexing, Langfuse, open-ended research agents and additional integrations
can follow demonstrated need; they are not prerequisites for native Android.

## GitHub CI — September 17

- [x] Add pull-request/main-push CI for backend correctness/tests, PostgreSQL
  migration/model agreement, frontend tests/build and desktop/mobile browser
  acceptance. Locked dependencies, standard Linux runners, synthetic data and
  failure artifacts only; no deployment or model-provider credentials.
- [x] Push and activate the workflow with the feature batch. The first hosted run
  passed frontend/browser checks and caught a worker-test log-directory assumption;
  the test now uses a temporary directory. Follow full-suite results in
  [GitHub Actions](https://github.com/rdavisdunham/jarvis/actions/workflows/ci.yml).
- [ ] After the first run, consider requiring both checks before merge and making
  Railway wait for CI before deployment; those settings are unchanged.

Details: [CI.md](CI.md).

## Configurable planner, routing and notifications — September 17

Deployed to production September 17 in `8e63e95`. Both Railway services report
SUCCESS; the public frontend matches the tested build and the API is ready.
This supersedes the fixed goals/projects/classification portion of the earlier
[task-routing PRD](TASK_ROUTING_PRD.md).

- [x] Editable default schema: record types, required descriptions, fields,
  workflows, main-home hierarchy, inheritance, named links and cardinality.
- [x] Work/content/timeline/metric capabilities with stable task/note services;
  best-effort import retains existing source data and adopts recurring/imported tasks.
- [x] Inline detail cards, list/board/timeline, touch movement, custom filters,
  saved views, task multi-select, links and conversational site controls.
- [x] Versioned structural preview/apply/history; guarded record and relationship
  receipts/Revert; stale external edits cannot be overwritten by an old undo.
- [x] Separate field understanding and routing evidence, explicit rules,
  weekly dream review, manual interview, daily offer and configurable work hours.
  Learned automatic rules remain gated on independent held-out human evidence;
  confirmed rules can be enabled directly.
- [x] Deadline alerts, quiet hours, urgent overrides, optional morning summary,
  same-notice snooze and unseen background-result grouping.
- [x] Scoped schema/custom-record API and MCP access, including Connected agents
  settings; existing keys do not gain new permissions automatically.
- [x] Isolated migrations, backend/frontend checks and desktop/mobile browser
  acceptance. See [evidence and limits](CUSTOM_PLANNER_IMPLEMENTATION.md).
- [x] Release and verify API + worker on Railway (`8e63e95`), including migrations,
  exact frontend asset hashes, API readiness and clean worker startup. Existing
  PITR archive uploads were verified in logs; an extra local export was blocked
  by automatic approval review and not performed.
- [ ] Physical-phone voice, board gestures, notification delivery and network
  recovery; actual model interviews and connected-account imports/updates.
- [ ] Real-user routing quality dataset: 100 held-out matches and ≥95% precision
  before claiming learned automatic-rule quality. Synthetic matcher tests are
  regression checks, not a substitute for this acceptance gate.
- [ ] Later expansion: occasional contextual clarifications during ordinary
  conversation, outside explicit field setup or the weekly interview.

## Profile navigation — September 16

- [x] Move Memory, Settings and Log out into the bottom-left profile menu. Preserve
  direct Eri navigation and personal-workspace-only memory access; support keyboard,
  touch, outside-click dismissal and Escape. Production build and ten desktop/mobile
  browser checks passed, including navigation, keyboard focus and logout.

## Clarification continuations and activity reset — September 16

- [x] Give each pending question a stable identity and route conversational answers
  through the existing backend's `work_answer` tool, without an interpretation model.
- [x] Start a fresh durable attempt beneath the original activity card; preserve the
  original request, question/answer history, saved command receipts and per-action Revert.
- [x] Reject duplicate/stale question claims, account/agent mismatches, cancelled work,
  and detached edits that attempt to bypass an explicitly linked pending question.
- [x] Keep unrelated requests parallel, preserve repeated clarification rounds, and
  notify the current voice session when an earlier session's request is continued.
- [x] Add Clear activity history with a confirmation: stop unfinished requests and
  hide old cards while retaining saved records and their audit receipts. Scope to the
  signed-in account and workspace; late model responses cannot recreate cleared cards.
- [x] Validate real Luna and Gemini continuations with synthetic records; both kept
  independent work separate and updated the original task without duplicating it.
- [x] Verify 96 focused backend tests, 121 frontend tests and six mobile browser
  checks; production build passes. Clearing preserves saved tasks; no browser errors.
- [ ] Complete the owner's requested production history reset after authenticated
  maintenance access is available; no production records should be deleted.

Implementation and test notes: [CLARIFICATION_CONTINUATIONS.md](CLARIFICATION_CONTINUATIONS.md).

## Compact planner UX follow-through — September 16

- [x] Checkpoint the external-agent release on main (`5c76eeb`) before starting this batch.
- [x] Combine task view/filter controls; move secondary actions into a menu; add
  removable filter chips, modified saved-view feedback and clearer empty results.
- [x] Make task capture title-first with optional details, a removable planned-day
  default, visible submit action and immediate access to the saved detail card.
- [x] Add board column navigation, remembered column, optional hidden empty/finished
  columns, and compact mobile timelines with sticky names and date controls.
- [x] Keep task drafts on failures; show field save state and compare conflicting
  versions before explicitly applying a draft or accepting the saved values.
- [x] Open notes for reading and save long-note drafts explicitly. Existing goals,
  projects, spaces, areas and assignees now use inline saves. Eri can navigate away
  from clean detail cards; dirty notes remain protected.
- [x] Standardize priority/status labels, explain scheduling and organization,
  show project open-task/deadline summaries and optional planner examples.
- [x] Add workspace-aware record links and a return trail for linked cards.
  Links use existing access checks and never grant permission.
- [x] Clarify sharing roles, integration summaries, memory sources, processing
  states and the exact scope of forgetting a fact versus deleting its source.
- [x] Share accessible tab navigation, improve touch/focus targets, scope search to
  the current page, and make mobile chat a scroll-locked sheet with Back to work.
- [x] Update Eri's site map and editor behavior descriptions for the new controls.
- [x] Verify 31 focused backend tests, 120 frontend tests and 25 browser acceptance
  checks using 260 synthetic tasks and 16 projects; production build passes.
- [ ] Complete real Android keyboard/microphone/background-notification and formal
  screen-reader/contrast acceptance; browser emulation is not physical-device proof.
- [ ] Rename the owner's existing Business space during the separately approved
  task-routing rollout. This UX batch preserves classification IDs and records.

Audit follow-through and verification: [UX_POLISH_VALIDATION.md](UX_POLISH_VALIDATION.md).
R2 activation, MCP OAuth-only clients and the seven-day usage pilot remain open.
Automatic task routing was subsequently implemented locally in the September 17 batch.

## External-agent API, MCP and backup preparation — September 16

- [x] Add named, workspace-bound bot keys in Settings → Integrations, with separate
  task/organization/note permissions, expiry, one-time reveal and immediate revocation.
- [x] Expose typed API and official-SDK MCP tools over the existing command service.
  Structured actions execute directly; retries share receipts across both transports.
- [x] Preserve revisions, membership checks, bot attribution and guarded Edit/Revert
  in Activity. Reject stale edits and mismatched idempotency keys.
- [x] Add keyword search, linked-record lookups and a paginated cursor change feed
  for external synchronization, including archives and permission filtering.
- [x] Add optional scoped requests to Eri's durable queue, status, clarification and
  cancellation. Keep personal memories/conversations and connected-account tools
  outside bot grants; recheck revocation before each subsequent tool write.
- [x] Verify parallel retries, conflicting edits, real MCP client interoperability,
  mid-run revocation, permission boundaries and mobile/desktop connection controls.
- [x] Prepare separate daily R2 cron configuration and a secret-safe, offline setup
  check. Preserve daily/weekly retention and the existing encrypted restore tooling.
- [ ] Activate R2 only after bucket credentials are supplied; verify a real R2
  upload, download, isolated restore and scheduled production run. Deferred by owner.
- [ ] Add MCP OAuth consent/discovery for clients that cannot supply Bearer headers.
- [ ] Consider signed webhook delivery after the initial polling integrations prove useful.

Connection and API contract: [EXTERNAL_AGENTS.md](EXTERNAL_AGENTS.md).
Backup handoff: [R2_BACKUPS.md](R2_BACKUPS.md).
Verification: [EXTERNAL_AGENTS_VALIDATION.md](EXTERNAL_AGENTS_VALIDATION.md).

## Direct backend execution and action receipts — September 16

- [x] Remove the interpretation-model call and exact-quote gate. Voice turns and
  text requests are saved unchanged and sent directly to the selected backend.
- [x] Keep independent requests concurrent. The backend resolves references with
  `work_followup`; the database supplies status, dependencies and saved record IDs.
  Waiting requests release their worker slot and resume from a saved checkpoint.
- [x] Reserve affected records before tools act, bind new IDs when commands commit,
  and retain short transaction locks/revision checks. Unknown and bulk scopes wait
  conservatively; an older request cannot expand into newer reserved work.
- [x] Keep earlier speech as reference data, exclude later speech from earlier
  requests, and avoid replaying earlier user turns as new instructions.
- [x] Make cards lead with verified saved changes and key fields. Collapse original
  speech; keep Edit/Revert on saved changes. Replace Mark reviewed with optional
  Dismiss notification for failures; this never approves or retries work.
- [x] Link follow-up receipts to the earlier request. Guard Revert against changed
  fields and incoming record links; record reversals without deleting history.
- [x] Verify natural speech and early follow-ups with Luna and Gemini; verify
  desktop/mobile Edit, Revert and notification dismissal on synthetic records.
- [ ] Owner voice acceptance after deployment: Alex/milk/follow-up, filtering from
  natural speech, wake words, goodbye, disconnect/reconnect and backgrounding.

This supersedes the original batch's separate intake-model classification. Earlier
failed intake cards can be retried through the direct backend; they are not replayed
automatically because their original times or intent may no longer be current.

## Current batch — reliable background work, action cards and onboarding

Deployed September 16 (implementation `e5d9234`); API, worker and public HTTPS
checks passed. Release checks and remaining acceptance
work are recorded in [BACKGROUND_WORK_VALIDATION.md](BACKGROUND_WORK_VALIDATION.md).
Design and decisions: [BACKGROUND_WORK_PRD.md](BACKGROUND_WORK_PRD.md).

- [x] Complete the delegated desktop/mobile audit: **34 recommendations and
  76 screenshots**, with production-login and synthetic authenticated coverage
  clearly separated in [recommendations.md](../recommendations.md).
- [x] Persist voice/text intake, queued requests, dependencies and execution
  checkpoints in PostgreSQL. New speech cannot replace an earlier request.
  Accepted work continues after voice ends, a browser disconnects or the worker
  restarts. Planned writes reuse stable command IDs after a crash.
- [x] Run up to two independent actions per account and four installation-wide;
  intake has a separate bounded lane. Related actions wait for their dependencies.
  Check current account/workspace access at execution and every server write.
- [x] Support targeted corrections, clarification, cancellation and continuation.
  Cancel stops unfinished work and preserves committed changes. A provider/browser
  operation already dispatched may finish; its actual outcome stays visible.
- [x] Add global Activity and compact per-request/per-record cards with progress,
  attention, Edit, Revise, Continue, Cancel and safe Revert. Local undo checks
  changed fields and preserves unrelated later edits. Unsupported relationship or
  external reversals explain why the record needs review. Remote sync remains
  pending until confirmed. Browser inline saves link back to their request.
- [x] Restore progress after reconnect and combine verified spoken results at a
  pause. Do not announce an unconfirmed external write as completed. Ordinary
  voice acknowledgments do not fill Activity with completed small-talk cards.
- [x] Apply related UX-01–10 and invitation UX-26: activity visibility, separate
  voice/work controls, accurate connection/worker/provider status, cloud setup
  copy, Tasks breadcrumb, clearer invitation/error states, and 30-second labels.
  Also compact saved-view controls, enlarge mobile completion/action targets,
  and shorten the mobile month calendar while retaining the selected-day agenda.
- [x] Build the public landing/privacy/terms/help pages and Google entry flow.
  Keep the planner at `app.eridani.app`, with an assets-only public site for
  `eridani.app` and `www.eridani.app`. Preserve invited destinations through OAuth;
  explain wrong-account, expired and revoked invitations. No automatic emails.
- [x] Remove personal private-chat creation controls and Eri's private-mode action.
  New conversations use account history preferences. Legacy private, history-off
  and shared-workspace retention/isolation boundaries remain intact.
- [x] Harden wake-plus-request capture, 30-second idle handling, contextual
  shutdown, repeated close, and immediate microphone release. Browser wake remains
  opt-in and foreground-only. Accepted work survives shutdown.
- [x] Update Eri's tool catalog and site map with Activity and account-scoped work
  listing, cancellation and supported reversal. Browser controls require a current
  originating-device acknowledgment; expired controls cannot navigate later.
- [x] Verify both backend profiles against synthetic create/edit/cancel/goodbye
  scenarios; verify exact-once command recovery by killing the actual worker.
  Full regression, build, browser and schema-upgrade evidence is in the validation
  document. Realtime stays disabled.
- [x] Owner reports basic phone voice is working well (September 16). Wake words
  and the updated goodbye sequence have not yet been tested on the phone.
- [ ] Finish remaining phone/desktop microphone trials: rapid A/B requests, late
  correction, wake plus immediate request, goodbye, reconnect and backgrounding.
  Automated microphone simulations are not real-device acceptance.
- [ ] Complete Google production consent verification and review the public
  operator/contact policy details before broad invitations. No Google approval
  is implied by publishing the pages.
- [ ] **R2 backup activation:** supply bucket-scoped S3 endpoint/access credentials,
  deploy daily/weekly encrypted exports and verify download/restore from
  `eridani-backups`. The bucket exists; credentials are still absent. Native
  Railway PITR remains active. No second database or paid queue was added.

The compact task/organization/note/settings recommendations are implemented in
the UX follow-through batch above. Remaining acceptance covers physical-device
and assistive-technology checks, not another redesign. Keep the seven-day usage pilot after the expansion
and real-device/operation checks. Notification controls and task routing were
subsequently implemented locally in the September 17 batch; external API/MCP is
implemented. Android, vector indexing and open-ended research remain expansions.

## Task routing — original decisions, implemented locally September 17

- [x] Inspect current backend memory injection, task organization and learning.
      Write [Task routing PRD](TASK_ROUTING_PRD.md) with agreed product decisions,
      implementation proposal, correction lifecycle and acceptance cases.
- [x] Confirm automatic strong matches with uncertain fields left unassigned;
      customizable classification fields; one client per project and standalone
      client tasks.
- [x] Implement customizable structure, inheritance, separate routing knowledge,
      evidence/corrections, reviewed reorganization and gated automatic routing.
      The September 17 implementation supersedes the fixed-field PRD; real-user
      held-out quality acceptance remains open in the current batch above.
- [ ] At implementation, rename Davis's Business space to Work in place; configure
      Client = ABC and the Andi / Transcript Intelligence project relationships.
      These are planned setup requirements; no records have been changed for them.
- [x] Add editable work windows as weak context and an independent routing-memory
      section. Do not mix personal-memory facts with routing rules or reinforce
      Eri's unconfirmed guesses.

## Backlog status — September 15

- [x] Add Backlog to new-task creation, inline task details, bulk edits, board
      drag/drop, status filters, saved views/links and Eri's task/site tools.
      New tasks still default to Open. Backlog means work captured for later;
      Open means ready to start; Deferred means previously planned work postponed.
- [x] Map Linear Backlog to Eridani Backlog in both directions, including publish.
      Existing Deferred-to-Linear-Backlog compatibility remains.
- [x] Preserve existing task statuses, dates and reminder behavior. Active still
      includes unfinished Backlog tasks; due/planned Backlog work stays visible in
      matching calendar/day views. No database migration or bulk rewrite needed.
- [x] Validate the exact local release: 84 backend tests, 101 frontend tests
      (one existing Realtime skip), production build and healthy API/worker.
      Deploy alongside the 30-second voice/end-conversation fix; cloud migration
      preparation remains undeployed.

## Voice shutdown update — September 15

- [x] Increase the quiet voice timeout to **30 seconds**, measured after actual
      speech/playout and task work, with activity resetting the window.
- [x] Replace conflicting instructions that prohibited farewell delegation.
      Live now delegates closing intent to the backend's session-bound
      `voice_end` tool. It accepts no account/device/session IDs, stops further
      calls in the turn, preserves committed task receipts, and closes provider
      media before releasing the microphone.
- [x] Harden browser fallback for natural farewells and same-bubble Live captions;
      preserve negation, added requests and contextual yes/no meaning.
- [x] Verify the backend tool/close path and browser timeout, cleanup and fresh
      session behavior with automated checks.
- [ ] Confirm natural spoken endings on the actual phone/desktop after refresh.
      Automated tests do not measure the Live model's real-world recognition rate.

## Cloud migration

Decision: retain point-in-time recovery alongside independent daily backups.
Decision: prepare the Railway migration now, with Google-only public sign-in.
Cloud production cutover is complete at **https://app.eridani.app**; device acceptance and optional R2 exports remain. See the
[migration runbook and owner checklist](CLOUD_MIGRATION.md).

- [x] Compare Railway, managed PostgreSQL options, Cloudflare and a VPS against
      the actual workload and Brainforge's deployment patterns. See the
      [hosting plan](CLOUD_HOSTING_PLAN.md) and
      [interactive cost comparison](cloud-hosting-plan.html).
- [x] Prepare Railway API/worker/backup service configuration, PostgreSQL 16
      image candidate with pgvector and inherited Railway recovery tooling,
      serialized migrations, bounded connection pools and a worker overlap lock.
- [x] Add Google-only cloud sign-in, revocation of old pairing sessions, safe
      staging pause flags, maintenance mode, schema readiness and private
      credential handoff/preflight. Existing local configuration is unchanged.
- [x] Add encrypted R2-compatible daily/weekly exports, scoped retention,
      authenticated downloads and atomic new/empty-database restores. Isolated
      S3-compatible rehearsal preserved all 55 public/DBOS tables and integration
      encryption; populated-target restore was refused.
- [x] Configure Railway project, API/worker, `app.eridani.app`, Google callback,
      private database references and PostgreSQL **16.15**, matching development.
      Use `.railway/railway.ts`; the old per-service JSON mechanism is rejected
      for new services. Explicit image pins prevent the import default selecting 18.
- [x] Deploy the source-only cloud preview against separate `eridani_preview`.
      HTTPS/readiness/authentication checks pass; Google-only login is configured.
      Worker and provider actions stay paused; the PC app remains active.
- [x] Enable native PITR; verify the initial full backup and continuous archive
      success with zero observed failures.
- [x] Verify native PITR timestamp restore: the sibling contains the synthetic
      `before` marker and the source retains `after`, on PostgreSQL 16.15.
      Desktop/mobile cloud login-page browser checks also pass.
- [x] Receive explicit transfer approval; stop local writers; take the final
      encrypted snapshot; restore production `eridani`; compare all **55 tables /
      11,245 rows** and verify Google linkage, DBOS, schema and credential decryption.
- [x] Activate only the Railway worker. Confirm current heartbeat, healthy API,
      private database connectivity and successful Google Calendar synchronization.
      Preserve local database and encrypted snapshot for recovery.
- [x] Remove migration-only public database access. Retire the local Windows
      startup shortcut to `.runtime/retired-startup` so a reboot cannot resume the
      stale local writers. Keep production independent of the PC.
- [ ] Owner acceptance: Google browser sign-in, Live voice, Android permissions /
      notifications, controlled integration writes and use with the PC offline.
      Re-register microphone/notification permissions on the new origin.
- [ ] Deferred by owner: supply R2 bucket-scoped S3 credentials, deploy the daily
      backup job and verify export/download/restore. `eridani-backups` exists.
- [x] Stop unused PostgreSQL 18 compute while preserving its volume. Remove the
      disposable recovery service and its synthetic test volume after verification.
- [ ] Retire the preserved PostgreSQL 18 volume after confirming it holds no needed
      data. Keep future local development separate from the preserved production copy.
- [ ] Automate periodic isolated recovery drills and stale/failed-backup alerts.

## Ordered delivery batches — next work

This order supersedes the older thematic priority lists below. Each implementation
batch includes the website, Eri's corresponding tools/context, focused regression
checks and release notes. Completed features are evidence, not new work.

### Batch 1 — finish daily task workflows and Eri reliability

- [x] Open existing tasks as inline detail cards from lists, boards, timelines,
      calendar and linked records. Click individual fields to change them; save on
      blur/Enter, Escape cancels a pending field. No Edit/Save buttons or edit mode.
      Main details sit beside compact property/assignment/project/goal attribution.
      Eri can navigate away after pending changes save; conflicts retain the field.
- [x] Save named task views with their tab, filters, grouping, sort and layout;
      extend current tab links to restore the complete view.
- [x] Add revision-guarded note append and exact anchored replacement, preserving
      untouched text, Unicode and source links.
- [x] Resolve assignee names to canonical actors before filtering. Acknowledge
      the actual layout and visible/empty results before claiming success.
- [x] Distinguish output truncation, malformed arguments, transport failures and
      remote application failures. Cover the missing calendar-sync worker path
      in the regression fixtures and finish archived-note filter parity.

Exit: task browsing is consistent, edits preserve unrelated content, saved views
restore faithfully, and Eri reports the browser/domain result accurately.

### Batch 2 — multi-user foundation

- [x] Support separate signed-in accounts, sessions, profiles and private records,
      preserving the existing owner's records and integration links.
- [x] Add invitations, shared space/project membership, roles and revocation;
      distinguish an assignee from a user who actually has access.
- [x] Enforce membership and ownership consistently in reads, writes, search,
      embeddings, Eri context/tools, notifications and background jobs.
- [x] Keep learned personal memory and integration credentials private by default;
      make access to shared notes/tasks and calendar information explicit.

Exit: two accounts can use private and intentionally shared work; tests prove
isolation and immediate revocation across the UI, API, agent and retrieval paths.

### Batch 3 — connected workflows

- [x] Add scoped, revocable external bot API access through the existing command
      service, then an MCP adapter. Preserve actor attribution, idempotency,
      revisions and the multi-user access rules from Batch 2.
- [ ] Expand notifications with natural-language snoozing, priority, quiet-time
      preferences and optional digests/bundling. Completion/attention delivery
      moves to the current background-work batch. Keep task completion and alert
      delivery history distinct.
- [ ] Document and test these contracts for Eri and future Android clients.

Exit: external bots and Eri follow the same rules, and notification preferences
produce predictable delivery without duplicate task effects.

### Batch 4 — device/operations checks, then the seven-day pilot

- [ ] Verify GPT-Live recovery, interruption, late corrections, voice ending,
      wake words and 30-second handoff on the actual phone and desktop.
      These checks now ship with the current background-work batch above.
      Realtime remains paused.
- [ ] Verify locked-phone Web Push, network gaps and cloud API/worker restart
      recovery, including normal use with the host PC off. Automate isolated
      restore drills and stale/failed-backup alerts; retain a separately
      protected recovery copy outside the primary database service.
- [ ] After those checks pass, run the seven-day PRD pilot with at least 50
      successful task/reminder interactions and record reliability, delivery and
      voice-quality results. Development cost tracking stays off; provider usage
      can be reviewed separately.

Exit: recorded device/operation evidence and a pilot findings list, with material
failures fixed before expanding the execution surface.

### Batch 5 — larger expansions after the foundation is proven

- Bounded request execution, progress, cancellation and saved results are now
  in the current batch above. Open-ended agent assignments/research remain later.
- [ ] Build native Android against the established account/action contracts.
- [ ] Upgrade memory/note retrieval with measured quality checks and pgvector;
      improve deep-sleep entity reconciliation and clarification follow-through.
- [ ] Revisit Langfuse if tracing/evaluation gaps justify it. Additional providers,
      Home Assistant, finance and richer goal/dependency features remain later.

## Current release: inline task workflows and invited accounts

- Batch 1 shipped as cee4ca8; [daily workflow validation](BATCH1_VALIDATION.md).
- Batch 2 is deployed; [release validation](BATCH2_VALIDATION.md). It adds invited Google accounts, private Personal scopes, shared space/
  project workspaces, owner/editor/viewer roles, private saved views and
  revocation across requests, Eri tools, Live sessions and semantic retrieval.
- Shared workspaces start empty. Personal memory and connected accounts stay
  private; shared chats are temporary. Local shared calendar/alert records are
  explicit. [Account guide and boundaries](ACCOUNTS_AND_SHARING.md).
- Follow-ups: intentional private-to-shared record moves with a relationship
  preview, combined workspace views, per-member shared push delivery and an
  account recovery method before allowing non-owner Google unlink.
- Task cards have no Edit/Save buttons: click individual values; valid changes
  save on blur/Enter and Escape cancels the pending field.

## Previous release: task tabs, calendar cards and touch boards — deployed

- [x] Combine Today, Inbox, Next 7 days and All (formerly Work) into one Tasks
      page. Preserve search, filters and layout; keep legacy links working.
- [x] Label calendar item types and open saved details from month chips and
      agenda rows. Put Edit at the upper right; include task relationships,
      reminders, notes and saved appointment fields.
- [x] Make task boards work with mouse, touch and keyboard grips, including
      edge scrolling and empty status columns. Moves save canonical status,
      project or assignee; sort continues to determine card order.
- [x] Explain task versus alert in the reminder form and
      [PRODUCTIVITY_SCHEMA.md](PRODUCTIVITY_SCHEMA.md). Tasks own completion;
      events and work blocks reserve time. No database migration.
- [x] Give Eri an acknowledged open-details action and distinguish read-only
      record cards from unsaved editable drafts.
- [x] Validate and deploy: 447 backend / 78 frontend tests pass; real browser
      checks cover touch/mouse/keyboard moves, saved details and the existing
      planner/Google/Linear flows. Healthy services, exact source hashes and
      unchanged saved records. [Validation](CALENDAR_TASKS_VALIDATION.md).

### Requested follow-ups

- [x] **Inline task detail cards.** Supersedes the initial Edit-button request:
      existing task cards have individually editable, automatically saved fields.
      Keep creation forms explicit and protect other unsaved form drafts.
- [x] **Multi-user support.** Add separate user accounts, private data and
      preferences, plus invited membership in shared spaces/projects with clear
      roles and permissions. Scope tasks, notes, memory, integrations and Eri's
      retrieval/actions to the current user's access.

## Previous batch: conversational planner workspace — deployed

- [x] Complete typed, acknowledged conversational controls for pages, search,
      filters, sort, grouping, layouts, calendar ranges, selection, organization,
      Settings sections and device voice/wake/density preferences.
- [x] Give Eri read/patch/save/close/discard access to task, alert, note,
      organization, appointment, Google event, bulk-task and memory editors.
      Drafts remain unsaved until save; navigation cannot silently discard them.
      OAuth consent, browser permissions and credential entry remain owner actions.
- [x] Add shared task list/boards/timelines to Work, Today, Inbox and This week.
      Add project status boards and start/target timelines. Accessible status
      selectors accompany drag and drop; undated work remains discoverable.
- [x] Review and apply compact UX: collapsed filters with visible applied chips,
      quick task capture, denser notes/organization rows, expandable relationship
      detail and five focused Settings sections. Google consent returns directly
      to Integrations. Review: [PLANNER_UX_ARCHITECTURE_REVIEW.md](PLANNER_UX_ARCHITECTURE_REVIEW.md).
- [x] Update the backend tool schemas, tool-local instructions and Live's delegated
      site context to the final website. No second agent runtime is required.
- [x] Preserve exact authored note whitespace and sparse updates; return precise,
      owner-scoped relationship errors without exposing foreign records.
- [x] Add verified scheduling for up to eight tasks/seven days, including time
      windows, dependencies, busy periods and proven earliest finish. Commit
      rechecks availability/revisions and saves all local blocks atomically.
      Short, owner-bound references replace model-copied encrypted payloads.
- [x] Review schema and architecture: retain canonical Task, authored Note,
      learned Memory, outcome Goal and finite Project. Alerts and work blocks
      stay distinct projections/attention records. No migration or new service.
- [x] Complete fresh 8 × 3 × 2 backend eval and audit every reply; keep the
      interrupted attempt, native scores, explicit grader corrections and the
      separate six-trial short-reference diagnostic visible.
      [Results](RELIABILITY_HELDOUT_RESULTS.md).
- [x] Verify real GPT-Live audio/delegation, captions, quiet timeout, global mobile
      voice dock, fresh-session restart and media cleanup in a disposable database
      with generated speech. Clarification, stale-input protection, interruption,
      backend failure recovery and receipts also have deterministic protocol tests.
- [x] Validation: 446 backend tests pass (one optional skip; paused Realtime
      module excluded), 75 frontend tests pass (one retained Realtime case skipped),
      build and application/changed-script Ruff pass. Real rendered acceptance:
      85 acknowledged controls, desktop/mobile density and overflow, Google/Linear
      consent, details, write/retry/conflict flows and 30-second scroll stability.
      See [PLANNER_VALIDATION.md](PLANNER_VALIDATION.md).
- [x] Deploy and verify: API/worker/PostgreSQL healthy; all 52 backend module
      hashes match the running container. HTTPS serves index-C5suLihr.js.
      Fifteen checked table hashes are unchanged, including all 55 existing tasks;
      cost tracking remains off.

### Immediate follow-ups from this release

- [ ] Add revision-guarded note append and exact anchored replacement so the model
      does not retranscribe unchanged body text. A Gemini trial changed Unicode
      characters into literal escape text; full-content writes remain an exposed risk.
- [ ] Resolve assignees to canonical actor IDs before filtering. Preferred profile
      name is not always the stored actor label; show/acknowledge empty results.
- [ ] Verify requested layout and visible results before claiming a timeline or
      filtered tasks are on screen. Keep UI acknowledgement as the authority.
- [ ] Distinguish model output truncation, malformed tool arguments, transport
      errors and remote application errors in user messages.
- [ ] Extend the eval's deliberately unmodeled calendar-sync worker path;
      current unavailable-calendar trial remains uncredited, not assumed successful.
- [ ] Add saved/persistent views and deep view links, then extract App.tsx's view
      and voice controllers; generate/parity-check shared API/browser contracts.
- [ ] Consider persistent task dependencies/blocked reasons, direct goal-to-task
      links and metric history when concrete use requires them.
- [ ] Browser/provider acceptance does not replace physical-phone microphone,
      background/wake and network-recovery testing. Keep the seven-day usage pilot
      after the agreed expansion and device/operation checks.

## Previous batch: tool design and paired regression — deployed

- [x] Research official OpenAI/Google guidance and implement a short global policy,
      detailed tool-local schemas and portable capability loading. Eighteen tools are
      initially exposed; 81 remain available across thirteen groups.
      See [TOOL_DESIGN_RESEARCH.md](TOOL_DESIGN_RESEARCH.md).
- [x] Implement all seven earlier tool findings: exact task selections, actionable
      ID failures, authoritative counts, DST resolution, explicit remote retries,
      relationship diffs/current peer revisions, and source-note provenance.
- [x] Retire GPT-5.4 mini from active routing and Settings. Previous OpenAI
      selections resolve to Luna; explicit Gemini preferences are preserved.
      Automatic memory/note extraction also uses Luna with low reasoning.
- [x] Validate both providers against the full 81-tool catalog. Fix strict-mode
      compatibility for unsupported uniqueItems while keeping server validation.
      Preserve the interrupted first evaluation as separate evidence.
- [x] Complete the fixed 20 scenarios × 3 repeats × 2 models and audit every
      reply. Keep the original baseline, diagnostic runs and matched comparison.
      Protocol: [TOOL_REFINEMENT_EVAL_PLAN.md](TOOL_REFINEMENT_EVAL_PLAN.md).
      Reviewed completions: Gemini 58/60, Luna 57/60; original scores and three
      explicit grader corrections remain visible. [Results](TOOL_REFINEMENT_RESULTS.md).
- [x] Deploy and verify the exact evaluated code on desktop/mobile. API, worker
      and PostgreSQL are healthy; all 19 evaluated application source hashes match
      the running container. HTTPS serves index-DCL2bqzL.js.
- [x] Verification: 379 backend tests passed (one optional skip; paused Realtime
      module excluded), 60 frontend tests, build, Ruff and browser checks passed.
      Separate production-error diagnostic: both models 3/3 scheduling passes;
      retain outside the 120 scored results. Real voice/device acceptance remains
      a follow-up. Development cost tracking remains off.

### New evidence to harden next

- [x] Return field/reference-kind-specific errors for invalid note links; teach
      sparse edits to avoid copying unchanged IDs. Luna once mistyped an existing
      goal ID and safely failed the note edit.
- [x] Align synthetic read validation with production errors and separately
      audit six scheduling trials. Preserve the original 120 scores and sources.
- [x] Align simulated calendar-list/availability connection state and remote
      polling behavior; never silently merge follow-up scores.

- [x] Add calendar planning preflight that checks a proposed block set against
      confirmed busy/free windows and dependencies. Normalize availability into
      the requested time zone while retaining exact UTC instants. One Luna
      diagnostic scheduled across a known busy interval; the tool stored the
      requested timestamps faithfully, so agent reasoning still needs a guard.
- [x] Add held-out agent scenarios for entity-type discovery (authored notes
      versus tasks) and valid natural-language explanations of infeasibility.
      Keep deterministic domain tests and a literal-effects reply audit.
- [ ] Complete real-device GPT-Live delegation/recovery and connected Google/
      Linear acceptance; synthetic evals do not replace those checks.

## Previous batch: GPT-Live only — deployed

- [x] Pause Realtime without deleting its controller, voice catalog or protocol code.
      The server rejects new Realtime sessions before any provider call or session
      replacement. Only GPT-Live is advertised to devices.
- [x] Default new and previously Realtime-selected devices to GPT-Live. Preserve
      the saved Live voice; replace unsupported names with Marin. Hide the provider
      selector while only one mode is enabled; keep the Live voice selector.
- [x] Update connection-error guidance and Eri's site map. API/worker are healthy;
      HTTPS serves index-DUXj-4wa.js. Eleven Live/availability backend tests,
      48 Live transcript/idle/sign-off tests, Ruff, production build, and deployed
      desktop/mobile preference recovery and voice-list checks passed.
- [ ] Continue real-device voice acceptance with GPT-Live only. Realtime comparisons
      and provider testing are paused until the owner asks to revisit them. Before
      re-enabling Realtime, adapt its direct-tool path to the new discovery contract
      (Live uses the backend loader) and rerun protocol acceptance.

## Previous batch: expert task-agent evaluation — complete

- [x] Build 24 difficult scenarios with deterministic state/scope graders and
      paired synthetic fixtures. Include multi-turn clarification, date edge
      cases, graph/note preservation, planning, untrusted content and injected
      recovery failures. See [EXPERT_AGENT_EVALUATION.md](EXPERT_AGENT_EVALUATION.md).
- [x] Preserve complete observable tool context for improvement work: arguments,
      outcomes, receipts, errors, conversation turns and timing. Keep native
      reasoning and credentials out of the artifacts.
- [x] Complete all 144 scored trials (24 × 3 repetitions × 2 models) and audit
      every reply. Gemini completed 71/72 workflows; Luna 69/72, with no unintended
      saved changes. Retain all traces, expected fixtures, diagnostics and review
      notes. Results, costs and limits: [EXPERT_AGENT_RESULTS.md](EXPERT_AGENT_RESULTS.md).
- [x] Verify 115 evaluation tests, fixture/prompt pairing, artifact integrity,
      cleanup, and desktop/mobile report behavior. Production selection remains
      GPT-5.4 mini; no deployment or production tool changes in this batch.
- [x] Sharpen the tools using the retained evidence: structured task filtering
      and selection counts, robust record references and lookup recovery,
      explicit batch receipts, read-only DST validation, clear remote retry
      semantics, affected relationship revisions and source-note provenance.
      Candidates and validation requirements are in
      [AGENT_TOOL_IMPROVEMENTS.md](AGENT_TOOL_IMPROVEMENTS.md).
- [x] Re-evaluate both models on the twenty fixed regression scenarios after tool
      improvements; retain the original baseline and explicit grader corrections.
- [x] Add fresh held-out scenarios before claiming broader reliability gains.

## Previous batch: Gemini task-agent trial — deployed

- [x] Add Gemini 3.8 Flash alongside the existing OpenAI task agent, selected in
      Settings. Persist the owner's choice across devices and pin each active turn
      to one provider. Text chat and GPT-Live delegation share the selection.
- [x] Add a blank GEMINI_API_KEY entry in the ignored .env, server-side key loading,
      missing-key guidance and an opt-in synthetic real-provider handshake.
      Setup: [GEMINI_SETUP.md](GEMINI_SETUP.md).
- [x] Preserve tool authorization, revisions, receipts, action limits and Gemini
      thought signatures across sequential/parallel calls. No automatic provider
      fallback or replay after a failed request. Cost tracking stays disabled.
- [x] Validation: 225 backend tests (one optional skip), 76 frontend tests,
      production build, Ruff and desktop/mobile browser Settings checks passed.
      API/worker/PostgreSQL are healthy; authenticated deployed bootstrap and HTTPS
      bundle index-CjYJ0is5.js verified. Task/memory counts and historical cost ledger
      are unchanged. The backup service is active.
- [x] Owner supplied Gemini key; API/worker recreated and healthy. Real synthetic
      handshake passed. Final task-loop comparison: Gemini 12/12 clean passes;
      Luna 11/12 clean, with one rejected ID typo corrected successfully. Final
      records correct in all 12 cases for each. See the model comparison/eval trace.
- [ ] Owner tries Gemini in real text and GPT-Live conversations, then compares
      against Luna. The 5.4-mini baseline is historical; audio/device and connected-service behavior
      were not part of the synthetic task-loop comparison.
- [x] Add reasoning-enabled Luna through Responses and expose it in Settings.
      Low reasoning, stateless encrypted reasoning continuity, existing tool
      safeguards and receipts. Both OpenAI profiles use the existing API key.
      No schema migration; older provider-only preferences remain compatible.
- [x] Luna Responses verification: 12 workflows completed without tool errors,
      470 reasoning tokens confirmed; one valid-offset string-grader mismatch
      corrected and a fresh timed-task check passed. 232 backend tests, 76 frontend
      tests, desktop/mobile model selection and Gemini handshake passed.
      See [LUNA_SETUP.md](LUNA_SETUP.md) and the retained eval evidence.
- [ ] Owner compares reasoning-enabled Luna and Gemini during real text/GPT-Live
      use. GPT-5.4 mini is retired; automatic memory/note extraction uses Luna
      and embeddings remain OpenAI.

## Database backups

- [x] **Automated database backups are already implemented and running.**
      Dedicated Docker backup service runs at startup and every 24 hours.
      Encrypted PostgreSQL custom dumps and SHA-256 manifests are written to
      C:/Users/davin/JarvisBackups; successful backups keep 30 days of history.
      Failed attempts retry after five minutes. Settings shows the latest backup.
- [x] Restore validation for the current schema, including the productivity graph,
      completed in an isolated database with workers disabled. Most recent verified
      dump: jarvis-20260913T042040Z.pgdump.enc.
- [ ] Keep an off-PC recovery copy and recovery key (see daily-use operations below).
- [ ] Automate periodic isolated restore drills and actionable backup-failure/stale
      backup alerts; current restore checks are operator-run.

## Previous batch: productivity graph — deployed

- [x] Private Personal/Business spaces and ongoing areas, with optional organization
      for standalone tasks. Existing work stays unclassified until assigned.
- [x] Goals with success criteria, parent goals, short/long horizons and optional
      outcome metrics. Projects gain lifecycle and start/target dates.
- [x] Many-to-many goals/projects, editable from either side. Outcome progress stays
      independent of task/project completion; revision checks protect linked edits.
- [x] Connected notes: multiple goals/projects, note-to-note links and backlinks,
      related notes on goal/project details, and scoped keyword/meaning search.
- [x] Planned task dates distinct from deadlines, reminders and time blocks;
      Today/This week/Calendar display planned work. Stable local assignee IDs.
- [x] Goals & projects page, mobile forms, space/area/goal filters, Eri commands,
      site navigation/highlighting and organization context.
- [x] Existing scheduler retains unified task completion and independently
      completable recurring occurrences. Assignment remains separate from agent execution.
- [x] Release validation: 213 backend tests (one optional test skipped), 76 frontend
      tests, production build and Ruff. Desktop/mobile browser checks cover
      many-to-many links, notes/backlinks, planned dates, scope filters, unsaved
      edits across refresh and Eri's navigation/highlighting. Existing Google/Linear
      acceptance also passes, including 30-second scroll stability and write recovery.
- [x] Deployed schema 0011 with healthy API/worker/PostgreSQL and HTTPS bundle
      index-GZSDsE7r.js. All 55 existing tasks and the old columns/data in 16 tables
      match their pre-migration hashes. Two private spaces are seeded; existing
      projects/tasks remain unclassified. The configured agent is still gpt-5.4-mini
      and cost tracking remains off.
- [x] Encrypted backups before and after the upgrade restored into isolated
      databases without workers. Verified post-upgrade backup:
      jarvis-20260913T042040Z.pgdump.enc, including every new graph table.
- [ ] Owner: try a real goal with two supporting projects, connect a note, and
      organize current work into Personal/Business areas. Shared membership remains
      a later feature.
- [ ] Model comparison recorded in [BACKGROUND_MODEL_COMPARISON.md](BACKGROUND_MODEL_COMPARISON.md).
      Initial Luna/Flash task acceptance is recorded; compare the candidates with
      Luna/Gemini during real use, with a stronger planner considered where needed.
      The default model and disabled cost tracking remain unchanged.
- [ ] Richer outcome check-ins and independently managed agent jobs remain later
      expansions. Multi-user support is scoped in the requested follow-ups above.
      See [PRODUCTIVITY_SCHEMA.md](PRODUCTIVITY_SCHEMA.md).

## Previous batch: unified tasks, calendar details and Linear — deployed

- [x] Tasks own completion. Reminders are alerts on a task; new standalone alerts
      create their task automatically. Migration carries old schedules, delivered
      notices and completion history forward. Repeating routines have a template
      and separately completable occurrences. Completing a task closes its alerts.
- [x] Local appointments and task work blocks, editable on mobile and through
      Eri's tools. A block reserves time without changing the task deadline.
      Calendar availability includes local busy entries and connected Google calendars.
- [x] Optional Google publication for selected appointments/blocks. Durable writes,
      stable IDs, no duplicate display of a linked copy, explicit conflict review,
      and unlinking that preserves both records. A remote deletion leaves local
      work available; it does not delete the task or its appointment.
- [x] Rich Google cache: descriptions, location, meeting links, organizer, guest
      responses and attachment links. Cache-first details remain readable when
      the fresh provider read fails. Historical recurrence UNTIL compatibility
      fixes the misleading incomplete-range warning; real issues identify a source.
- [x] Linear API integration: encrypted personal key in Settings, selectable teams,
      optional “assigned to me” scope, five-minute incremental polling and daily
      full reconciliation. Issues/projects/parent links import into local work.
- [x] Linear writes through task edits, explicit Publish to Linear, exact workflow
      controls and Eri's shared tools. Stable create IDs, durable receipts,
      pre-write conflict detection and owner-directed resolution. Local annotations
      and work are retained when remote access changes. See [LINEAR_SETUP.md](LINEAR_SETUP.md).
- [x] Release validation: 204 backend tests (one optional test skipped), 74 frontend tests, production build,
      mobile browser flows, real 30-second scroll stability, lost-response retry,
      Google/Linear conflict handling and migration round-trip with legacy reminders.
      Provider writes in acceptance use synthetic fixtures, not owner calendars/issues.
- [x] Deployed schema 0010 with healthy API/worker/PostgreSQL and verified HTTPS
      bundle index-CFF8CMib.js. Preserved all 47 original tasks; eight legacy
      standalone schedules gained tasks (55 total), with delivery history retained.
      All 2,680 cached Google events were enriched, including 267 descriptions,
      403 meeting links and 542 events with guests. September's projection reports
      no incomplete range or warnings. Credentials, selections and cost ledger match
      their pre-upgrade hashes; cost tracking remains off.
- [x] Encrypted backups before and after the migration restored into isolated
      databases. Latest verified backup: jarvis-20260913T025905Z.pgdump.enc,
      including planning/Linear tables, credentials, selections and record counts.
- [ ] Owner: connect a Linear key in Settings, choose teams, then try a real issue
      import/edit. No real Linear account is connected by automated acceptance.
- [ ] Owner: verify real Google editing consent and a personal published work block;
      confirm shared-calendar availability and richer event details on mobile.
- [ ] Later: native repeating appointments, more Linear project/label/cycle controls,
      OAuth/webhooks if needed, and Eri as an agent inside Linear. Public Linear
      updates have no conditional revision parameter; the documented preflight
      check cannot eliminate the narrow read/write race.
- [ ] Langfuse/evals integration remains deferred until this functionality is hardened.
      Smarter notifications and scoped external bot API/MCP remain the next larger
      expansions. Real-use voice recovery/device checks precede the seven-day pilot.

## Previous batch: mobile Calendar and Google editing — deployed

- [x] Owner confirmed real Google Calendar sync works on mobile.
- [x] Fixed the roughly 30-second scroll jump: background refresh retains the
      agenda instead of clearing it; assistant highlights scroll only once.
- [x] Month, Week and Day views, remembered on this browser. Double-tap a date
      to open Day, with an explicit Open day button and previous/next navigation.
- [x] Google event creation, editing/rescheduling and deletion through the website
      and Eri's shared tools. Choose a writable selected calendar; timed/all-day
      events, title, notes, location, availability and basic repeating events.
      Recurring edits explicitly target one occurrence or the entire series.
- [x] Separate optional Calendar editing consent, fresh Google permissions,
      conditional edits to prevent stale overwrites, durable write receipts and
      retry reconciliation to avoid duplicate events. Recent calendar changes
      shows confirmed, pending, failed or unconfirmed outcomes.
- [x] Automated validation: 183 backend tests plus the additional uncertain-outcome
      regression pass; 74 frontend tests, production build, scoped Ruff and isolated
      browser/migration acceptance pass. Browser checks cover a real 30-second
      refresh on a long mobile agenda, all views and create/edit/delete with a
      lost-response retry. No synthetic writes were made to the owner's calendar.
- [x] Deployed schema 0009 with healthy API/worker/PostgreSQL and the verified HTTPS
      bundle. Google credentials/selections, 2,680 cached events and the historical
      cost ledger are preserved; cost tracking remains off. Encrypted backup
      jarvis-20260913T002615Z.pgdump.enc restored successfully in an isolated database.
- [ ] Owner: Settings → Google → Enable Calendar editing, grant the additional
      Google permission, then try a real personal event. See [GOOGLE_SETUP.md](GOOGLE_SETUP.md).
      Events with guests and special Google event types remain managed in Google.
- [ ] Verify multiple selected calendars and combined availability on the real
      account, including shared calendars. Source selection and simultaneous sync
      are implemented; owner confirmed general sync, not this whole matrix.
- [x] Implement smarter notification controls locally in the September 17 batch;
      authenticated scoped API/MCP was implemented September 16. Real-device and
      external-client acceptance remain open.
- [ ] Real-use voice recovery, device/operations checks and the seven-day pilot stay
      after expansion. Internal cost tracking stays disabled during development.

## Previous batch: Google sign-in and Calendar — deployed and connected

- [x] Owner-only Google sign-in linked from an authenticated session, separate
      read-only Calendar consent and PIN pairing retained for recovery.
- [x] Encrypted refresh credentials, selected calendars, durable incremental sync,
      recurring/deleted-event handling, availability and disconnect/unlink controls.
- [x] Deployed schema 0008 with healthy API/worker/PostgreSQL. 157 backend and 71
      frontend tests, browser/migration acceptance and encrypted restore passed.
      Backup: jarvis-20260911T231255Z.pgdump.enc; historical cost ledger preserved.
- [x] Owner created the Google Cloud project/client and saved credentials locally.
      API/worker loaded the configuration; owner subsequently confirmed real sync.
- [ ] Verify Google sign-in from a second device. Pairing remains the recovery route.

## Previous batch: development mode, contextual tasks and notes — deployed

Owner order: keep expanding before the large testing round. Internal cost recording
and budget enforcement are disabled locally during development; the owner monitors
OpenAI Usage. Historical holds remain preserved and no longer block work.

- [x] Development switch in ignored .env.upgrade: JARVIS_COST_TRACKING_ENABLED=false.
      No new cost events/reservations, headroom checks or reconciliation writes while
      disabled. Settings shows the disabled state and links to OpenAI Usage.
- [x] Contextual task lookup from current selection, visible records and recently
      discussed tasks in the same conversation, with fresh revisions and ambiguity.
- [x] Select tasks and bulk-edit status, project, assignee, priority and due date.
      One stale task rejects the whole batch; no partial update. Eri shares the API.
- [x] Notes workspace with editable text, tags, project/task/conversation links,
      archive/restore, keyword search and explicit search by meaning.
- [x] Cloud note embeddings and durable indexing; extraction previews with exact
      source quotes, linked tasks and duplicate prevention. Authored notes do not
      silently become personal memory assertions.
- [x] Eri can read/search/edit notes, propose or create requested to-dos, select
      task groups and open linked records through the CopilotKit bridge.
- [x] Network-response-loss regression: repeat Save reuses the original command
      receipt. Search changes cannot mix old pagination or trigger meaning queries.
- [x] Deployed schema 0007; API/worker/PostgreSQL healthy; live HTTPS bundle checked.
      130 backend and 71 frontend tests, isolated browser/migration checks, and
      post-deploy encrypted backup/restore passed. Historical cost ledger unchanged.
- [x] Next expansion implemented: Google owner sign-in and read-only Calendar;
      real account sync has since been confirmed by the owner.
- [ ] Complete fine-grained CopilotKit field/settings controls, archived-note
      filter parity, broader contextual-reference quality and project board/timeline.
- [ ] After expansion: real-use voice recovery, physical devices/operations,
      then the seven-day owner pilot. Use OpenAI Usage for development costs.
- [ ] Before re-enabling internal accounting, reconcile historical holds and account
      for the unrecorded development period. Disabled tracking cannot backfill usage.

## Previous batch: unified workspace and calendar — deployed

Previous owner order: accounting fixes, workspace/calendar expansion, then acceptance.
The development-mode decision above now defers accounting reconciliation.

- [x] One Work list for tasks and standalone reminders, with linked reminders under
      their task; searchable metadata, status/project/kind filters and retained history.
- [x] Month calendar and selected-day agenda for date-only/timed deadlines, delivered
      reminders and future recurrence previews. Owner timezone, DST-aware schedules,
      completed items, month navigation and mobile layout. Reading never dispatches work.
- [x] Real projects with rename/archive, migrated existing project labels, subtasks
      with cycle checks, assignee labels, work types, tags and priority.
      Assignment metadata does not launch an agent.
- [x] Task details can open/create linked reminders. Reminder details support edits,
      rescheduling, completing one occurrence and cancelling a series.
      A metadata edit preserves a reminder already queued for delivery.
- [x] Shared Eri tools for projects, organization fields, calendar reads and calendar
      navigation/filtering. CopilotKit acknowledges the selected date and filters;
      an open editor blocks navigation.
- [x] Budget lifecycle: stale activity becomes unconfirmed rather than active; explicit
      text-model rejections release unused allowances, unknown outcomes remain held.
      Missing Realtime usage cannot silently count as zero; separate duration-based
      transcription usage is recorded once. Settings lists unconfirmed sessions.
- [x] Audited reconciliation operator command from provider evidence, retaining the
      original ledger and an idempotent adjustment. No historical charge is guessed.
- [ ] Deferred until accounting is re-enabled: historical reconciliation needs billing evidence. The existing
      project API key received HTTP 403 from the organization Costs endpoint.
      The 25 older uncertain Realtime sessions plus one abandoned active session
      retain their unknown headroom. This is held allowance, not confirmed spending.
      No budget limit was raised and no old hold was automatically forgiven.
- [x] Deployment, current encrypted restore, 116 backend tests, 36 frontend tests,
      migration round-trip and isolated desktop/mobile/CopilotKit checks.
- [x] Conversational voice sign-off: Eri may offer a natural closing question.
      The next complete reply follows the question's meaning (no to "anything else",
      yes to "will that be all"). Added requests/continued speech keep voice open.
      Live uses raw captions for context before animated text finishes appearing.
      Both providers retain their microphone cleanup, wake listening and quiet timeout.
- [ ] Verify the new natural sign-off with the owner's microphone on GPT-Live.
- [ ] Before accounting is re-enabled, reconcile allowances against a Jarvis-only OpenAI cost export for September 11
      UTC. The current command settles individual sessions; project/day aggregates
      need a matching period-level adjustment, not invented per-session costs.
      Preserve original usage, exclude unrelated project activity, and avoid double counting.
- [ ] Project board/timeline, deeper contextual references, Google sign-in/Calendar,
      notification bundling and scoped bot API/MCP remain later. The first notes and
      contextual editing release is implemented in the current batch.

## Current position

The daily-use task/reminder foundation is running in Docker at
https://davispc.tail957c2.ts.net:9443. Text task agents and GPT-Live voice are connected. Realtime is temporarily disabled; its code is retained.
The owner reports that real voice conversations and “Hey, Eri” work very well. This is the working
foundation, not completion of every integration in the upgrade PRD.

Detailed implementation, recovery instructions, evidence, and limitations:
[JARVIS_IMPLEMENTATION.md](JARVIS_IMPLEMENTATION.md).

## PRD comparison and current delivery order — September 11

The [PRD release boundaries](JARVIS_UPGRADE_PRD.md#3-release-boundaries) describe
the original sequence. Current implementation and owner feedback change that order:

- **R0, durable voice/text tasks:** implemented, with successful real-provider checks.
- **R1, daily use:** most features are implemented. Actual locked-phone delivery,
  Windows reboot recovery, a seven-day pilot and the full measured acceptance suite
  remain open. Passing automated tests is not full R1 acceptance.
- **R2, Home Assistant:** not implemented; defer behind the connected personal
  workspace because that is the owner's current priority.
- **R3, memory/context:** source-backed cloud learning, embeddings, retrieval,
  correction/deletion and the first weekly review are active. Retrieval quality,
  clarification follow-through, richer context brokering and indexed search remain.
- **R4, richer personal work:** projects and linked notes are implemented; durable
  research/planner jobs, finance and broader integrations remain expansions. GPT-Live's short task delegation
  does not implement the durable research system.

Later owner decisions supersede the PRD's local GPU/legacy-memory migration
proposals: use cloud inference, retire the active legacy Qdrant bridge, and consider
pgvector after base hardening. Keep the host awake, retain the 30-second quiet
timeout without the old total-session caps, and use the configured fixed pairing
PIN until Google sign-in. GPT-Live, full conversational app control, linked notes,
unified work items and weekly memory review extend the original PRD.

**Execution order:** development cost tracking off; contextual task editing and
linked notes (deployed); Google sign-in and read-only Calendar (implemented, real account setup pending); smarter
notifications; scoped bot access and bounded agent work. Then complete real-use
voice recovery and device/operations acceptance, followed by the seven-day pilot.
Android and broader integrations follow the core workspace. Expand CopilotKit
coverage alongside every feature. Accounting reconciliation and vector indexing
are deferred until base hardening; do not treat missing development costs as zero.

## Done

- [x] GPT-Live (gpt-live-1) alongside Realtime, selectable in Settings.
      GPT-Live delegates tasks to the existing gpt-5.4-mini backend; domain tools,
      owner checks, receipts, and budget accounting are shared.
- [x] Provider-specific voices in Settings, remembered per browser and validated
      on both client and server. Chat stays minimal.
- [x] New chat, a simplified conversation window, and a microphone-reactive
      iridescent voice mode with both speaker transcripts.
- [x] Remove ownership callout blobs and replace raw “unresolved” with a useful,
      plain-language voice status.
- [x] Complete reminders without deleting them. Delivered reminders remain
      pending until completed; completed recurring occurrences preserve the routine.
- [x] Eri can open app pages and open/highlight a requested task or reminder.
      This is the initial typed UI tool, not the full CopilotKit expansion below.
- [x] Foreground wake phrases “Eri,” “Eridani,” and “Hey, Eri” using browser
      recognition. Owner confirmed the existing Hey Eri behavior works.
- [x] A 30-second quiet window after responses, paused/reset by speech and task work.
      Actual audio playout extends the window beyond early transcript arrival.
      Wake-word listening resumes after voice has fully closed.
- [x] Global iridescent voice glow and a mobile/desktop voice dock outside chat.
- [x] CopilotKit frontend tool registry with an authenticated, per-device bridge:
      chat open/close, smart mobile visibility, searches, status/project filters,
      task/reminder forms, current screen context and UI acknowledgements.
- [x] Editable preferred name in Settings; shared by greetings and model prompts.
      No hard-coded owner name remains in the personality file.
- [x] Cloud fact extraction, verbatim source evidence, tags, deduplication,
      corrections, embeddings, semantic retrieval and early context injection.
- [x] Gradual assistant caption display for GPT-Live, keeping exact provider text.
- [x] Realtime close drains cancelled-response usage and releases idle headroom;
      an already-hung-up call no longer creates a spurious budget hold.

- [x] FastAPI domain API, PostgreSQL storage, and a separate durable DBOS worker.
- [x] Responsive web app with task capture, editing, completion, reminders,
      recurring schedules, notification Inbox, and export.
- [x] Persistent pairing sessions: secure HTTP-only cookie, 30-day expiry.
      The owner's chosen fixed PIN is set in ignored local configuration until
      OAuth replaces pairing. Setup preserves it; session tokens remain random.
- [x] OpenAI project key loaded from the ignored .env; live text actions verified.
- [x] Realtime browser voice with server-owned tools, silence gate, interruption,
      and spoken confirmation after a successful save.
- [x] Browser microphone capture waits for provider readiness; brief connection
      drops have a recovery window.
- [x] Live cloud speech captions in the conversation: partial text while speaking,
      reconciled final transcripts, and both speaker labels.
- [x] Clarify the manual voice control as “Respond now”; prevent repeating a
      completed action with it.
- [x] Clear recovered voice errors, ignore late responses from stopped sessions,
      and retry transient voice status failures.
- [x] Reconnect task event streams explicitly after HTTP failures, preserve their
      cursor, and keep task-sync warnings separate from voice status.
- [x] Private conversation mode and source-backed memory capture/correction/deletion.
      The legacy Qdrant bridge was retired at the owner's request on September 11.
- [x] Command retry deduplication, durable reminder recovery, and budget tracking.
- [x] Private Tailscale HTTPS access, Docker startup helper, encrypted backups,
      and a populated backup restore check.
- [x] Owner has tried a real microphone conversation successfully.
- [x] Replace competing Buster/Jarvis personality instructions with Eridani/Eri:
      witty and lightly playful, polished and formal, with assistance as her purpose.
- [x] Share one personality source between text and voice:
      [personality.py](../apps/api/jarvis/personality.py).
      Remove the old JARVIS_SYSTEM_PROMPT entry from environment configuration.
- [x] Remove the old five-minute and twenty-turn voice session caps.
      Keep browser-disconnect cleanup, provider failure handling, explicit End
      voice, and the existing budget controls. The new owner-requested 30-second
      quiet timeout is separate from the removed total-session limits.

## Latest validation

September 11 hardening: **99 backend tests and 33 frontend tests pass** (one
optional paid-provider test skipped). Ruff and the production build pass.
Migration 0005_task_due_time is deployed; API, worker and
PostgreSQL are healthy. Backend regression coverage includes bulk edits beyond the
old caps, partial results, correction/cancellation races, stale memory reads,
review failure/cooldown behavior, task-time/DST rules and budget deferral/resume.
Frontend coverage includes standalone endings and microphone release for both
providers. Mobile task-time editing and the budget display passed browser checks,
alongside the existing memory-review flow. Screenshots were visually inspected.
A current encrypted backup restored successfully with dispatch disabled.

The live snapshot at 19:16 UTC has 1 visible canonical memory; older assertions
remain as history/suppressed records. No test tasks or memories were written to the
live owner in this batch. The physical-phone/reboot checks and seven-day owner
pilot remain open. Use this deployment as the pilot baseline; do not count historic
synthetic acceptance records as owner interactions.


Earlier September 11 baseline: **76 backend tests and 18 frontend tests pass** (one optional
paid-provider test is skipped in the normal suite). Ruff and the production build
pass. Migration `0004_memory_review` preserved a populated memory during an
isolated upgrade/downgrade/upgrade check. Mobile browser review/correction and
Review now passed in a disposable database. Earlier opt-in real OpenAI
extraction/embedding/retrieval checks also passed; those were not repeated here.

Weekly review is deployed and enabled. The first live pass scanned 2 facts,
merged 0, and queued the Hayes/Haze clarification. Next run: September 13 at
03:00 America/Chicago. The active legacy Qdrant service has been removed, and
API/worker/PostgreSQL are healthy. Database checks show the new review queue.

Real-provider GPT-Live: one saved task, both transcripts, audible confirmation,
global mobile glow/dock, quiet timeout, final usage, New chat and switch to Realtime.
Realtime: one task, audio, interruption without losing the save, restart, partial
captions and injected SSE/status recovery. Its latest reservation closed correctly.

Real text-agent browser checks: chat closing, project/status filters, search,
mobile navigation closing the overlay, visible learned memories, editable-name
Settings and no mobile overflow. Both mobile screenshots were inspected.
The live history pass processed 65 eligible user sources and produced **2 embedded,
source-backed memories**. Most prior messages are task requests or conversational
backchannels and correctly do not become memories. Source transcripts contain a
name spelling variation; entity/alias reconciliation remains a quality follow-up.
Synthetic task fixtures were archived; no test memories were added to the live owner.

## Prioritized next work — owner usage feedback

Owner-requested feature expansions are also recorded in the app's “Eridani
roadmap” project. The code-review findings and release gates added here on
September 11 are tracked in this document; they have not been copied into new
app tasks. The ordered delivery batches at the top now set the work order;
this section retains detailed requirements and historical evidence.
The first hardening implementation batch is now deployed; remaining acceptance and expansion work stays open.

### Completed: current experience

- [x] Preserve working Hey Eri behavior; add standalone Eri and the 30-second timeout.
- [x] CopilotKit chat-panel controls, mobile behavior and current app context.
- [x] Page/capability map, selected/visible IDs, search/filter state and UI acknowledgements.
- [x] Global voice glow and controls while chat is closed.
- [x] Automatic extraction, embeddings, semantic retrieval and visible memory controls.
- [x] Editable preferred name across text and voice.
- [x] Smoother GPT-Live assistant captions.

### First: harden the daily-use foundation

Owner reordered the work: finish connected-workspace expansions before the large
voice/device/operations acceptance round, then the seven-day owner pilot. Internal
cost tracking and enforcement are off for development. Reconciliation before
re-enabling accounting and vector-index work remain later.
The first three code findings were confirmed by reading the current implementation;
device acceptance items are outstanding checks, not claims of observed failures.

- [x] **Checkpoint the current deployed work before the next coding batch.**
      Committed and pushed the memory/UI/deep-sleep baseline to main as
      b9d1a4cd63216790a473f1b94b7101ee93e205fa before changing app code.
- [x] **Raise the multi-action allowance and explain limits.** Replaced four
      calls with 100 tool calls and 30 planning rounds per request, configurable
      through JARVIS_MAX_TOOL_CALLS_PER_REQUEST and
      JARVIS_MAX_MODEL_ROUNDS_PER_REQUEST. Text/GPT-Live have one final summary
      call after the planning allowance. Reads count toward the tool allowance.
      Task reads paginate; limits preserve receipts and expose partial results.
      Larger allowances trade more possible model work for fewer interrupted
      batches; explicit cancellation still applies. Budget checks are disabled in local development.
- [x] **End voice by speaking.** Standalone "goodbye," "thank you" or "that's all"
      should end voice, release the microphone and return to wake listening for
      "Hey, Eri" or "Eri." Exclude quoted phrases and thanks followed by another
      request. Preserve committed actions and make pending work status clear.
- [x] **Optional task due times.** Add a time and timezone with the due date in
      storage, tools, forms and displays. Preserve date-only tasks; a due time
      does not create a reminder. Validate timezone and date/time edits.
- [x] **Complete Eri's access to existing actions.** Expose the existing
      memory.correct, memory.forget, notification.read, notification.snooze and
      notification.dismiss commands through the shared model-tool registry.
      Preserve current owner checks, revisions, idempotency and truthful results.
      Cover text, Realtime and GPT-Live delegation through the same contract.
      PRD: T01 and the shared tool gateway.
- [x] **Prevent stale memory context.** Retrieval currently snapshots facts before
      awaiting the cloud query embedding. Recheck canonical source visibility,
      suppression and memory revisions after that await and before returning or
      injecting context. Test correction/deletion during a delayed lookup.
      PRD: M05 and section 11.4.
- [x] **Weekly-review failure visibility and clarification cooldown.** Failed
      reviews now show their state, last successful run and Retry review.
      Preparing context no longer consumes the daily offer. The cooldown starts
      when an assistant question contains the candidate spellings; transcript
      production is not proof that audio was heard. Broader semantic follow-through
      remains part of the later memory-quality work.
- [x] **Cost visibility and optional-work controls.** Show active reservations,
      uncertain holds, calendar-month spend and a pace-based projection separately.
      Add the 80% warning, 95% optional-memory deferral and a version label on new
      usage estimates. Deferred memory jobs resume durably when room returns.
      Cancellation during initial retrieval releases the unused chat allowance;
      cancelling an in-flight paid request keeps its outcome marked uncertain.
- [ ] **Deferred: reconcile holds before re-enabling accounting.** September 11,
      19:16 UTC snapshot: $2.551407 recorded estimated spend and $124.267111 held
      for unconfirmed sessions. Do not treat held amounts as confirmed charges.
      Retain them until provider evidence supports settlement. Complete the
      provider-price/transcription accounting audit. The seven-day comparison moves to the post-expansion pilot.
- [ ] **Finish voice and site-control recovery acceptance.** Exercise GPT-Live
      through long conversations/provider session endings, interruptions,
      a correction arriving during a pending action, close/reopen, network loss and
      API restart. Confirm receipts restore committed outcomes without replaying
      actions. Check standalone Eri, 30-second mic handoff, current selection,
      browser action acknowledgments and recovery on the actual mobile device.
      Extend existing record editors/settings coverage after these paths are sound;
      search, filters, navigation and chat control already exist.
- [x] **Restore the current encrypted backup in isolation.** Verified
      jarvis-20260911T202655Z.pgdump.enc at migration 0006_workspace_accounting,
      including tasks, projects, schedules, notifications and memory reviews.
      No worker or reminder dispatch ran against the restored database.
- [ ] **Prove operational recovery and reminder delivery.** Test actual Windows
      reboot startup and locked-phone Web Push, including permission denial and
      opening the notification. Verify an encrypted backup of the current schema
      restores in an isolated stack with dispatch disabled. Keep an encrypted
      recovery copy and its separately stored recovery key off this PC.


The hardening exit is evidence of reliable saved actions, current memory context,
visible failures, recoverable state and understood spending. Automated checks,
real-device results and the pilot each establish different parts of that evidence.
The narrow GPT-Live context enhancement below follows correctness fixes and can
be exercised during the pilot.

- [x] **Weekly deep sleep, first version.** Durable DBOS review each Sunday at
      3 AM in the owner's home timezone, with one catch-up after downtime.
      Merge exact duplicate assertions while preserving source links. Queue
      similar-spelling candidates for clarification; never select a name just
      because it sounds similar. Memory shows review questions, a corrected-fact
      editor, “These are different,” “Ask next week,” and Review now.
      Eri receives at most one optional review offer per day and can resolve an
      explicit answer with the shared command tools. Settings can disable it.
- [x] **Retire the legacy Qdrant bridge.** Remove its active search adapter, UI,
      configuration, Compose service and backup mount. Old volumes/backups remain
      offline recovery artifacts; the running app does not read them.
- [x] **Quiet memory updates during GPT-Live.** Feed relevant retrieved facts
      through session.thinking.append while speech continues, with debouncing,
      small payloads, freshness/deletion checks and cancellation on close.
      Implemented with a one-second debounce, bounded factual payloads, revision
      checks and append acknowledgments. New protocol/lifecycle tests pass; the
      owner's real-provider pilot still needs to assess the resulting conversation.
      An append acknowledgment is estimated context delivery, not a guarantee
      that the end of the current response or the next words use the entire update.

### Next: a connected personal workspace

The following batch covers linked/searchable notes, contextual task editing,
projects/work types/subtasks/assignees, Google Calendar availability, smarter
notifications, and an authenticated API/MCP connection for other bots.

Build these as small successive batches in the listed order, carrying the same
actions into the web interface and Eri's tools with each release.

- [x] **Unified work-item experience and organization, first release.** Work combines
      tasks and alerts while preserving schedules, delivery records
      and completion history. Schema 0010 gives every alert a task and repeating routines a template. Real projects, work types, subtasks, assignee labels,
      tags, priorities and status filters are implemented. Tasks sort by priority
      and deadline; planned dates are separate and visible in day/week/calendar views. Reminders sort by scheduled time. Existing project labels are
      migrated. Scheduled agent execution and additional sort modes remain later.
- [x] **Contextual task editing, first release.** Resolve selected, visible and
      recently discussed task references within a conversation, re-read current
      records, preserve ambiguity and revision checks, and edit groups atomically.
      The web selection/bulk editor and shared Eri tools use the same commands.
- [x] **Typed editor and settings controls.** Deployed in the conversational
      planner release with 85 acknowledged browser actions.
- [ ] **Remaining reference/control quality.** Measure references in real use,
      finish archived-note filter parity and update controls for new detail cards.
      These follow-ups are included in Batch 1 above.
- [x] **Calendar and day agenda.** Tasks and reminder occurrences share a month
      calendar, selected-day agenda, filters and conversational controls without
      requiring Google sync.
- [x] **Project board and timeline views.** Deployed in the conversational
      planner release; see [PLANNER_VALIDATION.md](PLANNER_VALIDATION.md).
- [x] **Linked notes, first release.** Editable notes linked to tasks/projects and conversations,
      tags, Eri-readable source content, and to-do extraction with provenance and
      duplicate prevention. Cloud embeddings and hybrid semantic/keyword search
      use PostgreSQL JSONB chunks; keyword lookup remains available during provider
      failures. Goals/projects and note-to-note links with backlinks are now available. Authored notes stay separate from personal memory assertions.
- [ ] Measure note retrieval and extraction quality on real owner content before
      the later pgvector migration. Add richer source lifecycle and note-to-memory
      review only if the owner wants authored notes to supply learned facts.
- [x] **Google sign-in and read-only Calendar implementation.** Persistent owner
      sign-in, separate Calendar consent, current availability, deadline conflicts,
      source links/freshness and private-host-compatible polling are deployed.
      Supported recurring exceptions, deletions and invalid cursors are covered.
      Pairing remains a recovery route.
- [x] Configure the real Google client and complete first account-link/sync checks.
      Optional Google event edits and selective local-entry publication are implemented.
- [ ] **Smarter notifications.** Flexible/natural-language snoozing, bundling or
      digests, priority levels and preferences. Extend the existing simple snooze
      while preserving delivery/completion history and recurring behavior.
- [ ] **External bot API/MCP.** Personal-Linear-style task tracking for other AI
      agents. Reuse the current REST/domain command path, add revocable scoped
      bot access, documented contracts, idempotency, actor/audit attribution and
      an MCP adapter. Share the capability contracts with Android.
      Start after the work-item contracts stabilize, so outside bots use the same
      rules and recorded results as the website and Eri.
- [ ] **Bounded agent execution, then native Android.** The request execution,
      progress, cancellation and stored-results foundation moved to the current
      batch. Broader autonomous agent assignments remain later. Reuse established app/tool contracts
      for Android; native background wake and notifications need their own device
      acceptance. Home Assistant, finance and broader research/capture integrations
      follow the core workspace unless the owner changes priorities.

### After expansion: verify, then pilot

- [ ] Complete real-use voice recovery and actual-device/operations checks listed above
      after the unified workspace and calendar expansion.
- [ ] **Run the PRD owner pilot and record measured results.** Seven days with at
      least 50 successful task/reminder interactions; report voice failures,
      unwanted speech, interruptions, delivery results and per-category costs.
      Use section 16's fixed command/failure suites for duplicate effects,
      truthful confirmation, latency and source attribution. Mark unmeasured
      targets explicitly; cloud memory supersedes the old GPU-specific setup.

### Accepted design boundaries

- [x] **Notes and memory:** share source links, tags and retrieval infrastructure,
      but retain authored notes as editable source records and personal memory as
      derived assertions. Define revisions, back-links, deletion and reindexing.
      Owner accepted this direction. Authored notes, evidence-linked tasks and semantic search are implemented; richer graph links are covered by the productivity batch above.
- [x] **Tasks, reminders and agent work:** owner accepted one work-item interface with
      owner/agent assignment, optional schedules and notifications. Keep due dates,
      recurring schedules, execution attempts, delivery and completion as distinct
      concepts underneath. Preserve quick standalone reminder capture (which creates a linked task) and existing history.
      Do not delete the reminder system merely to reduce the number of UI sections.
- [ ] **Agent assignees:** define execution scope, allowed actions, running/failed/
      completed status and result references before assignments launch automation.

### Memory audit before this batch — September 11 (historical)

Read-only inspection of the live deployment found history and memory learning
enabled, a fresh worker heartbeat, **0 visible canonical memory entries**, and
**66 completed extraction jobs**. The saved sources included **65 user messages**
and **59 assistant messages**. None of the saved user messages matched the
old extractor's exact opening phrases, so the jobs completed without creating
memories. The worker is running; extraction is a deliberately narrow initial
implementation, not the full R3 memory system.

Explicit “remember this” tool capture, manual memory capture, correction and
forgetting exist. The old automatic capture checked only prefixes such as
“remember that,” “I prefer,” and “I live in.” Normal conversational wording may
never match. Private conversations are excluded from automatic history/learning.
At that audit, canonical search was lexical/full-text only. Legacy memory is
a separate read-only search bridge and is only queried when search text is given;
its records do not populate an empty canonical-memory view.

This batch replaced that implementation and backfilled eligible saved history.
The live system now has cloud extraction and embeddings, semantic search and
pre-response context. Saved counts and processing counts are reported separately.

## Requested next: full app control and Android

- [ ] Expand beyond today's open/highlight tool into full conversational site control:
      searches, filters, sort order, views, record editors, forms, and settings.
      CopilotKit's frontend tools and app context are integrated now. Extend richer
      editors/settings, multi-step recovery and Android contracts; adding another
      agent runtime remains explicitly acceptable to the owner.
- [x] Use typed, authorized app actions with completion acknowledgments and
      visible outcomes. Carry current page/filter/selection state into agent context.
      Further multi-step/editor/settings coverage remains in the expansion task.
- [ ] Design those capabilities as shared contracts for the web app and a future
      native Android app, so both can be operated by talking to Eri.
- [x] Move voice model and voice selectors out of chat into Settings; match the
      voice list to the selected model and validate it again on the server.

## Next: daily-use polish and account access

- [x] Make the owner's preferred/display name editable in Settings and use it
      consistently in greetings, text, GPT-Live, and Realtime. The current saved
      name stays until edited; prompts no longer hard-code Davin.
      Also saved in the app as “Make my preferred name editable in Settings.”

- [x] Implement Google OAuth/OpenID Connect restricted to the account linked by
      the owner, with persistent sessions and device revocation.
- [ ] Complete the real Google client setup and account-link acceptance before
      closing the app task “Add Google account sign-in to Jarvis.”
- [ ] Verify the added standalone “Eri” phrase on the owner's actual device,
      including mic handoff after the timeout. Existing Hey Eri is owner-confirmed;
      parser, idle behavior and automated voice checks pass.
- [ ] Verify GPT-Live across long conversations, interruptions, late task
      corrections and different voices. Realtime comparisons are paused.
      Earlier real-provider evidence remains recorded above.
- [ ] Test locked-phone Web Push on the actual phone, including permission,
      delivery, opening the notice, and recovery after a connection gap.
- [ ] Validate long conversations on the actual device, varied speech, pauses,
      and interruption timing. Provider/network interruptions remain possible
      with no total-duration/turn cap and the new 30-second quiet timeout.
- [ ] After expansion and voice/device/operations acceptance, run the seven-day usage/cost and voice-quality pilot; record issues and
      tune the gate and personality from actual use.
- [ ] Test startup after an actual Windows reboot.
- [ ] Keep a recovery copy of encrypted backups and the recovery key off this PC.

The CopilotKit and Android requests are also saved under the “Eridani roadmap”
project in the app:
“Expand Eri's site control with CopilotKit” and
“Plan Eridani's Android app with shared voice controls.”

## Later PRD work

- [ ] **PostgreSQL vector index after base hardening.** Use pgvector with a
      cosine HNSW index for canonical memories and later note chunks. Migrate the
      existing 512-dimensional cloud embeddings, retain lexical fallback, enforce
      owner/source/deletion filters, and benchmark recall and latency before rollout.
      PostgreSQL remains canonical storage; Qdrant is not the planned future store.
      Reference: [pgvector](https://github.com/pgvector/pgvector).
- [ ] Extend deep sleep beyond exact duplicates and spelling candidates:
      cross-sentence entity/alias reconciliation, linked-fact corrections,
      contradiction review, measured semantic candidate retrieval/reranking and
      reliable clarification follow-through. The first version does not silently
      merge merely similar facts or run an LLM over the whole memory collection.
- [ ] Connect Home Assistant with explicit device mappings.
- [x] Google Calendar and availability code is deployed with outbound polling;
      real credentials and owner consent remain open in the current batch above.
- [ ] Add bounded, durable research/planner jobs with progress, cancellation,
      saved artifacts, sourced web research, and replay-safe child delegation.
      GPT-Live's short conversational task delegation is not this R4 job system.
- [ ] Add read-only finance imports with idempotent ingestion and reconciled
      aggregates, then broader integrations.
- [ ] Expand memory context brokering, source/relevance feedback, and measured
      retrieval quality. The current capture/search layer is only part of R3.

## Git checkpoint

Before the caption/recovery changes, the working foundation was committed as
`8d41337de35ef5ad6dc01d5f01f2f9daade1c7c1` and pushed to `origin/main`.
The remote previously used `master`; `main` was created at the owner's request.
Before this batch, caption/recovery, GPT-Live, daily-use polish and the roadmap
were committed and pushed to `origin/main` as
`404c230ddd9b7dae48fab4c78d6a5c2e87ea3707`. Remote equality was verified before
new edits. The memory/UI/idle/name and weekly-review work was subsequently
committed and pushed to main as b9d1a4cd63216790a473f1b94b7101ee93e205fa before
this hardening batch. The following source commit records the hardening work.

## Maintenance rule

Record completed work and concrete next steps here after substantive changes.
Keep detailed operational evidence in JARVIS_IMPLEMENTATION.md; the PRD and
upgrade decisions remain the design references. Environment files hold secrets
and deployment settings, not an alternate personality.
