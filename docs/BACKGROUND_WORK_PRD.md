# Reliable background work and action cards

Status: implemented September 16; see [release validation](BACKGROUND_WORK_VALIDATION.md)
for shipped behavior, verification and remaining actual-device acceptance.
Updated: September 16, 2026.

## Outcome

Eri can keep listening while every accepted request has a visible, durable outcome.
Adding a second task must not silently replace the first. Speech interruption,
request correction, cancellation, and ending voice are separate operations.
This batch also includes action cards, public onboarding, and the closely related
UX/recovery work identified in the site audit.

## Baseline evidence before this change

The reported incident has not been replayed from production traces. These are
confirmed code paths that explain how it can occur:

- `apps/api/jarvis/live_voice.py`: every user transcript delta increments
  `input_revision`. The delegation guard blocks the next tool call whenever that
  revision changes and asks the model to reconsider newer speech. This does not
  distinguish a new request from a correction of the first request.
- The delegation prompt says to handle the latest outstanding request. Combined
  with mutable whole-conversation context, that can shift attention to request B
  before request A has been saved. A spoken acknowledgment is not a saved job.
- `delegation_lock` serializes entire backend turns. Delegations wait in memory,
  then read the current transcript; they are not independent durable requests.
  The recorded handled revision can cover newer speech without a per-request
  accounting of which intentions actually completed.
- Closing Live cancels pending work. `conversation.chat` creates a `Job` row but
  runs inside the API process, outside the worker outbox. The row is a result
  record, not a resumable queue entry; stale running chats become failed.
- Domain commands already have stable IDs, revision checks, transactional
  receipts, and a workspace mutation lock. DBOS jobs and an outbox already serve
  reminders, memory, and integrations. Reuse these foundations.
- Current receipts retain results, not a complete reversible change history.
  General Revert needs explicit before/after field changes and inverse rules.
- `apps/web/src/wake-word.ts` uses opt-in, foreground browser recognition and
  currently matches a standalone wake phrase. Test a wake phrase followed by an
  immediate request so words are not silently lost. Background/locked-phone wake
  is a later native-app capability, not a promise of this web implementation.
- The idle timer is 30 seconds, but the UI still contains a 15-second ending label.
  Align the status copy and behavior. Current Live shutdown mutes immediately but
  can retain the media track while awaiting server finalization; test release and
  wake re-entry during slow or failed close acknowledgment.

OpenAI's [Live delegation guide](https://developers.openai.com/api/docs/guides/live-delegation)
confirms that client delegation leaves context, execution, permissions and task
state with the application. Delegation events contain metadata, not task text.
The app can send quiet progress or concise verified results back to Live. Nothing
in that protocol requires cancelling accepted work when speech is interrupted.

## Interview decisions

The owner selected all three recommended behaviors on September 16:

1. Finish accepted jobs after goodbye/voice close and leave their result cards.
   Surface failures or questions without holding the microphone open.
2. Run independent requests concurrently; keep related changes in order.
3. Update cards immediately, then give a brief combined spoken confirmation at a
   natural pause, instead of speaking over the next request.

The owner also requested removal of personal per-conversation private mode. New
personal chats follow the existing account history settings. Existing private
conversations, history-off preferences and shared-workspace nonretention are not
retroactively converted into saved history. This removes the private-mode choice
from the interview; it does not remove account or workspace privacy boundaries.

4. Cancellation stops unfinished work. Saved effects stay in place, with explicit
   Revert for supported changes. Approved September 16.

Implemented defaults for this batch:

- When several jobs could match "cancel that" or "make it Friday", ask which one.
  Clear references can target a job directly; unclear references must not cancel
  the entire queue.
- Initial concurrency is two independent backend jobs per account,
  with a small installation-wide cap, a fair queue, and configurable limits.
  This is a starting point to measure, not a promise that every task is faster.

## Request lifecycle

1. **Capture.** Retain transcript event identity and timeline ranges. A serial
   intake step accounts for every unhandled span and all outstanding requests.
   Allow late transcript fragments to settle; never infer turn boundaries from
   the current two-second display grouping or fixed sleep alone.
2. **Classify.** Separate a new request, addition to an existing request,
   correction, cancellation, clarification answer, and conversational speech.
   Bind corrections to explicit request/record references. Assistant speech and
   imported record content never authorize new work. Unclear destructive changes
   wait for a clarification while unrelated work can continue.
3. **Accept.** Save a request, its initial revision, execution scope and outbox
   entry atomically. Only this state justifies "queued" / "working on it" in an
   action card. Live should avoid implying acceptance before it receives that
   status; it must never call unsaved work completed.
4. **Schedule.** Start eligible requests using bounded workers. Each job gets its
   own stable instruction, relevant context, dependencies and allowed tools.
   The latest utterance must not overwrite another job's instruction.
5. **Execute.** Persist planned action steps and stable command IDs before side
   effects. Recheck permissions and request revision before every write. Save
   results at checkpoints. Reuse command receipts when recovering interrupted
   steps rather than generating new creates from the full transcript again.
6. **Publish.** Persist progress/results, update action cards through the event
   stream, and give Live short verified updates. Reconnection fetches the same
   jobs/cards; delivery acknowledgment is distinct from successful execution.

Intake needs durable event deduplication and a handled-span ledger. Repeated
provider delegation notifications or reconnection must not create another job
for the same request. Repeating the same words intentionally later is not, by
itself, a duplicate. An intake/reconciliation pass must account for a second
request arriving during the first delegation even if no second delegation event
is delivered; unrecognized speech must remain visible as unresolved input.

## Queue, concurrency and corrections

- Parallel workers are bounded executions of the existing selected backend
  profile (Luna or Gemini), not a new agent framework or additional vendor.
  No recursive child-agent spawning in the first release.
- Separate clear requests can run concurrently. Split one larger request only
  where independence is established, with an explicit parent/child relationship
  and bounded fan-out. Simple mutations should not require extra model agents.
- Resolve dependencies before dispatch. "Add a task, then put it in ABC" is
  ordered. A read intended to reflect an earlier change also waits for it.
  Uncertain relationships stay sequential. Retain the current short workspace
  transaction lock even when planning/model calls run concurrently.
- Corrections increment only the target job's request revision. Queued work uses
  the corrected input. A running job checks for corrections before its next
  action. If the old action already committed, apply a new revision-checked
  correction and describe both effects; never claim the first did not happen.
- Cancellation stops work that has not committed. Already committed changes
  remain represented by receipts, with Revert where supported. A remote request
  already sent may finish; show reconciliation status instead of claiming it
  was cancelled. Cancellation races must be explicit and testable.
- An explicit "stop everything" targets the current actor's eligible jobs in the
  stated workspace. It never cancels another member's work by implication.
- Ending speech playback, goodbye, idle timeout, browser close and network loss
  must not silently erase accepted requests. Accepted jobs continue after voice
  ends, as selected by the owner; explicit cancellation is a separate command.
- Queue age, deadlines, provider timeouts, per-job action limits, and instance-wide
  capacity remain bounded. Expired or failed work is visible and recoverable.
  Provider rate limits back off; a noisy user cannot starve other accounts.

## Minimal architecture changes

Extend existing PostgreSQL jobs/outbox/DBOS execution and domain commands.
Do not add Redis, a second orchestration framework, or a separate vector store
for this feature.

Introduce a versioned request record (or explicit agent-job fields on the existing
job model) for actor account, workspace, originating conversation/device, request
revision, payload retention policy, dependencies, parent job, pinned model profile,
status, lease/checkpoint, cancellation flag and timestamps. Keep job steps and
receipt references queryable with stable unique keys. Persist the accepted plan
before a side effect; step replay with changed arguments must fail safely.

Suggested user-facing states: Queued, Working, Needs your input, Waiting for sync,
Done, Partly done, Failed, Cancelled. Keep integration delivery status separate
from the assistant finishing its local work. Avoid exposing internal job labels.

A durable execution principal must identify the real actor and workspace without
relying on a live browser cookie or granting the worker an owner bypass. Recheck
current membership/role, account state and integration authority at execution and
at each write. Role changes/revocation stop remaining work. Switching the visible
workspace never retargets an accepted request. Personal memories and integration
credentials never leak into shared jobs or another member's cards.

Browser-only actions (navigation, filters, opening cards, microphone control) stay
session/device-bound and ordered. Background workers cannot claim these succeeded
when no matching browser acknowledged them. Expire stale navigation instructions;
show an explicit Open result instead of unexpectedly moving a later session.
A fast voice-close/control path must not wait behind the work queue.

Personal per-chat private mode is being retired. Durable jobs still need minimal
persisted execution input, with a documented expiry and cleanup policy for scopes
where account/shared history is off. Do not republish legacy private speech or
shared-chat input as saved chat or learned memory. Preserve existing data-deletion
rules. Store no credentials, full reasoning or unrelated retrieval context in jobs.
Show actor-appropriate details in cards; request inputs are not shared transcripts.

## Action cards and Revert

One compact card per request, with expandable individual changes for batches:
short outcome, affected record links, progress/attention state, and Edit/Revert
when applicable. Active cards support Cancel; unclear targets support a concise
clarification. Cards remain reachable after voice ends and across reconnection.

Edit opens the current inline detail view (click fields to change them), with no
new whole-card Edit/Save mode. Show current record values, not a stale snapshot.

Implement Revert as a new audited compensating command. Capture the original
field-level changes and before/after revisions. Revert only fields still matching
the action's after-state; preserve unrelated later edits and surface conflicts.
For creates with new links/dependents, do not cascade-delete subsequent work.
For batches, list which changes reverted and which need attention. Recheck access.

Define inverse support per command type. Do not offer a universal working Revert
button for actions without an inverse. Google/Linear effects use their existing
outbox, conflict handling and reconciliation; show pending/failed reversal state.
Notification delivery, speech and other irreversible effects cannot be unsent.
Private content deletion still wins over retaining an undo history forever.

## Landing page and related UX in this batch

- Public `eridani.app` explains the product, with a clear sign-in to
  `app.eridani.app`, privacy/terms/support pages and accurate AI-provider disclosure.
- Keep Google login and invite-only accounts. Handle returning/invited users,
  deep links, expired invitations, consent rejection and recoverable login errors.
  Add public-domain DNS/HTTPS to deployment acceptance, not just app-domain checks.
- Provide completion/attention notifications and a compact work indicator. Broader
  notification priority/snooze/digest customization remains in connected workflows.
- Audit desktop/mobile signed-in views and onboarding. Track the audit in root
  `recommendations.md`, including screenshots, severity, evidence provenance and
  what could not be tested. Bring related defects into this batch after review.
- Bring audit UX-01–09, the Tasks breadcrumb fix UX-10, and the invitation portion
  of UX-26 into this batch: global pending/attention visibility, accurate worker/
  connection/provider/sync state, cloud-appropriate integration copy and public
  onboarding. Broader task/calendar/note/settings density work stays in the next
  focused UX batch. See [recommendations.md](../recommendations.md).
- Update Eri's tool descriptions/site context for jobs, cards and navigation only
  after the UI and action contracts are final.
- Harden wake-word and shutdown behavior: recognize "Eri" and "hey Eri" reliably,
  prevent the assistant's own audio from retriggering wake, keep one mic/session
  owner, and return to wake readiness only after voice cleanup finishes. Cover
  natural farewells and contextual agreement, negation/continued requests, repeated
  close, idle timeout, permissions, connection loss and provider-close failures.
  Test wake-plus-request capture, foreground/background transitions and unsupported
  browsers; make current listening/paused state clear without promising locked-phone
  wake support. Align the stale 15-second label with the actual 30-second timeout.
  Ending voice must release media promptly while durable jobs continue.
- Keep independent encrypted R2 backup activation tracked as a release-operations
  follow-up. Native Railway PITR already works; R2 bucket-scoped credentials and
  actual export/download/restore verification are still outstanding.

External bot API/MCP, native Android, automatic task routing, vector-index migration,
open-ended research/sandbox agents, public self-service signup, and Langfuse remain
separate expansions. They can later reuse durable job/receipt contracts.

## Acceptance scenarios

- Add A; while Eri acknowledges, add B: both appear exactly once, with separate cards.
- Add A; "actually make it Friday": one correct target is updated, no duplicate.
- Add A and B; "cancel that" is ambiguous: ask once, keep unrelated work intact.
- Goodbye, idle timeout, network loss, page reload, API restart and worker restart
  preserve accepted work according to the selected retention/close policy.
- Crash before/after command commit and before result delivery: replay gives the
  same receipt. Google/Linear timeout reconciles before attempting another create.
- Independent requests overlap; same-record changes and dependent reads stay in
  order. Multiple accounts remain fair and isolated under queue saturation.
- Revoked member, viewer, expired authority and workspace change never acquire
  broader permissions. UI controls never execute in another device/session.
- Edit opens the right live record; Revert preserves later edits/relationships and
  reports partial or remote failures truthfully.
- Captions, action cards and short voice confirmations stay coherent when results
  finish out of order or arrive while the user is speaking.
- No new personal private-mode toggle or agent control is available. Legacy
  private and shared/history-off boundaries still hold; job inputs follow their
  documented retention policy without leaked transcripts or unintended learning.
- Both backend profiles pass scripted edge cases, followed by actual phone/desktop
  Live interruption/reconnect/end-conversation trials. Realtime stays disabled.
- Public landing, invite flow, Google sign-in, mobile layout, keyboard/focus behavior
  and production origin links pass browser checks. UX screenshots state their
  environment; synthetic acceptance is not presented as production user evidence.

## Delivery order

1. Finish the interview and UX audit; settle partial-cancellation behavior and
   document remaining history-off job retention constraints.
2. Durable intake/queue and recovery, initially conservative execution ordering.
3. Bounded parallel workers, dependencies, targeted corrections and cancellation.
4. Action cards, edit/revert contracts and completion/attention delivery.
5. Public landing/login and related audited UX improvements; update Eri's controls.
6. Focused provider, crash/replay, permission and real-device acceptance, then ship.

## Implementation and release notes

The implementation uses `agent_work`, encrypted `voice_inboxes`, short-lived
`device_bridges` / `device_actions`, and encrypted `action_changes`, alongside the
existing jobs, outbox, commands and DBOS worker. It adds no Redis or agent vendor.
Request input and recovery checkpoints expire after 24 hours. Completed
history-off requests clear their input/checkpoint immediately; closed voice inboxes
clear raw capture after durable acceptance. Unanswered clarification input can
remain encrypted until expiry. Saved effects and their audit receipts remain.
Device context lasts 60 seconds; dispatched UI controls expire after 12 seconds.
Acknowledgment tombstones remain for 24 hours so crash replay cannot reissue an
old screen edit. Browser or remote actions already in flight may finish after a
cancel; cancellation is not a claim to retract an already-dispatched operation.

Revert supports verified sparse local changes and archiving unchanged creations.
Relationship rewrites and external calendar reversals require the record's own
controls; the card explains this limitation. Connected task changes use existing
integration validation/outboxes. Revert never overwrites a later change to one of
the same fields. General external undo is a later expansion.

Public pages use a Cloudflare static-assets Worker without database/API access.
The authenticated app, durable worker and PostgreSQL remain on Railway. See
[public-site operations](PUBLIC_SITE.md) for publishing and rollback.
