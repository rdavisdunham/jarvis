# Conversation action cards and notifications

September 17, 2026.

## Behavior

Chat cards summarize saved changes, rather than serving as the assistant's reply.
Navigation and ordinary read-only requests keep their normal transcript response.
Requests that need an answer or fail still have actionable controls. Completed
cards show the outcome and essential fields with Edit, Revert and a small Details
disclosure. Activity retains the fuller audit presentation.

Cards are anchored to the originating user message, not sorted by completion
or appended beneath all messages. Typed turns use durable source IDs; voice
uses the original request text and timestamp because its transcript IDs differ
from queued work IDs. Anchors are retained for the conversation. Clarifications
update the original card. A separate conversation-scoped, authenticated and
paginated work feed preserves older cards beyond Activity's latest 50 requests.
Typed backend replies merge into normal transcript messages with durable IDs to
avoid duplicates; Live's own spoken transcript remains the voice reply.

Routine success no longer creates a notification. Existing success entries are
excluded from the notification feed, push eligibility, pending delivery and
morning summaries. Completing a clarification still dismisses its old question.
Reminders, deadlines and actionable failures/questions retain their behavior.

The existing card navigation regression also exposed an asynchronous double-close
path. Detail cards now let the navigation bridge await the save once, then close
synchronously so a later Forward action cannot be closed by stale work.

## Verification

- Frontend: 130 tests passed; one deliberately disabled Realtime test skipped.
- Backend queue, notification and continuation checks: 60 tests passed.
- Production TypeScript/build and Python correctness checks pass.
- All three isolated browser suites pass: custom planner, shell navigation and
  chat activity. Chat checks cover silent navigation cards with visible replies,
  parallel completion in reverse order, collapsed details, real Revert, another
  request below previous cards, reload anchoring and no completion notifications.
- Phone and desktop screenshots were inspected. Browser acceptance runs against
  a disposable database, real application commands/receipts/authentication and
  synthetic queue completion. No model calls or production data are used.
- The new browser suite runs in existing GitHub CI. No database migration or
  dependency was added.

## Remaining acceptance

Use the physical phone with Live to check speech-to-card alignment, rapid requests
and clarification answers. Automated voice transcript cases do not replace a real
microphone session. Existing bundle-size warnings remain.
