# Task tabs, calendar details and touch boards

September 13, 2026. Deployed on the existing schema 0011_productivity_graph.
No database migration, new drag library or new agent runtime.

## Behavior

The Tasks page combines Today, Inbox, Next 7 days and All (formerly Work).
Their rules and the task/alert/event distinction are explained in
[PRODUCTIVITY_SCHEMA.md](PRODUCTIVITY_SCHEMA.md). Existing view links still
select the corresponding tab, and tab switching preserves filters and layout.

Calendar month chips and agenda rows open saved details with Edit at the upper
right. Local task cards include dates, classification, assignee, notes,
relationships and reminders. Created task occurrences can be completed; work
blocks and events do not have an independent completion state. Future routine
projections cannot accidentally complete their template.

Task board grips support mouse, touch and keyboard movement between columns.
Status, project and assignee changes use revision-guarded domain commands.
Empty status columns accept drops and touch dragging scrolls at the edges.
Keyboard: Space to pick up, arrows to choose a column, Space to drop, Escape to
cancel. Existing status selectors remain available. Card order follows the
selected sort; this release does not store manual ranks.

Eri can open a saved calendar card with ui_calendar open_details=true and read
it as mode=detail. ui_form opens the actual editable draft. Patching a read-only
card fails without changing records; normal sparse editing and dirty-draft
protection remain in place.

## Verification

- Backend: 447 passed, one optional skip. Paused Realtime module excluded.
- Frontend: 78 passed, one retained Realtime test skipped. Production build and
  application/test/changed-script Ruff pass.
- New rendered browser acceptance uses an isolated PostgreSQL database and real
  domain commands. It covers exact tab subsets and reload links, mouse dragging,
  keyboard movement/cancellation, Chromium touch input, project movement,
  saved-detail reads and patch refusal, transition to editable forms, sparse
  edits, completion closing the task alert, work-block semantics, month-chip
  clicks, Escape, mobile card layout and page overflow.
- Existing planner acceptance passes all 85 acknowledged actions, including
  boards/timelines, draft preservation, revision conflicts, organization, settings
  and mobile layout.
- Google/Linear browser acceptance passes consent, synced details, mobile views,
  30-second scroll stability, appointment publication/edit/cancellation,
  lost-response retries, work blocks, Linear publication/conflict review and
  unlink/disconnect. Provider calls are synthetic; browser/API/domain behavior
  and database migrations are real.
- The Google test deliberately delays the fresh event response. Cached Google
  date objects are kept out of the normalized date formatter, preventing a
  transient blank page. Guests remain expanded; links and attachments survive
  the transition to fresh details.
- Desktop and mobile screenshots were reviewed. No paid voice/backend eval was
  needed for this UI batch; no new real-device voice acceptance claim is made.

## Deployment

API, worker and PostgreSQL are healthy; the backup service remains running.
All 52 backend source hashes match the running image. HTTPS serves
index-iEx4UWjO.js. The live page passes a read-only desktop/mobile check.

All fifteen checked canonical table hashes are unchanged across deployment,
including 55 tasks and eight alert definitions. Schema remains
0011_productivity_graph and cost tracking remains disabled.

The existing JavaScript bundle-size advisory remains a separate optimization
follow-up. Touch behavior was exercised in Chromium with real pointer dispatch;
a physical Android interaction check is still useful during the next device round.
