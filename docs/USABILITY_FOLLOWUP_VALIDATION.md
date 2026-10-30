# Usability follow-up validation — 2026-10-27

Implemented against main a1e2ff4. This follow-up does not change the latency or reasoning policy.

## Behavior

- Opening chat and starting a new conversation focus the text input. The launcher focuses during its click gesture so mobile browsers can show the keyboard; the panel's delayed focus no longer steals it. Active voice mode retains its existing microphone/input behavior.
- The read-only `navigation_find` tool searches names across records, tasks, notes, and the calendar. Words such as “task” are hints rather than mandatory kinds. A unique “Weekly check-in” calendar event can therefore satisfy “show me the weekly check-in task.” It returns the exact UI command, never mutation authority. Multi-day projections of the same event count as one match. Distinct recurring or scheduled occurrences remain separate choices; exact calendar-entry IDs select the chosen occurrence even when several share a day. Stale or ambiguous detail selections refuse to open instead of falling back to a different occurrence. Multiple matches and incomplete calendar coverage require clarification. The default calendar range is today through 31 days; an explicit range can use the existing calendar limit. Current backing titles are used; archived records and templates are excluded.
- Google invitations expose Yes / No / Maybe for the connected account's own attendee entry. Organizer fields remain read-only. The existing write queue shows pending, confirmed, or conflict/failure; successful requests refresh the source cache.
- RSVP uses a separate owner/source/generation/ETag-bound token. Only the connected account's self attendee is changed, using `attendeesOmitted`. The selected occurrence/series scope is retained. A provider read confirms the result before completion, and the existing private marker makes retries idempotent.
- A confirmed RSVP has Revert in its detail view and agent Activity receipt. Revert queues a compensating provider write using the saved post-write ETag; an intervening Google edit causes a conflict rather than being overwritten. Activity says reverted only after that compensating write succeeds. Notifications default on and can be disabled for a response; Google still controls any mandatory service messages.

The invitation API behavior is documented in Google's [Events resource](https://developers.google.com/workspace/calendar/api/v3/reference/events) and [Events.patch reference](https://developers.google.com/workspace/calendar/api/v3/reference/events/patch).

## Verification

All database checks used disposable, synthetic PostgreSQL databases. No production calendar or model-provider calls were made.

- 120 targeted backend tests passed: Google writes, navigation, tool refinement, and agent work. Four navigation tests passed again after adding current-title and archived-target coverage.
- Calendar-navigation follow-up: 15 backend and 13 frontend tests passed, covering multi-day Google/local events, distinct same-day recurring/scheduled occurrences, exact-entry contract round trips, ambiguous/stale selection refusal, and legacy unique selections. Frontend type-check passed.
- Eval harness: 95 passed, 2 skipped; exposed command/tool inventory regenerated.
- Python correctness checks passed, including the new RSVP runner.
- Frontend type-check and production build passed.
- Existing shell acceptance passed at 390, 600, 768, 1000, and 1440 pixels, including input focus, New chat, navigation/drafts, and reduced-motion behavior.
- New synthetic Google browser acceptance passed at 1440 and 390 pixels: guest fields remain read-only; Yes/Maybe reaches confirmed state; Revert restores the previous response; no horizontal overflow or browser errors.
- Backend cases cover attendee-only patching, stale-source conflicts, safe inverse conflicts, lost-response retry, token-purpose separation, another guest's exclusion, Activity receipts, and occurrence/series scope.

The new `scripts/validate_google_rsvp.py` builds an isolated fixture from forward-only migrations and runs `apps/web/e2e/google-rsvp.mjs`. CI now runs this beside the existing planner acceptance. The older Google validator's migration-downgrade exercise was not used: current migration safety correctly rejects that historical downgrade.

## Remaining acceptance

- A real phone must confirm keyboard appearance and comfortable focus with its browser/virtual keyboard. Desktop Chromium mobile viewports verify focus, not an Android keyboard.
- A real Google invitation should be checked end-to-end for organizer delivery, recurring-instance/series behavior, and source-side Revert. The synthetic transport verifies our contract but cannot establish provider acceptance.
- Agent routing to the new navigation group is schema-tested; a live voice request remains a normal-use check. No paid eval was needed for this follow-up.

## Follow-up — 2026-10-30

- Chat focus: the launcher and New chat logic now live in `apps/web/src/chat-focus.tsx`
  (`ChatLauncher`, `startNewConversation`, `focusComposer`). A jsdom render test
  (`chat-focus.test.tsx`) checks that `document.activeElement` is the composer at the moment the
  click handler returns, for both the launcher and New chat, and again after New chat's async reset.
  A real phone keyboard is still a device check.
- Cross-kind navigation: a `task_resolve` search (always loaded) that finds no task returns
  `other_kinds`, the `navigation_find` lookup across events, notes and records, so the usual first
  lookup for "show me the weekly check-in task" also reaches the one event. A single selected match
  (from either tool) makes its open tool (`ui_calendar`, `ui_form` or `ui_records`) callable at once,
  without a discovery round. `navigation_find` stays in the navigation group: the always-loaded set
  keeps its under-20 budget. Results
  carry `requested_kind`, `kind_name` and `kind_matches_request` so the reply names the kind only when
  it matters. Typed chat gets the rule from the lookup tools' descriptions and every navigation result
  (the global prompt stays within its size budget); GPT-Live's instructions tell it to delegate
  show/open requests as said and treat a named kind as a hint. Eval `site_controls.26` covers the case with a component contract.
- GPT-Live memory updates: see [GPT-Live sessions](APP_FUNCTIONALITY.md#33-gpt-live-sessions).
