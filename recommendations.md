# Eridani UX recommendations

Audit date: 2026-09-16. Scope: the whole site, desktop and mobile, screenshots, and prioritized recommendations.

## Findings and limits

The useful foundation is already here: task tabs, inline task details, keyboard board movement, typed calendar entries, linked notes, and account/workspace separation. The next batch should make accepted work dependable and visible, then make joining and understanding the product easier.

**Production observed:** the login page at https://app.eridani.app at 1440×1000 and 390×844; its Google button successfully reached the real Google sign-in page. https://eridani.app returned `ERR_NAME_NOT_RESOLVED` in the audit browser.

**Authentication limit:** the browser-control runtime failed with a Windows sandbox initialization error. A fresh Playwright browser had no signed-in Google session and required human sign-in. This was **not** an authenticated production crawl as davinator321@gmail.com. No password was requested, identity forged, or production authentication changed.

**Authenticated UI observed:** the existing built frontend and real local API against a newly created, disposable PostgreSQL database. All records were synthetic. Workers and external services were disabled; local pairing was used only in this isolated environment. No production tasks, calendar events, Linear issues, accounts, or settings were modified.

**Coverage:** task presets, filters, list/board/timeline, creation and inline details, bulk editing, all organization tabs, project board/timeline, calendar month/week/day, task and event forms, notes, memory correction, all six Settings sections, sharing/workspace switching, notifications, chat shell and mobile navigation. Sizes: 1440×1000, 390×844, plus one 844×390 landscape detail check.

**Verified effects:** inline description persisted; quick task creation and completion; keyboard movement from Open to In progress; bulk selection; calendar task detail and task time-block form; memory correction; synthetic workspace/invitation creation and workspace switching. Navigating from task search to Notes cleared the prior search. The main walkthrough captured zero application JavaScript errors and no document-level horizontal overflow in its recorded mobile states.

**Not validated:** actual microphone/wake-word behavior, spoken interruption/end behavior, real Google/Linear writes, external-provider failures, accepting an invitation as another Google account, device push delivery, production data scale, or a formal screen-reader/contrast audit. Disabled-provider screens are configuration-edge evidence, not claims that production integrations are broken.

Initial audit-selector assumptions and a later unavailable assertion helper were corrected or checked separately. They are not product bugs. Quick-task reopening was attempted but not verified and is not counted as passing.

**Change during audit:** the parent task separately removed the personal private-chat control locally. A labeled post-change UI check confirms that control is absent and New chat remains. It was not deployed during this audit. Earlier chat screenshots show the baseline. This does not recommend removing shared-workspace, account, or history-retention privacy.

Screenshots and raw evidence are in the ignored local directory [.runtime/ux-audit](.runtime/ux-audit/); they will not appear in a fresh Git checkout. Full-page screenshots of fixed overlays may include background below the viewport. That alone does not prove the overlay is broken.

## Priority and delivery order

- **P1:** affects trust, task completion, access, or a frequent mobile action.
- **P2:** recurring usability friction.
- **P3:** refinement after the essential flows are stable.

Current batch: UX-01–09 and the onboarding portion of UX-26. Small accompanying fixes: UX-10 and stale labels. Next compact/mobile batch: UX-11–25 and remaining UX-26–31. Later: UX-32–34.

Recommendations are proposals and acceptance criteria, not claims these features already exist. The separate [background-work draft](docs/BACKGROUND_WORK_PRD.md) contains execution design.

## Implementation follow-through — September 16

The background-work/onboarding batch now implements UX-01–10 and the invitation
portion of UX-26. See [release validation](docs/BACKGROUND_WORK_VALIDATION.md).
Follow-up browser checks captured 13 public/activity/task/calendar states at desktop
and mobile sizes, with no application JavaScript errors or page-width overflow.
Edit opened the correct inline task detail; Revert archived its unchanged creation;
Cancel left saved effects intact. The private-chat option is removed.

Small density fixes also landed: a compact saved-view menu, larger mobile task and
activity targets, and a shorter month calendar with a selected-day agenda.
These do not claim completion of the broader UX-11–34 recommendations below.
Actual microphone/phone behavior and authenticated production Google/Linear
write trials remain separate from the synthetic browser evidence.

## Current batch — dependable work and onboarding

### UX-01 · P1 · Make each accepted instruction durable and visible

**Evidence:** owner-reported loss of the first add-task request when a second spoken instruction interrupted “let me add that.” The audio failure was not reproduced here; the existing chat shows a general Thinking state rather than an independent request list.

**Change:** persist before acknowledging acceptance. Give every request a card with Queued, Working, Waiting for you, Completed, Partly completed, Failed, or Cancelled. Include the original request and linked results. Accepted work survives voice closure. Independent work may run concurrently; related edits preserve dependencies and order.

**Acceptance:** “Add A,” immediately followed by “Add B,” yields two tracked requests and exactly one of each task after interruption/reconnect/reload. Unconfirmed acceptance is explicitly marked rather than implied.

### UX-02 · P1 · Action cards need exact Edit and Revert semantics

**Evidence:** the owner requested persistent summaries; ordinary mutations currently have brief feedback and generic autosave wording. [Task details](.runtime/ux-audit/task-details-desktop.png).

**Change:** group committed changes by request, show important task fields and before/after values, and open the existing inline detail card from Edit. Batch actions need per-item outcomes. Revert must describe the concrete reversal and preserve unrelated later changes. An external reversal may be a compensating action, not a true undo.

**Acceptance:** successful writes produce one persistent card; failed/uncommitted writes never appear successful. Stale/conflicting reversals explain the conflict. Google/Linear changes distinguish queued, confirmed, failed, and reversal-pending states.

### UX-03 · P1 · Show pending work even with chat closed

**Evidence:** chat can close and notifications are separate, but there is no durable work overview. [Mobile chat](.runtime/ux-audit/chat-mobile.png), [notifications](.runtime/ux-audit/notifications-desktop.png).

**Change:** add a small global activity entry with a pending count and needs-attention badge. Cards update immediately. Combine spoken confirmations at natural pauses instead of interrupting the user for every parallel completion.

**Acceptance:** closing voice leaves pending work visible; reopening or reconnecting restores the same cards; already-delivered confirmations are not repeated; failed jobs remain discoverable.

### UX-04 · P1 · Distinguish Stop speaking, End voice, and Cancel work

**Evidence:** execution is being decoupled from conversational interruption; the owner chose to finish accepted work after voice closes. Actual mic behavior remains untested.

**Change:** Stop speaking silences audio; End voice closes listening/speaking while work continues; Cancel work targets a particular request and reports already-completed effects.

**Acceptance:** Goodbye does not cancel accepted task writes. “Cancel the proposal update” identifies that job. “Stop talking” does not cancel either job. Late callbacks cannot reopen an ended microphone session.

### UX-05 · P1 · Correct timeout copy and explain wake-listening state

**Evidence:** settings visibly says 15 quiet seconds, while actual intended/configured timeout is 30. The terminal idle label also says 15. [Voice settings](.runtime/ux-audit/settings-voice-mobile.png); source: App.tsx and voice-idle.ts.

**Change:** use one timeout value for behavior and copy. Show wake listening as On, Paused during voice, Unsupported, or Permission needed. After a natural goodbye, re-arm wake listening only when enabled. Keep concise voice controls/glow available with chat closed.

**Acceptance:** all copy agrees with 30 seconds; Eri/Hey Eri starts a fresh session after ending; no self-trigger from Eri's output; permission failures offer recovery. Validate on real phone and desktop microphones.

### UX-06 · P1 · Provide a working public front door

**Evidence:** app.eridani.app is a login-only card; the apex failed DNS resolution. No product overview, help, privacy, or terms links are visible. [Production login](.runtime/ux-audit/production-app-desktop.png).

**Change:** put the landing page at eridani.app and authenticated work at app.eridani.app. Explain tasks/calendar/notes/voice with a real product example. Make Sign in primary; clearly disclose invite-only access. Add public privacy, terms, and support pages.

**Acceptance:** both domains resolve with HTTPS; visitors understand the product without authentication; policy/help pages work; Sign in has loading/error states and returns to the intended app destination.

### UX-07 · P1 · Design the complete Google/invitation journey

**Evidence:** production Google button reaches real sign-in. Sharing says after invitation creation that no email was sent. [Google sign-in](.runtime/ux-audit/production-google-signin.png), [invitation](.runtime/ux-audit/sharing-invitation.png).

**Change:** cover Sign in, Accept invitation, Wrong account, Expired/revoked invitation, and Request access. Make “no email is sent” clear before submission; offer a copyable invitation message/link. Show intended account/workspace before acceptance.

**Acceptance:** wrong-account and expired-link states offer recovery without exposing unrelated members. OAuth cancellation returns to a useful screen. Successful login preserves a valid destination/invitation. Automatic email sending is not implied.

### UX-08 · P2 · Remove misleading home-server and pairing copy

**Evidence:** Google setup refers to a home server and “Pairing still works”; Linear says credentials are encrypted on a home server. These edge states are exposed in the isolated fixture and confirmed in source. [Integrations](.runtime/ux-audit/settings-integrations-mobile.png).

**Change:** use cloud-appropriate language such as “Stored encrypted in Eridani.” Unavailable integration setup should direct the user to the workspace owner, not to local-server configuration.

**Acceptance:** cloud login/settings/errors contain no misleading local-machine promises. Technical operator instructions remain outside normal user flows.

### UX-09 · P1 · Make availability/completion labels truthful

**Evidence:** broad Connected, Up to date, and Thinking labels do not describe individual jobs. The disabled-provider fixture simultaneously says a key is required and “Text is ready.” Memory shows learning while its test worker is deliberately paused. [Chat](.runtime/ux-audit/chat-mobile.png), [memory](.runtime/ux-audit/memory-mobile.png).

**Change:** distinguish browser connection, worker availability, external sync, and model availability. Prefer contextual states: Saved; syncing to Google, Background work paused, or Reconnect to receive updates.

**Acceptance:** offline, worker-paused, provider-unavailable, external-write-pending, and completed states are distinct. Text-ready copy follows actual text capability. Failed work cannot coexist with an unexplained all-done indication.

## Compact UX polish recommendations

### UX-10 · P2 · Fix task breadcrumbs
**Observed:** Today shows “Your space › Notifications”; the source fallback also applies to Inbox/Next 7 days. [Today](.runtime/ux-audit/tasks-today-desktop.png).
**Change/acceptance:** resolve all presets to Tasks, optionally including the active tab. Direct links, clicks, and shared-workspace entry must agree.

### UX-11 · P1 mobile · Reduce the control stack before tasks
**Observed:** title, tabs, explanation, saved views, filters, quick add, layout, and creation buttons occupy roughly 500px before the mobile board begins. [Board mobile](.runtime/ux-audit/tasks-board-mobile.png).
**Change/acceptance:** combine view/filter controls; move Save view/Copy link into a view menu; retain one clear capture action. At 390×844, useful task content should be visible initially without losing advanced filters.

### UX-12 · P1 · Enlarge hit areas without enlarging every row
**Observed:** desktop completion buttons measure 18×18px; secondary links and project actions can be under 24px on one axis. Raw geometry counts include background controls behind modals and are not a formal accessibility failure total.
**Change/acceptance:** retain compact visuals but enlarge touch regions, especially completion/close/drag/overflow. Aim for at least 44px intended touch regions for essential mobile actions; check 320/390px, 200% zoom, contrast, focus, and screen readers separately.

### UX-13 · P2 · Simplify task creation
**Observed:** the new-task form exposes almost every property immediately; Save is below the initial mobile viewport. [Creation](.runtime/ux-audit/task-create-mobile.png).
**Change/acceptance:** title plus a few optional chips, then the existing inline detail card; advanced organization collapses. A simple task takes a title and one submit. Keep the primary action usable with the mobile keyboard open.

### UX-14 · P2 · Explain planned date, deadline, reminder, and time block
**Observed/source:** quick capture in Today/Next 7 days supplies today as the planned date. Full task initialization starts undated unless a date is passed. These defaults and four scheduling concepts need visible explanations.
**Change/acceptance:** show a removable “Planned today” chip and consistent capture defaults. Explain deadline = due, planned day = intended work day, reminder = notification, block = reserved time. Successful creation should leave the item visible or link to where it went.

### UX-15 · P2 · Improve autosave feedback and conflict recovery
**Observed:** inline description saving persisted correctly; the header is a generic “Changes save automatically.” [Details](.runtime/ux-audit/task-inline-edit-desktop.png).
**Change/acceptance:** field-level Saving/Saved/Failed, retain drafts on errors, explain Enter/blur/Escape, and compare conflicting versions before reload. Navigation must wait for an active commit without silently losing either version.

### UX-16 · P2 · Make boards practical on phones
**Observed:** mobile starts at Backlog with only part of Open visible; active work is offscreen. Empty project columns occupy large areas. Keyboard task movement successfully persisted In progress. [Task board](.runtime/ux-audit/tasks-board-mobile.png), [project board](.runtime/ux-audit/projects-board-desktop.png).
**Change/acceptance:** column switcher/counts, remembered horizontal position, hideable empty/terminal columns, and preferred active-column focus. Preserve status selectors and keyboard movement as alternatives to drag. Errors and action feedback must match across interaction methods.

### UX-17 · P2 · Improve timeline legibility
**Observed:** narrow task labels truncate heavily; date markers require horizontal exploration. [Timeline mobile](.runtime/ux-audit/tasks-timeline-mobile.png).
**Change/acceptance:** shorter default mobile span, sticky names, full label on focus/tap, persistent Today/navigation controls. Clearly retain the distinction between date markers and reserved blocks, and explain Unscheduled items.

### UX-18 · P2 · Standardize labels and account context
**Observed:** detail cards show lowercase open and owner; priority terms vary between forms/cards/bulk editing. “Personal” names both a private workspace and an organization space. [Task details](.runtime/ux-audit/task-details-desktop.png), [bulk editor](.runtime/ux-audit/bulk-task-mobile.png).
**Change/acceptance:** one status/priority dictionary; Me/person's name instead of owner; clear workspace versus classification wording. Rename Business to Work with the accepted routing work while preserving IDs. Lists, cards, action receipts, and spoken confirmations should agree.

### UX-19 · P2 · Make filter effects obvious
**Observed:** All still applies active filters; many controls sit in an expandable panel. [Filters](.runtime/ux-audit/tasks-filters-desktop.png).
**Change/acceptance:** removable chips, Reset to this tab, and empty-result explanations that distinguish no records from filtered-out records. Mark saved views as modified when appropriate. Preserve the observed clearing of incompatible search when navigating to Notes.

### UX-20 · P1 mobile · Shorten the monthly calendar
**Observed:** the tall month grid pushes selected-day entries far below the initial viewport; narrow cells show mostly icons. [Month mobile](.runtime/ux-audit/calendar-month-mobile.png).
**Change/acceptance:** compact month picker plus visible agenda, or a phone-specific agenda/week default. Keep Open day discoverable; double-tap is an extra shortcut. Show the first selected-day event without passing a screen of empty cells, and retain date/scroll during refresh.

### UX-21 · P2 · Connect a task's calendar appearances
**Observed:** one task appears as planned work, reminder, and deadline on the same day. Existing type badges help, but the rows can look like separate work. [Calendar](.runtime/ux-audit/calendar-month-mobile.png).
**Change/acceptance:** optionally group related entries or show a common task link. Detail cards show linked scheduling together. Completion changes the single task; pure events have no task-completion control; reminder rules remain distinct from tasks.

### UX-22 · P2 · Unify record detail behavior
**Observed:** tasks open readable inline cards; notes open a Save form; project/goal titles open editors. [Task](.runtime/ux-audit/task-details-desktop.png), [note](.runtime/ux-audit/note-detail-mobile.png).
**Change/acceptance:** a shared detail shell with type/title, content, attribution, relationships, and predictable close/back. Reading a record must not create an edit blocker. Long-note drafts may still use explicit Save, with clear dirty state and preserved drafts during Eri navigation.

### UX-23 · P2 · Explain goals and projects in context
**Observed:** organization views have counters/links but little onboarding; cards show “1 goals” and multiple terse actions. [Goals](.runtime/ux-audit/organization-goals-desktop.png).
**Change/acceptance:** short outcome-versus-deliverable examples, next deadline/open tasks on projects, success criterion/progress on goals, correct plurals, and secondary actions in overflow. Do not equate task-completion percentage with a measured goal outcome unless configured.

### UX-24 · P2 · Make notes readable before showing metadata
**Observed:** a short mobile note is surrounded by a large textarea and many metadata fields. [Note mobile](.runtime/ux-audit/note-detail-mobile.png).
**Change/acceptance:** prioritize reading/writing; compact secondary attribution/links. Keep Find to-dos visible with a review-before-create explanation. Show existing task links, avoid duplicate suggestions, and preserve return-to-note navigation.

### UX-25 · P2 · Explain memory sources and processing
**Observed:** “Source-backed,” a learning count, View source, Correct, and an icon-only forget action need more context. [Memory](.runtime/ux-audit/memory-mobile.png).
**Change/acceptance:** You told Eri/Learned from with date/source; separate queued/active/failed/review-needed processing; explain correction and forgetting scope. Future task-routing learning should stay separate from personal facts, as already proposed in the routing PRD.

### UX-26 · P1 onboarding · Polish sharing
**Observed:** sharing works with synthetic data but its long paragraphs and plain stacked forms are visually less finished than task cards. [Sharing mobile](.runtime/ux-audit/settings-sharing-mobile.png).
**Change/acceptance:** clear steps for private account versus shared workspace, name, access, invitation, and copied handoff. Show members/pending invitations/role meaning. Clearly state no email is sent; assignment never grants access; first workspace switch explains why it is empty and retains visible workspace identity.

### UX-27 · P2 · Make integration status actionable
**Observed:** unconfigured Google/Linear cards contain long technical copy. Real connected-provider screens were not authenticated in production during this audit. [Integrations](.runtime/ux-audit/settings-integrations-mobile.png).
**Change/acceptance:** Connected account, calendars/teams, last successful sync, read/write permissions, pending changes, one next action. Separate Google login from Calendar read/edit access. External failures link to action cards instead of looking complete.

### UX-28 · P2 · Complete tab keyboard semantics
**Observed:** Tasks tabs support arrows. In Settings, ArrowRight from Profile left focus/selection on Profile. Other tab sets similarly lack the shared implementation.
**Change/acceptance:** one accessible tab component with roving focus, Arrow/Home/End behavior and associated panels. Validate focus restoration, visible rings and modal trapping; complete a separate keyboard/screen-reader pass.

### UX-29 · P3 · Scope search and shortcut hints
**Observed:** the desktop hint always shows ⌘ K; mobile search is cramped; Search work remains in Settings. [Empty desktop](.runtime/ux-audit/local-empty-desktop.png).
**Change/acceptance:** platform-correct shortcut, mobile search sheet, and explicit per-section search or actual grouped global search. Hide irrelevant search where it has no effect.

### UX-30 · P2 · Make archive/delete/recovery language consistent
**Observed:** archive, completion, cancellation, memory forget, and external deletion differ; terse Edit labels may merely open a detail card. Destructive production actions were not exercised.
**Change/acceptance:** shared wording/placement, Undo for reversible local changes, and explicit scope for recurring-series/external/source deletion. Never claim irreversible side effects were restored. Action history records actor, time and outcome.

### UX-31 · P2 · Clarify phone setup, chat overlay, and permission recovery
**Observed:** the mobile chat consumes most of the viewport; full-page captures also contain background tasks below it. That capture artifact alone does not prove scrolling is broken. Real microphone/push use was not tested. [Post-change mobile chat](.runtime/ux-audit/post-change-chat-mobile.png).
**Change/acceptance:** make mobile chat visibly a sheet/fullscreen mode with consistent body-scroll behavior and an obvious return to work. Offer optional microphone/wake/notification setup with useful denied/unsupported states. Test real Android keyboards, background scroll, locked-screen notifications and permission recovery.

## Later refinements

### UX-32 · P3 · Add record links and navigation history
**Change/acceptance:** stable record links with workspace context, a small return trail for linked cards, preserved list scroll, and helpful inaccessible/missing-record states. Sign-in returns to authorized destinations; links never grant permission.

### UX-33 · P3 · Offer a short optional first-run example
**Change/acceptance:** explain one goal → project → task → note relationship and one reminder/time block. Allow skip/revisit and keep examples clearly separate from real records. Never silently seed a user's tasks or personal memories.

### UX-34 · P3 · Validate realistic scale and accessibility settings
**Change/acceptance:** review hundreds of tasks, long titles, many tags/projects, 200% zoom, reduced motion, screen readers, keyboard-only use and real Android browser controls. Include reconnect with a card open and concurrent account edits. Drafts, focus and list stability must survive refresh.

## Operational and implementation notes

R2 backups remain an operational TODO, not a landing-page feature. Keep the backup/recovery status truthful; this audit did not configure backups, deploy, commit, or push. The broader batch should not imply R2 is complete.

Preserve what already works: inline task cards, keyboard board movement, consistent event-type badges, explicit linked records, and account/workspace isolation. Prioritize visibility and clear behavior before adding more navigation.

The separately implemented local private-chat-control removal is confirmed in [post-change desktop](.runtime/ux-audit/post-change-chat-desktop.png) and [post-change mobile](.runtime/ux-audit/post-change-chat-mobile.png). Preserve established history-off/shared-workspace behavior as designed; this report does not request its removal.

## Screenshot index

All authenticated screenshots below are from the isolated synthetic environment; only the production-prefixed login images are production.

- Public: [desktop login](.runtime/ux-audit/production-app-desktop.png), [mobile login](.runtime/ux-audit/production-app-mobile.png), [Google login](.runtime/ux-audit/production-google-signin.png).
- Tasks: [Today](.runtime/ux-audit/tasks-today-desktop.png), [All](.runtime/ux-audit/tasks-all-desktop.png), [filters](.runtime/ux-audit/tasks-filters-desktop.png), [desktop detail](.runtime/ux-audit/task-details-desktop.png), [mobile detail](.runtime/ux-audit/task-details-mobile.png), [new task](.runtime/ux-audit/task-create-mobile.png), [bulk edit](.runtime/ux-audit/bulk-task-mobile.png).
- Work views: [task board mobile](.runtime/ux-audit/tasks-board-mobile.png), [keyboard move result](.runtime/ux-audit/board-keyboard-moved.png), [timeline mobile](.runtime/ux-audit/tasks-timeline-mobile.png), [project board](.runtime/ux-audit/projects-board-desktop.png), [project timeline](.runtime/ux-audit/projects-timeline-desktop.png).
- Organization: [goals](.runtime/ux-audit/organization-goals-desktop.png), [projects](.runtime/ux-audit/organization-projects-desktop.png), [areas](.runtime/ux-audit/organization-areas-desktop.png), [spaces](.runtime/ux-audit/organization-spaces-desktop.png).
- Calendar: [month](.runtime/ux-audit/calendar-month-mobile.png), [week](.runtime/ux-audit/calendar-week-mobile.png), [day](.runtime/ux-audit/calendar-day-mobile.png), [task details](.runtime/ux-audit/calendar-task-detail.png), [new event](.runtime/ux-audit/calendar-new-event-mobile.png), [time block](.runtime/ux-audit/task-time-block.png).
- Knowledge: [notes](.runtime/ux-audit/notes-desktop.png), [note detail](.runtime/ux-audit/note-detail-mobile.png), [memory](.runtime/ux-audit/memory-mobile.png), [corrected memory](.runtime/ux-audit/memory-corrected.png).
- Settings: [profile](.runtime/ux-audit/settings-profile-mobile.png), [voice](.runtime/ux-audit/settings-voice-mobile.png), [integrations](.runtime/ux-audit/settings-integrations-mobile.png), [privacy](.runtime/ux-audit/settings-privacy-mobile.png), [system](.runtime/ux-audit/settings-system-mobile.png), [sharing](.runtime/ux-audit/settings-sharing-mobile.png), [invitation](.runtime/ux-audit/sharing-invitation-mobile.png).
- Navigation/conversation: [baseline desktop chat](.runtime/ux-audit/chat-desktop.png), [baseline mobile chat](.runtime/ux-audit/chat-mobile.png), [mobile navigation](.runtime/ux-audit/navigation-mobile.png), [shared empty workspace](.runtime/ux-audit/shared-workspace-empty.png), [landscape task detail](.runtime/ux-audit/task-details-mobile-landscape.png).
