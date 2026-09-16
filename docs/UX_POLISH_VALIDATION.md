# Compact planner UX verification

September 16, 2026. This batch follows the UX audit in `recommendations.md`.
The previous external-agent release was already committed and pushed as `5c76eeb`
before implementation began.

## Changes

- Tasks: compact view/filter row, action menu, removable filters, modified saved
  views, title-first capture, planned-day chip and consistent scheduling copy.
- Boards/timelines: column selector with counts, remembered position, optional
  hidden empty/finished columns, shorter phone timeline defaults and sticky names.
- Details: task field feedback, preserved failure drafts and explicit conflict
  comparison; readable notes; inline organization saves; safe Eri navigation.
- Organization: open tasks and next deadline on projects, outcome criteria on goals,
  consistent plural/status/priority labels and optional examples without seeded data.
- Memory: separate queued/running/retrying/failed counts; source read cards; clear
  correction and fact/source deletion choices. No learning-model behavior changed.
- Navigation: workspace-aware task/note/organization links, return trail, shared
  keyboard tabs, page-specific search and mobile chat focus/body-scroll handling.
- Settings: compact sharing steps and roles, connection/account/permission summaries.

## Verification

- TypeScript and production build pass. Existing vendor annotation and large-bundle
  warnings remain; this batch adds no frontend dependency.
- Relevant backend memory, site-control and account tests: 31 passed.
- Frontend regression: 120 passed; the existing disabled-Realtime test remains skipped.
- Main browser walkthrough: 21 checks passed with zero JavaScript errors and no
  document-width overflow in its recorded mobile/desktop states.
- Additional navigation checks: 4 passed (board position after reload, Escape on
  a clean project, missing-record recovery, and rejected unauthorized workspace).
- Screenshot review caught and fixed task-title crowding, mobile organization
  toolbar overflow and intrinsic project-board grid overflow.
- A real concurrent task update exercised preserved local drafts, explicit saved/
  draft comparison and revision-guarded reapplication. Note drafts resisted a
  simulated Eri navigation request; clean notes/projects navigated successfully.
- Memory forgetting preserved the source by default; copying a note link retained
  its workspace context. These checks use the real local authenticated API.
- Browser test uses a separate disposable PostgreSQL database with 260 synthetic
  tasks, 16 projects, a goal and linked note. External services and workers are
  disabled. No production record or integration write is used for UX validation.

The browser walkthrough checks 390px/320px mobile and 1440px desktop layouts,
compact capture, board controls, filters, concurrent-edit recovery, linked-record
return, Eri navigation, protected note drafts, keyboard tabs and chat scroll locking.
A 720px viewport also exercises narrow layout equivalent to a zoomed desktop pane;
it is not a substitute for physical browser zoom or a screen-reader audit.
Evidence is stored in ignored `.runtime/ux-polish/` files.

## Limits

Real Android keyboards/microphones, wake-word/goodbye behavior, background and
lock-screen notifications, formal screen-reader and contrast checks, and live
Google/Linear mutations remain separate acceptance tasks. The owner-specific
Business space rename remains with task routing. R2 credentials/activation and
MCP OAuth consent are unchanged.
