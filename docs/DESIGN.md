# Eridani design system — "Orbit"

Eridani is a personal planner, task tracker, linked notebook and calendar with a conversational assistant (Eri). People open it to see what today holds, capture something quickly, or find and change a piece of their work. The interface should feel calm and legible like a well-kept paper planner, with one distinctive touch taken from the brand mark: the iridescent orbit (lavender → sky → mint → blush) around a small central star.

**Spend the boldness in one place.** The iridescent orbit appears only in three spots:
- the day ring on Today,
- the Eri orb (chat launcher and assistant avatar),
- the brand mark.

Everything else is quiet: paper surfaces, charcoal ink, a single violet action color, and hairlines. Never use gradient washes as decoration on cards, buttons or backgrounds.

## Tokens

Define these tokens once on `:root` in `src/theme.css`, and redefine them for dark mode under `:root[data-theme="dark"]` and under `@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) }`. The theme follows the system by default; Settings › Profile has a Light / Dark / System choice, stored in localStorage under `eridani-theme`.

| Token | Light | Dark | Use |
|---|---|---|---|
| `--paper` | `#F7F7FA` | `#121219` | page background |
| `--surface` | `#FFFFFF` | `#1A1A24` | panels, cards, rows on hover, dialogs |
| `--sunken` | `#EFEEF4` | `#15151E` | inputs, segmented track, code, empty wells |
| `--line` | `#E4E3EC` | `#2A2A39` | hairlines, borders |
| `--line-strong` | `#D2D0DE` | `#3A3950` | input borders, dividers that must be seen |
| `--ink` | `#23242C` | `#ECEBF5` | primary text |
| `--ink-2` | `#50525F` | `#A6A5B8` | secondary text, metadata |
| `--ink-3` | `#6E6F7C` | `#8C8B9D` | tertiary text (counts, dates, footnotes), placeholders, disabled; at least 4.5:1 on paper and surface |
| `--violet` | `#5B4BD6` | `#A99BFF` | primary actions, links, focus, selection |
| `--violet-ink` | `#FFFFFF` | `#16142B` | text on violet |
| `--violet-soft` | `#ECE9FD` | `#26223F` | selected row, active nav, soft buttons |
| `--due-overdue` | `#C2315B` | `#FF8FAE` | overdue dates (with a soft bg `--due-overdue-soft`: `#FBE8EE` / `#3A1E29`) |
| `--due-today` | `#9C5F0B` | `#F5C36B` | due today (`--due-today-soft`: `#FBF0DD` / `#352915`) |
| `--done` | `#1C7B60` | `#7FDDBF` | completed / success (`--done-soft`: `#E3F5EE` / `#16302A`) |
| `--danger` | `#B42318` | `#FF9C92` | destructive actions, errors |
| iridescent stops `--iri-1..4` | `#B9A4F8` `#8CC8FA` `#9BE3CF` `#F6B9CF` | same | only in orbit elements |
| `--orbit` | `conic-gradient(from 200deg, var(--iri-1), var(--iri-2), var(--iri-3), var(--iri-4), var(--iri-1))` | same | |

Shape and depth:
- Radius encodes hierarchy, not one value everywhere: `--r-sm: 6px` (chips inside rows, kbd), `--r-md: 10px` (inputs, buttons), `--r-lg: 16px` (panels, dialogs, note cards), `999px` (pills, avatar, orb).
- Shadows only on floating layers (dialogs, popovers, chat panel, toasts): `--shadow-float: 0 1px 2px rgb(20 18 40 / .06), 0 12px 32px -8px rgb(20 18 40 / .18)`. In dark mode use a darker shadow plus a 1px `--line` border. Panels and rows sit flat on hairlines.

Spacing uses a 4px base: 4, 8, 12, 16, 20, 24, 32, 40, 56. Rows are 44px minimum, and every touch target is at least 40×40.

## Type

One family: **Outfit** (self-hosted `@fontsource-variable/outfit`; the CSP allows `font-src 'self'` only), falling back to `ui-sans-serif, system-ui`. Its geometric single-storey forms match the wordmark.

| Role | Size / line-height | Weight | Tracking |
|---|---|---|---|
| Display (Today date) | 40/44 (28/32 phone) | 600 | -0.02em |
| Page title | 28/34 (24/30 phone) | 600 | -0.015em |
| Section title | 17/24 | 600 | -0.005em |
| Body / row title | 15/22 | 450 | 0 |
| Secondary / meta | 13/18 | 450 | 0.005em |
| Small chip / caption | 12/16 | 500 | 0.01em |

- Use sentence case everywhere. No all-caps labels, no tracked-out eyebrows above headings, and no middle-dot meta strings; use chips or separate spans with spacing instead.
- Use `font-variant-numeric: tabular-nums` for times, dates and counts.
- Keep prose at a measure of at most 72ch (note bodies, descriptions).

## Components (classes in `src/theme.css`)

- **Buttons:** `.btn` is the base: 36px tall (40px on touch devices), `--r-md`, 14px/500, icon gap 8px.
  - `.btn-primary`: violet fill, violet-ink text.
  - `.btn-soft`: violet-soft fill, violet text.
  - `.btn-ghost`: transparent, ink-2 text, sunken on hover.
  - `.btn-danger`: danger text, soft danger on hover.
  - `.btn-icon`: square 36px, `aria-label` required.
  - Hover darkens or lightens by about 6%. Active presses in 1px. Disabled is 45% opacity.
  - Focus is always a visible `outline: 2px solid var(--violet); outline-offset: 2px` on `:focus-visible`.
  - Buttons never use a gradient or a drop shadow, and their text never ends in an arrow glyph.
- **Inputs:** `.field` is a wrapper with a label above (13px ink-2) and an optional hint below. Inputs, selects and textareas sit on `--surface` with a 1px `--line-strong` border and `--r-md`, and are 40px tall. On focus the border turns violet with a 3px violet-soft ring. Selects use a custom chevron (no native arrow styling mismatch).
- **Chips:** `.chip` is 24px, `--r-sm`, 12px/500, on sunken with ink-2 text. Variants:
  - `.chip-due-overdue`, `.chip-due-today`, `.chip-done` use the soft backgrounds with their ink colors.
  - `.chip-home` shows the record's home path with a small type glyph.
  - Chips wrap; a row of chips never overflows horizontally.
- **Segmented:** `.segmented` is a track on sunken with `--r-md`. The selected option sits on surface with a subtle 1px line. Use it for view switches (List / Board / Timeline, Month / Week / Day).
- **Tabs:** `.tabs` are text tabs with a 2px violet underline on the active tab, 15px/500 ink-2, and the active tab in ink. They scroll horizontally on phones without a visible scrollbar.
- **List rows:** `.row` is a 44px minimum flex row, 12px horizontal padding, with a hairline between rows (not boxed cards). It contains:
  - a leading check ring (`.check`): 20px circle with a 1.5px `--line-strong` border, a violet border on hover, and a filled done color with a checkmark when complete;
  - the title (15px, ink, wraps to at most 2 lines);
  - a trailing area with meta chips that wrap below the title on narrow widths;
  - hover or focus-within reveals quiet `.btn-icon` actions.
  - A selected row uses `--violet-soft`.
- **Panels:** `.panel` is a surface with a 1px line and `--r-lg`, padded 20px. It has a header with a section title, an optional count in ink-3, and an optional trailing ghost action. Use panels to group information on Today and in detail views, not for every list item.
- **Note cards:** `.note-card` is a panel variant: title 16/22 600, a 3-line clamped preview in ink-2, and tags as chips at the bottom.
- **Dialogs** (shared `Dialog` in `ux.tsx`): surface, `--r-lg`, `--shadow-float`, a 24px header with title and close `.btn-icon`, a body, and a sticky footer with right-aligned actions (primary last). On phones (≤ 700px) dialogs become bottom sheets with a grab handle and a full-width sticky footer above the keyboard.
- **Empty states:** one short sentence that says what to do, plus one `.btn-soft` action. No illustrations and no jokes.
- **Toasts:** surface with float shadow, bottom-center, a 14px message, and an optional action.

## Shell

```text
desktop ≥ 1000px
┌────────────┬───────────────────────────────────────────────────────┐
│ ε eridani  │  [search tasks, notes, people…  ⌘K]     ◷ Activity  🔔 │
│            ├───────────────────────────────────────────────────────┤
│ ◉ Today    │                                                       │
│ ☐ Tasks  12│        page content, max-width 1120px, left-aligned   │
│ ◇ Organize │                                                       │
│ ▦ Calendar │                                                       │
│ ✎ Notes    │                                                       │
│            │                                                       │
│ workspace ▾│                                                       │
│ (D) Davin ▴│                                             (orb Eri) │
└────────────┴───────────────────────────────────────────────────────┘
```

- **Sidebar:** 232px, on `--paper` with a hairline right border.
  - The brand mark at the top is the orbit epsilon SVG plus the "eridani" wordmark in Outfit 600.
  - Nav items are 36px rows with an icon and label. The active item has violet-soft fill and violet ink.
  - Counts (open tasks, inbox) appear as small ink-3 tabular numbers.
  - The workspace switcher is a compact select near the bottom, above the profile.
- **Phones:** the sidebar becomes a bottom tab bar (Today, Tasks, Calendar, Notes, More). The top bar shows only the page title, search and notifications.
- **Eri orb:** a 52px circle filled with `--orbit`, a white star dot, and the float shadow, bottom-right. It's the only gradient-filled control. The chat panel is a floating surface (380×min(720px, 100vh − 120px)) anchored above it, which becomes a full sheet on phones.

## Pages

**Today** (view `today`) is the landing page, and it should feel like opening a planner to today's spread.

```text
┌───────────────────────────────────────────────────────────────┐
│ Friday, October 2                              ◯ day ring 3/8 │
│ 3 due today, 1 overdue, 2 events                              │
│ [ Add a task for today…                             ] (+ Add)  │
├──────────────────────────────┬────────────────────────────────┤
│ Schedule                     │ Due and overdue            4   │
│  09:30  Stand-up with ABC    │ ○ Send Yvette the summary  ⚑   │
│  ─ now ─────────────────     │ ○ Finish the docs   3:00 pm    │
│  15:00  Finish the docs      │ Inbox                      3   │
│  18:00  Water the plants     │ ○ Buy milk                     │
├──────────────────────────────┴────────────────────────────────┤
│ Recent notes   [card] [card] [card]                           │
└───────────────────────────────────────────────────────────────┘
```

- The quick add (placeholder "Add a task for today…") creates a task planned for today (its planned day is today; no deadline is set). Enter or the Add button submits it.
- The day ring is an SVG circle stroked with the orbit gradient. Its arc length is completed ÷ (completed + open due today), with a small star at the current time-of-day position. It shows a count, not a percentage.
- Schedule merges calendar events, work blocks, timed task deadlines and reminders in time order, with a "now" hairline.
- Panels stack in one column below 1000px.

**Tasks** (Inbox / Next 7 days / All, with the existing Today tab kept for compatibility):
- One toolbar row holds the tabs on the left, then the search field, a "Filter" button showing an active-filter count that opens a popover with Collection, Main home, Status and Archived, a "Saved views" menu button, a segmented List / Board / Timeline control, and the Structure button.
- The quick-add input sits directly above the list.
- Rows follow the `.row` spec, grouped under small section headers (Overdue, Today, Upcoming, No date) when sorted by date.
- Board columns are 280px sunken tracks with surface cards.

**Organization:** the same toolbar pattern. Records show type glyphs and their home path; projects show a timeline range bar.

**Calendar:**
- The month grid uses hairline cells. Today's date is a violet filled circle (not an oval).
- Events are compact pills: tasks use a check glyph, reminders a bell, events a solid violet-soft fill.
- The heading row holds the page title, the search field, a "Filters" button (funnel icon; the count appears only when a filter is active) and an "Actions" menu (New event, New task, Task reminder), using the same toolbar controls as Tasks.
- Week and day views use an hour grid with a now line. In the narrow week columns, overlapping items that start within 30 minutes share a row side by side (time hidden, title on up to two lines, full time in the tooltip); an item that starts later cascades over earlier ones with a small indent. The day view keeps side-by-side lanes.
- Month / Week / Day is a segmented control, and the month title sits beside the prev / today / next controls.

**Notes:**
- A list rail (All notes, saved lists, Uncategorized) sits on the left on desktop and becomes a select on phones.
- A responsive grid of note cards fills the main area (min 260px columns).
- The note editor is a wide reading measure with the title at page-title size.

**Detail cards** (task, record, note):
- A two-column dialog: the left column is content (title as an inline-editable 22px heading, then details and related records).
- The right column is a properties list of compact label/value rows that edit in place. Use a field look only on focus, not 12 stacked full inputs.

**Settings:** a left section rail on desktop, and stacked sections on phones. Each section is a panel with field rows.

## Accessibility and motion

- Contrast meets AA for text in both themes: body, meta and chip text (including `--ink-3` on paper and surface, and the due/done chip inks on their soft backgrounds) is at least 4.5:1.
- Focus is always visible.
- `prefers-reduced-motion` disables all transitions except opacity.
- Motion is limited to user-triggered changes: dialog open/close (120ms fade plus 4px rise), check completion (the ring fills and the title gets a strikethrough), and the chat panel opening. There are no entrance animations on page load.

## Compatibility constraints

The browser acceptance suites (`scripts/validate_custom_planner.py`, `apps/web/e2e/*.mjs` in the CI list) select elements by role and accessible name, plus these classes:
- `.login-card button.primary`
- `.companion` and `.companion.visible`
- `.messages` and `.message.assistant`
- `.settings-content`, `.usage-report` and `.search-alias-settings`
- `.note-grid`, `.note-card`, `.note-source` and `.note-saved-entries`
- `.timeline-bar`

Keep every accessible name, role and these classes unless you update the fixture in the same change.
