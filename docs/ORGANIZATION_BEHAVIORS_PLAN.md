# Organization behaviors plan

**Purpose:** a working reference for implementing the Organization upgrades the owner agreed with Astra (2026-10-03). The full spec is preserved in [Appendix A](#appendix-a-agreed-spec-2026-10-03). Before starting each phase:
1. read this file;
2. check the code against the "Current state" table, since `main` moves quickly;
3. update the status column when a phase lands.

**Ground rules from the spec and the owner:**
- **A behavior activates a feature** (completion, scheduling, a timer, a review cycle). Ordinary information stays a **field**: "Client industry" and "Department" are fields, not behaviors.
- **Calculated totals are calculated fields**, not behaviors.
- **One Home per record.** Behaviors and templates never add a second hierarchy or extra Homes.
- **Eri never silently redesigns types or rules.** Schema changes always go through preview → apply.
- **Keep it simple** (owner, 2026-10-02). Build the smallest useful version, measure and use it, and add complexity only with evidence.
- **No added running cost.** No new always-on services.

## Current state (verified against `main` @ 581d1c7, 2026-10-03; B2 updated in Phase 3, B3 in Phase 4)

Code references use `api/` for `apps/api/jarvis/` and `web/` for `apps/web/src/`.

| Area | Status | Evidence / gap |
|---|---|---|
| **A.** Existing behaviors: work, content, timeline, metric | ✅ Done | `api/structure_schema.py:63` capabilities, bindings `:38-55`, validation `:143-187`; `api/structure.py:82-142`; toggles in `web/TypeEditor.tsx` |
| **C.** One Home, derived path, Home picker, spaces not access boundaries | ✅ Done | `StructureRecord.parent_id`; `parent_chain` `api/structure.py:346`; `web/HomePicker.tsx` |
| **C.** Subtasks, ready-to-complete, no cascading completion | ✅ Done | `api/record_contents.py:74`; `web/RecordCard.tsx:122-127` |
| **C.** Extra relationships and advisory blocking | ✅ Done | relationship `behavior: related\|blocks`; cycle checks `api/record_contents.py:107-133`; surfaced in task recommendations |
| **C.** Reusable fields, per-attachment visibility/inherit/order, inheritance with override, operational fields never inherit | ✅ Done | `field_library` + `library_id`; `api/field_library.py`; `api/structure.py:448-468`; `api/structure_schema.py:168-169` |
| **D.** Browse default, container tabs, Add here, record tree | ✅ Done | `web/Structure.tsx:42`, `web/OrganizationBrowse.tsx`, `web/OrganizationTree.tsx`; `api/record_contents.browse` |
| **D.** Move with contents / move only this record / archive options, preview, guarded Revert, links excluded, external items untouched | ✅ Done | `api/record_contents.py:225-434`; `web/ContentsDialog.tsx`; `tests/test_record_contents.py` |
| **E.** Type editor: Visual + Advanced on one draft, preview/apply, types appear once, nests-itself badge, reuse/reorder fields | ✅ Done | `web/StructureEditor.tsx`, `web/TypeMap.tsx`, `web/TypeEditor.tsx`; `type_layout` validation `api/structure_schema.py:116-130` |
| **E.** "Also allowed in…" row | ✅ Done (Phase 2) | `alsoAllowedIn` `web/blueprint-model.ts`; shown under each type in `web/TypeMap.tsx` and on Blueprint nodes `web/Blueprint.tsx` |
| **E.** Fields as a clearly labeled collapsible group | ✅ Done (Phase 2) | "Fields" summary with count in `web/TypeMap.tsx`; Atlas type inspector Fields group marks "Operational, never inherited" / "Can inherit from home" (`web/AtlasInspector.tsx`) |
| **E.** Moving a type: rearrange the diagram vs change allowed homes | ✅ Done (Phase 2) | `typeMoveChoice`/`allowAndPlace` `web/blueprint-model.ts`; `TypeMoveChoices` in `web/TypeMap.tsx` and the Blueprint drop popover; cycles refused |
| **E.** Affected records shown before placement changes apply | ✅ Done (Phase 2) | placement issues carry `kind`, `title`, `home` (`api/structure.py` preview); `PlacementImpact` lists them with links (`web/StructureEditor.tsx`); Atlas type inspector lists them before a draft change |
| **F.** Learning from titles, descriptions and filing; no silent redesign; structure interview deferred | ✅ Done (record bodies not used) | `api/routing.py:96-148,427,443`; `docs/TODO.md:285-290` |
| **B1.** Organizing container behavior | ✅ Done (Phase 1) | `RecordType.opens_as` `api/structure_schema.py:69`; load upgrade `api/field_library.py` (`default_opens_as`); resolver `api/structure.py` `opens_as()` used by `record_contents.browse` groups and every record payload (`opens_as`: `container`/`item`); `openTarget` `web/structure-types.ts`; control in `web/TypeEditor.tsx`; `ui_records(record_id)` routes containers to Browse (`api/tools.py`); `tests/test_opens_as.py`, `e2e/custom-planner.mjs` |
| **Atlas / Blueprint** (Phase 2) | ✅ Done | `GET /structure/atlas` `api/atlas.py` (+ bot route); `web/AtlasView.tsx`, `AtlasMap.tsx`, `Blueprint.tsx`, `AtlasInspector.tsx`, `atlas-model.ts`, `blueprint-model.ts`; layout "atlas" (≥720px, lazy chunk; Browse stays default); `ui_records(layout="atlas")`; `tests/test_atlas.py`, `e2e/organization-atlas.mjs`. Hooks: template stamps go in `AddHere` (`web/AtlasInspector.tsx`); review pulses/lens use the `review` lens placeholder in `AtlasMap`/`AtlasView` and a skeleton field such as `review_due` |
| **B2.** Reusable template | ✅ Done (Phase 3) | `record_templates` table (migration `0022`, `RecordTemplate` in `api/structure_models.py`); `api/record_templates.py`: `template.create/update/archive/capture` (save as template drops dates, assignees, status, record links and default priority), `record.instantiate` plan → apply under one command with `provenance.template`, grouped Revert through `record.restore_contents` guarded by `restore_guard` (`api/record_contents.py`, `api/action_history.py`); routes `GET /structure/templates`, `POST /structure/templates/instantiate/preview` (+ bot `/external/structure/templates`); Eri `template_list`, `record_instantiate_preview`, `record_instantiate` (`templates` tool group); web `Templates.tsx`/`template-model.ts` (Types & fields Templates section and editor, Save as template in the record card, Start from template in Browse Add here and the collection quick-add, Atlas stamps in `AddHere`); `tests/test_record_templates.py`, `e2e/organization-templates.mjs`. `Task.is_template` remains the separate recurrence template |
| **B3.** Review cadence | ✅ Done (Phase 4) | `RecordType.review = {enabled, every}` `api/structure_schema.py` (load upgrade and older-client preservation in `api/field_library.py`); migration `0023` adds `last_reviewed_at`, `next_review_at`, `review_queued_at`, `review_paused` and an `(owner_id, next_review_at)` index; `api/record_reviews.py`: seeding on apply (spread over the first interval), `created + interval` for new records, `record.mark_reviewed` and `record.review` (snooze, pause, resume, restore) with guarded Revert (`api/action_history.py`), `queue_due_record_reviews` in `api/worker.py`, the `record_review` kind in `api/review_questions.py`; `GET /structure/reviews` (+ bot route), Atlas `review_due`/`next_review_at`; Eri `reviews_due`, `record_mark_reviewed`, `record_review` (`review_cadence` group); web `Reviews.tsx`, `review-model.ts`; `tests/test_record_reviews.py`, `e2e/organization-reviews.mjs` |
| **B4.** Recurring work for any actionable type | 🟡 Machinery exists | `Schedule`/`Occurrence`/template tasks (`api/models.py:187-214`, `api/domain.py:740-765`); not a type behavior; no handling of unfinished children |
| **B5.** Calendar scheduling behavior | 🟡 Machinery exists | `PlanningEntry` with start/end/timezone/all-day bound to `task_id`; not a type behavior |
| **B6.** Time tracking | ❌ Missing | |
| **Calculated fields** | ❌ Missing | `summary()` deliberately never sums effort (`api/record_contents.py:41`) |

**Key data-model facts:**
- **Type definitions** are one JSONB document per owner (`structure_schemas.definition`, versioned by `revision`), validated by the pydantic `Definition` in `api/structure_schema.py`. Adding type-level settings needs **no DB migration**, only a definition upgrade on load (pattern: `api/field_library.py`). The `capabilities` Literal currently allows 4 values with `max_length=4`.
- **Schema edits** go through `StructureProposal` (preview → apply, 30-minute expiry, impact and issues) and `structure.restore`.
- **Records** are `structure_records` (values JSONB keyed by attachment id, `parent_id`, `task_id`/`note_id`, provenance, `sort_order`).
- **Multi-record operations** use the `record_contents` pattern: plan with a hash, apply re-plans and compares, every change goes through `structure.mutate` under one `command_id`, and grouped Revert has a fingerprint guard.
- **Review inbox:** `api/review_questions.py` aggregates review kinds into one Questions inbox with defer/snooze and delivery leases. Periodic queueing is registered in `api/worker.py`.

## Plan

Phases are ordered by the spec's priority and by dependency. The container setting comes first because the visual Atlas needs it. Each phase:
- ships as **one PR to `main`** (branch protection requires the Backend, Frontend and Security checks);
- includes backend tests, web tests and an e2e assertion where the UI changes;
- updates Eri's tool descriptions when behavior changes;
- updates `docs/CUSTOM_PLANNER_IMPLEMENTATION.md`.

### Phase 1: Organizing container behavior (B1)

This comes first because the Atlas (Phase 2) needs an explicit answer to "is this a region or an item?".

**Design:** a presentation setting, not a new hierarchy.
- Add `RecordType.opens_as: "auto" | "container" | "item"`, defaulting to `auto` so existing schemas keep their current inferred behavior.
- It's shown in the type editor as the behavior toggle **"Organizing container"**. This needs no capability slot and has no bound fields: turning it on sets `container`, off sets `item`, and "Let Eridani decide" sets `auto`.
- Default types: Space, Area and Client get `container`; Project gets `container` (it's work plus timeline, but opens as its contents); Task and Note get `item`.
- Add a definition upgrade that sets those defaults only where the owner hasn't edited the type. `auto` everywhere else.

**Wiring:**
- `web/OrganizationBrowse.tsx:45` and `api/record_contents.py:180-184` consult `opens_as` first and fall back to the current heuristic only for `auto`.
- Opening a container record goes to its Browse contents page: Groups, Tasks, Notes, Related, All contents. It also shows the existing `summary()` progress and the details card accessible from a "Details" button.
- An item with children still opens its detail card, which shows a compact "Contents (N)" section.
- Eri: `record_browse` and the `ui_show` / `open_record` behavior respect the setting ("show me ABC" opens ABC's contents; "show me Finish central docs" opens the task).

**Acceptance:**
- Backend tests:
  - an explicit `container` with no children still lists as a group;
  - an explicit `item` with children doesn't;
  - `auto` is unchanged.
- Vitest for the open-target decision.
- e2e: a Client opens to its contents, and a Task with subtasks opens its details.

### Phase 2: Atlas and Blueprint, the visual Organization (the centerpiece)

**Concept prototype:** https://claude.ai/artifact/6ubyHh7DnsV1At9wvc2U4j (sample records). Its source is saved at `docs/prototypes/eridani-atlas.html`. It's a standalone page that uses d3 from a CDN; open it in a browser. Treat the prototype as the design reference, and refine it with the owner before building.

The goal is to replace the plain tree with a way of *seeing and handling* the structure that only Eridani has, grounded in its brand (a star system with orbits). Precise list views stay available for everyday work. **Browse stays the default on phones**; the Atlas is the desktop and tablet canvas.

**Atlas: your actual records**
- **Zoomable map of nested regions** (circle packing). Spaces, Clients and Projects (types that open as containers, from Phase 1) are regions you fly into with a click.
  - Breadcrumbs show the path. Double-click flies out. ⌘K search flies to any record.
  - Only the focused level and its children are labeled (semantic zoom), so large workspaces stay readable.
- **Items as marks**, styled by state:
  - Tasks are dots: hollow when open, half-filled when in progress, filled with the done color when done, and a rose ring when overdue.
  - Notes are rounded squares; Goals are target glyphs.
  - **Subtasks orbit their task as moons**, so a task with subtasks stays an item (Phase 1 rule) while showing that it has contents.
- **Progress rings** on regions show the share of work done inside. **Review-due pulses** appear once Phase 4 lands.
- **Relationships on demand.** Selecting a record draws its extra links: supports and related as dashed violet arcs, blocks as a rose tether with an arrow. Links never look like Homes.
- **Drag to re-home.** While dragging, every *legal* destination glows (computed from the type's allowed homes, excluding its own subtree) and illegal ones dim. Dropping opens the contents choice:
  - **Move with contents (N items move along)**;
  - **Move only this record (its items stay in {old home})**.

  It applies through the existing `record.contents` preview/apply with grouped Revert, followed by an Undo toast.
- **Add here in place.** A selected region's inspector offers "+ {allowed type}" buttons and **template stamps** (Phase 3).
- **Lenses** recolor the whole map: Status, Due (overdue/today), Reviews due (Phase 4) and Type.
- **Focus halo:** the orbit gradient appears only around the focused region (the design system's one bold place).

**Blueprint: the rules**
- **A constellation of types** along the primary diagram path (`type_layout`), each type appearing once. Each node shows:
  - its **behavior glyphs** (✓ work, ¶ content, ⟷ timeline, ◎ metric, ▢ container, ↻ review);
  - a **field count**;
  - a **"nests itself" loop**;
  - an **"Also allowed in …" line**.
- **"Show every allowed home"** draws all permitted placements as thin dashed edges, which makes the permissive defaults visible.
- **Dragging a type** onto another opens an explicit choice:
  - **Rearrange the diagram only**, enabled only when it's already an allowed home;
  - **Also allow {Type} inside {Target}**.

  It never silently changes rules, and cycles are refused.
- **Type inspector:**
  - behavior toggles, including Organizing container (Phase 1) and Review cadence (Phase 4);
  - allowed-home chips with remove (×);
  - a **Fields group**, explicitly labeled and collapsible, marking each field "Operational, never inherited" or "Can inherit from home".
- **Placement impact.** Removing an allowed home that records currently use shows **which records** (titles, links) live there, and Apply stays blocked until they're moved. Changes still go through the existing structure preview → apply.

**Linked views (the "Both" mode)**
- Hovering a type in Blueprint lights up its records on the Atlas and dims the rest.
- Selecting a record highlights its type and outlines its legal homes in Blueprint.
- Selection, lens and focus are shared, so the two trees read as one system with two jobs.

**Implementation notes**
- **Rendering:** SVG with `d3-hierarchy` (pack and tree layouts, about 10 KB). Zoom uses `interpolateZoom`. Add `d3-hierarchy`/`d3-interpolate` as the only new deps, not all of d3. Render only nodes visible at the current zoom.
- **Data:** a bounded server endpoint returns the structure skeleton (ids, titles, types, parent, status/due/review flags, counts) for the current workspace, paginated by depth for big workspaces. Details still load through the existing record APIs.
- **Mutations** reuse `record.contents`, `structure.preview`/`apply` and `record.create`. No new write paths.
- **Accessibility:** every mark is focusable with an accessible name. Arrow keys move between siblings, Enter flies in or selects, and Backspace flies out. Moves have a keyboard path ("Move to…" picker using `HomePicker`). Reduced motion turns off orbits, pulses and zoom animation.
- **Phones:** the Atlas isn't shown below about 720px. Browse (lists and drill-down) remains the phone experience, with the Blueprint drawn vertically.
- **Eri:** `ui_records(layout="atlas", focus=…)` lets Eri fly the map ("show me everything under ABC").
- **This phase absorbs the old "blueprint clarity" items:** the "Also allowed in" row, the Fields label, the explicit move choice and the placement-impact list.

**Acceptance:**
- Vitest for the legal-target computation, the move choice, the open-as rules and the label budget.
- Backend tests for the skeleton endpoint (bounded, scoped, revocation-safe).
- An e2e spec covering:
  - flying into ABC;
  - dragging a task to another project with "move only this";
  - Undo;
  - hovering Task in Blueprint;
  - removing an in-use home and seeing the blocking list.
- A performance check: 2,000 records render and zoom smoothly on a mid-range laptop.

### Phase 3: Reusable templates (B2) — ✅ done

**As built (differences from the design below):**
- Eri gets a read tool `record_instantiate_preview` beside `template_list` and the `record_instantiate` command, mirroring `record_contents_preview` (a preview inside a command would leave a misleading receipt). Template CRUD is owner UI and bot/MCP only; Eri does not edit templates.
- Payload bounds: 4 levels below the record, 100 child records, 200 active templates per workspace. Template values may not set date or assignee fields; captures also drop record links (instance-specific) and default priority 0.
- Instantiate re-validates types, allowed homes and field values against the current schema (`TEMPLATE_OUTDATED`, 409, names the type pair) and the home via `validate_parent`. The preview hash covers the request, schema revision, template revision and home chain.
- Child records are created with a `:template` command suffix so template structure never counts as the owner's own routing evidence; the root uses the plain command.
- Revert archives the created records deepest first through `record.restore_contents`; it refuses when the group guard changed (any edit, core task/note change or new record inside), when links touch the created records, or when already reverted.

**Design (simplest useful version):** templates are **saved blueprints per type**, stored separately from the schema so that editing a template doesn't need a schema preview/apply.
- **New table `record_templates`** (migration): `id`, `owner_id`, `type_id`, `name`, `description`, `payload` JSONB, `revision`, `archived`, timestamps.
- **`payload`:**
  ```text
  { values: {attachment_id: value},        # default field values
    body: "description outline (markdown)",
    children: [ {type_id, title, body, values, children:[...]} ] }   # optional child records
  ```
- **Create from a template:** a new domain command, **`record.instantiate`** (template id, home, title override). It creates the record and its children in one transaction using the `record_contents` pattern:
  - plan → apply;
  - each record created via `structure.mutate("record.create")` under one `command_id`;
  - one receipt;
  - grouped Revert reusing `restore_guard`, which archives the created subtree if nothing has changed since.

  The created records are **independent copies**. They record `provenance.template = {id, revision}` for reference only, and **editing the template never rewrites them**.
- **Save as template:** from any record's menu. Snapshot the record (values, body) and optionally its children, recursively, into a new template for its type. Exclude operational, date-like and assignee values by default, because templates shouldn't carry deadlines. The owner can edit the template afterwards.
- **Template management:** a "Templates" section on the type in the type editor (list, rename, edit outline and defaults, archive) and a simple template editor dialog. No schema proposal is involved.
- **UI entry points:** "Add here" (list Browse and the Atlas inspector, as **template stamps**) and the quick-add menu show "Start from template ▸" when the chosen type has templates.
- **Eri:** tools `template_list` and `record_instantiate` (with preview). "Start a client onboarding project for ABC" picks the template, previews it, then creates.
- **Type behavior flag:** none needed. A type "has templates" when templates exist; keep it simple.

**Acceptance:**
- Backend:
  - instantiate creates the expected tree with independent records;
  - template edits don't change existing instances;
  - Revert archives the created subtree only when unchanged;
  - viewer and bot scope rules apply.
- Web: a template editor test.
- e2e: create "Client onboarding" from a template and see contract/access/kickoff/discovery tasks under it.

### Phase 4: Review cadence (B3) — ✅ done

**As built (differences from the design below):**
- The per-record override interval was left out (keep it simple); a per-record **pause** (`review_paused`) covers "Stop reviewing this record".
- **Seeding:** enabling a type, or changing its interval, sets each active record's first review within the first interval (`now + 1 day + (interval − 1 day) × jitter(record id)`), so nothing comes due the day it is turned on and a large type spreads out evenly. A record reviewed recently keeps `last_reviewed + interval`. New records get `created + interval` in `record.create`; records that reach the type another way (task adoption, imports) are scheduled by the daily queue. The structure preview names the change (`impact.review_changes`) before apply. Disabling hides open items and stops the queue but keeps every date.
- **Questions:** a fourth column, `review_queued_at`, marks when the daily queue surfaced a due review. The inbox lists records queued after their last review: *pending* when due, *deferred* while snoozed (snoozing moves `next_review_at`, so it reuses the existing open/deferred semantics, and `review.defer` works on these keys too). The queue runs in the worker beside the memory and routing queues, at most once per account-local day, outside quiet hours, never while a delivery lease is reserved, and adds at most 20 items per day. Questions belong to the personal workspace, so shared-workspace reviews show on Today, cards, the Atlas and through Eri, not in Questions. Record reviews never trigger chat invitations.
- **Receipts:** `record.mark_reviewed` and `record.review` bump the record revision and journal like other record commands; Revert restores the previous review state through `record.review action=restore` only while the dates are unchanged. Seeding and queueing do not bump revisions.
- **Visibility:** "Due" for Today, cards, the Atlas and Eri is `next_review_at <= now` on an active, unpaused record of a reviewed type; the inbox is the queued subset. The Atlas pulse is a small amber dot on a mark's upper right whose pulse halo turns off under reduced motion; the Reviews lens keeps due records and the regions holding them lit.
- **Eri:** `reviews_due` (with `within_days` for "this week"), `record_mark_reviewed`, and `record_review` (snooze, pause, resume) in their own `review_cadence` group.
- **Bots:** `reviews_due` and `GET /external/structure/reviews` need `records:read`; both commands need `records:write`, plus `tasks:write`/`notes:write` for task- or note-backed records.

**Design:**
- **Type setting:** `RecordType.review = {enabled: bool, every: "1w"|"2w"|"1m"|"3m"|"6m"|"1y"}`. It's shown as the behavior toggle **"Review cadence"** with an interval select.
- **Per-record state:** two nullable columns on `structure_records` (migration), `last_reviewed_at` and `next_review_at`, plus an optional per-record override interval in `values`, or a column if simpler. Turning the behavior on seeds `next_review_at = now + interval`, spread over a few days so a large type doesn't all come due at once.
- **Due reviews feed the existing Questions inbox:** add a `record_review` kind adapter in `api/review_questions.py` with actions:
  - **Mark reviewed** sets `last_reviewed_at = now` and `next_review_at = now + interval`;
  - **Snooze** (1 day / 1 week);
  - **Open** opens the record;
  - **Stop reviewing this record** (per-record opt-out).
- **Queueing:** a periodic `queue_due_record_reviews` in the worker, next to the memory and routing review queues, using the existing delivery leases and quiet hours. Daily granularity is plenty, so no extra load.
- **Visibility:**
  - the record detail card shows "Last reviewed … · Next review …" as separate spans (no middle-dot strings);
  - Today shows a small "Reviews due (N)" panel only when N > 0;
  - the Atlas shows a review-due pulse on records and a **Reviews due** lens.
- **Eri:** tools `reviews_due` and `record_mark_reviewed`. "What should I review this week?" and "Mark ABC reviewed".

**Acceptance:**
- Backend:
  - due computation;
  - mark reviewed reschedules;
  - snooze;
  - per-record opt-out;
  - disabling the type behavior stops new items without deleting history.
- e2e: enable monthly review on Client, fast-forward via the test clock, see ABC in Questions, mark it reviewed.

### Later: only with evidence or an explicit owner request

| Item | Notes when it's picked up |
|---|---|
| **Calculated fields** (e.g. total estimated effort across a project, spending across a trip) | A field kind `calculated`: a sum, count or min/max of a numeric field over the contents (children or subtree), computed on read and cached on change. Read-only. Not a behavior. |
| **Recurring work (B4)** for any actionable type | Reuse `Schedule`/`Occurrence`. The open design question is what happens to unfinished children when the next occurrence is created (copy the template children fresh; leave the old ones with the old occurrence). |
| **Calendar scheduling (B5)** as a type behavior | Reuse `PlanningEntry` (start, end, timezone, all-day) bound to the record's task. Stays distinct from Do/Due. |
| **Time tracking (B6)** | A `time_entries` table and an optional timer. Show estimate vs actual and roll up through contents (needs calculated fields). No billing. |
| **Learning from record bodies (F)** | Routing currently uses titles, types and filing. Add bodies only if suggestions measurably improve; watch the prompt cost. |
| **"Chat about my structure" interview** | Deferred per spec. It would produce the same preview → apply proposal. |

## Cross-cutting checklist (every phase)

- [ ] Definition changes are backward compatible: an upgrade on load, defaults preserve current behavior, and a migration test lives in `tests/test_release_migrations.py`.
- [ ] Authority: viewers can't mutate; bots respect `records:*` plus the core scope rules from the audit; shared workspaces are fine.
- [ ] Every multi-record operation previews, uses one `command_id` and one receipt, and has guarded Revert.
- [ ] Eri: the tool descriptions in `api/tool_catalog.py` are updated; the new tools follow the fast-lane plan (`docs/ERI_LATENCY_PLAN.md`), with cheap reads and `background_request` for heavy multi-record work.
- [ ] UI follows `docs/DESIGN.md` (rows, chips, panels, sentence case, light and dark, phone).
- [ ] The full backend suite, **`pytest evals`** (the eval harness's own tests, including the surface-inventory guard), vitest, build, and all browser suites pass locally before the PR; CI is green before merging.
- [ ] New HTTP routes, commands or read tools: refresh `evals/app/surface-inventory.json` with `PYTHONPATH=apps/api uv run --no-sync python -m scripts.app_eval.inventory`, and make sure tests cover the new surfaces.
- [ ] Update this file's "Current state" table and `docs/CUSTOM_PLANNER_IMPLEMENTATION.md`.

## Appendix A: agreed spec (2026-10-03)

Preserved verbatim from the owner's conversation with Astra.

> I'd add a few behaviors that change what a record can actually do, while keeping ordinary information as fields. A behavior should activate a feature—such as completion, scheduling, or a timer. "Client industry" and "Department" should remain fields.
>
> **The four behaviors available today**
> - Actionable work: gives a type task functionality: completion, workflow statuses, assignee, priority, Do date, Due date, and effort estimate. A Client can enable this, just like a Project or Task. It also becomes eligible for work views.
> - Authored content: connects the record to the notes system. Every record already has a title and description; this behavior makes its content function as a note.
> - Timeline dates: adds a planned start and target finish. Those dates describe a span; they don't reserve time on your calendar.
> - Outcome metric: adds starting value, current value, target, and unit. For example, a Goal could track "3 retained clients → 8 retained clients." This measurement is independent of how many tasks you complete.
>
> You can combine them. Our default Project uses Actionable work + Timeline dates. A research deliverable could use Actionable work + Authored content + Timeline dates.
>
> **The additional behaviors I'd consider**
> 1. Organizing container — my first choice. Makes a record open primarily as a place containing other records, with a contents page, progress, and useful summaries. Currently, records can already contain children, and the UI infers whether to open them as containers. An explicit setting would give you control: a Client opens its projects; a Task opens its details, even if it happens to have two subtasks. This setting would control presentation without introducing another hierarchy.
> 2. Reusable template. Defines a starting structure for new instances: default field values, a description outline, and optional child records. For example, creating a "Client onboarding" project could offer a checklist containing contract, access, kickoff, and discovery tasks. Those become independent records after creation. Editing the template later shouldn't rewrite projects already underway.
> 3. Review cadence. Lets a type periodically resurface for consideration: review clients monthly, goals weekly, reference notes every six months. This would track "last reviewed" and "next review" and feed a review inbox. It would be useful for ongoing responsibilities that don't naturally become completed tasks.
> 4. Recurring work. Makes recurrence a first-class option for any actionable type. You could repeat a maintenance task or a project containing several steps. We already have recurrence machinery elsewhere in the app, but it isn't a general type behavior. This would need careful handling of each occurrence's history and unfinished children.
> 5. Calendar scheduling. Gives a record an actual scheduled start, end, timezone, and all-day option. Useful for appointments, work blocks, classes, and events. That remains separate from Do/Due: "work on this Friday" is different from "reserve Friday, 10–11."
> 6. Time tracking. Adds actual time entries and optional timers. You could compare estimated versus actual effort, then total time through a project or client's contents. This would be particularly useful for your Work setup. Billing could come later; recording time doesn't need to require a billing system.
>
> My priority would be Organizing container, Templates, and Review cadence. They would make the current product more useful without immediately adding a lot of scheduling complexity. Calculated totals would also help—estimated effort across a project, spending across a trip—but I'd implement those as calculated fields, rather than another behavior checkbox.
>
> **Organization starts with records, types, fields, and relationships.** A record is an actual thing: ABC, Transcript Intelligence, or "Finish the central docs." A type defines what kind of thing it is: Client, Project, or Task. It supplies the available behaviors, fields, statuses, and permitted homes. A field stores a piece of information about a record: priority, deadline, industry, budget, or a link to something else. A relationship connects records. We distinguish the primary Home from additional links because they serve different purposes.
>
> ```mermaid
> flowchart TD
>     W["Work · Space"] --> A["ABC · Client"]
>     A --> P["Transcript Intelligence · Project"]
>     P --> T["Finish central docs · Task"]
>     T --> S["Check examples · Task"]
>     P --> N["Discovery notes · Note"]
>     P -. supports .-> G["Improve client delivery · Goal"]
> ```
>
> The solid lines are Homes. The dashed line is an extra relationship. Each record has one immediate Home. "Finish central docs" lives inside Transcript Intelligence. Transcript Intelligence lives inside ABC. ABC lives inside Work. You only assign the task to Transcript Intelligence. The system can derive the rest of its path: Work → ABC → Transcript Intelligence → Finish central docs. You shouldn't need to separately assign Work, ABC, and Transcript Intelligence to the same task. That would create three opportunities for those assignments to disagree.
>
> The Home picker therefore lets you browse progressively—Work, then ABC, then its projects—or search directly. You can stop at any home permitted for that record's type. A task can live directly under Work if you don't need a project. Also, Work and Personal are organizational spaces inside your workspace. They don't themselves define sharing permissions.
>
> **Subtasks use this exact same structure.** A subtask is an ordinary Task whose Home is another Task. It has its own status, assignee, dates, and links. There isn't a separate subtask system you need to configure. The Task type simply permits Task as a home. Projects can therefore behave like larger pieces of work containing smaller pieces of work. Completing the parent leaves its children's statuses unchanged. When all children are done, the parent is ready for your final completion decision.
>
> **Extra relationships handle connections that shouldn't change where something lives.** A project can support multiple goals. A note can relate to several projects. A task can block another task. Those connections don't give the record additional Homes. This keeps the main tree understandable while allowing connections across branches. Blocking is advisory: Eri can flag an unfinished prerequisite, but you can still record progress or completion. Completing the dependent task doesn't complete its blocker.
>
> **Fields are reusable definitions with separate values.** You can define "Budget" once and attach it to Project and Trip. Each project or trip still has its own budget value. Changing the shared field's name or meaning affects the types using that definition. Visibility, placement, and inheritance can be configured on each attachment. Some custom fields can inherit through Home. For example, a classification can come from an ancestor unless the record overrides it. Operational fields such as deadlines, priority, and assignee remain independent; moving a task shouldn't silently change its deadline or who owns it. This preserves the separation you wanted: Types & fields builds the system. Detail cards use the system.
>
> **The two trees have different jobs, and the UI needs to make that unmistakable.** The actual-record tree is your filing structure. It contains Work, ABC, Transcript Intelligence, and the specific tasks and notes beneath them. Organization's default Browse view is a compact way to move through that structure. Opening a container gives you Groups, Tasks, Notes, Related, and All contents. "Add here" creates a record in the current Home. The tree layout provides a broader structural view of those same records.
>
> Moving an actual record changes its Home:
> - Move with contents: Transcript Intelligence moves to another client, and its children stay inside it.
> - Move only this record: Transcript Intelligence moves, while its direct children move up to ABC. Each child keeps its own descendants.
> - Archive: offers equivalent choices for retaining or including the contents, with a preview and guarded Revert.
>
> Extra links remain links; they aren't included merely because they point to the moved or archived record. Local filing also doesn't automatically reparent or delete the corresponding item in an external service.
>
> The Types & fields tree is the blueprint. It contains definitions such as Space, Client, Project, and Task, with their fields available beneath them. Clicking a type edits its name, description, behaviors, statuses, and allowed Home types. Clicking a field edits that field's definition and configuration. You can add types, reuse fields, and reorder fields. Visual and Advanced edit the same draft, followed by Preview and Apply.
>
> There is an important current limitation in how this is communicated: the type diagram shows one arrangement, but a type can have several permitted Home types. For example, Task might be allowed inside Project, Client, Space, or another Task. It appears once in the diagram so the diagram remains finite and readable. Dragging Task beneath Project in this builder changes its diagram placement. It doesn't move existing tasks or remove their other permitted homes. Currently, the destination must already be an allowed Home type. The defaults are also deliberately permissive. A tidy-looking Space → Client → Project → Task diagram does not mean those are the only legal placements.
>
> **How I want that tree to become clearer.** I'd retain the two views, but make the blueprint more explicit:
> - Each type appears once along a primary illustrated path.
> - A visible "Also allowed in…" row shows its other permitted homes.
> - A "Can nest itself" badge explains Task-inside-Task without drawing an endless branch.
> - Fields sit in a clearly labeled, collapsible Fields group so they don't look like child record types.
> - Moving a type clearly distinguishes rearranging the diagram from changing allowed homes.
> - Changes to placement rules show affected existing records before you apply them.
>
> Finally, Eri's organization learning should use your record titles, descriptions, and actual filing choices to improve suggestions. Learning that "Transcript Intelligence work belongs under this project" is useful. Silently redesigning your types or changing their rules would be a separate action. The guided "Chat about my structure" interview remains deferred. That will eventually provide another way to build this same schema through conversation, with a concrete preview before applying it.
