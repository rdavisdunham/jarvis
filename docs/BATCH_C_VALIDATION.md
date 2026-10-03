# Batch C — Quick lists and guided setup

Implemented October 3, 2026 on `codex/batch-c-capture-setup`, based on `cfc9c6e`
(the Batch B / PR #23 merge). This report records local synthetic verification,
not deployment, model-quality acceptance or a real-device test.

## What changed

- Quick lists reuse Task rows: one parent list, stable child Task IDs, optional
  sections, a deadline and Today pin. Ordinary Tasks/Today views show the parent;
  search can still find children. Completion, add/edit and reorder are inline.
  Alerts default off; a missed deadline never discards work.
- Promotion turns that same parent and its children into ordinary tasks/subtasks,
  optionally assigned to an existing project. It does not clone work or create a
  second project engine. Create a new project through the existing project tools.
- Capture/edit/promotion use the existing command transaction, revisions,
  idempotency, action history, Revert, owner lock, permissions and search observation.
  New typed tools load through the existing tool catalog. API/MCP bot access requires
  task scopes; revoked keys and shared viewers cannot mutate.
- Setup is account-scoped progress in OwnerSettings: name, timezone, purpose,
  optional custom grouping with a meaningful description/example, skip and resume.
  Preview and apply are separate commands using existing schema confirmation and
  stale-proposal guards. It creates no sample tasks. Current profile preferences
  remain authoritative when setup is resumed. Shared members configure their own
  Personal account, not the shared owner's profile.
- Setup exposes text/voice tools to Eri and reuses field-understanding/clarification
  and schema tools. The form offers a simple optional group; more elaborate schema
  design remains in Organization and conversational schema preview/apply.
- Explicitly confirmed filing rules work without waiting for automatic field
  assessment. Automatic rules retain understood-field and held-out quality gates,
  explicit assignments still win, and conflicting rules abstain. Rules show their
  destination, reason and evidence, with edit/confirm/pause/forget controls. Editing
  a pending rule resolves its weekly interview question. Nothing writes personal memory.
- Login/invitation/setup copy distinguishes owned personal records from a selected
  shared membership. Install the v19 peeling-note mark as SVG, monochrome SVG and
  PNG PWA icons; see [Brand](BRAND.md). Orbit styling is retained.

## Evidence

All commands used synthetic local data and empty application environment settings.
No paid model inference, connected-provider writes or production mutations ran.

| Check | Result |
| --- | --- |
| Full backend `pytest -q --tb=short` on PostgreSQL 16.15 | 836 passed, 1 skipped |
| Eval harness `python -m pytest -q evals --tb=short` | 82 passed, 2 skipped |
| Frontend `npm test -- --run` | 175 passed, 1 skipped |
| Frontend TypeScript/Vite build | Passed; existing large-chunk advisory remains |
| All browser acceptance fixtures | Eight suites passed; Batch C rerun after final UI polish |
| Python correctness lint and `git diff --check` | Passed |
| Migration 0019 → 0020 → 0019 → 0020 and model agreement | Passed in disposable database |

`tests/test_batch_c.py` covers atomic/idempotent capture, item IDs, stale revisions,
invalid reorder, promotion/history, foreign records, deadline retention, list cap,
setup isolation/skip/resume, explanation and separate confirmation, changed-answer
invalidation, timezone validation, current profile preferences, explicit versus
learned routing, edited-rule history/memory separation, shared roles and bot scopes.

The new browser fixture uses the real local API with synthetic authentication:
Houston capture → complete Hayes item → add/reorder → backend edit while open →
close/reopen → one summary → promotion with original IDs. It also exercises setup
answers, preview-before-apply, confirmation, persisted completion and 390/892px
layouts. Existing browser fixtures cover navigation, chat, notes, semantic search,
usage, Batch B source UI and custom planner workflows. Screenshots are local
artifacts; they do not substitute for physical Pixel Fold acceptance.

The catalog now contains **1,051 scenarios across 42 features**. Quick capture and
onboarding each add 25 scenarios. Twelve exact deterministic scenarios are fully
bound, raising the ratchet from 100 to **112**; component coverage stays **224**.
The other new cases retain incomplete/unrun acceptance status. Historical reports
and original outcomes remain unchanged. Existing calendar eval seed fingerprints
ignore new capture metadata only at its defaults, preserving old frozen seeds;
non-default capture state remains in snapshots/fingerprints.

## Migration and rollback

Migration `0020_capture_setup` adds `is_quick_list`, `quick_section` and `quick_order`
to Task with defaults that preserve ordinary tasks. Setup progress uses existing
JSON settings; no provider data or existing task rows are rewritten.

Upgrade the database before running the new app. Downgrading removes capture
metadata, so lists become ordinary parent/subtasks and section/order information
is lost; task rows/IDs remain. Back up before any production downgrade. Retaining
0020 while rolling back app code is preferable for preserving that metadata, but
an older app will display checklist children as normal tasks.

## Still requires acceptance

- Real Eri text/Live conversations: incomplete capture, follow-up edits, suggested
  supplies versus actual commitments, setup clarifications and weekly rule interview.
- A fresh invited person completes or skips/resumes setup without developer help.
- Physical phone/fold keyboard, scrolling, screen reader/focus and installed PWA icon.
- Batch A/B connected-provider, notifications, operations and voice checks in TODO.
- Release-core model reruns, latency measurement and the seven-day pilot remain Batch D.

A merge or green CI alone does not close these acceptance items.
