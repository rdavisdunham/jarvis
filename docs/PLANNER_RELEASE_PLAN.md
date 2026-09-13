# Planner workspace release

Requested September 13, 2026. Preserve existing personal data, active integration
bindings, disabled cost accounting and the paused Realtime implementation.

## Delivery order

- [x] Reliability: sparse note changes and precise relationship errors; deterministic
      calendar planning and atomic, revalidated plan saving.
- [x] Evaluation: align synthetic integration state and add fresh held-out cases.
- [x] Conversational controls: navigation, all view/filter/sort/group controls,
      editor drafts and device settings through typed acknowledged actions.
- [x] Project/work boards and timelines: shared task records, revisions and filters;
      accessible status changes, undated work and clear date semantics.
- [x] Review every main page and editor for compactness, useful density and consistent
      organization; record suggestions and apply the resulting UX changes.
- [x] Update Eri's tool definitions and site context to the final UI and capabilities.
- [x] Verify GPT-Live delegation/clarifications/interruption/result delivery and
      shutdown paths; separate automated protocol checks from physical-device tests.
- [x] Review the full schema and architecture against an all-in-one planner/task
      tracker. Apply necessary consistency fixes; document intentional boundaries
      and genuinely deferred capabilities.
- [x] Complete targeted/full validation, real paired backend eval and browser acceptance.
- [x] Deploy and verify health, exact application sources and preserved records.

## Design constraints

A task is the actionable unit. Alerts do not duplicate task completion; work blocks
reserve time, while planned dates and deadlines retain separate meanings. Projects,
goals and authored notes link to the same records. Board/timeline are projections,
not additional task stores. Assignment does not start an autonomous job.

Use native form/domain validation and stable command receipts for changes. Eri
must distinguish an unsaved draft, saved local state and pending remote state.
Device-reported context is bounded untrusted data; no integration credentials enter
it. An acknowledged action must describe what actually happened, including refusal
to discard an unsaved edit. Avoid arbitrary DOM selectors or executable UI actions.

Implementation/review: [PLANNER_UX_ARCHITECTURE_REVIEW.md](PLANNER_UX_ARCHITECTURE_REVIEW.md).
Verification: [PLANNER_VALIDATION.md](PLANNER_VALIDATION.md).
Paid paired evidence: [RELIABILITY_HELDOUT_RESULTS.md](RELIABILITY_HELDOUT_RESULTS.md).
Physical-device acceptance is distinguished from the completed generated-audio
and protocol checks; ongoing phone/network testing remains in TODO.
