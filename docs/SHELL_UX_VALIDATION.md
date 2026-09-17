# Chat, Settings, search and navigation verification

September 17, 2026. Follow-up to `recommendations.md` and
[the compact UX validation](UX_POLISH_VALIDATION.md).

## Changes

- Persistent pastel Eri launcher in the bottom-right on every signed-in page.
  Chat opens as a floating panel, with a short scale/slide animation and a
  reduced-motion alternative. Closing chat preserves active voice behavior.
- Top search always opens all actionable tasks, resets collection/status filters,
  and searches titles, details and stored field values. Other pages retain their
  own local search. Task embeddings remain planned, not implemented here.
- Settings has separate Profile, Organization, Notifications, Voice, Connections,
  Privacy, Sharing and System sections. Available content width selects vertical
  navigation or a compact section selector; controls reflow for foldable widths.
- Browser Back/Forward restores pages, filters, task cards, conversation and
  activity without a document reload. Closing a layer consumes its navigation
  entry. Unsaved note and structure drafts block leaving; inline detail changes
  use the existing save-before-leaving behavior. History state stores only an
  opaque session/index; snapshots stay in memory and reset on sign-out.
- Eri's UI contracts and screen instructions include the new Settings locations.

## Evidence

- Production build and TypeScript pass.
- Frontend regression: 123 passed, one intentionally disabled Realtime test skipped.
- Backend UI contract checks: 6 passed.
- Existing custom planner browser acceptance passes.
- New shell acceptance passes: page/card/chat Back and Forward, no document reload,
  protected unsaved note and structure drafts, task search from Notifications,
  all eight Settings sections, viewport containment and reduced motion.
- Widths checked: 390, 600, 768, 820, 1000 and 1440 CSS pixels. Screenshots of the
  unfolded-style Settings and chat layouts were inspected.
- Browser checks use the real local API with a disposable synthetic PostgreSQL
  database, disabled workers and disabled external services. Production records,
  integrations and model providers are not used.

The browser runner is part of GitHub CI through
`scripts/validate_custom_planner.py`. Screenshots are local ignored artifacts in
`artifacts/custom-planner/`. No dependency or database migration was added.

## Remaining acceptance

Test the physical Pixel 10 Pro Fold's Back gesture, folded/unfolded keyboard and
voice continuation while the panel is closed. Viewport simulation does not verify
Android keyboard/microphone behavior. Existing bundle-size warnings remain.
