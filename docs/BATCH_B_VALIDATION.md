# Batch B — organization, sources and daily workflows

October 2, 2026. Branch: `codex/batch-b-workspace`. Review before merge; this document does not certify production deployment.

## Implemented behavior

- Organization opens an actual-record tree. Drop above a record to reorder, or onto its center to change home. Keyboard/touch Move controls provide home selection and up/down ordering. Current owner/schema/record revisions and cycle/type checks remain authoritative. Types & fields is a separate schema editor with the existing preview/confirmation process.
- Client/project cards show contained work and clickable extra links. A main home controls containment/inheritance; extra links do not move anything. Native task moves clear stale legacy project membership when no legacy project remains in the ancestry. Connected Linear projects stay connector-owned; local home moves never silently send a Linear project update.
- A common source payload supplies literal provider names, stable account/container/item identity, original URLs, provider context, remote version and sync state. Task list/board/timeline/search and calendar/detail views use source badges. Account-level source colors are editable in Profile settings; text labels remain visible. Existing provider details are displayed without pretending unsupported remote fields can be edited.
- Supported title/description/status/date/assignee changes retain the durable connector paths. Pending, failed, disconnected and conflict states are distinguished from confirmation. Native records are not automatically published. Local archive is not remote deletion; existing provider-specific delete and safe-Revert restrictions remain in effect.
- Eridani-only notes are separate from synced descriptions. Task/custom-record annotations live on the canonical StructureRecord; appointments retain local notes on PlanningEntry. Google annotations use owner + Google account subject + provider calendar + provider event identity in a separate table, surviving replaceable cache rows, full sync and reselection. These are workspace-visible notes, not personal secrets within a shared workspace. Recurring-series annotations attach to the cached series unless an imported occurrence is explicitly selected.
- Settings distinguish scheduling availability from weak work-context filing hints. The verified planner applies work/personal windows independently, uses the profile timezone, handles overnight/DST windows and checks again at commit. Empty windows are unrestricted. Personal defaults are unrestricted; work defaults are weekdays 08:00–17:00. Explicit owner-requested exceptions carry a visible reason in the proposal and saved result.
- Existing natural-language snooze, quiet hours, explicit urgency and daily summaries are retained and regression-tested. Successful Eri actions remain in chat/Activity, not notifications.
- Record-card conflicts preserve drafts and offer compare/use-saved/reapply. Annotation editors participate in navigation guards. Mobile scheduling controls wrap without horizontal overflow. Eri's discovered tools/site map describe the tree, local notes and scheduling behavior while keeping prompt/discovery size limits.

## Verification

- New Batch B backend acceptance: 15 passed, including same-source Linear readback, Revert preserving later notes, owner/scope isolation, Google cache/account boundaries, move ordering, stale revision rejection, native unfiling, scheduling revalidation, explicit exceptions and DST/overnight cases.
- Frontend: 171 passed, 1 skipped; TypeScript/Vite build checked. Five new focused tests cover valid homes, ordering and literal/safe source rendering.
- Browser: all seven suites passed in a migrated disposable database with synthetic credentials and external services disabled. Includes the new Batch B suite plus existing shell navigation, chat activity, semantic search, notes lists, usage and custom planner suites. Phone 390px, foldable 892px and desktop widths exercised. Screenshots are local artifacts, not committed personal content.
- Google connector regression: 70 passed. Full backend suite: 817 passed, 1 skipped.
- Migration 0019: N-1 → head, model agreement, downgrade to N-1 and re-upgrade/model agreement passed on a disposable PostgreSQL 16 database. Columns/tables are additive with server defaults. Downgrading removes new annotations and ordering: take a backup before any production downgrade.
- Offline eval inventory/catalog/coverage checks passed; 82 meta-tests passed, 2 skipped. The 1,001-case catalog still has 100 fully bound cases. Additional regression coverage does not falsely mark unbound model scenarios as passed. No paid inference used.

## Remaining acceptance and deliberate limits

- Dedicated real Google/Linear items: perform an Eridani edit and source-side readback, interruption/retry/conflict, disconnect and reconnect, multi-calendar/recurrence scope, and verify local notes stay private to Eridani. Mock-provider success is not real-provider acceptance.
- Physical Pixel Fold and locked-phone push/quiet-hour/snooze acceptance, actual voice UI acknowledgements and production restart checks remain release gates in Batch D.
- Calendar/event reversals continue using their existing date/scope/provider-specific controls where generic Revert cannot safely apply. Google local-note commands are revision-checked local saves; they do not create a remote calendar write.
- Rank space exhaustion is rejected rather than silently rewriting other records; Move to end then reposition recovers it. Ordering is serialized with workspace graph mutations.
- No new live latency benchmark or paid model comparison was run. Conditional B5 (Luna no-reasoning/compact-tool trials) waits for A4 measurements; the model default is unchanged.
- This patch intentionally does not broaden Linear API fields beyond the supported snapshot, add multiple remote owners per field, or introduce an additional connector. Future connectors must supply the same source metadata and keep annotations outside synced payloads.
