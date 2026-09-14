# Daily workflow and reliability release

September 14, 2026.

Existing tasks now open the same inline detail card from task lists, boards, timelines, calendar entries and linked records. Text saves on blur/Enter (Ctrl/Cmd+Enter for descriptions); Escape cancels a pending field. Selectors save immediately. Eri can read, change and close these cards, with revision conflicts retaining unsaved text. Task creation and other forms keep their explicit save behavior.

The card retains completion, archive/restore, linked notes, reminder editing, work-block creation and Linear publication/conflict review. Main details and compact properties stack on phones. Named task views restore filters, tab, layout, grouping, sorting and timeline range, including on page reload.

Notes support revision-guarded append and unique anchored replacement without rewriting untouched text. Assignee filters resolve canonical IDs and reject ambiguous names. Browser acknowledgements include the observed layout and visible result sample. Archived-note keyword searches follow the UI filter. Tool failures distinguish malformed arguments, truncated output and provider/transport errors.

Validation:
- 456 backend tests passed; one optional skip. Paused Realtime suite excluded.
- 78 frontend tests passed; one paused Realtime test skipped. Production build passed.
- New browser acceptance covers inline editing, navigation, Escape, conflicts, saved views/deep links, empty-result acknowledgements, archived notes and mobile overflow.
- Planner acceptance: 81 acknowledged actions.
- Calendar/task acceptance covers mouse, keyboard and real touch board moves, tab subsets, details, completion and calendar editing.
- Google/Linear acceptance covers consent, details, stable scrolling, publication/CRUD, lost acknowledgements, import/publish/conflict review and work blocks.
- Calendar eval fixture now runs the real sync worker against synthetic transport; successful and unavailable sync paths are tested. Historical paid-eval results were not rewritten.
- API and worker deployed healthy. HTTPS smoke passed at desktop and phone widths, with no page errors or horizontal overflow.
- Served bundle: index-qO3DrBh3.js. Schema remains 0011_productivity_graph.
- Canonical record hashes unchanged across deployment: 56 tasks, 2 projects, 2 spaces, 2 actors, 8 schedules, 14 memory assertions and 322 historical budget reservations. No notes, goals or planning entries were added by validation.
- Development cost accounting stays off; Live remains the only offered voice mode.
