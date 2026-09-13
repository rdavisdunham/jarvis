# Fixed 20-scenario tool regression

This run reuses twenty questions/workflows from the September 13 expert suite,
including every earlier failing scenario. Each runs three times with both
reasoning-enabled GPT-5.6 Luna and Gemini 3.8 Flash: **120 scored trials**.
The two clarification scenarios retain their original second user message; a
trial is one scenario, not one provider request.

The original prompts, deterministic initial records, permitted changes and
database-state graders remain unchanged. Selection was fixed before new paid
results. This is a targeted before/after regression, not a fresh random estimate
of general model ability. Existing files in `docs/evals/expert-*` remain baseline
evidence and are never overwritten.

## Fixed selection

1. `bulk_pagination` — exact filtering, complete selection application and receipt counts.
2. `selected_not_visible` — explicit selected scope versus the visible screen.
3. `ambiguous_followup` — clarify duplicate titles before modifying the chosen task.
4. `singular_ambiguous` — do not guess which of several visible tasks is meant.
5. `stale_revision` — atomic conflict handling, including the new selection update path.
6. `lost_ack` — inspect current records instead of repeating an uncertain create.
7. `dst_gap` — resolve a nonexistent local time, clarify, then save the correction.
8. `dst_fold` — select the requested second occurrence of a repeated hour.
9. `clear_deadline` — clear the deadline while preserving planned work and alerts.
10. `goal_rewire` — maintain the exact many-to-many graph and unrelated metrics.
11. `note_preservation` — preserve text and every unmentioned note link.
12. `evidence_extraction` — create only definite commitments with real source-evidence links.
13. `note_injection` — treat forged instructions inside note content as data.
14. `unsupported_email` — do not invent an unavailable capability or substitute an unwanted record.
15. `calendar_unknown` — unavailable calendar evidence must not become free time.
16. `constraint_schedule` — discover planning tools and produce the optimal feasible four-block schedule.
17. `impossible_schedule` — prove the requested schedule cannot fit before writing.
18. `pending_remote` — preserve queued/retrying semantics without duplicate creation or invented monitoring.
19. `ui_refusal` — respect an unsaved editor instead of claiming navigation succeeded.
20. `recurring_occurrence` — complete one delivered occurrence while preserving its routine.

Four scenarios are excluded from this shorter regression. All passed every
baseline trial:

- `subtask_reparent`: straightforward reparenting overlaps the retained relationship-preservation cases.
- `cross_zone`: explicit Tokyo/Chicago conversion is less directly targeted than the retained DST gap and fold cases.
- `memory_override`: stale-memory precedence was already clean; untrusted-data restraint remains covered by note injection.
- `all_day_span`: exclusive all-day end handling was clean; retained scheduling cases put more pressure on the revised tools.

These exclusions do not declare the capabilities fully tested or permanently
retire them. They remain in the full twenty-four-scenario suite.

## What changes in the harness

The production conversation loop now discovers typed tools through
`ToolSession`. The runner observes loader calls without replacing their
behavior, and retains successful/failed loads separately from execution calls.
Provider responses are projected to function names, arguments and call IDs, so
attempts rejected before execution are still visible. Native reasoning content,
encrypted state and Gemini thought signatures are never retained.

Every provider request records the actual offered tool names/count, UTF-8 JSON
schema bytes, system-prompt bytes, input-context bytes and full request-body
bytes. Input byte counts include the size of opaque provider continuation state,
but never its content. These are byte measurements, not tokenizer estimates.
Provider token usage remains the authority for actual token counts.

Task queries and `task_selection_update` use the real application commands.
The synthetic adapter records all members and revisions of the authoritative
selection for later auditing, without adding those hidden members to model
context. Selection snapshots are cleared between fixtures. The concurrent-edit
fault applies to both existing task edits and the new atomic selection path.
Original exact-state and after-every-call safety checks stay in force.

Only instruction time is frozen, now in `agent_instructions.datetime`.
Monotonic clocks, request timeouts, database clocks and selection expiry remain
live. Calendar/remote services remain synthetic adapters. New remote status
metadata must match the production return contract before scored trials start.

## Execution and evidence

Do not start paid trials until implementation and local harness checks are
declared ready. Then run:

```sh
uv run python scripts/evaluate_expert_agents.py \
  --suite tool-refinement20 --repeats 3 \
  --output .runtime/tool-refinement-evaluation.json
```

Use a separately named `tool-refinement-diagnostic` artifact for any explicit
preflight. Diagnostics are marked and excluded from scored results. The runner
interleaves model order for each scenario/repeat, preserves both successful and
failed trials, checkpoints after each trial, stops the paired suite on a provider access, quota or schema rejection,
and records every remaining trial as skipped rather than counting it as a pass.
Schema rejection text is retained after credential scrubbing; ordinary provider
errors retain codes and safe parameter names only.

Compare each retained case/repeat against the same baseline fixture. Report
exact-state success, safe-but-recovered completion, tool errors, manual
truthfulness findings, discovery overhead, provider usage and latency
separately. A shorter prompt or fewer tools is not a quality win if it causes
missed work. Keep the raw traces and selection snapshots so future tool fixes
can target observed failure mechanisms.
