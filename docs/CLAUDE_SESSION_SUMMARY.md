# Claude session summary: audit, repairs, CI hardening and UI redesign

Covers the Claude Code session of 2026-10-01 to 2026-10-02 that began from [CLAUDE_AUDIT_HANDOFF.md](archive/2026-10-02/CLAUDE_AUDIT_HANDOFF.md). Prepared as the starting point for a follow-up audit. Treat it as a map, not proof: verify against the code.

Related documents:
- [CLAUDE_AUDIT_FINDINGS.md](archive/2026-10-02/CLAUDE_AUDIT_FINDINGS.md): every finding with file:line, plus repair status in section 8.
- [DESIGN.md](DESIGN.md): the new UI design system.
- [CI.md](CI.md): the CI jobs, the coverage ratchet and the branch-protection ruleset.

## 1. Where things stand

| Item | State |
|---|---|
| PR #1 `audit-fixes`: audit fixes, CI and UI redesign | **Merged** to `main` (merge `202ba10`, head `bac4200`) at 2026-10-02 08:21 UTC and deployed to Railway. |
| PR #15 `fix-voice-requests` | **Open.** Fixes a production regression introduced by #1 (section 5), restores the old Eri launcher, and removes the leaked keys from the tree. Contains this file. |
| Branch protection | GitHub ruleset "main protection" (id 24354023) on the default branch requires *Backend and migrations*, *Frontend and browser* and *Security audit*, plus an up-to-date branch, and blocks force-push and deletion. Read back after creation. Nothing can land on `main` without a passing PR. |
| Railway | Services: Eridani_Web, Eridani_Worker and **Postgres16 only**. The owner deleted the old PG18 service. `.railway/railway.ts` **still declares the PG18 service**, so applying that plan would recreate it. The file needs reconciling (not done). |
| Leaked secrets | Three Groq API keys from 2024 prototype commits were found by the full-history gitleaks scan. The **owner revoked them** on 2026-10-02. They are removed from the current tree, and `.gitleaksignore` lists exactly those three fingerprints. |
| Dependabot | 13 PRs opened (#2–#14) after Dependabot was enabled. **Several are risky majors and should not be bulk-merged:** #2 postgres image 16→18 (compose/CI only, and would mismatch Railway's PG16), #3 node 22→26, #6 TypeScript 5.7→7, #10 Vite 6→8, #11 Vitest 3→5, #7 plugin-react 4→6, #13 lucide-react 0.x→1.x, #9 SQLAlchemy 2.0→2.1. Minors (#4, #5, #8, #12, #14) are likely safe once CI is green. |
| Backups | Deliberately deferred by the owner until true multi-user work (encryption or RLS). R2, restore drills and key escrow are out of scope for now. |

## 2. Commits (oldest first)

| Commit | Summary |
|---|---|
| `2da5ff7` | Committed the owner's previously untracked eval harness (`scripts/app_eval/*`, `evals/test_*.py`), Langfuse export, audit handoff and source snapshot. `campaign.py` also carries the coverage ratchet, because both live in the same file. |
| `1f4d290` | Added `docs/CLAUDE_AUDIT_FINDINGS.md`. |
| `6e5e8ff` | CI security job, Dependabot, eval coverage ratchet, README rewrite, compose worker behind a profile, Dockerfile layer caching, deploy migration diagnostics, `railway.ts` warnings, docs. |
| `976ca81` | Revert timezones (EVAL-001), null policy (EVAL-002), task tools expose record homes (EVAL-004/008), core/record sync, agent UI edits are not human evidence. |
| `af69d12` | Memory review (EVAL-005), facts commit without embeddings (EVAL-006), extraction retry, list extraction coverage (EVAL-007), spoken-number normalizer. |
| `3800080` | Bot scope leaks, logout, invites, session and bot-key revocation, lock order. |
| `696fe18` | Durable work: per-revision budget reservations, run lock, catch-all and reaper, budget-hold leak, supervisor scan isolation, prose-question guard, push registration. |
| `a7ecff2` | Voice clarification ownership redesign. |
| `0cff896` | Google write fencing, stranded planning entries, morning summary, push resubscription, Linear stale-read guard. |
| `451702d` | Frontend load coalescing, visibility-aware polling, optimistic capture, Dialog primitive, CopilotKit replaced with a local registry, lazy loading. |
| `04971f2` | Base images pinned by digest, gitleaks action pinned to a commit SHA, Dependabot for Docker. |
| `c5229cd` | Removed the CopilotKit dependency; npm audit (prod, high) made blocking. |
| `54bac5f` | Upgraded oauthlib 3→4, pyjwt and urllib3 for pip-audit findings. |
| `fd47689` | Made the custom-planner e2e fixture date-relative. It had hardcoded Sept 2026 dates and would have failed CI on `main` too. |
| `11f2b63` | **UI redesign ("Orbit").** |
| `bac4200` | Fixed pip-audit export conflicting with the workflow-wide `UV_LOCKED`. |
| `1d1716e` | Restored the original Eri pill launcher at the owner's request *(PR #15)*. |
| `bd64367` | Removed hardcoded Groq keys from `backend/OLD/JARVISv0.{1,2}.py` *(PR #15)*. |
| `b94dd10` | Fixed the voice/pending-question regression *(PR #15, see section 5)*. |
| `c612dd8` | `.gitleaksignore` for the three revoked keys *(PR #15)*. |

## 3. Audit repairs by area

Full details and file:line references are in the findings document. This section summarizes what changed and where.

**Authority and privacy**
- Files: `bot_access.py`, `access.py`, `accounts.py`, `api.py`, `google_auth.py`, `external_*.py`, `search_service.py`, `tools.py`.
- `records:write` now also requires `tasks:write` or `notes:write` when the target is a task- or note-backed record.
- `records:read` no longer returns or ranks on core bodies without `tasks:read` / `notes:read` (search uses a `withhold` parameter).
- Logout works for viewers, revoked members and expired sessions.
- First Google login accepts only invites from the server owner.
- Replaced sessions are deleted on link/consent, and bot keys are revoked when a member is removed.
- Queued bot reads work for record and list tools.
- Viewers can use search.
- Lock order is consistently `work-order`, then `work`.

**Durable work and billing**
- Files: `work_runner.py`, `worker.py`, `agent_work.py`.
- Budget reservations are per revision, so corrections no longer fail with "session ended".
- A session advisory run lock prevents concurrent execution.
- Unexpected exceptions finish the work as failed, and a reaper clears stale running or dispatched rows.
- Budget holds are kept only when a provider outcome is unknown.
- Each supervisor scan is isolated, with dispatch running first.
- Recovered tool errors are cleared by affected entity.
- One shared per-run httpx client.
- Note: each running request holds one extra pooled DB connection for the lock.

**Revert, mutation semantics and structure**
- Files: `db.py`, `domain.py`, `action_history.py`, `structure.py`, `task_tools.py`, `task_context.py`, `tools.py`, `tool_catalog.py`, `productivity.py`, `routing.py`.
- DB sessions are pinned to UTC, timestamps use a UTC encoder, and Revert compares instants.
- One null policy (`NOT_NULL` / `CLEARED`), with a test that every nullable schema property is accepted by its command.
- Batch and selection edits sync to records.
- Minimal two-way parent ↔ `project_id` sync.
- `task_list`, `task_resolve` and `task_get` return `record_id`, a `home` chain and a `home_id` filter, and `task_resolve` returns compact rows.
- `record_list` defaults to 25 rows (maximum 50, compact), and reconciliation runs only when drift is found.
- Missing workflow statuses map to the nearest status.
- `ui-agent:` command IDs keep agent UI patches out of human routing evidence.

**Memory, search and notes**
- Files: `memory_*.py`, `search_*.py`, `note_lists.py`, `notes.py`, and the new `text_normalize.py`.
- Fact-key conflict review and numeric-contradiction review.
- Facts commit with null vectors when embedding fails, and an embed job runs later.
- Failed extractions retry with backoff.
- A superseded fact can be reasserted.
- Per-entry rejection reasons are recorded, and missing list items get a second pass or `possible_omissions`.
- `canon()` and `sql_filter()` match spoken numbers and punctuation, applied to semantic, memory, notes, task and record search.
- Batched note embeddings and query-embedding caches.

**Voice**
- Files: `work_intake.py`, `live_voice.py`, `voice.py`, `work_continuation.py`, `agent_instructions.py`, `voice*.ts`.
- The quiet flush now retains speech instead of enqueueing it, and Live's explicit delegation claims one combined job.
- Goodbye and chit-chat are not enqueued. At close, leftover turns are enqueued only if they contain an action verb (a heuristic).
- A question is spoken once, and announcement stamps no longer include the revision.
- Consistent inbox lock order, and expiry skips running work.
- The browser avoids a bare-"thanks" auto-close while busy, polling is calmer, and the idle suspension is capped at 120s.
- **Warning:** the deterministic answer binding added here caused the production regression and is removed in PR #15.

**Integrations and notifications**
- Files: `google_*.py`, `planning.py`, `notices.py`, `linear_*.py`, `worker.py`, `api.py`, `sw.js`, `push-subscription.ts`.
- A stale sync no longer bumps the account generation, so queued writes survive.
- Failed or cancelled publishes stay editable and deletable.
- Overnight reminders count toward the morning summary.
- A retryable `RefreshError` is treated as temporary.
- "Use Google" times are normalized.
- A per-row guard keeps one bad entry from breaking the calendar view.
- Linear ignores stale reads, and reassigned issues become out of scope.
- Duplicate-alert window is ±60s.
- Push re-subscribes after subscription changes, new devices get no backlog, and a 403 deactivates the subscription.

**Frontend platform**
- Coalesced, sequenced loads.
- Polling depends on SSE health and tab visibility.
- Queued quick-add and optimistic completion.
- Shared focus-managed `Dialog` and `Popover`.
- Local tool registry instead of CopilotKit.
- Heavy dialogs are lazy-loaded.

**CI, ops and repo**
- New `security` job (pip-audit, npm audit, gitleaks over full history), Dependabot (actions, uv, npm, docker), and an eval coverage ratchet (`runner coverage --baseline`).
- Digest pins, and a Dockerfile dependency layer.
- README describes the current app.
- `compose.upgrade.yml` worker requires `--profile worker` plus `ERIDANI_LOCAL_LIVE_WRITER`.
- `railway.ts` comments only.
- `.codex-remote-attachments/` ignored.

## 4. UI redesign ("Orbit")

- **Spec:** `docs/DESIGN.md`. Tokens and components are in `apps/web/src/theme.css`.
- **Foundation:**
  - Light and dark themes, with a Light/Dark/System setting stored in localStorage under `eridani-theme`.
  - Self-hosted Outfit (`@fontsource-variable/outfit`); the CSP allows `font-src 'self'` only.
  - A violet action color, and an iridescent orbit used only for the day ring, the brand mark and the chat avatar.
- **Pages:**
  - Today is the new landing page (`Today.tsx`): day ring, merged schedule with a now line, due/overdue, inbox and recent notes.
  - Tasks and Organization share a single toolbar with a Filter popover and date-grouped `.row` lists.
  - Calendar: single header row, cascading week overlaps, and dialogs on the field system.
  - Notes have a list rail and a card grid.
  - Memory facts are shown as rows.
  - Detail cards use a properties list.
  - Settings has a section rail with `SettingsGroup` / `SettingRow`.
- **Phones:** bottom tab bar and bottom-sheet dialogs.
- **Behaviour changes:** the app lands on Today instead of Tasks, and task lists sort by due date by default.
- **Bundle:** the PlannerApp chunk is about 139 KB gzip (252 KB before the audit, 126 KB after the CopilotKit removal and before the redesign).
- **Launcher:** the original Eri pill button was restored in PR #15.

## 5. Production regression found after deploy (fixed in PR #15)

- **Symptom:** the owner reported that Eri summarized overdue items but other voice requests (Linear task management and others) didn't work.
- **Evidence:**
  - Railway logs showed two Live sessions and successful model calls (20 of 20 returned 200), with no server errors.
  - A production DB read to confirm the stored per-request errors was blocked by the session's permission classifier. The cause is inferred from the code path, and the regression tests reproduce the scenario.
- **Cause:** two audit changes interacted.
  1. The prose-question guard (`work_runner.py`) forced a reply ending in "?" with no saved changes into `needs_input` after one nudge, including optional offers like "want me to reschedule any?".
  2. `claim_voice` (`work_intake.py`) attached the next delegated voice turn to any question the session had heard, as its answer.
  - Together, every request after one summary-with-an-offer was swallowed as an "answer".
- **Fix (`b94dd10`):**
  - Keep the single corrective nudge, then trust the reply.
  - Every delegated turn is a new request; the backend uses `work_answer` itself when a turn really answers a pending question (the pre-audit behaviour).
  - Tests: `test_heard_question_does_not_capture_the_next_request`, `test_offer_after_one_nudge_stays_a_finished_answer`.
- **After deploy:** requests already stuck as "needs your answer" remain in Activity and should be dismissed.

## 6. Verification performed

Run locally against a disposable PostgreSQL 16; the recipe is in project memory and `docs/CI.md`.

- **Backend:** 783 passed / 1 skipped under both `PGTZ=UTC` and the America/Chicago server default. The baseline was 688, plus 6 timezone failures.
- **Evals:** 82 passed / 2 skipped.
- **Web:** tsc clean, vitest 166 passed / 1 skipped, build OK.
- **Browser:** all 6 acceptance suites in `scripts/validate_custom_planner.py` pass.
- **Security:** pip-audit and npm audit (prod) clean. gitleaks 8.24.3 over the full history (103 commits): no leaks, with the ignore file in place.
- **GitHub CI:** all three jobs green on PR #1. PR #15 CI was running when this was written.
- **Visual:** about 100 screenshots reviewed (desktop light and dark, 884px fold, 390px phone), produced by the scripts under the session scratchpad.

**Not verified:**
- Real GPT-Live audio, and voice end to end (the regression was caught in production).
- Real Google or Linear writes.
- Physical phone or foldable, and locked-phone push.
- Production database state.

## 7. Suggested focus for the next audit

1. **Voice end to end in a real session**, with PR #15 deployed:
   - delegation timing (`settle()`), the close-time action-verb fallback, single announcements, and `work_answer` use on real follow-ups.
   - Live instructions now tell Live to gather the "what" and "when" before delegating. Check that it doesn't over-ask.
2. **Interaction risks between audit changes.** Section 5 shows the agents fixed areas in isolation. Look for similar cross-module effects:
   - question guard ↔ Live announcements;
   - run lock ↔ revise/cancel;
   - budget per-revision IDs ↔ usage reporting;
   - new `records:*` scope rules ↔ existing bot keys.
3. **Bot keys in production:** existing keys with only `records:write` lost write access to task- and note-backed records (intended). Confirm no integration relied on that.
4. **Linear:** the integration agent added a by-id re-read of missing issues during "only mine" full syncs. Measure the Linear API call volume (the logs show many GraphQL calls per cycle) against rate limits.
5. **Structure sync:** the H7 two-way parent sync is minimal. Legacy-table retirement (findings S1) is still the real fix for EVAL-004/008-class bugs.
6. **UI:**
   - App.tsx is still about 3.4k lines, and the conflict banner (S9) was not built.
   - Known small issues: cramped 3-way overlaps in the week view at fold width; the Eri pill can cover a right-aligned action mid-scroll; a row's hover reschedule button covers its due chip.
7. **Ops:** reconcile `railway.ts` with the actual services, triage the Dependabot majors, and turn off Postgres serverless (the worker polls every 5s, so it never sleeps; if it did, reminders would stop).

## 8. Deliberately not done

- Legacy organization table retirement.
- pgvector.
- Quick lists.
- App.tsx split.
- Conflict banner.
- The ~200-case eval core set and repeats≥3.
- An N-1 migration compatibility CI job.
- Archiving `backend/` and `client/`.
- Backups, R2 and key escrow (deferred by the owner).
