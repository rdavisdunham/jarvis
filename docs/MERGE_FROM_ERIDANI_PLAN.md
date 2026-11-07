# Merge from Eridani: plan

Status: approved 2026-11-07. Phase 1 and 1b are built: remotea #59 (host commands), the Eridani merge panel (#186) and auto-merge (PR C). Phase 2 (the GitHub App) is next.

## Goal

Davis can merge an agent run's pull request from the run drawer, without opening GitHub.

- **Merge appears for every run that has a pull request**, in every repository.
- **CI gates it** when the pull request has CI. The button is grayed out while CI runs. With no CI it is always live.
- **Merging completes the run.**
- **Update branch** is offered when GitHub refuses the merge because the branch is behind its base.
- **Auto-merge** is a per-run toggle, with a default in Settings. It merges by itself once the run passes its checks. It is only offered when the run has at least one real check: CI on the pull request, or a local verify entry.

Two ways to reach GitHub, built in this order:

1. **Through remotea (phase 1).** The host runs `gh` with its own login, the way `ci.status` already works. This is the path for every repository, including brainforge-ai org repositories, where Davis cannot install an app.
2. **A GitHub App (phase 2),** for Davis's own repositories. CI results arrive by webhook, and merges work while the host is asleep. remotea stays the fallback for every repository the app does not cover.

## Already in place

These ship today and are not rebuilt:

- **CI checks** (`run_ci.py`, #154; remotea `ci.status`, remotea #49). The host reads checks with `gh` and records them as `ci` attempts.
- **The verification ruleset** (#159). Under policy `code_ready` or `pr_open`:

  | The target has | The pull request has | What finishes the run |
  |---|---|---|
  | a quick check | CI | the quick check, then green CI |
  | no verify entry | CI | green CI |
  | a local check | no CI | the local pass |
  | no verify entry | no CI | Davis: "Needs your verification", **Approve as verified** |

- **"No CI" detection.** A pull request whose checks are absent on 2 polls in a row (about 4 minutes) has no CI.
- **Merge detection.** The host's worktrees report says when a branch merged. Policy `pr_merged` already ends a run on the merge.

## Facts that shape the design

- **eridani's `main`** requires Backend, Frontend and Security audit, with **strict up-to-date** branches. GitHub refuses to merge a pull request that is behind `main`, even when its CI was green. In practice "behind" will be the most common reason a merge is refused.
- **eridani's repository settings:** auto-merge off, the "update branch" suggestion off, merge commits allowed, branches deleted on merge.
- **Host `gh` 2.45** has `gh pr merge --match-head-commit`, but no `gh pr update-branch` command. The update goes through `gh api -X PUT repos/{o}/{r}/pulls/{n}/update-branch` with `expected_head_sha`, which works on any version.
- **Org repositories.** abchomeandcommercial and brainforge-platform belong to brainforge-ai. wmux belongs to someone else. These go through remotea only.

| Target | CI on pull requests | Local verify | Merge path |
|---|---|---|---|
| jarvis (eridani) | yes | `quick`, `full` | app, else remotea |
| remotea | yes | `server` (full suite) | app, else remotea |
| brainforge-platform | yes | none | remotea |
| abchomeandcommercial | workflows exist; PR triggers unconfirmed | none | remotea |
| budget, popscribe, the-connection | no | none | app, else remotea; manual merge only |
| wmux | release workflows only | `test` | remotea, if Davis can merge there at all |

## Phase 1: merge through remotea

### remotea (one PR in rdavisdunham/remotea)

- **`ci.status` reports mergeability**, so the button knows before a click whether to offer Merge or Update branch. It adds `gh pr view` fields `mergeable`, `mergeStateStatus` (CLEAN, BEHIND, DIRTY, BLOCKED, UNSTABLE, UNKNOWN), `baseRefName` and `state` (OPEN, MERGED, CLOSED) to the `pr` object.
- **New write command `pr.merge`** `{run_id, branch, head_sha, method: "merge"}`, run in the branch's worktree:
  `gh pr merge <branch> --merge --match-head-commit <head_sha>`. Outcomes:
  - `executed` with `{merged: true, number, url, base, merge_commit}`;
  - `refused` with a `reason`: `behind`, `conflicts`, `head_moved`, `checks_failing`, `not_open`, `blocked` (another rule), `permission`, `gh_unavailable`, plus `gh`'s own message.

  "Already merged" counts as success, so a retried command is safe. A lost answer is reconciled by the next `ci.status`, which reports the PR as `MERGED`.
- **New write command `pr.update_branch`** `{run_id, branch, head_sha}`:
  `gh api -X PUT repos/{o}/{r}/pulls/{n}/update-branch -f expected_head_sha=<head_sha>`. Outcome: `{new_head_sha}`, or a refusal (`conflicts`, `head_moved`, `permission`).
- **New capabilities** `pr.merge` and `pr.update_branch` in the hello.
- **Fencing.** Both commands are durable and journaled on the run timeline, unlike `ci.status`. They act on GitHub, not on the agent's session, so they are sent whoever has control of the session, and for a settled run too. `head_sha` is their fence.

### Eridani backend

- **Record the merge state** with each CI answer (`RunCi.merge`: `{state, base, number, head_sha}`). Add an on-demand refresh, `POST /remotea/runs/{id}/ci/refresh`, which the drawer calls when it opens. That keeps the merge state current on runs that are no longer polling.
- **`POST /remotea/runs/{id}/merge`** `{expected_head_sha}`. The server checks all of these before sending `pr.merge`:
  - the run has a pull request, from the CI record or the session's `pr_urls`;
  - when the pull request has CI: `ci.state == passed` on `expected_head_sha`, and that is still the PR head;
  - with no CI: no CI condition;
  - the host is online and advertises `pr.merge`;
  - the run is `awaiting_approval`, `needs_you` on a verify, CI, or no-plan-free reason, or already settled with an open PR. A settled run only merges; its state does not change.
- **Merged.** The run is approved with an override note "Merged PR #n into <base>" and becomes `completed`. Its task completes. The timeline gets "Merged PR #n".
- **Refused, behind.** The run stays where it is, and the card offers Update branch.
- **`POST /remotea/runs/{id}/update-branch`** sends `pr.update_branch`. The new head has no CI yet. A run in `awaiting_approval` moves to `awaiting_ci` (a new transition) and waits for green on the new head, then returns to approval. CI reruns on GitHub by itself.
- **Refused, conflicts.** The card offers "Send back to fix conflicts", which queues a note to the agent.
- **CI is preferred wherever it exists (a change to #159; Davis, 2026-11-07).** Quick check plus CI is the best setup. A full local entry (`full`, remotea's `server`) on a repository with CI is skipped: CI stands in for it after every turn and the judge reads CI's result. With no CI, the full local pass is enough. Whether a repository has CI is learned from the CI answers for its target.
- **CI polls were never sent (bug found while building this).** `agent_link.COMMANDS` did not list `ci.status`, so the host was never asked and a run waiting for CI could only time out. PR B adds `ci.status`, `pr.merge` and `pr.update_branch` there, and lets them through for settled runs and under terminal control.

### Eridani web: the Verify section

A merge panel sits beside the CI card, inside Final acceptance:

| Situation | What the panel shows |
|---|---|
| CI running | **Merge into main**, disabled, "Waiting on CI", with the CI chip |
| CI failing | disabled, "CI failing", plus Send back |
| CI green on the current head, or no CI | **Merge into main**, enabled |
| Behind its base | **Update branch** ("main has moved; update, then CI runs again") |
| Conflicts | **Send back to fix conflicts** |
| No pull request | "No pull request yet". Approve only |
| Host offline, or no `pr.merge` capability | disabled, with the reason |
| The host answered `ci.status` with a failure and there is no earlier answer | disabled, with the host's reason (e.g. an old `gh`) instead of "Checking the pull request…" |
| Merged | "Merged PR #n into main", with a link |

- **Approve without merging** stays as a secondary action, for work merged elsewhere or not meant to merge.
- **The "Needs your verification" card** (no CI, no verify) gets Merge beside Approve as verified.
- **The button names the base** ("Merge into main"). Merging into eridani's `main` deploys to Railway, so the label says exactly what happens. There is no extra confirm step.

## Phase 1b: auto-merge

- **A per-run toggle,** "Auto-merge", in the run drawer header, beside the stage track. **A Settings default** ("Auto-merge new runs", `remotea_auto_merge`) sets it for new runs.
- **Offered only with a real check:** CI on the pull request, or a local verify entry on the plan. With neither, the toggle is disabled ("Needs a verify entry or CI"), and only the manual Merge button exists.
- **When it fires:** at the point where the run would otherwise wait for approval or auto-complete. That is after the judge finds the criteria met, after integration, and after green CI when the pull request has CI.
- **Behind its base:** it updates the branch, waits for green CI on the new head, and merges. It tries this at most 3 times per run, then stops in `needs_you(merge_blocked)`.
- **Conflicts and other refusals** stop the run in `needs_you(merge_blocked)` with the reason and Send back.
- **Auto-merge implies completion.** `remotea_auto_complete` keeps its meaning for runs without auto-merge.
- **Docs.** Update the "Automatic merge" item in `docs/TODO.md`: Davis decided on 2026-11-07 to offer it behind the real-check rule.

## Phase 2: a GitHub App for Davis's repositories

- **Registration (once).** A GitHub App owned by rdavisdunham:
  - Permissions: Checks read, Commit statuses read, Actions read (for failed-job logs), Pull requests read and write, Contents read and write (merging needs it), Metadata read.
  - Events: `check_suite`, `check_run`, `workflow_run`, `pull_request`, `status`.
  - Webhook URL: `https://app.eridani.app/api/v1/integrations/github/webhook`.
- **Railway environment:** `GITHUB_APP_ID`, `GITHUB_APP_SLUG`, `GITHUB_APP_PRIVATE_KEY` and `GITHUB_WEBHOOK_SECRET`. No user token is stored. Eridani mints a 1-hour installation token per call.
- **Settings: a GitHub card.** Connect GitHub opens the app's install page. Davis picks repositories there, and the callback stores the installation id against his account, checked with a signed `state`. The card lists the covered repositories, the last webhook, and Disconnect.
- **The webhook route** verifies HMAC-SHA256 against the secret, drops repeated deliveries by delivery id, and maps each event to runs by repository plus head SHA or branch. It writes the same `ci` attempt as a host answer does, with `source: github_app`. The `run_ci` state machine stays unchanged.
- **Merge and update branch** go through REST: `PUT /pulls/{n}/merge` with `sha`, and `PUT /pulls/{n}/update-branch` with `expected_head_sha`.
- **Choosing the source per run.** Eridani uses the app when its installation covers the run's repository, and remotea otherwise. The drawer says which one it used ("via GitHub" or "via DavisPC"). If the app fails on a call, Eridani falls back to remotea for that call.
- **Security.** The app's private key lives only in Railway's environment. Access is limited to the repositories Davis picked. Tokens are short-lived and never stored.

## Delivery

| PR | Repository | Contents |
|---|---|---|
| A | remotea | mergeability in `ci.status`; `pr.merge`; `pr.update_branch`; capabilities; tests with a fake `gh` |
| B | eridani | merge and update-branch routes, states and transitions; CI for any local entry; the Verify merge panel; docs |
| C | eridani | auto-merge toggle, setting and loop; `merge_blocked`; TODO update |
| D | eridani | the GitHub App: Settings card, install callback, webhook, REST merge, source choice |

- **Local checks stay light:** typecheck, lint, and the touched tests. CI verifies each PR.
- **PR B needs PR A deployed** on the host: `npm run build` and a remotea service restart. Until then, a host without `pr.merge` shows Merge disabled with "Update remotea".
- **Acceptance after B:**
  - merge a the-connection run's PR, which has no CI;
  - merge a jarvis run's PR after green CI;
  - push to `main` while a run waits, so its PR falls behind, then use Update branch and watch CI rerun;
  - merge a brainforge-platform run through remotea.

## Decisions (Davis, 2026-11-07)

1. **CI is always preferred when it exists.** Quick check plus CI is optimal; CI alone when there is no local check; CI instead of a full local check; a full local check alone when there is no CI.
2. **Auto-merge on "behind"** updates the branch and retries, up to 3 times.
3. **Merge commits everywhere.** A per-repository choice can come later.
4. **Order:** phase 1 and 1b first, then the GitHub App.
