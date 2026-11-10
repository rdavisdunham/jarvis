# End-to-end upgrade plan: Eridani plans, remotea builds

Drafted 2026-11-10. Based on read-only surveys of this repo, the remotea repo, the Brainforge platform repo, your Linear issues and two weeks of Granola meetings. The code facts come from those surveys and were not re-checked line by line; items marked "unconfirmed" need a look before work starts.

## The diagnosis

Almost every part of the flow already exists. What is missing is the chain between the parts.

- **Inputs exist:** meetings become proposed tasks, Linear syncs both ways, calendar syncs, and remotea sessions are filed under records.
- **Planning exists:** tasks carry steps, success criteria, a verify command, a completion policy and a target repo.
- **Execution exists:** Eridani can spawn a run in a worktree on remotea, steer it, answer its questions, verify, judge, read CI and merge.
- **Nothing connects them.** A meeting proposal becomes a task, then stops. Nothing fills in its criteria or target, and nothing starts a run. A Linear issue assigned to you does the same.
- **Every hop is a click, on a different screen.** One task takes at least three clicks (start, approve plan, approve or merge) and often five or six (meeting review, filing, interview). Plan approval is required on every planned run, however small.
- **Human-only work gets nothing.** About half your open Linear issues are labelled `human-only` (follow-ups, demos, training, QA passes). The system tracks them but does no preparation for them.
- **Several inputs never arrive:** Slack, email, and meetings you did not record yourself.

So the jump is not another feature. It is one rule: **work moves forward by itself until it needs your direction, and everything that needs you lands in one queue.**

## Target flow

1. **Signal.** Something arrives: a meeting, a Linear issue, a Slack ask, an email, something you say to Dani.
2. **Triage.** Eridani dedupes it against existing tasks and Linear, files it under the right client, and decides what it is: ignore, for information, your task, or waiting on someone else.
3. **Route.** Each task gets one of four routes (table below).
4. **Run.** Agent work starts on remotea without you. Human work gets a prep pack.
5. **Decide.** Anything needing you goes into one queue, phrased as a decision with a recommendation.
6. **Close.** Results flow back out: Linear status and labels, a PR, a drafted update.

### Routes

| Route | What qualifies | What happens without you | Where you come in |
|---|---|---|---|
| Auto | `ai-assignable`, target has verify and CI, no migration, no production or client-visible behaviour change | Plans, runs, verifies, opens the PR; merges in your own repos | Client repos: review the PR |
| Direction check | Other agent-runnable work | Agent writes a five-line brief: how it reads the task, its approach, what it will leave alone, open questions | You answer the brief once; it then runs unattended |
| Full plan review | Migrations, production changes, client-visible scoring or rubric changes, anything that spends money | Plan drafted | Today's plan approval, unchanged |
| Human with prep | `human-only` | Prep pack: drafted message, meeting brief, checklist, the relevant transcript excerpts | You do the task |

The direction check is the answer to "I need to give input on direction": it is asked early, once, in a few lines, before any code is written.

## Phases

Ordered so that each phase makes the next one safe. Widening the inputs comes after the chain and the queue, because more volume is only useful once the system can absorb it.

### Phase 0: unblock (days)

- Deploy remotea `origin/main` (#67 `run.reopen`, #68 `spawn.resume`) and add a check that flags when the running build is behind.
- Finish and commit the manual-verification work sitting uncommitted on `feat/remotea-ledger`.
- Give the five targets that fail the doctor check a working verify command. Without one, every run ends in manual verification.
- Fix the Zoom transcript sync (PLT-2444) and backfill. No Zoom transcript has reached the vault since 2026-10-02. Linear records two causes: an AWS storage credential failure from 2026-10-05 (storage since moved to R2), then an OpenRouter 401 "User not found" on the summary step, still failing on 2026-11-10 (PLT-2500). The fix is a credential on the Cloudflare Worker, which needs an operator with Cloudflare and OpenRouter access. Zoom deletes cloud recordings after 60 days, so early October starts becoming unrecoverable around 2026-12-01.
- Move the judge onto its own queue; it currently shares one with memory and overview jobs.
- Record a baseline: clicks per completed task, time from signal to task, share of tasks finished without you.

Done when: remotea runs what is merged, all eight targets pass verify readiness, and the baseline numbers exist.

### Phase 1: the chain

- **Readiness pass.** When a task is created from any source, a background job fills in criteria, verify command, completion policy and target. The target comes from a mapping of Linear team and project to repo (ABCD to `abchomeandcommercial`, PLT to `brainforge-platform`), extending today's folder rules.
- **Route assignment** from the table above, shown on the task and changeable in one tap.
- **Auto-start** for the Auto route, within a per-host cap.
- **Direction brief** replaces mandatory plan approval for the Direction check route.
- **remotea: failure states.** Add errored, stalled, rate-limited and auth-expired, plus an inactivity watchdog. Unattended runs need these; today silence is never interpreted.
- **remotea: evidence.** Export the diff and full transcript per run, and have the judge read the diff. It currently judges the agent's own report.

Done when: an `ai-assignable` Linear issue in a ready target reaches an open PR with no click from you.

### Phase 2: one queue

- Merge meeting review, Questions, direction briefs, plan approvals, run questions, PRs ready for review and drafts awaiting send into a single "Needs you" list.
- Every item states the decision, the recommendation, the evidence and what happens if you accept.
- Batch delivery at set times (morning, after meetings, end of day); urgent items push to your phone immediately. Eridani already sends push notifications, so remotea's "needs you" should travel through it.
- Dani can read the queue aloud and take answers by voice.
- One assistant, one name. Dani appears in both apps with the same capabilities: the terminal tools Tea has today (search, read, name and move terminals) plus sight of tasks and runs. The Tea name is retired, so there are never two assistants that look alike and behave differently.

Done when: you can clear a day's decisions from your phone without opening remotea.

### Phase 3: widen the inputs

Start with an intake agent, not a connector per source. A scheduled harness run on the remotea host reads each source through the connectors the harness already has, then submits what it finds to Eridani's existing sources endpoint (`PUT /api/v1/external/sources/...`, the one `scripts/bridge_submit.py` uses). This works before the Brainforge gateway is set up on any harness. Direct gateway ingestion by Eridani is the later upgrade: it removes the dependency on the host being online.

- **Linear:** add webhooks, and treat issues assigned to you by others as signals to triage. Much of your work is created by other people, often in batches after meetings.
- **Meetings you did not record:** query the Brainforge transcript search by client and date. Link each meeting to a project in Eridani, because the transcript store attributes meetings to clients and attendees but not to projects.
- **Slack:** direct messages and mentions become signals.
- **Email:** last, and only if Slack and meetings leave a real gap.
- **Linear writes:** no approval step. Check for an existing issue first, update before creating, never create in bulk.

Done when: an ask made in a meeting you did not record, or in a Slack DM, shows up triaged within the hour.

### Phase 4: human work and reporting out

- **Prep packs** for `human-only` tasks.
- **Standup notes** for the Daily Delivery Sync, built from the session ledger and completed runs.
- **Client and weekly updates** as drafts.
- **`exec:` label at Done**, applied automatically. Eridani knows whether a task was agentic end to end, agent-assisted or human-only, and the company already requires this label.
- **Clockify draft** from session time per client (the ledger already files sessions by client folder).

### Phase 5: other people

Start only when phases 1 to 3 work for you. Roll out in two steps, because people who do not run agents are not blocked on the host work (see "Two kinds of user"):

- **5a, people using it without a host.** Tracking and meeting organisation work with no remotea host, so this needs only row security, per-user keys and a simple setup.
- **5b, people running agents.** Everything in 5a plus a host.

Prerequisites the surveys found missing:

- **A host that does not need WSL** (next section; 5b only).
- Database-level row security; today every query filters on owner by hand and `owner_id="davin"` is a config default.
- remotea per member; today the hub token is equivalent to shell access.
- Spawning OpenCode, since the company's setup guide steers staff toward it.
- Per-user model keys and spend limits.
- Sign-in restricted to `@brainforge.ai`, and client data kept separate per client.

### Native Windows host (and later Mac)

Recommendation: do not mirror the server's capabilities inside the app. Ship the same server inside the app.

- **One server core, three platforms.** The server is already Node. Package it with the Remotea app as a background process that starts at sign-in, keeps running when the window is closed, and shows in the tray. The app talks to it over the same protocol it uses for a remote host, on localhost. A second implementation inside the app would drift from the Linux one, which is the two-assistants problem again.
- **Platform adapters** for the parts that assume Linux: terminal spawning and shells, background service management (systemd today), paths and credential storage, and the verify sandbox. Linux assumptions currently appear in about 14 server source files.
- **Verify sandbox.** It uses bubblewrap, which is Linux only. On native Windows, verify would run unsandboxed, so the Auto route there should require CI as the gate.
- **WSL stays supported and stays your setup.** It is the better host for repos with Linux toolchains. Native Windows is for people who will not install WSL.
- **Mac.** The server side should be the easier port, since macOS is close to Linux. The app is the larger job, because it is a Windows terminal app. Out of scope for now; the adapter split keeps it possible.
- **Order.** A one-week spike during Phase 1 or 2 (run the server natively on Windows, spawn Claude Code, complete one Eridani run) to find the hard parts early. The full build, with installer and auto-update, belongs to Phase 5.

This matches the "Native Windows hosting and future macOS support" entry already in remotea's `docs/remotea/TODO.md`.

### Two kinds of user

One app, the same features for everyone. Nothing is hidden by role. The requirement is that the baseline is simple enough for a non-engineer to use.

**The baseline** is what everyone does daily: meetings become tasks, tasks are tracked, the queue asks for decisions, prep packs arrive, Dani answers questions. Rules for keeping it simple:

- It works with no configuration. Setup is sign in with Google, connect calendar and meetings, done.
- It works without a host. Tasks take the "Human with prep" route until one is connected; the agent features are still there to see and to set up.
- Baseline screens use plain words. A task says "an agent is working on this" and "ready for you to review"; the worktree, verify command and harness are one click deeper, on the run's own page, for anyone who wants them.
- No baseline action requires understanding an advanced one.

**Automations** are the flexible part, open to anyone. Each one is a trigger, a condition and an action, built from pieces the pipeline already has:

- triggers: a signal arrives, a task changes state, a run finishes, a time of day;
- conditions: source, client, label, repo, person;
- actions: set a route, start a run with a prompt, draft a message or report, create or update a task.

The route rules in Phase 1 are the first automations; Phase 4 generalises them. An engineer writes one by describing it to Dani or editing it directly.

**Recipes** make automations easy to adopt. Someone publishes an automation as a named recipe ("after every client meeting, draft the follow-up"); anyone else turns it on with one switch, and can open it to see or change how it is built.

Test for 5a: someone non-technical goes from sign-in to their first meeting with tasks in ten minutes, without help.

### Keeping Eridani general

Eridani should work for Brainforge without being built around it. Everything Brainforge-specific lives in a workspace profile, as settings and not code:

- label names that drive routes (`ai-assignable`, `human-only`, `exec:`),
- the Linear team and project to repo mapping,
- route rules and merge rules per repo,
- which sources the intake agent reads,
- report formats (standup notes, client updates, Clockify).

Your profile is the Brainforge one. A later product needs a different profile, not a rewrite.

## What to stop or defer

- Swarm follow-ups, until single runs flow unattended.
- The GitHub App (merge phase 2); the host has to be online to run anyway.
- New organization features; overviews, placement and people links are enough.
- Own-built Slack, Gmail or Drive connectors.
- Merging the two organization models and the two meanings of "work" (Dani requests and remotea runs) in the database. Unify them in the interface in Phase 2 and leave the tables alone for now.

## Decisions (settled 2026-11-10)

| Decision | Outcome |
|---|---|
| Is Linear still the record for Brainforge work? | Yes. Eridani is the cockpit over it and the record only for personal work. |
| Which work may skip the direction check? | The Auto row above. Start narrow (your own repos) and widen as the numbers hold. |
| Where may it merge without you? | Your own repos with green CI. Client repos stop at "PR ready". |
| How do outside sources reach Eridani? | An intake agent in a harness first. Direct gateway ingestion later. |
| One assistant or two? | One, with one name and one set of capabilities. |
| When do other people come in? | After Phase 3 works. |
| Is Eridani Brainforge-only? | No. Brainforge specifics live in a workspace profile. |

Still open: whether native Windows hosting is worth building, or WSL is required for everyone. The spike answers it.

## How to tell it worked

- Clicks per completed task: from three to six, down to zero or one.
- Share of `ai-assignable` issues finished without you touching them.
- Minutes per day spent in the queue.
- Time from a signal arriving to a triaged task.
- Linear issues created per week by Eridani, to keep the volume honest.
