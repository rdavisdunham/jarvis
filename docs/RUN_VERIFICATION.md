# How an agent run is verified

An Eridani agent run is checked in three layers, cheapest first:

1. A **quick local check** catches obvious breakage.
2. The **judge** decides whether the work does what the plan asked.
3. **GitHub CI** on the pull request proves the full suite passes.

The judge comes before CI: a run only waits for CI after the judge finds the success criteria met.

This page is the overview. The exact rules, states and payloads are in [agent control](AGENT_CONTROL.md):
- "The judge (`agent_judge.py`)";
- "CI as the final verification";
- "jarvis: `scripts/verify-quick.sh` after every turn".

The GitHub workflow itself is in [GitHub CI](CI.md). The decision to work this way is Davis's, from 2026-11-04. Running the full suite locally after every turn, and again on GitHub, did the same work twice.

## After every turn

1. **Quick local check** ([`scripts/verify-quick.sh`](../scripts/verify-quick.sh), target under 3 minutes, timeout 600 s). It runs:
   - ruff with CI's selection;
   - pytest on the tests related to the changed files, plus the drift guards;
   - `tsc -b` and `vitest related`, when the web app changed.

   A failing quick check becomes the judge's gap.
2. **Judge** (a model call, prompt `judge-v5`). It grades each success criterion **met**, **not met** or **unclear**. It treats the agent's own report as a claim, never as proof. It counts only three kinds of evidence:
   - the quick check's result, and GitHub CI's result for this turn;
   - the host's worktree and pull request record;
   - the plan's steps as ticked.

   A criterion of kind `verify` needs a passing verification on this turn. Either a local pass or green CI counts; green CI is recorded with rule `verify_by_ci`.
3. **Not met.** The judge writes a gap, the agent gets it as its next turn, and the loop repeats.
4. **Met.** Sub-task pull requests must merge first (`awaiting_integration`). Then the run waits for GitHub CI (`awaiting_ci`), unless green CI is already on record for this turn.

## Waiting for CI

Eridani holds no GitHub credential. Every 2 minutes it asks the run's host to read the pull request's checks with `gh`, using the host's own login. A run waiting on CI is not the owner's move, and the wait does not count toward its wall clock. In the drawer it shows in the **Verify** section, as the Awaiting CI card.

| CI result | What happens |
|---|---|
| **Green** on the pull request head | The run goes to approval, or completes itself when `auto_complete` is on. The judge is **not** run again: it already found the criteria met. |
| **Red**, as soon as any check fails | The agent gets the failed jobs, up to 40 log lines and the run link, and goes back to work. This step skips the judge. The agent fixes the code and pushes, and the next turn goes through the quick check, the judge and CI again. |
| **Red again on the same commit** | The agent did not push a fix. The run stops for you (`ci_failed_again`). |
| **No pull request, or the worktree is not pushed** | The agent is told to push and open or update the pull request. This happens at most once per turn and twice per run. |
| **Nothing finished within 90 minutes** | The run stops for you (`ci_timeout`), with what was still running. The limit is `JARVIS_AGENT_CI_WAIT_SECONDS`. |
| **The host cannot read CI** | The run stops for you (`ci_unavailable`). Update remotea, or check CI yourself and approve. |

## When green CI rescues a stuck run

Sometimes a run stops because the local check could not run at all, for example a missing test database or a timeout (`verify_infra`). Eridani keeps polling CI on such a run: every 2 minutes during the CI wait, then every 10 minutes, for up to 24 hours. If CI goes green on the current commit, the run goes back to the **judge**, with green CI as its verification evidence. This is the only path where CI's result goes through the judge.

The same polling applies to a run stopped on `ci_timeout` or `ci_unavailable`. Green CI moves either one straight to approval or completion, because the judge had already found the criteria met.

## When CI is not the gate

The CI wait applies only when both of these hold:
- the plan's verify entry is a quick check (its id is in `JARVIS_AGENT_CI_QUICK_VERIFY_IDS`, default `quick`), or the
  plan names no verify entry (CI is then the only verification; a pull request with no checks on two polls means the
  run needs your verification instead: the "Needs your verification" card with Verified, approve and Describe what's
  wrong). A full local entry never consults CI. See the ruleset in [AGENT_CONTROL.md](AGENT_CONTROL.md#ci-as-the-final-verification);
- the completion policy is `code_ready` or `pr_open`.

In every other case, the run finishes as before:
- A plan whose verify entry runs the full suite locally (`full`, `./scripts/verify.sh`) finishes on its local pass.
- A plan with policy `pr_merged` finishes when the pull request merges.

## Where it shows

- **Run drawer.** The Verify section shows each criterion with the judge's status, the latest quick check and judge notes, and GitHub CI's chip or card. A failed CI card sits in the Run section while the agent fixes it.
- **Run rows** carry a CI chip: pending, passing or failing.
- **Timeline.** Each CI record appears as "GitHub CI on 1a2b3c4 · passing" (or failing, running, no pull request…). It is never shown as a judge result.
