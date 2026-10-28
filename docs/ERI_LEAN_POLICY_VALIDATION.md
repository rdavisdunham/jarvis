# Lean tool-policy validation

October 27, 2026. Review and isolated validation for PR #40, reconciled with
main a1e2ff4 (session-ledger PRs #58–60). Discovery remains the application default.
The owner asked to defer checks requiring their physical phone.

## Review changes

- Keep the static prompt before changing request data, the smaller initial tool
  set, and the existing follow-up coordination path. No new queue, agent, early
  completion shortcut, migration or authorization bypass.
- Remove the proposed silent history/recent-work cuts. A long reference may end
  with a qualifier or correction. Existing history limits and complete recent-work
  data remain; the current request and clarification answer remain verbatim.
- Hash workspace identity, full tool definitions and static instructions into the
  optional cache key. This separates accounting without placing owner IDs in the
  key. Current OpenAI documentation describes the key as accounting separation on
  GPT-5.6 and later; the stable prefix itself enables reuse.
- Honor an explicitly empty external-bot group enum in the compact loader. It
  must not fall back to advertising every capability. Tool authorization remains
  enforced by the existing catalog and executor.
- Repair the synthetic seed to use current core-task revisions after record
  creation rather than assuming revision 1. Production revision checks are unchanged.

## Matched provider comparison

Four state-oracle scenarios (task creation, note creation, sparse deadline edit,
and duplicate-target clarification), three repetitions per policy, fresh synthetic
Rowan database clones. Nine serial blocks rotated discovery/reads/lean,
reads/lean/discovery, lean/discovery/reads. All used Luna low reasoning and requested
Fast; all returned priority. The 36 trials share one source/fixture/harness
fingerprint. No owner data, real voice or connected-source mutations were used.

All 36/36 passed. Twelve trials per arm:

- discovery-v1: median 4.100511 s, mean 4.8902805 s, 29 model rounds,
  256,942 input tokens, ledger cost $0.025931767.
- reads-v1: median 3.9791415 s, mean 4.49045775 s, 27 model rounds,
  246,375 input tokens, ledger cost $0.024956669.
- lean-v1: median 3.7413455 s, mean 3.784375083 s, 27 model rounds,
  208,394 input tokens, ledger cost $0.021605664.

Lean's median was about 8.8% lower than discovery (6.0% lower than reads); cost
was about 16.7% lower than discovery (13.4% lower than reads). These are small
synthetic backend samples, not a production p95 or audible-latency promise.

Every arm had first-round cache hits in 10/12 trials. The earlier production
zero-cache observation was not reproduced in this sample: do not describe this as
a demonstrated rescue from universally uncached requests. Cache writes and reads,
cold trials, returned tier and all attempted calls are included in the costs.

An additional two-turn smoke created Call Alex, then changed that same call to
tomorrow, once per policy. All 3/3 passed the saved-state oracle without duplicate
creation or unrelated mutation. Each used six model calls, so this smoke did not
demonstrate fewer follow-up rounds. Actual request inspection confirmed a cache
key on all six lean calls and no key on the other policies; the provider accepted
them. Deterministic coverage also exercises same-round linking plus lookup,
pending predecessors, resumed reads and Revert.

Primary trials cost $0.072494100. Follow-up smoke cost $0.022058421.
Total ledger spend: $0.094552521 (about 9.5 cents), 101 requests, no uncertain
charges. Nine $0.20 primary ledgers plus one $0.20 smoke ledger capped this work
at $2 in aggregate. No paid judge, Gemini, embeddings or live content tracing.

[Sanitized trial evidence](evals/2026-10-27-lean/comparison.json) retains individual
outcomes, latency, tool sequences, definitions sizes, cache reads/writes, tokens,
tiers, costs and fingerprints. Full synthetic artifacts remain in the validation
worktree under artifacts/lean-validation-2026-10-27/.

## Test environment and deterministic checks

The managed checkout is on Windows storage, tested from WSL with explicit
PYTHONPATH and Git metadata. Python uses the locked project environment.
The original local corpus on port 54340 had an older fixture hash and was preserved.
A separately named local PostgreSQL 16.15 container/volume on port 54341 holds the
new synthetic corpus. A failed initial seed was exported before rebuilding only
that new corpus. Both setup failures occurred before any provider calls.

Initial focused regression suite: 115 passed. Offline eval harness: 95 passed,
2 skipped (after correcting the Windows/WSL Git path in the test environment).
Correctness Ruff selection passed. Final focused suite after the empty-group fix: 134 passed, including external-bot
access, queue continuations, Responses accounting and lean tool regressions.

The post-benchmark empty-group fix changes only catalogs whose permitted group
list is explicitly empty. Owner catalog definitions used in the paid trials are
unchanged; this is recorded rather than claiming an identical final fingerprint.

## Rollout and remaining evidence

- discovery-v1 remains the default. reads-v1 and lean-v1 remain opt-in, pinned
  at request acceptance on API and worker. Accepted work retains its policy.
- Local correctness and the targeted provider comparison support reviewing and
  merging the opt-in implementation; they do not establish every workflow's model
  parity. Required CI still needs the final commit.
- Real phone typed/GPT-Live checks remain deferred: capture, sparse and full-body
  edits, agenda, bulk edits, related corrections, and clarification continuation.
  Actual source writes, device rendering and audible completion remain unmeasured.
- Enable a new policy consistently on API/worker only for an explicit rollout.
  Rollback changes newly accepted work; it does not rewrite already accepted jobs.

Official source: [OpenAI prompt caching](https://developers.openai.com/api/docs/guides/prompt-caching),
checked October 27, 2026. The existing Luna accounting includes cache-write
premiums and returned service tier.
