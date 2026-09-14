# Invited accounts and sharing release validation

September 14, 2026.

- 471 backend tests passed, with one optional skip. The paused Realtime suite remains excluded.
- 78 frontend tests passed, with one paused Realtime skip. Production build passed.
- Two-account browser acceptance passed: private task isolation, invitation acceptance, shared inline editing, phone workspace switching, and revocation clearing an open card.
- Access regressions cover viewer writes, owner-only membership changes, CSRF, expired/revoked/wrong-email invitations, canonical member assignment, private saved views, command account attribution, stale device contexts and namespace-claim attempts.
- Semantic retrieval tests use real note indexing with synthetic vectors and verify private/shared separation and revocation during a pending query.
- Live controller tests verify membership revocation closes a session; private interactive tools also stop when their session is revoked. No paid voice calls were used for these checks.
- Google OAuth tests cover invited registration with verified subject/email/nonce binding and preserve owner identity. Google/Linear browser acceptance passed, including calendar publication, conflict review, work blocks and unlink recovery.
- Migration round trips preserve earlier notes, sessions, reminder completion and task links.
- Encrypted pre-release backup jarvis-20260914T055324Z.pgdump.enc was restored to a separate database, then upgraded to 0012_accounts. It retained 56 tasks and preserved legacy session/command identities. No workers ran against the restore; the temporary database was removed afterward.
- API, worker, PostgreSQL and backup services are running. All 56 deployed backend Python files match source hashes.
- HTTPS smoke passed with the workspace selector and Sharing screen, task board/timeline projections, no browser errors and no phone overflow.
- Served JavaScript: index-B-j4qieG.js. Schema: 0012_accounts.
- Planner, notes, memory, settings and cost-record hashes are unchanged across deployment. Google connection fields match the encrypted pre-deployment backup; sync state may advance during routine polling.
- Development cost tracking remains off. GPT-Live remains the only offered voice mode.

See [Accounts and sharing](ACCOUNTS_AND_SHARING.md) for onboarding, role boundaries and explicitly deferred sharing extensions.
