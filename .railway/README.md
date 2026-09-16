# Railway infrastructure

`.railway/railway.ts` describes the existing Eridani Railway project. Install the
pinned root SDK with `npm ci`, link this checkout to project
`e0c8a9a5-c608-47c5-97c0-1e9d93f5878a` / `production`, then run
`railway config plan`. Review before applying. See `docs/CLOUD_MIGRATION.md`.

This is a whole-project definition. Omitted resources can be deleted. Keep
recovery buckets, volumes, legacy databases and temporary restore services in
mind when reviewing a plan. Never apply an unreviewed destructive plan.

Secrets use `preserve()`. Do not import decrypted variables into source control.
Both PostgreSQL image majors are explicit: the SDK's default is not safe for an
existing version-16 volume. Production candidate is Postgres16; the old Postgres
18 service has its deployment stopped, its volume preserved, and is not connected to the app.

The current services run production against the private eridani database. Variable
values remain managed in Railway. Local production writers and auto-start are
stopped; never restart them against the stale local copy. Follow the runbook for
future restores, staging environments and deliberate worker handoffs.

API and worker are configured to use rdavisdunham/jarvis on main. Apply reviewed
infrastructure changes separately from app releases; pushing app code and applying
an infrastructure plan are different operations.
