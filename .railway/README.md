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

The current services run a paused staging preview. Variable values remain managed
in Railway so infrastructure application cannot silently promote staging or
start duplicate workers. Only perform that switch through the cutover runbook.

GitHub autodeploy is disconnected during migration. Apply infrastructure changes
separately from app uploads; these are different operations.
