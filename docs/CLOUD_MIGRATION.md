# Eridani cloud migration runbook

Updated September 16, 2026. **Production is live at https://app.eridani.app.** The user approved the full private-data transfer. The final frozen source and cloud restore matched all **55 tables / 11,245 rows** before cloud workers resumed. API and worker now use the private `eridani` database on PostgreSQL 16.15. Google Calendar synchronized successfully from Railway. Local API/worker/backup containers are stopped; their database and encrypted snapshot are preserved.

Use Railway for the web/API, worker and PostgreSQL, with Railway point-in-time recovery (PITR) plus independent encrypted exports to Cloudflare R2. The [hosting plan](CLOUD_HOSTING_PLAN.md) explains the cost assumptions and Neon alternative.

## Owner setup and remaining input

- Railway project **Eridani**, services, the `app.eridani.app` domain and Google OAuth client are configured. The user confirmed the authorized redirect URI in Google Cloud Console; the deployed app generates exactly `https://app.eridani.app/api/v1/auth/google/callback`.
- Full transfer approval was received and cutover completed. Sign in to the new site using the existing Google account. Allow microphone and notifications on the new origin; actual spoken voice behavior and phone notifications still need a device check.
- R2 bucket `eridani-backups` exists. Its S3 access key/secret are still missing. R2 export deployment is deferred by the user's instruction; native Railway PITR remains enabled. Later, create an **Object Read & Write** token restricted to this private bucket and fill the matching fields in `.env.cloud`. [Cloudflare token instructions](https://developers.cloudflare.com/r2/api/tokens/)
- Store recovery copies of `JARVIS_BACKUP_KEY` and `JARVIS_INTEGRATION_ENCRYPTION_KEY` in your password manager. Keep the keys separate from encrypted archives.

The private handoff file is `/home/davin/jarvis/.env.cloud` in WSL. It is ignored by Git/Docker and is **not loaded by the running local app**. Its database URL now uses Railway private networking. The migration-only public PostgreSQL endpoint was removed. Read-only database checks from the PC require a new deliberate temporary connection or execution inside Railway. Do not paste secrets into chat or pass this python-dotenv file directly to Docker's differently parsed `--env-file`.

## Prepared code

- `.railway/railway.ts`: the complete imported Railway project, including explicit PostgreSQL major versions, preserved secret variables, domain, API/worker commands, migrations and readiness. The reviewed plan matched live configuration with no changes before the disposable recovery drill.
- `package.json` / `package-lock.json`: pinned Railway infrastructure SDK 3.11.0; independent of the frontend package.
- `deploy/railway/Dockerfile.postgres`: deferred pgvector 0.8.6 candidate preserving Railway's TLS/pgBackRest wrapper. Production currently uses the official PostgreSQL 16 registry image.
- `deploy/railway/Dockerfile.backup`: prepared independent encrypted export image. Add a cron service at **08:00 UTC** once R2 credentials are available; it is not currently deployed.
- `python -m jarvis.deploy migrate`: serialized Alembic migrations. Both API and worker pre-deploy hooks can invoke it safely.
- `python -m jarvis.deploy preflight --database`: read-only schema, DBOS presence, Google identity, encryption compatibility and pgvector availability checks.
- `scripts/prepare_cloud.py`: creates the handoff file without overwriting it; `--check` validates it without deploying anything.

Cloud startup rejects pairing login, missing OAuth/encryption configuration, invalid public origins and unsafe staging flags. Existing pairing sessions are also rejected when pairing is disabled. Google sign-in continues to require an existing linked account or a valid email-bound invitation.

The worker holds a PostgreSQL session advisory lock so overlapping releases cannot run two supervisors on the **same database**. This does not coordinate old and new databases during migration: stop the old worker before enabling the new one.

## Create the Railway services

Railway rejected the previous `railwayConfigFile` configuration on these new services: Config as Code (`railway.json` / `railway.toml`) is deprecated. Use the checked-in **`.railway/railway.ts`** infrastructure definition instead. Legacy per-service JSON examples have been removed. [Current Railway infrastructure configuration](https://docs.railway.com/infrastructure-as-code#migrating-from-config-as-code)

From the repository root:

```sh
npm ci
railway link --project e0c8a9a5-c608-47c5-97c0-1e9d93f5878a --environment production
railway config plan
```

Always review the plan before `railway config apply`. This file describes the **whole project**: omitting an existing resource can delete it. Never apply it while a disposable recovery service exists unless that service is deliberately included or its deletion has been reviewed. Secret values use `preserve()`; never import with `--include-variables`. The PostgreSQL factory defaults to a newer major version unless `image` is explicit; keep both existing database images explicitly pinned.

For app deployments, keep the repository root as build context and `Dockerfile.upgrade` as the Dockerfile. The API starts with `/app/.venv/bin/python -m jarvis.deploy api`, the worker with `... worker`; both use `... migrate` before deployment. API readiness is `/health/ready`, target port **8765**, with one process and one replica because Live controllers are process-local. The first upload used a curated source-only directory, excluding credentials, private snapshots, attachments and runtime files.

All current services use **us-west2**. Only the API has a public HTTP domain. Worker and future backup need no public endpoint or volume. The unused PostgreSQL 18 deployment is stopped to avoid idle compute charges; its original volume and service configuration remain preserved. It is not used by Eridani's cloud preview.

Deploy PostgreSQL first. **Do not run migrations against the intended restore target before its initial restore.** Preview migration hooks target `eridani_preview`; the intended restore databases remain empty. Keep the worker paused and external actions disabled throughout the rehearsal.

### PostgreSQL and PITR gate

Start from an official Railway PostgreSQL 16 service so database variables, its persistent volume and recovery controls are present. Preserve the volume mount and `PGDATA` layout provided by the template. Never point two running PostgreSQL containers at the same volume.

The candidate Dockerfile extends `ghcr.io/railwayapp-templates/postgres-ssl:16`. It preserves the upstream entrypoint, PGDATA and pgBackRest tooling. Local startup, pgvector queries, application migrations and logical restore passed. It does **not** prove that Railway's dashboard recognizes this GitHub-built image for managed PITR.

Before selecting that candidate for production, verify in the actual project that PITR can be enabled and that a timestamp restore boots the same extension-capable image. If Railway requires a supported registry image or other template setup, resolve that in staging. Keep the official PostgreSQL 16 image while vector indexing remains deferred, or use Neon with pgvector if the combined recovery/extension requirement cannot be met cleanly. Never replace reliable PITR with an unverified image.

Enable PITR in PostgreSQL's **Backups** tab. Railway creates its archive bucket and sets `WAL_ARCHIVE_*` variables. Keep those credentials on PostgreSQL only; this bucket is separate from R2. After the first base backup and healthy archiving, perform a timestamp restore into a sibling database and verify a harmless before/after record. A restored sibling needs PITR enabled again before it becomes the primary. [Railway PITR instructions](https://docs.railway.com/volumes/point-in-time-recovery)

A working `pgbackrest version` command is not proof of recoverability. Do not cut over until the actual archive and timestamp restore are verified.

## Service variables

Use variable references to the PostgreSQL service's **private** connection URL for API, worker and backup. `JARVIS_DATABASE_URL` accepts Railway's `postgresql://` or `postgres://` URL and selects the installed psycopg 3 driver. It takes precedence over `DATABASE_URL`; avoid setting conflicting copies. Use a direct connection, not a transaction pooler, because worker coordination and DBOS use session features.

The handoff file's database URL is for operator checks. A `*.railway.internal` address resolves inside Railway, not on your PC. Run database preflight in the deployed service, or use a deliberately temporary TLS-enabled public database connection for the local check. Remove that public database endpoint when the transfer is finished.

### API and worker

Share these values between API and worker:

- `JARVIS_DATABASE_URL`, `JARVIS_ORIGIN`
- Existing `JARVIS_OWNER_ID`, `JARVIS_OWNER_NAME`, `JARVIS_TIMEZONE`
- `JARVIS_GOOGLE_CLIENT_ID`, `JARVIS_GOOGLE_CLIENT_SECRET`, `JARVIS_INTEGRATION_ENCRYPTION_KEY`
- `JARVIS_OPENAI_API_KEY`, `JARVIS_GEMINI_API_KEY`
- Existing VAPID private/public keys and subject
- `JARVIS_COST_TRACKING_ENABLED=false`, preserving the current development choice
- `JARVIS_DATABASE_POOL_SIZE=5`, `JARVIS_DATABASE_MAX_OVERFLOW=2`
- `JARVIS_DBOS_POOL_SIZE=10`, `JARVIS_DBOS_CLIENT_POOL_SIZE=5`

Rehearsal-only values (production now enables worker/provider actions):

```dotenv
JARVIS_DEPLOYMENT_ENVIRONMENT=staging
JARVIS_PAIRING_ENABLED=false
JARVIS_WORKER_ENABLED=false
JARVIS_EXTERNAL_SERVICES_ENABLED=false
JARVIS_MAINTENANCE_MODE=false
WEB_CONCURRENCY=1
```

Do not set `JARVIS_OWNER_TOKEN` in cloud services. The public sign-in route is Google. API may set `FORWARDED_ALLOW_IPS=*` only while all inbound HTTP traffic enters through Railway's trusted HTTP proxy; do not expose a separate raw TCP route to that API. Railway supplies `PORT`.

Staging startup enforces both pause flags. The worker parks without starting DBOS; chat, voice, Calendar/Linear clients and linking are blocked from external effects. Google login remains available for checking access. Model and push credentials are cleared in memory while external services are paused. Edits may still change the staging database or queue work, so discard the rehearsal database and restore afresh before cutover.

For deliberately enabled controlled integration tests, use a separate synthetic environment labeled `production`, with test accounts and records. Do not enable a worker on a restored personal database while the original worker is active.

### Backup service only

Give the backup service `JARVIS_DATABASE_URL`, the existing `JARVIS_BACKUP_KEY`, and:

```dotenv
JARVIS_BACKUP_REQUIRE_REMOTE=true
JARVIS_BACKUP_S3_ENDPOINT=https://ACCOUNT_ID.r2.cloudflarestorage.com
JARVIS_BACKUP_S3_BUCKET=eridani-backups
JARVIS_BACKUP_S3_ACCESS_KEY_ID=SET_IN_RAILWAY
JARVIS_BACKUP_S3_SECRET_ACCESS_KEY=SET_IN_RAILWAY
JARVIS_BACKUP_S3_REGION=auto
JARVIS_BACKUP_S3_PREFIX=eridani/staging
JARVIS_BACKUP_DIRECTORY=/backups
```

Use `eridani/production` only for the production backup service. Keep HTTP storage endpoints disabled; the insecure flag exists solely for isolated tests. API/worker do not need R2 credentials or the backup encryption key.

Exports retain 30 days of daily snapshots and 12 weeks of Sunday snapshots. Pruning only considers recognized snapshot names within that service's prefix. Success is recorded in the database only after required remote archives have been uploaded and their length/hash metadata verified. Download additionally checks the payload checksum and authenticates Fernet encryption.

Never delete the PITR bucket or the R2 bucket as part of routine deployment cleanup.

## Rehearsal and initial data restore

These are operator commands for the relevant image/service after its variables have been securely populated. They contain no credentials.

On the development machine:

```sh
.venv/bin/python scripts/prepare_cloud.py
.venv/bin/python scripts/prepare_cloud.py --check
# Only with a target address reachable from this machine:
.venv/bin/python scripts/prepare_cloud.py --check --database
```

The first command does nothing if `.env.cloud` already exists. Configuration checks do not start services. `--database` is read-only and refuses a cloud cutover configuration without the owner's linked Google identity.

For the restore, run the backup image as a temporary operator job with its variables and a private archive staged under `/backups`. Override the scheduled command while restoring; never enable the recurring source worker. Use an existing local encrypted export, or first upload a **copy** into the staging R2 prefix using the backup tool. Do not expose an archive via a public download URL.

To download a known object:

```sh
python3 /usr/local/bin/jarvis-backup.py download \
  --object eridani/staging/daily/EXACT_BACKUP_FILENAME.pgdump.enc
```

Restore to a new disposable database for a drill:

```sh
python3 /usr/local/bin/jarvis-backup.py restore \
  --file /backups/EXACT_BACKUP_FILENAME.pgdump.enc \
  --target jarvis_restore_rehearsal
```

For an already-created but **empty** Railway application database, use its actual database name:

```sh
python3 /usr/local/bin/jarvis-backup.py restore-empty \
  --file /backups/EXACT_BACKUP_FILENAME.pgdump.enc \
  --target railway
```

Both restore modes authenticate the encrypted dump and use a single transaction with stop-on-error. The normal mode refuses an existing database; the empty mode refuses an occupied database and refuses system database names. Neither starts a worker or uses `--clean`. The archive includes public application tables, DBOS schema, sequences and Alembic version state; ownership and ACLs are remapped to the restore user.

Compare all reported row counts against the frozen source. Preserve and check both encryption keys, Google identity linkage, roles/workspace membership, tasks, notes, schedules, memories, pending jobs and DBOS workflows. Counts alone do not establish semantic correctness; sample representative records privately in the UI.

After the restore, run from the application image:

```sh
/app/.venv/bin/python -m jarvis.deploy migrate
/app/.venv/bin/python -m jarvis.deploy preflight --database
```

Then start API with the staging flags above. Check HTTPS, Google login, rejection of pairing sessions, task/note details, private/shared access, and calendar data. Start the paused worker and verify it does not consume jobs. In separate controlled tests, verify one scheduled action survives worker restart without duplication.

Run the separate Railway PITR timestamp drill. R2 export/download/restore remains deferred until its credentials are supplied. Configure deployment-failure notifications and verify Settings' backup/worker timestamps. Fully automated restore drills and dedicated stale-backup alerts remain follow-up work.

## Cutover order

1. Confirm cloud login, restore and device checks passed. Choose a short quiet period with no active voice session.
2. Freeze source writes. Maintenance mode blocks new API requests, but does not drain existing voice/SSE requests or stop a worker. Stop/drain the old API, worker and scheduled backup process before the final export.
3. Take the final encrypted export while the old writer processes are stopped. The database stays running to produce it. Record source row counts and backup hash.
4. Restore into a fresh cloud database. Do not overwrite the rehearsal database or merge two independently edited copies. Run migrations, compare counts and run preflight.
5. Set cloud `JARVIS_DEPLOYMENT_ENVIRONMENT=production`, keep pairing off, and enable `JARVIS_EXTERNAL_SERVICES_ENABLED=true`. Initially keep the worker paused while checking owner login.
6. Enable **only the cloud worker**, then verify one task edit, one due notification, Calendar/Linear synchronization and one controlled external write. Check PITR archive health. Enable the production R2 backup job and confirm its first remote copy only after its separate credentials are supplied.
7. On desktop and Android, verify Live start, tool use, interruption, transcript streaming and reconnect. Re-enable browser notifications on the new origin. Existing push subscriptions and cookies belong to the old origin; sign in and subscribe again on each device.
8. Test with the home PC offline. Keep the old deployment stopped as a recovery copy. Enable GitHub autodeploys only after the release behavior is understood.

The API and worker migration hooks are serialized, but migrations must still remain compatible with the previous release until it stops. Destructive schema changes need a separate deployment plan. Do not run major PostgreSQL upgrades as part of this move.

## Rollback

Before accepting cloud writes, rollback can restart the unchanged local deployment after stopping cloud API and worker.

After accepting cloud writes, the old database is stale. Freeze cloud writes, export the latest cloud database and restore it to a fresh compatible target before switching back. Simply restarting the old local database would lose new work and may duplicate scheduled effects.

A PITR restore also creates a separate database. Keep its workers paused while inspecting it. Enable archiving on the replacement before promoting it, and deliberately retire the old writer.

## Verification status

Completed locally with isolated synthetic records, without changing the running app:

- API/frontend, backup and PostgreSQL cloud images built.
- The former JSON configs passed the old provider schema but were rejected for new services; replaced with a reviewed `.railway/railway.ts` plan.
- PostgreSQL booted with its inherited Railway wrapper; pgBackRest 2.59.1 is present; pgvector 0.8.6 executed a similarity query.
- Alembic migrated an empty database to the current release; DBOS initialized and stopped cleanly.
- Encrypted S3-compatible export, upload, download, authentication and both restore modes preserved all **55 public/DBOS tables**. Repeating the empty restore was refused without altering records.
- Staging preflight on the restored database verified the linked fixture owner and decryption of Google and Linear credentials.
- Release verification: **521 backend tests passed, one skipped; 101 frontend tests passed, one skipped; production frontend build, Ruff and diff checks passed.**

Verified in Railway on September 16:

- Web/API and the active production worker deployed successfully from the prepared release; both connect to the private `eridani` database.
- `app.eridani.app` serves HTTPS with a valid certificate. Root/live/ready return 200; unauthenticated bootstrap returns 401; pairing is disabled; Google login generates the correct callback and a Secure/HttpOnly/SameSite cookie.
- Official PostgreSQL 16 image boots **16.15**, matching local 16.15. PITR has a completed base backup (`20260916-061241F`) and continuous WAL archiving with zero observed failures.
- Final encrypted snapshot `jarvis-20260916T064911481206Z.pgdump.enc` remains in the ignored local `.runtime/cloud-backups` directory. Full data transfer and exact per-table count comparison passed. Google identity linkage, current migrations, DBOS schema and credential decryption passed. The owner approved this transfer before it ran.
- Native timestamp restore **passed**: a disposable sibling restored the synthetic marker to `before`, while the source retained `after`; both ran PostgreSQL 16.15. The target was 2026-09-16T06:31:14.918014Z. The disposable restore service and its volume were removed after verification.
- Railway's separate manual-backup creation endpoint returned `OAUTH_INSUFFICIENT_GRANT`. Native PITR enablement, automatic base backup, archiving and timestamp-restore request worked with current access. No persistent SSH key was added.

Production worker heartbeat is current. Google Calendar status is ready, read/write permission was preserved, and a cloud sync completed at 2026-09-16T06:54:28Z without an error. No Linear connection is currently configured; its code remains available. Recovery archiving remained healthy after the final restore. Temporary public database access was removed; the API remained ready using private networking.

Local API, worker and backup containers are stopped. The Windows Startup `Jarvis.lnk` shortcut was moved to `.runtime/retired-startup` so a reboot cannot restart the old writer. Do not restart the stale local stack as production now that the cloud accepts writes. A separate development database should be used for future local work.

Still required from the owner: actual Google browser sign-in, spoken voice/microphone checks, Android notifications and any desired controlled integration writes. Desktop and phone-sized login-page checks passed without page errors or horizontal overflow. R2 exports remain deferred pending bucket-scoped credentials; native PITR is active and its restore drill passed.

## Keeping PostgreSQL versions aligned

Keep the major version **16** in the local Compose image, backup client image and cloud infrastructure image. Railway's recovery-enabled image uses its supported major tag; development pins the tested minor (`16.15-bookworm`). Before redeploying PostgreSQL, check the candidate image's actual `SHOW server_version`, test migrations and a logical restore on that version, then update the development/backup minor pins in the same reviewed change. A major upgrade is a separate restore/upgrade procedure, never a tag swap against an existing volume. The pgvector candidate stays deferred until indexing is implemented and its native recovery behavior is verified.
