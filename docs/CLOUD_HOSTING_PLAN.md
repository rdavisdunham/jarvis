# Eridani cloud hosting plan

Deployment update (September 16): the actual project is colocated in Railway us-west2. The original Virginia suggestion below was an initial planning candidate.

## Recommendation

Host Eridani's API, website, background worker and PostgreSQL together on Railway. Enable Railway's point-in-time recovery and keep a separate encrypted backup copy in Cloudflare R2. Begin with the Hobby plan for the current personal deployment, subject to verifying the selected PostgreSQL image and recovery controls in a staging project. Use Pro when its storage or operational features are needed.

Allow **$15–25 per month for this initial hosting arrangement**, excluding AI usage, domain registration, tax and temporary staging. This is a planning range, not a provider quote or a capacity guarantee. The illustrative model below totals **$14.15**; its RAM, CPU, traffic and archive assumptions need checking against the first cloud bill. On Pro the same modeled resources would cost **$20**, because its subscription is a minimum that credits resource usage. [Railway pricing](https://docs.railway.com/pricing)

The preferred alternative is **Railway for the app and Neon Launch for PostgreSQL**, around **$30–40 per month** under the small-workload assumptions. Choose that arrangement if reducing database maintenance is worth about $15–20 more per month, or if Railway's image and recovery rehearsal fails. Supabase is also a sound option, especially if its Auth or Storage products become useful, but it is not necessary for the existing app.

The user has accepted the migration direction and authorized code preparation. See the [migration runbook](CLOUD_MIGRATION.md) for prepared configuration, local verification and the owner setup checklist. No infrastructure has been provisioned, production settings changed or live data migrated. Prices and documentation were checked on September 14, 2026. The baseline is one owner and a small number of invited users, with hosted AI models and background work available all day.

## Actual workload and implications

The active deployment is described by `compose.upgrade.yml` and `Dockerfile.upgrade`. It uses a Python FastAPI service that also serves the website, a separate DBOS background worker, PostgreSQL 16, a migration job and an encrypted backup job. The older local inference stack is not a cloud requirement. There is no reason to rent a GPU for this migration.

Three read-only samples across 55 seconds showed approximately **161 MiB for the API, 203 MiB for the worker, 160 MiB for PostgreSQL and 16 MiB for the backup container**. The database occupied **19.1 MiB**. CPU varied during the sample. These measurements establish that the current installation is small; they do not establish peak capacity, future user capacity or a monthly bill.

The worker scans for scheduled work every five seconds. DBOS maintains database state and listeners. Each open event stream also polls for authorization changes and new events every half-second. Consequently, a serverless database would probably remain awake throughout the day. Neon's advertised intermittent-use examples should not be used as Eridani's cost estimate. [Neon scale-to-zero behavior](https://neon.com/docs/introduction/scale-to-zero)

Live audio travels **directly between the browser and OpenAI over WebRTC**. The API negotiates the session and maintains an outbound provider connection for transcripts and delegation. Hosting estimates should include application traffic and tool activity, but should not assume Railway relays the entire audio stream. This finding comes from `apps/web/src/voice.ts` and `apps/api/jarvis/live_voice.py`, not from a generic voice-hosting assumption.

Active voice controllers currently live in the API process's memory. Deploy **one API process in one replica** initially. Additional replicas would require session ownership and routing changes. Keep one DBOS worker deployment initially too; review recovery coordination before adding executors. Neither moving to a cloud host nor buying a larger plan removes these application constraints.

## Proposed deployment

Place the following services in one Railway project and region:

- **Web/API:** the existing Docker image, one process and replica, HTTPS on the public application domain. Serve the current frontend from the API initially.
- **Worker:** the same application image with its worker command, always running, without a public endpoint.
- **PostgreSQL:** a separate persistent service reached over the project's private network. Start from Railway's supported PostgreSQL 16 major image, after compatibility checks.
- **Migrations:** a controlled release step, executed once per release before compatible application services start.
- **Backup export:** a scheduled job that encrypts a logical database dump and uploads it to a private R2 bucket.

Use Railway's Virginia region as the initial candidate, with actual latency checked during staging. If the database is external, select the corresponding Northern Virginia region there. Geographic proximity is not the same as a shared private network or identical data center. [Railway regions](https://docs.railway.com/deployments/regions), [private networking](https://docs.railway.com/networking/private-networking/how-it-works)

Keep the primary database private. Store secrets in service variables with only the services that need them. Keep the database, OAuth credentials, encryption keys and billing accounts dedicated to Eridani. There is no reason to share Brainforge's production database or credentials.

Cloudflare can handle DNS and the independent backup bucket. Start with DNS pointing directly to Railway's HTTPS service; adding a second application proxy is optional and should follow streaming tests. The home PC remains a development machine and is not required for reminders, sync, voice sessions or backups to run.

## Database choice

### Railway PostgreSQL: best initial cash cost

Railway's database is a PostgreSQL container on persistent storage, with configuration and maintenance owned by the application operator. Its documentation explicitly describes the templates as unmanaged. That does not mean backups must be invented from scratch; it does mean updates, sizing, extension compatibility, monitoring and recovery ownership remain ours. [PostgreSQL service documentation](https://docs.railway.com/databases/postgresql)

Railway now documents native **point-in-time recovery**, which can restore to a chosen timestamp in the retained history. It uses pgBackRest, base backups and archived write-ahead logs, retains roughly four weeks, and restores into a separate service. There is no separate PITR fee; compressed archive storage and service upload traffic are billed. Minor-version pinning is unsupported by this workflow, so use its supported major image and rehearse updates. Confirm feature availability for the actual project before relying on it. [Railway recovery documentation](https://docs.railway.com/volumes/point-in-time-recovery)

Hobby's database volume ceiling is 5 GB. That is ample for today's 19 MB logical database, but PostgreSQL files, indexes, temporary work and WAL also consume storage. Upgrade before space becomes tight. Volume-backed services have deployment and replication constraints; this starting configuration is not a high-availability database. [Plan limits](https://docs.railway.com/pricing/plans), [volume reference](https://docs.railway.com/volumes/reference)

The default image does not promise pgvector. A candidate image now installs pgvector 0.8.6 on Railway's PostgreSQL 16 base while retaining its recovery wrapper; local startup and logical restore passed. Managed PITR recognition and timestamp restore still require a Railway rehearsal. Keep vector indexing as a planned upgrade, or move the database to Neon if the combined extension/recovery requirement is not met. Do not swap in an arbitrary community template and assume backup behavior survives. Standard PostgreSQL exports preserve the exit path. [Extension guidance](https://docs.railway.com/databases/postgresql)

### Neon Launch: preferred managed database alternative

Neon provides managed PostgreSQL and recovery history without requiring Eridani to adopt a new application API. Its pgvector extension is available on every plan. [Neon pgvector documentation](https://neon.com/docs/extensions/pgvector) Its Launch compute rate is **$0.106 per CU-hour**. At a continuously running 0.25 CU and 730 hours, compute alone is **$19.35 per month**. Storage is **$0.35 per GB-month**, retained history **$0.20 per GB-month**, and paid public transfer includes 500 GB before overage. [Neon pricing](https://neon.com/pricing)

Configure the retention window explicitly to seven days and turn production scale-to-zero off. Use a modest compute maximum initially and measure it: at 0.5 average CU, compute becomes $38.69; at 1 CU, $77.38. These are billing calculations, not estimates of how many users those sizes serve.

Connect DBOS through an **unpooled connection**, not Neon's transaction-pooled endpoint. Persistent session features are required. Managed recovery still needs a restore drill and an independent export. [Neon connection pooling](https://neon.com/docs/connect/connection-pooling), [backup and restore](https://neon.com/docs/manage/backups)

Neon becomes especially attractive when database maintenance or vector-image upkeep begins consuming time. It adds another provider and cross-provider database latency, so it is a deliberate operational tradeoff rather than an automatic performance upgrade.

### Supabase Pro: useful bundle, optional here

Supabase Pro starts at **$25 per month**, including credits for one Micro database, 8 GB database disk, 250 GB ordinary egress and daily backups retained for seven days. A separate additional Micro project starts at $10 if an existing paid organization's included credit is already used. That could improve incremental pricing if there is an appropriate personal organization; this plan does not assume access to Brainforge's account. [Supabase pricing](https://supabase.com/pricing)

The distinction between backups matters: daily backups do not provide recovery to an arbitrary moment between them. Supabase's seven-day point-in-time recovery add-on is about **$100 per month extra** and requires at least Small compute. Independent daily exports can improve resilience, but do not turn daily recovery into PITR. [Supabase backup documentation](https://supabase.com/docs/guides/platform/backups)

Eridani can use Supabase purely as PostgreSQL. It does not need to replace its existing Google sign-in, permissions, models or command service with Supabase Auth and browser database calls. In that configuration, **disable the Data API**, or otherwise prevent it from exposing restored application tables. The current app's authorization is enforced by its backend and cannot be assumed to exist as Supabase row-level policies. [Supabase hardening](https://supabase.com/blog/hardening-supabase), [Data API controls](https://supabase.com/blog/supabase-security-2025-retro)

For DBOS, choose a direct connection or the session pooler. Supabase's transaction pooler is unsuitable. Railway now offers opt-in outbound IPv6, which can reach Supabase's IPv6 direct endpoint without assuming a paid IPv4 add-on is necessary. Test the selected route and TLS configuration. [DBOS Supabase integration](https://docs.dbos.dev/integrations/supabase), [Supabase connection options](https://supabase.com/docs/guides/database/connecting-to-postgres), [Railway outbound networking](https://docs.railway.com/networking/outbound-networking)

## Other hosting options

**Render** is the strongest simple alternative with a managed database on the same platform. An illustrative configuration is two $7 services for web and worker, a $19 1 GB database, disk and bandwidth. That totals about **$36.55 per month** in the scenario below. The $6 database tier exists, but its 256 MB memory gives less room than the 1 GB comparison. Paid databases have PITR; retention depends on the workspace plan. The low service tiers also need load testing before assuming they fit. [Render pricing](https://render.com/pricing), [database backups](https://render.com/docs/postgresql-backups), [storage sizing](https://render.com/docs/postgresql-creating-connecting)

**A DigitalOcean virtual machine** gives straightforward Docker Compose portability. A 2 GB machine is $12 per month, or $15.60 with daily VM backups at the listed 30% surcharge. The price is competitive, but OS patching, firewall setup, PostgreSQL maintenance and recovery all become our responsibility. VM snapshots are not a replacement for database recovery and independent dumps. This is a reasonable choice for someone who wants to operate a server; it does not buy much relief from maintenance. [Droplet pricing](https://www.digitalocean.com/pricing/droplets)

**Cloudflare Workers/Containers** are not the best default for this existing Python worker architecture. Containers add Worker/Durable Object routing and use ephemeral disk, so the database still belongs elsewhere. Two always-on Basic containers with 1 GiB each model at about $21.51 before the external database and any additional Workers/Durable Objects usage. Cloudflare's low headline entry price therefore does not describe this complete application. Its strengths are useful at the edges, especially R2 storage. [Container architecture](https://developers.cloudflare.com/containers/concepts/architecture/), [container pricing](https://developers.cloudflare.com/containers/platform/pricing/)

Large cloud platforms and Kubernetes are not required to preserve future options. The portable assets here are the Docker image, PostgreSQL schema, migrations, HTTP API and standard object-storage interface. A low-level cloud deployment could be evaluated later if measured scale, compliance or availability requirements justify its operational cost.

## Cost model and uncertainty

The interactive companion report allows the principal assumptions to be changed. It is a sensitivity model, not a live billing connection.

The small-workload scenario assumes a 730-hour month, 0.60 GB mean API-plus-worker RAM, 0.05 mean vCPU, and 20 GB outgoing application/export traffic. The all-Railway database adds 0.25 GB mean RAM including a provisional archiver allowance, 0.03 mean vCPU and 5 GB billed volume storage. It assumes 20 GB compressed archive uploads per month and 20 GB-month archived storage. A separate $1 allowance covers backup-job compute. These are modeling assumptions, not measured monthly consumption.

For an external database, the model assumes 1 GB logical data, 5 GB of outbound database requests and 100 GB of database response traffic. Neon retains 2 GB-month of history in this example. The R2 backup copy uses 2 GB and low request volume, within its published free allowances if otherwise unused. R2's standard storage above the free allowance is $0.015 per GB-month. [R2 pricing](https://developers.cloudflare.com/r2/pricing/)

The resulting illustrative monthly totals are:

- **Railway together: $14.15.** API/worker $7 compute; database $3.10 compute; application/export egress $1; volume $0.75; backup job $1; archive upload $1; archive storage $0.30.
- **Railway plus Neon: $29.35.** Railway $9.25 including database-request egress and backup job; Neon $20.10 for compute, data and history.
- **Railway plus Supabase: $34.25.** Railway $9.25; Supabase $25 with daily backups.
- **Render: $36.55.** Web/worker $14; database $19; 1 GB disk $0.30; 15 GB beyond the modeled included 5 GB bandwidth $2.25; backup-job allowance $1.
- **DigitalOcean: $15.60.** 2 GB VM and daily VM backups; backup-job compute shares the machine. Independent small R2 storage is within the assumed free allowance.
- **Cloudflare Containers plus Neon: at least $42.61.** Two Basic containers and base Workers subscription $21.51, Neon $20.10 and export-job allowance $1. Additional platform usage is excluded, so this is a lower bound.

These options have different performance and recovery properties; the totals are not a benchmark of equivalent machines. Railway bills used RAM/CPU, Render and a VM sell instance sizes, and Cloudflare container RAM/disk are based on provisioning. The database image and pgBackRest may use more resources than the small local PostgreSQL sample. Backup churn can also exceed the illustrative archive allowance. [Railway resource rates](https://docs.railway.com/pricing), [archive billing](https://docs.railway.com/storage-buckets/billing)

The estimates exclude AI model usage, domain registration, taxes, outbound email/SMS subscriptions, paid monitoring, provider support upgrades, restore copies and temporary preview environments. They exclude labor. A staging rehearsal can briefly duplicate services and incur extra cost. Do not infer a monthly network bill from Docker's cumulative counters across different container uptimes.

## Brainforge patterns to reuse

Brainforge's platform repository already describes **Railway application hosting and Supabase databases**, with Google authentication through Supabase. Its Railway configuration declares explicit start and healthcheck behavior; its Dockerfile provides another deployable packaging route. The database guide describes several projects for different platform responsibilities.

Reuse the separation between application hosting and database credentials, explicit build/start paths, health checks, and server-only privileged credentials. Keep Eridani's existing authentication and one primary PostgreSQL database. Brainforge's several-database arrangement is not a reason to give every Eridani user or workspace a separate database.

This evidence is from repository configuration and documentation, not a review of Brainforge's production bills or availability. Relevant files are `brainforge-platform/apps/platform/railway.toml`, `Dockerfile.standalone`, `docs/databases.md`, `src/utils/supabase/server.ts` and `src/app/api/healthz/route.ts`. Its historical deployment investigation also supports rehearsing monorepo build paths and startup health checks rather than relying on defaults.

## Migration batches

### 1. Prepare deployment and public access

Add explicit Railway service configuration around the existing Dockerfile, correct monorepo build context, target port, API command and worker command. Review the Docker build context so local credentials, databases and backups cannot enter the image. Make migrations a serialized release operation. Separate liveness from dependency readiness and keep readiness independent of login redirects.

Before exposing the app publicly, review the existing Google sign-in and invitation flow. The old shared short pairing PIN must not remain an unrestricted public login path. Validate secure cookies, allowed origins, forwarded-proxy handling, request limits, bot-token scopes and workspace isolation. The host's HTTPS does not replace those application controls.

Update OAuth redirect URLs and any provider webhook destinations to the new domain. Preserve the keys required to decrypt stored Google/Linear credentials and backups. Existing browser cookies, notification permissions and push registrations belong to the old origin, so plan for sign-in and permission renewal on the new domain.

### 2. Rehearse PostgreSQL and recovery

Create an isolated staging database and restore an encrypted export. Include application tables, the **DBOS system schema**, migration state, sequences and the permissions needed to operate them. Do not restore only the task tables. Map roles deliberately rather than attempting to copy unavailable provider superusers.

Keep staged integration polling, scheduled notifications and external writes disabled. A restored queue can otherwise repeat real effects. Redact or restrict copied personal data; a staging URL is not a public demo. Verify row counts and integrity without exposing task or memory contents in logs.

Use a PostgreSQL dump client compatible with the target server. The current client is version 16; a target with a newer major version requires an appropriate client for future backups. Rehearse any major-version change separately from the hosting cutover.

Enable native recovery, create a harmless test record, restore to a chosen earlier point into an isolated database, and prove the expected record state. Separately download and decrypt an R2 export and restore it. The target is recovery in under an hour, measured in the drill. Successful backup creation alone does not satisfy this batch.

### 3. Verify application behavior in the cloud

Exercise task and note editing, private/shared permissions, immediate revocation, search, memory retrieval and project/calendar navigation. Test Google and Linear integrations only with controlled records and deliberately enabled connections. Verify a scheduled task executes exactly once across worker restart.

Tune database connection pools across the API, application worker engine and DBOS clients. The current SQLAlchemy engine allows 10 pooled connections plus 10 overflow per process, and DBOS has its own pools. Those totals can exhaust a small managed database even though the sample showed only 15 connections. Use direct or session-pooled PostgreSQL connections because DBOS uses session features such as LISTEN/NOTIFY. [DBOS production checklist](https://docs.dbos.dev/production/checklist), [pool and recovery guidance](https://docs.dbos.dev/faq)

Keep one API replica and test voice start, tools, interruption, transcript updates and recovery from connection loss on desktop and Android. A deploy may end an active voice session; verify a clear reconnect path. Test event-stream resumption across Railway's documented HTTP duration limit, preserving cursors and authorization checks. [Railway networking limits](https://docs.railway.com/networking/public-networking/specs-and-limits)

Measure database latency, pool utilization, CPU/RAM, backup storage and outgoing traffic. Preserve immediate access revocation while reducing avoidable polling work. Optimize noisy queries before assuming a larger database is the answer.

### 4. Cut over once, with a defined rollback

Freeze application writes and stop the old background worker and integration jobs. Take a final consistent export, restore it to the cloud, verify critical counts and credentials, then point the cloud services at that database. Enable only the cloud worker after the checks pass.

Change the public domain and OAuth configuration, then verify login, one task change, one scheduled action and voice. Retain the old environment stopped as a recovery option until the cloud deployment passes its initial checks.

Before new cloud writes, rollback can return to the old database. After new cloud writes, restarting the stale local database would lose changes. At that point, rollback requires freezing cloud writes and migrating the latest cloud data back or repairing the cloud deployment. This distinction belongs in the cutover runbook.

### 5. Operate and measure

Alert on failed or stale exports, unhealthy archive status, missed worker heartbeat, low disk space and persistent integration errors. Keep daily encrypted exports for 30 days and weekly copies for 12 weeks, subject to measured storage cost. Keep the encryption key outside the backup bucket and document how to retrieve it.

Set soft budget alerts first. Railway hard usage limits can stop services and suspend bucket access; an aggressive cutoff is inappropriate for an assistant responsible for reminders. Choose any hard ceiling deliberately after observing real usage. [Railway cost controls](https://docs.railway.com/pricing/cost-control)

A single-node deployment can be unavailable during failures or maintenance. Point-in-time recovery reduces potential data loss; it does not make the application highly available. Rehearse recovery monthly initially and after database changes. If the Railway archive is unhealthy, the independent export's actual age determines the fallback data-loss window.

## Future expansion and exit conditions

**Android, additional users and external bots** can use the same HTTPS API and permission model. Hosting does not require separate databases per user or a switch to Supabase Auth. Add scoped bot tokens and the MCP adapter through the existing command service.

**Notes, embeddings and memory** can remain in PostgreSQL with a future pgvector index. Verify the extension and backup combination before enabling it. Use R2 for larger attachments and exports rather than expanding the application container's disk. Add another vector database only when measured retrieval or scale requirements justify it.

**More background work** can use additional queues and eventually separate workers. DBOS multi-executor recovery and side-effect idempotency need an explicit review before enabling replicas. Long-running or untrusted agent code should run in an isolated execution service, not inside the public API container.

**Higher traffic** should first prompt query and polling measurements, then vertical scaling. Horizontal API scaling requires moving or routing live-session state. Migrate from Railway PostgreSQL to Neon or another managed PostgreSQL service if maintenance, pgvector image support or availability requirements outweigh the monthly savings.

**Future provider changes** remain practical if the domain model stays in ordinary PostgreSQL migrations, the API remains portable Docker, and files use an S3-compatible storage interface. Provider backup formats are useful for local recovery; independent logical exports are the cross-provider escape route. Supabase, Neon and Railway do not impose a feature ceiling on the planner itself, but every host has limits that should be measured before a scaling decision.

## Decision and completion criteria

Proceed with **all-Railway plus independent R2 backups** as the default migration design. Switch the database choice to **Neon Launch** if the supported image, archive behavior or restore drill does not meet the operational requirements. Choose Supabase instead if its bundled services or an appropriate existing personal paid organization materially improve value.

The migration is complete only when the cloud services work with the home PC offline, real device voice and notifications pass, private/shared data stays isolated, the scheduled worker recovers without duplicate effects, and both native and independent backup restores have been demonstrated. A public homepage loading is only the first check.

## Evidence and sources

Prices are in USD and were checked September 14, 2026. Official provider documentation describes offered behavior; plan/account availability, latency, failure recovery and monthly resource usage still require a cloud rehearsal. No provider account console or invoice was audited.

Local evidence: `compose.upgrade.yml`, `Dockerfile.upgrade`, `apps/api/jarvis/db.py`, `worker.py`, `api.py`, `voice.py`, `live_voice.py`, `apps/web/src/voice.ts`, `scripts/backup.py`, and three read-only resource samples at 22:24:42–22:25:37 UTC. Brainforge evidence is the platform repository files named above; no Brainforge files or infrastructure were changed.

1. Railway. [Pricing](https://docs.railway.com/pricing). Accessed September 14, 2026.
2. Railway. [Plan limits](https://docs.railway.com/pricing/plans). Accessed September 14, 2026.
3. Railway. [PostgreSQL responsibilities and extensions](https://docs.railway.com/databases/postgresql). Accessed September 14, 2026.
4. Railway. [Point-in-time recovery](https://docs.railway.com/volumes/point-in-time-recovery). Accessed September 14, 2026.
5. Railway. [Bucket billing](https://docs.railway.com/storage-buckets/billing). Accessed September 14, 2026.
6. Railway. [Volume limitations](https://docs.railway.com/volumes/reference). Accessed September 14, 2026.
7. Railway. [Public networking limits](https://docs.railway.com/networking/public-networking/specs-and-limits). Accessed September 14, 2026.
8. Railway. [Private networking](https://docs.railway.com/networking/private-networking/how-it-works). Accessed September 14, 2026.
9. Railway. [Outbound networking](https://docs.railway.com/networking/outbound-networking). Accessed September 14, 2026.
10. Railway. [Cost control](https://docs.railway.com/pricing/cost-control). Accessed September 14, 2026.
11. Railway. [Regions](https://docs.railway.com/deployments/regions). Accessed September 14, 2026.
12. Neon. [Pricing](https://neon.com/pricing). Accessed September 14, 2026.
13. Neon. [Scale to zero](https://neon.com/docs/introduction/scale-to-zero). Accessed September 14, 2026.
14. Neon. [Connection pooling](https://neon.com/docs/connect/connection-pooling). Accessed September 14, 2026.
15. Neon. [Backup and restore](https://neon.com/docs/manage/backups). Accessed September 14, 2026.
16. Supabase. [Pricing](https://supabase.com/pricing). Accessed September 14, 2026.
17. Supabase. [Database backups](https://supabase.com/docs/guides/platform/backups). Accessed September 14, 2026.
18. Supabase. [Connecting to Postgres](https://supabase.com/docs/guides/database/connecting-to-postgres). Accessed September 14, 2026.
19. Supabase. [Hardening Supabase](https://supabase.com/blog/hardening-supabase). Accessed September 14, 2026.
20. Supabase. [Security changes in 2025](https://supabase.com/blog/supabase-security-2025-retro). Accessed September 14, 2026.
21. DBOS. [Production checklist](https://docs.dbos.dev/production/checklist). Accessed September 14, 2026.
22. DBOS. [Supabase integration](https://docs.dbos.dev/integrations/supabase). Accessed September 14, 2026.
23. DBOS. [FAQ and pool configuration](https://docs.dbos.dev/faq). Accessed September 14, 2026.
24. Render. [Pricing](https://render.com/pricing). Accessed September 14, 2026.
25. Render. [Postgres backups](https://render.com/docs/postgresql-backups). Accessed September 14, 2026.
26. Render. [Postgres creation and storage](https://render.com/docs/postgresql-creating-connecting). Accessed September 14, 2026.
27. Cloudflare. [Container pricing](https://developers.cloudflare.com/containers/platform/pricing/). Accessed September 14, 2026.
28. Cloudflare. [Container architecture](https://developers.cloudflare.com/containers/concepts/architecture/). Accessed September 14, 2026.
29. Cloudflare. [R2 pricing](https://developers.cloudflare.com/r2/pricing/). Accessed September 14, 2026.
30. DigitalOcean. [Droplet pricing](https://www.digitalocean.com/pricing/droplets). Accessed September 14, 2026.
31. Neon. [pgvector extension](https://neon.com/docs/extensions/pgvector). Accessed September 14, 2026.
