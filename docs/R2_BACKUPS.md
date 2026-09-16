# Independent R2 backups

The bucket and encrypted backup implementation are ready. Activation is deliberately deferred until bucket credentials are supplied. Railway PITR remains active and separate.

## Prepared configuration

Use a **separate** Railway service sourced from this repository. Set its config file path to `/deploy/railway/backup.railway.toml`. It uses the PostgreSQL 16.15 backup image and runs once daily at 09:00 UTC, exits after completion, and has no public domain. No permanent backup process or new database is needed.

Set these variables on that backup service only:

- `JARVIS_DATABASE_URL`: reference the production database URL over Railway private networking.
- `JARVIS_BACKUP_KEY`: preserve the existing backup encryption key in the private cloud environment file. Keep an independent recovery copy; never rotate it without preserving old keys.
- `JARVIS_BACKUP_REQUIRE_REMOTE=true`.
- `JARVIS_BACKUP_S3_ENDPOINT`: the R2 account's HTTPS S3 endpoint.
- `JARVIS_BACKUP_S3_BUCKET=eridani-backups`.
- `JARVIS_BACKUP_S3_ACCESS_KEY_ID` and `JARVIS_BACKUP_S3_SECRET_ACCESS_KEY`: bucket-scoped Object Read & Write credentials.
- `JARVIS_BACKUP_S3_REGION=auto`.
- `JARVIS_BACKUP_S3_PREFIX=eridani/production`.
- `JARVIS_BACKUP_DIRECTORY=/backups`.

The container's temporary disk is sufficient: a run succeeds only after the encrypted archive and checksum manifest are uploaded and verified remotely. The script retains daily copies for 30 days and Sunday copies for 84 days. Its prefix isolates production from staging.

## Activation and verification

1. Run `python3 /usr/local/bin/jarvis-backup.py check` inside the configured backup image. This only validates configuration, prints missing/invalid variable names, and never connects, exports, or prints credentials. Exit 2 means configuration needs attention. A passing check is **not** proof of a working remote backup.
2. Trigger one `once` run. Confirm `remote_verified: true` and the backup heartbeat in Eridani. A failed upload must not mark a successful backup.
3. Download that exact archive with the backup script's `download --object` command. It checks the checksum and authenticated encryption before keeping the file.
4. Restore into a **disposable empty database**, compare application and DBOS table counts, and verify a synthetic credential can authenticate. Preserve the application's integration encryption key for encrypted connections, work, and action history. Never restore over the active writer.
5. Enable the daily schedule and observe its first scheduled success. Retain Railway PITR for recovery between daily exports.

Still outstanding: credentials, actual R2 upload/download/restore drill, and scheduled production run. No R2 credentials were requested or created in this batch.

Sources: [Cloudflare S3 access](https://developers.cloudflare.com/r2/get-started/s3/), [bucket-scoped tokens](https://developers.cloudflare.com/r2/api/tokens/), [Railway scheduled jobs](https://docs.railway.com/cron-jobs), [configuration reference](https://docs.railway.com/config-as-code/reference).
