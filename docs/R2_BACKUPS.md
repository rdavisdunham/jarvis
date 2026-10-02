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

## Key escrow checklist

Backups are only recoverable if the keys survive the loss of Railway and of this machine. Two keys matter: `JARVIS_BACKUP_KEY` (decrypts backup archives) and `JARVIS_INTEGRATION_ENCRYPTION_KEY` (decrypts Google/Linear credentials, work and action history inside a restored database). Record the answers here (names/locations only, never key material):

- [ ] **Holders.** Primary holder: ______. Second holder or sealed location (e.g. password manager vault shared with a trusted person, offline printed copy in a safe): ______. Neither copy may live only in Railway or only on this laptop.
- [ ] **Versions.** Each escrowed copy is labelled with its key fingerprint (first 8 hex of SHA-256 of the key) and the date it became active. Retired keys stay escrowed while any backup or PITR window encrypted under them is retained.
- [ ] **Test decrypt from escrow.** Using only the escrowed copy (not Railway variables), download the latest R2 archive with `download --object` and confirm checksum + authenticated decryption pass. Record date and result: ______.
- [ ] **Restore drill.** Restore that archive into a disposable empty database with the escrowed integration key; run `python -m jarvis.deploy preflight --database` with worker and external services disabled; confirm table counts and `integration_decryption: passed`. Record date and result: ______. Repeat at least quarterly and after any key rotation.
- [ ] **Rotation rule.** Never rotate either key without escrowing the new one and re-running the test decrypt first.

Still outstanding: credentials, actual R2 upload/download/restore drill, and scheduled production run. No R2 credentials were requested or created in this batch.

Sources: [Cloudflare S3 access](https://developers.cloudflare.com/r2/get-started/s3/), [bucket-scoped tokens](https://developers.cloudflare.com/r2/api/tokens/), [Railway scheduled jobs](https://docs.railway.com/cron-jobs), [configuration reference](https://docs.railway.com/config-as-code/reference).
