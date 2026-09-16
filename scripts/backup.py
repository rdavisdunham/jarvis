"""Encrypted PostgreSQL exports, S3/R2 copies and isolated, atomic restores."""

import argparse
import base64
import hashlib
import json
import os
import re
import signal
import subprocess
import tempfile
import threading
from datetime import UTC, datetime, timedelta
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlsplit

from cryptography.fernet import Fernet

stop = threading.Event()
BACKUP_NAME = re.compile(r"jarvis-(\d{8}T\d{6}(?:\d{6})?Z)\.pgdump\.enc")


def pg(*args, input=None):
    return subprocess.run(args, input=input, capture_output=True, check=True, timeout=1800).stdout


def configure_postgres():
    """Pass credentials through libpq environment, never command arguments."""
    raw = os.environ.get("JARVIS_DATABASE_URL") or os.environ.get("DATABASE_URL")
    if not raw:
        return
    url = urlsplit(raw.replace("postgresql+psycopg://", "postgresql://", 1))
    if url.scheme not in {"postgresql", "postgres"} or not url.hostname or not url.path.strip("/"):
        raise ValueError("Invalid PostgreSQL connection URL.")
    mapping = {
        "PGHOST": url.hostname,
        "PGPORT": str(url.port or 5432),
        "PGDATABASE": unquote(url.path.lstrip("/")),
        "PGUSER": unquote(url.username or ""),
        "PGPASSWORD": unquote(url.password or ""),
    }
    for key, env in {
        "sslmode": "PGSSLMODE",
        "sslrootcert": "PGSSLROOTCERT",
        "connect_timeout": "PGCONNECT_TIMEOUT",
        "options": "PGOPTIONS",
    }.items():
        values = parse_qs(url.query).get(key)
        if values:
            mapping[env] = values[-1]
    os.environ.update(mapping)
    os.environ.setdefault("PGCONNECT_TIMEOUT", "10")


def cipher():
    return Fernet(os.environ["JARVIS_BACKUP_KEY"].encode())


def counts(database):
    # Include DBOS and application tables. Do not print record contents.
    rows = (
        pg(
            "psql",
            "-XAt",
            "-d",
            database,
            "-c",
            """
        SELECT json_build_array(schemaname,tablename)
        FROM pg_tables WHERE schemaname IN ('public','dbos')
        ORDER BY schemaname,tablename
    """,
        )
        .decode()
        .splitlines()
    )
    result = {}
    for row in rows:
        schema, table = json.loads(row)
        identifier = ".".join('"' + part.replace('"', '""') + '"' for part in (schema, table))
        result[schema + "." + table] = int(
            pg("psql", "-XAt", "-d", database, "-c", "SELECT count(*) FROM " + identifier).strip()
        )
    return result


def remote_store():
    names = ["ENDPOINT", "BUCKET", "ACCESS_KEY_ID", "SECRET_ACCESS_KEY"]
    values = {name: os.environ.get("JARVIS_BACKUP_S3_" + name, "") for name in names}
    required = os.environ.get("JARVIS_BACKUP_REQUIRE_REMOTE", "").lower() == "true"
    if not any(values.values()) and not required:
        return None
    if not all(values.values()):
        raise ValueError("Complete the backup S3 credentials before running a remote backup.")
    endpoint = urlsplit(values["ENDPOINT"])
    insecure = os.environ.get("JARVIS_BACKUP_ALLOW_INSECURE_S3", "").lower() == "true"
    if not endpoint.hostname or (endpoint.scheme != "https" and not (insecure and endpoint.scheme == "http")):
        raise ValueError("Backup storage requires an HTTPS endpoint.")
    prefix = os.environ.get("JARVIS_BACKUP_S3_PREFIX", "eridani/production").strip("/")
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9/_-]{0,150}", prefix):
        raise ValueError("Invalid backup prefix.")
    import boto3
    from botocore.config import Config

    client = boto3.client(
        "s3",
        endpoint_url=values["ENDPOINT"],
        aws_access_key_id=values["ACCESS_KEY_ID"],
        aws_secret_access_key=values["SECRET_ACCESS_KEY"],
        region_name=os.environ.get("JARVIS_BACKUP_S3_REGION", "auto"),
        config=Config(
            connect_timeout=10,
            read_timeout=120,
            retries={"max_attempts": 3, "mode": "standard"},
            s3={"addressing_style": "path"},
        ),
    )
    return client, values["BUCKET"], prefix


def snapshot_time(name):
    match = BACKUP_NAME.fullmatch(name)
    if not match:
        return None
    stamp = match[1]
    fmt = "%Y%m%dT%H%M%S%fZ" if len(stamp) > 16 else "%Y%m%dT%H%M%SZ"
    return datetime.strptime(stamp, fmt).replace(tzinfo=UTC)


def upload(store, key, data, digest):
    client, bucket, _ = store
    client.put_object(
        Bucket=bucket,
        Key=key,
        Body=data,
        ContentType="application/octet-stream",
        ContentMD5=base64.b64encode(hashlib.md5(data).digest()).decode(),
        Metadata={"sha256": digest},
    )
    head = client.head_object(Bucket=bucket, Key=key)
    if head["ContentLength"] != len(data) or head.get("Metadata", {}).get("sha256") != digest:
        raise RuntimeError("Remote backup verification failed.")


def publish_remote(store, target, encoded, manifest, created):
    client, bucket, prefix = store
    for folder in ["daily", "weekly"] if created.weekday() == 6 else ["daily"]:
        key = f"{prefix}/{folder}/{target.name}"
        upload(store, key, encoded, manifest["sha256"])
        client.put_object(
            Bucket=bucket,
            Key=key + ".json",
            Body=json.dumps(manifest).encode(),
            ContentType="application/json",
        )


def prune_remote(store, created):
    client, bucket, prefix = store
    for folder, days in [("daily", 30), ("weekly", 84)]:
        path = f"{prefix}/{folder}/"
        for page in client.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=path):
            for item in page.get("Contents", []):
                name = item["Key"][len(path) :]
                dated = snapshot_time(name)
                # Only exact snapshot filenames in our prefix; never bucket-wide pruning.
                if dated and dated < created - timedelta(days=days):
                    client.delete_object(Bucket=bucket, Key=item["Key"])
                    client.delete_object(Bucket=bucket, Key=item["Key"] + ".json")


def write_private(path, data):
    # Fail on collision. A prior export must never be overwritten.
    with path.open("xb") as stream:
        path.chmod(0o600)
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def backup():
    store = remote_store()  # Fail before dumping if a required remote destination is incomplete.
    encryption = cipher()
    destination = Path(os.environ.get("JARVIS_BACKUP_DIRECTORY", "/backups"))
    destination.mkdir(mode=0o700, parents=True, exist_ok=True)
    database = os.environ.get("PGDATABASE", "jarvis")
    dump = pg("pg_dump", "--format=custom", "--no-owner", "--no-acl", "-d", database)
    encoded = encryption.encrypt(dump)
    created = datetime.now(UTC)
    stamp = created.strftime("%Y%m%dT%H%M%S%fZ")
    target = destination / f"jarvis-{stamp}.pgdump.enc"
    temporary = target.with_suffix(".tmp")
    write_private(temporary, encoded)
    temporary.replace(target)
    manifest = {
        "created_at": stamp,
        "sha256": hashlib.sha256(encoded).hexdigest(),
        "bytes": len(encoded),
        "format": 1,
    }
    write_private(target.with_suffix(".json"), (json.dumps(manifest, indent=2) + "\n").encode())
    if store:
        publish_remote(store, target, encoded, manifest, created)
    # Only report success after every required copy is present and verified.
    pg(
        "psql",
        "-XAt",
        "-d",
        database,
        "-c",
        "INSERT INTO worker_health (id,last_scan_at) VALUES ('backup',now()) "
        "ON CONFLICT (id) DO UPDATE SET last_scan_at=now()",
    )
    print(
        json.dumps({"backup": target.name, "bytes": len(encoded), "remote_verified": bool(store)}), flush=True
    )
    for old in destination.glob("jarvis-*.pgdump.enc"):
        dated = snapshot_time(old.name)
        if dated and dated < created - timedelta(days=30):
            old.unlink()
            old.with_suffix(".json").unlink(missing_ok=True)
    if store:
        prune_remote(store, created)
    return target


def download(key):
    store = remote_store()
    if not store:
        raise ValueError("Configure remote backup storage first.")
    client, bucket, prefix = store
    if not re.fullmatch(
        re.escape(prefix) + r"/(daily|weekly)/jarvis-\d{8}T\d{6}(?:\d{6})?Z\.pgdump\.enc", key
    ):
        raise ValueError("Select a backup object in this environment's prefix.")
    manifest = json.loads(client.get_object(Bucket=bucket, Key=key + ".json")["Body"].read())
    payload = client.get_object(Bucket=bucket, Key=key)["Body"].read()
    if hashlib.sha256(payload).hexdigest() != manifest["sha256"] or len(payload) != manifest["bytes"]:
        raise ValueError("Backup checksum does not match.")
    cipher().decrypt(payload)  # Authenticate before keeping the downloaded archive.
    destination = Path(os.environ.get("JARVIS_BACKUP_DIRECTORY", "/backups"))
    destination.mkdir(mode=0o700, parents=True, exist_ok=True)
    target = destination / key.rsplit("/", 1)[-1]
    write_private(target, payload)
    print(json.dumps({"downloaded": str(target), "authenticated": True}), flush=True)
    return target


def restore(path, target, *, existing_empty=False):
    if (
        not target
        or not re.fullmatch(r"[a-z][a-z0-9_]{0,62}", target)
        or target in {"postgres", "template0", "template1"}
    ):
        raise ValueError("Choose an application database target.")
    if not existing_empty and not re.fullmatch(r"jarvis_restore_[a-z0-9_]{1,40}", target):
        raise ValueError("Isolated restore targets must use jarvis_restore_*.")
    decoded = cipher().decrypt(Path(path).read_bytes())
    with tempfile.TemporaryFile() as dump:
        dump.write(decoded)
        dump.seek(0)
        subprocess.run(
            ["pg_restore", "--list"],
            stdin=dump,
            check=True,
            timeout=60,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        dump.seek(0)
        if existing_empty:
            occupied = pg(
                "psql",
                "-XAt",
                "-d",
                target,
                "-c",
                """
                SELECT EXISTS (
                    SELECT 1 FROM pg_class c JOIN pg_namespace n ON n.oid=c.relnamespace
                    WHERE n.nspname NOT IN ('pg_catalog','information_schema')
                    AND n.nspname NOT LIKE 'pg_toast%' AND c.relkind IN ('r','p','v','m','S','f')
                )
            """,
            ).strip()
            if occupied != b"f":
                raise ValueError("Restore refused: target contains records or schema objects.")
        else:
            pg("createdb", target)  # Fails if the target already exists.
        subprocess.run(
            ["pg_restore", "--exit-on-error", "--single-transaction", "--no-owner", "--no-acl", "-d", target],
            stdin=dump,
            check=True,
            timeout=1800,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
    print(
        json.dumps({"restored_database": target, "counts": counts(target), "workers_started": False}),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode", choices=["once", "loop", "download", "restore", "restore-empty"], default="loop", nargs="?"
    )
    parser.add_argument("--file")
    parser.add_argument("--object")
    parser.add_argument("--target")
    args = parser.parse_args()
    try:
        configure_postgres()
        if args.mode == "download":
            download(args.object or "")
        elif args.mode in {"restore", "restore-empty"}:
            if not args.file or not args.target:
                raise ValueError("Restore needs --file and --target.")
            restore(args.file, args.target, existing_empty=args.mode == "restore-empty")
        elif args.mode == "once":
            backup()
        else:
            for event in (signal.SIGTERM, signal.SIGINT):
                signal.signal(event, lambda *_: stop.set())
            while not stop.is_set():
                try:
                    backup()
                    stop.wait(86400)
                except Exception:  # noqa: BLE001 - never expose credentials from provider/subprocess errors
                    print(
                        "Backup failed; retrying in five minutes. Check database and storage configuration.",
                        flush=True,
                    )
                    stop.wait(300)
    except Exception:  # noqa: BLE001 - never expose credentials from provider/subprocess errors
        print(
            "Backup command failed. Check configuration, credentials and the restore target; "
            "no workers were started.",
            file=__import__("sys").stderr,
        )
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
