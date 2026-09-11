"""Encrypted PostgreSQL backup and isolated restore. Never restore over an existing database."""

import argparse
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

from cryptography.fernet import Fernet

stop = threading.Event()


def pg(*args, input=None):
    return subprocess.run(args, input=input, capture_output=True, check=True).stdout


def cipher():
    return Fernet(os.environ["JARVIS_BACKUP_KEY"].encode())


def counts(database):
    names = [
        "tasks",
        "schedules",
        "notifications",
        "sources",
        "memory_assertions",
        "memory_reviews",
        "commands",
    ]
    return {
        name: int(pg("psql", "-XAt", "-d", database, "-c", f"SELECT count(*) FROM {name}").strip())
        for name in names
    }


def backup():
    destination = Path(os.environ.get("JARVIS_BACKUP_DIRECTORY", "/backups"))
    destination.mkdir(parents=True, exist_ok=True)
    database = os.environ.get("PGDATABASE", "jarvis")
    dump = pg("pg_dump", "--format=custom", "--no-owner", "--no-acl", "-d", database)
    # Authenticate the whole snapshot. Keys and provider secrets are never part of the dump.
    encoded = cipher().encrypt(dump)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    target = destination / f"jarvis-{stamp}.pgdump.enc"
    temporary = target.with_suffix(".tmp")
    temporary.write_bytes(encoded)
    temporary.chmod(0o600)
    with temporary.open("rb") as stream:
        os.fsync(stream.fileno())
    temporary.replace(target)
    manifest = {
        "created_at": stamp,
        "sha256": hashlib.sha256(encoded).hexdigest(),
        "bytes": len(encoded),
        "format": 1,
    }
    target.with_suffix(".json").write_text(json.dumps(manifest, indent=2) + "\n")
    pg(
        "psql",
        "-XAt",
        "-d",
        database,
        "-c",
        "INSERT INTO worker_health (id,last_scan_at) VALUES ('backup',now()) ON CONFLICT (id) DO UPDATE SET last_scan_at=now()",
    )
    print(json.dumps({"backup": target.name, "bytes": len(encoded)}), flush=True)
    # Prune only our exact timestamped filenames, after a successful new snapshot.
    cutoff = datetime.now(UTC) - timedelta(days=30)
    for old in destination.glob("jarvis-*.pgdump.enc"):
        if (
            re.fullmatch(r"jarvis-\d{8}T\d{6}Z\.pgdump\.enc", old.name)
            and datetime.fromtimestamp(old.stat().st_mtime, UTC) < cutoff
        ):
            old.unlink()
            old.with_suffix(".json").unlink(missing_ok=True)
    return target


def restore(path, target):
    if not re.fullmatch(r"jarvis_restore_[a-z0-9_]{1,40}", target):
        raise ValueError("Restore target must be a new jarvis_restore_* database.")
    # createdb fails if the target exists. No overwrite, no automatic deletion, no worker startup.
    decoded = cipher().decrypt(Path(path).read_bytes())
    with tempfile.TemporaryFile() as dump:
        dump.write(decoded)
        dump.seek(0)
        subprocess.run(["pg_restore", "--list"], stdin=dump, check=True, stdout=subprocess.DEVNULL)
        dump.seek(0)
        pg("createdb", target)
        subprocess.run(
            ["pg_restore", "--exit-on-error", "--no-owner", "--no-acl", "-d", target], stdin=dump, check=True
        )
    print(
        json.dumps({"restored_database": target, "counts": counts(target), "workers_started": False}),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["once", "loop", "restore"], default="loop", nargs="?")
    parser.add_argument("--file")
    parser.add_argument("--target")
    args = parser.parse_args()
    if args.mode == "restore":
        restore(args.file, args.target)
        return
    if args.mode == "once":
        backup()
        return
    for event in (signal.SIGTERM, signal.SIGINT):
        signal.signal(event, lambda *_: stop.set())
    while not stop.is_set():
        try:
            backup()
            stop.wait(86400)
        except (OSError, subprocess.CalledProcessError):
            print("Backup failed; will retry in five minutes. Check storage and database access.", flush=True)
            stop.wait(300)


if __name__ == "__main__":
    main()
