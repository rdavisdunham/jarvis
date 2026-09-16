import importlib.util
import io
from datetime import UTC, datetime
from pathlib import Path

import pytest
from cryptography.fernet import Fernet

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("cloud_backup", ROOT / "scripts/backup.py")
backup = importlib.util.module_from_spec(spec)
spec.loader.exec_module(backup)


class MemoryS3:
    def __init__(self):
        self.objects = {}
        self.deleted = []
        self.fail = False

    def put_object(self, **kw):
        if self.fail:
            raise RuntimeError("Simulated upload failure")
        self.objects[kw["Key"]] = {"Body": kw["Body"], "Metadata": kw.get("Metadata", {})}

    def head_object(self, **kw):
        obj = self.objects[kw["Key"]]
        return {"ContentLength": len(obj["Body"]), "Metadata": obj["Metadata"]}

    def get_object(self, **kw):
        return {"Body": io.BytesIO(self.objects[kw["Key"]]["Body"])}

    def get_paginator(self, *args):
        return self

    def paginate(self, **kw):
        yield {"Contents": [{"Key": k} for k in list(self.objects) if k.startswith(kw["Prefix"])]}

    def delete_object(self, **kw):
        self.deleted.append(kw["Key"])
        self.objects.pop(kw["Key"], None)


@pytest.fixture
def exports(monkeypatch, tmp_path):
    key = Fernet.generate_key().decode()
    monkeypatch.setenv("JARVIS_BACKUP_KEY", key)
    monkeypatch.setenv("JARVIS_BACKUP_DIRECTORY", str(tmp_path))
    monkeypatch.setenv("PGDATABASE", "isolated")
    calls = []

    def pg(*args, **kw):
        calls.append(args)
        return b"test database snapshot" if args[0] == "pg_dump" else b""

    monkeypatch.setattr(backup, "pg", pg)
    store = MemoryS3(), "test-bucket", "eridani/staging"
    monkeypatch.setattr(backup, "remote_store", lambda: store)
    return store, calls, tmp_path, key


def test_remote_export_is_encrypted_verified_then_marks_health(exports):
    store, calls, _folder, key = exports
    path = backup.backup()
    data = path.read_bytes()
    assert b"test database snapshot" not in data
    assert Fernet(key.encode()).decrypt(data) == b"test database snapshot"
    assert any("worker_health" in str(call) for call in calls)
    assert any(k.startswith("eridani/staging/daily/") for k in store[0].objects)
    assert path.stat().st_mode & 0o077 == 0


def test_failed_upload_does_not_mark_backup_healthy_or_prune(exports):
    store, calls, folder, _key = exports
    store[0].fail = True
    old = folder / "jarvis-20000101T000000Z.pgdump.enc"
    old.write_bytes(b"old")
    with pytest.raises(RuntimeError):
        backup.backup()
    assert not any("worker_health" in str(call) for call in calls)
    assert old.exists()


def test_download_authenticates_and_cannot_overwrite(exports, monkeypatch, tmp_path):
    store, _calls, _folder, key = exports
    backup.backup()
    object_key = next(k for k in store[0].objects if k.endswith(".enc"))
    monkeypatch.setenv("JARVIS_BACKUP_DIRECTORY", str(tmp_path / "download"))
    result = backup.download(object_key)
    assert Fernet(key.encode()).decrypt(result.read_bytes()) == b"test database snapshot"
    with pytest.raises(FileExistsError):
        backup.download(object_key)
    with pytest.raises(ValueError):
        backup.download("another-project/daily/" + result.name)


def test_retention_only_deletes_exact_names_in_configured_prefix(exports):
    store, *_ = exports
    client = store[0]
    old = "eridani/staging/daily/jarvis-20000101T000000Z.pgdump.enc"
    others = [
        "eridani/production/daily/jarvis-20000101T000000Z.pgdump.enc",
        "eridani/staging/daily/important.pgdump.enc",
        "eridani/staging/daily/subdir/jarvis-20000101T000000Z.pgdump.enc",
    ]
    for name in [old, *others]:
        client.objects[name] = {}
    backup.prune_remote(store, datetime.now(UTC))
    assert client.deleted == [old, old + ".json"]
    assert all(name in client.objects for name in others)


def test_restore_empty_refuses_occupied_database_before_restoring(exports, monkeypatch):
    _store, _calls, folder, key = exports
    path = folder / "restore.enc"
    path.write_bytes(Fernet(key.encode()).encrypt(b"dump"))
    processes = []
    monkeypatch.setattr(backup.subprocess, "run", lambda args, **kw: processes.append(args))
    monkeypatch.setattr(backup, "pg", lambda *args, **kw: b"t")
    with pytest.raises(ValueError, match="contains"):
        backup.restore(path, "railway", existing_empty=True)
    assert processes == [["pg_restore", "--list"]]


def test_restore_uses_one_transaction(exports, monkeypatch):
    _, _, folder, key = exports
    path = folder / "restore.enc"
    path.write_bytes(Fernet(key.encode()).encrypt(b"dump"))
    processes = []
    monkeypatch.setattr(backup.subprocess, "run", lambda args, **kw: processes.append(args))
    monkeypatch.setattr(backup, "pg", lambda *args, **kw: b"f")
    monkeypatch.setattr(backup, "counts", lambda target: {"dbos.workflow_status": 2})
    backup.restore(path, "railway", existing_empty=True)
    assert "--single-transaction" in processes[-1]
    assert "--clean" not in processes[-1]


def test_connection_secrets_stay_in_environment(monkeypatch):
    # configure_postgres intentionally updates libpq env; restore each changed variable after this test.
    for name in ("PGHOST", "PGPORT", "PGDATABASE", "PGUSER", "PGPASSWORD", "PGSSLMODE", "PGCONNECT_TIMEOUT"):
        monkeypatch.setenv(name, backup.os.environ.get(name, ""))
    monkeypatch.setenv(
        "JARVIS_DATABASE_URL", "postgresql+psycopg://user:p%40ss@db:5432/railway?sslmode=require"
    )
    backup.configure_postgres()
    assert backup.os.environ["PGPASSWORD"] == "p@ss"
    assert backup.os.environ["PGDATABASE"] == "railway"
    assert backup.os.environ["PGSSLMODE"] == "require"


def test_snapshot_timestamp_accepts_old_and_new_formats():
    assert backup.snapshot_time("jarvis-20260914T120000Z.pgdump.enc").year == 2026
    assert backup.snapshot_time("jarvis-20260914T120000123456Z.pgdump.enc").microsecond == 123456
    assert backup.snapshot_time("unrelated.enc") is None


def test_cloud_handoff_preserves_keys_and_never_overwrites(tmp_path, capsys):
    spec = importlib.util.spec_from_file_location("prepare_cloud", ROOT / "scripts/prepare_cloud.py")
    prepare = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(prepare)
    (tmp_path / ".env.cloud.example").write_text(
        "JARVIS_BACKUP_KEY=\nJARVIS_PAIRING_ENABLED=false\nJARVIS_ORIGIN=\n"
    )
    (tmp_path / ".env.upgrade").write_text("JARVIS_BACKUP_KEY='keep-this-key'\nJARVIS_OWNER_TOKEN=5283\n")
    target = prepare.prepare(tmp_path)
    assert "keep-this-key" in target.read_text() and "5283" not in target.read_text()
    target.write_text("do not overwrite")
    prepare.prepare(tmp_path)
    assert target.read_text() == "do not overwrite"
    assert "keep-this-key" not in capsys.readouterr().out
