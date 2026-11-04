"""Migrations 0048-0050 (context tables) on a real database: upgrade over existing data, rows in every stage,
downgrade one stage and all the way back, and upgrade again cleanly. Never touches the supplied database."""

import os
import subprocess
import sys
from pathlib import Path
from uuid import uuid4

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine import make_url

ROOT = Path(__file__).resolve().parents[2]
SOURCES = {"context_connections", "context_sources", "context_source_revisions", "context_segments", "context_chunks",
           "context_runs", "context_cursors"}
ASSERTIONS = {"context_decisions", "context_assertions", "context_assertion_versions", "context_evidence",
              "context_entity_links", "context_actor_identities", "context_actor_merges"}
REVIEW = {"context_proposals", "context_publications", "context_briefs", "context_owner_state"}
VECTOR = "[" + ",".join(["0.5"] * 2 + ["0"] * 510) + "]"
ROWS = [
    "INSERT INTO context_owner_state (owner_id, created_at, updated_at) VALUES ('personal', now(), now())",
    ("INSERT INTO context_sources (id, owner_id, connector, external_id, created_at, updated_at)"
     " VALUES ('src', 'personal', 'api', 'zoom-1', now(), now())"),
    ("INSERT INTO context_source_revisions (id, owner_id, source_id, revision, content_hash, received_at, created_at)"
     " VALUES ('rev', 'personal', 'src', 1, 'h', now(), now())"),
    ("INSERT INTO context_segments (id, owner_id, revision_id, seq, text, char_start, char_end, created_at)"
     " VALUES ('seg', 'personal', 'rev', 0, 'Alex will send the contract on Friday.', 0, 38, now())"),
    ("INSERT INTO context_chunks (id, owner_id, revision_id, seq, segment_start, segment_end, text, embedding,"
     f" created_at, updated_at) VALUES ('chk', 'personal', 'rev', 0, 0, 0, 'Alex will send the contract.', '{VECTOR}',"
     " now(), now())"),
    ("INSERT INTO context_runs (id, owner_id, stage, source_id, created_at, updated_at)"
     " VALUES ('run', 'personal', 'extract', 'src', now(), now())"),
    ("INSERT INTO context_assertions (id, owner_id, kind, statement, subject_type, subject_id, predicate, run_id,"
     " embedding, created_at, updated_at) VALUES ('asr', 'personal', 'commitment', 'Alex sends the contract',"
     f" 'actor', 'alex', 'sends', 'run', '{VECTOR}', now(), now())"),
    ("INSERT INTO context_evidence (id, owner_id, assertion_id, assertion_version, source_id, revision_id, segment_id,"
     " quote, created_at) VALUES ('evd', 'personal', 'asr', 1, 'src', 'rev', 'seg', 'send the contract', now())"),
    ("INSERT INTO context_actor_identities (id, owner_id, actor_id, namespace, value, normalized, created_at,"
     " updated_at) VALUES ('idn', 'personal', 'alex', 'email', 'alex@example.test', 'alex@example.test', now(), now())"),
    ("INSERT INTO context_proposals (id, owner_id, source_id, kind, fingerprint, title, created_at, updated_at)"
     " VALUES ('prp', 'personal', 'src', 'create', 'fp', 'Send the contract', now(), now())"),
]


def test_context_migrations_upgrade_with_rows_downgrade_and_upgrade_again(test_database):
    name = "jarvis_context_" + uuid4().hex
    supplied = make_url(test_database)
    admin = create_engine(supplied.set(database="postgres"), isolation_level="AUTOCOMMIT")
    target = supplied.set(database=name)
    trial = None
    with admin.connect() as connection:
        connection.exec_driver_sql(f'CREATE DATABASE "{name}"')
    try:
        env = {**os.environ, "JARVIS_DATABASE_URL": target.render_as_string(hide_password=False), "JARVIS_ENV_FILE": ""}

        def migrate(*args):
            result = subprocess.run([sys.executable, "-m", "alembic", *args], cwd=ROOT, env=env, capture_output=True,
                                    text=True, check=False)
            assert result.returncode == 0, result.stdout + result.stderr

        def tables():
            return {t for t in inspect(trial).get_table_names() if t.startswith("context_")}

        migrate("upgrade", "0047_actor_name_uniqueness")
        trial = create_engine(target)
        with trial.begin() as db:  # Existing planner data the context migrations must leave alone.
            db.execute(text("INSERT INTO actors (id, owner_id, name, kind, archived, revision, created_at)"
                            " VALUES ('alex', 'personal', 'Alex', 'person', false, 1, now())"))
        assert not tables()
        migrate("upgrade", "head")
        migrate("check")
        assert tables() == SOURCES | ASSERTIONS | REVIEW
        with trial.begin() as db:
            for statement in ROWS:
                db.execute(text(statement))
            assert db.scalar(text("SELECT extversion FROM pg_extension WHERE extname = 'vector'"))
            assert db.scalar(text("SELECT tsv @@ plainto_tsquery('english', 'contracts') FROM context_segments"))
            assert db.scalar(text(f"SELECT embedding <=> '{VECTOR}' FROM context_chunks")) == 0
            assert db.scalar(text(f"SELECT embedding <=> '{VECTOR}' FROM context_assertions")) == 0
            row = db.execute(text("SELECT status, availability, current_revision FROM context_sources")).one()
            assert tuple(row) == ("received", "available", 0)
        migrate("downgrade", "0048_context_sources")
        assert tables() == SOURCES
        with trial.connect() as db:
            assert db.scalar(text("SELECT count(*) FROM context_segments")) == 1
        migrate("downgrade", "0047_actor_name_uniqueness")
        assert not tables()
        with trial.connect() as db:
            assert db.scalar(text("SELECT name FROM actors WHERE id = 'alex'")) == "Alex"
        migrate("upgrade", "head")
        migrate("check")
        assert tables() == SOURCES | ASSERTIONS | REVIEW
    finally:
        if trial:
            trial.dispose()
        with admin.connect() as connection:
            connection.exec_driver_sql(f'DROP DATABASE "{name}" WITH (FORCE)')
        admin.dispose()
