"""Explicit local database boundaries; never infer an eval target from production env."""

import base64
import hashlib
import json
import os
import re
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path
from uuid import uuid4

from sqlalchemy import create_engine, text
from sqlalchemy.engine import make_url

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_URL = "postgresql+psycopg://eridani_eval:synthetic-eval-only@127.0.0.1:54340/eridani_eval_corpus"
CORPUS = "eridani_eval_corpus"
PREFIX = "eridani_eval_trial_"


def validate_url(url, *, trial=False):
    parsed = make_url(url)
    valid_name = re.fullmatch(PREFIX + r"[a-f0-9]{32}", parsed.database or "")
    if parsed.drivername != "postgresql+psycopg" or parsed.host not in {"127.0.0.1", "::1", "localhost"}:
        raise ValueError("Eval databases must use local PostgreSQL; remote targets are refused.")
    if parsed.username != "eridani_eval" or parsed.query:
        raise ValueError("Use the dedicated eridani_eval database role without connection overrides.")
    if (trial and not valid_name) or (not trial and parsed.database != CORPUS and not valid_name):
        raise ValueError("Refusing a database outside the dedicated eval namespace.")
    return parsed


def settings_env(url, *, live=False):
    validate_url(url)
    values = {
        "JARVIS_ENV_FILE": "",
        "JARVIS_DATABASE_URL": url,
        "JARVIS_DEPLOYMENT_ENVIRONMENT": "local",
        "JARVIS_PAIRING_ENABLED": "true",
        "JARVIS_OWNER_ID": "eval-rowan",
        "JARVIS_OWNER_NAME": "Rowan",
        "JARVIS_OWNER_TOKEN": "synthetic-eval-rowan",
        "JARVIS_ORIGIN": "http://127.0.0.1:8766",
        "JARVIS_WORKER_ENABLED": "false",
        "JARVIS_COST_TRACKING_ENABLED": "false",
        "JARVIS_EXTERNAL_SERVICES_ENABLED": "true" if live else "false",
        "JARVIS_SEMANTIC_SEARCH_ENABLED": "false",
        "JARVIS_GOOGLE_CLIENT_ID": "",
        "JARVIS_GOOGLE_CLIENT_SECRET": "",
        "JARVIS_VAPID_PRIVATE_KEY": "",
        "JARVIS_HOME_ASSISTANT_TOKEN": "",
        "JARVIS_OPENAI_API_KEY": "",
        "JARVIS_GEMINI_API_KEY": "",
        "JARVIS_GROQ_API_KEY": "",
        "OPENAI_API_KEY": "",
        "GEMINI_API_KEY": "",
        "GROQ_API_KEY": "",
    }
    # Public synthetic-only key: corpus receipts must remain decryptable in every clone.
    # This key is never loaded by the application outside this opt-in eval environment.
    values["JARVIS_INTEGRATION_ENCRYPTION_KEY"] = base64.urlsafe_b64encode(
        hashlib.sha256(b"eridani synthetic evaluation corpus v1 -- never production").digest()
    ).decode()
    if live:
        from dotenv import dotenv_values

        secrets = {**dotenv_values(ROOT / ".env"), **os.environ}
        for provider in ("OPENAI", "GEMINI"):
            values[f"JARVIS_{provider}_API_KEY"] = (
                secrets.get(f"JARVIS_{provider}_API_KEY") or secrets.get(f"{provider}_API_KEY") or ""
            )
    return values


@contextmanager
def environment(url, *, live=False):
    overrides = settings_env(url, live=live)
    controlled = set(overrides) | {k for k in os.environ if k.startswith("JARVIS_")}
    previous = {k: os.environ.get(k) for k in controlled}
    for key in controlled:
        os.environ.pop(key, None)
    os.environ.update(overrides)
    from jarvis.config import get_settings
    from jarvis.db import engine

    if engine.cache_info().currsize:
        engine().dispose()
    engine.cache_clear()
    get_settings.cache_clear()
    try:
        yield
    finally:
        if engine.cache_info().currsize:
            engine().dispose()
        engine.cache_clear()
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        get_settings.cache_clear()


def corpus_hash():
    paths = [
        ROOT / "evals/app/personas/rowan-v1.json",
        ROOT / "evals/app/corpus/rowan-v1.json",
        Path(__file__).with_name("seed.py"),
    ]
    return hashlib.sha256(b"".join(p.read_bytes() for p in paths)).hexdigest()


def admin_for(url):
    parsed = validate_url(url)
    return create_engine(parsed.set(database="postgres"), isolation_level="AUTOCOMMIT")


def ensure_corpus(url):
    parsed = validate_url(url)
    if parsed.database != CORPUS:
        raise ValueError("Seeding requires the dedicated corpus database.")
    admin = admin_for(url)
    try:
        with admin.connect() as db:
            db.exec_driver_sql("ALTER ROLE eridani_eval SET timezone TO 'UTC'")
            exists = db.scalar(text("SELECT 1 FROM pg_database WHERE datname=:name"), {"name": CORPUS})
            if not exists:
                db.exec_driver_sql(f'CREATE DATABASE "{CORPUS}"')
        target = create_engine(url)
        try:
            with target.connect() as db:
                marker = db.scalar(text("SELECT to_regclass('eval_harness.metadata')"))
                if marker:
                    saved = db.scalar(text("SELECT value FROM eval_harness.metadata WHERE key='corpus'"))
                    if not saved:
                        raise ValueError(
                            "An earlier corpus build did not finish. Preserve/export it before explicitly rebuilding this synthetic baseline."
                        )
                    if saved["sha256"] != corpus_hash():
                        raise ValueError(
                            "Corpus version differs. Preserve this database; export it before creating a new baseline."
                        )
                    return saved
                if db.scalar(
                    text("SELECT count(*) FROM information_schema.tables WHERE table_schema='public'")
                ):
                    raise ValueError("Refusing to seed an occupied, unmarked database.")
            with target.begin() as db:
                db.exec_driver_sql("CREATE SCHEMA eval_harness")
                db.exec_driver_sql(
                    "CREATE TABLE eval_harness.metadata (key text PRIMARY KEY, value jsonb NOT NULL)"
                )
                db.execute(
                    text("INSERT INTO eval_harness.metadata VALUES ('building', CAST(:value AS jsonb))"),
                    {"value": json.dumps({"synthetic": True, "sha256": corpus_hash()})},
                )
            env = {
                **{k: v for k, v in os.environ.items() if not k.startswith("JARVIS_")},
                **settings_env(url),
            }
            subprocess.run(
                [sys.executable, "-m", "alembic", "upgrade", "head"], cwd=ROOT, env=env, check=True
            )
            with environment(url):
                from .seed import seed

                result = seed()
            result.update(sha256=corpus_hash(), version=1, synthetic=True)
            with target.begin() as db:
                db.execute(
                    text("INSERT INTO eval_harness.metadata VALUES ('corpus', CAST(:value AS jsonb))"),
                    {"value": json.dumps(result)},
                )
            return result
        finally:
            target.dispose()
    finally:
        admin.dispose()


@contextmanager
def trial_database(url):
    parsed = validate_url(url)
    if parsed.database != CORPUS:
        raise ValueError("Trials must clone the marked corpus, not another arbitrary database.")
    target = create_engine(url)
    try:
        with target.connect() as db:
            saved = db.scalar(text("SELECT value FROM eval_harness.metadata WHERE key='corpus'"))
            if saved["sha256"] != corpus_hash() or not saved.get("synthetic"):
                raise ValueError("Corpus marker mismatch.")
    finally:
        target.dispose()
    name = PREFIX + uuid4().hex
    trial_url = parsed.set(database=name).render_as_string(hide_password=False)
    validate_url(trial_url, trial=True)
    admin = admin_for(url)
    created = False
    try:
        with admin.connect() as db:
            db.exec_driver_sql(f'CREATE DATABASE "{name}" TEMPLATE "{CORPUS}"')
        created = True
        yield trial_url, saved
    finally:
        if created:
            validate_url(trial_url, trial=True)
            with admin.connect() as db:
                db.exec_driver_sql(f'DROP DATABASE "{name}" WITH (FORCE)')
        admin.dispose()
