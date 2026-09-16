"""Explicit cloud entry points and secret-safe deployment checks."""

import argparse
import json
import os
import signal
import sys
import threading
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
from urllib.parse import urlsplit

from alembic import command
from alembic.config import Config
from alembic.script import ScriptDirectory
from cryptography.fernet import Fernet
from sqlalchemy import text
from sqlalchemy.engine import make_url
from sqlalchemy.exc import ArgumentError

from .config import get_settings
from .db import engine, session_scope

ROOT = Path(__file__).resolve().parents[3]


class DeploymentConfigurationError(RuntimeError):
    """Only include operator-safe configuration guidance in this error."""


def validate_deployment(settings):
    cloud = settings.deployment_environment != "local" or bool(os.getenv("RAILWAY_ENVIRONMENT_ID"))
    if not cloud:
        return
    failures = []
    if settings.deployment_environment == "local":
        failures.append("Set JARVIS_DEPLOYMENT_ENVIRONMENT explicitly for Railway.")
    origin = urlsplit(settings.origin)
    if (
        origin.scheme != "https"
        or not origin.hostname
        or origin.hostname in {"localhost", "127.0.0.1", "::1"}
        or origin.username
        or origin.password
        or origin.path not in {"", "/"}
        or origin.query
        or origin.fragment
    ):
        failures.append("JARVIS_ORIGIN must be the public HTTPS origin without a path.")
    if settings.pairing_enabled:
        failures.append("Disable JARVIS_PAIRING_ENABLED for public deployments.")
    if not settings.google_client_id or not settings.google_client_secret:
        failures.append("Google OAuth credentials are required for cloud sign-in.")
    try:
        Fernet(settings.integration_encryption_key.encode())
    except (ValueError, TypeError):
        failures.append("Preserve a valid JARVIS_INTEGRATION_ENCRYPTION_KEY from the current installation.")
    try:
        db = make_url(settings.database_url)
        if db.drivername != "postgresql+psycopg" or not db.host or not db.database or not db.password:
            raise ValueError()
        if db.host in {"localhost", "127.0.0.1", "::1"} and os.getenv("RAILWAY_ENVIRONMENT_ID"):
            raise ValueError()
    except (ArgumentError, ValueError, TypeError):
        failures.append("Set a PostgreSQL connection URL for the intended database.")
    if settings.deployment_environment == "staging" and (
        settings.worker_enabled or settings.external_services_enabled
    ):
        failures.append("Staging must keep JARVIS_WORKER_ENABLED and JARVIS_EXTERNAL_SERVICES_ENABLED false.")
    if os.getenv("WEB_CONCURRENCY", "1") != "1":
        failures.append("Live sessions currently require WEB_CONCURRENCY=1.")
    if failures:
        # Never include settings, connection strings, credentials, or exception reprs.
        raise DeploymentConfigurationError("Deployment checks failed: " + " ".join(failures))


@lru_cache
def expected_heads():
    return set(ScriptDirectory.from_config(alembic_config()).get_heads())


def alembic_config():
    config = Config(str(ROOT / "alembic.ini"))
    config.set_main_option("script_location", str(ROOT / "migrations"))
    return config


def check_schema(db):
    actual = set(db.scalars(text("SELECT version_num FROM alembic_version")))
    if actual != expected_heads():
        raise DeploymentConfigurationError("Database migrations do not match this release.")


def preflight(database=False):
    settings = get_settings()
    validate_deployment(settings)
    result = {
        "environment": settings.deployment_environment,
        "configuration": "passed",
        "pairing_enabled": settings.pairing_enabled,
        "worker_enabled": settings.worker_enabled,
        "external_services_enabled": settings.external_services_enabled,
    }
    if database:
        with session_scope() as db:
            check_schema(db)
            result["schema"] = "current"
            result["dbos_schema_present"] = bool(
                db.scalar(text("SELECT to_regclass('dbos.workflow_status') IS NOT NULL"))
            )
            linked = bool(
                db.scalar(
                    text("SELECT EXISTS (SELECT 1 FROM google_identities WHERE owner_id=:owner)"),
                    {"owner": settings.owner_id},
                )
            )
            result["owner_google_linked"] = linked
            result["pgvector_available"] = bool(
                db.scalar(text("SELECT EXISTS (SELECT 1 FROM pg_available_extensions WHERE name='vector')"))
            )
            if settings.deployment_environment != "local" and not linked:
                raise DeploymentConfigurationError(
                    "Restore and verify the owner's existing Google identity before cutover."
                )
            # Check encryption compatibility without outputting token material.
            from sqlalchemy import select

            from .google_auth import unseal
            from .models import GoogleIdentity, LinearConnection

            for account in db.scalars(select(GoogleIdentity)):
                if account.credentials:
                    unseal(account.credentials)
            for account in db.scalars(select(LinearConnection)):
                if account.credentials:
                    unseal(account.credentials)
            result["integration_decryption"] = "passed"
    return result


@contextmanager
def worker_lease(stop):
    """Prevent overlapping deployments from running two supervisors on one database."""
    with engine().connect() as connection:
        acquired = False
        try:
            while not stop.is_set():
                acquired = bool(
                    connection.scalar(
                        text("SELECT pg_try_advisory_lock(hashtext('eridani:worker-supervisor'))")
                    )
                )
                connection.commit()
                if acquired:
                    break
                stop.wait(1)
            yield connection if acquired else None
        finally:
            if acquired and not connection.invalidated:
                connection.execute(text("SELECT pg_advisory_unlock(hashtext('eridani:worker-supervisor'))"))
                connection.commit()


def stop_event():
    event = threading.Event()
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: event.set())
    return event


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["api", "worker", "migrate", "preflight"])
    parser.add_argument("--database", action="store_true", help="Run read-only database checks.")
    args = parser.parse_args()
    try:
        if args.mode == "migrate":
            validate_deployment(get_settings())
            command.upgrade(alembic_config(), "head")
        elif args.mode == "preflight":
            print(json.dumps(preflight(args.database)))
        elif args.mode == "api":
            validate_deployment(get_settings())
            port = int(os.getenv("PORT", "8765"))
            if not 1 <= port <= 65535:
                raise RuntimeError("PORT must be between 1 and 65535.")
            os.execv(
                sys.executable,
                [
                    sys.executable,
                    "-m",
                    "uvicorn",
                    "jarvis.api:app",
                    "--host",
                    "0.0.0.0",
                    "--port",
                    str(port),
                    "--workers",
                    "1",
                    "--no-access-log",
                    "--proxy-headers",
                    "--forwarded-allow-ips",
                    os.getenv("FORWARDED_ALLOW_IPS", "127.0.0.1"),
                ],
            )
        else:
            from .worker import main as run_worker

            run_worker()
    except DeploymentConfigurationError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1) from None
    except Exception:  # noqa: BLE001 - never expose credentials from provider/subprocess errors
        # Detailed failure types are useful; payloads may contain secrets.
        print(
            "Deployment command failed. Check configuration, database access and the migration runbook.",
            file=sys.stderr,
        )
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
