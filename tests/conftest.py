import os
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from jarvis.config import get_settings
from jarvis.db import engine
from jarvis.models import Base
from sqlalchemy import create_engine
from sqlalchemy.engine import make_url


@pytest.fixture(scope="session", autouse=True)
def test_database():
    supplied = os.environ.get("ERIDANI_EVAL_OWNED_DB")
    if supplied:
        from scripts.app_eval.environment import validate_url
        from sqlalchemy import text
        validate_url(supplied, trial=True)
        if get_settings().database_url != supplied:
            raise ValueError("Eval trial environment mismatch")
        with engine().connect() as connection:
            marker = connection.scalar(text("SELECT value FROM eval_harness.metadata WHERE key='corpus'"))
            if not marker.get("synthetic"):
                raise ValueError("Missing synthetic eval marker")
        os.environ.update(JARVIS_COST_TRACKING_ENABLED="true", JARVIS_OWNER_TOKEN="test-owner-token", JARVIS_ORIGIN="http://testserver")
        get_settings.cache_clear()
        Base.metadata.create_all(engine())
        # Match the original pytest fixture: model tables, not a migrated schema.
        # Migration/readiness tests deliberately expect no alembic_version.
        with engine().begin() as connection:
            connection.exec_driver_sql("DROP TABLE IF EXISTS alembic_version")
        yield supplied
        engine().dispose()
        engine.cache_clear()
        return
    original = get_settings().database_url
    name = "jarvis_test_" + uuid4().hex
    admin = create_engine(original, isolation_level="AUTOCOMMIT")
    with admin.connect() as connection:
        connection.exec_driver_sql(f'CREATE DATABASE "{name}"')
    testing = make_url(original).set(database=name).render_as_string(hide_password=False)
    os.environ["JARVIS_DATABASE_URL"] = testing
    os.environ["JARVIS_COST_TRACKING_ENABLED"] = "true"
    os.environ["JARVIS_OWNER_TOKEN"] = "test-owner-token"
    os.environ["JARVIS_ORIGIN"] = "http://testserver"
    get_settings.cache_clear()
    engine.cache_clear()
    Base.metadata.create_all(engine())
    yield testing
    engine().dispose()
    engine.cache_clear()
    with admin.connect() as connection:
        connection.exec_driver_sql(f'DROP DATABASE "{name}" WITH (FORCE)')
    admin.dispose()
    os.environ.pop("JARVIS_DATABASE_URL", None)
    get_settings.cache_clear()


@pytest.fixture(autouse=True)
def clean_database(test_database):
    with engine().begin() as connection:
        for table in reversed(Base.metadata.sorted_tables):
            connection.execute(table.delete())


@pytest.fixture
def client():
    from jarvis.api import app

    with TestClient(app) as client:
        r = client.post("/api/v1/auth/login", json={"token": "test-owner-token"})
        assert r.status_code == 200
        client.headers["X-CSRF-Token"] = r.json()["csrf"]
        yield client
