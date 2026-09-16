import threading

import pytest
from cryptography.fernet import Fernet
from jarvis import deploy
from jarvis.config import Settings, get_settings, require_external_services
from jarvis.db import engine
from jarvis.domain import DomainError
from sqlalchemy import text


def cloud_settings(**values):
    return Settings(
        _env_file=None,
        **{
            "deployment_environment": "staging",
            "origin": "https://eri.example.com",
            "database_url": "postgresql://user:secret@postgres.railway.internal/railway",
            "pairing_enabled": False,
            "worker_enabled": False,
            "external_services_enabled": False,
            "google_client_id": "test-client",
            "google_client_secret": "test-secret",
            "integration_encryption_key": Fernet.generate_key().decode(),
            **values,
        },
    )


def test_native_postgres_url_and_pool_configuration():
    s = cloud_settings()
    assert s.database_url == "postgresql+psycopg://user:secret@postgres.railway.internal/railway"
    assert Settings(_env_file=None, database_url="postgres://u:p@db/d?sslmode=require").database_url.endswith(
        "u:p@db/d?sslmode=require"
    )
    deploy.validate_deployment(s)


@pytest.mark.parametrize(
    "change",
    [
        {"pairing_enabled": True},
        {"origin": "http://eri.example.com"},
        {"origin": "https://eri.example.com/path"},
        {"google_client_secret": ""},
        {"worker_enabled": True},
        {"external_services_enabled": True},
        {"integration_encryption_key": "bad"},
        {"database_url": "sqlite:///tmp.db"},
        {"database_url": ""},
    ],
)
def test_cloud_startup_rejects_unsafe_or_incomplete_configuration(change):
    with pytest.raises(RuntimeError, match="Deployment checks failed") as error:
        deploy.validate_deployment(cloud_settings(**change))
    assert "test-secret" not in str(error.value)
    assert "user:secret@" not in str(error.value)


def test_staging_clears_model_and_push_credentials(monkeypatch):
    for name in ("OPENAI_API_KEY", "GEMINI_API_KEY", "JARVIS_VAPID_PRIVATE_KEY"):
        monkeypatch.setenv(name, "must-not-be-used")
    monkeypatch.setenv("JARVIS_OPENAI_API_KEY", "also-blocked")
    monkeypatch.setenv("JARVIS_EXTERNAL_SERVICES_ENABLED", "false")
    get_settings.cache_clear()
    try:
        s = get_settings()
        assert not s.openai_api_key and not s.gemini_api_key and not s.vapid_private_key
        with pytest.raises(DomainError, match="paused"):
            require_external_services()
    finally:
        get_settings.cache_clear()


def test_pairing_flag_hides_login_and_revokes_old_pairing_sessions(client, monkeypatch):
    assert client.get("/api/v1/auth/options").json()["pairing"] is True
    monkeypatch.setattr(get_settings(), "pairing_enabled", False)
    assert client.get("/api/v1/auth/options").json()["pairing"] is False
    assert client.post("/api/v1/auth/login", json={"token": "test-owner-token"}).status_code == 401
    assert client.get("/api/v1/bootstrap").status_code == 401


def test_staging_blocks_provider_clients_before_network(monkeypatch):
    from jarvis.google_auth import begin, exchange
    from jarvis.google_calendar import CalendarClient
    from jarvis.linear_client import LinearClient

    monkeypatch.setattr(get_settings(), "external_services_enabled", False)
    for call in [
        lambda: CalendarClient({"refresh_token": "never-send"}),
        lambda: LinearClient("never-send"),
        lambda: begin("calendar"),
        lambda: exchange("calendar_write", "state", "verifier", "code"),
    ]:
        with pytest.raises(DomainError) as error:
            call()
        assert error.value.code == "EXTERNAL_SERVICES_DISABLED"


def test_maintenance_blocks_api_but_keeps_health_available(client, monkeypatch):
    monkeypatch.setattr(get_settings(), "maintenance_mode", True)
    response = client.get("/api/v1/bootstrap")
    assert response.status_code == 503 and response.headers["retry-after"] == "60"
    assert client.get("/health/live").status_code == 200
    assert client.get("/health/ready").status_code == 200


def test_staging_chat_is_blocked(client, monkeypatch):
    from uuid import uuid4

    monkeypatch.setattr(get_settings(), "external_services_enabled", False)
    response = client.post(
        "/api/v1/chat",
        json={
            "turn_id": str(uuid4()),
            "conversation_id": str(uuid4()),
            "message": "Do something",
        },
    )
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "EXTERNAL_SERVICES_DISABLED"


def test_worker_lease_prevents_overlap_and_releases():
    with deploy.worker_lease(threading.Event()) as lease:
        assert lease is not None
        with engine().connect() as other:
            assert (
                other.scalar(text("SELECT pg_try_advisory_lock(hashtext('eridani:worker-supervisor'))"))
                is False
            )
    with deploy.worker_lease(threading.Event()) as lease:
        assert lease is not None


def test_paused_worker_does_not_initialize_dbos(monkeypatch):
    from jarvis import worker

    monkeypatch.setattr(get_settings(), "worker_enabled", False)
    event = threading.Event()
    event.set()
    monkeypatch.setattr(deploy, "stop_event", lambda: event)
    monkeypatch.setattr(worker, "DBOS", lambda *a, **k: pytest.fail("Paused worker initialized DBOS"))
    worker.main()


def test_readiness_rejects_missing_cloud_schema(monkeypatch):
    from jarvis.api import ready

    monkeypatch.setattr(get_settings(), "deployment_environment", "staging")
    response = ready()
    assert response.status_code == 503


def test_cloud_check_never_falls_back_to_local_secrets(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text("JARVIS_GOOGLE_CLIENT_SECRET=local-only\nOPENAI_API_KEY=local-only\n")
    monkeypatch.delenv("JARVIS_GOOGLE_CLIENT_SECRET", raising=False)
    monkeypatch.delenv("JARVIS_OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("JARVIS_ENV_FILE", "")
    get_settings.cache_clear()
    try:
        assert get_settings().google_client_secret == ""
        assert get_settings().openai_api_key == ""
    finally:
        get_settings.cache_clear()


def test_staging_voice_is_blocked_before_session_creation(client, monkeypatch):
    from uuid import uuid4

    from jarvis.voice import controllers

    monkeypatch.setattr(get_settings(), "external_services_enabled", False)
    monkeypatch.setattr(get_settings(), "openai_api_key", "never-send")
    existing = set(controllers)
    response = client.post(
        "/api/v1/voice/sessions",
        json={
            "provider": "live",
            "conversation_id": str(uuid4()),
            "voice": "ara",
            "sdp": "test",
        },
    )
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "EXTERNAL_SERVICES_DISABLED"
    assert set(controllers) == existing
