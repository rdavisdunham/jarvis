"""/api/v1/context/settings: owner-session routes for the Sources panel (file import, deliveries, optional Granola).

No network: the Granola check goes through an httpx MockTransport. Keys never appear in responses or rows.
"""

import httpx
import pytest
from cryptography.fernet import Fernet
from fastapi.testclient import TestClient
from jarvis.api import app
from jarvis.config import get_settings
from jarvis.context import models as m
from jarvis.context.connectors import granola
from jarvis.context.routes import settings as routes
from jarvis.db import session_scope
from jarvis.models import Job
from sqlalchemy import func, select

from .conftest import MEETING_AT

BASE = "/api/v1/context/settings"
KEY = "gk_live_0123456789secret"
ACCOUNT = "davin@example.com"
VTT = "WEBVTT\n\n00:00:01.000 --> 00:00:03.000\n<v Davin>Kick-off.</v>\n\n00:00:04.000 --> 00:00:06.000\n<v Priya>Ready.</v>\n"


@pytest.fixture(autouse=True)
def enabled(monkeypatch):
    monkeypatch.setenv("JARVIS_CONTEXT_ENABLED", "true")
    monkeypatch.setenv("JARVIS_INTEGRATION_ENCRYPTION_KEY", Fernet.generate_key().decode())
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def granola_on(monkeypatch):
    monkeypatch.setenv("JARVIS_CONTEXT_GRANOLA_ENABLED", "true")
    get_settings.cache_clear()


class Upstream:
    """Answers GET /v1/notes with `status` (and one note owned by ACCOUNT on success)."""

    def __init__(self, status: int = 200):
        self.status, self.calls = status, []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.calls.append(request)
        assert request.url.path == "/v1/notes"
        if self.status != 200:
            return httpx.Response(self.status, json={"error": "no"})
        return httpx.Response(200, json={"notes": [{"id": "not_A1b2C3d4E5f6G7", "owner": {"email": ACCOUNT}},
                                                   {"id": "not_H8i9J0k1L2m3N4", "owner": {"email": "x@y.z"}}],
                                         "hasMore": False})


@pytest.fixture
def upstream(monkeypatch):
    server = Upstream()
    monkeypatch.setattr(routes, "granola_client", lambda key: granola.GranolaClient(
        key, transport=httpx.MockTransport(server), max_retries=0, sleep=lambda _s: None))
    return server


def code(response):
    return response.status_code, response.json()["error"]["code"]


def upload(client, data=VTT, content_type="text/vtt", **params):
    params = {k: v for k, v in {"filename": "kickoff.vtt", "occurred_at": MEETING_AT.isoformat(), **params}.items()
              if v is not None}
    return client.post(BASE + "/import", params=params, content=data.encode(), headers={"Content-Type": content_type})


def connect(client, **body):
    return client.post(BASE + "/granola", json={"api_key": KEY, "account": ACCOUNT, **body})


def jobs(kind):
    with session_scope() as db:
        return list(db.scalars(select(Job).where(Job.kind == kind)))


def test_overview_shows_file_import_and_endpoint_as_live_sources_without_granola(client):
    data = client.get(BASE).json()
    live = {a["key"]: a["live"] for a in data["adapters"]}
    assert live == {"upload": True, "api": True, "granola": False, "zoom": False}
    assert data["deliveries"] == [] and data["granola"] == {"enabled": False, "connections": []}
    assert data["submission"]["url"] == "http://testserver/api/v1/external/sources/{connector}/{external_id}"
    assert data["submission"]["scope"] == "sources:write" and "bridge_submit.py" in data["submission"]["bridge"]
    assert data["limits"]["max_source_chars"] == get_settings().context_max_source_chars
    assert set(data["limits"]["formats"]) == {"txt", "markdown", "vtt", "srt", "json"}


def test_file_import_is_idempotent_and_deliveries_count_per_label(client):
    first = upload(client)
    assert first.status_code == 200, first.text
    created = first.json()
    assert (created["title"], created["format"], created["ingestion_status"], created["revision"]) == (
        "kickoff", "vtt", "created", 1)
    again = upload(client).json()
    assert (again["id"], again["ingestion_status"]) == (created["id"], "none")
    markdown = upload(client, "# Retro\n\nDavin: Went well.\n", "application/octet-stream", filename="retro.md")
    assert markdown.json()["title"] == "Retro" and markdown.json()["format"] == "markdown"
    token = client.post("/api/v1/bot-keys", json={"name": "rec", "scopes": ["sources:write"],
                                                  "expires_in_days": 30}).json()["token"]
    put = TestClient(app).put("/api/v1/external/sources/recorder/m1", content=VTT.encode(),
                              params={"occurred_at": MEETING_AT.isoformat()},
                              headers={"Authorization": "Bearer " + token, "Content-Type": "text/vtt"})
    assert put.status_code == 200, put.text
    delivered = {(d["connector"], d["label"]): d for d in client.get(BASE).json()["deliveries"]}
    assert set(delivered) == {("upload", "upload"), ("api", "recorder")}
    assert delivered[("upload", "upload")]["sources"] == 2 and delivered[("api", "recorder")]["sources"] == 1
    assert delivered[("api", "recorder")]["last_delivery_at"]
    with session_scope() as db:
        assert db.scalar(select(func.count()).select_from(m.ContextSource).where(
            m.ContextSource.connector == "upload")) == 2


def test_file_import_refuses_bad_input(client):
    assert code(upload(client, "plain words", "text/plain", occurred_at=None)) == (400, "INVALID_ARGUMENT")
    assert code(upload(client, "", "text/plain")) == (400, "INVALID_ARGUMENT")
    assert code(upload(client, "x", "application/pdf", filename="deck.pdf")) == (415, "UNSUPPORTED_MEDIA_TYPE")
    with session_scope() as db:
        assert db.scalar(select(func.count()).select_from(m.ContextSource)) == 0


def test_routes_refuse_when_context_is_off(client, monkeypatch):
    monkeypatch.setenv("JARVIS_CONTEXT_ENABLED", "false")
    get_settings.cache_clear()
    assert code(client.get(BASE)) == (403, "CONTEXT_DISABLED")
    assert code(upload(client)) == (403, "CONTEXT_DISABLED")
    assert code(connect(client)) == (409, "GRANOLA_DISABLED")


def test_routes_refuse_in_shared_workspaces(client, granola_on, upstream):
    from test_accounts import client_for, shared

    shared(client, client_for("guest"))
    assert code(client.get(BASE)) == (403, "PERSONAL_WORKSPACE")
    assert code(upload(client)) == (403, "PERSONAL_WORKSPACE")
    assert code(connect(client)) == (403, "PERSONAL_WORKSPACE")
    assert upstream.calls == []


def test_routes_need_a_session(client):
    assert code(TestClient(app).get(BASE)) == (401, "NOT_AUTHORIZED")


def test_granola_routes_refuse_unless_the_adapter_is_enabled(client, upstream):
    assert code(connect(client)) == (409, "GRANOLA_DISABLED")
    assert code(client.post(BASE + "/granola/x/sync")) == (409, "GRANOLA_DISABLED")
    assert upstream.calls == []


def test_granola_key_is_verified_and_stored_only_as_ciphertext(client, granola_on, upstream):
    response = connect(client, account="Davin@Example.com")
    assert response.status_code == 200, response.text
    data = response.json()
    assert KEY not in response.text
    assert (data["account"], data["health"], data["has_key"]) == (ACCOUNT, "ok", True)
    assert data["verification"] == {"listed": 2, "owned": 1}
    assert upstream.calls[0].headers["Authorization"] == "Bearer " + KEY
    with session_scope() as db:
        row = db.get(m.ContextConnection, data["id"])
        assert row.credential_ciphertext and KEY not in row.credential_ciphertext
        assert granola.open_credential(row) == KEY
    overview = client.get(BASE).json()
    assert [c["id"] for c in overview["granola"]["connections"]] == [data["id"]]
    assert {a["key"]: a["live"] for a in overview["adapters"]}["granola"] is True
    assert KEY not in client.get(BASE).text
    assert code(connect(client, account="someone@else.com")) == (409, "GRANOLA_CONNECTED")
    assert connect(client).json()["id"] == data["id"]  # same account: the key is replaced in place


@pytest.mark.parametrize(("status", "expected"), [(401, (400, "GRANOLA_KEY_REJECTED")),
                                                  (403, (400, "GRANOLA_NOT_ENTITLED")),
                                                  (503, (503, "GRANOLA_UNAVAILABLE"))])
def test_granola_connect_explains_upstream_refusals(client, granola_on, upstream, status, expected):
    upstream.status = status
    response = connect(client)
    assert code(response) == expected
    if status == 403:
        assert "Business or Enterprise" in response.json()["error"]["message"]
    with session_scope() as db:
        assert db.scalar(select(func.count()).select_from(m.ContextConnection)) == 0


def test_granola_test_records_health(client, granola_on, upstream):
    connection = connect(client).json()["id"]
    upstream.status = 401
    result = client.post(f"{BASE}/granola/{connection}/test").json()
    assert (result["ok"], result["health"]) == (False, "auth_failed") and "did not accept" in result["last_error"]
    upstream.status = 200
    result = client.post(f"{BASE}/granola/{connection}/test").json()
    assert (result["ok"], result["health"], result["last_error"]) == (True, "ok", None)


def test_sync_and_backfill_enqueue_jobs_once(client, granola_on, upstream):
    connection = connect(client).json()["id"]
    first = client.post(f"{BASE}/granola/{connection}/sync").json()
    again = client.post(f"{BASE}/granola/{connection}/sync").json()
    assert (first["already_queued"], again["already_queued"], again["job_id"]) == (False, True, first["job_id"])
    assert [j.payload for j in jobs(routes.SYNC_JOB)] == [{"connection_id": connection}]
    assert client.post(f"{BASE}/granola/{connection}/backfill", json={"days": 30}).status_code == 200
    assert [j.payload for j in jobs(routes.BACKFILL_JOB)] == [{"connection_id": connection, "days": 30}]
    assert client.post(f"{BASE}/granola/{connection}/backfill", json={"days": 400}).status_code == 422
    assert code(client.post(f"{BASE}/granola/unknown/sync")) == (404, "NOT_FOUND")


def test_remove_forgets_the_key_and_keeps_imported_meetings(client, granola_on, upstream):
    connection = connect(client).json()["id"]
    upload(client)
    assert client.delete(f"{BASE}/granola/{connection}").json() == {"id": connection, "removed": True}
    with session_scope() as db:
        row = db.get(m.ContextConnection, connection)
        assert (row.credential_ciphertext, row.enabled) == (None, False)
        assert db.scalar(select(func.count()).select_from(m.ContextSource)) == 1
    assert client.get(BASE).json()["granola"]["connections"] == []
    assert code(client.post(f"{BASE}/granola/{connection}/sync")) == (404, "NOT_FOUND")
    reconnected = connect(client).json()
    assert (reconnected["id"], reconnected["has_key"]) == (connection, True)


def test_owner_routes_are_mounted_on_the_app_with_session_csrf(client):
    paths = set(app.openapi()["paths"])
    assert BASE in paths and BASE + "/import" in paths
    assert not any(p.startswith("/api/v1/external/settings") for p in paths)
    csrf = client.headers.pop("X-CSRF-Token")
    try:
        assert code(upload(client)) == (403, "NOT_AUTHORIZED")
        assert code(connect(client)) == (403, "NOT_AUTHORIZED")
    finally:
        client.headers["X-CSRF-Token"] = csrf
    with session_scope() as db:
        assert db.scalar(select(func.count()).select_from(m.ContextSource)) == 0


def test_import_has_the_transcript_body_cap_not_the_100_kb_default(client, monkeypatch):
    from jarvis.context.routes.sources import body_limit

    large = "\n\n".join(f"Davin: Line {i}. " + "x" * 1000 for i in range(150))
    assert len(large) > 150_000
    accepted = upload(client, large, "text/plain", filename="long.txt")
    assert accepted.status_code == 200, accepted.text
    monkeypatch.setenv("JARVIS_CONTEXT_MAX_SOURCE_CHARS", "1000")
    get_settings.cache_clear()
    assert len(large) > body_limit()
    assert upload(client, large + " more", "text/plain", filename="long2.txt").status_code == 413


def stored_settings():
    with session_scope() as db:
        state = db.get(m.ContextOwnerState, "davin")
        return dict(state.settings or {}) if state else {}


def test_preferences_show_defaults_and_save_only_allow_listed_fields(client):
    from jarvis.context.lint import DEFAULT_BUDGET_USD

    data = client.get(BASE).json()
    assert data["preferences"] == {"synthesize_briefs": False, "semantic_supersede": False,
                                   "lint_budget_usd": DEFAULT_BUDGET_USD, "review_destination": "eridani"}
    assert data["preference_limits"]["lint_budget_usd"]["max"] == routes.LINT_BUDGET_MAX_USD
    with session_scope() as db:  # keys the route must keep and never expose
        db.add(m.ContextOwnerState(owner_id="davin", settings={"engine": "double", "lint": {"spent": 0.1}}))
    saved = client.patch(BASE + "/preferences", json={"synthesize_briefs": True, "lint_budget_usd": 1.25})
    assert saved.status_code == 200, saved.text
    assert saved.json()["preferences"] == {"synthesize_briefs": True, "semantic_supersede": False,
                                           "lint_budget_usd": 1.25, "review_destination": "eridani"}
    client.patch(BASE + "/preferences", json={"review_destination": "linear", "semantic_supersede": True})
    assert stored_settings() == {"engine": "double", "lint": {"spent": 0.1}, "synthesize_briefs": True,
                                 "lint_budget_usd": 1.25, "review_destination": "linear", "semantic_supersede": True}
    assert "engine" not in client.get(BASE).json()["preferences"]
    from jarvis.context import pipeline, proposals, temporal
    with session_scope() as db:
        assert pipeline.synthesize_briefs(db, "davin") and temporal.semantic_supersede_enabled(db, "davin")
        assert proposals.destination(db, "davin") == "linear"


@pytest.mark.parametrize("body", [
    {}, {"engine": "double"}, {"synthesize_briefs": "yes"}, {"synthesize_briefs": None},
    {"lint_budget_usd": -1}, {"lint_budget_usd": 999}, {"review_destination": "jira"},
    {"synthesize_briefs": True, "lint": {}},
])
def test_preferences_refuse_anything_outside_the_allow_list(client, body):
    assert client.patch(BASE + "/preferences", json=body).status_code == 422
    assert stored_settings() == {}


def test_preferences_need_a_session_csrf_the_flag_and_a_personal_workspace(client, monkeypatch):
    change = {"synthesize_briefs": True}
    assert code(TestClient(app).patch(BASE + "/preferences", json=change)) == (401, "NOT_AUTHORIZED")
    token = client.post("/api/v1/bot-keys", json={"name": "ctx", "scopes": ["context:write"],
                                                  "expires_in_days": 30}).json()["token"]
    bot = TestClient(app).patch(BASE + "/preferences", json=change, headers={"Authorization": "Bearer " + token})
    assert code(bot) == (401, "NOT_AUTHORIZED")
    csrf = client.headers.pop("X-CSRF-Token")
    try:
        assert code(client.patch(BASE + "/preferences", json=change)) == (403, "NOT_AUTHORIZED")
    finally:
        client.headers["X-CSRF-Token"] = csrf
    monkeypatch.setenv("JARVIS_CONTEXT_ENABLED", "false")
    get_settings.cache_clear()
    assert code(client.patch(BASE + "/preferences", json=change)) == (403, "CONTEXT_DISABLED")
    monkeypatch.setenv("JARVIS_CONTEXT_ENABLED", "true")
    get_settings.cache_clear()
    from test_accounts import client_for, shared

    shared(client, client_for("guest"))
    assert code(client.patch(BASE + "/preferences", json=change)) == (403, "PERSONAL_WORKSPACE")
    assert stored_settings() == {}
