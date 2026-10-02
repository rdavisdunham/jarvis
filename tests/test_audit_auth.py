"""Audit regressions: bot scopes, record privacy, logout, invites, sessions, lock order."""

from urllib.parse import parse_qs, urlsplit
from uuid import uuid4

import pytest
from cryptography.fernet import Fernet
from fastapi.testclient import TestClient
from jarvis import bot_access, external_service
from jarvis import google_auth as auth
from jarvis.api import app
from jarvis.auth import digest
from jarvis.config import get_settings
from jarvis.db import session_scope
from jarvis.domain import DomainError, execute
from jarvis.models import AuthSession, BotCredential, GoogleOAuthAttempt, UserAccount
from test_accounts import client_for, post, shared
from test_external_agents import call, key


@pytest.fixture(autouse=True)
def settings(monkeypatch):
    monkeypatch.setenv("JARVIS_INTEGRATION_ENCRYPTION_KEY", Fernet.generate_key().decode())
    monkeypatch.setenv("JARVIS_OPENAI_API_KEY", "synthetic")
    monkeypatch.setenv("JARVIS_GEMINI_API_KEY", "synthetic")
    monkeypatch.setenv("JARVIS_COST_TRACKING_ENABLED", "false")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def schema_revision(client):
    return client.get("/api/v1/structure").json()["revision"]


def owner_task(client, title="Visible title", notes="classified-body"):
    r = client.post(
        "/api/v1/commands",
        json={
            "command_id": str(uuid4()),
            "tool": "task.create",
            "arguments": {"title": title, "notes": notes, "due_date": "2030-03-04"},
        },
    )
    assert r.status_code == 200, r.text
    task = r.json()["data"]
    listed = client.get("/api/v1/structure/records?capability=work").json()["items"]
    return task, next(x for x in listed if x["task_id"] == task["id"])


def test_records_write_alone_cannot_mutate_core_tasks_or_notes(client):
    task, record = owner_task(client)
    revision = schema_revision(client)
    _, records_only = key(client, ["records:write", "schema:read"])
    create = {"title": "Backdoor", "schema_revision": revision}
    assert call(records_only, "record.create", {**create, "type_id": "task"}).status_code == 403
    assert call(records_only, "record.create", {**create, "type_id": "note"}).status_code == 403
    update = {"record_id": record["id"], "expected_revision": record["revision"], "schema_revision": revision}
    assert call(records_only, "record.update", {**update, "title": "Renamed"}).status_code == 403
    assert call(records_only, "record.update", {**update, "archived": True}).status_code == 403
    assert call(records_only, "record.create", {**create, "type_id": "client"}).status_code == 200
    _, with_tasks = key(client, ["records:write", "tasks:write"])
    assert call(with_tasks, "record.create", {**create, "type_id": "note"}).status_code == 403
    renamed = call(with_tasks, "record.update", {**update, "title": "Renamed"})
    assert renamed.status_code == 200, renamed.text
    assert client.get("/api/v1/tasks/" + task["id"]).json()["title"] == "Renamed"
    # Queued bot work runs commands under the credential principal: the same rule applies.
    row = client.get("/api/v1/structure/records/" + record["id"]).json()
    credential = next(
        c for c in client.get("/api/v1/bot-keys").json()["items"] if c["scopes"] == ["records:read", "records:write", "schema:read"]
    )
    with bot_access.bind(credential["id"]), session_scope() as db, pytest.raises(DomainError) as exc:
        execute(db, "davin", "queued:0", "record.update", {**update, "expected_revision": row["revision"], "title": "Queued"})
    assert exc.value.code == "INSUFFICIENT_SCOPE"


def test_records_read_hides_core_bodies_and_task_fields(client):
    task, record = owner_task(client)
    external = TestClient(app)
    _, records_only = key(client, ["records:read", "schema:read"])
    got = external.get("/api/v1/external/structure/records/" + record["id"], headers=records_only)
    assert got.status_code == 200 and got.json()["title"] == "Visible title"
    assert "classified-body" not in got.text and "2030-03-04" not in got.text
    listed = external.get("/api/v1/external/structure/records", headers=records_only)
    assert "Visible title" in listed.text and "classified-body" not in listed.text
    feed = external.get("/api/v1/external/changes", headers=records_only)
    assert any(e["kind"] == "record" for e in feed.json()["items"]) and "classified-body" not in feed.text
    search = external.post("/api/v1/external/search", headers=records_only, json={"query": "Visible title"})
    assert search.status_code == 200, search.text
    assert "Visible title" in search.text and "classified-body" not in search.text
    credential = next(c for c in client.get("/api/v1/bot-keys").json()["items"])["id"]
    from jarvis.external_mcp import dispatch

    with bot_access.bind(credential):
        for name, args in (
            ("record_get", {"record_id": record["id"]}),
            ("record_list", {}),
            ("record_search", {"query": "Visible title"}),
        ):
            assert "classified-body" not in str(dispatch(name, args))
            assert "classified-body" not in str(external_service.backend_read(name, args, None))
    _, readers = key(client, ["records:read", "tasks:read"])
    full = external.get("/api/v1/external/structure/records/" + record["id"], headers=readers)
    assert "classified-body" in full.text and "2030-03-04" in full.text


def test_queued_bot_reads_cover_every_advertised_read_tool(client):
    _, record = owner_task(client)
    credential, _ = key(client, ["records:read", "schema:read", "notes:read", "work:run"])
    with bot_access.bind(credential["id"]):
        assert external_service.backend_read("structure_schema", {}, None)["revision"] >= 1
        assert external_service.backend_read("record_list", {}, None)["items"]
        got = external_service.backend_read("record_get", {"record_id": record["id"]}, None)
        assert got["id"] == record["id"] and "body" not in got
        external_service.backend_read("note_lists", {}, None)
        with pytest.raises(DomainError) as exc:
            external_service.backend_read("note_list_items", {"list_id": str(uuid4())}, None)
        assert exc.value.code != "INVALID_ARGUMENT"


def test_viewer_and_revoked_member_can_log_out_and_viewer_can_search(client):
    guest = client_for("guest", "guest@example.test")
    workspace = shared(client, guest, "viewer")
    assert guest.post("/api/v1/search/records", json={"query": "launch"}).status_code == 200
    assert guest.post("/api/v1/tasks", json={}).status_code in {403, 404, 405, 422}
    token = guest.cookies.get("jarvis_session")
    assert guest.post("/api/v1/auth/logout", headers={"X-CSRF-Token": "wrong"}).status_code == 403
    assert guest.post("/api/v1/auth/logout").status_code == 200
    with session_scope() as db:
        assert db.get(AuthSession, digest(token)) is None
    other = client_for("guest2", "guest2@example.test")
    invite = post(client, "/accounts/invitations", {"workspace_id": workspace["id"], "email": "guest2@example.test", "role": "editor"})
    post(other, "/accounts/accept", {"invite_id": invite["id"]})
    post(other, "/accounts/switch", {"workspace_id": workspace["id"]})
    member = next(m for m in client.get("/api/v1/accounts/members/" + workspace["id"]).json()["members"] if m["account_id"] == "guest2")
    post(client, "/accounts/members", {"workspace_id": workspace["id"], "account_id": "guest2", "role": "editor", "expected_revision": member["revision"], "active": False})
    token = other.cookies.get("jarvis_session")
    assert other.get("/api/v1/bootstrap").status_code == 403
    assert other.post("/api/v1/auth/logout").status_code == 200
    with session_scope() as db:
        assert db.get(AuthSession, digest(token)) is None


def test_member_removal_revokes_bot_keys_permanently(client):
    guest = client_for("guest", "guest@example.test")
    workspace = shared(client, guest)
    credential, headers = key(guest)
    assert call(headers).status_code == 200
    member = next(m for m in client.get("/api/v1/accounts/members/" + workspace["id"]).json()["members"] if m["account_id"] == "guest")
    post(client, "/accounts/members", {"workspace_id": workspace["id"], "account_id": "guest", "role": "editor", "expected_revision": member["revision"], "active": False})
    with session_scope() as db:
        assert db.get(BotCredential, credential["id"]).revoked_at is not None
    invite = post(client, "/accounts/invitations", {"workspace_id": workspace["id"], "email": "guest@example.test", "role": "editor"})
    post(guest, "/accounts/accept", {"invite_id": invite["id"]})
    assert call(headers).status_code == 401


@pytest.fixture
def google(monkeypatch):
    from jarvis.google_routes import oauth_attempts

    oauth_attempts.clear()
    s = get_settings()
    monkeypatch.setattr(s, "origin", "http://localhost")
    monkeypatch.setattr(s, "google_client_id", "fixture.apps.googleusercontent.com")
    monkeypatch.setattr(s, "google_client_secret", "fixture-client-secret")
    monkeypatch.setattr(s, "integration_encryption_key", Fernet.generate_key().decode())

    def run(c, purpose, subject, email, **tokens):
        response = c.post("/api/v1/auth/google/start", json={"purpose": purpose})
        assert response.status_code == 200, response.text
        state = parse_qs(urlsplit(response.json()["url"]).query)["state"][0]
        with session_scope() as db:
            nonce = db.get(GoogleOAuthAttempt, auth.digest(state)).nonce
        monkeypatch.setattr(auth, "exchange", lambda *_: {"id_token": "fixture", **tokens})
        claims = {"sub": subject, "email": email, "email_verified": True, "nonce": nonce}
        monkeypatch.setattr(auth, "verify_identity", lambda _: claims)
        return auth.finish(state, c.cookies.get(auth.COOKIE), "code")

    return run


def test_only_server_owner_invites_create_accounts(client, google):
    guest = client_for("guest", "guest@example.test")
    workspace = post(guest, "/accounts/workspaces", {"name": "Guest club", "kind": "space"})
    post(guest, "/accounts/invitations", {"workspace_id": workspace["id"], "email": "new@example.test", "role": "editor"})
    with pytest.raises(DomainError, match="invitation"):
        google(TestClient(app), "login", "new-subject", "new@example.test")
    with session_scope() as db:
        before = db.query(UserAccount).count()
    owned = post(client, "/accounts/workspaces", {"name": "Owner team", "kind": "space"})
    post(client, "/accounts/invitations", {"workspace_id": owned["id"], "email": "new@example.test", "role": "editor"})
    token, _ = google(TestClient(app), "login", "new-subject", "new@example.test")
    with session_scope() as db:
        assert db.query(UserAccount).count() == before + 1
        assert db.get(AuthSession, digest(token))


def test_google_link_and_consent_delete_the_replaced_session(client, google):
    old = client.cookies.get("jarvis_session")
    token, csrf = google(client, "link", "owner-subject", "owner@example.test")
    with session_scope() as db:
        assert db.get(AuthSession, digest(old)) is None and db.get(AuthSession, digest(token))
    client.cookies.set("jarvis_session", token)
    client.headers["X-CSRF-Token"] = csrf
    newer, _ = google(
        client, "calendar", "owner-subject", "owner@example.test",
        scope="openid " + auth.CALENDAR_SCOPE, refresh_token="fixture-refresh",
    )
    with session_scope() as db:
        assert db.get(AuthSession, digest(token)) is None and db.get(AuthSession, digest(newer))


def test_bot_cancel_reply_and_submit_take_work_order_before_work(client, monkeypatch):
    credential, headers = key(client, ["tasks:write", "work:run"])
    seen = []
    real = external_service.advisory
    monkeypatch.setattr(external_service, "advisory", lambda db, name: (seen.append(name.split(":")[0]), real(db, name))[1])
    submitted = TestClient(app).post("/api/v1/external/requests", headers=headers, json={"request_id": str(uuid4()), "message": "Plan my week"})
    assert submitted.status_code == 202, submitted.text
    assert seen.index("work-order") < seen.index("work")
    seen.clear()
    work_id = submitted.json()["id"]
    reply = TestClient(app).post(f"/api/v1/external/requests/{work_id}/reply", headers=headers, json={"request_id": str(uuid4()), "message": "Also Friday", "expected_revision": submitted.json()["revision"]})
    assert reply.status_code in {200, 400, 409}, reply.text
    assert seen.index("work-order") < seen.index("work")
    seen.clear()
    assert TestClient(app).post(f"/api/v1/external/requests/{work_id}/cancel", headers=headers).status_code == 200
    assert seen.index("work-order") < seen.index("work")


def test_records_only_search_cannot_probe_core_bodies(client):
    owner_task(client, title="Visible title", notes="pineapple-ledger")
    external = TestClient(app)
    _, records_only = key(client, ["records:read"])
    _, with_tasks = key(client, ["records:read", "tasks:read"])

    def found(headers, query):
        r = external.post("/api/v1/external/search", headers=headers, json={"query": query})
        assert r.status_code == 200, r.text
        return r.text

    assert "Visible title" not in found(records_only, "pineapple-ledger")
    assert "Visible title" in found(records_only, "Visible title")
    assert "Visible title" in found(with_tasks, "pineapple-ledger")
