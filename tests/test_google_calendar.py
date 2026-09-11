import json
import time
from datetime import date, timedelta
from types import SimpleNamespace
from urllib.parse import parse_qs, urlsplit
from uuid import uuid4

import pytest
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from google.auth import crypt, jwt
from google.auth.exceptions import GoogleAuthError
from jarvis import google_auth as auth
from jarvis import google_calendar as calendar
from jarvis.config import get_settings
from jarvis.db import session_scope
from jarvis.domain import DomainError, execute
from jarvis.google_projection import project
from jarvis.models import (
    AuthSession,
    GoogleCalendar,
    GoogleCalendarEvent,
    GoogleIdentity,
    GoogleOAuthAttempt,
    Job,
    now,
)
from sqlalchemy import delete, select


@pytest.fixture(autouse=True)
def google_setup(monkeypatch):
    from jarvis.google_routes import oauth_attempts

    oauth_attempts.clear()
    settings = get_settings()
    monkeypatch.setattr(settings, "origin", "http://localhost")
    monkeypatch.setattr(settings, "google_client_id", "fixture.apps.googleusercontent.com")
    monkeypatch.setattr(settings, "google_client_secret", "fixture-client-secret")
    monkeypatch.setattr(settings, "integration_encryption_key", Fernet.generate_key().decode())
    # No provider call is allowed unless that test supplies an explicit fake.
    monkeypatch.setattr(
        calendar, "CalendarClient", lambda *_: (_ for _ in ()).throw(AssertionError("Unexpected Google call"))
    )
    monkeypatch.setattr(
        auth, "exchange", lambda *_: (_ for _ in ()).throw(AssertionError("Unexpected OAuth exchange"))
    )


def linked(selected=True):
    with session_scope() as db:
        account = GoogleIdentity(
            owner_id="davin",
            subject="google-subject",
            email="owner@example.test",
            credentials=auth.seal({"refresh_token": "fixture-refresh"}),
            calendar_enabled=True,
            status="ready",
            last_sync_at=now(),
            next_sync_at=now(),
        )
        db.add(account)
        db.flush()
        source = GoogleCalendar(
            owner_id="davin",
            provider_id="owner@example.test",
            title="Personal",
            timezone="America/Chicago",
            selected=selected,
            primary=True,
            last_sync_at=now(),
        )
        db.add(source)
        db.flush()
        return source.id


def event(source, provider_id="event-one", **payload):
    with session_scope() as db:
        row = GoogleCalendarEvent(
            calendar_id=source,
            provider_id=provider_id,
            payload={
                "id": provider_id,
                "summary": "Planning",
                "start": {"dateTime": "2026-09-18T09:00:00-05:00", "timeZone": "America/Chicago"},
                "end": {"dateTime": "2026-09-18T10:00:00-05:00", "timeZone": "America/Chicago"},
                **payload,
            },
        )
        db.add(row)
        db.flush()
        return row.id


def start(client, purpose="link"):
    response = client.post("/api/v1/auth/google/start", json={"purpose": purpose})
    assert response.status_code == 200, response.text
    state = parse_qs(urlsplit(response.json()["url"]).query)["state"][0]
    cookie = client.cookies.get(auth.COOKIE)
    with session_scope() as db:
        row = db.get(GoogleOAuthAttempt, auth.digest(state))
        nonce = row.nonce
    return state, cookie, nonce, response


def successful_exchange(monkeypatch, nonce, subject="google-subject", **tokens):
    monkeypatch.setattr(auth, "exchange", lambda *_: {"id_token": "fixture-id", **tokens})
    monkeypatch.setattr(
        auth,
        "verify_identity",
        lambda _: {
            "sub": subject,
            "email": "owner@example.test",
            "email_verified": True,
            "nonce": nonce,
        },
    )


def test_google_link_requires_pairing_and_login_cannot_claim_first_account(client):
    assert client.get("/api/v1/auth/options").json()["google"] is False
    assert client.post("/api/v1/auth/google/start", json={"purpose": "login"}).status_code == 401
    client.cookies.clear()
    assert client.post("/api/v1/auth/google/start", json={"purpose": "link"}).status_code == 401


def test_oauth_start_uses_bound_cookie_pkce_nonce_and_separate_permissions(client):
    _state, cookie, nonce, response = start(client)
    params = parse_qs(urlsplit(response.json()["url"]).query)
    assert params["code_challenge_method"] == ["S256"]
    assert params["nonce"] == [nonce] and len(cookie) > 32
    assert calendar.configured()
    assert "HttpOnly" in response.headers["set-cookie"] and "SameSite=lax" in response.headers["set-cookie"]
    assert auth.CALENDAR_SCOPE not in params["scope"][0]
    assert "fixture-client-secret" not in response.text
    _, _, _, response = start(client, "calendar")
    params = parse_qs(urlsplit(response.json()["url"]).query)
    assert auth.CALENDAR_SCOPE in params["scope"][0] and params["access_type"] == ["offline"]


def test_oauth_binding_is_one_use_and_uses_subject_not_email(client, monkeypatch):
    state, browser, nonce, _ = start(client)
    successful_exchange(monkeypatch, nonce)
    token, csrf = auth.finish(state, browser, "code")
    assert token and csrf
    with pytest.raises(DomainError):
        auth.finish(state, browser, "code")
    with session_scope() as db:
        row = db.get(GoogleIdentity, "davin")
        assert row.subject == "google-subject" and not row.calendar_enabled and row.credentials is None
    state, browser, nonce, _ = start(client, "login")
    successful_exchange(monkeypatch, nonce, subject="other-account-same-email")
    with pytest.raises(DomainError, match="already linked"):
        auth.finish(state, browser, "code")


def test_oauth_wrong_browser_denial_expiry_and_nonce_never_link(client, monkeypatch):
    state, browser, _nonce, _ = start(client)
    with pytest.raises(DomainError):
        auth.finish(state, "wrong-browser", "code")
    with pytest.raises(DomainError, match="cancelled"):
        auth.finish(state, browser, error="access_denied")
    state, browser, _nonce, _ = start(client)
    successful_exchange(monkeypatch, "wrong-nonce")
    with pytest.raises(DomainError):
        auth.finish(state, browser, "code")
    state, browser, _nonce, _ = start(client)
    with session_scope() as db:
        db.get(GoogleOAuthAttempt, auth.digest(state)).expires_at = now() - timedelta(seconds=1)
    with pytest.raises(DomainError):
        auth.finish(state, browser, "code")
    with session_scope() as db:
        assert db.get(GoogleIdentity, "davin") is None


def test_oauth_cannot_link_after_original_pairing_session_ends(client, monkeypatch):
    state, browser, nonce, _ = start(client)
    successful_exchange(monkeypatch, nonce)
    with session_scope() as db:
        db.execute(delete(AuthSession))
    with pytest.raises(DomainError):
        auth.finish(state, browser, "code")
    with session_scope() as db:
        assert db.get(GoogleIdentity, "davin") is None


def test_calendar_grant_is_separate_and_credentials_are_encrypted(client, monkeypatch):
    state, browser, nonce, _ = start(client)
    successful_exchange(monkeypatch, nonce)
    auth.finish(state, browser, "code")
    state, browser, nonce, _ = start(client, "calendar")
    successful_exchange(monkeypatch, nonce, scope="openid", refresh_token="fixture-refresh")
    with pytest.raises(DomainError, match="permission"):
        auth.finish(state, browser, "code")
    with session_scope() as db:
        assert db.get(GoogleIdentity, "davin").credentials is None
    state, browser, nonce, _ = start(client, "calendar")
    successful_exchange(
        monkeypatch, nonce, scope="openid " + auth.CALENDAR_SCOPE, refresh_token="fixture-refresh"
    )
    auth.finish(state, browser, "code")
    with session_scope() as db:
        row = db.get(GoogleIdentity, "davin")
        assert row.calendar_enabled and "fixture-refresh" not in row.credentials
        assert auth.unseal(row.credentials)["refresh_token"] == "fixture-refresh"
    assert "fixture-refresh" not in client.get("/api/v1/integrations/google").text
    assert "fixture-refresh" not in client.get("/api/v1/export").text


def test_real_google_verifier_checks_signature_audience_issuer_and_expiration(monkeypatch):
    private = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    pem = private.private_bytes(
        serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()
    )
    public = (
        private.public_key()
        .public_bytes(serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo)
        .decode()
    )
    signer = crypt.RSASigner.from_string(pem, key_id="fixture")
    response = SimpleNamespace(status=200, data=json.dumps({"fixture": public}).encode())
    monkeypatch.setattr(auth, "VerificationRequest", lambda: lambda *a, **kw: response)
    claims = {
        "iss": "https://accounts.google.com",
        "aud": get_settings().google_client_id,
        "iat": int(time.time()) - 10,
        "exp": int(time.time()) + 300,
        "sub": "google-subject",
    }
    assert auth.verify_identity(jwt.encode(signer, claims))["sub"] == "google-subject"
    for patch in [
        {"aud": "wrong-client"},
        {"iss": "https://attacker.invalid"},
        {"exp": int(time.time()) - 120},
    ]:
        with pytest.raises((ValueError, GoogleAuthError)):
            auth.verify_identity(jwt.encode(signer, {**claims, **patch}))


class Provider:
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def request(self, path, params=None, body=None):
        self.calls.append((path, params, body))
        response = self.replies.pop(0)
        if isinstance(response, Exception):
            raise response
        return response

    def close(self):
        pass


def run_sync(monkeypatch, provider):
    monkeypatch.setattr(calendar, "CalendarClient", lambda _: provider)
    with session_scope() as db:
        jid = calendar.queue_sync(db, "davin", force=True)
    calendar.process(jid)
    with session_scope() as db:
        return db.get(Job, jid).status


def remote_calendar():
    return {
        "items": [
            {
                "id": "owner@example.test",
                "summary": "Personal",
                "primary": True,
                "timeZone": "America/Chicago",
            }
        ]
    }


def test_paged_full_sync_incremental_deletion_and_invalid_cursor_recovery(client, monkeypatch):
    source = linked()
    p = Provider(
        [
            remote_calendar(),
            {"items": [{"id": "a", "summary": "A"}], "nextPageToken": "page2"},
            {"items": [{"id": "b", "summary": "B"}], "nextSyncToken": "cursor1"},
        ]
    )
    assert run_sync(monkeypatch, p) == "succeeded"
    assert p.calls[-1][1]["pageToken"] == "page2"
    p = Provider(
        [remote_calendar(), {"items": [{"id": "a", "status": "cancelled"}], "nextSyncToken": "cursor2"}]
    )
    assert run_sync(monkeypatch, p) == "succeeded"
    assert p.calls[-1][1]["syncToken"] == "cursor1"
    assert "timeMin" not in p.calls[-1][1] and "orderBy" not in p.calls[-1][1]
    with session_scope() as db:
        rows = {r.provider_id: r.payload for r in db.scalars(select(GoogleCalendarEvent))}
        assert rows["a"]["status"] == "cancelled" and "b" in rows
    p = Provider(
        [
            remote_calendar(),
            calendar.SyncFailure("gone", 410),
            {"items": [{"id": "c", "summary": "Replacement"}], "nextSyncToken": "new"},
        ]
    )
    assert run_sync(monkeypatch, p) == "succeeded"
    assert "syncToken" not in p.calls[-1][1]
    with session_scope() as db:
        assert [r.provider_id for r in db.scalars(select(GoogleCalendarEvent))] == ["c"]
        assert db.get(GoogleCalendar, source).sync_token == "new"


def test_partial_sync_failure_keeps_previous_snapshot_and_cursor(client, monkeypatch):
    source = linked()
    event(source)
    with session_scope() as db:
        db.get(GoogleCalendar, source).sync_token = "old"
    provider = Provider(
        [
            remote_calendar(),
            {"items": [{"id": "new"}], "nextPageToken": "next"},
            calendar.SyncFailure("unavailable", 500),
        ]
    )
    assert run_sync(monkeypatch, provider) == "failed"
    with session_scope() as db:
        assert db.get(GoogleCalendar, source).sync_token == "old"
        assert [r.provider_id for r in db.scalars(select(GoogleCalendarEvent))] == ["event-one"]
        assert db.get(GoogleIdentity, "davin").status == "error"


def test_removed_calendar_and_acl_failure_purge_cached_events(client, monkeypatch):
    source = linked()
    event(source)
    assert (
        run_sync(monkeypatch, Provider([remote_calendar(), calendar.SyncFailure("unavailable", 403)]))
        == "succeeded"
    )
    with session_scope() as db:
        assert not db.get(GoogleCalendar, source).available
        assert list(db.scalars(select(GoogleCalendarEvent))) == []
    assert run_sync(monkeypatch, Provider([{"items": []}])) == "succeeded"
    with session_scope() as db:
        assert not db.get(GoogleCalendar, source).available


def test_sync_result_cannot_restore_data_after_disconnect(client, monkeypatch):
    source = linked()
    event(source)

    class Disconnecting(Provider):
        def request(self, path, params=None, body=None):
            if path.endswith("/events"):
                with session_scope() as db:
                    row = db.get(GoogleIdentity, "davin")
                    row.calendar_enabled = False
                    row.generation += 1
                    db.execute(delete(GoogleCalendarEvent))
            return super().request(path, params, body)

    p = Disconnecting([remote_calendar(), {"items": [{"id": "late"}], "nextSyncToken": "new"}])
    assert run_sync(monkeypatch, p) == "cancelled"
    with session_scope() as db:
        assert list(db.scalars(select(GoogleCalendarEvent))) == []


def test_calendar_selection_revision_owner_checks_and_cache_removal(client):
    source = linked()
    event(source)
    with session_scope() as db:
        result = execute(
            db,
            "davin",
            str(uuid4()),
            "calendar.select",
            {"calendar_id": source, "expected_revision": 1, "selected": False},
        )
        assert result["data"]["selected"] is False
        assert list(db.scalars(select(GoogleCalendarEvent))) == []
    with session_scope() as db, pytest.raises(DomainError):
        execute(
            db,
            "davin",
            str(uuid4()),
            "calendar.select",
            {"calendar_id": source, "expected_revision": 1, "selected": True},
        )
    with session_scope() as db, pytest.raises(DomainError):
        execute(
            db,
            "another-owner",
            str(uuid4()),
            "calendar.select",
            {"calendar_id": source, "expected_revision": 2, "selected": True},
        )


def test_recurring_dst_cancelled_and_moved_instances(client):
    source = linked()
    event(
        source,
        "series",
        start={"dateTime": "2026-03-06T09:00:00-06:00", "timeZone": "America/Chicago"},
        end={"dateTime": "2026-03-06T10:00:00-06:00", "timeZone": "America/Chicago"},
        recurrence=["RRULE:FREQ=DAILY;COUNT=5"],
    )
    event(
        source,
        "cancel",
        status="cancelled",
        recurringEventId="series",
        originalStartTime={"dateTime": "2026-03-07T09:00:00-06:00"},
    )
    event(
        source,
        "moved",
        summary="Moved planning",
        recurringEventId="series",
        originalStartTime={"dateTime": "2026-03-08T09:00:00-05:00"},
        start={"dateTime": "2026-03-09T13:00:00-05:00"},
        end={"dateTime": "2026-03-09T14:00:00-05:00"},
    )
    with session_scope() as db:
        rows, truncated = project(db, "davin", date(2026, 3, 6), date(2026, 3, 11), "America/Chicago")
        assert not truncated and len(rows) == 4
        assert not any(r["date"] in {"2026-03-07", "2026-03-08"} for r in rows)
        assert any(r["at"].endswith("-05:00") and "T09:00" in r["at"] for r in rows)
        assert len([r for r in rows if r["date"] == "2026-03-09"]) == 2


def test_all_day_multiday_exclusions_transparency_and_unsafe_links(client):
    source = linked()
    event(
        source,
        "days",
        start={"date": "2026-09-18"},
        end={"date": "2026-09-20"},
        recurrence=["RRULE:FREQ=WEEKLY;COUNT=2", "EXDATE;VALUE=DATE:20260925"],
        htmlLink="javascript:alert(1)",
        transparency="transparent",
    )
    with session_scope() as db:
        rows, truncated = project(db, "davin", date(2026, 9, 18), date(2026, 9, 28), "America/Chicago")
        assert not truncated and [r["date"] for r in rows] == ["2026-09-18", "2026-09-19"]
        assert all(r["all_day"] and not r["busy"] and r["url"] is None for r in rows)


def test_cancelled_series_suppresses_old_exception_and_deadline_conflicts(client):
    source = linked()
    event(source, "series", status="cancelled")
    event(
        source,
        "override",
        recurringEventId="series",
        originalStartTime={"dateTime": "2026-09-18T09:00:00-05:00"},
    )
    with session_scope() as db:
        assert project(db, "davin", date(2026, 9, 18), date(2026, 9, 19), "America/Chicago")[0] == []
    event(source, "meeting")
    with session_scope() as db:
        execute(
            db,
            "davin",
            str(uuid4()),
            "task.create",
            {"title": "Deadline", "due_date": "2026-09-18", "due_time": "09:30"},
        )
    response = client.get("/api/v1/calendar?start=2026-09-18&end=2026-09-19")
    assert response.status_code == 200
    items = response.json()["items"]
    assert next(r for r in items if r["kind"] == "task")["conflicts"] == ["Planning"]


def test_fresh_availability_merges_busy_intervals_and_never_guesses_on_partial_error(client, monkeypatch):
    linked()
    response = {
        "calendars": {
            "owner@example.test": {
                "busy": [
                    {"start": "2026-09-18T10:00:00-05:00", "end": "2026-09-18T11:00:00-05:00"},
                    {"start": "2026-09-18T10:30:00-05:00", "end": "2026-09-18T12:00:00-05:00"},
                ]
            }
        }
    }
    monkeypatch.setattr(calendar, "CalendarClient", lambda _: Provider([response]))
    result = calendar.availability("davin", "2026-09-18T09:00:00-05:00", "2026-09-18T13:00:00-05:00")
    assert result["status"] == "fresh" and len(result["busy"]) == 1 and len(result["free"]) == 2
    monkeypatch.setattr(
        calendar,
        "CalendarClient",
        lambda _: Provider([{"calendars": {"owner@example.test": {"errors": [{"reason": "notFound"}]}}}]),
    )
    result = calendar.availability("davin", "2026-09-18T09:00:00-05:00", "2026-09-18T13:00:00-05:00")
    assert result["status"] == "unavailable" and result["free"] == []


def test_availability_rechecks_connection_and_validates_timezones(client, monkeypatch):
    linked()

    class Changed(Provider):
        def request(self, *args, **kwargs):
            with session_scope() as db:
                db.get(GoogleIdentity, "davin").generation += 1
            return {"calendars": {"owner@example.test": {"busy": []}}}

    monkeypatch.setattr(calendar, "CalendarClient", lambda _: Changed([]))
    result = calendar.availability("davin", "2026-09-18T09:00:00-05:00", "2026-09-18T13:00:00-05:00")
    assert result["status"] == "unavailable"
    with pytest.raises(DomainError):
        calendar.availability("davin", "2026-09-18T09:00:00", "2026-09-18T13:00:00")
    assert (
        client.post(
            "/api/v1/calendar/availability/day",
            json={
                "date": "2026-09-18",
                "start_time": "09:00",
                "end_time": "13:00",
                "timezone": "Wrong/Zone",
                "minutes": 30,
            },
        ).status_code
        == 400
    )


def test_disconnect_clears_calendar_data_but_keeps_identity(client, monkeypatch):
    source = linked()
    event(source)
    monkeypatch.setattr(auth.httpx, "post", lambda *a, **kw: SimpleNamespace(status_code=200))
    assert auth.disconnect_calendar("davin") == {"disconnected": True, "revoked": True}
    with session_scope() as db:
        row = db.get(GoogleIdentity, "davin")
        assert row.subject == "google-subject" and not row.calendar_enabled and row.credentials is None
        assert list(db.scalars(select(GoogleCalendar))) == []
        assert list(db.scalars(select(GoogleCalendarEvent))) == []


def test_unlink_removes_google_sessions_and_inflight_consent_cannot_rebind(client, monkeypatch):
    source = linked()
    event(source)
    pair_cookie = client.cookies.get("jarvis_session")
    google_cookie, _ = auth.new_session("davin", "google")
    state, browser, nonce, _ = start(client, "calendar")

    def revoke_during_exchange(*_):
        auth.unlink_google("davin")
        return {"id_token": "fixture", "refresh_token": "fixture", "scope": auth.CALENDAR_SCOPE}

    successful_exchange(monkeypatch, nonce)
    monkeypatch.setattr(auth, "exchange", revoke_during_exchange)
    monkeypatch.setattr(auth.httpx, "post", lambda *a, **kw: SimpleNamespace(status_code=200))
    with pytest.raises(DomainError, match="changed"):
        auth.finish(state, browser, "code")
    with session_scope() as db:
        assert db.get(AuthSession, auth.digest(pair_cookie)) is not None
        assert db.get(AuthSession, auth.digest(google_cookie)) is None
        assert db.get(GoogleIdentity, "davin") is None
        assert list(db.scalars(select(GoogleCalendarEvent))) == []


def test_calendar_disconnect_generation_invalidates_pending_consent(client, monkeypatch):
    linked()
    state, browser, nonce, _ = start(client, "calendar")
    successful_exchange(monkeypatch, nonce, scope=auth.CALENDAR_SCOPE, refresh_token="fixture")
    monkeypatch.setattr(auth.httpx, "post", lambda *a, **kw: SimpleNamespace(status_code=200))
    auth.disconnect_calendar("davin")
    with pytest.raises(DomainError, match="changed"):
        auth.finish(state, browser, "code")


def test_revoked_refresh_grant_needs_reconnect_and_poll_does_not_loop(client, monkeypatch):
    linked()
    monkeypatch.setattr(
        calendar, "CalendarClient", lambda _: (_ for _ in ()).throw(calendar.SyncFailure("reconnect", 401))
    )
    with session_scope() as db:
        jid = calendar.queue_sync(db, "davin", force=True)
    calendar.process(jid)
    with session_scope() as db:
        assert db.get(GoogleIdentity, "davin").status == "needs_reconnect"
        assert calendar.queue_sync(db, "davin", force=True) is None


def test_callback_does_not_return_codes_or_tokens_and_access_log_redacts_query(client):
    import logging

    from jarvis.google_routes import OAuthLogFilter

    response = client.get(
        "/api/v1/auth/google/callback?state=invalid&code=private-code", follow_redirects=False
    )
    assert response.status_code == 303 and response.headers["location"] == "/?view=settings&google=failed"
    record = logging.LogRecord(
        "uvicorn.access",
        logging.INFO,
        "",
        1,
        "%s %s %s %s %s",
        ("127.0.0.1", "GET", "/api/v1/auth/google/callback?code=private-code", "1.1", 303),
        None,
    )
    assert OAuthLogFilter().filter(record)
    assert "private-code" not in record.getMessage()
    assert "private-code" not in response.text


@pytest.mark.asyncio
async def test_eri_calendar_tools_use_owner_scope_and_current_availability(client, monkeypatch):
    from jarvis.tools import call_tool, registry

    source = linked()
    eid = event(source)
    status = await call_tool("davin", str(uuid4()), 0, "calendar_connection", {})
    assert status["calendars"][0]["id"] == source and "credentials" not in status
    assert "calendar_select" in {t["name"] for t in registry()}
    assert client.get("/api/v1/calendar/events/" + eid).status_code == 200
    with session_scope() as db, pytest.raises(DomainError):
        calendar.event_detail(db, "other-owner", eid)
    monkeypatch.setattr(
        calendar, "CalendarClient", lambda _: Provider([{"calendars": {"owner@example.test": {"busy": []}}}])
    )
    result = await call_tool(
        "davin",
        str(uuid4()),
        0,
        "calendar_availability",
        {"start": "2026-09-18T09:00:00-05:00", "end": "2026-09-18T12:00:00-05:00", "minutes": 60},
    )
    assert result["status"] == "fresh" and len(result["free"]) == 1


def test_dense_unsupported_recurrence_is_flagged_instead_of_silently_claiming_complete(client):
    source = linked()
    event(source, "dense", recurrence=["RRULE:FREQ=MINUTELY"])
    with session_scope() as db:
        rows, truncated = project(db, "davin", date(2026, 9, 18), date(2026, 9, 19), "America/Chicago")
        assert truncated and rows == []


def test_reconnect_replaces_unreadable_old_credentials(client, monkeypatch):
    linked()
    with session_scope() as db:
        db.get(GoogleIdentity, "davin").credentials = "unreadable-old-ciphertext"
    state, browser, nonce, _ = start(client, "calendar")
    successful_exchange(monkeypatch, nonce, scope=auth.CALENDAR_SCOPE, refresh_token="replacement-refresh")
    auth.finish(state, browser, "code")
    with session_scope() as db:
        assert (
            auth.unseal(db.get(GoogleIdentity, "davin").credentials)["refresh_token"] == "replacement-refresh"
        )


def test_timed_out_sync_allows_recovery_and_invalidates_late_result(client):
    linked()
    with session_scope() as db:
        first = calendar.queue_sync(db, "davin", force=True)
        db.get(Job, first).created_at = now() - timedelta(minutes=16)
        old_generation = db.get(GoogleIdentity, "davin").generation
    with session_scope() as db:
        second = calendar.queue_sync(db, "davin", force=True)
        assert second != first
        assert db.get(Job, first).status == "failed"
        assert db.get(GoogleIdentity, "davin").generation == old_generation + 1


def test_oauth_link_and_session_issuance_commit_atomically(client, monkeypatch):
    state, browser, nonce, _ = start(client)
    successful_exchange(monkeypatch, nonce)

    def failed_session(owner, method, *, db=None):
        assert db is not None
        assert db.get(GoogleIdentity, owner).subject == "google-subject"
        raise RuntimeError("Session storage failed")

    monkeypatch.setattr(auth, "new_session", failed_session)
    with pytest.raises(RuntimeError, match="Session storage failed"):
        auth.finish(state, browser, "code")
    with session_scope() as db:
        assert db.get(GoogleIdentity, "davin") is None
        assert not db.scalars(select(AuthSession).where(AuthSession.auth_method == "google")).all()
