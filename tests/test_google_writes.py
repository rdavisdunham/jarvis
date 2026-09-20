import copy
from datetime import timedelta
from urllib.parse import parse_qs, urlsplit
from uuid import uuid4

import pytest
from cryptography.fernet import Fernet
from jarvis import google_auth, google_calendar
from jarvis.config import get_settings
from jarvis.db import session_scope
from jarvis.domain import DomainError, execute
from jarvis.google_schema import CalendarCreate, CalendarRead
from jarvis.google_writes import event_body, process_write, read_event, write_status
from jarvis.models import GoogleCalendar, GoogleCalendarEvent, GoogleIdentity, GoogleOAuthAttempt, Job, now
from sqlalchemy import select


class Provider:
    def __init__(self):
        self.events = {}
        self.calls = []
        self.role = "owner"
        self.fail = ""
        self.version = 1
        self.originals = []

    def request(self, path, params=None, body=None, *, method=None, headers=None):
        method = method or ("POST" if body is not None else "GET")
        self.calls.append((method, path, copy.deepcopy(body), headers))
        if path.startswith("users/me/calendarList/"):
            return {"accessRole": self.role}
        if path.endswith("/instances"):
            return {"items": [copy.deepcopy(e) for e in self.originals]}
        event_id = path.rsplit("/", 1)[-1]
        if method == "GET":
            if self.fail == "read":
                raise google_calendar.SyncFailure("unavailable")
            if event_id not in self.events:
                raise google_calendar.SyncFailure("missing", 404)
            return copy.deepcopy(self.events[event_id])
        if method == "POST":
            event_id = body["id"]
            if event_id in self.events:
                raise google_calendar.SyncFailure("duplicate", 409)
            if self.fail == "before_create":
                raise google_calendar.SyncFailure("unavailable")
            self.events[event_id] = copy.deepcopy(body)
        else:
            if self.fail == "conflict" or headers.get("If-Match") != self.events[event_id]["etag"]:
                raise google_calendar.SyncFailure("conflict", 412)
            if method == "DELETE":
                del self.events[event_id]
                if self.fail == "after_delete":
                    self.fail = ""
                    raise google_calendar.SyncFailure("unavailable")
                return {}
            # Google PATCH merges nested objects and removes explicit null fields.
            def merge(current, patch):
                result = copy.deepcopy(current)
                for key, value in patch.items():
                    if value is None:
                        result.pop(key, None)
                    elif isinstance(value, dict):
                        result[key] = merge(result.get(key, {}), value)
                    else:
                        result[key] = copy.deepcopy(value)
                return result
            saved = merge(self.events[event_id], body)
            kinds = [{key for key in ("date", "dateTime") if point.get(key)} for point in (saved["start"], saved["end"])]
            if kinds not in [[{"date"}, {"date"}], [{"dateTime"}, {"dateTime"}]]:
                raise google_calendar.SyncFailure("invalid_event_time", 400)
            self.events[event_id] = saved
        self.version += 1
        self.events[event_id]["etag"] = '"' + str(self.version) + '"'
        self.events[event_id].setdefault("status", "confirmed")
        self.events[event_id]["id"] = event_id
        if self.fail in {"after_create", "after_update"}:
            self.fail = ""
            raise google_calendar.SyncFailure("unavailable")
        return copy.deepcopy(self.events[event_id])

    def close(self):
        pass


@pytest.fixture
def provider(monkeypatch):
    settings = get_settings()
    monkeypatch.setattr(settings, "origin", "http://localhost")
    monkeypatch.setattr(settings, "google_client_id", "fixture.apps.googleusercontent.com")
    monkeypatch.setattr(settings, "google_client_secret", "fixture-secret")
    monkeypatch.setattr(settings, "integration_encryption_key", Fernet.generate_key().decode())
    fake = Provider()
    monkeypatch.setattr(google_calendar, "CalendarClient", lambda *_: fake)
    with session_scope() as db:
        account = GoogleIdentity(
            owner_id="davin",
            subject="owner-subject",
            email="owner@example.test",
            credentials=google_auth.seal({"refresh_token": "fixture-refresh"}),
            calendar_enabled=True,
            calendar_write_enabled=True,
            status="ready",
            next_sync_at=now(),
        )
        db.add(account)
        db.flush()
        source = GoogleCalendar(
            owner_id="davin",
            provider_id="personal",
            title="Personal",
            timezone="America/Chicago",
            selected=True,
            available=True,
            access_role="owner",
            primary=True,
        )
        db.add(source)
        db.flush()
        fake.source = source.id
    return fake


def command(tool, args, command_id=None):
    with session_scope() as db:
        return execute(db, "davin", command_id or str(uuid4()), tool, args)["data"]


def fields(**changes):
    return {
        "title": "Planning",
        "start": "2026-09-18T09:00",
        "end": "2026-09-18T10:00",
        "timezone": "America/Chicago",
        "location": "Desk",
        **changes,
    }


def existing(provider, **changes):
    remote = {
        "id": "existing",
        "summary": "Planning",
        "etag": '"1"',
        "status": "confirmed",
        "start": {"dateTime": "2026-09-18T09:00:00-05:00", "timeZone": "America/Chicago"},
        "end": {"dateTime": "2026-09-18T10:00:00-05:00", "timeZone": "America/Chicago"},
        **changes,
    }
    provider.events[remote["id"]] = remote
    with session_scope() as db:
        row = GoogleCalendarEvent(
            calendar_id=provider.source, provider_id=remote["id"], payload=google_calendar.clean_event(remote)
        )
        db.add(row)
        db.flush()
        return row.id


def preview(provider, **changes):
    eid = existing(provider, **changes)
    return read_event("davin", CalendarRead(event_id=eid))


def status(job):
    with session_scope() as db:
        return write_status(db, "davin", job["job_id"])


def test_create_receipt_and_worker_replay_create_exactly_one_event(provider):
    args = {**fields(), "calendar_id": provider.source}
    cid = str(uuid4())
    job = command("calendar.create", args, cid)
    assert command("calendar.create", args, cid) == job
    process_write(job["job_id"])
    process_write(job["job_id"])
    assert status(job)["status"] == "succeeded"
    assert len(provider.events) == 1
    assert sum(c[0] == "POST" for c in provider.calls) == 1
    with session_scope() as db:
        assert len(list(db.scalars(select(Job)))) == 1
        saved = db.scalar(select(GoogleCalendarEvent))
        assert saved.payload["summary"] == "Planning"


@pytest.mark.parametrize(
    "operation,fail", [("create", "after_create"), ("update", "after_update"), ("delete", "after_delete")]
)
def test_response_loss_recovers_from_google_without_repeating_write(provider, operation, fail):
    args = (
        {**fields(), "calendar_id": provider.source}
        if operation == "create"
        else {"edit_token": preview(provider)["edit_token"]}
    )
    if operation == "update":
        args.update(fields(title="Updated"))
    job = command("calendar." + operation, args)
    provider.fail = fail
    with pytest.raises(RuntimeError, match="retry"):
        process_write(job["job_id"])
    assert status(job)["status"] == "retrying"
    process_write(job["job_id"])
    assert status(job)["status"] == "succeeded"
    assert sum(c[0] in {"POST", "PATCH", "DELETE"} for c in provider.calls) == 1
    assert len(provider.events) == (0 if operation == "delete" else 1)


def test_google_etag_conflict_does_not_overwrite_newer_event(provider):
    token = preview(provider)["edit_token"]
    provider.events["existing"].update(etag='"other"', summary="Edited elsewhere")
    job = command("calendar.update", {**fields(title="My edit"), "edit_token": token})
    process_write(job["job_id"])
    assert status(job)["status"] == "failed"
    assert "changed" in status(job)["result"]["message"]
    assert provider.events["existing"]["summary"] == "Edited elsewhere"
    assert not any(c[0] == "PATCH" for c in provider.calls)


def test_google_if_match_catches_change_between_read_and_patch(provider):
    token = preview(provider)["edit_token"]
    provider.fail = "conflict"
    job = command("calendar.update", {**fields(), "edit_token": token})
    process_write(job["job_id"])
    assert status(job)["status"] == "failed"
    assert next(c for c in provider.calls if c[0] == "PATCH")[3] == {"If-Match": '"1"'}


def test_permission_is_checked_in_google_before_write(provider):
    job = command("calendar.create", {**fields(), "calendar_id": provider.source})
    provider.role = "reader"
    process_write(job["job_id"])
    assert status(job)["status"] == "failed" and not provider.events


@pytest.mark.parametrize("change", ["scope", "role", "selection"])
def test_local_read_only_or_unselected_calendars_cannot_queue(provider, change):
    with session_scope() as db:
        if change == "scope":
            db.get(GoogleIdentity, "davin").calendar_write_enabled = False
        elif change == "role":
            db.get(GoogleCalendar, provider.source).access_role = "reader"
        else:
            db.get(GoogleCalendar, provider.source).selected = False
    with pytest.raises(DomainError):
        command("calendar.create", {**fields(), "calendar_id": provider.source})


def test_connection_generation_change_cancels_queued_write(provider):
    job = command("calendar.create", {**fields(), "calendar_id": provider.source})
    with session_scope() as db:
        db.get(GoogleIdentity, "davin").generation += 1
    process_write(job["job_id"])
    assert status(job)["status"] == "cancelled" and provider.calls == []


def test_edit_token_is_owner_bound_and_expires(provider):
    token = preview(provider)["edit_token"]
    decoded = google_auth.unseal(token)
    for change in [{"owner": "other"}, {"expires_at": (now() - timedelta(seconds=1)).isoformat()}]:
        altered = google_auth.seal({**decoded, **change})
        with pytest.raises(DomainError):
            command("calendar.delete", {"edit_token": altered})
    assert not any(c[0] == "DELETE" for c in provider.calls)


def test_all_day_exclusive_end_and_repeat_rules(provider):
    data = event_body(
        CalendarCreate(
            calendar_id=provider.source,
            **fields(start="2026-09-18", end="2026-09-20", all_day=True),
            repeat="weekly",
        )
    )
    assert data["start"] == {"date": "2026-09-18"} and data["end"] == {"date": "2026-09-20"}
    assert data["recurrence"] == ["RRULE:FREQ=WEEKLY"]
    assert "attendees" not in data


@pytest.mark.parametrize(
    "changes",
    [
        {"start": "2026-09-18T10:00", "end": "2026-09-18T09:00"},
        {"start": "2026-03-08T02:30", "end": "2026-03-08T04:00"},
        {"start": "2026-09-18T09:00+00:00", "end": "2026-09-18T10:00+00:00"},
        {"all_day": True, "start": "2026-09-18", "end": "2026-09-18"},
    ],
)
def test_invalid_or_dst_inconsistent_times_are_rejected(provider, changes):
    with pytest.raises(DomainError):
        command("calendar.create", {**fields(**changes), "calendar_id": provider.source})


def test_guest_events_are_readable_but_not_mutated(provider):
    detail = preview(provider, attendees=[{"email": "guest@example.test"}])
    assert not detail["editable"] and detail["edit_token"] is None
    assert "guests" in detail["read_only_reason"]


def test_guest_added_after_preview_blocks_write(provider):
    detail = preview(provider)
    provider.events["existing"]["attendees"] = [{"email": "guest@example.test"}]
    job = command("calendar.delete", {"edit_token": detail["edit_token"]})
    process_write(job["job_id"])
    assert status(job)["status"] == "failed" and "existing" in provider.events


def test_recurring_occurrence_and_series_have_separate_targets(provider):
    eid = existing(provider, recurrence=["RRULE:FREQ=WEEKLY"])
    child = {
        **provider.events["existing"],
        "id": "existing_occurrence",
        "recurringEventId": "existing",
        "originalStartTime": {"dateTime": "2026-09-18T09:00:00-05:00"},
        "etag": '"instance"',
    }
    child.pop("recurrence")
    provider.events[child["id"]] = child
    provider.originals = [child]
    with pytest.raises(DomainError, match="occurrence"):
        read_event("davin", CalendarRead(event_id=eid))
    single = read_event(
        "davin", CalendarRead(event_id=eid, scope="occurrence", occurrence_start="2026-09-18T09:00:00-05:00")
    )
    series = read_event("davin", CalendarRead(event_id=eid, scope="series"))
    assert google_auth.unseal(single["edit_token"])["provider_event"] == child["id"]
    assert google_auth.unseal(series["edit_token"])["provider_event"] == "existing"
    job = command("calendar.delete", {"edit_token": single["edit_token"]})
    process_write(job["job_id"])
    assert "existing" in provider.events and child["id"] not in provider.events


def test_all_day_recurring_occurrence_resolves_by_original_date(provider):
    eid = existing(
        provider, start={"date": "2026-09-18"}, end={"date": "2026-09-19"}, recurrence=["RRULE:FREQ=DAILY"]
    )
    child = {
        **provider.events["existing"],
        "id": "all_day_instance",
        "recurringEventId": "existing",
        "originalStartTime": {"date": "2026-09-18"},
    }
    child.pop("recurrence")
    provider.events[child["id"]] = child
    provider.originals = [child]
    result = read_event(
        "davin", CalendarRead(event_id=eid, scope="occurrence", occurrence_start="2026-09-18")
    )
    assert result["all_day"] and result["start"] == "2026-09-18"


def test_repeated_unknown_provider_outcome_remains_unconfirmed(provider):
    job = command("calendar.create", {**fields(), "calendar_id": provider.source})
    provider.fail = "before_create"
    for _ in range(4):
        with pytest.raises(RuntimeError):
            process_write(job["job_id"])
    process_write(job["job_id"])
    assert status(job)["status"] == "unconfirmed"
    ids = [c[2]["id"] for c in provider.calls if c[0] == "POST"]
    assert len(ids) == 5 and len(set(ids)) == 1


def test_write_result_and_preview_enforce_owner(provider):
    eid = existing(provider)
    with pytest.raises(DomainError):
        read_event("other", CalendarRead(event_id=eid))
    job = command("calendar.create", {**fields(), "calendar_id": provider.source})
    with session_scope() as db, pytest.raises(DomainError):
        write_status(db, "other", job["job_id"])


def test_write_scope_is_separate_and_partial_grants_preserve_read_access(client, provider, monkeypatch):
    with session_scope() as db:
        db.get(GoogleIdentity, "davin").calendar_write_enabled = False
    response = client.post("/api/v1/auth/google/start", json={"purpose": "calendar_write"})
    params = parse_qs(urlsplit(response.json()["url"]).query)
    assert google_auth.WRITE_SCOPE in params["scope"][0] and google_auth.CALENDAR_SCOPE in params["scope"][0]
    state = params["state"][0]
    with session_scope() as db:
        nonce = db.get(GoogleOAuthAttempt, google_auth.digest(state)).nonce
    monkeypatch.setattr(
        google_auth,
        "exchange",
        lambda *_: {"id_token": "fixture", "refresh_token": "fixture", "scope": google_auth.CALENDAR_SCOPE},
    )
    monkeypatch.setattr(
        google_auth,
        "verify_identity",
        lambda _: {
            "sub": "owner-subject",
            "email": "owner@example.test",
            "email_verified": True,
            "nonce": nonce,
        },
    )
    with pytest.raises(DomainError, match="editing permission"):
        google_auth.finish(state, client.cookies.get(google_auth.COOKIE), "code")
    with session_scope() as db:
        account = db.get(GoogleIdentity, "davin")
        assert account.calendar_enabled and not account.calendar_write_enabled


def test_write_cannot_be_mistaken_for_synchronous_confirmation(client, provider):
    response = client.post(
        "/api/v1/commands",
        json={
            "command_id": str(uuid4()),
            "tool": "calendar.create",
            "arguments": {**fields(), "calendar_id": provider.source},
        },
    )
    job = response.json()["data"]
    assert job["status"] == "queued" and provider.events == {}
    assert client.get("/api/v1/calendar/writes/" + job["job_id"]).json()["status"] == "queued"
    assert client.get("/api/v1/calendar/writes").json()["items"][0]["job_id"] == job["job_id"]


def test_write_race_invalidates_prewrite_sync_snapshot(provider, monkeypatch):
    eid = existing(provider)
    with session_scope() as db:
        source = db.get(GoogleCalendar, provider.source)
        source.sync_token = "old"
        generation = db.get(GoogleIdentity, "davin").generation
        from jarvis.domain import enqueue_job

        job_id = enqueue_job(db, "davin", "google_sync", {"generation": generation}).id

    def request(path, params=None, body=None):
        if path == "users/me/calendarList":
            return {
                "items": [{"id": "personal", "primary": True, "summary": "Personal", "accessRole": "owner"}]
            }
        with session_scope() as db:
            db.get(GoogleCalendar, provider.source).revision += 1
            db.get(GoogleCalendarEvent, eid).payload = {
                **provider.events["existing"],
                "summary": "Fresh local write",
            }
        return {"items": [provider.events["existing"]], "nextSyncToken": "stale"}

    monkeypatch.setattr(provider, "request", request)
    google_calendar.process(job_id)
    with session_scope() as db:
        assert db.get(GoogleCalendarEvent, eid).payload["summary"] == "Fresh local write"
        assert db.get(GoogleCalendar, provider.source).sync_token == "old"


def test_lost_response_then_lost_access_does_not_report_definite_failure(provider):
    job = command("calendar.create", {**fields(), "calendar_id": provider.source})
    provider.fail = "after_create"
    with pytest.raises(RuntimeError):
        process_write(job["job_id"])
    provider.role = "reader"
    process_write(job["job_id"])
    assert status(job)["status"] == "unconfirmed"
    assert len(provider.events) == 1
    assert sum(call[0] == "POST" for call in provider.calls) == 1


@pytest.mark.parametrize("to_all_day", [False, True])
@pytest.mark.parametrize("lose_response", [False, True])
def test_convert_event_time_format_and_preserve_remote_metadata(provider, to_all_day, lose_response):
    current = {} if to_all_day else {
        "start": {"date": "2026-09-18"}, "end": {"date": "2026-09-20"},
    }
    token = preview(provider, **current, extendedProperties={"private": {"otherApp": "keep"}},
                    colorId="3")["edit_token"]
    replacement = fields(all_day=to_all_day, description="Keep the agenda")
    if to_all_day:
        replacement.update(start="2026-09-21", end="2026-09-24")
    job = command("calendar.update", {**replacement, "edit_token": token})
    if lose_response:
        provider.fail = "after_update"
        with pytest.raises(RuntimeError, match="retry"):
            process_write(job["job_id"])
    process_write(job["job_id"])
    assert status(job)["status"] == "succeeded"
    saved = provider.events["existing"]
    for key in ("start", "end"):
        assert set(saved[key]) == ({"date"} if to_all_day else {"dateTime", "timeZone"})
    assert saved["start"] == ({"date": "2026-09-21"} if to_all_day else
                             {"dateTime": "2026-09-18T14:00:00+00:00", "timeZone": "America/Chicago"})
    assert saved["end"] == ({"date": "2026-09-24"} if to_all_day else
                           {"dateTime": "2026-09-18T15:00:00+00:00", "timeZone": "America/Chicago"})
    assert saved["description"] == "Keep the agenda" and saved["location"] == "Desk"
    assert saved["colorId"] == "3" and saved["extendedProperties"]["private"]["otherApp"] == "keep"
    assert sum(c[0] == "PATCH" for c in provider.calls) == 1
    with session_scope() as db:
        cached = db.scalar(select(GoogleCalendarEvent))
        assert cached.payload["start"] == saved["start"] and cached.payload["end"] == saved["end"]


def test_reschedule_all_day_event_keeps_exclusive_multiday_end(provider):
    token = preview(provider, start={"date": "2026-09-18"}, end={"date": "2026-09-19"})["edit_token"]
    job = command("calendar.update", {**fields(all_day=True, start="2026-09-21", end="2026-09-24"),
                                      "edit_token": token})
    process_write(job["job_id"])
    assert status(job)["status"] == "succeeded"
    assert provider.events["existing"]["start"] == {"date": "2026-09-21"}
    assert provider.events["existing"]["end"] == {"date": "2026-09-24"}
