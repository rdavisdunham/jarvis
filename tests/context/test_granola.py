"""Optional Granola REST adapter: client, limiter, mapping and poller against recorded synthetic responses.

No network: every request goes through an httpx MockTransport serving `fixtures/granola/`. `sources.receive`
(package B1) is replaced by a recorder.
"""

import json
import sys
import types
from datetime import UTC, date, datetime
from pathlib import Path

import httpx
import jarvis.context as context_pkg
import pytest
from cryptography.fernet import Fernet
from jarvis.config import get_settings
from jarvis.context import models as m
from jarvis.context.connectors import granola
from jarvis.context.schema import content_hash
from jarvis.db import session_scope
from sqlalchemy import select

FIXTURES = Path(__file__).parent / "fixtures" / "granola"
OWNER = "davin"
ACCOUNT = "alex.morgan@fernhill.example"
NOTE, FOREIGN, NO_TRANSCRIPT, DELETED = ("not_A1b2C3d4E5f6G7", "not_H8i9J0k1L2m3N4", "not_P5q6R7s8T9u0V1",
                                         "not_W2x3Y4z5A6b7C8")
NOW = datetime(2026, 11, 4, 12, 0, tzinfo=UTC)


def load(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text())


class FakeGranola:
    """Serves the fixtures. `failures` maps a path to a list of responses (status or exception) served first."""

    def __init__(self):
        self.requests: list[httpx.Request] = []
        self.failures: dict[str, list] = {}

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        path, params = request.url.path, request.url.params
        queued = self.failures.get(path) or self.failures.get("*")
        if queued:
            failure = queued.pop(0)
            if isinstance(failure, Exception):
                raise failure
            status, headers = failure if isinstance(failure, tuple) else (failure, {})
            return httpx.Response(status, headers=headers, json={"message": "nope"})
        if path == "/v1/notes":
            return httpx.Response(200, json=load("notes_page2.json" if params.get("cursor") else "notes_page1.json"))
        parts = path.strip("/").split("/")
        if len(parts) == 3:
            name = FIXTURES / f"note_{parts[2]}.json"
            return httpx.Response(200, json=json.loads(name.read_text())) if name.exists() else httpx.Response(404)
        if len(parts) == 4 and parts[3] == "transcript":
            page = 2 if params.get("cursor") else 1
            return httpx.Response(200, json=load(f"transcript_{parts[2]}_{page}.json"))
        return httpx.Response(404)

    @property
    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self)

    def paths(self) -> list[str]:
        return [r.url.path for r in self.requests]


def note_envelope():
    return granola.to_envelope(load(f"note_{NOTE}.json"),
                               load(f"transcript_{NOTE}_1.json")["transcript"]
                               + load(f"transcript_{NOTE}_2.json")["transcript"])


# ---------------------------------------------------------------- mapping


def test_mapping_attribution_offsets_and_metadata():
    envelope = note_envelope()
    assert envelope.connector == "granola" and envelope.external_id == NOTE
    labels = [(s.speaker_label, s.attribution) for s in envelope.segments]
    assert labels == [("Alex Morgan", "owner"), ("Priya Natarajan", "other"), ("Speaker B", "other"),
                      (None, None), ("Alex Morgan", "owner")]  # blank item dropped; unknown speaker unnamed
    assert [s.seq for s in envelope.segments] == [0, 1, 2, 3, 4]
    assert [(s.start_ms, s.end_ms) for s in envelope.segments][:2] == [(0, 4750), (6250, 14500)]
    assert envelope.occurred_at == datetime(2026, 11, 3, 15, 1, 4, 250000, tzinfo=UTC)
    assert envelope.ended_at == datetime(2026, 11, 3, 15, 1, 36, tzinfo=UTC)
    assert envelope.private_notes == "- Check whether Northwind's SSO needs **SCIM**."
    assert envelope.summary.startswith("### Summary")
    assert envelope.calendar_event_id == "5q2vlk0e8m1c3b7d9f4a6h2j0k_20261103T150000Z"
    assert envelope.external_revision == "2026-11-03T16:02:40Z"
    assert envelope.content_hash == content_hash([s.text for s in envelope.segments] + [envelope.private_notes])
    assert (envelope.upstream_owner_email, envelope.upstream_owner_name) == (ACCOUNT, "Alex Morgan")
    assert envelope.metadata["granola"]["scheduled_start_time"] == "2026-11-03T15:00:00Z"
    people = {(p.email or p.speaker_label): p for p in envelope.participants}
    assert people["priya.natarajan@northwind.example"].role == "organizer"
    assert people["priya.natarajan@northwind.example"].name == "Priya Natarajan"
    assert people["tomas.reyes@northwind.example"].name is None  # invitee only: no guessed name
    assert people[ACCOUNT].speaker_label == "Alex Morgan"
    assert people["Speaker B"].role == "speaker" and people["Speaker B"].email is None


def test_mapping_is_deterministic():
    assert note_envelope().model_dump() == note_envelope().model_dump()


def test_me_is_the_note_owner_not_the_importer():
    note = load(f"note_{NOTE}.json") | {"owner": {"name": "Sam Okafor", "email": "sam@elsewhere.example"}}
    envelope = granola.to_envelope(note, load(f"transcript_{NOTE}_1.json")["transcript"])
    assert envelope.segments[0].speaker_label == "Sam Okafor" and envelope.segments[0].attribution == "owner"
    assert all(s.speaker_label != OWNER for s in envelope.segments)


def test_summary_edit_keeps_content_hash():
    note = load(f"note_{NOTE}.json")
    items = load(f"transcript_{NOTE}_1.json")["transcript"]
    edited = note | {"summary_markdown": "Rewritten summary", "updated_at": "2026-11-04T08:00:00Z"}
    first, second = granola.to_envelope(note, items), granola.to_envelope(edited, items)
    assert first.content_hash == second.content_hash and first.external_revision != second.external_revision


def test_note_without_transcript_is_not_an_envelope():
    with pytest.raises(ValueError):
        granola.to_envelope(load(f"note_{NO_TRANSCRIPT}.json"), [])


# ---------------------------------------------------------------- client


class Clock:
    def __init__(self):
        self.now, self.slept = 0.0, []

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.now += seconds


def test_token_bucket_holds_five_per_second():
    clock = Clock()
    bucket = granola.TokenBucket(clock=clock, sleep=clock.sleep)
    for _ in range(11):
        bucket.acquire()
    assert clock.now == pytest.approx(1.2)  # 5 burst, then 6 more at 5/s


def client(fake: FakeGranola, clock: Clock | None = None) -> granola.GranolaClient:
    clock = clock or Clock()
    return granola.GranolaClient("gk_test", transport=fake.transport,
                                 limiter=granola.TokenBucket(clock=clock, sleep=clock.sleep), sleep=clock.sleep)


def test_client_pages_notes_and_transcripts():
    fake = FakeGranola()
    with client(fake) as api:
        notes = list(api.iter_notes(date(2026, 11, 3)))
        items = api.get_transcript(NOTE)
    assert [n["id"] for n in notes] == [NOTE, FOREIGN, NO_TRANSCRIPT, DELETED]
    assert len(items) == 6
    lists = [r for r in fake.requests if r.url.path == "/v1/notes"]
    assert [dict(r.url.params) for r in lists] == [
        {"updated_after": "2026-11-03", "page_size": "30"},
        {"updated_after": "2026-11-03", "page_size": "30", "cursor": "cur_f3k9QpZ"}]
    transcripts = [r for r in fake.requests if r.url.path.endswith("/transcript")]
    assert [r.url.params.get("cursor") for r in transcripts] == [None, "tc_8sd7Fh2"]
    assert all(r.url.params["page_size"] == "100" for r in transcripts)
    assert all("include" not in r.url.params for r in fake.requests)
    assert fake.requests[0].headers["Authorization"] == "Bearer gk_test"


def test_client_backs_off_on_429_and_5xx():
    fake, clock = FakeGranola(), Clock()
    fake.failures["/v1/notes"] = [(429, {"Retry-After": "3"}), 503, httpx.ReadTimeout("slow")]
    with client(fake, clock) as api:
        page = api.list_notes(date(2026, 11, 3))
    assert page.has_more and len(fake.requests) == 4
    assert clock.slept[0] == 3.0 and clock.slept[1:3] == [2.0, 4.0]


def test_client_gives_up_after_retries():
    fake, clock = FakeGranola(), Clock()
    fake.failures["/v1/notes"] = [500] * 10
    with client(fake, clock) as api, pytest.raises(granola.GranolaError) as raised:
        api.list_notes()
    assert raised.value.retryable and len(fake.requests) == granola.MAX_RETRIES + 1


def test_client_401_is_auth_failure_and_404_is_not_retryable():
    fake = FakeGranola()
    fake.failures["/v1/notes"] = [401]
    with client(fake) as api:
        with pytest.raises(granola.GranolaAuthFailed):
            api.list_notes()
        with pytest.raises(granola.GranolaError) as missing:
            api.get_note("not_Zz9Zz9Zz9Zz9Zz")
        with pytest.raises(granola.GranolaError) as invalid:
            api.get_transcript("../admin")
    assert missing.value.code == "not_found" and not missing.value.retryable
    assert invalid.value.code == "invalid_id"
    assert len(fake.requests) == 2


# ---------------------------------------------------------------- poller


class Calls(list):
    receive = None


@pytest.fixture
def received(monkeypatch):
    """Replace `jarvis.context.sources.receive` (package B1) with a recorder."""
    calls = Calls()

    def receive(db, owner, envelope, connection_id=None):
        if getattr(receive, "fail", None) == envelope.external_id:
            raise RuntimeError("boom")
        calls.append((owner, envelope, connection_id))
        return {"source_id": envelope.external_id}

    module = getattr(context_pkg, "sources", None) or sys.modules.get("jarvis.context.sources")
    if module is None:
        module = types.ModuleType("jarvis.context.sources")
        monkeypatch.setitem(sys.modules, "jarvis.context.sources", module)
        monkeypatch.setattr(context_pkg, "sources", module, raising=False)
    monkeypatch.setattr(module, "receive", receive, raising=False)
    calls.receive = receive
    return calls


@pytest.fixture
def enabled(monkeypatch):
    settings = get_settings()
    monkeypatch.setattr(settings, "context_enabled", True)
    monkeypatch.setattr(settings, "context_granola_enabled", True)
    monkeypatch.setattr(settings, "integration_encryption_key", Fernet.generate_key().decode())
    monkeypatch.setattr(granola.TokenBucket, "acquire", lambda self: None)  # verified on its own above


def make_connection(db, **values) -> m.ContextConnection:
    connection = m.ContextConnection(owner_id=OWNER, connector="granola", external_account=ACCOUNT,
                                     credential_ciphertext=granola.seal_credential("gk_live_secret"), **values)
    db.add(connection)
    db.flush()
    return connection


def cursor_value(db, connection) -> dict:
    return db.scalar(select(m.ContextCursor.value).where(m.ContextCursor.connection_id == connection.id,
                                                         m.ContextCursor.name == "notes"))


def test_adapter_is_off_by_default(received):
    fake = FakeGranola()
    with session_scope() as db:
        connection = m.ContextConnection(owner_id=OWNER, connector="granola", external_account=ACCOUNT)
        db.add(connection)
        db.flush()
        report = granola.sync(db, OWNER, connection, transport=fake.transport)
    assert get_settings().context_granola_enabled is False
    assert report.status == "disabled" and fake.requests == [] and received == []


def test_sync_imports_own_notes_and_advances(received, enabled):
    fake = FakeGranola()
    with session_scope() as db:
        connection = make_connection(db)
        report = granola.sync(db, OWNER, connection, now=NOW, transport=fake.transport, sleep=lambda s: None)
        state = cursor_value(db, connection)
        health = connection.health, connection.last_success_at
    assert (report.status, report.listed, report.delivered, report.foreign, report.deleted_upstream) == (
        "ok", 4, 1, 1, 1)
    assert report.pending == [NO_TRANSCRIPT] and report.advanced and report.watermark == "2026-11-04"
    assert health == ("ok", NOW)
    assert state["scan_from"] == "2026-11-04" and state["pending"][NO_TRANSCRIPT]["reason"] == "no_transcript"
    [(owner, envelope, connection_id)] = received
    assert owner == OWNER and connection_id == connection.id and envelope == note_envelope()
    # First scan starts the day before today; the foreign note's details are never fetched.
    assert fake.requests[0].url.params["updated_after"] == "2026-11-03"
    assert f"/v1/notes/{FOREIGN}" not in fake.paths() and f"/v1/notes/{DELETED}" not in fake.paths()


def test_repeat_sync_dedupes_and_scans_from_last_success(received, enabled):
    with session_scope() as db:
        connection = make_connection(db)
        granola.sync(db, OWNER, connection, now=NOW, transport=FakeGranola().transport, sleep=lambda s: None)
        fake = FakeGranola()
        report = granola.sync(db, OWNER, connection, now=datetime(2026, 11, 6, 9, tzinfo=UTC),
                              transport=fake.transport, sleep=lambda s: None)
    assert (report.delivered, report.unchanged, report.pending) == (0, 1, [NO_TRANSCRIPT])
    assert len(received) == 1
    assert fake.requests[0].url.params["updated_after"] == "2026-11-03"  # last success 11-04, minus a day
    assert report.watermark == "2026-11-06"


@pytest.mark.parametrize(("failure", "health"), [([429] * 10, "degraded"), ([502] * 10, "degraded"),
                                                  ([httpx.ConnectTimeout("t")] * 10, "degraded"),
                                                  ([401], "auth_failed")])
def test_upstream_failures_pause_without_advancing(received, enabled, failure, health):
    with session_scope() as db:
        connection = make_connection(db)
        granola.sync(db, OWNER, connection, now=NOW, transport=FakeGranola().transport, sleep=lambda s: None)
        fake = FakeGranola()
        fake.failures[f"/v1/notes/{NO_TRANSCRIPT}/transcript"] = list(failure)
        report = granola.sync(db, OWNER, connection, now=datetime(2026, 11, 7, tzinfo=UTC),
                              transport=fake.transport, sleep=lambda s: None)
        state = cursor_value(db, connection)
        assert connection.health == health and connection.last_success_at == NOW
        assert connection.last_error and "gk_live_secret" not in connection.last_error
    assert report.status == health and not report.advanced
    assert state["scan_from"] == "2026-11-04" and report.watermark == "2026-11-04"
    assert len(received) == 1


def test_ambiguous_404_and_receive_errors_park_the_note(received, enabled):
    fake = FakeGranola()
    fake.failures[f"/v1/notes/{NOTE}"] = [404]
    with session_scope() as db:
        connection = make_connection(db)
        report = granola.sync(db, OWNER, connection, now=NOW, transport=fake.transport, sleep=lambda s: None)
        assert report.advanced and set(report.pending) == {NOTE, NO_TRANSCRIPT}
        assert cursor_value(db, connection)["pending"][NOTE]["reason"] == "not_found"
        received.receive.fail = NOTE
        report = granola.sync(db, OWNER, connection, now=NOW, transport=FakeGranola().transport,
                              sleep=lambda s: None)
        assert cursor_value(db, connection)["pending"][NOTE]["reason"] == "receive_failed"
        received.receive.fail = None
        report = granola.sync(db, OWNER, connection, now=NOW, transport=FakeGranola().transport,
                              sleep=lambda s: None)
    assert report.delivered == 1 and report.pending == [NO_TRANSCRIPT] and len(received) == 1


def test_pending_notes_are_abandoned_after_a_week(received, enabled):
    with session_scope() as db:
        connection = make_connection(db)
        granola.sync(db, OWNER, connection, now=NOW, transport=FakeGranola().transport, sleep=lambda s: None)
        report = granola.sync(db, OWNER, connection, now=datetime(2026, 11, 12, tzinfo=UTC),
                              transport=FakeGranola().transport, sleep=lambda s: None)
    assert report.abandoned == [NO_TRANSCRIPT] and report.pending == []


def test_connection_must_belong_to_owner(received, enabled):
    with session_scope() as db:
        connection = make_connection(db)
        with pytest.raises(ValueError):
            granola.sync(db, "someone-else", connection, transport=FakeGranola().transport)


def test_unreadable_credential_marks_auth_failed(received, enabled):
    fake = FakeGranola()
    with session_scope() as db:
        connection = make_connection(db)
        connection.credential_ciphertext = "not-a-fernet-token"
        report = granola.sync(db, OWNER, connection, now=NOW, transport=fake.transport)
        assert connection.health == "auth_failed"
    assert report.status == "auth_failed" and fake.requests == []


def test_backfill_scans_created_window_once(received, enabled):
    fake = FakeGranola()
    with session_scope() as db:
        connection = make_connection(db)
        report = granola.backfill(db, OWNER, connection, days=90, now=NOW, transport=fake.transport,
                                  sleep=lambda s: None)
        again = granola.backfill(db, OWNER, connection, days=90, now=NOW, transport=fake.transport,
                                 sleep=lambda s: None)
    assert report.delivered == 1 and report.advanced and again.status == "complete"
    assert fake.requests[0].url.params["created_after"] == "2026-08-06"
    assert "updated_after" not in fake.requests[0].url.params


GROUP = "not_G7r8o9u0p1C2a3"  # a colleague's single-device recording of a four-person call


def test_one_audio_source_in_a_group_call_leaves_speakers_unknown():
    note, items = load(f"note_{GROUP}.json"), load(f"transcript_{GROUP}_1.json")["transcript"]
    envelope = granola.to_envelope(note, items)
    assert {(s.speaker_label, s.attribution) for s in envelope.segments} == {(None, None)}
    assert envelope.metadata["granola"]["speakers_separated"] is False
    assert all(p.speaker_label is None for p in envelope.participants)  # nobody is the "me" voice
    assert {p.email for p in envelope.participants} >= {"sam.okafor@fernhill.example", "tomas.reyes@northwind.example"}
    # Two people on one microphone is the owner's own call: lines stay the owner's.
    pair = note | {"attendees": note["attendees"][:2], "calendar_event": None}
    assert {(s.speaker_label, s.attribution) for s in granola.to_envelope(pair, items).segments} == {
        ("Sam Okafor", "owner")}
    # Any name or diarization label means Granola separated the voices.
    named = [items[0], {**items[1], "speaker": {"source": "speaker", "attribution": "them", "name": "Alex Morgan"}},
             *items[2:]]
    labels = [s.speaker_label for s in granola.to_envelope(note, named).segments]
    assert labels[:2] == ["Sam Okafor", "Alex Morgan"]
    assert "speakers_separated" not in granola.to_envelope(note, named).metadata["granola"]
