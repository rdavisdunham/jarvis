"""PUT/GET/DELETE /api/v1/external/sources: scopes, flag, personal-workspace rule, formats and idempotency."""

import json

import pytest
from fastapi.testclient import TestClient
from jarvis.api import app
from jarvis.config import get_settings
from jarvis.context import models as m
from jarvis.db import session_scope
from jarvis.models import BotCredential, Job
from sqlalchemy import func, select

from .conftest import MEETING_AT

BASE = "/api/v1/external/sources"
VTT = "WEBVTT\n\n00:00:01.000 --> 00:00:03.000\n<v Davin>Kick-off.</v>\n\n00:00:04.000 --> 00:00:06.000\n<v Priya>Ready.</v>\n"


@pytest.fixture(autouse=True)
def enabled(monkeypatch):
    monkeypatch.setenv("JARVIS_CONTEXT_ENABLED", "true")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def key(client, *scopes):
    response = client.post("/api/v1/bot-keys", json={"name": "ctx", "scopes": list(scopes), "expires_in_days": 30})
    assert response.status_code == 201, response.text
    return {"Authorization": "Bearer " + response.json()["token"]}


def body(**values):
    return {"version": 1, "title": "Pilot scope", "occurred_at": MEETING_AT.isoformat(), "external_revision": "r1",
            "segments": [{"seq": 0, "speaker_label": "Davin", "text": "We settle scope today."},
                         {"seq": 1, "speaker_label": "Priya", "text": "The pilot covers 40 clinics."}],
            "private_notes": "Check the contract.", **values}


def put(headers, path="recorder/m1", payload=None, content_type=None, **kwargs):
    if payload is not None:
        kwargs["json"] = payload
    headers = {**headers, "Content-Type": content_type} if content_type else headers
    return TestClient(app).put(f"{BASE}/{path}", headers=headers, **kwargs)


def code(response):
    return response.status_code, response.json()["error"]["code"]


def count(model):
    with session_scope() as db:
        return db.scalar(select(func.count()).select_from(model))


def test_json_submission_is_idempotent_and_revises_on_changed_content(client):
    writer = key(client, "sources:write")
    first = put(writer, payload=body())
    assert first.status_code == 200, first.text
    created = first.json()
    assert (created["connector"], created["label"], created["external_id"], created["revision"]) == ("api", "recorder", "m1", 1)
    assert (created["ingestion_status"], created["processing_status"]) == ("created", "queued") and created["job_id"]
    assert set(created) == {"id", "connector", "label", "external_id", "revision", "revision_id", "ingestion_status",
                            "processing_status", "job_id"}
    again = put(writer, payload=body()).json()
    assert (again["id"], again["ingestion_status"], again["job_id"]) == (created["id"], "none", None)
    meta = put(writer, payload=body(summary="Display only", external_revision="r2")).json()
    assert (meta["ingestion_status"], meta["revision"], meta["job_id"]) == ("metadata", 1, None)
    revised = put(writer, payload=body(segments=[{"seq": 0, "text": "Entirely new content."}])).json()
    assert (revised["id"], revised["ingestion_status"], revised["revision"]) == (created["id"], "content", 2)
    assert count(m.ContextSource) == 1 and count(m.ContextSourceRevision) == 2
    with session_scope() as db:
        assert db.scalar(select(func.count()).select_from(Job).where(Job.kind == "context_ingest")) == 2


def test_known_connectors_keep_their_key_and_other_labels_fall_back_to_api(client):
    from jarvis.context import sources
    from jarvis.context.schema import SourceEnvelope

    writer = key(client, "sources:write")
    zoom = put(writer, "Zoom/123", body()).json()
    recorder = put(writer, "recorder/123", body()).json()
    smuggled = put(writer, "api/123", body(metadata={"connector_label": "recorder"})).json()
    assert (zoom["connector"], zoom["label"]) == ("zoom", "zoom")
    assert (recorder["connector"], recorder["label"], recorder["external_id"]) == ("api", "recorder", "123")
    assert len({zoom["id"], recorder["id"], smuggled["id"]}) == 3
    bridged = put(writer, "granola/note-1", body()).json()
    with session_scope() as db:  # the in-app poller delivering the same Granola note dedupes to one source
        polled = sources.receive(db, "davin", SourceEnvelope.model_validate({
            **{k: v for k, v in body().items() if k != "version"}, "connector": "granola", "external_id": "note-1"}))
        assert db.get(m.ContextSource, recorder["id"]).external_id == "recorder:123"
    assert (polled.source_id, polled.change) == (bridged["id"], "none")
    assert code(put(writer, "bad label/1", body())) == (400, "INVALID_ARGUMENT")


def test_raw_formats_dispatch_by_content_type(client):
    writer = key(client, "sources:write")
    params = {"occurred_at": MEETING_AT.isoformat(), "title": "Standup"}
    vtt = put(writer, "zoom/v1", content=VTT, params=params, content_type="text/vtt")
    assert vtt.status_code == 200, vtt.text
    srt = "1\n00:00:01,000 --> 00:00:02,000\nMarco: Hi.\n"
    assert put(writer, "zoom/s1", content=srt, params=params,
               content_type="application/x-subrip").status_code == 200
    md = put(writer, "upload/n1", content="# Notes\n\n**Davin:** Go.", params=params,
             content_type="text/markdown")
    assert md.status_code == 200
    plain = put(writer, "other/t1", content="Davin: Plain.", params=params,
                content_type="text/plain; charset=utf-8")
    assert plain.json()["ingestion_status"] == "created"
    missing = put(writer, "other/t2", content="Davin: hi", content_type="text/plain")
    assert code(missing) == (400, "INVALID_ARGUMENT") and "occurred_at" in missing.json()["error"]["message"]
    assert code(put(writer, "other/t3", content=b"%PDF", params=params,
                    content_type="application/pdf")) == (415, "UNSUPPORTED_MEDIA_TYPE")
    bad = put(writer, "zoom/v2", content="WEBVTT\n\nnot a cue", params=params, content_type="text/vtt")
    assert code(bad) == (400, "INVALID_ARGUMENT")
    reader = key(client, "context:read")
    with session_scope() as db:
        source = db.scalar(select(m.ContextSource).where(m.ContextSource.connector == "zoom",
                                                         m.ContextSource.external_id == "v1"))
    detail = TestClient(app).get(f"{BASE}/{source.id}", headers=reader).json()
    assert [(s["speaker_label"], s["text"], s["start_ms"]) for s in detail["segments"]] == [
        ("Davin", "Kick-off.", 1000), ("Priya", "Ready.", 4000)]
    assert detail["title"] == "Standup" and detail["label"] == "zoom" and detail["external_id"] == "v1"


def test_oversized_and_invalid_envelopes_are_rejected(client, monkeypatch):
    writer = key(client, "sources:write")
    monkeypatch.setenv("JARVIS_CONTEXT_MAX_SOURCE_CHARS", "1000")
    get_settings.cache_clear()
    big = put(writer, payload=body(segments=[{"seq": 0, "text": "x" * 1001}]))
    assert code(big) == (413, "SOURCE_TOO_LARGE") and big.json()["error"]["data"]["max_source_chars"] == 1000
    invalid = put(writer, payload=body(occurred_at="2026-11-03T15:00:00"))
    assert code(invalid) == (400, "INVALID_ARGUMENT") and invalid.json()["error"]["data"][0]["loc"] == ["occurred_at"]
    assert code(put(writer, content="{", content_type="application/json")) == (400, "INVALID_ARGUMENT")
    assert count(m.ContextSource) == 0


def test_scopes_separate_submission_reading_and_deletion(client):
    writer, reader, editor = key(client, "sources:write"), key(client, "context:read"), key(client, "context:write")
    source_id = put(writer, payload=body()).json()["id"]
    http = TestClient(app)
    assert code(http.get(BASE, headers=writer)) == (403, "INSUFFICIENT_SCOPE")
    assert code(http.get(f"{BASE}/{source_id}", headers=writer)) == (403, "INSUFFICIENT_SCOPE")
    assert code(http.delete(f"{BASE}/{source_id}", headers=writer)) == (403, "INSUFFICIENT_SCOPE")
    assert code(put(reader, payload=body())) == (403, "INSUFFICIENT_SCOPE")
    assert code(http.delete(f"{BASE}/{source_id}", headers=reader)) == (403, "INSUFFICIENT_SCOPE")
    listing = http.get(BASE, headers=reader).json()
    assert listing["total"] == 1 and "segments" not in listing["items"][0]
    assert "We settle scope today." not in json.dumps(listing)
    detail = http.get(f"{BASE}/{source_id}", headers=reader).json()
    assert [s["kind"] for s in detail["segments"]] == ["transcript", "transcript", "private_note"]
    assert code(http.get(BASE, params={"status": "bogus"}, headers=reader)) == (400, "INVALID_ARGUMENT")
    deleted = http.delete(f"{BASE}/{source_id}", headers=editor)
    assert deleted.status_code == 200 and deleted.json()["status"] == "deleted"
    assert code(http.get(f"{BASE}/{source_id}", headers=editor)) == (404, "NOT_FOUND")
    assert http.get(BASE, headers=reader).json()["items"] == []
    redelivered = put(writer, payload=body()).json()
    assert (redelivered["ingestion_status"], redelivered["processing_status"]) == ("deleted", "deleted")
    with session_scope() as db:
        assert db.scalar(select(func.count()).select_from(Job).where(Job.kind == "context_delete")) == 1


def test_routes_refuse_when_context_is_disabled(client, monkeypatch):
    writer, reader = key(client, "sources:write"), key(client, "context:read")
    monkeypatch.setenv("JARVIS_CONTEXT_ENABLED", "false")
    get_settings.cache_clear()
    assert code(put(writer, payload=body())) == (403, "CONTEXT_DISABLED")
    assert code(TestClient(app).get(BASE, headers=reader)) == (403, "CONTEXT_DISABLED")
    assert count(m.ContextSource) == 0


def test_routes_refuse_in_shared_workspaces(client):
    from test_accounts import client_for, shared

    guest = client_for("guest")
    shared(client, guest)
    headers = key(client, "tasks:read")
    with session_scope() as db:  # a key minted before the personal-only rule
        row = db.scalar(select(BotCredential).where(BotCredential.name == "ctx"))
        row.scopes = ["context:read", "sources:write", "tasks:read"]
    assert code(put(headers, payload=body())) == (403, "PERSONAL_WORKSPACE")
    assert code(TestClient(app).get(BASE, headers=headers)) == (403, "PERSONAL_WORKSPACE")
    assert count(m.ContextSource) == 0


def test_transcript_route_has_its_own_body_cap(client, monkeypatch):
    from jarvis.context.routes.sources import body_limit

    writer = key(client, "sources:write")
    large = body(segments=[{"seq": i, "speaker_label": "Davin", "text": f"Line {i}. " + "x" * 1000} for i in range(150)])
    assert len(json.dumps(large)) > 150_000
    accepted = put(writer, payload=large)
    assert accepted.status_code == 200, accepted.text
    session = TestClient(app).put("/api/v1/external/sessions/claude_code/s1", headers=writer,
                                  json={"title": "x" * 150_000})
    assert code(session) == (413, "INVALID_ARGUMENT")  # every other route keeps the 100 KB cap
    monkeypatch.setenv("JARVIS_CONTEXT_MAX_SOURCE_CHARS", "1000")
    get_settings.cache_clear()
    assert body_limit() == 102_000
    over = put(writer, "recorder/big", content=b"x" * 110_000, content_type="text/plain")
    assert code(over) == (413, "INVALID_ARGUMENT")
    chunked = put(writer, "recorder/big", content=iter([b"x" * 60_000, b"x" * 60_000]), content_type="text/plain",
                  params={"occurred_at": MEETING_AT.isoformat()})
    assert code(chunked) == (413, "SOURCE_TOO_LARGE")
