import json
from uuid import uuid4

from jarvis import agent_work, latency
from jarvis.db import session_scope
from jarvis.models import AgentWork, Job
from sqlalchemy.orm import Session


def events(caplog):
    return [json.loads(r.message.split("latency ", 1)[1])
            for r in caplog.records if r.name == "jarvis.latency"]


def test_commit_events_do_not_report_rolled_back_saves(caplog):
    with Session() as db:
        db.begin()
        latency.after_commit(db, "work_finished", "rollback", outcome="succeeded")
        db.rollback()
        db.begin()
        latency.after_commit(db, "work_finished", "commit", outcome="succeeded")
        db.commit()
        db.begin()
        latency.after_commit(db, "work_finished", "another", outcome="failed")
        db.commit()
    assert [e["request_id"] for e in events(caplog)] == ["commit", "another"]


def test_nested_commit_is_not_final_and_context_is_content_free(caplog):
    with Session() as db:
        db.begin()
        nested = db.begin_nested()
        with latency.context(run_id="run"):
            latency.after_commit(db, "work_finished", "test", outcome="succeeded", message="private")
        nested.commit()
        assert not events(caplog)
        db.commit()
    assert events(caplog)[0]["run_id"] == "run"
    assert "private" not in str(events(caplog))


def test_browser_timing_requires_scoped_terminal_work(client, monkeypatch, caplog):
    from jarvis.config import get_settings
    from jarvis.api import client_latency_seen, client_latency_rates
    monkeypatch.setenv("JARVIS_OPENAI_API_KEY", "synthetic")
    monkeypatch.setenv("JARVIS_EXTERNAL_SERVICES_ENABLED", "true")
    get_settings.cache_clear()
    client_latency_seen.clear()
    client_latency_rates.clear()
    try:
        conv = client.post("/api/v1/conversations", json={}).json()
        identity = str(uuid4())
        accepted = client.post("/api/v1/work", json={"turn_id": identity,
            "conversation_id": conv["id"], "message": "Hello"})
        assert accepted.status_code == 200, accepted.text
        url = "/api/v1/work/" + identity + "/latency"
        body = {"stage": "work_reply_rendered", "revision": 1, "elapsed_ms": 120}
        assert client.post(url, json=body).status_code == 403
        with session_scope() as db:
            agent_work.finish(db, db.get(AgentWork, identity), "succeeded", "Hello")
        assert client.post(url, json={**body, "revision": 2}).status_code == 403
        assert client.post(url, json={**body, "elapsed_ms": -1}).status_code == 422
        assert client.post(url, json={**body, "elapsed_ms": 300001}).status_code == 422
        assert client.post(url, json={**body, "content": "private"}).status_code == 422
        assert client.post(url, json=body).json() == {"recorded": True}
        assert client.post(url, json=body).json() == {"recorded": False}
        with session_scope() as db:
            db.get(AgentWork, identity).device_id = "another-device"
        assert client.post(url, json=body).status_code == 403
        emitted = [e for e in events(caplog) if e["stage"] == "work_reply_rendered"]
        assert len(emitted) == 1 and emitted[0]["source"] == "client_reported"
    finally:
        get_settings.cache_clear()
