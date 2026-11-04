"""/api/v1/context/reviews (owner session) and /api/v1/external/context/reviews (bot keys, read-only): review
read, approval batches, separate update approval, dismiss, later, Wrong versus Changed, retry, and the rules that
agents never approve and stale revisions refuse the whole batch."""

import pytest
from fastapi.testclient import TestClient
from jarvis.api import app
from jarvis.config import get_settings
from jarvis.context import models as m
from jarvis.db import session_scope
from jarvis.models import Task
from sqlalchemy import func, select

from .test_pipeline import drain, use_doubles, vectors  # noqa: F401
from .test_proposals import by_title, meeting, process

BASE = "/api/v1/context/reviews"
EXTERNAL = "/api/v1/external/context/reviews"


@pytest.fixture(autouse=True)
def enabled(monkeypatch, vectors):  # noqa: F811
    monkeypatch.setenv("JARVIS_CONTEXT_ENABLED", "true")
    get_settings.cache_clear()
    use_doubles()
    yield
    get_settings.cache_clear()


def key(client, *scopes):
    response = client.post("/api/v1/bot-keys", json={"name": "ctx", "scopes": list(scopes), "expires_in_days": 30})
    assert response.status_code == 201, response.text
    return {"Authorization": "Bearer " + response.json()["token"]}


def code(response):
    return response.status_code, response.json()["error"]["code"]


def review(client, source_id):
    response = client.get(f"{BASE}/{source_id}")
    assert response.status_code == 200, response.text
    return response.json()


def item(proposal, **extra):
    return {"proposal_id": proposal["id"], "expected_revision": proposal["revision"], **extra}


def tasks():
    with session_scope() as db:
        return db.scalar(select(func.count()).select_from(Task).where(Task.title.not_like("owner%")))


def test_review_read_inbox_and_bot_twins(client):
    got = process(meeting())
    data = review(client, got.source_id)
    assert data["source"]["id"] == got.source_id and "segments" not in data["source"]
    mine = by_title(data["proposals"], "circulate")
    assert mine["preselected"] and mine["responsible"]["id"] and mine["evidence"][0]["quote"]
    assert {c["kind"] for c in data["context"]} >= {"commitment"} and data["destinations"]["default"] == "eridani"
    inbox = client.get(BASE).json()["items"]
    assert inbox == [{"source_id": got.source_id, "title": "Pilot sync", "status": "ready", "occurred_at":
                      data["source"]["occurred_at"], "open": 3, "preselected": 1}]
    reader = key(client, "context:read")
    bot = TestClient(app)
    assert bot.get(f"{EXTERNAL}/{got.source_id}", headers=reader).json()["proposals"] == data["proposals"]
    assert bot.get(EXTERNAL, headers=reader).json()["items"] == inbox
    assert code(bot.get(EXTERNAL, headers=key(client, "sources:write")))[0] == 403
    writer = key(client, "context:read", "context:write")
    refused = bot.post(f"{EXTERNAL}/{got.source_id}/approve", headers=writer,
                       json={"command_id": "c1", "items": [item(mine)]})
    assert refused.status_code in (404, 405) and tasks() == 0


def test_approve_selected_publishes_once_and_agents_are_refused(client):
    got = process(meeting())
    mine = by_title(review(client, got.source_id)["proposals"], "circulate")
    agent = client.post(f"{BASE}/{got.source_id}/approve", json={"command_id": "ui-agent:1", "items": [item(mine)]})
    assert code(agent) == (403, "HUMAN_REQUIRED")
    body = {"command_id": "approve-1", "items": [item(mine, edits={"due_date": "2026-11-07"})]}
    first = client.post(f"{BASE}/{got.source_id}/approve", json=body)
    assert first.status_code == 200, first.text
    assert client.post(f"{BASE}/{got.source_id}/approve", json=body).json() == first.json()  # replay
    drain()
    assert tasks() == 1
    after = by_title(review(client, got.source_id)["proposals"], "circulate")
    assert after["disposition"] == "published" and after["task"]["due_date"] == "2026-11-07"
    assert after["publication"]["status"] == "published"


def test_stale_revision_refuses_the_whole_batch(client):
    got = process(meeting())
    proposals = review(client, got.source_id)["proposals"]
    mine, priya = by_title(proposals, "circulate"), by_title(proposals, "budget")
    stale = client.post(f"{BASE}/{got.source_id}/approve", json={
        "command_id": "approve-2", "items": [item(priya), {**item(mine), "expected_revision": 9}]})
    assert code(stale) == (409, "REVISION_CONFLICT")
    with session_scope() as db:
        assert db.scalar(select(func.count()).select_from(m.ContextPublication)) == 0
        assert db.get(m.ContextProposal, priya["id"]).disposition == "pending"


def test_updates_need_their_own_approval(client):
    got = process(meeting())
    mine = by_title(review(client, got.source_id)["proposals"], "circulate")
    refused = client.post(f"{BASE}/{got.source_id}/apply-updates", json={"command_id": "u1", "items": [item(mine)]})
    assert code(refused) == (400, "INVALID_ARGUMENT")


def test_dismiss_and_later(client):
    got = process(meeting())
    proposals = review(client, got.source_id)["proposals"]
    priya, marco = by_title(proposals, "budget"), by_title(proposals, "checklist")
    dismissed = client.post(f"{BASE}/{got.source_id}/dismiss", json={"command_id": "d1", "items": [item(priya)]})
    assert dismissed.status_code == 200 and dismissed.json()["proposals"][0]["disposition"] == "dismissed"
    later = client.post(f"{BASE}/{got.source_id}/later", json={"command_id": "l1", "items": [item(marco)],
                                                               "until": "2026-11-05T17:00:00+00:00"})
    assert later.json()["proposals"][0]["disposition"] == "later"
    other = process(meeting(external_id="m-2"))
    foreign = by_title(review(client, other.source_id)["proposals"], "circulate")
    wrong_source = client.post(f"{BASE}/{got.source_id}/dismiss", json={"command_id": "d2", "items": [item(foreign)]})
    assert code(wrong_source) == (404, "NOT_FOUND")


def test_wrong_and_changed_behave_differently(client):
    got = process(meeting())
    data = review(client, got.source_id)
    mine, priya = by_title(data["proposals"], "circulate"), by_title(data["proposals"], "budget")
    own = next(c for c in data["context"] if c["id"] == mine["assertion_ids"][0])
    theirs = next(c for c in data["context"] if c["id"] == priya["assertion_ids"][0])

    wrong = client.post(f"{BASE}/{got.source_id}/assertions/{theirs['id']}/wrong",
                        json={"command_id": "w1", "expected_revision": theirs["revision"]})
    assert wrong.status_code == 200 and wrong.json()["superseded_proposals"] == [priya["id"]]

    changed = client.post(f"{BASE}/{got.source_id}/assertions/{own['id']}/changed", json={
        "command_id": "c1", "expected_revision": own["revision"], "due_at": "2026-11-13T00:00:00+00:00",
        "statement": "I'll circulate the notes by 2026-11-13."})
    assert changed.status_code == 200, changed.text
    assert changed.json()["superseded_proposals"] == []
    after = review(client, got.source_id)["proposals"]
    revised = next(p for p in after if p["id"] == mine["id"])
    assert revised["disposition"] == "pending" and revised["due_date"] == "2026-11-13" and revised["preselected"]
    assert revised["revision"] == mine["revision"] + 1 and "2026-11-13" in revised["explanation"]
    assert next(p for p in after if p["id"] == priya["id"])["disposition"] == "superseded"
    stale = client.post(f"{BASE}/{got.source_id}/approve", json={"command_id": "a9", "items": [item(mine)]})
    assert code(stale) == (409, "REVISION_CONFLICT")


def test_changed_after_publication_proposes_an_update(client):
    got = process(meeting())
    data = review(client, got.source_id)
    mine = by_title(data["proposals"], "circulate")
    own = next(c for c in data["context"] if c["id"] == mine["assertion_ids"][0])
    client.post(f"{BASE}/{got.source_id}/approve", json={"command_id": "a1", "items": [item(mine)]})
    drain()
    client.post(f"{BASE}/{got.source_id}/assertions/{own['id']}/changed", json={
        "command_id": "c1", "expected_revision": own["revision"], "due_at": "2026-11-13T00:00:00+00:00",
        "statement": "I'll circulate the notes by 2026-11-13."})
    follow = next(p for p in review(client, got.source_id)["proposals"] if p["follows"] == mine["id"])
    assert follow["kind"] == "update" and follow["after"] == {"due_date": "2026-11-13"}
    applied = client.post(f"{BASE}/{got.source_id}/apply-updates", json={"command_id": "u1", "items": [item(follow)]})
    assert applied.status_code == 200, applied.text
    drain()
    with session_scope() as db:
        assert str(db.get(Task, follow["task"]["id"]).due_date) == "2026-11-13"
    assert tasks() == 1


def test_retry_route_is_human_only(client):
    got = process(meeting())
    mine = by_title(review(client, got.source_id)["proposals"], "circulate")
    published = client.post(f"{BASE}/{got.source_id}/approve", json={"command_id": "a1", "items": [item(mine)]})
    pid = published.json()["publications"][0]["id"]
    assert code(client.post(f"{BASE}/publications/{pid}/retry", json={"command_id": "ui-agent:r"})) == \
        (403, "HUMAN_REQUIRED")
    assert client.post(f"{BASE}/publications/{pid}/retry", json={"command_id": "r1"}).json()["status"] == "queued"
    drain()
    assert tasks() == 1


def test_flag_off_refuses(client, monkeypatch):
    monkeypatch.setenv("JARVIS_CONTEXT_ENABLED", "false")
    get_settings.cache_clear()
    assert code(client.get(BASE)) == (403, "CONTEXT_DISABLED")
