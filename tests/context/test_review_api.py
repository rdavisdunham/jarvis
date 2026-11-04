"""/api/v1/context/reviews (owner session) and /api/v1/external/context/reviews (bot keys, read-only), in the
shapes the web client (`apps/web/src/context/context-api.ts`) reads: review read and list, approval batches,
separate update approval, dismiss, later, done, Wrong versus Changed, speaker corrections, retry, and the rules
that agents never approve, resent commands change nothing and stale revisions refuse the whole batch."""

import pytest
from fastapi.testclient import TestClient
from jarvis.api import app
from jarvis.config import get_settings
from jarvis.context import identity, notices
from jarvis.context import models as m
from jarvis.db import session_scope
from jarvis.models import Task
from sqlalchemy import func, select

from .test_pipeline import drain, use_doubles, vectors  # noqa: F401
from .test_proposals import OWNER, by_title, meeting, process

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
    return {"proposal_id": proposal["id"], "revision": proposal["revision"], **extra}


def post(client, source_id, route, **body):
    return client.post(f"{BASE}/{source_id}/{route}", json=body)


def tasks():
    with session_scope() as db:
        return db.scalar(select(func.count()).select_from(Task).where(Task.title.not_like("owner%")))


def test_review_read_list_and_bot_twins(client):
    got = process(meeting())
    data = review(client, got.source_id)
    assert data["source_id"] == got.source_id and data["source"]["id"] == got.source_id
    assert "segments" not in data["source"] and data["status"] == "pending"
    assert data["revision"] == sum(p["revision"] for p in data["proposals"])
    assert data["destination"] == "eridani" and data["destinations"] == ["eridani"]
    mine = by_title(data["proposals"], "circulate")
    assert mine["preselected"] and mine["assignee"]["id"] and mine["assignee"]["name"]
    assert mine["evidence"][0]["quote"] and mine["evidence"][0]["segment"] == 0
    assert mine["evidence"][0]["speaker"] == "Davin" and mine["target"] is None and mine["publication"] is None
    assert mine["due_at"].startswith("2026-11-06")
    assert {i["kind"] for i in data["items"]} >= {"commitment"} and all(i["citation"] for i in data["items"])
    loops = next(s for s in data["brief"]["sections"] if s["name"] == "open_loops")
    assert loops["claims"][0]["citations"][0]["source_id"] == got.source_id
    assert {s["label"] for s in data["speakers"]} == {"Davin", "Priya"}
    assert all(s["link_revision"] == 1 and s["decision_id"] is None for s in data["speakers"])

    listed = client.get(BASE).json()["items"]
    assert listed == [{"source_id": got.source_id, "title": "Pilot sync", "occurred_at": data["source"]["occurred_at"],
                       "status": "pending", "pending": 3, "later": 0, "published": 0}]
    reader = key(client, "context:read")
    bot = TestClient(app)
    assert bot.get(f"{EXTERNAL}/{got.source_id}", headers=reader).json()["proposals"] == \
        [{**p, "task": None} for p in data["proposals"]]
    assert bot.get(EXTERNAL, headers=reader).json()["items"] == listed
    assert code(bot.get(EXTERNAL, headers=key(client, "sources:write")))[0] == 403
    writer = key(client, "context:read", "context:write")
    refused = bot.post(f"{EXTERNAL}/{got.source_id}/approve", headers=writer,
                       json={"command_id": "c1", "items": [item(mine)]})
    assert refused.status_code in (404, 405) and tasks() == 0


def test_approve_publishes_once_and_agents_are_refused(client):
    got = process(meeting())
    mine = by_title(review(client, got.source_id)["proposals"], "circulate")
    agent = post(client, got.source_id, "approve", command_id="ui-agent:approve:1", destination="eridani",
                 items=[item(mine)])
    assert code(agent) == (403, "HUMAN_REQUIRED")
    body = {"command_id": "approve-1", "destination": "eridani", "items": [item(mine, edits={"due_date": "2026-11-07"})]}
    first = client.post(f"{BASE}/{got.source_id}/approve", json=body)
    assert first.status_code == 200, first.text
    again = client.post(f"{BASE}/{got.source_id}/approve", json=body)  # the UI resends after a lost connection
    assert again.json()["publications"] == first.json()["publications"]
    approved = by_title(first.json()["review"]["proposals"], "circulate")
    assert approved["disposition"] == "approved" and approved["publication"]["status"] == "queued"
    drain()
    assert tasks() == 1
    after = by_title(review(client, got.source_id)["proposals"], "circulate")
    assert after["disposition"] == "published" and after["task"]["due_date"] == "2026-11-07"
    assert after["publication"]["status"] == "published" and after["publication"]["task_id"]


def test_stale_revision_refuses_the_whole_batch(client):
    got = process(meeting())
    proposals = review(client, got.source_id)["proposals"]
    mine, priya = by_title(proposals, "circulate"), by_title(proposals, "budget")
    stale = post(client, got.source_id, "approve", command_id="approve-2",
                 items=[item(priya), {**item(mine), "revision": 9}])
    assert code(stale) == (409, "REVISION_CONFLICT")
    with session_scope() as db:
        assert db.scalar(select(func.count()).select_from(m.ContextPublication)) == 0
        assert db.get(m.ContextProposal, priya["id"]).disposition == "pending"


def test_updates_need_their_own_approval(client):
    got = process(meeting())
    mine = by_title(review(client, got.source_id)["proposals"], "circulate")
    refused = post(client, got.source_id, "apply-updates", command_id="u1", items=[item(mine)])
    assert code(refused) == (400, "INVALID_ARGUMENT")


def test_dismiss_later_and_resent_commands(client):
    got = process(meeting())
    proposals = review(client, got.source_id)["proposals"]
    priya, marco = by_title(proposals, "budget"), by_title(proposals, "checklist")
    dismissed = post(client, got.source_id, "dismiss", command_id="d1", items=[item(priya)])
    assert by_title(dismissed.json()["review"]["proposals"], "budget")["disposition"] == "dismissed"
    resent = post(client, got.source_id, "dismiss", command_id="d1", items=[item(priya)])
    assert resent.status_code == 200 and resent.json()["review"] == dismissed.json()["review"]
    later = post(client, got.source_id, "later", command_id="l1", items=[item(marco)],
                 until="2026-11-05T17:00:00+00:00")
    assert by_title(later.json()["review"]["proposals"], "checklist")["disposition"] == "later"
    with session_scope() as db:
        decision = db.scalar(select(m.ContextDecision).where(m.ContextDecision.command_id == "l1"))
        assert decision.effective_at.isoformat() == "2026-11-05T17:00:00+00:00"  # E1's digest reads this
    other = process(meeting(external_id="m-2"))
    foreign = by_title(review(client, other.source_id)["proposals"], "circulate")
    assert code(post(client, got.source_id, "dismiss", command_id="d2", items=[item(foreign)])) == (404, "NOT_FOUND")


def test_later_without_items_defers_the_whole_review(client):
    got = process(meeting())
    data = post(client, got.source_id, "later", command_id="l-all", items=[]).json()["review"]
    assert data["status"] == "later" and {p["disposition"] for p in data["proposals"]} == {"later"}
    with session_scope() as db:
        assert got.source_id not in notices.unreviewed(db, OWNER)
    assert client.get(BASE).json()["items"][0]["later"] == 3


def test_done_dismisses_what_is_open_and_checks_the_review_revision(client):
    got = process(meeting())
    data = review(client, got.source_id)
    selection = [item(p) for p in data["proposals"]]
    stale = post(client, got.source_id, "done", command_id="done-0", review_revision=data["revision"] - 1,
                 items=selection)
    assert code(stale) == (409, "REVISION_CONFLICT")
    finished = post(client, got.source_id, "done", command_id="done-1", review_revision=data["revision"],
                    items=selection)
    assert finished.status_code == 200, finished.text
    after = finished.json()["review"]
    assert after["status"] == "done" and {p["disposition"] for p in after["proposals"]} == {"dismissed"}
    assert after["items"] == data["items"] and tasks() == 0  # the context stays; no task


def test_wrong_and_changed_behave_differently(client):
    got = process(meeting())
    data = review(client, got.source_id)
    mine, priya = by_title(data["proposals"], "circulate"), by_title(data["proposals"], "budget")
    own = next(c for c in data["items"] if c["id"] == mine["assertion_ids"][0])
    theirs = next(c for c in data["items"] if c["id"] == priya["assertion_ids"][0])

    wrong = post(client, got.source_id, "wrong", command_id="w1", assertion_id=theirs["id"],
                 expected_revision=theirs["revision"], reason="")
    assert wrong.status_code == 200 and wrong.json()["superseded_proposals"] == [priya["id"]]
    assert by_title(wrong.json()["review"]["proposals"], "budget")["disposition"] == "superseded"

    changed = post(client, got.source_id, "changed", command_id="c1", assertion_id=own["id"],
                   expected_revision=own["revision"], due_at="2026-11-13T00:00:00+00:00",
                   statement="I'll circulate the notes by 2026-11-13.", effective_at=None)
    assert changed.status_code == 200, changed.text
    assert changed.json()["superseded_proposals"] == []
    revised = next(p for p in changed.json()["review"]["proposals"] if p["id"] == mine["id"])
    assert revised["disposition"] == "pending" and revised["due_date"] == "2026-11-13" and revised["preselected"]
    assert revised["revision"] == mine["revision"] + 1 and "2026-11-13" in revised["explanation"]
    stale = post(client, got.source_id, "approve", command_id="a9", items=[item(mine)])
    assert code(stale) == (409, "REVISION_CONFLICT")


def test_changed_after_publication_proposes_an_update(client):
    got = process(meeting())
    data = review(client, got.source_id)
    mine = by_title(data["proposals"], "circulate")
    own = next(c for c in data["items"] if c["id"] == mine["assertion_ids"][0])
    post(client, got.source_id, "approve", command_id="a1", items=[item(mine)])
    drain()
    post(client, got.source_id, "changed", command_id="c1", assertion_id=own["id"],
         expected_revision=own["revision"], due_at="2026-11-13T00:00:00+00:00",
         statement="I'll circulate the notes by 2026-11-13.")
    follow = next(p for p in review(client, got.source_id)["proposals"] if p["follows"] == mine["id"])
    assert follow["kind"] == "update" and follow["after"] == {"due_date": "2026-11-13"}
    assert follow["before"] == {"due_date": "2026-11-06"} and follow["target"]["entity_type"] == "task"
    applied = post(client, got.source_id, "apply-updates", command_id="u1", items=[item(follow)])
    assert applied.status_code == 200, applied.text
    drain()
    with session_scope() as db:
        assert str(db.get(Task, follow["target"]["id"]).due_date) == "2026-11-13"
    assert tasks() == 1


def speaker(data, label):
    return next(s for s in data["speakers"] if s["label"] == label)


def test_speaker_confirm_reject_and_undo(client):
    got = process(meeting())
    data = review(client, got.source_id)
    davin = speaker(data, "Davin")
    assert davin["actor_id"] and davin["status"] in ("exact", "inferred")

    confirmed = post(client, got.source_id, "speakers", command_id="s1", link_id=davin["link_id"],
                     expected_revision=davin["link_revision"], action="confirm")
    assert confirmed.status_code == 200, confirmed.text
    now_ = speaker(confirmed.json()["review"], "Davin")
    assert now_["status"] == "confirmed" and now_["decision_id"] and now_["link_revision"] == 2
    stale = post(client, got.source_id, "speakers", command_id="s2", link_id=davin["link_id"],
                 expected_revision=1, action="reject")
    assert code(stale) == (409, "REVISION_CONFLICT")

    undone = post(client, got.source_id, "speakers", command_id="s3", link_id=davin["link_id"],
                  expected_revision=2, action="undo", decision_id=now_["decision_id"])
    assert undone.status_code == 200, undone.text
    back = speaker(undone.json()["review"], "Davin")
    assert back["status"] == davin["status"] and back["actor_id"] == davin["actor_id"] and back["decision_id"] is None

    rejected = post(client, got.source_id, "speakers", command_id="s4", link_id=davin["link_id"],
                    expected_revision=back["link_revision"], action="reject")
    assert rejected.status_code == 200, rejected.text
    gone = speaker(rejected.json()["review"], "Davin")
    assert gone["status"] == "rejected" and gone["actor_id"] is None and gone["decision_id"]
    with session_scope() as db:
        segments = list(db.scalars(select(m.ContextSegment).where(
            m.ContextSegment.revision_id == got.revision_id, m.ContextSegment.speaker_label == "Davin")))
        assert len(segments) == 2 and all(s.speaker_link_status == "rejected" and s.speaker_actor_id is None
                                          and s.speaker_provenance["decision_id"] == gone["decision_id"]
                                          for s in segments)
        negatives = identity.rejected_links(db, OWNER)
        assert negatives.blocks("Davin", "actor", davin["actor_id"], scope=got.source_id)  # durable, this meeting
        assert not negatives.blocks("Davin", "actor", davin["actor_id"])

    other = review(client, process(meeting(external_id="m-2")).source_id)
    assert code(post(client, got.source_id, "speakers", command_id="s5", link_id=speaker(other, "Davin")["link_id"],
                     expected_revision=1, action="confirm")) == (404, "NOT_FOUND")


def test_retry_route_is_human_only(client):
    got = process(meeting())
    mine = by_title(review(client, got.source_id)["proposals"], "circulate")
    published = post(client, got.source_id, "approve", command_id="a1", items=[item(mine)])
    pid = published.json()["publications"][0]["id"]
    assert code(client.post(f"{BASE}/publications/{pid}/retry", json={"command_id": "ui-agent:retry:r"})) == \
        (403, "HUMAN_REQUIRED")
    retried = client.post(f"{BASE}/publications/{pid}/retry", json={"command_id": "r1"}).json()
    assert retried["publication"]["status"] == "queued" and retried["review"]["source_id"] == got.source_id
    drain()
    assert tasks() == 1


def test_flag_off_refuses(client, monkeypatch):
    monkeypatch.setenv("JARVIS_CONTEXT_ENABLED", "false")
    get_settings.cache_clear()
    assert code(client.get(BASE)) == (403, "CONTEXT_DISABLED")
