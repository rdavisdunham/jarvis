"""Person pages for the owner's session (/api/v1/context/people/*): the re-checked person view, the people list
and human-only identity corrections (merge, unmerge, not the same person, confirm or reject a link)."""

from uuid import uuid4

import pytest
from jarvis.config import get_settings
from jarvis.context import service
from jarvis.context.models import ContextActorMerge, ContextDecision, ContextEntityLink
from jarvis.context.native import retrieve as native_retrieve
from jarvis.db import session_scope
from jarvis.domain import execute
from jarvis.models import Actor, Task
from sqlalchemy import func, select

from .test_service import OWNER, seed, vec

BASE = "/api/v1/context/people"


@pytest.fixture(autouse=True)
def enabled(monkeypatch):
    monkeypatch.setenv("JARVIS_CONTEXT_ENABLED", "true")
    get_settings.cache_clear()
    monkeypatch.setattr(native_retrieve, "query_vector", lambda owner, text: vec(0))
    yield
    get_settings.cache_clear()


def code(response):
    return response.status_code, response.json()["error"]["code"]


def namesake(name="Priya Shah", owner=OWNER) -> str:
    with session_scope() as db:
        actor = Actor(owner_id=owner, name=name)
        db.add(actor)
        db.flush()
        return actor.id


def inferred_link(assertion_id: str, actor_id: str, mention="Priya") -> str:
    with session_scope() as db:
        row = ContextEntityLink(owner_id=OWNER, assertion_id=assertion_id, subject_type="assertion",
                                subject_id=assertion_id, target_type="actor", target_id=actor_id, mention=mention,
                                role="participant", status="inferred", score=0.71)
        db.add(row)
        db.flush()
        return row.id


def decisions(operation: str) -> int:
    with session_scope() as db:
        return db.scalar(select(func.count()).select_from(ContextDecision).where(
            ContextDecision.owner_id == OWNER, ContextDecision.operation == operation))


def test_person_page_shows_cited_context_meetings_tasks_and_merge_candidates(client):
    ids = seed()
    twin = namesake()
    with session_scope() as db:
        task = execute(db, OWNER, f"test-{uuid4()}", "task.create",
                       {"title": "Send the pilot budget", "assignee_id": ids.priya})["data"]["id"]
    page = client.get(f"{BASE}/{ids.priya}")
    assert page.status_code == 200, page.text
    data = page.json()
    assert data["actor"] == {**data["actor"], "id": ids.priya, "name": "Priya Shah", "is_me": False}
    context = {c["assertion_id"]: c for c in data["context"]}
    assert set(context) == {ids.new, ids.why, ids.question}  # superseded MySQL left out
    cite = context[ids.new]["citation"]
    assert cite["source_id"] and cite["segment"] == 0 and cite["quote"].startswith("We decided to launch the pilot")
    assert "evidence" not in context[ids.new]
    assert {m["source_id"] for m in data["meetings"]} >= {cite["source_id"]}
    assert [t["id"] for t in data["tasks"]] == [task]
    assert [(c["actor_id"], c["name"]) for c in data["merge_candidates"]] == [(twin, "Priya Shah")]
    assert "Priya Shah" in data["merge_candidates"][0]["reason"].title()

    listed = client.get(BASE, params={"q": "priya"}).json()["items"]
    assert sorted(p["id"] for p in listed) == sorted([ids.priya, twin])
    assert client.get(BASE, params={"q": "nobody"}).json()["items"] == []


def test_the_person_view_is_rechecked_before_return(client, monkeypatch):
    ids = seed()
    real = service.visible
    monkeypatch.setattr(service, "visible", lambda db, scope, wanted, at=None: real(db, scope, wanted, at=at)
                        - {ids.why})
    data = client.get(f"{BASE}/{ids.priya}").json()
    assert ids.why not in {c["assertion_id"] for c in data["context"]}
    assert any("1 statement(s) withheld" in g for g in data["gaps"])


def test_reads_refuse_unknown_people_and_when_context_is_off(client, monkeypatch):
    ids = seed()
    stranger = namesake("Priya Shah", owner="someone-else")
    assert code(client.get(f"{BASE}/{stranger}")) == (404, "NOT_FOUND")
    assert code(client.post(f"{BASE}/{stranger}/not-same", json={"command_id": str(uuid4()),
                                                                  "other_actor_id": ids.priya})) == (404, "NOT_FOUND")
    monkeypatch.setenv("JARVIS_CONTEXT_ENABLED", "false")
    get_settings.cache_clear()
    assert code(client.get(f"{BASE}/{ids.priya}")) == (403, "CONTEXT_DISABLED")
    assert code(client.get(BASE)) == (403, "CONTEXT_DISABLED")


def test_merge_is_human_only_idempotent_and_unmerge_restores(client):
    ids = seed()
    twin = namesake()
    with session_scope() as db:
        task = execute(db, OWNER, f"test-{uuid4()}", "task.create",
                       {"title": "Book the venue", "assignee_id": twin})["data"]["id"]
    body = {"winner_actor_id": ids.priya, "loser_actor_id": twin}

    agent = client.post(f"{BASE}/{twin}/merge", json={**body, "command_id": f"ui-agent:dani:{uuid4()}"})
    assert code(agent) == (403, "NOT_AUTHORIZED")
    assert code(client.post(f"{BASE}/{ids.priya}/merge", json={"winner_actor_id": twin, "loser_actor_id": "x",
                                                                "command_id": str(uuid4())})) == (
        400, "INVALID_ARGUMENT")

    command_id = str(uuid4())
    merged = client.post(f"{BASE}/{twin}/merge", json={**body, "command_id": command_id})
    assert merged.status_code == 200, merged.text
    person = merged.json()["person"]
    assert person["actor"]["id"] == ids.priya  # the page of the person kept
    assert [t["id"] for t in person["tasks"]] == [task]
    [entry] = person["merges"]
    assert (entry["loser_name"], entry["undone_at"]) == ("Priya Shah", None)
    assert person["merge_candidates"] == []
    again = client.post(f"{BASE}/{twin}/merge", json={**body, "command_id": command_id})
    assert again.status_code == 200 and decisions("merge") == 1  # a retry changes nothing
    with session_scope() as db:
        assert db.get(Actor, twin).archived and db.get(Task, task).assignee_id == ids.priya

    assert code(client.post(f"{BASE}/{ids.priya}/unmerge", json={"merge_id": entry["merge_id"],
                                                                  "command_id": f"ui-agent:x:{uuid4()}"})) == (
        403, "NOT_AUTHORIZED")
    undone = client.post(f"{BASE}/{ids.priya}/unmerge", json={"merge_id": entry["merge_id"],
                                                               "command_id": str(uuid4())})
    assert undone.status_code == 200, undone.text
    assert undone.json()["person"]["merges"][0]["undone_at"] is not None
    with session_scope() as db:
        assert not db.get(Actor, twin).archived and db.get(Task, task).assignee_id == twin
        assert db.get(ContextActorMerge, entry["merge_id"]).undone_at is not None
    assert code(client.post(f"{BASE}/{ids.priya}/unmerge", json={"merge_id": entry["merge_id"],
                                                                  "command_id": str(uuid4())})) == (
        409, "REVISION_CONFLICT")


def test_not_the_same_person_is_durable(client):
    ids = seed()
    twin = namesake()
    assert client.get(f"{BASE}/{ids.priya}").json()["merge_candidates"]
    response = client.post(f"{BASE}/{ids.priya}/not-same", json={"other_actor_id": twin, "command_id": str(uuid4()),
                                                                  "reason": "Different teams."})
    assert response.status_code == 200, response.text
    assert response.json()["person"]["merge_candidates"] == []
    assert client.get(f"{BASE}/{twin}").json()["merge_candidates"] == []
    with session_scope() as db:
        row = db.scalar(select(ContextDecision).where(ContextDecision.operation == "no_merge"))
        assert (row.origin, row.negative, row.reason) == ("human", True, "Different teams.")


def test_inferred_links_are_confirmed_or_rejected_against_their_revision(client):
    ids = seed()
    keep, drop = inferred_link(ids.new, ids.priya), inferred_link(ids.why, ids.priya, mention="P.")
    data = client.get(f"{BASE}/{ids.priya}").json()
    review = {r["link_id"]: r for r in data["links_to_review"]}
    assert set(review) == {keep, drop}
    assert review[keep]["statement"].startswith("We decided to launch the pilot on Postgres")

    stale = client.post(f"{BASE}/{ids.priya}/links/{keep}", json={"action": "confirm", "expected_revision": 9,
                                                                   "command_id": str(uuid4())})
    assert code(stale) == (409, "REVISION_CONFLICT")
    assert code(client.post(f"{BASE}/{ids.priya}/links/{keep}", json={
        "action": "confirm", "expected_revision": 1, "command_id": f"ui-agent:x:{uuid4()}"})) == (403, "NOT_AUTHORIZED")
    other = namesake("Marco")
    assert code(client.post(f"{BASE}/{other}/links/{keep}", json={
        "action": "confirm", "expected_revision": 1, "command_id": str(uuid4())})) == (404, "NOT_FOUND")

    confirmed = client.post(f"{BASE}/{ids.priya}/links/{keep}", json={"action": "confirm", "expected_revision": 1,
                                                                       "command_id": str(uuid4())})
    assert confirmed.status_code == 200, confirmed.text
    rejected = client.post(f"{BASE}/{ids.priya}/links/{drop}", json={"action": "reject", "expected_revision": 1,
                                                                      "command_id": str(uuid4())})
    assert rejected.json()["person"]["links_to_review"] == []
    with session_scope() as db:
        assert db.get(ContextEntityLink, keep).status == "confirmed"
        assert db.get(ContextEntityLink, drop).status == "rejected"
    assert decisions("reject_link") == 1 and decisions("confirm_link") == 1


def test_writes_need_csrf(client):
    ids = seed()
    del client.headers["X-CSRF-Token"]
    response = client.post(f"{BASE}/{ids.priya}/not-same", json={"other_actor_id": namesake(),
                                                                  "command_id": str(uuid4())})
    assert response.status_code == 403
