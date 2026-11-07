"""People linked to clients and projects, and Dani's record_overview read (organization knowledge, phase 4)."""

import pytest
from context.test_briefs import People
from jarvis import people_links, record_overviews
from jarvis.db import session_scope
from jarvis.models import Actor, Task
from jarvis.structure_models import PersonBucketLink, StructureRecord
from jarvis.tools import READ_TOOLS, _call_tool, registry
from sqlalchemy import select
from test_organization_foundation import OWNER, create


def setup():
    client = create("client", "ABC", body="Home services. We work with their Customer Service team.")
    project = create("project", "Transcript Intelligence", parent_id=client["id"], body="AI call grading.")
    task = create("task", "Rubric co-design with trainers", parent_id=project["id"])
    with session_scope() as db:
        world = People(db, owner=OWNER)
        db.get(Task, task["task_id"]).assignee_id = world.priya
        world.say("Marco will pull the 8x8 call data.", subject=("project", project["id"]), responsible=[world.marco],
                  kind="commitment")
        world.say("The owner reviews every rubric.", subject=("project", project["id"]), responsible=[world.me],
                  kind="commitment")
        people = {"priya": world.priya, "marco": world.marco, "me": world.me}
    return client, project, people


def refresh(record_id):
    with session_scope() as db:
        return people_links.refresh_bucket(db, OWNER, db.get(StructureRecord, record_id))


def links(record_id):
    with session_scope() as db:
        return {row["name"]: row for row in people_links.for_bucket(db, OWNER, record_id)}


def test_links_come_from_assignments_and_meeting_facts_and_follow_the_evidence(monkeypatch):
    client, project, people = setup()
    monkeypatch.setattr("jarvis.context.views.me", lambda db, owner: people["me"])
    refresh(project["id"])
    refresh(client["id"])
    found = links(project["id"])
    assert set(found) == {"Priya", "Marco"}  # The owner is on everything and is never linked.
    assert (found["Priya"]["tasks"], found["Priya"]["meeting_facts"]) == (1, 0)
    assert (found["Marco"]["tasks"], found["Marco"]["meeting_facts"]) == (0, 1)
    assert found["Priya"]["evidence"]["tasks"] == ["Rubric co-design with trainers"]
    assert links(client["id"])["Priya"]["tasks"] == 1  # Non-exclusive: the client sees its project's work.
    with session_scope() as db:
        assert {b["title"] for b in people_links.for_actor(db, OWNER, people["priya"])} == {"ABC", "Transcript Intelligence"}
        # The task is reassigned: Priya's inferred link goes with the evidence.
        task = db.scalar(select(Task).where(Task.title == "Rubric co-design with trainers"))
        task.assignee_id = None
    refresh(project["id"])
    assert set(links(project["id"])) == {"Marco"}


def test_owner_decisions_survive_refreshes(client, monkeypatch):
    task = create("task", "Not a bucket")  # Before the meeting world adds an actor named "owner".
    _, project, people = setup()
    monkeypatch.setattr("jarvis.context.views.me", lambda db, owner: people["me"])
    refresh(project["id"])
    url = f"/api/v1/structure/records/{project['id']}/people"
    assert client.post(f"{url}/{people['marco']}", json={"status": "rejected"}).json()["status"] == "rejected"
    with session_scope() as db:
        kim = Actor(owner_id=OWNER, name="Kim")
        db.add(kim)
        db.flush()
        kim_id = kim.id
    assert client.post(f"{url}/{kim_id}", json={"status": "confirmed"}).json()["status"] == "confirmed"
    refresh(project["id"])
    found = links(project["id"])
    assert set(found) == {"Priya", "Kim"}  # Rejected stays hidden; confirmed stays without evidence.
    with session_scope() as db:
        assert db.scalar(select(PersonBucketLink.status).where(PersonBucketLink.actor_id == people["marco"])) == "rejected"
    # Back to following the evidence.
    client.post(f"{url}/{people['marco']}", json={"status": "inferred"})
    assert "Marco" in links(project["id"])
    assert client.post(f"{url}/{people['marco']}", json={"status": "maybe"}).status_code == 422
    assert client.post(f"/api/v1/structure/records/{task['id']}/people/{kim_id}",
                       json={"status": "confirmed"}).status_code == 400


def test_person_page_lists_linked_buckets(client, monkeypatch):
    from jarvis.config import get_settings

    monkeypatch.setattr(get_settings(), "context_enabled", True)
    _, project, people = setup()
    monkeypatch.setattr("jarvis.context.views.me", lambda db, owner: people["me"])
    refresh(project["id"])
    page = client.get(f"/api/v1/context/people/{people['priya']}")
    assert page.status_code == 200, page.text
    assert [b["title"] for b in page.json()["buckets"]] == ["Transcript Intelligence"]


def test_daily_sweep_refreshes_links_even_without_a_model(monkeypatch):
    _, project, people = setup()
    monkeypatch.setattr("jarvis.context.views.me", lambda db, owner: people["me"])
    monkeypatch.setattr(record_overviews, "enabled", lambda: False)
    with session_scope() as db:
        assert record_overviews.queue_due(db) == 0
    assert set(links(project["id"])) == {"Priya", "Marco"}


@pytest.mark.asyncio
async def test_dani_reads_what_eridani_knows_about_a_bucket(monkeypatch):
    client, project, people = setup()
    monkeypatch.setattr("jarvis.context.views.me", lambda db, owner: people["me"])
    refresh(project["id"])
    assert "record_overview" in {t["name"] for t in registry()} and len(READ_TOOLS["record_overview"]["description"]) < 400
    data = await _call_tool(OWNER, "turn", 0, "record_overview", {"record_id": project["id"]})
    assert data["title"] == "Transcript Intelligence" and data["description"] == "AI call grading."
    assert data["overview"] is None  # None written yet; the description and evidence still answer.
    assert {p["name"] for p in data["people"]} == {"Priya", "Marco"}
    assert any("8x8" in f["statement"] for f in data["meeting_facts"])
    assert data["tasks"]["open"] == 1 and data["tasks"]["recent_open"][0]["title"] == "Rubric co-design with trainers"
    parent = await _call_tool(OWNER, "turn", 1, "record_overview", {"record_id": client["id"]})
    assert parent["child_buckets"] == [{"record_id": project["id"], "type_id": "project", "title": "Transcript Intelligence"}]


def test_naming_a_person_recalls_their_clients_and_projects(monkeypatch):
    client, project, people = setup()
    monkeypatch.setattr("jarvis.context.views.me", lambda db, owner: people["me"])
    refresh(project["id"])
    refresh(client["id"])
    with session_scope() as db:
        db.add(Actor(owner_id=OWNER, name="Priya Shah"))  # Shares the first name: "Priya" alone is not unique.
        db.add(Actor(owner_id=OWNER, name="Jo"))  # Too short to match on its own.
        db.get(Actor, people["marco"]).name = "Marco Diaz"
    with session_scope() as db:
        # A unique first name matches; the latest-named person comes first; possessives still match.
        assert people_links.mentioned(db, OWNER, "Ask priya about it, then check Marco's numbers") == [
            (people["marco"], "Marco Diaz works on or around: Transcript Intelligence (project).")]
        assert [a for a, _ in people_links.mentioned(db, OWNER, "Priya and Marco Diaz")] == [people["marco"]]
        assert people_links.mentioned(db, OWNER, "Priya") == []  # Two Priyas; neither is named in full.
        assert people_links.mentioned(db, OWNER, "Jo said hi. Also the owner.") == []
        assert people_links.mentioned(db, OWNER, "Remarcoing the plan") == []  # Whole words only.
        # Confirmed links count; rejected links stay out.
        people_links.decide(db, OWNER, people["marco"], client["id"], "confirmed")
        people_links.decide(db, OWNER, people["marco"], project["id"], "rejected")
        assert people_links.mentioned(db, OWNER, "Marco") == [
            (people["marco"], "Marco Diaz works on or around: ABC (client).")]


async def test_typed_chat_context_carries_named_people(monkeypatch):
    from unittest.mock import AsyncMock

    from jarvis import memory_service

    _, project, people = setup()
    monkeypatch.setattr("jarvis.context.views.me", lambda db, owner: people["me"])
    refresh(project["id"])
    monkeypatch.setattr(memory_service, "semantic_search", AsyncMock(return_value=[]))
    context = await memory_service.prompt_context(OWNER, "What is Marco working on?")
    assert context.startswith("People named here and where they work (DATA): ")
    assert "Marco works on or around: Transcript Intelligence (project)." in context
    assert await memory_service.prompt_context(OWNER, "What is on my plate?") == ""


def test_a_misheard_name_is_recalled_with_what_was_heard(monkeypatch):
    _, project, people = setup()
    monkeypatch.setattr("jarvis.context.views.me", lambda db, owner: people["me"])
    refresh(project["id"])
    with session_scope() as db:
        db.get(Actor, people["priya"]).name = "Sreya"
        db.get(Actor, people["marco"]).name = "Janiece"
        db.add(Actor(owner_id=OWNER, name="Sarah"))  # No links: never recalled, but still a known name.
    refresh(project["id"])
    with session_scope() as db:
        assert people_links.mentioned(db, OWNER, "What did Janice say about the rubric?") == [
            (people["marco"], '"Janice" is probably Janiece, who works on or around: Transcript Intelligence (project).')]
        assert [a for a, _ in people_links.mentioned(db, OWNER, "Ask Shreya, then Janiece")] == [
            people["marco"], people["priya"]]
        # Exact wins: no "probably" when the right spelling is also said.
        assert "probably" not in people_links.mentioned(db, OWNER, "Janice, I mean Janiece")[0][1]
        # Lowercase words and common words never fuzzy-match; nor do ambiguous ones.
        assert people_links.mentioned(db, OWNER, "janice said hi") == []
        assert people_links.mentioned(db, OWNER, "Sorry, wrong number") == []
        db.add(Actor(owner_id=OWNER, name="Janise"))  # Now "Janice" could be either.
        db.flush()
        assert people_links.mentioned(db, OWNER, "What did Janice say?") == []


async def test_person_lookup_tool_finds_misspelled_names_with_their_work(monkeypatch):
    _, project, people = setup()
    monkeypatch.setattr("jarvis.context.views.me", lambda db, owner: people["me"])
    with session_scope() as db:
        db.get(Actor, people["marco"]).name = "Janiece Moreno"
        db.add(Actor(owner_id=OWNER, name="Janet"))
    refresh(project["id"])
    assert "person_lookup" in {t["name"] for t in registry()} and len(READ_TOOLS["person_lookup"]["description"]) < 400
    found = await _call_tool(OWNER, "turn", 0, "person_lookup", {"name": "Janice"})
    assert [p["name"] for p in found["people"]] == ["Janiece Moreno"]
    best = found["people"][0]
    assert best["actor_id"] == people["marco"] and best["match"] == "sounds_like" and best["score"] >= 0.8
    assert best["works_on"] == [{"record_id": project["id"], "title": "Transcript Intelligence", "type": "project",
                                 "status": "inferred"}]
    exact = await _call_tool(OWNER, "turn", 1, "person_lookup", {"name": "janiece moreno"})
    assert exact["people"][0]["match"] == "exact"
    assert (await _call_tool(OWNER, "turn", 2, "person_lookup", {"name": "Zed"}))["people"] == []
