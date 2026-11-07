"""Set up with Dani: finishing a run and re-running it (docs/SETUP_WITH_DANI_PLAN.md, PR 7)."""
from uuid import uuid4

import pytest
from cryptography.fernet import Fernet
from sqlalchemy import select

import jarvis.domain  # noqa: F401  (load the command registry first)
from jarvis import setup
from jarvis.config import get_settings
from jarvis.db import session_scope
from jarvis.models import Job, OwnerSettings, Source
from jarvis.setup_models import SetupSession


@pytest.fixture(autouse=True)
def model_settings(monkeypatch):
    monkeypatch.setenv("JARVIS_INTEGRATION_ENCRYPTION_KEY", Fernet.generate_key().decode())
    monkeypatch.setenv("JARVIS_OPENAI_API_KEY", "synthetic")
    monkeypatch.setenv("JARVIS_EXTERNAL_SERVICES_ENABLED", "true")
    monkeypatch.setenv("JARVIS_COST_TRACKING_ENABLED", "false")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def owner():
    return get_settings().owner_id


def command(client, tool, arguments):
    response = client.post("/api/v1/commands", json={"command_id": str(uuid4()), "tool": tool, "arguments": arguments})
    assert response.status_code == 200, response.text
    return response.json()["data"]


def applied(client):
    session = client.post("/api/v1/setup/start", json={}).json()
    proposal = command(client, "setup.tree_preview", {"tree": [{"id": "s1", "kind": "space", "title": "Work"}]})
    command(client, "setup.tree_apply", {"proposal_id": proposal["id"], "preview_hash": proposal["preview_hash"]})
    return session


def said(conversation_id):
    with session_scope() as db:
        return db.scalars(select(Source.content).where(Source.conversation_id == conversation_id, Source.deleted_at.is_(None),
                                                      Source.role == "assistant").order_by(Source.created_at)).all()


def onboarding_status():
    with session_scope() as db:
        row = db.get(OwnerSettings, owner())
        return ((row.values if row else {}).get("onboarding") or {}).get("status")


def test_finishing_after_creating_completes_setup_and_says_so(client):
    session = applied(client)
    state = client.post("/api/v1/setup/finish", json={"status": "completed"}).json()
    assert state["status"] == "completed" and state["finished_at"]
    assert said(session["conversation_id"])[-1] == setup.FINISHING
    assert onboarding_status() == "completed"
    # Finishing twice changes nothing.
    client.post("/api/v1/setup/finish", json={"status": "completed"})
    assert said(session["conversation_id"]).count(setup.FINISHING) == 1


def test_finishing_without_changes_leaves_the_form_path_alone(client):
    session = client.post("/api/v1/setup/start", json={}).json()
    client.post("/api/v1/setup/finish", json={"status": "completed"})
    assert onboarding_status() is None
    assert said(session["conversation_id"])[-1] == setup.FINISHING


def test_finishing_with_history_off_can_delete_the_transcript(client):
    session = applied(client)
    client.post("/api/v1/setup/finish", json={"status": "completed", "delete_conversation": True})
    assert said(session["conversation_id"]) == []
    assert onboarding_status() == "completed"


def test_finishing_queues_the_about_you_rebuild_when_allowed(client, monkeypatch):
    from jarvis import about_you

    monkeypatch.setattr(about_you, "enabled", lambda: True)
    applied(client)
    client.post("/api/v1/setup/finish", json={"status": "completed"})
    with session_scope() as db:
        assert db.scalar(select(Job).where(Job.owner_id == owner(), Job.kind == "about_you"))


def test_a_rerun_opens_with_what_changed_and_keeps_its_focus(client):
    applied(client)
    client.post("/api/v1/setup/finish", json={"status": "completed"})
    again = client.post("/api/v1/setup/start", json={"mode": "typed", "focus": "organization"}).json()
    assert again["status"] == "interviewing" and again["focus"] == "organization" and again["notes"]["rerun"] is True
    assert said(again["conversation_id"]) == [setup.RERUN_OPENING]
    with session_scope() as db:
        block = setup.interview_block(db, owner(), again["conversation_id"])
        assert "focuses on the organization" in block and "This is a re-run" in block
        assert db.get(SetupSession, owner()).tree_proposal_id is None
