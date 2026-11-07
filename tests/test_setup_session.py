"""Set up with Dani: the interview session (docs/SETUP_WITH_DANI_PLAN.md, PR 1)."""
from uuid import uuid4

import pytest
from cryptography.fernet import Fernet
from sqlalchemy import select

from jarvis import agent_work, setup, work_runner
from jarvis.agent_instructions import backend_instructions
from jarvis.config import get_settings
from jarvis.db import session_scope
from jarvis.models import AgentWork, OwnerSettings, SharedWorkspace, Source


@pytest.fixture(autouse=True)
def model_settings(monkeypatch):
    monkeypatch.setenv("JARVIS_INTEGRATION_ENCRYPTION_KEY", Fernet.generate_key().decode())
    monkeypatch.setenv("JARVIS_OPENAI_API_KEY", "synthetic")
    monkeypatch.setenv("JARVIS_GEMINI_API_KEY", "synthetic")
    monkeypatch.setenv("JARVIS_EXTERNAL_SERVICES_ENABLED", "true")
    monkeypatch.setenv("JARVIS_COST_TRACKING_ENABLED", "false")
    get_settings.cache_clear()

    async def memories(*args):
        return ""

    monkeypatch.setattr(work_runner, "prompt_context", memories)
    yield
    get_settings.cache_clear()


def owner():
    return get_settings().owner_id


def history_off():
    with session_scope() as db:
        row = db.get(OwnerSettings, owner())
        if not row:
            row = OwnerSettings(owner_id=owner(), values={})
            db.add(row)
        row.values = {**row.values, "history_enabled": False}


def messages(client, conversation_id):
    return client.get("/api/v1/conversations/" + conversation_id).json()["messages"]


def send(client, conversation_id, text, purpose="setup"):
    return client.post("/api/v1/work", json={"turn_id": str(uuid4()), "conversation_id": conversation_id,
                                             "message": text, "purpose": purpose})


async def state_for(work_id):
    with session_scope() as db:
        row = db.get(AgentWork, work_id)
        db.expunge(row)
    with agent_work.principal_for(row):
        return await work_runner.initial_state(row, work_runner.model_for("luna"), tool_policy="lean-v1")


def test_start_opens_a_setup_conversation_with_the_fixed_opening(client):
    assert client.get("/api/v1/setup").json()["status"] == "not_started"
    started = client.post("/api/v1/setup/start", json={"mode": "typed"}).json()
    assert started["status"] == "interviewing" and started["mode"] == "typed"
    assert [m["role"] for m in messages(client, started["conversation_id"])] == ["assistant"]
    assert messages(client, started["conversation_id"])[0]["content"] == setup.OPENING


def test_resume_keeps_the_conversation_and_restart_begins_a_new_one(client):
    first = client.post("/api/v1/setup/start", json={"mode": "voice"}).json()
    resumed = client.post("/api/v1/setup/start", json={"mode": "typed"}).json()
    assert resumed["conversation_id"] == first["conversation_id"] and resumed["mode"] == "typed"
    assert len(messages(client, first["conversation_id"])) == 1
    switched = client.post("/api/v1/setup/mode", json={"mode": "voice"}).json()
    assert switched["mode"] == "voice" and switched["revision"] > resumed["revision"]
    fresh = client.post("/api/v1/setup/start", json={"restart": True}).json()
    assert fresh["conversation_id"] != first["conversation_id"]


def test_finished_run_reopens_as_a_rerun(client):
    client.post("/api/v1/setup/start", json={})
    done = client.post("/api/v1/setup/finish", json={"status": "completed"}).json()
    assert done["status"] == "completed" and done["finished_at"]
    again = client.post("/api/v1/setup/start", json={"focus": "organization"}).json()
    assert again["focus"] == "organization"
    assert messages(client, again["conversation_id"])[0]["content"] == setup.RERUN_OPENING


def test_setup_is_personal_only(client):
    from jarvis.access import tool_access
    from jarvis.domain import DomainError
    with session_scope() as db:
        db.add(SharedWorkspace(id="davin-shared", name="Shared", kind="project", creator_id=owner()))
    with session_scope() as db:
        for check in (lambda: setup.state(db, "davin-shared"), lambda: tool_access(db, "davin-shared", "setup_propose")):
            with pytest.raises(DomainError) as exc:
                check()
            assert exc.value.code == "PERSONAL_WORKSPACE"


def test_setup_purpose_needs_the_active_setup_conversation(client):
    other = client.post("/api/v1/conversations", json={}).json()
    refused = send(client, other["id"], "I teach high school chemistry.")
    assert refused.status_code == 409 and refused.json()["error"]["code"] == "SETUP_NOT_ACTIVE"
    started = client.post("/api/v1/setup/start", json={}).json()
    assert send(client, started["conversation_id"], "I teach high school chemistry.").status_code == 200
    client.post("/api/v1/setup/finish", json={"status": "abandoned"})
    assert send(client, started["conversation_id"], "One more thing").status_code == 409


async def test_history_off_still_keeps_the_setup_transcript(client):
    history_off()
    started = client.post("/api/v1/setup/start", json={}).json()
    assert started["history_off"]
    work = send(client, started["conversation_id"], "I run a small bakery.").json()
    assert [m["role"] for m in messages(client, started["conversation_id"])] == ["assistant", "user"]
    state = await state_for(work["id"])
    assert any("SETUP INTERVIEW" in (m.get("content") or "") and "history is off" in (m.get("content") or "")
               for m in state["messages"])
    # An ordinary conversation still keeps nothing.
    other = client.post("/api/v1/conversations", json={}).json()
    assert send(client, other["id"], "Add Alpha", purpose=None).status_code == 200
    with session_scope() as db:
        assert not db.scalars(select(Source).where(Source.conversation_id == other["id"])).all()
    finished = client.post("/api/v1/setup/finish", json={"delete_conversation": True}).json()
    assert finished["status"] == "completed"
    with session_scope() as db:
        kept = db.scalars(select(Source).where(Source.conversation_id == started["conversation_id"])).all()
        assert kept and all(s.deleted_at and s.content == "" for s in kept)


async def test_interview_block_only_in_setup_turns_and_only_in_the_dynamic_half(client):
    started = client.post("/api/v1/setup/start", json={}).json()
    setup_work = send(client, started["conversation_id"], "I'm a nurse.").json()
    setup_state = await state_for(setup_work["id"])
    other = client.post("/api/v1/conversations", json={}).json()
    plain = send(client, other["id"], "Add Alpha", purpose=None).json()
    plain_state = await state_for(plain["id"])
    text = lambda state: "\n".join(m.get("content") or "" for m in state["messages"])  # noqa: E731
    assert "SETUP INTERVIEW" in text(setup_state) and "SETUP INTERVIEW" not in text(plain_state)
    # The opening is part of the transcript the typed model reads.
    assert setup.OPENING[:40] in text(setup_state)
    prefs = {"timezone": "UTC", "default_reminder_hour": 9}
    static, dynamic = backend_instructions(prefs, parts=True, tool_policy="lean-v1")
    static_setup, dynamic_setup = backend_instructions(prefs, parts=True, tool_policy="lean-v1", setup=setup.SETUP_INTERVIEW)
    assert static == static_setup and "SETUP INTERVIEW" in dynamic_setup and "SETUP INTERVIEW" not in static_setup


def test_onboarding_completes_with_an_applied_tree(client):
    from datetime import timedelta
    from jarvis.domain import DomainError, execute
    from jarvis.models import now
    from jarvis.setup_models import OrganizationProposal, SetupSession
    client.post("/api/v1/setup/start", json={})
    with session_scope() as db:
        with pytest.raises(DomainError) as exc:
            execute(db, owner(), "c-1", "onboarding.save", {"expected_revision": 0, "status": "completed"})
        assert exc.value.code == "PREVIEW_REQUIRED"
    with session_scope() as db:
        proposal = OrganizationProposal(owner_id=owner(), account_id=owner(), tree=[], impact={}, preview_hash="x",
                                        applied_at=now(), expires_at=now() + timedelta(hours=1))
        db.add(proposal)
        db.flush()
        db.get(SetupSession, owner()).tree_proposal_id = proposal.id
    with session_scope() as db:
        result = execute(db, owner(), "c-2", "onboarding.save", {"expected_revision": 0, "status": "completed"})
        assert result["data"]["status"] == "completed"
