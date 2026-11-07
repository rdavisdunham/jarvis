"""Set up with Dani: confirming what Dani always keeps in mind (docs/SETUP_WITH_DANI_PLAN.md, PR 6)."""
from uuid import uuid4

import pytest
from cryptography.fernet import Fernet
from sqlalchemy import select

import jarvis.domain  # noqa: F401  (load the command registry before the modules that register into it)
from jarvis import about_you
from jarvis.config import get_settings
from jarvis.db import session_scope
from jarvis.domain import DomainError
from jarvis.memory_learning import fingerprint
from jarvis.models import Job, Memory, SharedWorkspace, Source
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


def command(client, tool, arguments, status=200):
    response = client.post("/api/v1/commands", json={"command_id": str(uuid4()), "tool": tool, "arguments": arguments})
    assert response.status_code == status, response.text
    return response.json()


def fact(db, content, **values):
    source = Source(owner_id=owner(), kind="explicit", role="user", native_id="f:" + str(uuid4()), content=content)
    db.add(source)
    db.flush()
    row = Memory(owner_id=owner(), source_id=source.id, content=content, fingerprint=fingerprint(content),
                 attribution="automatic", **values)
    db.add(row)
    db.flush()
    return row.id


def with_candidates(client, rows):
    session = client.post("/api/v1/setup/start", json={}).json()
    with session_scope() as db:
        evidence = db.scalar(select(Source.id).where(Source.conversation_id == session["conversation_id"]))
        row = db.get(SetupSession, owner())
        row.memory_candidates = [{"id": str(i), "content": c, "fact_key": "k" + str(i), "evidence_source_id": evidence,
                                  "evidence": "quote", "existing_memory_id": m, "action": a}
                                 for i, (c, m, a) in enumerate(rows)]
    return session


def test_saving_confirms_edits_dedupes_and_discards(client):
    with session_scope() as db:
        kept = fact(db, "Davis coaches robotics.")
        old = fact(db, "Davis works at ABC.")
        retired = fact(db, "Davis lives in Austin.", core=True)
        learned = fact(db, "Davis teaches chemistry.")  # Luna already saved it from the same turns.
    with_candidates(client, [("Davis teaches chemistry.", None, "add"), ("Davis coaches robotics.", kept, "keep"),
                             ("Davis works at XYZ.", old, "update"), ("Davis lives in Austin.", retired, "retire"),
                             ("Davis likes jazz.", None, "add"), ("Davis has a dog.", None, "add")])
    result = command(client, "setup.memories_save", {"choices": [
        {"id": "0", "content": "Davis teaches chemistry."}, {"id": "1", "content": "Davis coaches robotics."},
        {"id": "2", "content": "Davis works at XYZ."}, {"id": "3", "content": "Davis lives in Austin."},
        {"id": "4", "content": "Davis loves jazz."}, {"id": "5", "content": "Davis has a dog.", "include": False}]})["data"]
    assert result["memory_candidates"] == [] and result["retired_ids"] == [retired]
    with session_scope() as db:
        assert db.get(Memory, learned).core and db.get(Memory, kept).core
        assert result["saved_ids"][:2] == [learned, kept]
        updated = db.get(Memory, result["saved_ids"][2])
        assert (updated.content, updated.supersedes_id, updated.core) == ("Davis works at XYZ.", old, True)
        assert db.get(Memory, old).suppressed and not db.get(Memory, retired).core
        jazz = db.get(Memory, result["saved_ids"][3])
        # Confirmed in the review: the owner's own statement, on its own source (survives deleting the transcript).
        assert (jazz.content, jazz.attribution, jazz.core) == ("Davis loves jazz.", "owner_statement", True)
        assert db.get(Source, jazz.source_id).conversation_id is None
        assert not db.scalar(select(Memory).where(Memory.content == "Davis has a dog."))
        assert db.get(SetupSession, owner()).confirmed_memory_ids == result["saved_ids"]
        assert db.scalar(select(Job).where(Job.kind == "about_you", Job.owner_id == owner()))


def test_stale_candidate_ids_are_refused(client):
    with_candidates(client, [("Davis teaches chemistry.", None, "add")])
    command(client, "setup.memories_save", {"choices": [{"id": "9", "content": "x"}]}, 409)


def test_pin_and_unpin_feed_about_you(client):
    with session_scope() as db:
        pinned = fact(db, "Davis is training for a marathon.")
        other = fact(db, "Davis prefers mornings.")
    assert command(client, "memory.pin", {"memory_id": pinned})["data"]["core"] is True
    assert client.get("/api/v1/memory").json()["about"]["core"] == [{"id": pinned, "content": "Davis is training for a marathon."}]
    with session_scope() as db:
        material, ids = about_you.gather(db, owner(), {"preferred_name": "Davis"})
        assert material["core_facts"] == ["Davis is training for a marathon."]
        assert "Davis is training for a marathon." not in material["saved_facts_newest_first"]
        assert set(ids) >= {pinned, other}
    assert command(client, "memory.unpin", {"memory_id": pinned})["data"]["core"] is False
    with session_scope() as db:
        assert "core_facts" not in about_you.gather(db, owner(), {"preferred_name": "Davis"})[0]


def test_correcting_a_core_fact_keeps_it_core(client):
    with session_scope() as db:
        pinned = fact(db, "Davis lives in Austin.", core=True)
    corrected = command(client, "memory.correct", {"memory_id": pinned, "content": "Davis lives in Denver."})["data"]
    with session_scope() as db:
        assert db.get(Memory, corrected["id"]).core


def test_shared_workspace_is_refused(client):
    from jarvis import setup_memory
    from jarvis.access import PERSONAL_TOOLS

    assert "setup.memories_save".startswith(PERSONAL_TOOLS) and "memory.pin".startswith(PERSONAL_TOOLS)
    with session_scope() as db:
        db.add(SharedWorkspace(id="davin-shared", name="Shared", kind="project", creator_id="davin"))
        db.flush()
        with pytest.raises(DomainError) as exc:
            setup_memory.save(db, "davin-shared", setup_memory.MemoriesSave(choices=[]), "c1")
        assert exc.value.code == "PERSONAL_WORKSPACE"
