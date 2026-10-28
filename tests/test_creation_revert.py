"""Creation undo distinguishes note indexing from edits, while retaining revision/link guards."""
from uuid import uuid4

import pytest
from sqlalchemy import select
from jarvis import notes
from jarvis.action_history import public_change
from jarvis.db import session_scope
from jarvis.domain import DomainError
from jarvis.models import ActionChange, Job, Note, NoteTaskLink, StructureRecord
from test_agent_work import recovery_settings  # noqa: F401


def command(client, tool, arguments):
    result = client.post("/api/v1/commands", json={
        "command_id": str(uuid4()), "tool": tool, "arguments": arguments,
    })
    assert result.status_code == 200, result.text
    return result.json()


def created(client, kind="note", *, generic=False):
    if generic:
        schema = client.get("/api/v1/structure").json()
        result = command(client, "record.create", {
            "type_id": kind, "title": "Synthetic creation", "body": "Synthetic body",
            "schema_revision": schema["revision"],
        })
        entity_kind = "record"
    else:
        result = command(client, kind + ".create", {
            "title": "Synthetic creation", **({"content": "Synthetic body"} if kind == "note" else {}),
        })
        entity_kind = kind
    with session_scope() as db:
        change = db.scalar(select(ActionChange).where(
            ActionChange.command_id == result["command_id"], ActionChange.entity_kind == entity_kind,
        ))
        return result["data"], change.id


def state(change_id):
    with session_scope() as db:
        return public_change(db, db.get(ActionChange, change_id))


def revert(client, change_id):
    return client.post("/api/v1/work/actions/" + change_id + "/revert", json={"command_id": str(uuid4())})


def index(monkeypatch, note_id, outcome="ready"):
    def embeddings(owner, texts):
        if outcome == "deferred":
            raise DomainError("BUDGET_DEFERRED", "Synthetic budget pause")
        if outcome in {"retrying", "failed"}:
            raise RuntimeError("Synthetic provider outage")
        return [[1.0] + [0.0] * 511 for _ in texts]
    monkeypatch.setattr(notes, "embeddings", embeddings)
    with session_scope() as db:
        job = db.scalar(select(Job).where(Job.kind == "embed_note", Job.payload["note_id"].astext == note_id))
        if outcome == "failed":
            job.payload = {**job.payload, "attempts": 2}
        job_id = job.id
    if outcome == "retrying":
        with pytest.raises(RuntimeError, match="Synthetic provider outage"):
            notes.index_note(job_id)
    else:
        notes.index_note(job_id)


@pytest.mark.parametrize("outcome", ["ready", "retrying", "failed", "deferred"])
def test_note_creation_revert_survives_indexing_and_canonical_reads(client, monkeypatch, outcome):
    note, change_id = created(client)
    assert state(change_id)["can_revert"]
    index(monkeypatch, note["id"], outcome)
    canonical = client.get("/api/v1/structure/by-core/note/" + note["id"])
    assert canonical.status_code == 200, canonical.text
    assert client.get("/api/v1/structure/records/" + canonical.json()["id"]).status_code == 200
    with session_scope() as db:
        row = db.get(Note, note["id"])
        assert row.index_state == outcome and row.revision == note["revision"]
    card = state(change_id)
    assert card["can_revert"], card
    undone = revert(client, change_id)
    assert undone.status_code == 200, undone.text
    with session_scope() as db:
        assert db.get(Note, note["id"]).archived


def test_task_creation_revert_survives_ordinary_canonical_reads(client):
    task, change_id = created(client, "task")
    assert client.get("/api/v1/structure/by-core/task/" + task["id"]).status_code == 200
    assert client.get("/api/v1/structure/records/" + task["id"]).status_code == 200
    assert state(change_id)["can_revert"]
    assert revert(client, change_id).status_code == 200


@pytest.mark.parametrize("restore_original", [False, True])
def test_indexing_does_not_hide_later_note_edits_or_revision_changes(client, monkeypatch, restore_original):
    note, change_id = created(client)
    index(monkeypatch, note["id"])
    changed = command(client, "note.update", {
        "note_id": note["id"], "expected_revision": note["revision"], "content": "Later user work",
    })["data"]
    if restore_original:
        command(client, "note.update", {
            "note_id": note["id"], "expected_revision": changed["revision"], "content": "Synthetic body",
        })
    assert not state(change_id)["can_revert"]
    assert revert(client, change_id).status_code == 409
    with session_scope() as db:
        assert not db.get(Note, note["id"]).archived


def test_indexed_note_creation_revert_still_checks_unversioned_backlinks(client, monkeypatch):
    note, change_id = created(client)
    index(monkeypatch, note["id"])
    task, _ = created(client, "task")
    with session_scope() as db:
        db.add(NoteTaskLink(note_id=note["id"], task_id=task["id"], linked=True))
        assert db.get(Note, note["id"]).revision == note["revision"]
    card = state(change_id)
    assert not card["can_revert"] and "linked" in card["revert_reason"]
    assert revert(client, change_id).status_code == 409


def test_generic_note_creation_keeps_core_revision_provenance_guard(client, monkeypatch):
    record, change_id = created(client, generic=True)
    index(monkeypatch, record["note_id"])
    assert state(change_id)["can_revert"]
    with session_scope() as db:
        note = db.get(Note, record["note_id"])
        revision = note.revision
    command(client, "note.update", {
        "note_id": record["note_id"], "expected_revision": revision, "content": "Changed through note view",
    })
    card = state(change_id)
    assert not card["can_revert"] and "another view" in card["revert_reason"]
    assert revert(client, change_id).status_code == 409
    with session_scope() as db:
        assert not db.get(StructureRecord, record["id"]).archived
