import time
from uuid import uuid4

import pytest
from jarvis.db import session_scope
from jarvis.models import Job, Memory, Note, NoteEmbedding, Task, TaskReference
from jarvis.notes import chunks, index_note, search_notes, suggest_tasks
from jarvis.tools import call_tool
from jarvis.ui_control import states
from sqlalchemy import func, select


def cmd(client, tool, arguments, identity=None):
    return client.post(
        "/api/v1/commands",
        json={"command_id": identity or str(uuid4()), "tool": tool, "arguments": arguments},
    )


def note(client, content="Call the dentist."):
    r = cmd(client, "note.create", {"title": "Personal planning", "content": content})
    assert r.status_code == 200, r.text
    return r.json()["data"]


def task(client, title="Dentist"):
    return cmd(client, "task.create", {"title": title}).json()["data"]


def test_authored_notes_links_revision_archive_and_owner_checks(client):
    project = cmd(client, "project.create", {"name": "Home"}).json()["data"]
    t = task(client)
    n = note(client)
    updated = cmd(
        client,
        "note.update",
        {
            "note_id": n["id"],
            "expected_revision": 1,
            "project_id": project["id"],
            "task_ids": [t["id"]],
            "tags": ["health", "health"],
        },
    ).json()["data"]
    assert updated["tags"] == ["health"] and updated["tasks"][0]["id"] == t["id"]
    assert client.get("/api/v1/notes", params={"task_id": t["id"]}).json()["items"][0]["id"] == n["id"]
    stale = cmd(client, "note.update", {"note_id": n["id"], "expected_revision": 1, "content": "Stale"})
    assert stale.status_code == 409
    with session_scope() as db:
        other = Note(owner_id="someone-else", title="Private", content="secret", tags=[])
        db.add(other)
        db.flush()
        other_id = other.id
    assert client.get("/api/v1/notes/" + other_id).status_code == 404
    assert (
        cmd(
            client, "note.update", {"note_id": other_id, "expected_revision": 1, "content": "bad"}
        ).status_code
        == 404
    )
    archived = cmd(client, "note.update", {"note_id": n["id"], "expected_revision": 2, "archived": True})
    assert archived.status_code == 200
    assert client.get("/api/v1/notes").json()["items"] == []
    assert len(client.get("/api/v1/notes?archived=true").json()["items"]) == 1
    with session_scope() as db:
        assert db.scalar(select(func.count()).select_from(Memory)) == 0


def test_search_literal_wildcards_tags_and_pagination(client):
    n = note(client, "Check 100% coverage.")
    note(client, "Something else.")
    assert client.get("/api/v1/notes", params={"q": "%"}).json()["items"][0]["id"] == n["id"]
    assert len(client.get("/api/v1/notes", params={"q": "%"}).json()["items"]) == 1
    result = client.get("/api/v1/notes?limit=1").json()
    assert len(result["items"]) == 1 and result["next_offset"] == 1
    assert len(client.get("/api/v1/notes?limit=1&offset=1").json()["items"]) == 1


def test_extraction_is_preview_verifies_quotes_and_never_creates_tasks(client, monkeypatch):
    n = note(client, "Call the dentist. Buy printer paper.")
    monkeypatch.setattr(
        "jarvis.notes.extraction_request",
        lambda *a, **k: {
            "choices": [
                {
                    "message": {
                        "content": '{"items":[{"title":"Call dentist","evidence":"Call the dentist."},{"title":"Invented","evidence":"Not in note"}]}'
                    }
                }
            ]
        },
    )
    result = suggest_tasks("davin", n["id"])
    assert len(result["items"]) == 1
    with session_scope() as db:
        assert db.scalar(select(func.count()).select_from(Task)) == 0
    result = cmd(
        client,
        "note.tasks",
        {
            "note_id": n["id"],
            "expected_revision": 1,
            "items": [{"title": "Call dentist", "evidence": "Call the dentist."}],
        },
    ).json()["data"]
    assert result["tasks"][0]["existing"] is False
    again = cmd(
        client,
        "note.tasks",
        {
            "note_id": n["id"],
            "expected_revision": 1,
            "items": [{"title": "Call dental office", "evidence": "Call the dentist."}],
        },
    ).json()["data"]
    assert again["tasks"][0]["id"] == result["tasks"][0]["id"] and again["tasks"][0]["existing"]
    exported = client.get("/api/v1/export").json()
    assert exported["notes"][0]["id"] == n["id"]
    assert exported["note_task_links"][0]["evidence"] == "Call the dentist."


def test_extraction_invalid_quote_rolls_back_entire_batch(client):
    n = note(client)
    result = cmd(
        client,
        "note.tasks",
        {
            "note_id": n["id"],
            "expected_revision": 1,
            "items": [
                {"title": "Good", "evidence": "Call the dentist."},
                {"title": "Bad", "evidence": "Invented"},
            ],
        },
    )
    assert result.status_code == 400
    with session_scope() as db:
        assert db.scalar(select(func.count()).select_from(Task)) == 0


def test_extraction_reply_cannot_use_note_edited_during_cloud_request(client, monkeypatch):
    n = note(client)

    def model(*a, **k):
        cmd(client, "note.update", {"note_id": n["id"], "expected_revision": 1, "content": "Already done."})
        return {
            "choices": [
                {
                    "message": {
                        "content": '{"items":[{"title":"Call dentist","evidence":"Call the dentist."}]}'
                    }
                }
            ]
        }

    monkeypatch.setattr("jarvis.notes.extraction_request", model)
    from jarvis.domain import DomainError

    with pytest.raises(DomainError, match="note changed"):
        suggest_tasks("davin", n["id"])


def test_revisioned_embeddings_discard_stale_result_and_keyword_fallback(client, monkeypatch):
    n = note(client)
    with session_scope() as db:
        job_id = db.scalar(select(Job.id).where(Job.kind == "embed_note"))

    def embedding(*a, **k):
        cmd(client, "note.update", {"note_id": n["id"], "expected_revision": 1, "content": "Plan a picnic"})
        return [[1.0, 0.0]]

    monkeypatch.setattr("jarvis.notes.embeddings", embedding)
    index_note(job_id)
    with session_scope() as db:
        assert db.get(Job, job_id).status == "cancelled"
        assert db.scalar(select(func.count()).select_from(NoteEmbedding)) == 0
    monkeypatch.setattr(
        "jarvis.notes.embeddings", lambda *a, **k: (_ for _ in ()).throw(ValueError("offline"))
    )
    result = search_notes("davin", "picnic")
    assert result["mode"] == "keyword_fallback" and result["items"][0]["id"] == n["id"]


def test_semantic_index_search_and_archive_during_query(client, monkeypatch):
    n = note(client, "Put the tent in the car.")
    monkeypatch.setattr("jarvis.notes.embeddings", lambda *a, **k: [[1.0, 0.0]])
    with session_scope() as db:
        job = db.scalar(select(Job.id).where(Job.kind == "embed_note"))
    index_note(job)
    assert search_notes("davin", "camping")["items"][0]["id"] == n["id"]

    def delayed(*a, **k):
        cmd(client, "note.update", {"note_id": n["id"], "expected_revision": 1, "archived": True})
        return [[1.0, 0.0]]

    monkeypatch.setattr("jarvis.notes.embeddings", delayed)
    assert search_notes("davin", "camping")["items"] == []


def test_note_cannot_link_other_owners_records(client):
    with session_scope() as db:
        other = Task(owner_id="other", title="Private")
        db.add(other)
        db.flush()
        tid = other.id
    response = cmd(client, "note.create", {"title": "Test", "task_ids": [tid]})
    assert response.status_code == 404
    assert client.get("/api/v1/notes").json()["items"] == []


def test_batch_edit_is_atomic_revision_checked_and_retry_safe(client):
    a, b = task(client, "A"), task(client, "B")
    items = [
        {"task_id": t["id"], "expected_revision": t["revision"], "due_date": "2026-09-18"} for t in (a, b)
    ]
    cmd(client, "task.update", {"task_id": b["id"], "expected_revision": 1, "title": "Updated B"})
    assert cmd(client, "task.batch", {"items": items}).status_code == 409
    assert client.get("/api/v1/tasks/" + a["id"]).json()["due_date"] is None
    items[1]["expected_revision"] = 2
    first = cmd(client, "task.batch", {"items": items}, "batch-1")
    assert first.status_code == 200
    retry = cmd(client, "task.batch", {"items": items}, "batch-1")
    assert retry.json() == first.json()
    assert len(first.json()["data"]["tasks"]) == 2


async def test_context_selected_recent_and_ambiguous_targets_keep_current_revisions(client):
    conv = client.post("/api/v1/conversations", json={"private": False}).json()["id"]
    a, b = task(client, "Call dentist"), task(client, "Call dentist")
    states[("davin", "context-device")] = {
        "at": time.monotonic(),
        "context": {"selected_task_ids": [a["id"], b["id"]], "view": "all"},
    }
    result = await call_tool(
        "davin",
        "turn",
        0,
        "task_resolve",
        {"scope": "selected"},
        device="context-device",
        conversation_id=conv,
    )
    assert result["ambiguous"] and len(result["tasks"]) == 2
    await call_tool("davin", "turn", 1, "task_get", {"task_id": a["id"]}, conversation_id=conv)
    cmd(client, "task.update", {"task_id": a["id"], "expected_revision": 1, "title": "Book dentist"})
    result = await call_tool("davin", "turn", 2, "task_resolve", {"scope": "recent"}, conversation_id=conv)
    assert result["tasks"][0]["revision"] == 2 and result["tasks"][0]["title"] == "Book dentist"
    other_conv = client.post("/api/v1/conversations", json={"private": False}).json()["id"]
    assert (
        await call_tool("davin", "turn", 3, "task_resolve", {"scope": "recent"}, conversation_id=other_conv)
    )["tasks"] == []


async def test_private_task_references_are_not_persisted(client):
    conv = client.post("/api/v1/conversations", json={"private": True}).json()["id"]
    t = task(client)
    await call_tool("davin", "private", 0, "task_get", {"task_id": t["id"]}, conversation_id=conv)
    with session_scope() as db:
        assert db.scalar(select(func.count()).select_from(TaskReference)) == 0


def test_chunks_cover_long_note_and_overlap():
    content = "x" * 5000
    result = chunks("Title", content, ["tag"])
    assert len(result) == 4 and all(len(c) <= 1810 for c in result)


def test_cost_tracking_disabled_writes_nothing_and_ignores_old_holds(client, monkeypatch):
    from jarvis import budget
    from jarvis.config import get_settings
    from jarvis.models import BudgetReservation, Usage

    with session_scope() as db:
        budget.reserve(db, "davin", "old", 149, "test")
        budget.close(db, "davin", "old", uncertain=True)
    monkeypatch.setattr(get_settings(), "cost_tracking_enabled", False)
    with session_scope() as db:
        budget.reserve(db, "davin", "new", 1000, "test", optional=True)
        budget.ensure_room(db, "davin", "new", 1000)
        budget.record_usage(db, "davin", "new", "report", "test", {}, 100)
        budget.touch(db, "davin", "old")
        budget.close(db, "davin", "old")
        budget.expire_abandoned(db, "davin")
        assert db.get(BudgetReservation, "new") is None
        assert db.get(BudgetReservation, "old").state == "uncertain"
        assert db.scalar(select(func.count()).select_from(Usage)) == 0
        assert budget.holds(db, "davin") == []
        assert budget.summary(db, "davin")["spent_usd"] is None
    assert client.get("/api/v1/bootstrap").json()["budget"]["tracking_enabled"] is False


@pytest.mark.asyncio
async def test_semantic_note_search_keeps_linked_task_scope(client, monkeypatch):
    first, second = task(client, "First"), task(client, "Second")
    wanted, unrelated = note(client), note(client)
    for n, t in [(wanted, first), (unrelated, second)]:
        assert cmd(client, "note.update", {
            "note_id": n["id"], "expected_revision": 1, "task_ids": [t["id"]]
        }).status_code == 200
    monkeypatch.setattr("jarvis.notes.embeddings", lambda *a: [[1.0, 0.0]])
    response = client.get("/api/v1/notes/search", params={"q": "dentist", "task_id": first["id"]})
    assert response.status_code == 200
    assert [n["id"] for n in response.json()["items"]] == [wanted["id"]]
    result = await call_tool("davin", str(uuid4()), 0, "note_search", {
        "semantic": True, "query": "dentist", "task_id": first["id"]
    })
    assert [n["id"] for n in result["items"]] == [wanted["id"]]
