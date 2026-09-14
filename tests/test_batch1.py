from uuid import uuid4

import pytest
from jarvis.db import session_scope
from jarvis.domain import DomainError, execute, preferences
from jarvis.models import OwnerSettings
from jarvis.saved_views import SaveView, ViewState, collection, save_view
from jarvis.task_tools import list_tasks


def run(db, tool, **args):
    return execute(db, "owner", str(uuid4()), tool, args)["data"]


def test_exact_note_operations_and_conflicts():
    with session_scope() as db:
        content = "  Hāyes 🐈\n\nKeep \\u1234 literal.\n"
        n = run(db, "note.create", title="Notes", content=content, tags=["home"])
        out = run(db, "note.append", note_id=n["id"], expected_revision=1, text="\nMore ✓\n")
        assert out["content"] == content + "\nMore ✓\n"
        assert out["tags"] == ["home"]
        out = run(
            db, "note.replace", note_id=n["id"], expected_revision=2, old_text="Hāyes 🐈", new_text="Hayes 🐈"
        )
        assert out["content"] == (content + "\nMore ✓\n").replace("Hāyes", "Hayes")
    with pytest.raises(DomainError), session_scope() as db:
        run(db, "note.append", note_id=n["id"], expected_revision=1, text="stale")
    with pytest.raises(DomainError, match="occur once"), session_scope() as db:
        run(db, "note.replace", note_id=n["id"], expected_revision=3, old_text="\n", new_text="x")


def test_assignee_alias_resolves_to_id_and_preserves_ambiguity():
    with session_scope() as db:
        run(db, "settings.update", preferred_name="Morgan")
        t = run(db, "task.create", title="Assigned to owner")
        result = list_tasks(db, "owner", {"assignee": "Morgan"})
        assert result["match_count"] == 1 and result["tasks"][0]["id"] == t["id"]
        assert list_tasks(db, "owner", {"assignee": "me"})["match_count"] == 1
        run(db, "actor.create", name="Morgan", kind="person")
        with pytest.raises(DomainError, match="ambiguous"):
            list_tasks(db, "owner", {"assignee": "Morgan"})


def test_saved_view_isolation_revision_and_retry():
    args = SaveView(
        id=str(uuid4()),
        name="My board",
        expected_revision=0,
        state=ViewState(tab="inbox", layout="board", tag="home"),
    )
    with session_scope() as db:
        first = save_view(db, "owner", args)
        assert save_view(db, "owner", args) == first
        assert collection(db, "other") == []
        assert preferences(db, "owner")["task_views"][0]["state"]["layout"] == "board"
    with session_scope() as db:
        with pytest.raises(DomainError):
            save_view(db, "owner", args.model_copy(update={"name": "Stale replacement"}))
        row = db.get(OwnerSettings, "owner")
        assert len(row.values["task_views"]) == 1


def test_saved_views_api_requires_auth_and_csrf(client):
    body = {"id": str(uuid4()), "name": "Board", "expected_revision": 0, "state": {"layout": "board"}}
    result = client.post("/api/v1/task-views", json=body)
    assert result.status_code == 200
    assert client.post("/api/v1/task-views", json=body).json() == result.json()
    client.headers.pop("X-CSRF-Token")
    assert client.post("/api/v1/task-views", json=body).status_code == 403
