from uuid import uuid4

from jarvis.db import session_scope
from jarvis.domain import execute
from jarvis.models import Task
from jarvis.saved_views import SaveView, ViewState, collection, save_view
from jarvis.tools import call_tool


def command(tool, **args):
    with session_scope() as db:
        return execute(db, "davin", str(uuid4()), tool, args)["data"]


async def test_backlog_tool_create_filter_and_bulk_update_preserve_details():
    future = await call_tool("davin", "backlog-create", 0, "task_create", {
        "title": "Future work", "status": "backlog", "notes": "Keep this",
        "due_date": "2030-01-15", "due_time": "14:00", "due_timezone": "America/Chicago",
    })
    future = future["data"]
    assert future["status"] == "backlog"
    ready = command("task.create", title="Ready work")
    assert ready["status"] == "open"
    listed = await call_tool("davin", "backlog-list", 0, "task_list", {"status": "backlog"})
    assert [t["id"] for t in listed["tasks"]] == [future["id"]]
    await call_tool("davin", "backlog-move", 0, "task_update", {
        "task_id": ready["id"], "expected_revision": ready["revision"], "status": "backlog",
    })
    command("task.batch", items=[
        {"task_id": future["id"], "expected_revision": future["revision"], "status": "open"},
        {"task_id": ready["id"], "expected_revision": ready["revision"] + 1, "status": "open"},
    ])
    with session_scope() as db:
        task = db.get(Task, future["id"])
        assert task.status == "open" and task.notes == "Keep this"
        assert task.due_date.isoformat() == "2030-01-15" and task.due_time == "14:00"
        assert task.completed_at is None


async def test_backlog_site_controls_and_saved_view():
    result = await call_tool("davin", "backlog-ui", 0, "ui_filter", {"status": "backlog"})
    assert result["ui_action"]["status"] == "backlog"
    with session_scope() as db:
        save_view(db, "davin", SaveView(
            id="backlog", name="Backlog", expected_revision=0,
            state=ViewState(status="backlog", layout="board"),
        ))
    with session_scope() as db:
        saved = collection(db, "davin")[0]
        assert saved["state"]["status"] == "backlog" and saved["state"]["layout"] == "board"
