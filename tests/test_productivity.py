from uuid import uuid4

from jarvis.db import session_scope
from jarvis.models import Goal, Project, Space
from jarvis.tools import registry


def cmd(client, tool, **args):
    r = client.post("/api/v1/commands", json={"command_id": str(uuid4()), "tool": tool, "arguments": args})
    assert r.status_code == 200, r.text
    return r.json()["data"]


def reject(client, tool, **args):
    return client.post("/api/v1/commands", json={"command_id": str(uuid4()), "tool": tool, "arguments": args})


def test_many_to_many_revisions_and_outcome_independence(client):
    g1 = cmd(client, "goal.create", name="Five clients", metric_target=5, metric_current=1)
    g2 = cmd(client, "goal.create", name="Reduce support load")
    p1 = cmd(client, "project.create", name="Website", goal_ids=[g1["id"], g2["id"]])
    p2 = cmd(client, "project.create", name="Referrals", goal_ids=[g1["id"]])
    graph = client.get("/api/v1/organization").json()
    a = next(g for g in graph["goals"] if g["id"] == g1["id"])
    assert set(a["project_ids"]) == {p1["id"], p2["id"]}
    assert a["progress"] == 0.2 and a["revision"] == 3
    assert (
        reject(client, "goal.update", goal_id=g1["id"], expected_revision=1, project_ids=[]).status_code
        == 409
    )
    cmd(client, "project.update", project_id=p1["id"], expected_revision=1, status="completed")
    a = next(g for g in client.get("/api/v1/organization").json()["goals"] if g["id"] == g1["id"])
    assert a["status"] == "planned" and a["progress"] == 0.2
    cmd(client, "goal.update", goal_id=g1["id"], expected_revision=a["revision"], project_ids=[p2["id"]])
    p = next(p for p in client.get("/api/v1/projects").json()["items"] if p["id"] == p1["id"])
    assert p["goal_ids"] == [g2["id"]] and p["status"] == "completed"
    assert p["revision"] == 3


def test_home_and_assignment_follow_project_without_changing_deadline(client):
    graph = client.get("/api/v1/organization").json()
    personal, business = graph["spaces"]
    area = cmd(client, "area.create", name="Health", space_id=personal["id"])
    project = cmd(client, "project.create", name="Training", area_id=area["id"])
    task = cmd(
        client,
        "task.create",
        title="Register",
        project_id=project["id"],
        planned_date="2030-01-02",
        due_date="2030-01-10",
        assignee="Eri",
    )
    assert task["space_id"] == personal["id"] and task["assignee_id"]
    cmd(
        client,
        "project.update",
        project_id=project["id"],
        expected_revision=1,
        space_id=business["id"],
        area_id=None,
    )
    task = client.get("/api/v1/tasks/" + task["id"]).json()
    assert task["space_id"] == business["id"] and task["area_id"] is None
    assert task["due_date"] == "2030-01-10" and task["planned_date"] == "2030-01-02"
    cmd(
        client,
        "task.update",
        task_id=task["id"],
        expected_revision=task["revision"],
        planned_date="2030-01-05",
    )
    assert client.get("/api/v1/tasks/" + task["id"]).json()["due_date"] == "2030-01-10"
    assert (
        reject(
            client, "task.create", title="Mismatch", area_id=area["id"], space_id=business["id"]
        ).status_code
        == 400
    )
    cmd(client, "space.update", space_id=personal["id"], expected_revision=1, name="Private")
    assert len(client.get("/api/v1/organization").json()["spaces"]) == 2


def test_goal_cycles_owner_boundaries_and_atomic_links(client):
    one = cmd(client, "goal.create", name="Long term")
    two = cmd(client, "goal.create", name="This month", parent_goal_id=one["id"])
    assert (
        reject(
            client, "goal.update", goal_id=one["id"], expected_revision=1, parent_goal_id=two["id"]
        ).status_code
        == 400
    )
    with session_scope() as db:
        foreign = Space(owner_id="other", name="Private")
        goal = Goal(owner_id="other", name="Hidden")
        project = Project(owner_id="other", name="Hidden project")
        db.add_all([foreign, goal, project])
        db.flush()
        ids = foreign.id, goal.id, project.id
    assert reject(client, "goal.create", name="Bad", space_id=ids[0]).status_code == 404
    assert reject(client, "project.create", name="Bad", goal_ids=[one["id"], ids[1]]).status_code == 404
    assert (
        reject(
            client, "goal.update", goal_id=one["id"], expected_revision=1, project_ids=[ids[2]]
        ).status_code
        == 404
    )
    graph = client.get("/api/v1/organization").json()
    assert not graph["projects"] and len(graph["goals"]) == 2


def test_connected_notes_backlinks_and_export(client):
    g = cmd(client, "goal.create", name="Outcome")
    p = cmd(client, "project.create", name="Initiative")
    n1 = cmd(client, "note.create", title="Research", content="Call the client.")
    n2 = cmd(
        client,
        "note.create",
        title="Meeting",
        goal_ids=[g["id"]],
        project_ids=[p["id"]],
        related_note_ids=[n1["id"]],
    )
    assert n2["goals"][0]["id"] == g["id"] and n2["projects"][0]["id"] == p["id"]
    assert client.get("/api/v1/notes/" + n1["id"]).json()["backlinks"][0]["id"] == n2["id"]
    assert client.get("/api/v1/notes", params={"project_id": p["id"]}).json()["items"][0]["id"] == n2["id"]
    graph = client.get("/api/v1/organization").json()
    assert graph["goals"][0]["notes"][0]["id"] == n2["id"]
    assert graph["projects"][0]["notes"][0]["id"] == n2["id"]
    export = client.get("/api/v1/export").json()
    assert export["note_goal_links"] and export["note_note_links"]
    assert (
        reject(
            client, "note.update", note_id=n1["id"], expected_revision=1, related_note_ids=[n1["id"]]
        ).status_code
        == 400
    )
    cmd(client, "note.update", note_id=n2["id"], expected_revision=1, related_note_ids=[])
    assert not client.get("/api/v1/notes/" + n1["id"]).json()["backlinks"]


def test_productivity_commands_replay_and_registered_tools(client):
    args = {"command_id": str(uuid4()), "tool": "goal.create", "arguments": {"name": "Outcome"}}
    first = client.post("/api/v1/commands", json=args).json()
    assert client.post("/api/v1/commands", json=args).json() == first
    assert len(client.get("/api/v1/organization").json()["goals"]) == 1
    names = {r["name"] for r in registry()}
    assert {
        "organization_list",
        "goal_create",
        "goal_update",
        "area_create",
        "space_update",
        "actor_create",
    } <= names
    assert (
        reject(
            client, "project.create", name="Impossible", start_date="2030-02-01", target_date="2030-01-01"
        ).status_code
        == 400
    )


def test_archive_preserves_links_history_and_rejects_new_assignments(client):
    g = cmd(client, "goal.create", name="Outcome")
    p = cmd(client, "project.create", name="Work", goal_ids=[g["id"]])
    task = cmd(client, "task.create", title="Work", project_id=p["id"])
    cmd(client, "goal.update", goal_id=g["id"], expected_revision=2, archived=True)
    assert reject(client, "project.create", name="Other", goal_ids=[g["id"]]).status_code == 400
    cmd(client, "project.update", project_id=p["id"], expected_revision=1, archived=True)
    assert client.get("/api/v1/tasks/" + task["id"]).json()["status"] == "open"
    assert client.get("/api/v1/organization").json()["goals"][0]["project_ids"] == [p["id"]]


def test_planning_date_calendar_does_not_become_deadline_or_booking(client):
    task = cmd(
        client, "task.create", title="Draft proposal", planned_date="2030-01-02", due_date="2030-01-10"
    )
    entries = client.get("/api/v1/calendar?start=2030-01-01&end=2030-01-12").json()["items"]
    own = [e for e in entries if e["task_id"] == task["id"]]
    assert {e["timing"]: e["date"] for e in own} == {"planned": "2030-01-02", "deadline": "2030-01-10"}
    planned = next(e for e in own if e["timing"] == "planned")
    assert planned["at"] is None and planned["busy"] is False
    assert client.get("/api/v1/tasks/" + task["id"]).json()["due_time"] is None


def test_notes_scopes_apply_before_pagination_and_semantic_fallback(client, monkeypatch):
    from jarvis import notes

    def unavailable(*_):
        raise RuntimeError("Synthetic provider outage")

    monkeypatch.setattr(notes, "embeddings", unavailable)
    space = cmd(client, "space.create", name="Private work")
    area = cmd(client, "area.create", name="Operations", space_id=space["id"])
    goal = cmd(client, "goal.create", name="Outcome")
    project = cmd(client, "project.create", name="Project", area_id=area["id"], goal_ids=[goal["id"]])
    first = cmd(
        client, "note.create", title="Shared context", content="Planning evidence", project_id=project["id"]
    )
    second = cmd(client, "note.create", title="Other context", content="Planning evidence")
    for key, value in (("space_id", space["id"]), ("area_id", area["id"]), ("goal_id", goal["id"])):
        items = client.get("/api/v1/notes", params={key: value, "limit": 1}).json()["items"]
        assert [n["id"] for n in items] == [first["id"]]
        result = client.get("/api/v1/notes/search", params={key: value, "q": "Planning"}).json()
        assert result["mode"] == "keyword_fallback" and [n["id"] for n in result["items"]] == [first["id"]]
    assert second["id"] != first["id"]


async def test_assistant_reads_scoped_tasks_and_notes(client):
    from jarvis.tools import call_tool

    goal = cmd(client, "goal.create", name="Outcome")
    project = cmd(client, "project.create", name="Project", goal_ids=[goal["id"]])
    task = cmd(client, "task.create", title="Scoped action", project_id=project["id"])
    cmd(client, "task.create", title="Unrelated action")
    note = cmd(client, "note.create", title="Scoped note", goal_ids=[goal["id"]])
    tasks = await call_tool("davin", str(uuid4()), 0, "task_list", {"goal_id": goal["id"]})
    assert [t["id"] for t in tasks["tasks"]] == [task["id"]]
    notes = await call_tool("davin", str(uuid4()), 1, "note_search", {"goal_id": goal["id"]})
    assert [n["id"] for n in notes["items"]] == [note["id"]]
