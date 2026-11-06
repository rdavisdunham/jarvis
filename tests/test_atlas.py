from fastapi.testclient import TestClient
from jarvis.api import app
from jarvis import atlas
from jarvis.db import session_scope
from jarvis.models import BotCredential, Task, WorkspaceMember, now
from test_accounts import client_for, command, shared
from test_external_agents import key
from test_organization_foundation import OWNER, apply, create, run, schema


def tree():
    c = create("client", "ABC")
    p = create("project", "Transcript Intelligence", parent_id=c["id"])
    t = create("task", "Finish the central docs", parent_id=p["id"], values={"due_date": "2026-01-02"})
    s = create("task", "Check examples", parent_id=t["id"])
    n = create("note", "Discovery notes", parent_id=p["id"])
    g = create("goal", "Improve delivery")
    run("record.link", {"source_id": p["id"], "target_id": g["id"], "relationship_id": "supports",
                        "expected_revision": p["revision"], "schema_revision": schema()["revision"]})
    return c, p, t, s, n, g


def skeleton(**args):
    with session_scope() as db:
        return atlas.skeleton(db, OWNER, **args)


def test_skeleton_has_homes_state_links_and_resolved_presentation(client):
    c, p, t, s, n, g = tree()
    data = client.get("/api/v1/structure/atlas").json()
    rows = {r["id"]: r for r in data["records"]}
    assert rows[t["id"]]["parent_id"] == p["id"] and rows[s["id"]]["parent_id"] == t["id"]
    assert rows[c["id"]]["opens_as"] == "container" and rows[t["id"]]["opens_as"] == "item"
    assert rows[t["id"]]["work"] and rows[t["id"]]["status_meaning"] == "open"
    assert rows[t["id"]]["due_date"] == "2026-01-02" and not rows[n["id"]]["work"]
    assert rows[c["id"]]["depth"] == 0 and rows[g["id"]]["depth"] == 0
    assert rows[s["id"]]["depth"] == 3
    assert data["links"] == [{**data["links"][0], "source_id": p["id"], "target_id": g["id"], "behavior": "related", "label": "Supports"}]
    assert {t_["id"] for t_ in data["types"]} >= {"client", "project", "task", "note", "goal"}
    assert data["truncated"] is False and data["total"] == 6
    assert "body" not in rows[t["id"]] and "values" not in rows[t["id"]]


def test_skeleton_excludes_archived_and_templates_and_is_breadth_first_bounded():
    c, p, t, s, n, g = tree()
    run("record.update", dict(record_id=n["id"], expected_revision=n["revision"], schema_revision=schema()["revision"], archived=True))
    with session_scope() as db:
        db.get(Task, s["task_id"]).is_template = True
    ids = [r["id"] for r in skeleton()["records"]]
    assert n["id"] not in ids and s["id"] not in ids
    capped = skeleton(limit=3)
    assert capped["truncated"] and len(capped["records"]) == 3
    # The top of the hierarchy survives truncation; deeper levels go first.
    assert {r["id"] for r in capped["records"]} == {c["id"], g["id"], p["id"]}
    assert all(r["parent_id"] in {None, *[x["id"] for x in capped["records"]]} for r in capped["records"])
    assert all(link["source_id"] != t["id"] for link in capped["links"])


def test_skeleton_is_scoped_to_the_workspace(client):
    tree()
    other = client_for("stranger")
    assert other.get("/api/v1/structure/atlas").json()["records"] == []
    assert client.get("/api/v1/structure/atlas?limit=5001").status_code == 422


def test_viewer_reads_and_revoked_member_cannot(client):
    guest = client_for("guest", "guest@example.test")
    w = shared(client, guest)
    row = command(client, "record.create", type_id="client", title="Shared client", schema_revision=1)
    with session_scope() as db:
        db.get(WorkspaceMember, (w["id"], "guest")).role = "viewer"
    seen = guest.get("/api/v1/structure/atlas")
    assert seen.status_code == 200 and row["id"] in [r["id"] for r in seen.json()["records"]]
    with session_scope() as db:
        db.get(WorkspaceMember, (w["id"], "guest")).active = False
    response = guest.get("/api/v1/structure/atlas")
    assert response.status_code == 403 and "Shared client" not in response.text


def test_bot_scope_and_revocation(client):
    c, p, t, s, n, g = tree()
    external = TestClient(app)
    _, denied = key(client, ["tasks:write"])
    assert external.get("/api/v1/external/structure/atlas", headers=denied).status_code == 403
    credential, headers = key(client, ["records:read"])
    data = external.get("/api/v1/external/structure/atlas", headers=headers).json()
    rows = {r["id"]: r for r in data["records"]}
    # records:read sees the skeleton but not task dates without tasks:read.
    assert rows[t["id"]]["title"] == "Finish the central docs" and rows[t["id"]]["due_date"] is None
    _, full = key(client, ["records:read", "tasks:read"])
    assert {r["id"]: r for r in external.get("/api/v1/external/structure/atlas", headers=full).json()["records"]}[t["id"]]["due_date"] == "2026-01-02"
    assert external.get("/api/v1/structure/atlas", headers=headers).status_code == 401
    with session_scope() as db:
        db.get(BotCredential, credential["id"]).revoked_at = now()
    assert external.get("/api/v1/external/structure/atlas", headers=headers).status_code == 401


def test_placement_preview_lists_blocking_records_with_titles_and_homes():
    c, p, t, s, n, g = tree()
    d = schema()
    for item in d["types"]:
        if item["id"] == "task":
            item["parent_types"] = [x for x in item["parent_types"] if x != "task"]
    proposal = run("structure.preview", dict(expected_revision=d["revision"],
        definition={k: d[k] for k in ("types", "relationships", "field_library", "type_layout")}))
    placement = [i for i in proposal["impact"]["issues"] if i.get("kind") == "placement"]
    assert placement == [{"record_id": s["id"], "kind": "placement", "type_id": "task", "title": "Check examples",
                          "archived": False, "home": {"id": t["id"], "title": "Finish the central docs", "type_id": "task"},
                          "message": "Move this record before disallowing its parent type."}]
    assert proposal["impact"]["blocking_count"] >= 1
    # Moving the record out unblocks the change.
    run("record.update", dict(record_id=s["id"], expected_revision=s["revision"], schema_revision=d["revision"], parent_id=p["id"]))
    apply(d)


async def test_ui_records_can_fly_the_atlas(monkeypatch):
    from jarvis import tools

    sent = []

    async def fake(owner, device, action):
        sent.append(action)
        return {"status": "displayed"}

    monkeypatch.setattr(tools, "dispatch", fake)
    c = create("client", "ABC")
    await tools.call_tool(OWNER, "turn", 0, "ui_records", {"record_id": c["id"], "layout": "atlas"})
    assert sent[0]["record_id"] == c["id"] and sent[0]["layout"] == "atlas" and "parent_id" not in sent[0]
    assert "atlas" in tools.READ_TOOLS["ui_records"]["parameters"]["properties"]["layout"]["enum"]


def test_finished_filter_precedes_budget_and_keeps_only_needed_ancestor_context():
    c, p, t, s, n, g = tree()
    with session_scope() as db:
        db.get(Task, t["task_id"]).status = "cancelled"
    data = skeleton()
    rows = {r["id"]: r for r in data["records"]}
    assert rows[t["id"]]["context_only"] and rows[t["id"]]["opens_as"] == "container"
    assert rows[s["id"]]["parent_id"] == t["id"] and not rows[s["id"]]["context_only"]
    assert not rows[t["id"]]["work"]
    assert data["total"] == 5 and data["context_count"] == 1
    assert not {r["id"]: r for r in skeleton(show_finished=True)["records"]}[t["id"]]["context_only"]
    with session_scope() as db:
        db.get(Task, s["task_id"]).status = "completed"
    assert t["id"] not in {r["id"] for r in skeleton()["records"]}
    capped = skeleton(limit=4)
    assert capped["total"] == 4 and not capped["truncated"]
    assert {r["id"] for r in capped["records"]} == {c["id"], p["id"], n["id"], g["id"]}


def test_archived_parent_preserves_active_child_and_archive_lookup(client):
    c, p, t, s, n, g = tree()
    with session_scope() as db:
        task = db.get(Task, t["task_id"])
        task.status = "completed"
        task.archived = True
    rows = {r["id"]: r for r in skeleton()["records"]}
    assert rows[t["id"]]["context_only"] and rows[t["id"]]["archived"]
    assert rows[s["id"]]["parent_id"] == t["id"]
    archived = client.get("/api/v1/structure/atlas?archived=true").json()
    assert {r["id"] for r in archived["records"] if not r["context_only"]} == {t["id"]}
    browse = client.get(f'/api/v1/structure/browse?parent_id={p["id"]}&status=active').json()
    context = next(r for r in browse["items"] if r["id"] == t["id"])
    assert context["context_only"] and context["opens_as"] == "container"
    inside = client.get(f'/api/v1/structure/browse?parent_id={t["id"]}&status=active').json()
    assert s["id"] in {r["id"] for r in inside["items"]}


def test_historical_completion_summary_survives_task_archive():
    from jarvis.record_contents import summary
    from jarvis.structure_models import StructureRecord
    from jarvis import structure
    c, p, t, s, n, g = tree()
    with session_scope() as db:
        for task_id in (t["task_id"], s["task_id"]):
            task = db.get(Task, task_id)
            task.status = "completed"
            task.archived = True
        structure.sync_core_records(db, OWNER, structure.ensure(db, OWNER))
        result = summary(db, db.get(StructureRecord, p["id"]), structure.ensure(db, OWNER))
        assert result["work_done"] == 2 and result["work_total"] == 2 and result["work_open"] == 0


def test_explicit_terminal_status_filters_before_limit_and_preserves_non_task_context(client):
    c, p, t, s, n, g = tree()
    with session_scope() as db:
        db.get(Task, t["task_id"]).status = "completed"
        db.get(Task, s["task_id"]).status = "cancelled"
    data = client.get("/api/v1/structure/atlas?show_finished=true&status=cancelled").json()
    rows = {r["id"]: r for r in data["records"]}
    assert rows[t["id"]]["context_only"] and not rows[s["id"]]["context_only"]
    assert data["total"] == 5
    assert not any(r["work"] and r["type_id"] == "task" and r["status_meaning"] != "cancelled" for r in data["records"])


def test_root_groups_keep_hidden_task_home_as_navigation_context(client):
    parent = create("task", "Finished root home")
    child = create("task", "Still needs work", parent_id=parent["id"])
    with session_scope() as db:
        task = db.get(Task, parent["task_id"])
        task.status = "completed"
        task.archived = True
    data = client.get("/api/v1/structure/browse?section=groups&status=active").json()
    assert data["total"] == 0 and data["context_count"] == 1
    assert data["items"][0]["id"] == parent["id"] and data["items"][0]["context_only"]
    children = client.get(f'/api/v1/structure/browse?parent_id={parent["id"]}&status=active').json()
    assert children["items"][0]["id"] == child["id"]


def test_completed_organizational_container_remains_visible_without_active_children(client):
    project = create("project", "Finished project context")
    task = create("task", "Finished ordinary task")
    with session_scope() as db:
        db.get(Task, project["task_id"]).status = "completed"
        db.get(Task, task["task_id"]).status = "completed"
    rows = {r["id"]: r for r in skeleton()["records"]}
    assert project["id"] in rows and not rows[project["id"]]["context_only"]
    assert task["id"] not in rows
    browse = client.get("/api/v1/structure/browse?status=active").json()
    assert project["id"] in {r["id"] for r in browse["items"]}
    assert task["id"] not in {r["id"] for r in browse["items"]}


def test_explicit_custom_container_work_type_remains_visible(client):
    from jarvis.structure_models import StructureSchema
    custom = create("task", "A custom organizational container")
    with session_scope() as db:
        definition = db.get(StructureSchema, OWNER)
        import copy
        value = copy.deepcopy(definition.definition)
        task_type = next(t for t in value["types"] if t["id"] == "task")
        task_type["opens_as"] = "container"
        definition.definition = value
        db.get(Task, custom["task_id"]).status = "completed"
    rows = {r["id"]: r for r in skeleton()["records"]}
    assert custom["id"] in rows and not rows[custom["id"]]["context_only"]


def test_browse_counts_follow_visibility_without_losing_historical_progress(client):
    group = create("client", "Counts")
    cancelled = create("task", "Canceled child", parent_id=group["id"])
    completed = create("task", "Completed child", parent_id=group["id"])
    with session_scope() as db:
        db.get(Task, cancelled["task_id"]).status = "cancelled"
        db.get(Task, completed["task_id"]).status = "completed"
    active = client.get(f'/api/v1/structure/browse?parent_id={group["id"]}&status=active').json()
    assert active["total"] == active["parent"]["contents"]["direct"] == 0
    assert active["parent"]["contents"]["descendants"] == 0
    assert active["parent"]["contents"]["work_total"] == 2
    assert active["parent"]["contents"]["work_done"] == 1
    finished = client.get(f'/api/v1/structure/browse?parent_id={group["id"]}&status=all').json()
    assert finished["total"] == finished["parent"]["contents"]["direct"] == 2
    child = create("task", "Active nested child", parent_id=cancelled["id"])
    context = client.get(f'/api/v1/structure/browse?parent_id={group["id"]}&status=active').json()
    assert context["parent"]["contents"]["direct"] == context["total"] == 0
    assert context["parent"]["contents"]["context_direct"] == context["context_count"] == 1
    assert context["parent"]["contents"]["descendants"] == 1
    assert context["items"][0]["contents"]["direct"] == 1
    with session_scope() as db:
        db.get(Task, completed["task_id"]).archived = True
    archive = client.get(f'/api/v1/structure/browse?parent_id={group["id"]}&archived=true&status=active').json()
    assert archive["total"] == archive["parent"]["contents"]["direct"] == 1
    assert archive["parent"]["contents"]["work_total"] == 3
    assert archive["parent"]["contents"]["work_done"] == 1


def test_atlas_progress_counts_finished_work_like_browse(client):
    # The Atlas loads without finished records; its progress must still match Browse ("100/155 complete").
    c, p, t, s, n, g = tree()
    shipped = create("task", "Shipped", parent_id=p["id"])
    filed = create("task", "Filed outcome", parent_id=p["id"])
    shelved = create("task", "Shelved open work", parent_id=p["id"])
    with session_scope() as db:
        db.get(Task, s["task_id"]).status = "completed"
        db.get(Task, shipped["task_id"]).status = "completed"
        outcome = db.get(Task, filed["task_id"])
        outcome.status, outcome.archived = "completed", True
        db.get(Task, shelved["task_id"]).archived = True
    rows = {r["id"]: r for r in client.get("/api/v1/structure/atlas").json()["records"]}
    assert shipped["id"] not in rows and filed["id"] not in rows
    counts = lambda r: (r["work_done"], r["work_open"], r["work_total"])  # noqa: E731
    # Under the project: t open; s, shipped and the filed outcome done; archived open work drops out.
    assert counts(rows[p["id"]]) == (3, 1, 4)
    assert counts(rows[c["id"]]) == (3, 2, 5)
    assert counts(rows[g["id"]]) == (0, 0, 0)
    for record in (c, p):
        browse = client.get(f'/api/v1/structure/browse?parent_id={record["id"]}&status=active').json()["parent"]["contents"]
        assert (browse["work_done"], browse["work_open"], browse["work_total"]) == counts(rows[record["id"]])
    finished = {r["id"]: r for r in client.get("/api/v1/structure/atlas?show_finished=true").json()["records"]}
    assert counts(finished[p["id"]]) == counts(rows[p["id"]])
