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
