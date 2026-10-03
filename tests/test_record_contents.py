from uuid import uuid4
import pytest
from cryptography.fernet import Fernet
from sqlalchemy import select
from jarvis import record_contents as contents, structure
from jarvis.db import session_scope
from jarvis.domain import DomainError
from jarvis.models import Task, ActionChange, now
from jarvis.structure_models import StructureRecord
from test_organization_foundation import create, run, change, schema, OWNER, apply


@pytest.fixture(autouse=True)
def crypto(monkeypatch):
    from jarvis.config import get_settings

    monkeypatch.setattr(get_settings(), "integration_encryption_key", Fernet.generate_key().decode())


def tree():
    a = create("client", "ABC")
    b = create("client", "Other")
    p = create("project", "Transcript Intelligence", parent_id=a["id"])
    t = create("task", "Write docs", parent_id=p["id"])
    sub = create("task", "Draft intro", parent_id=t["id"])
    note = create("note", "Specifications", parent_id=p["id"])
    return a, b, p, t, sub, note


def preview(row, **args):
    request = dict(
        record_id=row["id"],
        expected_revision=row["revision"],
        schema_revision=schema()["revision"],
        **{"operation": "move", **args},
    )
    with session_scope() as db:
        return request, contents.plan(db, OWNER, contents.ContentsPlan(**request))


def act(row, **args):
    request, p = preview(row, **args)
    return run("record.contents", {**request, "preview_hash": p["preview_hash"]})


def saved(identity):
    with session_scope() as db:
        return structure.data(db, db.get(StructureRecord, identity))


def test_move_with_contents_changes_one_edge_and_preserves_dates():
    a, b, p, t, sub, n = tree()
    t = change(t, values={"due_date": "2027-02-01", "planned_date": "2027-01-30"})
    result = act(p, parent_id=b["id"])
    assert result["affected_count"] == 1 and result["descendant_count"] == 3
    assert saved(t["id"])["parent_id"] == p["id"]
    assert saved(sub["id"])["parent_id"] == t["id"]
    assert saved(t["id"])["values"]["due_date"] == "2027-02-01"


def test_move_item_only_promotes_direct_children_preserving_subtree():
    a, b, p, t, sub, n = tree()
    result = act(p, parent_id=b["id"], mode="item")
    assert result["promoted_count"] == 2
    assert saved(t["id"])["parent_id"] == a["id"] and saved(n["id"])["parent_id"] == a["id"]
    assert saved(sub["id"])["parent_id"] == t["id"]


@pytest.mark.parametrize("mode", ["item", "subtree"])
def test_archive_contents_choices(mode):
    a, b, p, t, sub, n = tree()
    result = act(p, operation="archive", mode=mode)
    assert saved(p["id"])["archived"]
    assert saved(t["id"])["archived"] == (mode == "subtree")
    assert saved(sub["id"])["archived"] == (mode == "subtree")
    assert saved(n["id"])["archived"] == (mode == "subtree")
    assert not saved(a["id"])["archived"]


def test_stale_child_or_new_child_invalidates_preview_atomically():
    a, b, p, t, sub, n = tree()
    request, plan = preview(p, parent_id=b["id"], mode="item")
    create("task", "New child", parent_id=p["id"])
    with pytest.raises(DomainError, match="Contents changed"):
        run("record.contents", {**request, "preview_hash": plan["preview_hash"]})
    assert saved(p["id"])["parent_id"] == a["id"]
    assert saved(t["id"])["parent_id"] == p["id"]


def test_invalid_promoted_home_is_reported_before_writes():
    a, b, p, t, sub, n = tree()
    d = schema()
    next(x for x in d["types"] if x["id"] == "task")["parent_types"] = ["project", "task"]
    apply(d)
    request, plan = preview(p, parent_id=b["id"], mode="item")
    assert plan["issues"]
    with pytest.raises(DomainError):
        run("record.contents", {**request, "preview_hash": plan["preview_hash"]})
    assert saved(t["id"])["parent_id"] == p["id"]


def test_related_link_is_not_archived_with_contents():
    a, b, p, t, sub, n = tree()
    run(
        "record.link",
        dict(
            source_id=p["id"],
            target_id=b["id"],
            relationship_id="related",
            expected_revision=p["revision"],
            schema_revision=schema()["revision"],
        ),
    )
    p = saved(p["id"])
    act(p, operation="archive")
    assert not saved(b["id"])["archived"]


@pytest.mark.parametrize("mode,operation", [("item", "move"), ("subtree", "archive"), ("item", "archive")])
def test_grouped_revert_restores_entire_operation(mode, operation):
    from jarvis.action_history import revert

    a, b, p, t, sub, n = tree()
    args = {"operation": operation, "mode": mode}
    if operation == "move":
        args["parent_id"] = b["id"]
    result = act(p, **args)
    with session_scope() as db:
        receipt = db.scalar(
            select(ActionChange).where(
                ActionChange.tool == "record.contents", ActionChange.entity_id == p["id"]
            )
        )
        reverted = revert(db, OWNER, OWNER, receipt.id, str(uuid4()))
        assert reverted["data"]["restored"]
    assert saved(p["id"])["parent_id"] == a["id"] and not saved(p["id"])["archived"]
    assert saved(t["id"])["parent_id"] == p["id"] and not saved(t["id"])["archived"]
    assert saved(n["id"])["parent_id"] == p["id"]


def test_grouped_revert_refuses_newer_change_to_any_member():
    from jarvis.action_history import revert

    a, b, p, t, sub, n = tree()
    act(p, parent_id=b["id"], mode="item")
    change(saved(t["id"]), parent_id=None)
    with session_scope() as db:
        receipt = db.scalar(
            select(ActionChange).where(
                ActionChange.tool == "record.contents", ActionChange.entity_id == p["id"]
            )
        )
        with pytest.raises(DomainError):
            revert(db, OWNER, OWNER, receipt.id, str(uuid4()))
    assert saved(p["id"])["parent_id"] == b["id"]


def link(a, b):
    a = saved(a["id"])
    return run(
        "record.link",
        dict(
            source_id=a["id"],
            target_id=b["id"],
            relationship_id="blocks",
            expected_revision=a["revision"],
            schema_revision=schema()["revision"],
        ),
    )


def test_blockers_are_directed_acyclic_and_advisory():
    a = create("task", "Get approval")
    b = create("task", "Publish")
    c = create("task", "Announce")
    link(a, b)
    link(b, c)
    with pytest.raises(DomainError):
        link(c, a)
    with pytest.raises(DomainError):
        link(a, a)
    assert saved(b["id"])["blockers"][0]["id"] == a["id"]
    b = change(saved(b["id"]), status_id="completed")
    assert b["status_meaning"] == "completed" and saved(a["id"])["status_meaning"] == "open"
    assert not saved(c["id"])["blockers"]


def test_parent_progress_is_advisory_and_counts_each_record_once():
    a, b, p, t, sub, n = tree()
    with session_scope() as db:
        stats = contents.summary(db, db.get(StructureRecord, p["id"]), structure.ensure(db, OWNER))
        assert stats["work_total"] == 2 and stats["descendants"] == 3 and stats["work_open"] == 2
    change(p, status_id="completed")
    assert saved(t["id"])["status_meaning"] == "open"
    change(t, status_id="completed")
    change(sub, status_id="completed")
    with session_scope() as db:
        assert contents.summary(db, db.get(StructureRecord, p["id"]), structure.ensure(db, OWNER))[
            "ready_to_complete"
        ]


def test_browse_paginates_and_distinguishes_direct_subtree_related(client):
    a, b, p, t, sub, n = tree()
    path = "/api/v1/structure/browse?parent_id=" + p["id"]
    first = client.get(path + "&limit=1").json()
    assert first["total"] == 2 and first["next_offset"] == 1
    second = client.get(path + "&limit=1&offset=1").json()
    assert first["items"][0]["id"] != second["items"][0]["id"]
    assert client.get(path + "&scope=subtree").json()["total"] == 3
    assert client.get(path + "&scope=related").json()["total"] == 0
    assert client.get(path + "&scope=subtree&section=work").json()["total"] == 2


def test_foreign_home_is_not_browsable_or_mutable(client):
    from jarvis.domain import execute

    with session_scope() as db:
        other = execute(
            db,
            "other",
            str(uuid4()),
            "record.create",
            dict(type_id="client", title="Private", schema_revision=1),
        )["data"]
    assert client.get("/api/v1/structure/browse?parent_id=" + other["id"]).status_code == 404
    with pytest.raises(DomainError):
        act(other, parent_id=None)


def test_source_archive_and_move_preview_is_explicitly_local():
    a, b, p, t, sub, n = tree()
    with session_scope() as db:
        task = db.get(Task, t["task_id"])
        task.external = {"provider": "linear"}
    request, plan = preview(p, parent_id=b["id"])
    assert plan["providers"][0]["provider"] == "linear"
    assert "not deleted" in plan["source_effect"]


def test_grouped_revert_does_not_move_a_new_child():
    from jarvis.action_history import revert

    a, b, p, t, sub, n = tree()
    act(p, parent_id=b["id"])
    create("task", "New commitment", parent_id=p["id"])
    with session_scope() as db:
        receipt = db.scalar(
            select(ActionChange).where(
                ActionChange.tool == "record.contents", ActionChange.entity_id == p["id"]
            )
        )
        with pytest.raises(DomainError):
            revert(db, OWNER, OWNER, receipt.id, str(uuid4()))


def test_contents_bot_scope_includes_nested_note_and_task():
    from jarvis.bot_access import core_scopes

    a, b, p, t, sub, n = tree()
    with session_scope() as db:
        assert core_scopes(db, OWNER, "record.contents", {"record_id": p["id"]}) == {
            "tasks:write",
            "notes:write",
        }


def test_archive_subtree_preserves_already_archived_children_on_revert():
    from jarvis.action_history import revert
    a,b,p,t,sub,n=tree()
    change(n,archived=True)
    act(p,operation="archive")
    with session_scope() as db:
        receipt=db.scalar(select(ActionChange).where(ActionChange.tool=="record.contents",ActionChange.entity_id==p["id"]))
        revert(db,OWNER,OWNER,receipt.id,str(uuid4()))
    assert saved(n["id"])["archived"]
    assert not saved(t["id"])["archived"]
