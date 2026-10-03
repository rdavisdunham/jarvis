import copy
import pytest
from pydantic import ValidationError
from jarvis import record_contents as contents, structure
from jarvis.db import session_scope
from jarvis.domain import DomainError
from jarvis.field_library import upgrade
from jarvis.structure_models import StructureSchema
from jarvis.structure_schema import Definition
from test_organization_foundation import create, run, schema, OWNER, apply


def set_opens_as(**modes):
    d = schema()
    for t in d["types"]:
        if t["id"] in modes:
            t["opens_as"] = modes[t["id"]]
    apply(d)


def groups(parent=None):
    with session_scope() as db:
        return {r["title"]: r["opens_as"] for r in contents.browse(db, OWNER, parent_id=parent, section="groups")["items"]}


def test_default_types_have_explicit_presentation():
    modes = {t["id"]: t["opens_as"] for t in schema()["types"]}
    assert modes == {"space": "container", "area": "container", "client": "container",
                     "project": "container", "goal": "auto", "task": "item", "note": "item"}


def test_explicit_container_without_children_lists_as_group():
    set_opens_as(task="container")
    create("task", "Lonely task")
    assert groups() == {"Lonely task": "container"}


def test_explicit_item_with_children_is_not_a_group():
    c = create("client", "ABC")
    p = create("project", "Transcript Intelligence", parent_id=c["id"])
    t = create("task", "Finish the central docs", parent_id=p["id"])
    create("task", "Check examples", parent_id=t["id"])
    assert groups(p["id"]) == {}
    assert run("record.update", dict(record_id=t["id"], expected_revision=t["revision"],
                                     schema_revision=schema()["revision"], title="Finish docs"))["opens_as"] == "item"
    set_opens_as(client="item")
    assert "ABC" not in groups()
    with session_scope() as db:
        assert structure.data(db, db.get(contents.StructureRecord, c["id"]))["opens_as"] == "item"


def test_auto_keeps_inferred_rule():
    set_opens_as(task="auto", client="auto")
    create("client", "ABC")
    t = create("task", "Parent task")
    create("task", "Loose task")
    assert groups() == {"ABC": "container"}
    create("task", "Child", parent_id=t["id"])
    assert groups() == {"ABC": "container", "Parent task": "container"}


def test_upgrade_sets_defaults_only_for_untouched_default_types():
    legacy = copy.deepcopy(structure.default_definition())
    for t in legacy["types"]:
        t.pop("opens_as")
    by_id = {t["id"]: t for t in legacy["types"]}
    by_id["client"]["name"] = "Customer"
    by_id["note"]["capabilities"] = ["content", "work"]
    by_id["area"]["opens_as"] = "item"
    custom = copy.deepcopy(by_id["space"]) | {"id": "trip", "name": "Trip", "plural": "Trips"}
    custom.pop("opens_as", None)
    legacy["types"].append(custom)
    modes = {t["id"]: t["opens_as"] for t in upgrade(legacy)["types"]}
    assert modes == {"space": "container", "area": "item", "client": "auto", "project": "container",
                     "goal": "auto", "task": "item", "note": "auto", "trip": "auto"}
    assert upgrade(upgrade(legacy)) == upgrade(legacy)


def test_stored_legacy_definition_upgrades_on_load():
    schema()
    with session_scope() as db:
        row = db.get(StructureSchema, OWNER)
        legacy = copy.deepcopy(row.definition)
        for t in legacy["types"]:
            t.pop("opens_as")
        row.definition = legacy
    modes = {t["id"]: t["opens_as"] for t in schema()["types"]}
    assert modes["client"] == "container" and modes["task"] == "item" and modes["goal"] == "auto"


def test_older_client_preview_keeps_current_setting():
    set_opens_as(task="container")
    d = schema()
    for t in d["types"]:
        t.pop("opens_as")
    p = run("structure.preview", dict(expected_revision=d["revision"],
        definition={k: d[k] for k in ("types", "relationships", "field_library", "type_layout")}))
    assert {t["id"]: t["opens_as"] for t in p["definition"]["types"]}["task"] == "container"


def test_invalid_value_rejected():
    d = schema()
    d["types"][0]["opens_as"] = "folder"
    with pytest.raises(ValidationError):
        Definition.model_validate({k: d[k] for k in ("types", "relationships", "field_library", "type_layout")})
    with pytest.raises((DomainError, ValidationError)):
        apply(d)


async def test_ui_records_follows_opens_as(monkeypatch):
    from jarvis import tools

    sent = []

    async def fake(owner, device, action):
        sent.append(action)
        return {"status": "displayed"}

    monkeypatch.setattr(tools, "dispatch", fake)
    c = create("client", "ABC")
    t = create("task", "Finish the central docs", parent_id=c["id"])
    create("task", "Check examples", parent_id=t["id"])
    await tools.call_tool(OWNER, "turn", 0, "ui_records", {"record_id": c["id"]})
    await tools.call_tool(OWNER, "turn", 1, "ui_records", {"record_id": t["id"]})
    await tools.call_tool(OWNER, "turn", 2, "ui_records", {"record_id": c["id"], "open_details": True})
    assert sent[0]["parent_id"] == c["id"] and sent[0]["layout"] == "browse" and "record_id" not in sent[0]
    assert sent[1]["record_id"] == t["id"] and "parent_id" not in sent[1]
    assert sent[2]["record_id"] == c["id"] and "open_details" not in sent[2]
