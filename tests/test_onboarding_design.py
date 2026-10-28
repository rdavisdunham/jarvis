"""Guided setup uses the same bounded schema/proposal contract as the type editor."""
from copy import deepcopy
import pytest
from sqlalchemy import select
from jarvis.db import session_scope
from jarvis.domain import DomainError
from jarvis import onboarding
from jarvis.models import Task, Note
from jarvis.structure_models import StructureRecord
from jarvis.structure_schema import Definition
from test_structure import run, definition, create, apply, propose


def state():
    with session_scope() as db:
        return onboarding.state(db, "davin")


def proposed_design():
    current = definition()
    design = {key: deepcopy(current[key]) for key in ("types", "relationships", "field_library", "type_layout")}
    client = next(t for t in design["types"] if t["id"] == "client")
    client["name"], client["plural"] = "Customer", "Customers"
    field = {"id": "region", "name": "Region", "description": "The operating region for a customer or project.",
             "kind": "text"}
    design["field_library"].append(field)
    for kind in ("client", "project"):
        next(t for t in design["types"] if t["id"] == kind)["fields"].append(
            {**field, "id": "region", "library_id": "region"})
    design["type_layout"] = [{"type_id": "client", "parent_type_id": None},
                             {"type_id": "project", "parent_type_id": "client"},
                             {"type_id": "task", "parent_type_id": "project"}]
    return current, design


def full_preview(**extra):
    current, design = proposed_design()
    return run("onboarding.preview", {"expected_revision": state()["revision"],
        "schema_revision": current["revision"], "definition": design, **extra}, key="design-turn:preview")


def test_full_design_is_linked_resumable_and_creates_no_example_records():
    saved = run("onboarding.save", {"expected_revision": 0, "design_notes":
        "Customers have projects. ABC and Transcript Intelligence are examples, not records to create."})
    current, design = proposed_design()
    project = next(t for t in design["types"] if t["id"] == "project")
    project["parent_types"] = ["client"]
    design["relationships"][0]["description"] = "An advisory link to related work, never a second Home."
    result = run("onboarding.preview", {"expected_revision": state()["revision"],
        "schema_revision": current["revision"], "definition": design}, key="design-turn:preview")
    proposal = result["proposal"]
    assert proposal["definition"] == Definition.model_validate(design).model_dump(mode="json")
    assert result["progress"]["proposal_id"] == proposal["id"]
    assert result["progress"]["design_notes"] == saved["design_notes"]
    assert "example" not in result
    assert definition()["revision"] == 1
    with pytest.raises(DomainError) as exc:
        run("structure.apply", {"proposal_id": proposal["id"], "expected_revision": 1}, key="design-turn:apply")
    assert exc.value.code == "SCHEMA_CONFIRMATION_REQUIRED"
    with pytest.raises(DomainError):
        run("onboarding.save", {"expected_revision": result["progress"]["revision"], "status": "completed"})
    apply(proposal)
    done = run("onboarding.save", {"expected_revision": result["progress"]["revision"],
                                   "status": "completed", "step": "done"})
    assert done["status"] == "completed"
    schema = definition()
    assert next(t for t in schema["types"] if t["id"] == "client")["name"] == "Customer"
    assert next(t for t in schema["types"] if t["id"] == "project")["fields"][-1]["library_id"] == "region"
    assert {r["type_id"]: r["parent_type_id"] for r in schema["type_layout"]}["task"] == "project"
    with session_scope() as db:
        assert not list(db.scalars(select(StructureRecord)))
        assert not list(db.scalars(select(Task)))
        assert not list(db.scalars(select(Note)))


@pytest.mark.parametrize("missing", ["schema_revision", "definition"])
def test_full_design_requires_its_schema_revision(missing):
    current, design = proposed_design()
    args = {"expected_revision": 0, "schema_revision": current["revision"], "definition": design}
    args.pop(missing)
    with pytest.raises(DomainError):
        run("onboarding.preview", args)
    assert state()["revision"] == 0


def test_full_design_rejects_stale_schema_and_preserves_saved_answers():
    current, design = proposed_design()
    changed = definition()
    changed["types"][0]["description"] += " Updated elsewhere."
    apply(propose(changed))
    with pytest.raises(DomainError):
        run("onboarding.preview", {"expected_revision": 0,
                                  "schema_revision": current["revision"], "definition": design})
    assert state()["revision"] == 0


@pytest.mark.parametrize("remove", ["type", "field", "library", "relationship"])
def test_generated_setup_never_silently_drops_existing_definitions(remove):
    current, design = proposed_design()
    if remove == "type":
        # No current record uses this type, so the ordinary impact checker alone would allow deletion.
        target = next(t for t in design["types"] if t["id"] == "goal")
        design["types"].remove(target)
        for t in design["types"]:
            t["parent_types"] = [x for x in t["parent_types"] if x != "goal"]
            for field in t["fields"]:
                field["target_types"] = [x for x in field.get("target_types", []) if x != "goal"]
        design["relationships"] = [r for r in design["relationships"]
                                   if "goal" not in r["source_types"] + r["target_types"]]
    elif remove == "field":
        next(t for t in design["types"] if t["id"] == "task")["fields"].pop()
    elif remove == "library":
        used = {f.get("library_id") for t in design["types"] for f in t["fields"]}
        unused = {"id": "kept-field", "name": "Kept", "description": "Saved reusable field.", "kind": "text"}
        first = definition()
        first["field_library"].append(unused)
        initial = run("structure.preview", {"expected_revision": first["revision"],
            "definition": {k: first[k] for k in ("types", "relationships", "field_library", "type_layout")}})
        apply(initial)
        current, design = proposed_design()
        design["field_library"] = [f for f in design["field_library"] if f["id"] != "kept-field"]
        assert "kept-field" not in used
    else:
        design["relationships"].pop()
    with pytest.raises(DomainError) as exc:
        run("onboarding.preview", {"expected_revision": 0,
                                  "schema_revision": current["revision"], "definition": design})
    assert exc.value.code == "SETUP_PRESERVE_DEFINITIONS"


def test_full_setup_status_mappings_preserve_current_work_and_no_duplicate_types_on_refinement():
    task = create("task", "Keep this existing task")
    current, design = proposed_design()
    kind = next(t for t in design["types"] if t["id"] == "task")
    next(s for s in kind["statuses"] if s["id"] == "open")["id"] = "ready"
    result = run("onboarding.preview", {"expected_revision": 0,
        "schema_revision": current["revision"], "definition": design,
        "status_mappings": {"task": {"open": "ready"}}})
    assert result["proposal"]["impact"]["blocking_count"] == 0
    apply(result["proposal"])
    with session_scope() as db:
        assert db.get(StructureRecord, task["id"]).status_id == "ready"
        assert db.get(Task, task["task_id"]).title == "Keep this existing task"
    run("onboarding.save", {"expected_revision": result["progress"]["revision"], "status": "completed"})
    current = definition()
    result = run("onboarding.preview", {"expected_revision": state()["revision"],
        "schema_revision": current["revision"],
        "definition": {k: current[k] for k in ("types", "relationships", "field_library", "type_layout")}})
    apply(result["proposal"])
    assert len(definition()["types"]) == len(current["types"])
    with session_scope() as db:
        assert len(list(db.scalars(select(StructureRecord)))) == 1


def test_design_answer_change_invalidates_review_and_old_snapshot_cannot_apply():
    result = full_preview()
    updated = run("onboarding.save", {"expected_revision": result["progress"]["revision"],
                                      "design_notes": "Actually add a course type instead."})
    assert "proposal_id" not in updated
    with pytest.raises(DomainError):
        run("onboarding.save", {"expected_revision": updated["revision"], "status": "completed"})
    with pytest.raises(DomainError) as exc:
        apply(result["proposal"])
    assert exc.value.code == "STALE_PROPOSAL"


def test_new_setup_preview_retires_the_previous_proposal_and_rechecks_record_snapshot():
    old = full_preview()
    current, design = proposed_design()
    revised = run("onboarding.preview", {"expected_revision": state()["revision"],
        "schema_revision": current["revision"], "definition": design})
    with pytest.raises(DomainError) as exc:
        apply(old["proposal"])
    assert exc.value.code == "STALE_PROPOSAL"
    create("client", "New real client")
    with pytest.raises(DomainError) as exc:
        apply(revised["proposal"])
    assert exc.value.code == "STALE_PROPOSAL"
