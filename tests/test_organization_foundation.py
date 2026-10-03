import copy
from uuid import uuid4
import pytest
from sqlalchemy import select
from jarvis.db import session_scope
from jarvis.domain import execute, DomainError
from jarvis import structure
from jarvis.models import Task
from jarvis.structure_models import StructureRecord

OWNER = "davin"

def run(tool, args):
    with session_scope() as db:
        return execute(db, OWNER, str(uuid4()), tool, args)["data"]

def schema():
    with session_scope() as db:
        return copy.deepcopy(structure.schema_data(db, OWNER))

def create(kind, title, **args):
    return run("record.create", dict(type_id=kind, title=title, schema_revision=schema()["revision"], **args))

def change(row, **args):
    return run("record.update", dict(record_id=row["id"], expected_revision=row["revision"],
                                    schema_revision=schema()["revision"], **args))

def apply(d):
    p = run("structure.preview", dict(expected_revision=d["revision"],
        definition={k:d[k] for k in ("types", "relationships", "field_library", "type_layout")}))
    return run("structure.apply", dict(proposal_id=p["id"], expected_revision=p["schema_revision"]))

def test_projects_are_actionable_with_independent_do_due_and_timeline():
    p = create("project", "Transcript Intelligence", values={
        "planned_date":"2027-01-03", "due_date":"2027-01-09",
        "start_date":"2027-01-01", "target_date":"2027-01-08"})
    assert p["task_id"] and {"work","timeline"} <= set(p["capabilities"])
    assert p["values"]["planned_date"] != p["values"]["due_date"]
    assert change(p, status_id="completed")["status_meaning"] == "completed"

def test_library_edit_propagates_definition_but_not_values():
    d=schema()
    fields={t["id"]:next((f for f in t["fields"] if f["binding"]=="priority"),None) for t in d["types"]}
    key=fields["task"]["library_id"]
    assert key == fields["project"]["library_id"]
    for f in d["field_library"]:
        if f["id"]==key: f["name"]="Importance"
    apply(d)
    a=create("task","Task",values={"priority":1})
    b=create("project","Project",values={"priority":3})
    assert a["values"]["priority"]==1 and b["values"]["priority"]==3
    assert all(next(f for f in t["fields"] if f["binding"]=="priority")["name"]=="Importance"
               for t in schema()["types"] if t["id"] in {"task","project"})

def test_legacy_schema_preview_retains_library_and_layout():
    d=schema(); d["type_layout"]=[{"type_id":"project","parent_type_id":"client"}]; apply(d)
    d=schema(); d["types"][0]["name"]="Context"
    p=run("structure.preview",dict(expected_revision=d["revision"],
        definition={k:d[k] for k in ("types","relationships")}))
    assert p["definition"]["field_library"]
    assert p["definition"]["type_layout"]==d["type_layout"]

def test_visual_layout_does_not_move_records():
    c=create("client","ABC"); p=create("project","Andi",parent_id=c["id"])
    d=schema(); d["type_layout"]=[{"type_id":"project","parent_type_id":"client"}];apply(d)
    with session_scope() as db:
        assert db.get(StructureRecord,p["id"]).parent_id==c["id"]

def test_type_layout_cycle_rejected():
    d=schema();d["type_layout"]=[{"type_id":"task","parent_type_id":"project"},
                                {"type_id":"project","parent_type_id":"task"}]
    with pytest.raises(DomainError):apply(d)

def test_missing_shared_field_rejected():
    d=schema(); d["field_library"]=[]
    with pytest.raises(DomainError):apply(d)

def test_custom_fields_with_equal_labels_not_merged():
    from jarvis.field_library import upgrade
    d=schema()
    for t in d["types"][:2]:
        t["fields"].append(dict(id="same",name="Value",description="Classification",kind="text",
                               binding=None, options=[],target_types=[],multiple=False,
                               inherit=False,visible=True,archived=False))
    d=upgrade(d)
    assert d["types"][0]["fields"][-1]["library_id"] != d["types"][1]["fields"][-1]["library_id"]

def test_canonical_subtasks_map_distinct_core_ids_in_both_directions():
    parent=create("task","Prepare")
    child=create("task","Pack",parent_id=parent["id"])
    assert parent["task_id"] != parent["id"]
    with session_scope() as db:
        t=db.get(Task,child["task_id"])
        assert t.parent_task_id==parent["task_id"]
        rev=t.revision
    result=run("task.update",dict(task_id=child["task_id"],expected_revision=rev,parent_task_id=None))
    with session_scope() as db:
        assert db.get(StructureRecord,child["id"]).parent_id is None
        assert result["revision"]==db.get(Task,child["task_id"]).revision

def test_legacy_task_create_nests_under_custom_actionable_record():
    parent=create("project","Release")
    child=run("task.create",dict(title="Write notes",parent_task_id=parent["task_id"]))
    with session_scope() as db:
        row=db.scalar(select(StructureRecord).where(StructureRecord.task_id==child["id"]))
        assert row.parent_id==parent["id"]

def test_quick_list_promotion_retains_child_homes_and_identity():
    result=run("quicklist.create",{"title":"Houston","items":[{"title":"Pack for Hayes"}]})
    child=result["items"][0]
    with session_scope() as db:
        row=db.scalar(select(StructureRecord).where(StructureRecord.task_id==child["id"]))
        root=db.scalar(select(StructureRecord).where(StructureRecord.task_id==result["id"]))
        assert row.parent_id==root.id
    run("quicklist.promote",{"list_id":result["id"],"expected_revision":result["revision"]})
    with session_scope() as db:
        row=db.scalar(select(StructureRecord).where(StructureRecord.task_id==child["id"]))
        assert row.parent_id==root.id


def test_migration_reports_conflicting_homes_without_overwriting():
    a=create("client","First"); b=create("task","Parent"); c=create("task","Child",parent_id=a["id"])
    from jarvis.hierarchy import adopt, report
    with session_scope() as db:
        row=db.get(StructureRecord,c["id"])
        row.provenance={k:v for k,v in row.provenance.items() if k!="home_version"}
        db.get(Task,c["task_id"]).parent_task_id=b["task_id"]
        adopt(db,OWNER,structure.ensure(db,OWNER))
        assert row.parent_id==a["id"]
        assert report(db,OWNER)["conflicts"][0]["record_id"]==c["id"]

def test_cosmetic_type_layout_does_not_trigger_field_assessment():
    from jarvis.models import Job
    with session_scope() as db:
        structure.schema_data(db,OWNER)
        before=list(db.scalars(select(Job.id).where(Job.kind=="assess_field")))
    d=schema();d["type_layout"]=[{"type_id":"task","parent_type_id":"project"}];apply(d)
    with session_scope() as db:
        assert list(db.scalars(select(Job.id).where(Job.kind=="assess_field")))==before

def test_shared_library_value_incompatibility_blocks_apply():
    row=create("task","Task",values={"priority":2})
    d=schema()
    priority=next(f for f in d["field_library"] if f["binding"]=="priority")
    priority["kind"]="text"
    with pytest.raises(DomainError):apply(d)


def test_inheritance_uses_library_identity_not_attachment_name():
    d=schema()
    shared=dict(id="shared_region",name="Region",description="Operating region",kind="text")
    d["field_library"].append(shared)
    for kind, local, inherit in [("client","region",False),("task","operating_region",True)]:
        t=next(t for t in d["types"] if t["id"]==kind)
        t["fields"].append({
            **shared,"id":local,"library_id":"shared_region","inherit":inherit})
    apply(d)
    parent=create("client","ABC",values={"region":"South"})
    child=create("task","Review",parent_id=parent["id"])
    assert child["values"]["operating_region"]=="South"
    assert child["inherited"]["operating_region"]==parent["id"]


def test_recurrence_origin_is_not_a_containment_edge():
    parent=create("task","Hidden template")
    child=create("task","Visible occurrence")
    # Real recurrence lineage is retained in occurrence_id, separate from Home.
    with session_scope() as db:
        task=db.get(Task,child["task_id"])
        task.parent_task_id=parent["task_id"]
        # Exercise adapter branch without manufacturing an invalid FK.
        row=db.get(StructureRecord,child["id"])
        row.provenance={k:v for k,v in row.provenance.items() if k!="home_version"}
        with db.no_autoflush:
            task.occurrence_id="synthetic-lineage"
            from jarvis.hierarchy import from_task
            from_task(db,OWNER,row,task,structure.ensure(db,OWNER))
            assert row.parent_id is None and task.parent_task_id is None
            task.occurrence_id=None
