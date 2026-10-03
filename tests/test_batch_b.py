"""Batch B acceptance: canonical homes, provider ownership, annotations and scheduling."""
from datetime import UTC, datetime
from uuid import uuid4

import pytest
from cryptography.fernet import Fernet
from sqlalchemy import delete, select
from jarvis import structure, planner, linear_sync
from jarvis.config import get_settings
from jarvis.db import session_scope
from jarvis.domain import DomainError, execute, serial
from jarvis.models import GoogleCalendarEvent, GoogleEventAnnotation, GoogleIdentity, Job, Task
from jarvis.structure_models import StructureRecord
from jarvis.scheduling_windows import allowed_intervals
from jarvis.google_calendar import event_detail
from test_structure import create, run, definition
from test_linear import linear as linear_fixture, imported
from test_google_calendar import linked, event
from test_planner_release import request, task

@pytest.fixture
def linear(monkeypatch):
    return linear_fixture.__wrapped__(monkeypatch)


@pytest.fixture(autouse=True)
def encryption(monkeypatch):
    monkeypatch.setattr(get_settings(), "integration_encryption_key", Fernet.generate_key().decode())

def update(row, **changes):
    return run("record.update", {"record_id":row["id"],"expected_revision":row["revision"],"schema_revision":definition()["revision"],**changes})

def record(task_id):
    with session_scope() as db:
        schema=structure.ensure(db,"davin")
        structure.sync_core_records(db,"davin",schema)
        row=db.scalar(select(StructureRecord).where(StructureRecord.task_id==task_id))
        return structure.data(db,row,schema)

def test_tree_moves_reorder_preserve_links_and_reject_stale_revision():
    home=create("client","ABC")
    a=create("project","A",parent_id=home["id"])
    b=create("project","B",parent_id=home["id"])
    c=create("project","C",parent_id=home["id"])
    moved=update(c,move_before_id=a["id"])
    assert moved["sort_order"]<a["sort_order"]<b["sort_order"]
    with pytest.raises(DomainError):update(c,move_before_id=None)
    moved=update(moved,move_before_id=None)
    assert moved["sort_order"]>b["sort_order"]
    assert moved["home"][-1]["id"]==home["id"]
    with pytest.raises(DomainError):update(moved,parent_id=moved["id"])
    with pytest.raises(DomainError):update(moved,move_before_id=str(uuid4()))

def test_native_record_unfiling_clears_legacy_project():
    project=run("project.create",{"name":"Legacy project"})
    task=run("task.create",{"title":"Move me","project_id":project["id"]})
    row=record(task["id"])
    update(row,parent_id=None)
    with session_scope() as db:
        assert db.get(Task,task["id"]).project_id is None


def test_annotations_do_not_replace_canonical_task_description():
    row=create("task","Details",body="Public description")
    updated=update(row,local_notes="Workspace-only thought")
    with session_scope() as db:
        assert db.get(Task,row["task_id"]).notes=="Public description"
        assert structure.data(db,db.get(StructureRecord,row["id"]))["local_notes"]=="Workspace-only thought"
    run("task.update",{"task_id":row["task_id"],"expected_revision":updated["task_revision"],"notes":"Edited via task API"})
    current=record(row["task_id"])
    assert current["body"]=="Edited via task API"
    assert current["local_notes"]=="Workspace-only thought"
    with pytest.raises(DomainError):
        run("record.update",{"record_id":row["id"],"schema_revision":1,"expected_revision":1,"local_notes":"intrusion"},owner="other")


def test_linear_local_home_and_annotations_do_not_queue_remote_project_edit(linear):
    tid,_=imported()
    row=record(tid)
    home=create("client","Independent local client")
    update(row,parent_id=home["id"],local_notes="Internal fee estimate")
    with session_scope() as db:
        assert not db.scalar(select(Job).where(Job.kind=="linear_write"))
        source=serial(db.get(Task,tid))["source"]
        assert source["provider"]=="linear" and source["account_id"]=="ws"
        assert source["container_id"]=="team" and source["item_id"]=="remote"
    row=record(tid)
    changed=update(row,body="Source description edit")
    assert changed["source"]["sync_state"]=="pending"
    with session_scope() as db:
        job=db.scalar(select(Job).where(Job.kind=="linear_write"))
        job_id=job.id
        assert job.payload["patch"]=={"description":"Source description edit"}
    linear_sync.process_write(job_id)
    assert linear.rows["remote"]["description"]=="Source description edit"
    current=record(tid)
    assert current["local_notes"]=="Internal fee estimate" and current["parent_id"]==home["id"]
    assert current["source"]["sync_state"]=="synced"


def test_google_annotations_survive_cache_replacement_and_are_account_scoped():
    source=linked()
    identity=event(source)
    result=run("calendar.annotate",{"event_id":identity,"expected_revision":0,"local_notes":"Never upload"})
    assert result["annotation_revision"]==1
    with pytest.raises(DomainError):run("calendar.annotate",{"event_id":identity,"expected_revision":0,"local_notes":"Stale"})
    with pytest.raises(DomainError):run("calendar.annotate",{"event_id":identity,"expected_revision":1,"local_notes":"Foreign"},owner="other")
    with session_scope() as db:
        payload=db.get(GoogleCalendarEvent,identity).payload
        assert "local_notes" not in payload
        db.execute(delete(GoogleCalendarEvent).where(GoogleCalendarEvent.id==identity))
    replacement=event(source,description="Source now says this")
    with session_scope() as db:
        detail=event_detail(db,"davin",replacement)
        assert detail["local_notes"]=="Never upload" and detail["annotation_revision"]==1
        assert detail["source"]["provider"]=="google"
        assert not detail["source"]["editable_fields"]  # Read-only source still permits local annotations.
        db.get(GoogleIdentity,"davin").subject="different-google-account"
        db.flush()
        assert event_detail(db,"davin",replacement)["local_notes"]==""
        assert db.scalar(select(GoogleEventAnnotation)).local_notes=="Never upload"


def test_planning_annotations_do_not_change_synced_fields_or_enqueue():
    entry=run("planning.create",{"title":"Meeting","start":"2027-01-02","end":"2027-01-03","timezone":"UTC","all_day":True})
    result=run("planning.annotate",{"entry_id":entry["id"],"expected_revision":1,"local_notes":"Preparation"})
    assert result["fields"]==entry["fields"] and result["local_notes"]=="Preparation"
    with session_scope() as db:
        assert not db.scalar(select(Job).where(Job.kind=="google_write"))


def test_source_colors_validate_and_preserve_other_providers():
    changed=run("settings.update",{"source_colors":{"linear":"#112233"}})
    assert changed["source_colors"]["linear"]=="#112233" and "google" in changed["source_colors"]
    for bad in [{"linear":"red; background:url(x)"},{"unknown":"#112233"}]:
        with pytest.raises(DomainError):run("settings.update",{"source_colors":bad})


def test_work_personal_windows_and_commit_revalidation():
    work=task("Work",60,availability="work")
    personal=task("Personal",60,availability="personal")
    run("settings.update",{"timezone":"America/Chicago","scheduling_windows":{"work":[{"days":[3],"start":"12:00","end":"17:00"}]}})
    result=planner.propose("davin",request([work,personal]))
    starts={b["task_id"]:b["start"] for b in result["blocks"]}
    assert "T12:00" in starts[work["task_id"]] and "T09:00" in starts[personal["task_id"]]
    run("settings.update",{"scheduling_windows":{"work":[{"days":[3],"start":"14:00","end":"17:00"}]}})
    with pytest.raises(DomainError,match="scheduling hours"):
        run("planning.commit",{"plan_token":result["plan_token"]})


def test_unavailable_hours_are_infeasible_and_explicit_override_is_visible():
    item=task("Work",60,availability="work")
    run("settings.update",{"scheduling_windows":{"work":[{"days":[3],"start":"18:00","end":"19:00"}]}})
    result=planner.propose("davin",request([item]))
    assert result["status"]=="infeasible"
    result=planner.propose("davin",request([item],override_reason="Owner explicitly requested morning outside work hours"))
    assert result["status"]=="ready" and result["scheduling"]["override_reason"]
    saved=run("planning.commit",{"plan_token":result["plan_token"]})
    assert saved["override_reason"]==result["scheduling"]["override_reason"]


@pytest.mark.parametrize("begin,end,days,start,finish,minutes",[
    ("2026-10-03T03:00Z","2026-10-03T10:00Z",[4],"22:00","02:00",240),
    ("2026-03-08T06:00Z","2026-03-08T12:00Z",[6],"01:00","04:00",120),
    ("2026-11-01T05:00Z","2026-11-01T12:00Z",[6],"01:00","03:00",180),
])
def test_windows_respect_overnight_and_dst(begin,end,days,start,finish,minutes):
    free=[(datetime.fromisoformat(begin),datetime.fromisoformat(end))]
    intervals=allowed_intervals(free,[{"days":days,"start":start,"end":finish}],"America/Chicago")
    assert sum((b-a).total_seconds()/60 for a,b in intervals)==minutes


def test_invalid_scheduling_settings_rejected():
    for windows in [{"course":[]},{"work":[{"days":[8],"start":"08:00","end":"17:00"}]},{"work":[{"days":[],"start":"08:00","end":"17:00"}]}]:
        with pytest.raises(DomainError):run("settings.update",{"scheduling_windows":windows})


def test_reverting_linear_description_preserves_later_local_notes(linear):
    from jarvis.action_history import revert
    from jarvis.models import ActionChange
    tid,_=imported()
    row=record(tid)
    command_id="source-description-change"
    run("record.update",{"record_id":row["id"],"expected_revision":row["revision"],"schema_revision":1,"body":"New shared description"},key=command_id)
    with session_scope() as db:
        job=db.scalar(select(Job).where(Job.kind=="linear_write"));first=job.id
    linear_sync.process_write(first)
    update(record(tid),local_notes="Keep this later annotation")
    with session_scope() as db:
        change=db.scalar(select(ActionChange).where(ActionChange.command_id==command_id,ActionChange.entity_kind=="record"))
        revert(db,"davin","davin",change.id,str(uuid4()))
    with session_scope() as db:
        job=db.scalar(select(Job).where(Job.kind=="linear_write",Job.id!=first));second=job.id
        assert job.payload["patch"]=={"description":"Notes"}
    linear_sync.process_write(second)
    assert linear.rows["remote"]["description"]=="Notes"
    assert record(tid)["local_notes"]=="Keep this later annotation"


def test_read_only_bot_cannot_see_scoped_annotations(client):
    from jarvis.external_service import custom_scrub
    row=create("task","Scope check",body="Task content")
    row=update(row,local_notes="Task private context")
    with session_scope() as db:
        scrubbed=custom_scrub(db,"davin",row,{"records:read"})
    assert "source" not in scrubbed and "local_notes" not in scrubbed and "body" not in scrubbed
