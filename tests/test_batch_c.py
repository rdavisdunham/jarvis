"""Batch C acceptance: capture identity, setup confirmation and explicit-rule authority."""
from uuid import uuid4
import pytest
from sqlalchemy import select
from cryptography.fernet import Fernet
from jarvis.db import session_scope
from jarvis.domain import DomainError, execute
from jarvis.config import get_settings
from jarvis.models import Task, Note, Schedule, ActionChange, OwnerSettings, Memory
from jarvis.structure_models import FieldUnderstanding, RoutingPattern
from jarvis import quick_lists, onboarding, structure, routing
from test_structure import run, create, definition, apply

@pytest.fixture(autouse=True)
def crypto(monkeypatch):
    monkeypatch.setattr(get_settings(),"integration_encryption_key",Fernet.generate_key().decode())

def capture(**args):
    return run("quicklist.create",{"title":"Get ready for Houston","items":[{"title":"Pack Hayes food","section":"Hayes"},{"title":"Pack clothes","section":"Personal"},{"title":"Gather moped tools","section":"Moped"},{"title":"Pack work laptop","section":"Work"}],**args})

def read(identity,owner="davin"):
    with session_scope() as db:return quick_lists.read(db,owner,identity)

def setup(owner="davin"):
    with session_scope() as db:return onboarding.state(db,owner)

def test_houston_capture_is_atomic_idempotent_searchable_and_no_alerts():
    args={"title":"Houston","items":[{"title":"Hayes food","section":"Hayes"}]}
    first=run("quicklist.create",args,key="capture-1")
    assert first==run("quicklist.create",args,key="capture-1")
    assert first["total"]==1 and first["items"][0]["quick_list_parent_id"]==first["id"]
    with session_scope() as db:
        assert len(list(db.scalars(select(Task))))==2
        assert not list(db.scalars(select(Schedule)))
        assert all(t.deadline_alert=="off" for t in db.scalars(select(Task)))
        assert quick_lists.listing(db,"davin","Hayes")["items"][0]["id"]==first["id"]
        assert not quick_lists.listing(db,"other")["items"]
        assert len(list(db.scalars(select(ActionChange))))>=2

def test_check_add_reorder_and_reopen_keep_item_ids():
    row=capture();item=row["items"][0]
    updated=run("quicklist.item",{"list_id":row["id"],"expected_revision":row["revision"],"operation":"edit","item_id":item["id"],"expected_item_revision":item["revision"],"completed":True})
    assert updated["done"]==1
    added=run("quicklist.item",{"list_id":row["id"],"expected_revision":updated["revision"],"operation":"add","title":"Phone charger","section":"Personal"})
    order=[t["id"] for t in added["items"]][::-1]
    moved=run("quicklist.item",{"list_id":row["id"],"expected_revision":added["revision"],"operation":"reorder","order":order})
    assert [t["id"] for t in read(row["id"])["items"]]==order
    item=next(t for t in moved["items"] if t["id"]==item["id"])
    reopened=run("quicklist.item",{"list_id":row["id"],"expected_revision":moved["revision"],"operation":"edit","item_id":item["id"],"expected_item_revision":item["revision"],"completed":False})
    assert reopened["done"]==0 and reopened["total"]==5

def test_stale_list_or_item_never_overwrites():
    row=capture();item=row["items"][0]
    run("task.update",{"task_id":item["id"],"expected_revision":item["revision"],"title":"Human changed this"})
    with pytest.raises(DomainError):run("quicklist.item",{"list_id":row["id"],"expected_revision":row["revision"],"operation":"edit","item_id":item["id"],"expected_item_revision":item["revision"],"completed":True})
    assert read(row["id"])["items"][0]["title"]=="Human changed this"
    added=run("quicklist.item",{"list_id":row["id"],"expected_revision":row["revision"],"operation":"add","title":"New"})
    with pytest.raises(DomainError):run("quicklist.item",{"list_id":row["id"],"expected_revision":row["revision"],"operation":"add","title":"Stale"})
    assert read(row["id"])["total"]==added["total"]

@pytest.mark.parametrize("order",[[],["foreign"],["duplicate","duplicate"]])
def test_invalid_reorder_is_atomic(order):
    row=capture()
    with pytest.raises(DomainError):run("quicklist.item",{"list_id":row["id"],"expected_revision":row["revision"],"operation":"reorder","order":order})
    assert read(row["id"])["revision"]==row["revision"]

def test_promote_preserves_ids_completion_and_history():
    row=capture();item=row["items"][0]
    row=run("quicklist.item",{"list_id":row["id"],"expected_revision":row["revision"],"operation":"edit","item_id":item["id"],"expected_item_revision":item["revision"],"completed":True})
    project=run("project.create",{"name":"Trip"})
    promoted=run("quicklist.promote",{"list_id":row["id"],"expected_revision":row["revision"],"project_id":project["id"]})
    assert promoted["task_ids"]==[t["id"] for t in row["items"]]
    with session_scope() as db:
        assert len(list(db.scalars(select(Task))))==5
        assert db.get(Task,item["id"]).status=="completed"
        assert all(t.project_id==project["id"] for t in db.scalars(select(Task)))
        assert not db.get(Task,row["id"]).is_quick_list
        assert len(list(db.scalars(select(ActionChange).where(ActionChange.entity_id==item["id"]))))>=2

def test_foreign_list_item_and_project_are_rejected():
    row=capture();other=run("quicklist.create",{"title":"Private"},owner="other")
    with pytest.raises(DomainError):read(row["id"],"other")
    with pytest.raises(DomainError):run("quicklist.item",{"list_id":row["id"],"expected_revision":row["revision"],"operation":"edit","item_id":other["id"],"expected_item_revision":1,"title":"steal"})
    project=run("project.create",{"name":"Other"},owner="other")
    with pytest.raises(DomainError):run("quicklist.promote",{"list_id":row["id"],"expected_revision":row["revision"],"project_id":project["id"]})
    assert read(row["id"])["is_quick_list"]

def test_past_deadline_is_retained_with_opt_in_alert():
    row=capture(due_date="2020-01-01",planned_date="2020-01-01")
    assert row["status"]=="open" and row["total"]==4 and row["deadline_alert"]=="off"
    assert all(t["due_date"] is None for t in row["items"])

def test_capture_limit_rejects_whole_transaction():
    with pytest.raises(DomainError):capture(items=[{"title":"x"}]*101)
    with session_scope() as db:assert not list(db.scalars(select(Task)))

def test_setup_is_account_scoped_skip_resume_and_stale_protected():
    row=run("onboarding.save",{"expected_revision":0,"preferred_name":"Morgan","timezone":"America/Chicago","purpose":"Work and home","status":"skipped"})
    assert setup()["preferred_name"]=="Morgan" and setup()["status"]=="skipped"
    assert setup("other")["revision"]==0
    with pytest.raises(DomainError):run("onboarding.save",{"expected_revision":0,"preferred_name":"Overwrite"})
    resumed=run("onboarding.save",{"expected_revision":row["revision"],"status":"in_progress"})
    assert resumed["purpose"]=="Work and home"

def test_setup_requires_explanation_and_separate_confirmation():
    row=run("onboarding.save",{"expected_revision":0,"classification_name":"Client","description":"Short"})
    with pytest.raises(DomainError):run("onboarding.preview",{"expected_revision":row["revision"]})
    row=run("onboarding.save",{"expected_revision":row["revision"],"description":"Organizations I do paid projects for at work.","examples":"ABC is a client; Andi is an ABC project."})
    result=run("onboarding.preview",{"expected_revision":row["revision"]},key="setup-turn:1")
    proposal=result["proposal"]
    assert definition()["revision"]==1
    with pytest.raises(DomainError):run("structure.apply",{"proposal_id":proposal["id"],"expected_revision":1},key="setup-turn:2")
    with pytest.raises(DomainError):run("onboarding.save",{"expected_revision":result["progress"]["revision"],"status":"completed"})
    apply(proposal)
    completed=run("onboarding.save",{"expected_revision":result["progress"]["revision"],"status":"completed","step":"done"})
    assert completed["status"]=="completed"
    assert "setup-group" in [t["id"] for t in definition()["types"]]
    with session_scope() as db:
        assert not list(db.scalars(select(Task))) and not list(db.scalars(select(Note)))
        assert db.get(FieldUnderstanding,("davin","type:setup-group"))

def test_changed_setup_answers_invalidate_preview():
    result=run("onboarding.preview",{"expected_revision":0})
    apply(result["proposal"])
    with pytest.raises(DomainError):run("onboarding.save",{"expected_revision":1,"status":"completed","classification_name":"Different"})
    assert setup()["status"]=="in_progress"

def test_setup_invalid_timezone_never_changes_profile():
    with pytest.raises(DomainError):run("onboarding.save",{"expected_revision":0,"preferred_name":"Nope","timezone":"invalid/zone"})
    assert setup()["revision"]==0

def test_explicit_rules_work_without_waiting_for_model_assessment():
    home=create("client","ABC")
    with session_scope() as db:
        for field in db.scalars(select(FieldUnderstanding)):field.status="needs_input"
    rule=run("routing.create",{"phrase":"Transcript Intelligence","type_id":"task","parent_id":home["id"],"reason":"Confirmed ABC project"})
    task=create("task","Finish Transcript Intelligence docs")
    assert task["parent_id"]==home["id"]
    with session_scope() as db:
        stored=db.get(RoutingPattern,rule["id"]);stored.origin="learned"
    assert create("task","Transcript Intelligence follow-up")["parent_id"] is None

def test_rule_edit_has_revision_guard_and_keeps_memory_separate():
    home=create("client","ABC")
    rule=run("routing.create",{"phrase":"Old phrase","type_id":"task","parent_id":home["id"],"reason":"Old"})
    edited=run("routing.change",{"pattern_id":rule["id"],"expected_revision":rule["revision"],"action":"edit","phrase":"New phrase","reason":"Corrected by me"})
    assert edited["origin"]=="explicit"
    assert create("task","New phrase docs")["parent_id"]==home["id"]
    assert create("task","Old phrase docs")["parent_id"] is None
    with pytest.raises(DomainError):run("routing.change",{"pattern_id":rule["id"],"expected_revision":rule["revision"],"action":"pause"})
    with session_scope() as db:
        assert not list(db.scalars(select(Memory)))
        state=routing.state(db,"davin")
        assert state["patterns"][0]["destination"]=="ABC"
        assert "Confirmed" in state["patterns"][0]["explanation"]


def test_shared_list_roles_and_personal_setup_boundary(client):
    from test_accounts import client_for,command,shared
    guest=client_for("guest")
    shared(client,guest)
    row=command(guest,"quicklist.create",title="Team checklist",items=[{"title":"Team item"}])
    assert guest.get("/api/v1/quick-lists/"+row["id"]).status_code==200
    assert guest.get("/api/v1/onboarding").status_code==403
    assert guest.post("/api/v1/commands",json={"command_id":str(uuid4()),"tool":"onboarding.save","arguments":{"expected_revision":0}}).status_code==403
    from jarvis.models import WorkspaceMember
    with session_scope() as db:
        member=db.scalar(select(WorkspaceMember).where(WorkspaceMember.account_id=="guest"));member.role="viewer"
    assert guest.post("/api/v1/commands",json={"command_id":str(uuid4()),"tool":"quicklist.item","arguments":{"list_id":row["id"],"expected_revision":row["revision"],"operation":"add","title":"No"}}).status_code==403
    personal=client_for("stranger")
    assert personal.get("/api/v1/quick-lists/"+row["id"]).status_code==404


def test_scoped_bot_can_capture_but_not_change_setup(client):
    from test_external_agents import key,call
    from jarvis import bot_access,external_service
    from jarvis.api import app
    from fastapi.testclient import TestClient
    credential,headers=key(client)
    response=call(headers,"quicklist.create",{"title":"Bot checklist","items":[{"title":"One"}]})
    assert response.status_code==200,response.text
    assert call(headers,"onboarding.save",{"expected_revision":0}).status_code==403
    assert client.post("/api/v1/bot-keys/"+credential["id"]+"/revoke").status_code==200
    assert call(headers,"quicklist.create",{"title":"Revoked"}).status_code==401


def test_resumed_setup_uses_current_profile_preferences():
    run("onboarding.save",{"expected_revision":0,"preferred_name":"Morgan","timezone":"America/Chicago","status":"skipped"})
    run("settings.update",{"preferred_name":"Rowan","timezone":"Europe/London"})
    resumed=setup()
    assert resumed["preferred_name"]=="Rowan" and resumed["timezone"]=="Europe/London"
    assert resumed["status"]=="skipped"
