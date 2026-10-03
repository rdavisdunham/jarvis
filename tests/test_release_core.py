"""Release-core contracts for flexible schemas and N-1 data, using disposable databases."""
import copy
from uuid import uuid4
import pytest
from sqlalchemy import select
from jarvis import structure
from jarvis.db import session_scope
from jarvis.domain import DomainError
from jarvis.models import Task, Note
from jarvis.structure_models import StructureRecord
from test_structure import run, create, definition, propose, apply


def fields():
    return [
        {"id":"discipline","name":"Discipline","description":"Kind of work","kind":"select","options":[{"id":str(i),"name":f"Discipline {i}"} for i in range(5)]},
        {"id":"genres","name":"Genres","description":"Genres to browse","kind":"multiselect","options":[{"id":"scifi","name":"Science fiction"},{"id":"drama","name":"Drama"}]},
        {"id":"flag","name":"Flag","description":"A boolean choice","kind":"boolean"},
        {"id":"score","name":"Score","description":"A numeric score","kind":"number"},
        {"id":"client_ref","name":"Client","description":"Client this work belongs to","kind":"relation","target_types":["client"],"inherit":True},
    ]


def install_fields():
    d=definition()
    for t in d['types']:
        if t['id'] in {'task','note'}:t['fields']+=fields()
    apply(propose(d))


def read(row):
    with session_scope() as db:return structure.data(db,db.get(StructureRecord,row['id']))


def update(row,**changes):
    return run('record.update',{'record_id':row['id'],'expected_revision':row['revision'],'schema_revision':definition()['revision'],**changes})


def test_schema_preview_rename_stable_ids_and_full_definition():
    d=definition();before=copy.deepcopy(d)
    project=next(t for t in d['types'] if t['id']=='project');project['name']='Initiative';project['plural']='Initiatives'
    p=propose(d)
    assert definition()==before
    apply(p)
    after=definition()
    assert {t['id'] for t in after['types']}=={t['id'] for t in before['types']}
    assert next(t for t in after['types'] if t['id']=='project')['name']=='Initiative'
    assert [t for t in after['types'] if t['id']!='project']==[t for t in before['types'] if t['id']!='project']
    # A replay cannot duplicate metadata, whether returned idempotently or rejected stale.
    try:apply(p)
    except DomainError:pass
    assert definition()==after


@pytest.mark.parametrize('invalid',['description','duplicate_type','duplicate_field','unknown_target'])
def test_invalid_schema_is_atomic(invalid):
    d=definition();before=copy.deepcopy(d);t=next(t for t in d['types'] if t['id']=='task')
    if invalid=='description':t['fields'].append({**fields()[0],'description':''})
    elif invalid=='duplicate_type':d['types'].append(copy.deepcopy(t))
    elif invalid=='duplicate_field':t['fields'] += [fields()[0],fields()[0]]
    else:t['fields'].append({**fields()[-1],'target_types':['not-a-type']})
    with pytest.raises(DomainError):propose(d)
    assert definition()==before


def test_schema_options_scalars_content_and_rename_preserve_identity():
    install_fields()
    row=create('note','Movies',body='Keep this authored text.',values={'genres':['scifi','drama','scifi'],'discipline':'0','flag':False,'score':0})
    assert row['values']=={'genres':['scifi','drama'],'discipline':'0','flag':False,'score':0}
    with session_scope() as db:
        assert db.get(Note,row['note_id']).content=='Keep this authored text.'
        assert not list(db.scalars(select(Task)))
    d=definition();t=next(t for t in d['types'] if t['id']=='note')
    f=next(f for f in t['fields'] if f['id']=='discipline');assert len(f['options'])==5
    f['options'][0]['name']='Renamed discipline';apply(propose(d))
    assert read(row)['values']['discipline']=='0'
    d=definition();t=next(t for t in d['types'] if t['id']=='note');next(f for f in t['fields'] if f['id']=='discipline')['archived']=True
    apply(propose(d))
    assert read(row)['values']['discipline']=='0'


@pytest.mark.parametrize('values',[{'discipline':'unknown'},{'score':float('nan')},{'score':float('inf')},{'score':float('-inf')}])
def test_invalid_custom_values_reject_without_effect(values):
    install_fields();row=create('task','Preserve',values={'score':0})
    with pytest.raises(DomainError):update(row,values=values)
    assert read(row)==row


def test_client_lineage_override_reset_move_and_wrong_type():
    install_fields();a=create('client','Juniper');b=create('client','Lens')
    project=create('project','Beacon',parent_id=a['id']);row=create('task','Docs',parent_id=project['id'])
    assert row['values']['client_ref']==a['id'] and row['inherited']['client_ref']==a['id']
    row=update(row,values={'client_ref':b['id']})
    assert row['values']['client_ref']==b['id'] and 'client_ref' not in row['inherited']
    row=update(row,reset_fields=['client_ref'])
    assert row['values']['client_ref']==a['id']
    update(project,parent_id=b['id'])
    assert read(row)['values']['client_ref']==b['id']
    with pytest.raises(DomainError):update(read(row),values={'client_ref':project['id']})
    assert read(row)['values']['client_ref']==b['id']


def test_classification_record_has_no_phantom_task_and_stale_edits_reject():
    row=create('client','Juniper');assert not row['task_id'] and not row['note_id']
    changed=update(row,body='Current notes')
    with pytest.raises(DomainError):update(row,title='Stale')
    with pytest.raises(DomainError):run('record.update',{'record_id':row['id'],'expected_revision':changed['revision'],'schema_revision':999,'title':'Wrong schema'})
    assert read(changed)==changed
    with session_scope() as db:assert not list(db.scalars(select(Task)))


def test_custom_archive_restore_keeps_backing_task_and_active_reads():
    row=create('task','Archive release fixture')
    row=update(row,archived=True)
    with session_scope() as db:
        assert db.get(Task,row['task_id']).archived
        assert row['id'] not in {r['id'] for r in structure.records(db,'davin')['items']}
        assert row['id'] in {r['id'] for r in structure.records(db,'davin',archived=True)['items']}
    row=update(row,archived=False)
    with session_scope() as db:
        assert not db.get(Task,row['task_id']).archived
        assert row['id'] in {r['id'] for r in structure.records(db,'davin')['items']}


def test_custom_records_paginate_past_100_without_loss():
    ids={create('client',f'Client {i}')['id'] for i in range(105)}
    seen=[];offset=0
    with session_scope() as db:
        while True:
            page=structure.records(db,'davin',type_id='client',limit=25,offset=offset)
            seen.extend(r['id'] for r in page['items'])
            if not page['has_more']:break
            offset=page['next_offset']
    assert len(seen)==105 and set(seen)==ids


def test_status_labels_change_without_operational_drift_and_used_status_mapping():
    row=create('task','Status fixture');d=definition();t=next(t for t in d['types'] if t['id']=='task')
    next(s for s in t['statuses'] if s['id']=='open')['name']='Ready to go'
    apply(propose(d));assert read(row)['status_meaning']=='open'
    d=definition();t=next(t for t in d['types'] if t['id']=='task')
    next(s for s in t['statuses'] if s['id']=='open')['id']='ready'
    p=propose(d);assert p['impact']['blocking_count']>0
    with pytest.raises(DomainError):apply(p)
    p=propose(d,status_mappings={'task':{'open':'ready'}});apply(p)
    saved=read(row);assert saved['status_id']=='ready' and saved['status_meaning']=='open'
    with session_scope() as db:assert db.get(Task,row['task_id']).status=='open'


def test_schema_apply_after_concurrent_schema_change_rejects():
    d=definition();one=copy.deepcopy(d);two=copy.deepcopy(d)
    one['types'][0]['name']='First';two['types'][0]['name']='Second'
    p=propose(one);q=propose(two);apply(p)
    with pytest.raises(DomainError):apply(q)
    assert definition()['types'][0]['name']=='First'


def test_schema_viewer_editor_cannot_redesign_via_tools(client):
    import asyncio
    from test_accounts import client_for,shared,post
    from jarvis.models import WorkspaceMember
    from jarvis.tools import call_tool
    guest=client_for('guest');w=shared(client,guest);boot=guest.get('/api/v1/bootstrap').json()
    with session_scope() as db:d=structure.schema_data(db,w['id'])
    args={'expected_revision':d['revision'],'definition':{k:d[k] for k in ('types','relationships')}}
    for role in ('editor','viewer'):
        with session_scope() as db:db.get(WorkspaceMember,(w['id'],'guest')).role=role
        response=guest.post('/api/v1/commands',json={'command_id':str(uuid4()),'tool':'structure.preview','arguments':args})
        assert response.status_code==403
        with pytest.raises(DomainError):asyncio.run(call_tool(w['id'],str(uuid4()),0,'structure_preview',args,device=boot['device_id']))
    with session_scope() as db:assert structure.schema_data(db,w['id'])['revision']==d['revision']


def test_source_annotations_recheck_viewer_removed_member_and_foreign_account(client):
    from test_accounts import client_for,shared,command
    from jarvis.models import WorkspaceMember
    guest=client_for('guest');w=shared(client,guest)
    row=command(guest,'record.create',type_id='task',title='Shared',schema_revision=1)
    row=command(guest,'record.update',record_id=row['id'],expected_revision=row['revision'],schema_revision=1,local_notes='Workspace-only note')
    with session_scope() as db:db.get(WorkspaceMember,(w['id'],'guest')).role='viewer'
    args={'record_id':row['id'],'expected_revision':row['revision'],'schema_revision':1,'local_notes':'No'}
    assert guest.post('/api/v1/commands',json={'command_id':str(uuid4()),'tool':'record.update','arguments':args}).status_code==403
    assert guest.get('/api/v1/structure/records/'+row['id']).json()['local_notes']=='Workspace-only note'
    with session_scope() as db:db.get(WorkspaceMember,(w['id'],'guest')).active=False
    response=guest.get('/api/v1/structure/records/'+row['id']);assert response.status_code==403 and 'Workspace-only note' not in response.text
    stranger=client_for('stranger');response=stranger.get('/api/v1/structure/records/'+row['id'])
    assert response.status_code==404 and 'Workspace-only note' not in response.text


def test_setup_two_devices_stale_preview_and_current_profile(client):
    from test_accounts import client_for,command
    second=client_for('davin')
    saved=command(client,'onboarding.save',expected_revision=0,preferred_name='Rowan',purpose='Release testing')
    assert second.get('/api/v1/onboarding').json()['purpose']=='Release testing'
    proposal=command(second,'onboarding.preview',expected_revision=saved['revision'])['proposal']
    d=definition();d['types'][0]['name']='Changed concurrently';apply(propose(d))
    with pytest.raises(DomainError):apply(proposal)
    assert second.get('/api/v1/onboarding').json()['status']!='completed'
