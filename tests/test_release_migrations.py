"""N-1 migration with real data and old SQL shape; never touches the supplied database."""
import os
import subprocess
import sys
from pathlib import Path
from uuid import uuid4
from sqlalchemy import MetaData, Table, create_engine, select, inspect
from sqlalchemy.engine import make_url
from sqlalchemy.orm import Session
from jarvis.models import Task, OwnerSettings

ROOT=Path(__file__).resolve().parents[1]


def test_dani_migration_preserves_ids_assignments_history_and_collisions(test_database):
    from jarvis.models import Actor
    name = "jarvis_dani_" + uuid4().hex
    supplied = make_url(test_database)
    admin = create_engine(supplied.set(database="postgres"), isolation_level="AUTOCOMMIT")
    target = supplied.set(database=name)
    trial = None
    with admin.connect() as connection:
        connection.exec_driver_sql(f'CREATE DATABASE "{name}"')
    try:
        env = {**os.environ, "JARVIS_DATABASE_URL": target.render_as_string(hide_password=False), "JARVIS_ENV_FILE": ""}
        def migrate(*args):
            result = subprocess.run([sys.executable, "-m", "alembic", *args], cwd=ROOT, env=env, capture_output=True, text=True)
            assert result.returncode == 0, result.stdout + result.stderr
        migrate("upgrade", "0045_durable_runs")
        trial = create_engine(target)
        with Session(trial) as db, db.begin():
            db.add_all([
                Actor(id="default", owner_id="personal", name="Eri", kind="agent", revision=3),
                Actor(id="archived", owner_id="shared", name="Eri", kind="agent", archived=True),
                Actor(id="person", owner_id="shared", name="Dani", kind="person"),
                Actor(id="custom", owner_id="shared", name="Dani (assistant)", kind="agent"),
                Actor(id="old-person", owner_id="other", name="Eri", kind="person"),
                Actor(id="renamed", owner_id="custom-workspace", name="My helper", kind="agent"),
            ])
            db.flush()
            db.add_all([
                Task(id="active", owner_id="personal", title="Ask Eri about this", notes="Eri wrote the old plan.",
                     assignee_id="default", assignee="Eri", revision=4),
                Task(id="finished", owner_id="shared", title="History", assignee_id="archived", assignee="Eri",
                     archived=True, status="completed", revision=6),
                Task(id="person-task", owner_id="other", title="Person", assignee_id="old-person", assignee="Eri"),
            ])
        migrate("upgrade", "head")
        migrate("upgrade", "head")
        migrate("check")
        with Session(trial) as db:
            assert (db.get(Actor, "default").name, db.get(Actor, "default").revision) == ("Dani", 4)
            assert db.get(Actor, "archived").name == "Dani (assistant 2)"
            assert db.get(Actor, "archived").archived
            assert db.get(Actor, "person").name == "Dani"
            assert db.get(Actor, "custom").name == "Dani (assistant)"
            assert db.get(Actor, "renamed").name == "My helper"
            active = db.get(Task, "active")
            assert (active.assignee_id, active.assignee, active.revision) == ("default", "Dani", 5)
            assert active.title == "Ask Eri about this" and active.notes == "Eri wrote the old plan."
            finished = db.get(Task, "finished")
            assert (finished.assignee_id, finished.assignee, finished.revision) == ("archived", "Dani (assistant 2)", 7)
            assert finished.archived and finished.status == "completed"
            assert db.get(Task, "person-task").assignee == "Eri"
        # Rolling back the app must not rewrite names again or lose actor relationships.
        migrate("downgrade", "0045_durable_runs")
        migrate("upgrade", "head")
        with Session(trial) as db:
            assert db.get(Task, "active").assignee_id == "default"
            assert db.get(Actor, "default").revision == 4
    finally:
        if trial:
            trial.dispose()
        with admin.connect() as connection:
            connection.exec_driver_sql(f'DROP DATABASE "{name}" WITH (FORCE)')
        admin.dispose()


def test_n_minus_one_data_upgrade_old_writer_and_downgrade(test_database):
    name='jarvis_release_'+uuid4().hex
    supplied=make_url(test_database)
    admin=create_engine(supplied.set(database='postgres'),isolation_level='AUTOCOMMIT')
    target=supplied.set(database=name)
    trial=None
    with admin.connect() as c:c.exec_driver_sql(f'CREATE DATABASE "{name}"')
    try:
        env={**os.environ,'JARVIS_DATABASE_URL':target.render_as_string(hide_password=False),'JARVIS_ENV_FILE':''}
        def migrate(*args):
            result=subprocess.run([sys.executable,'-m','alembic',*args],cwd=ROOT,env=env,capture_output=True,text=True)
            assert result.returncode==0,result.stdout+result.stderr
        migrate('upgrade','0019_workspace_details')
        trial=create_engine(target);meta=MetaData()
        old=Table('tasks',meta,autoload_with=trial)
        assert 'is_quick_list' not in old.c
        def old_values(title):
            values={'id':str(uuid4()),'owner_id':'release-owner','title':title}
            for col in Task.__table__.columns:
                if col.name in old.c and col.name not in values and col.default is not None:
                    values[col.name]=col.default.arg(None) if col.default.is_callable else col.default.arg
            return values
        first=old_values('Existing task');second=old_values('Old writer after upgrade')
        with trial.begin() as c:
            c.execute(old.insert().values(**first))
            c.execute(OwnerSettings.__table__.insert().values(owner_id='release-owner',values={'preferred_name':'Rowan','source_colors':{'linear':'#aabbcc'}}))
            c.exec_driver_sql("INSERT INTO google_event_annotations (owner_id,account_subject,calendar_id,event_id,local_notes) VALUES ('release-owner','synthetic','calendar','event','Keep local notes')")
        migrate('upgrade','head');migrate('upgrade','head');migrate('check')
        # A previous-version insert omits new fields. DB defaults must make it readable by current code.
        with trial.begin() as c:c.execute(old.insert().values(**second))
        with Session(trial) as db:
            rows=list(db.scalars(select(Task)))
            assert {r.id for r in rows}=={first['id'],second['id']}
            assert all((r.is_quick_list,r.quick_section,r.quick_order)==(False,'',0) for r in rows)
            prefs=db.get(OwnerSettings,'release-owner')
            prefs.values={**prefs.values,'onboarding':{'revision':1,'status':'skipped','purpose':'Keep setup'}}
            db.commit()
        # Old-column reads/updates leave new data alone. Downgrade necessarily discards quick-list metadata.
        with trial.begin() as c:c.execute(old.update().where(old.c.id==first['id']).values(title='Edited by old shape'))
        migrate('downgrade','0019_workspace_details')
        assert 'is_quick_list' not in {c['name'] for c in inspect(trial).get_columns('tasks')}
        migrate('upgrade','head')
        with Session(trial) as db:
            assert db.get(Task,first['id']).title=='Edited by old shape'
            assert db.get(OwnerSettings,'release-owner').values['onboarding']['purpose']=='Keep setup'
            assert db.get(OwnerSettings,'release-owner').values['source_colors']=={'linear':'#aabbcc'}
            assert db.connection().exec_driver_sql('SELECT local_notes FROM google_event_annotations').scalar_one()=='Keep local notes'
    finally:
        if trial:trial.dispose()
        with admin.connect() as c:c.exec_driver_sql(f'DROP DATABASE "{name}" WITH (FORCE)')
        admin.dispose()


def test_record_templates_migration_upgrades_and_downgrades(test_database):
    name='jarvis_templates_'+uuid4().hex
    supplied=make_url(test_database)
    admin=create_engine(supplied.set(database='postgres'),isolation_level='AUTOCOMMIT')
    target=supplied.set(database=name)
    trial=None
    with admin.connect() as c:c.exec_driver_sql(f'CREATE DATABASE "{name}"')
    try:
        env={**os.environ,'JARVIS_DATABASE_URL':target.render_as_string(hide_password=False),'JARVIS_ENV_FILE':''}
        def migrate(*args):
            result=subprocess.run([sys.executable,'-m','alembic',*args],cwd=ROOT,env=env,capture_output=True,text=True)
            assert result.returncode==0,result.stdout+result.stderr
        migrate('upgrade','0021_review_delivery')
        trial=create_engine(target)
        assert 'record_templates' not in inspect(trial).get_table_names()
        with trial.begin() as c:
            c.exec_driver_sql("INSERT INTO structure_records (id,owner_id,type_id,title,body,sort_order,local_notes,values,archived,revision,schema_revision,provenance,created_at,updated_at) VALUES ('00000000-0000-0000-0000-000000000001','release-owner','project','Existing','',0,'','{}',false,1,1,'{}',now(),now())")
        migrate('upgrade','head');migrate('check')
        from jarvis.structure_models import RecordTemplate, StructureRecord
        with Session(trial) as db:
            db.add(RecordTemplate(owner_id='release-owner',type_id='project',name='Client onboarding',payload={'children':[{'type_id':'task','title':'Kickoff'}]},created_by='release-owner'))
            db.commit()
            row=db.scalar(select(RecordTemplate))
            assert (row.revision,row.archived,row.description)==(1,False,'') and row.payload['children'][0]['title']=='Kickoff'
            assert db.get(StructureRecord,'00000000-0000-0000-0000-000000000001').title=='Existing'
        assert 'ix_record_templates_owner_id_type_id' in {i['name'] for i in inspect(trial).get_indexes('record_templates')}
        migrate('downgrade','0021_review_delivery')
        assert 'record_templates' not in inspect(trial).get_table_names()
        # The current model has later columns, so read the downgraded row with plain SQL.
        with trial.connect() as c:
            assert c.exec_driver_sql("SELECT title FROM structure_records WHERE id='00000000-0000-0000-0000-000000000001'").scalar_one()=='Existing'
        migrate('upgrade','head')
    finally:
        if trial:trial.dispose()
        with admin.connect() as c:c.exec_driver_sql(f'DROP DATABASE "{name}" WITH (FORCE)')
        admin.dispose()


def test_record_reviews_migration_upgrades_and_downgrades(test_database):
    name='jarvis_reviews_'+uuid4().hex
    supplied=make_url(test_database)
    admin=create_engine(supplied.set(database='postgres'),isolation_level='AUTOCOMMIT')
    target=supplied.set(database=name)
    trial=None
    with admin.connect() as c:c.exec_driver_sql(f'CREATE DATABASE "{name}"')
    try:
        env={**os.environ,'JARVIS_DATABASE_URL':target.render_as_string(hide_password=False),'JARVIS_ENV_FILE':''}
        def migrate(*args):
            result=subprocess.run([sys.executable,'-m','alembic',*args],cwd=ROOT,env=env,capture_output=True,text=True)
            assert result.returncode==0,result.stdout+result.stderr
        insert="INSERT INTO structure_records (id,owner_id,type_id,title,body,sort_order,local_notes,values,archived,revision,schema_revision,provenance,created_at,updated_at) VALUES ('{}','release-owner','client','{}','',0,'','{{}}',false,1,1,'{{}}',now(),now())"
        migrate('upgrade','0022_record_templates')
        trial=create_engine(target)
        assert 'next_review_at' not in {c['name'] for c in inspect(trial).get_columns('structure_records')}
        with trial.begin() as c:c.exec_driver_sql(insert.format('00000000-0000-0000-0000-000000000001','Existing'))
        migrate('upgrade','head');migrate('check')
        # A previous-version writer omits the review columns; defaults keep the row readable.
        with trial.begin() as c:c.exec_driver_sql(insert.format('00000000-0000-0000-0000-000000000002','Old writer'))
        from jarvis.structure_models import StructureRecord
        from jarvis.models import now
        with Session(trial) as db:
            rows=list(db.scalars(select(StructureRecord).order_by(StructureRecord.id)))
            assert [(r.title,r.review_paused,r.next_review_at,r.last_reviewed_at,r.review_queued_at) for r in rows]==[
                ('Existing',False,None,None,None),('Old writer',False,None,None,None)]
            rows[0].next_review_at=now();rows[0].review_paused=True
            db.commit()
        assert 'ix_structure_records_owner_id_next_review_at' in {i['name'] for i in inspect(trial).get_indexes('structure_records')}
        migrate('downgrade','0022_record_templates')
        assert 'review_paused' not in {c['name'] for c in inspect(trial).get_columns('structure_records')}
        migrate('upgrade','head')
        with Session(trial) as db:
            assert db.get(StructureRecord,'00000000-0000-0000-0000-000000000001').title=='Existing'
            assert db.get(StructureRecord,'00000000-0000-0000-0000-000000000001').review_paused is False
    finally:
        if trial:trial.dispose()
        with admin.connect() as c:c.exec_driver_sql(f'DROP DATABASE "{name}" WITH (FORCE)')
        admin.dispose()


def test_agent_sessions_migration_upgrades_and_downgrades(test_database):
    name='jarvis_sessions_'+uuid4().hex
    supplied=make_url(test_database)
    admin=create_engine(supplied.set(database='postgres'),isolation_level='AUTOCOMMIT')
    target=supplied.set(database=name)
    trial=None
    with admin.connect() as c:c.exec_driver_sql(f'CREATE DATABASE "{name}"')
    try:
        env={**os.environ,'JARVIS_DATABASE_URL':target.render_as_string(hide_password=False),'JARVIS_ENV_FILE':''}
        def migrate(*args):
            result=subprocess.run([sys.executable,'-m','alembic',*args],cwd=ROOT,env=env,capture_output=True,text=True)
            assert result.returncode==0,result.stdout+result.stderr
        migrate('upgrade','0023_record_reviews')
        trial=create_engine(target)
        assert 'agent_sessions' not in inspect(trial).get_table_names()
        migrate('upgrade','head');migrate('check')
        from jarvis.session_models import AgentSession
        with Session(trial) as db:
            db.add(AgentSession(owner_id='release-owner',tool='claude_code',session_id='s1'))
            db.commit()
            assert db.scalar(select(AgentSession)).binding_source==''
        assert 'ix_agent_sessions_owner_id_last_active_at' in {i['name'] for i in inspect(trial).get_indexes('agent_sessions')}
        migrate('downgrade','0023_record_reviews')
        assert not {'agent_sessions','agent_session_rules'} & set(inspect(trial).get_table_names())
        migrate('upgrade','head')
    finally:
        if trial:trial.dispose()
        with admin.connect() as c:c.exec_driver_sql(f'DROP DATABASE "{name}" WITH (FORCE)')
        admin.dispose()


def test_notification_bundles_migration_backfills_delivery_and_downgrades(test_database):
    name='jarvis_notices_'+uuid4().hex
    supplied=make_url(test_database)
    admin=create_engine(supplied.set(database='postgres'),isolation_level='AUTOCOMMIT')
    target=supplied.set(database=name)
    trial=None
    with admin.connect() as c:c.exec_driver_sql(f'CREATE DATABASE "{name}"')
    try:
        env={**os.environ,'JARVIS_DATABASE_URL':target.render_as_string(hide_password=False),'JARVIS_ENV_FILE':''}
        def migrate(*args):
            result=subprocess.run([sys.executable,'-m','alembic',*args],cwd=ROOT,env=env,capture_output=True,text=True)
            assert result.returncode==0,result.stdout+result.stderr
        notice="INSERT INTO notifications (id,owner_id,title,body,scheduled_at,created_at,generation) VALUES ('{}','release-owner','{}','',now(),now(),{})"
        migrate('upgrade','0036_run_summaries')
        trial=create_engine(target)
        with trial.begin() as c:
            c.exec_driver_sql(notice.format('00000000-0000-0000-0000-000000000001','Pushed',1))
            c.exec_driver_sql(notice.format('00000000-0000-0000-0000-000000000002','Snoozed since its push',2))
            c.exec_driver_sql("INSERT INTO push_subscriptions (id,owner_id,device_id,subscription,active) VALUES ('sub','release-owner','d','{}',true)")
            for identity,generation in (('00000000-0000-0000-0000-000000000001',1),('00000000-0000-0000-0000-000000000002',1)):
                c.exec_driver_sql(f"INSERT INTO delivery_attempts (id,notification_id,subscription_id,generation,status,attempts,next_attempt_at) VALUES ('{uuid4()}','{identity}','sub',{generation},'submitted',1,now())")
        migrate('upgrade','head');migrate('check')
        # A previous-version writer omits the new columns; they stay empty.
        with trial.begin() as c:c.exec_driver_sql(notice.format('00000000-0000-0000-0000-000000000003','Old writer',1))
        from jarvis.models import Notification
        from jarvis.notices import state
        with Session(trial) as db:
            rows={r.title:r for r in db.scalars(select(Notification))}
            # Only a push at the notice's current generation counts as delivered.
            assert {t:state(r) for t,r in rows.items()}=={'Pushed':'delivered','Snoozed since its push':'pending','Old writer':'pending'}
            assert all(r.snoozed_until is None and r.bundle_id is None for r in rows.values())
        migrate('downgrade','0036_run_summaries')
        assert 'delivered_at' not in {c['name'] for c in inspect(trial).get_columns('notifications')}
        migrate('upgrade','head')
    finally:
        if trial:trial.dispose()
        with admin.connect() as c:c.exec_driver_sql(f'DROP DATABASE "{name}" WITH (FORCE)')
        admin.dispose()


def test_task_archive_migration_preserves_outcomes_and_legacy_unknown_age(test_database):
    from datetime import UTC, datetime
    from sqlalchemy import text
    name = 'jarvis_archive_' + uuid4().hex
    supplied = make_url(test_database)
    admin = create_engine(supplied.set(database='postgres'), isolation_level='AUTOCOMMIT')
    target = supplied.set(database=name)
    trial = None
    with admin.connect() as c:
        c.exec_driver_sql(f'CREATE DATABASE "{name}"')
    try:
        env = {**os.environ, 'JARVIS_DATABASE_URL': target.render_as_string(hide_password=False), 'JARVIS_ENV_FILE': ''}
        def migrate(*args):
            result = subprocess.run([sys.executable, '-m', 'alembic', *args], cwd=ROOT, env=env, capture_output=True, text=True)
            assert result.returncode == 0, result.stdout + result.stderr
        migrate('upgrade', '0043_run_owner_wait')
        trial = create_engine(target)
        old = Table('tasks', MetaData(), autoload_with=trial)
        instant = datetime(2026, 10, 1, 12, tzinfo=UTC)
        for status in ('completed', 'cancelled', 'open'):
            values = {'id': status, 'owner_id': 'release-owner', 'title': status, 'status': status, 'archived': status == 'cancelled'}
            for col in Task.__table__.columns:
                if col.name in old.c and col.name not in values and col.default is not None:
                    values[col.name] = col.default.arg(None) if col.default.is_callable else col.default.arg
            if status == 'completed':
                values['completed_at'] = instant
            with trial.begin() as c:
                c.execute(old.insert().values(**values))
        migrate('upgrade', 'head')
        migrate('check')
        with trial.connect() as c:
            rows = {r['id']: r for r in c.execute(text('SELECT * FROM tasks')).mappings()}
        assert rows['completed']['finished_at'] == instant
        assert (rows['completed']['archive_eligible_at'] - instant).total_seconds() == 86400
        assert rows['cancelled']['finished_at'] is None and rows['cancelled']['archive_eligible_at'] is None
        assert rows['cancelled']['archived'] and rows['cancelled']['archive_reason'] == 'manual'
        assert rows['open']['finished_at'] is None
        migrate('downgrade', '0043_run_owner_wait')
        assert 'finished_at' not in {c['name'] for c in inspect(trial).get_columns('tasks')}
        with trial.connect() as c:
            assert c.scalar(text("SELECT status FROM tasks WHERE id='cancelled'")) == 'cancelled'
        migrate('upgrade', 'head')
    finally:
        if trial:
            trial.dispose()
        with admin.connect() as c:
            c.exec_driver_sql(f'DROP DATABASE "{name}" WITH (FORCE)')
        admin.dispose()


def test_in_review_migration_adds_status_to_work_types_and_downgrades(test_database):
    import copy
    import json
    from sqlalchemy import text
    from jarvis.structure import default_definition
    name = 'jarvis_in_review_' + uuid4().hex
    supplied = make_url(test_database)
    admin = create_engine(supplied.set(database='postgres'), isolation_level='AUTOCOMMIT')
    target = supplied.set(database=name)
    trial = None
    with admin.connect() as c:
        c.exec_driver_sql(f'CREATE DATABASE "{name}"')
    try:
        env = {**os.environ, 'JARVIS_DATABASE_URL': target.render_as_string(hide_password=False), 'JARVIS_ENV_FILE': ''}
        def migrate(*args):
            result = subprocess.run([sys.executable, '-m', 'alembic', *args], cwd=ROOT, env=env, capture_output=True, text=True)
            assert result.returncode == 0, result.stdout + result.stderr
        migrate('upgrade', '0057_memory_core')
        trial = create_engine(target)
        old = copy.deepcopy(default_definition())
        for t in old['types']:
            t['statuses'] = [s for s in t['statuses'] if s['meaning'] != 'in_review']
        old['types'].append({**copy.deepcopy(next(t for t in old['types'] if t['id'] == 'task')), 'id': 'notes_only',
                             'capabilities': ['contents'],
                             'statuses': [{'id': 'open', 'name': 'Open', 'meaning': 'open'}]})
        has = copy.deepcopy(default_definition())
        with trial.begin() as c:
            for owner, definition in (('old-owner', old), ('has-owner', has)):
                c.execute(text("INSERT INTO structure_schemas (owner_id, revision, definition, updated_at) "
                               "VALUES (:o, 3, CAST(:d AS jsonb), now())"), {'o': owner, 'd': json.dumps(definition)})
            c.exec_driver_sql("INSERT INTO structure_records (id,owner_id,type_id,title,body,sort_order,local_notes,values,archived,revision,schema_revision,provenance,created_at,updated_at,status_id) VALUES ('00000000-0000-0000-0000-000000000001','old-owner','task','Review me','',0,'','{}',false,1,1,'{}',now(),now(),'in_progress')")

        def schemas():
            with trial.connect() as c:
                return {r.owner_id: (r.revision, r.definition) for r in c.execute(text('SELECT owner_id, revision, definition FROM structure_schemas'))}

        migrate('upgrade', 'head')
        migrate('check')
        after = schemas()
        revision, definition = after['old-owner']
        assert revision == 4
        types = {t['id']: t for t in definition['types']}
        ids = [s['id'] for s in types['task']['statuses']]
        assert ids[ids.index('in_progress') + 1] == 'in_review'
        assert types['task']['statuses'][ids.index('in_review')] == {'id': 'in_review', 'name': 'In review', 'meaning': 'in_review'}
        assert types['notes_only']['statuses'] == [{'id': 'open', 'name': 'Open', 'meaning': 'open'}]
        assert after['has-owner'] == (3, has)

        with trial.begin() as c:
            c.exec_driver_sql("UPDATE structure_records SET status_id='in_review' WHERE owner_id='old-owner'")
            tasks = Table('tasks', MetaData(), autoload_with=c)
            values = {'id': 'review-task', 'owner_id': 'old-owner', 'title': 'Review', 'status': 'in_review'}
            for col in Task.__table__.columns:
                if col.name in tasks.c and col.name not in values and col.default is not None:
                    values[col.name] = col.default.arg(None) if col.default.is_callable else col.default.arg
            c.execute(tasks.insert().values(**values))
        migrate('downgrade', '0057_memory_core')
        restored = schemas()
        assert restored['old-owner'][0] == 5
        assert all(s['meaning'] != 'in_review' for t in restored['old-owner'][1]['types'] for s in t['statuses'])
        assert [s['id'] for s in restored['old-owner'][1]['types'][0]['statuses']] == [s['id'] for s in old['types'][0]['statuses']]
        with trial.connect() as c:
            assert c.scalar(text("SELECT status_id FROM structure_records WHERE owner_id='old-owner'")) == 'in_progress'
            assert c.scalar(text("SELECT status FROM tasks WHERE id='review-task'")) == 'in_progress'
        migrate('upgrade', 'head')
    finally:
        if trial:
            trial.dispose()
        with admin.connect() as c:
            c.exec_driver_sql(f'DROP DATABASE "{name}" WITH (FORCE)')
        admin.dispose()
