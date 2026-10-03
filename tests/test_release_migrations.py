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
