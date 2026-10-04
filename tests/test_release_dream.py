"""Exercise scheduled memory review through the real worker scan, not queue helpers alone."""
import asyncio
from datetime import UTC, datetime, timedelta
from uuid import uuid4
import pytest
from sqlalchemy import select
from jarvis import memory_review, memory_service, worker
from jarvis.db import session_scope
from jarvis.domain import execute
from jarvis.models import Job, Memory, Source

OWNER='davin'

def remember(content):
    with session_scope() as db:
        row=execute(db,OWNER,str(uuid4()),'memory.capture',{'content':content})['data']
        db.get(Memory,row['id']).embedding=[1.0,0.0]
        return row


def tick(monkeypatch,instant):
    monkeypatch.setattr(memory_review,'now',lambda:instant)
    isolated=worker.isolated
    # Keep the production supervisor -> in_session -> queue_due_reviews path.
    # Unrelated network/maintenance services are excluded from this scheduler fixture.
    monkeypatch.setattr(worker,'isolated',lambda name,fn,*args: isolated(name,fn,*args) if name=='memory_reviews' else None)
    worker.supervisor_cycle(None,1,dispatch=False)
    monkeypatch.setattr(worker,'isolated',isolated)
    with session_scope() as db:return list(db.scalars(select(Job.id).where(Job.owner_id==OWNER,Job.kind==memory_review.KIND).order_by(Job.created_at,Job.id)))


def test_disabled_dream_scheduler_never_queues_on_worker_ticks(monkeypatch):
    remember('The user likes tea.')
    with session_scope() as db:execute(db,OWNER,str(uuid4()),'settings.update',{'deep_sleep_enabled':False})
    assert tick(monkeypatch,datetime(2030,1,14,tzinfo=UTC))==[]
    assert tick(monkeypatch,datetime(2030,1,21,tzinfo=UTC))==[]


def test_offline_week_catches_up_once_across_worker_ticks(monkeypatch):
    remember('The user likes tea.')
    first=tick(monkeypatch,datetime(2030,1,14,tzinfo=UTC));assert len(first)==1
    memory_review.process(first[0])
    later=datetime(2030,2,4,tzinfo=UTC)
    resumed=tick(monkeypatch,later);assert len(resumed)==2
    memory_review.process(next(j for j in resumed if j not in first))
    assert set(tick(monkeypatch,later+timedelta(minutes=1)))==set(resumed)
    with session_scope() as db:
        rows=list(db.scalars(select(Job).where(Job.kind==memory_review.KIND).order_by(Job.created_at,Job.id)))
        assert len({r.payload['period'] for r in rows})==2
        assert all(r.status=='succeeded' for r in rows)


def test_dst_review_worker_ticks_keep_one_job_per_local_week(monkeypatch):
    remember('The user likes tea.')
    with session_scope() as db:execute(db,OWNER,str(uuid4()),'settings.update',{'timezone':'America/Chicago'})
    before=datetime(2030,3,10,7,59,tzinfo=UTC)
    first=tick(monkeypatch,before);assert len(first)==1
    memory_review.process(first[0])
    after=datetime(2030,3,10,8,0,tzinfo=UTC)
    second=tick(monkeypatch,after);assert len(second)==2
    memory_review.process(next(j for j in second if j not in first))
    assert set(tick(monkeypatch,after+timedelta(hours=1)))==set(second)
    with session_scope() as db:
        periods=sorted(datetime.fromisoformat(r.payload['period']) for r in db.scalars(select(Job).where(Job.kind==memory_review.KIND).order_by(Job.created_at,Job.id)))
    assert periods[1]-periods[0]==timedelta(hours=167)
    from zoneinfo import ZoneInfo
    assert all(p.astimezone(ZoneInfo('America/Chicago')).hour==3 for p in periods)


def test_forget_source_removes_review_vector_retrieval_and_prompt(monkeypatch):
    first=remember("The user's cat is named Miso.");remember("The user's cat is named Mizo.")
    with session_scope() as db:job=memory_review.queue_review(db,OWNER,manual=True).id
    memory_review.process(job)
    with session_scope() as db:
        assert memory_review.pending_reviews(db,OWNER)
        execute(db,OWNER,str(uuid4()),'memory.forget',{'memory_id':first['id'],'delete_source':True})
        row=db.get(Memory,first['id'])
        assert row.suppressed and row.embedding is None and not row.content
        assert db.get(Source,row.source_id).deleted_at
        assert not memory_review.pending_reviews(db,OWNER)
        assert first['id'] not in {r['id'] for r in memory_service.search(db,OWNER,'Miso')}
    monkeypatch.setattr('jarvis.memory_learning.query_embedding',lambda *a,**k:[1.0,0.0])
    assert first['id'] not in {r['id'] for r in asyncio.run(memory_service.semantic_search(OWNER,'Miso'))}
    context=asyncio.run(memory_service.prompt_context(OWNER,'Miso'))
    assert first['id'] not in context and 'named Miso.' not in context
