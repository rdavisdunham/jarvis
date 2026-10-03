import json
import logging
import threading
from datetime import timedelta
from unittest.mock import Mock
from uuid import uuid4

import pytest
from sqlalchemy import select, text
from jarvis import worker
from jarvis.config import get_settings
from jarvis.db import engine, session_scope
from jarvis.domain import enqueue_job
from jarvis.latency import span
from jarvis.models import Job, Outbox, now
from jarvis.work_wakeup import wake_dispatch


def test_notify_is_commit_delivered_and_rollback_is_silent():
    with engine().connect() as listener:
        listener.execute(text('LISTEN eridani_outbox')); listener.commit()
        raw = listener.connection.driver_connection
        with session_scope() as db:
            wake_dispatch(db)
            assert list(raw.notifies(timeout=.02, stop_after=1)) == []
        assert len(list(raw.notifies(timeout=.2, stop_after=1))) == 1
        with pytest.raises(RuntimeError), session_scope() as db:
            wake_dispatch(db)
            raise RuntimeError('rollback')
        assert list(raw.notifies(timeout=.02, stop_after=1)) == []
        listener.execute(text('UNLISTEN eridani_outbox')); listener.commit()


def test_dispatch_scans_startup_and_wakes_for_committed_work(monkeypatch):
    monkeypatch.setattr(get_settings(), 'worker_interval_seconds', 60)
    ready, stop, received = threading.Event(), threading.Event(), []
    original = worker.dispatch_outbox
    def scan(client, after=None):
        result = original(client, after)
        ready.set()
        return result
    monkeypatch.setattr(worker, 'dispatch_outbox', scan)
    class Client:
        def enqueue(self, options, identity):
            received.append(identity)
            if len(received) == 2: stop.set()
    with session_scope() as db:
        first = enqueue_job(db, 'synthetic', 'reminder', {}).id
    errors = []
    def run():
        try:
            with engine().connect() as lease:
                worker.dispatch_loop(Client(), stop, lease)
        except BaseException as e: errors.append(e)
    thread = threading.Thread(target=run)
    thread.start()
    try:
        assert ready.wait(5)
        assert received == [first]  # startup recovers persisted work before LISTEN existed
        with session_scope() as db:
            second = enqueue_job(db, 'synthetic', 'reminder', {}).id
        assert stop.wait(5)  # fallback is 60s; this requires a committed notification
        assert received == [first, second]
    finally:
        stop.set(); thread.join(5)
    assert not thread.is_alive() and not errors


def test_dispatch_pages_past_blocked_first_thousand(monkeypatch):
    import jarvis.agent_work
    monkeypatch.setattr(jarvis.agent_work, 'eligible', lambda db, job: False)
    with session_scope() as db:
        for i in range(1001):
            identity = str(uuid4())
            db.add(Job(id=identity, owner_id='synthetic', kind='agent_action', payload={},
                       created_at=now()-timedelta(seconds=2000-i)))
            db.add(Outbox(job_id=identity))
        db.flush()
        ready = enqueue_job(db, 'synthetic', 'reminder', {}).id
    sent = []
    client = Mock()
    client.enqueue.side_effect = lambda options, identity: sent.append(identity)
    cursor = worker.dispatch_outbox(client)
    assert cursor is not None and sent == []
    assert worker.dispatch_outbox(client, after=cursor) is None
    assert sent == [ready]


def test_outbox_crash_retry_uses_same_workflow_identity():
    with session_scope() as db:
        identity = enqueue_job(db, 'synthetic', 'reminder', {}).id
    sent = []
    class Client:
        def enqueue(self, options, _):
            sent.append(options['workflow_id'])
            if len(sent)==1: raise RuntimeError('DBOS accepted but response was lost')
    with pytest.raises(RuntimeError): worker.dispatch_outbox(Client())
    with session_scope() as db:
        assert db.get(Outbox, identity).submitted_at is None
    worker.dispatch_outbox(Client())
    assert sent == [identity, identity]  # DBOS deduplicates this durable identity


def test_lost_lease_stops_dispatch(monkeypatch):
    lease = Mock(invalidated=True)
    monkeypatch.setattr(worker, 'dispatch_outbox', Mock())
    with pytest.raises(RuntimeError, match='lease was lost'):
        worker.dispatch_loop(Mock(), threading.Event(), lease)
    worker.dispatch_outbox.assert_not_called()


def test_timings_are_monotonic_bounded_and_include_failure(caplog, monkeypatch):
    ticks = iter([10, 10.25])
    monkeypatch.setattr('jarvis.latency.time.monotonic', lambda: next(ticks))
    with caplog.at_level(logging.INFO, logger='jarvis.latency'):
        with pytest.raises(ValueError), span('model', 'synthetic-id', revision=2, message='private speech'):
            raise ValueError('secret provider body')
    event = json.loads(caplog.records[-1].message.split('latency ',1)[1])
    assert event['duration_ms'] == 250 and event['outcome'] == 'error'
    assert 'private speech' not in caplog.text and 'secret provider body' not in caplog.text
