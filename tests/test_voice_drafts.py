from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from uuid import uuid4

import pytest
from cryptography.fernet import Fernet
from sqlalchemy import select
from jarvis import work_intake
from jarvis.config import get_settings
from jarvis.db import session_scope
from jarvis.domain import DomainError
from jarvis.models import AgentWork, AuthSession, Job, Outbox, VoiceInbox, now
from jarvis.work_crypto import unseal


@pytest.fixture(autouse=True)
def setup(monkeypatch):
    monkeypatch.setattr(get_settings(), "integration_encryption_key", Fernet.generate_key().decode())
    monkeypatch.setattr(get_settings(), "openai_api_key", "synthetic")


def draft(client, message="Introduction email to Greg, head of marketing at ABC"):
    conv = client.post('/api/v1/conversations', json={}).json()['id']
    with session_scope() as db:
        session = db.scalar(select(AuthSession))
        row = work_intake.open_voice(db, str(uuid4()), session.owner_id, session.owner_id, session.device_id, conv)
        work_intake.append_voice(db, row.id, 'speech', 'user', message, 0, 1000)
        work_intake.claim_voice(db, row, close=True)
        return row.id, row.owner_id, row.account_id, row.device_id


def test_draft_is_not_work_and_send_replays_without_duplicate(client):
    identity, *_ = draft(client)
    listed = client.get('/api/v1/voice/drafts').json()['items']
    assert len(listed) == 1 and listed[0]['id'] == identity
    assert client.get('/api/v1/work').json()['items'] == []
    url = '/api/v1/voice/drafts/' + identity + '/send'
    body = {'message': 'Add a task to email Greg at ABC'}
    first = client.post(url, json=body)
    assert first.status_code == 200, first.text
    assert client.post(url, json=body).json() == first.json()
    assert client.post(url, json={'message': 'A different request'}).status_code == 409
    assert client.get('/api/v1/voice/drafts').json()['items'] == []
    with session_scope() as db:
        work = db.scalar(select(AgentWork))
        assert unseal(work.input_ciphertext)['message'] == body['message']
        assert db.get(Outbox, work.id)
        assert len(list(db.scalars(select(Job)))) == 1
        assert 'draft' not in unseal(db.get(VoiceInbox, identity).content_ciphertext)


def test_discard_prevents_late_send_and_removes_speech(client):
    identity, *_ = draft(client)
    url = '/api/v1/voice/drafts/' + identity
    assert client.post(url + '/discard').json() == {'status': 'discarded'}
    assert client.post(url + '/send', json={'message':'Add Greg'}).json() == {'status':'discarded'}
    with session_scope() as db:
        assert db.scalar(select(AgentWork)) is None
        assert 'Greg' not in str(unseal(db.get(VoiceInbox, identity).content_ciphertext))


@pytest.mark.parametrize('index', [0,1,2])
def test_draft_owner_account_and_device_are_all_required(client, index):
    identity, *scope = draft(client)
    scope[index] = 'other'
    with session_scope() as db:
        assert work_intake.list_drafts(db, *scope)['items'] == []
        with pytest.raises(DomainError) as exc:
            work_intake.resolve_draft(db, *scope, identity, message='Add Greg')
        assert exc.value.status == 404


def test_draft_expiry_and_invalid_send_do_not_create_work(client):
    identity, *_ = draft(client)
    url = '/api/v1/voice/drafts/' + identity
    assert client.post(url+'/send', json={'message':'  '}).status_code == 400
    with session_scope() as db:
        db.get(VoiceInbox, identity).expires_at = now()-timedelta(seconds=1)
    assert client.get('/api/v1/voice/drafts').json()['items'] == []
    assert client.post(url+'/send', json={'message':'Add Greg'}).status_code == 410
    with session_scope() as db:
        assert db.scalar(select(AgentWork)) is None


def test_concurrent_send_and_discard_have_one_durable_disposition(client):
    identity, *scope = draft(client)
    def resolve(message):
        with session_scope() as db:
            return work_intake.resolve_draft(db, *scope, identity, message=message)
    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = list(pool.map(resolve, ['Add Greg', None]))
    assert outcomes[0] == outcomes[1]
    with session_scope() as db:
        assert len(list(db.scalars(select(AgentWork)))) == (1 if outcomes[0]['status']=='sent' else 0)


def test_rolled_back_send_retains_draft_and_can_retry(client):
    identity, *scope = draft(client)
    with pytest.raises(RuntimeError), session_scope() as db:
        work_intake.resolve_draft(db, *scope, identity, message='Add Greg')
        raise RuntimeError('simulate failed transaction')
    assert len(client.get('/api/v1/voice/drafts').json()['items']) == 1
    with session_scope() as db:
        assert db.scalar(select(AgentWork)) is None
        assert work_intake.resolve_draft(db, *scope, identity, message='Add Greg')['status'] == 'sent'


@pytest.mark.parametrize('answer', ['Yes', 'No', 'Okay', '9', 'ABC'])
def test_short_answer_is_preserved_without_an_action_verb(client, answer):
    identity, *_ = draft(client, answer)
    with session_scope() as db:
        assert unseal(db.get(VoiceInbox, identity).content_ciphertext)['draft']['message'] == answer
