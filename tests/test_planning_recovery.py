import pytest
from jarvis import agent_planning_recovery as recovery
from jarvis import external_mcp
from jarvis import run_coordinator as coordinator
from jarvis.db import session_scope
from jarvis.domain import DomainError
from jarvis.session_models import AgentCommand, AgentRun
from sqlalchemy import select
from test_agent_plan import BOT
from test_agent_runs import HUMAN
from test_agent_runs import host as host  # noqa: PLC0414
from test_agent_runs import quiet as quiet  # noqa: PLC0414
from test_durable_runs import durable as durable  # noqa: PLC0414
from test_durable_runs import new_run


def test_bound_tools_do_not_require_the_model_to_copy_a_token(durable, monkeypatch):
    rid = new_run(durable)
    with session_scope() as db:
        token = coordinator.run_token(db.get(AgentRun, rid))
    monkeypatch.setattr(external_mcp.bot_access, 'authorize', lambda *a, **k: BOT)
    context = external_mcp._run_binding.set(token)
    try:
        result = external_mcp.dispatch('task_plan', {})
        assert result['actor']['verified'] is True
        assert result['run']['id'] == rid
        with pytest.raises(DomainError) as error:
            external_mcp.dispatch('task_plan', {'run_token': 'different'})
        assert error.value.code == 'RUN_CONTEXT_CONFLICT'
    finally:
        external_mcp._run_binding.reset(context)
    assert external_mcp._run_binding.get() is None


def failed(db, rid, outcome='rejected'):
    run = db.get(AgentRun, rid)
    coordinator.transition(db, run, 'needs_you', reason='no_plan')
    run.sync_state = {**(run.sync_state or {}), 'coordination_failure': {
        'code': 'RUN_TOKEN_REQUIRED', 'tool': 'task_interview', 'outcome': outcome,
        'request_id': 'original-call', 'binding': True}}
    return run


def test_automatic_recovery_is_durable_and_only_runs_once(durable):
    rid = new_run(durable)
    with session_scope() as db:
        run = failed(db, rid)
        assert recovery.retry(db, run, actor=HUMAN, automatic=True)
        assert run.state == 'interviewing'
    with session_scope() as db:
        run = db.get(AgentRun, rid)
        assert not recovery.retry(db, run, actor=HUMAN, automatic=True)
        queued = list(db.scalars(select(AgentCommand).where(AgentCommand.run_id == rid, AgentCommand.kind == 'input')))
        assert len(queued) == 1
        assert 'SAME prepared interview' in queued[0].payload['text']


@pytest.mark.parametrize('outcome', ['uncertain'])
def test_uncertain_writes_are_never_replayed(durable, outcome):
    rid = new_run(durable)
    with session_scope() as db:
        run = failed(db, rid, outcome)
        with pytest.raises(DomainError, match='Reconcile'):
            recovery.retry(db, run, actor=HUMAN, automatic=True)


def test_operator_pause_is_not_overridden(durable):
    rid = new_run(durable)
    with session_scope() as db:
        run = failed(db, rid)
        run.paused = True
        run.sync_state = {**run.sync_state, 'operator_paused': True}
        with pytest.raises(DomainError, match='Resume'):
            recovery.retry(db, run, actor=HUMAN, automatic=True)


def test_verified_planner_can_restore_rejected_interview_without_a_new_run(durable):
    from jarvis import agent_plan
    from test_agent_plan import call
    rid = new_run(durable)
    with session_scope() as db:
        failed(db, rid)
    result = call(agent_plan.task_interview, agent_plan.InterviewInput, rid,
                  questions=[{'id':'original','question':'Which scope?','free_text':True}])
    assert result['status'] == 'open'
    with session_scope() as db:
        run = db.get(AgentRun, rid)
        assert run.state == 'interviewing'
        assert 'coordination_failure' not in run.sync_state


def test_root_turn_automatically_retries_and_new_provider_ids_cannot_loop(durable):
    from types import SimpleNamespace
    rid = new_run(durable)
    with session_scope() as db:
        run = failed(db, rid)
        run.paused = True
        coordinator.record_turn(db, run, SimpleNamespace(generation=run.generation, interim=False, synthetic=False))
        assert run.state == 'interviewing'
        assert run.sync_state['planning_auto_generation'] == run.generation
        run.sync_state = {**run.sync_state, 'coordination_failure': {**run.sync_state['coordination_failure'], 'recovering': False, 'request_id': 'new-provider-id'}}
        assert not recovery.retry(db, run, actor=HUMAN, automatic=True)
        coordinator.settle_planning(db, run)
        assert run.state == 'needs_you'
        assert not run.sync_state['coordination_failure']['recovering']


def test_mcp_header_binding_is_request_local(monkeypatch):
    import asyncio

    from jarvis import bot_access
    monkeypatch.setattr(bot_access, 'authenticate', lambda request: 'credential')
    observations = []
    async def application(scope, receive, send):
        before = external_mcp._run_binding.get()
        await asyncio.sleep(0)
        observations.append((before, external_mcp._run_binding.get()))
    mount = external_mcp.MCPMount()
    mount.app = application
    async def invoke(value):
        await mount({'type':'http', 'headers':[(b'x-eridani-run-token',value.encode())]}, None, None)
    async def both():
        await asyncio.gather(invoke('first-run'), invoke('second-run'))
    asyncio.run(both())
    assert sorted(observations) == [('first-run','first-run'),('second-run','second-run')]
    assert external_mcp._run_binding.get() is None


@pytest.mark.parametrize("reason", ["no_plan", "exited"])
def test_real_mcp_transport_binds_the_run_and_saves_its_interview(durable, client, reason):
    from test_external_agents import key, rpc
    rid = new_run(durable)
    with session_scope() as db:
        run = failed(db, rid)
        if reason == 'exited':
            from jarvis.session_models import AgentSession
            run.state_reason = 'exited'
            row = db.get(AgentSession, run.session_pk)
            row.state, row.surf, row.incarnation = 'idle', run.surf, run.incarnation + 1
            run.incarnation = row.incarnation
        bound_token = coordinator.run_token(run)
    _, headers = key(client, scopes=['agents:session'])
    headers = {**headers, 'X-Eridani-Run-Token': bound_token}
    response = rpc(client, headers, 'tools/call', {'name':'task_plan','arguments':{}})
    assert response.status_code == 200
    result = response.json()['result']
    assert not result.get('isError'), result
    assert result['structuredContent']['actor']['verified'] is True
    assert result['structuredContent']['run']['id'] == rid
    response = rpc(client, headers, 'tools/call', {'name':'task_interview','arguments':{
        'questions':[{'id':'existing-question','question':'Which scope?','free_text':True}]}})
    result = response.json()['result']
    assert not result.get('isError'), result
    assert result['structuredContent']['status'] == 'open'
    assert external_mcp._run_binding.get() is None


def test_an_exited_planner_cannot_submit_until_its_host_confirms_it_alive(durable):
    from jarvis import agent_plan
    from jarvis.session_models import AgentSession
    from test_agent_plan import call
    rid = new_run(durable)
    with session_scope() as db:
        run = failed(db, rid)
        run.state_reason = 'exited'
        db.get(AgentSession, run.session_pk).state = 'exited'
    with pytest.raises(DomainError, match='needs_you'):
        call(agent_plan.task_interview, agent_plan.InterviewInput, rid,
             questions=[{'id':'original','question':'Which scope?','free_text':True}])


def test_validation_recovery_preserves_answers_and_requires_approval(durable):
    rid = new_run(durable)
    with session_scope() as db:
        run = failed(db, rid)
        run.sync_state = {**run.sync_state, 'coordination_failure': {
            'code': 'INVALID_ARGUMENT', 'tool': 'submit_plan', 'outcome': 'rejected', 'request_id': 'bad-plan'}}
        assert recovery.retry(db, run, actor=HUMAN)
        prompt = db.scalar(select(AgentCommand).where(AgentCommand.run_id == rid, AgentCommand.kind == 'input')).payload['text']
        assert 'validation error' in prompt
        assert "new submission_id" in prompt
        assert 'wait for human approval' in prompt
        assert 'token' not in prompt


def test_a_planning_tool_error_is_not_presented_as_a_missing_interview_answer_receipt():
    from types import SimpleNamespace
    from jarvis import run_attention
    run = SimpleNamespace(state='needs_you', state_reason='receipt_uncertain',
                          sync_state={'coordination_failure': {'outcome': 'uncertain'}})
    result = run_attention.attention(None, run)
    assert 'planning submission' in result['headline']
    assert 'answer' not in result['headline']
    run.sync_state['coordination_failure']['outcome'] = 'rejected'
    assert 'rejected' in run_attention.attention(None, run)['headline']
    run.sync_state = {}
    assert 'answer was saved' in run_attention.attention(None, run)['headline']
