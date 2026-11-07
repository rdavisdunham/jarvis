"""Run one request through the production backend loop, in process, for tests and offline evals.

The message is accepted as durable work (`agent_work.enqueue`) and executed by `work_runner.run`, the
same path typed chat and delegated voice take. The result has the shape the removed in-process chat loop
(`conversation.chat`) returned, so evals that grade it keep working.
"""

from contextlib import ExitStack, nullcontext
from unittest.mock import patch


async def run_turn(owner, device, turn_id, conversation_id, message, *, account=None, voice_session_id=None,
                   stream=False):
    """Accept and run one request. `stream=False` sends typed requests as one provider call, so a fake
    HTTP client needs only `post`; voice requests never stream. A repeated `turn_id` with the same message
    returns the saved outcome without calling the model again."""
    from jarvis import agent_work, work_runner
    from jarvis.db import session_scope
    from jarvis.models import AgentWork, Job, UserAccount

    with session_scope() as db:
        # Backend work runs for an account (access.authorize_execution); a bare eval owner gets one.
        if not account and not db.get(UserAccount, owner):
            db.add(UserAccount(id=owner, name="Evaluation owner"))
            db.flush()
        agent_work.enqueue(db, owner, account or owner, device, conversation_id, turn_id, message,
                           voice_session_id=voice_session_id)
    ui_actions, finished = [], {}
    real_tool, real_finish = work_runner.call_tool, work_runner.finish

    async def observed(*args, **kwargs):
        outcome = await real_tool(*args, **kwargs)
        if isinstance(outcome, dict) and outcome.get("ui_action"):
            ui_actions.append(outcome["ui_action"])
        return outcome

    def observed_finish(db, row, status, message, **result):
        # A private request keeps only receipts after it finishes; the caller still sees the reply.
        finished.update(message=message, errors=result.get("errors", []))
        return real_finish(db, row, status, message, **result)

    with ExitStack() as stack:
        stack.enter_context(patch.object(work_runner, "call_tool", observed))
        stack.enter_context(patch.object(work_runner, "finish", observed_finish))
        if not stream:
            stack.enter_context(patch.object(work_runner, "streaming", lambda _row: nullcontext()))
        # work_answer hands the turn to a queued continuation; run it in the same turn.
        for _ in range(4):
            await work_runner.run(turn_id)
            with session_scope() as db:
                if db.get(Job, turn_id).status != "queued":
                    break
    with session_scope() as db:
        row, job = db.get(AgentWork, turn_id), db.get(Job, turn_id)
        agent = work_runner.model_for(job.payload.get("profile"), job.payload.get("service_tier", "default"))
        return {
            "turn_id": turn_id,
            "profile": agent.profile_id,
            "provider": agent.provider,
            "model": agent.model,
            "status": job.status,
            "message": finished.get("message", row.result.get("message", "")),
            "actions": row.result.get("actions", []),
            "ui_actions": ui_actions,
            "tool_calls": row.result.get("tool_calls", 0),
            "tool_errors": finished.get("errors", row.result.get("errors", [])),
        }
