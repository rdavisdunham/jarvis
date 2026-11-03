"""Synthetic host events for browser acceptance; never mounted in production.

Authentication, run starts, decision responses, commands, drafts and receipts use the real app.
This fixture supplies provider events in place of paid agents and terminal processes.
"""
from datetime import timedelta
from types import SimpleNamespace
from uuid import uuid4

from jarvis import agent_control, agent_link, agent_plan, run_coordinator as coordinator
from jarvis.api import User, app
from jarvis.db import session_scope
from jarvis.domain import owned
from jarvis.models import BotCredential, Task, now
from jarvis.session_models import AgentCommand, AgentHost, AgentQuestion, AgentSession, ExecutionTarget
from sqlalchemy import select

CAPS = ["session.report", "turn", "status", "exit", "sessions.snapshot", "command.outcome", "question", "context",
        "spawn", "input", "answer", "stop", "runs.snapshot", "decisions.durable", "commands.durable", "run.pause",
        "role.options", "managed.sessions", "activity", "spawn.roles"]
CATALOG = [{"harness": "codex", "model": "gpt-6.1-sol", "label": "GPT-6.1 Sol", "default": True,
            "efforts": ["low", "medium", "high"], "permission_profiles": ["ask"]},
           {"harness": "claude", "model": "opus", "label": "Opus", "default": True,
            "efforts": ["low", "high"], "permission_profiles": ["ask"]}]


def bot_for(run):
    return SimpleNamespace(owner_id=run.owner_id, id=None, scopes=["agents:session"])


@app.post("/api/v1/__test_durable/host")
def host_fixture(user: User):
    with session_scope() as db:
        credential = db.scalar(select(BotCredential).where(BotCredential.owner_id == user.owner_id)
                               .order_by(BotCredential.created_at.desc()))
        if not credential:
            raise ValueError("Enable remotea before creating the synthetic host")
        host = AgentHost(owner_id=user.owner_id, account_id=credential.account_id, credential_id=credential.id,
                         server_id="durable-browser-" + uuid4().hex, name="Synthetic desktop", machine="Acceptance",
                         boot_id="synthetic-boot", conn_epoch=1, lease_until=now() + timedelta(hours=1), last_seen_at=now(),
                         caps={"caps": CAPS, "caps_version": 2, "model_catalog": CATALOG,
                               "harnesses": {name: {"version": "acceptance", "spawn": True, "capture": True,
                                   "observe": True, "question": False, "managed_questions": True, "context": True,
                                   "resume": True, "mcp": True, "managed": True} for name in ("claude", "codex")},
                               "targets": [{"id": "fixture", "repo": "/synthetic/repo", "worktree_root": "/synthetic/worktrees",
                                            "enabled": ["claude", "codex"], "launch": {name: {"ready": True, "managed": True} for name in ("claude", "codex")}, "mcp": {"configured": True}}]})
        db.add(host)
        db.flush()
        target = ExecutionTarget(owner_id=user.owner_id, host_id=host.id, host_target="fixture", name="Synthetic repository",
                                 repo="/synthetic/repo", worktree_root="/synthetic/worktrees", launch={}, verify=[])
        db.add(target)
        db.flush()
        return {"host_id": host.id, "target_id": target.id}


@app.post("/api/v1/__test_durable/runs/{run_id}/{event}")
def event_fixture(run_id: str, event: str, body: dict, user: User):
    with session_scope() as db:
        run = coordinator.load(db, user.owner_id, run_id)
        host = db.get(AgentHost, run.host_id)
        if event == "attach":
            coordinator.attach(db, host.id, run.id, run.generation, run.tool, "synthetic-" + run.id + "-" + str(run.attempt),
                               surf=coordinator.surf_for(run.id, run.generation, run.attempt), incarnation=1)
        elif event == "interview":
            return agent_plan.task_interview(db, bot_for(run), agent_plan.InterviewInput(
                run_token=coordinator.run_token(run), questions=[
                    {"id": "scope", "question": "Which widgets should we improve?", "options": [{"label": "All widgets"}, {"label": "Charts"}]},
                    {"id": "target", "question": "What latency target?"}]))
        elif event == "plan":
            task = owned(db, Task, run.task_id, user.owner_id)
            return agent_plan.submit_plan(db, bot_for(run), agent_plan.SubmitPlanInput(
                run_token=coordinator.run_token(run), submission_id=uuid4(), expected_plan_revision=task.plan_revision,
                prd=body.get("prd", "Measure widget latency, implement the focused fix, and open a pull request."),
                steps=[{"title": "Measure widget latency"}, {"title": "Implement and test the fix"}],
                success_criteria=[{"text": "The latency target is measured and met"}]))
        elif event == "activity":
            agent_link.record_activity(db, bot_for(run), host.id, agent_link.Activity(
                run_id=run.id, generation=run.generation, surf=run.surf, incarnation=run.incarnation,
                kind=body.get("kind", "editing"), label=body.get("label", "Editing"), sequence=body.get("sequence", 1),
                session_id="synthetic-" + run.id + "-" + str(run.attempt), at=now()))
        elif event == "permission":
            question, _ = coordinator.record_question(db, user.owner_id, host.id, agent_link.Question(
                run_id=run.id, generation=run.generation, provider=run.tool, provider_session_id="synthetic-" + run.id + "-" + str(run.attempt),
                question_id="permission-" + uuid4().hex, provider_request_id="request-" + uuid4().hex,
                surf=run.surf, incarnation=run.incarnation, durable=True, kind="permission",
                action={"tool": "shell", "cwd": "/synthetic/repo", "input": {"command": "npm test", "purpose": "Verify the latency fix"},
                        "permissions": {"network": False, "write_paths": []}, "digest": "a" * 64},
                request_digest="a" * 64,
                items=[{"id": "permission", "text": "Run the project tests?", "choices": [
                    {"id": "allow_once", "label": "Approve once"}, {"id": "deny", "label": "Deny"}]}]))
            db.flush()
            return {"question_id": question.id}
        elif event == "consume":
            for command in db.scalars(select(AgentCommand).where(AgentCommand.run_id == run.id,
                                      AgentCommand.status.in_(["queued", "sent", "acked", "deferred"]), AgentCommand.kind.in_(["input", "answer"]))):
                coordinator.on_outcome(db, command, "executed", "Synthetic provider acknowledgement")
            for question in db.scalars(select(AgentQuestion).where(AgentQuestion.run_id == run.id, AgentQuestion.status == "submitted")):
                coordinator.question_resolved(db, host.id, agent_link.QuestionResolved(
                    question_id=question.question_id, generation=question.generation, outcome="accepted"))
        elif event == "finish":
            for command in db.scalars(select(AgentCommand).where(AgentCommand.run_id == run.id,
                                      AgentCommand.kind == "stop", AgentCommand.status.in_(["queued", "sent", "acked"]))):
                coordinator.on_outcome(db, command, "executed", "Synthetic process termination confirmed")
        elif event == "takeover":
            session = db.get(AgentSession, run.session_pk)
            coordinator.on_status(db, session, agent_link.Status(
                surf=run.surf, incarnation=run.incarnation, tool=run.tool,
                session_id="synthetic-" + run.id + "-" + str(run.attempt), state="needs_you", since=now(),
                execution="managed", controller="terminal"), host_id=host.id)
        elif event == "absent":
            agent_link.snapshot(db, bot_for(run), host.id, agent_link.Snapshot(surfaces=[], complete=True, at=now()))
        elif event == "exit":
            session = db.get(AgentSession, run.session_pk)
            agent_link.exited(db, bot_for(run), host.id, agent_link.Exit(
                surf=run.surf, incarnation=run.incarnation, provider=run.tool, provider_session_id=session.session_id,
                code=0, exited_at=now()))
        elif event == "state":
            # Fixture-only provider/coordinator boundary; UI lifecycle actions always use production routes.
            state = body["state"]
            if state not in {"verifying", "awaiting_approval", "completed"}:
                raise ValueError("Unsupported synthetic state")
            run.state = state
            run.state_revision += 1
            if state == "completed":
                run.finished_at = now()
        else:
            raise ValueError("Unknown synthetic provider event")
        db.flush()
        return agent_control.public_run(db, run, detail=True)
