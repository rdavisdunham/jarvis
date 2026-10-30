"""Retention and redaction of run content (Eridani task 18e3a5f3, Astra's review E-S3; agent_retention.py)."""

import re
from datetime import timedelta
from pathlib import Path

from fastapi.testclient import TestClient
from jarvis import agent_control as control
from jarvis import agent_retention as retention
from jarvis import session_models
from jarvis.agent_actor import Actor
from jarvis.api import app
from jarvis.db import session_scope
from jarvis.models import Base, Notification, OwnerSettings, now
from jarvis.session_models import (
    AgentActionReceipt,
    AgentAttempt,
    AgentCommand,
    AgentInterview,
    AgentPendingControl,
    AgentQuestion,
    AgentRun,
    AgentRunPlan,
    AgentSession,
    AgentTurn,
)
from sqlalchemy import func, select
from test_agent_link import bearer
from test_agent_runs import (
    OWNER,
    host,  # noqa: F401 - fixture
    quiet,  # noqa: F401 - fixture
    start,
    task,
)
from test_linear import linear  # noqa: F401 - fixture

SECRET = "the customer's private roadmap"
DOC = Path(__file__).resolve().parents[1] / "docs/AGENT_CONTROL.md"


def seeded(host, title, *, finished_days):  # noqa: F811
    """A settled run with a copy of SECRET in every place run content lives."""
    run_id = start(host, task(title)["id"])["id"]
    with session_scope() as db:
        run = db.get(AgentRun, run_id)
        session = AgentSession(owner_id=OWNER, tool="claude", session_id="s-" + run_id, run_id=run_id,
                               summary="ledger summary")
        db.add(session)
        db.flush()
        run.session_pk = session.id
        control.sent(run, Actor("human"), "please ship " + SECRET, mode="queue")
        from jarvis import run_coordinator as coordinator

        coordinator.note(run, "outcome", kind="input", op_id="op", state="failed", detail="host said " + SECRET)
        plan = db.scalar(select(AgentRunPlan).where(AgentRunPlan.run_id == run_id))
        if plan is None:
            plan = AgentRunPlan(owner_id=OWNER, run_id=run_id, revision=1)
            db.add(plan)
            run.plan_revision = 1
        plan.prd_text = "PRD: " + SECRET
        plan.criteria = [{"id": "c1", "text": "Tests pass", "by": "human"}]
        plan.proposals = {"criteria": [{"id": "c1", "text": SECRET, "because": SECRET}]}
        for i, (role, text) in enumerate((("user", SECRET), ("assistant", SECRET))):
            db.add(AgentTurn(owner_id=OWNER, session_pk=session.id, host_id=host.id, provider_session_id="p",
                             provider_turn_id=f"{run_id}-{i}", run_id=run_id, seq=i, role=role, text=text))
        db.add(AgentTurn(owner_id=OWNER, session_pk=session.id, host_id=host.id, provider_session_id="p",
                         provider_turn_id=f"{run_id}-loose", run_id=None, seq=9, text=SECRET))  # No run id on the frame.
        db.add(AgentAttempt(owner_id=OWNER, run_id=run_id, kind="judge", status="done", result={
            "verdict": "not_met", "gap": SECRET, "summary": SECRET, "tail": SECRET, "rules": ["r"],
            "criteria": [{"id": "c1", "status": "not_met", "evidence": SECRET}]}))
        db.add(AgentQuestion(owner_id=OWNER, session_pk=session.id, host_id=host.id, run_id=run_id, question_id="q",
                             provider_request_id="pr-" + run_id, items=[{"id": "i", "text": SECRET}]))
        db.add(AgentInterview(owner_id=OWNER, run_id=run_id, questions=[{"id": "i", "text": SECRET}],
                              answers={"i": SECRET}))
        db.add(Notification(owner_id=OWNER, dedup_key="agent:agent_question:" + run_id, category="agent_question",
                            title="An agent asked you something", body=SECRET, scheduled_at=now(), eligible_at=now(),
                            target={"view": "remotea", "run_id": run_id, "task_id": run.task_id}))
        db.add(AgentPendingControl(owner_id=OWNER, account_id=OWNER, tool="control.message", run_id=run_id,
                                   arguments={"run_id": run_id, "text": SECRET}, summary="Send " + SECRET,
                                   status="declined", expires_at=now() + timedelta(minutes=10)))
        db.add(AgentActionReceipt(owner_id=OWNER, action_id="a-" + run_id[:30], run_id=run_id, kind="message",
                                  digest="d", result={"history": [{"text": SECRET}]}))
        run.state, run.finished_at = "completed", now() - timedelta(days=finished_days)
    return run_id


def leaks(run_id):
    """Every row that still holds SECRET for this run, by table."""
    with session_scope() as db:
        run = db.get(AgentRun, run_id)
        found = {"history": SECRET in str(run.history)}
        for model in (AgentTurn, AgentAttempt, AgentQuestion, AgentInterview, AgentPendingControl, AgentActionReceipt):
            found[model.__tablename__] = any(SECRET in str(retention_view(row))
                                             for row in db.scalars(select(model).where(model.run_id == run_id)))
        found["agent_turns_unattributed"] = bool(db.scalar(select(func.count(AgentTurn.id)).where(
            AgentTurn.session_pk == run.session_pk, AgentTurn.text.contains(SECRET))))
        found["agent_run_plans"] = any(SECRET in p.prd_text + str(p.proposals)
                                       for p in db.scalars(select(AgentRunPlan).where(AgentRunPlan.run_id == run_id)))
        found["notifications"] = bool(db.scalar(select(func.count(Notification.id)).where(
            Notification.target["run_id"].astext == run_id, Notification.body.contains(SECRET))))
        return {k for k, v in found.items() if v}


def retention_view(row):
    return {c.name: getattr(row, c.name) for c in row.__table__.columns}


def test_an_expired_run_loses_its_children_and_every_copy_of_its_text(host):  # noqa: F811
    old = seeded(host, "Old run", finished_days=91)
    young = seeded(host, "Young run", finished_days=10)
    assert leaks(old) == leaks(young) == {"history", "agent_turns", "agent_attempts", "agent_questions",
                                         "agent_interviews", "agent_pending_controls", "agent_action_receipts",
                                         "agent_turns_unattributed", "agent_run_plans", "notifications"}
    with session_scope() as db:
        before = list(db.get(AgentRun, old).history)
        counts = retention.sweep(db)
    assert counts["run_turns"] == 3 and counts["run_attempts"] == 1 and counts["run_questions"] == 1
    assert counts["run_interviews"] == 1 and counts["redacted_runs"] == 1
    assert leaks(old) == set()
    with session_scope() as db:
        for model in (AgentTurn, AgentAttempt, AgentQuestion, AgentInterview, AgentCommand):
            assert not db.scalars(select(model).where(model.run_id == old)).all()
        run = db.get(AgentRun, old)
        # The skeleton: every earlier entry is still there with its actor, time, event and reason.
        assert len(run.history) == len(before) + 1
        for kept, was in zip(run.history, before, strict=False):
            assert {k: v for k, v in kept.items() if k not in ("text", "detail")} == \
                   {k: v for k, v in was.items() if k not in ("text", "detail")}
        sent = next(e for e in run.history if e.get("event") == "sent")
        assert sent["text"] == retention.MARKER and sent["actor"]["kind"] == "human" and sent["mode"] == "queue"
        outcome = next(e for e in run.history if e.get("event") == "outcome")
        assert outcome["detail"] == retention.MARKER and outcome["state"] == "failed"
        assert run.history[-1]["event"] == "redacted" and run.history[-1]["reason"] == "retention"
        plan = db.scalar(select(AgentRunPlan).where(AgentRunPlan.run_id == old))
        assert plan.prd_text == retention.MARKER and plan.proposals is None
        assert plan.criteria == [{"id": "c1", "text": "Tests pass", "by": "human"}]  # What the judge used.
        assert db.scalar(select(AgentSession.summary).where(AgentSession.run_id == old)) == "ledger summary"
        detail = control.public_run(db, run, detail=True)
        assert SECRET not in str(detail) and detail["plan"]["prd_text"] == retention.MARKER
    with session_scope() as db:
        assert retention.sweep(db)["redacted_runs"] == 0  # Once per run.
    assert leaks(young) == {"history", "agent_turns", "agent_attempts", "agent_questions", "agent_interviews",
                            "agent_pending_controls", "agent_action_receipts", "agent_turns_unattributed",
                            "agent_run_plans", "notifications"}  # Untouched.


def test_the_owner_window_is_configurable(client, host):  # noqa: F811
    run_id = seeded(host, "Short window", finished_days=10)
    assert client.get("/api/v1/remotea/status").json()["retention_days"] == 90
    assert client.post("/api/v1/remotea/retention", json={"days": 0}).status_code == 422
    response = client.post("/api/v1/remotea/retention", json={"days": 7})
    assert response.status_code == 200 and response.json()["retention_days"] == 7
    with session_scope() as db:
        assert db.get(OwnerSettings, OWNER).values[retention.SETTING] == 7
        assert retention.sweep(db)["redacted_runs"] == 1
    assert leaks(run_id) == set()


def test_the_redact_route_scrubs_one_run_now_and_keeps_verdicts(client, host):  # noqa: F811
    run_id = seeded(host, "Redact me", finished_days=1)
    other = seeded(host, "Leave me", finished_days=1)
    response = client.post(f"/api/v1/remotea/runs/{run_id}/redact")
    assert response.status_code == 200, response.text
    body = response.json()
    assert SECRET not in str(body["run"]) and body["redacted"]["run_turns"] == 3
    assert body["run"]["history"][-1]["event"] == "redacted" and body["run"]["history"][-1]["reason"] == "owner"
    assert leaks(run_id) == set() and "history" in leaks(other)
    with session_scope() as db:
        attempt = db.scalar(select(AgentAttempt).where(AgentAttempt.run_id == run_id))
        assert attempt.result["verdict"] == "not_met" and attempt.result["rules"] == ["r"]
        assert attempt.result["criteria"] == [{"id": "c1", "status": "not_met", "evidence": retention.MARKER}]
        assert attempt.result["gap"] == attempt.result["tail"] == retention.MARKER
    # Bots read the same rows: agents:read shows the marker.
    reader = client.post("/api/v1/bot-keys", json={"name": "reader", "expires_in_days": 30,
                                                   "scopes": ["agents:read", "tasks:read"]}).json()["token"]
    read = TestClient(app).get(f"/api/v1/external/agents/runs/{run_id}", headers=bearer(reader))
    assert read.status_code == 200, read.text
    assert SECRET not in read.text and retention.MARKER in read.text
    # An unsettled run is refused.
    live = start(host, task("Still running")["id"])["id"]
    refused = client.post(f"/api/v1/remotea/runs/{live}/redact")
    assert refused.status_code == 409 and refused.json()["error"]["code"] == "RUN_ACTIVE"


def test_the_redact_route_refuses_anyone_but_the_owner(client, host):  # noqa: F811
    from test_accounts import client_for, shared

    run_id = seeded(host, "Mine", finished_days=1)
    stranger = client_for("stranger")
    assert stranger.post(f"/api/v1/remotea/runs/{run_id}/redact").status_code == 404  # Another person's run.
    assert stranger.post("/api/v1/remotea/retention", json={"days": 1}).status_code == 200  # Their own setting.
    guest = client_for("guest")
    shared(client, guest)
    for who in (client, guest):  # In a shared workspace neither the owner nor an editor may redact.
        refused = who.post(f"/api/v1/remotea/runs/{run_id}/redact")
        assert refused.status_code == 403
        assert who.post("/api/v1/remotea/retention", json={"days": 1}).status_code == 403
    assert "history" in leaks(run_id)
    with session_scope() as db:
        assert db.get(OwnerSettings, OWNER) is None or retention.SETTING not in db.get(OwnerSettings, OWNER).values


def test_the_account_export_includes_the_run_tables(client, host):  # noqa: F811
    run_id = seeded(host, "Exported", finished_days=1)
    exported = client.get("/api/v1/export").json()
    for name in ("agent_runs", "agent_run_plans", "agent_attempts", "agent_questions", "agent_interviews",
                 "agent_sessions"):
        assert name in exported, name
    run = next(r for r in exported["agent_runs"] if r["id"] == run_id)
    assert "run_token_hash" not in run and "run_token_salt" not in run and SECRET in str(run["history"])
    assert any(p["run_id"] == run_id and SECRET in p["prd_text"] for p in exported["agent_run_plans"])
    # Excluded by name, with the reason in the doc.
    section = retention_section()
    for name in ("agent_turns", "agent_commands"):
        assert name not in exported and re.search(rf"leaves out[^.]*`{name}`", section, re.S), name
    client.post(f"/api/v1/remotea/runs/{run_id}/redact")
    assert SECRET not in client.get("/api/v1/export").text  # Redacted rows export the marker.


def retention_section():
    text = DOC.read_text()
    start_at = text.index("### Retention and redaction")
    return text[start_at:text.index("\n### ", start_at + 1)]


def test_every_agent_table_is_in_the_retention_doc():
    """Drift guard: every table in session_models.py is named in AGENT_CONTROL.md "Retention and redaction"."""
    tables = {m.class_.__tablename__ for m in Base.registry.mappers if m.class_.__module__ == session_models.__name__}
    assert {"agent_runs", "agent_run_plans", "agent_turns", "agent_action_receipts"} <= tables
    section = retention_section()
    missing = sorted(t for t in tables if f"`{t}`" not in section)
    assert not missing, f"Name these tables in docs/AGENT_CONTROL.md 'Retention and redaction': {missing}"
    for model in retention.EXPORTED:  # And the export list matches what the doc says is exported.
        assert model.__tablename__ in section.split("**Account export.**")[1].split("It leaves out")[0]
