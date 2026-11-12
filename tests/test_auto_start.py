"""Auto-start: Auto-route tasks start their own run and their plan goes ahead without the owner (auto_start.py)."""

from datetime import timedelta

import pytest
from jarvis import agent_plan, auto_start
from jarvis import run_coordinator as coordinator
from jarvis.db import session_scope
from jarvis.domain import DomainError
from jarvis.models import Task, now
from jarvis.session_models import AgentHost, AgentRun, ExecutionTarget
from sqlalchemy import select
from test_agent_plan import PLAN, call
from test_agent_runs import OWNER, attached, quiet, task  # noqa: F401 - pytest fixtures
from test_agent_runs import host as host  # noqa: F401 - pytest fixture
from test_linear import run


def allow(host, **values):
    with session_scope() as db:
        return auto_start.save(db, OWNER, auto_start.Settings(enabled=True, targets=[host.target_id], **values))


def ready(host, title="Rename a helper", **values):
    """A task the readiness pass would have left on the Auto route, ready to run."""
    made = task(title, execution_target_id=host.target_id, run_route="auto",
                success_criteria=[{"text": "The helper has its new name everywhere", "by": "agent"}], **values)
    with session_scope() as db:
        row = db.get(Task, made["id"])
        row.readiness = {**row.readiness, "state": "ready"}
    return made


def submit(run_id, task_id, **extra):
    with session_scope() as db:
        revision = db.get(Task, task_id).plan_revision
    return call(agent_plan.submit_plan, agent_plan.SubmitPlanInput, run_id,
                **{**PLAN, "expected_plan_revision": revision, **extra})


def sweep():
    with session_scope() as db:
        auto_start.sweep(db)


def runs(task_id=None):
    with session_scope() as db:
        query = select(AgentRun).order_by(AgentRun.created_at)
        rows = list(db.scalars(query.where(AgentRun.task_id == task_id) if task_id else query))
        db.expunge_all()
        return rows


def test_nothing_starts_until_the_owner_turns_it_on_and_names_a_repository(host):
    made = ready(host)
    sweep()
    assert runs() == []
    with session_scope() as db:
        assert auto_start.public(db, OWNER)["enabled"] is False
        auto_start.save(db, OWNER, auto_start.Settings(enabled=True))  # On, but no repository named.
        with pytest.raises(DomainError):
            auto_start.save(db, OWNER, auto_start.Settings(enabled=True, targets=["not-a-target"]))
    sweep()
    assert runs(made["id"]) == []


def test_an_auto_route_task_starts_plans_and_executes_without_the_owner(host):
    made = ready(host)
    shown = allow(host)
    assert shown["choices"][0]["id"] == host.target_id and shown["started_today"] == 0
    sweep()
    [row] = runs(made["id"])
    assert row.sync_state["auto_start"] and row.sync_state["start_phase"] == "planning" and row.state == "spawning"
    with session_scope() as db:
        assert db.get(Task, made["id"]).readiness["auto_start"]["run_id"] == row.id
        assert auto_start.public(db, OWNER)["started_today"] == 1
    sweep()  # One attempt: no second run, now or after this one ends.
    assert len(runs(made["id"])) == 1
    attached(host, row.id)
    submit(row.id, made["id"])
    with session_scope() as db:
        after = db.get(AgentRun, row.id)
        plan = coordinator.frozen(db, after)
        assert plan.approval_status == "approved" and after.state == "executing"
        assert [h["actor"] for h in after.history if h.get("event") == "plan_approved"] == [{"kind": "auto_start"}]


def test_a_migration_or_a_changed_route_still_stops_for_the_owner(host):
    allow(host)
    first, second = ready(host, "Add a column"), ready(host, "Rename another helper")
    sweep()
    risky, moved = runs(first["id"])[0], runs(second["id"])[0]
    attached(host, risky.id)
    submit(risky.id, first["id"], migration=True)
    with session_scope() as db:
        row = db.get(AgentRun, risky.id)
        assert (row.state, row.state_reason) == ("needs_you", "plan_approval")
    attached(host, moved.id, session_id="sess-2")
    with session_scope() as db:
        row = db.get(Task, second["id"])
        row.run_route = "direction"  # The owner moved it off the Auto route while the agent planned.
    submit(moved.id, second["id"])
    with session_scope() as db:
        row = db.get(AgentRun, moved.id)
        assert (row.state, row.state_reason) == ("needs_you", "plan_approval")


def test_caps_allowed_repositories_and_an_offline_host_hold_tasks_back(host):
    with session_scope() as db:
        other = ExecutionTarget(owner_id=OWNER, host_id=host.id, host_target="acme", name="acme",
                                repo="/home/davin/acme", worktree_root="/home/davin/wt", launch={}, verify=[])
        db.add(other)
        db.flush()
        other_id = other.id
    elsewhere = ready(host, "Fix the acme importer")
    run("task.update", task_id=elsewhere["id"], expected_revision=elsewhere["revision"], execution_target_id=other_id)
    not_ready = task("Not looked at yet", execution_target_id=host.target_id, run_route="auto",
                     success_criteria=[{"text": "Done"}])
    backlog = ready(host, "Later", status="backlog")
    one, two = ready(host, "First"), ready(host, "Second")
    allow(host, daily_cap=1)
    with session_scope() as db:
        db.get(AgentHost, host.id).lease_until = now() - timedelta(minutes=1)
    sweep()
    assert runs() == []  # Offline: nothing starts, and it is not counted as an attempt.
    with session_scope() as db:
        assert "auto_start" not in db.get(Task, one["id"]).readiness
        db.get(AgentHost, host.id).lease_until = now() + timedelta(hours=1)
    sweep()
    assert [r.task_id for r in runs()] == [one["id"]]  # Oldest first, and the daily cap of one holds the rest.
    allow(host, daily_cap=5, host_cap=1)
    sweep()
    assert [r.task_id for r in runs()] == [one["id"]]  # The host already has its one working run.
    allow(host, daily_cap=5, host_cap=3)
    sweep()
    assert [r.task_id for r in runs()] == [one["id"], two["id"]]
    for skipped in (elsewhere, not_ready, backlog):
        assert runs(skipped["id"]) == []
