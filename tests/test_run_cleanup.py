"""Only explicit terminal run state authorizes exact, replay-safe terminal cleanup."""
from datetime import timedelta

import pytest
from jarvis import agent_cleanup, agent_control, agent_link, run_coordinator as coordinator
from jarvis.db import session_scope
from jarvis.domain import DomainError
from jarvis.models import now
from jarvis.session_models import AgentCommand, AgentHost, AgentRun
from sqlalchemy import select
from test_agent_runs import HUMAN, OWNER, attached, start, task
from test_agent_runs import host as host  # noqa: F401 - fixture
from test_agent_hyperplan import WORKER, hyper, submit
from test_agent_hyperplan import roles_host as roles_host  # noqa: F401 - fixture
from test_agent_plan import turn


@pytest.fixture
def cleanup_host(host):
    with session_scope() as db:
        row = db.get(AgentHost, host.id)
        # Cleanup is independently supported, including native hosts without managed.sessions.
        row.caps = {**row.caps, "caps": [*row.caps["caps"], "run.cleanup"]}
    return host


def running(host):
    identity = start(host, task()["id"])["id"]
    attached(host, identity)
    return identity


def cleanups(db, identity):
    return db.scalars(select(AgentCommand).where(AgentCommand.run_id == identity,
                                                AgentCommand.kind == "run.cleanup")
                      .order_by(AgentCommand.created_at, AgentCommand.id)).all()


@pytest.mark.parametrize("final", ["completed", "stopped", "capped", "failed"])
def test_all_final_states_queue_exact_cleanup_without_expiry(cleanup_host, final):
    identity = running(cleanup_host)
    with session_scope() as db:
        run = db.get(AgentRun, identity)
        primary = {"surf": run.surf, "incarnation": 1, "generation": 1}
        agent_cleanup.remember(run, "retired-planner", 2, 1, "exited")
        agent_cleanup.remember(run, "reviewer", 1, 1, "exited")
        if final in ("completed", "stopped"):
            coordinator.request_stop(db, run, actor=HUMAN, final=final)
            coordinator.finish(db, run)
        else:
            coordinator.transition(db, run, final, reason="test")
        rows = cleanups(db, identity)
        assert len(rows) == 1
        command = rows[0]
        assert command.status == "queued"  # finish must not cancel the cleanup it just created.
        assert command.expires_at is None and command.payload["expires_at"] == 0
        assert command.payload["terminal_state"] == final
        assert primary in command.payload["surfaces"]
        assert {s["surf"] for s in command.payload["surfaces"]} == {run.surf, "retired-planner", "reviewer"}
        assert coordinator.stale(db, command, run) is None
        assert not agent_cleanup.queue(db, run)


def test_plan_approval_pause_and_disconnection_never_queue_cleanup(cleanup_host):
    identity = running(cleanup_host)  # The fixture just approved the initial plan.
    with session_scope() as db:
        run = db.get(AgentRun, identity)
        assert run.state == "executing"
        assert not agent_cleanup.queue(db, run)
        run.paused = True
        db.get(AgentHost, cleanup_host.id).lease_until = now() - timedelta(hours=1)
        assert not agent_cleanup.queue(db, run)
        with pytest.raises(DomainError, match="Only an ended run"):
            coordinator.command(db, run, "run.cleanup", {"terminal_state": "completed", "surfaces": []})
        assert not cleanups(db, identity)


def test_old_hosts_keep_existing_behavior(host):
    identity = running(host)
    with session_scope() as db:
        run = db.get(AgentRun, identity)
        coordinator.transition(db, run, "failed", reason="test")
        assert not cleanups(db, identity)


def test_reconnect_retires_known_ended_surfaces_and_does_not_infer_from_absence(cleanup_host):
    identity = running(cleanup_host)
    with session_scope() as db:
        run = db.get(AgentRun, identity)
        run.state = "stopped"  # An old ended run first encountered after upgrading the host.
        run.generation = 2
        run.sync_state = {}
        agent_cleanup.snapshot(db, OWNER, cleanup_host.id, [])
        assert not cleanups(db, identity)
        surface = agent_link.SnapshotSurface(surf=run.surf, incarnation=3, generation=2,
                                             run_id=identity, state="exited")
        agent_cleanup.snapshot(db, OWNER, "different-host", [surface])
        agent_cleanup.snapshot(db, "different-owner", cleanup_host.id, [surface])
        assert not cleanups(db, identity)
        coordinator.reconcile(db, OWNER, cleanup_host.id, [surface])
        first = cleanups(db, identity)[0]
        assert first.payload["surfaces"] == [{"surf": run.surf, "incarnation": 3, "generation": 2}]
        digest, op_id = first.digest, first.op_id
        frames = agent_link.drain(db, db.get(AgentHost, cleanup_host.id), 1)
        assert any(frame["op_id"] == op_id for frame in frames)
        first.updated_at = now() - timedelta(days=2)
        frames = agent_link.drain(db, db.get(AgentHost, cleanup_host.id), 2)
        assert any(frame["op_id"] == op_id and frame["digest"] == digest for frame in frames)
        agent_cleanup.snapshot(db, OWNER, cleanup_host.id, [surface])
        assert len(cleanups(db, identity)) == 1
        run.generation = 3
        assert coordinator.stale(db, first, run) is not None


def test_terminal_takeover_requires_new_positive_exit_fact_for_another_cleanup(cleanup_host):
    identity = running(cleanup_host)
    with session_scope() as db:
        run = db.get(AgentRun, identity)
        run.state = "stopped"
        run.sync_state = {**run.sync_state, "controller": "terminal"}
        surface = agent_link.SnapshotSurface(surf=run.surf, incarnation=1, generation=1,
            run_id=identity, state="idle", execution="managed", controller="terminal")
        agent_cleanup.snapshot(db, OWNER, cleanup_host.id, [surface])
        first = cleanups(db, identity)[0]
        coordinator.on_outcome(db, first, "executed", result={"surfaces": [{"surf": run.surf, "status": "operator_control"}]})
        agent_cleanup.snapshot(db, OWNER, cleanup_host.id, [])
        agent_cleanup.snapshot(db, OWNER, cleanup_host.id, [surface])
        assert len(cleanups(db, identity)) == 1
        agent_cleanup.snapshot(db, OWNER, cleanup_host.id, [surface.model_copy(update={"state": "exited"})])
        assert len(cleanups(db, identity)) == 2
        assert run.state == "stopped" and run.sync_state["controller"] == "terminal"


def test_final_approval_of_already_exited_run_queues_cleanup(cleanup_host):
    identity = running(cleanup_host)
    with session_scope() as db:
        run = db.get(AgentRun, identity)
        session = db.get(agent_control.AgentSession, run.session_pk)
        session.state = "exited"
        run.state = "awaiting_approval"
        agent_control.approve(db, OWNER, identity,
            agent_control.ApproveInput(expected_revision=run.state_revision), actor=HUMAN)
        assert run.state == "completed"
        assert len(cleanups(db, identity)) == 1


def test_steered_surface_exit_then_approval_uses_confirmed_new_generation(cleanup_host):
    identity = running(cleanup_host)
    with session_scope() as db:
        run = db.get(AgentRun, identity)
        run.generation += 1
        command = coordinator.command(db, run, "input", {"surf": run.surf, "incarnation": run.incarnation,
                                                        "mode": "steer", "text": "Revise the approach"})
        coordinator.on_outcome(db, command, "executed", result={
            "surf": run.surf, "incarnation": run.incarnation, "generation": run.generation})
        actual_generation = run.generation
        session = db.get(agent_control.AgentSession, run.session_pk)
        session.state = "exited"
        coordinator.on_status(db, session, agent_link.Status(
            surf=run.surf, incarnation=run.incarnation, state="exited", since=now()), host_id=cleanup_host.id)
        run.state = "awaiting_approval"
        agent_control.approve(db, OWNER, identity,
            agent_control.ApproveInput(expected_revision=run.state_revision), actor=HUMAN)
        assert run.state == "completed"
        assert cleanups(db, identity)[0].payload["surfaces"] == [{
            "surf": run.surf, "incarnation": run.incarnation, "generation": actual_generation}]


def test_live_takeover_exit_retries_cleanup_without_reconnect(cleanup_host):
    identity = running(cleanup_host)
    with session_scope() as db:
        run = db.get(AgentRun, identity)
        run.state = "stopped"
        run.sync_state = {**run.sync_state, "controller": "terminal"}
        first = agent_cleanup.queue(db, run)[0]
        coordinator.on_outcome(db, first, "executed", result={
            "surfaces": [{"surf": run.surf, "status": "operator_control"}]})
        session = db.get(agent_control.AgentSession, run.session_pk)
        frame = agent_link.Status(surf=run.surf, incarnation=run.incarnation, state="exited", since=now(),
                                  controller="terminal")
        coordinator.on_status(db, session, frame, host_id=cleanup_host.id)
        assert len(cleanups(db, identity)) == 2
        coordinator.on_status(db, session, frame, host_id=cleanup_host.id)
        assert len(cleanups(db, identity)) == 2
        assert run.state == "stopped" and run.sync_state["controller"] == "terminal"


@pytest.fixture
def handoff_host(roles_host):
    with session_scope() as db:
        host = db.get(AgentHost, roles_host.id)
        host.caps = {**host.caps, "caps": [*host.caps["caps"], "run.cleanup", "run.cleanup.handoff"]}
    return roles_host


def split_handoff(host, *, generation=1):
    created, identity = hyper(host, reviewer=None, worker=WORKER)
    submit(identity, created["id"])
    turn(identity, 1)
    with session_scope() as db:
        run = db.get(AgentRun, identity)
        run.generation = generation
        planner = run.surf
        coordinator.approve_plan(db, run, actor=HUMAN)
    return identity, planner


def retired_cleanups(db, identity):
    return [row for row in cleanups(db, identity) if row.payload.get("scope") == "retired_planner"]


def test_distinct_worker_attach_retires_only_the_saved_planner_and_replays_durably(handoff_host):
    identity, planner = split_handoff(handoff_host, generation=2)
    with session_scope() as db:
        run = db.get(AgentRun, identity)
        saved = {"surf": planner, "incarnation": 1, "generation": 1}
        assert run.sync_state["handoff"]["from_binding"] == saved  # Steer did not rewrite the old binding.
        assert not cleanups(db, identity)
        worker = coordinator.handoff_surf(run)
        agent_cleanup.remember(run, "unrelated-reviewer", 1, 1)
        coordinator.attach(db, handoff_host.id, identity, 2, "claude", "worker-1", surf=worker, incarnation=1)
        rows = retired_cleanups(db, identity)
        assert len(rows) == 1
        operation = rows[0]
        assert operation.payload["surfaces"] == [saved]
        assert operation.payload["handoff_to"] == {"surf": worker, "incarnation": 1, "generation": 2}
        assert "terminal_state" not in operation.payload
        assert operation.expires_at is None and operation.payload["expires_at"] == 0
        assert coordinator.stale(db, operation, run) is None
        digest, op_id = operation.digest, operation.op_id
        coordinator.attach(db, handoff_host.id, identity, 2, "claude", "worker-1", surf=worker, incarnation=1)
        assert not agent_cleanup.queue(db, run)
        assert len(retired_cleanups(db, identity)) == 1
        frames = agent_link.drain(db, db.get(AgentHost, handoff_host.id), 1)
        assert any(frame["op_id"] == op_id for frame in frames)
        operation.updated_at = now() - timedelta(days=2)
        frames = agent_link.drain(db, db.get(AgentHost, handoff_host.id), 2)
        assert any(frame["op_id"] == op_id and frame["digest"] == digest for frame in frames)
        assert run.state == "executing" and run.surf == worker


def test_exited_worker_snapshot_is_not_handoff_cleanup_proof(handoff_host):
    identity, planner = split_handoff(handoff_host)
    with session_scope() as db:
        run = db.get(AgentRun, identity)
        worker = coordinator.handoff_surf(run)
        exited = agent_link.SnapshotSurface(surf=worker, incarnation=1, generation=1, run_id=identity,
            tool="claude", session_id="worker-1", state="exited")
        coordinator.reconcile(db, OWNER, handoff_host.id, [exited])
        assert not retired_cleanups(db, identity)
        assert not run.sync_state["handoff"].get("attached_binding")
        live = exited.model_copy(update={"state": "working"})
        coordinator.reconcile(db, OWNER, handoff_host.id, [live])
        assert retired_cleanups(db, identity)[0].payload["surfaces"][0]["surf"] == planner
        assert run.sync_state["handoff"]["attached_binding"]["surf"] == worker
        coordinator.reconcile(db, OWNER, handoff_host.id, [live])
        assert len(retired_cleanups(db, identity)) == 1


@pytest.mark.parametrize("outcome,detail,result", [
    ("rejected", "target_busy", {"limit": 1, "running": 1}),
    ("failed", "harness missing", {}),
])
def test_worker_startup_failure_does_not_claim_successful_handoff(handoff_host, outcome, detail, result):
    identity, _ = split_handoff(handoff_host)
    with session_scope() as db:
        run = db.get(AgentRun, identity)
        spawn = db.scalars(select(AgentCommand).where(AgentCommand.run_id == identity,
            AgentCommand.kind == "spawn").order_by(AgentCommand.created_at.desc())).first()
        coordinator.on_outcome(db, spawn, outcome, detail, result)
        assert not retired_cleanups(db, identity)
        assert not run.sync_state["handoff"].get("attached_binding")
        assert not run.sync_state["handoff"].get("done")


def test_same_session_execution_and_old_handoff_hosts_never_retire_planner(handoff_host):
    identity = running(handoff_host)
    with session_scope() as db:
        run = db.get(AgentRun, identity)
        assert run.state == "executing" and not run.sync_state.get("handoff")
        assert not agent_cleanup.queue(db, run)
    separate, _ = split_handoff(handoff_host)
    with session_scope() as db:
        host = db.get(AgentHost, handoff_host.id)
        host.caps = {**host.caps, "caps": [cap for cap in host.caps["caps"] if cap != "run.cleanup.handoff"]}
        run = db.get(AgentRun, separate)
        coordinator.attach(db, host.id, separate, 1, "claude", "worker-old-host",
                           surf=coordinator.handoff_surf(run), incarnation=1)
        assert not retired_cleanups(db, separate)


def test_retired_cleanup_rejects_changed_proof_and_retries_only_positive_planner_exit(handoff_host):
    identity, planner = split_handoff(handoff_host)
    with session_scope() as db:
        run = db.get(AgentRun, identity)
        worker = coordinator.handoff_surf(run)
        coordinator.attach(db, handoff_host.id, identity, 1, "claude", "worker-1", surf=worker, incarnation=1)
        first = retired_cleanups(db, identity)[0]
        for change in ({"surfaces": [first.payload["handoff_to"]]},
                       {"handoff_to": first.payload["surfaces"][0]}, {"terminal_state": "completed"}):
            with pytest.raises(DomainError, match="confirmed separate worker"):
                coordinator.command(db, run, "run.cleanup", {**first.payload, **change})
        coordinator.on_outcome(db, first, "executed", result={
            "surfaces": [{"surf": planner, "status": "operator_control"}]})
        assert not agent_cleanup.queue(db, run)
        agent_cleanup.snapshot(db, OWNER, handoff_host.id, [])
        assert len(retired_cleanups(db, identity)) == 1
        agent_cleanup.snapshot(db, OWNER, handoff_host.id, [agent_link.SnapshotSurface(
            surf=planner, incarnation=1, generation=1, run_id=identity, state="exited", controller="terminal")])
        assert len(retired_cleanups(db, identity)) == 2
        agent_cleanup.remember(run, worker, 2, 1, "working")
        assert coordinator.stale(db, first, run) is not None
        coordinator.attach(db, handoff_host.id, identity, 1, "claude", "worker-1", surf=worker, incarnation=2)
        replacement = retired_cleanups(db, identity)[-1]
        assert replacement.payload["handoff_to"]["incarnation"] == 2
        assert replacement.payload["surfaces"][0]["surf"] == planner
        run.generation = 2
        assert coordinator.stale(db, replacement, run) is not None
        delivered = coordinator.command(db, run, "input", {
            "surf": worker, "incarnation": 2, "mode": "steer", "text": "Continue the approved work"})
        coordinator.on_outcome(db, delivered, "executed", result={
            "surf": worker, "incarnation": 2, "generation": 2})
        refreshed = retired_cleanups(db, identity)[-1]
        assert refreshed.payload["handoff_to"] == {"surf": worker, "incarnation": 2, "generation": 2}
        assert coordinator.stale(db, refreshed, run) is None


def test_late_planner_spawn_receipt_cannot_rebind_or_acknowledge_worker(handoff_host):
    identity, planner = split_handoff(handoff_host)
    with session_scope() as db:
        run = db.get(AgentRun, identity)
        spawns = list(db.scalars(select(AgentCommand).where(AgentCommand.run_id == identity,
            AgentCommand.kind == "spawn").order_by(AgentCommand.created_at)))
        planner_spawn, worker_spawn = spawns
        worker = coordinator.handoff_surf(run)
        # SessionStart already attached the planner and approval created its worker,
        # but the provider's original spawn receipt arrives only now.
        coordinator.on_outcome(db, planner_spawn, "executed", result={
            "surf": planner, "incarnation": 1, "generation": 1})
        assert run.state == "handing_off" and run.surf == "" and run.session_pk is None
        planner_snapshot = agent_link.SnapshotSurface(surf=planner, incarnation=1, generation=1,
            run_id=identity, tool="claude", session_id="sess-1", state="exited")
        # Also repair a row produced by the old race rather than recognizing the
        # retired planner as evidence that the worker's spawn was accepted.
        run.surf, run.incarnation = planner, 1
        coordinator.reconcile(db, OWNER, handoff_host.id, [planner_snapshot])
        assert worker_spawn.status == "queued"
        assert (worker_spawn.outcome or {}).get("via") != "reconcile"
        assert run.surf == "" and run.session_pk is None
        assert not retired_cleanups(db, identity)
        coordinator.attach(db, handoff_host.id, identity, 1, "claude", "worker-race", surf=worker, incarnation=1)
        session_pk = run.session_pk
        coordinator.on_outcome(db, planner_spawn, "failed", "late retired-planner failure")
        assert run.state == "executing" and run.surf == worker and run.session_pk == session_pk
        assert retired_cleanups(db, identity)[0].payload["surfaces"][0]["surf"] == planner


def test_positive_worker_status_attaches_before_its_first_completed_turn(handoff_host):
    from types import SimpleNamespace

    identity, planner = split_handoff(handoff_host)
    with session_scope() as db:
        run = db.get(AgentRun, identity)
        worker = coordinator.handoff_surf(run)
        bot = SimpleNamespace(owner_id=OWNER)
        base = {"tool": "claude", "surf": worker, "incarnation": 1, "generation": 1,
                "run_id": identity, "since": now()}
        for change in ({"state": "exited", "session_id": "worker-exited"},
                       {"state": "working", "session_id": "worker-stale", "generation": 0},
                       {"state": "working", "session_id": "planner-late", "surf": planner}):
            agent_link.status(db, bot, handoff_host.id, agent_link.Status(**{**base, **change}))
            assert run.session_pk is None and not retired_cleanups(db, identity)
        agent_link.status(db, bot, handoff_host.id, agent_link.Status(**{
            **base, "state": "working", "session_id": "worker-live"}))
        assert run.state == "executing" and run.session_pk and run.surf == worker
        assert run.sync_state["handoff"]["attached_binding"]["surf"] == worker
        assert retired_cleanups(db, identity)[0].payload["surfaces"][0]["surf"] == planner


def test_obsolete_planner_accepted_ack_cannot_bind_pending_handoff(handoff_host):
    from types import SimpleNamespace

    identity, planner = split_handoff(handoff_host)
    with session_scope() as db:
        run = db.get(AgentRun, identity)
        planner_spawn = db.scalars(select(AgentCommand).where(AgentCommand.run_id == identity,
            AgentCommand.kind == "spawn").order_by(AgentCommand.created_at)).first()
        planner_spawn.status = "sent"  # An old journal acceptance is replayed before the final receipt.
        agent_link.acked(db, handoff_host.id, SimpleNamespace(op_id=planner_spawn.op_id,
            acked_id=planner_spawn.op_id, ok=True, outcome={"state": "accepted", "surf": planner, "incarnation": 1}))
        assert run.surf == "" and run.incarnation == 0 and run.session_pk is None
