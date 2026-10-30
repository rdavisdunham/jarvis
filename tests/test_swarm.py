"""Swarm mode (swarm.py): planning with the model mocked and the fallback, edits, Start gating, batch sequencing,
the per-host cap and the integrate step."""

import json
from types import SimpleNamespace

import pytest
from jarvis import swarm
from jarvis import run_coordinator as coordinator
from jarvis.agent_actor import Actor
from jarvis.db import session_scope
from jarvis.domain import DomainError
from jarvis.models import OwnerSettings, Task
from jarvis.session_models import AgentCommand, AgentRun, AgentSwarm, ExecutionTarget
from sqlalchemy import select
from test_agent_runs import quiet  # noqa: F401 - fixture
from test_agent_runs import host as host  # noqa: F401 - pytest fixture
from test_linear import run as command
from test_organization_foundation import create

OWNER = "davin"
HUMAN = Actor("human")


# ---------------------------------------------------------------- the fallback and the model's answer (pure)


def facts(**tasks):
    """{id: (key, blocked_by)} → planner facts."""
    return {t: {"title": t.upper(), "key": key, "repo": key, "blocked_by": list(deps), "areas": []}
            for t, (key, deps) in tasks.items()}


def test_fallback_puts_one_repository_in_series_and_other_repositories_beside_it():
    groups, reasons, why = swarm.fallback(facts(a=("jarvis", ()), b=("jarvis", ()), c=("remotea", ()), d=("web", ())))
    assert groups == [["a", "c", "d"], ["b"]]
    assert reasons[0].startswith("Different repositories")
    assert "same repository (jarvis)" in why["b"]


def test_fallback_orders_by_blocks_and_never_runs_a_task_beside_its_blocker():
    groups, _, why = swarm.fallback(facts(a=("x", ("c",)), b=("y", ()), c=("z", ()), d=("w", ("a",))))
    assert groups == [["b", "c"], ["a"], ["d"]]
    assert why["a"].startswith("after “C”")


def test_fallback_treats_tasks_without_a_target_as_one_repository_and_survives_a_cycle():
    groups, _, _ = swarm.fallback(facts(a=(swarm.UNKNOWN_KEY, ()), b=(swarm.UNKNOWN_KEY, ())))
    assert groups == [["a"], ["b"]]
    groups, _, _ = swarm.fallback(facts(a=("x", ("b",)), b=("y", ("a",))))
    assert sorted(t for g in groups for t in g) == ["a", "b"] and len(groups) == 2


def answer(batches, areas=None):
    return swarm.ModelPlan(batches=[swarm.ModelBatch(refs=r, reason=why) for r, why in batches],
                           tasks=[swarm.ModelTask(ref=k, areas=v) for k, v in (areas or {}).items()])


def test_a_model_plan_must_place_every_task_once_and_after_its_blockers():
    f = facts(a=("jarvis", ()), b=("jarvis", ("a",)), c=("remotea", ()))
    groups, reasons, areas = swarm.from_model(answer([(["T1", "T3"], "disjoint"), (["T2"], "needs T1")],
                                                     {"T1": ["apps/web"], "T2": ["apps/api/x.py"]}), f)
    assert groups == [["a", "c"], ["b"]] and reasons == ["disjoint", "needs T1"]
    assert areas == {"a": ["apps/web"], "b": ["apps/api/x.py"]}
    for bad in ([(["T1", "T3"], "x")],                          # T2 lost
                [(["T1", "T2", "T3"], "x")],                    # T2 beside its blocker
                [(["T1", "T1"], "x"), (["T2", "T3"], "y")],     # repeated
                [(["T1", "T9"], "x"), (["T2", "T3"], "y")]):    # invented
        with pytest.raises(swarm.NoPlan):
            swarm.from_model(answer(bad), f)


def test_overlap_matches_equal_paths_and_folders():
    assert swarm.overlap(["apps/api/"], ["apps/api/jarvis/swarm.py", "docs"]) == ["apps/api/jarvis/swarm.py"]
    assert swarm.overlap(["apps/web"], ["apps/webby"]) == []


# ---------------------------------------------------------------- with a database


def enable_remotea(on=True):
    with session_scope() as db:
        row = db.get(OwnerSettings, OWNER)
        row.values = {**row.values, "remotea_enabled": on}


def project(host, *titles, record_target=True):
    """A project record with child tasks; the host's jarvis target is saved for the project (every child resolves
    to it) unless `record_target` is false."""
    p = create("project", "Swarm project")
    children = [create("task", t, parent_id=p["id"]) for t in titles]
    if record_target:
        with session_scope() as db:
            db.get(ExecutionTarget, host.target_id).record_id = p["id"]
    return SimpleNamespace(record=p, tasks=[c["task_id"] for c in children], records=[c["id"] for c in children])


def second_target(host, task_id, name="remotea"):
    with session_scope() as db:
        target = ExecutionTarget(owner_id=OWNER, host_id=host.id, host_target=name, name=name, repo=f"/home/davin/{name}",
                                 worktree_root="/home/davin/wt", launch={}, verify=[])
        db.add(target)
        db.flush()
        target_id = target.id
    with session_scope() as db:
        revision = db.get(Task, task_id).revision
    command("task.update", task_id=task_id, expected_revision=revision, execution_target_id=target_id)
    return target_id


def plan(project, monkeypatch, model=None, **body):
    """Create through swarm.create; `model` is a ModelPlan the mocked planner answers (None: no model)."""
    monkeypatch.setattr(swarm, "model_ready", lambda: model is not None)
    if model is not None:
        monkeypatch.setattr(swarm, "ask", lambda owner, f: (model, {"cost": 0.001, "latency_ms": 5}))
    return swarm.create(session_scope, OWNER, None, swarm.CreateInput(record_id=project.record["id"], **body))


def edit(s, *ops):
    with session_scope() as db:
        return swarm.edit(db, OWNER, s["id"], swarm.EditInput(expected_revision=s["revision"], ops=list(ops)))


def start(s, **body):
    with session_scope() as db:
        return swarm.start(db, OWNER, s["id"], swarm.StartInput(expected_revision=s["revision"], **body), actor=HUMAN)


def detail(s):
    with session_scope() as db:
        return swarm.public(db, swarm.load(db, OWNER, s["id"], lock=False))


def advance():
    with session_scope() as db:
        swarm.sweep(db)


def move(run_id, *states, reason="", detail_text=""):
    with session_scope() as db:
        row = db.get(AgentRun, run_id, with_for_update=True)
        for state in states:
            coordinator.transition(db, row, state, reason=reason, detail=detail_text)


def run_ids(s, batch=0):
    return [i["run"]["id"] for i in s["batches"][batch]["items"] if i["run"]]


def test_fallback_plan_when_no_model_is_configured(host, monkeypatch):
    p = project(host, "Alpha", "Beta")
    s = plan(p, monkeypatch)
    assert s["state"] == "draft" and s["planner"]["source"] == "fallback" and "No planning model" in s["planner"]["note"]
    # Both on the same repository: one after the other.
    assert [[i["task_id"] for i in b["items"]] for b in s["batches"]] == [[p.tasks[0]], [p.tasks[1]]]
    assert all(i["status"] == "planned" and i["target"]["name"] == "jarvis" for b in s["batches"] for i in b["items"])
    assert s["batches"][0]["cost"]["basis"] == "no finished runs yet"
    # Nothing spawned.
    with session_scope() as db:
        assert db.scalar(select(AgentRun)) is None
    # The project's open plan comes back for a second Play; `fresh` replaces a plan nothing of which started.
    assert plan(p, monkeypatch)["id"] == s["id"]
    again = plan(p, monkeypatch, fresh=True)
    assert again["id"] != s["id"] and detail(s)["state"] == "canceled"


def test_the_model_plan_is_used_and_a_bad_one_falls_back(host, monkeypatch):
    p = project(host, "Alpha", "Beta")
    s = plan(p, monkeypatch, answer([(["T1", "T2"], "Separate folders")], {"T1": ["apps/web"], "T2": ["apps/api"]}))
    assert s["planner"]["source"] == "model" and s["planner"]["profile"] == "luna"
    assert len(s["batches"]) == 1 and s["batches"][0]["reason"] == "Separate folders"
    assert s["batches"][0]["items"][0]["areas"] == ["apps/web"]
    assert not [w for w in s["warnings"] if w["code"] == "SAME_REPO"]  # Disjoint areas: no overlap warning.
    s2 = plan(p, monkeypatch, answer([(["T1"], "only one")]), fresh=True)
    assert s2["planner"]["source"] == "fallback" and "not used (invalid)" in s2["planner"]["note"]
    assert len(s2["batches"]) == 2


def test_a_project_without_open_tasks_and_tasks_with_runs(host, monkeypatch):
    empty = project(host)
    with pytest.raises(DomainError) as error:
        plan(empty, monkeypatch)
    assert error.value.code == "NO_TASKS"
    p = project(host, "Alpha", "Beta")
    from test_agent_runs import start as start_run

    start_run(host, p.tasks[0])
    s = plan(p, monkeypatch)
    assert [e["task_id"] for e in s["excluded"]] == [p.tasks[0]]
    assert [i["task_id"] for b in s["batches"] for i in b["items"]] == [p.tasks[1]]


def test_blocks_links_order_the_plan(host, monkeypatch):
    p = project(host, "Alpha", "Beta")
    second_target(host, p.tasks[0])  # Different repositories, so only the link keeps them apart.
    from jarvis import structure
    from jarvis.structure_models import StructureRecord

    with session_scope() as db:
        revision = structure.data(db, db.get(StructureRecord, p.records[1]))["revision"]
        schema_revision = structure.schema_data(db, OWNER)["revision"]
    command("record.link", source_id=p.records[1], target_id=p.records[0], relationship_id="blocks",
            expected_revision=revision, schema_revision=schema_revision)
    s = plan(p, monkeypatch)
    assert [[i["task_id"] for i in b["items"]] for b in s["batches"]] == [[p.tasks[1]], [p.tasks[0]]]
    assert s["batches"][1]["items"][0]["blocked_by"] == [p.tasks[1]]
    # Moving it beside its blocker is allowed (Blocks is advisory) but warned about.
    s = edit(s, swarm.Op(op="move", task_id=p.tasks[0], batch_id=s["batches"][0]["id"]))
    assert len(s["batches"]) == 1 and any(w["code"] == "ORDER" for w in s["warnings"])


def test_edits_before_start(host, monkeypatch):
    p = project(host, "Alpha", "Beta", "Gamma")
    s = plan(p, monkeypatch, answer([(["T1", "T2", "T3"], "all apart")], {"T1": ["a"], "T2": ["b"], "T3": ["c"]}))
    one = s["batches"][0]["id"]
    s = edit(s, swarm.Op(op="split", batch_id=one, task_ids=[p.tasks[2]]))
    assert [[i["task_id"] for i in b["items"]] for b in s["batches"]] == [p.tasks[:2], [p.tasks[2]]]
    s = edit(s, swarm.Op(op="order", batch_id=s["batches"][1]["id"], index=0))
    assert s["batches"][0]["items"][0]["task_id"] == p.tasks[2]
    s = edit(s, swarm.Op(op="merge", batch_id=s["batches"][0]["id"], into=s["batches"][1]["id"]))
    assert len(s["batches"]) == 1 and len(s["batches"][0]["items"]) == 3
    s = edit(s, swarm.Op(op="exclude", task_id=p.tasks[1]), swarm.Op(op="cap", value=2),
             swarm.Op(op="auto_continue", value=True), swarm.Op(op="harness", value="codex"))
    assert [e["task_id"] for e in s["excluded"]] == [p.tasks[1]]
    assert (s["cap"], s["auto_continue"], s["harness"]) == (2, True, "codex")
    s = edit(s, swarm.Op(op="include", task_id=p.tasks[1]))
    assert s["batches"][-1]["items"][0]["task_id"] == p.tasks[1] and not s["excluded"]
    s = edit(s, swarm.Op(op="move", task_id=p.tasks[1], batch_id=s["batches"][0]["id"]))
    assert len(s["batches"]) == 1  # The emptied batch is dropped.
    # A stale revision, a bad cap, and one bad op among good ones change nothing.
    with pytest.raises(DomainError) as error:
        edit({**s, "revision": s["revision"] - 1}, swarm.Op(op="cap", value=3))
    assert error.value.code == "REVISION_CONFLICT"
    for op in (swarm.Op(op="cap", value=99), swarm.Op(op="include", task_id=p.tasks[0])):
        with pytest.raises(DomainError):
            edit(s, swarm.Op(op="cap", value=4), op)
    assert detail(s)["cap"] == 2 and detail(s)["revision"] == s["revision"]


def test_start_goes_through_the_normal_start_path_and_its_gates(host, monkeypatch):
    p = project(host, "Alpha", "Beta")
    s = plan(p, monkeypatch)
    enable_remotea(False)
    with pytest.raises(DomainError) as error:
        start(s)
    assert error.value.code == "REMOTEA_DISABLED"
    enable_remotea(True)
    with pytest.raises(DomainError) as error:
        start(s, batch_id=s["batches"][1]["id"])
    assert error.value.code == "BATCH_ORDER"
    s = start(s)
    assert s["state"] == "running" and s["batches"][0]["state"] == "running"
    item = s["batches"][0]["items"][0]
    assert item["status"] == "spawning" and item["run"]["id"]
    with session_scope() as db:
        spawn = db.scalar(select(AgentCommand).where(AgentCommand.run_id == item["run"]["id"],
                                                     AgentCommand.kind == "spawn"))
        prompt = spawn.payload["prompt"]
    assert "## Swarm (batch 1 of 2" in prompt and "never choose the next migration number" in prompt
    # The second batch waits: Start refuses while batch 1 runs, editing batch 1 is refused.
    with pytest.raises(DomainError) as error:
        start(s)
    assert error.value.code == "SWARM_BUSY"
    with pytest.raises(DomainError) as error:
        edit(s, swarm.Op(op="exclude", task_id=p.tasks[0]))
    assert error.value.code == "BATCH_STARTED"
    # A started plan is not replaced by planning again.
    with pytest.raises(DomainError) as error:
        plan(p, monkeypatch, fresh=True)
    assert error.value.code == "SWARM_RUNNING"


def test_a_task_that_cannot_start_is_recorded_and_settles_its_slot(host, monkeypatch):
    p = project(host, "Alpha", record_target=False)
    s = plan(p, monkeypatch)
    assert any(w["code"] == "NEEDS_TARGET" for w in s["warnings"])
    s = start(s)
    item = s["batches"][0]["items"][0]
    assert item["status"] == "not_started" and item["error"]["code"] == "NEEDS_TARGET"
    assert s["state"] == "done" and s["batches"][0]["state"] == "settled"


def test_batches_run_in_sequence_and_wait_for_start_unless_continuing_automatically(host, monkeypatch):
    p = project(host, "Alpha", "Beta", "Gamma")
    s = start(plan(p, monkeypatch))  # Same repository: three batches of one.
    first = run_ids(s)[0]
    move(first, "executing")
    advance()
    assert detail(s)["batches"][0]["state"] == "running"
    move(first, "awaiting_approval")  # Done working, waiting for review and merge: the batch settles.
    advance()
    s = detail(s)
    assert s["batches"][0]["state"] == "settled" and s["batches"][1]["state"] == "pending"
    assert s["next_batch"] == s["batches"][1]["id"]
    s = start(s, all=True)  # Start the rest: batch 2 now, batch 3 when batch 2 settles.
    assert s["auto_continue"] and s["batches"][1]["state"] == "running"
    move(run_ids(s, 1)[0], "failed", reason="spawn_failed")
    advance()
    s = detail(s)
    assert s["batches"][1]["state"] == "settled" and s["batches"][2]["state"] == "running"
    move(run_ids(s, 2)[0], "stopping")
    with session_scope() as db:
        coordinator.finish(db, db.get(AgentRun, run_ids(s, 2)[0], with_for_update=True))
    advance()
    assert detail(s)["state"] == "done"


def test_the_per_host_cap_holds_tasks_until_a_slot_frees(host, monkeypatch):
    p = project(host, "Alpha", "Beta", "Gamma")
    s = plan(p, monkeypatch, answer([(["T1", "T2", "T3"], "apart")], {"T1": ["a"], "T2": ["b"], "T3": ["c"]}), cap=2)
    s = start(s)
    statuses = [i["status"] for i in s["batches"][0]["items"]]
    assert statuses == ["spawning", "spawning", "queued"]
    advance()
    assert detail(s)["batches"][0]["items"][2]["status"] == "queued"
    move(run_ids(s)[0], "executing", "awaiting_approval")  # Waiting for review: no longer takes a slot.
    advance()
    assert detail(s)["batches"][0]["items"][2]["status"] == "spawning"


def test_a_spawn_remotea_refused_as_busy_is_queued_again(host, monkeypatch):
    p = project(host, "Alpha")
    s = start(plan(p, monkeypatch))
    first = run_ids(s)[0]
    move(first, "failed", reason="spawn_rejected", detail_text="target_busy: 3 of 3 runs")
    advance()
    item = detail(s)["batches"][0]["items"][0]
    assert item["retries"] == 1 and item["run"]["id"] != first and item["status"] == "spawning"


def test_integrate_flags_overlapping_work_and_holds_continue_automatically(host, monkeypatch):
    p = project(host, "Alpha", "Beta", "Gamma")
    model = answer([(["T1", "T2"], "risky"), (["T3"], "later")],
                   {"T1": ["apps/api/jarvis"], "T2": ["apps/api/jarvis/swarm.py"], "T3": ["docs"]})
    s = start(plan(p, monkeypatch, model), all=True)
    for run_id in run_ids(s):
        move(run_id, "executing", "awaiting_approval")
    advance()
    s = detail(s)
    check = s["batches"][0]["integrate"]
    assert s["batches"][0]["state"] == "settled" and s["batches"][1]["state"] == "pending"  # Held for Davin.
    assert check["blocking"] and check["repos"][0]["conflicts"][0]["basis"] == "planner areas"
    assert check["repos"][0]["conflicts"][0]["shared"] == ["apps/api/jarvis/swarm.py"]
    assert sorted(check["repos"][0]["merge_order"]) == sorted(p.tasks[:2])
    s = start(s)
    assert s["batches"][1]["state"] == "running"


def test_routes_are_gated_and_owner_scoped(client, host, monkeypatch):
    p = project(host, "Alpha")
    monkeypatch.setattr(swarm, "model_ready", lambda: False)
    created = client.post("/api/v1/remotea/swarms", json={"record_id": p.record["id"]})
    assert created.status_code == 200, created.text
    s = created.json()
    assert client.get("/api/v1/remotea/swarms?active=true").json()["items"][0]["id"] == s["id"]
    assert client.get(f"/api/v1/remotea/swarms/{s['id']}").json()["state"] == "draft"
    edited = client.post(f"/api/v1/remotea/swarms/{s['id']}/edit",
                         json={"expected_revision": s["revision"], "ops": [{"op": "cap", "value": 1}]}).json()
    assert edited["cap"] == 1
    assert client.post(f"/api/v1/remotea/swarms/{s['id']}/edit",
                       json={"expected_revision": edited["revision"], "ops": [{"op": "nope"}]}).status_code == 422
    enable_remotea(False)
    refused = client.post(f"/api/v1/remotea/swarms/{s['id']}/start", json={"expected_revision": edited["revision"]})
    assert refused.status_code == 403 and refused.json()["error"]["code"] == "REMOTEA_DISABLED"
    # Cancel works while remotea is off; it keeps later batches from starting.
    canceled = client.post(f"/api/v1/remotea/swarms/{s['id']}/cancel", json={"expected_revision": edited["revision"]})
    assert canceled.json()["state"] == "canceled"
    with session_scope() as db:
        db.get(AgentSwarm, s["id"]).owner_id = "someone-else"
    assert client.get(f"/api/v1/remotea/swarms/{s['id']}").status_code == 404


def test_swarm_plans_are_in_the_account_export(host, monkeypatch):
    from jarvis import agent_retention as retention

    s = plan(project(host, "Alpha"), monkeypatch)
    with session_scope() as db:
        rows = retention.export_tables(db, OWNER)["agent_swarms"]
    assert [r["id"] for r in rows] == [s["id"]] and json.dumps(rows)
