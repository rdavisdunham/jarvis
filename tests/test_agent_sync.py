"""Stage E: Linear outcomes of agent runs (agent_sync.py, linear_sync ordering guard) and completion policies.

These are the plan's test_linear.py additions; they live here because they need the run fixtures of
test_agent_runs.py, which itself imports test_linear.py.
"""

import copy

import pytest
from jarvis import agent_control as control
from jarvis import agent_judge as judge
from jarvis import agent_sync
from jarvis import linear_sync as sync
from jarvis import run_coordinator as coordinator
from jarvis.db import session_scope
from jarvis.linear_client import LinearFailure
from jarvis.models import Job, LinearIssue, Task
from jarvis.session_models import AgentAttempt, AgentCommand, AgentHost, AgentRun, AgentSession
from sqlalchemy import select
from test_agent_judge import criteria_run, judged, stub, verdict
from test_agent_plan import turn
from test_agent_runs import HUMAN, OWNER, attached, start, state
from test_agent_runs import host as host  # noqa: F401 - pytest fixture
from test_agent_runs import quiet as quiet  # noqa: F401 - pytest fixture
from test_agent_runs import started_state as started_state  # noqa: F401 - pytest fixture
from test_linear import Fake, imported
from test_linear import linear as linear  # noqa: F401 - pytest fixture


@pytest.fixture
def fake(started_state, linear, monkeypatch):  # noqa: F811
    """The Linear fake with an In Progress state and comments."""
    plain = Fake.query

    def query(self, text, variables):
        if "commentCreate" in text:
            data = variables["input"]
            created = {"id": f"comment-{len(self.comments) + 1}", "url": "https://linear.app/c", "body": data["body"]}
            self.comments.append({**created, "issue": data["issueId"], "user": {"id": "me"}})
            if self.lose_comment:
                self.lose_comment = False
                raise LinearFailure("unavailable", retry=True, unknown=True)  # Accepted, response lost.
            return {"commentCreate": {"success": True, "comment": created}}
        if "EriComments" in text:
            nodes = [c for c in self.comments if c["issue"] == variables["id"]]
            return {"issue": {"comments": {"nodes": copy.deepcopy(nodes), "pageInfo": {"hasNextPage": False}}}}
        result = plain(self, text, variables)
        for value in result.values():
            if value["issue"]["state"]["id"] == "doing":
                value["issue"]["state"] = {"id": "doing", "name": "In Progress", "type": "started"}
                self.rows[value["issue"]["id"]]["state"] = value["issue"]["state"]
        return result

    monkeypatch.setattr(Fake, "query", query)
    linear.comments, linear.lose_comment = [], False
    return linear


def client_factory(fake):  # noqa: F811
    from jarvis.linear_client import LinearClient

    class Client(LinearClient):
        def __init__(self, key):
            self.http = None

        def query(self, text, variables=None):
            return fake.query(text, variables or {})

        def close(self):
            pass
    return Client


def linear_jobs(kind="linear_write"):
    with session_scope() as db:
        return [(j.id, j.status, j.payload.get("purpose"), j.payload.get("patch")) for j in
                db.scalars(select(Job).where(Job.kind == kind).order_by(Job.created_at))]


def sync_state(run_id):
    with session_scope() as db:
        return db.get(AgentRun, run_id).sync_state.get("linear") or {}


def settle_stop(run_id):
    with session_scope() as db:
        stop = db.scalar(select(AgentCommand).where(AgentCommand.run_id == run_id, AgentCommand.kind == "stop"))
        coordinator.on_outcome(db, stop, "executed", "terminated")


def approved_run(host, fake):  # noqa: F811
    tid, _ = imported()
    run_id = start(host, tid)["id"]
    attached(host, run_id)
    with session_scope() as db:
        control.approve(db, OWNER, run_id, control.ApproveInput(expected_revision=db.get(AgentRun, run_id).state_revision), actor=HUMAN)
    settle_stop(run_id)
    assert state(run_id)[0] == "completed"
    return tid, run_id


# ---------------------------------------------------------------- the completion comment


def test_a_completed_run_comments_once_with_a_stable_marker(host, fake):  # noqa: F811
    tid, run_id = approved_run(host, fake)
    [(job_id, status, _, _)] = linear_jobs("linear_comment")
    comment = sync_state(run_id)["comment"]
    assert status == "queued" and comment["status"] == "queued" and comment["marker"] == "eridani-run:" + run_id
    agent_sync.process_comment(job_id, client_factory(fake))
    agent_sync.process_comment(job_id, client_factory(fake))  # A redelivered job does nothing.
    with session_scope() as db:
        run = db.get(AgentRun, run_id)
        assert agent_sync.queue_comment(db, run) is None  # Once per run.
    assert len(fake.comments) == 1 and fake.comments[0]["issue"] == "remote"
    assert fake.comments[0]["body"].endswith("eridani-run:" + run_id)
    saved = sync_state(run_id)["comment"]
    assert (saved["status"], saved["comment_id"], saved["reconciled"]) == ("posted", "comment-1", False)


def test_an_accepted_but_lost_comment_is_reconciled_not_duplicated(host, fake):  # noqa: F811
    _, run_id = approved_run(host, fake)
    [(job_id, *_)] = linear_jobs("linear_comment")
    fake.lose_comment = True
    with pytest.raises(RuntimeError):
        agent_sync.process_comment(job_id, client_factory(fake))  # The worker retries this step.
    lost = sync_state(run_id)["comment"]
    assert lost["status"] == "retrying" and lost["attempted"] is True and len(fake.comments) == 1
    agent_sync.process_comment(job_id, client_factory(fake))
    saved = sync_state(run_id)["comment"]
    assert (saved["status"], saved["comment_id"], saved["reconciled"]) == ("posted", "comment-1", True)
    assert len(fake.comments) == 1


def test_a_failed_comment_never_undoes_the_completion(host, fake):  # noqa: F811
    tid, run_id = approved_run(host, fake)
    [(job_id, *_)] = linear_jobs("linear_comment")

    def refuse(self, text, variables):
        raise LinearFailure("permission")
    Client = client_factory(fake)
    Client.query = refuse
    agent_sync.process_comment(job_id, Client)
    assert sync_state(run_id)["comment"]["status"] == "failed" and state(run_id)[0] == "completed"
    with session_scope() as db:
        assert db.get(Task, tid).status == "completed"


def test_a_task_without_a_linear_issue_gets_no_comment(host):  # noqa: F811
    from test_agent_runs import task

    run_id = start(host, task()["id"])["id"]
    attached(host, run_id)
    with session_scope() as db:
        control.approve(db, OWNER, run_id, control.ApproveInput(expected_revision=db.get(AgentRun, run_id).state_revision), actor=HUMAN)
    settle_stop(run_id)
    assert state(run_id)[0] == "completed" and "comment" not in sync_state(run_id)
    assert linear_jobs("linear_comment") == []


# ---------------------------------------------------------------- ordering: start and finish


def test_finishing_supersedes_an_in_progress_write_that_has_not_started(host, fake):  # noqa: F811
    tid, run_id = approved_run(host, fake)
    jobs = linear_jobs()
    start_job = next(j for j in jobs if j[2] == "in_progress")
    done_job = next(j for j in jobs if j[2] == "done")
    assert start_job[1] == "cancelled" and done_job[3] == {"stateId": "done"}
    assert sync_state(run_id)["in_progress"]["status"] == "queued"
    assert sync_state(run_id)["done"] == {**sync_state(run_id)["done"], "status": "queued", "job_id": done_job[0]}
    sync.process_write(start_job[0])  # A late delivery of the cancelled start does nothing.
    sync.process_write(done_job[0])
    assert [w[1] for w in fake.writes] == [{"stateId": "done"}]
    with session_scope() as db:
        assert db.get(Task, tid).status == "completed"


def test_a_start_write_in_flight_cannot_reopen_a_completed_task(host, fake):  # noqa: F811
    tid, _ = imported()
    run_id = start(host, tid)["id"]
    attached(host, run_id)
    [(start_id, *_)] = [j for j in linear_jobs() if j[2] == "in_progress"]
    with session_scope() as db:  # The worker has taken it and may have written to Linear.
        job = db.get(Job, start_id)
        job.status, job.result = "running", {"attempts": 1, "attempted": True}
    with session_scope() as db:
        control.approve(db, OWNER, run_id, control.ApproveInput(expected_revision=db.get(AgentRun, run_id).state_revision), actor=HUMAN)
    settle_stop(run_id)
    assert sync_state(run_id)["done"]["status"] == "deferred" and sync_state(run_id)["done"]["after"] == start_id
    assert [j for j in linear_jobs() if j[2] == "done"] == []
    sync.process_write(start_id)  # The stale start lands in Linear now.
    with session_scope() as db:
        assert db.get(Task, tid).status == "completed"  # ...and does not reopen the task here.
        link = db.scalar(select(LinearIssue))
        assert link.snapshot["state"]["type"] == "started" and link.sync_state == "pending"
    [(done_id, status, _, patch)] = [j for j in linear_jobs() if j[2] == "done"]
    assert status == "queued" and patch == {"stateId": "done"}
    assert sync_state(run_id)["done"]["status"] == "queued" and sync_state(run_id)["done"]["job_id"] == done_id
    sync.process_write(done_id)
    with session_scope() as db:
        assert db.get(Task, tid).status == "completed"
        assert db.scalar(select(LinearIssue)).snapshot["state"]["type"] == "completed"


def test_a_refused_linear_finish_never_fails_the_approval(host, fake):  # noqa: F811
    tid, _ = imported()
    run_id = start(host, tid)["id"]
    attached(host, run_id)
    with session_scope() as db:
        for job in db.scalars(select(Job).where(Job.kind == "linear_write")):
            job.status = "cancelled"
        link = db.scalar(select(LinearIssue))
        link.sync_state, link.pending_job_id = "conflict", None
    with session_scope() as db:
        control.approve(db, OWNER, run_id, control.ApproveInput(expected_revision=db.get(AgentRun, run_id).state_revision), actor=HUMAN)
    done = sync_state(run_id)["done"]
    assert done["status"] == "blocked" and done["code"] == "LINEAR_CONFLICT"
    with session_scope() as db:
        assert db.get(Task, tid).status == "completed"


# ---------------------------------------------------------------- completion policy


def completion(policy, prs=(), merged=False):
    return {"policy": policy, "branch": "user/eri-1-x", "pr_urls": list(prs), "merged": merged}


PR = "https://github.com/rdavisdunham/jarvis/pull/66"


@pytest.mark.parametrize("policy,prs,merged,expected,rule", [
    ("code_ready", (), False, "met", None),
    ("pr_open", (), False, "not_met", "pr_required"),
    ("pr_open", (PR,), False, "met", None),
    ("pr_merged", (), True, "not_met", "pr_required"),
    ("pr_merged", (PR,), False, "needs_you", "pr_not_merged"),
    ("pr_merged", (PR,), True, "met", None),
])
def test_completion_policy_rules(policy, prs, merged, expected, rule):
    evidence = {"plan": {"criteria": [{"id": "c1", "text": "done", "kind": "text"}]},
                "turn": {"text": "Done."}, "completion": completion(policy, prs, merged)}
    final = judge.decide(evidence, stub(verdict()))
    assert final["verdict"] == expected
    assert (rule in final["rules"]) if rule else not final["rules"]


def test_a_not_met_verdict_is_never_upgraded_by_the_policy():
    evidence = {"plan": {"criteria": [{"id": "c1", "text": "done"}]}, "turn": {"text": "x"},
                "completion": completion("pr_open", (PR,), True)}
    assert judge.decide(evidence, stub(verdict("not_met", "more", criteria=(("c1", "not_met"),))))["verdict"] == "not_met"


def test_pull_request_links_are_validated():
    from jarvis.agent_sessions import pull_requests

    assert pull_requests([PR, PR, "https://evil.example/x", "http://github.com/a/b/pull/1",
                          "https://gitlab.com/a/b/-/merge_requests/7", None]) == \
        [PR, "https://gitlab.com/a/b/-/merge_requests/7"]


def test_the_plan_freezes_the_policy_and_pr_merged_waits_for_the_host_merge_report(host):  # noqa: F811
    run_id = criteria_run(host)
    with session_scope() as db:
        run = db.get(AgentRun, run_id)
        plan = coordinator.frozen(db, run)
        assert plan.completion_policy == run.completion_policy == "code_ready"
        plan.completion_policy = "pr_merged"
        db.get(AgentSession, run.session_pk).pr_urls = [PR]
        branch = run.branch
    turn(run_id, 1)
    assert judged(run_id, stub(verdict())) == ("needs_you", "awaiting_merge")
    with session_scope() as db:
        run = db.get(AgentRun, run_id)
        attempt = db.scalars(select(AgentAttempt).where(AgentAttempt.run_id == run_id, AgentAttempt.kind == "judge")).first()
        evidence = judge.evidence_for(db, attempt)
        assert evidence["completion"] == {"policy": "pr_merged", "branch": branch, "pr_urls": [PR], "merged": False}
        host_row = db.get(AgentHost, run.host_id)
        host_row.caps = {**host_row.caps, "worktrees": {"worktrees": [{"branch": branch, "base": "main", "merged": True}]}}
        db.flush()
        coordinator.check_integrations(db, OWNER, host_row.id)
    assert state(run_id) == ("executing", "merged")
    turn(run_id, 2)
    assert judged(run_id, stub(verdict())) == ("awaiting_approval", "judge_met")


def test_completion_policy_is_a_gate_field_while_a_run_owns_the_task(host):  # noqa: F811
    from jarvis.agent_actor import Actor, acting
    from jarvis.domain import DomainError, execute

    run_id = criteria_run(host)
    with session_scope() as db:
        task = db.get(Task, db.get(AgentRun, run_id).task_id)
        with acting(Actor("bot", verified=False)), pytest.raises(DomainError) as refused:
            execute(db, OWNER, "policy-edit", "task.update",
                    {"task_id": task.id, "expected_revision": task.revision, "completion_policy": "pr_merged"})
    assert refused.value.code == "RUN_OWNS_COMPLETION"


def test_the_comment_carries_structured_facts_only_and_reconciles_only_our_own(host, fake):  # noqa: F811
    """Stage F, L5 + info: no judge summary, title or branch in Linear; a stranger's marker comment is not adopted."""
    from jarvis.session_models import AgentAttempt

    _, run_id = approved_run(host, fake)
    with session_scope() as db:
        run = db.get(AgentRun, run_id)
        db.add(AgentAttempt(owner_id=run.owner_id, run_id=run_id, kind="judge", status="done", generation=1,
                            turn_seq=1, plan_revision=1, result={"summary": "internal path /home/davin/secret",
                                                                  "criteria": [{"id": "c1", "status": "met"}]}))
        db.flush()
        body = agent_sync.comment_body(db, run)
    assert "c1 met" in body and "/home/davin" not in body and "Judge" not in body and "Branch" not in body
    assert agent_sync.ours({"body": "x eridani-run:r", "user": {"id": "me"}}, "me", "eridani-run:r")
    assert not agent_sync.ours({"body": "x eridani-run:r", "user": {"id": "stranger"}}, "me", "eridani-run:r")

