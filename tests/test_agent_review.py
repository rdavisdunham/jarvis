"""E-Q3 and ba56714a: the review packet beside Approve (criteria, changes, test evidence, questions, the judge's
reason), Edit criteria on a parked run and Judge again."""

import asyncio

from jarvis import agent_control as control
from jarvis import agent_sessions as sessions
from jarvis import run_coordinator as coordinator
from jarvis.db import session_scope
from jarvis.models import Task
from jarvis.session_models import AgentRun, AgentSession
from jarvis.tools import call_tool
from test_agent_attempts import attempts, planned, result
from test_agent_judge import judged, stub, verdict
from test_agent_plan import turn, verifying_host  # noqa: F401 - fixture
from test_agent_runs import OWNER, quiet, state  # noqa: F401 - fixture
from test_agent_runs import host as host  # noqa: F401 - pytest fixture


def pr(run_id, *urls):
    with session_scope() as db:
        row = db.get(AgentSession, db.get(AgentRun, run_id).session_pk)
        sessions.upsert(db, OWNER, row.tool, row.session_id, sessions.SessionInput(pr_urls=list(urls), turn_count=2))


def revision(run_id):
    with session_scope() as db:
        return db.get(AgentRun, run_id).state_revision


def test_the_review_packet_reads_criteria_tests_pr_and_questions(verifying_host, client):  # noqa: F811
    run_id = planned(verifying_host, title="Review me")
    assert turn(run_id, 3) == ("verifying", "turn")
    [attempt] = attempts(run_id)
    result(verifying_host, run_id, attempt.id, isolated=True, isolation="bwrap", workspace_complete=False,
           workspace_reasons=["untracked files capped at 2000 (2400 found)"])
    pr(run_id, "https://github.com/rdavisdunham/jarvis/pull/12")
    assert judged(run_id, stub(verdict())) == ("awaiting_approval", "judge_met")
    detail = client.get(f"/api/v1/remotea/runs/{run_id}").json()
    verify = next(a for a in detail["attempts"] if a["kind"] == "verify")
    assert verify["isolated"] is True and verify["isolation"] == "bwrap" and verify["workspace_complete"] is False
    assert verify["workspace_reasons"] == ["untracked files capped at 2000 (2400 found)"]
    assert detail["changes"]["pr_urls"] == ["https://github.com/rdavisdunham/jarvis/pull/12"]
    assert detail["changes"]["branch"] == detail["branch"]
    status = asyncio.run(call_tool(OWNER, "turn-r1", 0, "control_status", {}))
    [spoken] = status["runs"]
    packet = spoken["awaiting_approval"]["review_packet"]
    assert packet["text"] == ("1 of 1 criteria met; tests passed just now in a sandbox (workspace incomplete: untracked "
                              "files capped at 2000 (2400 found)); PR #12 open; no open questions")
    assert packet["criteria"] == [{"id": "c1", "text": "tests pass", "status": "met"}]
    assert packet["tests"]["exit_code"] == 0 and packet["reason"] == "s"
    next_text = spoken["awaiting_approval"]["next"]
    assert "Review me" in next_text and "owner's decision" in next_text


def test_an_older_host_says_nothing_about_isolation(verifying_host):  # noqa: F811
    run_id = planned(verifying_host, title="Old host")
    turn(run_id, 3)
    [attempt] = attempts(run_id)
    result(verifying_host, run_id, attempt.id, exit_code=1, tail="FAILED")
    with session_scope() as db:
        packet = control.review_packet(db, db.get(AgentRun, run_id))
        [shown] = [control.public_attempt(a) for a in [db.get(control.AgentAttempt, attempt.id)]]
    assert shown["isolated"] is None and shown["workspace_complete"] is None and shown["workspace_reasons"] == []
    assert "tests failed (exit 1) just now;" in packet["text"] and "sandbox" not in packet["text"]
    assert "no PR recorded" in packet["text"]


def test_a_check_that_could_not_run_names_the_host_fix(verifying_host):  # noqa: F811
    run_id = planned(verifying_host, title="Infra")
    turn(run_id, 3)
    [attempt] = attempts(run_id)
    result(verifying_host, run_id, attempt.id, exit_code=127, tail="bash: uv: command not found")
    with session_scope() as db:
        packet = control.review_packet(db, db.get(AgentRun, run_id))
    assert "verification could not run (" in packet["text"] and "tests failed" not in packet["text"]


def needs_you(host, gap):  # noqa: F811
    run_id = planned(host, verify=False, title="Parked")
    assert turn(run_id, 1)[0] == "judging"
    assert judged(run_id, stub(verdict("needs_you", gap=gap, criteria=(("c1", "not_met"),)))) == \
        ("needs_you", "judge_needs_you")
    return run_id


def test_the_page_shows_the_judges_full_needs_you_reason(host, client):  # noqa: F811
    gap = "The turn says the endpoint works, but nothing shows it. " * 12  # Longer than a transition detail (300).
    run_id = needs_you(host, gap)
    detail = client.get(f"/api/v1/remotea/runs/{run_id}").json()
    assert detail["reason_detail"] == gap.strip() or detail["reason_detail"] == gap
    with session_scope() as db:
        packet = control.review_packet(db, db.get(AgentRun, run_id))
    assert packet["text"].startswith("0 of 1 criteria met; no verify command")
    assert gap.strip() in packet["reason"]


def test_edit_criteria_then_judge_again(host, client):  # noqa: F811
    run_id = needs_you(host, "Say what done means.")
    before = revision(run_id)
    stale = client.post(f"/api/v1/remotea/runs/{run_id}/criteria",
                        json={"criteria": [{"id": "c1", "text": "the endpoint returns 200"}], "expected_revision": before - 1})
    assert stale.status_code == 409 and stale.json()["error"]["code"] == "REVISION_CONFLICT"
    test_kind = client.post(f"/api/v1/remotea/runs/{run_id}/criteria",
                            json={"criteria": [{"text": "tests pass", "kind": "verify"}], "expected_revision": before})
    assert test_kind.status_code == 400 and "verify command" in test_kind.json()["error"]["message"]
    r = client.post(f"/api/v1/remotea/runs/{run_id}/criteria", json={
        "criteria": [{"id": "c1", "text": "the endpoint returns 200"}, {"text": "the docs mention it"}],
        "expected_revision": before})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["state"] == "needs_you" and body["plan_revision"] == 2 and body["state_revision"] == before + 1
    assert [c["text"] for c in body["plan"]["criteria"]] == ["the endpoint returns 200", "the docs mention it"]
    assert all(c.get("by") is None for c in body["plan"]["criteria"])  # The human's now.
    assert [h for h in body["history"] if h.get("event") == "criteria_set"]
    with session_scope() as db:
        task = db.get(Task, db.get(AgentRun, run_id).task_id)
        assert [c["text"] for c in task.success_criteria] == ["the endpoint returns 200", "the docs mention it"]
    old = client.post(f"/api/v1/remotea/runs/{run_id}/approve", json={"expected_revision": before})
    assert old.status_code == 409  # An approval sent against the old criteria conflicts.
    again = client.post(f"/api/v1/remotea/runs/{run_id}/judge", json={"expected_revision": before + 1})
    assert again.status_code == 200, again.text
    assert state(run_id) == ("judging", "judge_again")
    [*_, attempt] = attempts(run_id, "judge")
    assert attempt.plan_revision == 2 and attempt.turn_seq == 1
    criteria = (("c1", "met"), (body["plan"]["criteria"][1]["id"], "met"))
    assert judged(run_id, stub(verdict(criteria=criteria))) == ("awaiting_approval", "judge_met")
    busy = client.post(f"/api/v1/remotea/runs/{run_id}/judge", json={"expected_revision": revision(run_id)})
    assert busy.status_code == 200  # awaiting_approval may be judged again too.
    assert state(run_id)[0] == "judging"
    refused = client.post(f"/api/v1/remotea/runs/{run_id}/judge", json={"expected_revision": revision(run_id)})
    assert refused.status_code == 409 and refused.json()["error"]["code"] == "RUN_NOT_PARKED"


def test_judge_again_verifies_first_when_the_plan_has_a_verify_command(verifying_host, client):  # noqa: F811
    run_id = planned(verifying_host, title="Reverify")
    turn(run_id, 3)
    [attempt] = attempts(run_id)
    result(verifying_host, run_id, attempt.id, exit_code=1, tail="FAILED")
    judged(run_id, stub(verdict("needs_you", gap="Tests fail.", criteria=(("c1", "not_met"),))))
    assert state(run_id)[0] == "needs_you"
    r = client.post(f"/api/v1/remotea/runs/{run_id}/judge", json={"expected_revision": revision(run_id)})
    assert r.status_code == 200, r.text
    assert state(run_id) == ("verifying", "judge_again")
    latest = attempts(run_id)[-1]
    assert latest.status == "running" and latest.turn_seq == 3 and latest.id != attempt.id


def test_edit_criteria_is_only_for_a_parked_run(verifying_host, client):  # noqa: F811
    run_id = planned(verifying_host, title="Busy")
    with session_scope() as db:
        row = db.get(AgentRun, run_id)
        assert coordinator.frozen(db, row) is not None and row.state == "executing"
    r = client.post(f"/api/v1/remotea/runs/{run_id}/criteria",
                    json={"criteria": [{"text": "x"}], "expected_revision": revision(run_id)})
    assert r.status_code == 409 and r.json()["error"]["code"] == "RUN_NOT_PARKED"
    judge = client.post(f"/api/v1/remotea/runs/{run_id}/judge", json={"expected_revision": revision(run_id)})
    assert judge.status_code == 409 and judge.json()["error"]["code"] == "RUN_NOT_PARKED"
