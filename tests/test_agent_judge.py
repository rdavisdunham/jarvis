"""Stage D: the judge with a stubbed model: hard rules, non-verdicts, CAS, completion policy, steps, caps, no progress."""

import pytest
from jarvis import agent_control as control
from jarvis import agent_judge as judge
from jarvis import run_coordinator as coordinator
from jarvis.db import session_scope
from jarvis.models import Job, Notification, OwnerSettings, Task
from jarvis.session_models import AgentAttempt, AgentCommand, AgentRun, AgentTurn
from sqlalchemy import select
from test_agent_attempts import attempts, planned, result
from test_agent_plan import HUMAN, turn, verifying_host  # noqa: F401 - fixture
from test_agent_runs import OWNER, quiet, state  # noqa: F401 - fixture
from test_agent_runs import host as host  # noqa: F401 - pytest fixture


def verdict(kind="met", gap="", criteria=(("c1", "met"),), steps=()):
    return judge.Verdict(verdict=kind, gap=gap, summary="s",
                         criteria=[judge.CriterionVerdict(id=i, status=s, evidence="seen") for i, s in criteria],
                         steps=[judge.StepVerdict(id=i, status=s) for i, s in steps])


def stub(*answers):
    calls = []

    def model(evidence):
        calls.append(evidence)
        answer = answers[min(len(calls), len(answers)) - 1]
        if isinstance(answer, Exception):
            raise answer
        return answer
    model.calls = calls
    return model


def job_for(run_id):
    with session_scope() as db:
        return db.scalars(select(Job).where(Job.kind == "agent_judge").order_by(Job.created_at.desc())).first().id \
            if run_id else None


def judged(run_id, model):
    judge.process(job_for(run_id), model=model)
    return state(run_id)


def last_judge(run_id):
    return attempts(run_id, "judge")[-1]


EVIDENCE = {"plan": {"criteria": [{"id": "c1", "text": "tests pass", "kind": "verify", "verify_id": "tests"},
                                  {"id": "c2", "text": "docs", "kind": "text"}],
                     "verify": {"target_id": "jarvis", "verify_id": "tests"}},
            "turn": {"text": "All done!", "truncated": False}}


def fresh(status="passed", **values):
    return {"verify_id": "tests", "status": status, "fresh": True, "exit_code": 0 if status == "passed" else 1,
            "tail": "FAILED test_x", "timed_out": False, "detail": "", **values}


# ---------------------------------------------------------------- hard rules (pure)


def test_no_criteria_is_never_met_and_skips_the_model():
    model = stub(verdict())
    final = judge.decide({"plan": {"criteria": []}}, model)
    assert final["verdict"] == "needs_you" and final["rules"] == ["no_criteria"] and not model.calls


def test_required_verify_must_be_fresh_and_passing():
    met = verdict(criteria=(("c1", "met"), ("c2", "met")))
    assert judge.decide({**EVIDENCE, "verify": fresh()}, stub(met))["verdict"] == "met"
    failed = judge.decide({**EVIDENCE, "verify": fresh("failed")}, stub(met))
    assert failed["verdict"] == "not_met" and "FAILED test_x" in failed["gap"]
    assert "required_verify_not_passing" in failed["rules"]
    for verify in (None, fresh("timed_out"), fresh("refused"), fresh("uncertain"), fresh(fresh=False)):
        final = judge.decide({**EVIDENCE, "verify": verify}, stub(met))
        assert final["verdict"] == "needs_you", verify  # Never met; an agent cannot fix a missing check.
    stale = judge.decide({**EVIDENCE, "verify": fresh(fresh=False)}, stub(met))
    assert "stale" in stale["gap"]


def test_verify_criteria_need_a_matching_passing_attempt_and_all_criteria_must_be_met():
    evidence = {**EVIDENCE, "plan": {**EVIDENCE["plan"], "verify": None}}
    final = judge.decide({**evidence, "verify": None}, stub(verdict(criteria=(("c1", "met"), ("c2", "met")))))
    assert final["verdict"] == "not_met" and final["criteria"][0]["status"] == "not_met"
    assert "verify_criterion_needs_passing_attempt" in final["rules"]
    partial = judge.decide({**EVIDENCE, "verify": fresh()}, stub(verdict(criteria=(("c1", "met"),))))
    assert partial["verdict"] == "not_met" and partial["criteria"][1]["evidence"] == "No evidence given."
    unknown = judge.decide({**EVIDENCE, "verify": fresh()}, stub(verdict(criteria=(("c1", "met"), ("c2", "met"),
                                                                                   ("evil", "met")))))
    assert [c["id"] for c in unknown["criteria"]] == ["c1", "c2"]


def test_disputed_steps_need_a_contradicting_verification():
    out = verdict("not_met", "fix", steps=(("s1", "disputed"), ("s2", "done")))
    assert judge.decide({**EVIDENCE, "verify": fresh()}, stub(out))["steps"][0]["status"] == "unclear"
    assert judge.decide({**EVIDENCE, "verify": fresh("failed")}, stub(out))["steps"][0]["status"] == "disputed"


def test_the_request_is_strict_json_on_the_extraction_model():
    from jarvis.memory_learning import EXTRACTION_MODEL

    body = judge.request(EVIDENCE)
    assert body["model"] == EXTRACTION_MODEL
    schema = body["response_format"]["json_schema"]
    assert schema["strict"] and schema["schema"]["additionalProperties"] is False
    assert "untrusted data" in body["messages"][0]["content"]


def test_model_failures_are_non_verdicts(monkeypatch):
    from jarvis import memory_learning
    from jarvis.domain import DomainError

    def reply(message):
        return lambda *a, **k: {"choices": [{"message": message}], "usage": {}}

    for fake, kind in ((reply({"refusal": "no"}), "refusal"), (reply({"content": "{\"verdict\": \"maybe\"}"}), "invalid")):
        monkeypatch.setattr(memory_learning, "extraction_request", fake)
        with pytest.raises(judge.NonVerdict) as caught:
            judge.ask(OWNER, EVIDENCE)
        assert caught.value.kind == kind

    def incomplete(*a, **k):
        raise DomainError("PROVIDER_INCOMPLETE", "x", 503)

    monkeypatch.setattr(memory_learning, "extraction_request", incomplete)
    with pytest.raises(judge.NonVerdict) as caught:
        judge.ask(OWNER, EVIDENCE)
    assert caught.value.kind == "incomplete"
    monkeypatch.setattr(memory_learning, "extraction_request",
                        reply({"content": verdict().model_dump_json()}))
    parsed, meta = judge.ask(OWNER, EVIDENCE)
    assert parsed.verdict == "met" and "latency_ms" in meta


# ---------------------------------------------------------------- the job and the coordinator


def test_an_infra_verification_needs_you_with_its_action_whatever_the_model_says():
    infra = fresh("error", exit_code=None, category="infra_database", infra=True,
                  action="Point the verify entry at a disposable test database.")
    for answer in (verdict(), verdict("not_met", "Fix the database code", criteria=(("c1", "not_met"),),
                                       steps=(("s1", "disputed"),))):
        model = stub(answer)
        final = judge.decide({**EVIDENCE, "verify": infra}, model)
        assert final["verdict"] == "needs_you" and "verify_infra" in final["rules"]
        assert final["gap"] == "Point the verify entry at a disposable test database."
        assert not final["verify_failed"] and all(s["status"] != "disputed" for s in final["steps"])
        assert model.calls[0]["verify"]["infra"] is True  # The model sees it could not run.
    failing = judge.decide({**EVIDENCE, "verify": fresh("failed", category="tests_failed", infra=False)},
                           stub(verdict(criteria=(("c1", "met"), ("c2", "met")))))
    assert failing["verdict"] == "not_met" and "verify_infra" not in failing["rules"]


def criteria_run(host, title="Judged", steps=None):  # noqa: F811
    run_id = planned(host, verify=False, title=title)
    if steps is not None:
        with session_scope() as db:
            row = db.get(AgentRun, run_id)
            task = db.get(Task, row.task_id)
            task.steps = steps
    return run_id


def test_met_goes_to_awaiting_approval_when_auto_complete_is_off(host):  # noqa: F811
    run_id = criteria_run(host)
    assert turn(run_id, 1)[0] == "judging"
    assert judged(run_id, stub(verdict())) == ("awaiting_approval", "judge_met")
    attempt = last_judge(run_id)
    assert attempt.status == "done" and attempt.result["verdict"] == "met"
    with session_scope() as db:
        row = db.get(AgentAttempt, attempt.id)
        assert row.prompt_version == judge.PROMPT_VERSION and row.model
        assert db.scalar(select(Notification).where(Notification.category == "agent_done"))
        assert db.get(Task, db.get(AgentRun, run_id).task_id).status != "completed"
    with session_scope() as db:
        control.approve(db, OWNER, run_id, control.ApproveInput(expected_revision=db.get(AgentRun, run_id).state_revision), actor=HUMAN)
    assert state(run_id)[0] in ("stopping", "completed")


def test_met_with_auto_complete_completes_as_the_judge(host):  # noqa: F811
    with session_scope() as db:
        db.get(OwnerSettings, OWNER).values = {**db.get(OwnerSettings, OWNER).values, "remotea_auto_complete": True}
    run_id = criteria_run(host)
    turn(run_id, 1)
    assert judged(run_id, stub(verdict())) == ("stopping", "judge_met")
    with session_scope() as db:
        row = db.get(AgentRun, run_id)
        task = db.get(Task, row.task_id)
        assert task.status == "completed" and row.stop_final == "completed"
        assert any(h.get("actor", {}).get("kind") == "run_judge" and h.get("to") == "stopping" for h in row.history)
        stop = db.scalar(select(AgentCommand).where(AgentCommand.run_id == run_id, AgentCommand.kind == "stop"))
        coordinator.on_outcome(db, stop, "executed", "terminated")
    assert state(run_id)[0] == "completed"


def test_auto_complete_refuses_a_gate_the_agent_wrote_alone(host):  # noqa: F811
    """Stage F, M1: criteria that are all agent-authored, with no required verify, wait for a human tap."""
    with session_scope() as db:
        db.get(OwnerSettings, OWNER).values = {**db.get(OwnerSettings, OWNER).values, "remotea_auto_complete": True}
    run_id = criteria_run(host)
    with session_scope() as db:
        from jarvis.session_models import AgentRunPlan

        plan = db.scalar(select(AgentRunPlan).where(AgentRunPlan.run_id == run_id))
        plan.criteria = [{**c, "by": "agent"} for c in plan.criteria]
    turn(run_id, 1)
    assert judged(run_id, stub(verdict())) == ("awaiting_approval", "judge_met")


def test_not_met_sends_the_gap_as_the_next_execution_turn(host):  # noqa: F811
    run_id = criteria_run(host)
    turn(run_id, 1)
    assert judged(run_id, stub(verdict("not_met", "Add the docs page", criteria=(("c1", "not_met"),)))) == \
        ("executing", "not_met")
    with session_scope() as db:
        row = db.get(AgentRun, run_id)
        gap = db.scalars(select(AgentCommand).where(AgentCommand.run_id == run_id, AgentCommand.kind == "input")
                         .order_by(AgentCommand.created_at.desc())).first()
        assert "Add the docs page" in gap.payload["text"] and gap.payload["mode"] == "queue"
        assert row.history[-1].get("source") == "eridani" or any(h.get("source") == "eridani" for h in row.history)
    assert turn(run_id, 2)[0] == "judging"
    with session_scope() as db:
        assert db.get(AgentRun, run_id).turn_count == 2


def test_needs_you_pushes_agent_stuck(host):  # noqa: F811
    run_id = criteria_run(host)
    turn(run_id, 1)
    assert judged(run_id, stub(verdict("needs_you", "Which database?"))) == ("needs_you", "judge_needs_you")
    with session_scope() as db:
        assert db.scalar(select(Notification).where(Notification.category == "agent_stuck"))


def test_non_verdicts_retry_twice_then_need_you(host):  # noqa: F811
    run_id = criteria_run(host)
    turn(run_id, 1)
    model = stub(judge.NonVerdict("invalid", "bad json"))
    assert judged(run_id, model) == ("needs_you", "judge_unavailable")
    assert len(model.calls) == 3 and last_judge(run_id).result["non_verdict"] == "invalid"
    other = criteria_run(host, title="Refused")
    turn(other, 1)
    refusal = stub(judge.NonVerdict("refusal", "no"))
    assert judged(other, refusal)[0] == "needs_you" and len(refusal.calls) == 1
    third = criteria_run(host, title="Recovers")
    turn(third, 1)
    assert judged(third, stub(judge.NonVerdict("timeout", "slow"), verdict()))[0] == "awaiting_approval"
    assert last_judge(third).result["verdict"] == "met"


def test_a_verdict_for_a_stale_tuple_is_discarded(host):  # noqa: F811
    run_id = criteria_run(host)
    turn(run_id, 1)
    with session_scope() as db:
        row = db.get(AgentRun, run_id, with_for_update=True)
        attempt = db.scalar(select(AgentAttempt).where(AgentAttempt.run_id == run_id, AgentAttempt.kind == "judge"))
        row.turn_seq += 1  # Another turn landed meanwhile.
        assert coordinator.on_verdict(db, attempt, {"verdict": "met", "gap": "", "criteria": [], "steps": []}) is None
        assert attempt.status == "stale" and row.state == "judging"
    with session_scope() as db:
        db.get(AgentRun, run_id).turn_seq -= 1
        control.send_message(db, OWNER, run_id, control.MessageInput(text="wait", mode="steer"), actor=HUMAN)
    model = stub(verdict())
    judge.process(job_for(run_id), model=model)
    assert not model.calls and state(run_id)[0] == "executing"  # The job saw the superseded attempt and stood down.


def test_step_inference_marks_done_and_disputed_but_never_downgrades(verifying_host):  # noqa: F811
    steps = [{"id": "s1", "title": "a", "status": "todo", "revision": 1}, {"id": "s2", "title": "b", "status": "done",
                                                                          "revision": 1},
             {"id": "s3", "title": "c", "status": "done", "revision": 1}]
    run_id = planned(verifying_host)
    with session_scope() as db:
        db.get(Task, db.get(AgentRun, run_id).task_id).steps = steps
    turn(run_id, 3)
    result(verifying_host, run_id, attempts(run_id)[0].id, exit_code=1)
    judged(run_id, stub(verdict("not_met", "tests fail", criteria=(("c1", "not_met"),),
                                steps=(("s1", "done"), ("s2", "disputed"), ("s3", "not_done")))))
    with session_scope() as db:
        merged = {s["id"]: s for s in db.get(Task, db.get(AgentRun, run_id).task_id).steps}
        assert merged["s1"]["status"] == "done" and merged["s1"]["by"].startswith("judge:")
        assert merged["s2"]["status"] == "disputed" and merged["s2"]["revision"] == 2
        assert merged["s3"]["status"] == "done" and merged["s3"]["revision"] == 1


def test_the_last_allowed_turn_is_judged_before_capping(host):  # noqa: F811
    run_id = criteria_run(host)
    with session_scope() as db:
        db.get(AgentRun, run_id).turn_cap = 2
    turn(run_id, 1)
    judged(run_id, stub(verdict("not_met", "more", criteria=(("c1", "not_met"),))))
    assert turn(run_id, 2)[0] == "judging"  # The cap does not cut the last turn short.
    assert judged(run_id, stub(verdict("not_met", "still more", criteria=(("c1", "not_met"),)))) == ("capped", "turn_cap")
    met = criteria_run(host, title="Met at cap")
    with session_scope() as db:
        db.get(AgentRun, met).turn_cap = 1
    turn(met, 1)
    assert judged(met, stub(verdict()))[0] == "awaiting_approval"


def test_repeated_identical_gaps_are_no_progress(host):  # noqa: F811
    run_id = criteria_run(host)
    same = verdict("not_met", "The login page still 500s.", criteria=(("c1", "not_met"),))
    for seq in (1, 2):
        turn(run_id, seq)
        assert judged(run_id, stub(same))[0] == "executing"
    turn(run_id, 3)
    assert judged(run_id, stub(verdict("not_met", "  the login page still 500s. ", criteria=(("c1", "not_met"),)))) == \
        ("needs_you", "no_progress_gap")


def test_evidence_flags_truncated_turns(host):  # noqa: F811
    run_id = criteria_run(host)
    with session_scope() as db:
        row = db.get(AgentRun, run_id)
        db.add(AgentTurn(owner_id=OWNER, session_pk=row.session_pk, host_id=row.host_id,
                                     provider_session_id="sess-1", provider_turn_id="t1", run_id=run_id, generation=1,
                                     seq=1, role="assistant", text="x" * (judge.TURN_CHARS + 10), truncated=False))
    turn(run_id, 1)
    model = stub(verdict())
    judged(run_id, model)
    assert model.calls[0]["turn"]["truncated"] and len(model.calls[0]["turn"]["text"]) == judge.TURN_CHARS


def test_wall_clock_covers_planning_and_a_stuck_judge_needs_you(host):  # noqa: F811
    from datetime import timedelta

    from jarvis.models import now
    from test_agent_plan import interviewing

    _, planning = interviewing(host)
    with session_scope() as db:
        coordinator.sweep(db, now() + timedelta(hours=7))
    assert state(planning) == ("needs_you", "wall_clock")
    run_id = criteria_run(host)
    turn(run_id, 1)
    with session_scope() as db:
        coordinator.sweep(db, now() + timedelta(minutes=16))
    assert state(run_id) == ("needs_you", "judge_unavailable") and last_judge(run_id).result["non_verdict"] == "timeout"


# ---------------------------------------------------------------- verify_missing: an operator-only gap (run 75661a23)


def test_a_gap_asking_for_a_recorded_verification_on_a_plan_without_verify_needs_you():
    """The judge wrote not_met six times for 'provide an independently recorded verification result' on a plan with
    no verify command. Nothing the agent does can record one, so the rule turns it into needs_you once."""
    evidence = {**EVIDENCE, "plan": {"criteria": [{"id": "c1", "text": "the full suite passes", "kind": "text"}], "verify": None}}
    model = stub(verdict("not_met", "Provide an independently recorded verification result for the compatibility suite.",
                         criteria=(("c1", "not_met"),)))
    final = judge.decide(evidence, model)
    assert final["verdict"] == "needs_you" and "verify_missing" in final["rules"]
    assert final["gap"].startswith("No verify command is set") and "independently recorded" in final["gap"]
    # A gap the agent can close stays not_met, and a plan with a verify command is judged by the existing rules.
    plain = judge.decide(evidence, stub(verdict("not_met", "The login page still 500s; fix the handler.", criteria=(("c1", "not_met"),))))
    assert plain["verdict"] == "not_met" and "verify_missing" not in plain["rules"]
    with_verify = judge.decide({**EVIDENCE, "verify": None}, stub(verdict("not_met", "Verification has not run yet.", criteria=(("c1", "not_met"),))))
    assert "verify_missing" not in with_verify["rules"]


def test_repeated_rule_gaps_count_as_no_progress_even_when_reworded():
    same_rule = [{"verdict": "not_met", "gap": f"wording {i}", "rules": ["criteria_incomplete"]} for i in range(3)]
    assert coordinator.stalled(same_rule)
    assert not coordinator.stalled([{"verdict": "not_met", "gap": f"wording {i}", "rules": []} for i in range(3)])
    assert not coordinator.stalled([*same_rule[:2], {"verdict": "met", "gap": "", "rules": ["criteria_incomplete"]}])


def test_verify_missing_parks_the_run_once_instead_of_looping(host):  # noqa: F811
    run_id = criteria_run(host)  # The plain host offers no verify id, so the plan has none.
    turn(run_id, 1)
    assert judged(run_id, stub(verdict("not_met", "Run and record the verification for the test suite independently.",
                                      criteria=(("c1", "not_met"),)))) == ("needs_you", "verify_missing")
    with session_scope() as db:
        row = db.get(AgentRun, run_id)
        assert row.turn_count == 1
        assert not db.scalar(select(AgentCommand).where(AgentCommand.run_id == run_id, AgentCommand.kind == "input",
                                                        AgentCommand.payload["text"].astext.like("%ERIDANI_GAP%")))


def input_count(run_id):
    with session_scope() as db:
        return len(list(db.scalars(select(AgentCommand).where(AgentCommand.run_id == run_id, AgentCommand.kind == "input"))))


def test_an_infra_verification_parks_the_run_without_a_gap_prompt(verifying_host):  # noqa: F811
    run_id = planned(verifying_host)
    turn(run_id, 3)
    result(verifying_host, run_id, attempts(run_id)[0].id, exit_code=1,
           tail="psycopg.OperationalError: connection to server at \"127.0.0.1\", port 5432 failed: Connection refused")
    sent = input_count(run_id)
    assert judged(run_id, stub(verdict("not_met", "Fix the database layer", criteria=(("c1", "not_met"),)))) == \
        ("needs_you", "verify_infra")
    assert input_count(run_id) == sent  # No gap prompt: the agent cannot fix the host.
    action = attempts(run_id)[0].result["action"]
    assert "disposable test database" in action
    with session_scope() as db:
        row = db.get(AgentRun, run_id)
        assert row.history[-1]["detail"] == action and control.reason_detail(db, row) == action
        public = control.public_run(db, row)
        assert public["reason_text"] == coordinator.REASONS["verify_infra"] and public["reason_detail"] == action
        assert control.spoken_run(db, row)["why"] == "host setup problem: " + action
    assert "verify_infra" in last_judge(run_id).result["rules"]


def test_failing_tests_still_send_the_gap_prompt(verifying_host):  # noqa: F811
    run_id = planned(verifying_host)
    turn(run_id, 3)
    result(verifying_host, run_id, attempts(run_id)[0].id, exit_code=1, tail="FAILED test_x - assert 1 == 2\n1 failed")
    assert attempts(run_id)[0].result["category"] == "tests_failed"
    sent = input_count(run_id)
    assert judged(run_id, stub(verdict(criteria=(("c1", "met"),)))) == ("executing", "not_met")
    assert input_count(run_id) == sent + 1
    with session_scope() as db:
        gap = db.scalars(select(AgentCommand).where(AgentCommand.run_id == run_id, AgentCommand.kind == "input")
                         .order_by(AgentCommand.created_at.desc())).first()
        assert "1 failed" in gap.payload["text"]


# ---------------------------------------------------------------- freestyle: talk to the agent, Eridani stays out


def test_freestyle_records_turns_without_verifying_judging_or_capping(host, client):  # noqa: F811
    run_id = criteria_run(host)
    with session_scope() as db:
        revision = db.get(AgentRun, run_id).state_revision
    r = client.post(f"/api/v1/remotea/runs/{run_id}/freestyle", json={"enabled": True, "expected_revision": revision})
    assert r.status_code == 200 and r.json()["freestyle"] is True, r.text
    assert turn(run_id, 1) == ("executing", "started")
    with session_scope() as db:
        row = db.get(AgentRun, run_id)
        assert row.turn_count == 0 and row.turn_seq == 1
        assert [h for h in row.history if h.get("event") == "turn" and h.get("freestyle")]
        assert db.scalar(select(Job).where(Job.kind == "agent_judge")) is None
    off = client.post(f"/api/v1/remotea/runs/{run_id}/freestyle", json={"enabled": False})
    assert off.status_code == 200 and off.json()["freestyle"] is False
    assert turn(run_id, 2) == ("judging", "turn")


def test_freestyle_resumes_a_run_the_judge_parked(host, client):  # noqa: F811
    run_id = criteria_run(host)
    turn(run_id, 1)
    assert judged(run_id, stub(verdict("needs_you", "Only you can decide.", criteria=(("c1", "not_met"),)))) == ("needs_you", "judge_needs_you")
    r = client.post(f"/api/v1/remotea/runs/{run_id}/freestyle", json={"enabled": True})
    assert r.status_code == 200, r.text
    assert state(run_id) == ("executing", "freestyle")


# ---------------------------------------------------------------- evidence the loop can check (task 6e753e0f)


@pytest.mark.parametrize("gap", [
    "Paste the full pytest output in your final response so I can confirm the suite passes.",
    "Provide the test output showing all tests pass.",
    "Include the git diff of the handler change in your reply.",
    "Please share the logs from the failing run in your next message.",
    "Provide evidence that the migration ran.",
])
def test_a_gap_asking_for_pasted_proof_is_rewritten_to_the_verify_action(gap):
    """The agent's reply is a claim: a gap asking it to paste output, logs or diffs can never close. With a verify
    command the gap becomes "make it pass"; the request to paste is dropped, the rest of the judge's note stays."""
    final = judge.decide({**EVIDENCE, "verify": fresh("failed")},
                         stub(verdict("not_met", gap + " The KeyError in test_x is still there.",
                                      criteria=(("c1", "not_met"),))))
    assert final["verdict"] == "not_met" and "proof_request" in final["rules"]
    assert "`tests` pass" in final["gap"] and "do not paste" in final["gap"]
    assert "KeyError in test_x" in final["gap"] and not judge.PROOF_ASK.search(final["gap"].split("missing:")[-1])


def test_a_pasted_proof_gap_without_any_check_needs_you():
    no_verify = {"plan": {"criteria": [{"id": "c1", "text": "the suite passes", "kind": "text"}], "verify": None},
                 "turn": {"text": "All tests pass:\n===== 40 passed ====="}}
    tests = judge.decide(no_verify, stub(verdict("not_met", "Show the test results.", criteria=(("c1", "not_met"),))))
    assert tests["verdict"] == "needs_you" and {"proof_request", "verify_missing"} <= set(tests["rules"])
    diff = judge.decide(no_verify, stub(verdict("not_met", "Attach the diff.", criteria=(("c1", "not_met"),))))
    assert diff["verdict"] == "needs_you" and "verify_missing" in diff["rules"]
    # Something the agent can still do that Eridani checks: tick its steps, open the PR.
    steps = {**no_verify, "plan": {**no_verify["plan"], "steps": [{"id": "s1", "status": "todo"}]},
             "completion": {"policy": "pr_open", "pr_urls": ["https://github.com/a/b/pull/1"]}}
    ticked = judge.decide(steps, stub(verdict("not_met", "Include the git diff of the handler change in your reply.",
                                              criteria=(("c1", "not_met"),))))
    assert ticked["verdict"] == "not_met" and "task_step" in ticked["gap"] and "pull request" in ticked["gap"]


def test_a_gap_that_only_mentions_output_is_left_alone():
    for gap in ("Fix the paste handler so pasted logs keep their newlines.",
                "The CLI should print the diff of the two configs; add that.", "Add a log line when the job retries."):
        final = judge.decide(EVIDENCE, stub(verdict("not_met", gap, criteria=(("c1", "not_met"),))))
        assert final["verdict"] == "not_met" and final["gap"] == gap and "proof_request" not in final["rules"]


def test_the_judge_and_gap_prompts_ask_only_for_checkable_actions():
    from jarvis import agent_prompts as prompts

    assert "claim" in judge.SYSTEM and "Never ask the agent to paste" in judge.SYSTEM
    text = prompts.gap("Make tests pass.", [{"id": "c1", "status": "not_met", "evidence": "x"}])
    assert "task_step" in text and "do not paste test output" in text


PR_URL = "https://github.com/rdavisdunham/jarvis/pull/90"


def pr_completion(policy, prs=(), merged=None, *, pr_fresh=True, merged_fresh=True, **extra):
    return {"policy": policy, "branch": "user/eri-1-x", "pr_urls": list(prs), "merged": merged,
            "pr_fresh": pr_fresh, "merged_fresh": merged_fresh, "as_of": {}, **extra}


@pytest.mark.parametrize("completion,expected,rule", [
    (pr_completion("pr_open", pr_fresh=False), "needs_you", "pr_unknown"),  # A stale record: never a gap.
    (pr_completion("pr_open"), "not_met", "pr_required"),  # Reported after the turn, no PR: the agent opens one.
    (pr_completion("pr_open", (PR_URL,), pr_fresh=False), "met", None),  # A PR on record does not go stale.
    (pr_completion("pr_merged", (PR_URL,), None, merged_reason="no commits of its own yet"), "needs_you", "pr_unknown"),
    (pr_completion("pr_merged", (PR_URL,), False, merged_fresh=False), "needs_you", "pr_unknown"),
    (pr_completion("pr_merged", (PR_URL,), False), "needs_you", "pr_not_merged"),
    (pr_completion("pr_merged", (PR_URL,), True, merged_fresh=False), "met", None),
])
def test_pr_evidence_freshness(completion, expected, rule):
    final = judge.decide({"plan": {"criteria": [{"id": "c1", "text": "done"}]}, "turn": {"text": "Opened the PR."},
                          "completion": completion}, stub(verdict()))
    assert final["verdict"] == expected
    assert (rule in final["rules"]) if rule else not final["rules"]
    if completion.get("merged_reason"):
        assert "no commits of its own yet" in final["gap"]


def pr_open_run(host):  # noqa: F811
    run_id = criteria_run(host, title="PR open")
    with session_scope() as db:
        coordinator.frozen(db, db.get(AgentRun, run_id)).completion_policy = "pr_open"
    return run_id


def report_session(run_id, prs=()):
    """The collector's session.report for the run's session, stored through agent_sessions.upsert."""
    from jarvis import agent_sessions as sessions
    from jarvis.session_models import AgentSession

    with session_scope() as db:
        row = db.get(AgentSession, db.get(AgentRun, run_id).session_pk)
        sessions.upsert(db, OWNER, row.tool, row.session_id, sessions.SessionInput(pr_urls=list(prs), turn_count=2))


def judge_jobs(run_id):
    with session_scope() as db:
        ids = {a.id for a in db.scalars(select(AgentAttempt).where(AgentAttempt.run_id == run_id,
                                                                   AgentAttempt.kind == "judge"))}
        return [j.id for j in db.scalars(select(Job).where(Job.kind == "agent_judge").order_by(Job.created_at))
                if j.payload.get("attempt") in ids]


def scans(run_id):
    with session_scope() as db:
        return [(c.id, c.op_id, c.payload) for c in db.scalars(select(AgentCommand).where(
            AgentCommand.run_id == run_id, AgentCommand.kind == "worktrees.scan"))]


def test_merged_null_is_unknown_in_the_evidence(host):  # noqa: F811
    from datetime import timedelta

    from jarvis.models import now
    from jarvis.session_models import AgentHost

    run_id = criteria_run(host)
    with session_scope() as db:
        run = db.get(AgentRun, run_id)
        row = db.get(AgentHost, run.host_id)
        row.caps = {**row.caps, "worktrees": {"at": now().isoformat(), "worktrees": [
            {"branch": run.branch, "base": "main", "merged": None, "merged_reason": "no commits of its own yet"}]}}
        db.flush()
        completion = judge.completion_evidence(db, run, coordinator.frozen(db, run), since=now() - timedelta(minutes=1))
        assert completion["merged"] is None and completion["merged_fresh"] is True
        assert completion["merged_reason"] == "no commits of its own yet"
        assert not coordinator.branch_merged(db, run)


def test_pr_open_with_a_fresh_record_proceeds(host):  # noqa: F811
    run_id = pr_open_run(host)
    turn(run_id, 1)
    report_session(run_id, [PR_URL])
    assert judged(run_id, stub(verdict())) == ("awaiting_approval", "judge_met")
    assert last_judge(run_id).result["evidence"]["completion"]["pr_fresh"] is True
    # A record after the turn that has no PR: the agent can open one, so it is a gap, not a question for Davin.
    other = pr_open_run(host)
    turn(other, 1)
    report_session(other)
    assert judged(other, stub(verdict())) == ("executing", "not_met")
    assert "pr_required" in last_judge(other).result["rules"]


def test_pr_open_with_a_stale_record_waits_then_needs_you_instead_of_not_met(host):  # noqa: F811
    from datetime import timedelta

    from jarvis.models import now

    run_id = pr_open_run(host)
    report_session(run_id)  # Before the turn: stale for it.
    turn(run_id, 1)
    model = stub(verdict())
    assert judged(run_id, model) == ("judging", "turn")  # Held, not judged from the old record.
    held = last_judge(run_id)
    assert held.status == "queued" and held.result["held"]["final"]["verdict"] == "met"
    assert held.result["held"]["scan"] is None and not scans(run_id)  # This host does not offer worktrees.scan.
    with session_scope() as db:
        coordinator.sweep(db)  # Inside the wait: nothing yet.
    assert len(judge_jobs(run_id)) == 1
    with session_scope() as db:
        coordinator.sweep(db, clock=now() + judge.EVIDENCE_WAIT + timedelta(seconds=5))
    assert len(judge_jobs(run_id)) == 2
    judge.process(judge_jobs(run_id)[-1], model=model)
    assert state(run_id) == ("needs_you", "pr_unknown") and len(model.calls) == 1  # The model was asked once.
    assert "no pull request on record" in last_judge(run_id).result["gap"]
    # The record arrives later with the PR: the run moves on as the held verdict would have.
    report_session(run_id, [PR_URL])
    assert state(run_id) == ("awaiting_approval", "judge_met")


def test_a_held_judge_runs_again_when_the_record_arrives(host):  # noqa: F811
    from jarvis.session_models import AgentHost

    run_id = pr_open_run(host)
    with session_scope() as db:
        row = db.get(AgentHost, host.id)
        row.caps = {**row.caps, "caps": [*row.caps.get("caps", []), "worktrees.scan"]}
    turn(run_id, 1)
    assert judged(run_id, stub(verdict())) == ("judging", "turn")
    [(scan_id, op_id, payload)] = scans(run_id)
    assert payload["branch"] and payload["session_id"] == "sess-1" and payload["turn_seq"] == 1
    assert last_judge(run_id).result["held"]["scan"] == op_id
    with session_scope() as db:
        from jarvis import agent_link

        row = db.get(AgentHost, host.id)
        sent = [f for f in agent_link.drain(db, row, row.conn_epoch) if f["t"] == "worktrees.scan"]
        assert [f["op_id"] for f in sent] == [op_id] and sent[0]["digest"]
    report_session(run_id, [PR_URL])
    report_session(run_id, [PR_URL])  # Released once until it runs.
    assert len(judge_jobs(run_id)) == 2
    judge.process(judge_jobs(run_id)[-1], model=stub(AssertionError("the held verdict is reused")))
    assert state(run_id) == ("awaiting_approval", "judge_met")
    # Whatever the host did with the scan, its outcome never moves the run.
    with session_scope() as db:
        coordinator.on_outcome(db, db.get(AgentCommand, scan_id), "uncertain", "lost")
    assert state(run_id) == ("awaiting_approval", "judge_met")
