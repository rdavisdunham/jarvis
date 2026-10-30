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
        control.approve(db, OWNER, run_id, control.ApproveInput(), actor=HUMAN)
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
