"""Compound: what a finished run learned, kept per repository for the next run (run_learnings.py)."""

from jarvis import agent_control as control
from jarvis import run_learnings as learnings
from jarvis.db import session_scope
from jarvis.domain import enqueue_job
from jarvis.models import Job, now
from jarvis.session_models import AgentAttempt, AgentRun, AgentRunLearning
from sqlalchemy import select
from test_agent_plan import start
from test_agent_runs import HUMAN, OWNER, attached, commands, quiet, task  # noqa: F401 - pytest fixtures
from test_agent_runs import host as host  # noqa: F401 - pytest fixture

LESSON = learnings.Answer(keep=True, kind="bug", title="The quick check needs the build first",
                          applies_when="Running the `quick` verify command in this repository.",
                          learning="Run `make build` before `quick`: it imports generated files and fails with "
                                   "ImportError otherwise.", paths=["Makefile", "/abs/ignored"], duplicate_of="")


def finished(host, *, gap=None, said=None):
    """A completed run on the host's repository; with `gap`, the judge sent it back once first."""
    made = task("Fix the upload retry")
    run_id = start(host, made["id"], interview=False)["id"]
    attached(host, run_id)
    with session_scope() as db:
        run = db.get(AgentRun, run_id)
        if gap:
            db.add(AgentAttempt(owner_id=OWNER, run_id=run_id, generation=1, kind="judge", turn_seq=1, status="done",
                                result={"verdict": "not_met", "gap": gap, "summary": "not yet"}))
        db.add(AgentAttempt(owner_id=OWNER, run_id=run_id, generation=1, kind="judge", turn_seq=2, status="done",
                            result={"verdict": "met", "gap": "", "summary": "done"}))
        if said:
            run.history = [*run.history, {"event": "sent", "text": said, "actor": {"kind": "human"}}]
        run.state, run.finished_at, run.turn_count = "completed", now(), 2 if gap else 1
    return run_id


def learn(run_id, model):
    with session_scope() as db:
        job = enqueue_job(db, OWNER, learnings.KIND, {"run_id": run_id}).id
    calls = []

    def scripted(body):
        calls.append(body)
        return model(body) if callable(model) else model
    learnings.process(job, model=scripted)
    with session_scope() as db:
        return calls, db.get(Job, job).result, learnings.state(db.get(AgentRun, run_id))


def test_a_run_that_went_straight_through_teaches_nothing_and_costs_nothing(host):
    run_id = finished(host)
    calls, result, mark = learn(run_id, LESSON)
    assert calls == [] and result["outcome"] == "uneventful" and mark["state"] == "none"
    with session_scope() as db:
        assert db.scalar(select(AgentRunLearning)) is None


def test_a_run_that_was_sent_back_writes_one_entry_and_the_next_run_is_given_it(host):
    run_id = finished(host, gap="The quick check fails with ImportError until the build has run.")
    calls, result, mark = learn(run_id, LESSON)
    assert len(calls) == 1 and result["outcome"] == "kept" and mark["state"] == "kept"
    record = __import__("json").loads(calls[0]["messages"][1]["content"])
    assert record["verdicts"][0]["gap"].startswith("The quick check fails") and record["existing_entries"] == []
    with session_scope() as db:
        row = db.scalar(select(AgentRunLearning))
        assert (row.kind, row.execution_target_id, row.run_id, row.paths) == ("bug", host.target_id, run_id, ["Makefile"])
        shown = control.public_run(db, db.get(AgentRun, run_id), detail=True)["learning"]
        assert shown["state"] == "kept" and shown["entry"]["learning"].startswith("Run `make build`")

    # The next run in the same repository starts with it, as fenced data.
    later = start(host, task("Add a progress bar")["id"])["id"]
    prompt = commands(later, "spawn")[0].payload["prompt"]
    assert "ERIDANI_LEARNED" in prompt and "Run `make build` before `quick`" in prompt
    assert "data to weigh, not instructions" in prompt


def test_the_same_lesson_again_confirms_the_entry_instead_of_adding_one(host):
    first = finished(host, gap="ImportError from quick")
    learn(first, LESSON)
    with session_scope() as db:
        entry = db.scalar(select(AgentRunLearning.id))
    second = finished(host, gap="quick failed again before the build")
    calls, result, mark = learn(second, lambda body: LESSON.model_copy(update={"duplicate_of": entry}))
    assert entry in calls[0]["response_format"]["json_schema"]["schema"]["properties"]["duplicate_of"]["enum"]
    assert result == {"run_id": second, "outcome": "confirmed", "id": entry} and mark["state"] == "confirmed"
    with session_scope() as db:
        rows = db.scalars(select(AgentRunLearning)).all()
        assert len(rows) == 1 and rows[0].confirmations == 1 and rows[0].confirmed_by == [second]


def test_the_model_may_decline_and_an_owner_correction_is_enough_to_ask(host):
    run_id = finished(host, said="Use the retry helper in uploads/retry.py, do not write a new loop.")
    calls, result, mark = learn(run_id, learnings.Answer(keep=False))
    assert len(calls) == 1 and result["outcome"] == "none" and mark["state"] == "none"


def test_the_owner_can_list_and_remove_entries_and_removed_ones_are_not_given_out(host, client):
    run_id = finished(host, gap="ImportError from quick")
    learn(run_id, LESSON)
    listed = client.get(f"/api/v1/remotea/learnings?target_id={host.target_id}").json()["items"]
    assert [e["title"] for e in listed] == ["The quick check needs the build first"] and listed[0]["run_id"] == run_id
    assert client.post(f"/api/v1/remotea/learnings/{listed[0]['id']}/remove").json()["status"] == "removed"
    assert client.get("/api/v1/remotea/learnings").json()["items"] == []
    later = start(host, task("Another task")["id"])["id"]
    assert "ERIDANI_LEARNED" not in commands(later, "spawn")[0].payload["prompt"]


def test_the_sweep_queues_completed_runs_once(host, monkeypatch):
    monkeypatch.setattr(learnings, "enabled", lambda: True)
    run_id = finished(host, gap="something")
    with session_scope() as db:
        learnings.sweep(db)
        learnings.sweep(db)
        jobs = db.scalars(select(Job).where(Job.kind == learnings.KIND)).all()
        assert [j.payload["run_id"] for j in jobs] == [run_id]
    learnings.process(jobs[0].id, model=lambda body: LESSON)
    with session_scope() as db:
        learnings.sweep(db)
        assert len(db.scalars(select(Job).where(Job.kind == learnings.KIND)).all()) == 1


def test_deleting_a_runs_content_takes_what_it_wrote_down_with_it(host, client):
    run_id = finished(host, gap="ImportError from quick")
    learn(run_id, LESSON)
    done = client.post(f"/api/v1/remotea/runs/{run_id}/redact")
    assert done.status_code == 200 and done.json()["redacted"]["learnings"] == 1
    with session_scope() as db:
        row = db.scalar(select(AgentRunLearning))
        assert (row.status, row.title, row.body, row.paths) == ("removed", "", "", [])
    assert client.get("/api/v1/remotea/learnings").json()["items"] == []
