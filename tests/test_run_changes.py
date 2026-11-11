"""What a run changed, as evidence for the judge (run_changes.py): the request after a turn, the host's answer stored
for that turn, and the judge's `changes` evidence."""

import pytest
from jarvis import run_changes
from jarvis import run_coordinator as coordinator
from jarvis.db import session_scope
from jarvis.session_models import AgentCommand, AgentHost, AgentRun
from sqlalchemy import select
from test_agent_attempts import attempts, planned, result
from test_agent_judge import judged, stub, verdict
from test_agent_plan import HUMAN, inputs, turn, verifying_host  # noqa: F401 - fixture
from test_agent_runs import OWNER, quiet, state  # noqa: F401 - fixture
from test_agent_runs import host as host

PATCH = "diff --git a/app.py b/app.py\n+print('hello')\n"
ANSWER = {"head": "a" * 40, "base_ref": "origin/main", "base_sha": "b" * 40,
          "files": [{"path": "app.py", "added": 1, "removed": 0}], "files_total": 1,
          "untracked": ["notes/"], "untracked_total": 1, "uncommitted": True,
          "patch": PATCH, "patch_truncated": False, "patch_chars": len(PATCH)}


@pytest.fixture
def changes_host(verifying_host):  # noqa: F811
    with session_scope() as db:
        row = db.get(AgentHost, verifying_host.id)
        row.caps = {**row.caps, "caps": [*row.caps["caps"], "run.changes"]}
    return verifying_host


def commands(run_id):
    with session_scope() as db:
        return list(db.scalars(select(AgentCommand).where(
            AgentCommand.run_id == run_id, AgentCommand.kind == "run.changes").order_by(AgentCommand.created_at)))


def answer(run_id, state_="executed", detail="", **values):
    with session_scope() as db:
        row = db.scalar(select(AgentCommand).where(AgentCommand.run_id == run_id, AgentCommand.kind == "run.changes")
                        .order_by(AgentCommand.created_at.desc()).limit(1))
        coordinator.on_outcome(db, row, state_, detail, {**ANSWER, **values} if state_ == "executed" else None)


def test_a_turn_asks_the_host_for_the_changes_and_the_judge_reads_them(changes_host):
    run_id = planned(changes_host)
    turn(run_id, 3)
    [command] = commands(run_id)
    assert command.payload["turn_seq"] == 3 and command.payload["max_chars"] == run_changes.HOST_PATCH_CHARS
    assert command.payload["branch"] and command.payload["target"] == "jarvis"
    with session_scope() as db:
        run = db.get(AgentRun, run_id)
        assert not any(e.get("kind") == "run.changes" for e in run.history)  # Not on the timeline.
        assert run_changes.pending(db, run_id, run.generation, 3)
    before = state(run_id)
    answer(run_id)
    assert state(run_id) == before  # The answer is evidence; it does not move the run.
    with session_scope() as db:
        assert not run_changes.pending(db, run_id, db.get(AgentRun, run_id).generation, 3)
    [record] = attempts(run_id, "changes")
    assert record.status == "done" and record.turn_seq == 3 and record.result["patch"] == PATCH
    result(changes_host, run_id, attempts(run_id, "verify")[-1].id, turn_seq=3)
    model = stub(verdict())
    judged(run_id, model)
    changes = model.calls[0]["changes"]
    assert changes == {"available": True, "head": "a" * 40, "base": "origin/main",
                       "files": [{"path": "app.py", "added": 1, "removed": 0}], "files_total": 1,
                       "untracked": ["notes/"], "uncommitted": True, "patch": PATCH, "patch_truncated": False}
    recorded = attempts(run_id, "judge")[-1].result["evidence"]["changes"]
    assert recorded["available"] and recorded["files_total"] == 1 and "patch" not in recorded


def test_the_judge_is_told_why_there_are_no_changes_to_read(changes_host):
    run_id = planned(changes_host)
    turn(run_id, 3)
    answer(run_id, "failed", "no worktree for task/x on this host")
    result(changes_host, run_id, attempts(run_id, "verify")[-1].id, turn_seq=3)
    model = stub(verdict())
    judged(run_id, model)
    assert model.calls[0]["changes"] == {"available": False, "reason": "no worktree for task/x on this host"}


def test_a_host_without_the_capability_is_not_asked(verifying_host):  # noqa: F811
    run_id = planned(verifying_host)
    turn(run_id, 3)
    assert commands(run_id) == []
    result(verifying_host, run_id, attempts(run_id, "verify")[-1].id, turn_seq=3)
    model = stub(verdict())
    judged(run_id, model)
    assert model.calls[0]["changes"] == {"available": False, "reason": "this host does not report changes"}


def test_a_long_patch_is_cut_for_the_judge_and_the_wait_ends_with_the_answer(changes_host):
    run_id = planned(changes_host)
    turn(run_id, 3)
    result(changes_host, run_id, attempts(run_id, "verify")[-1].id, turn_seq=3)
    judge_id = attempts(run_id, "judge")[-1].id
    naps = []

    def nap(seconds):  # The answer arrives during the second nap.
        naps.append(seconds)
        if len(naps) == 2:
            answer(run_id, patch="x" * 50_000, patch_chars=50_000)

    assert run_changes.wait(judge_id, sleep=nap) == 2.0
    assert run_changes.wait(judge_id, sleep=nap) == 0.0 and len(naps) == 2
    model = stub(verdict())
    judged(run_id, model)
    changes = model.calls[0]["changes"]
    assert len(changes["patch"]) == run_changes.JUDGE_PATCH_CHARS and changes["patch_truncated"]
