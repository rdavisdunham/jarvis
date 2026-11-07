"""Merge from Eridani (run_merge.py): the merge panel's states, Merge completing the run, Update branch sending the
run back to CI on the new head, refusals, a lost answer reconciled by the next CI answer, and a settled run merging."""

import pytest
from jarvis import agent_control as control
from jarvis import run_ci, run_merge
from jarvis import run_coordinator as coordinator
from jarvis.db import session_scope
from jarvis.domain import DomainError
from jarvis.models import Task
from jarvis.session_models import AgentCommand, AgentHost, AgentRun
from sqlalchemy import select
from test_agent_attempts import planned
from test_agent_plan import HUMAN, inputs, turn, verifying_host  # noqa: F401 - fixture
from test_agent_runs import OWNER, quiet, state  # noqa: F401 - fixture
from test_agent_runs import host as host  # noqa: F401 - fixture
from test_run_ci import HEAD, PR, checks, met, met_without_verify

NEW = "9" * 40
MERGE_COMMIT = "c" * 40


@pytest.fixture
def merge_host(verifying_host, monkeypatch):  # noqa: F811
    """The `tests` entry is the quick check; the host answers ci.status, pr.merge and pr.update_branch."""
    monkeypatch.setattr(run_ci, "quick_ids", lambda: {"tests"})
    with session_scope() as db:
        row = db.get(AgentHost, verifying_host.id)
        row.caps = {**row.caps, "caps": [*row.caps["caps"], "ci.status", "pr.merge", "pr.update_branch"]}
    return verifying_host


def pr(**values):
    return {**PR, "state": "OPEN", "base": "main", "draft": False, "mergeable": "MERGEABLE", "merge_state": "CLEAN",
            **values}


def answer(run_id, pr_=None, **values):
    """The host's ci.status answer for the newest poll (a fresh one when that was answered)."""
    with session_scope() as db:
        row = db.scalar(select(AgentCommand).where(AgentCommand.run_id == run_id, AgentCommand.kind == "ci.status")
                        .order_by(AgentCommand.created_at.desc()).limit(1))
        if row is None or row.status not in ("queued", "sent"):
            row = run_ci.request(db, db.get(AgentRun, run_id, with_for_update=True))
        result = {"pr": pr() if pr_ is None else pr_, "checks": checks("pass", "pass", "pass"), "worktree_head": HEAD,
                  **values}
        coordinator.on_outcome(db, row, "executed", "", result)
    return state(run_id)


def panel(run_id):
    with session_scope() as db:
        return control.public_run(db, db.get(AgentRun, run_id))["merge"]


def click(run_id, kind="merge", head=HEAD):
    with session_scope() as db:
        body = control.MergeInput(expected_head_sha=head)
        fn = run_merge.merge if kind == "merge" else run_merge.update_branch
        return fn(db, OWNER, run_id, body, actor=HUMAN)


def command(run_id, kind):
    with session_scope() as db:
        return db.scalar(select(AgentCommand).where(AgentCommand.run_id == run_id, AgentCommand.kind == kind)
                         .order_by(AgentCommand.created_at.desc()).limit(1))


def outcome(run_id, kind, state_, result=None, detail=""):
    with session_scope() as db:
        row = db.scalar(select(AgentCommand).where(AgentCommand.run_id == run_id, AgentCommand.kind == kind)
                        .order_by(AgentCommand.created_at.desc()).limit(1))
        coordinator.on_outcome(db, row, state_, detail, result)
    return state(run_id)


def green_awaiting_approval(merge_host, **pr_values):
    run_id = planned(merge_host)
    assert met(merge_host, run_id) == ("awaiting_ci", "ci_pending")
    assert answer(run_id, pr(**pr_values)) == ("awaiting_approval", "judge_met")
    return run_id


def test_the_panel_waits_on_ci_then_merge_completes_the_run(merge_host):
    run_id = planned(merge_host)
    assert panel(run_id)["status"] == "unknown"  # No CI answer yet.
    met(merge_host, run_id)
    answer(run_id, checks=checks("pass", "pending"))
    view = panel(run_id)
    assert view["status"] == "waiting_ci" and not view["can_merge"] and view["detail"] == "Waiting on CI."
    assert answer(run_id) == ("awaiting_approval", "judge_met")
    view = panel(run_id)
    assert view["status"] == "ready" and view["can_merge"] and view["pr"]["head_sha"] == HEAD
    assert view["detail"] == "Merge PR #42 into main."

    click(run_id)
    sent = command(run_id, "pr.merge")
    assert sent.payload["head_sha"] == HEAD and sent.payload["target"] == "jarvis"
    assert panel(run_id)["status"] == "merging"
    outcome(run_id, "pr.merge", "executed", {"merged": True, "already": False, "pr": pr(state="MERGED"),
                                              "merge_commit": MERGE_COMMIT})
    with session_scope() as db:
        run = db.get(AgentRun, run_id)
        assert run.state in ("stopping", "completed")
        assert db.get(Task, run.task_id).status == "completed"
        assert run.sync_state["merge"]["state"] == "merged"
        assert any(h.get("event") == "merged_pr" and h.get("number") == 42 for h in run.history)
    assert panel(run_id)["status"] == "merged" and panel(run_id)["detail"] == "Merged PR #42 into main."


def test_merge_is_refused_on_a_moved_head_or_while_ci_runs(merge_host):
    run_id = green_awaiting_approval(merge_host)
    with pytest.raises(DomainError) as moved:
        click(run_id, head=NEW)
    assert moved.value.code == "HEAD_MOVED"
    answer(run_id, checks=checks("pass", "pending"))
    with pytest.raises(DomainError) as waiting:
        click(run_id)
    assert waiting.value.code == "MERGE_UNAVAILABLE" and waiting.value.message == "Waiting on CI."


def test_behind_offers_update_branch_and_ci_decides_again_on_the_new_head(merge_host):
    run_id = green_awaiting_approval(merge_host, merge_state="BEHIND")
    view = panel(run_id)
    assert view["status"] == "behind" and view["can_update"] and not view["can_merge"]
    with pytest.raises(DomainError):
        click(run_id)
    click(run_id, "update")
    assert command(run_id, "pr.update_branch").payload["head_sha"] == HEAD
    assert panel(run_id)["status"] == "updating"
    assert outcome(run_id, "pr.update_branch", "executed",
                   {"updated": True, "new_head_sha": NEW, "worktree_synced": False}) == ("awaiting_ci", "ci_update")
    # GitHub's new head: the worktree is still on the old one, which is not unpushed work (Eridani updated it).
    assert answer(run_id, pr(head_sha=NEW), checks=checks("pass", "pending")) == ("awaiting_ci", "ci_update")
    assert answer(run_id, pr(head_sha=NEW)) == ("awaiting_approval", "judge_met")
    view = panel(run_id)
    assert view["status"] == "ready" and view["pr"]["head_sha"] == NEW


def test_a_refusal_is_shown_and_a_lost_answer_is_reconciled(merge_host):
    run_id = green_awaiting_approval(merge_host)
    click(run_id)
    outcome(run_id, "pr.merge", "rejected", {"reason": "blocked", "message": "the base branch policy prohibits the merge"},
            "blocked")
    view = panel(run_id)
    assert view["op"]["state"] == "refused" and view["op"]["reason"] == "blocked"
    assert state(run_id) == ("awaiting_approval", "judge_met")
    # Click again; the host's answer is lost, and the next CI answer shows the pull request merged.
    answer(run_id)
    click(run_id)
    assert outcome(run_id, "pr.merge", "uncertain", None, "interrupted by a server restart")[0] == "awaiting_approval"
    answer(run_id, pr(state="MERGED"))
    with session_scope() as db:
        run = db.get(AgentRun, run_id)
        assert run.state in ("stopping", "completed") and run.sync_state["merge"]["state"] == "merged"


def test_a_repository_without_ci_merges_from_needs_your_verification(merge_host):
    run_id = planned(merge_host, verify=False)
    met_without_verify(run_id)
    answer(run_id, checks=[])
    assert answer(run_id, checks=[]) == ("needs_you", "verify_manual")
    view = panel(run_id)
    assert view["status"] == "ready" and view["ci"] == "none" and view["can_merge"]
    click(run_id)
    outcome(run_id, "pr.merge", "executed", {"merged": True, "already": False, "pr": pr(state="MERGED")})
    with session_scope() as db:
        run = db.get(AgentRun, run_id)
        [entry] = [h for h in run.history if h.get("event") == "approve_override"]
        assert entry["reason"] == "verify_manual" and entry["note"] == "Merged PR #42 into main"


def test_a_settled_run_only_merges(merge_host):
    run_id = green_awaiting_approval(merge_host)
    with session_scope() as db:
        db.get(AgentRun, run_id).state = "completed"
    assert panel(run_id)["can_merge"]
    click(run_id)
    outcome(run_id, "pr.merge", "executed", {"merged": True, "already": True, "pr": pr(state="MERGED")})
    assert state(run_id)[0] == "completed"
    assert panel(run_id)["status"] == "merged"


def test_the_agent_still_working_or_an_old_host_grays_merge_out(merge_host):
    run_id = green_awaiting_approval(merge_host)
    with session_scope() as db:
        control.send_message(db, OWNER, run_id, control.MessageInput(text="One more thing"), actor=HUMAN)
    assert panel(run_id)["status"] == "working"
    with session_scope() as db:
        run = db.get(AgentRun, run_id)
        run.state = "awaiting_approval"
        row = db.get(AgentHost, merge_host.id)
        row.caps = {**row.caps, "caps": [c for c in row.caps["caps"] if not c.startswith("pr.")]}
    view = panel(run_id)
    assert view["status"] == "host" and "Update remotea" in view["detail"] and not view["can_merge"]


def test_conflicts_and_failed_ci_are_the_agents_to_fix(merge_host):
    run_id = green_awaiting_approval(merge_host, mergeable="CONFLICTING", merge_state="DIRTY")
    assert panel(run_id)["status"] == "conflicts"
    answer(run_id, checks=checks("pass", "fail"))
    with session_scope() as db:
        db.get(AgentRun, run_id).state = "awaiting_approval"
    answer(run_id, checks=checks("pass", "fail"))
    assert panel(run_id)["status"] == "ci_failed"


def test_refresh_asks_the_host_at_most_every_30_seconds(merge_host):
    run_id = green_awaiting_approval(merge_host)
    with session_scope() as db:
        run = db.get(AgentRun, run_id)
        run.branch = "eri/task-1"
        run_ci.save(run, requested_at=None)  # The last poll was long ago.
    before = len([c for c in commands(run_id) if c.kind == "ci.status"])
    with session_scope() as db:
        run_merge.refresh(db, OWNER, run_id)
        run_merge.refresh(db, OWNER, run_id)
    assert len([c for c in commands(run_id) if c.kind == "ci.status"]) == before + 1


def commands(run_id):
    with session_scope() as db:
        return list(db.scalars(select(AgentCommand).where(AgentCommand.run_id == run_id)))


# ---------------------------------------------------------------- auto-merge (phase 1b)


def auto(run_id, enabled=True):
    with session_scope() as db:
        return run_merge.set_auto(db, OWNER, run_id, control.AutoMergeInput(enabled=enabled), actor=HUMAN)


def test_auto_merge_merges_on_green_ci_and_completes_like_auto_complete(merge_host):
    from jarvis.remotea_routes import set_enabled

    with session_scope() as db:
        set_enabled(db, OWNER, True, "remotea_auto_merge")  # New runs follow the setting.
    run_id = planned(merge_host)
    assert panel_auto(run_id) == {"enabled": True, "available": True, "why": None, "tries": 0, "active": False}
    met(merge_host, run_id)
    assert answer(run_id) == ("awaiting_approval", "auto_merge")  # Green: it waits there while it merges.
    answer(run_id)  # The fresh answer auto_start asked for: merge.
    sent = command(run_id, "pr.merge")
    assert sent is not None and sent.payload["head_sha"] == HEAD and sent.payload.get("origin") in (None, "automation")
    outcome(run_id, "pr.merge", "executed", {"merged": True, "already": False, "pr": pr(state="MERGED")})
    with session_scope() as db:
        run = db.get(AgentRun, run_id)
        assert run.state in ("stopping", "completed") and db.get(Task, run.task_id).status == "completed"
        assert not any(h.get("event") == "approve_override" for h in run.history)  # The judge completed it.


def test_auto_merge_updates_a_branch_behind_its_base_then_stops_after_three_tries(merge_host):
    run_id = planned(merge_host)
    auto(run_id)
    met(merge_host, run_id)
    answer(run_id)
    head = HEAD
    for n in range(3):
        assert answer(run_id, pr(head_sha=head, merge_state="BEHIND")) == ("awaiting_approval", "auto_merge")
        assert command(run_id, "pr.update_branch").payload["head_sha"] == head
        new = str(n + 1) * 40
        assert outcome(run_id, "pr.update_branch", "executed", {"updated": True, "new_head_sha": new}) == ("awaiting_ci", "ci_update")
        assert answer(run_id, pr(head_sha=new)) == ("awaiting_approval", "auto_merge")  # Green on the new head.
        head = new
    assert answer(run_id, pr(head_sha=head, merge_state="BEHIND")) == ("needs_you", "merge_blocked")
    assert panel_auto(run_id)["tries"] == 3


def test_auto_merge_stops_for_conflicts_and_refusals(merge_host):
    run_id = planned(merge_host)
    auto(run_id)
    met(merge_host, run_id)
    answer(run_id)
    assert answer(run_id, pr(mergeable="CONFLICTING", merge_state="DIRTY")) == ("needs_you", "merge_blocked")
    with session_scope() as db:
        assert "conflicts" in control.public_run(db, db.get(AgentRun, run_id))["reason_detail"]

    other = green_awaiting_approval(merge_host)
    auto(other)  # Turned on while the run already waits for approval: it starts now.
    answer(other)
    outcome(other, "pr.merge", "rejected", {"reason": "permission", "message": "must have write access"}, "permission")
    assert state(other) == ("needs_you", "merge_blocked")


def test_auto_merge_needs_a_real_check(merge_host):
    run_id = planned(merge_host, verify=False)
    with session_scope() as db:
        assert run_merge.auto_available(db, db.get(AgentRun, run_id)) == "Needs a verify entry or CI."
    with pytest.raises(DomainError) as refused:
        auto(run_id)
    assert refused.value.code == "AUTO_MERGE_UNAVAILABLE"
    met_without_verify(run_id)
    answer(run_id, checks=[])
    assert answer(run_id, checks=[]) == ("needs_you", "verify_manual")  # Manual merge only.
    assert panel(run_id)["can_merge"]


def panel_auto(run_id):
    with session_scope() as db:
        return control.public_run(db, db.get(AgentRun, run_id))["auto_merge"]
