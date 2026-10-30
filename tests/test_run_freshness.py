"""Run freshness: the base a run started from, Eridani-reserved migration numbers, remotea version pairing."""

import threading
from types import SimpleNamespace

import httpx
import pytest
from jarvis import agent_control as control
from jarvis import agent_link, agent_plan, remotea_release
from jarvis import migration_numbers as numbers
from jarvis import run_coordinator as coordinator
from jarvis.db import session_scope
from jarvis.session_models import AgentCommand, AgentHost, AgentRun, MigrationReservation
from sqlalchemy import select
from test_agent_plan import PLAN, call, interviewing
from test_agent_runs import OWNER, quiet, start, task  # noqa: F401 - fixture
from test_agent_runs import host as host  # noqa: F401 - pytest fixture
from test_linear import run  # noqa: F401

BOT = SimpleNamespace(owner_id=OWNER, id=None, scopes=["agents:session"])
SHA = "1a2b3c4d5e6f7a8b9c0d1a2b3c4d5e6f7a8b9c0d"


def configure(host_id, head=None):
    """The host lists `migrations` on its target (hello) and reports the chain head (worktrees.report)."""
    with session_scope() as db:
        row = db.get(AgentHost, host_id)
        row.caps = {**row.caps, "targets": [{"id": "jarvis", "repo": "/home/davin/jarvis", "worktree_root": "/home/davin/wt",
                                             "migrations": {"dir": "migrations/versions"}}]}
        if head is not None:
            report(db, row, [], heads=[{"target": "jarvis", "dir": "migrations/versions", "ref": "origin/main", "head": head}])


def report(db, host_row, worktrees, **extra):
    agent_link.worktrees(db, BOT, host_row.id, agent_link.WorktreesReport.model_validate(
        {"v": 1, "id": "w", "t": "worktrees.report", "worktrees": worktrees, **extra}))


def spawned(run_id, **facts):
    with session_scope() as db:
        row = db.scalar(select(AgentCommand).where(AgentCommand.run_id == run_id, AgentCommand.kind == "spawn"))
        coordinator.on_outcome(db, row, "executed", "", {"surf": "s1", "incarnation": 1, "worktree": "/wt/x",
                                                          "branch": f"task/{run_id}", "base": "main", **facts})


def public(run_id):
    with session_scope() as db:
        return control.public_run(db, db.get(AgentRun, run_id))


def end(run_id, to="failed"):
    with session_scope() as db:
        row = db.get(AgentRun, run_id, with_for_update=True)
        row.state = next(s for s in ("executing", "stopping", "awaiting_approval") if to in coordinator.TRANSITIONS[s])
        row.stop_final = "completed" if to == "completed" else "stopped"
        db.flush()
        coordinator.transition(db, row, to, reason="test")


def reservation(run_id):
    with session_scope() as db:
        row = db.scalar(select(MigrationReservation).where(MigrationReservation.run_id == run_id))
        return (row.number, row.status) if row else None


# ---------------------------------------------------------------- the base a run started from


def test_the_run_shows_its_base_with_behind_live_from_the_worktrees_report(host):  # noqa: F811
    run_id = start(host, task()["id"])["id"]
    spawned(run_id, base_sha=SHA, base_ref="origin/main", fetch={"ok": True}, behind=0, ahead=0)
    base = public(run_id)["base"]
    assert base["sha"] == SHA and base["ref"] == "origin/main" and base["behind"] == 0 and base["live"] is False
    with session_scope() as db:
        run_row = db.get(AgentRun, run_id)
        report(db, db.get(AgentHost, host.id), [
            {"target": "jarvis", "path": "/wt/x", "branch": run_row.branch, "base": "main", "base_ref": "origin/main",
             "behind": 4, "ahead": 0, "merged": None, "merged_reason": "no commits of its own yet", "candidate": True}])
    base = public(run_id)["base"]
    assert base["behind"] == 4 and base["live"] is True and base["as_of"] and base["behind_ref"] == "origin/main"


def test_an_older_host_without_base_facts_shows_nothing_and_a_failed_fetch_is_kept(host):  # noqa: F811
    first = start(host, task("Older host")["id"])["id"]
    spawned(first, base_sha=None)
    assert public(first)["base"] is None
    end(first, "stopped")
    second = start(host, task("Offline")["id"])["id"]
    spawned(second, base_sha=SHA, base_ref="main", fetch={"ok": False, "error": "could not read from remote"}, behind=2)
    assert public(second)["base"]["fetch"] == {"ok": False, "error": "could not read from remote"}


def test_a_branch_with_no_commits_of_its_own_is_not_merged(host):  # noqa: F811
    run_id = start(host, task("Fresh branch")["id"])["id"]
    with session_scope() as db:
        run_row = db.get(AgentRun, run_id)
        report(db, db.get(AgentHost, host.id), [{"target": "jarvis", "branch": run_row.branch, "base": "main",
                                                 "ahead": 0, "merged": None, "merged_reason": "no commits of its own yet"}])
        assert coordinator.branch_merged(db, run_row) is False
        report(db, db.get(AgentHost, host.id), [{"target": "jarvis", "branch": run_row.branch, "base": "main",
                                                 "ahead": 2, "merged": True}])
        assert coordinator.branch_merged(db, run_row) is True


def test_hello_stores_the_remotea_version(client):
    from test_agent_link import bearer, connect, enable, hello

    token = enable(client)["token"]
    with connect(bearer(token)) as ws:
        hello(ws, version="0.1.0")
        assert client.get("/api/v1/remotea/status").json()["hosts"][0]["version"] == "0.1.0"


# ---------------------------------------------------------------- migration numbers


def test_a_plan_with_a_migration_reserves_the_next_number_and_the_agent_is_told(host):  # noqa: F811
    configure(host.id, head=32)
    created, run_id = interviewing(host)
    plan = call(agent_plan.submit_plan, agent_plan.SubmitPlanInput, run_id, **PLAN, migration=True)
    assert plan["migration"] == {"number": 33, "dir": "migrations/versions"}
    read = call(agent_plan.task_plan, agent_plan.PlanQuery, run_id)
    assert read["run"]["migration"]["number"] == 33 and "0033" in read["run"]["migration_note"]
    with session_scope() as db:
        row = db.get(AgentRun, run_id)
        assert "number 0033 in migrations/versions" in coordinator.render_plan(db, row, coordinator.frozen(db, row))


def test_override_and_held_reservations_raise_the_next_number(host):  # noqa: F811
    configure(host.id, head=32)
    with session_scope() as db:
        listed = numbers.configure(db, OWNER, host.id, "jarvis", override=40)["items"]
    assert listed[0]["next"] == 41 and listed[0]["head"] == 32
    _, first = interviewing(host, "First migration")
    assert call(agent_plan.submit_plan, agent_plan.SubmitPlanInput, first, **PLAN, migration=True)["migration"]["number"] == 41
    end(first, "stopped")  # Released: nothing higher was reserved since.
    assert reservation(first) == (41, "released")
    _, second = interviewing(host, "Second migration")
    assert call(agent_plan.submit_plan, agent_plan.SubmitPlanInput, second, **PLAN, migration=True)["migration"]["number"] == 41


def test_settings_set_the_folder_and_pattern_without_host_config(host, client):  # noqa: F811
    with session_scope() as db:
        listed = numbers.listing(db, OWNER)
    assert [(m["host_target"], m["configured"], m["dir"], m["next"]) for m in listed] == [("jarvis", False, None, None)]
    saved = client.post("/api/v1/remotea/migrations", json={"host_id": host.id, "host_target": "jarvis",
                                                             "dir": "/db/migrations/", "pattern": "^(\\d{3})-", "override": 7})
    assert saved.status_code == 400  # Absolute folder refused.
    saved = client.post("/api/v1/remotea/migrations", json={"host_id": host.id, "host_target": "jarvis",
                                                             "dir": "db/migrations/", "pattern": "^(\\d{3})-", "override": 7})
    assert saved.status_code == 200, saved.text
    item = saved.json()["items"][0]
    assert item | {"reported_at": None} == {
        "host_id": host.id, "host_name": item["host_name"], "host_target": "jarvis", "configured": True,
        "source": "eridani", "dir": "db/migrations", "pattern": "^(\\d{3})-", "host_dir": None, "head": None,
        "override": 7, "last_reserved": None, "next": 8, "reported_at": None}
    _, run_id = interviewing(host)
    plan = call(agent_plan.submit_plan, agent_plan.SubmitPlanInput, run_id, **PLAN, migration=True)
    assert plan["migration"] == {"number": 8, "dir": "db/migrations"}
    for bad in ({"dir": "../up"}, {"dir": "x", "pattern": "("}, {"dir": "x", "pattern": "^\\d+_"}, {"pattern": "^(\\d)"}):
        refused = client.post("/api/v1/remotea/migrations", json={"host_id": host.id, "host_target": "jarvis", **bad})
        assert refused.status_code == 400, bad
    assert client.get("/api/v1/remotea/migrations").json()["items"][0]["dir"] == "db/migrations"


def test_a_settings_folder_overrides_the_host_and_ignores_its_head_for_another_folder(host):  # noqa: F811
    configure(host.id, head=32)
    with session_scope() as db:
        item = numbers.listing(db, OWNER)[0]
        assert (item["source"], item["dir"], item["head"], item["next"]) == ("host", "migrations/versions", 32, 33)
        item = numbers.configure(db, OWNER, host.id, "jarvis", dir="alembic/versions")["items"][0]
        # The host read its head from migrations/versions, not this folder: it does not count.
        assert (item["source"], item["dir"], item["host_dir"], item["head"], item["next"]) == (
            "eridani", "alembic/versions", "migrations/versions", None, 1)
        item = numbers.configure(db, OWNER, host.id, "jarvis")["items"][0]  # Cleared: back to the host's.
        assert (item["source"], item["head"]) == ("host", 32)


def test_without_a_configured_folder_nothing_is_reserved(host):  # noqa: F811
    _, run_id = interviewing(host)
    plan = call(agent_plan.submit_plan, agent_plan.SubmitPlanInput, run_id, **PLAN, migration=True)
    assert "migration" not in plan and "No migrations folder" in plan["migration_note"]
    assert reservation(run_id) is None


def test_two_concurrent_freezes_never_get_the_same_number(host):  # noqa: F811
    configure(host.id, head=32)
    runs = [interviewing(host, f"Concurrent {i}")[1] for i in range(4)]
    barrier, got, errors = threading.Barrier(len(runs)), [], []

    def freeze(run_id):
        try:
            with session_scope() as db:
                row = db.get(AgentRun, run_id)
                barrier.wait()
                got.append(numbers.reserve(db, row)["number"])
        except Exception as exc:  # noqa: BLE001 - surfaced below
            errors.append(exc)

    threads = [threading.Thread(target=freeze, args=(r,)) for r in runs]
    for t in threads:
        t.start()
    for t in threads:
        t.join(30)
    assert not errors and sorted(got) == [33, 34, 35, 36]


def test_release_keeps_the_gap_when_a_higher_number_was_reserved_since(host):  # noqa: F811
    configure(host.id, head=32)
    _, first = interviewing(host, "Lower")
    _, second = interviewing(host, "Higher")
    call(agent_plan.submit_plan, agent_plan.SubmitPlanInput, first, **PLAN, migration=True)
    call(agent_plan.submit_plan, agent_plan.SubmitPlanInput, second, **PLAN, migration=True)
    end(first, "failed")
    assert reservation(first) == (33, "held")  # 34 exists: the gap stays rather than reusing 33.
    end(second, "capped")
    assert reservation(second) == (34, "released")


def test_a_later_plan_without_the_migration_releases_it(host):  # noqa: F811
    configure(host.id, head=32)
    created, run_id = interviewing(host)
    call(agent_plan.submit_plan, agent_plan.SubmitPlanInput, run_id, **PLAN, migration=True)
    from jarvis.models import Task

    with session_scope() as db:
        expected = db.get(Task, created["id"]).plan_revision
    call(agent_plan.submit_plan, agent_plan.SubmitPlanInput, run_id, **{**PLAN, "expected_plan_revision": expected})
    assert reservation(run_id) == (33, "released")


def test_a_later_run_of_the_task_keeps_its_number_and_the_spawn_carries_it(host):  # noqa: F811
    configure(host.id, head=32)
    created, run_id = interviewing(host)
    call(agent_plan.submit_plan, agent_plan.SubmitPlanInput, run_id, **PLAN, migration=True)
    end(run_id, "completed")  # Completed keeps the number (the migration shipped with the work).
    assert reservation(run_id) == (33, "held")
    again = start(host, created["id"])["id"]
    with session_scope() as db:
        spawn = db.scalar(select(AgentCommand).where(AgentCommand.run_id == again, AgentCommand.kind == "spawn"))
        assert spawn.payload["migration"] == {"number": 33, "dir": "migrations/versions"}
        assert "0033" in spawn.payload["prompt"]
    assert reservation(again) == (33, "held")


# ---------------------------------------------------------------- version pairing


@pytest.mark.parametrize(("have", "latest", "nag"), [
    ("0.1.0", "0.2.0", False), ("0.1.0", "0.3.0", True), ("0.3.0", "0.3.0", False), ("0.4.0", "0.3.0", False),
    ("0.9.0", "1.0.0", True), ("v0.2.1", "v0.3.5", False), (None, "0.3.0", False), ("dev", "0.3.0", False)])
def test_update_available_only_past_one_minor_release(have, latest, nag):
    assert remotea_release.behind(have, latest) is nag


def test_the_cached_release_drives_the_host_note(host):  # noqa: F811
    with session_scope() as db:
        row = db.get(AgentHost, host.id)
        row.version = "0.1.0"
        remotea_release.set_cache("0.3.0", "https://github.com/rdavisdunham/remotea/releases/tag/v0.3.0")
        assert agent_link.public_host(row)["update"] == {
            "latest": "0.3.0", "url": "https://github.com/rdavisdunham/remotea/releases/tag/v0.3.0"}
        remotea_release.set_cache(None)  # No release published.
        assert agent_link.public_host(row)["update"] is None


def test_release_lookup_404_and_failure_show_no_note():
    class Client:
        def __init__(self, response=None, error=None):
            self.response, self.error = response, error

        def get(self, url, **kwargs):
            if self.error:
                raise self.error
            return self.response

    request = httpx.Request("GET", remotea_release.LATEST_URL)
    assert remotea_release.refresh(Client(httpx.Response(404, request=request)))["version"] is None
    assert remotea_release.update_for("0.0.1") is None
    assert remotea_release.refresh(Client(error=httpx.ConnectError("offline")))["ok"] is False
    assert remotea_release.update_for("0.0.1") is None
    found = remotea_release.refresh(Client(httpx.Response(200, request=request, json={
        "tag_name": "v0.5.0", "html_url": "https://github.com/rdavisdunham/remotea/releases/tag/v0.5.0"})))
    assert found["version"] == "0.5.0" and remotea_release.update_for("0.1.0")["latest"] == "0.5.0"
    remotea_release.set_cache(None)
