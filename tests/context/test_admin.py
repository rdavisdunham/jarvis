"""Owner processing controls (context/routes/admin.py) and the history CLI (scripts/context_export.py): pause bumps
the epoch so runs started before it cannot apply, reads and manual corrections continue while paused, the rollback
drill pauses and exports an importable history, an owner's history round-trips through the CLI, and the account
export carries the context history."""

import json

import pytest
from fastapi.testclient import TestClient
from jarvis.config import get_settings
from jarvis.context import decisions, export
from jarvis.context import models as m
from jarvis.context.routes import admin
from jarvis.db import session_scope
from jarvis.models import Job
from sqlalchemy import func, select

from scripts import context_export

from .test_export_roundtrip import graph

BASE = "/api/v1/context/admin"


def owner() -> str:
    return get_settings().owner_id


def state(owner_id: str) -> m.ContextOwnerState | None:
    with session_scope() as db:
        row = db.get(m.ContextOwnerState, owner_id)
        if row is not None:
            db.expunge(row)
        return row


def run_started(owner_id: str) -> str:
    """An automatic run that started under the owner's current epoch and has not applied yet."""
    with session_scope() as db:
        current = db.get(m.ContextOwnerState, owner_id)
        run = m.ContextRun(owner_id=owner_id, stage="extract", status="running", epoch=current.epoch if current else 0)
        db.add(run)
        db.flush()
        return run.id


def applies(run_id: str) -> bool:
    """The pause contract the pipeline enforces before applying: same epoch and not paused (B6 tests it end to end)."""
    with session_scope() as db:
        run = db.get(m.ContextRun, run_id)
        current = db.get(m.ContextOwnerState, run.owner_id)
        return run.epoch == (current.epoch if current else 0) and not (current and current.paused)


def test_pause_bumps_the_epoch_once_and_resume_keeps_it():
    with session_scope() as db:
        admin.pause(db, "a")  # creates the state row
    assert (state("a").epoch, state("a").paused) == (1, True)
    assert state("a").paused_at is not None
    with session_scope() as db:
        admin.pause(db, "a")  # already paused: nothing changes
    assert state("a").epoch == 1
    with session_scope() as db:
        admin.resume(db, "a")
    assert (state("a").epoch, state("a").paused, state("a").paused_at) == (1, False, None)
    with session_scope() as db:
        admin.pause(db, "a")
    assert (state("a").epoch, state("a").paused) == (2, True)


def test_pause_rejects_results_of_runs_started_before_it():
    with session_scope() as db:
        graph(db, "a")  # owner state at epoch 3
    before = run_started("a")
    assert applies(before)
    with session_scope() as db:
        admin.pause(db, "a")
    assert not applies(before)
    with session_scope() as db:
        admin.resume(db, "a")
    assert not applies(before)  # resuming does not revive a result from before the pause
    assert applies(run_started("a"))
    with session_scope() as db:
        runs = {r["id"]: r for r in admin.status(db, "a")["last_runs"]}
    assert runs[before]["epoch"] == 3 and not runs[before]["current_epoch"]


def test_sources_reads_and_manual_corrections_continue_while_paused(envelope):
    from jarvis.context import sources

    with session_scope() as db:
        ids = graph(db, "a")
        admin.pause(db, "a")
    with session_scope() as db:
        received = sources.receive(db, "a", envelope)
        assert sources.get_source(db, "a", received.source_id)["id"] == received.source_id
        assert export.export_history(db, "a")["assertions"]
        row = db.get(m.ContextAssertion, ids["new"])
        decision = decisions.wrong(db, "a", row.id, row.current_version, command_id="cmd-wrong", principal="a")
        assert decision.operation == "wrong"
    with session_scope() as db:
        assert db.get(m.ContextAssertion, ids["new"]).lifecycle == "retracted"
        assert db.get(m.ContextOwnerState, "a").paused


def test_status_reports_queue_runs_and_failures():
    with session_scope() as db:
        db.add_all([Job(owner_id="a", kind="context_extract", payload={}, status="queued"),
                    Job(owner_id="a", kind="context_extract", payload={}, status="failed",
                        result={"error": "model timeout"}),
                    Job(owner_id="a", kind="contextXextract", payload={}, status="queued"),  # LIKE wildcard
                    Job(owner_id="b", kind="context_extract", payload={}, status="queued"),
                    m.ContextRun(owner_id="a", stage="extract", status="failed", error="x" * 2000),
                    m.ContextRun(owner_id="a", stage="embed", status="succeeded")])
    with session_scope() as db:
        report = admin.status(db, "a")
    assert (report["epoch"], report["paused"], report["paused_at"]) == (0, False, None)
    assert report["queue"] == {"jobs": {"queued": 1, "failed": 1}, "runs": {"failed": 1, "succeeded": 1}}
    assert len(report["last_runs"]) == 2
    assert [r["stage"] for r in report["failures"]["runs"]] == ["extract"]
    assert len(report["failures"]["runs"][0]["error"]) == admin.ERROR_CHARS
    assert [j["error"] for j in report["failures"]["jobs"]] == ["model timeout"]


# ---------------------------------------------------------------- routes


def test_routes_pause_resume_and_report(client):
    response = client.get(f"{BASE}/status")
    assert response.status_code == 200 and response.json()["paused"] is False
    paused = client.post(f"{BASE}/pause").json()
    assert (paused["epoch"], paused["paused"]) == (1, True)
    assert client.post(f"{BASE}/pause").json()["epoch"] == 1
    resumed = client.post(f"{BASE}/resume").json()
    assert (resumed["epoch"], resumed["paused"]) == (1, False)


def test_rollback_drill_pauses_and_returns_an_importable_history(client):
    with session_scope() as db:
        graph(db, owner())
        graph(db, "other")
    started = run_started(owner())
    response = client.post(f"{BASE}/rollback-drill")
    assert response.status_code == 200, response.text
    drill = response.json()
    assert drill["verified"] is True and drill["problem"] is None
    assert drill["status"]["paused"] and drill["status"]["epoch"] == 4
    assert not applies(started)
    history = drill["history"]
    assert history["owner_id"] == owner() and history["owner_state"]["paused"] is True
    assert drill["counts"] == {name: len(history[name]) for name in export.SECTION_NAMES}
    assert drill["counts"]["assertions"] == 2
    with session_scope() as db:  # the verification import left nothing behind
        assert db.scalar(select(func.count()).select_from(m.ContextAssertion)) == 4
        assert not db.scalar(select(m.ContextOwnerState).where(m.ContextOwnerState.owner_id.like("drill:%")))
    # The document is the canonical format: it imports as another owner.
    with session_scope() as db:
        report = export.import_history(db, "copy", history, preserve_ids=False)
    assert report.counts["assertions"] == 2


def test_routes_need_a_session_and_csrf_and_refuse_shared_workspaces(client):
    from jarvis.api import app

    anonymous = TestClient(app)
    assert anonymous.get(f"{BASE}/status").status_code == 401
    assert anonymous.post(f"{BASE}/pause").status_code in {401, 403}
    no_csrf = client.headers.pop("X-CSRF-Token")
    assert client.post(f"{BASE}/pause").status_code == 403
    client.headers["X-CSRF-Token"] = no_csrf
    assert state(owner()) is None

    from test_accounts import client_for, shared

    guest = client_for("guest")
    shared(client, guest)
    for method, path in (("get", "status"), ("post", "pause"), ("post", "resume"), ("post", "rollback-drill")):
        response = getattr(client, method)(f"{BASE}/{path}")
        assert response.status_code == 403 and response.json()["error"]["code"] == "PERSONAL_WORKSPACE", path
    assert state(owner()) is None


# ---------------------------------------------------------------- account export and CLI


def test_account_export_includes_the_context_history(client):
    with session_scope() as db:
        graph(db, owner())
        graph(db, "other")
        expected = export.export_history(db, owner())
    exported = client.get("/api/v1/export").json()["context_history"]
    assert exported == json.loads(json.dumps(expected))
    assert exported["connections"] and "credential_ciphertext" not in exported["connections"][0]
    assert "gAAAA-secret" not in client.get("/api/v1/export").text


def test_cli_round_trips_an_owner_and_refuses_non_empty_targets(tmp_path, capsys):
    with session_scope() as db:
        graph(db, "a")
        original = json.loads(json.dumps(export.export_history(db, "a", include_embeddings=True)))
    out = tmp_path / "history.json"
    assert context_export.main(["--owner", "a", "--out", str(out), "--include-embeddings"]) == 0
    assert json.loads(capsys.readouterr().out)["counts"]["assertions"] == 2
    assert json.loads(out.read_text()) == original
    with pytest.raises(SystemExit, match="already exists"):
        context_export.main(["--owner", "a", "--out", str(out)])

    assert context_export.main(["--owner", "a", "--in", str(out)]) == 1  # owner already has rows
    assert "already has context rows" in capsys.readouterr().err

    with session_scope() as db:
        export.delete_history(db, "a")
    assert context_export.main(["--owner", "a", "--in", str(out), "--dry-run"]) == 0
    assert json.loads(capsys.readouterr().out)["dry_run"] is True
    with session_scope() as db:
        assert export.export_history(db, "a")["assertions"] == []

    assert context_export.main(["--owner", "a", "--in", str(out)]) == 0
    capsys.readouterr()
    with session_scope() as db:
        assert json.loads(json.dumps(export.export_history(db, "a", include_embeddings=True))) == original


def test_cli_imports_an_account_export_under_new_ids(client, tmp_path, capsys):
    with session_scope() as db:
        graph(db, owner())
    account = tmp_path / "account.json"
    account.write_text(client.get("/api/v1/export").text)
    assert context_export.main(["--owner", "copy", "--in", str(account), "--new-ids"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["counts"]["assertions"] == 2 and report["new_ids"] is True
    with session_scope() as db:
        copy = export.export_history(db, "copy")
    assert len(copy["decisions"]) == 2 and copy["owner_state"]["epoch"] == 3


def test_cli_rejects_mixed_flags(tmp_path):
    with pytest.raises(SystemExit):
        context_export.main(["--owner", "a", "--out", str(tmp_path / "x.json"), "--dry-run"])
    with pytest.raises(SystemExit):
        context_export.main(["--owner", "a", "--in", str(tmp_path / "x.json"), "--include-embeddings"])
    with pytest.raises(SystemExit):
        context_export.main(["--owner", "a"])
