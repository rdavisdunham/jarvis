import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from xml.etree import ElementTree as ET

import pytest

from scripts.app_eval.campaign import exit_status, import_evidence
from scripts.app_eval.ledger import Ledger
from scripts.app_eval.registry import jobs, select
from scripts.app_eval.reporting import atomic_json, compare, report


def campaign(tmp_path):
    rows = select(cases="task_capture.01")
    work = jobs(rows, ["live-model"], include_support=False)
    manifest = {
        "cases": rows,
        "jobs": work,
        "commit": "test",
        "fingerprint": "source",
        "catalog_sha256": "catalog",
        "corpus_sha256": "corpus",
        "harness_sha256": "harness",
        "model": "luna",
        "modes": ["live-model"],
        "repeats": 1,
        "created_at": (datetime.now(UTC) - timedelta(minutes=5)).isoformat(),
    }
    atomic_json(tmp_path / "manifest.json", manifest)
    (tmp_path / "results").mkdir()
    Ledger(tmp_path / "budget.sqlite").initialize()
    return manifest


def test_reports_keep_component_failure_visible_and_shared_cost(tmp_path):
    manifest = campaign(tmp_path)
    job = manifest["jobs"][0]
    atomic_json(tmp_path / "results/one.json", {**job, "job_id": job["id"], "status": "failed"})
    ledger = Ledger(tmp_path / "budget.sqlite")
    paid = ledger.reserve("trial", "agent", "luna", ".02")
    ledger.settle(paid, ".001")
    ledger.reserve("trial", "embedding", "embedding", ".01")
    result = report(tmp_path)
    assert result["summary"]["failed"] == 1
    assert result["by_type"]["contracts"]["failed"] == 1
    assert result["cost_by_kind"]["agent"]["estimated_usd"] == "0.001"
    assert result["cost_by_kind"]["embedding"]["including_uncertain_usd"] == "0.01"
    assert len(ET.parse(tmp_path / "junit.xml").findall(".//failure")) == 2
    assert exit_status(result) == 1


def test_incomplete_is_nonzero_without_inventing_failure():
    assert exit_status({"jobs": {"passed": 1}, "summary": {"blocked": 1}}) == 2
    assert exit_status({"jobs": {"passed": 1}, "summary": {"passed": 1}}) == 0


def test_comparison_refuses_changed_oracle(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    b.mkdir()
    campaign(a)
    m = campaign(b)
    assert compare(a, b)["changes"] == []
    m["harness_sha256"] = "changed"
    atomic_json(b / "manifest.json", m)
    with pytest.raises(ValueError, match="harness"):
        compare(a, b)


def test_manual_evidence_requires_current_source_real_attachment_and_device(tmp_path):
    m = campaign(tmp_path)
    item = {
        "case_id": "task_capture.01",
        "criteria": list(m["cases"][0]["criteria_map"]),
        "status": "passed",
        "observed_at": datetime.now(UTC).isoformat(),
        "observer": "Tester",
        "device": "Physical phone / browser version",
        "commit": "test",
        "fingerprint": "source",
        "attachments": ["evidence.txt"],
    }
    source = tmp_path / "input.json"
    source.write_text(json.dumps(item))
    with pytest.raises(ValueError, match="attachments"):
        import_evidence(tmp_path, source)
    (tmp_path / "evidence.txt").write_text("Actual observation")
    assert import_evidence(tmp_path, source)["summary"]["partial"] == 1
    assert len(list((tmp_path / "manual").glob("*.json"))) == 1
    for patch, message in [
        ({"device": ""}, "device"),
        ({"fingerprint": "old"}, "fingerprint"),
        ({"observed_at": (datetime.now(UTC) + timedelta(days=1)).isoformat()}, "observed"),
        ({"attachments": ["../outside.txt"]}, "attachments"),
    ]:
        source.write_text(json.dumps({**item, **patch}))
        with pytest.raises(ValueError, match=message):
            import_evidence(tmp_path, source)
