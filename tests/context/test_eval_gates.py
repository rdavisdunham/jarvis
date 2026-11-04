"""Evaluation gates (package F1): scoring math (nDCG, paired bootstrap, Wilson intervals), the gate rules
(unverified data is never a pass; critical temporal and permission failures override averages), the question sets,
the runner's database mode on one history through the real pipeline, the three-arm comparison on it, and the
adjudication worksheet round trip. No model is called: `offline()` refuses every provider call."""

import ast
import csv
import json
import shutil
from pathlib import Path

import pytest
from jarvis.config import get_settings

from evals.context import compare, gates, report, runner, worksheet
from evals.context.compare import Index, covers, ndcg, paired_bootstrap, score_hits
from evals.context.gates import Count, Critical, Quality

ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------- scoring math


def test_ndcg_uses_binary_gains_and_the_ideal_ranking():
    assert ndcg([1, 0, 1], 2) == pytest.approx((1 + 1 / 2) / (1 + 1 / 1.5849625007211562))
    assert ndcg([0, 0, 0, 0, 0, 1], 1) == 0.0  # outside the cut-off
    assert ndcg([1, 1], 1) == pytest.approx(1 + 1 / 1.5849625007211562)  # the caller credits each target once
    assert ndcg([], 0) is None


def test_paired_bootstrap_is_deterministic_and_brackets_the_mean():
    a = [0.2, 0.5, 0.1, 0.9, 0.4, None, 0.3, 0.6]
    b = [0.4, 0.5, 0.3, 1.0, 0.3, 0.8, 0.6, 0.9]
    first = paired_bootstrap(a, b, samples=500, seed=3)
    assert first == paired_bootstrap(a, b, samples=500, seed=3)
    assert first["n"] == 7  # the pair with a missing value is dropped
    assert first["low"] <= first["difference"] <= first["high"] and first["difference"] == pytest.approx(0.1429, 1e-3)
    assert paired_bootstrap([0.0] * 4, [1.0] * 4, samples=100) | {} == {
        "n": 4, "mean_a": 0.0, "mean_b": 1.0, "difference": 1.0, "low": 1.0, "high": 1.0}
    assert paired_bootstrap([None], [1.0])["n"] == 0


def test_wilson_interval():
    low, high = gates.wilson(9, 10)
    assert (low, high) == pytest.approx((0.5958, 0.9821), abs=1e-3)
    assert gates.wilson(0, 0) is None and gates.wilson(5, 5)[1] == 1.0


def test_hits_cover_targets_by_evidence_overlap_segment_or_chunk_range():
    index = Index(segments={"s1": ("m1", 2, "r1", "Decision: we host on AWS.")},
                  chunks={"c1": ("m1", 0, 4, "r1")}, assertions={"a1": [("m1", 2, 0, 24)], "a2": [("m1", 3, 0, 9)]},
                  live={"r1"}, texts={("r1", 2): "Decision: we host on AWS.", ("r1", 3): "Old: Azure"})
    target = [("m1", 2, 10, 24)]
    assert covers(index, {"ref": "assertion", "id": "a1"}, target)
    assert not covers(index, {"ref": "assertion", "id": "a2"}, target)
    assert covers(index, {"ref": "segment", "id": "s1"}, target) and covers(index, {"ref": "chunk", "id": "c1"}, target)
    q = {"targets": {"expected": [target], "stale": [[("m1", 3, 0, 9)]], "forbidden": [[("m9", 0, 0, 5)]]}}
    stale_first = score_hits(index, q, [{"ref": "assertion", "id": "a2", "citation": None},
                                        {"ref": "assertion", "id": "a1", "citation": {
                                            "revision_id": "r1", "segment": 2, "quote": "we host on AWS"}}])
    assert stale_first["stale"] == "stale" and stale_first["recall"] == 1.0 and not stale_first["leak"]
    assert stale_first["ndcg"] == pytest.approx(1 / 1.5849625007211562)
    assert (stale_first["citations"], stale_first["valid_citations"]) == (2, 1)  # no citation is not valid
    current = score_hits(index, q, [{"ref": "segment", "id": "s1", "citation": {
        "revision_id": "r1", "segment_start": 2, "segment_end": 2}}])
    assert current["stale"] == "current" and current["ndcg"] == 1.0 and current["valid_citations"] == 1


# ---------------------------------------------------------------- gate rules


def graded_quality(verified=(0, 0), unverified=(10, 10), target=0.95):
    return gates.grade_quality(Quality("q", "q", target, Count(*verified), Count(*unverified)))


def test_unverified_data_is_not_gradeable_never_a_pass():
    perfect_but_unverified = graded_quality(verified=(0, 0), unverified=(10, 10))
    assert perfect_but_unverified["status"] == gates.NOT_GRADEABLE
    assert perfect_but_unverified["unverified"]["rate"] == 1.0
    assert graded_quality(verified=(19, 20))["status"] == gates.PASS
    assert graded_quality(verified=(18, 20))["status"] == gates.FAIL
    assert gates.grade_critical(Critical("t", "t", "temporal", warnings=["unverified stale answer"]))["status"] == \
        gates.NOT_GRADEABLE
    assert gates.verdict([], [perfect_but_unverified], []) == "NOT GRADEABLE"


def test_critical_failures_override_passing_averages():
    passing = [graded_quality(verified=(100, 100))]
    mechanical = [{"status": gates.PASS}]
    ok = gates.grade_critical(Critical("p", "p", "permission", checked=50))
    assert gates.verdict(mechanical, passing, [ok]) == "PASS"
    temporal = gates.grade_critical(Critical("t", "t", "temporal", failures=["as-of q1 answered stale"], checked=99))
    assert gates.verdict(mechanical, passing, [ok, temporal]) == "FAIL"
    assert gates.verdict([{"status": gates.NOT_RUN}], passing, [ok]) == "NOT GRADEABLE"


def test_mechanical_gates_read_junit_and_name_existing_tests(tmp_path):
    gate = gates.MECHANICAL[2]
    assert gates.grade_mechanical(gate, None)["status"] == gates.NOT_RUN
    cases = "".join(f'<testcase classname="{n.split("::")[0][:-3].replace("/", ".")}" name="{n.split("::")[1]}"/>'
                    for n in gate.tests)
    (tmp_path / "ok.xml").write_text(f"<testsuites><testsuite>{cases}</testsuite></testsuites>")
    assert gates.grade_mechanical(gate, gates.junit_outcomes(tmp_path / "ok.xml"))["status"] == gates.PASS
    first = gate.tests[0].split("::")
    failed = cases.replace(f'name="{first[1]}"/>', f'name="{first[1]}"><failure message="x"/></testcase>', 1)
    (tmp_path / "bad.xml").write_text(f"<testsuites><testsuite>{failed}</testsuite></testsuites>")
    assert gates.grade_mechanical(gate, gates.junit_outcomes(tmp_path / "bad.xml"))["status"] == gates.FAIL
    assert gates.grade_mechanical(gate, gates.junit_outcomes(tmp_path / "ok.xml"),
                                  [{"name": "probe", "ok": False}])["status"] == gates.FAIL
    for g in gates.MECHANICAL:  # every listed test exists, so a rename cannot silently drop coverage
        for node in g.tests:
            path, name = node.split("::")
            tree = ast.parse((ROOT / path).read_text())
            assert name in {f.name for f in ast.walk(tree) if isinstance(f, ast.FunctionDef)}, node


# ---------------------------------------------------------------- question sets


@pytest.mark.parametrize("name", runner.HISTORIES)
def test_question_sets_resolve_and_are_marked_unverified(name):
    history = runner.load_history(runner.FIXTURES / name)
    questions = compare.load_questions(compare.question_file(runner.FIXTURES / name), history)
    assert len(questions) >= 35 and len({q["id"] for q in questions}) == len(questions)
    assert all(q["verified"] is False for q in questions)
    assert all(q["targets"]["expected"] or q["targets"]["forbidden"] for q in questions)
    assert all(q["valid_at"] for q in questions if q["category"] == "temporal")
    deleted = {x["id"] for x in history.labels["assertions"] if x["from_deleted_source"]}
    assert {t for q in questions for t in q["forbidden"]} <= deleted
    assert any(q["scope"]["entity"] for q in questions) and any(q["stale"] for q in questions)


def test_question_sets_total_at_least_one_hundred():
    assert sum(len(json.loads(compare.question_file(runner.FIXTURES / n).read_text())["questions"])
               for n in runner.HISTORIES) >= 100


def test_live_comparison_refuses_without_a_budget(capsys):
    assert compare.main(["--live"]) == 2 and "--budget-usd" in capsys.readouterr().err
    assert compare.main(["--live", "--budget-usd", "0"]) == 2


def test_throwaway_database_refuses_a_remote_server(monkeypatch):
    monkeypatch.setattr(get_settings(), "database_url", "postgresql+psycopg://u:p@db.example.com:5432/jarvis")
    with pytest.raises(SystemExit, match="refusing"), runner.throwaway_database():
        pass


# ---------------------------------------------------------------- the database mode, comparison and report


def verified_copy(tmp_path: Path) -> Path:
    """history_alpha with the hosting change adjudicated: its two decisions, the change, the two as-of questions
    and the comparison questions about hosting are verified; everything else stays unverified."""
    path = tmp_path / "history_alpha"
    shutil.copytree(runner.FIXTURES / "history_alpha", path)
    labels = json.loads((path / "labels.json").read_text())
    hosting = {"a01-dec-azure", "a04-dec-aws"}
    for item in labels["assertions"]:
        item["verified"] = item["id"] in hosting
    for item in labels["changes"] + labels["as_of"]:
        item["verified"] = item["id"] in ("x-alpha-hosting", "q-alpha-host-aug", "q-alpha-host-sep")
    (path / "labels.json").write_text(json.dumps(labels))
    questions = json.loads(compare.question_file(runner.FIXTURES / "history_alpha").read_text())
    for q in questions["questions"]:
        q["verified"] = bool({x for x in q["expected"] if isinstance(x, str)} & hosting)
    (tmp_path / "questions.json").write_text(json.dumps(questions))
    return path


def test_db_mode_compare_and_report_on_one_history(tmp_path):
    path = verified_copy(tmp_path)
    with runner.offline():
        result, deployment = runner.db_run(path, "arrival", maintenance_runs=3)
        comparison = compare.compare(path, deployment, tmp_path / "questions.json", samples=200)
        data = {"runs": [result.to_dict()], "compare": [comparison],
                "pooled": compare.pooled([comparison], samples=200),
                "costs": {"alpha": report.costs(deployment, comparison)},
                "probes": {"alpha": {"cross_owner": report.cross_owner_probe(runner.FIXTURES / "history_alpha",
                                                                             deployment)}}}
    p, stab, everything = result.processing, result.stability, result.scores["all"]
    assert result.mode == "db" and p["covered_segments"] == p["segments"] > 30
    assert set(p["sources"]) == {"ready", "deleted"} and p["runs"]["extract:succeeded"] == 7
    assert everything["deletion_leaks"] == [] and everything["entity_links"]["auto_applied"] > 0
    assert everything["speakers"]["labels"] == 6 and everything["fulfilment"]["measured"] is True
    assert everything["as_of"]["questions"] == 4 and everything["proposals"] is not None
    assert set(stab["supersession_errors_by_order"]) == set(runner.THREE_ORDERS)
    assert stab["state_agreement"] == 1.0 and stab["no_change_on_repeat"] is True
    assert stab["maintenance"]["unchanged"] == 6 and not any(stab["maintenance"]["created"].values())
    # Only verified labels reach the release side.
    release = result.scores["verified"]
    assert release["per_kind"]["decision"]["labels"] == 2 and release["as_of"]["questions"] == 2
    assert "commitment" not in release["per_kind"] or release["per_kind"]["commitment"]["labels"] == 0

    assert comparison["questions"] == {"total": 37, "verified": 4, "unverified_excluded": 33}
    for arm in compare.ARMS:
        arm_all = comparison["all"]["arms"][arm]
        assert arm_all["questions"] == 37 and arm_all["leaks"] == 0
        assert arm_all["citations"] and arm_all["citation_validity"] == 1.0
    assert comparison["verified"]["arms"]["raw"]["questions"] == 4
    assert set(comparison["all"]["paired"]) == {"structured_vs_raw", "combined_vs_raw", "combined_vs_structured"}
    assert data["costs"]["alpha"]["by_meeting"].keys() >= {"alpha-01", "alpha-07"}
    assert data["probes"]["alpha"]["cross_owner"]["asked"] == 74
    assert data["probes"]["alpha"]["cross_owner"]["disclosed"] == []

    graded = report.grade(data, None, [])
    temporal, permission = graded["critical"]
    # The doubles never supersede decisions, so the verified hosting change is a critical temporal failure, and it
    # fails the release however the averages look.
    assert temporal["status"] == gates.FAIL and any("supersession" in x for x in temporal["failures"])
    assert permission["status"] == gates.PASS and graded["verdict"] == "FAIL"
    assert graded["unverified_excluded"]["total"] == (result.labels["total"] - result.labels["verified"]) + 33
    link_target = next(q for q in graded["targets"] if q["key"] == "entity_link_precision")
    assert link_target["status"] == gates.NOT_GRADEABLE and link_target["unverified"]["n"] > 0
    text = report.markdown(data, graded, junit=None, series=[])
    assert f"Unverified labels excluded: {graded['unverified_excluded']['total']}" in text
    assert "Verdict: **FAIL**" in text and "NOT RUN" in text


# ---------------------------------------------------------------- adjudication worksheet


def fill(directory: Path) -> dict:
    """Adjudicate the exported sheet as a person would: every engine item judged, one statement corrected, one
    hypothetical marked, one missing fact added, one question kept."""
    with (directory / "worksheet.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    marked = {"changed": None, "must_not": None}
    for row in rows:
        section = row["section"]
        if section == "meeting":
            row["minutes"] = "6"
        elif section == "speaker":
            row["verdict"] = "unknown" if row["engine"].startswith("unresolved") else "correct"
        elif section == "assertion":
            row["verdict"] = "correct"
            if "SMS reminders" in row["quote"]:
                row["verdict"], row["notes"] = "must_not", "hypothetical"
                marked["must_not"] = row["ref"]
            elif row["kind"] == "decision" and marked["changed"] is None:
                row["verdict"], row["correction"] = "changed", "Host the portal on Azure."
                marked["changed"] = row["ref"]
        elif section == "link":
            row["verdict"] = "correct" if row["engine"].startswith(("exact", "inferred")) else "skip"
    rows.append({**dict.fromkeys(worksheet.COLUMNS, ""), "section": "assertion", "meeting": "alpha-01",
                 "kind": "fact", "statement": "The monthly retainer is 18,500 dollars.",
                 "quote": "the monthly retainer is 18,500 dollars", "verdict": "missing"})
    question = next(r for r in rows if r["section"] == "question")
    question.update(verdict="keep", statement="Which cloud hosts the portal?", kind="decision",
                    expected=f"A:{marked['changed']} || Q:alpha-01|the monthly retainer is 18,500 dollars")
    with (directory / "worksheet.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=worksheet.COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return marked


def test_worksheet_round_trip(tmp_path):
    series = tmp_path / "series"
    with runner.offline():
        deployment = runner.deploy(runner.load_history(runner.FIXTURES / "history_alpha"), "arrival")
        exported = worksheet.export(deployment.owner, series, questions=[{"question": "Which cloud hosts it?"}],
                                    blank_questions=1)
    assert exported["meetings"] == 6  # the deleted meeting is not exported
    sheet = (series / "worksheet.csv").read_text()
    assert "alpha-06" not in sheet and "Speaker 2" in sheet and "statement of work" in sheet
    rebuilt = runner.load_history(runner.FIXTURES / "history_alpha")
    exported_meeting = json.loads((series / "meetings" / "alpha-01.json").read_text())
    assert [s["text"] for s in exported_meeting["segments"]] == [
        s.text for s in rebuilt.meeting("alpha-01").envelope.segments]

    marked = fill(series)
    counts = worksheet.import_sheet(series)
    labels = json.loads((series / "labels.json").read_text())
    assert counts["assertions"] == len(labels["assertions"]) and counts["must_not"] == 1 and counts["questions"] == 1
    assert all(x["verified"] for section in ("assertions", "must_not", "links") for x in labels[section])
    assert all(v["verified"] for mapping in labels["speakers"].values() for v in mapping.values())
    changed = next(x for x in labels["assertions"] if x["id"] == f"w-{marked['changed']}")
    assert changed["statement"] == "Host the portal on Azure." and changed["verdict"] == "changed"
    commitments = [x for x in labels["assertions"] if x["kind"] == "commitment"]
    assert commitments and all("responsible" in x and "due" in x for x in commitments)
    question = json.loads((series / "questions.json").read_text())["questions"][0]
    assert question["verified"] and question["expected"][0] == f"w-{marked['changed']}"

    # The series is a runner history with verified labels, and the comparison can load its questions.
    history = runner.load_history(series)
    scored = runner.run(series, "doubles", "chronological", shuffles=1)
    assert scored.labels["verified"] >= len(labels["assertions"]) and scored.scores["verified"]["per_kind"]
    assert compare.load_questions(series / "questions.json", history)[0]["targets"]["expected"]

    adjudicated = worksheet.score(series)
    assert adjudicated["fabricated_actions"] == 1 and adjudicated["per_kind"]["fact"]["missing"] == 1
    assert adjudicated["per_kind"]["fact"]["recall"] == 0.0 and adjudicated["overturn"]["overturned"] == 2
    assert adjudicated["correction_minutes"] == {"meetings": 6, "per_meeting": 6.0}
    assert adjudicated["links"]["correct"] > 0 and adjudicated["links"]["wrong"] == 0

    # A bad quote is refused with every problem listed, and no labels file is left behind.
    rows = list(csv.DictReader((series / "worksheet.csv").open(newline="")))
    rows[-1].update(section="assertion", verdict="missing", meeting="alpha-01", kind="fact", quote="not in the meeting")
    with (series / "worksheet.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=worksheet.COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError, match="not in the meeting"):
        worksheet.import_sheet(series)
    assert not (series / "labels.json").exists()
