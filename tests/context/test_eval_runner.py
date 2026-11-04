"""Fixture histories and the evaluation runner (evals/context/runner.py).

The fixtures are synthetic. These tests check that they validate against SourceEnvelope, that every label points
at exact text, that each history contains the required scenarios, that history.json matches the envelopes, and
that the runner scores all three histories with the doubles, deterministically and with teeth."""

import json
import shutil
from datetime import datetime, timedelta

import pytest
from jarvis.context.doubles import DoubleExtractor, DoubleReconciler
from jarvis.context.schema import ExtractionResult, ReconcileOp

from evals.context import runner
from evals.context.runner import (
    FIXTURES,
    HISTORIES,
    EngineUnavailable,
    build_export,
    load_history,
    make_engines,
    schedule,
)

PATHS = [FIXTURES / name for name in HISTORIES]


@pytest.fixture(scope="module", params=PATHS, ids=HISTORIES)
def history(request):
    return load_history(request.param)


def test_histories_are_six_to_ten_valid_meetings_over_about_sixty_days(history):
    assert 6 <= len(history.meetings) <= 10
    times = sorted(m.occurred_at for m in history.meetings)
    assert timedelta(days=45) <= times[-1] - times[0] <= timedelta(days=62)
    manifest = json.loads((history.path / "manifest.json").read_text())
    assert manifest["synthetic"] is True
    domains = {e.split("@")[1] for a in history.actors for e in a["emails"]}
    assert all(d.endswith(".example") for d in domains), domains
    for meeting in history.meetings:
        assert meeting.received_at >= meeting.occurred_at
        assert [s.seq for s in meeting.segments] == sorted({s.seq for s in meeting.segments})


def test_every_label_points_at_exact_text_and_known_ids(history):
    labels = history.labels
    by_id = {x["id"]: x for x in labels["assertions"]}
    actor_ids = {a["id"] for a in history.actors}
    assert len(by_id) == len(labels["assertions"])
    for item in labels["assertions"]:
        assert isinstance(item["verified"], bool)
        for evidence in item["evidence"]:
            runner._locate(history, evidence["meeting"], evidence["quote"])
        if item["kind"] == "commitment":
            assert item["responsible"] and set(item["responsible"]) <= actor_ids
            assert item["due"] is None or datetime.fromisoformat(item["due"])
            if item["fulfilled"]:
                runner._locate(history, item["fulfilled"]["meeting"], item["fulfilled"]["quote"])
    for item in labels["must_not"]:
        runner._locate(history, item["meeting"], item["quote"])
        assert item["reason"] in ("hypothetical", "negated")
    for change in labels["changes"]:
        assert by_id[change["from"]]["kind"] == by_id[change["to"]]["kind"]
    for question in labels["as_of"]:
        assert question["expected"] in question["family"] and set(question["family"]) <= set(by_id)
        assert datetime.fromisoformat(question["valid_at"]).tzinfo is not None
    for loop in labels["open_loops"]:
        assert all(by_id[x]["kind"] == "commitment" and not by_id[x]["from_deleted_source"] for x in loop["expected"])
    for meeting, speakers in labels["speakers"].items():
        assert set(speakers.values()) <= actor_ids
        assert set(speakers) <= {s.speaker_label for s in history.meeting(meeting).segments}


def test_each_history_contains_the_required_scenarios(history):
    s, labels = history.labels["scenarios"], history.labels
    by_id = {x["id"]: x for x in labels["assertions"]}
    change = next(c for c in labels["changes"] if c["id"] == s["decision_change"])
    assert change["key"] == "decision" and by_id[change["from"]]["meeting"] != by_id[change["to"]]["meeting"]
    fulfilled = by_id[s["fulfilled_commitment"]]
    assert fulfilled["kind"] == "commitment" and fulfilled["fulfilled"]["meeting"] != fulfilled["meeting"]

    one, two = (next(a for a in history.actors if a["id"] == x) for x in s["shared_display_name"])
    assert one["name"] == two["name"] and set(one["emails"]).isdisjoint(two["emails"])

    reused = s["reused_speaker_label"]
    people = {labels["speakers"][m][reused["label"]] for m in reused["meetings"]}
    assert len(people) == 2

    late = history.meeting(s["late_arrival"])
    assert any(m.occurred_at > late.occurred_at and m.received_at < late.received_at for m in history.meetings)

    gone = history.meeting(s["deleted_source"])
    assert gone.deleted_at and gone.deleted_at > gone.received_at
    assert {x["id"] for x in labels["assertions"] if x["meeting"] == gone.key} == {
        x["id"] for x in labels["assertions"] if x["from_deleted_source"]}

    note = by_id[s["private_note"]]
    meeting_key, seq, *_ = runner._locate(history, note["meeting"], note["evidence"][0]["quote"])
    segment = next(x for x in history.meeting(meeting_key).segments if x.seq == seq)
    assert note["segment_kind"] == segment.kind == "private_note" and segment.attribution == "owner"

    assert by_id[s["number"]]["numbers"]
    relative = by_id[s["relative_date"]]
    assert relative["due_expression"] and relative["due"]
    reasons = {x["id"]: x["reason"] for x in labels["must_not"]}
    assert (reasons[s["hypothetical"]], reasons[s["negated"]]) == ("hypothetical", "negated")


def test_history_json_is_the_canonical_export_of_the_envelopes(history):
    on_disk = json.loads((history.path / "history.json").read_text())
    assert build_export(history) == on_disk
    deleted = [s for s in on_disk["sources"] if s["status"] == "deleted"]
    assert len(deleted) == 1
    tombstone = next(r for r in on_disk["revisions"] if r["source_id"] == deleted[0]["id"])
    assert tombstone["payload"] == {} and not any(s["revision_id"] == tombstone["id"] for s in on_disk["segments"])


def test_schedules_reflect_arrival_deletion_and_seeded_shuffles(history):
    late = history.labels["scenarios"]["late_arrival"]
    gone = history.labels["scenarios"]["deleted_source"]
    arrival = [f"{e}:{m.key}" for e, m, _ in schedule(history, "arrival", 0)]
    chronological = [f"{e}:{m.key}" for e, m, _ in schedule(history, "chronological", 0)]
    assert arrival != chronological and arrival.index(f"receive:{late}") > chronological.index(f"receive:{late}")
    assert chronological[-1] == f"delete:{gone}" and arrival[-1] != f"delete:{gone}"
    shuffled = [m.key for _, m, _ in schedule(history, "shuffled", 7)]
    assert shuffled == [m.key for _, m, _ in schedule(history, "shuffled", 7)]
    assert len(shuffled) == len(history.meetings) + 1 and shuffled[-1] == gone


@pytest.mark.parametrize("path", PATHS, ids=HISTORIES)
def test_runner_reports_every_history_with_the_doubles(path, tmp_path):
    report = runner.run(path, "doubles", "chronological", 0)
    data = json.loads(json.dumps(report.to_dict()))
    assert data["labels"]["total"] > 20 and data["labels"]["verified"] == 0
    everything, release = data["scores"]["all"], data["scores"]["verified"]
    assert everything["per_kind"]["commitment"]["recall"] > 0 and everything["per_kind"]["decision"]["recall"] > 0
    assert release["per_kind"]["commitment"]["recall"] is None  # no verified labels yet
    assert everything["deletion_leaks"] == [] and data["processing"]["extraction_problems"] == []
    assert data["processing"]["covered_segments"] == data["processing"]["segments"]
    assert data["stability"]["deterministic"] and data["stability"]["no_change_on_repeat"]
    assert data["stability"]["compared_runs"] == 4
    assert {"stale_answer_rate", "results"} <= set(everything["as_of"])
    json_path, md_path = runner.write_report(report, tmp_path)
    assert json.loads(json_path.read_text())["history"] == report.history
    assert md_path.read_text().startswith(f"# Context eval: {report.history}")


def test_scoring_catches_a_hypothetical_the_double_turns_into_a_commitment():
    report = runner.run(FIXTURES / "history_alpha")
    violations = report.scores["all"]["must_not"]["violations"]
    assert [v["label"] for v in violations] == ["n-alpha-sms"]
    attribution = report.scores["all"]["commitments"]["attribution"]
    assert attribution.get("unresolved", 0) > 0  # diarized labels and the shared name stay unresolved
    assert report.scores["all"]["per_kind"]["fact"]["recall"] == 0


def test_an_engine_that_finds_nothing_scores_zero_recall():
    class Empty:
        def extract(self, request):
            return ExtractionResult(coverage={s.seq: "no_content" for s in request.segments})

    report = runner.run(FIXTURES / "history_beta", engines=(Empty(), DoubleReconciler()))
    kinds = report.scores["all"]["per_kind"]
    assert all(v["recall"] == 0 for v in kinds.values()) and report.processing["assertions"] == 0


class HostingReconciler(DoubleReconciler):
    """Test engine: a decision naming AWS supersedes the one naming Azure, whatever the arrival order."""

    def reconcile(self, request):
        ops = super().reconcile(request)
        pool = [a for a in (*request.related, *request.changed) if a.kind == "decision" and a.lifecycle == "current"]
        azure = [a for a in pool if "Azure" in a.statement and "AWS" not in a.statement]
        aws = [a for a in pool if "AWS" in a.statement]
        for old in azure:
            for new in aws:
                ops.append(ReconcileOp(op="supersede", targets=[old.id, new.id],
                                       target_revisions={old.id: old.revision, new.id: new.revision}))
        return ops


@pytest.mark.parametrize("order", ["chronological", "arrival", "shuffled"])
def test_supersession_is_scored_and_converges_across_orders(order):
    report = runner.run(FIXTURES / "history_alpha", order=order, seed=3,
                        engines=(DoubleExtractor(), HostingReconciler()))
    sup = report.scores["all"]["supersession"]
    assert sup["agreed"] == 1 and [e["change"] for e in sup["errors"]] == ["x-alpha-staging"]
    assert report.stability["supersession_agreement"] == 1.0
    question = next(r for r in report.scores["all"]["as_of"]["results"] if r["id"] == "q-alpha-host-aug")
    assert question["verdict"] == "correct"


def test_verified_labels_are_scored_separately(tmp_path):
    copy = tmp_path / "history_alpha"
    shutil.copytree(FIXTURES / "history_alpha", copy)
    labels = json.loads((copy / "labels.json").read_text())
    for item in labels["assertions"]:
        item["verified"] = item["kind"] == "decision"
    (copy / "labels.json").write_text(json.dumps(labels))
    report = runner.run(copy)
    release = report.scores["verified"]["per_kind"]
    assert report.labels["verified"] == sum(1 for x in labels["assertions"] if x["kind"] == "decision")
    assert release["decision"]["recall"] == 1.0 and release["commitment"]["recall"] is None
    assert release["commitment"]["ignored"] > 0  # matches unverified labels: neither right nor wrong yet


def test_native_engine_is_guarded():
    with pytest.raises(EngineUnavailable):
        make_engines("native", load_history(FIXTURES / "history_gamma"))
    with pytest.raises(ValueError):
        make_engines("oracle", load_history(FIXTURES / "history_gamma"))


def test_cli_builds_and_runs(tmp_path, capsys):
    copy = tmp_path / "history_gamma"
    shutil.copytree(FIXTURES / "history_gamma", copy)
    (copy / "history.json").unlink()
    assert runner.main(["build", str(copy)]) == 0
    assert (copy / "history.json").read_text() == (FIXTURES / "history_gamma" / "history.json").read_text()
    assert runner.main(["run", str(copy), "--order", "arrival", "--shuffles", "1", "--out", str(tmp_path / "out")]) == 0
    assert (tmp_path / "out" / "gamma-doubles-arrival.json").exists()
    assert "Context eval: gamma" in capsys.readouterr().out
