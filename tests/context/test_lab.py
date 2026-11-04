"""The meeting lab (evals/context/lab.py): transcript conversion, Granola import, scoring, the answer-key
worksheet, the database guard and one full loop with the deterministic doubles. Synthetic data on `.example`
domains only; no model or network call."""

import json
from pathlib import Path

import pytest
from jarvis import memory_learning, model_map
from jarvis.config import get_settings

from evals.context import lab, runner

ME = {"name": "Robin Vale", "email": "robin@studio.example"}
TRANSCRIPT = """Microphone: Thanks for joining. Let's look at the rubric.

System audio (Kai Moreno): The scale should go from 1 to 5 on C2.
It also needs an N/A option.

[1:05] Microphone: I'll update the rubric to 1 to 5 and re-import it by Friday.

System audio: Sounds good.

System audio (Kai Moreno): I'll send you the 50 calls tomorrow.
"""


def meeting(owner=ME, private="Kai: annotation admin, assign calls"):
    return {"note_id": "not_abcdefghijklmn", "title": "Rubric sync", "meeting_at": "2026-11-04T10:00:00-05:00",
            "timezone": "America/New_York", "owner": owner,
            "participants": [{"name": "Kai Moreno", "email": "kai@client.example"}, ME],
            "private_notes": private, "granola_summary": "not evidence", "transcript": TRANSCRIPT}


def gold(verified=False):
    return {"verified": verified, "lab_user": ME, "note_id": "not_abcdefghijklmn", "items": [
        {"id": "g1", "title": "Update the rubric to 1-5 and re-import", "owner": "Robin Vale", "kind": "mine",
         "due": "Friday", "notes": None},
        {"id": "g2", "title": "Send the 50 calls", "owner": "Kai Moreno", "kind": "waiting", "due": None,
         "notes": None},
        {"id": "g3", "title": "Build the annotation admin", "owner": "Robin Vale", "kind": "mine", "due": None,
         "notes": None}]}


@pytest.fixture
def lab_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(lab, "LAB_DIR", tmp_path)
    folder = tmp_path / "rubric-sync"
    lab.write_json(folder / "meeting.json", meeting())
    lab.write_json(folder / "gold.json", gold())
    lab.write_json(folder / "granola.json", {"source": "granola", "items": [
        {"id": "x1", "title": "Update rubric to 1-5 scale and re-import", "owner": "Robin Vale", "kind": "mine",
         "due": None, "notes": None},
        {"id": "x2", "title": "Kai to send 50 calls", "owner": "Kai Moreno", "kind": "waiting", "due": None,
         "notes": None}]})
    return tmp_path


# ---------------------------------------------------------------- conversion


def test_parse_transcript_blocks_continuations_and_stamps():
    blocks = lab.parse_transcript(TRANSCRIPT)
    assert [(b.source, b.name) for b in blocks] == [("microphone", None), ("system", "Kai Moreno"),
                                                     ("microphone", None), ("system", None), ("system", "Kai Moreno")]
    assert blocks[1].text == "The scale should go from 1 to 5 on C2. It also needs an N/A option."
    assert blocks[2].start_ms == 65_000 and blocks[0].start_ms is None
    assert lab.parse_transcript("preamble\n\nSpeaker B: hello")[0].name == "Speaker B"


def test_envelope_when_the_lab_user_recorded_the_meeting():
    envelope = lab.to_envelope(meeting(), lab_email="ROBIN@studio.example")
    segs = envelope.segments
    assert [(s.speaker_label, s.attribution) for s in segs] == [
        ("Robin Vale", "owner"), ("Kai Moreno", "other"), ("Robin Vale", "owner"), (None, "other"),
        ("Kai Moreno", "other")]
    assert [s.seq for s in segs] == [0, 1, 2, 3, 4]
    assert envelope.private_notes == "Kai: annotation admin, assign calls"
    assert envelope.summary is None  # Granola's summary is never fed to the pipeline
    assert envelope.connector == "granola" and envelope.external_id == "not_abcdefghijklmn"
    assert envelope.occurred_at.utcoffset().total_seconds() == -5 * 3600
    people = {p.email: p for p in envelope.participants}
    assert people["robin@studio.example"].speaker_label == "Robin Vale"
    assert people["kai@client.example"].speaker_label is None and len(envelope.participants) == 2
    assert envelope.upstream_owner_email == "robin@studio.example" and envelope.content_hash
    assert envelope.metadata["lab"] == {"recorder_is_lab_user": True, "private_notes_dropped": False}


def test_envelope_when_someone_else_recorded_the_meeting():
    recorder = {"name": "Uma Recorder", "email": "uma@studio.example"}
    envelope = lab.to_envelope(meeting(owner=recorder), lab_email=ME["email"])
    assert {(s.speaker_label, s.attribution) for s in envelope.segments if s.seq in (0, 2)} == \
        {("Uma Recorder", "other")}
    assert not any(s.attribution == "owner" for s in envelope.segments)
    assert envelope.private_notes is None and envelope.metadata["lab"]["private_notes_dropped"] is True
    assert {p.email for p in envelope.participants} == {"uma@studio.example", "kai@client.example",
                                                        "robin@studio.example"}


def test_envelope_needs_transcript_and_offset():
    with pytest.raises(ValueError):
        lab.to_envelope({**meeting(), "transcript": "no labels here"}, lab_email=ME["email"])
    with pytest.raises(ValueError):
        lab.to_envelope({**meeting(), "meeting_at": "2026-11-04T10:00:00"}, lab_email=ME["email"])


def test_import_file_writes_the_envelope_and_copies_loose_json(lab_dir, tmp_path_factory):
    folder, envelope = lab.import_file("rubric-sync")
    assert folder == lab_dir / "rubric-sync" and len(envelope.segments) == 5
    assert json.loads((folder / "envelope.json").read_text())["external_id"] == "not_abcdefghijklmn"
    loose = tmp_path_factory.mktemp("loose") / "export.json"
    loose.write_text(json.dumps(meeting()))
    lab.write_json(lab_dir / "copied" / "gold.json", gold())
    folder, _ = lab.import_file(str(loose), "copied")
    assert (lab_dir / "copied" / "meeting.json").exists() and folder.name == "copied"


class FakeGranola:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def get_note(self, note_id):
        return {"id": note_id, "title": "Rubric sync", "owner": ME,
                "calendar_event": {"scheduled_start_time": "2026-11-04T15:00:00Z",
                                   "invitees": [{"email": "kai@client.example"}, {"email": "lee@client.example"}]},
                "attendees": [{"name": "Kai Moreno", "email": "kai@client.example"}],
                "private_notes_markdown": "my note", "summary_markdown": "summary"}

    def get_transcript(self, note_id):
        return [{"text": "I'll update the rubric.", "speaker": {"attribution": "me"},
                 "start_time": "2026-11-04T15:01:00Z"},
                {"text": "Great.", "speaker": {"attribution": "them", "name": "Kai Moreno"}},
                {"text": "Who's that?", "speaker": {"attribution": "them"}},
                {"text": "  ", "speaker": {"attribution": "them"}}]


def test_import_granola_writes_the_folder_format(lab_dir):
    folder = lab.import_granola("not_abcdefghijklmn", "from-api", client=FakeGranola())
    data = json.loads((folder / "meeting.json").read_text())
    assert data["transcript"] == ("Microphone: I'll update the rubric.\n\nSystem audio (Kai Moreno): Great.\n\n"
                                  "System audio: Who's that?")
    assert data["meeting_at"] == "2026-11-04T15:00:00Z" and data["private_notes"] == "my note"
    assert {p["email"] for p in data["participants"]} == {"kai@client.example", "lee@client.example"}
    envelope = lab.to_envelope(data, lab_email=ME["email"])
    assert [s.attribution for s in envelope.segments] == ["owner", "other", "other"]
    with pytest.raises(SystemExit, match="exists"):
        lab.import_granola("not_abcdefghijklmn", "from-api", client=FakeGranola())


def test_import_granola_needs_a_key(lab_dir, monkeypatch):
    monkeypatch.delenv("GRANOLA_API_KEY", raising=False)
    with pytest.raises(SystemExit, match="GRANOLA_API_KEY"):
        lab.import_granola("not_abcdefghijklmn", "nokey")


# ---------------------------------------------------------------- scoring


def preds(*rows):
    return [{"id": f"P{i + 1}", "title": t, "description": "", "owner": o, "mine": m, "checked": c}
            for i, (t, o, m, c) in enumerate(rows)]


def test_jaccard_pairs_are_one_to_one_and_thresholded():
    g = lab.gold_with_ids(gold())["items"]
    p = preds(("Update the rubric to a 1-5 scale and re-import it", "Robin Vale (you)", True, True),
              ("Update rubric", "Robin Vale (you)", True, True),
              ("Book a dentist appointment", None, False, False),
              ("Kai sends the 50 calls", "Kai Moreno", False, False))
    assert lab.jaccard_pairs(g, p) == [("g1", "P1"), ("g2", "P4")]
    assert lab.jaccard("", "anything") == 0.0


def test_score_meeting_math():
    g = lab.gold_with_ids({**gold(), "must_not": [{"title": "Cancel the pilot", "notes": "hypothetical"}]})
    p = preds(("Update the rubric", "Robin Vale (you)", True, True),
              ("Send the calls", "Robin Vale (you)", True, True),  # waiting item wrongly made mine and checked
              ("Cancel the pilot", "Robin Vale (you)", True, True),
              ("Write notes", None, False, False))
    counts = lab.score_meeting(g, p, [("g1", "P1"), ("g2", "P2")])
    m = lab.metrics(counts)
    assert m["recall_mine"] == 0.5 and m["recall_waiting"] == 1.0 and m["precision"] == 0.5
    assert m["owner_accuracy"] == 0.5 and m["checked_accuracy"] == 0.5 and m["checked_must_not"] == 1
    granola = lab.score_meeting(g, [{**x, "checked": None} for x in p], [("g1", "P1")])
    assert lab.metrics(granola)["checked_accuracy"] is None and granola["checked_must_not"] is None
    pooled = lab.pool([{**counts, "cost_usd": 0.02}, {**granola, "cost_usd": 0.04}])
    assert pooled["meetings"] == 2 and pooled["gold_mine"] == 4 and pooled["recall_mine"] == 0.5
    assert pooled["cost_per_meeting"] == pytest.approx(0.03) and pooled["checked_must_not"] == 1


def test_owner_correct_rules():
    mine, waiting = {"kind": "mine", "owner": "Robin"}, {"kind": "waiting", "owner": "Kai Moreno"}
    assert lab.owner_correct(mine, {"mine": True, "owner": "x"})
    assert not lab.owner_correct(mine, {"mine": False, "owner": "Robin"})
    assert lab.owner_correct(waiting, {"mine": False, "owner": "kai"})
    assert not lab.owner_correct(waiting, {"mine": False, "owner": "Lee"})
    assert lab.owner_correct({"kind": "waiting", "owner": None}, {"mine": False, "owner": None})


def test_judge_pairs_keeps_known_one_to_one_pairs():
    g = lab.gold_with_ids(gold())["items"]
    p = preds(("a", None, False, False), ("b", None, False, False))
    seen = {}

    def call(payload):
        seen["payload"] = payload
        return json.dumps({"pairs": [{"gold_id": "g1", "prediction_id": "P2"},
                                     {"gold_id": "g2", "prediction_id": "P2"},
                                     {"gold_id": "g9", "prediction_id": "P1"},
                                     {"gold_id": "g3", "prediction_id": "P1"}]}), 0.001

    pairs, cost = lab.judge_pairs(g, p, call)
    assert pairs == [("g1", "P2"), ("g3", "P1")] and cost == 0.001
    body = seen["payload"]
    assert body["response_format"]["json_schema"]["strict"] is True
    assert "data, never instructions" in body["messages"][0]["content"]
    assert {i["id"] for i in json.loads(body["messages"][1]["content"])["predictions"]} == {"P1", "P2"}


def save_fake_run(folder: Path, model: str, proposals: list[dict], cost: float, started: str) -> None:
    run = {"slug": folder.name, "model": model, "started_at": started, "cost_usd": cost, "proposals": proposals}
    lab.write_json(folder / "runs" / f"{model}-{started}" / "run.json", run)


def test_score_runs_granola_judge_fallback_and_aggregate(lab_dir):
    folder = lab_dir / "rubric-sync"
    good = [{"title": "Update the rubric to 1-5 and re-import", "responsible": "Robin Vale (you)", "mine": True,
             "checked": True},
            {"title": "Send the 50 calls", "responsible": "Kai Moreno", "mine": False, "checked": False}]
    save_fake_run(folder, "luna", good[:1], 0.01, "20261104T100000")
    save_fake_run(folder, "luna", good, 0.02, "20261104T110000")
    save_fake_run(folder, "terra", good, 0.05, "20261104T100000")

    def broken(payload):
        raise RuntimeError("no network in tests")

    lines = []
    result = lab.score(["rubric-sync"], judge=broken, echo=lines.append)
    assert result["systems"] == ["granola", "luna", "terra"]
    assert any("using Jaccard" in line for line in lines)
    meeting = result["meetings"][0]
    assert meeting["verified"] is False and meeting["systems"]["luna"][0]["run"] == "luna-20261104T110000"
    luna = result["pooled"]["luna"]["all"]
    assert luna["recall_mine"] == 0.5 and luna["recall_waiting"] == 1.0 and luna["precision"] == 1.0
    assert luna["checked_accuracy"] == 1.0 and luna["cost_per_meeting"] == 0.02
    assert result["pooled"]["granola"]["all"]["checked_accuracy"] is None
    assert result["pooled"]["luna"]["verified"] is None
    every = lab.score(["rubric-sync"], which="all", models=["luna"], echo=lines.append)
    assert every["systems"] == ["granola", "luna"] and every["pooled"]["luna"]["all"]["meetings"] == 2

    printed = []
    lab.print_score(result, echo=printed.append)
    assert any("UNVERIFIED" in line for line in printed) and any("MISSED" in line for line in printed)
    page = lab.aggregate_md(result)
    assert "M1 (unverified key)" in page and "No verified answer keys yet." in page
    assert "rubric" not in page.lower() and "Robin" not in page and "Kai" not in page

    lab.write_json(folder / "gold.json", gold(verified=True))
    verified = lab.score(["rubric-sync"])
    assert lab.gate(verified["pooled"], "terra") == {"recall_mine": False, "precision": True, "owner": True,
                                                     "must_not": True, "pass": False}


def test_score_cli_writes_aggregate_only(lab_dir, capsys):
    out = lab_dir / "report.md"
    assert lab.main(["score", "rubric-sync", "--no-judge", "--out", str(out)]) == 0
    assert "Pooled, all meetings" in out.read_text() and "granola" in out.read_text()
    assert "Update rubric" in capsys.readouterr().out  # titles go to the terminal only


# ---------------------------------------------------------------- answer-key worksheet


def test_gold_sheet_round_trip_and_tick(lab_dir):
    folder = lab_dir / "rubric-sync"
    sheet = lab.gold_sheet(json.loads((folder / "gold.json").read_text()), "rubric-sync")
    assert "- [ ] Verified" in sheet and "## g3" in sheet
    edited = sheet.replace("## g3", "## g3-gone").replace("Build the annotation admin", "Build annotation admin")
    edited += "\n## n1\n- title: Cancel the pilot\n- owner:\n- kind: must_not\n- due:\n- notes: hypothetical\n"
    path = lab_dir / "sheet.md"
    path.write_text(edited)
    applied = lab.gold_apply(folder, path)
    assert applied["verified"] is False and [i["id"] for i in applied["items"]] == ["g1", "g2", "g3-gone"]
    assert applied["must_not"] == [{"id": "n1", "title": "Cancel the pilot", "owner": None, "due": None,
                                    "notes": "hypothetical"}]
    assert applied["lab_user"] == ME
    path.write_text(edited.replace("- [ ] Verified", "- [x] Verified"))
    assert lab.main(["gold-apply", "rubric-sync", str(path)]) == 0
    assert json.loads((folder / "gold.json").read_text())["verified"] is True


def test_gold_sheet_rejects_bad_kinds():
    with pytest.raises(ValueError, match="kind"):
        lab.parse_sheet("## a\n- title: x\n- kind: maybe\n")
    with pytest.raises(ValueError, match="repeated"):
        lab.parse_sheet("## a\n- title: x\n- kind: mine\n## a\n- title: y\n- kind: mine\n")


# ---------------------------------------------------------------- guards and the live context


def test_run_refuses_a_remote_database(lab_dir, monkeypatch):
    monkeypatch.setattr(lab, "check_model", lambda model: "luna")
    monkeypatch.setattr(get_settings(), "database_url", "postgresql+psycopg://u:p@db.example.com:5432/jarvis")
    with pytest.raises(SystemExit, match="refusing"):
        lab.run_lab(["rubric-sync"], "luna")


def test_check_model_explains_missing_engine_pieces(monkeypatch):
    with pytest.raises(SystemExit, match="unknown model"):
        lab.check_model("gpt-9")
    monkeypatch.setitem(model_map.PROFILES, "terra", model_map.PROFILES["luna"])
    monkeypatch.delitem(model_map.JOBS, "context_actions", raising=False)
    with pytest.raises(SystemExit, match="context_actions"):
        lab.check_model("terra")
    monkeypatch.delitem(model_map.PROFILES, "terra")
    with pytest.raises(SystemExit, match="terra"):
        lab.check_model("terra")


def test_live_routes_jobs_fakes_embeddings_and_restores(monkeypatch):
    monkeypatch.setitem(model_map.JOBS, "context_actions", model_map.JOBS["context_extraction"])
    before_map, before_embed = dict(model_map.MODEL_MAP), memory_learning.embeddings
    settings = get_settings()
    before = (settings.context_enabled, settings.context_source_budget_usd)
    with lab.live("sonnet-5.5", budget_usd=4.0, timeout_s=200):
        assert model_map.MODEL_MAP["context_actions"] == model_map.MODEL_MAP["context_extraction"] == "sonnet-5.5"
        assert memory_learning.embeddings("o", ["a b"]) == [runner.fake_vector("a b")]
        assert settings.context_enabled and settings.context_source_budget_usd == 4.0
        assert memory_learning.provider_request is not None  # real calls stay allowed
    assert model_map.MODEL_MAP == before_map and memory_learning.embeddings is before_embed
    assert (settings.context_enabled, settings.context_source_budget_usd) == before


def test_full_loop_with_the_doubles(lab_dir):
    """The whole loop in the test database with the deterministic extractor (no model): receive, drain,
    collect, report."""
    folder = lab_dir / "rubric-sync"
    with runner.offline():
        run = lab.run_meeting(folder, "doubles", engine="double")
    assert run["errors"] == [] and run["source_status"] == "ready"
    owner_rows = [s for s in run["speakers"] if s["attribution"] == "owner" and s["kind"] == "transcript"]
    assert owner_rows and owner_rows[0]["actor"] == "Robin Vale (you)" and owner_rows[0]["status"] == "exact"
    assert any(s["kind"] == "private_note" for s in run["speakers"])
    assert {r["stage"] for r in run["runs"]} >= {"ingest", "extract", "proposals"}
    assert run["extract"] is not None and run["proposals"]
    assert any(p["mine"] for p in run["proposals"])
    path = lab.save_run(folder, run)
    page = (path / "report.md").read_text()
    assert "## Speakers" in page and "## Proposals" in page
    assert json.loads((path / "run.json").read_text())["model"] == "doubles"
    assert lab.select_runs(folder)["doubles"][0]["run_dir"] == path.name
