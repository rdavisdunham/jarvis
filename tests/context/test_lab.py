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
    assert envelope.metadata["lab"] == {"recorder_is_lab_user": True, "speakers_separated": True,
                                        "speakers_decided_by": "lab", "microphone_as": "recorder", "private_notes_status": "kept",
                                        "private_notes_dropped": False}
    without = lab.to_envelope(meeting(), lab_email=ME["email"], private_notes=False)
    assert without.private_notes is None and without.metadata["lab"]["private_notes_status"] == \
        "dropped: --no-private-notes"
    assert without.content_hash != envelope.content_hash


def test_envelope_when_someone_else_recorded_the_meeting():
    recorder = {"name": "Uma Recorder", "email": "uma@studio.example"}
    envelope = lab.to_envelope(meeting(owner=recorder), lab_email=ME["email"])
    assert {(s.speaker_label, s.attribution) for s in envelope.segments if s.seq in (0, 2)} == \
        {("Uma Recorder", "other")}
    assert not any(s.attribution == "owner" for s in envelope.segments)
    assert envelope.private_notes is None and envelope.metadata["lab"]["private_notes_dropped"] is True
    assert {p.email for p in envelope.participants} == {"uma@studio.example", "kai@client.example",
                                                        "robin@studio.example"}


def test_unseparated_speakers_stay_unknown():
    """Room or mixed audio: every block is Microphone although several people spoke."""
    mixed = {**meeting(owner={"name": "Uma Recorder", "email": "uma@studio.example"}),
             "transcript": "Microphone: Let's start.\n\nMicrophone: I'll send the calls.\n\nMicrophone: Thanks."}
    auto = lab.to_envelope(mixed, lab_email=ME["email"])
    assert {(s.speaker_label, s.attribution) for s in auto.segments} == {(None, None)}
    assert auto.metadata["lab"]["speakers_separated"] is False and auto.metadata["lab"]["microphone_as"] == "unknown"
    assert not any(p.speaker_label for p in auto.participants)
    forced = lab.to_envelope(mixed, lab_email=ME["email"], speakers="recorder")
    assert {(s.speaker_label, s.attribution) for s in forced.segments} == {("Uma Recorder", "other")}
    # One person alone, or both sources present: the recorder as before, unless forced unknown.
    solo = {**mixed, "participants": []}
    assert {s.speaker_label for s in lab.to_envelope(solo, lab_email=ME["email"]).segments} == {"Uma Recorder"}
    split = lab.to_envelope(meeting(), lab_email=ME["email"], speakers="unknown")
    assert [(s.speaker_label, s.attribution) for s in split.segments][:2] == [(None, None), ("Kai Moreno", "other")]
    with pytest.raises(ValueError, match="speakers"):
        lab.to_envelope(meeting(), lab_email=ME["email"], speakers="everyone")


def test_import_file_reports_unseparated_speakers(lab_dir, capsys):
    folder = lab_dir / "rubric-sync"
    lab.write_json(folder / "meeting.json", {**meeting(), "transcript": "Microphone: a\n\nMicrophone: b"})
    assert lab.main(["import-file", "rubric-sync"]) == 0
    assert "speakers not separated" in capsys.readouterr().out
    assert lab.main(["import-file", "rubric-sync", "--speakers", "recorder"]) == 0
    assert "speakers not separated" in capsys.readouterr().out  # still reported, mapped to the recorder


def test_envelope_needs_transcript_and_offset():
    with pytest.raises(ValueError):
        lab.to_envelope({**meeting(), "transcript": "no labels here"}, lab_email=ME["email"])
    with pytest.raises(ValueError):
        lab.to_envelope({**meeting(), "meeting_at": "2026-11-04T10:00:00"}, lab_email=ME["email"])


def test_import_file_writes_the_envelope_and_copies_loose_json(lab_dir, tmp_path_factory):
    folder, envelope, built = lab.import_file("rubric-sync")
    assert folder == lab_dir / "rubric-sync" and len(envelope.segments) == 5 and built == "text"
    assert json.loads((folder / "envelope.json").read_text())["external_id"] == "not_abcdefghijklmn"
    loose = tmp_path_factory.mktemp("loose") / "export.json"
    loose.write_text(json.dumps(meeting()))
    lab.write_json(lab_dir / "copied" / "gold.json", gold())
    folder, _, _ = lab.import_file(str(loose), "copied")
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


def match(*pairs):
    return [{"gold": g, "pred": p, "parts": list(parts)} for g, p, *parts in pairs]


def test_score_meeting_math():
    g = lab.gold_with_ids({**gold(), "must_not": [{"title": "Cancel the pilot", "notes": "hypothetical"}]})
    p = preds(("Update the rubric", "Robin Vale (you)", True, True),
              ("Send the calls", "Robin Vale (you)", True, True),  # waiting item wrongly made mine and checked
              ("Cancel the pilot", "Robin Vale (you)", True, True),
              ("Write notes", None, False, False),
              ("Build annotation admin", "Kai Moreno", False, False))  # the user's todo given to Kai
    counts = lab.score_meeting(g, p, match(("g1", "P1"), ("g2", "P2"), ("g3", "P5")))
    m = lab.metrics(counts)
    assert m["recall_mine"] == 0.5 and m["candidate_recall_mine"] == 1.0  # g3 matched, wrong owner
    assert m["recall_waiting"] == 0.0 and m["candidate_recall_waiting"] == 1.0  # g2 matched as the user's
    assert m["candidate_recall"] == 1.0 and m["precision"] == 0.6
    assert m["owner_accuracy"] == pytest.approx(1 / 3) and m["checked_accuracy"] == pytest.approx(1 / 3)
    # Three checked proposals: one is the user's todo, one someone else's, one matches nothing.
    assert m["checked_precision"] == pytest.approx(1 / 3) and m["unmatched_checked"] == 1
    assert m["checked_must_not"] == 1 and m["parts_recall"] == 1.0 and m["full_recall"] == 1.0
    granola = lab.score_meeting(g, [{**x, "checked": None} for x in p], match(("g1", "P1")))
    gm = lab.metrics(granola)
    assert gm["checked_accuracy"] is None and gm["checked_precision"] is None
    assert granola["checked_must_not"] is None and granola["unmatched_checked"] is None
    pooled = lab.pool([{**counts, "cost_usd": 0.02}, {**granola, "cost_usd": 0.04}])
    assert pooled["meetings"] == 2 and pooled["gold_mine"] == 4 and pooled["recall_mine"] == 0.5
    assert pooled["candidate_recall_mine"] == 0.75 and pooled["unmatched_checked"] == 1
    assert pooled["cost_per_meeting"] == pytest.approx(0.03) and pooled["checked_must_not"] == 1


def test_grouped_items_match_on_parts():
    grouped = {"id": "g1", "title": "Rubric rework", "owner": "Robin Vale", "kind": "mine",
               "parts": ["Change the C2 scale to 1-5", "Add an N/A option", "Re-import the rubric to TI"]}
    g = lab.gold_with_ids({"items": [grouped, gold()["items"][1]]})
    p = preds(("Change the C2 scale to 1-5", "Robin Vale (you)", True, True),
              ("Add an N/A option to the rubric", "Robin Vale (you)", True, True),
              ("Change C2 scale to 1-5 now", "Robin Vale (you)", True, True))  # a duplicate of part 1
    matches = lab.jaccard_matches(g["items"], p)
    assert [(x["gold"], x["pred"], x["parts"]) for x in matches] == [("g1", "P1", [0]), ("g1", "P2", [1])]
    m = lab.metrics(lab.score_meeting(g, p, matches))
    assert m["candidate_recall"] == 0.5 and m["full_recall"] == 0.0 and m["parts_recall"] == 0.5  # 2 of 4 parts
    assert m["precision"] == pytest.approx(2 / 3) and m["unmatched_checked"] == 1
    whole = lab.accept(g["items"], p, [("g1", "P3", [])])
    assert whole == [{"gold": "g1", "pred": "P3", "parts": [0, 1, 2]}]
    assert lab.metrics(lab.score_meeting(g, p, whole))["full_recall"] == 0.5


def test_owner_correct_rules():
    mine, waiting = {"kind": "mine", "owner": "Robin"}, {"kind": "waiting", "owner": "Kai Moreno"}
    assert lab.owner_correct(mine, {"mine": True, "owner": "x"})
    assert not lab.owner_correct(mine, {"mine": False, "owner": "Robin"})
    assert lab.owner_correct(waiting, {"mine": False, "owner": "kai"})
    assert not lab.owner_correct(waiting, {"mine": False, "owner": "Lee"})
    assert lab.owner_correct({"kind": "waiting", "owner": None}, {"mine": False, "owner": None})
    people = [{"name": "Kai Moreno", "email": "kai@client.example"}, {"name": "Kai Lund", "email": "kl@x.example"},
              {"name": "Lee Park", "email": None}]
    # Emails decide when both are known, directly or through a full name among the participants.
    assert lab.same_person(("K. Moreno", "KAI@client.example"), ("Kai", "kai@client.example"), people)
    assert not lab.same_person(("Kai Moreno", None), ("Someone", "kl@x.example"), people)
    # Full names match; a shared first name does not.
    assert lab.same_person(("Lee  Park", None), ("lee park (you)", None), people)
    assert not lab.owner_correct(waiting, {"mine": False, "owner": "Kai"}, people)
    assert lab.same_person(("Lee Park", None), ("Lee", None), people)  # unique first name


def test_judge_matches_keep_known_pairs_and_parts():
    items = gold()["items"]
    items[0] = {**items[0], "parts": ["Change the scale", "Re-import"]}
    g = lab.gold_with_ids({"items": items})["items"]
    p = preds(("a", None, False, False), ("b", None, False, False), ("c", None, False, False),
              ("d", None, False, False))
    seen = {}

    def call(payload):
        seen["payload"] = payload
        return json.dumps({"matches": [
            {"gold_id": "g2", "prediction_id": "P2", "parts": []},
            {"gold_id": "g2", "prediction_id": "P4", "parts": []},  # an ungrouped item takes one prediction
            {"gold_id": "g3", "prediction_id": "P2", "parts": []},  # a prediction matches one item
            {"gold_id": "g9", "prediction_id": "P1", "parts": []},
            {"gold_id": "g1", "prediction_id": "P1", "parts": [2]},
            {"gold_id": "g1", "prediction_id": "P3", "parts": [1, 2]}]}), 0.001

    matches, banned, cost = lab.judge_matches(g, p, call)
    assert banned == []
    assert matches == [{"gold": "g2", "pred": "P2", "parts": []}, {"gold": "g1", "pred": "P1", "parts": [1]},
                       {"gold": "g1", "pred": "P3", "parts": [0, 1]}] and cost == 0.001
    key = json.loads(seen["payload"]["messages"][1]["content"])["answer_key"]
    assert key[0]["parts"] == [{"number": 1, "text": "Change the scale"}, {"number": 2, "text": "Re-import"}]
    assert "parts" not in key[1]
    body = seen["payload"]
    assert body["response_format"]["json_schema"]["strict"] is True
    assert "data, never instructions" in body["messages"][0]["content"]
    assert {i["id"] for i in json.loads(body["messages"][1]["content"])["predictions"]} == {"P1", "P2", "P3", "P4"}


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
    with pytest.raises(SystemExit, match="no 'gpt-9' profile"):
        lab.check_model("gpt-9")
    monkeypatch.delitem(model_map.JOBS, "context_actions", raising=False)
    with pytest.raises(SystemExit, match="context_actions"):
        lab.check_model("luna")


def test_model_accepts_any_profile_request_profile(monkeypatch):
    served = lab.routable_profiles()
    assert {"luna", "sonnet-5.5"} <= set(served) and "luna-fast" not in served  # Fast tier: not profile_request
    for pid in served:
        assert lab.check_model(pid) == pid
    assert lab.check_model("sonnet") == "sonnet-5.5"
    with pytest.raises(SystemExit, match="not served by profile_request"):
        lab.check_model("deepseek")  # a placeholder without a price
    high = next((p for p in ("sonnet-5.5-high", "terra-high") if p in served), None)
    if high:
        assert lab.model_label(lab.check_model(high), "luna") == f"{high}+luna"
    seen = {}
    monkeypatch.setattr(lab, "run_lab", lambda slugs, model, **kw: seen.update(model=model))
    assert lab.main(["run", "a", "--model", "luna-none"]) == 0 and seen["model"] == "luna-none"


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


def test_a_colleagues_recording_never_resolves_to_the_user(lab_dir):
    """Recorded by someone else: the recorder's lines resolve to the recorder's seeded actor, never to the lab
    user, and nothing of theirs becomes the user's checked work."""
    folder = lab_dir / "rubric-sync"
    lab.write_json(folder / "meeting.json", meeting(owner={"name": "Uma Recorder", "email": "uma@studio.example"}))
    with runner.offline():
        run = lab.run_meeting(folder, "doubles", engine="double")
    assert run["errors"] == [] and run["meeting"]["recorder_is_lab_user"] is False
    rows = {s["label"]: s for s in run["speakers"] if s["kind"] == "transcript"}
    assert rows["Uma Recorder"]["attribution"] == "other"
    assert rows["Uma Recorder"]["actor"] == "Uma Recorder" and rows["Uma Recorder"]["status"] == "exact"
    assert not any(s["actor"] == "Robin Vale (you)" for s in run["speakers"])
    assert not any(p["mine"] or p["checked"] for p in run["proposals"])


# ---------------------------------------------------------------- facts model and owner timezone


def test_facts_model_routes_separately_and_labels_runs(monkeypatch):
    monkeypatch.setitem(model_map.JOBS, "context_actions", model_map.JOBS["context_extraction"])
    with lab.live("sonnet-5.5", facts_profile="luna"):
        assert model_map.MODEL_MAP["context_actions"] == "sonnet-5.5"
        assert model_map.MODEL_MAP["context_extraction"] == "luna"
    with lab.live("sonnet-5.5"):
        assert model_map.MODEL_MAP["context_extraction"] == "sonnet-5.5"
    assert lab.model_label("terra", "luna") == "terra+luna"
    assert lab.model_label("terra", "terra") == lab.model_label("terra", None) == "terra"
    assert lab.check_profile("sonnet") == "sonnet-5.5" and lab.check_profile("luna-none") == "luna-none"
    with pytest.raises(SystemExit, match="no 'gpt-9' profile"):
        lab.check_profile("gpt-9")


def test_score_rows_use_the_model_pair_label(lab_dir):
    folder = lab_dir / "rubric-sync"
    save_fake_run(folder, "terra+luna", [{"title": "Send the 50 calls", "responsible": "Kai Moreno",
                                          "mine": False, "checked": False}], 0.03, "20261104T100000")
    result = lab.score(["rubric-sync"], models=["terra+luna"])
    assert result["systems"] == ["granola", "terra+luna"]


def test_timezone_falls_back_to_the_lab_user(lab_dir):
    no_zone = {**meeting(), "timezone": None}
    assert lab.to_envelope(no_zone, lab_email=ME["email"]).timezone is None
    assert lab.to_envelope(no_zone, lab_email=ME["email"], default_timezone="America/Chicago").timezone == \
        "America/Chicago"
    assert lab.to_envelope(meeting(), lab_email=ME["email"], default_timezone="UTC").timezone == "America/New_York"
    assert lab.lab_timezone(ME) == "America/Chicago"
    assert lab.lab_timezone({**ME, "timezone": "Europe/Berlin"}) == "Europe/Berlin"


def test_seeded_owner_has_the_timezone_preference():
    from jarvis.db import session_scope
    from jarvis.domain import preferences

    owner = "lab-tz-test"
    lab.seed_owner(owner, ME, timezone="Europe/Berlin", others=[{"name": "Kai Moreno",
                                                                  "email": "kai@client.example"}])
    with session_scope() as db:
        assert preferences(db, owner)["timezone"] == "Europe/Berlin"


def test_full_loop_records_models_and_timezones(lab_dir):
    folder = lab_dir / "rubric-sync"
    lab.write_json(folder / "meeting.json", {**meeting(), "timezone": None})
    lab.write_json(folder / "gold.json", {**gold(), "lab_user": {**ME, "timezone": "America/Denver"}})
    with runner.offline():
        run = lab.run_meeting(folder, "terra+luna", engine="double", profile="terra", facts_profile="luna")
    assert run["actions_model"] == "terra" and run["facts_model"] == "luna" and run["model"] == "terra+luna"
    assert run["owner_timezone"] == run["envelope_timezone"] == "America/Denver"
    assert "Actions model terra, facts model luna" in lab.report_md(run)


# ---------------------------------------------------------------- the production connector path


def raw_note(owner=ME, private="my note"):
    return {"id": "not_abcdefghijklmn", "title": "Rubric sync", "owner": owner, "created_at": "2026-11-04T14:50:00Z",
            "updated_at": "2026-11-04T16:00:00Z",
            "calendar_event": {"scheduled_start_time": "2026-11-04T15:00:00Z",
                               "scheduled_end_time": "2026-11-04T15:30:00Z",
                               "invitees": [{"email": "kai@client.example"}]},
            "attendees": [{"name": "Kai Moreno", "email": "kai@client.example"}, ME],
            "private_notes_markdown": private, "summary_markdown": "Granola's own summary"}


def raw_item(text, attribution, name=None, second=0):
    speaker = {"source": "microphone" if attribution == "me" else "speaker", "attribution": attribution}
    if name:
        speaker["name"] = name
    return {"text": text, "speaker": speaker, "start_time": f"2026-11-04T15:00:{second:02d}Z",
            "end_time": f"2026-11-04T15:00:{second + 1:02d}Z"}


SPLIT = [raw_item("Let's look at the rubric.", "me", second=1),
         raw_item("I'll send you the 50 calls tomorrow.", "them", "Kai Moreno", 3),
         raw_item("I'll update the rubric by Friday.", "me", second=5)]
MIXED = [raw_item("Let's look at the rubric.", "me", second=1),
         raw_item("I'll send you the 50 calls tomorrow.", "me", second=3)]


def raw_folder(lab_dir, slug, note, transcript):
    folder = lab_dir / slug
    lab.write_json(folder / "gold.json", gold())
    lab.write_json(folder / lab.RAW_NOTE, note)
    lab.write_json(folder / lab.RAW_TRANSCRIPT, transcript)
    return folder


def test_raw_folders_go_through_the_production_connector(lab_dir):
    from jarvis.context.connectors import granola

    folder = raw_folder(lab_dir, "raw-split", raw_note(), SPLIT)
    envelope, built = lab.build_envelope(folder, ME)
    production = granola.to_envelope(raw_note(), SPLIT)
    assert built == "connector" and envelope.external_revision == production.external_revision
    assert [(s.speaker_label, s.attribution, s.text, s.start_ms) for s in envelope.segments] == \
        [(s.speaker_label, s.attribution, s.text, s.start_ms) for s in production.segments]
    assert envelope.metadata["granola"] == production.metadata["granola"]
    assert envelope.summary == production.summary  # whatever production passes on, the lab passes on
    assert envelope.private_notes == "my note" and envelope.timezone == "America/Chicago"
    assert envelope.metadata["lab"]["speakers_separated"] is True
    nonotes, _ = lab.build_envelope(folder, ME, private_notes=False)
    assert nonotes.private_notes is None


def test_connector_path_applies_the_lab_rules_after_the_connector(lab_dir):
    recorder = {"name": "Uma Recorder", "email": "uma@studio.example"}
    colleague = raw_folder(lab_dir, "raw-colleague", raw_note(owner=recorder), SPLIT)
    envelope, _ = lab.build_envelope(colleague, ME)
    assert {(s.speaker_label, s.attribution) for s in envelope.segments if s.seq in (0, 2)} == \
        {("Uma Recorder", "other")}
    assert envelope.private_notes is None
    assert envelope.metadata["lab"]["private_notes_status"] == "dropped: recorded by someone else"
    mixed = raw_folder(lab_dir, "raw-mixed", raw_note(owner=recorder), MIXED)
    auto, _ = lab.build_envelope(mixed, ME)
    assert {(s.speaker_label, s.attribution) for s in auto.segments} == {(None, None)}
    # The connector decided (more than two people, one audio source, no names): the lab defers to it.
    assert auto.metadata["granola"]["speakers_separated"] is False
    assert auto.metadata["lab"]["speakers_separated"] is False
    assert auto.metadata["lab"]["speakers_decided_by"] == "connector"
    assert auto.metadata["lab"]["microphone_as"] == "unknown"
    assert not any(p.speaker_label for p in auto.participants)
    forced, _ = lab.build_envelope(mixed, ME, speakers="recorder")
    assert {(s.speaker_label, s.attribution) for s in forced.segments} == {("Uma Recorder", "other")}
    assert forced.metadata["lab"]["microphone_as"] == "recorder"
    assert {p.email: p.speaker_label for p in forced.participants}["uma@studio.example"] == "Uma Recorder"
    # The lab user's own unseparated recording, forced to the recorder, is the owner speaking.
    own, _ = lab.build_envelope(raw_folder(lab_dir, "raw-own-mixed", raw_note(), MIXED), ME, speakers="recorder")
    assert {(s.speaker_label, s.attribution) for s in own.segments} == {("Robin Vale", "owner")}
    # The connector's call stands when it finds the speakers separated, even with "me" lines only (two people).
    pair = {**raw_note(owner=recorder), "attendees": [{"name": "Kai Moreno", "email": "kai@client.example"}]}
    two, _ = lab.build_envelope(raw_folder(lab_dir, "raw-pair", pair, MIXED), ME)
    assert two.metadata["lab"]["speakers_separated"] is True and two.metadata["lab"]["speakers_decided_by"] == \
        "connector"
    assert {(s.speaker_label, s.attribution) for s in two.segments} == {("Uma Recorder", "other")}
    forced_unknown, _ = lab.build_envelope(colleague, ME, speakers="unknown")
    assert [(s.speaker_label, s.attribution) for s in forced_unknown.segments] == \
        [(None, None), ("Kai Moreno", "other"), (None, None)]


def test_import_granola_saves_the_raw_responses(lab_dir):
    folder = lab.import_granola("not_abcdefghijklmn", "from-raw", client=FakeGranola())
    assert json.loads((folder / lab.RAW_NOTE).read_text()) == FakeGranola().get_note("not_abcdefghijklmn")
    assert json.loads((folder / lab.RAW_TRANSCRIPT).read_text()) == FakeGranola().get_transcript("x")
    lab.write_json(folder / "gold.json", gold())
    _, envelope, built = lab.import_file("from-raw")
    assert built == "connector" and len(envelope.segments) == 3


def test_full_loop_on_the_connector_path_without_notes(lab_dir):
    folder = raw_folder(lab_dir, "raw-loop", raw_note(), SPLIT)
    with runner.offline():
        run = lab.run_meeting(folder, lab.model_label("terra", "luna", False), engine="double",
                              private_notes=False)
    assert run["envelope_path"] == "connector" and run["private_notes_enabled"] is False
    assert run["model"] == "terra+luna-nonotes" and run["meeting"]["private_notes"] is False
    assert not any(s["kind"] == "private_note" for s in run["speakers"])
    kai = [p for p in run["proposals"] if p["responsible"] == "Kai Moreno"]
    assert kai and kai[0]["responsible_email"] == "kai@client.example" and not kai[0]["checked"]
    assert "envelope from connector" in lab.report_md(run)


def test_run_cli_passes_notes_and_facts_options(monkeypatch):
    seen = {}
    monkeypatch.setattr(lab, "run_lab", lambda slugs, model, **kw: seen.update(slugs=slugs, model=model, **kw))
    assert lab.main(["run", "a", "b", "--model", "terra", "--facts-model", "luna", "--no-private-notes"]) == 0
    assert seen["slugs"] == ["a", "b"] and seen["facts_model"] == "luna" and seen["private_notes"] is False


def test_gold_sheet_round_trips_parts_and_owner_email(lab_dir):
    folder = lab_dir / "rubric-sync"
    key = gold()
    key["items"][0] = {**key["items"][0], "parts": ["Change C2 to 1-5", "Add N/A"], "owner_email": ME["email"]}
    lab.write_json(folder / "gold.json", key)
    sheet = lab.gold_sheet(key, "rubric-sync")
    assert "- parts:\n  - Change C2 to 1-5\n  - Add N/A" in sheet
    path = lab_dir / "sheet.md"
    path.write_text(sheet.replace("  - Add N/A", "  - Add N/A\n  - Re-import to TI"))
    applied = lab.gold_apply(folder, path)
    assert applied["items"][0]["parts"] == ["Change C2 to 1-5", "Add N/A", "Re-import to TI"]
    assert applied["items"][0]["owner_email"] == ME["email"] and "parts" not in applied["items"][1]
    assert lab.parse_sheet("## a\n- title: x\n- kind: mine\n- parts: inline part\n")[1][0]["parts"] == \
        ["inline part"]


def test_recall_waiting_needs_the_owner_unless_the_key_has_none():
    items = [{"id": "w1", "title": "Send the 50 calls", "owner": "Kai Moreno", "kind": "waiting"},
             {"id": "w2", "title": "Book the venue", "owner": None, "kind": "waiting"}]
    g = {"items": items}
    p = preds(("Send the calls", "Lee Park", False, False), ("Book the venue", "Robin Vale (you)", True, True))
    m = lab.metrics(lab.score_meeting(g, p, match(("w1", "P1"), ("w2", "P2"))))
    assert m["candidate_recall_waiting"] == 1.0 and m["recall_waiting"] == 0.5  # w1 wrong owner; w2 on title
    assert m["owner_accuracy"] == 0.0  # owner accuracy still counts the user taking an unowned item
    right = preds(("Send the calls", "Kai", False, False))
    assert lab.metrics(lab.score_meeting(g, right, match(("w1", "P1"))))["recall_waiting"] == 0.5


# ---------------------------------------------------------------- inferred owners and new rejection codes


def test_inferred_owners_count_as_owners_and_are_right_unchecked():
    g = lab.gold_with_ids(gold())
    p = [{**x, "inferred": inferred} for x, inferred in zip(
        preds(("Update the rubric to 1-5", "Robin Vale (you)", True, False),  # inferred mine, unchecked: right
              ("Send the 50 calls", "Kai Moreno", False, False),  # inferred Kai, unchecked
              ("Build annotation admin", "Robin Vale (you)", True, True)),  # proven mine, checked
        (True, True, False), strict=True)]
    m = lab.metrics(lab.score_meeting(g, p, match(("g1", "P1"), ("g2", "P2"), ("g3", "P3"))))
    assert m["recall_mine"] == 1.0 and m["recall_waiting"] == 1.0 and m["owner_accuracy"] == 1.0
    assert m["checked_accuracy"] == 1.0 and m["inferred_owner_share"] == pytest.approx(2 / 3)
    checked_inferred = [{**p[0], "checked": True}, *p[1:]]
    wrong = lab.metrics(lab.score_meeting(g, checked_inferred, match(("g1", "P1"), ("g2", "P2"), ("g3", "P3"))))
    assert wrong["checked_accuracy"] == pytest.approx(2 / 3)
    assert "inferred owner" in "\n".join(lab.table([("x", lab.pool([{**lab.score_meeting(g, p, match(
        ("g1", "P1"))), "cost_usd": None}]))]))


def test_inferred_owner_detection_from_proposals():
    assert lab.inferred_owner({"reason": "Likely yours, inferred from the discussion. Check it if so."})
    assert lab.inferred_owner({"reason": "Likely Kai Moreno, inferred from the discussion."})
    assert lab.inferred_owner({"owner_basis": "inferred", "reason": ""})
    assert not lab.inferred_owner({"reason": "Waiting on Kai Moreno. Track it only if you need to."})
    run = {"proposals": [{"title": "t", "responsible": "Kai", "mine": False, "checked": False,
                          "reason": "Likely Kai, inferred from the discussion."}]}
    assert lab.predictions_from_run(run)[0]["inferred"] is True
    assert lab.predictions_from_items([{"title": "t", "kind": "mine"}])[0]["inferred"] is False


def test_report_renders_progress_report_rejections():
    run = {"slug": "m", "model": "terra", "started_at": "2026-11-04T10:00:00+00:00", "wall_ms": 1000,
           "cost_usd": 0.01, "source_status": "ready", "errors": [], "speakers": [], "runs": [], "assertions": [],
           "meeting": {"segments": 3, "private_notes": False},
           "extract": {"diagnostics": {"model": "x", "actions": {"model": "terra", "items": [], "rejected": [
               {"code": "progress_report", "title": "Shared the dashboard status", "reason": "a status update",
                "segments": [4]},
               {"code": "not_a_follow_up", "title": "Lunch", "reason": "no", "segments": [5]}]}},
               "candidates": [], "actions": [], "refused": [], "admitted": 0},
           "proposals": [{"title": "Update the rubric", "responsible": "Robin Vale (you)", "checked": False,
                          "due": None, "reason": "Likely yours, inferred from the discussion. Check it if so."}]}
    page = lab.report_md(run)
    assert "Rejected (not_a_follow_up 1, progress_report 1):" in page
    assert "- `progress_report` Shared the dashboard status: a status update (segments [4])" in page
    assert "Likely yours" in page


# ---------------------------------------------------------------- lab user aliases

ALIASED = {**ME, "aliases": ["Robin", "R. Vale", "Robin A. Vale"]}


def test_aliases_are_the_user_but_a_bare_other_first_name_is_not():
    assert lab.user_names(ALIASED) == ["Robin Vale", "Robin", "R. Vale", "Robin A. Vale"]
    assert lab.is_user("r. vale", None, ALIASED) and lab.is_user("Robin Vale (you)", None, ALIASED)
    assert lab.is_user(None, "ROBIN@studio.example", ALIASED)
    assert not lab.is_user("Vale", None, ALIASED) and not lab.is_user("Robin", None, ME)  # not listed
    people = [ALIASED, {"name": "Robin Marsh", "email": "rm@client.example"}]
    mine = {"kind": "mine", "owner": "Robin Vale"}
    # A proposal naming the user by alias (a mention, no actor) is the user's.
    assert lab.owner_correct(mine, {"mine": False, "owner": "R. Vale"}, people, ALIASED)
    assert not lab.owner_correct(mine, {"mine": False, "owner": "Robin Marsh"}, people, ALIASED)
    # Someone else's item given to an alias of the user is wrong; the other Robert-style namesake is not the user.
    marsh = {"kind": "waiting", "owner": "Robin Marsh"}
    assert not lab.owner_correct(marsh, {"mine": False, "owner": "Robin A. Vale"}, people, ALIASED)
    assert lab.owner_correct(marsh, {"mine": False, "owner": "robin marsh"}, people, ALIASED)
    assert lab.same_person(("R. Vale", None), ("Robin A. Vale", None), people, ALIASED)
    assert not lab.same_person(("Robin Marsh", None), ("Robin Vale", None), [], ALIASED)


def test_bare_first_name_is_not_the_user_without_an_alias():
    user = {**ME, "aliases": ["R. Vale"]}  # "Robin" alone is not an alias: another Robin works there
    mine = {"kind": "mine", "owner": "Robin Vale"}
    assert not lab.owner_correct(mine, {"mine": False, "owner": "Robin"}, [], user)
    assert lab.owner_correct(mine, {"mine": False, "owner": "R. Vale"}, [], user)
    assert not lab.same_person(("Robin", None), ("Robin Vale", None), [], user)
    g = {"lab_user": user, "items": [{"id": "g1", "title": "Update the rubric", "owner": "Robin Vale",
                                      "kind": "mine"}]}
    p = preds(("Update the rubric", "R. Vale", False, False), ("Update rubric again", "Robin", False, False))
    m = lab.metrics(lab.score_meeting(g, p, match(("g1", "P1"))))
    assert m["recall_mine"] == 1.0 and m["owner_accuracy"] == 1.0
    m = lab.metrics(lab.score_meeting(g, p, match(("g1", "P2"))))
    assert m["recall_mine"] == 0.0 and m["candidate_recall_mine"] == 1.0


def test_seeded_owner_carries_every_alias():
    from jarvis.context import identity
    from jarvis.db import session_scope

    owner = "lab-alias-test"
    actor = lab.seed_owner(owner, ALIASED)
    with session_scope() as db:
        for alias in ("Robin Vale", "Robin", "R. Vale", "Robin A. Vale"):
            assert identity.actors_for(db, owner, "alias", alias) == [actor]
        assert identity.resolve_exact(db, owner, "email", ME["email"]) == actor
        assert identity.actors_for(db, owner, "alias", "Vale") == []


# ---------------------------------------------------------------- name spelling and must-not visibility


def test_names_compare_without_spaces_dots_or_hyphens():
    people = [{"name": "YvetteRuiz", "email": None}, {"name": "JanieceGarcia", "email": None}]
    yvette = {"kind": "waiting", "owner": "Yvette Ruiz"}
    for spelling in ("YvetteRuiz", "yvette-ruiz", "Yvette  Ruiz", "yvette.ruiz"):
        assert lab.owner_correct(yvette, {"mine": False, "owner": spelling}, people)
    assert not lab.owner_correct(yvette, {"mine": False, "owner": "JanieceGarcia"}, people)
    # The first-name rule still splits on spaces: "Yvette" alone matches "Yvette Ruiz", not "YvetteRuiz".
    assert lab.same_person(("Yvette", None), ("Yvette Ruiz", None), people)
    assert not lab.same_person(("Yvette", None), ("YvetteRuiz", None), people)
    # A compact name still finds its email among the participants.
    assert lab.same_person(("Yvette Ruiz", None), ("x", "y@client.example"),
                           [{"name": "YvetteRuiz", "email": "y@client.example"}])
    user = {**ME, "aliases": ["R. Vale"]}
    assert lab.is_user("RobinVale", None, user) and lab.is_user("r-vale", None, user)
    assert not lab.is_user("Robin", None, user)


def test_checked_must_not_items_are_printed(lab_dir):
    folder = lab_dir / "rubric-sync"
    key = {**gold(), "must_not": [{"id": "n1", "title": "Cancel the pilot", "owner": "Robin Vale",
                                   "notes": "hypothetical"}]}
    lab.write_json(folder / "gold.json", key)
    save_fake_run(folder, "terra", [
        {"title": "Cancel the whole pilot", "responsible": "Robin Vale (you)", "mine": True, "checked": True},
        {"title": "Cancel the pilot", "responsible": "Kai Moreno", "mine": False, "checked": False}],
        0.01, "20261104T100000")

    def judge(payload):
        data = json.loads(payload["messages"][1]["content"])
        assert data["must_not"] == [{"id": "n1", "title": "Cancel the pilot", "owner": "Robin Vale",
                                     "notes": "hypothetical"}]
        return json.dumps({"matches": [], "must_not_matches": [{"must_not_id": "n1", "prediction_id": "P1"},
                                                                {"must_not_id": "n1", "prediction_id": "P2"}]}), 0.0

    result = lab.score(["rubric-sync"], models=["terra"], judge=judge, echo=lambda line: None)
    run = result["meetings"][0]["systems"]["terra"][0]
    assert run["method"] == "judge" and run["checked_must_not"] == 1 and run["must_not_checked"] == [("n1", "P1")]
    lines = []
    lab.print_score(result, echo=lines.append)
    assert "   MUST-NOT CHECKED Cancel the pilot <- Cancel the whole pilot" in lines
    assert sum("MUST-NOT CHECKED" in line for line in lines) == 1  # the unchecked match stays silent


def test_must_not_fallback_needs_near_identical_titles_and_the_same_owner():
    user = {**ME, "aliases": ["R. Vale"]}
    people = [user, {"name": "Kai Moreno", "email": "kai@client.example"}]
    must_not = lab.gold_with_ids({"items": [], "must_not": [
        {"id": "n1", "title": "Fix the sticky-note app UI", "owner": "Robin Vale", "notes": "done in the meeting"},
        {"id": "n2", "title": "Revoke the trainers' central doc access", "owner": "Robin Vale", "notes": "rejected"},
        {"id": "n3", "title": "Run the data sync from the earliest state", "owner": "Robin Vale",
         "notes": "rejected: only from February onward"},
        {"id": "n4", "title": "Send Kai the full call archive", "owner": "Kai Moreno", "notes": "hypothetical"}]})
    rows = [("Share the sticky note app with the wider team", "Robin Vale (you)", True, True),
            ("Sweep central docs to remove duplicated information", "Robin Vale (you)", True, True),
            ("Run the data sync for February 2026 onward", "Robin Vale (you)", True, True),
            ("Fix the sticky note app UI", "Robin Vale (you)", True, True),  # the same action: a hit
            ("Send Kai the full call archive", "Robin Vale (you)", True, True)]  # same title, wrong owner
    p = preds(*rows)
    assert lab.must_not_pairs(must_not["must_not"], p, people, user) == [("n1", "P4")]
    g = {"lab_user": user, "items": [], "must_not": must_not["must_not"]}
    counts = lab.score_meeting(g, p, [], people)
    assert counts["checked_must_not"] == 1 and counts["must_not_checked"] == [("n1", "P4")]
    # Kai's own hypothetical, proposed for Kai, is the same owner.
    kai = preds(("Send Kai the full call archive", "Kai Moreno", False, True))
    assert lab.must_not_pairs(must_not["must_not"], kai, people, user) == [("n4", "P1")]
    # The judge's pairs replace the fallback; unknown IDs and repeated predictions are dropped.
    answer = {"matches": [], "must_not_matches": [{"must_not_id": "n2", "prediction_id": "P2"},
                                                  {"must_not_id": "n9", "prediction_id": "P1"},
                                                  {"must_not_id": "n3", "prediction_id": "P2"}]}
    _, banned, _ = lab.judge_matches([], p, lambda payload: (json.dumps(answer), None), must_not["must_not"])
    assert banned == [("n2", "P2")]
    assert lab.score_meeting(g, p, [], people, banned)["must_not_checked"] == [("n2", "P2")]
