"""scripts/bridge_submit.py against a fake source endpoint (httpx MockTransport, no network)."""

import json
from pathlib import Path

import httpx
import pytest

from scripts import bridge_submit

VTT = "WEBVTT\n\n00:00:01.000 --> 00:00:04.000\n<v Priya>We ship the export on Friday.\n"
SRT = "1\n00:00:01,000 --> 00:00:03,000\nMarco: Checklist is drafted.\n"
EXPORT = {"meetings": [
    {"id": "9f1c2b7e-0d4a-4c55-9e1f-1a2b3c4d5e6f", "title": "Northwind kickoff", "date": "2026-11-02T16:00:00Z",
     "attendees": [{"name": "Priya Natarajan", "email": "priya.natarajan@northwind.example"}, "Tomás Reyes"],
     "transcript": [{"speaker": "Priya Natarajan", "text": "Analytics team goes first."},
                    {"speaker": {"name": "Tomás Reyes"}, "text": "Agreed."}, {"speaker": "x", "text": " "}]},
    {"id": "not_K3l4M5n6O7p8Q9", "title": "1:1", "date": "2026-11-03",
     "attendees": [], "transcript": "Alex: Let's review hiring.\nSam: Sure."},
    {"id": "empty-meeting", "title": "No recording", "date": "2026-11-03", "transcript": None},
]}


class FakeEndpoint:
    def __init__(self):
        self.store: dict[tuple[str, str], str] = {}
        self.requests: list[httpx.Request] = []
        self.busy = 0

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if self.busy:
            self.busy -= 1
            return httpx.Response(429, headers={"Retry-After": "1"})
        if request.headers.get("Authorization") != "Bearer sk-test":
            return httpx.Response(401, json={"error": {"code": "UNAUTHORIZED"}})
        _, _, _, _, _, connector, external_id = request.url.path.split("/")
        body = request.content
        revision = request.url.params.get("external_revision") or json.loads(body).get("external_revision")
        key = (connector, external_id)
        change = "created" if key not in self.store else "none" if self.store[key] == revision else "content"
        self.store[key] = revision
        return httpx.Response(200, json={"id": "src", "ingestion_status": change})


@pytest.fixture
def folder(tmp_path: Path) -> Path:
    (tmp_path / "2026-11-03 10-30 standup.srt").write_text(SRT)
    (tmp_path / "call.vtt").write_text(VTT)
    (tmp_path / "notes.md").write_text("# Planning\n\nAlex: Ship it.\n")
    (tmp_path / "notes.meta.json").write_text(json.dumps({"title": "Planning", "occurred_at": "2026-11-01T09:00:00",
                                                           "timezone": "Europe/Lisbon"}))
    (tmp_path / "envelope.json").write_text(json.dumps({
        "version": 1, "external_id": "rec-77", "title": "Recorder", "occurred_at": "2026-11-04T08:00:00+00:00",
        "text": "Sam: Budget approved."}))
    (tmp_path / "granola_export.json").write_text(json.dumps(EXPORT))
    (tmp_path / "ignored.pdf").write_bytes(b"%PDF")
    return tmp_path


def run(folder, fake, *args):
    return bridge_submit.main([str(folder), "--base", "https://eridani.test/", *args],
                              transport=httpx.MockTransport(fake))




def test_folder_round_trip(folder, monkeypatch, capsys):
    monkeypatch.setenv(bridge_submit.KEY_ENV, "sk-test")
    fake = FakeEndpoint()
    assert run(folder, fake, "--connector", "recorder", "--timezone", "America/New_York") == 0
    first = capsys.readouterr().out
    assert "created=6" in first and "sk-test" not in first and "Ship it" not in first
    assert sorted(fake.store) == sorted([
        ("recorder", "2026-11-03 10-30 standup"), ("recorder", "call"), ("recorder", "notes"),
        ("recorder", "rec-77"), ("recorder", "9f1c2b7e-0d4a-4c55-9e1f-1a2b3c4d5e6f"),
        ("recorder", "not_K3l4M5n6O7p8Q9")])
    by_path = {r.url.raw_path.decode().split("?")[0].rsplit("/", 1)[1]: r for r in fake.requests}
    srt = by_path["2026-11-03%2010-30%20standup"]
    assert srt.headers["Content-Type"] == "application/x-subrip"
    assert srt.url.params["occurred_at"] == "2026-11-03T10:30:00-05:00"
    assert srt.url.params["external_revision"].startswith("sha256:")
    notes = by_path["notes"]
    assert notes.url.params["title"] == "Planning" and notes.url.params["occurred_at"] == "2026-11-01T09:00:00+00:00"
    assert by_path["call"].headers["Content-Type"] == "text/vtt"
    kickoff = json.loads(by_path["9f1c2b7e-0d4a-4c55-9e1f-1a2b3c4d5e6f"].content)
    assert kickoff["version"] == 1 and [s["speaker_label"] for s in kickoff["segments"]] == [
        "Priya Natarajan", "Tomás Reyes"]
    assert kickoff["participants"] == [{"name": "Priya Natarajan", "email": "priya.natarajan@northwind.example"},
                                       {"name": "Tomás Reyes"}]
    one_on_one = json.loads(by_path["not_K3l4M5n6O7p8Q9"].content)
    assert one_on_one["text"].startswith("Alex:") and one_on_one["occurred_at"] == "2026-11-03T00:00:00-05:00"

    assert run(folder, fake, "--connector", "recorder", "--timezone", "America/New_York") == 0
    assert "unchanged=6" in capsys.readouterr().out and len(fake.store) == 6

    (folder / "call.vtt").write_text(VTT + "\n00:00:05.000 --> 00:00:06.000\n<v Alex>Thanks.\n")
    assert run(folder, fake, "--connector", "recorder", "--timezone", "America/New_York") == 0
    out = capsys.readouterr().out
    assert "revised=1" in out and "unchanged=5" in out


def test_default_labels(folder, monkeypatch, capsys):
    monkeypatch.setenv(bridge_submit.KEY_ENV, "sk-test")
    fake = FakeEndpoint()
    assert run(folder, fake) == 0
    labels = {connector for connector, _ in fake.store}
    assert labels == {"other", "granola"} and ("granola", "not_K3l4M5n6O7p8Q9") in fake.store


def test_dry_run_sends_nothing(folder, capsys):
    fake = FakeEndpoint()
    assert run(folder, fake, "--dry-run") == 0
    out = capsys.readouterr().out
    assert fake.requests == [] and "would_submit=6" in out and "skip  granola_export.json#empty-meeting" in out


def test_failures_are_counted(folder, monkeypatch, capsys):
    monkeypatch.setenv(bridge_submit.KEY_ENV, "wrong")
    (folder / "broken.json").write_text("{not json")
    assert run(folder, FakeEndpoint()) == 1
    out = capsys.readouterr().out
    assert "failed=7" in out and "fail  broken.json: JSONDecodeError" in out and "UNAUTHORIZED" in out


def test_retries_rate_limits(folder, monkeypatch):
    monkeypatch.setenv(bridge_submit.KEY_ENV, "sk-test")
    slept = []
    monkeypatch.setattr(bridge_submit.time, "sleep", slept.append)
    fake = FakeEndpoint()
    fake.busy = 2
    counts = bridge_submit.run(folder, "https://eridani.test", "sk-test", None, bridge_submit.ZoneInfo("UTC"),
                               transport=httpx.MockTransport(fake), sleep=slept.append)
    assert counts["created"] == 6 and slept == [1.0, 1.0]


def test_requires_key_unless_dry_run(folder, monkeypatch):
    monkeypatch.delenv(bridge_submit.KEY_ENV, raising=False)
    with pytest.raises(SystemExit):
        bridge_submit.main([str(folder), "--base", "https://eridani.test"])
