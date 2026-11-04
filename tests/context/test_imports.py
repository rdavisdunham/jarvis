"""Imports: TXT, Markdown, WebVTT, SRT and versioned JSON become SourceEnvelopes; malformed input is refused."""

import json
from datetime import UTC, datetime

import pytest
from jarvis.context import imports
from jarvis.domain import DomainError

AT = datetime(2026, 11, 3, 15, 0, tzinfo=UTC)


def parse(body, fmt, **values):
    return imports.parse(body, fmt, connector="api", external_id="recorder:m1", occurred_at=AT, **values)


def test_plain_text_reads_speakers_stamps_and_continuations():
    envelope = parse("[00:01:02] Davin: We settle scope today.\nIt is due Friday.\n\nPriya: Agreed.\n"
                     "\nA line with no label.", "txt", title="Scope")
    assert [(s.speaker_label, s.text, s.start_ms) for s in envelope.segments] == [
        ("Davin", "We settle scope today.\nIt is due Friday.", 62_000), ("Priya", "Agreed.", None),
        (None, "A line with no label.", None)]
    assert envelope.title == "Scope" and envelope.content_hash and envelope.metadata == {"import_format": "txt"}


def test_urls_and_mid_line_colons_are_not_speakers():
    segments = imports.text_segments("See https://example.com/x for details\nMarco: Decision: ship it")
    assert segments[0].speaker_label is None
    assert segments[1].speaker_label == "Marco" and segments[1].text == "Decision: ship it"


def test_markdown_unwraps_bold_speakers_and_takes_the_heading_as_title():
    envelope = parse("# Pilot sync\n\n- **Davin:** Let us start.\n\n**Priya**: The pilot covers 40 clinics.\n", "markdown")
    assert envelope.title == "Pilot sync"
    assert [(s.speaker_label, s.text) for s in envelope.segments] == [
        ("Davin", "Let us start."), ("Priya", "The pilot covers 40 clinics.")]


def test_webvtt_voices_timings_and_notes():
    body = ("WEBVTT - meeting\nKind: captions\n\nNOTE exported by a recorder\n\n1\n00:00:01.000 --> 00:00:04.500 align:start\n"
            "<v Davin>Hello <b>there</b></v>\n\n00:05.000 --> 00:07.250\nPriya: Hi\nsecond line\n")
    segments = parse(body, "vtt").segments
    assert [(s.speaker_label, s.text, s.start_ms, s.end_ms) for s in segments] == [
        ("Davin", "Hello there", 1000, 4500), ("Priya", "Hi\nsecond line", 5000, 7250)]


def test_srt_cues():
    body = "1\n00:00:01,000 --> 00:00:02,000\nMarco: First.\n\n2\n00:00:03,000 --> 00:00:04,000\n<i>Unlabelled</i>\n"
    segments = parse(body, "srt").segments
    assert [(s.speaker_label, s.text, s.start_ms) for s in segments] == [("Marco", "First.", 1000), (None, "Unlabelled", 3000)]


@pytest.mark.parametrize("body,fmt,message", [
    ("", "txt", "no text"),
    ("1\n00:00:01,000 --> 00:00:02,000\nHi", "vtt", "starts with WEBVTT"),
    ("WEBVTT\n\n00:00:01.000 -> 00:00:02.000\nHi", "vtt", "no timing line"),
    ("WEBVTT\n\n00:00:01.000 --> 00:00:99.000\nHi", "vtt", "malformed timing"),
    ("WEBVTT\n\n00:00:05.000 --> 00:00:01.000\nHi", "vtt", "ends before it starts"),
    ("Hello\n00:00:01,000 --> 00:00:02,000\nHi", "srt", "cue number"),
    ("WEBVTT\n", "vtt", "no cue text"),
])
def test_malformed_input_is_refused_clearly(body, fmt, message):
    with pytest.raises(DomainError) as error:
        parse(body, fmt)
    assert error.value.code == "INVALID_ARGUMENT" and message in error.value.message


def test_raw_text_needs_an_occurrence_time():
    with pytest.raises(DomainError, match="occurred_at"):
        imports.parse("Davin: hi", "txt", connector="api", external_id="x:1")


def test_versioned_json_envelope():
    body = {"version": 1, "external_id": "m1", "title": "Pilot", "occurred_at": AT.isoformat(),
            "segments": [{"seq": 0, "speaker_label": "Davin", "text": "Hello"}], "private_notes": "Remember X."}
    envelope = parse(json.dumps(body), "json")
    assert envelope.connector == "api" and envelope.external_id == "recorder:m1"
    assert envelope.content_hash and envelope.private_notes == "Remember X."


@pytest.mark.parametrize("body,message", [
    ("not json", "not valid JSON"),
    ("[1]", "must be an object"),
    (json.dumps({"title": "x"}), '"version": 1'),
    (json.dumps({"version": 1, "external_id": "other", "occurred_at": AT.isoformat(), "text": "x",
                 "external_revision": "r"}), "different external_id"),
])
def test_bad_json_is_refused(body, message):
    with pytest.raises(DomainError, match=message):
        parse(body, "json")


def test_invalid_envelope_reports_locations_without_echoing_values():
    secret_text = "do-not-echo-this"
    body = json.dumps({"version": 1, "occurred_at": "2026-11-03T15:00:00", "text": secret_text, "content_hash": "h"})
    with pytest.raises(DomainError) as error:
        parse(body, "json")
    assert error.value.message == "The source envelope is not valid."
    assert secret_text not in json.dumps(error.value.data) and error.value.data[0]["loc"] == ["occurred_at"]


def test_detect_by_type_extension_and_sniffing():
    assert imports.detect("text/vtt; charset=utf-8") == "vtt"
    assert imports.detect(None, "call.SRT") == "srt"
    assert imports.detect("text/plain", data="WEBVTT\n\n00:01.000 --> 00:02.000\nx") == "vtt"
    assert imports.detect("text/plain", data="1\n00:00:01,000 --> 00:00:02,000\nx") == "srt"
    assert imports.detect("text/plain", data="Davin: hi") == "txt"
    with pytest.raises(DomainError) as error:
        imports.detect("application/pdf")
    assert error.value.status == 415


def test_decode_refuses_non_utf8():
    assert imports.decode("﻿Davin: hi".encode()) == "Davin: hi"
    with pytest.raises(DomainError, match="UTF-8"):
        imports.decode(b"\xff\xfe\x00bad")
