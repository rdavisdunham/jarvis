"""File and raw-body imports: TXT, Markdown, WebVTT, SRT and versioned JSON become a `SourceEnvelope`.

Imported text is data. Speaker labels come from "Name: text" lines and VTT `<v Name>` voice spans; timestamps
from VTT and SRT cue timings or a leading `[hh:mm:ss]` on a text line. Malformed input is refused with a
`DomainError` naming the problem and where it is; content is never echoed back.
"""

import json
import re
from datetime import datetime
from typing import Any

from pydantic import ValidationError

from ..domain import DomainError
from .schema import Segment, SourceEnvelope, content_hash

FORMATS = ("txt", "markdown", "vtt", "srt", "json")
CONTENT_TYPES = {
    "text/plain": "txt",
    "text/markdown": "markdown",
    "text/x-markdown": "markdown",
    "text/vtt": "vtt",
    "application/x-subrip": "srt",
    "application/srt": "srt",
    "text/srt": "srt",
    "application/json": "json",
}
EXTENSIONS = {".txt": "txt", ".md": "markdown", ".markdown": "markdown", ".vtt": "vtt", ".srt": "srt", ".json": "json"}

_STAMP = r"(?:(\d{1,2}):)?(\d{1,2}):(\d{2})(?:[.,](\d{1,3}))?"
_LINE_STAMP = re.compile(r"^\[?\(?(" + _STAMP + r")\)?\]?\s*[-–]?\s*")
_SPEAKER = re.compile(r"^([^\W\d_][\w.'’()&-]*(?: [\w.'’()&-]+){0,4}):\s+(\S.*)$", re.DOTALL)
_TIMING = re.compile(r"^\s*(" + _STAMP + r")\s+-->\s+(" + _STAMP + r")(?:\s.*)?$")
_VOICE = re.compile(r"<v(?:\.[\w.-]+)?\s+([^>]+)>")
_TAG = re.compile(r"</?[a-zA-Z][^>]*>|<\d[\d:.]*>")
_MD_MARK = re.compile(r"^\s*(?:[-*+]\s+|>\s*)?")
_MD_BOLD_SPEAKER = re.compile(r"^(\*\*|__)([^*_\n]{1,80}?):?\1:?\s+")


def bad(message: str) -> DomainError:
    return DomainError("INVALID_ARGUMENT", message, 400)


def detect(content_type: str | None = None, filename: str | None = None, data: str = "") -> str:
    """The import format from a Content-Type, else a file extension; plain text that is really VTT or SRT is
    recognised by its header or first cue."""
    kind = CONTENT_TYPES.get((content_type or "").split(";")[0].strip().lower())
    if kind is None and filename:
        kind = next((v for k, v in EXTENSIONS.items() if filename.lower().endswith(k)), None)
    if kind is None:
        raise DomainError("UNSUPPORTED_MEDIA_TYPE", "Send JSON, plain text, Markdown, WebVTT or SRT.", 415)
    if kind == "txt":
        head = data.lstrip("﻿").lstrip()[:200]
        if head.startswith("WEBVTT"):
            return "vtt"
        lines = head.splitlines()
        if len(lines) > 1 and lines[0].strip().isdigit() and _TIMING.match(lines[1]):
            return "srt"
    return kind


def decode(data: bytes) -> str:
    try:
        return data.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise bad("The body is not valid UTF-8 text.") from error


def _ms(groups: tuple) -> int:
    hours, minutes, seconds, fraction = groups
    if int(minutes) > 59 or int(seconds) > 59:
        raise ValueError("timestamp out of range")
    fraction = (fraction or "0").ljust(3, "0")
    return ((int(hours or 0) * 60 + int(minutes)) * 60 + int(seconds)) * 1000 + int(fraction)


def _split_speaker(line: str) -> tuple[str | None, str]:
    match = _SPEAKER.match(line)
    return (match.group(1).strip(), match.group(2)) if match else (None, line)


def text_segments(text: str) -> list[Segment]:
    """Plain text: a "Name: text" line (optionally after a `[hh:mm:ss]` stamp) starts a segment; following lines
    continue it; a blank line ends it. Lines without a label form unlabelled segments."""
    segments: list[Segment] = []
    current: dict | None = None

    def close():
        nonlocal current
        if current and current["text"].strip():
            segments.append(Segment(seq=len(segments), speaker_label=current["speaker"],
                                    text=current["text"].strip(), start_ms=current["start"]))
        current = None

    for raw in text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        line = raw.strip()
        if not line:
            close()
            continue
        start = None
        stamp = _LINE_STAMP.match(line)
        if stamp and stamp.group(1) and stamp.end() < len(line):
            try:
                start = _ms(stamp.groups()[1:5])
                line = line[stamp.end():]
            except ValueError:
                start = None
        speaker, body = _split_speaker(line)
        if speaker or start is not None or current is None:
            close()
            current = {"speaker": speaker, "text": body, "start": start}
        else:
            current["text"] += "\n" + line
    close()
    if not segments:
        raise bad("The transcript has no text.")
    return segments


def markdown_segments(text: str) -> tuple[list[Segment], str]:
    """Markdown: bullets, quotes and bold speaker names (`**Name:**`) are unwrapped, then read as plain text.
    The first level-one heading is returned as a title and not kept as a segment."""
    title, lines = "", []
    for raw in text.replace("\r\n", "\n").split("\n"):
        if not title and raw.startswith("# "):
            title = raw[2:].strip()
            continue
        line = _MD_MARK.sub("", raw, count=1) if raw.strip() else ""
        line = _MD_BOLD_SPEAKER.sub(lambda m: m.group(2).strip() + ": ", line, count=1)
        lines.append(line)
    return text_segments("\n".join(lines)), title


def _cue_text(lines: list[str]) -> tuple[str | None, str]:
    joined = "\n".join(lines)
    voice = _VOICE.search(joined)
    speaker = voice.group(1).strip() if voice else None
    body = _TAG.sub("", joined).strip()
    if speaker is None:
        speaker, body = _split_speaker(body)
    return speaker, body


def _cues(blocks: list[list[str]], label: str) -> list[Segment]:
    segments: list[Segment] = []
    for number, block in enumerate(blocks, 1):
        timing = next((i for i, line in enumerate(block[:2]) if "-->" in line), None)
        if timing is None:
            raise bad(f"{label} cue {number} has no timing line.")
        match = _TIMING.match(block[timing])
        try:
            if not match:
                raise ValueError
            start, end = _ms(match.groups()[1:5]), _ms(match.groups()[6:10])
        except ValueError as error:
            raise bad(f"{label} cue {number} has a malformed timing line.") from error
        if end < start:
            raise bad(f"{label} cue {number} ends before it starts.")
        speaker, body = _cue_text(block[timing + 1:])
        if body:
            segments.append(Segment(seq=len(segments), speaker_label=speaker, text=body, start_ms=start, end_ms=end))
    if not segments:
        raise bad(f"The {label} file has no cue text.")
    return segments


def _blocks(text: str) -> list[list[str]]:
    blocks, block = [], []
    for line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        if line.strip():
            block.append(line.rstrip())
        elif block:
            blocks.append(block)
            block = []
    if block:
        blocks.append(block)
    return blocks


def vtt_segments(text: str) -> list[Segment]:
    text = text.lstrip("﻿")
    if not re.match(r"WEBVTT(?:[ \t].*)?(?:\n|\r|$)", text):
        raise bad("A WebVTT file starts with WEBVTT.")
    blocks = _blocks(text)[1:]  # the header block
    blocks = [b for b in blocks if not b[0].startswith(("NOTE", "STYLE", "REGION"))]
    return _cues(blocks, "WebVTT")


def srt_segments(text: str) -> list[Segment]:
    blocks = _blocks(text.lstrip("﻿"))
    for number, block in enumerate(blocks, 1):
        if "-->" not in block[0] and not block[0].strip().isdigit():
            raise bad(f"SRT cue {number} does not start with a cue number.")
    return _cues(blocks, "SRT")


def envelope_from_json(data: str | bytes | dict, *, connector: str, external_id: str) -> SourceEnvelope:
    """Versioned JSON: `{"version": 1, ...SourceEnvelope fields}`. `connector` and `external_id` come from the
    caller (the URL or the upload), and a body that names a different external ID is refused."""
    if not isinstance(data, dict):
        try:
            data = json.loads(data)
        except (ValueError, UnicodeDecodeError) as error:
            raise bad("The body is not valid JSON.") from error
    if not isinstance(data, dict):
        raise bad("The JSON body must be an object.")
    data = dict(data)
    version = data.pop("version", data.pop("envelope_version", None))
    if version != 1:
        raise bad('The JSON body needs "version": 1.')
    data.pop("connector", None)
    named = data.pop("external_id", None)
    if named is not None and named != external_id:
        raise bad("The body names a different external_id from the URL.")
    if data.get("segments") is not None and not (data.get("external_revision") or data.get("content_hash")):
        texts = [s.get("text", "") for s in data["segments"] if isinstance(s, dict)]
        data["content_hash"] = content_hash(texts)
    return _validated(connector=connector, external_id=external_id, envelope_version=1, **data)


def _validated(**fields: Any) -> SourceEnvelope:
    try:
        return SourceEnvelope(**fields)
    except ValidationError as error:
        # Locations and messages only: the rejected input value is never echoed back.
        details = [{"loc": list(e["loc"]), "msg": e["msg"]} for e in error.errors(include_url=False)]
        raise DomainError("INVALID_ARGUMENT", "The source envelope is not valid.", 400, data=details) from error
    except TypeError as error:
        raise bad("The source envelope has unexpected fields.") from error


def parse(data: bytes | str, fmt: str, *, connector: str, external_id: str, title: str = "",
          occurred_at: datetime | None = None, timezone: str | None = None,
          external_revision: str | None = None) -> SourceEnvelope:
    """One body in `fmt` (see `detect`) as an envelope. Raw text formats need `occurred_at` from the caller;
    the content hash of the parsed segments stands in for a missing `external_revision`."""
    text = decode(data) if isinstance(data, bytes) else data
    if fmt == "json":
        return envelope_from_json(text, connector=connector, external_id=external_id)
    if fmt not in FORMATS:
        raise DomainError("UNSUPPORTED_MEDIA_TYPE", "Send JSON, plain text, Markdown, WebVTT or SRT.", 415)
    if occurred_at is None:
        raise bad("Say when the meeting happened: occurred_at (ISO 8601 with a timezone offset).")
    if fmt == "markdown":
        segments, heading = markdown_segments(text)
        title = title or heading
    else:
        segments = {"txt": text_segments, "vtt": vtt_segments, "srt": srt_segments}[fmt](text)
    return _validated(connector=connector, external_id=external_id, title=title[:500], occurred_at=occurred_at,
                      timezone=timezone, segments=segments, external_revision=external_revision,
                      content_hash=content_hash(s.text for s in segments), metadata={"import_format": fmt})
