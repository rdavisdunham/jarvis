"""Submit a folder of meeting exports to Eridani's source endpoint (standalone: stdlib and httpx only).

    ERIDANI_SOURCES_KEY=<bot key with sources:write> \\
    python scripts/bridge_submit.py ./exports --base https://eridani.example --connector recorder [--dry-run]

Each file becomes `PUT {base}/api/v1/external/sources/{connector}/{external_id}`:

- `.vtt`, `.srt`, `.md`/`.markdown`, `.txt`: sent raw with the matching Content-Type. `external_id` is the file
  stem. Title, start time and timezone come from an optional sidecar `<stem>.meta.json`
  (`{"title", "occurred_at", "timezone"}`), else a `YYYY-MM-DD[ HH-MM]` prefix in the file name (read in
  `--timezone`), else the file's modification time.
- `.json` with `"version": 1`: a source envelope, sent as is (`external_id` from the body or the stem).
- `.json` holding a Granola MCP export (a list of meetings, or `{"meetings": [...]}`, each with `id`, `title`,
  `date`, `attendees` and `transcript` text or `[{"speaker", "text"}]` items): one envelope per meeting, keyed by
  the meeting id, connector label `granola` unless `--connector` is given.

Every submission carries `external_revision = sha256:<content hash>`, so resubmitting an unchanged folder creates
nothing new and a changed file creates a revision. The script keeps no state and is not scheduled. File contents
and the key are never printed.
"""

import argparse
import hashlib
import json
import os
import re
import sys
import time
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import quote
from zoneinfo import ZoneInfo

import httpx

KEY_ENV = "ERIDANI_SOURCES_KEY"
BASE_ENV = "ERIDANI_BASE_URL"
RAW_TYPES = {".vtt": "text/vtt", ".srt": "application/x-subrip", ".md": "text/markdown",
             ".markdown": "text/markdown", ".txt": "text/plain"}
NAME_TIME = re.compile(r"^(\d{4}-\d{2}-\d{2})(?:[ T_](\d{2})[-:.h](\d{2}))?")
RETRIES = 3
# The endpoint's ingestion_status: a new source, a content revision, a display-only refresh, or nothing new
# (a deleted source ignores content it already held).
CHANGES = {"created": "created", "content": "revised", "metadata": "refreshed", "none": "unchanged",
           "deleted": "unchanged"}


@dataclass
class Submission:
    connector: str
    external_id: str
    origin: str  # file name, plus the meeting id for exports holding several meetings
    content_type: str
    body: bytes
    params: dict[str, str]


def digest(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def parse_time(value: str, zone: ZoneInfo) -> datetime:
    """An ISO date or datetime; naive values are read in `zone`, date-only values at midnight."""
    parsed = datetime.fromisoformat(value.strip())
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=zone)


def raw_submission(path: Path, connector: str, zone: ZoneInfo) -> Submission:
    body = path.read_bytes()
    meta_path = path.with_name(path.stem + ".meta.json")
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    if meta.get("occurred_at"):
        occurred = parse_time(meta["occurred_at"], ZoneInfo(meta.get("timezone") or zone.key))
    elif match := NAME_TIME.match(path.stem):
        day, hour, minute = match.groups()
        occurred = parse_time(f"{day}T{hour or '00'}:{minute or '00'}", zone)
    else:
        occurred = datetime.fromtimestamp(path.stat().st_mtime, UTC)
    params = {"occurred_at": occurred.isoformat(), "external_revision": digest(body)}
    if meta.get("title"):
        params["title"] = str(meta["title"])[:500]
    if meta.get("timezone") or zone.key != "UTC":
        params["timezone"] = meta.get("timezone") or zone.key
    return Submission(connector, path.stem, path.name, RAW_TYPES[path.suffix.lower()], body, params)


def _segments(transcript: list) -> list[dict]:
    segments = []
    for item in transcript:
        if not isinstance(item, dict) or not str(item.get("text") or "").strip():
            continue
        speaker = item.get("speaker") or item.get("name")
        if isinstance(speaker, dict):
            speaker = speaker.get("name") or speaker.get("diarization_label")
        segment = {"seq": len(segments), "text": str(item["text"])}
        if speaker:
            segment["speaker_label"] = str(speaker)[:200]
        segments.append(segment)
    return segments


def meeting_envelope(meeting: dict, zone: ZoneInfo) -> dict | None:
    """A Granola MCP export meeting as a version 1 envelope, or None when it has no transcript text."""
    transcript = meeting.get("transcript")
    envelope: dict = {"version": 1, "title": str(meeting.get("title") or "")[:500]}
    if isinstance(transcript, list):
        segments = _segments(transcript)
        if not segments:
            return None
        envelope["segments"] = segments
        content = json.dumps(segments, sort_keys=True, ensure_ascii=False)
    elif isinstance(transcript, str) and transcript.strip():
        envelope["text"] = content = transcript
    else:
        return None
    when = meeting.get("date") or meeting.get("meeting_at") or meeting.get("created_at")
    if not when:
        return None
    envelope["occurred_at"] = parse_time(str(when), zone).isoformat()
    participants = []
    for person in meeting.get("attendees") or meeting.get("participants") or []:
        if isinstance(person, str):
            person = {"email": person} if "@" in person else {"name": person}
        if isinstance(person, dict) and (person.get("name") or person.get("email")):
            participants.append({k: str(person[k]) for k in ("name", "email") if person.get(k)})
    envelope["participants"] = participants
    notes = meeting.get("private_notes") or meeting.get("notes")
    if isinstance(notes, str) and notes.strip():
        envelope["private_notes"] = notes
    envelope["external_revision"] = digest((content + "\n" + (envelope.get("private_notes") or "")).encode())
    envelope["metadata"] = {"bridge": "granola_mcp_export"}
    return envelope


def json_submissions(path: Path, connector: str | None, zone: ZoneInfo) -> Iterator[Submission]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and data.get("version", data.get("envelope_version")) == 1:
        body = dict(data)
        body.pop("connector", None)
        external_id = str(body.get("external_id") or path.stem)
        if not (body.get("external_revision") or body.get("content_hash")):
            body["external_revision"] = digest(path.read_bytes())
        yield Submission(connector or "other", external_id, path.name, "application/json",
                         json.dumps(body, ensure_ascii=False).encode(), {})
        return
    meetings = data.get("meetings") if isinstance(data, dict) else data
    if not isinstance(meetings, list):
        raise ValueError("not a version 1 envelope or a list of meetings")  # noqa: TRY004 - reported as a bad file
    for meeting in meetings:
        if not isinstance(meeting, dict) or not meeting.get("id"):
            continue
        envelope = meeting_envelope(meeting, zone)
        if envelope is None:
            print(f"skip  {path.name}#{meeting['id']}: no transcript text or date")
            continue
        yield Submission(connector or "granola", str(meeting["id"]), f"{path.name}#{meeting['id']}",
                         "application/json", json.dumps(envelope, ensure_ascii=False).encode(), {})


def collect(folder: Path, connector: str | None, zone: ZoneInfo) -> Iterator[Submission | tuple[str, str]]:
    """Submissions in file-name order; unreadable files yield `(origin, error)`."""
    for path in sorted(p for p in folder.iterdir() if p.is_file()):
        suffix = path.suffix.lower()
        if path.name.endswith(".meta.json"):
            continue
        try:
            if suffix in RAW_TYPES:
                yield raw_submission(path, connector or "other", zone)
            elif suffix == ".json":
                yield from json_submissions(path, connector, zone)
        except (OSError, ValueError, UnicodeDecodeError) as error:
            yield path.name, type(error).__name__ + (f": {error}" if isinstance(error, ValueError) else "")


def put(client: httpx.Client, item: Submission, sleep=time.sleep) -> httpx.Response:
    url = f"/api/v1/external/sources/{quote(item.connector, safe='')}/{quote(item.external_id, safe='')}"
    for attempt in range(RETRIES):
        response = client.put(url, params=item.params, content=item.body,
                              headers={"Content-Type": item.content_type})
        if response.status_code != 429 and response.status_code < 500 or attempt == RETRIES - 1:
            return response
        try:
            wait = float(response.headers.get("Retry-After", 2 ** attempt))
        except ValueError:
            wait = 2.0 ** attempt
        sleep(min(wait, 60.0))
    return response


def run(folder: Path, base: str, key: str, connector: str | None, zone: ZoneInfo, dry_run: bool = False,
        transport: httpx.BaseTransport | None = None, sleep=time.sleep) -> dict[str, int]:
    counts = {"created": 0, "revised": 0, "refreshed": 0, "unchanged": 0, "failed": 0, "would_submit": 0}
    client = None if dry_run else httpx.Client(base_url=base.rstrip("/"), transport=transport, timeout=60,
                                               headers={"Authorization": f"Bearer {key}"})
    try:
        for item in collect(folder, connector, zone):
            if isinstance(item, tuple):
                counts["failed"] += 1
                print(f"fail  {item[0]}: {item[1]}")
                continue
            if client is None:
                counts["would_submit"] += 1
                print(f"would {item.origin} -> {item.connector}/{item.external_id} ({len(item.body)} bytes)")
                continue
            try:
                response = put(client, item, sleep)
            except httpx.HTTPError as error:
                counts["failed"] += 1
                print(f"fail  {item.origin}: {type(error).__name__}")
                continue
            if response.status_code >= 400:
                counts["failed"] += 1
                try:
                    detail = response.json().get("error", {}).get("code") or response.status_code
                except (ValueError, AttributeError):
                    detail = response.status_code
                print(f"fail  {item.origin}: {detail}")
                continue
            change = str(response.json().get("ingestion_status") or "")
            outcome = CHANGES.get(change, "created")
            counts[outcome] += 1
            print(f"{outcome[:5]:5} {item.origin} -> {item.connector}/{item.external_id}")
    finally:
        if client is not None:
            client.close()
    print(" ".join(f"{k}={v}" for k, v in counts.items() if v or k == "failed"))
    return counts


def main(argv: list[str] | None = None, transport: httpx.BaseTransport | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("folder", type=Path)
    parser.add_argument("--base", default=os.environ.get(BASE_ENV, ""), help=f"Eridani origin (or ${BASE_ENV})")
    parser.add_argument("--connector", help="label in the URL: recorder, zoom, granola, other (default per file)")
    parser.add_argument("--timezone", default="UTC", help="zone for naive times in names and exports")
    parser.add_argument("--dry-run", action="store_true", help="list what would be submitted; send nothing")
    args = parser.parse_args(argv)
    key = os.environ.get(KEY_ENV, "")
    if not args.folder.is_dir():
        parser.error(f"{args.folder} is not a folder")
    if not args.dry_run and not (args.base and key):
        parser.error(f"--base (or ${BASE_ENV}) and ${KEY_ENV} are required unless --dry-run")
    counts = run(args.folder, args.base, key, args.connector and args.connector.strip().lower(),
                 ZoneInfo(args.timezone), args.dry_run, transport)
    return 1 if counts["failed"] else 0


if __name__ == "__main__":
    sys.exit(main())
