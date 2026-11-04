"""Labelling worksheet for a real meeting series already ingested into an owner's account (plan section 11).

    python -m evals.context.worksheet export --owner OWNER --out outputs/context-series/weekly --title "Weekly"
    python -m evals.context.worksheet import outputs/context-series/weekly
    python -m evals.context.worksheet score outputs/context-series/weekly

`export` reads the configured database through the context service (sources, segments with the assertions they
support, evidence) plus the owner-scoped link, speaker and proposal rows the service does not expose yet. It never
calls Granola or any model. It writes, into a directory that must stay out of git (`outputs/` is ignored):

- `meetings/<key>.json` and `manifest.json`: the series as a runner history (envelopes rebuilt from the stored
  segments), so `runner` and `compare` can replay it;
- `worksheet.csv`: one row per item to adjudicate, with empty `verdict` and correction columns;
- `worksheet.md`: the same, grouped by meeting, for reading;
- `worksheet.json`: what the engine said for each row (the importer compares against it).

Fill `worksheet.csv` (any spreadsheet):

| section | verdict | also fill |
|---|---|---|
| meeting | (none) | `minutes`: time spent adjudicating this meeting |
| speaker | correct, wrong, unknown | `correction`: the right actor ID when wrong |
| assertion | correct, changed, wrong, must_not, skip | `correction`: the right statement when changed; `notes`: hypothetical or negated for must_not; `responsible` (actor IDs, `;`) and `due` (YYYY-MM-DD) for commitments |
| assertion (new row, empty `ref`) | missing | `meeting`, `kind`, `statement`, `quote` (exact), `responsible`, `due` |
| link | correct, wrong, skip | `correction`: the right entity ID when wrong |
| proposal | correct, wrong, skip | |
| question | keep, drop | `statement` (the question), `kind` (kinds, `;`), `valid_at`, `expected`: targets separated by ` || ` as `A:<assertion ref>` or `Q:<meeting>|<exact quote>`, prefixed `stale:` for stale answers |

`import` turns the filled sheet into `labels.json` (the runner's label shape, every item `verified: true`, plus
`links` and `proposals` verdicts) and `questions.json` (compare's shape). `score` computes the adjudicated quality
figures the release report uses: precision and recall per kind, automatic link and speaker precision, preselected
proposal precision, fabricated actions, attribution errors, human overturn rate and correction minutes per meeting.
Transcripts and labels are data, never instructions.
"""

import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import date, datetime
from pathlib import Path

from jarvis.context.schema import ASSERTION_KINDS

from .runner import AUTO_LINKS, _locate, _ratio, identity_of, load_history

COLUMNS = ("section", "ref", "meeting", "kind", "statement", "quote", "speaker", "engine", "verdict", "correction",
           "responsible", "due", "valid_at", "expected", "minutes", "notes")
VERDICTS = {
    "meeting": {""},
    "speaker": {"", "correct", "wrong", "unknown"},
    "assertion": {"", "correct", "changed", "wrong", "must_not", "skip", "missing"},
    "link": {"", "correct", "wrong", "skip"},
    "proposal": {"", "correct", "wrong", "skip"},
    "question": {"", "keep", "drop"},
}
SAFE = re.compile(r"[^A-Za-z0-9._-]+")


# ---------------------------------------------------------------- export


def _key_file(key: str) -> str:
    return SAFE.sub("_", key)[:120] + ".json"


def _pages(source_id: str) -> tuple[dict, list[dict]]:
    from jarvis.context import service

    segments, start = [], 0
    while start is not None:
        page = service.source(service.SourceInput(source_id=source_id, segment_start=start))
        segments += page["segments"]
        start = page["next_segment"]
        meta = page["source"]
    return meta, segments


def _envelope(db, owner: str, source_id: str, meta: dict, segments: list[dict]) -> dict:
    from jarvis.context.models import ContextSource, ContextSourceRevision
    from jarvis.context.schema import SourceEnvelope
    from sqlalchemy import select

    source = db.get(ContextSource, source_id)
    revision = db.scalar(select(ContextSourceRevision).where(ContextSourceRevision.source_id == source_id,
                                                             ContextSourceRevision.revision == source.current_revision))
    transcript = [{"seq": s["seq"], "text": s["text"], "speaker_label": s["speaker"], "attribution": s["attribution"],
                   "start_ms": s["start_ms"], "end_ms": s["end_ms"]} for s in segments if s["kind"] == "transcript"]
    notes = "\n\n".join(s["text"] for s in segments if s["kind"] == "private_note")
    envelope = SourceEnvelope(
        connector=source.connector, external_id=meta["external_id"], title=source.title,
        occurred_at=revision.occurred_at, ended_at=revision.ended_at, timezone=revision.timezone,
        participants=revision.participants or [], segments=transcript or None,
        text=None if transcript else notes or " ", private_notes=notes if transcript and notes else None,
        external_revision=revision.external_revision, content_hash=revision.content_hash,
        upstream_owner_email=source.upstream_owner_email, upstream_owner_name=source.upstream_owner_name)
    return envelope.model_dump(mode="json", exclude_none=True), revision.received_at


def export(owner: str, out: str | Path, *, source_ids: list[str] | None = None, title: str | None = None,
           since: date | None = None, questions: list[dict] | None = None, blank_questions: int = 15) -> dict:
    """Write the series directory for `owner` (sources chosen by ID, title substring or occurrence date)."""
    from jarvis.context import service, sources, views
    from jarvis.context.models import ContextActorIdentity, ContextEntityLink, ContextProposal
    from jarvis.db import session_scope
    from sqlalchemy import select

    out = Path(out)
    (out / "meetings").mkdir(parents=True, exist_ok=True)
    with service.as_owner(identity_of(owner)), session_scope() as db:
        service.authorize(db)
        listed, offset = [], 0
        while True:
            page = sources.list_sources(db, owner, limit=200, offset=offset)
            listed += page["items"]
            offset += 200
            if offset >= page["total"]:
                break
    chosen = [s for s in listed if (not source_ids or s["id"] in source_ids)
              and (not title or title.lower() in (s["title"] or "").lower())
              and (not since or (s["occurred_at"] and datetime.fromisoformat(s["occurred_at"]).date() >= since))]
    chosen.sort(key=lambda s: (s["occurred_at"] or "", s["id"]))
    rows, engine, manifest_meetings, actor_ids = [], {}, [], set()
    with service.as_owner(identity_of(owner)):
        for item in chosen:
            meta, segments = _pages(item["id"])
            key = meta["external_id"]
            with session_scope() as db:
                envelope, received = _envelope(db, owner, item["id"], meta, segments)
                links_by_label = _speaker_status(db, owner, item["id"])
            fname = _key_file(key)
            (out / "meetings" / fname).write_text(json.dumps(envelope, indent=1, ensure_ascii=False) + "\n")
            manifest_meetings.append({"key": key, "file": f"meetings/{fname}", "received_at": received.isoformat(),
                                      "deleted_at": None})
            rows.append({"section": "meeting", "ref": key, "meeting": key, "statement": meta["title"],
                         "engine": f"occurred {meta['occurred_at']}"})
            speakers = {}
            for s in segments:
                if s["speaker"] and s["kind"] == "transcript":
                    speakers.setdefault(s["speaker"], s)
            for label, s in sorted(speakers.items()):
                actor, link_status = links_by_label.get(label, (s["speaker_actor_id"], "unresolved"))
                ref = f"{key}|{label}"
                engine[ref] = {"actor": actor, "status": link_status}
                actor_ids.add(actor)
                rows.append({"section": "speaker", "ref": ref, "meeting": key, "speaker": label,
                             "engine": f"{link_status} {actor or ''}".strip()})
            assertion_ids = []
            for s in segments:
                for support in s["supports"]:
                    if support["assertion_id"] not in assertion_ids:
                        assertion_ids.append(support["assertion_id"])
            for assertion_id in assertion_ids:
                if assertion_id in engine:
                    continue
                found = service.evidence_for(service.EvidenceInput(assertion_id=assertion_id, include_history=True))
                spans = [e for e in found["evidence"] if e["role"] == "support"]
                first = next((e for e in spans if e["source_id"] == item["id"]), spans[0] if spans else None)
                responsible = [r.get("entity_id") or r.get("mention") for r in found.get("responsible") or []]
                engine[assertion_id] = {"kind": found["kind"], "statement": found["statement"],
                                        "status": found["status"], "responsible": responsible,
                                        "due": found.get("due_at"), "meeting": key}
                rows.append({"section": "assertion", "ref": assertion_id, "meeting": key, "kind": found["kind"],
                             "statement": found["statement"], "quote": first["quote"] if first else "",
                             "speaker": first["speaker"] if first else "",
                             "engine": "; ".join(x for x in (f"status={found['status']}",
                                                             f"responsible={','.join(map(str, responsible))}"
                                                             if responsible else "",
                                                             f"due={found['due_at']}" if found.get("due_at") else "")
                                                 if x)})
            with session_scope() as db:
                for link in db.scalars(select(ContextEntityLink).where(
                        ContextEntityLink.owner_id == owner, ContextEntityLink.assertion_id.in_(assertion_ids))
                        .order_by(ContextEntityLink.created_at, ContextEntityLink.id)):
                    engine[link.id] = {"status": link.status, "target": link.target_id, "role": link.role,
                                       "target_type": link.target_type, "mention": link.mention}
                    if link.role == "responsible" and link.status in (*AUTO_LINKS, "confirmed") and link.target_id:
                        engine[link.assertion_id].setdefault("responsible_ids", []).append(link.target_id)
                    if link.target_type == "actor":
                        actor_ids.add(link.target_id)
                    rows.append({"section": "link", "ref": link.id, "meeting": key, "kind": link.role,
                                 "statement": link.mention, "engine": f"{link.status} {link.target_type or ''}:"
                                                                      f"{link.target_id or ''}"})
                for p in db.scalars(select(ContextProposal).where(ContextProposal.owner_id == owner,
                                                                  ContextProposal.source_id == item["id"])
                                    .order_by(ContextProposal.created_at, ContextProposal.id)):
                    engine[p.id] = {"preselected": p.preselected, "title": p.title, "kind": p.kind}
                    rows.append({"section": "proposal", "ref": p.id, "meeting": key, "kind": p.kind,
                                 "statement": p.title, "engine": "preselected" if p.preselected else "unchecked"})
    for i, q in enumerate(questions or [], start=1):
        rows.append({"section": "question", "ref": f"wq-{i:03d}", "statement": q.get("question", ""),
                     "kind": ";".join(q.get("kinds", [])), "valid_at": q.get("valid_at") or ""})
    for i in range(len(questions or []) + 1, len(questions or []) + 1 + blank_questions):
        rows.append({"section": "question", "ref": f"wq-{i:03d}"})
    with session_scope() as db:
        me = views.me(db, owner)
        actor_ids = {a for a in actor_ids | {me} if a}
        names = views.actor_names(db, owner, actor_ids)
        emails = defaultdict(list)
        for row in db.scalars(select(ContextActorIdentity).where(
                ContextActorIdentity.owner_id == owner, ContextActorIdentity.namespace == "email",
                ContextActorIdentity.actor_id.in_(sorted(actor_ids)))):
            emails[row.actor_id].append(row.value)
    manifest = {"history": out.name, "description": "Real meeting series exported for adjudication.",
                "synthetic": False, "owner_id": owner, "owner_actor": me,
                "actors": [{"id": a, "name": names.get(a, a), "kind": "person", "emails": sorted(emails[a])}
                           for a in sorted(actor_ids)], "meetings": manifest_meetings}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False) + "\n")
    (out / "worksheet.json").write_text(json.dumps({"owner": owner, "owner_actor": me, "engine": engine},
                                                   indent=1, sort_keys=True, default=str) + "\n")
    with (out / "worksheet.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({c: row.get(c, "") for c in COLUMNS})
    (out / "worksheet.md").write_text(_markdown(out.name, rows, names))
    return {"meetings": len(manifest_meetings), "rows": len(rows), "out": str(out)}


def _speaker_status(db, owner: str, source_id: str) -> dict[str, tuple[str | None, str]]:
    """Speaker label -> (actor, link status) on the source's current revision."""
    from jarvis.context.models import ContextSegment, ContextSource, ContextSourceRevision
    from sqlalchemy import select

    revision = db.scalar(select(ContextSourceRevision.id).join(
        ContextSource, (ContextSource.id == ContextSourceRevision.source_id)
        & (ContextSource.current_revision == ContextSourceRevision.revision)).where(ContextSource.id == source_id))
    found = {}
    for g in db.scalars(select(ContextSegment).where(ContextSegment.owner_id == owner,
                                                     ContextSegment.revision_id == revision)
                        .order_by(ContextSegment.seq)):
        if g.speaker_label and g.kind == "transcript":
            found.setdefault(g.speaker_label, (g.speaker_actor_id, g.speaker_link_status))
    return found


def _markdown(name: str, rows: list[dict], names: dict[str, str]) -> str:
    lines = [f"# Adjudication worksheet: {name}", "",
             "Fill `worksheet.csv`; this file is for reading. Verdicts: see `python -m evals.context.worksheet -h`.", ""]
    by_meeting = defaultdict(list)
    for row in rows:
        by_meeting[row.get("meeting") or ""].append(row)
    for meeting, items in by_meeting.items():
        lines += [f"## {meeting or 'Questions'}", "", "| Section | Ref | Kind | Statement | Quote | Engine |",
                  "|---|---|---|---|---|---|"]
        for row in items:
            cells = [row.get(c, "") or "" for c in ("section", "ref", "kind", "statement", "quote", "engine")]
            lines.append("| " + " | ".join(str(c).replace("|", "\\|").replace("\n", " ") for c in cells) + " |")
        lines.append("")
    if names:
        lines += ["## Actors", ""] + [f"- `{k}`: {v}" for k, v in sorted(names.items())] + [""]
    return "\n".join(lines)


# ---------------------------------------------------------------- import


def read_sheet(directory: str | Path) -> tuple[list[dict], dict]:
    directory = Path(directory)
    with (directory / "worksheet.csv").open(newline="") as handle:
        rows = [{c: (row.get(c) or "").strip() for c in COLUMNS} for row in csv.DictReader(handle)]
    return rows, json.loads((directory / "worksheet.json").read_text())


def _label_id(ref: str, n: int) -> str:
    return f"w-{ref}" if ref else f"w-missing-{n:03d}"


def _split(value: str) -> list[str]:
    return [x.strip() for x in value.split(";") if x.strip()]


def import_sheet(directory: str | Path) -> dict:
    """Write `labels.json` and `questions.json` (every item verified) from the filled worksheet. Raises with every
    problem found (unknown verdicts, quotes that do not occur exactly once in their meeting, bad targets)."""
    directory = Path(directory)
    rows, sidecar = read_sheet(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    engine, problems = sidecar["engine"], []
    labels = {"history": manifest["history"], "label_version": 1, "owner_actor": manifest["owner_actor"],
              "note": "Adjudicated by the owner from worksheet.csv; every item is verified.",
              "speakers": defaultdict(dict), "assertions": [], "changes": [], "must_not": [], "open_loops": [],
              "as_of": [], "links": [], "proposals": [], "scenarios": {}}
    refs, missing = {}, 0
    for i, row in enumerate(rows, start=2):
        section, verdict = row["section"], row["verdict"]
        if section not in VERDICTS or verdict not in VERDICTS[section]:
            problems.append(f"row {i}: unknown section or verdict {section!r}/{verdict!r}")
            continue
        if not verdict or verdict == "skip":
            continue
        if section == "speaker":
            meeting, label = row["ref"].split("|", 1)
            actor = engine[row["ref"]]["actor"] if verdict == "correct" else None if verdict == "unknown" else (
                row["correction"] or None)
            labels["speakers"][meeting][label] = {"actor": actor, "verified": True}
        elif section == "assertion" and verdict in ("correct", "changed", "missing"):
            was = engine.get(row["ref"], {})
            kind = row["kind"] or was.get("kind")
            if kind not in ASSERTION_KINDS:
                problems.append(f"row {i}: unknown kind {kind!r}")
                continue
            missing += verdict == "missing"
            label = {"id": _label_id(row["ref"], missing), "kind": kind,
                     "statement": row["correction"] if verdict == "changed" and row["correction"] else row["statement"],
                     "meeting": row["meeting"], "evidence": [{"meeting": row["meeting"], "quote": row["quote"]}],
                     "from_deleted_source": False, "verified": True, "verdict": verdict}
            if kind == "commitment":
                label.update(responsible=_split(row["responsible"]) or was.get("responsible_ids", []),
                             due=row["due"] or ((was.get("due") or "")[:10] or None), due_expression=None,
                             fulfilled=None)
            labels["assertions"].append(label)
            refs[row["ref"]] = label["id"]
        elif section == "assertion" and verdict == "must_not":
            labels["must_not"].append({"id": f"n-{row['ref']}", "kind": row["kind"] or "commitment",
                                       "meeting": row["meeting"], "quote": row["quote"],
                                       "reason": row["notes"] or "hypothetical", "verified": True})
        elif section == "link":
            labels["links"].append({"id": row["ref"], "meeting": row["meeting"], "role": row["kind"],
                                    "mention": row["statement"], "engine": engine.get(row["ref"], {}),
                                    "verdict": verdict, "expected": row["correction"] or None, "verified": True})
        elif section == "proposal":
            labels["proposals"].append({"id": row["ref"], "meeting": row["meeting"], "title": row["statement"],
                                        "preselected": bool(engine.get(row["ref"], {}).get("preselected")),
                                        "verdict": verdict, "verified": True})
    questions = []
    for i, row in enumerate(rows, start=2):
        if row["section"] != "question" or row["verdict"] != "keep":
            continue
        expected, stale = [], []
        for part in (p.strip() for p in row["expected"].split("||") if p.strip()):
            bucket = stale if part.startswith("stale:") else expected
            part = part.removeprefix("stale:").strip()
            if part.startswith("A:") and part[2:] in refs:
                bucket.append(refs[part[2:]])
            elif part.startswith("Q:") and "|" in part:
                meeting, quote = part[2:].split("|", 1)
                bucket.append({"meeting": meeting.strip(), "quote": quote.strip()})
            else:
                problems.append(f"row {i}: target {part!r} is neither A:<adjudicated assertion> nor Q:<meeting>|<quote>")
        questions.append({"id": row["ref"], "question": row["statement"], "category": "adjudicated",
                          "scope": {"kinds": _split(row["kind"]), "entity": None}, "valid_at": row["valid_at"] or None,
                          "expected": expected, "stale": stale, "forbidden": [], "verified": True})
    labels["speakers"] = dict(labels["speakers"])
    if problems:
        raise ValueError("worksheet problems:\n" + "\n".join(problems))
    (directory / "labels.json").write_text(json.dumps(labels, indent=1, ensure_ascii=False) + "\n")
    history = load_history(directory)
    for label in history.labels["assertions"]:
        for e in label["evidence"]:
            try:
                _locate(history, e["meeting"], e["quote"])
            except (ValueError, StopIteration) as error:
                problems.append(f"{label['id']}: {error}")
    if problems:
        (directory / "labels.json").unlink()
        raise ValueError("worksheet problems:\n" + "\n".join(problems))
    (directory / "questions.json").write_text(json.dumps(
        {"history": manifest["history"], "question_version": 1, "note": "Adjudicated questions; every item verified.",
         "questions": questions}, indent=1, ensure_ascii=False) + "\n")
    return {"assertions": len(labels["assertions"]), "must_not": len(labels["must_not"]),
            "speakers": sum(len(v) for v in labels["speakers"].values()), "links": len(labels["links"]),
            "proposals": len(labels["proposals"]), "questions": len(questions)}


# ---------------------------------------------------------------- adjudicated scores


def score(directory: str | Path) -> dict:
    """Quality figures straight from the owner's verdicts on what the engine produced (every one verified)."""
    rows, sidecar = read_sheet(directory)
    engine = sidecar["engine"]
    kinds = defaultdict(Counter)
    links, speakers, proposals, outcome = Counter(), Counter(), Counter(), Counter()
    minutes = [float(r["minutes"]) for r in rows if r["section"] == "meeting" and r["minutes"]]
    for row in rows:
        section, verdict = row["section"], row["verdict"]
        if not verdict or verdict == "skip":
            continue
        if section == "assertion":
            kind = row["kind"] or engine.get(row["ref"], {}).get("kind") or "unknown"
            kinds[kind][verdict] += 1
            if kind == "commitment" and verdict in ("correct", "changed") and row["responsible"]:
                said = set(engine.get(row["ref"], {}).get("responsible_ids", []))
                outcome["attribution_judged"] += 1
                outcome["attribution_errors"] += set(_split(row["responsible"])) != said
        elif section == "link" and engine.get(row["ref"], {}).get("status") in AUTO_LINKS:
            links[verdict] += 1
        elif section == "speaker" and verdict != "unknown" and engine.get(row["ref"], {}).get("status") in AUTO_LINKS:
            speakers[verdict] += 1
        elif section == "proposal" and engine.get(row["ref"], {}).get("preselected"):
            proposals[verdict] += 1
    per_kind = {}
    for kind, c in sorted(kinds.items()):
        found = c["correct"] + c["changed"]
        per_kind[kind] = {**dict(c), "precision": _ratio(found, found + c["wrong"] + c["must_not"]),
                          "recall": _ratio(found, found + c["missing"]), "found": found,
                          "judged_extracted": found + c["wrong"] + c["must_not"], "labelled": found + c["missing"]}
    judged = sum(c["correct"] + c["changed"] + c["wrong"] + c["must_not"] for c in kinds.values()) + \
        links["correct"] + links["wrong"] + proposals["correct"] + proposals["wrong"]
    overturned = sum(c["changed"] + c["wrong"] + c["must_not"] for c in kinds.values()) + links["wrong"] + \
        proposals["wrong"]
    return {"series": Path(directory).name, "per_kind": per_kind,
            "links": {"correct": links["correct"] + speakers["correct"], "wrong": links["wrong"] + speakers["wrong"]},
            "speakers": dict(speakers), "preselected": {"correct": proposals["correct"], "wrong": proposals["wrong"]},
            "fabricated_actions": sum(c["must_not"] for c in kinds.values()),
            "attribution": {"errors": outcome["attribution_errors"], "judged": outcome["attribution_judged"]},
            "overturn": {"overturned": overturned, "judged": judged, "rate": _ratio(overturned, judged)},
            "correction_minutes": {"meetings": len(minutes),
                                   "per_meeting": round(sum(minutes) / len(minutes), 2) if minutes else None}}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m evals.context.worksheet", description=__doc__.split("\n\n")[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    ex = sub.add_parser("export", help="write the series directory and an empty worksheet")
    ex.add_argument("--owner", required=True)
    ex.add_argument("--out", type=Path, required=True)
    ex.add_argument("--source", action="append", dest="sources", help="a source ID (repeatable)")
    ex.add_argument("--title", help="only sources whose title contains this")
    ex.add_argument("--since", type=date.fromisoformat, help="only meetings on or after this date")
    ex.add_argument("--questions", type=Path, help="JSON list of {question, kinds?, valid_at?} to pre-fill")
    ex.add_argument("--blank-questions", type=int, default=15)
    im = sub.add_parser("import", help="turn the filled worksheet into labels.json and questions.json")
    im.add_argument("directory", type=Path)
    sc = sub.add_parser("score", help="adjudicated quality figures")
    sc.add_argument("directory", type=Path)
    args = parser.parse_args(argv)
    if args.command == "export":
        if args.out.resolve().is_relative_to(Path(__file__).resolve().parents[2]) and "outputs" not in args.out.parts:
            print("write real meeting data under outputs/ (ignored by git), not into the repository", file=sys.stderr)
            return 2
        questions = json.loads(args.questions.read_text()) if args.questions else None
        print(json.dumps(export(args.owner, args.out, source_ids=args.sources, title=args.title, since=args.since,
                                questions=questions, blank_questions=args.blank_questions)))
    elif args.command == "import":
        try:
            print(json.dumps(import_sheet(args.directory)))
        except ValueError as error:
            print(error, file=sys.stderr)
            return 2
    else:
        print(json.dumps(score(args.directory), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
