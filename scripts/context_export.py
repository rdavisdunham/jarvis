"""Export or import one owner's meeting-context history in the canonical format (jarvis.context.export).

  export:  --owner OWNER --out history.json [--include-embeddings]
  import:  --owner OWNER --in history.json [--new-ids] [--dry-run]

Export never overwrites an existing file and leaves out connection credentials. Import refuses an owner that
already has context rows and writes nothing when any check fails; `--in` also takes an account export
(GET /api/v1/export), whose `context_history` section is used. `--new-ids` replaces every context ID (needed to
copy a history into a database that still holds the original rows). `--dry-run` runs the import and rolls it back.
Only counts are printed, never record contents.
"""

import argparse
import json
import sys
from pathlib import Path

from jarvis.context.export import HistoryImportError, export_history, import_history
from jarvis.db import session_scope


def run_export(owner: str, out: Path, include_embeddings: bool) -> dict:
    if out.exists():
        raise SystemExit(f"{out} already exists; choose a new file.")
    with session_scope() as db:
        history = export_history(db, owner, include_embeddings=include_embeddings)
    out.write_text(json.dumps(history, ensure_ascii=False, indent=1, sort_keys=True), encoding="utf-8")
    counts = {k: len(v) for k, v in history.items() if isinstance(v, list)}
    return {"owner_id": owner, "out": str(out), "include_embeddings": include_embeddings, "counts": counts}


class DryRun(Exception):
    """Raised inside the transaction so a dry run rolls back."""


def run_import(owner: str, source: Path, new_ids: bool, dry_run: bool) -> dict:
    data = json.loads(source.read_text(encoding="utf-8"))
    if isinstance(data, dict) and "context_history" in data:  # an account export
        data = data["context_history"]
    try:
        with session_scope() as db:
            report = import_history(db, owner, data, preserve_ids=not new_ids)
            if dry_run:
                raise DryRun
    except DryRun:
        pass
    return {"owner_id": owner, "in": str(source), "dry_run": dry_run, "new_ids": new_ids, "counts": report.counts,
            "missing_actors": report.missing_actors}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--owner", required=True, help="Owner ID to export from or import into")
    direction = parser.add_mutually_exclusive_group(required=True)
    direction.add_argument("--out", type=Path, help="Write the owner's history to this new file")
    direction.add_argument("--in", dest="source", type=Path, help="Import a history file into the owner")
    parser.add_argument("--include-embeddings", action="store_true", help="Export: keep chunk embeddings")
    parser.add_argument("--new-ids", action="store_true", help="Import: give every row a new ID")
    parser.add_argument("--dry-run", action="store_true", help="Import: check and roll back")
    args = parser.parse_args(argv)
    if args.out and (args.new_ids or args.dry_run):
        parser.error("--new-ids and --dry-run apply to --in")
    if args.source and args.include_embeddings:
        parser.error("--include-embeddings applies to --out (an import takes whatever the file holds)")
    try:
        result = (run_export(args.owner, args.out, args.include_embeddings) if args.out
                  else run_import(args.owner, args.source, args.new_ids, args.dry_run))
    except HistoryImportError as exc:
        print(f"Import refused: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
