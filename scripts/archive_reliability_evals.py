"""Archive immutable reliability eval evidence; no provider calls or production edits."""

import ast
import gzip
import hashlib
import json
from pathlib import Path

from jarvis.config import get_settings

ROOT = Path(__file__).resolve().parents[1]
DATE = "2026-09-13"
FILES = {
    "reliability-heldout-v1.json": ("reliability-heldout-aborted-v1", True),
    "reliability-heldout-v1-interruption.json": ("reliability-heldout-v1-interruption", False),
    "reliability-heldout-v1-manual-review.json": ("reliability-heldout-v1-response-review", False),
    "reliability-heldout-v1-sources.json.gz": ("reliability-heldout-v1-frozen-sources", False),
    "reliability-heldout-fixtures-v1.json.gz": ("reliability-heldout-v1-fixtures", False),
    "reliability-heldout-v2.json": ("reliability-heldout-agents", True),
    "reliability-heldout-v2-integrity.json": ("reliability-heldout-integrity", False),
    "reliability-heldout-v2-manual-review.json": ("reliability-heldout-response-review", False),
    "reliability-heldout-v2-sources.json.gz": ("reliability-heldout-frozen-sources", False),
    "reliability-heldout-fixtures-v2.json.gz": ("reliability-heldout-fixtures", False),
    "reliability-plan-reference-diagnostic-v1.json": ("reliability-plan-reference-diagnostic", True),
    "reliability-plan-reference-diagnostic-v1-integrity.json": (
        "reliability-plan-reference-integrity",
        False,
    ),
    "reliability-plan-reference-diagnostic-v1-manual-review.json": (
        "reliability-plan-reference-response-review",
        False,
    ),
    "reliability-plan-reference-diagnostic-v1-sources.json.gz": (
        "reliability-plan-reference-frozen-sources",
        False,
    ),
}
FORBIDDEN = {"reasoning_content", "reasoning_details", "encrypted_content", "thought_signature", "thought"}


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def write_immutable(path, raw):
    if path.exists():
        if path.read_bytes() != raw:
            raise FileExistsError(f"Refusing to overwrite different evidence: {path.name}")
    else:
        path.write_bytes(raw)


def forbidden_payload(value):
    if isinstance(value, dict):
        return bool(set(value) & FORBIDDEN) or any(forbidden_payload(v) for v in value.values())
    if isinstance(value, list):
        return any(forbidden_payload(v) for v in value)
    return False


def archive():
    settings = get_settings()
    secrets = []
    for name in type(settings).model_fields:
        if any(
            part in name
            for part in (
                "api_key",
                "client_secret",
                "encryption_key",
                "owner_token",
                "private_key",
                "home_assistant_token",
            )
        ):
            value = getattr(settings, name)
            if hasattr(value, "get_secret_value"):
                value = value.get_secret_value()
            if isinstance(value, str) and len(value) >= 12:
                secrets.append((name, value))
    destination = ROOT / "docs/evals"
    index = {"date": DATE, "provider_calls": 0, "raw_results_preserved": True, "artifacts": []}
    for source_name, (stem, compress) in FILES.items():
        source = ROOT / ".runtime" / source_name
        raw = source.read_bytes()
        payload = gzip.decompress(raw) if source.suffix == ".gz" else raw
        document = json.loads(payload)
        if forbidden_payload(document):
            raise ValueError(f"Native reasoning payload found in {source_name}")
        for secret_name, secret in secrets:
            if secret.encode() in payload:
                raise ValueError(f"Credential field {secret_name} found in {source_name}; no value printed")
        suffix = ".json.gz" if compress or source.suffix == ".gz" else ".json"
        target = destination / (stem + "-" + DATE + suffix)
        archived = gzip.compress(raw, mtime=0) if compress else raw
        write_immutable(target, archived)
        decoded = gzip.decompress(archived) if target.suffix == ".gz" else archived
        assert decoded == payload
        index["artifacts"].append(
            {
                "source": str(source.relative_to(ROOT)),
                "archive": str(target.relative_to(ROOT)),
                "source_sha256": digest(raw),
                "archive_sha256": digest(archived),
                "json_payload_sha256": digest(payload),
                "bytes": len(archived),
            }
        )
    full = json.loads((ROOT / ".runtime/reliability-heldout-v2.json").read_text())
    full_review = json.loads((ROOT / ".runtime/reliability-heldout-v2-manual-review.json").read_text())
    diagnostic = json.loads((ROOT / ".runtime/reliability-plan-reference-diagnostic-v1.json").read_text())
    diagnostic_review = json.loads(
        (ROOT / ".runtime/reliability-plan-reference-diagnostic-v1-manual-review.json").read_text()
    )
    assert full["state"] == diagnostic["state"] == "completed"
    assert full["database_removed"] and diagnostic["database_removed"]
    assert full_review["source_sha256"] == digest(
        (ROOT / ".runtime/reliability-heldout-v2.json").read_bytes()
    )
    assert diagnostic_review["source_sha256"] == digest(
        (ROOT / ".runtime/reliability-plan-reference-diagnostic-v1.json").read_bytes()
    )
    assert full_review["coverage"]["trials_reviewed"] == 48 and len(diagnostic_review["reviews"]) == 6
    frozen = json.loads(
        gzip.decompress(
            (ROOT / ".runtime/reliability-plan-reference-diagnostic-v1-sources.json.gz").read_bytes()
        )
    )
    changes = []
    for name, expected in frozen["manifest"]["source_sha256"].items():
        actual = (ROOT / name).read_bytes()
        if digest(actual) != expected:
            equivalent = name.endswith(".py") and ast.dump(ast.parse(actual.decode())) == ast.dump(
                ast.parse(frozen["sources"][name])
            )
            changes.append(
                {"path": name, "python_ast_identical": equivalent, "current_sha256": digest(actual)}
            )
    assert all(
        change["path"] == "apps/api/jarvis/note_schema.py" and change["python_ast_identical"]
        for change in changes
    )
    index["after_diagnostic_source_changes"] = changes
    index["note"] = (
        "Archives preserve runtime paths inside original metadata. Use this index to resolve their immutable repository copies. Source text is retained exactly as evaluated."
    )
    index["credential_scan"] = "Configured long provider/secret values absent; no values printed."
    index["native_reasoning_scan"] = (
        "No forbidden native reasoning/continuation payload keys; numerical usage details retained."
    )
    path = destination / ("reliability-heldout-artifact-index-" + DATE + ".json")
    write_immutable(path, json.dumps(index, indent=2).encode())
    print(
        json.dumps(
            {
                "archived_files": len(FILES),
                "index": str(path.relative_to(ROOT)),
                "reviewed_full_trials": 48,
                "reviewed_diagnostic_trials": 6,
                "after_diagnostic_source_changes": changes,
                "provider_calls": 0,
            }
        )
    )
    return index


if __name__ == "__main__":
    archive()
