"""CLI for the app acceptance catalog, synthetic corpus and executable checks."""

import argparse
import json
import os
import subprocess
import sys
import time
import traceback
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from .catalog import ROOT, html_report, load, validate
from .environment import DEFAULT_URL, ensure_corpus, environment, settings_env, trial_database


def artifacts(path=None):
    p = (
        Path(path)
        if path
        else ROOT
        / "artifacts/app-evals"
        / (datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8])
    )
    p.mkdir(parents=True, exist_ok=False)
    import hashlib

    from .catalog import digest

    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout.strip()
    sources = {}
    for folder in ("scripts/app_eval", "apps/api/jarvis"):
        for source in sorted((ROOT / folder).glob("*.py")):
            sources[source.relative_to(ROOT).as_posix()] = hashlib.sha256(source.read_bytes()).hexdigest()
    (p / "run-context.json").write_text(
        json.dumps(
            {
                "app_commit": revision,
                "catalog_sha256": digest(),
                "source_hashes": sources,
                "started_at": datetime.now(UTC).isoformat(),
                "synthetic_only": True,
            },
            indent=2,
        )
    )
    return p


def selected(available, requested):
    if not requested:
        return sorted(available)
    wanted = [x.strip() for x in requested.split(",")]
    missing = set(wanted) - set(available)
    if missing:
        raise ValueError("Unavailable cases: " + ", ".join(sorted(missing)))
    return wanted


def contracts(url, ids, output):
    from .contracts import CASES, Harness, private_state, snapshot

    results = []
    for identity in selected(CASES, ids):
        started = time.monotonic()
        result = {"case_id": identity, "layer": "command_contract", "status": "infra_error"}
        try:
            with trial_database(url) as (trial, fixture), environment(trial):
                h = Harness(fixture)
                before = snapshot()
                try:
                    CASES[identity](h, identity)
                    after = snapshot()
                    assert private_state(before) == private_state(after), "Foreign account data changed"
                    result.update(
                        status="passed", checks="Declared component assertions and private-account snapshot"
                    )
                except AssertionError as exc:
                    after = snapshot()
                    result.update(status="failed", error=str(exc) or "State assertion failed")
                    if private_state(before) != private_state(after):
                        result["status"] = "safety_failure"
                    result["traceback"] = traceback.format_exc()
                result["evidence"] = {"before": before, "after": after, "tools": h.trace}
        except Exception as exc:  # noqa: BLE001 - classify infrastructure failures, retain other trials
            result.update(status="infra_error", error=type(exc).__name__, traceback=traceback.format_exc())
        result["duration_seconds"] = round(time.monotonic() - started, 3)
        (output / (identity + ".json")).write_text(json.dumps(result, indent=2, default=str))
        results.append({k: v for k, v in result.items() if k not in {"evidence", "traceback"}})
        print(json.dumps(results[-1]), flush=True)
    (output / "results.json").write_text(json.dumps(results, indent=2))
    html_report(output / "catalog.html", results)
    return 0 if all(r["status"] == "passed" for r in results) else 1


def regressions(url, output, scope):
    # The existing suites create/delete their own UUID databases. Never point them at prod.
    from .environment import validate_url

    validate_url(url)
    env = {
        **{k: v for k, v in os.environ.items() if not k.startswith("JARVIS_")},
        **settings_env(url),
        "JARVIS_OWNER_ID": "davin",
        "JARVIS_OWNER_NAME": "Davis",
    }
    from sqlalchemy.engine import make_url

    # Existing pytest/browser runners create their own disposable DBs. Connect their admin to
    # postgres, not the corpus, so it remains cloneable while regressions are running.
    env["JARVIS_DATABASE_URL"] = make_url(url).set(database="postgres").render_as_string(hide_password=False)
    env.update(
        JARVIS_EXTERNAL_SERVICES_ENABLED="true",
        JARVIS_OPENAI_API_KEY="synthetic-openai",
        JARVIS_GEMINI_API_KEY="synthetic-gemini",
        JARVIS_GROQ_API_KEY="synthetic-groq",
    )
    env.pop("JARVIS_WORKER_ENABLED", None)  # recovery tests explicitly launch their isolated worker
    commands = {
        "backend": [
            sys.executable,
            "-m",
            "pytest",
            "-p",
            "evals.offline",
            "-q",
            "--tb=short",
            "--junitxml=" + str(output / "backend.xml"),
            "tests",
        ],
        "frontend": ["npm", "test", "--", "--run"],
        "browser": [sys.executable, "scripts/validate_custom_planner.py"],
    }
    results = []
    for kind in list(commands) if scope == "all" else [scope]:
        start = time.monotonic()
        cwd = ROOT / "apps/web" if kind == "frontend" else ROOT
        with (output / (kind + ".log")).open("w") as log:
            child_env = dict(env)
            if kind == "browser":
                child_env.update(JARVIS_WORKER_ENABLED="false", JARVIS_EXTERNAL_SERVICES_ENABLED="false")
            proc = subprocess.run(
                commands[kind], cwd=cwd, env=child_env, stdout=log, stderr=subprocess.STDOUT, check=False
            )
        item = {
            "layer": "existing_regressions",
            "suite": kind,
            "status": "passed" if proc.returncode == 0 else "failed",
            "returncode": proc.returncode,
            "duration_seconds": round(time.monotonic() - start, 2),
        }
        print(json.dumps(item), flush=True)
        results.append(item)
    (output / "regressions.json").write_text(json.dumps(results, indent=2))
    return int(any(r["status"] != "passed" for r in results))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database-url", default=DEFAULT_URL, help="Dedicated local corpus URL only.")
    sub = parser.add_subparsers(dest="mode", required=True)
    sub.add_parser("validate")
    sub.add_parser("seed")
    p = sub.add_parser("catalog")
    p.add_argument("--output", default="artifacts/app-evals/catalog.html")
    p = sub.add_parser("case")
    p.add_argument("id")
    p = sub.add_parser("contracts")
    p.add_argument("--cases")
    p.add_argument("--output")
    p = sub.add_parser("regressions")
    p.add_argument("--scope", choices=["backend", "frontend", "browser", "all"], default="all")
    p.add_argument("--output")
    p = sub.add_parser("models")
    p.add_argument("--cases")
    p.add_argument("--models", default="luna,gemini")
    p.add_argument("--run-paid", action="store_true")
    p.add_argument("--max-provider-requests", type=int, default=20)
    p.add_argument("--max-usd-per-model", type=float, default=2.0)
    p.add_argument("--output")
    p.add_argument("--repeats", type=int, default=1)
    args = parser.parse_args()
    if args.mode == "validate":
        print(json.dumps(validate()))
        return 0
    if args.mode == "case":
        _, cases = load()
        match = next((c for c in cases if c["id"] == args.id), None)
        if not match:
            parser.error("Unknown case")
        print(json.dumps(match, indent=2))
        return 0
    if args.mode == "catalog":
        validate()
        html_report(args.output)
        print(args.output)
        return 0
    if args.mode == "seed":
        data = ensure_corpus(args.database_url)
        print(json.dumps({k: v for k, v in data.items() if k != "refs"}, indent=2))
        return 0
    output = artifacts(args.output)
    print("Evidence directory: " + str(output), flush=True)
    if args.mode == "contracts":
        return contracts(args.database_url, args.cases, output)
    if args.mode == "regressions":
        return regressions(args.database_url, output, args.scope)
    if args.mode == "models":
        if not args.run_paid:
            parser.error(
                "Real inference requires explicit --run-paid; catalog/contract checks never spend API credits."
            )
        if not 1 <= args.max_provider_requests <= 1000:
            parser.error("Provider request limit must be 1–1000")
        if not 1 <= args.repeats <= 10:
            parser.error("Repeats must be 1–10")
        from .model_runner import run

        return run(args, output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
