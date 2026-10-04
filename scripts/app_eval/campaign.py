"""On-demand campaign coordinator. Workers never share application globals."""

import argparse
import concurrent.futures
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from .catalog import ROOT, digest
from .environment import DEFAULT_URL, corpus_hash, ensure_corpus, settings_env, trial_database, validate_url
from .ledger import Ledger
from .registry import MODES, coverage, jobs, select, validate_registry
from .reporting import TERMINAL, atomic_json, report


def fingerprint():
    paths = sorted(
        {
            p
            for folder in (
                "scripts/app_eval",
                "apps/api/jarvis",
                "apps/web/src",
                "apps/web/e2e",
                "tests",
                "evals",
            )
            for p in (ROOT / folder).rglob("*")
            if p.is_file() and p.suffix in {".py", ".json", ".mjs", ".ts", ".tsx", ".md", ".wav"}
        }
    )
    paths = [
        p
        for p in paths
        if not (
            p.is_relative_to(ROOT / "evals/app")
            and (
                p.suffix == ".md"
                and p.name != "protocols.md"
                or p.name
                in {"automation-coverage.json", "baseline-2026-09-19.json", "paid-baseline-2026-09-19.json"}
            )
        )
    ]
    paths.extend(
        ROOT / name
        for name in (
            "uv.lock",
            "apps/web/package-lock.json",
            "scripts/validate_custom_planner.py",
            "scripts/summarize_latency.py",
            "scripts/expert_eval_cases.py",
            "scripts/reliability_eval_cases.py",
        )
    )
    return hashlib.sha256(
        b"".join(p.relative_to(ROOT).as_posix().encode() + p.read_bytes() for p in paths)
    ).hexdigest()


def config_fingerprint(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest() if path else None


def harness_fingerprint():
    # Application source changes are what comparisons measure; fixture/oracle
    # changes instead invalidate a like-for-like comparison.
    paths = sorted(
        p
        for folder in ("scripts/app_eval", "evals", "tests", "apps/web/e2e")
        for p in (ROOT / folder).rglob("*")
        if p.is_file()
        and p.suffix in {".py", ".json", ".mjs"}
        and p.name
        not in {"automation-coverage.json", "baseline-2026-09-19.json", "paid-baseline-2026-09-19.json"}
    )
    paths.append(ROOT / "scripts/summarize_latency.py")
    return hashlib.sha256(
        b"".join(p.relative_to(ROOT).as_posix().encode() + p.read_bytes() for p in paths)
    ).hexdigest()


def plan(args):
    validate_url(args.database_url)
    summary = validate_registry()
    rows = select(args.types, args.features, args.cases)
    modes = args.mode.split(",")
    if not set(modes) <= set(MODES):
        raise ValueError("Unknown execution mode")
    work = jobs(rows, modes, args.repeats, not args.no_support)
    if getattr(args, "service_tier", None) and any(j["adapter"] != "agent" for j in work):
        raise ValueError("Service-tier comparison requires agent-only jobs (--no-support)")
    return {
        "version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "fingerprint": fingerprint(),
        "catalog_sha256": digest(),
        "harness_sha256": harness_fingerprint(),
        "corpus_sha256": corpus_hash(),
        "registry": summary,
        "cases": rows,
        "jobs": work,
        "modes": modes,
        "cap_usd": args.max_usd,
        "judge_cap_usd": 0 if getattr(args, "judge", "luna") == "external" else min(1, args.max_usd),
        "judge": getattr(args, "judge", "luna"),
        "max_provider_requests": args.max_provider_requests,
        "repeats": args.repeats,
        "workers": args.workers,
        "model": args.model,
        "service_tier": getattr(args, "service_tier", None),
        "database_url": args.database_url,
        "live_config": str(Path(args.live_config).resolve()) if args.live_config else None,
        "live_config_sha256": config_fingerprint(args.live_config),
    }


def summary(manifest):
    from collections import Counter

    return {
        "selected_cases": len(manifest["cases"]),
        "jobs": len(manifest["jobs"]),
        "jobs_by_adapter": dict(Counter(j["adapter"] for j in manifest["jobs"])),
        "fully_bound_cases": sum(c["requirement"] is None for c in manifest["cases"]),
        "requirements": dict(
            Counter(c["requirement"]["kind"] for c in manifest["cases"] if c["requirement"])
        ),
        "modes": manifest["modes"],
        "cap_usd": manifest["cap_usd"],
        "model": manifest["model"],
        "service_tier": manifest.get("service_tier"),
        "workers": manifest["workers"],
        "paid_jobs": sum(j["mode"] == "live-model" for j in manifest["jobs"]),
        "connected_jobs": sum(j["mode"] == "live-service" for j in manifest["jobs"]),
        "note": "Supporting suite success does not pass an acceptance scenario. Paid jobs reserve actual request bounds; no fixed full-run price guarantee.",
    }


@contextmanager
def lock(directory):
    path = directory / "RUNNING"
    try:
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        pid = int(path.read_text())
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            path.unlink()
            descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        else:
            raise ValueError("Campaign coordinator is still running; resume refused") from exc
    os.write(descriptor, str(os.getpid()).encode())
    os.close(descriptor)
    try:
        yield
    finally:
        path.unlink(missing_ok=True)


def clean_env():
    # No accidental production settings, proxies or credentials in child processes.
    keep = {
        "PATH",
        "HOME",
        "USER",
        "LANG",
        "LC_ALL",
        "VIRTUAL_ENV",
        "PYTHONPATH",
        "DISPLAY",
        "XDG_RUNTIME_DIR",
        "PLAYWRIGHT_BROWSERS_PATH",
        "LD_LIBRARY_PATH",
    }
    return {k: v for k, v in os.environ.items() if k in keep or k.startswith("ERIDANI_EVAL_")}


def execute(directory, manifest, job):
    output = directory / "results" / (job["id"] + ".json")
    previous = json.loads(output.read_text()) if output.exists() else None
    if previous and previous["status"] in TERMINAL:
        return {**previous, "reused": True}
    started = time.monotonic()
    result = {
        "job_id": job["id"],
        "adapter": job["adapter"],
        "target": job["target"],
        "mode": job["mode"],
        "repeat": job["repeat"],
        "status": "infra_error",
    }
    if (directory / "STOP").exists():
        result.update(status="not_completed", reason="STOP file present")
        atomic_json(output, result)
        return result
    attempts = directory / "attempts" / job["id"]
    attempts.mkdir(parents=True, exist_ok=True)
    from .resources import mark_trial, reap

    attempt = attempts / (str(len(list(attempts.iterdir())) + 1).zfill(3))
    attempt.mkdir()
    process = None
    try:
        for old in attempts.iterdir():
            if old != attempt:
                reap(old, directory, job["id"])
        with trial_database(manifest["database_url"]) as (url, fixture):
            mark_trial(url, directory, job["id"], attempt)
            config = {
                "job": job,
                "database_url": url,
                "fixture": fixture,
                "directory": str(directory),
                "attempt": str(attempt),
                "live_config": manifest.get("live_config"),
                "judge": manifest.get("judge", "luna"),
                "model": manifest["model"],
                "service_tier": manifest.get("service_tier"),
                "fingerprint": manifest["fingerprint"],
            }
            atomic_json(attempt / "input.json", config)
            env = {**clean_env(), **settings_env(url)}
            env.update(PYTHONUNBUFFERED="1", ERIDANI_EVAL_OWNED_DB=url)
            with (attempt / "worker.log").open("w") as log:
                process = subprocess.Popen(
                    [sys.executable, "-m", "scripts.app_eval.worker", str(attempt / "input.json")],
                    cwd=ROOT,
                    env=env,
                    stdout=log,
                    stderr=log,
                    start_new_session=True,
                )
                (attempt / "worker.pid").write_text(str(process.pid))
                deadline = time.monotonic() + (900 if job["adapter"] in {"pytest", "browser"} else 240)
                while process.poll() is None:
                    if (directory / "STOP").exists() or time.monotonic() > deadline:
                        os.killpg(process.pid, signal.SIGTERM)
                        try:
                            process.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            os.killpg(process.pid, signal.SIGKILL)
                            process.wait(timeout=10)
                        result.update(
                            status="not_completed", reason="STOP requested or worker deadline reached"
                        )
                        break
                    time.sleep(0.1)
                evidence = attempt / "result.json"
                if evidence.exists() and result["status"] != "not_completed":
                    result.update(json.loads(evidence.read_text()))
                elif result["status"] != "not_completed":
                    result["reason"] = f"Worker exited {process.returncode} without evidence"
    except Exception as exc:
        result.update(status="infra_error", reason=type(exc).__name__)
    finally:
        if process and process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=10)
    result.update(
        duration_seconds=round(time.monotonic() - started, 3),
        evidence_directory=str(attempt.relative_to(directory)),
    )
    atomic_json(attempt / "summary.json", result)
    atomic_json(output, result)
    return result


def run(directory, manifest):
    import threading

    paid, browser = threading.Semaphore(2), threading.Semaphore(2)

    def gated(job):
        semaphore = paid if job["mode"] != "offline" else browser if job["adapter"] == "browser" else None
        if semaphore:
            with semaphore:
                return execute(directory, manifest, job)
        return execute(directory, manifest, job)

    with lock(directory):
        ensure_corpus(manifest["database_url"])
        with concurrent.futures.ThreadPoolExecutor(max_workers=manifest["workers"]) as pool:
            futures = {pool.submit(gated, job): job for job in manifest["jobs"]}
            try:
                for future in concurrent.futures.as_completed(futures):
                    row = future.result()
                    print(
                        json.dumps(
                            {
                                k: row.get(k)
                                for k in (
                                    "job_id",
                                    "adapter",
                                    "target",
                                    "status",
                                    "duration_seconds",
                                    "reused",
                                )
                            }
                        ),
                        flush=True,
                    )
            except KeyboardInterrupt:
                (directory / "STOP").touch()
                print("Stopping workers; uncertain charges remain reserved.", flush=True)
        result = report(directory)
    print(json.dumps(result, indent=2))
    return exit_status(result)


def exit_status(result):
    failures = {"failed", "safety_failure", "infra_error", "component_failed"}
    if any(result[part].get(status, 0) for part in ("jobs", "summary") for status in failures):
        return 1
    if any(count for status, count in result["summary"].items() if status != "passed"):
        return 2
    return 0


def import_evidence(directory, source):
    manifest = json.loads((directory / "manifest.json").read_text())
    item = json.loads(Path(source).read_text())
    required = {"case_id", "criteria", "status", "observed_at", "observer", "device", "commit", "attachments"}
    if not required <= item.keys() or item["status"] not in {"passed", "failed", "needs_review"}:
        raise ValueError("Incomplete manual evidence")
    case = next((c for c in manifest["cases"] if c["id"] == item["case_id"]), None)
    if not case or not item["criteria"] or not set(item["criteria"]) <= case["criteria_map"].keys():
        raise ValueError("Unknown case/criteria")
    if item["commit"] != manifest["commit"] or item.get("fingerprint") != manifest["fingerprint"]:
        raise ValueError("Evidence must match the campaign source fingerprint and commit")
    if any(not isinstance(item[k], str) or not item[k].strip() for k in ("observer", "device")):
        raise ValueError("Observer and device descriptions are required")
    observed = datetime.fromisoformat(item["observed_at"])
    if (
        observed.tzinfo is None
        or observed < datetime.fromisoformat(manifest["created_at"])
        or observed > datetime.now(UTC)
    ):
        raise ValueError("Evidence must be timezone-aware and observed during this campaign")
    if not item["attachments"] or any(
        not (directory / p).resolve().is_relative_to(directory.resolve()) or not (directory / p).is_file()
        for p in item["attachments"]
    ):
        raise ValueError("Evidence attachments must exist inside this campaign")
    item.update(level="acceptance", variant="manual", evidence_origin="human", repeat=1)
    (directory / "manual").mkdir(exist_ok=True)
    atomic_json(directory / "manual" / (uuid4().hex + ".json"), item)
    return report(directory)


def coverage_regressions(baseline, current):
    """Ratchet: fully bound cases may grow but never shrink or silently become unbound."""
    problems = []
    before, after = baseline["registry"]["fully_bound"], current["registry"]["fully_bound"]
    if after < before:
        problems.append(f"fully_bound total dropped from {before} to {after}")
    for feature, old in sorted(baseline["features"].items()):
        new = current["features"].get(feature)
        if new is None:
            problems.append(f"{feature}: feature missing (baseline fully_bound={old['fully_bound']})")
            continue
        if new["fully_bound"] < old["fully_bound"]:
            problems.append(f"{feature}: fully_bound dropped from {old['fully_bound']} to {new['fully_bound']}")
        # A case that was bound at baseline is absent from the baseline backlog. New catalog
        # entries are also absent, so only enforce per-case identity when the catalog did not grow.
        if new["catalog"] <= old["catalog"]:
            was_unbound = {r["case_id"] for r in old["remaining"]}
            regressed = sorted({r["case_id"] for r in new["remaining"]} - was_unbound)
            if regressed:
                problems.append(f"{feature}: previously bound cases are now unbound: {', '.join(regressed)}")
    return problems


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("coverage")
    p.add_argument("--output")
    p.add_argument(
        "--baseline",
        help="Committed coverage JSON; exit non-zero if fully bound coverage regresses against it.",
    )
    p = sub.add_parser("compare")
    p.add_argument("baseline")
    p.add_argument("candidate")
    p.add_argument("--output")
    for command in ("plan", "run"):
        p = sub.add_parser(command)
        p.add_argument("--database-url", default=DEFAULT_URL)
        p.add_argument("--types", default="all")
        p.add_argument("--features")
        p.add_argument("--cases")
        p.add_argument("--mode", default="offline")
        p.add_argument("--model", choices=["luna", "luna-none"], default="luna")
        p.add_argument("--service-tier", choices=["default", "fast"], help="Explicit tier for agent evals only; production settings are untouched.")
        p.add_argument("--judge", choices=["luna", "external"], default="luna")
        p.add_argument("--max-usd", type=float, default=10)
        p.add_argument("--max-provider-requests", type=int, default=10000)
        p.add_argument("--repeats", type=int, default=1)
        p.add_argument("--workers", type=int, default=4)
        p.add_argument("--no-support", action="store_true")
        p.add_argument("--run-paid", action="store_true")
        p.add_argument("--live-config")
        p.add_argument("--output")
    for command in ("resume", "report", "import-evidence"):
        p = sub.add_parser(command)
        p.add_argument("directory")
        if command == "import-evidence":
            p.add_argument("evidence")
    args = parser.parse_args(argv)
    if args.command == "coverage":
        data = coverage()
        if args.output:
            atomic_json(Path(args.output), data)
        print(json.dumps(data["registry"], indent=2))
        if args.baseline:
            problems = coverage_regressions(json.loads(Path(args.baseline).read_text()), data)
            for problem in problems:
                print(f"coverage regression: {problem}", file=sys.stderr)
            return int(bool(problems))
        return 0
    if args.command == "compare":
        from .reporting import compare

        data = compare(Path(args.baseline), Path(args.candidate))
        if args.output:
            atomic_json(Path(args.output), data)
        print(json.dumps(data, indent=2))
        return 0
    if args.command in {"report", "import-evidence", "resume"}:
        directory = Path(args.directory).resolve()
        if args.command == "report":
            print(json.dumps(report(directory), indent=2))
            return 0
        if args.command == "import-evidence":
            print(json.dumps(import_evidence(directory, args.evidence), indent=2))
            return 0
        manifest = json.loads((directory / "manifest.json").read_text())
        if manifest["fingerprint"] != fingerprint():
            raise ValueError("Source/fixture/grader changes require a new campaign; resume refused")
        if manifest.get("live_config_sha256") != config_fingerprint(manifest.get("live_config")):
            raise ValueError("Live resource configuration changed; resume refused")
        return run(directory, manifest)
    if not 1 <= args.workers <= 8 or not 1 <= args.repeats <= 10 or not 0 < args.max_usd <= 10:
        parser.error("Use 1–8 workers, 1–10 repeats and a positive allowance no larger than $10")
    manifest = plan(args)
    print(json.dumps(summary(manifest), indent=2))
    if args.command == "plan":
        if args.output:
            atomic_json(Path(args.output), manifest)
        return 0
    if any(j["mode"] == "live-model" for j in manifest["jobs"]) and not args.run_paid:
        parser.error("Paid execution requires --run-paid; planning never spends money")
    directory = (
        Path(args.output).resolve()
        if args.output
        else ROOT
        / "artifacts/app-evals"
        / (datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8])
    )
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "results").mkdir()
    atomic_json(directory / "manifest.json", manifest)
    Ledger(directory / "budget.sqlite").initialize(
        args.max_usd, manifest["judge_cap_usd"], args.max_provider_requests
    )
    print("Campaign: " + str(directory), flush=True)
    return run(directory, manifest)


if __name__ == "__main__":
    raise SystemExit(main())
