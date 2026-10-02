"""Single-trial process entrypoint. No application globals cross trial boundaries."""

import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path
from xml.etree import ElementTree as ET

from .catalog import ROOT
from .environment import environment, validate_url
from .ledger import Ledger
from .reporting import atomic_json
from .spending import EvalLimit


def main(path):
    config = json.loads(Path(path).read_text())
    job, attempt = config["job"], Path(config["attempt"])
    url = config["database_url"]
    validate_url(url, trial=True)
    result = {"status": "infra_error"}
    trace = []
    try:
        from .campaign import fingerprint

        if config.get("fingerprint") and config["fingerprint"] != fingerprint():
            raise ValueError("Campaign source changed before worker execution")
        adapter = job["adapter"]
        with environment(url, live=job["mode"] == "live-model", providers=("OPENAI",)):
            if adapter == "contract":
                from .contracts import CASES, Harness, private_state, snapshot
                from .offline import offline

                h = Harness(config["fixture"])
                with offline():
                    before = snapshot()
                    try:
                        CASES[job["target"]](h, job["target"])
                        result["status"] = "passed"
                    except AssertionError as exc:
                        import traceback

                        result.update(
                            status="failed",
                            reason=str(exc),
                            stack=[
                                {"file": f.filename, "line": f.lineno, "function": f.name}
                                for f in traceback.extract_tb(exc.__traceback__)
                            ],
                        )
                    after = snapshot()
                    if private_state(before) != private_state(after):
                        result.update(status="safety_failure", reason="Foreign account changed")
                    atomic_json(attempt / "state.json", {"before": before, "after": after, "tools": h.trace})
            elif adapter == "agent":
                from jarvis.agent_models import catalog
                from .model_runner import PROBES, execute_probe
                from .transport import metered

                if not catalog()["luna"].available:
                    result.update(status="blocked", reason="OPENAI_API_KEY unavailable")
                else:
                    with metered(Ledger(Path(config["directory"]) / "budget.sqlite"), job["id"], trace):
                        details = asyncio.run(
                            execute_probe(
                                job["target"], PROBES[job["target"]], "luna", config["fixture"], trace
                            )
                        )
                    atomic_json(attempt / "state.json", details)
                    result.update({k: v for k, v in details.items() if k not in {"before", "after", "reply"}})
            elif adapter in {"pytest", "browser"}:
                env = dict(os.environ)
                env["PYTHONPATH"] = str(ROOT) + os.pathsep + str(ROOT / "apps/api")
                env.update(
                    JARVIS_OWNER_ID="davin",
                    JARVIS_OWNER_NAME="Davis",
                    JARVIS_OPENAI_API_KEY="synthetic-openai",
                    JARVIS_GEMINI_API_KEY="synthetic-gemini",
                    JARVIS_GROQ_API_KEY="synthetic-groq",
                    JARVIS_WORKER_ENABLED="true"
                    if job["target"] == "tests/test_worker_recovery.py"
                    else "false",
                    JARVIS_EXTERNAL_SERVICES_ENABLED="true" if adapter == "pytest" else "false",
                    ERIDANI_EVAL_OWNED_DB=url,
                )
                if adapter == "pytest":
                    command = [
                        sys.executable,
                        "-m",
                        "pytest",
                        "-p",
                        "evals.offline",
                        "-q",
                        "--tb=short",
                        "--junitxml=" + str(attempt / "junit.xml"),
                    ]
                    command += (
                        [job["target"]]
                        if "" in job.get("selectors", [""])
                        else [job["target"] + "::" + selector for selector in job["selectors"]]
                    )
                else:
                    env["JARVIS_BROWSER_FIXTURE"] = job["target"]
                    env["ERIDANI_EVAL_ARTIFACT_DIR"] = str(attempt / "browser")
                    command = [sys.executable, "scripts/validate_custom_planner.py"]
                with (attempt / "suite.log").open("w") as log:
                    completed = subprocess.run(
                        command, cwd=ROOT, env=env, stdout=log, stderr=log, check=False
                    )
                result.update(
                    status="passed"
                    if completed.returncode == 0
                    else "failed"
                    if completed.returncode == 1
                    else "infra_error",
                    returncode=completed.returncode,
                )
                if adapter == "pytest" and (attempt / "junit.xml").exists():
                    root = ET.parse(attempt / "junit.xml").getroot()
                    nodes = {}
                    for node in root.iter("testcase"):
                        nodes[node.attrib["name"]] = (
                            "failed"
                            if node.find("failure") is not None
                            else "infra_error"
                            if node.find("error") is not None
                            else "blocked"
                            if node.find("skipped") is not None
                            else "passed"
                        )
                    result["tests"] = nodes
                    if nodes and all(s == "blocked" for s in nodes.values()):
                        result["status"] = "blocked"
            elif adapter == "pipeline":
                from .pipelines import execute

                from .judge import external_grading

                with external_grading(attempt if config.get("judge") == "external" else None):
                    result.update(execute(config, trace))
            elif adapter == "connected":
                from .connected import execute

                result.update(execute(config, trace))
            else:
                raise ValueError("Unknown adapter")
    except EvalLimit as exc:
        result.update(status="not_completed", reason=str(exc))
    except AssertionError as exc:
        result.update(status="failed", reason=str(exc))
    except Exception as exc:
        import traceback

        result.update(
            status="infra_error",
            reason=type(exc).__name__,
            stack=[
                {"file": f.filename, "line": f.lineno, "function": f.name}
                for f in traceback.extract_tb(exc.__traceback__)
            ],
        )
    finally:
        if config.get("fingerprint") and config["fingerprint"] != fingerprint():
            result.update(status="infra_error", reason="Source fingerprint changed during campaign")
        atomic_json(attempt / "trace.json", trace)
        atomic_json(attempt / "result.json", result)


if __name__ == "__main__":
    main(sys.argv[1])
