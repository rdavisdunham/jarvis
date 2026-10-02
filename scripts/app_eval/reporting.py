"""Evidence-first JSON/JUnit/HTML reports. Never collapse variants or retries."""

import html
import json
from collections import Counter
from decimal import Decimal
from pathlib import Path
from xml.etree import ElementTree as ET

from .catalog import load
from .ledger import Ledger

TERMINAL = {"passed", "failed", "safety_failure", "needs_review"}
FAILURES = {"failed", "safety_failure", "infra_error"}


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, default=str))
    temporary.replace(path)


def aggregate(manifest, results, imported=()):
    cases = []
    for case in manifest["cases"]:
        evidence = []
        for job in manifest["jobs"]:
            for consumer in job["consumers"]:
                if consumer["case_id"] == case["id"]:
                    result = dict(results.get(job["id"], {"status": "not_run"}))
                    selector = consumer.get("selector")
                    if selector and "tests" in result:
                        matches = [
                            status
                            for name, status in result["tests"].items()
                            if name == selector or name.startswith(selector + "[")
                        ]
                        result["status"] = (
                            "infra_error"
                            if not matches
                            else "failed"
                            if "failed" in matches
                            else "infra_error"
                            if "infra_error" in matches
                            else "blocked"
                            if "blocked" in matches
                            else "passed"
                        )
                    evidence.append(
                        {
                            **consumer,
                            "job_id": job["id"],
                            "repeat": job["repeat"],
                            "mode": job["mode"],
                            **result,
                        }
                    )
        evidence.extend(r for r in imported if r["case_id"] == case["id"])
        covered = set()
        for item in evidence:
            if item["level"] == "acceptance" and item["status"] == "passed":
                covered.update(item["criteria"])
        missing = sorted(set(case["criteria_map"]) - covered)
        acceptance = [e for e in evidence if e["level"] == "acceptance"]
        status = (
            "failed"
            if any(e["status"] in {"failed", "safety_failure"} for e in acceptance)
            else "infra_error"
            if any(e["status"] == "infra_error" for e in acceptance)
            else "needs_review"
            if any(e["status"] == "needs_review" for e in acceptance)
            else "passed"
            if not missing and acceptance and all(e["status"] == "passed" for e in acceptance)
            else "component_failed"
            if any(
                e["status"] in {"failed", "safety_failure"} and e["level"] == "component" for e in evidence
            )
            else "partial"
            if any(e["status"] == "passed" and e["level"] != "supporting" for e in evidence)
            else "blocked"
            if any(e["status"] == "blocked" for e in evidence) or case["requirement"]
            else "not_run"
        )
        cases.append(
            {
                "case_id": case["id"],
                "feature": case["feature"],
                "type": case["type"],
                "title": case["title"],
                "status": status,
                "missing_criteria": missing,
                "requirement": case["requirement"],
                "evidence": evidence,
            }
        )
    return cases


def report(directory):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    results = {}
    for path in sorted((directory / "results").glob("*.json")):
        row = json.loads(path.read_text())
        results[row["job_id"]] = row
    imported = [json.loads(p.read_text()) for p in sorted((directory / "manual").glob("*.json"))]
    cases = aggregate(manifest, results, imported)
    spending = Ledger(directory / "budget.sqlite").snapshot()
    total = len(load()[1])
    data = {
        "run_id": directory.name,
        "catalog_total": total,
        "selected": len(cases),
        "not_selected": total - len(cases),
        "summary": dict(Counter(c["status"] for c in cases)),
        "jobs": dict(Counter(r["status"] for r in results.values())),
        "by_type": grouped(cases, "type"),
        "by_feature": grouped(cases, "feature"),
        "regressions": dict(
            Counter(status for r in results.values() for status in r.get("tests", {}).values())
        ),
        "cost_by_kind": cost_groups(spending["calls"], "kind"),
        "duration_seconds": round(sum(r.get("duration_seconds", 0) for r in results.values()), 3),
        "spending": spending,
        "cases": cases,
    }
    atomic_json(directory / "report.json", data)
    atomic_json(directory / "spending.json", spending)
    suites = ET.Element("testsuites")
    suite = ET.SubElement(suites, "testsuite", name="Eridani acceptance", tests=str(len(cases)))
    for case in cases:
        node = ET.SubElement(suite, "testcase", classname=case["feature"], name=case["case_id"])
        if case["status"] == "failed":
            ET.SubElement(node, "failure", message="Acceptance assertion failed").text = json.dumps(
                case["evidence"]
            )
        elif case["status"] == "infra_error":
            ET.SubElement(node, "error", message="Infrastructure failure")
        elif case["status"] != "passed":
            ET.SubElement(node, "skipped", message=case["status"]).text = json.dumps(case["missing_criteria"])
    component_suite = ET.SubElement(
        suites, "testsuite", name="Eridani execution jobs", tests=str(len(results))
    )
    for row in results.values():
        node = ET.SubElement(
            component_suite,
            "testcase",
            classname=row["adapter"],
            name=row["target"],
            time=str(row.get("duration_seconds", 0)),
        )
        if row["status"] in {"failed", "safety_failure"}:
            ET.SubElement(node, "failure", message=row.get("reason", "Component or supporting suite failed"))
        elif row["status"] == "infra_error":
            ET.SubElement(node, "error", message=row.get("reason", "Infrastructure failure"))
        elif row["status"] != "passed":
            ET.SubElement(node, "skipped", message=row["status"])
    ET.ElementTree(suites).write(directory / "junit.xml", encoding="utf-8", xml_declaration=True)
    rows = []
    for case in cases:
        search = html.escape(
            " ".join(str(case[k]) for k in ("case_id", "type", "title", "status")).lower(), quote=True
        )
        details = html.escape(json.dumps(case, indent=2))
        rows.append(
            f'<details data-search="{search}"><summary><b>{html.escape(case["case_id"])}</b> · {html.escape(case["status"])} · {html.escape(case["title"])}</summary><pre>{details}</pre></details>'
        )
    body = """<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>Eridani eval results</title><style>body{font:15px/1.5 system-ui;background:#111820;color:#dfece9;max-width:1100px;margin:auto;padding:24px}input{width:95%;padding:12px}details{padding:12px;border-bottom:1px solid #345}pre{white-space:pre-wrap;overflow-wrap:anywhere}summary{cursor:pointer}</style>
<h1>Eridani evaluation results</h1>"""
    body += (
        "<p>"
        + html.escape(
            json.dumps(
                {k: v for k, v in data.items() if k in {"selected", "not_selected", "summary", "jobs"}}
            )
        )
        + "</p>"
    )
    body += (
        "<p>Cost: $"
        + spending["estimated_usd"]
        + " estimated; $"
        + spending["including_uncertain_usd"]
        + " including uncertain calls. Cap: $"
        + spending["cap_usd"]
        + ".</p>"
    )
    body += (
        "<h2>Execution types</h2><ul>"
        + "".join(
            "<li><strong>" + html.escape(kind) + "</strong>: " + html.escape(json.dumps(counts)) + "</li>"
            for kind, counts in data["by_type"].items()
        )
        + "</ul>"
    )
    body += (
        '<input id="filter" aria-label="Filter by feature, type or status" placeholder="Filter by feature, type or status">'
        + "".join(rows)
    )
    body += """<script>document.getElementById('filter').addEventListener('input',e=>document.querySelectorAll('details').forEach(d=>d.hidden=!d.dataset.search.includes(e.target.value.toLowerCase())))</script>"""
    (directory / "report.html").write_text(body)
    return {
        **{k: v for k, v in data.items() if k not in {"cases", "spending"}},
        "spending": {k: v for k, v in spending.items() if k != "calls"},
    }


def grouped(cases, key):
    return {
        value: dict(Counter(c["status"] for c in cases if c[key] == value))
        for value in sorted({c[key] for c in cases})
    }


def cost_groups(calls, key):
    return {
        value: {
            "requests": len(rows),
            "estimated_usd": str(sum((Decimal(r["actual_usd"] or "0") for r in rows), Decimal(0))),
            "including_uncertain_usd": str(sum((Decimal(r["charged_usd"]) for r in rows), Decimal(0))),
        }
        for value in sorted({r[key] for r in calls})
        if (rows := [r for r in calls if r[key] == value])
    }


def compare(baseline, candidate):
    manifests = [json.loads((p / "manifest.json").read_text()) for p in (baseline, candidate)]
    for key in ("catalog_sha256", "corpus_sha256", "harness_sha256", "model", "repeats", "modes"):
        if not manifests[0].get(key) or manifests[0][key] != manifests[1].get(key):
            raise ValueError("Comparison requires matching " + key)
    if sorted(c["id"] for c in manifests[0]["cases"]) != sorted(c["id"] for c in manifests[1]["cases"]):
        raise ValueError("Comparison requires the same selected cases")
    for path in (baseline, candidate):
        report(path)
    a, b = [json.loads((p / "report.json").read_text()) for p in (baseline, candidate)]
    old = {c["case_id"]: c for c in a["cases"]}
    changes = [
        {"case_id": c["case_id"], "before": old[c["case_id"]]["status"], "after": c["status"]}
        for c in b["cases"]
        if old[c["case_id"]]["status"] != c["status"]
    ]
    return {
        "baseline": str(baseline),
        "candidate": str(candidate),
        "changes": changes,
        "estimated_cost_delta_usd": str(
            Decimal(b["spending"]["estimated_usd"]) - Decimal(a["spending"]["estimated_usd"])
        ),
        "note": "Matching fixtures and graders; application revisions may differ. One repeat is not a statistically stable model ranking.",
    }
