"""Versioned acceptance catalog; scenario coverage is separate from execution coverage."""

import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CATALOG = ROOT / "evals/app"


def load():
    features = json.loads((CATALOG / "features.json").read_text())
    cases = [
        case
        for feature in features
        for case in json.loads((CATALOG / "features" / (feature["id"] + ".json")).read_text())
    ]
    return features, cases


def digest():
    paths = sorted(CATALOG.rglob("*.json"))
    return hashlib.sha256(
        b"".join(p.relative_to(CATALOG).as_posix().encode() + p.read_bytes() for p in paths)
    ).hexdigest()


def validate():
    features, cases = load()
    errors = []
    ids = Counter(c["id"] for c in cases)
    feature_ids = {f["id"] for f in features}
    for identity, n in ids.items():
        if n != 1:
            errors.append(f"Duplicate case: {identity}")
    for feature in features:
        subset = [c for c in cases if c["feature"] == feature["id"]]
        if not 25 <= len(subset) <= 50:
            errors.append(f"{feature['id']}: expected 25–50 cases, found {len(subset)}")
        for path in feature["source_files"] + feature["regression_files"]:
            if not (ROOT / path).is_file():
                errors.append(f"Missing source: {path}")
        if len({c["title"] for c in subset}) != len(subset):
            errors.append(f"Duplicate scenario title in {feature['id']}")
    for case in cases:
        if case["feature"] not in feature_ids:
            errors.append(f"Unknown feature: {case['id']}")
        for field in ("inputs", "expected", "invariants", "setup", "steps", "evidence"):
            if not case.get(field):
                errors.append(f"{case['id']}: missing {field}")
        if case["partition"] not in {"development", "holdout"}:
            errors.append(f"Unknown partition: {case['id']}")
        if case["execution"]["status"] != "not_run":
            errors.append(f"Results belong in run artifacts, not catalog: {case['id']}")
    if errors:
        raise ValueError("\n".join(errors))
    return {"features": len(features), "cases": len(cases), "sha256": digest()}


def report(results=()):
    features, cases = load()
    # A result is for a specific layer. Contract success never marks the model/browser scenario passed.
    supplied = {r["case_id"]: r for r in results if r.get("layer") == "acceptance"}
    return {
        "catalog_sha256": digest(),
        "features": features,
        "cases": [{**c, "result": supplied.get(c["id"], {"status": "not_run"})} for c in cases],
        "component_results": list(results),
        "summary": dict(Counter(supplied.get(c["id"], {}).get("status", "not_run") for c in cases)),
    }


def html_report(destination, results=()):
    import html

    data = report(results)
    rows = []
    if results:
        counts = Counter(r.get("status", "not_run") for r in results)
        rows.append(
            "<h2>Component run</h2><p>"
            + html.escape(str(dict(counts)))
            + "</p><p>These results check a specific layer; the full acceptance protocols below remain unrun.</p>"
        )
        for result in results:
            rows.append(
                "<p><code>"
                + html.escape(result.get("case_id", ""))
                + "</code> · "
                + html.escape(result.get("layer", ""))
                + " · <strong>"
                + html.escape(result.get("status", "not_run"))
                + "</strong></p>"
            )
    for f in data["features"]:
        rows.append(
            f"<h2>{html.escape(f['title'])} <small>{sum(c['feature'] == f['id'] for c in data['cases'])} scenarios</small></h2><p>{html.escape(f['description'])}</p>"
        )
        for c in (c for c in data["cases"] if c["feature"] == f["id"]):
            details = "".join(
                f"<h4>{field.capitalize()}</h4><ul>"
                + "".join("<li>" + html.escape(str(v)) + "</li>" for v in c[field])
                + "</ul>"
                for field in ("setup", "inputs", "steps", "expected", "invariants", "evidence")
            )
            rows.append(
                f'<details data-search="{html.escape((c["id"] + " " + c["title"] + " " + f["title"]).lower(), quote=True)}"><summary><code>{c["id"]}</code> {html.escape(c["title"])} <small>{c["result"]["status"]} · {c["partition"]}</small></summary>{details}</details>'
            )
    body = (
        """<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Eridani evaluation catalog</title><style>
body{font:16px/1.5 system-ui;background:#111820;color:#dfebe9;max-width:1050px;margin:auto;padding:24px}
h1,h2{color:#b9e0d5}h2{margin-top:40px}small{color:#9cabba;font-size:12px}
input{box-sizing:border-box;width:100%;padding:14px;background:#233039;border:1px solid #58766b;color:white;border-radius:12px;position:sticky;top:8px}
details{padding:12px;border-bottom:1px solid #34424b}summary{cursor:pointer}code{color:#b9cafa}li{margin:4px 0}
</style><h1>Eridani · 1,001 evaluation scenarios</h1>
<p>40 feature areas; 25–26 cases each. Synthetic Rowan corpus. These are acceptance specifications; unexecuted cases are not passes. Automated component evidence is reported separately.</p>
<input aria-label="Filter scenarios" placeholder="Search a feature, case ID, or scenario…" id="search">
"""
        + "".join(rows)
        + """
<script>document.getElementById('search').addEventListener('input',e=>{const q=e.target.value.toLowerCase();document.querySelectorAll('details').forEach(d=>d.hidden=!d.dataset.search.includes(q))})</script>"""
    )
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(body)
    destination.with_suffix(".json").write_text(json.dumps(data, indent=2))
