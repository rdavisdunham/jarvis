"""Executable evidence registry. Related regressions are never acceptance evidence."""

import ast
import hashlib
import json
from collections import Counter

from .catalog import CATALOG, ROOT, load
from .contracts import CASES
from .model_runner import PROBES

GROUPS = {
    "contracts": "quick_capture onboarding task_capture task_edit task_lifecycle time_deadlines routines organization custom_schema custom_records planner_views planning notifications accounts workspaces privacy operations budget",
    "agent": "queue clarifications receipts chat_agent",
    "pipeline": "notes note_tasks note_lists note_organization search search_aliases memory_capture memory_management memory_dream routing_rules routing_dream",
    "integration": "google_sync google_writes linear_sync linear_writes external_agents",
    "browser": "site_controls browser_settings",
    "voice": "live_voice wake_shutdown",
}
FEATURE_TYPE = {feature: group for group, features in GROUPS.items() for feature in features.split()}
MODES = ("offline", "live-model", "live-service")
BROWSERS = {
    "quick_capture":["batch-c.mjs"], "onboarding":["batch-c.mjs"],
    "planner_views": ["custom-planner.mjs"],
    "browser_settings": ["shell-navigation.mjs", "usage-report.mjs"],
    "site_controls": ["custom-planner.mjs", "shell-navigation.mjs"],
    "note_lists": ["note-lists.mjs"],
    "note_organization": ["note-lists.mjs"],
    "search": ["semantic-search.mjs"],
    "search_aliases": ["semantic-search.mjs"],
    "receipts": ["chat-activity.mjs"],
    "clarifications": ["chat-activity.mjs"],
    "budget": ["usage-report.mjs"],
}
DEVICE_CASES = {"live_voice.25", "wake_shutdown.25", "browser_settings.24"}
PROVIDER_FEATURES = {
    "google_sync": "google",
    "google_writes": "google",
    "linear_sync": "linear",
    "linear_writes": "linear",
    "external_agents": "external_agents",
    "live_voice": "voice",
}


def criteria(case):
    return {
        f"{field}.{i + 1}": text for field in ("expected", "invariants") for i, text in enumerate(case[field])
    }


def registry():
    features, cases = load()
    by_feature = {f["id"]: f for f in features}
    bindings_file = CATALOG / "bindings.json"
    explicit = json.loads(bindings_file.read_text()) if bindings_file.exists() else {}
    rows = []
    for case in cases:
        identity = case["id"]
        variants = []
        if identity in CASES:
            variants.append(
                {
                    "id": "contract",
                    "mode": "offline",
                    "adapter": "contract",
                    "target": identity,
                    "level": "component",
                    "criteria": [],
                }
            )
        if identity in PROBES:
            variants.append(
                {
                    "id": "luna",
                    "mode": "live-model",
                    "adapter": "agent",
                    "target": identity,
                    "level": "component",
                    "criteria": [],
                }
            )
        # These suites supply useful supporting evidence, not proof of this individual scenario.
        for path in by_feature[case["feature"]]["regression_files"]:
            variants.append(
                {
                    "id": "support:" + path,
                    "mode": "offline",
                    "adapter": "pytest",
                    "target": path,
                    "level": "supporting",
                    "criteria": [],
                }
            )
        for path in BROWSERS.get(case["feature"], []):
            variants.append(
                {
                    "id": "browser:" + path,
                    "mode": "offline",
                    "adapter": "browser",
                    "target": path,
                    "level": "supporting",
                    "criteria": [],
                }
            )
        service = {
            "google_sync": "google",
            "google_writes": "google",
            "linear_sync": "linear",
            "linear_writes": "linear",
            "operations": "r2",
        }.get(case["feature"])
        if service:
            variants.append(
                {
                    "id": "connected:" + service,
                    "mode": "live-service",
                    "adapter": "connected",
                    "target": service,
                    "level": "supporting",
                    "criteria": [],
                }
            )
        variants.extend(explicit.get(identity, []))
        bound = {c for v in variants if v["level"] == "acceptance" for c in v["criteria"]}
        unbound = sorted(set(criteria(case)) - bound)
        requirement = None
        if unbound:
            requirement = {
                "kind": "device" if identity in DEVICE_CASES else "acceptance_binding",
                "reason": "Physical phone evidence required"
                if identity in DEVICE_CASES
                else "Scenario-specific acceptance assertions are not yet fully bound; supporting tests cannot substitute",
                "criteria": unbound,
            }
        rows.append(
            {
                **case,
                "type": FEATURE_TYPE[case["feature"]],
                "criteria_map": criteria(case),
                "variants": variants,
                "requirement": requirement,
            }
        )
    return rows


def test_nodes(path):
    source = ROOT / path
    tree = ast.parse(source.read_text())
    return {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test_")
    }


def validate_registry():
    rows = registry()
    errors = []
    cached = {}
    for row in rows:
        names = set()
        for variant in row["variants"]:
            if variant["id"] in names:
                errors.append(f"{row['id']}: duplicate variant")
            names.add(variant["id"])
            if variant["mode"] not in MODES or variant["level"] not in {
                "component",
                "supporting",
                "acceptance",
            }:
                errors.append(f"{row['id']}: invalid variant mode/level")
            if not set(variant["criteria"]) <= row["criteria_map"].keys():
                errors.append(f"{row['id']}: unknown criterion")
            adapter, target = variant["adapter"], variant["target"]
            if adapter == "pytest":
                path, _, node = target.partition("::")
                if path not in cached:
                    cached[path] = test_nodes(path)
                if node and node not in cached[path]:
                    errors.append(f"{row['id']}: missing pytest node {target}")
            elif adapter == "browser":
                if not (ROOT / "apps/web/e2e" / target).is_file():
                    errors.append(f"{row['id']}: missing browser fixture")
            elif adapter == "contract":
                if target not in CASES:
                    errors.append(f"{row['id']}: missing contract")
            elif adapter == "agent":
                if target not in PROBES:
                    errors.append(f"{row['id']}: missing probe")
            elif adapter == "pipeline":
                from .pipelines import SUPPORTED

                if target not in SUPPORTED:
                    errors.append(f"{row['id']}: missing pipeline adapter")
            elif adapter == "connected":
                if target not in {"google", "linear", "r2"}:
                    errors.append(f"{row['id']}: unknown connected service")
            else:
                errors.append(f"{row['id']}: unknown adapter")
            if variant["level"] == "acceptance" and not variant["criteria"]:
                errors.append(f"{row['id']}: acceptance binding has no criteria")
    if errors:
        raise ValueError("\n".join(errors))
    return {
        "cases": len(rows),
        "types": dict(Counter(row["type"] for row in rows)),
        "fully_bound": sum(row["requirement"] is None for row in rows),
        "component_cases": sum(any(v["level"] == "component" for v in row["variants"]) for row in rows),
        "sha256": hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest(),
    }


def select(types="all", features=None, cases=None):
    rows = registry()

    def values(value, available):
        if not value or value == "all":
            return set(available)
        chosen = set(value.split(","))
        if not chosen <= set(available):
            raise ValueError("Unknown selector: " + ", ".join(sorted(chosen - set(available))))
        return chosen

    ts = values(types, GROUPS)
    fs = values(features, {r["feature"] for r in rows})
    cs = values(cases, {r["id"] for r in rows})
    selected = [r for r in rows if r["type"] in ts and r["feature"] in fs and r["id"] in cs]
    if not selected:
        raise ValueError("Selectors matched no cases")
    return selected


def jobs(rows, modes, repeats=1, include_support=True):
    """Deduplicate shared suites within each repeat, retaining every consumer."""
    result = {}
    for repeat in range(1, repeats + 1):
        for row in rows:
            for variant in row["variants"]:
                if variant["mode"] not in modes or (variant["level"] == "supporting" and not include_support):
                    continue
                target, _, selector = (
                    variant["target"].partition("::")
                    if variant["adapter"] == "pytest"
                    else (variant["target"], "", "")
                )
                key = (variant["adapter"], target, variant["mode"], repeat)
                identity = hashlib.sha256(json.dumps(key).encode()).hexdigest()[:20]
                job = result.setdefault(
                    identity,
                    {
                        "id": identity,
                        "adapter": key[0],
                        "target": key[1],
                        "mode": key[2],
                        "repeat": repeat,
                        "consumers": [],
                        "selectors": [],
                    },
                )
                if selector not in job["selectors"]:
                    job["selectors"].append(selector)
                job["consumers"].append(
                    {
                        "case_id": row["id"],
                        "variant": variant["id"],
                        "selector": selector,
                        "level": variant["level"],
                        "criteria": variant["criteria"],
                    }
                )
    return list(result.values())


def coverage():
    """Machine-readable automation backlog; no runtime pass claims."""
    rows = registry()
    return {
        "version": 2,
        "registry": validate_registry(),
        "features": {
            feature: {
                "catalog": len(group),
                "fully_bound": sum(r["requirement"] is None for r in group),
                "component_cases": sum(any(v["level"] == "component" for v in r["variants"]) for r in group),
                "supporting_suites": sorted(
                    {v["target"] for r in group for v in r["variants"] if v["level"] == "supporting"}
                ),
                "remaining": [{"case_id": r["id"], **r["requirement"]} for r in group if r["requirement"]],
            }
            for feature in sorted(FEATURE_TYPE)
            if (group := [r for r in rows if r["feature"] == feature])
        },
    }
