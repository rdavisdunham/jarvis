"""Opt-in export of completed synthetic eval evidence; never runs models or opens a DB.

Historical spans use OTLP/HTTP JSON so their recorded timings and IDs are preserved.
No global SDK instrumentation is installed in isolated eval workers.
"""

import argparse
import hashlib
import json
import os
import re
import time
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlsplit

import httpx
from dotenv import dotenv_values

from .catalog import ROOT
from .environment import validate_url
from .reporting import atomic_json

VERSION = 2
SECRET_KEY = re.compile(
    r"(?:password|secret|authorization|cookie|api[_-]?key|access_token|refresh_token|database_url|encryption_key)",
    re.IGNORECASE,
)
BUSINESS_TABLES = {
    "tasks",
    "notes",
    "memories",
    "structure_records",
    "structure_links",
    "work_requests",
    "clarifications",
    "note_lists",
    "note_list_items",
    "memory_reviews",
}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def identity(*parts, size=32):
    return hashlib.sha256(":".join(map(str, parts)).encode()).hexdigest()[:size]


class ExportError(RuntimeError):
    pass


class Config:
    def __init__(self, base, public, secret):
        u = urlsplit(base)
        if (
            u.scheme != "https"
            or not u.netloc
            or u.username
            or u.password
            or u.query
            or u.fragment
            or u.path not in {"", "/"}
        ):
            raise ExportError("LANGFUSE_BASE_URL must be an HTTPS origin without credentials or a path")
        if not public or not secret:
            raise ExportError("Missing LANGFUSE_PUBLIC_KEY or LANGFUSE_SECRET_KEY")
        self.base, self.public, self.secret = base.rstrip("/"), public, secret

    @classmethod
    def load(cls):
        values = {**dotenv_values(ROOT / ".env"), **{k: v for k, v in os.environ.items() if v}}
        return cls(
            *(values.get(k, "") for k in ("LANGFUSE_BASE_URL", "LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY"))
        )

    def __repr__(self):
        return "LangfuseConfig(credentials=[REDACTED])"


def clean(value, secrets=()):
    if isinstance(value, dict):
        return {k: "[REDACTED]" if SECRET_KEY.search(k) else clean(v, secrets) for k, v in value.items()}
    if isinstance(value, list):
        return [clean(v, secrets) for v in value]
    if isinstance(value, str):
        for secret in secrets:
            if secret:
                value = value.replace(secret, "[REDACTED]")
        value = re.sub(r"(?i)Bearer\s+[^\s\"']+", "Bearer [REDACTED]", value)
        value = re.sub(r"(?:sk|pk)-(?:lf-)?[A-Za-z0-9_-]{16,}", "[REDACTED]", value)
        return re.sub(r"(://)[^/\s:@]+:[^/\s@]+@", r"\1[REDACTED]@", value)
    return value


def encoded(value, limit=45000):
    text = json.dumps(value, ensure_ascii=False, default=str)
    if len(text.encode()) <= limit:
        return text
    return json.dumps(
        {
            "truncated": True,
            "sha256": digest(value),
            "preview": text[: limit // 4],
            "note": "Full evidence retained in local campaign artifacts.",
        }
    )


def read(directory, relative):
    path = (directory / relative).resolve()
    if not path.is_relative_to(directory.resolve()):
        raise ExportError("Evidence path escapes the campaign")
    return json.loads(path.read_text())


def changes(state):
    """Upload business-record changes, never whole DB snapshots/auth tables."""
    result = []
    for table in sorted(BUSINESS_TABLES):
        before = {r["id"]: r for r in state.get("before", {}).get(table, []) if "id" in r}
        after = {r["id"]: r for r in state.get("after", {}).get(table, []) if "id" in r}
        for key in sorted(before.keys() | after.keys()):
            a, b = before.get(key, {}), after.get(key, {})
            fields = [f for f in a.keys() | b.keys() if a.get(f) != b.get(f)]
            if fields:
                result.append(
                    {
                        "table": table,
                        "id": key,
                        "before": {f: a.get(f) for f in sorted(fields)},
                        "after": {f: b.get(f) for f in sorted(fields)},
                    }
                )
    return result


def attr(key, value):
    if isinstance(value, bool):
        value = {"boolValue": value}
    elif isinstance(value, int):
        value = {"intValue": str(value)}
    elif isinstance(value, float):
        value = {"doubleValue": value}
    else:
        value = {"stringValue": value if isinstance(value, str) else encoded(value)}
    return {"key": key, "value": value}


def span(
    trace_id,
    span_id,
    name,
    common,
    start,
    end,
    *,
    parent=None,
    kind="span",
    input=None,
    output=None,
    extra=None,
):
    attributes = {**common, "langfuse.observation.type": kind, **(extra or {})}
    if input is not None:
        attributes["langfuse.observation.input"] = encoded(input)
    if output is not None:
        attributes["langfuse.observation.output"] = encoded(output)
    item = {
        "traceId": trace_id,
        "spanId": span_id,
        "name": name,
        "kind": 1,
        "startTimeUnixNano": str(start),
        "endTimeUnixNano": str(max(start + 1, end)),
        "attributes": [attr(k, v) for k, v in attributes.items() if v is not None],
    }
    if parent:
        item["parentSpanId"] = parent
    return item


def build(directory):
    directory = Path(directory).resolve()
    manifest = read(directory, "manifest.json")
    validate_url(manifest["database_url"])  # Validate synthetic namespace; does not connect.
    report = read(directory, "report.json")
    grader = read(directory, "grader.json") if (directory / "grader.json").exists() else None
    results = {j["id"]: read(directory, "results/" + j["id"] + ".json") for j in manifest["jobs"]}
    if any(r["status"] in {"not_run", "running", "queued", "not_completed"} for r in results.values()):
        raise ExportError("Finish the campaign before exporting immutable observations")
    cases = {c["id"]: c for c in manifest["cases"]}
    reported = {c["case_id"]: c for c in report["cases"]}
    if set(cases) != set(reported) or len(cases) != len(report["cases"]):
        raise ExportError("Report case identities do not match the manifest")
    grades = {c["case_id"]: c for c in grader["cases"]} if grader else {}
    if grader and (
        set(grades) != set(cases)
        or len(grades) != len(grader["cases"])
        or grader["catalog_sha256"] != manifest["catalog_sha256"]
        or grader["harness_sha256"] != manifest["harness_sha256"]
        or grader["code_summary"] != report["summary"]
    ):
        raise ExportError("Grading evidence does not match this campaign")
    evidence = {}
    for key, result in results.items():
        attempt = result["evidence_directory"]
        evidence[key] = {}
        for name in ("state.json", "trace.json"):
            path = (directory / attempt / name).resolve()
            if not path.is_relative_to(directory):
                raise ExportError("Evidence path escapes the campaign")
            evidence[key][name] = read(directory, str(path.relative_to(directory))) if path.exists() else {}
    # Frozen content signature prevents reuse when any exported evidence changes.
    signature = digest(
        {"version": VERSION, "manifest": manifest, "report": report, "grader": grader, "evidence": evidence}
    )
    export_id = "eri-eval-" + signature[:24]
    created = datetime.fromisoformat(manifest["created_at"])
    start = int(created.timestamp() * 1e9)
    common = {
        "langfuse.environment": "eval",
        "langfuse.session.id": export_id,
        "langfuse.release": manifest["commit"],
        "langfuse.version": manifest["harness_sha256"],
        "langfuse.trace.metadata.campaign": directory.name,
        "langfuse.trace.metadata.export_id": export_id,
        "langfuse.trace.metadata.synthetic": True,
    }
    spans, scores = [], []
    job_traces = {key: identity(export_id, "job", key) for key in results}
    calls = report.get("spending", {}).get("calls", [])
    costs = {c["id"]: c for c in calls}
    for key, result in sorted(results.items()):
        attempt = result["evidence_directory"]
        state = evidence[key]["state.json"]
        trace = evidence[key]["trace.json"] or []
        # Historical trace logs do not record tool start times; do not invent real latency.
        local = {
            **common,
            "langfuse.trace.name": "eval-job/" + result["target"],
            "langfuse.trace.metadata.job_id": key,
            "langfuse.trace.metadata.adapter": result["adapter"],
            "langfuse.trace.metadata.mode": result["mode"],
            "langfuse.observation.metadata.timing": "reconstructed; job duration recorded, tool positions approximate",
        }
        end = start + int(result.get("duration_seconds", 0) * 1e9)
        tid, sid = job_traces[key], identity(export_id, key, "root", size=16)
        output = {k: v for k, v in result.items() if k not in {"tests"}}
        output.update(
            reply=state.get("reply"),
            changes=changes(state),
            pipeline=state.get("pipeline"),
            local_evidence=attempt,
        )
        spans.append(
            span(
                tid,
                sid,
                result["target"],
                local,
                start,
                end,
                output=output,
                extra={
                    "langfuse.observation.level": "ERROR"
                    if result["status"] in {"failed", "safety_failure", "infra_error"}
                    else "DEFAULT"
                },
            )
        )
        for index, item in enumerate(trace):
            if not isinstance(item, dict):
                continue
            # Cache bookkeeping may carry a model name but is not a provider call.
            provider_call = item.get("reservation") in costs
            model = item.get("model") if provider_call else None
            extra = {"langfuse.observation.metadata.sequence": index}
            if model:
                extra["langfuse.observation.model.name"] = model
                # Explicit ledger cost prevents duplicate/default model pricing estimates.
                cost = costs.get(item.get("reservation"), {})
                if cost.get("actual_usd") is not None:
                    extra["langfuse.observation.cost_details"] = encoded({"total": float(cost["actual_usd"])})
                if cost.get("actual_usd") is None:
                    extra["langfuse.observation.cost_details"] = encoded({"total": 0})
                    extra["langfuse.observation.metadata.cost_status"] = "unsettled; not a free call"
                    extra["langfuse.observation.metadata.reserved_usd"] = cost.get("charged_usd")
                extra["langfuse.observation.metadata.provider_usage"] = item.get("usage", {})
                usage = item.get("usage", {})
                extra["langfuse.observation.usage_details"] = encoded(
                    {
                        "input": usage.get("prompt_tokens", usage.get("input_tokens", 0)),
                        "output": usage.get("completion_tokens", usage.get("output_tokens", 0)),
                    }
                )

            spans.append(
                span(
                    tid,
                    identity(export_id, key, index, size=16),
                    item.get("name", model or "provider-event"),
                    local,
                    start,
                    start + 1,
                    parent=sid,
                    kind=("embedding" if item.get("kind") == "embedding" else "generation")
                    if model
                    else "tool",
                    input=item.get("arguments"),
                    output=item.get("result", item),
                    extra=extra,
                )
            )
        # Contract tools are stored in state, not the transport trace.
        for index, item in enumerate(state.get("tools", [])):
            spans.append(
                span(
                    tid,
                    identity(export_id, key, "contract", index, size=16),
                    item.get("name", item.get("tool", "command")),
                    local,
                    start,
                    start + 1,
                    parent=sid,
                    kind="tool",
                    output=item,
                )
            )
    for key, case in sorted(cases.items()):
        raw, grade = reported[key], grades.get(key)
        tid, sid = identity(export_id, "case", key), identity(export_id, "case-root", key, size=16)
        job_ids = sorted({e["job_id"] for e in raw["evidence"] if e.get("job_id")})
        out = {
            "automated_status": raw["status"],
            "review": grade,
            "missing_criteria": raw["missing_criteria"],
            "requirement": raw["requirement"],
            "execution_evidence": [
                {"job_id": j, "trace_id": job_traces[j], "status": results[j]["status"]} for j in job_ids
            ],
        }
        attrs = {
            **common,
            "langfuse.trace.name": "eval-case/" + key,
            "langfuse.experiment.id": export_id,
            "langfuse.experiment.name": directory.name + "-" + signature[:8],
            "langfuse.experiment.dataset.id": "eridani-catalog-" + manifest["catalog_sha256"][:20],
            "langfuse.experiment.item.id": key,
            "langfuse.experiment.item.root_observation_id": sid,
            "langfuse.experiment.item.expected_output": encoded(case["criteria_map"]),
            "langfuse.experiment.item.metadata.feature": case["feature"],
            "langfuse.experiment.item.metadata.automated_status": raw["status"],
            "langfuse.experiment.item.metadata.review_status": grade["review_disposition"]
            if grade
            else "not_reviewed",
            "langfuse.experiment.metadata.coverage": report["summary"],
            "langfuse.experiment.metadata.review_coverage": grader["review_summary"] if grader else {},
            "langfuse.experiment.metadata.estimated_usd": report["spending"]["estimated_usd"],
            "langfuse.experiment.metadata.including_uncertain_usd": report["spending"][
                "including_uncertain_usd"
            ],
            "langfuse.trace.metadata.feature": case["feature"],
            "langfuse.trace.metadata.case_id": key,
        }
        spans.append(
            span(
                tid,
                sid,
                key + ": " + case["title"],
                attrs,
                start,
                start + 1,
                input={"inputs": case["inputs"], "setup": case["setup"], "steps": case["steps"]},
                output=out,
            )
        )
        # Unassessable cases remain visible in all experiment items/coverage. Do not
        # turn missing checks into correctness scores or dilute pass/fail averages.
        if raw["status"] in {"blocked", "not_run"}:
            continue
        values = [
            ("automated_status", raw["status"], "Code assertions; see missing criteria and evidence scope.")
        ]
        if grade:
            values.append(("external_review_status", grade["review_disposition"], grade["reason"]))
        for name, value, comment in values:
            scores.append(
                {
                    "id": identity(export_id, key, name),
                    "traceId": tid,
                    "observationId": sid,
                    "name": name,
                    "value": value,
                    "dataType": "CATEGORICAL",
                    "environment": "eval",
                    "comment": comment,
                    "metadata": {
                        "case_id": key,
                        "scope": grade.get("scope") if grade else None,
                        "grader": grader.get("grader_model") if name == "external_review_status" else "code",
                        "missing_criteria": raw["missing_criteria"],
                    },
                }
            )
    return {
        "version": VERSION,
        "export_id": export_id,
        "signature": signature,
        "campaign": directory.name,
        "from_time": manifest["created_at"],
        "case_count": len(cases),
        "job_count": len(results),
        "spans": spans,
        "scores": scores,
    }


class Client:
    def __init__(self, config, transport=None):
        self.config = config
        self.http = httpx.Client(
            base_url=config.base,
            auth=(config.public, config.secret),
            timeout=30,
            trust_env=False,
            follow_redirects=False,
            transport=transport,
        )
        self.last_score = 0.0

    def close(self):
        self.http.close()

    def request(self, method, path, **kwargs):
        for attempt in range(4):
            try:
                response = self.http.request(method, path, **kwargs)
            except httpx.HTTPError:
                raise ExportError(
                    "Langfuse transport failed; upload receipt retains any uncertain request"
                ) from None
            if response.status_code == 429 and attempt < 3:
                try:
                    delay = min(30, max(1, float(response.headers.get("retry-after", "10"))))
                except ValueError:
                    delay = 10
                time.sleep(delay)
                continue
            if not response.is_success:
                raise ExportError(f"Langfuse HTTP {response.status_code} on {path}; response body omitted")
            try:
                return response.json()
            except ValueError:
                raise ExportError("Langfuse returned a non-JSON response") from None
        raise ExportError("Langfuse rate limit exhausted")

    def project(self):
        projects = self.request("GET", "/api/public/projects").get("data", [])
        if len(projects) != 1:
            raise ExportError("Expected one project-scoped Langfuse key")
        p = projects[0]
        return {k: p.get(k) for k in ("id", "name", "organization")}

    def observations(self, plan):
        found, cursor = {}, None
        for _ in range(100):
            params = {
                "sessionId": plan["export_id"],
                "fromStartTime": plan["from_time"],
                "fields": "core,basic",
                "limit": 1000,
            }
            if cursor:
                params["cursor"] = cursor
            data = self.request("GET", "/api/public/v2/observations", params=params)
            for row in data.get("data", []):
                if row["id"] in found:
                    raise ExportError("Duplicate observation returned; inspect immutable-span replay")
                found[row["id"]] = row
            next_cursor = data.get("meta", {}).get("nextCursor") or data.get("meta", {}).get("cursor")
            if not next_cursor:
                return found
            if next_cursor == cursor:
                raise ExportError("Observation pagination did not advance")
            cursor = next_cursor
        raise ExportError("Observation pagination limit reached")

    def score(self, body):
        time.sleep(max(0, 0.7 - (time.monotonic() - self.last_score)))
        self.last_score = time.monotonic()
        return self.request("POST", "/api/public/scores", json=body)


@contextmanager
def export_lock(directory):
    import fcntl

    with (directory / ".langfuse.lock").open("w") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ExportError("Another Langfuse export is running") from None
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def export(directory, client, *, verify_only=False):
    directory = Path(directory).resolve()
    plan = clean(build(directory), (client.config.public, client.config.secret))
    project = client.project()
    target = identity(client.config.base, project["id"], size=12)
    path = directory / ("langfuse-" + target + "-" + plan["signature"][:12] + ".json")
    with export_lock(directory):
        receipt = (
            json.loads(path.read_text())
            if path.exists()
            else {
                "version": VERSION,
                "export_id": plan["export_id"],
                "signature": plan["signature"],
                "project": project,
                "base_url": client.config.base,
                "sent_spans": [],
                "sent_scores": [],
                "pending": None,
                "verified": False,
            }
        )
        if receipt["signature"] != plan["signature"]:
            raise ExportError("Export receipt belongs to a different evidence snapshot")

        def save():
            atomic_json(path, receipt)

        if receipt["pending"]:
            pending = receipt["pending"]
            if pending["kind"] == "spans":
                visible = client.observations(plan)
                if not set(pending["ids"]) <= visible.keys():
                    raise ExportError(
                        "Prior upload is uncertain. Verify later; immutable spans will not be blindly resent."
                    )
                receipt["sent_spans"] = sorted(set(receipt["sent_spans"]) | set(pending["ids"]))
            else:
                data = client.request(
                    "GET", "/api/public/v3/scores", params={"id": pending["ids"][0], "limit": 1}
                )
                if not data.get("data"):
                    raise ExportError("Prior score upload is uncertain. Verify later before retrying.")
                receipt["sent_scores"] = sorted(set(receipt["sent_scores"]) | set(pending["ids"]))
            receipt["pending"] = None
            save()
        if not verify_only:
            sent = set(receipt["sent_spans"])
            remaining = [s for s in plan["spans"] if s["spanId"] not in sent]
            for offset in range(0, len(remaining), 25):
                batch = remaining[offset : offset + 25]
                receipt["pending"] = {"kind": "spans", "ids": [s["spanId"] for s in batch]}
                save()
                answer = client.request(
                    "POST",
                    "/api/public/otel/v1/traces",
                    headers={"x-langfuse-ingestion-version": "4"},
                    json={
                        "resourceSpans": [
                            {
                                "resource": {"attributes": [attr("service.name", "eridani-evals")]},
                                "scopeSpans": [
                                    {
                                        "scope": {"name": "eridani.eval.export", "version": str(VERSION)},
                                        "spans": batch,
                                    }
                                ],
                            }
                        ]
                    },
                )
                if answer.get("partialSuccess", {}).get("rejectedSpans") not in {None, 0, "0"}:
                    raise ExportError(
                        "Langfuse partially rejected spans; receipt retained for reconciliation"
                    )
                receipt["sent_spans"].extend(receipt["pending"]["ids"])
                receipt["pending"] = None
                save()
            sent_scores = set(receipt["sent_scores"])
            for body in plan["scores"]:
                if body["id"] in sent_scores:
                    continue
                receipt["pending"] = {"kind": "score", "ids": [body["id"]]}
                save()
                client.score(body)
                receipt["sent_scores"].append(body["id"])
                receipt["pending"] = None
                save()
                if len(receipt["sent_scores"]) % 25 == 0:
                    print(
                        json.dumps(
                            {
                                "langfuse_scores_uploaded": len(receipt["sent_scores"]),
                                "total": len(plan["scores"]),
                            }
                        ),
                        flush=True,
                    )
        visible = client.observations(plan)
        expected = {s["spanId"] for s in plan["spans"]}
        matched = expected & visible.keys()
        score_ids = set()
        score_mismatches = []
        for offset in range(0, len(plan["scores"]), 50):
            group = plan["scores"][offset : offset + 50]
            data = client.request(
                "GET", "/api/public/v3/scores", params={"id": ",".join(s["id"] for s in group), "limit": 100}
            )
            score_ids.update(s["id"] for s in data.get("data", []))
            expected_scores = {s["id"]: s for s in group}
            for actual in data.get("data", []):
                expected_score = expected_scores.get(actual["id"], {})
                if actual.get("name") != expected_score.get("name") or actual.get(
                    "value"
                ) != expected_score.get("value"):
                    score_mismatches.append(actual["id"])

        receipt.update(
            verified=matched == expected
            and score_ids == {s["id"] for s in plan["scores"]}
            and not score_mismatches,
            observed_spans=len(matched),
            expected_spans=len(expected),
            observed_scores=len(score_ids),
            expected_scores=len(plan["scores"]),
            score_value_mismatches=score_mismatches,
            case_count=plan["case_count"],
            job_count=plan["job_count"],
            project_url=client.config.base + "/project/" + project["id"],
            verified_at=datetime.now(UTC).isoformat(),
        )
        save()
        return {k: v for k, v in receipt.items() if k not in {"sent_spans", "sent_scores", "pending"}}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", nargs="?")
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.dry_run:
            if not args.directory:
                parser.error("Specify a campaign directory")
            plan = build(args.directory)
            print(
                json.dumps(
                    {k: v for k, v in plan.items() if k not in {"spans", "scores"}}
                    | {"observations": len(plan["spans"]), "scores": len(plan["scores"])},
                    indent=2,
                )
            )
            return 0
        client = Client(Config.load())
        try:
            if args.status:
                print(json.dumps(client.project(), indent=2))
                return 0
            if not args.directory:
                parser.error("Specify a campaign directory")
            result = export(args.directory, client, verify_only=args.verify)
            print(json.dumps(result, indent=2))
            return 0 if result["verified"] else 2
        finally:
            client.close()
    except (ExportError, ValueError, KeyError, OSError) as exc:
        # Never echo configuration, response bodies, or arbitrary saved evidence.
        message = str(exc) if isinstance(exc, ExportError) else type(exc).__name__
        print("Langfuse export: " + message, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
