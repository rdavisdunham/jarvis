import json

import httpx
import pytest

from scripts.app_eval.environment import DEFAULT_URL
from scripts.app_eval.langfuse_export import Client, Config, ExportError, build, clean, export, main


def put(root, name, value):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def campaign(tmp_path):
    case = {
        "id": "tasks.01",
        "feature": "tasks",
        "title": "Create task",
        "inputs": ["Create task"],
        "setup": ["Synthetic"],
        "steps": ["Create"],
        "criteria_map": {"expected.1": "Saved task"},
    }
    blocked = {**case, "id": "tasks.02"}
    put(
        tmp_path,
        "manifest.json",
        {
            "database_url": DEFAULT_URL,
            "created_at": "2026-09-23T15:00:00+00:00",
            "commit": "abc",
            "catalog_sha256": "catalog",
            "harness_sha256": "harness",
            "cases": [case, blocked],
            "jobs": [{"id": "job"}],
        },
    )
    result = {
        "status": "passed",
        "target": "tasks.01",
        "adapter": "agent",
        "mode": "live-model",
        "duration_seconds": 4,
        "evidence_directory": "attempts/job/001",
    }
    put(tmp_path, "results/job.json", result)
    raw = {
        "case_id": "tasks.01",
        "status": "passed",
        "missing_criteria": [],
        "requirement": None,
        "evidence": [{"job_id": "job"}],
    }
    put(
        tmp_path,
        "report.json",
        {
            "summary": {"passed": 1, "blocked": 1},
            "cases": [
                raw,
                {
                    **raw,
                    "case_id": "tasks.02",
                    "status": "blocked",
                    "missing_criteria": ["expected.1"],
                    "requirement": {"kind": "acceptance_binding"},
                },
            ],
            "spending": {
                "estimated_usd": "0.01",
                "including_uncertain_usd": "0.01",
                "calls": [{"id": "call", "actual_usd": "0.01"}],
            },
        },
    )
    grade = {
        "case_id": "tasks.01",
        "review_disposition": "needs_review",
        "reason": "Not enough evidence",
        "scope": "acceptance",
    }
    put(
        tmp_path,
        "grader.json",
        {
            "catalog_sha256": "catalog",
            "harness_sha256": "harness",
            "code_summary": {"passed": 1, "blocked": 1},
            "review_summary": {"needs_review": 1, "unassessable": 1},
            "grader_model": "gpt-6-sol",
            "cases": [grade, {**grade, "case_id": "tasks.02", "review_disposition": "unassessable"}],
        },
    )
    put(
        tmp_path,
        "attempts/job/001/state.json",
        {
            "before": {"tasks": [], "auth_sessions": []},
            "after": {
                "tasks": [{"id": "task", "title": "Create task"}],
                "auth_sessions": [{"id": "session", "token": "private-session"}],
            },
            "reply": "Saved",
        },
    )
    put(
        tmp_path,
        "attempts/job/001/trace.json",
        [
            {"model": "gpt-5.6-luna", "reservation": "call", "usage": {"prompt_tokens": 100}},
            {"name": "task_create", "arguments": {"title": "Create task"}, "result": {"status": "succeeded"}},
        ],
    )
    return tmp_path


def test_complete_catalog_separate_grades_and_deduplicated_cost(campaign):
    plan = build(campaign)
    assert plan["case_count"] == 2 and plan["job_count"] == 1
    assert len(plan["spans"]) == 5  # one shared job, two children, two case roots
    assert [(s["name"], s["value"]) for s in plan["scores"]] == [
        ("automated_status", "passed"),
        ("external_review_status", "needs_review"),
    ]
    text = json.dumps(plan)
    assert "unassessable" in text and "private-session" not in text
    cost_attrs = [
        a for s in plan["spans"] for a in s["attributes"] if a["key"] == "langfuse.observation.cost_details"
    ]
    assert len(cost_attrs) == 1
    assert json.loads(cost_attrs[0]["value"]["stringValue"]) == {"total": 0.01}


def test_evidence_change_creates_distinct_immutable_export(campaign):
    original = build(campaign)
    put(campaign, "attempts/job/001/trace.json", [{"name": "different_tool"}])
    changed = build(campaign)
    assert changed["export_id"] != original["export_id"]
    assert changed["spans"][0]["spanId"] != original["spans"][0]["spanId"]


def test_rejects_wrong_grader_and_path_escape(campaign):
    p = campaign / "grader.json"
    value = json.loads(p.read_text())
    value["harness_sha256"] = "wrong"
    p.write_text(json.dumps(value))
    with pytest.raises(ExportError, match="Grading"):
        build(campaign)
    p.unlink()
    result = json.loads((campaign / "results/job.json").read_text())
    result["evidence_directory"] = "../../outside"
    put(campaign, "results/job.json", result)
    with pytest.raises(ExportError, match="escapes"):
        build(campaign)


def test_rejects_production_or_unfinished_runs(campaign):
    p = campaign / "manifest.json"
    m = json.loads(p.read_text())
    m["database_url"] = DEFAULT_URL.replace("127.0.0.1", "production.example.com")
    p.write_text(json.dumps(m))
    with pytest.raises(ValueError, match="local PostgreSQL"):
        build(campaign)
    m["database_url"] = DEFAULT_URL
    p.write_text(json.dumps(m))
    result = json.loads((campaign / "results/job.json").read_text())
    result["status"] = "running"
    put(campaign, "results/job.json", result)
    with pytest.raises(ExportError, match="Finish"):
        build(campaign)


def test_redaction_keeps_usage_counts():
    value = {
        "api_key": "key",
        "nested": {"refresh_token": "secret"},
        "text": "Bearer abc123 sk-lf-12345678901234567890 postgresql://user:pass@localhost/db REAL-SECRET",
        "usage": {"prompt_tokens": 42},
    }
    result = clean(value, ("REAL-SECRET",))
    assert result["usage"]["prompt_tokens"] == 42
    assert all(
        s not in json.dumps(result) for s in ["abc123", "12345678901234567890", "user:pass", "REAL-SECRET"]
    )
    assert result["nested"]["refresh_token"] == "[REDACTED]"


def test_dry_run_never_loads_keys_or_uses_network(campaign, monkeypatch):
    def forbidden():
        raise AssertionError("Network configuration should not be loaded")

    monkeypatch.setattr(Config, "load", forbidden)
    assert main([str(campaign), "--dry-run"]) == 0


@pytest.mark.parametrize(
    "url", ["http://cloud.langfuse.com", "https://user:pass@host", "https://host/path", "https://host?key=x"]
)
def test_credentials_require_secure_origin(url):
    with pytest.raises(ExportError):
        Config(url, "public", "secret")


def test_http_error_does_not_print_provider_body_or_follow_redirects():
    def handler(request):
        return httpx.Response(302, headers={"location": "https://other.example.com"}, text="secret-response")

    client = Client(
        Config("https://cloud.langfuse.com", "public", "secret"), transport=httpx.MockTransport(handler)
    )
    try:
        with pytest.raises(ExportError, match="HTTP 302") as error:
            client.project()
        assert "secret-response" not in str(error.value)
    finally:
        client.close()


class FakeClient:
    def __init__(self):
        self.config = Config("https://cloud.langfuse.com", "public", "secret")
        self.spans, self.scores = {}, {}
        self.posts = 0
        self.lose_response = False

    def project(self):
        return {"id": "project", "name": "Test"}

    def observations(self, plan):
        return self.spans

    def request(self, method, path, **kwargs):
        if method == "POST":
            self.posts += 1
            for s in kwargs["json"]["resourceSpans"][0]["scopeSpans"][0]["spans"]:
                assert s["spanId"] not in self.spans, "Duplicate immutable span"
                self.spans[s["spanId"]] = s
            if self.lose_response:
                self.lose_response = False
                raise ExportError("Lost response")
            return {}
        ids = kwargs["params"]["id"].split(",")
        return {"data": [self.scores[i] for i in ids if i in self.scores]}

    def score(self, body):
        self.scores[body["id"]] = body
        return {"id": body["id"]}


def test_export_resume_does_not_duplicate_accepted_spans(campaign):
    client = FakeClient()
    client.lose_response = True
    with pytest.raises(ExportError, match="Lost response"):
        export(campaign, client)
    result = export(campaign, client)
    assert result["verified"] and result["observed_spans"] == 5 and result["observed_scores"] == 2
    assert client.posts == 1
    assert export(campaign, client)["verified"]
    assert client.posts == 1
    assert export(campaign, client, verify_only=True)["verified"]


def test_uncertain_spans_are_not_blindly_retried(campaign):
    client = FakeClient()
    client.lose_response = True
    with pytest.raises(ExportError):
        export(campaign, client)
    client.spans.clear()  # Read path cannot yet prove acceptance.
    with pytest.raises(ExportError, match="uncertain"):
        export(campaign, client)
    assert client.posts == 1


def test_cache_bookkeeping_does_not_create_billable_model_calls(campaign):
    path = campaign / "attempts/job/001/trace.json"
    rows = json.loads(path.read_text())
    rows.append({"kind": "embedding_cache", "model": "text-embedding-3-small", "hits": 2, "misses": 0})
    path.write_text(json.dumps(rows))
    plan = build(campaign)
    model_spans = [
        s
        for s in plan["spans"]
        if any(a["key"] == "langfuse.observation.model.name" for a in s["attributes"])
    ]
    assert len(model_spans) == 1
    assert any(a["key"] == "langfuse.observation.usage_details" for a in model_spans[0]["attributes"])


def test_verification_checks_actual_score_values(campaign):
    client = FakeClient()
    assert export(campaign, client)["verified"]
    key = next(iter(client.scores))
    client.scores[key] = {**client.scores[key], "value": "wrong"}
    result = export(campaign, client, verify_only=True)
    assert not result["verified"]
    assert result["score_value_mismatches"] == [key]


def test_observation_cursor_and_duplicate_detection():
    calls = []

    def handler(request):
        calls.append(str(request.url))
        if "cursor=next" in str(request.url):
            return httpx.Response(200, json={"data": [{"id": "second"}], "meta": {}})
        return httpx.Response(200, json={"data": [{"id": "first"}], "meta": {"cursor": "next"}})

    client = Client(
        Config("https://cloud.langfuse.com", "public", "secret"), transport=httpx.MockTransport(handler)
    )
    try:
        assert set(client.observations({"export_id": "x", "from_time": "2026-09-23"})) == {"first", "second"}
        assert len(calls) == 2
    finally:
        client.close()


def test_uncertain_usage_is_marked_without_automatic_cost_estimation(campaign):
    p = campaign / "report.json"
    report = json.loads(p.read_text())
    report["spending"]["calls"][0].update(actual_usd=None, charged_usd="0.03")
    report["spending"].update(estimated_usd="0", including_uncertain_usd="0.03")
    p.write_text(json.dumps(report))
    plan = build(campaign)
    attrs = [a for span in plan["spans"] for a in span["attributes"]]
    status = [a for a in attrs if a["key"] == "langfuse.observation.metadata.cost_status"]
    assert status and "unsettled" in status[0]["value"]["stringValue"]
    costs = [a for a in attrs if a["key"] == "langfuse.observation.cost_details"]
    assert json.loads(costs[0]["value"]["stringValue"]) == {"total": 0}
