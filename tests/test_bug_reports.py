"""The Report a bug button's backend: report + task, limits, owner-only access, bot reads, scrubbing, retention."""

import base64
import json
from datetime import timedelta

import pytest
from fastapi.testclient import TestClient
from jarvis import bug_reports
from jarvis.api import app
from jarvis.db import session_scope
from jarvis.models import BugReport, Project, Task, now
from jarvis.worker import housekeeping_owner
from sqlalchemy import select
from test_accounts import client_for, shared
from test_external_agents import key, settings  # noqa: F401 - pytest fixture

JPEG = b"\xff\xd8\xff\xe0" + b"\x00" * 2000


@pytest.fixture(autouse=True)
def fresh_throttle():
    bug_reports.reports_by_owner.clear()
    yield
    bug_reports.reports_by_owner.clear()
SHOT = "data:image/jpeg;base64," + base64.b64encode(JPEG).decode()


def context(**extra):
    return {
        "page": {"view": "remotea", "url": "http://testserver/?view=remotea&run=abc&q=private-words", "run_id": "abc",
                 "dialogs": ["Run task"]},
        "build": {"version": "0.1.0", "commit": "abc1234", "built_at": "2026-10-31T00:00:00Z"},
        "device": {"platform": "Linux", "viewport": "1280x800", "standalone": False, "online": True},
        "errors": [{"level": "error", "message": "TypeError: x is undefined", "stack": "at a (App.tsx:1)"}],
        "requests": [
            {"method": "GET", "path": "/remotea/runs/abc", "status": 500, "error": "INTERNAL", "request_id": "r-1"},
            {"method": "GET", "path": "/tasks", "status": 200},
        ],
        **extra,
    }


def report(client, status=201, **body):
    response = client.post("/api/v1/bug-reports", json={"description": "Run drawer is blank\nmore detail",
                                                          "expected": "The timeline", "context": context(),
                                                          "screenshot": SHOT, **body})
    assert response.status_code == status, response.text
    return response.json()


def test_a_report_opens_a_tagged_task_in_the_bug_report_project(client):
    created = report(client)
    assert created["title"] == "Run drawer is blank" and created["status"] == "open"
    assert created["screenshot"]["bytes"] == len(JPEG) and created["screenshot"]["mime"] == "image/jpeg"
    with session_scope() as db:
        task = db.get(Task, created["task_id"])
        project = db.scalar(select(Project).where(Project.name == bug_reports.PROJECT))
        assert task.title == "Run drawer is blank" and task.tags == ["bug-report"] and task.project_id == project.id
        assert "Expected:\nThe timeline" in task.notes and "View: remotea (run abc)" in task.notes
        assert "TypeError: x is undefined" in task.notes and "GET /remotea/runs/abc → 500" in task.notes
        assert f"/api/v1/bug-reports/{created['id']}/screenshot" in task.notes
    # The second report reuses the project.
    second = report(client, description="x" * 200, screenshot=None, screenshot_error="capture failed: tainted canvas")
    assert len(second["title"]) == 80 and second["title"].endswith("…") and second["screenshot"] is None
    assert second["screenshot_error"] == "capture failed: tainted canvas"
    with session_scope() as db:
        assert len(db.scalars(select(Project).where(Project.name == bug_reports.PROJECT)).all()) == 1
        assert "Screenshot: not captured (capture failed: tainted canvas)" in db.get(Task, second["task_id"]).notes

    shot = client.get(f"/api/v1/bug-reports/{created['id']}/screenshot")
    assert shot.status_code == 200 and shot.content == JPEG and shot.headers["content-type"] == "image/jpeg"
    assert client.get(f"/api/v1/bug-reports/{second['id']}/screenshot").status_code == 404
    listing = client.get("/api/v1/bug-reports").json()
    assert [i["id"] for i in listing["items"]] == [second["id"], created["id"]] and "context" not in listing["items"][0]
    assert client.get("/api/v1/bug-reports", params={"task_id": created["task_id"]}).json()["items"][0]["id"] == created["id"]
    detail = client.get(f"/api/v1/bug-reports/{created['id']}").json()
    assert detail["description"].startswith("Run drawer") and detail["context"]["page"]["view"] == "remotea"


def test_completing_the_task_reads_as_fixed_and_status_can_be_set(client):
    created = report(client)
    with session_scope() as db:
        db.get(Task, created["task_id"]).status = "completed"
    assert client.get(f"/api/v1/bug-reports/{created['id']}").json()["status"] == "fixed"
    assert client.get("/api/v1/bug-reports", params={"status": "open"}).json()["items"] == []
    assert client.post(f"/api/v1/bug-reports/{created['id']}/status", json={"status": "dismissed"}).json()["status"] == "dismissed"
    assert client.post(f"/api/v1/bug-reports/{created['id']}/status", json={"status": "gone"}).status_code == 422


def test_size_limits(client):
    huge = "data:image/jpeg;base64," + base64.b64encode(b"\xff\xd8\xff" + b"\x00" * (bug_reports.SCREENSHOT_MAX + 10)).decode()
    assert report(client, 413, screenshot=huge)["error"]["code"] == "INVALID_ARGUMENT"
    too_long = client.post("/api/v1/bug-reports", content=b"{" + b" " * (bug_reports.BODY_MAX + 10) + b"}",
                           headers={"Content-Type": "application/json"})
    assert too_long.status_code == 413
    assert report(client, 413, context={"blob": ["y" * 3000] * 100})["error"]["code"] == "INVALID_ARGUMENT"
    # Other routes keep the 100,000-byte cap.
    assert client.post("/api/v1/commands", content=b"{" + b" " * 200_000 + b"}",
                       headers={"Content-Type": "application/json"}).status_code == 413
    assert report(client, 400, screenshot="data:image/png;base64," + base64.b64encode(JPEG).decode())
    assert report(client, 400, screenshot="data:text/html;base64,PGgxPg==")
    assert report(client, 400, description="   ")
    with session_scope() as db:
        assert db.scalars(select(BugReport)).all() == []


def test_owner_only_personal_workspace_and_csrf(client):
    created = report(client)
    other = client_for("guest")
    assert other.get(f"/api/v1/bug-reports/{created['id']}").status_code == 404
    assert other.get(f"/api/v1/bug-reports/{created['id']}/screenshot").status_code == 404
    assert other.get("/api/v1/bug-reports").json()["items"] == []
    assert other.delete(f"/api/v1/bug-reports/{created['id']}").status_code == 404
    anonymous = TestClient(app)
    assert anonymous.get("/api/v1/bug-reports").status_code == 401
    no_csrf = TestClient(app, cookies=client.cookies)
    assert no_csrf.post("/api/v1/bug-reports", json={"description": "x"}).status_code == 403
    # In a shared workspace the button refuses rather than filing into the team's task list.
    owner = client_for("owner-a")
    shared(owner, client_for("guest", "guest@example.test"))
    refused = owner.post("/api/v1/bug-reports", json={"description": "x"})
    assert refused.status_code == 403 and refused.json()["error"]["code"] == "PERSONAL_WORKSPACE"


def test_bots_read_reports_with_tasks_read(client):
    created = report(client)
    _, reader = key(client, ["tasks:read"])
    _, notes_only = key(client, ["notes:read"], name="Notes")
    bot = TestClient(app)
    listing = bot.get("/api/v1/external/bug-reports", headers=reader)
    assert listing.status_code == 200 and listing.json()["items"][0]["id"] == created["id"]
    detail = bot.get(f"/api/v1/external/bug-reports/{created['id']}", headers=reader).json()
    assert detail["context"]["errors"][0]["message"].startswith("TypeError") and "Failing requests" in detail["summary"]
    shot = bot.get(f"/api/v1/external/bug-reports/{created['id']}/screenshot", headers=reader)
    assert shot.status_code == 200 and shot.content == JPEG
    for path in ("", f"/{created['id']}", f"/{created['id']}/screenshot"):
        refused = bot.get("/api/v1/external/bug-reports" + path, headers=notes_only)
        assert refused.status_code == 403 and refused.json()["error"]["code"] == "INSUFFICIENT_SCOPE"
        assert bot.get("/api/v1/external/bug-reports" + path).status_code == 401
    # The task itself is readable the usual way.
    task = bot.get(f"/api/v1/external/records/task/{created['task_id']}", headers=reader)
    assert task.status_code == 200 and "bug-report" in json.dumps(task.json())


def test_secrets_are_stripped_server_side(client):
    created = report(client, description="Broken after pasting sk-proj-abcdefghijklmnopqrstuvwx",
                     context=context(
                         headers={"Authorization": "Bearer abcdefghijklmnopqrstuvwxyz123456", "Accept": "json"},
                         storage={"eri-density": "compact", "api_key": "plain-value", "csrfToken": "zzz"},
                         cookie="jarvis_session=abc",
                         note="token=supersecretvalue123",
                         page={"view": "tasks", "url": "http://testserver/?view=tasks&code=oauth-code-123&run=r1"},
                     ))
    detail = client.get(f"/api/v1/bug-reports/{created['id']}").json()
    stored = json.dumps(detail)
    for secret in ("abcdefghijklmnopqrstuvwxyz123456", "plain-value", "zzz", "jarvis_session=abc", "supersecretvalue123",
                   "oauth-code-123", "sk-proj-abcdefghijklmnopqrstuvwx"):
        assert secret not in stored, secret
    ctx = detail["context"]
    assert ctx["headers"] == {"Authorization": "[redacted]", "Accept": "json"} and ctx["cookie"] == "[redacted]"
    assert ctx["storage"]["eri-density"] == "compact" and ctx["page"]["url"] == "http://testserver/?view=tasks&code=…&run=r1"
    with session_scope() as db:
        assert "oauth-code-123" not in db.get(Task, created["task_id"]).notes


def test_export_delete_and_screenshot_retention(client):
    created = report(client)
    exported = client.get("/api/v1/export").json()["bug_reports"]
    assert exported[0]["id"] == created["id"] and base64.b64decode(exported[0]["screenshot_base64"]) == JPEG
    old = report(client)
    with session_scope() as db:
        db.get(BugReport, old["id"]).created_at = now() - timedelta(days=bug_reports.SCREENSHOT_DAYS + 1)
        housekeeping_owner(db, db.get(BugReport, old["id"]).owner_id)
    assert client.get(f"/api/v1/bug-reports/{old['id']}/screenshot").status_code == 404
    kept = client.get(f"/api/v1/bug-reports/{old['id']}").json()
    assert kept["description"].startswith("Run drawer") and kept["screenshot_error"].startswith("removed after")
    assert client.get(f"/api/v1/bug-reports/{created['id']}/screenshot").status_code == 200
    assert client.delete(f"/api/v1/bug-reports/{created['id']}").json()["deleted"] is True
    assert client.get(f"/api/v1/bug-reports/{created['id']}").status_code == 404
    with session_scope() as db:
        assert db.get(Task, created["task_id"]) is not None


def test_reports_are_rate_limited_per_owner(client):
    for _ in range(30):
        report(client, screenshot=None)
    assert report(client, 429)["error"]["code"] == "RATE_LIMITED"


@pytest.mark.parametrize("text,expected", [
    ("/tasks?status=open&q=secret", "/tasks?status=…&q=…"),
    ("/?view=remotea&run=11111111-1111-4111-8111-111111111111", "/?view=remotea&run=11111111-1111-4111-8111-111111111111"),
    ("/?view=<script>", "/?view=…"),
])
def test_strip_query(text, expected):
    assert bug_reports.strip_query(text) == expected
