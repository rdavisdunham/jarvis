"""Optional Linear structure onboarding: teams/projects mapped onto Homes (linear_structure.py).

Linear is a fake throughout; no test reaches the network."""

import copy
from uuid import uuid4

import pytest
from cryptography.fernet import Fernet
from jarvis import linear_commands as commands
from jarvis import linear_structure as mapping
from jarvis import linear_sync as sync
from jarvis.config import get_settings
from jarvis.db import session_scope
from jarvis.domain import DomainError, execute
from jarvis.hierarchy import for_task
from jarvis.linear_client import LinearFailure
from jarvis.models import Job, LinearConnection, LinearIssue, Project, Task
from jarvis.structure import ensure, schema_data
from jarvis.structure_models import StructureRecord, StructureSchema
from sqlalchemy import func, select
from test_linear import Fake, issue

OWNER = "davin"


def run(tool, **args):
    with session_scope() as db:
        return execute(db, OWNER, str(uuid4()), tool, args)["data"]


def remote(id, project=None, parent=None, title=None):
    row = issue(id)
    row["identifier"] = "ERI-" + id
    row["title"] = title or "Issue " + id
    row["project"] = PROJECTS[project] | {} if project else None
    row["parent"] = {"id": parent} if parent else None
    return row


PROJECTS = {
    "acme": {"id": "acme", "name": "Acme Web", "description": "", "archivedAt": None},
    "beta": {"id": "beta", "name": "Beta Launch", "description": "", "archivedAt": None},
}


class Structured(Fake):
    def __init__(self):
        super().__init__()
        self.rows = {
            "r1": remote("r1", "acme"),
            "r2": remote("r2", "beta"),
            "r3": remote("r3"),
            "r4": remote("r4", "acme", parent="r1"),
        }
        self.projects_list = copy.deepcopy(list(PROJECTS.values()))
        self.org = "ws"
        self.fail_discover = False

    def identity(self):
        meta = super().identity()
        meta["organization"]["id"] = self.org
        return meta

    def projects(self):
        if self.fail_discover:
            raise LinearFailure("unavailable", retry=True)
        return [{**p, "teams": {"nodes": [{"id": "team"}]}} for p in copy.deepcopy(self.projects_list)]

    def issue_places(self, filter):
        return [
            {"id": r["id"], "team": r["team"], "project": r["project"] and {"id": r["project"]["id"]},
             "parent": r["parent"]}
            for r in self.rows.values()
        ]


@pytest.fixture
def linear(monkeypatch):
    monkeypatch.setattr(get_settings(), "integration_encryption_key", Fernet.generate_key().decode())
    fake = Structured()
    for module in (sync, commands, mapping):
        monkeypatch.setattr(module, "LinearClient", lambda *_: fake)
    conn = sync.connect(OWNER, "fixture-key-never-returned")
    selection = run("linear.select", expected_revision=conn["revision"], team_ids=["team"])
    sync.process_sync(selection["job_id"])
    return fake


def resync():
    with session_scope() as db:
        job = sync.queue_sync(db, OWNER, force=True)
    sync.process_sync(job)


def task_of(remote_id):
    with session_scope() as db:
        return db.scalar(select(LinearIssue.task_id).where(LinearIssue.remote_id == remote_id))


def home_of(remote_id):
    with session_scope() as db:
        record = for_task(db, OWNER, task_of(remote_id))
        return db.get(StructureRecord, record.parent_id).title if record and record.parent_id else None


def plan(team_action="create", project_action="create", **meanings):
    return {
        **meanings,
        "rules": [
            {"source": "team", "source_id": "team", "action": team_action},
            {"source": "project", "source_id": "acme", "action": project_action},
            {"source": "project", "source_id": "beta", "action": project_action},
        ],
    }


MEANINGS = {"team": {"new_type": "Workstream"}, "project": {"type_id": "project"}}


def preview(body):
    with session_scope() as db:
        return mapping.preview(db, OWNER, mapping.Mapping.model_validate(body))


def apply(body, fingerprint):
    with session_scope() as db:
        return mapping.apply(db, OWNER, mapping.Apply.model_validate({**body, "fingerprint": fingerprint}))


def setup(body=None):
    body = body or plan(**MEANINGS)
    shown = preview(body)
    return shown, apply(body, shown["fingerprint"])


def counts():
    with session_scope() as db:
        return {
            "records": db.scalar(select(func.count()).select_from(StructureRecord)),
            "tasks": db.scalar(select(func.count()).select_from(Task)),
            "projects": db.scalar(select(func.count()).select_from(Project)),
            "types": len(db.get(StructureSchema, OWNER).definition["types"]),
            "links": db.scalar(select(func.count()).select_from(LinearIssue)),
        }


def test_connection_offers_the_review_and_meaning_is_never_assumed(linear, client):
    status = client.get("/api/v1/integrations/linear").json()
    assert status["structure"]["offer"] is True and status["structure"]["status"] == ""
    with pytest.raises(DomainError) as missing:
        preview(plan())
    assert missing.value.code == "LINEAR_DISCOVER"
    view = mapping.discover(OWNER)
    assert {p["name"]: p["issues"] for p in view["catalog"]["projects"]} == {"Acme Web": 2, "Beta Launch": 1}
    assert view["catalog"]["teams"][0]["issues"] == 4
    # Nothing defaults a team or project to Client (or any type): the owner must say what it is.
    assert view["meanings"] == {}
    with pytest.raises(DomainError) as unanswered:
        preview(plan(project={"type_id": "project"}))
    assert unanswered.value.code == "LINEAR_MEANING" and unanswered.value.data == {"source": "team"}
    # Skipping everything needs no meaning at all.
    assert preview(plan("skip", "skip"))["schema_changes"] == []


def test_preview_is_a_dry_run_showing_tree_counts_and_schema_changes(linear):
    mapping.discover(OWNER)
    with session_scope() as db:
        ensure(db, OWNER)
    before = counts()
    shown = preview(plan(**MEANINGS))
    assert counts() == before  # Dry run: no Home, type or task was created.
    assert [c["kind"] for c in shown["schema_changes"]] == ["new_type", "allow_home", "allow_home"]
    assert shown["proposal"]["id"] and shown["proposal"]["blocking_count"] == 0
    [team] = shown["tree"]["homes"]
    assert (team["title"], team["type_name"], team["new"], team["issues"]) == ("Team", "Workstream", True, 1)
    assert {(c["title"], c["issues"]) for c in team["children"]} == {("Acme Web", 2), ("Beta Launch", 1)}
    assert shown["tree"]["unfiled"] == 0
    # Already-imported issues: three would be filed; the subtask follows its Linear parent.
    assert shown["placement"]["moved"] == 3 and shown["placement"]["subtasks"] == 1
    assert {c["to"] for c in shown["placement"]["changes"]} == {"Team", "Acme Web", "Beta Launch"}


def test_apply_files_existing_issues_and_new_issues_inherit_their_home(linear):
    mapping.discover(OWNER)
    project_before = {r: task_of(r) for r in ("r1", "r2", "r3")}
    with session_scope() as db:
        legacy = {r: db.get(Task, t).project_id for r, t in project_before.items()}
    _, result = setup()
    assert result["applied"] and result["placement"]["moved"] == 3
    assert (home_of("r1"), home_of("r2"), home_of("r3")) == ("Acme Web", "Beta Launch", "Team")
    with session_scope() as db:
        schema_data(db, OWNER)  # Any record list adopts subtasks under their parent issue.
    assert home_of("r4") == "Issue r1"  # A subtask stays under its Linear parent issue.
    with session_scope() as db:
        acme = db.scalar(select(StructureRecord).where(StructureRecord.title == "Acme Web"))
        assert db.get(StructureRecord, acme.parent_id).title == "Team"
        assert acme.provenance["linear_source"] == {"kind": "project", "id": "acme"}
        schema = db.get(StructureSchema, OWNER).definition
        assert any(t["name"] == "Workstream" for t in schema["types"])
        # Filing never touched the Linear project and queued no Linear write.
        assert {r: db.get(Task, t).project_id for r, t in project_before.items()} == legacy
        assert db.scalar(select(func.count()).select_from(Job).where(Job.kind == "linear_write")) == 0
        assert db.get(LinearConnection, OWNER).structure["status"] == "applied"
    linear.rows["r5"] = remote("r5", "beta")
    resync()
    assert home_of("r5") == "Beta Launch"
    with session_scope() as db:
        # The project Home is actionable; sync keeps the issue under it rather than clearing the parent.
        assert db.get(Task, task_of("r1")).parent_task_id == acme.task_id is not None


def test_repeated_setup_and_sync_reuse_mappings_without_duplicates(linear):
    mapping.discover(OWNER)
    setup()
    after = counts()
    resync()
    resync()
    mapping.discover(OWNER)
    again = preview(plan(**MEANINGS))
    assert again["schema_changes"] == [] and again["proposal"] is None
    assert {h["action"] for h in again["homes"]} == {"existing"}
    assert again["placement"]["moved"] == 0 and again["placement"]["unchanged"] == 3
    result = apply(plan(**MEANINGS), again["fingerprint"])
    assert result["placement"]["moved"] == 0
    assert counts() == after
    assert (home_of("r1"), home_of("r3")) == ("Acme Web", "Team")


def test_partial_failure_rolls_back_and_retries_apply_once(linear, monkeypatch):
    mapping.discover(OWNER)
    with session_scope() as db:
        ensure(db, OWNER)
    before = counts()
    shown = preview(plan(**MEANINGS))
    real = mapping.place

    def broken(*args, **kwargs):
        if kwargs.get("write"):
            raise RuntimeError("worker lost mid-apply")
        return real(*args, **kwargs)

    monkeypatch.setattr(mapping, "place", broken)
    with pytest.raises(RuntimeError):
        apply(plan(**MEANINGS), shown["fingerprint"])
    assert counts() == before  # Schema, Homes and placements all rolled back together.
    with session_scope() as db:
        assert db.get(LinearConnection, OWNER).structure.get("status") != "applied"
    monkeypatch.setattr(mapping, "place", real)
    first = apply(plan(**MEANINGS), shown["fingerprint"])
    # The response was lost; the browser retries the same apply.
    second = apply(plan(**MEANINGS), shown["fingerprint"])
    assert first["repeated"] is False and second["repeated"] is True
    with session_scope() as db:
        titles = list(db.scalars(select(StructureRecord.title).where(StructureRecord.title.in_(
            ["Team", "Acme Web", "Beta Launch"]))))
    assert sorted(titles) == ["Acme Web", "Beta Launch", "Team"]
    # A failed sync keeps every Home; its retry files the new issue once.
    linear.rows["r5"] = remote("r5", "acme")
    linear.fail_sync = True
    resync()
    assert task_of("r5") is None and home_of("r1") == "Acme Web"
    linear.fail_sync = False
    resync()
    resync()
    assert home_of("r5") == "Acme Web"
    with session_scope() as db:
        assert db.scalar(select(func.count()).select_from(LinearIssue).where(LinearIssue.remote_id == "r5")) == 1


def test_discovery_failure_changes_nothing_and_can_be_retried(linear):
    linear.fail_discover = True
    with pytest.raises(DomainError) as failed:
        mapping.discover(OWNER)
    assert failed.value.code == "LINEAR_UNAVAILABLE"
    with session_scope() as db:
        assert db.get(LinearConnection, OWNER).structure == {}
    linear.fail_discover = False
    assert mapping.discover(OWNER)["catalog"]["projects"]


def test_reconnect_keeps_the_mapping_for_the_same_workspace(linear, client):
    mapping.discover(OWNER)
    setup()
    after = counts()
    assert client.post("/api/v1/integrations/linear/disconnect").status_code == 200
    sync.connect(OWNER, "fixture-key-again")
    with session_scope() as db:
        assert mapping.overview(db.get(LinearConnection, OWNER))["status"] == "applied"
    linear.rows["r5"] = remote("r5", "beta")
    resync()
    assert home_of("r5") == "Beta Launch"
    assert counts()["records"] == after["records"] + 1 and counts()["types"] == after["types"]
    # Another Linear workspace has other IDs: its mapping starts fresh and is offered again.
    linear.org = "other-ws"
    sync.connect(OWNER, "fixture-key-other")
    with session_scope() as db:
        conn = db.get(LinearConnection, OWNER)
        assert mapping.overview(conn)["status"] == ""
        conn.team_ids = ["team"]
    with session_scope() as db:
        assert mapping.overview(db.get(LinearConnection, OWNER))["offer"] is True


def test_renamed_and_archived_source_projects(linear):
    mapping.discover(OWNER)
    body = plan(**MEANINGS)
    body["rules"][2]["title"] = "Beta (client)"
    setup(body)
    after = counts()
    for row in linear.rows.values():
        if row["project"] and row["project"]["id"] == "acme":
            row["project"]["name"] = "Acme Website"
        if row["project"] and row["project"]["id"] == "beta":
            row["project"]["name"] = "Beta Relaunch"
            row["project"]["archivedAt"] = "2026-10-01T00:00:00Z"
    resync()
    assert home_of("r1") == "Acme Website"  # Untouched Home follows the Linear rename.
    assert home_of("r2") == "Beta (client)"  # A title chosen here is kept.
    assert counts() == after  # Same Homes, same tasks, same legacy project rows.
    with session_scope() as db:
        rules = db.get(LinearConnection, OWNER).structure["rules"]
        beta = db.get(StructureRecord, rules["project:beta"]["record_id"])
        assert rules["project:beta"]["archived"] is True and not beta.archived
        assert rules["project:acme"]["name"] == "Acme Website"
    linear.projects_list[1]["archivedAt"] = "2026-10-01T00:00:00Z"
    view = mapping.discover(OWNER)
    assert next(p for p in view["catalog"]["projects"] if p["id"] == "beta")["archived"] is True
    assert preview(body)["schema_changes"] == []


def test_user_corrections_are_kept_and_filing_never_moves_the_linear_project(linear):
    mapping.discover(OWNER)
    setup()
    with session_scope() as db:
        schema = ensure(db, OWNER)
        elsewhere = db.scalar(select(StructureRecord).where(StructureRecord.title == "Beta Launch"))
        record = for_task(db, OWNER, task_of("r1"))
        args = {"record_id": record.id, "expected_revision": record.revision,
                "schema_revision": schema.revision, "parent_id": elsewhere.id}
        project_before = db.get(Task, record.task_id).project_id
    run("record.update", **args)
    with session_scope() as db:
        assert db.get(Task, task_of("r1")).project_id == project_before
        assert db.scalar(select(func.count()).select_from(Job).where(Job.kind == "linear_write")) == 0
    resync()
    assert home_of("r1") == "Beta Launch"  # Sync keeps the owner's correction.
    # Remap Acme to a different existing Home: the preview lists the kept exception.
    target = run("record.create", type_id="client", title="Acme Corp", parent_id=None,
                 schema_revision=schema.revision)
    body = plan(**MEANINGS)
    body["rules"][1] = {"source": "project", "source_id": "acme", "action": "reuse", "record_id": target["id"]}
    shown = preview(body)
    outcomes = {(c["title"], c["outcome"]) for c in shown["placement"]["changes"]}
    assert ("Issue r1", "kept") in outcomes
    assert shown["placement"]["moved"] == 0  # r4 is a subtask; r1 is the owner's exception.
    apply(body, shown["fingerprint"])
    assert home_of("r1") == "Beta Launch"
    linear.rows["r6"] = remote("r6", "acme")
    resync()
    assert home_of("r6") == "Acme Corp"


def test_source_project_edit_uses_write_back_and_the_issue_follows_its_mapping(linear):
    mapping.discover(OWNER)
    setup()
    beta_local = mapping.legacy_project_id(OWNER, "ws", "beta")
    with session_scope() as db:
        conn = db.get(LinearConnection, OWNER)
        conn.directory = {**conn.directory, "projects": [
            {**p, "local_id": mapping.legacy_project_id(OWNER, "ws", p["id"])} for p in PROJECTS.values()]}
        task = db.get(Task, task_of("r1"))
        revision = task.revision
    saved = run("task.update", task_id=task_of("r1"), expected_revision=revision, project_id=beta_local)
    assert saved["external"]["sync_state"] == "pending"
    assert home_of("r1") == "Acme Web"  # Nothing moves until Linear confirms.
    sync.process_write(saved["external"]["job_id"])
    assert linear.writes[-1][1] == {"projectId": "beta"}
    assert home_of("r1") == "Beta Launch"  # Filed as soon as Linear confirmed the project.
    linear.rows["r1"]["project"] = copy.deepcopy(PROJECTS["beta"])
    resync()
    assert home_of("r1") == "Beta Launch"


def test_skip_imports_normally_and_routes_need_a_preview(linear, client):
    assert client.post("/api/v1/integrations/linear/mapping/skip").json()["status"] == "skipped"
    assert client.get("/api/v1/integrations/linear").json()["structure"]["offer"] is False
    linear.rows["r5"] = remote("r5", "beta")
    resync()
    assert home_of("r5") is None
    found = client.post("/api/v1/integrations/linear/mapping/discover").json()
    assert {t["id"] for t in found["types"]} >= {"client", "project"}
    body = plan(**MEANINGS)
    shown = client.post("/api/v1/integrations/linear/mapping/preview", json=body).json()
    stale = client.post("/api/v1/integrations/linear/mapping/apply", json={**body, "fingerprint": "0" * 64})
    assert stale.status_code == 409
    done = client.post("/api/v1/integrations/linear/mapping/apply", json={**body, "fingerprint": shown["fingerprint"]})
    assert done.status_code == 200 and done.json()["structure"]["status"] == "applied"
    view = client.get("/api/v1/integrations/linear/mapping").json()
    assert {r["key"]: bool(r["home"]) for r in view["rules"]} == {
        "team:team": True, "project:acme": True, "project:beta": True}
    assert home_of("r5") == "Beta Launch"
