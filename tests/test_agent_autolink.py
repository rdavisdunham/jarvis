"""Stage E: the dream sequence's session auto-link (agent_autolink.py) and its durable Undo."""

import math
from datetime import timedelta

import pytest
from jarvis import agent_autolink as autolink
from jarvis import agent_sessions as sessions
from jarvis.db import session_scope
from jarvis.memory_learning import EMBEDDING_MODEL
from jarvis.models import Job, LinearConnection, LinearIssue, OwnerSettings, Task, now
from jarvis.search_models import SearchDocument
from jarvis.session_models import AgentSession, AgentSessionRule, AutolinkRejection
from sqlalchemy import func, select
from test_organization_foundation import OWNER, create

DIMS = 8


def unit(angle_from_axis, axis=0):
    """A unit vector at a given angle from basis axis `axis` (cosine with that axis = cos(angle))."""
    v = [0.0] * DIMS
    v[axis] = math.cos(angle_from_axis)
    v[(axis + 1) % DIMS] = math.sin(angle_from_axis)
    return v


def similar(cos, axis=0):
    return unit(math.acos(cos), axis)


# ---------------------------------------------------------------- labelled ranking fixtures (pure)

# (label, scored candidates, rejected, expected task or None). Scores are cosine + issue bonus.
LABELLED = [
    ("clear winner", {"login": 0.81, "billing": 0.52}, set(), "login"),
    ("sibling projects too close", {"login-web": 0.78, "login-ios": 0.71}, set(), None),
    ("duplicate tasks never auto-link", {"dup-a": 0.74, "dup-b": 0.74}, set(), None),
    ("weak similarity", {"login": 0.55}, set(), None),
    ("single candidate above threshold", {"login": 0.66}, set(), "login"),
    ("abbreviation: issue key named, modest text match", {"eng-12": 0.41 + autolink.ISSUE_BONUS, "other": 0.40},
     set(), "eng-12"),
    ("two issue keys named: neither wins", {"eng-12": 0.5 + autolink.ISSUE_BONUS, "eng-13": 0.48 + autolink.ISSUE_BONUS},
     set(), None),
    ("closed task still counts", {"closed-last-week": 0.80, "open-other": 0.40}, set(), "closed-last-week"),
    ("rejected top never wins and never promotes the runner-up", {"undone": 0.90, "next": 0.70}, {"undone"}, None),
    ("empty home", {}, set(), None),
]


@pytest.mark.parametrize("label,scored,rejected,expected", LABELLED, ids=[c[0] for c in LABELLED])
def test_labelled_ranking_fixtures(label, scored, rejected, expected):
    picked = autolink.choose(scored, rejected)
    assert (picked[0] if picked else None) == expected, label


def test_thresholds_are_the_documented_provisional_constants():
    assert (autolink.THRESHOLD, autolink.MARGIN) == (0.62, 0.12)
    assert autolink.choose({"a": 0.62, "b": 0.50}) == ("a", 0.62, 0.12)  # Both bounds inclusive.
    assert autolink.choose({"a": 0.6199}) is None
    assert autolink.choose({"a": 0.70, "b": 0.5801}) is None


def test_score_is_best_chunk_cosine_plus_issue_bonus():
    session = unit(0)
    assert autolink.score(session, [similar(0.2), similar(0.7)], False) == pytest.approx(0.7)
    assert autolink.score(session, [], True) == pytest.approx(autolink.ISSUE_BONUS)


# ---------------------------------------------------------------- the dream cycle


@pytest.fixture
def home():
    """A project with two tasks under it, a folder rule for its repo, and search vectors for both tasks."""
    project = create("project", "Acme portal")
    login = create("task", "Fix the flaky login test", parent_id=project["id"])
    billing = create("task", "Billing export", parent_id=project["id"])
    with session_scope() as db:
        db.get(OwnerSettings, OWNER) or db.add(OwnerSettings(owner_id=OWNER, values={}))
        db.flush()
        row = db.get(OwnerSettings, OWNER)
        row.values = {**(row.values or {}), "remotea_enabled": True}
        db.add(AgentSessionRule(owner_id=OWNER, path_prefix="/home/davin/acme", record_id=project["id"]))
        for record, vector in ((login, similar(0.9)), (billing, similar(0.3))):
            db.add(SearchDocument(owner_id=OWNER, target_key="record:" + record["id"], fingerprint="f", content="x",
                                  embedding_model=EMBEDDING_MODEL, vectors=[vector]))
    return {"project": project, "login": login, "billing": billing}


def session(session_id="s1", vector=None, **values):
    with session_scope() as db:
        row = sessions.claim(db, OWNER, "claude_code", session_id)
        row.cwd, row.title, row.first_prompt = "/home/davin/acme/web", "Investigate login flake", "login test fails"
        row.last_prompt, row.branch, row.repo, row.binding_source = "", "", "", ""
        row.last_active_at = now()
        for key, value in values.items():
            setattr(row, key, value)
        row.embedding, row.embedding_model = vector or unit(0), EMBEDDING_MODEL
        row.summary_fingerprint = "sum-1"
        sessions.resolve(db, row)
        db.flush()
        return row.id


def filed(session_pk):
    with session_scope() as db:
        row = db.get(AgentSession, session_pk)
        return row.binding_source, row.task_id, row.record_id


def cycle():
    with session_scope() as db:
        return autolink.cycle(db, OWNER)


def test_a_clear_session_is_linked_by_the_dream_and_survives_refresh_and_rebind(home):
    pk = session()
    assert filed(pk) == ("path_rule", home["project"]["task_id"], home["project"]["id"])
    assert cycle()["linked"] == 1
    source, task_id, record_id = filed(pk)
    assert (source, task_id, record_id) == ("auto", home["login"]["task_id"], home["login"]["id"])
    with session_scope() as db:  # A collector refresh: the turn count and last prompt change, the evidence doesn't.
        row = db.get(AgentSession, pk)
        row.turn_count, row.last_prompt = 42, "and now the CI run"
        sessions.resolve(db, row)
    assert filed(pk)[0] == "auto"
    with session_scope() as db:
        sessions.rebind(db, OWNER)  # A rule edit re-files unprotected sessions; auto stays.
    assert filed(pk) == ("auto", home["login"]["task_id"], home["login"]["id"])
    assert cycle()["linked"] == 0  # Already linked.


def test_undo_tombstones_the_pair_and_survives_resolve_rebind_and_the_next_cycle(home, client):
    pk = session()
    cycle()
    assert filed(pk)[0] == "auto"
    response = client.post(f"/api/v1/sessions/{pk}/autolink/undo")
    assert response.status_code == 200, response.text
    assert response.json()["filed"]["source"] == "path_rule"  # Back under its folder rule's home.
    with session_scope() as db:
        tomb = db.scalar(select(AutolinkRejection))
        assert (tomb.session_pk, tomb.task_id) == (pk, home["login"]["task_id"]) and tomb.evidence_revision
        sessions.resolve(db, db.get(AgentSession, pk))
        sessions.rebind(db, OWNER)
    counts = cycle()
    assert counts["linked"] == 0 and counts["blocked_by_undo"] == 1
    assert filed(pk) == ("path_rule", home["project"]["task_id"], home["project"]["id"])
    again = client.post(f"/api/v1/sessions/{pk}/autolink/undo")
    assert again.status_code == 409  # Only a dream link can be undone.


def test_clearing_or_replacing_a_dream_link_by_hand_is_an_undo(home):
    pk = session()
    cycle()
    with session_scope() as db:
        sessions.link(db, OWNER, "claude_code", "s1", sessions.LinkInput(task_id=home["billing"]["task_id"]))
        assert db.scalar(select(func.count()).select_from(AutolinkRejection)) == 1
    assert filed(pk)[0] == "manual"


def test_changed_evidence_lets_resolve_refile_and_the_dream_reconsider(home):
    pk = session()
    cycle()
    with session_scope() as db:
        row = db.get(AgentSession, pk)
        row.title = "Something else entirely"
        sessions.resolve(db, row)
        assert row.binding_source == "path_rule" and row.binding_evidence == ""


def test_candidates_include_recently_completed_tasks_only(home):
    with session_scope() as db:
        task = db.get(Task, home["login"]["task_id"])
        task.status, task.completed_at = "completed", now() - timedelta(days=3)
    session()
    assert cycle()["linked"] == 1
    session("s2")
    with session_scope() as db:
        db.get(Task, home["login"]["task_id"]).completed_at = now() - timedelta(days=45)
    assert cycle()["linked"] == 0  # The old task dropped out; billing alone is far below the threshold.


def test_exact_identifiers_count_only_in_the_connected_workspace(home):
    loose = create("task", "Unfiled issue task")
    with session_scope() as db:
        db.add(LinearConnection(owner_id=OWNER, enabled=True, workspace_id="acme", workspace_name="Acme",
                                viewer_id="me", credentials="sealed"))
        db.add(LinearIssue(owner_id=OWNER, workspace_id="other", remote_id="r-9", task_id=loose["task_id"],
                           snapshot={"identifier": "ENG-9"}))
    pk = session("s3", cwd="/elsewhere", title="ENG-9 and ENG-10 follow-up", first_prompt="see ENG-9")
    with session_scope() as db:
        row = db.get(AgentSession, pk)
        assert [r["identifier"] for r in row.issue_refs] == ["ENG-9"]
        # Another workspace's ENG-9 is not this workspace's: it never earns the issue bonus.
        assert autolink.workspace_issues(db, OWNER, row.issue_refs) == set()
        db.get(LinearIssue, db.scalar(select(LinearIssue.id))).workspace_id = "acme"
        assert autolink.workspace_issues(db, OWNER, row.issue_refs) == {loose["task_id"]}


def test_inferred_links_never_start_work(home):
    session()
    cycle()
    with session_scope() as db:
        from jarvis.session_models import AgentCommand, AgentRun

        assert db.scalar(select(func.count()).select_from(AgentRun)) == 0
        assert db.scalar(select(func.count()).select_from(AgentCommand)) == 0


def test_the_daily_job_is_queued_at_most_every_twenty_hours_for_enabled_owners(home):
    with session_scope() as db:
        assert len(autolink.queue_autolink(db)) == 1
        assert autolink.queue_autolink(db) == []
        assert len(autolink.queue_autolink(db, clock=now() + timedelta(hours=21))) == 1
        db.get(OwnerSettings, OWNER).values = {"remotea_enabled": False}
        assert autolink.queue_autolink(db, clock=now() + timedelta(days=3)) == []
    session()
    with session_scope() as db:
        db.get(OwnerSettings, OWNER).values = {"remotea_enabled": True}
        job = db.scalars(select(Job).where(Job.kind == "agent_autolink")).first()
        job_id = job.id
    autolink.process(job_id)
    with session_scope() as db:
        job = db.get(Job, job_id)
        assert job.status == "succeeded" and job.result["linked"] == 1
