"""Set up with Dani: the Terra synthesis that turns the interview into a proposal (docs/SETUP_WITH_DANI_PLAN.md, PR 4)."""
import json
from uuid import uuid4

import pytest
from cryptography.fernet import Fernet
from sqlalchemy import func, select

import jarvis.domain  # noqa: F401  (load the command registry before the module that registers into it)
from jarvis import setup_synthesis as synthesis
from jarvis.config import get_settings
from jarvis.db import session_scope
from jarvis.models import Area, Job, Memory, Source, Space
from jarvis.setup_models import OrganizationProposal, SetupSession


@pytest.fixture(autouse=True)
def model_settings(monkeypatch):
    monkeypatch.setenv("JARVIS_INTEGRATION_ENCRYPTION_KEY", Fernet.generate_key().decode())
    monkeypatch.setenv("JARVIS_OPENAI_API_KEY", "synthetic")
    monkeypatch.setenv("JARVIS_EXTERNAL_SERVICES_ENABLED", "true")
    monkeypatch.setenv("JARVIS_COST_TRACKING_ENABLED", "false")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def owner():
    return get_settings().owner_id


def started(client):
    session = client.post("/api/v1/setup/start", json={}).json()
    with session_scope() as db:
        db.add(Source(owner_id=owner(), conversation_id=session["conversation_id"], kind="transcript", role="user",
                      native_id="t:" + str(uuid4()), content="I teach chemistry and coach the robotics team."))
    return session


def node(id, kind, title, parent=None, existing=None, action="create"):
    return {"id": id, "parent_id": parent, "kind": kind, "title": title, "description": "", "existing_ref": existing,
            "action": action}


def answer(tree, memories=(), covered=("you", "week")):
    return synthesis.SetupProposal.model_validate(
        {"notes": {"covered": list(covered), "missing": ["Current projects"]}, "tree": tree,
         "core_memories": list(memories)})


def test_propose_builds_a_preview_from_the_interview_and_changes_no_records(client):
    with session_scope() as db:
        db.add(Space(owner_id=owner(), name="Work"))
    session = started(client)
    state = client.post("/api/v1/setup/propose").json()
    assert state["status"] == "proposing" and state["job_id"] and state["can_propose"] is False
    # Pressing again while it is being built reuses the same job.
    assert client.post("/api/v1/setup/propose").json()["job_id"] == state["job_id"]
    seen = {}

    def model(body):
        seen["body"] = body
        material = json.loads(body["messages"][1]["content"])
        work = next(r["ref"] for r in material["organization"] if r["name"] == "Work")
        user = next(t["ref"] for t in material["conversation"] if t["role"] == "user")
        return answer([
            node("n1", "space", "Work", existing=work, action="keep"),
            node("n2", "area", "Teaching", "n1"),
            node("n3", "project", "Robotics season", "n2"),
            node("n4", "area", "Orphan", "missing"),
            node("n5", "area", "Invented", "n1", existing="r99", action="rename"),
        ], [{"content": "Davin teaches chemistry.", "fact_key": "job", "evidence_source_id": user,
             "evidence": "I teach chemistry", "existing_memory_id": None, "action": "add"},
            {"content": "Made up.", "fact_key": "x", "evidence_source_id": "s404", "evidence": "",
             "existing_memory_id": "m404", "action": "retire"}]), {"model": "gpt-6.1-terra", "cost": 0.04}

    synthesis.process(state["job_id"], model=model)
    body = seen["body"]
    assert body["response_format"]["json_schema"]["strict"] is True
    assert body["model"] == "gpt-6.1-terra"
    material = json.loads(body["messages"][1]["content"])
    # The model sees refs, never record or source IDs.
    with session_scope() as db:
        space_id = db.scalar(select(Space.id))
    assert space_id not in body["messages"][1]["content"] and session["conversation_id"] not in body["messages"][1]["content"]
    assert material["conversation"][0]["role"] == "assistant"  # The fixed opening comes first.

    after = client.get("/api/v1/setup").json()
    assert after["status"] == "reviewing" and after["tree_proposal_id"] and after["can_propose"] is True
    assert after["notes"]["covered"] == ["you", "week"] and after["notes"]["propose_error"] is None
    assert [m["content"] for m in after["memory_candidates"]] == ["Davin teaches chemistry."]
    assert after["memory_candidates"][0]["evidence_source_id"]
    with session_scope() as db:
        proposal = db.get(OrganizationProposal, after["tree_proposal_id"])
        [work] = proposal.tree
        assert (work["existing_id"], work["action"]) == (space_id, "keep")
        # The orphan is dropped; an invented record ref becomes a new row.
        assert [(c["title"], c["action"]) for c in work["children"]] == [("Teaching", "create"), ("Invented", "create")]
        assert work["children"][0]["children"][0]["title"] == "Robotics season"
        assert db.scalar(select(func.count()).select_from(Area)) == 0
        row = db.get(SetupSession, owner())
        assert row.model == "gpt-6.1-terra" and row.cost == pytest.approx(0.04)
        said = db.scalars(select(Source.content).where(Source.conversation_id == session["conversation_id"],
                                                       Source.role == "assistant").order_by(Source.created_at)).all()
        assert said[-1] == synthesis.DRAFT
        assert db.get(Job, state["job_id"]).result == {"stored": True, "rows": 4, "memories": 1}


def test_a_failed_proposal_returns_to_the_interview_and_says_so(client):
    started(client)
    job = client.post("/api/v1/setup/propose").json()["job_id"]

    def broke(body):
        raise synthesis.NoProposal("budget", "BUDGET_EXCEEDED")

    synthesis.process(job, model=broke)
    state = client.get("/api/v1/setup").json()
    assert state["status"] == "interviewing" and state["notes"]["propose_error"] == "budget"
    with session_scope() as db:
        last = db.scalars(select(Source.content).where(Source.role == "assistant").order_by(Source.created_at.desc())).first()
        assert last == synthesis.FAILED["budget"]


def test_a_restarted_run_discards_the_old_job(client):
    started(client)
    job = client.post("/api/v1/setup/propose").json()["job_id"]
    client.post("/api/v1/setup/start", json={"restart": True})
    synthesis.process(job, model=lambda body: pytest.fail("no model call for a stale run"))
    with session_scope() as db:
        assert db.get(Job, job).result == {"stored": False, "reason": "stale"}
        assert db.get(SetupSession, owner()).status == "interviewing"


def test_nothing_to_propose_keeps_interviewing(client):
    started(client)
    job = client.post("/api/v1/setup/propose").json()["job_id"]
    synthesis.process(job, model=lambda body: answer([], covered=()))
    state = client.get("/api/v1/setup").json()
    assert state["status"] == "interviewing" and state["tree_proposal_id"] is None


def test_memory_candidates_only_trust_known_refs():
    refs = {"source": {"s1": "src"}, "record": {}, "memory": {"m1": "mem"}}
    rows = [synthesis.CoreMemory(content=c, fact_key="k", evidence_source_id=s, evidence="", existing_memory_id=m, action=a)
            for c, s, m, a in [("Kept.", "s1", "m1", "add"), ("New.", "s9", None, "add"), ("Fix.", None, "m9", "update"),
                               ("Gone.", None, "m9", "retire")]]
    out = synthesis.candidates(rows, refs)
    assert [(r["content"], r["action"], r["existing_memory_id"], r["evidence_source_id"]) for r in out] == [
        ("Kept.", "keep", "mem", "src"), ("New.", "add", None, None), ("Fix.", "add", None, None)]


def test_build_tree_places_rows_by_kind_and_depth():
    refs = {"source": {}, "record": {"r1": ("area", "area-id")}, "memory": {}}
    rows = [synthesis.FlatNode(**n) for n in [
        node("s", "space", "Home"), node("g", "goal", "Run a marathon", "s"), node("p", "project", "Training plan", "g"),
        node("a", "area", "Health", "s", existing="r1", action="keep"),
        node("x", "space", "Wrong kind", existing="r1", action="keep"),
        node("bad", "area", "Under a goal", "g"), node("gone", "area", "Archive me", "s", existing="r2", action="archive"),
        node("s", "space", "Duplicate id")]]
    [home, wrong] = synthesis.build_tree(rows, refs)
    assert [c.title for c in home.children] == ["Run a marathon", "Health"]
    assert home.children[0].children[0].title == "Training plan"
    assert (home.children[1].existing_id, home.children[1].action) == ("area-id", "keep")
    assert (wrong.existing_id, wrong.action) == (None, "create")


def test_setup_tools_are_registered():
    from jarvis.tool_catalog import DESCRIPTIONS, GROUPS
    from jarvis.tools import READ_TOOLS, VOICE_MUTATIONS, registry

    names = {t["name"] for t in registry()}
    assert {"setup_state", "setup_propose", "setup_tree_apply"} <= names <= set(DESCRIPTIONS) | names
    assert set(GROUPS["setup_interview"][1]) == {"setup_state", "setup_propose", "setup_tree_apply"}
    assert "setup_state" in READ_TOOLS and {"setup.propose", "setup.tree_apply"} <= VOICE_MUTATIONS
    assert all(name in DESCRIPTIONS for name in GROUPS["setup_interview"][1])


def test_memory_reads_are_bounded(client):
    session = started(client)
    with session_scope() as db:
        source = db.scalar(select(Source).where(Source.conversation_id == session["conversation_id"], Source.role == "user"))
        db.add_all([Memory(owner_id=owner(), source_id=source.id, content="Fact " + str(i) + " " + "x" * 400)
                    for i in range(synthesis.MEMORIES + 5)])
    with session_scope() as db:
        material, refs = synthesis.gather(db, owner(), db.get(SetupSession, owner()))
    assert len(material["saved_facts_newest_first"]) == synthesis.MEMORIES
    assert all(len(f["fact"]) <= synthesis.MEMORY_CHARS for f in material["saved_facts_newest_first"])
    assert set(refs["memory"]) == {f["ref"] for f in material["saved_facts_newest_first"]}
