from datetime import timedelta
from uuid import uuid4
import pytest
from sqlalchemy import select
from jarvis.config import get_settings
from jarvis.db import session_scope
from jarvis.models import Job, Source, Conversation, AuthSession, now
from jarvis.structure_models import StructureRecord, RoutingObservation, RoutingPattern
from jarvis.search_models import SearchAlias, SearchSession, SearchIndexState, SearchDocument
from jarvis.search_schema import SearchSelection, SearchEvent, AliasChange
from jarvis import search_service as service, search_index as indexing, search_learning as learning
from test_structure import create, run, definition, propose, apply, OWNER


@pytest.fixture(autouse=True)
def enabled(monkeypatch):
    monkeypatch.setattr(get_settings(), "semantic_search_enabled", True)

    def vectors(owner, texts, *args):
        return [
            [1.0, 0.0, 0.0]
            if any(w in t.lower() for w in ("pest", "insect", "extermin", "spraying"))
            else [0.0, 1.0, 0.0]
            if "software" in t.lower()
            else [0.0, 0.0, 1.0]
            for t in texts
        ]

    monkeypatch.setattr(indexing, "embeddings", vectors)
    monkeypatch.setattr(service, "embeddings", vectors)


def indexed():
    with session_scope() as db:
        job = indexing.queue_index(db, OWNER, force=True)
    indexing.index_workspace(job)


def search(query, **args):
    return service.search(OWNER, OWNER, dict(query=query, **args))


def ids(group):
    return {m["record"]["id"] for m in group["items"]}


def test_structured_and_possible_lanes_keep_wrongly_filed_records():
    abc = create(title="ABC", body="A pest control company")
    other = create(title="Other client", body="A software company")
    linked = create("task", "Renew spraying plan", parent_id=abc["id"])
    loose = create("task", "Review insect treatment", body="Discuss the extermination contract")
    wrong = create("note", "Pest service notes", parent_id=other["id"])
    unrelated = create("task", "Ship software update", parent_id=other["id"])
    indexed()
    result = search("pest control company", resolved=["record:" + abc["id"]])
    assert linked["id"] in ids(result["structured"])
    assert {loose["id"], wrong["id"]} <= ids(result["possible"])
    assert unrelated["id"] not in ids(result["possible"])
    assert wrong["parent_id"] == other["id"]
    strict = search("pest control", resolved=["record:" + abc["id"]], strict=True)
    assert not strict["possible"]["items"]
    filtered = search("pest control", resolved=["record:" + abc["id"]], home_id=other["id"])
    assert loose["id"] not in ids(filtered["possible"])
    assert wrong["id"] in ids(filtered["possible"])


def test_current_revision_and_provider_failure_preserve_keyword_search(monkeypatch):
    task = create("task", "Pest visit")
    indexed()

    def change_while_waiting(*args):
        run(
            "record.update",
            dict(
                record_id=task["id"],
                expected_revision=task["revision"],
                schema_revision=1,
                title="Software release",
            ),
        )
        raise TimeoutError()

    monkeypatch.setattr(service, "embeddings", change_while_waiting)
    result = search("pest")
    assert task["id"] not in ids(result["possible"])
    assert result["mode"] == "keyword_fallback" and result["index"]["pending"] > 0


def selected(conversation=None):
    abc = create(title="ABC", body="Pest control company")
    task = create("task", "Spraying visit", parent_id=abc["id"])
    result = service.search(
        OWNER,
        OWNER,
        dict(query="pest control company", resolved=["record:" + abc["id"]]),
        conversation_id=conversation,
    )
    with session_scope() as db:
        learning.choose(
            db,
            OWNER,
            OWNER,
            SearchSelection(
                search_id=result["search_id"],
                target_key="record:" + abc["id"],
                phrase="pest control company",
                record_ids=[task["id"]],
            ),
        )
    return result, abc, task


def test_learning_requires_presentation_and_use_and_forget_suppresses_recreation():
    result, abc, task = selected()
    with session_scope() as db:
        before_observations = set(db.scalars(select(RoutingObservation.id)))
        row = db.get(SearchSession, result["search_id"])
        learning.accept_search(db, row, "continued:test")
        assert not list(db.scalars(select(SearchAlias)))
        learning.event(db, OWNER, OWNER, SearchEvent(search_id=row.id, kind="used", record_id=task["id"]))
        learning.event(db, OWNER, OWNER, SearchEvent(search_id=row.id, kind="used", record_id=task["id"]))
        alias = db.scalar(select(SearchAlias))
        assert alias.status == "provisional"
        assert learning.state(db, OWNER, OWNER)["items"][0]["positive_count"] == 1
        learning.change_alias(
            db, OWNER, OWNER, alias.id, AliasChange(expected_revision=alias.revision, action="forget")
        )
        assert row.suppressed
        row.outcome = "unknown"
        row.suppressed = False
        learning.accept_search(db, row, "continued:another")
        assert row.suppressed and alias.status == "forgotten"
        assert set(db.scalars(select(RoutingObservation.id))) == before_observations


def test_continuation_and_late_correction_retract_alias():
    with session_scope() as db:
        db.add(
            AuthSession(
                token_hash="test-search",
                owner_id=OWNER,
                device_id="synthetic",
                csrf="test",
                expires_at=now() + timedelta(days=1),
            )
        )
        conv = Conversation(owner_id=OWNER, device_id="synthetic", private=False, learning=False)
        db.add(conv)
        db.flush()
        cid = conv.id
    result, abc, task = selected(cid)
    with session_scope() as db:
        row = db.get(SearchSession, result["search_id"])
        row.presented_at = now() - timedelta(seconds=2)
        source = Source(
            owner_id=OWNER,
            kind="text",
            native_id="test:continue",
            conversation_id=cid,
            role="user",
            content="Great, add another task",
            created_at=now(),
        )
        db.add(source)
        db.flush()
        learning.observe_source(db, source)
        assert row.outcome == "accepted"
        correction = Source(
            owner_id=OWNER,
            kind="text",
            native_id="test:correct",
            conversation_id=cid,
            role="user",
            content="No, I meant a different client",
            created_at=now(),
        )
        db.add(correction)
        db.flush()
        learning.observe_source(db, correction)
        assert row.outcome == "corrected" and db.get(SearchAlias, row.alias_id).status == "conflict"


def test_alias_routes_only_after_weekly_review_confirmation():
    result, abc, task = selected()
    with session_scope() as db:
        learning.event(
            db, OWNER, OWNER, SearchEvent(search_id=result["search_id"], kind="used", record_id=task["id"])
        )
        questions = learning.routing_candidates(db, OWNER)
        assert questions and all(q["evidence_count"] == 0 and q["held_out"] == 0 for q in questions)
        rules = list(db.scalars(select(RoutingPattern)))
        assert rules and all(
            r.status == "candidate" and r.origin == "search_candidate" and not r.evidence_ids for r in rules
        )
    fresh = create("task", "pest control company renewal")
    assert fresh["parent_id"] is None


def test_duplicate_names_remain_ambiguous_and_schema_aliases_survive_rename():
    first = create(title="ABC")
    second = create(title="ABC")
    result = search("ABC")
    assert result["ambiguous"] and result["resolved"] == []
    assert {"record:" + first["id"], "record:" + second["id"]} <= {r["key"] for r in result["resolutions"]}
    original = definition()
    t = next(t for t in original["types"] if t["id"] == "task")
    t["fields"].append(
        dict(
            id="client",
            name="Client",
            description="The customer purchasing this work",
            kind="relation",
            target_types=["client"],
            inherit=True,
        )
    )
    apply(propose(original))
    r = search("customer", resolved=["field:task:client"])
    with session_scope() as db:
        learning.choose(
            db,
            OWNER,
            OWNER,
            SearchSelection(search_id=r["search_id"], target_key="field:task:client", phrase="customer"),
        )
        session = db.get(SearchSession, r["search_id"])
        session.presented_at = now()
        learning.accept_search(db, session, "continued:test")
    current = definition()
    next(f for t in current["types"] if t["id"] == "task" for f in t["fields"] if f["id"] == "client")[
        "name"
    ] = "Account"
    apply(propose(current))
    assert "field:task:client" in search("customer")["resolved"]


def test_full_collection_over_one_thousand_and_pagination():
    definition()
    with session_scope() as db:
        db.add_all(
            StructureRecord(
                owner_id=OWNER,
                type_id="note",
                title="Older research " + str(i),
                body="rareterm evidence",
                values={},
                schema_revision=1,
                provenance={},
            )
            for i in range(1007)
        )
    first = search("rareterm", capability="content", limit=100)
    assert first["possible"]["total"] == 1007 and first["possible"]["next_offset"] == 100
    last = search("rareterm", capability="content", limit=100, offset=1000)
    assert len(last["possible"]["items"]) == 7 and last["possible"]["next_offset"] is None


def test_other_account_cannot_read_search_evidence_or_influence_aliases():
    result, _, _ = selected()
    with session_scope() as db:
        from jarvis.domain import DomainError

        with pytest.raises(DomainError):
            learning.require_session(db, OWNER, "another-account", result["search_id"])
    result = service.search("another-account", "another-account", dict(query="pest"))
    assert not result["structured"]["items"] and not result["possible"]["items"]


def test_index_stale_generation_never_publishes_old_vectors(monkeypatch):
    task = create("task", "Pest visit")
    calls = []
    original = indexing.embeddings

    def changed(owner, texts, *args):
        if not calls:
            calls.append(1)
            run(
                "record.update",
                dict(
                    record_id=task["id"],
                    expected_revision=task["revision"],
                    schema_revision=1,
                    title="Software visit",
                ),
            )
        return original(owner, texts, *args)

    monkeypatch.setattr(indexing, "embeddings", changed)
    indexed()
    with session_scope() as db:
        state = db.get(SearchIndexState, OWNER)
        assert state.indexed_generation != state.generation and state.status == "queued"
        next_job = state.job_id
    indexing.index_workspace(next_job)
    with session_scope() as db:
        assert db.get(SearchIndexState, OWNER).status == "ready"
        assert (
            "Software visit"
            in db.scalar(
                select(SearchDocument).where(SearchDocument.target_key == "record:" + task["id"])
            ).content
        )


def test_preferences_pause_learning_and_expired_conversation_is_unknown():
    from jarvis.search_models import SearchPreference

    result, abc, task = selected()
    with session_scope() as db:
        db.add(SearchPreference(owner_id=OWNER, account_id=OWNER, learning=False))
        db.flush()
        learning.event(
            db, OWNER, OWNER, SearchEvent(search_id=result["search_id"], kind="used", record_id=task["id"])
        )
        assert not list(db.scalars(select(SearchAlias)))
        row = db.get(SearchSession, result["search_id"])
        assert row.outcome == "unknown"


def test_inherited_fields_explicit_clear_and_option_identity():
    d = definition()
    for t in d["types"]:
        if t["id"] in ("task", "project"):
            t["fields"].append(
                dict(
                    id="client_ref",
                    name="Customer",
                    description="The client buying this work",
                    kind="relation",
                    target_types=["client"],
                    inherit=True,
                )
            )
        if t["id"] == "task":
            t["fields"].append(
                dict(
                    id="sector",
                    name="Sector",
                    description="Industry classification",
                    kind="select",
                    options=[dict(id="pest", name="Pest control"), dict(id="software", name="Software")],
                )
            )
    apply(propose(d))
    abc = create(title="ABC")
    project = create("project", "Intelligence", values={"client_ref": abc["id"]})
    inherited = create("task", "Build central docs", parent_id=project["id"], values={"sector": "pest"})
    cleared = create("task", "Explicitly unassigned", parent_id=project["id"], values={"client_ref": None})
    found = search("customer", resolved=["record:" + abc["id"]], capability="work")
    assert inherited["id"] in ids(found["structured"]) and cleared["id"] not in ids(found["structured"])
    option = search("pest control", resolved=["option:task:sector:pest"], strict=True)
    assert ids(option["structured"]) == {inherited["id"]}
    filtered = search("central", values={"client_ref": abc["id"]})
    assert inherited["id"] in ids(filtered["possible"]) and cleared["id"] not in ids(filtered["possible"])


def test_shared_membership_revoked_during_embedding_never_returns_data(monkeypatch):
    from jarvis.models import SharedWorkspace, WorkspaceMember, UserAccount
    from jarvis.domain import DomainError

    indexed_task = create("task", "Pest visit")
    indexed()
    with session_scope() as db:
        db.add(UserAccount(id="member", name="Test member"))
        db.flush()
        db.add(SharedWorkspace(id=OWNER, name="Shared search", creator_id="member", kind="space"))
        db.flush()
        db.add(WorkspaceMember(workspace_id=OWNER, account_id="member", role="viewer", active=True))

    def revoked(*_):
        with session_scope() as db:
            db.get(WorkspaceMember, (OWNER, "member")).active = False
        return [[1.0, 0.0, 0.0]]

    monkeypatch.setattr(service, "embeddings", revoked)
    with pytest.raises(DomainError, match="Workspace access ended"):
        service.search(OWNER, "member", dict(query="pest"))


def test_api_and_mcp_enforce_scopes_and_bots_never_teach(client):
    from test_external_agents import key
    from fastapi.testclient import TestClient
    from jarvis.api import app
    from jarvis import bot_access, external_mcp

    task = create("task", "Rareterm synthetic")
    denied, headers = key(client, ["tasks:read"])
    assert (
        TestClient(app)
        .post("/api/v1/external/search", headers=headers, json={"query": "Rareterm"})
        .status_code
        == 403
    )
    credential, headers = key(client, ["records:read"])
    found = TestClient(app).post("/api/v1/external/search", headers=headers, json={"query": "Rareterm"})
    assert found.status_code == 200, found.text
    assert found.json()["search_id"] is None and task["id"] in ids(found.json()["possible"])
    with bot_access.bind(credential["id"]):
        found = external_mcp.dispatch("record_search", {"query": "Rareterm"})
        assert found["search_id"] is None and task["id"] in ids(found["possible"])
    with session_scope() as db:
        assert not list(db.scalars(select(SearchSession))) and not list(db.scalars(select(SearchAlias)))
    ordinary = client.post("/api/v1/search/records", json={"query": "Rareterm"})
    assert ordinary.status_code == 200 and ordinary.json()["search_id"]


def test_weekly_search_rule_requires_explicit_answer_and_forget_clears_pending(monkeypatch):
    from jarvis import routing
    from jarvis.structure_models import RoutingReview

    result, abc, task = selected()
    with session_scope() as db:
        learning.event(
            db, OWNER, OWNER, SearchEvent(search_id=result["search_id"], kind="used", record_id=task["id"])
        )
        # Human-created fixture assignments are not part of this search-only review.
        for e in db.scalars(select(RoutingObservation)):
            e.suppressed = True
        review = RoutingReview(owner_id=OWNER, period="search-only-test")
        db.add(review)
        db.flush()
        job = Job(owner_id=OWNER, kind="review_routing", payload={"review_id": review.id})
        db.add(job)
        db.flush()
        jid = job.id
        rid = review.id
    monkeypatch.setattr(
        routing, "infer", lambda *a: pytest.fail("Search-only evidence must not train the extractor")
    )
    routing.process(jid)
    with session_scope() as db:
        review = db.get(RoutingReview, rid)
        assert review.status == "pending" and review.questions
        q = review.questions[0]
        revision = review.revision
        assert db.get(RoutingPattern, q["id"]).status == "candidate"
    run(
        "routing.answer",
        dict(review_id=rid, question_id=q["id"], expected_revision=revision, action="accept"),
    )
    routed = create("task", "pest control company renewal")
    assert routed["parent_id"] == abc["id"]
    with session_scope() as db:
        alias = db.scalar(select(SearchAlias))
        learning.change_alias(
            db, OWNER, OWNER, alias.id, AliasChange(expected_revision=alias.revision, action="forget")
        )
        # Explicitly approved routing remains an independent user instruction.
        assert db.get(RoutingPattern, q["id"]).status == "active"
        assert not db.get(RoutingReview, rid).questions
