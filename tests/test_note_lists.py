from uuid import uuid4

import pytest
from jarvis.db import session_scope
from jarvis.models import Job, Memory, Note, NoteEntrySource, NoteList, NoteOrganization, RoutingObservation
from jarvis.note_list_schema import Classification, OrganizationResult
from jarvis.note_lists import process
from jarvis.tools import call_tool
from sqlalchemy import func, select


def cmd(client, tool, args, identity=None):
    return client.post(
        "/api/v1/commands", json={"command_id": identity or str(uuid4()), "tool": tool, "arguments": args}
    )


def setup(client):
    r = cmd(client, "notelist.setup", {})
    assert r.status_code == 200, r.text
    return r.json()["data"]["items"]


def note(client, title="Recommendations", content="Sam recommended Arrival and Dune—watch these.", **extra):
    r = cmd(client, "note.create", {"title": title, "content": content, **extra})
    assert r.status_code == 200, r.text
    return r.json()["data"]


def job_for(nid):
    with session_scope() as db:
        return db.scalar(
            select(Job.id)
            .where(Job.kind == "organize_note", Job.payload["note_id"].as_string() == nid)
            .order_by(Job.created_at.desc())
            .limit(1)
        )


def response(movies, source, titles=("Arrival", "Dune"), existing=None):
    return OrganizationResult(
        classifications=[],
        entries=[
            {
                "title": t,
                "evidence": source,
                "list_ids": [movies["id"]],
                "confidence": 0.99,
                "save_intent": True,
                "existing_note_id": existing,
            }
            for t in titles
        ],
    )


def test_lists_are_editable_filters_and_safe_permissions(client):
    lists = setup(client)
    assert len(setup(client)) == 5
    movies = next(l for l in lists if l["name"] == "Movies")
    n = note(client, title="Arrival", content="Watch this", tags=["movies"])
    note(client, title="Novel", content="Read this", tags=["books"])
    found = client.get("/api/v1/notes", params={"list_id": movies["id"]}).json()
    assert [v["id"] for v in found["items"]] == [n["id"]]
    assert client.get("/api/v1/note-lists/" + movies["id"] + "/notes").json()["items"][0]["id"] == n["id"]
    changed = cmd(
        client,
        "notelist.save",
        {
            **{k: movies[k] for k in ("id", "description", "filters", "automatic", "extract_entries")},
            "name": "Cinema",
            "expected_revision": 1,
        },
    )
    assert changed.status_code == 200, changed.text
    assert client.get("/api/v1/notes", params={"list_id": movies["id"]}).json()["items"][0]["id"] == n["id"]
    assert (
        cmd(
            client, "notelist.save", {"name": "Empty", "description": "", "filters": {"tags": ["x"]}}
        ).status_code
        == 400
    )
    assert (
        cmd(client, "notelist.save", {"name": "Everything", "description": "x", "filters": {}}).status_code
        == 400
    )
    with session_scope() as db:
        other = NoteList(
            owner_id="elsewhere", name="Private", description="Secret", filters={"tags": ["movies"]}
        )
        db.add(other)
        db.flush()
        other_id = other.id
    assert client.get("/api/v1/notes", params={"list_id": other_id}).status_code == 404
    assert (
        cmd(
            client, "note.file", {"note_id": n["id"], "expected_revision": 1, "list_id": other_id}
        ).status_code
        == 404
    )


def test_extraction_keeps_source_and_retries_do_not_duplicate(client, monkeypatch):
    movies = setup(client)[0]
    n = note(client)
    monkeypatch.setattr("jarvis.routing.infer", lambda *a: response(movies, n["content"]))
    process(job_for(n["id"]))
    details = client.get("/api/v1/notes/" + n["id"]).json()
    assert details["content"] == n["content"]
    assert {e["title"] for e in details["saved_entries"]} == {"Arrival", "Dune"}
    items = client.get("/api/v1/notes", params={"list_id": movies["id"]}).json()["items"]
    assert {i["title"] for i in items} == {"Arrival", "Dune"}
    entry = client.get("/api/v1/notes/" + items[0]["id"]).json()
    assert entry["sources"][0]["evidence"] == n["content"]
    assert entry["organization"]["generated"]
    cmd(client, "note.organize", {"note_id": n["id"], "expected_revision": details["revision"]})
    process(job_for(n["id"]))
    with session_scope() as db:
        assert db.scalar(select(func.count()).select_from(Note)) == 3
        assert db.scalar(select(func.count()).select_from(NoteEntrySource)) == 2
        assert db.scalar(select(func.count()).select_from(Memory)) == 0
        assert db.scalar(select(func.count()).select_from(RoutingObservation)) == 0
    exported = client.get("/api/v1/export").json()
    assert len(exported["note_entry_sources"]) == 2 and len(exported["note_lists"]) == 5


def test_manual_clear_survives_model_and_counts_as_human_evidence(client, monkeypatch):
    movies = setup(client)[0]
    n = note(client, title="Arrival", content="A film I want to watch.")
    answer = OrganizationResult(
        classifications=[{"list_id": movies["id"], "evidence": n["content"], "confidence": 0.99}], entries=[]
    )
    monkeypatch.setattr("jarvis.routing.infer", lambda *a: answer)
    process(job_for(n["id"]))
    n = client.get("/api/v1/notes/" + n["id"]).json()
    assert n["tags"] == ["movies"]
    cleared = cmd(
        client, "note.update", {"note_id": n["id"], "expected_revision": n["revision"], "tags": []}
    ).json()["data"]
    process(job_for(n["id"]))
    current = client.get("/api/v1/notes/" + n["id"]).json()
    assert current["tags"] == [] and current["organization"]["tags_locked"]
    with session_scope() as db:
        obs = list(db.scalars(select(RoutingObservation)))
        assert len(obs) == 1 and obs[0].evidence["assignment"]["note_tags"] == []
    assert cleared["content"] == n["content"]


def test_invalid_quotes_passing_mentions_and_low_confidence_are_not_extracted(client, monkeypatch):
    movies = setup(client)[0]
    n = note(client, content="Arrival was mentioned at work.")
    monkeypatch.setattr(
        "jarvis.routing.infer",
        lambda *a: OrganizationResult(
            classifications=[],
            entries=[
                {
                    "title": "Arrival",
                    "evidence": n["content"],
                    "list_ids": [movies["id"]],
                    "confidence": 0.99,
                    "save_intent": False,
                    "existing_note_id": None,
                },
                {
                    "title": "Dune",
                    "evidence": "not in note",
                    "list_ids": [movies["id"]],
                    "confidence": 0.99,
                    "save_intent": True,
                    "existing_note_id": None,
                },
                {
                    "title": "Alien",
                    "evidence": n["content"],
                    "list_ids": [movies["id"]],
                    "confidence": 0.4,
                    "save_intent": True,
                    "existing_note_id": None,
                },
            ],
        ),
    )
    process(job_for(n["id"]))
    assert client.get("/api/v1/notes/" + n["id"]).json()["saved_entries"] == []


@pytest.mark.parametrize("change", ["note", "list"])
def test_results_are_discarded_when_source_or_definition_changes(client, monkeypatch, change):
    movies = setup(client)[0]
    n = note(client)
    jid = job_for(n["id"])

    def delayed(*args):
        if change == "note":
            assert (
                cmd(
                    client,
                    "note.update",
                    {"note_id": n["id"], "expected_revision": 1, "content": "Forget those recommendations."},
                ).status_code
                == 200
            )
        else:
            assert (
                cmd(
                    client,
                    "notelist.save",
                    {
                        "id": movies["id"],
                        "expected_revision": 1,
                        "name": "Movies",
                        "description": "Different criteria",
                        "filters": movies["filters"],
                    },
                ).status_code
                == 200
            )
        return response(movies, n["content"])

    monkeypatch.setattr("jarvis.routing.infer", delayed)
    process(jid)
    with session_scope() as db:
        assert db.get(Job, jid).status == "cancelled"
        assert db.get(NoteOrganization, n["id"]).status == ("queued" if change == "note" else "stale")
        assert db.scalar(select(func.count()).select_from(NoteEntrySource)) == 0


def test_disabling_extraction_keeps_classification_and_one_item_multiple_lists(client, monkeypatch):
    lists = setup(client)
    movies = lists[0]
    date = cmd(
        client,
        "notelist.save",
        {"name": "Date night", "description": "Things for date night", "filters": {"tags": ["date-night"]}},
    ).json()["data"]
    n = note(client)
    answer = response(movies, n["content"], ("Arrival",))
    answer.entries[0].list_ids.append(date["id"])
    monkeypatch.setattr("jarvis.routing.infer", lambda *a: answer)
    process(job_for(n["id"]))
    left = client.get("/api/v1/notes", params={"list_id": movies["id"]}).json()["items"]
    right = client.get("/api/v1/notes", params={"list_id": date["id"]}).json()["items"]
    assert len(left) == len(right) == 1 and left[0]["id"] == right[0]["id"]


def test_ambiguous_existing_titles_never_merge_or_duplicate(client, monkeypatch):
    movies = setup(client)[0]
    a = note(client, title="Dune", content="1984 film", tags=["movies"])
    note(client, title="Dune", content="2021 film", tags=["movies"])
    n = note(client, content="Save Dune for later.")
    monkeypatch.setattr(
        "jarvis.routing.infer", lambda *args: response(movies, n["content"], ("Dune",), a["id"])
    )
    process(job_for(n["id"]))
    assert client.get("/api/v1/notes/" + n["id"]).json()["saved_entries"] == []
    with session_scope() as db:
        assert db.scalar(select(func.count()).select_from(Note)) == 3


def test_new_lists_accept_custom_fields(client):
    setup(client)
    schema = client.get("/api/v1/structure").json()
    from copy import deepcopy

    definition = deepcopy({k: schema[k] for k in ("types", "relationships")})
    definition["types"][-1]["fields"] = [
        {
            "id": "topic",
            "name": "Topic",
            "description": "Saved thing category",
            "kind": "select",
            "options": [{"id": "film", "name": "Film"}],
        }
    ]
    preview = cmd(
        client, "structure.preview", {"expected_revision": schema["revision"], "definition": definition}
    ).json()["data"]
    assert (
        cmd(
            client, "structure.apply", {"proposal_id": preview["id"], "expected_revision": schema["revision"]}
        ).status_code
        == 200
    )
    lst = cmd(
        client,
        "notelist.save",
        {
            "name": "Films",
            "description": "Film notes",
            "filters": {"type_id": "note", "values": {"topic": "film"}},
        },
    )
    assert lst.status_code == 200, lst.text
    n = note(client, title="Arrival")
    filed = cmd(
        client,
        "note.file",
        {"note_id": n["id"], "expected_revision": n["revision"], "list_id": lst.json()["data"]["id"]},
    )
    assert filed.status_code == 200, filed.text
    assert (
        client.get("/api/v1/notes", params={"list_id": lst.json()["data"]["id"]}).json()["items"][0]["id"]
        == n["id"]
    )


@pytest.mark.asyncio
async def test_tools_discover_and_browse_lists_without_mutation(client):
    from jarvis.tools import registry

    lists = setup(client)
    names = {r["name"] for r in registry()}
    assert {"note_lists", "note_list_items", "notelist_save", "note_file", "note_organize"} <= names
    data = await call_tool("davin", "test", 0, "note_lists", {})
    assert len(data["items"]) == 5
    result = await call_tool("davin", "test", 1, "note_list_items", {"list_id": lists[0]["id"]})
    assert result["items"] == []


def test_classification_can_be_disabled_and_failures_are_retryable(client, monkeypatch):
    movies = setup(client)[0]
    n = note(client)

    def offline(*args):
        raise TimeoutError("provider unavailable")

    monkeypatch.setattr("jarvis.routing.infer", offline)
    jid = job_for(n["id"])
    for _ in range(4):
        with pytest.raises(TimeoutError):
            process(jid)
    process(jid)
    with session_scope() as db:
        assert db.get(Job, jid).status == "failed"
        assert db.get(NoteOrganization, n["id"]).status == "failed"
    assert client.get("/api/v1/notes/" + n["id"]).json()["content"] == n["content"]
    monkeypatch.setattr("jarvis.routing.infer", lambda *a: response(movies, n["content"]))
    assert cmd(client, "note.organize", {"note_id": n["id"], "expected_revision": 1}).status_code == 200
    process(job_for(n["id"]))
    assert len(client.get("/api/v1/notes/" + n["id"]).json()["saved_entries"]) == 2


def test_shared_viewer_cannot_configure_or_file_and_revoked_creator_cannot_extract(client, monkeypatch):
    from jarvis.models import WorkspaceMember
    from test_accounts import client_for, shared

    guest = client_for("guest")
    workspace = shared(client, guest)
    movies = setup(client)[0]
    n = note(guest)
    jid = job_for(n["id"])
    with session_scope() as db:
        db.get(WorkspaceMember, (workspace["id"], "guest")).role = "viewer"
    assert cmd(guest, "notelist.setup", {}).status_code == 403
    assert (
        cmd(
            guest, "note.file", {"note_id": n["id"], "expected_revision": 1, "list_id": movies["id"]}
        ).status_code
        == 403
    )
    assert guest.get("/api/v1/note-lists").status_code == 200
    monkeypatch.setattr(
        "jarvis.routing.infer", lambda *a: pytest.fail("Revoked work must not invoke a model")
    )
    process(jid)
    with session_scope() as db:
        assert db.get(Job, jid).status == "cancelled"
        assert db.get(NoteOrganization, n["id"]).status == "cancelled"
        assert db.scalar(select(func.count()).select_from(NoteEntrySource)) == 0


def test_optional_candidates_do_not_change_membership(client, monkeypatch):
    movies = setup(client)[0]
    member = note(client, title="Arrival", content="Watch Arrival", tags=["movies"])
    missed = note(client, title="Dune", content="Watch Dune", tags=["wrong-list"])
    monkeypatch.setattr(
        "jarvis.notes.search_notes", lambda *a, **k: {"items": [member, missed], "mode": "hybrid"}
    )
    result = client.get("/api/v1/note-lists/" + movies["id"] + "/suggestions")
    assert result.status_code == 200, result.text
    assert [i["id"] for i in result.json()["items"]] == [missed["id"]]
    assert client.get("/api/v1/notes/" + missed["id"]).json()["tags"] == ["wrong-list"]


def test_archiving_list_does_not_archive_notes_and_invalid_filter_can_be_removed(client):
    movies = setup(client)[0]
    n = note(client, title="Arrival", tags=["movies"])
    with session_scope() as db:
        row = db.get(NoteList, movies["id"])
        row.filters = {"type_id": "deleted-type", "tags": [], "values": {}}
    assert client.get("/api/v1/note-lists").json()["items"][0]["error"]
    r = cmd(
        client,
        "notelist.save",
        {
            "id": movies["id"],
            "expected_revision": 1,
            "name": movies["name"],
            "description": movies["description"],
            "filters": {"type_id": "deleted-type"},
            "archived": True,
        },
    )
    assert r.status_code == 200, r.text
    assert not client.get("/api/v1/notes/" + n["id"]).json()["archived"]


def test_extracted_evidence_is_searchable_without_copying_authored_content(client, monkeypatch):
    from jarvis.search_index import snapshot

    movies = setup(client)[0]
    n = note(client)
    monkeypatch.setattr("jarvis.routing.infer", lambda *a: response(movies, n["content"], ("Arrival",)))
    process(job_for(n["id"]))
    child = client.get("/api/v1/notes", params={"list_id": movies["id"]}).json()["items"][0]
    assert client.get("/api/v1/notes/" + child["id"]).json()["content"] == ""
    with session_scope() as db:
        documents, records = snapshot(db, "davin")
        record = next(r for r in records.values() if r["record"]["note_id"] == child["id"])
        assert "Sam recommended" in documents["record:" + record["record"]["id"]]["text"]


def test_weekly_note_rules_need_human_review_and_preserve_corrections(client, monkeypatch):
    from jarvis import routing
    from jarvis.models import RoutingPattern, RoutingReview

    setup(client)
    for i in range(12):
        note(
            client, title=f"Space opera recommendation {i}", content="An authored film note.", tags=["movies"]
        )
    review = cmd(client, "routing.run", {}).json()["data"]
    with session_scope() as db:
        jid = review["job_id"]
        review_id = db.get(Job, jid).payload["review_id"]
    monkeypatch.setattr(
        routing,
        "infer",
        lambda *a: routing.Candidates(
            candidates=[
                {
                    "phrase": "Space opera",
                    "type_id": "note",
                    "parent_id": None,
                    "values": [],
                    "note_tags": ["movies"],
                    "reason": "Independent manual classifications",
                }
            ]
        ),
    )
    routing.process(jid)
    with session_scope() as db:
        rule = db.scalar(select(RoutingPattern))
        assert rule.status == "candidate"
        rid = rule.id
        pending = db.get(RoutingReview, review_id)
        revision = pending.revision
    assert (
        cmd(
            client,
            "routing.answer",
            {"review_id": review_id, "expected_revision": revision, "question_id": rid, "action": "accept"},
        ).status_code
        == 200
    )
    new = note(client, title="Space opera archive", content="My reference.")
    monkeypatch.setattr(routing, "infer", lambda *a: OrganizationResult(classifications=[], entries=[]))
    process(job_for(new["id"]))
    current = client.get("/api/v1/notes/" + new["id"]).json()
    assert current["tags"] == ["movies"]
    assert (
        cmd(
            client,
            "note.update",
            {"note_id": new["id"], "expected_revision": current["revision"], "tags": ["books"]},
        ).status_code
        == 200
    )
    with session_scope() as db:
        assert db.get(RoutingPattern, rid).status == "conflict"


def test_extraction_switch_keeps_whole_note_classification(client, monkeypatch):
    movies = setup(client)[0]
    changed = cmd(
        client,
        "notelist.save",
        {
            **{k: movies[k] for k in ("id", "name", "description", "filters", "automatic")},
            "expected_revision": movies["revision"],
            "extract_entries": False,
        },
    )
    assert changed.status_code == 200
    source = note(client)
    result = response(movies, source["content"])
    result.classifications = [
        Classification(list_id=movies["id"], evidence=source["content"], confidence=0.99)
    ]
    monkeypatch.setattr("jarvis.routing.infer", lambda *a: result)
    process(job_for(source["id"]))
    found = client.get("/api/v1/notes", params={"list_id": movies["id"]}).json()["items"]
    assert [n["id"] for n in found] == [source["id"]]
    with session_scope() as db:
        assert db.scalar(select(func.count()).select_from(NoteEntrySource)) == 0
