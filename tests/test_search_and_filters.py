"""Roadmap bugs of 2026-10-28: slow top-bar title search (b14920b4), Tasks search missing a
mid-title word (ce2e0e8d), and the Main home filter missing a project's tasks (b8d2e28e)."""

import pytest
from sqlalchemy import event, select
from jarvis.config import get_settings
from jarvis.db import engine, session_scope
from jarvis import search_index as indexing, search_service as service, structure
from jarvis.structure_models import StructureRecord
from test_structure import create, run, OWNER

PARITY = "Parity: ignore time-varying parts of a session's resume when comparing"


def records(**params):
    with session_scope() as db:
        return structure.records(db, OWNER, limit=200, **params)["items"]


def search(query, **args):
    return service.search(OWNER, OWNER, dict(query=query, **args))


def found(result):
    return [m["record"]["title"] for m in result["structured"]["items"] + result["possible"]["items"]]


@pytest.fixture
def semantic(monkeypatch):
    """Semantic search on, with an embedding provider that relates nothing to anything."""
    monkeypatch.setattr(get_settings(), "semantic_search_enabled", True)
    calls = []

    def vectors(owner, texts, *args):
        calls.append(list(texts))
        return [[0.0, 1.0, 0.0] if len(texts) == 1 else [1.0, 0.0, 0.0] for _ in texts]

    monkeypatch.setattr(indexing, "embeddings", vectors)
    monkeypatch.setattr(service, "embeddings", vectors)
    service.query_vector.cache_clear()
    yield calls
    service.query_vector.cache_clear()


def roadmap():
    """A legacy project imported into the registry, as "Eridani roadmap" was."""
    project = run("project.create", {"name": "Eridani roadmap"})
    records()  # the first registry read imports legacy rows under their own ids
    return project


def index():
    with session_scope() as db:
        job = indexing.queue_index(db, OWNER, force=True)
    indexing.index_workspace(job)


# ---- ce2e0e8d: a word in the middle of a title ------------------------------------------------


@pytest.mark.parametrize("enabled", [False, True])
def test_search_finds_a_word_in_the_middle_of_a_title(enabled, semantic, monkeypatch):
    monkeypatch.setattr(get_settings(), "semantic_search_enabled", enabled)
    project = roadmap()
    run("task.create", {"title": PARITY, "project_id": project["id"]})
    run("task.create", {"title": "Ship the release", "notes": "Nothing ignorable here"})
    run("task.create", {"title": "Plan the week"})
    if enabled:
        index()
    for query in ("ignore", "IGNORE", "time-varying", "gnore", "resume when"):
        assert PARITY in found(search(query, capability="work", limit=100)), query
    # Substrings of notes count too; type names and field labels in the index text do not.
    assert "Ship the release" in found(search("ignorable", capability="work"))
    assert found(search("ignore", capability="work")).index(PARITY) == 0
    assert "Plan the week" not in found(search("task", capability="work"))
    assert "Plan the week" not in found(search("abel", capability="work"))  # "Project label: "


def test_tasks_list_returns_every_task_for_the_browser_filter():
    project = roadmap()
    for i in range(30):
        run("task.create", {"title": f"Filler {i}", "project_id": project["id"]})
    run("task.create", {"title": PARITY, "project_id": project["id"]})
    seen, offset = [], 0
    while offset is not None:
        with session_scope() as db:
            page = structure.records(db, OWNER, limit=7, offset=offset)
        seen += page["items"]
        offset = page["next_offset"]
    assert PARITY in [r["title"] for r in seen]
    assert len({r["id"] for r in seen}) == len(seen) == 32


# ---- b8d2e28e: Main home = a project -----------------------------------------------------------


def test_main_home_filter_includes_every_task_of_a_project(semantic):
    project = roadmap()
    direct = run("task.create", {"title": "Roadmap: direct task", "project_id": project["id"]})
    parent = run("task.create", {"title": "Roadmap: parent task", "project_id": project["id"]})
    child = run(
        "task.create",
        {"title": "Roadmap: child step", "project_id": project["id"], "parent_task_id": parent["id"]},
    )
    drifted = run("task.create", {"title": "Roadmap: drifted record", "project_id": project["id"]})
    filed = create("task", "Roadmap: filed under the project record", parent_id=project["id"])
    run("task.create", {"title": "Roadmap: elsewhere"})
    with session_scope() as db:
        # A record whose saved home drifted from its task's project (a home conflict).
        row = db.scalar(select(StructureRecord).where(StructureRecord.task_id == drifted["id"]))
        row.parent_id = None
    by_title = {r["title"]: r for r in records()}
    # The browser filter keeps a row when the home is its parent, an ancestor or its project.
    def in_home(r):
        return (
            r["parent_id"] == project["id"]
            or any(h["id"] == project["id"] for h in r["home"])
            or r.get("project_id") == project["id"]
        )

    for title in (
        direct["title"], parent["title"], child["title"], drifted["title"], filed["title"]
    ):
        assert in_home(by_title[title]), title
    assert not in_home(by_title["Roadmap: elsewhere"])
    assert by_title[child["title"]]["parent_id"] == by_title[parent["title"]]["id"]
    # The same rule on the server (Eri's search_records and the bot search route).
    index()
    result = search("roadmap", capability="work", home_id=project["id"], limit=100)
    assert set(found(result)) == {
        direct["title"], parent["title"], child["title"], drifted["title"], filed["title"]
    }


# ---- b14920b4: fast title lookup ---------------------------------------------------------------


def test_lexical_only_search_skips_the_provider_and_reports_timing(semantic):
    roadmap()
    run("task.create", {"title": PARITY})
    run("task.create", {"title": "Renew insurance"})
    index()
    semantic.clear()
    fast = search("parity", capability="work", lexical_only=True)
    assert semantic == [], "the lexical pass must not call the embedding provider"
    assert fast["mode"] == "lexical" and found(fast) == [PARITY]
    assert fast["search_id"] is None, "the fast pass must not open a second learning session"
    assert {"snapshot_ms", "rank_ms", "total_ms"} <= set(fast["timing"])
    assert "embedding_ms" not in fast["timing"] and "refresh_ms" not in fast["timing"]
    full = search("parity", capability="work")
    assert full["mode"] == "hybrid" and PARITY in found(full)
    assert {"snapshot_ms", "embedding_ms", "refresh_ms", "rank_ms", "total_ms"} <= set(full["timing"])
    assert all(isinstance(v, float) and v >= 0 for v in full["timing"].values())
    # A repeated short query reuses its vector instead of calling the provider again.
    search("parity", capability="work")
    assert semantic == [["parity"]]


def test_search_route_returns_timing(client, semantic):
    roadmap()
    run("task.create", {"title": PARITY})
    response = client.post("/api/v1/search/records", json={"query": "ignore", "lexical_only": True})
    assert response.status_code == 200
    body = response.json()
    assert body["timing"]["total_ms"] >= 0 and PARITY in found(body)


def test_record_list_reads_in_bulk_and_matches_single_record_reads():
    client_row = create("client", "ABC")
    project = create("project", "Website", parent_id=client_row["id"])
    tasks = [create("task", f"Step {i}", parent_id=project["id"]) for i in range(12)]
    note = create("note", "Kickoff notes", body="Agenda", parent_id=project["id"])
    a = tasks[0]
    with session_scope() as db:
        schema_revision = structure.schema_data(db, OWNER)["revision"]
        current = structure.data(db, db.get(StructureRecord, a["id"]))
    run(
        "record.link",
        dict(
            source_id=a["id"],
            target_id=tasks[1]["id"],
            relationship_id="blocks",
            expected_revision=current["revision"],
            schema_revision=schema_revision,
        ),
    )
    quick = run("quicklist.create", {"title": "Packing", "items": [{"title": "Socks"}, {"title": "Charger"}]})
    item = quick["items"][0]
    run(
        "quicklist.item",
        {
            "list_id": quick["id"],
            "expected_revision": quick["revision"],
            "operation": "edit",
            "item_id": item["id"],
            "expected_item_revision": item["revision"],
            "completed": True,
        },
    )
    records()  # settle registry adoption before counting
    statements = []

    def count(conn, cursor, statement, *args):
        statements.append(statement)

    listener = engine()
    event.listen(listener, "before_cursor_execute", count)
    try:
        listed = records()
    finally:
        event.remove(listener, "before_cursor_execute", count)
    # Bounded by the bulk reads, not by the number of rows (18 here).
    assert len(listed) == 18
    assert len(statements) <= 16, "\n".join(x[:160] for x in statements)
    with session_scope() as db:
        for item in listed:
            single = structure.data(db, db.get(StructureRecord, item["id"]))
            for key in ("links",):
                item[key] = sorted(item[key], key=lambda link: link["id"])
                single[key] = sorted(single[key], key=lambda link: link["id"])
            assert item == single, item["title"]
    by_id = {r["id"]: r for r in listed}
    assert by_id[tasks[1]["id"]]["blockers"][0]["id"] == a["id"]
    assert [h["title"] for h in by_id[note["id"]]["home"]] == ["ABC", "Website"]
    packing = next(r for r in listed if r["title"] == "Packing")
    assert (packing["quick_total"], packing["quick_done"]) == (2, 1)
    socks = next(r for r in listed if r["title"] == "Socks")
    assert socks["quick_list_parent_id"] == packing["task_id"]
