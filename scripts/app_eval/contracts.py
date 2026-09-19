"""Executable component oracles. They do not grade natural-language model behavior."""

import json
from uuid import uuid4

CASES = {}


def case(*identities):
    def register(fn):
        for identity in identities:
            CASES[identity] = fn
        return fn

    return register


class Harness:
    def __init__(self, fixture):
        self.refs = fixture["refs"]
        self.owner = fixture["owner"]
        self.trace = []

    def command(self, name, args, *, key=None, owner=None, error=None):
        from jarvis.db import session_scope
        from jarvis.domain import DomainError, execute

        try:
            with session_scope() as db:
                result = execute(db, owner or self.owner, key or str(uuid4()), name, args)["data"]
            self.trace.append({"command": name, "arguments": args, "result": result})
        except DomainError as exc:
            self.trace.append({"command": name, "arguments": args, "error": exc.code})
            if error is None or exc.code not in error:
                raise AssertionError(f"{name}: unexpected {exc.code}; expected {error}") from exc
            return {"error": exc.code}
        assert error is None, f"{name} unexpectedly succeeded; expected {error}"
        return result

    def row(self, model, identity):
        from jarvis import models
        from jarvis.db import session_scope
        from jarvis.domain import serial

        with session_scope() as db:
            return serial(db.get(getattr(models, model), identity))

    def rows(self, model):
        from jarvis import models
        from jarvis.db import session_scope
        from jarvis.domain import serial
        from sqlalchemy import select

        with session_scope() as db:
            from jarvis import structure_models

            cls = getattr(models, model, None) or getattr(structure_models, model)
            return [serial(r) for r in db.scalars(select(cls))]

    def task(self, key="beacon-docs"):
        return self.row("Task", self.refs["tasks"][key]["id"])

    def update(self, row, **changes):
        return self.command(
            "task.update", {"task_id": row["id"], "expected_revision": row["revision"], **changes}
        )

    def note(self):
        return self.command(
            "note.create",
            {
                "title": "Oracle fixture",
                "content": "  First paragraph.\n\nRepeated word.  \n",
                "tags": ["original"],
            },
        )

    def record(self, type_id="task", **kw):
        return self.command(
            "record.create", {"type_id": type_id, "schema_revision": 1, "title": "Oracle record", **kw}
        )

    @staticmethod
    def preserved(before, after, allowed):
        ignored = {"revision", "updated_at", *allowed}
        assert {k: v for k, v in before.items() if k not in ignored} == {
            k: v for k, v in after.items() if k not in ignored
        }


@case(
    "task_capture.01",
    "task_capture.02",
    "task_capture.03",
    "task_capture.06",
    "task_capture.07",
    "task_capture.15",
    "task_capture.17",
    "task_capture.18",
    "task_capture.19",
    "task_capture.20",
    "task_capture.21",
    "task_capture.23",
)
def capture(h, identity):
    n = int(identity.rsplit(".", 1)[1])
    args = {"title": "Buy oat milk"}
    if n == 2:
        args = {"title": "Review Beacon retry policy", "notes": "compare two timeout strategies"}
    if n == 3:
        args = {"title": "Investigate speaker overlap", "status": "backlog"}
    if n == 6:
        args = {"title": "Draft launch email", "planned_date": "2030-01-16"}
    if n == 7:
        args = {"title": "Test Beacon retries", "estimate_minutes": 45}
    if n == 15:
        args = {"title": "Review café résumé — 東京"}
    if n in {17, 18}:
        args = {"title": "x" * (500 if n == 17 else 501)}
    if n == 19:
        args = {"title": ""}
    if n == 23:
        args["status"] = "mystery"
    before = h.rows("Task")
    if n in {18, 19, 23}:
        h.command("task.create", args, error={"INVALID_ARGUMENT"})
        assert h.rows("Task") == before
        return
    key = str(uuid4())
    row = h.command("task.create", args, key=key)
    assert row["title"] == args["title"]
    for field in ("notes", "status", "planned_date", "estimate_minutes"):
        if field in args:
            assert row[field] == args[field]
    assert row["due_date"] is None and row["due_time"] is None
    assert row["completed_at"] is None
    if n == 20:
        assert h.command("task.create", args, key=key) == row
    if n == 21:
        h.command("task.create", {"title": "Different"}, key=key, error={"REVISION_CONFLICT"})
    after = h.rows("Task")
    assert len(after) == len(before) + 1
    assert [r for r in after if r["id"] != row["id"]] == before


@case(
    "task_edit.01",
    "task_edit.04",
    "task_edit.11",
    "task_edit.12",
    "task_edit.13",
    "task_edit.15",
    "task_edit.21",
    "task_edit.23",
    "task_edit.24",
    "task_edit.25",
)
def edit(h, identity):
    n = int(identity.rsplit(".", 1)[1])
    row = h.task()
    changes = {
        1: {"title": "Finish Beacon operations guide"},
        4: {"due_date": "2030-01-18"},
        11: {"estimate_minutes": 90},
        12: {"priority": 3},
        13: {"tags": ["qa", "regression"]},
        15: {"work_type": "documentation"},
        24: {"title": "Title only"},
        25: {"notes": ""},
    }
    if n == 21:
        h.update(row, title="Concurrent title")
        updated = h.task()
        h.command(
            "task.update",
            {"task_id": row["id"], "expected_revision": row["revision"], "title": "Stale"},
            error={"REVISION_CONFLICT"},
        )
        assert h.task() == updated
        return
    if n == 23:
        foreign = h.row("Task", h.refs["private_task"]["id"])
        h.command(
            "task.update",
            {"task_id": foreign["id"], "expected_revision": foreign["revision"], "title": "Intrusion"},
            error={"NOT_FOUND"},
        )
        assert h.row("Task", foreign["id"]) == foreign
        return
    after = h.update(row, **changes[n])
    for k, v in changes[n].items():
        assert after[k] == v
    h.preserved(row, after, set(changes[n]))


@case(
    "task_lifecycle.01",
    "task_lifecycle.02",
    "task_lifecycle.03",
    "task_lifecycle.04",
    "task_lifecycle.05",
    "task_lifecycle.06",
    "task_lifecycle.07",
    "task_lifecycle.08",
    "task_lifecycle.10",
    "task_lifecycle.12",
    "task_lifecycle.25",
)
def lifecycle(h, identity):
    n = int(identity.rsplit(".", 1)[1])
    row = h.task(
        "groceries"
        if n == 1
        else "completed"
        if n in {2, 12}
        else "inventory-map"
        if n == 4
        else "archive"
        if n == 10
        else "beacon-docs"
    )
    if n == 25:
        h.command(
            "task.update",
            {"task_id": row["id"], "expected_revision": row["revision"], "status": "unknown"},
            error={"INVALID_ARGUMENT"},
        )
        assert h.row("Task", row["id"]) == row
        return
    if n == 10:
        after = h.update(row, archived=False)
        assert not after["archived"]
        h.preserved(row, after, {"archived"})
        return
    if n == 12:
        after = h.update(row, notes="Completion retained.")
        assert after["completed_at"] == row["completed_at"]
        h.preserved(row, after, {"notes"})
        return
    status = {
        1: "completed",
        2: "open",
        3: "backlog",
        4: "open",
        5: "in_progress",
        6: "waiting",
        7: "deferred",
        8: "cancelled",
    }[n]
    after = h.update(row, status=status)
    assert after["status"] == status
    assert bool(after["completed_at"]) == (status == "completed")
    h.preserved(row, after, {"status", "completed_at"})


@case(
    "notes.01",
    "notes.02",
    "notes.03",
    "notes.04",
    "notes.05",
    "notes.06",
    "notes.07",
    "notes.08",
    "notes.10",
    "notes.11",
    "notes.18",
    "notes.20",
    "notes.21",
    "notes.22",
    "notes.23",
    "notes.24",
)
def notes(h, identity):
    n = int(identity.rsplit(".", 1)[1])
    if n in {1, 2, 23, 24}:
        content = (
            ""
            if n == 2
            else "x" * (30000 if n == 23 else 30001)
            if n in {23, 24}
            else "  # Title\n\n- item  \n"
        )
        before = h.rows("Note")
        row = h.command(
            "note.create",
            {"title": "Markdown", "content": content},
            error={"INVALID_ARGUMENT"} if n == 24 else None,
        )
        if n == 24:
            assert h.rows("Note") == before
        else:
            assert row["content"] == content
        return
    row = h.note()
    args = {"note_id": row["id"], "expected_revision": row["revision"]}
    if n in {3, 22}:
        key = str(uuid4())
        after = h.command("note.append", {**args, "text": "Next paragraph."}, key=key)
        assert after["content"].startswith(row["content"])
        assert after["content"].count("Next paragraph.") == 1
        if n == 22:
            assert h.command("note.append", {**args, "text": "Next paragraph."}, key=key) == after
    elif n in {4, 5, 6, 7}:
        if n == 6:
            row = h.command("note.update", {**args, "content": "same same"})
            args["expected_revision"] = row["revision"]
        old = "Absent" if n == 5 else "same" if n == 6 else "First paragraph."
        after = h.command(
            "note.replace",
            {**args, "old_text": old, "new_text": "" if n == 7 else "Changed."},
            error={"ANCHOR_MISMATCH"} if n in {5, 6} else None,
        )
        if n in {5, 6}:
            assert h.row("Note", row["id"])["content"] == row["content"]
            return
        assert after["content"] == row["content"].replace(old, "" if n == 7 else "Changed.")
    elif n == 20:
        foreign = h.row("Note", h.refs["private_note"]["id"])
        from jarvis.db import session_scope
        from jarvis.domain import DomainError, owned
        from jarvis.models import Note

        with session_scope() as db:
            try:
                owned(db, Note, foreign["id"], h.owner)
            except DomainError as exc:
                assert exc.code == "NOT_FOUND"
            else:
                raise AssertionError("Foreign note was readable")
        return
    elif n == 21:
        after = h.command("note.update", {**args, "title": "Concurrent title"})
        h.command("note.update", {**args, "title": "Stale"}, error={"REVISION_CONFLICT"})
        assert h.row("Note", row["id"])["title"] == after["title"]
        return
    elif n == 18:
        row = h.command("note.update", {**args, "archived": True})
        after = h.command(
            "note.update", {"note_id": row["id"], "expected_revision": row["revision"], "archived": False}
        )
        assert not after["archived"] and after["content"] == row["content"]
        return
    else:
        change = {8: {"title": "Renamed"}, 10: {"content": ""}, 11: {"tags": ["one", "two"]}}[n]
        after = h.command("note.update", {**args, **change})
        for k, v in change.items():
            assert after[k] == v
    for k in ("id", "task_ids", "project_ids", "goal_ids", "related_note_ids"):
        assert after.get(k) == row.get(k)
    if n != 8:
        assert after["title"] == row["title"]
    if n != 11:
        assert after["tags"] == row["tags"]


@case(
    "custom_records.03",
    "custom_records.04",
    "custom_records.05",
    "custom_records.08",
    "custom_records.09",
    "custom_records.10",
    "custom_records.11",
    "custom_records.12",
    "custom_records.15",
    "custom_records.19",
    "custom_records.20",
    "custom_records.22",
)
def records(h, identity):
    n = int(identity.rsplit(".", 1)[1])
    if n == 3:
        before = len(h.rows("Task"))
        rec = h.record("client", title="Synthetic extra client")
        assert rec["type_id"] == "client" and len(h.rows("Task")) == before
        return
    if n == 4:
        rec = h.record(parent_id=h.refs["projects"]["beacon"]["id"])
        assert rec["values"]["client"] == h.refs["records"]["juniper"]["id"]
        return
    if n == 5:
        rec = h.record(
            parent_id=h.refs["projects"]["beacon"]["id"], values={"client": h.refs["records"]["harbor"]["id"]}
        )
        assert rec["values"]["client"] == h.refs["records"]["harbor"]["id"]
        return
    if n in {8, 9, 10, 11, 12, 22}:
        values = {
            8: {"watch_state": "unknown"},
            9: {"genre": ["science-fiction", "jazz"]},
            10: {"favorite": False},
            11: {"rating": 0},
            12: {"rating": float("inf")},
            22: {"client": h.refs["projects"]["beacon"]["id"]},
        }[n]
        rec = h.command(
            "record.create",
            {"type_id": "note", "title": "Field oracle", "schema_revision": 1, "values": values},
            error={"INVALID_FIELD_VALUE"} if n in {8, 12, 22} else None,
        )
        if n not in {8, 12, 22}:
            for k, v in values.items():
                assert rec["values"][k] == v
        return
    rec = h.record()
    args = {"record_id": rec["id"], "expected_revision": rec["revision"], "schema_revision": 1}
    if n == 15:
        h.command("record.update", {**args, "parent_id": rec["id"]}, error={"HIERARCHY_CYCLE"})
    elif n == 19:
        after = h.command("record.update", {**args, "title": "Fresh"})
        h.command("record.update", {**args, "title": "Stale"}, error={"REVISION_CONFLICT"})
        assert h.row("Task", rec["task_id"])["title"] == after["title"]
    elif n == 20:
        h.command(
            "record.update",
            {**args, "schema_revision": 2, "title": "Wrong schema"},
            error={"SCHEMA_CHANGED", "REVISION_CONFLICT"},
        )


@case(
    "note_lists.01",
    "note_lists.02",
    "note_lists.03",
    "note_lists.04",
    "note_lists.05",
    "note_lists.06",
    "note_lists.15",
)
def lists(h, identity):
    n = int(identity.rsplit(".", 1)[1])
    before = h.rows("Note")
    lists = h.command("notelist.setup", {})["items"]
    if n in {1, 2}:
        assert {r["name"] for r in lists} >= {"Movies", "Books", "Shows", "Restaurants", "Recipes"}
        again = h.command("notelist.setup", {})["items"]
        assert {r["id"] for r in again} == {r["id"] for r in lists}
    elif n in {3, 4, 5}:
        args = {
            "name": "Jazz venues",
            "description": "" if n == 4 else "Small places to hear jazz.",
            "filters": {} if n == 5 else {"tags": ["jazz-venues"]},
        }
        result = h.command(
            "notelist.save",
            args,
            error={"INVALID_ARGUMENT"} if n == 4 else {"INVALID_FILTER"} if n == 5 else None,
        )
        if n == 3:
            for k in ("name", "description"):
                assert result[k] == args[k]
            assert result["filters"]["tags"] == ["jazz-venues"]
    else:
        row = next(r for r in lists if r["name"] == "Movies")
        args = {
            k: row[k]
            for k in ("id", "name", "description", "filters", "automatic", "extract_entries", "archived")
        }
        args["expected_revision"] = row["revision"]
        args.update({"name": "Cinema"} if n == 6 else {"archived": True})
        after = h.command("notelist.save", args)
        assert after["id"] == row["id"]
        assert after["filters"] == row["filters"]
        assert after["name"] == "Cinema" if n == 6 else after["archived"]
    assert h.rows("Note") == before


@case("memory_dream.01", "memory_dream.02", "memory_dream.05", "memory_dream.06", "memory_dream.23")
def dreams(h, identity):
    from jarvis import memory_review
    from jarvis.db import session_scope
    from jarvis.models import MemoryReview
    from sqlalchemy import select

    n = int(identity.rsplit(".", 1)[1])
    before = h.rows("Source")
    with session_scope() as db:
        job = memory_review.queue_review(db, h.owner, manual=True)
        jid = job.id
    memory_review.process(jid)
    if n == 23:
        memory_review.process(jid)
    memories = {r["id"]: r for r in h.rows("Memory")}
    jazz = [memories[h.refs["memories"][k]] for k in ("jazz", "jazz-duplicate")]
    assert sum(not m["suppressed"] for m in jazz) == 1
    assert h.rows("Source") == before
    a, b = (h.refs["memories"][k] for k in ("cat", "cat-spelling"))
    assert not memories[a]["suppressed"] and not memories[b]["suppressed"]
    with session_scope() as db:
        reviews = list(db.scalars(select(MemoryReview).where(MemoryReview.owner_id == h.owner)))
        assert reviews, "Miso/Mizo ambiguity needs an explicit review"
        review = next(r for r in reviews if a in r.memory_ids and b in r.memory_ids)
        rid, rev = review.id, review.revision
    if n in {5, 6}:
        h.command(
            "memory.resolve",
            {"review_id": rid, "expected_revision": rev, "action": "distinct" if n == 5 else "defer"},
        )
        assert not h.row("Memory", a)["suppressed"] and not h.row("Memory", b)["suppressed"]


def snapshot():
    from fastapi.encoders import jsonable_encoder
    from jarvis.db import engine
    from jarvis.models import Base
    from sqlalchemy import select

    with engine().connect() as db:
        return {
            t.name: jsonable_encoder([dict(r) for r in db.execute(select(t)).mappings()])
            for t in Base.metadata.sorted_tables
        }


def private_state(state):
    """Compare every row directly owned by the inaccessible persona."""
    return {
        name: sorted(
            (r for r in rows if r.get("owner_id") == "eval-jules"),
            key=lambda r: json.dumps(r, sort_keys=True),
        )
        for name, rows in state.items()
    }


@case(
    "routing_rules.01",
    "routing_rules.09",
    "routing_rules.10",
    "routing_rules.11",
    "routing_rules.12",
    "routing_rules.13",
    "routing_rules.18",
    "routing_rules.19",
    "routing_rules.20",
)
def rules(h, identity):
    n = int(identity.rsplit(".", 1)[1])
    before = h.rows("Memory")
    parent = h.refs["projects"]["beacon"]["id"]
    args = {
        "phrase": "Beacon dispatch",
        "type_id": "task",
        "parent_id": parent,
        "reason": "Explicit Rowan instruction",
    }
    if n in {18, 19, 20}:
        args["values"] = {18: {"priority": 3}, 19: {"due_date": "2030-01-18"}, 20: {"permission": "owner"}}[n]
        h.command("routing.create", args, error={"INVALID_RULE"})
        assert h.rows("Memory") == before
        return
    rule = h.command("routing.create", args)
    if n in {9, 10, 11}:
        changed = h.command(
            "routing.change",
            {
                "pattern_id": rule["id"],
                "expected_revision": rule["revision"],
                "action": "forget" if n == 11 else "pause",
            },
        )
        if n == 10:
            h.command(
                "routing.change",
                {"pattern_id": rule["id"], "expected_revision": changed["revision"], "action": "activate"},
            )
    if n == 13:
        h.command("routing.create", {**args, "parent_id": h.refs["projects"]["lens"]["id"]})
    kwargs = {"title": "Beacon dispatch new task"}
    if n == 12:
        kwargs["parent_id"] = h.refs["records"]["personal"]["id"]
    row = h.record(**kwargs)
    expected = h.refs["records"]["personal"]["id"] if n == 12 else None if n in {9, 11, 13} else parent
    assert row["parent_id"] == expected
    assert h.rows("Memory") == before


@case("routing_dream.01", "routing_dream.04", "routing_dream.14", "routing_dream.23", "routing_dream.24")
def rule_dream(h, identity):
    from unittest.mock import patch

    from jarvis import routing
    from jarvis.db import session_scope
    from jarvis.structure_models import RoutingPattern, RoutingReview, StructureRecord
    from sqlalchemy import select

    n = int(identity.rsplit(".", 1)[1])
    parent = h.refs["projects"]["beacon"]["id"]
    # Distinct human labeled records, with a distinctive phrase not present in the baseline.
    training = []
    for i in range(16):
        row = h.command(
            "record.create",
            {
                "type_id": "task",
                "title": f"Beacon sentinel review {i}",
                "parent_id": parent,
                "schema_revision": 1,
            },
            key=("backend:" if n == 4 else "") + str(uuid4()),
        )
        training.append(row["id"])
    original_memories = h.rows("Memory")
    captured = []

    def infer(owner, model, prompt, payload):
        captured.append(payload)
        return routing.Candidates(
            candidates=[
                routing.Candidate(
                    phrase="Beacon sentinel",
                    type_id="task",
                    parent_id=str(uuid4()) if n == 14 else parent,
                    values=[],
                    reason="Independent human examples",
                )
            ]
        )

    job = h.command("routing.run", {})
    with patch.object(routing, "infer", infer):
        routing.process(job["job_id"])
    with session_scope() as db:
        patterns = list(db.scalars(select(RoutingPattern).where(RoutingPattern.owner_id == h.owner)))
        matching = [p for p in patterns if p.condition.get("phrase") == "beacon sentinel"]
        review = db.get(
            RoutingReview,
            db.get(__import__("jarvis.models", fromlist=["Job"]).Job, job["job_id"]).payload["review_id"],
        )
        if n in {4, 14}:
            assert not matching
            if n == 4:
                assert all(
                    not ({e["id"] for e in payload["examples"]} & set(training)) for payload in captured
                )
        else:
            assert len(matching) == 1 and matching[0].status == "candidate"
            assert matching[0].evidence_ids
            assert all(db.get(StructureRecord, rid).parent_id == parent for rid in training)
            question = next(q for q in review.questions if q.get("id") == matching[0].id)
            answer = {
                "review_id": review.id,
                "expected_revision": review.revision,
                "question_id": question["id"],
                "action": "accept" if n == 23 else "dismiss",
            }
    if n in {23, 24}:
        h.command("routing.answer", answer)
        row = h.record(title="Beacon sentinel follow-up")
        assert row["parent_id"] == (parent if n == 23 else None)
    assert h.rows("Memory") == original_memories


@case("time_deadlines.26")
def non_utc_revert(h, identity):
    """Known app defect: receipt equality compares offset-formatted timestamps."""
    from jarvis.action_history import inverse
    from jarvis.db import engine, session_scope
    from jarvis.models import ActionChange
    from sqlalchemy import event, select

    eng = engine()

    def zone(dbapi_connection, connection_record):
        with dbapi_connection.cursor() as cur:
            cur.execute("SET TIME ZONE 'America/Chicago'")
        dbapi_connection.commit()

    eng.dispose()
    event.listen(eng, "connect", zone)
    try:
        row = h.command("task.create", {"title": "Non-UTC revert probe"})
        with session_scope() as db:
            change = db.scalar(
                select(ActionChange).where(
                    ActionChange.owner_id == h.owner,
                    ActionChange.entity_id == row["id"],
                    ActionChange.entity_kind == "task",
                )
            )
            assert change is not None
            command, reason = inverse(db, change)
            assert command, reason
    finally:
        event.remove(eng, "connect", zone)
        eng.dispose()
