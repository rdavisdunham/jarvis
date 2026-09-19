"""Seed only synthetic records, using production command validation."""

import copy
import json
from datetime import datetime, timedelta
from itertools import count
from unittest.mock import patch
from uuid import NAMESPACE_URL, uuid5

from .environment import ROOT

OWNER = "eval-rowan"
CLOCK = datetime.fromisoformat("2030-01-14T15:00:00+00:00")


def seed():
    from jarvis import models
    from jarvis.db import engine, session_scope
    from jarvis.domain import capture_source, execute
    from jarvis.memory_learning import fingerprint
    from jarvis.models import (
        Base,
        Memory,
        Note,
        OwnerSettings,
        SharedWorkspace,
        Task,
        UserAccount,
        WorkspaceMember,
    )
    from jarvis.structure import ensure
    from jarvis.structure_schema import Definition
    from sqlalchemy import select

    corpus = json.loads((ROOT / "evals/app/corpus/rowan-v1.json").read_text())
    persona = json.loads((ROOT / "evals/app/personas/rowan-v1.json").read_text())
    refs = {"tasks": {}, "notes": {}, "records": {}, "projects": {}, "lists": {}, "memories": {}}
    serial = count()

    def uid():
        return uuid5(NAMESPACE_URL, f"eridani-eval-rowan-v1:{next(serial)}")

    def cmd(tool, args, owner=OWNER):
        with session_scope() as db:
            return execute(db, owner, str(uid()), tool, args)["data"]

    with patch.object(models, "uuid4", side_effect=uid):
        with session_scope() as db:
            for key, name in [
                (OWNER, "Rowan Chen"),
                ("eval-jules", "Jules"),
                ("eval-sam", "Sam"),
                ("eval-lee", "Lee"),
            ]:
                db.add(UserAccount(id=key, name=name))
            db.flush()
            shared_id = str(uid())
            db.add(SharedWorkspace(id=shared_id, name="LumenWorks QA", kind="team", creator_id=OWNER))
            db.flush()
            for account, role in [(OWNER, "owner"), ("eval-sam", "editor"), ("eval-lee", "viewer")]:
                db.add(WorkspaceMember(workspace_id=shared_id, account_id=account, role=role))
            prefs = {
                "preferred_name": "Rowan",
                "timezone": "America/Chicago",
                "memory_learning": True,
                "deep_sleep_enabled": True,
                "history_enabled": True,
                "routing_learning": True,
                "routing_review_enabled": True,
                "routing_mode": "automatic",
                "quiet_enabled": True,
                "quiet_start": "21:00",
                "quiet_end": "08:00",
            }
            db.add(OwnerSettings(owner_id=OWNER, values=prefs))
            schema = ensure(db, OWNER)
            definition = copy.deepcopy(schema.definition)
            for t in definition["types"]:
                if t["id"] in {"task", "project", "note"}:
                    t["fields"].append(
                        {
                            "id": "client",
                            "name": "Client",
                            "description": persona["organization"]["custom_fields"]["client"],
                            "kind": "relation",
                            "target_types": ["client"],
                            "inherit": True,
                        }
                    )
                if t["id"] == "task":
                    t["fields"].append(
                        {
                            "id": "discipline",
                            "name": "Discipline",
                            "description": persona["organization"]["custom_fields"]["discipline"],
                            "kind": "select",
                            "options": [
                                {"id": k, "name": k.capitalize()} for k in persona["job"]["work_types"]
                            ],
                        }
                    )
                if t["id"] == "note":
                    t["fields"] += [
                        {
                            "id": "genre",
                            "name": "Genre",
                            "description": "Genre of saved film/book; not employment context.",
                            "kind": "multiselect",
                            "options": [
                                {"id": k, "name": k}
                                for k in ["science-fiction", "jazz", "vegetarian", "nonfiction"]
                            ],
                        },
                        {
                            "id": "watch_state",
                            "name": "Watch state",
                            "description": "Personal viewing preference, separate from task completion.",
                            "kind": "select",
                            "options": [{"id": k, "name": k} for k in ["want", "watched", "paused"]],
                        },
                        {
                            "id": "favorite",
                            "name": "Favorite",
                            "description": "Whether Rowan explicitly marked the item as a favorite.",
                            "kind": "boolean",
                        },
                        {
                            "id": "rating",
                            "name": "Rating",
                            "description": "Explicit rating, never invented from a mention.",
                            "kind": "number",
                        },
                    ]
            schema.definition = Definition.model_validate(definition).model_dump(mode="json")

        def record(key, type_id, title, **kw):
            value = cmd("record.create", dict(type_id=type_id, title=title, schema_revision=1, **kw))
            refs["records"][key] = value
            return value

        work = record("work", "space", "Work")
        personal = record("personal", "space", "Personal")
        record("community", "space", "Community")
        clients = {}
        for c in persona["job"]["clients"]:
            clients[c["key"]] = record(
                c["key"], "client", c["name"], body=c["description"], parent_id=work["id"]
            )
        for key, name, client in [
            ("beacon", "Beacon Dispatch", "juniper"),
            ("lens", "Transcript Lens", "juniper"),
            ("intake", "Intake Bridge", "cedar"),
            ("inventory", "Inventory Compass", "harbor"),
        ]:
            p = record(
                key,
                "project",
                name,
                parent_id=clients[client]["id"],
                values={"client": clients[client]["id"]},
            )
            refs["projects"][key] = p
        record(
            "delivery-goal",
            "goal",
            "Reduce manual dispatch handling",
            values={
                "metric_baseline": 12,
                "metric_current": 12,
                "metric_target": 24,
                "metric_unit": "hours/week",
            },
        )
        record(
            "learning-goal",
            "goal",
            "Read twelve thoughtful books",
            parent_id=personal["id"],
            values={"metric_target": 12, "metric_current": 1, "metric_unit": "books"},
        )
        refs["actors"] = {}
        for name, kind in [
            ("Rowan Chen", "person"),
            ("Alex Morgan", "person"),
            ("Alex Rivera", "person"),
            ("Research agent", "agent"),
        ]:
            refs["actors"][name] = cmd("actor.create", {"name": name, "kind": kind})
        for t in corpus["tasks"]:
            project = refs["projects"].get(t["project"])
            parent = project or (personal if t["work_type"] == "personal" else work)
            values = {k: t[k] for k in ("due_date", "due_time") if t[k]}
            if t["due_time"]:
                values["due_timezone"] = "Asia/Tokyo" if t["key"] == "international" else "America/Chicago"
            if t["key"] == "plan-no-due":
                values["planned_date"] = "2030-01-16"
            rec = record(
                "task-" + t["key"],
                "task",
                t["title"],
                parent_id=parent["id"],
                values=values,
                body=(
                    "Juniper Pest Care termite inspection invoice, intentionally misfiled."
                    if t["key"] == "misfiled"
                    else "Synthetic Rowan evaluation task."
                ),
            )
            changes = {
                "task_id": rec["task_id"],
                "expected_revision": 1,
                "status": t["status"],
                "work_type": t["work_type"],
                "priority": 1,
            }
            if t["key"] == "archive":
                changes["archived"] = True
            task = cmd("task.update", changes)
            refs["tasks"][t["key"]] = task
        for i in range(corpus["pagination_rows"]):
            rec = record(
                f"page-{i}",
                "task",
                f"Audit Beacon sample {i:03}",
                parent_id=refs["projects"]["beacon"]["id"],
                body=f"Synthetic QA sample; sample number {i}.",
            )
            cmd(
                "task.update",
                {
                    "task_id": rec["task_id"],
                    "expected_revision": 1,
                    "tags": ["pagination", "even" if i % 2 == 0 else "odd"],
                    "priority": i % 4,
                },
            )
        for n in corpus["notes"]:
            row = cmd("note.create", {k: n[k] for k in ("title", "content", "tags")})
            if n["key"] == "archive":
                row = cmd(
                    "note.update",
                    {"note_id": row["id"], "expected_revision": row["revision"], "archived": True},
                )
            refs["notes"][n["key"]] = row
        refs["private_task"] = cmd(
            "task.create", {"title": persona["privacy"]["private_canary"]}, owner="eval-jules"
        )
        refs["private_note"] = cmd(
            "note.create",
            {"title": "Private Jules record", "content": persona["privacy"]["private_canary"]},
            owner="eval-jules",
        )
        refs["shared_workspace"] = shared_id
        # Membership boundary fixtures are inserted directly, not via an invented session.
        with session_scope() as db:
            shared_task = Task(owner_id=shared_id, title="Shared QA checklist")
            db.add(shared_task)
            db.flush()
            refs["shared_task"] = {"id": shared_task.id, "title": shared_task.title}
            ensure(db, shared_id)
        for l in cmd("notelist.setup", {})["items"]:
            refs["lists"][l["name"].lower()] = l
        # Configure fixtures without running or impersonating the dream model.
        with session_scope() as db:
            for key, content in [
                ("cat", "The user has a cat named Miso."),
                ("cat-spelling", "The user has a cat named Mizo."),
                ("jazz", "The user prefers small jazz venues."),
                ("jazz-duplicate", "The user prefers small jazz venues."),
                ("company", "The user works at LumenWorks."),
            ]:
                source = capture_source(db, OWNER, content, str(uid()), explicit=True)
                m = Memory(
                    owner_id=OWNER,
                    source_id=source.id,
                    content=content,
                    evidence=content,
                    fingerprint=fingerprint(content),
                    attribution="owner_statement",
                    embedding=None,
                )
                db.add(m)
                db.flush()
                refs["memories"][key] = m.id
        # Stable record/source chronology; runtime leases and provider timers are not frozen.
        with engine().begin() as db:
            for table in Base.metadata.sorted_tables:
                if table.name in {"auth_sessions", "agent_work", "jobs"}:
                    continue
                changes = {k: CLOCK - timedelta(days=7) for k in ("created_at", "updated_at") if k in table.c}
                if changes:
                    db.execute(table.update().values(**changes))
        with session_scope() as db:
            counts = {
                "tasks": len(list(db.scalars(select(Task)))),
                "notes": len(list(db.scalars(select(Note)))),
                "memories": len(list(db.scalars(select(Memory)))),
            }
    return {"refs": refs, "counts": counts, "clock": CLOCK.isoformat(), "owner": OWNER}
