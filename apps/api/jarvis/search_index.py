"""Current canonical search projections and retryable, revision-checked embeddings."""

import hashlib
import json
from sqlalchemy import delete, select
from .db import session_scope
from .models import Actor, Job, Note, Task, now
from .structure_models import StructureRecord, StructureLink, StructureSchema
from .search_models import SearchDocument, SearchIndexState
from .memory_learning import EMBEDDING_MODEL, embeddings


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def snapshot(db, owner):
    """Bulk reads: no per-record SQL, no newest-N window, and no learned guesses."""
    from .domain import serial

    schema = db.get(StructureSchema, owner)
    if not schema:
        return {}, {}
    types = {t["id"]: t for t in schema.definition["types"] if not t.get("archived")}
    rows = {r.id: r for r in db.scalars(select(StructureRecord).where(StructureRecord.owner_id == owner))}
    tasks = {r.id: r for r in db.scalars(select(Task).where(Task.owner_id == owner))}
    notes = {r.id: r for r in db.scalars(select(Note).where(Note.owner_id == owner))}
    actors = {r.id: r for r in db.scalars(select(Actor).where(Actor.owner_id == owner))}
    links = list(db.scalars(select(StructureLink).where(StructureLink.owner_id == owner)))
    relations = {r["id"]: r for r in schema.definition["relationships"] if not r.get("archived")}
    from .models import NoteEntrySource
    note_sources = {}
    for source in db.scalars(select(NoteEntrySource).where(NoteEntrySource.owner_id == owner)):
        origin = notes.get(source.source_id)
        if origin and not origin.archived:
            note_sources.setdefault(source.entry_id, []).append((origin.title, source.evidence))
    documents, records = {}, {}

    def put(key, kind, label, content, **metadata):
        documents[key] = dict(
            key=key,
            kind=kind,
            label=label,
            text=content,
            fingerprint=digest(content),
            definition_fingerprint=digest(
                [key, metadata.get("type_id"), metadata.get("field_kind"), metadata.get("target_types")]
            ),
            **metadata,
        )

    def title(row):
        return (
            tasks[row.task_id].title
            if row.task_id in tasks
            else notes[row.note_id].title
            if row.note_id in notes
            else row.title
        )

    def inherited_valid(f, value):
        values = value if isinstance(value, list) else [value]
        if f["kind"] in ("select", "multiselect"):
            return all(v in {o["id"] for o in f["options"]} for v in values)
        if f["kind"] == "relation":
            return all(
                v in rows and not rows[v].archived and rows[v].type_id in f["target_types"] for v in values
            )
        return True

    for t in types.values():
        put(
            "type:" + t["id"],
            "type",
            t["name"],
            t["name"] + "\n" + t["plural"] + "\n" + t["description"],
            type_id=t["id"],
        )
        for f in t["fields"]:
            if f.get("archived"):
                continue
            key = "field:" + t["id"] + ":" + f["id"]
            put(
                key,
                "field",
                f["name"],
                t["name"] + "\n" + f["name"] + "\n" + f["description"],
                type_id=t["id"],
                field_id=f["id"],
                field_kind=f["kind"],
                target_types=f["target_types"],
            )
            for o in f["options"]:
                put(
                    "option:" + t["id"] + ":" + f["id"] + ":" + o["id"],
                    "option",
                    o["name"],
                    f["name"] + "\n" + o["name"] + "\n" + f["description"],
                    type_id=t["id"],
                    field_id=f["id"],
                    option_id=o["id"],
                )
    for rel in relations.values():
        put(
            "relationship:" + rel["id"],
            "relationship",
            rel["name"],
            rel["name"] + "\n" + rel["description"],
            relationship_id=rel["id"],
        )
    linkmap = {}
    for link in links:
        if link.relationship_id in relations:
            linkmap.setdefault(link.source_id, []).append(link)
            linkmap.setdefault(link.target_id, []).append(link)
    for row in rows.values():
        if row.type_id not in types:
            continue
        t = types[row.type_id]
        item = serial(row)
        item.update(type_name=t["name"], capabilities=t["capabilities"], schema_revision=schema.revision)
        values = dict(row.values)
        task, note = tasks.get(row.task_id), notes.get(row.note_id)
        if task:
            item.update(
                title=task.title,
                body=task.notes,
                archived=task.archived,
                task_revision=task.revision,
                status_meaning=task.status,
            )
            current_status = next(
                (s for s in t["statuses"] if s["id"] == row.status_id and s["meaning"] == task.status), None
            )
            current_status = current_status or next(
                (s for s in t["statuses"] if s["meaning"] == task.status), None
            )
            item["status_id"] = current_status["id"] if current_status else None
            for f in t["fields"]:
                if f.get("binding") and hasattr(task, f["binding"]):
                    v = getattr(task, f["binding"])
                    values[f["id"]] = v.isoformat() if hasattr(v, "isoformat") else v
        else:
            item["status_meaning"] = next(
                (s["meaning"] for s in t["statuses"] if s["id"] == row.status_id), None
            )
        if note:
            item.update(
                title=note.title, body=note.content, note_revision=note.revision, archived=note.archived
            )
        chain, seen, parent = [], {row.id}, row.parent_id
        while parent and parent in rows and parent not in seen:
            seen.add(parent)
            ancestor = rows[parent]
            chain.append(ancestor)
            parent = ancestor.parent_id
        inherited = {}
        for f in t["fields"]:
            if f.get("inherit") and not f.get("archived") and f["id"] not in values:
                for ancestor in chain:
                    if (
                        f["kind"] == "relation"
                        and ancestor.type_id in f["target_types"]
                        and not ancestor.archived
                    ):
                        values[f["id"]] = [ancestor.id] if f["multiple"] else ancestor.id
                        inherited[f["id"]] = ancestor.id
                        break
                    if f["id"] in ancestor.values and inherited_valid(f, ancestor.values[f["id"]]):
                        values[f["id"]] = ancestor.values[f["id"]]
                        inherited[f["id"]] = ancestor.id
                        break
        targets = {"type:" + t["id"]}
        home = [dict(id=p.id, title=title(p), type_id=p.type_id) for p in reversed(chain) if not p.archived]
        targets.update("record:" + p["id"] for p in home)
        parts = [item["title"], t["name"], item["body"], " / ".join(p["title"] for p in home)]
        if task:
            parts.extend(
                [
                    "Tags: " + ", ".join(task.tags),
                    "Work type: " + task.work_type,
                    "Project label: " + (task.project or ""),
                ]
            )
            if task.assignee_id in actors:
                parts.append("Assigned to: " + actors[task.assignee_id].name)
        if note:
            parts.append("Tags: " + ", ".join(note.tags))
            parts.extend("Saved from " + title + ": " + quote for title, quote in note_sources.get(note.id, []))
        labels, bindings = {}, {}
        for f in t["fields"]:
            if f.get("archived"):
                continue
            value = values.get(f["id"])
            if f.get("binding"):
                bindings[f["binding"]] = value
            if value is None or value == "" or value == []:
                continue
            targets.add("field:" + t["id"] + ":" + f["id"])
            vals = value if isinstance(value, list) else [value]
            names = []
            for v in vals:
                if f["kind"] == "relation" and str(v) in rows and not rows[str(v)].archived:
                    names.append(title(rows[str(v)]))
                    targets.add("record:" + str(v))
                elif f["kind"] in ("select", "multiselect"):
                    names.append(next((o["name"] for o in f["options"] if o["id"] == v), str(v)))
                    targets.add("option:" + t["id"] + ":" + f["id"] + ":" + str(v))
                else:
                    names.append(str(v))
            labels[f["id"]] = ", ".join(names)
            parts.append(f["name"] + ": " + labels[f["id"]])
        outgoing = []
        for link in linkmap.get(row.id, []):
            other = link.target_id if link.source_id == row.id else link.source_id
            if other in rows and not rows[other].archived:
                targets.add("record:" + other)
                targets.add("relationship:" + link.relationship_id)
                parts.append(relations[link.relationship_id]["name"] + ": " + title(rows[other]))
                outgoing.append(serial(link))
        item.update(values=values, inherited=inherited, home=home, links=outgoing)
        records[row.id] = dict(record=item, targets=targets, bindings=bindings, field_labels=labels)
        if not item["archived"]:
            put(
                "record:" + row.id,
                "record",
                item["title"],
                "\n".join(str(p or "") for p in parts),
                record_id=row.id,
                type_id=t["id"],
                capabilities=t["capabilities"],
            )
    return documents, records


def queue_index(db, owner, *, dirty=True, force=False):
    from .config import get_settings
    from .domain import advisory, enqueue_job

    if not force and not get_settings().semantic_search_enabled:
        return None
    advisory(db, "search-index:" + owner)
    state = db.get(SearchIndexState, owner)
    if not state:
        state = SearchIndexState(owner_id=owner)
        db.add(state)
        db.flush()
    elif dirty:
        state.generation += 1
    job = db.get(Job, state.job_id) if state.job_id else None
    if job and job.status in ("queued", "running", "dispatched", "retrying"):
        return job.id
    if not dirty and state.indexed_generation == state.generation and state.status == "ready":
        return None
    job = enqueue_job(db, owner, "index_search", {})
    state.job_id, state.status, state.updated_at = job.id, "queued", now()
    return job.id


def index_workspace(job_id):
    from .domain import advisory

    with session_scope() as db:
        job = db.get(Job, job_id)
        if not job or job.status in ("succeeded", "cancelled"):
            return
        owner = job.owner_id
        # Adopt imported/recurring core records before the index lock, preserving
        # the workspace -> events -> index lock order used by normal mutations.
        from .structure import sync_core_records

        schema = db.get(StructureSchema, owner)
        if schema:
            sync_core_records(db, owner, schema)
        advisory(db, "search-index:" + owner)
        state = db.get(SearchIndexState, owner)
        generation = state.generation
        documents, _ = snapshot(db, owner)
        existing = {
            d.target_key: d
            for d in db.scalars(select(SearchDocument).where(SearchDocument.owner_id == owner))
        }
        todo = [
            d
            for key, d in documents.items()
            if key not in existing
            or existing[key].fingerprint != d["fingerprint"]
            or existing[key].embedding_model != EMBEDDING_MODEL
            or not existing[key].vectors
        ]
        job.status, state.status = "running", "indexing"
    try:
        # Batch across documents as well as within long bodies. Complete chunks
        # are persisted only against the same generation observed before the call.
        chunks = [
            (doc, doc["text"][i : i + 1800]) for doc in todo for i in range(0, max(1, len(doc["text"])), 1600)
        ]
        accumulated = {}
        counts = {doc["key"]: max(1, (len(doc["text"]) + 1599) // 1600) for doc in todo}
        for offset in range(0, len(chunks), 16):
            batch = chunks[offset : offset + 16]
            vectors = embeddings(owner, [text for doc, text in batch])
            if len(vectors) != len(batch):
                raise ValueError("Incomplete search embeddings")
            with session_scope() as db:
                advisory(db, "search-index:" + owner)
                state = db.get(SearchIndexState, owner)
                if state.generation != generation:
                    break
                for (doc, _), vector in zip(batch, vectors, strict=True):
                    accumulated.setdefault(doc["key"], []).append(vector)
                    if len(accumulated[doc["key"]]) != counts[doc["key"]]:
                        continue
                    stored = db.scalar(
                        select(SearchDocument).where(
                            SearchDocument.owner_id == owner, SearchDocument.target_key == doc["key"]
                        )
                    )
                    if not stored:
                        stored = SearchDocument(owner_id=owner, target_key=doc["key"])
                        db.add(stored)
                    stored.fingerprint, stored.content = doc["fingerprint"], doc["text"]
                    stored.embedding_model, stored.vectors, stored.updated_at = (
                        EMBEDDING_MODEL,
                        accumulated.pop(doc["key"]),
                        now(),
                    )
        with session_scope() as db:
            advisory(db, "search-index:" + owner)
            state, job = db.get(SearchIndexState, owner), db.get(Job, job_id)
            job.status, job.finished_at = "succeeded", now()
            if state.generation != generation:
                queue_index(db, owner, dirty=False, force=True)
                return
            db.execute(
                delete(SearchDocument).where(
                    SearchDocument.owner_id == owner, SearchDocument.target_key.not_in(list(documents))
                )
            )
            state.indexed_generation, state.status, state.error = generation, "ready", None
            state.document_count, state.updated_at = len(documents), now()
    except Exception as exc:
        with session_scope() as db:
            advisory(db, "search-index:" + owner)
            state, job = db.get(SearchIndexState, owner), db.get(Job, job_id)
            attempts = int(job.payload.get("attempts", 0)) + 1
            job.payload = {**job.payload, "attempts": attempts}
            job.status = "failed" if attempts >= 3 else "retrying"
            state.status, state.error = "unavailable", type(exc).__name__
            job.result = {"error": type(exc).__name__}
            if attempts >= 3:
                job.finished_at = now()
                return
        raise


def backfill(db, *, force=False):
    from .config import get_settings

    if not force and not get_settings().semantic_search_enabled:
        return
    for owner in db.scalars(select(StructureSchema.owner_id)):
        state = db.get(SearchIndexState, owner)
        if (
            force
            or not state
            or (state.status != "ready" and (now() - state.updated_at).total_seconds() > 300)
        ):
            queue_index(db, owner, dirty=force, force=force)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Backfill or inspect the derived search index.")
    parser.add_argument("--status", action="store_true", help="Read aggregate status without queuing work")
    args = parser.parse_args()
    with session_scope() as db:
        if args.status:
            from .config import get_settings

            states = list(db.scalars(select(SearchIndexState)))
            print(
                json.dumps(
                    dict(
                        enabled=get_settings().semantic_search_enabled,
                        workspaces=len(states),
                        ready=sum(
                            s.status == "ready" and s.indexed_generation == s.generation for s in states
                        ),
                        documents=sum(s.document_count for s in states),
                        statuses=[
                            dict(
                                status=s.status,
                                pending_generations=s.generation - s.indexed_generation,
                                error=s.error,
                            )
                            for s in states
                        ],
                    )
                )
            )
        else:
            backfill(db, force=True)
            print("Search backfill queued. Existing records were not changed.")
