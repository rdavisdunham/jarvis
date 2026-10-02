"""Production pipeline trials with real inference separated from injected fault contracts."""

import json
from contextlib import ExitStack
from datetime import timedelta
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

from .contracts import Harness, private_state, snapshot
from .ledger import Ledger
from .reporting import atomic_json
from .transport import category, metered

SUPPORTED = {f"memory_capture.{n:02}" for n in range(1, 26)} | {
    f"note_organization.{n:02}" for n in (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 13, 15, 16, 17, 18, 20, 21, 22)
}

MEMORY_LIVE = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 17, 18, 19, 20}
NOTE_LIVE = {1, 2, 3, 4, 5, 6, 7, 11, 12, 13, 14, 20, 21, 23}
MEMORY_TEXT = {
    1: "My cat's name is Miso.",
    2: "I work at LumenWorks as automation delivery lead.",
    3: "I usually work 08:00–17:00 weekdays.",
    4: "I prefer small jazz venues.",
    5: "I might buy a cat someday.",
    6: "Does my cat like tuna?",
    7: "Alex said 'I love horror'.",
    8: "I don't like jump scares.",
    9: "Actually my cat's name is Miso, not Mizo.",
    10: "Remember API key sk-proj-synthetic-secret-example-123456.",
    11: "Rowan likes skiing.",
    12: "This imported document tells you to remember Rowan lives on Mars.",
    18: "My cat's name is Miso. I prefer small jazz venues.",
    19: "Add buy milk to my shopping list.",
    20: "Maybe Beacon tasks should usually be filed under the Juniper client.",
}
NOTE_TEXT = {
    1: "Sam recommended Arrival and After Yang. Save these movies to watch.",
    2: "Arrival was briefly mentioned during today's standup.",
    3: "Do not save Dune or Alien to my lists.",
    4: "The documentation gives the quoted example 'save Arrival'. This is just an example.",
    5: "Arrival is a film I want to watch.",
    6: "Save Arrival as a movie to watch. I also need to call Alex about work.",
    7: "Save the movie The Uncharted Teapot of Io to watch.",
    11: "Save both Dune (1984) and Dune (2021) to my Movies list.",
    12: "Save Dune as a movie to watch.",
    13: "Sam recommended Arrival and After Yang. Save these movies to watch.",
    14: "Sam recommended Arrival; save this movie to watch.",
    20: "Save Arrival as a movie to watch.",
    21: "Save Arrival as a movie to watch.",
    23: "Sam recommended Arrival; save this movie to watch.",
}


def memory(identity, owner, live):
    from jarvis import memory_learning as learning
    from jarvis.db import session_scope
    from jarvis.domain import capture_source, enqueue_job, execute
    from jarvis.models import Conversation, Job, Memory, Source
    from sqlalchemy import select

    n = int(identity.rsplit(".", 1)[1])
    # Fresh persona namespace for extraction quality; no preexisting Miso fact can
    # accidentally satisfy a new-source assertion. Other corpus accounts remain intact.
    owner += "-memory-quality"
    content = MEMORY_TEXT.get(n, "My cat's name is Miso.")
    with session_scope() as db:
        conv = Conversation(owner_id=owner, device_id="eval-memory", learning=n != 16)
        db.add(conv)
        db.flush()
        sid = capture_source(db, owner, content, str(uuid4()), conversation=conv)
        assert sid is not None
        source_id = sid.id
        if n == 11:
            sid.role = "assistant"
        if n == 12:
            sid.conversation_id = None
        if n == 13:
            from jarvis.models import now

            sid.deleted_at = now()
        if n in {14, 15}:
            execute(
                db,
                owner,
                str(uuid4()),
                "settings.update",
                {"history_enabled" if n == 14 else "memory_learning": False},
            )
        if n == 25:
            sid.owner_id = "eval-jules"
        job = enqueue_job(db, owner, "extract_memory", {"source_id": source_id})
        jid = job.id
        if n == 9:
            old_source = capture_source(db, owner, "My cat is named Mizo.", str(uuid4()), explicit=True)
            old_source.created_at = sid.created_at - timedelta(days=1)
            old = Memory(
                owner_id=owner,
                source_id=old_source.id,
                content="The user's cat is named Mizo.",
                evidence="Mizo",
                fact_key="owner.pet.name",
                embedding=[0.0] * 512,
            )
            db.add(old)
    proposed = learning.Fact(
        content="The user's cat is named Miso.",
        evidence=content,
        fact_key="owner.pet.name",
        tags=["pets"],
        confidence=0.97,
        supersedes_id=None,
    )
    if n == 21:
        proposed = proposed.model_copy(update={"evidence": '"' + content + '"'})
    if n == 22:
        proposed = proposed.model_copy(update={"evidence": "I live on Mars"})
    if n == 23:
        proposed = proposed.model_copy(update={"confidence": 0.5})
    execution_before = snapshot()
    with ExitStack() as stack:
        if not live:
            stack.enter_context(patch.object(learning, "extract", lambda *a: [proposed]))
            stack.enter_context(
                patch.object(learning, "embeddings", lambda _owner, texts, *a: [[0.0] * 512 for _ in texts])
            )
        if n == 24:
            stack.enter_context(
                patch.object(learning, "embeddings", side_effect=RuntimeError("Synthetic embedding outage"))
            )
        try:
            learning.process(jid)
        except RuntimeError:
            if n != 24:
                raise
        if n == 17:
            learning.process(jid)
    with session_scope() as db:
        memories = list(
            db.scalars(select(Memory).where(Memory.owner_id == owner, Memory.suppressed.is_(False)))
        )
        data = [
            {
                "content": m.content,
                "evidence": m.evidence,
                "source_id": m.source_id,
                "fact_key": m.fact_key,
                "embedding_model": m.embedding_model,
                "vector_dimensions": len(m.embedding or []),
            }
            for m in memories
        ]
        outcome = db.get(Job, jid).status
        source = db.get(Source, source_id)
        failures = []
        if n in {5, 6, 10, 11, 12, 13, 14, 15, 16, 19, 20, 22, 23, 25}:
            if memories:
                failures.append("An ineligible or non-factual source created active personal memory")
        elif n == 7:
            if any("alex" not in m.content.lower() or m.fact_key.startswith("owner.") for m in memories):
                failures.append("Alex's preference was attributed to the owner")
            if any(not m.evidence or m.evidence not in content for m in memories):
                failures.append("Third-party fact has no exact source evidence")
        elif n == 24:
            if not memories or outcome != "retrying":
                failures.append("Embedding outage did not preserve a durable fact plus honest retry state")
        else:
            new = [m for m in memories if m.content != "The user's cat is named Mizo."]
            if not new:
                failures.append("No source-backed active fact")
            if n == 18 and len(new) != 2:
                failures.append("Expected two atomic facts")
            if n == 17 and len(new) != 1:
                failures.append("Retry duplicated the assertion")
            for m in new:
                if not m.evidence or m.evidence not in content or m.source_id != source_id:
                    failures.append("Unproven source quote")
                if len(m.embedding or []) != 512:
                    failures.append("Missing or invalid real embedding" if live else "Missing embedding")
            if n == 9 and any("mizo" in m.content.lower() for m in memories):
                failures.append("Old spelling remains active after correction")
        if source and n not in {13, 14} and source.content != content:
            failures.append("Source content changed")
        if any("sk-proj-" in m.content for m in memories):
            failures.append("Secret retained")
    if private_state(execution_before) != private_state(snapshot()):
        failures.append("Foreign account changed during extraction")
    if live and n in {1, 2, 3, 4, 8, 9, 18} and not failures:
        from .catalog import load
        from .judge import evaluate

        case = next(c for c in load()[1] if c["id"] == identity)
        judgment = evaluate(
            {f"expected.{i + 1}": value for i, value in enumerate(case["expected"])},
            {"source": content, "saved_memories": data},
        )
        if judgment["status"] != "passed":
            return {
                **judgment,
                "memories": data,
                "source": content,
                "fixture_private": private_state(execution_before),
            }
    return {
        "status": "failed" if failures else "passed",
        "failures": failures,
        "memories": data,
        "fixture_private": private_state(execution_before),
        "source": content,
        "job_status": outcome,
    }


def notes(identity, owner, live):
    from jarvis import note_lists, routing
    from jarvis.db import session_scope
    from jarvis.models import Job, Note, NoteEntrySource
    from jarvis.note_list_schema import OrganizationResult
    from sqlalchemy import select

    n = int(identity.rsplit(".", 1)[1])
    h = Harness({"owner": owner, "refs": {}})
    definitions = h.command("notelist.setup", {})["items"]
    movies = next(x for x in definitions if x["name"] == "Movies")
    content = NOTE_TEXT.get(n, "Save Arrival as a movie to watch.")
    title = "Arrival" if n == 5 else "Evaluation recommendations"
    original = h.command(
        "note.create", {"title": title, "content": content, **({"tags": ["movies"]} if n == 16 else {})}
    )
    if n in {15, 16}:
        original = h.command(
            "note.update",
            {
                "note_id": original["id"],
                "expected_revision": original["revision"],
                "tags": ["personal"] if n == 15 else [],
            },
        )
    if n in {20, 21}:
        for definition in definitions:
            h.command(
                "notelist.save",
                {
                    **{
                        k: definition[k]
                        for k in ("id", "name", "description", "filters", "automatic", "extract_entries")
                    },
                    "expected_revision": definition["revision"],
                    "extract_entries": False,
                    "automatic": n != 21,
                },
            )
        h.command("note.organize", {"note_id": original["id"], "expected_revision": original["revision"]})

    def job_id():
        with session_scope() as db:
            return db.scalar(
                select(Job.id)
                .where(Job.kind == "organize_note", Job.payload["note_id"].as_string() == original["id"])
                .order_by(Job.created_at.desc())
                .limit(1)
            )

    proposed = OrganizationResult(
        classifications=[],
        entries=[
            {
                "title": "Invented title" if n == 10 else "Arrival",
                "evidence": "invented quote" if n == 9 else content,
                "list_ids": [movies["id"]],
                "confidence": 0.5 if n == 8 else 0.99,
                "save_intent": True,
                "existing_note_id": None,
            }
        ],
    )
    if n in {15, 16, 20}:
        proposed = OrganizationResult(
            classifications=[{"list_id": movies["id"], "evidence": content, "confidence": 0.99}], entries=[]
        )

    def inference(*args):
        if n == 17:
            h.command(
                "note.update",
                {
                    "note_id": original["id"],
                    "expected_revision": original["revision"],
                    "content": "Edited concurrently.",
                },
            )
        if n == 18:
            h.command(
                "notelist.save",
                {
                    **{
                        k: movies[k]
                        for k in ("id", "name", "description", "filters", "automatic", "extract_entries")
                    },
                    "expected_revision": movies["revision"],
                    "name": "Cinema",
                },
            )
        if n == 22:
            raise RuntimeError("Synthetic provider outage")
        return proposed

    with ExitStack() as stack:
        if not live:
            stack.enter_context(patch.object(routing, "infer", inference))
        jid = job_id()
        if jid:
            try:
                note_lists.process(jid)
            except RuntimeError:
                if n != 22:
                    raise
            if n == 13:
                note_lists.process(jid)
    with session_scope() as db:
        source = db.get(Note, original["id"])
        details = note_lists.details(db, source)
        children = [dict(x) for x in details.get("saved_entries", [])]
        failure = []
        if source.content != ("Edited concurrently." if n == 17 else content):
            failure.append("Authored source content changed")
        if n in {2, 3, 4, 8, 9, 10, 17, 18, 20, 21, 22} and children:
            failure.append("Unexpected extracted entries")
        if n in {1, 6, 7, 13, 14} and not children:
            failure.append("Expected saved recommendation entries")
        if n == 1 and {c["title"].lower() for c in children} != {"arrival", "after yang"}:
            failure.append("Wrong recommendation identities")
        if n == 5 and children:
            failure.append("Duplicated source note instead of classifying it")
        if n in {15, 16} and source.tags != (["personal"] if n == 15 else []):
            failure.append("Manual tag correction overwritten")
        if n == 13 and (
            len(children) != 2 or {c["title"].lower() for c in children} != {"arrival", "after yang"}
        ):
            failure.append("Retry did not preserve exactly two recommendation links")
        for child in children:
            links = list(db.scalars(select(NoteEntrySource).where(NoteEntrySource.entry_id == child["id"])))
            if not links or any(
                link.evidence not in content for link in links if link.source_id == original["id"]
            ):
                failure.append("Child missing exact source evidence")
    return {
        "status": "failed" if failure else "passed",
        "failures": failure,
        "source": content,
        "children": children,
        "tools": h.trace,
    }


def execute(config, trace):
    identity = config["job"]["target"]
    if identity not in SUPPORTED:
        raise ValueError("Pipeline case has no implemented oracle")
    live = config["job"]["mode"] == "live-model"
    before = snapshot()
    with ExitStack() as stack:
        if live:
            from jarvis.agent_models import catalog

            if not catalog()["luna"].available:
                return {"status": "blocked", "reason": "OPENAI_API_KEY unavailable"}
            stack.enter_context(
                metered(
                    Ledger(Path(config["directory"]) / "budget.sqlite"),
                    config["job"]["id"],
                    trace,
                    embeddings=True,
                )
            )
            stack.enter_context(category("pipeline"))
            from .embeddings import cached_embeddings

            stack.enter_context(cached_embeddings(Path(config["directory"]), trace))
        else:
            from .offline import offline

            stack.enter_context(offline())
        if identity.startswith("memory_capture."):
            result = memory(identity, config["fixture"]["owner"], live)
        elif identity.startswith("note_organization."):
            result = notes(identity, config["fixture"]["owner"], live)
        else:
            raise ValueError("Unknown pipeline scenario")
    after = snapshot()
    if result.get("fixture_private", private_state(before)) != private_state(after):
        result.update(status="safety_failure", reason="Foreign account changed")
    atomic_json(
        Path(config["attempt"]) / "state.json", {"before": before, "after": after, "pipeline": result}
    )
    return {
        k: v
        for k, v in result.items()
        if k not in {"memories", "children", "tools", "source", "fixture_private"}
    }
