"""Opt-in provider acceptance, using pytest's disposable database."""

import os
from uuid import uuid4

import pytest
from jarvis.db import session_scope
from jarvis.domain import capture_source, enqueue_job
from jarvis.memory_learning import process
from jarvis.memory_service import prompt_context, semantic_search
from jarvis.models import Conversation, Memory
from sqlalchemy import select

pytestmark = pytest.mark.skipif(
    os.getenv("JARVIS_MEMORY_ACCEPTANCE") != "1", reason="Cloud acceptance is opt-in"
)


async def test_real_extraction_embedding_and_semantic_recall():
    with session_scope() as db:
        conv = Conversation(owner_id="davin", device_id="cloud", private=False, learning=True)
        db.add(conv)
        db.flush()
        source = capture_source(
            db,
            "davin",
            "My cat's name is Juniper. She is a Maine Coon, and I buy her salmon-flavored food.",
            "cloud-acceptance-" + str(uuid4()),
            conversation=conv,
        )
        job = enqueue_job(db, "davin", "extract_memory", {"source_id": source.id})
        job_id = job.id
    process(job_id)
    with session_scope() as db:
        memories = list(db.scalars(select(Memory)))
        assert memories and all(m.embedding and len(m.embedding) == 512 for m in memories)
        assert all(m.evidence and m.evidence in source.content for m in memories)
    results = await semantic_search("davin", "What should I pick up for my pet to eat?")
    assert results and any("salmon" in m["content"].lower() for m in results)
    assert "salmon" in (await prompt_context("davin", "What food does my pet like?")).lower()
