"""Canonical export and import (jarvis.context.export): IDs, decisions and versions survive a round trip across
every context table; credentials, generated columns and (by default) embeddings stay out; bad documents are
refused before anything is written; the fixture histories load through import_history."""

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from jarvis.context import export
from jarvis.context import models as m
from jarvis.context.export import HistoryImportError, delete_history, export_history, import_history
from jarvis.db import session_scope
from jarvis.models import Actor, Base
from sqlalchemy import func, select

OWNER, OTHER = "owner-a", "owner-b"
T0 = datetime(2026, 9, 1, 15, 0, tzinfo=UTC)
FIXTURES = Path(__file__).parent / "fixtures"


def at(minutes: int) -> datetime:
    return T0 + timedelta(minutes=minutes)


def vector(seed: int) -> list[float]:
    return [((seed * 7 + i) % 13) / 8 for i in range(m.DIMENSIONS)]


def graph(db, owner=OWNER) -> dict:
    """Rows in every context table, with self references, two revisions, a decision undo and embeddings."""
    conn = m.ContextConnection(owner_id=owner, connector="granola", external_account="robin@example.test",
                               credential_ciphertext="gAAAA-secret", checkpoint={"updated_after": "2026-08-31"},
                               last_success_at=at(0), created_at=at(0), updated_at=at(0))
    db.add(conn)
    db.flush()
    source = m.ContextSource(owner_id=owner, connection_id=conn.id, connector="granola", external_id="note-1",
                             title="Pricing review", current_revision=2, status="ready", created_at=at(1),
                             updated_at=at(5))
    db.add_all([source, m.ContextCursor(owner_id=owner, connection_id=conn.id, name="notes",
                                        value={"date": "2026-09-01"}, created_at=at(0), updated_at=at(0))])
    db.flush()
    revs = [m.ContextSourceRevision(owner_id=owner, source_id=source.id, revision=n, content_hash=str(n) * 64,
                                    external_revision=f"r{n}", payload={"title": "Pricing review", "n": n},
                                    occurred_at=T0, timezone="Europe/London",
                                    participants=[{"name": "Sam", "email": "sam@example.test"}],
                                    received_at=at(n), created_at=at(n)) for n in (1, 2)]
    db.add_all(revs)
    db.flush()
    segs = [m.ContextSegment(owner_id=owner, revision_id=revs[1].id, seq=0, speaker_label="Sam",
                             speaker_actor_id="actor-sam", speaker_link_status="inferred",
                             speaker_provenance={"rule": "attendee"},
                             text="We decided to raise the enterprise price next quarter.", char_start=0,
                             char_end=55, start_ms=0, end_ms=4000, created_at=at(2)),
            m.ContextSegment(owner_id=owner, revision_id=revs[1].id, seq=1, kind="private_note",
                             speaker_attribution="owner", text="Sam owns the pricing memo.", char_start=56,
                             char_end=82, created_at=at(2))]
    db.add_all(segs)
    run = m.ContextRun(owner_id=owner, stage="extract", status="succeeded", source_id=source.id,
                       revision_id=revs[1].id, engine="double", config_version="c1", decision_watermark=2,
                       coverage={"0": "extracted"}, stage_results={"detect": [1]}, cost_usd=0.0125,
                       created_at=at(3), updated_at=at(3))
    db.add_all([run, m.ContextChunk(owner_id=owner, revision_id=revs[1].id, seq=0, segment_start=0,
                                    segment_end=1, token_count=20, text="We decided ... memo.",
                                    embedding=vector(1), embedding_model="text-embedding-3-small",
                                    embedded_at=at(4), created_at=at(2), updated_at=at(4))])
    db.flush()
    d1 = m.ContextDecision(owner_id=owner, seq=1, target_type="assertion", target_id="tbd", operation="wrong",
                           origin="human", principal=owner, negative=True, reason="misheard",
                           recorded_at=at(6), created_at=at(6))
    db.add(d1)
    db.flush()
    d2 = m.ContextDecision(owner_id=owner, seq=2, target_type="assertion", target_id="tbd", operation="undo",
                           origin="human", principal=owner, undoes_id=d1.id, recorded_at=at(7), created_at=at(7))
    old = m.ContextAssertion(owner_id=owner, kind="decision", statement="Keep the enterprise price.",
                             subject_type="project", subject_id="project-1", predicate="selected_option",
                             valid_from=at(-1000), valid_to=T0, lifecycle="superseded", current_version=2,
                             run_id=run.id, created_at=at(3), updated_at=at(8))
    db.add_all([d2, old])
    db.flush()
    new = m.ContextAssertion(owner_id=owner, kind="commitment", statement="Sam drafts the pricing memo.",
                             subject_type="actor", subject_id="actor-sam", predicate="commitment",
                             responsible=[{"entity_type": "actor", "entity_id": "actor-sam"}], due_at=at(10_000),
                             due_precision="day", supersedes_id=old.id, valid_from=T0, run_id=run.id,
                             embedding=vector(2), embedding_model="text-embedding-3-small", embedded_at=at(4),
                             created_at=at(4), updated_at=at(4))
    db.add(new)
    db.flush()
    d1.target_id = d2.target_id = new.id
    db.add_all([
        m.ContextAssertionVersion(owner_id=owner, assertion_id=old.id, version=1, statement=old.statement,
                                  lifecycle="current", valid_from=at(-1000), recorded_from=at(3), recorded_to=at(8),
                                  run_id=run.id, created_at=at(3)),
        m.ContextAssertionVersion(owner_id=owner, assertion_id=old.id, version=2, statement=old.statement,
                                  lifecycle="superseded", valid_from=at(-1000), valid_to=T0, recorded_from=at(8),
                                  decision_id=d1.id, created_at=at(8)),
        m.ContextAssertionVersion(owner_id=owner, assertion_id=new.id, version=1, statement=new.statement,
                                  lifecycle="current", valid_from=T0, recorded_from=at(4), run_id=run.id,
                                  created_at=at(4)),
        m.ContextEvidence(owner_id=owner, assertion_id=new.id, assertion_version=1, source_id=source.id,
                          revision_id=revs[1].id, segment_id=segs[1].id, quote=segs[1].text, char_start=56,
                          char_end=82, created_at=at(4)),
        m.ContextEvidence(owner_id=owner, assertion_id=old.id, assertion_version=1, source_id=source.id,
                          revision_id=revs[0].id, quote="", origin="derived",
                          derivation={"depends_on": [new.id]}, created_at=at(3)),
        m.ContextEntityLink(owner_id=owner, assertion_id=new.id, subject_type="assertion", subject_id=new.id,
                            target_type="actor", target_id="actor-sam", mention="Sam", role="responsible",
                            status="inferred", score=0.91, margin=0.4, run_id=run.id, decision_id=d2.id,
                            created_at=at(4), updated_at=at(4)),
        m.ContextActorIdentity(owner_id=owner, actor_id="actor-sam", namespace="email", value="Sam@Example.test",
                               normalized="sam@example.test", status="confirmed", created_at=at(0),
                               updated_at=at(0)),
        m.ContextActorMerge(owner_id=owner, winner_actor_id="actor-sam", loser_actor_id="actor-sam-2",
                            snapshot={"tasks": ["t1"]}, decision_id=d1.id, created_at=at(9), updated_at=at(9)),
        m.ContextOwnerState(owner_id=owner, epoch=3, decision_seq=2, settings={"digest": "17:00"},
                            created_at=at(0), updated_at=at(9)),
    ])
    proposal = m.ContextProposal(owner_id=owner, source_id=source.id, revision_id=revs[1].id, kind="create",
                                 fingerprint="f" * 64, title="Draft the pricing memo", responsible="actor-sam",
                                 assertion_ids=[new.id], evidence=[{"segment_id": segs[1].id}],
                                 disposition="approved", created_at=at(5), updated_at=at(5))
    db.add(proposal)
    db.flush()
    db.add_all([
        m.ContextPublication(owner_id=owner, proposal_id=proposal.id, proposal_revision=1, destination="eridani",
                             principal=owner, operation_id=f"op-{proposal.id}", task_id="task-1",
                             status="succeeded", created_at=at(6), updated_at=at(6)),
        m.ContextBrief(owner_id=owner, entity_type="project", entity_id="project-1", fingerprint="b" * 64,
                       sections=[{"name": "open_loops", "claims": [{"assertion_ids": [new.id]}]}],
                       dependencies={new.id: 1}, run_id=run.id, created_at=at(7), updated_at=at(7)),
    ])
    db.flush()
    return {"old": old.id, "new": new.id, "d1": d1.id, "d2": d2.id, "proposal": proposal.id}


def actors(db, owner, *ids):
    db.add_all([Actor(id=i, owner_id=owner, name=i.removeprefix("actor-").title()) for i in ids])
    db.flush()


def canonical(data: dict) -> dict:
    """Rows of each section as a sorted list, for comparing documents whose ID-based ordering differs."""
    return {k: sorted(v, key=lambda r: json.dumps(r, sort_keys=True)) if isinstance(v, list) else v
            for k, v in data.items()}


def test_export_covers_every_context_table_and_is_deterministic_json():
    context_tables = {t for t in Base.metadata.tables if t.startswith("context_")}
    assert {model.__tablename__ for _, model, _ in export.SECTIONS} | {"context_owner_state"} == context_tables
    with session_scope() as db:
        graph(db)
        first, second = export_history(db, OWNER), export_history(db, OWNER)
    assert first == second and json.loads(json.dumps(first)) == first
    assert (first["format"], first["version"]) == ("eridani-context-history", 1)
    assert all(first[name] for name in export.SECTION_NAMES), [n for n in export.SECTION_NAMES if not first[n]]
    text = json.dumps(first)
    assert "gAAAA-secret" not in text and "credential_ciphertext" not in first["connections"][0]
    assert not any("tsv" in row for name in export.SECTION_NAMES for row in first[name])
    assert not any(c in row for name in ("chunks", "assertions") for row in first[name]
                   for c in export.EMBEDDING_COLUMNS)
    assert [d["seq"] for d in first["decisions"]] == [1, 2]
    assert first["owner_state"]["decision_seq"] == 2
    assert [a["id"] for a in first["actors"]] == ["actor-sam", "actor-sam-2"]


def test_round_trip_preserves_ids_decisions_versions_and_self_references():
    with session_scope() as db:
        ids = graph(db)
        actors(db, OWNER, "actor-sam", "actor-sam-2")
        before = export_history(db, OWNER)
        delete_history(db, OWNER)
        assert export_history(db, OWNER)["decisions"] == []
        report = import_history(db, OWNER, json.loads(json.dumps(before)))
        after = export_history(db, OWNER)
        assert after == before
        assert report.missing_actors == [] and report.id_map == {}
        assert report.counts["assertion_versions"] == 3 and report.counts["owner_state"] == 1
        assert db.get(m.ContextAssertion, ids["new"]).supersedes_id == ids["old"]
        assert db.get(m.ContextDecision, ids["d2"]).undoes_id == ids["d1"]
        assert float(db.get(m.ContextRun, after["runs"][0]["id"]).cost_usd) == 0.0125
        state = db.get(m.ContextOwnerState, OWNER)
        assert (state.epoch, state.decision_seq) == (3, 2)


def test_import_into_another_owner_with_fresh_ids_rewrites_every_reference():
    with session_scope() as db:
        ids = graph(db)
        source = export_history(db, OWNER)
        report = import_history(db, OTHER, source, preserve_ids=False)
        copied = export_history(db, OTHER)
        assert export_history(db, OWNER) == source  # the original owner is untouched
    mapping = report.id_map
    assert len(mapping) == sum(len(source[n]) for n in export.SECTION_NAMES) + len(source["publications"])
    assert set(mapping.values()).isdisjoint(mapping)
    expected = export._rewrite({k: v for k, v in source.items()}, mapping)
    for name in export.SECTION_NAMES:
        for row in expected[name]:
            row["owner_id"] = OTHER
            if row.get("principal") == OWNER:
                row["principal"] = OTHER
    expected["owner_id"] = OTHER
    expected["owner_state"] = {**expected["owner_state"], "owner_id": OTHER}
    assert canonical(copied) == canonical(expected)
    new_id = mapping[ids["new"]]
    proposal = next(p for p in copied["proposals"])
    assert proposal["assertion_ids"] == [new_id]
    assert next(d for d in copied["decisions"] if d["seq"] == 1)["target_id"] == new_id
    assert next(d for d in copied["decisions"] if d["seq"] == 2)["undoes_id"] == mapping[ids["d1"]]
    assert copied["briefs"][0]["dependencies"] == {new_id: 1}


def test_embeddings_travel_only_when_asked():
    with session_scope() as db:
        graph(db)
        with_vectors = export_history(db, OWNER, include_embeddings=True)
        chunk = with_vectors["chunks"][0]
        assert chunk["embedding"] == pytest.approx(vector(1)) and chunk["embedding_model"]
        delete_history(db, OWNER)
        import_history(db, OWNER, with_vectors)
        assert export_history(db, OWNER, include_embeddings=True) == with_vectors
        without = export_history(db, OWNER)
        delete_history(db, OWNER)
        import_history(db, OWNER, without)
        row = db.scalar(select(m.ContextChunk).where(m.ContextChunk.owner_id == OWNER))
        assert (row.embedding, row.embedding_model, row.embedded_at) == (None, None, None)


def test_missing_actors_are_reported_not_created():
    with session_scope() as db:
        graph(db)
        data = export_history(db, OWNER)
        actors(db, OTHER, "actor-sam")
        report = import_history(db, OTHER, data, preserve_ids=False)
        assert report.missing_actors == ["actor-sam-2"]
        assert db.scalar(select(func.count()).select_from(Actor).where(Actor.owner_id == OTHER)) == 1


def _corrupt(data, change):
    data = json.loads(json.dumps(data))
    change(data)
    return data


@pytest.mark.parametrize(("change", "message"), [
    (lambda d: d.update(format="something-else"), "not an eridani-context-history"),
    (lambda d: d.update(version=2), "unsupported"),
    (lambda d: d.update(extras=[]), "unknown sections"),
    (lambda d: d["connections"][0].update(credential_ciphertext="x"), "non-importable columns"),
    (lambda d: d["segments"][0].update(tsv="'x'"), "non-importable columns"),
    (lambda d: d["assertions"][0].update(kind="opinion"), "is not one of"),
    (lambda d: d["evidence"][0].update(segment_id="missing"), "refers to rows not in the document"),
    (lambda d: d["assertions"].append(dict(d["assertions"][0])), "duplicate ids"),
    (lambda d: d["owner_state"].update(decision_seq=1), "below decision seq"),
    (lambda d: d["revisions"][0].update(received_at="2026-09-01T15:00:00"), "timezone-aware"),
])
def test_bad_documents_are_refused_before_anything_is_written(change, message):
    with session_scope() as db:
        graph(db)
        data = _corrupt(export_history(db, OWNER), change)
        with pytest.raises(HistoryImportError, match=message):
            import_history(db, OTHER, data, preserve_ids=False)
        assert db.scalar(select(func.count()).select_from(m.ContextSource).where(m.ContextSource.owner_id == OTHER)) == 0


def test_import_needs_an_empty_owner_and_unused_ids():
    with session_scope() as db:
        graph(db)
        data = export_history(db, OWNER)
        with pytest.raises(HistoryImportError, match="already has context rows"):
            import_history(db, OWNER, data)
        with pytest.raises(HistoryImportError, match="ids already exist"):
            import_history(db, OTHER, data)


@pytest.mark.parametrize("name", ["history_alpha", "history_beta", "history_gamma"])
def test_fixture_histories_load_through_import_and_export_back_unchanged(name):
    data = json.loads((FIXTURES / name / "history.json").read_text())
    owner = data["owner_id"]
    with session_scope() as db:
        db.add_all([Actor(id=a["id"], owner_id=owner, name=a["name"], kind=a["kind"]) for a in data["actors"]])
        db.flush()
        report = import_history(db, owner, data)
        assert report.missing_actors == []
        assert report.counts["sources"] >= 6 and report.counts["segments"] > 0
        assert export_history(db, owner) == data
        deleted = db.scalars(select(m.ContextSource).where(m.ContextSource.status == "deleted")).all()
        assert len(deleted) == 1 and deleted[0].deleted_at is not None
