"""Canonical export and import of one owner's context history. The same format is the fixture format.

    {"format": "eridani-context-history", "version": 1, "owner_id": ..., "include_embeddings": false,
     "actors": [{"id", "name", "kind"}], "owner_state": {...} | null, "connections": [...], ...}

Each table section is a list of rows (column name -> JSON value) in a deterministic order. Not exported:
connection credentials (`credential_ciphertext`), generated `tsv` columns, and embeddings unless asked for (then
`embedding`, `embedding_model` and `embedded_at` travel together; without them the importer leaves the rows
unembedded so the embedding stage picks them up). Datetimes are UTC ISO strings, `Numeric` values strings.

Actors are Eridani rows outside this module: the export lists the ones it references as plain strings and the
importer reports those the target owner lacks instead of creating them. On import every `owner_id` becomes the
target owner, and decision and publication principals equal to the exported owner follow it. With
`preserve_ids=False` every context ID and publication operation ID (unique across owners) is replaced, and every
exact occurrence of an old one (foreign keys, decision targets, link subjects, proposal assertion lists, JSON
payload values and keys) is rewritten.
"""

from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

from sqlalchemy import DateTime, Numeric, insert, select
from sqlalchemy.orm import Session

from ..models import Actor, uid
from . import models as m
from .schema import (
    ASSERTION_KINDS,
    ASSERTION_LIFECYCLES,
    CONNECTORS,
    DECISION_OPERATIONS,
    DECISION_ORIGINS,
    EVIDENCE_ORIGINS,
    EVIDENCE_ROLES,
    LINK_STATUSES,
    MODALITIES,
    PROPOSAL_DISPOSITIONS,
    PROPOSAL_KINDS,
    RUN_STAGES,
    RUN_STATUSES,
    SEGMENT_KINDS,
    SOURCE_STATUSES,
)

FORMAT = "eridani-context-history"
VERSION = 1
EMBEDDING_COLUMNS = ("embedding", "embedding_model", "embedded_at")
NEVER_EXPORTED = {"context_connections": {"credential_ciphertext"}}

# (section, model, sort columns). Order is insertion order: every foreign key points to an earlier section, except
# the self references decisions.undoes_id and assertions.supersedes_id, which import orders topologically.
SECTIONS: tuple[tuple[str, type, tuple[str, ...]], ...] = (
    ("connections", m.ContextConnection, ("connector", "external_account", "id")),
    ("cursors", m.ContextCursor, ("connection_id", "name", "id")),
    ("sources", m.ContextSource, ("connector", "external_id", "id")),
    ("revisions", m.ContextSourceRevision, ("source_id", "revision", "id")),
    ("segments", m.ContextSegment, ("revision_id", "seq", "id")),
    ("chunks", m.ContextChunk, ("revision_id", "seq", "id")),
    ("runs", m.ContextRun, ("created_at", "id")),
    ("decisions", m.ContextDecision, ("seq", "id")),
    ("assertions", m.ContextAssertion, ("created_at", "id")),
    ("assertion_versions", m.ContextAssertionVersion, ("assertion_id", "version", "id")),
    ("evidence", m.ContextEvidence, ("assertion_id", "assertion_version", "id")),
    ("entity_links", m.ContextEntityLink, ("subject_type", "subject_id", "role", "id")),
    ("actor_identities", m.ContextActorIdentity, ("actor_id", "namespace", "normalized", "id")),
    ("actor_merges", m.ContextActorMerge, ("created_at", "id")),
    ("proposals", m.ContextProposal, ("created_at", "id")),
    ("publications", m.ContextPublication, ("created_at", "id")),
    ("briefs", m.ContextBrief, ("entity_type", "entity_id", "version", "id")),
)
SECTION_NAMES = tuple(name for name, _, _ in SECTIONS)
SELF_REFERENCES = {"decisions": "undoes_id", "assertions": "supersedes_id"}

# Columns whose values are enumerations in schema.py; import rejects anything else.
ENUMS: dict[tuple[str, str], tuple[str, ...]] = {
    ("connections", "connector"): CONNECTORS, ("sources", "connector"): CONNECTORS,
    ("sources", "status"): SOURCE_STATUSES, ("segments", "kind"): SEGMENT_KINDS,
    ("runs", "stage"): RUN_STAGES, ("runs", "status"): RUN_STATUSES,
    ("decisions", "operation"): DECISION_OPERATIONS, ("decisions", "origin"): DECISION_ORIGINS,
    ("assertions", "kind"): ASSERTION_KINDS, ("assertions", "modality"): MODALITIES,
    ("assertions", "lifecycle"): ASSERTION_LIFECYCLES, ("assertion_versions", "lifecycle"): ASSERTION_LIFECYCLES,
    ("evidence", "role"): EVIDENCE_ROLES, ("evidence", "origin"): EVIDENCE_ORIGINS,
    ("entity_links", "status"): LINK_STATUSES,
    ("proposals", "kind"): PROPOSAL_KINDS, ("proposals", "disposition"): PROPOSAL_DISPOSITIONS,
}


class HistoryImportError(ValueError):
    """The data is not an importable history for this database (format, references, collisions)."""


@dataclass(frozen=True)
class ImportReport:
    owner_id: str
    counts: dict[str, int]
    missing_actors: list[str] = field(default_factory=list)
    id_map: dict[str, str] = field(default_factory=dict)  # old -> new; empty when IDs were preserved


# ---------------------------------------------------------------- columns and values


def _columns(section: str, model: type, include_embeddings: bool) -> list:
    skip = NEVER_EXPORTED.get(model.__tablename__, set())
    return [c for c in model.__table__.columns
            if c.computed is None and c.name not in skip
            and (include_embeddings or c.name not in EMBEDDING_COLUMNS)]


def _dump(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.astimezone(UTC).isoformat()
    if isinstance(value, Decimal):
        return str(value)
    if hasattr(value, "tolist"):  # pgvector returns numpy arrays
        return [float(x) for x in value.tolist()]
    return value


def _load(column, value: Any) -> Any:
    if value is None:
        return None
    if isinstance(column.type, DateTime):
        parsed = datetime.fromisoformat(value)
        if parsed.tzinfo is None:
            raise HistoryImportError(f"{column.table.name}.{column.name} is not timezone-aware: {value!r}")
        return parsed
    if isinstance(column.type, Numeric):
        return Decimal(str(value))
    return value


def _actor_refs(sections: dict[str, list[dict]]) -> set[str]:
    """Every actor ID the history refers to."""
    refs = {r["actor_id"] for r in sections["actor_identities"]}
    refs |= {r[k] for r in sections["actor_merges"] for k in ("winner_actor_id", "loser_actor_id")}
    refs |= {r["speaker_actor_id"] for r in sections["segments"] if r.get("speaker_actor_id")}
    refs |= {r["responsible"] for r in sections["proposals"] if r.get("responsible")}
    for row in sections["entity_links"]:
        for kind, key in (("subject_type", "subject_id"), ("target_type", "target_id")):
            if row.get(kind) == "actor" and row.get(key):
                refs.add(row[key])
    for row in sections["assertions"]:
        for kind, key in (("subject_type", "subject_id"), ("object_type", "object_id")):
            if row.get(kind) == "actor" and row.get(key):
                refs.add(row[key])
        refs |= {e["entity_id"] for e in row.get("responsible") or []
                 if isinstance(e, dict) and e.get("entity_type") == "actor" and e.get("entity_id")}
    return refs


# ---------------------------------------------------------------- export


def export_history(db: Session, owner: str, *, include_embeddings: bool = False) -> dict:
    """Every context row of `owner` as a JSON-serializable dict in the canonical format."""
    data: dict[str, Any] = {"format": FORMAT, "version": VERSION, "owner_id": owner,
                            "include_embeddings": include_embeddings}
    sections = {}
    for name, model, order in SECTIONS:
        columns = _columns(name, model, include_embeddings)
        table = model.__table__
        rows = db.execute(select(*columns).where(table.c.owner_id == owner)
                          .order_by(*(table.c[c] for c in order))).mappings()
        sections[name] = [{c.name: _dump(row[c.name]) for c in columns} for row in rows]
    state = db.get(m.ContextOwnerState, owner)
    data["owner_state"] = None if state is None else {
        c.name: _dump(getattr(state, c.name)) for c in m.ContextOwnerState.__table__.columns}
    refs = sorted(_actor_refs(sections))
    named = {a.id: a for a in db.scalars(select(Actor).where(Actor.owner_id == owner, Actor.id.in_(refs)))}
    data["actors"] = [{"id": ref, "name": named[ref].name if ref in named else None,
                       "kind": named[ref].kind if ref in named else None} for ref in refs]
    data.update(sections)
    return data


# ---------------------------------------------------------------- import


def _check_shape(data: dict) -> None:
    if not isinstance(data, dict) or data.get("format") != FORMAT:
        raise HistoryImportError(f"not an {FORMAT} document")
    if data.get("version") != VERSION:
        raise HistoryImportError(f"unsupported {FORMAT} version {data.get('version')!r}; this build reads {VERSION}")
    known = {"format", "version", "owner_id", "include_embeddings", "actors", "owner_state", *SECTION_NAMES}
    if unknown := sorted(set(data) - known):
        raise HistoryImportError(f"unknown sections: {unknown}")
    for name, model, _ in SECTIONS:
        allowed = {c.name for c in _columns(name, model, True)}
        for row in data.get(name) or []:
            if extra := sorted(set(row) - allowed):
                raise HistoryImportError(f"{name} has unknown or non-importable columns: {extra}")
            if not row.get("id"):
                raise HistoryImportError(f"{name} row without an id")
            for (section, column), values in ENUMS.items():
                if section == name and column in row and row[column] not in values:
                    raise HistoryImportError(f"{name}.{column} = {row[column]!r} is not one of {values}")


def _topological(rows: list[dict], column: str) -> list[dict]:
    """Rows ordered so that a row's self reference precedes it (stable otherwise)."""
    by_id, done, ordered = {r["id"]: r for r in rows}, set(), []

    def visit(row, path=()):
        if row["id"] in done:
            return
        if row["id"] in path:
            raise HistoryImportError(f"cycle through {column} at {row['id']}")
        parent = by_id.get(row.get(column))
        if parent is not None:
            visit(parent, (*path, row["id"]))
        done.add(row["id"])
        ordered.append(row)

    for row in rows:
        visit(row)
    return ordered


def _rewrite(value: Any, mapping: dict[str, str]) -> Any:
    if isinstance(value, str):
        return mapping.get(value, value)
    if isinstance(value, list):
        return [_rewrite(v, mapping) for v in value]
    if isinstance(value, dict):
        return {mapping.get(k, k): _rewrite(v, mapping) for k, v in value.items()}
    return value


def _has_rows(db: Session, owner: str) -> bool:
    if db.get(m.ContextOwnerState, owner) is not None:
        return True
    return any(db.scalar(select(model.id).where(model.owner_id == owner).limit(1)) for _, model, _ in SECTIONS)


def import_history(db: Session, owner: str, data: dict, *, preserve_ids: bool = True) -> ImportReport:
    """Insert an exported history as `owner`'s context. The owner must have no context rows yet. Every foreign
    key must resolve inside the document; nothing is written when a check fails. Does not commit."""
    _check_shape(data)
    sections = {name: [dict(r) for r in data.get(name) or []] for name in SECTION_NAMES}
    ids = {name: {r["id"] for r in rows} for name, rows in sections.items()}
    if any(len(ids[name]) != len(rows) for name, rows in sections.items()):
        raise HistoryImportError("duplicate ids inside a section")
    tables = {model.__tablename__: name for name, model, _ in SECTIONS}
    for name, model, _ in SECTIONS:
        for column in model.__table__.columns:
            for fk in column.foreign_keys:
                target = tables[fk.column.table.name]
                dangling = sorted({r[column.name] for r in sections[name]
                                   if r.get(column.name) and r[column.name] not in ids[target]})
                if dangling:
                    raise HistoryImportError(f"{name}.{column.name} refers to rows not in the document: {dangling[:5]}")
    state = data.get("owner_state")
    top_seq = max((r["seq"] for r in sections["decisions"]), default=0)
    if state is not None and state.get("decision_seq", 0) < top_seq:
        raise HistoryImportError(f"owner_state.decision_seq {state.get('decision_seq')} is below decision seq {top_seq}")
    if _has_rows(db, owner):
        raise HistoryImportError(f"owner {owner!r} already has context rows; import needs an empty owner")

    mapping: dict[str, str] = {}
    if not preserve_ids:
        mapping = {old: uid() for name in SECTION_NAMES for old in sorted(ids[name])}
        # Operation IDs are unique across owners too; a copy gets its own.
        mapping |= {r["operation_id"]: uid() for r in sections["publications"] if r.get("operation_id")}
    else:
        for name, model, _ in SECTIONS:
            if ids[name] and db.scalar(select(model.id).where(model.id.in_(ids[name])).limit(1)):
                raise HistoryImportError(f"{name} ids already exist in this database; import with preserve_ids=False")
    source_owner = data.get("owner_id")

    prepared: dict[str, list[dict]] = {}  # parse everything before the first write
    for name, model, _ in SECTIONS:
        rows = sections[name]
        if name in SELF_REFERENCES:
            rows = _topological(rows, SELF_REFERENCES[name])
        columns = {c.name: c for c in _columns(name, model, True)}
        prepared[name] = []
        for row in rows:
            row = _rewrite(row, mapping) if mapping else row
            row = {k: _load(columns[k], v) for k, v in row.items()}
            row["owner_id"] = owner
            if "principal" in row and source_owner and row["principal"] == source_owner:
                row["principal"] = owner
            prepared[name].append(row)
    state_columns = {c.name: c for c in m.ContextOwnerState.__table__.columns}
    state_row = None if state is None else {
        **{k: _load(state_columns[k], v) for k, v in state.items() if k in state_columns}, "owner_id": owner}

    counts = {}
    for name, model, _ in SECTIONS:
        # One statement per row keeps self references ordered and server defaults per missing column.
        for row in prepared[name]:
            db.execute(insert(model.__table__).values(**row))
        counts[name] = len(prepared[name])
    if state_row is not None:
        db.execute(insert(m.ContextOwnerState.__table__).values(**state_row))
    counts["owner_state"] = int(state is not None)
    db.flush()

    wanted = [a["id"] for a in data.get("actors") or []]
    wanted = sorted(set(wanted) | _actor_refs(sections))
    present = set(db.scalars(select(Actor.id).where(Actor.owner_id == owner, Actor.id.in_(wanted))))
    return ImportReport(owner_id=owner, counts=counts, missing_actors=[a for a in wanted if a not in present],
                        id_map=mapping)


def delete_history(db: Session, owner: str) -> None:
    """Remove every context row of `owner` (tests and re-imports). Does not touch actors. Does not commit."""
    for name, model, _ in reversed(SECTIONS):
        table = model.__table__
        if name in SELF_REFERENCES:
            db.execute(table.update().where(table.c.owner_id == owner).values({SELF_REFERENCES[name]: None}))
        db.execute(table.delete().where(table.c.owner_id == owner))
    db.execute(m.ContextOwnerState.__table__.delete().where(m.ContextOwnerState.owner_id == owner))
    db.flush()

