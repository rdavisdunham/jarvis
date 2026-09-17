"""Authenticated schema discovery and generic record reads; writes use commands."""

from typing import Annotated
from fastapi import APIRouter, Depends, Query
from .auth import Identity, authenticate
from .db import session_scope
from .domain import owned
from . import structure
from .structure_models import StructureProposal, StructureRecord

router = APIRouter(prefix="/api/v1/structure")
User = Annotated[Identity, Depends(authenticate)]


@router.get("")
def schema(user: User):
    with session_scope() as db:
        return structure.schema_data(db, user.owner_id)


@router.get("/records")
def records(
    user: User,
    type_id: str | None = None,
    capability: str | None = None,
    parent_id: str | None = None,
    query: str = "",
    archived: bool = False,
    limit: int = Query(default=100, ge=1, le=200),
    offset: int = Query(default=0, ge=0, le=100000),
):
    with session_scope() as db:
        return structure.records(
            db,
            user.owner_id,
            type_id=type_id,
            capability=capability,
            parent_id=parent_id,
            query=query,
            archived=archived,
            limit=limit,
            offset=offset,
        )


@router.get("/records/{record_id}")
def record(record_id: str, user: User):
    with session_scope() as db:
        structure.ensure(db, user.owner_id)
        return structure.data(db, owned(db, StructureRecord, record_id, user.owner_id))


@router.get("/proposals/{proposal_id}")
def proposal(proposal_id: str, user: User):
    with session_scope() as db:
        return structure.proposal_data(owned(db, StructureProposal, proposal_id, user.owner_id))


@router.get("/by-core/{kind}/{identity}")
def by_core(kind: str, identity: str, user: User):
    from sqlalchemy import select
    from .models import Task, Note
    from .domain import DomainError, serial

    if kind not in {"task", "note"}:
        raise DomainError("INVALID_ARGUMENT", "Choose task or note.")
    with session_scope() as db:
        core = owned(db, Task if kind == "task" else Note, identity, user.owner_id)
        structure.ensure(db, user.owner_id)
        col = StructureRecord.task_id if kind == "task" else StructureRecord.note_id
        row = db.scalar(
            select(StructureRecord).where(StructureRecord.owner_id == user.owner_id, col == identity)
        )
        if not row:
            structure.observe_core(db, user.owner_id, kind + ".update", serial(core), "registry-import")
            row = db.scalar(
                select(StructureRecord).where(StructureRecord.owner_id == user.owner_id, col == identity)
            )
        return structure.data(db, row)


@router.get("/routing/state")
def routing_state(user: User):
    from .routing import state

    with session_scope() as db:
        return state(db, user.owner_id)


@router.get("/history/applied")
def history(user: User):
    from sqlalchemy import select

    with session_scope() as db:
        return {
            "items": [
                structure.proposal_data(p)
                for p in db.scalars(
                    select(StructureProposal)
                    .where(
                        StructureProposal.owner_id == user.owner_id, StructureProposal.applied_at.is_not(None)
                    )
                    .order_by(StructureProposal.applied_at.desc())
                    .limit(30)
                )
            ]
        }
