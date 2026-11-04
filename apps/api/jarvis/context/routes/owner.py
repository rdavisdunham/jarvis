"""Meeting context for the signed-in owner's browser session: /api/v1/context/* (mounted through `owner_router`).

Session twins of the bot-key routes in `context.py` and `sources.py`, for Eridani's own web app (plan section 2:
the UI reads context only through the context API). Every call goes through the same `service` functions under
`service.as_owner(user)`: the flag (CONTEXT_DISABLED), the personal workspace (PERSONAL_WORKSPACE), the scope
predicate before assembly and the re-check before return. Non-GET requests need the CSRF header
(`auth.authenticate`); the security middleware re-authenticates the session after the handler. The session sees
linked task state (its owner can read tasks). Agent feedback has no session twin: the owner answers Questions.
"""

from typing import Annotated

from fastapi import APIRouter, Depends, Path, Query

from ...auth import Identity, authenticate
from ...db import session_scope
from ...domain import DomainError
from .. import service, sources
from ..schema import SOURCE_STATUSES

User = Annotated[Identity, Depends(authenticate)]
ContextId = Annotated[str, Path(max_length=100)]
router = APIRouter(tags=["Meeting context (owner)"])


@router.post("/search")
def search(body: service.SearchInput, user: User):
    with service.as_owner(user):
        return service.search(body)


@router.post("/brief")
def brief(body: service.BriefInput, user: User):
    with service.as_owner(user):
        return service.brief(body)


@router.get("/evidence/{assertion_id}")
def evidence(assertion_id: ContextId, user: User, include_history: bool = False):
    with service.as_owner(user):
        return service.evidence_for(service.EvidenceInput(assertion_id=assertion_id,
                                                          include_history=include_history))


@router.post("/open-loops")
def open_loops(body: service.OpenLoopsInput, user: User):
    with service.as_owner(user):
        return service.open_loops(body)


@router.post("/meeting-prep")
def meeting_prep(body: service.MeetingPrepInput, user: User):
    with service.as_owner(user):
        return service.meeting_prep(body)


@router.get("/changes")
def changes(user: User, query: Annotated[service.ChangesInput, Query()]):
    with service.as_owner(user):
        return service.changes(query)


# ---------------------------------------------------------------- sources


@router.get("/sources")
def list_sources(user: User, status: str | None = Query(None, max_length=20),
                 limit: int = Query(50, ge=1, le=100), offset: int = Query(0, ge=0, le=100_000)):
    if status is not None and status not in SOURCE_STATUSES:
        raise DomainError("INVALID_ARGUMENT", "Unknown source status.")
    with service.as_owner(user), session_scope() as db:
        caller = service.authorize(db)
        return sources.list_sources(db, caller.owner, status=status, limit=limit, offset=offset)


@router.get("/sources/{source_id}")
def get_source(source_id: ContextId, user: User):
    with service.as_owner(user), session_scope() as db:
        caller = service.authorize(db)
        return sources.get_source(db, caller.owner, source_id)


@router.get("/sources/{source_id}/segments")
def source_segments(source_id: ContextId, user: User, segment_start: int = Query(0, ge=0),
                    segment_end: int | None = Query(None, ge=0), revision: int | None = Query(None, ge=1)):
    """`context_source`: a segment range with the status of the assertions each segment supports."""
    with service.as_owner(user):
        return service.source(service.SourceInput(source_id=source_id, segment_start=segment_start,
                                                  segment_end=segment_end, revision=revision))


@router.delete("/sources/{source_id}")
def delete_source(source_id: ContextId, user: User):
    with service.as_owner(user), session_scope() as db:
        caller = service.authorize(db, service.WRITE, write=True)
        return sources.delete_source(db, caller.owner, source_id)
