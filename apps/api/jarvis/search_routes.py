"""Browser search and personal vocabulary controls; all reads remain workspace-scoped."""

from typing import Annotated
from fastapi import APIRouter, Depends
from starlette.concurrency import run_in_threadpool
from .auth import Identity, authenticate
from .db import session_scope
from .search_models import SearchPreference, SearchIndexState
from .search_schema import (
    SearchQuery,
    SearchSelection,
    SearchFeedback,
    SearchEvent,
    AliasChange,
    SearchPreferences,
)
from . import search_service, search_learning

router = APIRouter(prefix="/api/v1/search")
User = Annotated[Identity, Depends(authenticate)]


@router.post("/records")
async def records(body: SearchQuery, user: User):
    return await run_in_threadpool(search_service.search, user.owner_id, user.account_id, body)


@router.post("/selection")
def selection(body: SearchSelection, user: User):
    with session_scope() as db:
        return search_learning.choose(db, user.owner_id, user.account_id, body)


@router.post("/feedback")
def feedback(body: SearchFeedback, user: User):
    with session_scope() as db:
        return search_learning.feedback(db, user.owner_id, user.account_id, body)


@router.post("/events")
def event(body: SearchEvent, user: User):
    with session_scope() as db:
        return search_learning.event(db, user.owner_id, user.account_id, body)


@router.get("/aliases")
def aliases(user: User):
    with session_scope() as db:
        result = search_learning.state(db, user.owner_id, user.account_id)
        state = db.get(SearchIndexState, user.owner_id)
        result["index"] = {
            "status": state.status if state else "not_started",
            "documents": state.document_count if state else 0,
            "error": state.error if state else None,
        }
        from .config import get_settings

        result["enabled"] = get_settings().semantic_search_enabled
        return result


@router.post("/aliases/{alias_id}")
def change(alias_id: str, body: AliasChange, user: User):
    with session_scope() as db:
        return search_learning.change_alias(db, user.owner_id, user.account_id, alias_id, body)


@router.post("/preferences")
def preferences(body: SearchPreferences, user: User):
    with session_scope() as db:
        search_service.permitted(db, user.owner_id, user.account_id)
        pref = db.get(SearchPreference, (user.owner_id, user.account_id))
        if not pref:
            pref = SearchPreference(owner_id=user.owner_id, account_id=user.account_id)
            db.add(pref)
        pref.learning = body.learning
        if not body.learning:
            from sqlalchemy import select
            from .search_models import SearchSession

            for pending in db.scalars(
                select(SearchSession).where(
                    SearchSession.owner_id == user.owner_id,
                    SearchSession.account_id == user.account_id,
                    SearchSession.outcome == "unknown",
                )
            ):
                pending.suppressed = True
        return {"learning": body.learning}
