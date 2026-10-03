from typing import Annotated, Literal
from pydantic import BaseModel, ConfigDict, Field
from fastapi import APIRouter, Depends, Query
from .auth import Identity, authenticate
from .db import session_scope
from . import review_questions as reviews

router = APIRouter(prefix="/api/v1/questions")
User = Annotated[Identity, Depends(authenticate)]


class Reserve(BaseModel):
    model_config = ConfigDict(extra="forbid")
    conversation_id: str = Field(min_length=36, max_length=36)


class Ack(BaseModel):
    model_config = ConfigDict(extra="forbid")
    event: Literal["presented", "interrupted"]


@router.get("")
def questions(
    user: User,
    category: Literal["all", "memory", "organization", "field", "routing"] = "all",
    status: Literal["open", "pending", "deferred", "resolved", "stale", "all"] = "open",
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=40, ge=1, le=100),
):
    with session_scope() as db:
        return reviews.listing(
            db, user.owner_id, category=category, status=status, offset=offset, limit=limit
        )


@router.post("/reserve")
def reserve(body: Reserve, user: User):
    with session_scope() as db:
        return {
            "invitation": reviews.reserve(db, user.owner_id, user.device_id, body.conversation_id, "text")
        }


@router.post("/delivery/{identity}")
def acknowledge(identity: str, body: Ack, user: User):
    with session_scope() as db:
        return reviews.acknowledge(db, user.owner_id, user.device_id, identity, body.event)
