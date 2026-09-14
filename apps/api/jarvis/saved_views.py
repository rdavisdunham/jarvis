"""Account-private saved task views, with revision checks and bounded typed state."""

from typing import Annotated, Literal

from fastapi import APIRouter, Depends
from pydantic import BaseModel, ConfigDict, Field

from .auth import Identity, authenticate
from .db import session_scope
from .domain import DomainError, advisory
from .models import OwnerSettings


class ViewState(BaseModel):
    model_config = ConfigDict(extra="forbid")
    tab: Literal["today", "inbox", "week", "all"] = "all"
    query: str = Field(default="", max_length=300)
    status: Literal[
        "all", "active", "open", "in_progress", "waiting", "deferred", "completed", "cancelled"
    ] = "active"
    project: str = Field(default="", max_length=200)
    space: str = Field(default="", max_length=36)
    area: str = Field(default="", max_length=36)
    goal: str = Field(default="", max_length=36)
    assignee: str = Field(default="", max_length=100)
    work_type: str = Field(default="", max_length=80)
    tag: str = Field(default="", max_length=40)
    due_from: str = Field(default="", pattern=r"^$|^\d{4}-\d{2}-\d{2}$")
    due_through: str = Field(default="", pattern=r"^$|^\d{4}-\d{2}-\d{2}$")
    kind: Literal["all", "task", "reminder"] = "all"
    layout: Literal["list", "board", "timeline"] = "list"
    sort: Literal["priority", "due", "planned", "title", "updated"] = "priority"
    group: Literal["status", "project", "assignee"] = "status"
    timeline_date: str = Field(default="", pattern=r"^$|^\d{4}-\d{2}-\d{2}$")
    timeline_span: Literal[14, 30, 90] = 30


class SaveView(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    id: str = Field(min_length=1, max_length=36)
    name: str = Field(min_length=1, max_length=80)
    expected_revision: int = Field(ge=0)
    state: ViewState


class RemoveView(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str = Field(max_length=36)
    expected_revision: int = Field(ge=1)


def collection(db, owner):
    row = db.get(OwnerSettings, owner)
    return list((row.values if row else {}).get("task_views", []))


def save_view(db, owner, args, delete=False):
    advisory(db, f"workspace:{owner}")
    row = db.get(OwnerSettings, owner)
    if row is None:
        row = OwnerSettings(owner_id=owner, values={})
        db.add(row)
    items = collection(db, owner)
    prior = next((x for x in items if x["id"] == args.id), None)
    if (
        not delete
        and prior
        and prior["revision"] == args.expected_revision + 1
        and prior["name"] == args.name
        and prior["state"] == args.state.model_dump()
    ):
        return prior
    if delete and not prior:
        return {"deleted": True}
    if (prior["revision"] if prior else 0) != args.expected_revision:
        raise DomainError(
            "REVISION_CONFLICT", "This saved view changed. Reload the list before updating it.", 409
        )
    if delete and not prior:
        raise DomainError("NOT_FOUND", "That saved view is unavailable.", 404)
    if not delete and not prior and len(items) >= 30:
        raise DomainError("LIMIT_EXCEEDED", "Keep up to 30 saved views per account.")
    updated = (
        None
        if delete
        else {
            "id": args.id,
            "name": args.name,
            "revision": args.expected_revision + 1,
            "state": args.state.model_dump(),
        }
    )
    # Identical retries are safe even if the first reply was lost.
    if prior and updated and prior["name"] == updated["name"] and prior["state"] == updated["state"]:
        return prior
    items = [x for x in items if x["id"] != args.id]
    if updated:
        items.append(updated)
    row.values = {**row.values, "task_views": items}
    db.flush()
    return updated or {"deleted": True}


router = APIRouter()
User = Annotated[Identity, Depends(authenticate)]


@router.get("/api/v1/task-views")
def list_views(user: User):
    with session_scope() as db:
        return {"items": collection(db, user.owner_id)}


@router.post("/api/v1/task-views")
def put_view(args: SaveView, user: User):
    with session_scope() as db:
        return save_view(db, user.owner_id, args)


@router.post("/api/v1/task-views/remove")
def remove_view(args: RemoveView, user: User):
    with session_scope() as db:
        return save_view(db, user.owner_id, args, True)
