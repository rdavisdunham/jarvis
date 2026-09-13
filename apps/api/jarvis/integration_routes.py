import asyncio
from typing import Annotated

from fastapi import APIRouter, Depends
from pydantic import BaseModel, ConfigDict, Field

from . import linear_commands, linear_sync, planning
from .auth import Identity, authenticate
from .db import session_scope
from .domain import DomainError, advisory, emit, owned
from .models import Job, LinearConnection, PlanningEntry

router = APIRouter()
User = Annotated[Identity, Depends(authenticate)]


@router.get("/api/v1/planning/{entry_id}")
def planning_get(entry_id: str, user: User):
    with session_scope() as db:
        return planning.data(db, owned(db, PlanningEntry, entry_id, user.owner_id))


@router.get("/api/v1/planning/{entry_id}/comparison")
async def planning_compare(entry_id: str, user: User):
    return await asyncio.to_thread(planning.comparison, user.owner_id, entry_id)


@router.get("/api/v1/integrations/linear")
def linear_status(user: User):
    with session_scope() as db:
        return linear_sync.summary(db, user.owner_id)


class LinearKey(BaseModel):
    model_config = ConfigDict(extra="forbid")
    api_key: str = Field(min_length=10, max_length=1000)


@router.post("/api/v1/integrations/linear/connect")
async def linear_connect(body: LinearKey, user: User):
    return await asyncio.to_thread(linear_sync.connect, user.owner_id, body.api_key.strip())


@router.post("/api/v1/integrations/linear/sync")
def linear_sync_now(user: User):
    with session_scope() as db:
        advisory(db, f"workspace:{user.owner_id}")
        job_id = linear_sync.queue_sync(db, user.owner_id, force=True)
        if not job_id:
            raise DomainError("LINEAR_SCOPE", "Connect Linear and select a team first.", 409)
        return {"job_id": job_id, "status": "queued"}


@router.post("/api/v1/integrations/linear/disconnect")
def linear_disconnect(user: User):
    with session_scope() as db:
        advisory(db, f"workspace:{user.owner_id}")
        conn = db.get(LinearConnection, user.owner_id)
        if conn:
            conn.credentials = None
            conn.enabled = False
            conn.generation += 1
            conn.revision += 1
            emit(db, user.owner_id, "linear.changed", user.owner_id)
        return {"connected": False}


@router.get("/api/v1/linear/tasks/{task_id}/comparison")
async def linear_compare(task_id: str, user: User):
    return await asyncio.to_thread(linear_commands.comparison, user.owner_id, task_id)


@router.get("/api/v1/linear/writes/{job_id}")
def linear_write_status(job_id: str, user: User):
    with session_scope() as db:
        job = owned(db, Job, job_id, user.owner_id)
        if job.kind != "linear_write":
            raise DomainError("NOT_FOUND", "Linear operation not found.", 404)
        return {"job_id": job.id, "status": job.status, "result": job.result}
