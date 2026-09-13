"""Small, explicit scheduling requests; dependencies here constrain this plan only."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class PlanTask(BaseModel):
    model_config = ConfigDict(extra="forbid")
    task_id: str = Field(min_length=36, max_length=36)
    expected_revision: int = Field(ge=1)
    minutes: int = Field(ge=1, le=480)
    not_before: str | None = Field(default=None, max_length=64)
    not_after: str | None = Field(default=None, max_length=64)
    after_task_ids: list[str] = Field(default_factory=list, max_length=7)


class PlanRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    start: str = Field(max_length=64)
    end: str = Field(max_length=64)
    timezone: str = Field(max_length=100)
    scope: Literal["selected_calendars", "local_only"] = "selected_calendars"
    tasks: list[PlanTask] = Field(min_length=1, max_length=8)


class PlanCommit(BaseModel):
    model_config = ConfigDict(extra="forbid")
    plan_token: str = Field(min_length=36, max_length=36, description="Short proposal reference returned by planning_suggest; copy exactly.")
