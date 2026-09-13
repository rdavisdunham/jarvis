from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class Args(BaseModel):
    model_config = ConfigDict(extra="forbid")


class LinearSelect(Args):
    expected_revision: int = Field(ge=1)
    team_ids: list[str] = Field(max_length=30)
    only_mine: bool = True


class LinearPublish(Args):
    task_id: str
    expected_revision: int = Field(ge=1)
    team_id: str
    state_id: str | None = None
    project_id: str | None = None
    assignee_id: str | None = None


class LinearCreate(Args):
    title: str = Field(min_length=1, max_length=500)
    description: str = Field(default="", max_length=20000)
    team_id: str
    state_id: str | None = None
    project_id: str | None = None
    assignee_id: str | None = None
    due_date: str | None = None
    priority: int = Field(default=0, ge=0, le=4)


class LinearUpdate(Args):
    task_id: str
    expected_revision: int = Field(ge=1)
    title: str | None = Field(default=None, min_length=1, max_length=500)
    description: str | None = Field(default=None, max_length=20000)
    state_id: str | None = None
    assignee_id: str | None = None
    project_id: str | None = None
    due_date: str | None = None
    priority: int | None = Field(default=None, ge=0, le=4)


class LinearResolve(Args):
    task_id: str
    expected_revision: int = Field(ge=1)
    choice: Literal["linear", "eridani", "unlink"]
    edit_token: str | None = Field(default=None, max_length=90000)


LINEAR_COMMANDS = {
    "linear.select": LinearSelect,
    "linear.publish": LinearPublish,
    "linear.create": LinearCreate,
    "linear.update": LinearUpdate,
    "linear.resolve": LinearResolve,
}
