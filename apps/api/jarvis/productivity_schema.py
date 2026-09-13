"""Productivity contracts shared by HTTP and Eri's tools."""

from datetime import date
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class Args(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True, allow_inf_nan=False)


class Home(Args):
    space_id: str | None = None
    area_id: str | None = None


class SpaceCreate(Args):
    name: str = Field(min_length=1, max_length=200)
    description: str = Field(default="", max_length=10000)


class SpaceUpdate(Args):
    space_id: str
    expected_revision: int = Field(ge=1)
    name: str | None = Field(default=None, min_length=1, max_length=200)
    description: str | None = Field(default=None, max_length=10000)
    archived: bool | None = None


class AreaCreate(SpaceCreate):
    space_id: str


class AreaUpdate(SpaceUpdate):
    area_id: str
    space_id: str | None = None


class GoalCreate(Home):
    name: str = Field(min_length=1, max_length=200)
    description: str = Field(default="", max_length=10000)
    success_criteria: str = Field(default="", max_length=10000)
    status: Literal["planned", "active", "on_hold", "achieved", "abandoned"] = "planned"
    horizon: Literal["unspecified", "short_term", "long_term"] = "unspecified"
    parent_goal_id: str | None = None
    target_date: date | None = None
    metric_unit: str = Field(default="", max_length=80)
    metric_baseline: float = 0
    metric_current: float | None = None
    metric_target: float | None = None
    project_ids: list[str] = Field(default_factory=list, max_length=200)


class GoalUpdate(GoalCreate):
    goal_id: str
    expected_revision: int = Field(ge=1)
    name: str | None = Field(default=None, min_length=1, max_length=200)
    archived: bool = False


class ProjectCreate(Home):
    name: str = Field(min_length=1, max_length=200)
    description: str = Field(default="", max_length=10000)
    success_criteria: str = Field(default="", max_length=10000)
    status: Literal["planned", "active", "on_hold", "completed", "cancelled"] = "planned"
    start_date: date | None = None
    target_date: date | None = None
    goal_ids: list[str] = Field(default_factory=list, max_length=200)


class ProjectUpdate(ProjectCreate):
    project_id: str
    expected_revision: int = Field(ge=1)
    name: str | None = Field(default=None, min_length=1, max_length=200)
    archived: bool = False


class ActorCreate(Args):
    name: str = Field(min_length=1, max_length=100)
    kind: Literal["person", "agent"] = "person"


class ActorUpdate(Args):
    actor_id: str
    expected_revision: int = Field(ge=1)
    name: str | None = Field(default=None, min_length=1, max_length=100)
    archived: bool | None = None


PRODUCTIVITY_COMMANDS = {
    "space.create": SpaceCreate,
    "space.update": SpaceUpdate,
    "area.create": AreaCreate,
    "area.update": AreaUpdate,
    "goal.create": GoalCreate,
    "goal.update": GoalUpdate,
    "project.create": ProjectCreate,
    "project.update": ProjectUpdate,
    "actor.create": ActorCreate,
    "actor.update": ActorUpdate,
}
