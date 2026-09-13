from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from .google_schema import EventFields


class PlanningCreate(EventFields):
    kind: Literal["event", "block"] = "event"
    task_id: str | None = None
    google_calendar_id: str | None = None


class PlanningUpdate(EventFields):
    entry_id: str
    expected_revision: int = Field(ge=1)
    task_id: str | None = None


class PlanningChange(BaseModel):
    model_config = ConfigDict(extra="forbid")
    entry_id: str
    expected_revision: int = Field(ge=1)


class PlanningPublish(PlanningChange):
    calendar_id: str


class PlanningResolve(PlanningChange):
    choice: Literal["google", "eridani"]
    edit_token: str = Field(max_length=60000)


PLANNING_COMMANDS = {
    "planning.create": PlanningCreate,
    "planning.update": PlanningUpdate,
    "planning.delete": PlanningChange,
    "planning.publish": PlanningPublish,
    "planning.unlink": PlanningChange,
    "planning.resolve": PlanningResolve,
}
