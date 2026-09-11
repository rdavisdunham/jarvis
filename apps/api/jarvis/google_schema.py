from pydantic import BaseModel, ConfigDict, Field


class CalendarSelection(BaseModel):
    model_config = ConfigDict(extra="forbid")
    calendar_id: str
    expected_revision: int = Field(ge=1)
    selected: bool
