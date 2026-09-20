from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class CalendarSelection(BaseModel):
    model_config = ConfigDict(extra="forbid")
    calendar_id: str
    expected_revision: int = Field(ge=1)
    selected: bool


class EventFields(BaseModel):
    model_config = ConfigDict(extra="forbid")
    title: str = Field(min_length=1, max_length=500)
    start: str = Field(
        max_length=64, description="YYYY-MM-DD when all_day=true; otherwise a local date/time in timezone."
    )
    end: str = Field(
        max_length=64,
        description="Same format as start, later than start. For all-day events use the day AFTER the last included day.",
    )
    timezone: str = Field(max_length=100)
    all_day: bool = Field(
        default=False,
        description="Set false with BOTH start/end date-times to give an all-day event hours; set true with BOTH dates to make it all-day.",
    )
    location: str = Field(default="", max_length=1000)
    description: str = Field(default="", max_length=10000)
    busy: bool = True


class CalendarCreate(EventFields):
    calendar_id: str = Field(max_length=36)
    repeat: Literal["none", "daily", "weekly", "monthly"] = "none"


class CalendarUpdate(EventFields):
    edit_token: str = Field(
        max_length=60000,
        description="Use the fresh edit token from calendar_event_read for the exact event/occurrence/series.",
    )


class CalendarDelete(BaseModel):
    model_config = ConfigDict(extra="forbid")
    edit_token: str = Field(max_length=60000)


class CalendarRead(BaseModel):
    model_config = ConfigDict(extra="forbid")
    event_id: str = Field(max_length=36)
    scope: Literal["event", "occurrence", "series"] = "event"
    occurrence_start: str | None = Field(default=None, max_length=64)
