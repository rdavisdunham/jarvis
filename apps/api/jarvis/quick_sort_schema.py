"""Intent/IDs only; no model-authored answers or human-evidence flags."""
from typing import Literal
from uuid import UUID
from pydantic import BaseModel, ConfigDict, Field, model_validator

class Input(BaseModel):
    model_config = ConfigDict(extra="forbid")

class Start(Input):
    scope: Literal["unfiled", "home", "selected"] = "unfiled"
    kind: Literal["tasks", "notes", "both"] = "tasks"
    home_id: UUID | None = None
    record_ids: list[UUID] = Field(default_factory=list, max_length=50)
    limit: int = Field(default=20, ge=1, le=50)

    @model_validator(mode="after")
    def scope_inputs(self):
        if self.scope == "home" and self.home_id is None:
            raise ValueError("Choose a home to review.")
        if self.scope == "selected" and not self.record_ids:
            raise ValueError("Choose records to review.")
        if self.scope != "home" and self.home_id is not None:
            raise ValueError("home_id belongs to home scope.")
        if self.scope != "selected" and self.record_ids:
            raise ValueError("record_ids belongs to selected scope.")
        return self

class Reply(Input):
    session_id: UUID
    home_id: UUID | None = None

class Control(Input):
    session_id: UUID
    action: Literal["skip", "pause", "resume", "stop"]

COMMANDS = {"quicksort.start": Start, "quicksort.reply": Reply, "quicksort.control": Control}
