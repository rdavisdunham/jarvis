"""Shared note command contracts; independent of the storage and provider layers."""

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field


class NoteArgs(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class NoteCreate(NoteArgs):
    title: str = Field(min_length=1, max_length=200)
    content: str = Field(default="", max_length=30000)
    tags: list[Annotated[str, Field(max_length=40)]] = Field(default_factory=list, max_length=20)
    space_id: str | None = None
    area_id: str | None = None
    goal_ids: list[str] = Field(default_factory=list, max_length=200)
    project_ids: list[str] = Field(default_factory=list, max_length=200)
    related_note_ids: list[str] = Field(default_factory=list, max_length=200)
    project_id: str | None = None
    conversation_id: str | None = None
    task_ids: list[str] = Field(default_factory=list, max_length=100)


class NoteUpdate(NoteArgs):
    note_id: str
    expected_revision: int = Field(ge=1)
    title: str | None = Field(default=None, min_length=1, max_length=200)
    content: str | None = Field(default=None, max_length=30000)
    tags: list[Annotated[str, Field(max_length=40)]] = Field(default_factory=list, max_length=20)
    space_id: str | None = None
    area_id: str | None = None
    goal_ids: list[str] = Field(default_factory=list, max_length=200)
    project_ids: list[str] = Field(default_factory=list, max_length=200)
    related_note_ids: list[str] = Field(default_factory=list, max_length=200)
    project_id: str | None = None
    conversation_id: str | None = None
    task_ids: list[str] = Field(default_factory=list, max_length=100)
    archived: bool | None = None


class NoteTaskDraft(NoteArgs):
    title: str = Field(min_length=1, max_length=500)
    evidence: str = Field(min_length=1, max_length=2000)


class NoteTasks(NoteArgs):
    note_id: str
    expected_revision: int = Field(ge=1)
    items: list[NoteTaskDraft] = Field(min_length=1, max_length=30)


class Suggestions(NoteArgs):
    items: list[NoteTaskDraft] = Field(max_length=20)


NOTE_COMMANDS = {"note.create": NoteCreate, "note.update": NoteUpdate, "note.tasks": NoteTasks}
