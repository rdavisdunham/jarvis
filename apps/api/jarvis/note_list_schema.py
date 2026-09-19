"""Bounded saved filters and extraction output contracts."""

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field

from .structure_schema import Input

Tag = Annotated[str, Field(min_length=1, max_length=40)]


class ListFilter(Input):
    tags: list[Tag] = Field(default_factory=list, max_length=20)
    type_id: str | None = Field(default=None, max_length=80)
    values: dict = Field(default_factory=dict, max_length=80)


class ListSave(Input):
    id: str | None = Field(default=None, min_length=36, max_length=36)
    expected_revision: int = Field(default=0, ge=0)
    name: str = Field(min_length=1, max_length=80)
    description: str = Field(min_length=1, max_length=5000)
    filters: ListFilter
    automatic: bool = True
    extract_entries: bool = True
    archived: bool = False


class ListSetup(Input):
    pass


class OrganizeNote(Input):
    note_id: str
    expected_revision: int = Field(ge=1)


class FileNote(Input):
    note_id: str
    expected_revision: int = Field(ge=1)
    list_id: str


class Output(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Classification(Output):
    list_id: str
    evidence: str
    confidence: float = Field(ge=0, le=1)


class Entry(Output):
    title: str = Field(min_length=1, max_length=200)
    evidence: str = Field(min_length=1, max_length=2000)
    list_ids: list[str] = Field(min_length=1, max_length=10)
    confidence: float = Field(ge=0, le=1)
    save_intent: bool
    existing_note_id: str | None


class OrganizationResult(Output):
    classifications: list[Classification] = Field(max_length=20)
    entries: list[Entry] = Field(max_length=20)


COMMANDS = {
    "notelist.save": ListSave,
    "notelist.setup": ListSetup,
    "note.organize": OrganizeNote,
    "note.file": FileNote,
}
