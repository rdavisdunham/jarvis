"""Search uses existing identities; it never grants access or rewrites records."""

from datetime import date
from typing import Literal
from pydantic import Field, model_validator
from .structure_schema import Input


class SearchQuery(Input):
    query: str = Field(min_length=1, max_length=1200)
    capability: Literal["work", "content", "timeline", "metric"] | None = None
    type_ids: list[str] = Field(default_factory=list, max_length=50)
    resolved: list[str] = Field(default_factory=list, max_length=5)
    strict: bool = False
    home_id: str | None = None
    values: dict = Field(default_factory=dict)
    statuses: list[str] = Field(default_factory=list, max_length=40)
    due_from: date | None = None
    due_through: date | None = None
    archived: bool = False
    limit: int = Field(default=30, ge=1, le=100)
    offset: int = Field(default=0, ge=0, le=1000000)

    @model_validator(mode="after")
    def dates(self):
        if self.due_from and self.due_through and self.due_from > self.due_through:
            raise ValueError("The start date must precede the end date")
        return self


class SearchSelection(Input):
    search_id: str
    target_key: str
    phrase: str = Field(min_length=2, max_length=200)
    record_ids: list[str] = Field(default_factory=list, max_length=100)


class SearchFeedback(Input):
    search_id: str
    outcome: Literal["corrected", "accepted"]
    replacement_key: str | None = None


class SearchEvent(Input):
    search_id: str | None = None
    work_id: str | None = None
    kind: Literal["presented", "used"]
    record_id: str | None = None


class AliasChange(Input):
    expected_revision: int = Field(ge=1)
    action: Literal["pause", "forget", "confirm", "correct"]
    target_key: str | None = None


class SearchPreferences(Input):
    learning: bool
