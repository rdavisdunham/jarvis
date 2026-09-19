"""Explicit routing and review commands; organization never grants authority."""

from typing import Literal
from pydantic import Field
from .structure_schema import Input


class RunReview(Input):
    pass


class PatternCreate(Input):
    phrase: str = Field(min_length=2, max_length=200)
    type_id: str = Field(min_length=1, max_length=80)
    parent_id: str | None = None
    values: dict = Field(default_factory=dict)
    note_tags: list[str] = Field(default_factory=list, max_length=20)
    reason: str = Field(min_length=1, max_length=2000)


class PatternChange(Input):
    pattern_id: str
    expected_revision: int = Field(ge=1)
    action: Literal["activate", "pause", "forget"]


class ReviewAnswer(Input):
    review_id: str
    expected_revision: int = Field(ge=1)
    question_id: str
    action: Literal["accept", "dismiss", "defer_today", "defer_tomorrow", "defer_week"]


class UnderstandAnswer(Input):
    definition_id: str
    expected_revision: int = Field(ge=1)
    answer: str = Field(min_length=1, max_length=5000)


class PreviewExisting(Input):
    pattern_id: str


class ApplyExisting(Input):
    preview_id: str


COMMANDS = {
    "routing.run": RunReview,
    "routing.create": PatternCreate,
    "routing.change": PatternChange,
    "routing.answer": ReviewAnswer,
    "routing.understand": UnderstandAnswer,
    "routing.preview": PreviewExisting,
    "routing.apply": ApplyExisting,
}
