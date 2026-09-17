"""Bounded, typed definitions; custom schemas cannot execute code or grant access."""

from typing import Annotated, Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator

Key = Annotated[str, Field(min_length=1, max_length=80, pattern=r"^[a-zA-Z0-9_-]+$")]
Description = Annotated[str, Field(min_length=1, max_length=5000)]
Meaning = Literal["backlog", "open", "in_progress", "waiting", "deferred", "completed", "cancelled"]


class Input(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True, allow_inf_nan=False)


class Option(Input):
    id: Key
    name: str = Field(min_length=1, max_length=120)


class Status(Option):
    meaning: Meaning


class FieldDefinition(Input):
    id: Key
    name: str = Field(min_length=1, max_length=120)
    description: Description
    kind: Literal[
        "text", "long_text", "number", "boolean", "date", "datetime", "select", "multiselect", "relation"
    ]
    options: list[Option] = Field(default_factory=list, max_length=100)
    target_types: list[Key] = Field(default_factory=list, max_length=50)
    multiple: bool = False
    inherit: bool = False
    visible: bool = True
    archived: bool = False
    binding: (
        Literal[
            "due_date",
            "due_time",
            "due_timezone",
            "planned_date",
            "priority",
            "estimate_minutes",
            "assignee",
            "start_date",
            "target_date",
            "metric_baseline",
            "metric_current",
            "metric_target",
            "metric_unit",
        ]
        | None
    ) = None


class RecordType(Input):
    id: Key
    name: str = Field(min_length=1, max_length=120)
    plural: str = Field(min_length=1, max_length=120)
    description: Description
    capabilities: list[Literal["work", "content", "timeline", "metric"]] = Field(
        default_factory=list, max_length=4
    )
    parent_types: list[Key] = Field(default_factory=list, max_length=50)
    fields: list[FieldDefinition] = Field(default_factory=list, max_length=60)
    statuses: list[Status] = Field(default_factory=list, max_length=40)
    archived: bool = False


class Relationship(Input):
    id: Key
    name: str = Field(min_length=1, max_length=120)
    description: Description
    source_types: list[Key] = Field(min_length=1, max_length=50)
    target_types: list[Key] = Field(min_length=1, max_length=50)
    cardinality: Literal["one_to_one", "one_to_many", "many_to_one", "many_to_many"] = "many_to_many"
    archived: bool = False


class Definition(Input):
    types: list[RecordType] = Field(min_length=1, max_length=50)
    relationships: list[Relationship] = Field(default_factory=list, max_length=100)

    @model_validator(mode="after")
    def consistent(self):
        def unique(rows, label):
            if len({x.id for x in rows}) != len(rows):
                raise ValueError(f"Duplicate {label} identity")

        unique(self.types, "type")
        unique(self.relationships, "relationship")
        ids = {t.id for t in self.types}
        for t in self.types:
            unique(t.fields, "field")
            unique(t.statuses, "status")
            if len(set(t.capabilities)) != len(t.capabilities):
                raise ValueError("Capabilities must be unique")
            if set(t.parent_types) - ids:
                raise ValueError("Unknown parent type")
            bindings = [f.binding for f in t.fields if f.binding]
            if len(set(bindings)) != len(bindings):
                raise ValueError("Each operational field has one binding per type")
            if "work" in t.capabilities and not any(s.meaning in {"open", "backlog"} for s in t.statuses):
                raise ValueError("Actionable types need an initial status")
            if "work" in t.capabilities and not any(s.meaning == "completed" for s in t.statuses):
                raise ValueError("Actionable types need a completed status")
            for f in t.fields:
                unique(f.options, "option")
                if set(f.target_types) - ids:
                    raise ValueError("Unknown relationship target type")
                if f.kind == "relation" and not f.target_types:
                    raise ValueError("A relationship field needs target types")
                if f.binding:
                    expected = (
                        "date"
                        if f.binding in {"due_date", "planned_date", "start_date", "target_date"}
                        else "number"
                        if f.binding
                        in {
                            "priority",
                            "estimate_minutes",
                            "metric_baseline",
                            "metric_current",
                            "metric_target",
                        }
                        else "text"
                    )
                    if f.kind != expected or f.inherit:
                        raise ValueError("Operational fields must keep their scalar type and cannot inherit")
                if (
                    f.binding
                    in {
                        "due_date",
                        "due_time",
                        "due_timezone",
                        "planned_date",
                        "priority",
                        "estimate_minutes",
                        "assignee",
                    }
                    and "work" not in t.capabilities
                ):
                    raise ValueError("Task fields require actionable work")
                if f.binding in {"start_date", "target_date"} and "timeline" not in t.capabilities:
                    raise ValueError("Timeline fields require timeline behavior")
                if f.binding and f.binding.startswith("metric_") and "metric" not in t.capabilities:
                    raise ValueError("Metric fields require outcome behavior")
        for r in self.relationships:
            if (set(r.source_types) | set(r.target_types)) - ids:
                raise ValueError("Unknown relationship type")
        return self


class Preview(Input):
    expected_revision: int = Field(ge=1)
    definition: Definition
    status_mappings: dict[str, dict[str, str]] = Field(default_factory=dict)


class Apply(Input):
    proposal_id: str = Field(min_length=36, max_length=36)
    expected_revision: int = Field(ge=1)


class RecordCreate(Input):
    type_id: Key
    title: str = Field(min_length=1, max_length=500)
    body: str = Field(default="", max_length=30000)
    values: dict = Field(default_factory=dict)
    parent_id: str | None = None
    status_id: Key | None = None
    schema_revision: int = Field(ge=1)


class RecordUpdate(Input):
    reset_fields: list[Key] = Field(default_factory=list, max_length=60)
    record_id: str = Field(min_length=36, max_length=36)
    expected_revision: int = Field(ge=1)
    schema_revision: int = Field(ge=1)
    title: str | None = Field(default=None, min_length=1, max_length=500)
    body: str | None = Field(default=None, max_length=30000)
    values: dict = Field(default_factory=dict)
    parent_id: str | None = None
    status_id: Key | None = None
    archived: bool = False


class LinkChange(Input):
    source_id: str = Field(min_length=36, max_length=36)
    target_id: str = Field(min_length=36, max_length=36)
    relationship_id: Key
    expected_revision: int = Field(ge=1)
    schema_revision: int = Field(ge=1)
    remove: bool = False


class SchemaRestore(Input):
    proposal_id: str = Field(min_length=36, max_length=36)
    expected_revision: int = Field(ge=1)


COMMANDS = {
    "structure.preview": Preview,
    "structure.apply": Apply,
    "structure.restore": SchemaRestore,
    "record.create": RecordCreate,
    "record.update": RecordUpdate,
    "record.link": LinkChange,
}
