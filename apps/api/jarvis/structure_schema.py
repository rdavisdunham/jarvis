"""Bounded, typed definitions; custom schemas cannot execute code or grant access."""

from datetime import datetime
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
    library_id: Key | None = None
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


ReviewEvery = Literal["1w", "2w", "1m", "3m", "6m", "1y"]


class ReviewCadence(Input):
    """Behavior that periodically resurfaces a type's records for review. Off by default."""

    enabled: bool = False
    every: ReviewEvery = "1m"


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
    opens_as: Literal["auto", "container", "item"] = "auto"
    review: ReviewCadence = Field(default_factory=ReviewCadence)
    archived: bool = False


class Relationship(Input):
    behavior: Literal["related", "blocks"] = "related"
    id: Key
    name: str = Field(min_length=1, max_length=120)
    description: Description
    source_types: list[Key] = Field(min_length=1, max_length=50)
    target_types: list[Key] = Field(min_length=1, max_length=50)
    cardinality: Literal["one_to_one", "one_to_many", "many_to_one", "many_to_many"] = "many_to_many"
    archived: bool = False


class TypePlacement(Input):
    type_id: Key
    parent_type_id: Key | None = None


class Definition(Input):
    types: list[RecordType] = Field(min_length=1, max_length=50)
    relationships: list[Relationship] = Field(default_factory=list, max_length=100)
    field_library: list[FieldDefinition] = Field(default_factory=list, max_length=1000)
    type_layout: list[TypePlacement] = Field(default_factory=list, max_length=50)

    @model_validator(mode="after")
    def consistent(self):
        def unique(rows, label):
            if len({x.id for x in rows}) != len(rows):
                raise ValueError(f"Duplicate {label} identity")

        unique(self.field_library, "library field")
        library = {f.id: f for f in self.field_library}
        for field in self.field_library:
            if field.library_id:
                raise ValueError("Library fields cannot reference another library field")
        for record_type in self.types:
            for field in record_type.fields:
                if field.library_id and "field_library" in self.model_fields_set:
                    shared = library.get(field.library_id)
                    if not shared:
                        raise ValueError("Unknown reusable field")
                    for key in ("name", "description", "kind", "options", "target_types", "multiple", "binding"):
                        setattr(field, key, getattr(shared, key))
        unique(self.types, "type")
        unique(self.relationships, "relationship")
        ids = {t.id for t in self.types}
        placements = {p.type_id: p.parent_type_id for p in self.type_layout}
        if len(placements) != len(self.type_layout):
            raise ValueError("A type may appear once in the layout")
        for key, parent in placements.items():
            if key not in ids or (parent is not None and parent not in ids):
                raise ValueError("Unknown type in layout")
            seen = {key}
            while parent is not None:
                if parent in seen:
                    raise ValueError("The visual layout cannot contain a cycle")
                seen.add(parent)
                parent = placements.get(parent)
        for t in self.types:
            if placements.get(t.id) and placements[t.id] not in t.parent_types:
                raise ValueError("Visual parent must be an allowed home type")
            refs = [f.library_id for f in t.fields if f.library_id]
            if len(refs) != len(set(refs)):
                raise ValueError("A reusable field can be attached once per type")
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
    sort_order: float | None = Field(default=None, allow_inf_nan=False, description="Internal order restored by Revert; prefer move_before_id for ordinary moves.")
    move_before_id: str | None = Field(default=None, description="Move before this sibling, or null to the end. Combine with parent_id to change home.")
    local_notes: str | None = Field(default=None, max_length=30000, description="Eridani-only notes, never synced to a provider. Visible to this workspace.")
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


class ContentsPlan(Input):
    record_id: str = Field(min_length=36, max_length=36)
    expected_revision: int = Field(ge=1)
    schema_revision: int = Field(ge=1)
    operation: Literal["move", "archive"]
    mode: Literal["subtree", "item"] = "subtree"
    parent_id: str | None = None


class ContentsApply(ContentsPlan):
    preview_hash: str = Field(min_length=64, max_length=64)


class ContentsRestore(Input):
    source_command_id: str = Field(min_length=1, max_length=100)


TEMPLATE_DEPTH = 4
TEMPLATE_NODES = 100


class TemplateNode(Input):
    type_id: Key
    title: str = Field(min_length=1, max_length=500)
    body: str = Field(default="", max_length=30000)
    values: dict = Field(default_factory=dict)
    children: list["TemplateNode"] = Field(default_factory=list, max_length=TEMPLATE_NODES)


class TemplatePayload(Input):
    """Default values and outline for the record, plus optional child records."""

    values: dict = Field(default_factory=dict)
    body: str = Field(default="", max_length=30000)
    children: list[TemplateNode] = Field(default_factory=list, max_length=TEMPLATE_NODES)

    @model_validator(mode="after")
    def bounded(self):
        count = 0

        def visit(nodes, depth):
            nonlocal count
            if nodes and depth > TEMPLATE_DEPTH:
                raise ValueError(f"Templates nest at most {TEMPLATE_DEPTH} levels below the record")
            for node in nodes:
                count += 1
                visit(node.children, depth + 1)

        visit(self.children, 1)
        if count > TEMPLATE_NODES:
            raise ValueError(f"Templates hold at most {TEMPLATE_NODES} child records")
        return self


class TemplateCreate(Input):
    type_id: Key
    name: str = Field(min_length=1, max_length=120)
    description: str = Field(default="", max_length=2000)
    payload: TemplatePayload = Field(default_factory=TemplatePayload)


class TemplateUpdate(Input):
    template_id: str = Field(min_length=36, max_length=36)
    expected_revision: int = Field(ge=1)
    name: str | None = Field(default=None, min_length=1, max_length=120)
    description: str | None = Field(default=None, max_length=2000)
    payload: TemplatePayload | None = None


class TemplateArchive(Input):
    template_id: str = Field(min_length=36, max_length=36)
    expected_revision: int = Field(ge=1)
    archived: bool = True


class TemplateCapture(Input):
    record_id: str = Field(min_length=36, max_length=36)
    name: str = Field(min_length=1, max_length=120)
    description: str = Field(default="", max_length=2000)
    include_children: bool = True


class InstantiatePlan(Input):
    template_id: str = Field(min_length=36, max_length=36)
    parent_id: str | None = Field(default=None, description="Main home for the new record; null leaves it unfiled.")
    title: str | None = Field(default=None, min_length=1, max_length=500, description="Defaults to the template name.")
    schema_revision: int = Field(ge=1)


class InstantiateApply(InstantiatePlan):
    preview_hash: str = Field(min_length=64, max_length=64)


class MarkReviewed(Input):
    record_id: str = Field(min_length=36, max_length=36)


class ReviewState(Input):
    """Exact review state, used only to restore a previous receipt."""

    last_reviewed_at: datetime | None = None
    next_review_at: datetime | None = None
    review_paused: bool = False
    review_queued_at: datetime | None = None


class RecordReview(Input):
    record_id: str = Field(min_length=36, max_length=36)
    action: Literal["snooze", "pause", "resume", "restore"] = Field(
        description="snooze moves the next review out by `until`; pause stops reviewing this record; resume restarts it; restore is for Revert only."
    )
    until: Literal["day", "week"] | None = None
    state: ReviewState | None = None

    @model_validator(mode="after")
    def complete(self):
        if self.action == "snooze" and not self.until:
            raise ValueError("Choose how long to snooze: day or week")
        if self.action == "restore" and self.state is None:
            raise ValueError("Restore needs the previous state")
        return self


COMMANDS = {
    "record.mark_reviewed": MarkReviewed,
    "record.review": RecordReview,
    "record.instantiate": InstantiateApply,
    "template.create": TemplateCreate,
    "template.update": TemplateUpdate,
    "template.archive": TemplateArchive,
    "template.capture": TemplateCapture,
    "record.contents": ContentsApply, "record.restore_contents": ContentsRestore,
    "structure.preview": Preview,
    "structure.apply": Apply,
    "structure.restore": SchemaRestore,
    "record.create": RecordCreate,
    "record.update": RecordUpdate,
    "record.link": LinkChange,
}
