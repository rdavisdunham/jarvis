"""Typed contracts for the context module: source envelopes, engine requests and results, retrieval, briefs.

Every model is a frozen value with `extra="forbid"`. The enumerations below are `Literal` types and are also
exported as tuples (`ASSERTION_KINDS`, ...) because the `context_*` tables store them as plain strings.
`json_schemas()` publishes the models for the frontend.

Time rules: every datetime is timezone-aware. Effective intervals are half-open `[valid_from, valid_to)`. An
engine never asserts validity as of "now": a candidate's `valid_from` is either a value supported by evidence
or `None`, which means the source's `occurred_at` (see `CandidateAssertion.effective_from`).
"""

import hashlib
from collections.abc import Iterable
from datetime import datetime
from itertools import pairwise
from typing import Any, Literal, Self, get_args
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, field_validator, model_validator

Connector = Literal["upload", "api", "granola", "zoom"]
SegmentKind = Literal["transcript", "private_note"]
SourceStatus = Literal["received", "queued", "processing", "partial", "ready", "failed", "awaiting_input", "deleted"]
AssertionKind = Literal["fact", "preference", "decision", "rationale", "risk", "open_question", "procedure",
                        "relationship", "commitment"]
Modality = Literal["asserted", "proposed", "hypothetical", "negated", "question"]
AssertionLifecycle = Literal["current", "superseded", "retracted", "contested", "shadow"]
TimePrecision = Literal["instant", "day", "week", "month", "quarter", "year", "unknown"]
LinkStatus = Literal["exact", "inferred", "confirmed", "rejected", "unresolved"]
EvidenceRole = Literal["support", "contradiction"]
EvidenceOrigin = Literal["independent", "derived"]
DecisionOperation = Literal["wrong", "changed", "confirm", "confirm_link", "reject_link", "no_merge", "merge",
                            "unmerge", "approve", "dismiss", "later", "undo"]
DecisionOrigin = Literal["human", "agent", "system"]
ProposalKind = Literal["create", "update", "link", "suggestion"]
ProposalDisposition = Literal["pending", "selected", "approved", "dismissed", "later", "published", "superseded"]
RunStage = Literal["ingest", "extract", "link", "reconcile", "embed", "brief", "lint", "proposals"]
RunStatus = Literal["queued", "running", "succeeded", "partial", "failed", "stale", "cancelled"]
ParticipantRole = Literal["organizer", "attendee", "speaker"]
SpeakerAttribution = Literal["owner", "other"]
EntityType = Literal["actor", "project", "client", "area", "goal", "task", "calendar_event", "linear_issue"]
Coverage = Literal["extracted", "no_content", "skipped", "failed"]
ReconcileOpKind = Literal["equivalent", "conflict", "supersede", "link_change", "no_change"]
HitRefType = Literal["assertion", "segment", "chunk"]
BriefSectionName = Literal["current_state_and_why", "open_loops", "next_meeting_prep", "who_is_who",
                           "recent_changes", "unknowns"]

CONNECTORS = get_args(Connector)
SEGMENT_KINDS = get_args(SegmentKind)
SOURCE_STATUSES = get_args(SourceStatus)
ASSERTION_KINDS = get_args(AssertionKind)
MODALITIES = get_args(Modality)
ASSERTION_LIFECYCLES = get_args(AssertionLifecycle)
TIME_PRECISIONS = get_args(TimePrecision)
LINK_STATUSES = get_args(LinkStatus)
EVIDENCE_ROLES = get_args(EvidenceRole)
EVIDENCE_ORIGINS = get_args(EvidenceOrigin)
DECISION_OPERATIONS = get_args(DecisionOperation)
DECISION_ORIGINS = get_args(DecisionOrigin)
PROPOSAL_KINDS = get_args(ProposalKind)
PROPOSAL_DISPOSITIONS = get_args(ProposalDisposition)
RUN_STAGES = get_args(RunStage)
RUN_STATUSES = get_args(RunStatus)
PARTICIPANT_ROLES = get_args(ParticipantRole)
SPEAKER_ATTRIBUTIONS = get_args(SpeakerAttribution)
ENTITY_TYPES = get_args(EntityType)
COVERAGE = get_args(Coverage)
RECONCILE_OPS = get_args(ReconcileOpKind)
HIT_REF_TYPES = get_args(HitRefType)
BRIEF_SECTIONS = get_args(BriefSectionName)

# Keys whose values the pilot may replace deterministically (plan section 5, reconciliation policy).
EXCLUSIVE_KEYS = ("due_date", "responsible")


def content_hash(texts: Iterable[str]) -> str:
    """sha256 hex of the segment texts joined by newlines: the identity of a revision's content."""
    return hashlib.sha256("\n".join(texts).encode()).hexdigest()


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


# ---------------------------------------------------------------- sources


class Participant(Contract):
    name: str | None = Field(default=None, max_length=200)
    email: str | None = Field(default=None, max_length=320)
    role: ParticipantRole = "attendee"
    speaker_label: str | None = Field(default=None, max_length=200)

    @model_validator(mode="after")
    def _identified(self) -> Self:
        if not (self.name or self.email or self.speaker_label):
            raise ValueError("participant needs a name, email or speaker label")
        return self


class Segment(Contract):
    """One utterance or note block. Offsets index the revision text: `text == revision_text[char_start:char_end]`."""

    seq: int = Field(ge=0)
    kind: SegmentKind = "transcript"
    speaker_label: str | None = Field(default=None, max_length=200)
    attribution: SpeakerAttribution | None = None  # relative to the source owner, when the connector knows it
    text: str = Field(min_length=1)
    start_ms: int | None = Field(default=None, ge=0)
    end_ms: int | None = Field(default=None, ge=0)
    char_start: int | None = Field(default=None, ge=0)
    char_end: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def _ranges(self) -> Self:
        if self.start_ms is not None and self.end_ms is not None and self.end_ms < self.start_ms:
            raise ValueError("end_ms precedes start_ms")
        if (self.char_start is None) != (self.char_end is None):
            raise ValueError("char_start and char_end go together")
        if self.char_start is not None and self.char_end - self.char_start != len(self.text):
            raise ValueError("char offsets do not span the segment text")
        return self


class SourceEnvelope(Contract):
    """What every connector delivers to `sources.receive`. Exactly one of `segments` or `text`.

    Private notes travel in `private_notes` (they become `private_note` segments attributed to the owner);
    `summary` is display-only and never evidence.
    """

    envelope_version: Literal[1] = 1
    connector: Connector
    external_id: str = Field(min_length=1, max_length=500)
    title: str = Field(default="", max_length=500)
    occurred_at: AwareDatetime
    ended_at: AwareDatetime | None = None
    timezone: str | None = None
    participants: list[Participant] = Field(default_factory=list)
    segments: list[Segment] | None = None
    text: str | None = None
    private_notes: str | None = None
    summary: str | None = None
    external_revision: str | None = Field(default=None, max_length=200)
    content_hash: str | None = Field(default=None, max_length=200)
    calendar_event_id: str | None = Field(default=None, max_length=500)
    upstream_owner_email: str | None = Field(default=None, max_length=320)
    upstream_owner_name: str | None = Field(default=None, max_length=200)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("timezone")
    @classmethod
    def _zone(cls, value: str | None) -> str | None:
        if value is not None:
            try:
                ZoneInfo(value)
            except (ZoneInfoNotFoundError, ValueError) as error:
                raise ValueError(f"unknown timezone {value!r}") from error
        return value

    @model_validator(mode="after")
    def _shape(self) -> Self:
        if (self.segments is None) == (self.text is None):
            raise ValueError("supply exactly one of segments or text")
        if self.segments is not None:
            if not self.segments:
                raise ValueError("segments is empty")
            seqs = [s.seq for s in self.segments]
            if any(b <= a for a, b in pairwise(seqs)):
                raise ValueError("segment seq must strictly increase")
            if any(s.kind != "transcript" for s in self.segments):
                raise ValueError("private notes go in private_notes, not segments")
        elif not self.text.strip():
            raise ValueError("text is empty")
        if not (self.external_revision or self.content_hash):
            raise ValueError("supply external_revision or content_hash")
        if self.ended_at is not None and self.ended_at < self.occurred_at:
            raise ValueError("ended_at precedes occurred_at")
        return self


class Chunk(Contract):
    """Consecutive segments `[segment_start, segment_end]` (inclusive seqs) of one revision; the embedding unit."""

    seq: int = Field(ge=0)
    segment_start: int = Field(ge=0)
    segment_end: int = Field(ge=0)
    token_count: int = Field(ge=0)
    text: str

    @model_validator(mode="after")
    def _range(self) -> Self:
        if self.segment_end < self.segment_start:
            raise ValueError("segment_end precedes segment_start")
        return self


# ---------------------------------------------------------------- interpretation


class EntityRef(Contract):
    """An Eridani entity as `(entity_type, entity_id)`, or an unresolved mention of one (`entity_id` absent)."""

    entity_type: EntityType
    entity_id: str | None = Field(default=None, max_length=100)
    mention: str | None = Field(default=None, max_length=300)

    @model_validator(mode="after")
    def _identified(self) -> Self:
        if not (self.entity_id or self.mention):
            raise ValueError("entity reference needs an entity_id or a mention")
        return self


class EvidenceRef(Contract):
    """An exact quote: `segment.text[char_start:char_end] == quote`. `source_revision_id` is absent inside an
    extraction result, where it means the revision being extracted."""

    segment_seq: int = Field(ge=0)
    quote: str = Field(min_length=1)
    char_start: int = Field(ge=0)
    char_end: int = Field(ge=0)
    role: EvidenceRole = "support"
    origin: EvidenceOrigin = "independent"
    source_revision_id: str | None = None

    @model_validator(mode="after")
    def _span(self) -> Self:
        if self.char_end - self.char_start != len(self.quote):
            raise ValueError("char offsets do not span the quote")
        return self

    def matches(self, segment_text: str) -> bool:
        return segment_text[self.char_start:self.char_end] == self.quote


class TimeBound(Contract):
    """A normalized time with its precision and, when it came from words, the raw expression ("in two weeks")."""

    at: AwareDatetime
    precision: TimePrecision = "instant"
    expression: str | None = Field(default=None, max_length=200)


class CandidateAssertion(Contract):
    kind: AssertionKind
    statement: str = Field(min_length=1, max_length=2000)
    subject: EntityRef | None = None
    object: EntityRef | None = None
    key: str | None = Field(default=None, max_length=100)  # predicate, e.g. "due_date"; see EXCLUSIVE_KEYS
    modality: Modality = "asserted"
    inferred: bool = False  # False: stated outright; True: inferred from what was said
    valid_from: TimeBound | None = None  # None: the source's occurred_at
    valid_to: TimeBound | None = None  # exclusive; None: open-ended
    responsible: list[EntityRef] = Field(default_factory=list)  # commitments only; unresolved stays a mention
    recipient: EntityRef | None = None  # commitments only
    due: TimeBound | None = None  # commitments only
    evidence: list[EvidenceRef] = Field(min_length=1)
    confidence: float | None = Field(default=None, ge=0, le=1)  # diagnostic, never a gate

    @model_validator(mode="after")
    def _rules(self) -> Self:
        if not any(e.role == "support" for e in self.evidence):
            raise ValueError("an assertion needs at least one supporting evidence reference")
        if self.kind == "commitment":
            if not self.responsible:
                raise ValueError("a commitment names who is responsible (an unresolved mention is fine)")
        elif self.responsible or self.recipient or self.due:
            raise ValueError("responsible, recipient and due apply only to commitments")
        if self.valid_from and self.valid_to and self.valid_to.at <= self.valid_from.at:
            raise ValueError("valid_to must follow valid_from")
        return self

    def effective_from(self, occurred_at: datetime) -> TimeBound:
        return self.valid_from or TimeBound(at=occurred_at, precision="instant")


class AssertionView(CandidateAssertion):
    """A canonical assertion as engines see it. `valid_from` is always resolved."""

    id: str
    revision: int = Field(ge=1)
    lifecycle: AssertionLifecycle = "current"
    valid_from: TimeBound
    recorded_at: AwareDatetime
    source_revision_id: str | None = None


class LinkProposal(Contract):
    """An engine's proposed resolution of a mention. Engines propose `exact`, `inferred` or `unresolved`;
    `confirmed` and `rejected` come only from decisions."""

    mention: str = Field(min_length=1, max_length=300)
    candidate: EntityRef
    status: LinkStatus
    score: float | None = None
    margin: float | None = None
    evidence: list[EvidenceRef] = Field(default_factory=list)

    @model_validator(mode="after")
    def _status(self) -> Self:
        if self.status in ("confirmed", "rejected"):
            raise ValueError("engines cannot confirm or reject links")
        if (self.status == "unresolved") != (self.candidate.entity_id is None):
            raise ValueError("resolved links name an entity_id; unresolved ones do not")
        return self


class ProposedAction(Contract):
    kind: ProposalKind
    title: str = Field(min_length=1, max_length=500)
    description: str = Field(default="", max_length=5000)
    responsible: list[EntityRef] = Field(default_factory=list)
    due: TimeBound | None = None
    target: EntityRef | None = None  # the existing task or issue for update and link
    assertion_indexes: list[int] = Field(default_factory=list)  # into ExtractionResult.assertions
    evidence: list[EvidenceRef] = Field(min_length=1)

    @model_validator(mode="after")
    def _target(self) -> Self:
        if self.kind in ("update", "link") and not (self.target and self.target.entity_id):
            raise ValueError(f"{self.kind} proposals name a resolved target")
        return self


class Decision(Contract):
    """A prior human, agent or system decision an engine must respect (e.g. a scoped "wrong" or "reject_link")."""

    id: str
    seq: int = Field(ge=0)  # monotonic per owner; RunMeta.decision_watermark is the highest seq considered
    operation: DecisionOperation
    origin: DecisionOrigin
    target_type: str = Field(max_length=50)  # assertion, link, proposal, actor, ...
    target_id: str
    expected_revision: int | None = None
    statement: str | None = None  # the interpretation decided on, so engines do not propose it again
    reason: str = ""
    recorded_at: AwareDatetime


class EntityCandidate(Contract):
    """An authorized entity an engine may link to."""

    ref: EntityRef
    name: str = Field(min_length=1, max_length=300)
    aliases: list[str] = Field(default_factory=list)
    detail: str = ""

    @model_validator(mode="after")
    def _resolved(self) -> Self:
        if not self.ref.entity_id:
            raise ValueError("candidates are existing entities")
        return self


class ExtractLimits(Contract):
    max_assertions: int = Field(default=200, ge=0)
    max_actions: int = Field(default=50, ge=0)


class ExtractRequest(Contract):
    owner_id: str
    source_revision_id: str
    occurred_at: AwareDatetime
    timezone: str | None = None
    title: str = ""
    participants: list[Participant] = Field(default_factory=list)
    segments: list[Segment] = Field(min_length=1)
    candidates: list[EntityCandidate] = Field(default_factory=list)
    decisions: list[Decision] = Field(default_factory=list)
    limits: ExtractLimits = ExtractLimits()

    @model_validator(mode="after")
    def _numbered(self) -> Self:
        if len({s.seq for s in self.segments}) != len(self.segments):
            raise ValueError("segment seq values repeat")
        return self

    @property
    def content_hash(self) -> str:
        return content_hash(s.text for s in self.segments)


class ExtractionResult(Contract):
    assertions: list[CandidateAssertion] = Field(default_factory=list)
    links: list[LinkProposal] = Field(default_factory=list)
    actions: list[ProposedAction] = Field(default_factory=list)
    coverage: dict[int, Coverage] = Field(default_factory=dict)  # segment seq -> disposition
    diagnostics: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _indexes(self) -> Self:
        for action in self.actions:
            if any(not 0 <= i < len(self.assertions) for i in action.assertion_indexes):
                raise ValueError("action refers to an assertion index out of range")
        return self


def check_extraction(request: ExtractRequest, result: ExtractionResult) -> list[str]:
    """Structural problems of a result against its request: unknown segments, inexact quotes, missing coverage,
    exceeded limits. Empty when sound. Semantic validation (entailment, negation, dates) is not done here."""
    texts = {s.seq: s.text for s in request.segments}
    problems = [f"segment {seq} has no coverage disposition" for seq in texts if seq not in result.coverage]
    problems += [f"coverage names unknown segment {seq}" for seq in result.coverage if seq not in texts]
    refs = [e for a in result.assertions for e in a.evidence] + [e for x in result.actions for e in x.evidence]
    refs += [e for link in result.links for e in link.evidence]
    for ref in refs:
        if ref.segment_seq not in texts:
            problems.append(f"evidence cites unknown segment {ref.segment_seq}")
        elif not ref.matches(texts[ref.segment_seq]):
            problems.append(f"evidence quote is not exact in segment {ref.segment_seq}: {ref.quote!r}")
    if len(result.assertions) > request.limits.max_assertions:
        problems.append("too many assertions")
    if len(result.actions) > request.limits.max_actions:
        problems.append("too many actions")
    return problems


class ReconcileRequest(Contract):
    owner_id: str
    changed: list[AssertionView] = Field(min_length=1)
    related: list[AssertionView] = Field(default_factory=list)
    decisions: list[Decision] = Field(default_factory=list)
    exclusive_keys: list[str] = Field(default_factory=lambda: list(EXCLUSIVE_KEYS))


class ReconcileOp(Contract):
    """`targets` are assertion IDs; for `supersede` they are `[old, new]`. `target_revisions` pins the revisions
    the op was computed against so core can refuse stale ops."""

    op: ReconcileOpKind
    targets: list[str] = Field(min_length=1)
    target_revisions: dict[str, int] = Field(default_factory=dict)
    link: LinkProposal | None = None  # link_change only
    explanation: str = ""
    dependencies: list[str] = Field(default_factory=list)  # other assertion or decision IDs relied on

    @model_validator(mode="after")
    def _arity(self) -> Self:
        if self.op == "supersede" and len(self.targets) != 2:
            raise ValueError("supersede names [old, new]")
        if self.op in ("equivalent", "conflict") and len(self.targets) < 2:
            raise ValueError(f"{self.op} needs at least two targets")
        if (self.op == "link_change") != (self.link is not None):
            raise ValueError("link is required for link_change and only there")
        return self


# ---------------------------------------------------------------- retrieval


class ScopePredicate(Contract):
    """Authorization and temporal scope, applied inside the engine's candidate query.

    Defaults exclude superseded and retracted assertions; `shadow` assertions are never returned.
    `valid_at` filters effective time (`None`: current state by lifecycle); `known_at` filters recorded time
    (`None`: current recorded state)."""

    owner_id: str = Field(min_length=1)
    entities: list[EntityRef] = Field(default_factory=list)
    include_superseded: bool = False
    include_retracted: bool = False
    valid_at: AwareDatetime | None = None
    known_at: AwareDatetime | None = None
    kinds: list[AssertionKind] = Field(default_factory=list)

    @model_validator(mode="after")
    def _resolved(self) -> Self:
        if any(not e.entity_id for e in self.entities):
            raise ValueError("scope entities must be resolved")
        return self


class RetrieveQuery(Contract):
    text: str = Field(min_length=1, max_length=2000)
    scope: ScopePredicate
    limit: int = Field(default=20, ge=1, le=100)
    include_raw: bool = False  # also return segments and chunks, annotated with superseding status


class Citation(Contract):
    source_id: str
    source_revision_id: str
    segment_start: int = Field(ge=0)
    segment_end: int = Field(ge=0)
    quote: str | None = None


class Hit(Contract):
    ref_type: HitRefType
    id: str
    revision: int = Field(ge=1)
    score: float
    rank: dict[str, float] = Field(default_factory=dict)  # e.g. lexical_rank, semantic_rank, rrf
    snippet: str = ""
    citation: Citation | None = None
    kind: AssertionKind | None = None
    lifecycle: AssertionLifecycle | None = None  # for raw hits: the status of the assertion they support
    superseded_by: str | None = None
    valid_from: AwareDatetime | None = None
    valid_to: AwareDatetime | None = None
    entities: list[EntityRef] = Field(default_factory=list)


# ---------------------------------------------------------------- briefs


class BriefRequest(Contract):
    scope: ScopePredicate
    entity: EntityRef | None = None
    query: str | None = Field(default=None, max_length=2000)
    sections: list[BriefSectionName] = Field(default_factory=lambda: list(BRIEF_SECTIONS))

    @model_validator(mode="after")
    def _subject(self) -> Self:
        if not (self.entity or self.query):
            raise ValueError("a brief is about an entity or a query")
        if self.entity and not self.entity.entity_id:
            raise ValueError("brief entity must be resolved")
        return self


class TaskView(Contract):
    id: str
    revision: int = Field(ge=1)
    title: str
    status: str
    due_at: AwareDatetime | None = None
    assertion_ids: list[str] = Field(default_factory=list)  # assertions the task is linked to


class SynthesizeRequest(Contract):
    brief: BriefRequest
    assertions: list[AssertionView] = Field(default_factory=list)  # current, authorized
    history: list[AssertionView] = Field(default_factory=list)  # superseded or retracted, for "what changed"
    tasks: list[TaskView] = Field(default_factory=list)


class BriefClaim(Contract):
    text: str = Field(min_length=1)
    assertion_ids: list[str] = Field(default_factory=list)
    task_ids: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _cited(self) -> Self:
        if not (self.assertion_ids or self.task_ids):
            raise ValueError("every brief claim cites an assertion or task")
        return self


class BriefSection(Contract):
    name: BriefSectionName
    claims: list[BriefClaim] = Field(default_factory=list)
    gaps: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------- runs


class RunMeta(Contract):
    stage: RunStage
    engine: str = Field(min_length=1)  # "native", "double"
    model: str | None = None
    prompt_version: str | None = None
    config_version: str
    # e.g. {"source_revision": "<id>", "actor:<id>": "3"}
    input_revisions: dict[str, str] = Field(default_factory=dict)
    decision_watermark: int = Field(default=0, ge=0)


PUBLIC_MODELS: tuple[type[BaseModel], ...] = (
    Participant, Segment, SourceEnvelope, Chunk, EntityRef, EvidenceRef, TimeBound, CandidateAssertion,
    AssertionView, LinkProposal, ProposedAction, Decision, EntityCandidate, ExtractLimits, ExtractRequest,
    ExtractionResult, ReconcileRequest, ReconcileOp, ScopePredicate, RetrieveQuery, Citation, Hit, BriefRequest,
    TaskView, SynthesizeRequest, BriefClaim, BriefSection, RunMeta,
)


def json_schemas() -> dict[str, dict]:
    return {model.__name__: model.model_json_schema() for model in PUBLIC_MODELS}
