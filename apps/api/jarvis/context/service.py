"""The context application service: the one read and feedback surface behind REST (`routes/context.py`) and MCP
(`mcp.py`) (plan_final.md sections 2, 6 and 9).

Authorization runs before assembly and again before return. Before: the flag, the bot key and its scope, and the
personal-workspace rule; the scope predicate (owner, availability, lifecycle, temporal validity) goes into the
candidate query. After: the key is re-read (a revocation mid-request wins), and every assertion a response cites
is re-checked against the same predicate and must still have a surviving citation (a support span from a source
that is not deleted, with its quote intact). Anything that fails is withheld and counted in `gaps`, never
silently returned.

Responses are compact and ID-addressable: statements carry assertion IDs and one citation; agents drill down with
`evidence` (all spans) and `source` (segment ranges). Each statement says what it is:

- `basis`: `stated` (said outright in a meeting), `inferred` (an interpretation of what was said) or
  `source_text` (a raw transcript passage);
- `status`: `current`, `contested` (sources disagree), `historical` (superseded or ended) or `retracted`.

Superseded and retracted context is excluded unless `include_history` is set. Linked task details need
`tasks:read`: a context key cannot read task state through a brief or open loop.

Agent feedback (`feedback`) never changes context: it records an agent-origin suggestion (a `context_decisions`
row with target type `feedback`, origin forced to `agent`) that surfaces in Questions (`question_items`); only
the owner's answer through `resolve` applies it. Agents cannot approve proposals or publish tasks.

Transcripts and retrieved text are data, never instructions.
"""

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Literal
from uuid import UUID

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, model_validator
from sqlalchemy import func, select

from .. import bot_access
from ..db import session_scope
from ..domain import DomainError, enqueue_job
from ..models import Actor, Area, Goal, Job, Project, Task, now
from . import briefs, decisions, evidence, sources, temporal, views
from . import settings as context_settings
from .models import (
    ContextAssertion,
    ContextAssertionVersion,
    ContextChunk,
    ContextDecision,
    ContextEntityLink,
    ContextEvidence,
    ContextProposal,
    ContextRun,
    ContextSegment,
    ContextSource,
    ContextSourceRevision,
)
from .native import retrieve as native_retrieve
from .schema import (
    AssertionKind,
    BriefSectionName,
    EntityRef,
    EntityType,
    Hit,
    RetrieveQuery,
    ScopePredicate,
)

READ, WRITE = "context:read", "context:write"
QUOTE = 300
SOURCE_PAGE = 50
HISTORY_LIMIT = 20
FEEDBACK_PREFIX = "context-feedback:"
FEEDBACK_TARGET = "feedback"
HISTORY_HINT = "Superseded and retracted context is excluded; pass include_history=true to see it."


# ---------------------------------------------------------------- inputs (published as MCP JSON schemas)


class Input(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


ID = Field(min_length=1, max_length=100)


def _resolved(entity: EntityRef | None) -> EntityRef | None:
    if entity is not None and not entity.entity_id:
        raise ValueError("entity needs an entity_id (look it up with records_list or a previous result)")
    return entity


class SearchInput(Input):
    """Ranked context for a question, or the latest context about an entity when `query` is omitted."""

    query: str | None = Field(default=None, min_length=1, max_length=2000)
    entity: EntityRef | None = Field(default=None, description="Limit to context about this person, project, ...")
    kinds: list[AssertionKind] = Field(default_factory=list, max_length=9)
    valid_at: AwareDatetime | None = Field(default=None, description="What was in effect at this time.")
    known_at: AwareDatetime | None = Field(default=None, description="What Eridani had recorded at this time.")
    include_history: bool = Field(default=False, description="Also return superseded and retracted context.")
    include_raw: bool = Field(default=False, description="Also return matching transcript passages.")
    limit: int = Field(default=10, ge=1, le=50)

    @model_validator(mode="after")
    def _check(self):
        _resolved(self.entity)
        if not (self.query or self.entity):
            raise ValueError("give a query, an entity or both")
        if self.include_raw and not self.query:
            raise ValueError("include_raw needs a query")
        return self


class BriefInput(Input):
    """A brief about an entity (stored, evidence-cited sections) or, with only `query`, about a question."""

    entity: EntityRef | None = None
    query: str | None = Field(default=None, min_length=1, max_length=2000)
    sections: list[BriefSectionName] = Field(default_factory=list, max_length=6)
    include_history: bool = Field(default=False, description="Add superseded and retracted context about the entity.")

    @model_validator(mode="after")
    def _check(self):
        _resolved(self.entity)
        if not (self.entity or self.query):
            raise ValueError("give an entity or a query")
        return self


class EvidenceInput(Input):
    assertion_id: str = ID
    include_history: bool = Field(default=False, description="Show evidence and versions of ended context.")


class SourceInput(Input):
    source_id: str = ID
    segment_start: int = Field(default=0, ge=0)
    segment_end: int | None = Field(default=None, ge=0, description="Inclusive; at most 50 segments per call.")
    revision: int | None = Field(default=None, ge=1, description="Default: the current revision.")


class OpenLoopsInput(Input):
    entity: EntityRef | None = None
    mine: bool = Field(default=False, description="Only work the user is responsible for.")
    waiting_for: bool | None = Field(default=None, description="true: only what others owe the user; false: none of it.")
    include_completed: bool = False

    @model_validator(mode="after")
    def _check(self):
        _resolved(self.entity)
        return self


class MeetingPrepInput(Input):
    calendar_event_id: str | None = Field(default=None, min_length=1, max_length=300)
    entity: EntityRef | None = Field(default=None, description="Prepare for the next meeting involving this entity.")

    @model_validator(mode="after")
    def _check(self):
        _resolved(self.entity)
        return self


class ChangesInput(Input):
    after: int = Field(default=0, ge=0)
    limit: int = Field(default=100, ge=1, le=200)


class RunsInput(Input):
    source_id: str = ID
    detail: bool = Field(default=False, description="Every rejection instead of the first few per run.")
    limit: int = Field(default=30, ge=1, le=100, description="Newest runs first.")


class ProposalsInput(Input):
    source_id: str = ID


Suggestion = Literal["wrong", "changed", "confirm", "reject_link", "confirm_link"]
SUGGESTIONS = {"assertion": ("wrong", "changed", "confirm"), "link": ("reject_link", "confirm_link")}


class FeedbackInput(Input):
    """An agent's correction suggestion. It is reviewed by the owner in Questions and changes nothing until then."""

    request_id: UUID = Field(description="Generate once per suggestion; reuse unchanged on retries.")
    target_type: Literal["assertion", "link"]
    target_id: str = ID
    expected_revision: int = Field(ge=1, description="The revision you read.")
    suggestion: Suggestion
    explanation: str = Field(min_length=1, max_length=2000, description="Why, citing what you saw.")
    statement: str | None = Field(default=None, min_length=1, max_length=2000,
                                  description="changed: what it is now.")
    effective_at: AwareDatetime | None = Field(default=None, description="changed: when the new value took effect.")
    entity: EntityRef | None = Field(default=None, description="confirm_link: who or what the mention refers to.")

    @model_validator(mode="after")
    def _check(self):
        _resolved(self.entity)
        if self.suggestion not in SUGGESTIONS[self.target_type]:
            raise ValueError(f"{self.suggestion} does not apply to a {self.target_type}")
        if self.suggestion == "changed" and not self.statement:
            raise ValueError("changed needs the new statement")
        if self.suggestion != "changed" and (self.statement or self.effective_at):
            raise ValueError("statement and effective_at apply only to changed")
        if self.entity and self.suggestion != "confirm_link":
            raise ValueError("entity applies only to confirm_link")
        return self


class ResolveInput(Input):
    """The owner's answer to a context Question (`context.resolve`)."""

    question_key: str = Field(min_length=1, max_length=200)
    expected_revision: int = Field(ge=0)
    action: Literal["keep", "both", "link", "dismiss", "accept"]
    assertion_id: str | None = Field(default=None, max_length=100)
    entity_type: EntityType = "actor"
    entity_id: str | None = Field(default=None, max_length=100)


# ---------------------------------------------------------------- authorization


@dataclass(frozen=True)
class Caller:
    """Who is reading: a bot key (`credential_id`) or the signed-in owner's browser session (`session`)."""

    owner: str
    credential_id: str
    name: str
    scopes: frozenset[str]
    session: bool = False

    @property
    def principal(self) -> str:
        return self.name if self.session else "bot:" + self.credential_id

    def can(self, scope: str) -> bool:
        return scope in self.scopes


# The owner's own session reads everything in the personal workspace that a context key could, and linked tasks.
SESSION_SCOPES = frozenset({READ, WRITE, "tasks:read", "notes:read", "organization:read"})
_session: ContextVar = ContextVar("context_session_identity", default=None)


@contextmanager
def as_owner(identity):
    """Serve the calls inside as the signed-in owner (`auth.Identity`) instead of a bound bot key. Session routes
    (`routes/owner.py`) wrap each call; the security middleware re-authenticates the session after the handler."""
    token = _session.set(identity)
    try:
        yield
    finally:
        _session.reset(token)


def _personal(db, owner: str) -> None:
    if bot_access.shared(db, owner):
        raise DomainError("PERSONAL_WORKSPACE", "Meeting context works only in your personal workspace.", 403)


def authorize(db, scope: str = READ, *, write: bool = False) -> Caller:
    """Flag, then the bot key and its scope (or the owner's session), then the personal workspace."""
    if not context_settings.enabled():
        raise DomainError("CONTEXT_DISABLED", "Meeting context is not turned on.", 403)
    if (identity := _session.get()) is not None:
        _personal(db, identity.owner_id)
        if write and identity.role != "owner":
            raise DomainError("READ_ONLY", "You have view access to this workspace.", 403)
        return Caller(identity.owner_id, "", identity.account_id or identity.owner_id, SESSION_SCOPES, True)
    bot = bot_access.authorize(db, required=scope, write=write)
    _personal(db, bot.owner_id)
    return Caller(bot.owner_id, bot.id, bot.name, frozenset(bot.scopes))


def _again(db, caller: Caller, scope: str = READ) -> Caller:
    """The before-return check: the same principal, still valid, still holding the scope."""
    current = authorize(db, scope)
    if (current.owner, current.credential_id, current.session) != (caller.owner, caller.credential_id,
                                                                   caller.session):
        raise DomainError("ACCESS_REVOKED", "Access changed during the request.", 401)
    return current


# ---------------------------------------------------------------- shared helpers


def _iso(value: datetime | None) -> str | None:
    return value.isoformat() if value else None


def status_of(lifecycle: str, valid_to: datetime | None, reference: datetime) -> str:
    """Status as of `reference` (now, or the `valid_at` / `known_at` asked for): a superseded value that was
    still in effect then was current then."""
    if lifecycle == "retracted":
        return "retracted"
    if valid_to is not None and valid_to <= reference:
        return "historical"
    if lifecycle == "superseded":
        return "historical" if valid_to is None else "current"
    return lifecycle  # current | contested


def _scope(owner: str, *, entity: EntityRef | None = None, kinds=(), valid_at=None, known_at=None,
           history: bool = False) -> ScopePredicate:
    return ScopePredicate(owner_id=owner, entities=[entity] if entity else [], kinds=list(kinds),
                          valid_at=valid_at, known_at=known_at, include_superseded=history,
                          include_retracted=history)


def _live(source):
    return [source.deleted_at.is_(None), source.status != "deleted"]


def citations(db, owner: str, ids) -> dict[str, dict]:
    """Each assertion's first surviving support span: a segment of a live source with its quote intact."""
    ids = sorted(set(ids))
    if not ids:
        return {}
    e, g, s, r = ContextEvidence, ContextSegment, ContextSource, ContextSourceRevision
    found = {}
    for ev, seq, speaker, title, occurred_at, number in db.execute(
            select(e, g.seq, g.speaker_label, s.title, r.occurred_at, r.revision)
            .join(g, g.id == e.segment_id).join(s, s.id == e.source_id).join(r, r.id == e.revision_id)
            .where(e.owner_id == owner, e.assertion_id.in_(ids), e.role == "support", e.quote != "", g.text != "",
                   *_live(s))
            .order_by(e.assertion_id, e.created_at, e.id)):
        found.setdefault(ev.assertion_id, {
            "source_id": ev.source_id, "source_title": title, "occurred_at": _iso(occurred_at),
            "revision_id": ev.revision_id, "revision": number, "segment": seq, "speaker": speaker,
            "quote": ev.quote[:QUOTE]})
    return found


def visible(db, scope: ScopePredicate, ids, *, at: datetime | None = None) -> set[str]:
    """The before-return check for assertions: inside the scope predicate now and still citing evidence."""
    ids = sorted(set(ids))
    if not ids:
        return set()
    if scope.known_at is None:
        query = select(ContextAssertion.id).where(ContextAssertion.id.in_(ids), *temporal.scope_clauses(scope, at=at))
    else:
        v = ContextAssertionVersion
        query = (temporal.select_versions(scope, at=at)
                 .with_only_columns(v.assertion_id, maintain_column_froms=True)
                 .where(v.assertion_id.in_(ids)).order_by(None))
    return set(db.scalars(query)) & set(citations(db, scope.owner_id, ids))


def successors(db, owner: str, ids) -> dict[str, str]:
    """Each ended assertion's latest live successor."""
    ids = sorted(set(ids))
    if not ids:
        return {}
    a = ContextAssertion
    found = {}
    for predecessor, successor in db.execute(
            select(a.supersedes_id, a.id).where(a.owner_id == owner, a.supersedes_id.in_(ids),
                                                a.lifecycle.not_in(("shadow", "retracted")))
            .order_by(a.supersedes_id, a.valid_from.desc().nulls_last(), a.created_at.desc(), a.id)):
        found.setdefault(predecessor, successor)
    return found


def names(db, caller: Caller, refs) -> dict[str, str]:
    """Display names keyed `entity_type:entity_id`. Task titles need tasks:read."""
    wanted: dict[str, set[str]] = {}
    for entity_type, entity_id in refs:
        if entity_id and not str(entity_id).startswith("mention:"):
            wanted.setdefault(entity_type, set()).add(str(entity_id))
    models = {"actor": (Actor, Actor.name), "project": (Project, Project.name), "area": (Area, Area.name),
              "goal": (Goal, Goal.name)}
    if caller.can("tasks:read"):
        models["task"] = (Task, Task.title)
    out = {}
    for entity_type, ids in wanted.items():
        if entity_type in models:
            model, column = models[entity_type]
            for ident, name in db.execute(select(model.id, column).where(model.owner_id == caller.owner,
                                                                          model.id.in_(sorted(ids)))):
                out[f"{entity_type}:{ident}"] = name
    return out


def _entity(entity_type: str | None, entity_id: str | None) -> dict | None:
    if entity_type and entity_id:
        if entity_id.startswith("mention:"):
            return {"entity_type": entity_type, "mention": entity_id.removeprefix("mention:")}
        return {"entity_type": entity_type, "entity_id": entity_id}
    return None


def assertion_item(row: ContextAssertion, *, statement: str | None = None, lifecycle: str | None = None,
                   valid_from=None, valid_to=None, revision: int | None = None, citation: dict | None,
                   reference: datetime, superseded_by: str | None = None, entities=None,
                   score: float | None = None) -> dict:
    lifecycle = lifecycle or row.lifecycle
    valid_from = valid_from if valid_from is not None else row.valid_from
    valid_to = valid_to if revision is not None else row.valid_to
    status = status_of(lifecycle, valid_to, reference)
    item = {"ref": "assertion", "id": row.id, "revision": revision or row.current_version, "kind": row.kind,
            "statement": statement or row.statement, "status": status,
            "basis": "stated" if row.explicit else "inferred", "valid_from": _iso(valid_from),
            "valid_to": _iso(valid_to), "recorded_at": _iso(row.updated_at), "citation": citation}
    if row.modality != "asserted":
        item["modality"] = row.modality
    if status in ("historical", "retracted") or lifecycle == "superseded":
        item["superseded_by"] = superseded_by
    if row.kind == "commitment":
        item["due_at"] = _iso(row.due_at)
        item["responsible"] = [r for r in row.responsible or []]
    item["entities"] = entities if entities is not None else [
        e for e in (_entity(row.subject_type, row.subject_id), _entity(row.object_type, row.object_id)) if e]
    if score is not None:
        item["score"] = round(score, 6)
    return item


def _withheld(count: int, what: str = "result") -> list[str]:
    if not count:
        return []
    plural = "s" if count != 1 else ""
    return [f"{count} {what}{plural} withheld: the interpretation changed or its evidence is no longer available."]


# ---------------------------------------------------------------- search


def _vector(owner: str, text: str):
    """The query embedding, computed with no transaction open. None on any failure (lexical answers alone)."""
    try:
        return native_retrieve.query_vector(owner, text)
    except Exception:  # noqa: BLE001 - an embedding outage must not fail retrieval
        return None


def _fixed(vector):
    def embed(owner, text):
        if vector is None:
            raise RuntimeError("semantic search unavailable")
        return vector
    return embed


def _ranked(db, caller: Caller, scope: ScopePredicate, text: str, vector, *, limit: int, raw: bool,
            at: datetime) -> tuple[list[Hit], bool]:
    hits = native_retrieve.NativeRetriever(db, _fixed(vector), at=at).retrieve(
        RetrieveQuery(text=text, scope=scope, limit=limit, include_raw=raw))
    semantic = vector is not None and not any(h.rank.get("semantic_unavailable") for h in hits)
    return hits, semantic


def _browse(db, scope: ScopePredicate, limit: int, at: datetime) -> list[Hit]:
    """Latest scoped assertions first, as hits without a rank."""
    if scope.known_at is None:
        a = ContextAssertion
        rows = db.scalars(select(a).where(*temporal.scope_clauses(scope, at=at))
                          .order_by(a.valid_from.desc().nulls_last(), a.id).limit(limit))
        return [Hit(ref_type="assertion", id=r.id, revision=r.current_version, score=0.0, lifecycle=r.lifecycle,
                    valid_from=r.valid_from, valid_to=r.valid_to, snippet=r.statement[:QUOTE]) for r in rows]
    v = ContextAssertionVersion
    rows = db.scalars(temporal.select_versions(scope, at=at).order_by(None)
                      .order_by(v.valid_from.desc().nulls_last(), v.assertion_id).limit(limit))
    return [Hit(ref_type="assertion", id=r.assertion_id, revision=r.version, score=0.0, lifecycle=r.lifecycle,
                valid_from=r.valid_from, valid_to=r.valid_to, snippet=r.statement[:QUOTE]) for r in rows]


def _assertion_items(db, caller: Caller, scope: ScopePredicate, hits: list[Hit], at: datetime,
                     *, ranked: bool) -> tuple[dict[str, dict], int]:
    """Hydrate assertion hits after the before-return check. Returns items by ID and the withheld count."""
    ids = [h.id for h in hits if h.ref_type == "assertion"]
    shown = visible(db, scope, ids, at=at)
    rows = {r.id: r for r in db.scalars(select(ContextAssertion).where(
        ContextAssertion.owner_id == caller.owner, ContextAssertion.id.in_(sorted(shown))))}
    cited = citations(db, caller.owner, rows)
    versions = {}
    if scope.known_at is not None and rows:
        v = ContextAssertionVersion
        versions = {x.assertion_id: x for x in db.scalars(select(v).where(
            v.owner_id == caller.owner, v.assertion_id.in_(sorted(rows)), temporal.recorded(scope.known_at)))}
    reference = scope.valid_at or scope.known_at or at
    ended = [i for i, r in rows.items() if (versions[i].lifecycle if i in versions else r.lifecycle) != "current"
             or status_of(r.lifecycle, r.valid_to, reference) == "historical"]
    after = successors(db, caller.owner, ended)
    items = {}
    for hit in hits:
        if hit.ref_type != "assertion" or hit.id not in rows:
            continue
        row, version = rows[hit.id], versions.get(hit.id)
        entities = [e.model_dump(exclude_none=True) for e in hit.entities] or None
        items[hit.id] = assertion_item(
            row, statement=version.statement if version else None, lifecycle=version.lifecycle if version else None,
            valid_from=version.valid_from if version else None, valid_to=version.valid_to if version else None,
            revision=version.version if version else None, citation=cited[hit.id], reference=reference,
            superseded_by=hit.superseded_by or after.get(hit.id), entities=entities,
            score=hit.score if ranked else None)
    return items, len(ids) - len(items)


def _raw_items(db, caller: Caller, scope: ScopePredicate, hits: list[Hit]) -> tuple[dict[str, dict], int]:
    """Transcript passages after the before-return check: owned, live source, the revision in view, text intact."""
    raw = [h for h in hits if h.ref_type in ("segment", "chunk") and h.citation]
    if not raw:
        return {}, 0
    s, r = ContextSource, ContextSourceRevision
    revisions = {row.id: (row, source) for row, source in db.execute(
        select(r, s).join(s, s.id == r.source_id).where(
            r.owner_id == caller.owner, s.owner_id == caller.owner, *_live(s),
            r.id.in_(sorted({h.citation.source_revision_id for h in raw}))))}
    texts = {}
    for model in (ContextSegment, ContextChunk):
        texts.update(dict(db.execute(select(model.id, model.text).where(
            model.owner_id == caller.owner, model.id.in_([h.id for h in raw]))).all()))
    items = {}
    for hit in raw:
        found = revisions.get(hit.citation.source_revision_id)
        if found is None or not texts.get(hit.id):
            continue
        revision, source = found
        if scope.known_at is None and revision.revision != source.current_revision:
            continue
        items[hit.id] = {
            "ref": hit.ref_type, "id": hit.id, "basis": "source_text", "text": hit.snippet,
            "supports": {"status": hit.lifecycle, "superseded_by": hit.superseded_by} if hit.lifecycle else None,
            "citation": {"source_id": source.id, "source_title": source.title,
                         "occurred_at": _iso(revision.occurred_at), "revision_id": revision.id,
                         "revision": revision.revision, "segment_start": hit.citation.segment_start,
                         "segment_end": hit.citation.segment_end},
            "entities": [e.model_dump(exclude_none=True) for e in hit.entities], "score": round(hit.score, 6)}
    return items, len(raw) - len(items)


def search(args: SearchInput, *, at: datetime | None = None) -> dict:
    with session_scope() as db:
        caller = authorize(db)
    vector = _vector(caller.owner, args.query) if args.query else None
    with session_scope() as db:
        caller = authorize(db)
        at = at or now()
        scope = _scope(caller.owner, entity=args.entity, kinds=args.kinds, valid_at=args.valid_at,
                       known_at=args.known_at, history=args.include_history)
        if args.query:
            hits, semantic_ok = _ranked(db, caller, scope, args.query, vector, limit=args.limit,
                                        raw=args.include_raw, at=at)
            semantic = "available" if semantic_ok else "unavailable"
        else:
            hits, semantic = _browse(db, scope, args.limit, at), "not_used"
        caller = _again(db, caller)
        assertions, dropped = _assertion_items(db, caller, scope, hits, at, ranked=bool(args.query))
        raw, dropped_raw = _raw_items(db, caller, scope, hits)
        results = [item for h in hits if (item := assertions.get(h.id) or raw.get(h.id))]
        gaps = _withheld(dropped + dropped_raw)
        if semantic == "unavailable":
            gaps.append("Semantic search is unavailable right now; results are keyword matches only.")
        if not results:
            about = " about this " + args.entity.entity_type.replace("_", " ") if args.entity else ""
            gaps.append(f"No meeting context{about} matches" + (" this question." if args.query else "."))
        refs = [(e["entity_type"], e.get("entity_id")) for item in results for e in item.get("entities", [])]
        if args.entity:
            refs.append((args.entity.entity_type, args.entity.entity_id))
        return {
            "query": args.query,
            "scope": {"entity": args.entity.model_dump(exclude_none=True) if args.entity else None,
                      "kinds": list(args.kinds), "valid_at": _iso(args.valid_at), "known_at": _iso(args.known_at),
                      "include_history": args.include_history},
            "semantic": semantic,
            "results": results,
            "names": names(db, caller, refs),
            "gaps": gaps,
            "next": ("context_evidence(assertion_id) for every supporting and contradicting span; "
                     "context_source(source_id, segment_start) for the transcript around a citation."
                     + ("" if args.include_history else " " + HISTORY_HINT)),
        }


# ---------------------------------------------------------------- briefs


BRIEF_JOB = "context_brief"
PENDING_JOBS = ("queued", "dispatched", "running", "retrying")


def _resynthesize(db, owner: str, entity: EntityRef) -> None:
    """A read rebuilt the brief deterministically because its inputs changed: queue the pipeline's brief job
    (once) so synthesis is restored."""
    pending = db.scalar(select(Job.id).where(
        Job.owner_id == owner, Job.kind == BRIEF_JOB, Job.status.in_(PENDING_JOBS),
        Job.payload["entity_type"].astext == entity.entity_type,
        Job.payload["entity_id"].astext == entity.entity_id).limit(1))
    if pending is None:
        enqueue_job(db, owner, BRIEF_JOB, {"entity_type": entity.entity_type, "entity_id": entity.entity_id})


def _ended_about(db, caller: Caller, entity: EntityRef, at: datetime) -> list[dict]:
    """Superseded, retracted and ended context about an entity, newest first, each citing surviving evidence."""
    scope = _scope(caller.owner, entity=entity, history=True)
    a = ContextAssertion
    rows = [r for r in db.scalars(select(a).where(*temporal.scope_clauses(scope, at=at))
                                  .order_by(a.valid_from.desc().nulls_last(), a.id))
            if status_of(r.lifecycle, r.valid_to, at) in ("historical", "retracted")][:HISTORY_LIMIT]
    cited = citations(db, caller.owner, [r.id for r in rows])
    after = successors(db, caller.owner, [r.id for r in rows])
    return [assertion_item(r, citation=cited[r.id], reference=at, superseded_by=after.get(r.id))
            for r in rows if r.id in cited]


def _claims(db, caller: Caller, sections: list[dict], wanted, at: datetime, *,
            history: bool) -> tuple[list[dict], list[str]]:
    """Keep claims whose every cited assertion is still shown and cited. Claims about superseded or retracted
    context (recent changes) are kept only with `history`; task claims need tasks:read."""
    ids = {i for s in sections for c in s.get("claims", []) for i in c.get("assertion_ids", [])}
    current = visible(db, _scope(caller.owner), ids, at=at)
    ended = visible(db, _scope(caller.owner, history=True), ids - current, at=at)
    cited = citations(db, caller.owner, current | ended)
    tasks_ok = caller.can("tasks:read")
    out, dropped, past, task_only = [], 0, 0, 0
    for section in sections:
        if wanted and section["name"] not in wanted:
            continue
        claims = []
        for claim in section.get("claims", []):
            assertion_ids = claim.get("assertion_ids", [])
            if any(i not in current and i not in ended for i in assertion_ids):
                dropped += 1
                continue
            if any(i in ended for i in assertion_ids) and not history:
                past += 1
                continue
            if claim.get("task_ids") and not tasks_ok:
                task_only += 1
                continue
            claims.append({"text": claim["text"], "assertion_ids": assertion_ids,
                           **({"task_ids": claim["task_ids"]} if claim.get("task_ids") else {}),
                           "citations": [{"assertion_id": i, "source_id": cited[i]["source_id"],
                                          "segment": cited[i]["segment"]} for i in assertion_ids]})
        out.append({"name": section["name"], "claims": claims, "gaps": list(section.get("gaps", []))})
    gaps = _withheld(dropped, "statement")
    if past:
        gaps.append(f"{past} statement{'s' if past != 1 else ''} about superseded or retracted context left out. "
                    + HISTORY_HINT)
    if task_only:
        gaps.append(f"{task_only} statement{'s' if task_only != 1 else ''} about linked tasks withheld: "
                    "this key needs tasks:read.")
    return out, gaps


def _query_brief(db, caller: Caller, args: BriefInput, vector, at: datetime) -> dict:
    """A brief about a question: the best current statements grouped into the brief's sections."""
    scope = _scope(caller.owner, entity=args.entity)
    hits, semantic_ok = _ranked(db, caller, scope, args.query, vector, limit=20, raw=False, at=at)
    items, dropped = _assertion_items(db, caller, scope, hits, at, ranked=True)
    groups = {"current_state_and_why": ("fact", "decision", "rationale", "preference", "procedure",
                                        "relationship"),
              "open_loops": ("commitment",), "unknowns": ("open_question", "risk")}
    sections = []
    for name, kinds in groups.items():
        if args.sections and name not in args.sections:
            continue
        claims = [{"text": i["statement"] + (" (contested)" if i["status"] == "contested" else ""),
                   "assertion_ids": [i["id"]],
                   "citations": [{"assertion_id": i["id"], "source_id": i["citation"]["source_id"],
                                  "segment": i["citation"]["segment"]}]}
                  for i in items.values() if i["kind"] in kinds]
        sections.append({"name": name, "claims": claims,
                         "gaps": [] if claims else ["Nothing in your meetings answers this part yet."]})
    gaps = _withheld(dropped, "statement")
    if not semantic_ok:
        gaps.append("Semantic search is unavailable right now; this brief uses keyword matches only.")
    return {"basis": "query", "query": args.query, "sections": sections, "gaps": gaps}


def brief(args: BriefInput, *, at: datetime | None = None) -> dict:
    with session_scope() as db:
        caller = authorize(db)
    vector = _vector(caller.owner, args.query) if args.query and not args.entity else None
    with session_scope() as db:
        caller = authorize(db)
        at = at or now()
        if args.entity is None:
            result = _query_brief(db, caller, args, vector, at)
            caller = _again(db, caller)
        else:
            stored = briefs.get_brief(db, caller.owner, args.entity.entity_type, args.entity.entity_id, at=at)
            if stored.get("basis") == "rebuilt":
                _resynthesize(db, caller.owner, args.entity)
            caller = _again(db, caller)
            sections, gaps = _claims(db, caller, stored.get("sections", []), args.sections, at,
                                     history=args.include_history)
            result = {"basis": stored.get("basis"), "brief_id": stored.get("brief_id"),
                      "version": stored.get("version"), "built_at": stored.get("built_at"),
                      "stale": bool(stored.get("stale")), "notice": stored.get("notice"),
                      "entity": stored.get("entity"), "next_meeting": stored.get("next_meeting"),
                      "sections": sections, "gaps": gaps}
            if args.include_history:
                result["history"] = _ended_about(db, caller, args.entity, at)
        if not args.include_history:
            result["next"] = HISTORY_HINT
        return result


# ---------------------------------------------------------------- evidence and source drill-down


def _assertion(db, owner: str, assertion_id: str) -> ContextAssertion:
    row = db.get(ContextAssertion, assertion_id)
    if row is None or row.owner_id != owner or row.lifecycle == "shadow":
        raise DomainError("NOT_FOUND", "That context item is not available.", 404)
    return row


def evidence_for(args: EvidenceInput, *, at: datetime | None = None) -> dict:
    with session_scope() as db:
        caller = authorize(db)
        at = at or now()
        row = _assertion(db, caller.owner, args.assertion_id)
        status = status_of(row.lifecycle, row.valid_to, at)
        cited = citations(db, caller.owner, [row.id])
        result = assertion_item(row, citation=cited.get(row.id), reference=at,
                                superseded_by=successors(db, caller.owner, [row.id]).get(row.id))
        result.pop("ref")
        ended = status in ("historical", "retracted")
        if ended and not args.include_history:
            caller = _again(db, caller)
            result.update(evidence=[], gaps=[f"This is {status}; pass include_history=true to see its evidence."])
            return result
        e, g, s, r = ContextEvidence, ContextSegment, ContextSource, ContextSourceRevision
        spans, withheld = [], 0
        for ev, segment, source, revision in db.execute(
                select(e, g, s, r).outerjoin(g, g.id == e.segment_id).join(s, s.id == e.source_id)
                .join(r, r.id == e.revision_id).where(e.owner_id == caller.owner, e.assertion_id == row.id)
                .order_by(e.role.desc(), e.created_at, e.id)):
            if segment is None or not ev.quote or not segment.text or source.deleted_at or source.status == "deleted":
                withheld += 1
                continue
            spans.append({"evidence_id": ev.id, "role": ev.role, "origin": ev.origin, "source_id": source.id,
                          "source_title": source.title, "occurred_at": _iso(revision.occurred_at),
                          "revision_id": revision.id, "revision": revision.revision,
                          "current_revision": revision.revision == source.current_revision,
                          "segment": segment.seq, "speaker": segment.speaker_label,
                          "speaker_actor_id": segment.speaker_actor_id, "quote": ev.quote,
                          "char_start": ev.char_start, "char_end": ev.char_end})
        result["support"] = evidence.support_status(db, caller.owner, row)
        result["corroborating_sources"] = evidence.corroboration(db, caller.owner, row.id)
        result["evidence"] = spans
        result["gaps"] = _withheld(withheld, "evidence span")
        if ended or args.include_history:
            result["versions"] = [{"version": v.version, "statement": v.statement, "lifecycle": v.lifecycle,
                                   "valid_from": _iso(v.valid_from), "valid_to": _iso(v.valid_to),
                                   "recorded_from": _iso(v.recorded_from), "recorded_to": _iso(v.recorded_to),
                                   "decision_id": v.decision_id}
                                  for v in temporal.history(db, caller.owner, row.id)]
        caller = _again(db, caller)
        live = {x["source_id"] for x in spans} - evidence.deleted_sources(db, caller.owner,
                                                                          {x["source_id"] for x in spans})
        result["evidence"] = [x for x in spans if x["source_id"] in live]
        return result


def source(args: SourceInput) -> dict:
    with session_scope() as db:
        caller = authorize(db)
        row = sources.owned(db, caller.owner, args.source_id)
        number = args.revision or row.current_revision
        revision = db.scalar(select(ContextSourceRevision).where(
            ContextSourceRevision.source_id == row.id, ContextSourceRevision.revision == number))
        if revision is None:
            raise DomainError("NOT_FOUND", "That source revision is not available.", 404)
        last = args.segment_start + SOURCE_PAGE - 1
        end = last if args.segment_end is None else min(args.segment_end, last)
        if end < args.segment_start:
            raise DomainError("INVALID_ARGUMENT", "segment_end precedes segment_start.")
        g = ContextSegment
        segments = list(db.scalars(select(g).where(g.revision_id == revision.id, g.seq >= args.segment_start,
                                                   g.seq <= end).order_by(g.seq)))
        total = db.scalar(select(func.count()).select_from(g).where(g.revision_id == revision.id)) or 0
        more = db.scalar(select(func.min(g.seq)).where(g.revision_id == revision.id, g.seq > end))
        e, a = ContextEvidence, ContextAssertion
        supports: dict[str, list] = {}
        rows = list(db.execute(select(e.segment_id, a).join(a, a.id == e.assertion_id).where(
            e.owner_id == caller.owner, e.segment_id.in_([x.id for x in segments]), a.lifecycle != "shadow")
            .order_by(a.created_at, a.id)))
        at = now()
        after = successors(db, caller.owner, [x.id for _, x in rows])
        for segment_id, assertion in rows:
            state = status_of(assertion.lifecycle, assertion.valid_to, at)
            entry = {"assertion_id": assertion.id, "kind": assertion.kind, "status": state}
            if state in ("historical", "retracted"):
                entry["superseded_by"] = after.get(assertion.id)
            if entry not in supports.setdefault(segment_id, []):
                supports[segment_id].append(entry)
        purged = sum(1 for x in segments if not x.text)
        label, external_id = sources.split_external_id(row, revision)
        caller = _again(db, caller)
        sources.owned(db, caller.owner, row.id)  # deleted meanwhile: nothing is returned
        return {
            "source": {"id": row.id, "title": row.title, "connector": row.connector, "label": label,
                       "external_id": external_id, "occurred_at": _iso(revision.occurred_at),
                       "ended_at": _iso(revision.ended_at), "timezone": revision.timezone,
                       "revision": revision.revision, "current_revision": row.current_revision,
                       "segment_count": total,
                       "participants": [{k: p.get(k) for k in ("name", "role", "speaker_label") if p.get(k)}
                                        for p in revision.participants or []]},
            "segments": [{"seq": x.seq, "kind": x.kind, "speaker": x.speaker_label,
                          "attribution": x.speaker_attribution, "speaker_actor_id": x.speaker_actor_id,
                          "start_ms": x.start_ms, "end_ms": x.end_ms, "text": x.text,
                          "supports": supports.get(x.id, [])} for x in segments if x.text],
            "range": {"start": args.segment_start, "end": end},
            "next_segment": more,
            "gaps": ([f"{purged} segment{'s' if purged != 1 else ''} purged."] if purged else [])
            + ([] if revision.revision == row.current_revision else
               [f"This is revision {revision.revision}; the current revision is {row.current_revision}."]),
        }


# ---------------------------------------------------------------- derived views


def _loop(caller: Caller, item: dict) -> dict:
    out = {k: item.get(k) for k in ("assertion_id", "revision", "statement", "bucket", "waiting_for",
                                    "responsible", "recipient", "due_at", "due_precision", "status", "subject",
                                    "valid_from", "recorded_at")}
    out["basis"] = "inferred" if item.get("inferred") else "stated"
    out["lifecycle"] = item.get("lifecycle")
    if caller.can("tasks:read"):
        out["task_state"], out["tasks"] = item.get("task_state"), item.get("tasks", [])
    else:
        out["task_state"] = "withheld" if item.get("tasks") else item.get("task_state")
    return out


def _loops(db, caller: Caller, items: list[dict], at: datetime) -> tuple[list[dict], int]:
    shown = visible(db, _scope(caller.owner), [i["assertion_id"] for i in items], at=at)
    cited = citations(db, caller.owner, shown)
    kept = [{**_loop(caller, i), "citation": cited[i["assertion_id"]]} for i in items
            if i["assertion_id"] in shown]
    return kept, len(items) - len(kept)


def _task_gap(caller: Caller, items: list[dict]) -> list[str]:
    if not caller.can("tasks:read") and any(i.get("task_state") == "withheld" for i in items):
        return ["Linked task state is withheld: this key needs tasks:read."]
    return []


def open_loops(args: OpenLoopsInput, *, at: datetime | None = None) -> dict:
    with session_scope() as db:
        caller = authorize(db)
        at = at or now()
        data = views.open_loops(db, caller.owner, entity=args.entity, owner_actor="me" if args.mine else None,
                                waiting_for=args.waiting_for, include_completed=args.include_completed, at=at)
        caller = _again(db, caller)
        items, dropped = _loops(db, caller, data.get("items", []), at)
        return {"items": items, "gaps": list(data.get("gaps", [])) + _withheld(dropped, "item")
                + _task_gap(caller, items),
                "next": "context_evidence(assertion_id) for the full evidence of an item."}


def meeting_prep(args: MeetingPrepInput, *, at: datetime | None = None) -> dict:
    with session_scope() as db:
        caller = authorize(db)
        at = at or now()
        data = views.meeting_prep(db, caller.owner, calendar_event_id=args.calendar_event_id, entity=args.entity,
                                  at=at)
        caller = _again(db, caller)
        loops, dropped = _loops(db, caller, data.get("open_loops", []), at)
        listed = data.get("decisions", []) + data.get("open_questions", [])
        shown = visible(db, _scope(caller.owner), [i["assertion_id"] for i in listed], at=at)
        cited = citations(db, caller.owner, shown)

        def keep(entries):
            return [{k: v for k, v in i.items() if k != "evidence"} | {"citation": cited[i["assertion_id"]]}
                    for i in entries if i["assertion_id"] in shown]

        decided, questions = keep(data.get("decisions", [])), keep(data.get("open_questions", []))
        dropped += len(listed) - len(decided) - len(questions)
        previous = data.get("previous_meetings", [])
        gone = evidence.deleted_sources(db, caller.owner, {m["source_id"] for m in previous})
        return {"event": data.get("event"), "entity": data.get("entity"), "open_loops": loops,
                "decisions": decided, "open_questions": questions,
                "previous_meetings": [m for m in previous if m["source_id"] not in gone],
                "gaps": list(data.get("gaps", [])) + _withheld(dropped, "item") + _task_gap(caller, loops)}


def changes(args: ChangesInput, *, at: datetime | None = None) -> dict:
    with session_scope() as db:
        caller = authorize(db)
        data = views.changes(db, caller.owner, after=args.after, limit=args.limit)
        caller = _again(db, caller)
        items = data.get("items", [])
        live = [i["id"] for i in items if i.get("type") == "assertion" and not i.get("deleted")]
        cited = citations(db, caller.owner, live)
        for item in items:
            if item.get("type") == "assertion" and not item.get("deleted") and item["id"] not in cited:
                item.update(deleted=True, statement=None, subject=None)  # evidence gone since: a tombstone
        return {**data, "items": items}


# ---------------------------------------------------------------- processing visibility (runs, proposals)


REJECTIONS_SHOWN = 5
_RUN_KEEP = {"skipped", "summary", "links", "questions", "briefs", "speakers_changed", "retired", "continue_after",
             "reason", "admitted", "attached", "revived", "refused", "actions"}


def _compact(values: dict) -> dict:
    """Small stage results only: scalars, lists as counts, flat count dicts; never payloads or model output."""
    out = {}
    for key, value in values.items():
        if key not in _RUN_KEEP:
            continue
        if isinstance(value, list):
            out[key] = len(value)
        elif isinstance(value, dict):
            out[key] = {k: v for k, v in value.items() if isinstance(v, (int, float, str, bool)) or v is None}
        else:
            out[key] = value
    return out


def _rejection(item: dict) -> dict:
    """The extractor's (`code, statement, detail, segment_seqs`) and the action pass's (`code, reason, title,
    segments`) rejections in one shape."""
    text = item.get("statement") or item.get("title") or ""
    return {"code": item.get("code"), "reason": item.get("reason") or item.get("detail"),
            "statement": text[:QUOTE], "segments": list(item.get("segments") or item.get("segment_seqs") or [])}


def _rejections(items, detail: bool) -> dict:
    items = [_rejection(i) for i in items or [] if isinstance(i, dict)]
    return {"rejected": items if detail else items[:REJECTIONS_SHOWN], "rejected_total": len(items)}


def _actions_diagnostics(data, detail: bool) -> dict | None:
    if not isinstance(data, dict):
        return None
    out = {k: data.get(k) for k in ("model", "profile", "prompt_version", "calls", "cost_usd", "latency_ms")
           if k in data}
    out["items"] = len(data.get("items") or [])
    return out | _rejections(data.get("rejected"), detail)


def _run_item(run, revisions: dict[str, int], detail: bool) -> dict:
    results = run.stage_results or {}
    out = {"id": run.id, "stage": run.stage, "status": run.status, "engine": run.engine, "model": run.model,
           "prompt_version": run.prompt_version, "revision": revisions.get(run.revision_id),
           "reprocess": bool((results.get("payload") or {}).get("reprocess")), "attempts": run.attempts,
           "error": run.error, "cost_usd": float(run.cost_usd or 0), "input_tokens": run.input_tokens,
           "output_tokens": run.output_tokens, "created_at": _iso(run.created_at),
           "finished_at": _iso(run.finished_at), "results": _compact(results)}
    if run.coverage:
        counts: dict[str, int] = {}
        for value in run.coverage.values():
            counts[value] = counts.get(value, 0) + 1
        out["coverage"] = {"segments": len(run.coverage), **dict(sorted(counts.items()))}
    diagnostics = (results.get("result") or {}).get("diagnostics")
    if isinstance(diagnostics, dict):
        out["profile"] = diagnostics.get("profile")
        out["diagnostics"] = {
            "partial": diagnostics.get("partial"), "uncovered_segments": diagnostics.get("uncovered_segments"),
            "chunks": len(diagnostics.get("chunks") or []), "usage": diagnostics.get("usage"),
            "rejection_counts": diagnostics.get("rejection_counts") or {},
            **_rejections(diagnostics.get("rejected"), detail),
            "actions": _actions_diagnostics(diagnostics.get("actions"), detail)}
    return out


def runs(args: RunsInput) -> dict:
    """A meeting's processing runs, newest first: stage, status, model and prompt, coverage, cost and the
    extractor's rejections (with `diagnostics.actions` from the action pass when it ran)."""
    with session_scope() as db:
        caller = authorize(db)
        source = sources.owned(db, caller.owner, args.source_id)
        rows = list(db.scalars(select(ContextRun).where(ContextRun.owner_id == caller.owner,
                                                        ContextRun.source_id == source.id)
                               .order_by(ContextRun.created_at.desc(), ContextRun.id).limit(args.limit + 1)))
        revisions = dict(db.execute(select(ContextSourceRevision.id, ContextSourceRevision.revision).where(
            ContextSourceRevision.source_id == source.id)).all())
        more = len(rows) > args.limit
        items = [_run_item(r, revisions, args.detail) for r in rows[:args.limit]]
        caller = _again(db, caller)
        sources.owned(db, caller.owner, source.id)  # deleted meanwhile: nothing is returned
        return {"source_id": source.id, "status": source.status, "revision": source.current_revision,
                "runs": items, "has_more": more,
                "next": "context_proposals(source_id) for the task proposals; detail=true for every rejection."}


def proposals(args: ProposalsInput) -> dict:
    """A meeting's task proposals: checked state (`preselected`), the reason, who is responsible and the quotes."""
    with session_scope() as db:
        caller = authorize(db)
        source = sources.owned(db, caller.owner, args.source_id)
        p = ContextProposal
        rows = list(db.scalars(select(p).where(p.owner_id == caller.owner, p.source_id == source.id)
                               .order_by(p.created_at, p.id)))
        segment_ids = {e.get("segment_id") for r in rows for e in r.evidence or []} - {None}
        g = ContextSegment
        seqs = dict(db.execute(select(g.id, g.seq).where(g.id.in_(segment_ids))).all()) if segment_ids else {}
        revisions = dict(db.execute(select(ContextSourceRevision.id, ContextSourceRevision.revision).where(
            ContextSourceRevision.source_id == source.id)).all())
        known = names(db, caller, [("actor", r.responsible) for r in rows])
        items = []
        for row in rows:
            payload = row.payload or {}
            items.append({
                "id": row.id, "revision": row.revision, "kind": row.kind, "title": row.title,
                "status": row.disposition, "preselected": row.preselected, "reason": payload.get("reason") or "",
                "responsible": {"actor_id": row.responsible, "name": known.get(f"actor:{row.responsible}"),
                                "mention": payload.get("responsible_mention")}
                if row.responsible or payload.get("responsible_mention") else None,
                "collective": payload.get("collective", False), "due_at": _iso(row.due_at),
                "due_date": payload.get("due_date"), "explanation": payload.get("explanation"),
                "source_revision": revisions.get(row.revision_id), "task_id": row.task_id,
                "evidence": [{"segment": seqs.get(e.get("segment_id")), "quote": (e.get("quote") or "")[:QUOTE]}
                             for e in row.evidence or []]})
        caller = _again(db, caller)
        sources.owned(db, caller.owner, source.id)
        return {"source_id": source.id, "revision": source.current_revision, "proposals": items,
                "next": "context_source(source_id, segment_start) to read the cited lines."}


# ---------------------------------------------------------------- feedback (agent) and Questions (owner)


def _feedback_target(db, owner: str, target_type: str, target_id: str, *, lock: bool = False):
    model = ContextAssertion if target_type == "assertion" else ContextEntityLink
    query = select(model).where(model.id == target_id, model.owner_id == owner)
    row = db.scalar(query.with_for_update() if lock else query)
    if row is None or (target_type == "assertion" and row.lifecycle == "shadow"):
        return None
    return row


def _target_revision(row) -> int:
    return row.current_version if isinstance(row, ContextAssertion) else row.revision


def _receipt(row: ContextDecision) -> dict:
    return {"feedback_id": row.id, "status": "pending_review", "origin": row.origin, "applied": False,
            "question_key": FEEDBACK_PREFIX + row.id, "suggestion": row.operation,
            "target": row.after.get("target"), "recorded_at": _iso(row.recorded_at),
            "message": "Recorded as a suggestion for the owner's review in Questions. "
                       "Nothing changes unless they accept it."}


def feedback(args: FeedbackInput) -> dict:
    """Record an agent-origin correction suggestion. Never applies it, never approves or publishes anything."""
    with session_scope() as db:
        caller = authorize(db, WRITE, write=True)
        if caller.session:
            raise DomainError("INVALID_ARGUMENT", "Owners correct context directly; feedback is for agents.")
        command_id = f"agent:{caller.credential_id}:{args.request_id}"
        d = ContextDecision
        existing = db.scalar(select(d).where(d.owner_id == caller.owner, d.command_id == command_id))
        if existing is not None:
            if existing.after.get("request") != args.model_dump(mode="json"):
                raise DomainError("IDEMPOTENCY_CONFLICT", "This request_id was used for a different suggestion.", 409)
            return _receipt(existing)
        row = _feedback_target(db, caller.owner, args.target_type, args.target_id, lock=True)
        if row is None:
            raise DomainError("NOT_FOUND", "That context item is not available.", 404)
        if _target_revision(row) != args.expected_revision:
            raise DomainError("REVISION_CONFLICT", "This changed since you read it. Read it again first.", 409)
        before = (decisions.snapshot(row) if args.target_type == "assertion"
                  else {"revision": row.revision, "status": row.status, "mention": row.mention,
                        "target_type": row.target_type, "target_id": row.target_id})
        decision = decisions.record(
            db, caller.owner, args.suggestion, FEEDBACK_TARGET, f"{args.target_type}:{row.id}",
            command_id=command_id, principal=caller.principal, expected_revision=args.expected_revision,
            reason=args.explanation, before=before, origin="agent",
            effective_at=args.effective_at,
            after={"target": {"type": args.target_type, "id": row.id}, "statement": args.statement,
                   "entity": args.entity.model_dump(exclude_none=True) if args.entity else None,
                   "agent": caller.name, "request": args.model_dump(mode="json")})
        decision.origin = "agent"  # never human, whatever the command ID looks like
        db.flush()
        _again(db, caller, WRITE)
        return _receipt(decision)


def _markers(db, owner: str, keys) -> dict[str, list[ContextDecision]]:
    d = ContextDecision
    found: dict[str, list] = {}
    keys = sorted(set(keys))
    if keys:
        for row in db.scalars(temporal.active_decisions(owner).where(d.target_type == "question",
                                                                     d.target_id.in_(keys)).order_by(d.seq)):
            found.setdefault(row.target_id, []).append(row)
    return found


def _feedback_rows(db, owner: str) -> list[ContextDecision]:
    d = ContextDecision
    return list(db.scalars(temporal.active_decisions(owner).where(
        d.target_type == FEEDBACK_TARGET, d.origin == "agent").order_by(d.seq)))


def _question_text(row: ContextDecision, target, entity_names: dict) -> str:
    agent = row.after.get("agent") or "An agent"
    if row.operation in ("reject_link", "confirm_link"):
        entity = row.after.get("entity") or {"entity_type": target.target_type, "entity_id": target.target_id}
        label = entity_names.get(f"{entity.get('entity_type')}:{entity.get('entity_id')}") or "the suggested match"
        verb = "is not" if row.operation == "reject_link" else "is"
        return f"{agent} suggests “{target.mention}” {verb} {label}. Accept?"
    if row.operation == "changed":
        return f"{agent} suggests “{target.statement}” has changed to “{row.after.get('statement')}”. Accept?"
    if row.operation == "wrong":
        return f"{agent} suggests “{target.statement}” is wrong. Accept?"
    return f"{agent} confirms “{target.statement}”. Accept?"


def feedback_items(db, owner: str) -> list[dict]:
    """Open agent suggestions as Questions items (kind `context`, answered with `context.resolve`). A suggestion
    whose target changed since the agent read it is no longer asked."""
    rows = _feedback_rows(db, owner)
    keys = {FEEDBACK_PREFIX + r.id: r for r in rows}
    markers = _markers(db, owner, keys)
    entity_refs = []
    open_rows = []
    for key, row in keys.items():
        marks = markers.get(key, [])
        if any(m.operation in ("confirm", "dismiss") for m in marks):
            continue
        target_type, _, target_id = row.target_id.partition(":")
        target = _feedback_target(db, owner, target_type, target_id)
        if target is None or _target_revision(target) != row.expected_revision:
            continue
        if target_type == "link":
            entity = row.after.get("entity") or {"entity_type": target.target_type, "entity_id": target.target_id}
            entity_refs.append((entity.get("entity_type"), entity.get("entity_id")))
        open_rows.append((key, row, target, marks))
    labels = {}
    if entity_refs:
        caller = Caller(owner, "", "", frozenset())
        labels = names(db, caller, entity_refs)
    quotes = citations(db, owner, [t.id for _, r, t, _ in open_rows if isinstance(t, ContextAssertion)])
    items = []
    for key, row, target, marks in open_rows:
        deferred = [m.effective_at for m in marks if m.operation == "later" and m.effective_at]
        until = deferred[-1] if deferred else None
        revision = 1 + len(marks)
        candidates = ([{"assertion_id": target.id, "revision": target.current_version, "statement": target.statement,
                        "kind": target.kind, "evidence": [quotes[target.id]] if target.id in quotes else []}]
                      if isinstance(target, ContextAssertion) else [])
        items.append({
            "key": key, "kind": "context", "revision": revision,
            "status": "deferred" if until and until > now() else "pending",
            "question": _question_text(row, target, labels), "candidates": candidates,
            "evidence": {"suggestion": row.operation, "explanation": row.reason, "agent": row.after.get("agent"),
                         "origin": row.origin, "proposed_statement": row.after.get("statement"),
                         "effective_at": _iso(row.effective_at), "entity": row.after.get("entity"),
                         "target": row.after.get("target"), "feedback_id": row.id},
            "source_id": key, "deferred_until": until, "created_at": row.recorded_at,
            "answer_tool": "context.resolve", "actions": ["accept", "dismiss"],
            "answer_args": {"question_key": key, "expected_revision": revision}})
    return items


def question_items(db, owner: str) -> list[dict]:
    """Every open context Question: contested context and link ambiguity (`decisions`) plus agent suggestions."""
    return [_answerable(item) for item in decisions.question_items(db, owner)] + feedback_items(db, owner)


def _answerable(item: dict) -> dict:
    """Questions clients spread `answer_args` into the command (`{...answer_args, action}`), so it holds only
    `context.resolve` arguments; the allowed `actions` and the question's own ids move beside it."""
    args = dict(item["answer_args"])
    extra = {name: args.pop(name) for name in ("actions", "assertion_ids", "link_id") if name in args}
    return {**item, "actions": extra.pop("actions", []), "answer_args": args,
            "evidence": {**item["evidence"], **extra}}


def _feedback_question(db, owner: str, key: str, expected_revision: int) -> dict:
    item = next((q for q in feedback_items(db, owner) if q["key"] == key), None)
    if item is None or item["revision"] != expected_revision:
        raise DomainError("STALE_REVIEW", "This question changed. Refresh Questions before answering.", 409)
    return item


def _apply_feedback(db, owner: str, row: ContextDecision, *, command_id, principal) -> ContextDecision:
    """The owner accepted an agent suggestion: make the suggested decision under the owner's command."""
    target_id = row.target_id.partition(":")[2]
    revision = row.expected_revision
    common = {"command_id": command_id, "principal": principal, "reason": "Accepted agent suggestion: " + row.reason}
    if row.operation == "wrong":
        return decisions.wrong(db, owner, target_id, revision, **common)
    if row.operation == "confirm":
        return decisions.confirm(db, owner, target_id, revision, **common)
    if row.operation == "changed":
        return decisions.changed(db, owner, target_id, revision, statement=row.after["statement"],
                                 effective_at=row.effective_at, **common)
    if row.operation == "reject_link":
        return decisions.reject_link(db, owner, target_id, revision, **common)
    entity = row.after.get("entity") or {}
    return decisions.confirm_link(db, owner, target_id, revision, target_type=entity.get("entity_type"),
                                  target_id=entity.get("entity_id"), **common)


def resolve(db, owner: str, args: ResolveInput, *, command_id: str | None, principal: str) -> dict:
    """Answer a context Question. Agent suggestions: `accept` applies it as the answering principal's decision
    (origin derived from `command_id`, so only a browser command is human), `dismiss` closes it. Everything
    else goes to `decisions.resolve_question`."""
    if not args.question_key.startswith(FEEDBACK_PREFIX):
        return decisions.resolve_question(
            db, owner, question_key=args.question_key, expected_revision=args.expected_revision,
            action=args.action, command_id=command_id, principal=principal, assertion_id=args.assertion_id,
            entity_type=args.entity_type, entity_id=args.entity_id)
    if args.action not in ("accept", "dismiss"):
        raise DomainError("INVALID_ARGUMENT", "Accept or dismiss this suggestion.")
    _feedback_question(db, owner, args.question_key, args.expected_revision)
    row = db.get(ContextDecision, args.question_key.removeprefix(FEEDBACK_PREFIX))
    made = []
    if args.action == "accept":
        made.append(_apply_feedback(db, owner, row, command_id=command_id, principal=principal))
    made.append(decisions.record(db, owner, "confirm" if args.action == "accept" else "dismiss", "question",
                                 args.question_key, command_id=command_id, principal=principal,
                                 expected_revision=args.expected_revision,
                                 after={"feedback_id": row.id, "applied": [d.id for d in made]}))
    return {"question_key": args.question_key, "status": "resolved", "decision_ids": [d.id for d in made]}


def defer_question(db, owner: str, key: str, expected_revision: int, delta: timedelta, *,
                   command_id: str | None = None, principal: str | None = None) -> dict:
    """Remind me later for any context Question, agent suggestions included."""
    if not key.startswith(FEEDBACK_PREFIX):
        return decisions.defer_question(db, owner, key, expected_revision, delta, command_id=command_id,
                                        principal=principal)
    _feedback_question(db, owner, key, expected_revision)
    until = now() + delta
    decisions.record(db, owner, "later", "question", key, command_id=command_id,
                     principal=principal or bot_access.current_id() or owner, expected_revision=expected_revision,
                     effective_at=until)
    return {"question_key": key, "status": "deferred", "revision": expected_revision + 1,
            "deferred_until": until.isoformat()}


__all__ = [
    "BriefInput", "ChangesInput", "EvidenceInput", "FeedbackInput", "MeetingPrepInput", "OpenLoopsInput",
    "ResolveInput", "SearchInput", "SourceInput", "authorize", "brief", "changes", "defer_question",
    "evidence_for", "feedback", "feedback_items", "meeting_prep", "open_loops", "question_items", "resolve",
    "search", "source",
]
