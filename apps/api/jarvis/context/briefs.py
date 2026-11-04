"""Entity briefs: versioned, evidence-cited summaries of what is known about an actor, project, client or other
entity (plan section 6).

A brief has the six sections of `schema.BRIEF_SECTIONS`. Version 0 is deterministic (no model): every claim
restates current assertions with surviving evidence and cites them, plus linked task state read from tasks.
Synthesis is split so no transaction or lock is held during the model call (plan section 9): `prepare_brief`
reads the inputs into a detached `BriefPlan`, the caller closes its session and runs `run_synthesis`, and
`commit_brief` re-checks the fingerprint and writes (grounded sections; the deterministic ones if synthesis
failed, or if the inputs changed meanwhile, then flagged for re-synthesis). Next-meeting preparation is always
deterministic (calendar data).

Dependencies and freshness. `dependencies` records what the brief was built from: assertion versions (an object
keyed by assertion ID, which `temporal.stale_briefs` matches), entity link revisions, source availability,
linked task and actor revisions, the highest decision seq on those items, the next calendar occurrence and the
format version. Its sha256 is the `fingerprint`: an identical fingerprint never creates a new version (except a
deterministic brief being upgraded to a synthesized one). Corrections and deletions call `mark_stale` (or B5's
`temporal.stale_briefs`) in their own transaction. A stale brief is never served as current: `get_brief`
rebuilds it, or, when asked not to, serves the deterministic current view marked stale, so nothing a deletion
removed can reappear from a cached brief.
"""

import hashlib
import json
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime, time, timedelta

from sqlalchemy import func, or_, select, update
from sqlalchemy.dialects.postgresql import array

from ..domain import DomainError, advisory, emit
from ..models import Actor, Task, now
from . import decisions, temporal, views
from .engine import Synthesizer
from .models import (
    ContextAssertion,
    ContextBrief,
    ContextDecision,
    ContextEntityLink,
    ContextEvidence,
    ContextSource,
)
from .native.synthesize import SynthesisFailed, ground
from .schema import (
    BRIEF_SECTIONS,
    ENTITY_TYPES,
    BriefClaim,
    BriefRequest,
    BriefSection,
    EntityRef,
    ScopePredicate,
    SynthesizeRequest,
    TaskView,
)

log = logging.getLogger(__name__)

FORMAT = "brief-v0.1"
RECENT_DAYS = 14
SECTION_LIMIT = 12
STATE_KINDS = ("decision", "rationale", "fact", "preference", "procedure", "risk")
MODALITY_PREFIX = {"proposed": "Proposed: ", "hypothetical": "Hypothetically: ", "negated": "Not so: ",
                   "question": "Asked: "}
STALE_NOTICE = ("The stored brief is out of date after a correction, deletion or new context; this is the current "
                "view built from surviving evidence until it is rebuilt.")


# ---------------------------------------------------------------- inputs


@dataclass
class Inputs:
    owner: str
    entity: EntityRef
    name: str | None
    at: datetime
    rows: list[ContextAssertion]  # current (or contested), in scope, with surviving evidence
    history: list[ContextAssertion]  # superseded predecessors of rows that took effect recently
    dropped: int  # in-scope rows whose evidence did not survive
    withdrawn: int  # in-scope rows retracted recently
    loops: list[dict]
    tasks: dict[str, list[Task]]
    actors: dict[str, Actor]
    links: dict[str, int]
    sources: dict[str, str]
    decision_seq: int
    occurrence: views.Occurrence | None
    me: str | None
    names: dict[str, str] = field(default_factory=dict)

    @property
    def window(self) -> datetime:
        return self.at - timedelta(days=RECENT_DAYS)


def _effective(row: ContextAssertion) -> datetime:
    return row.valid_from or row.created_at


def gather(db, owner: str, entity: EntityRef, *, at: datetime | None = None) -> Inputs:
    """Read everything a brief depends on, owner-scoped, as of `at` (default now)."""
    at = at or now()
    a = ContextAssertion
    in_scope = views.scoped(db, owner, entity, at=at)
    rows = views.supported(db, owner, in_scope)
    window = at - timedelta(days=RECENT_DAYS)
    previous = {r.supersedes_id for r in rows if r.supersedes_id and _effective(r) >= window}
    history = views.supported(db, owner, list(db.scalars(select(a).where(
        a.owner_id == owner, a.id.in_(previous), a.lifecycle == "superseded").order_by(a.id)))) if previous else []
    retracted = ScopePredicate(owner_id=owner, entities=[entity], include_retracted=True)
    withdrawn = db.scalar(select(func.count()).select_from(
        temporal.select_assertions(retracted, at=at).where(a.lifecycle == "retracted", a.updated_at >= window)
        .subquery())) or 0
    ids = [r.id for r in rows + history]
    commitments = [r for r in rows if r.kind == "commitment" and r.modality not in views.NOT_COMMITTED]
    loops = [loop for loop in views.loops(db, owner, commitments) if loop["status"] == "open"]
    tasks = views.linked_tasks(db, owner, rows)
    if entity.entity_type == "task":
        task = db.get(Task, entity.entity_id)
        if task is not None and task.owner_id == owner:
            tasks.setdefault("", []).append(task)  # key "": the brief is about this task itself
    link = ContextEntityLink
    links = dict(db.execute(select(link.id, link.revision).where(
        link.owner_id == owner, or_(link.assertion_id.in_(ids), (link.target_type == entity.entity_type)
                                    & (link.target_id == entity.entity_id)))).all())
    e, s = ContextEvidence, ContextSource
    sources = {sid: "deleted" if deleted_at or status == "deleted" else availability
               for sid, availability, status, deleted_at in db.execute(
                   select(s.id, s.availability, s.status, s.deleted_at).where(
                       s.owner_id == owner, s.id.in_(select(e.source_id).where(e.assertion_id.in_(ids)))))}
    d = ContextDecision
    decision_seq = db.scalar(select(func.max(d.seq)).where(
        d.owner_id == owner, d.target_id.in_([*ids, *links, entity.entity_id]))) or 0
    try:
        occurrence = views.next_occurrence(db, owner, entity, at=at)
    except DomainError as error:  # calendar unreadable: the brief still builds, with a gap
        log.warning("brief calendar lookup failed: %s", error.code)
        occurrence = None
    actor_ids = {i.get("entity_id") for r in rows for i in r.responsible} | {r.recipient for r in rows}
    actor_ids |= {i for r in rows for t, i in ((r.subject_type, r.subject_id), (r.object_type, r.object_id))
                  if t == "actor"}
    actor_ids |= set(db.scalars(select(link.target_id).where(link.owner_id == owner, link.assertion_id.in_(ids),
                                                             link.target_type == "actor",
                                                             link.status.in_(views.LINKED))))
    if occurrence:
        actor_ids |= {g["actor_id"] for g in occurrence.attendees}
    if entity.entity_type == "actor":
        actor_ids.add(entity.entity_id)
    actor_ids.discard(None)
    actors = {r.id: r for r in db.scalars(select(Actor).where(Actor.owner_id == owner,
                                                              Actor.id.in_(actor_ids)))} if actor_ids else {}
    return Inputs(owner=owner, entity=entity, name=views.entity_name(db, owner, entity.entity_type, entity.entity_id),
                  at=at, rows=rows, history=history, dropped=len(in_scope) - len(rows), withdrawn=withdrawn,
                  loops=loops, tasks=tasks, actors=actors, links=links, sources=sources, decision_seq=decision_seq,
                  occurrence=occurrence, me=views.me(db, owner),
                  names={i: actor.name for i, actor in actors.items()})


def dependencies(inputs: Inputs) -> dict:
    occurrence = inputs.occurrence
    return {
        "format": FORMAT,
        "entity_name": inputs.name,
        "assertions": {r.id: r.current_version for r in inputs.rows + inputs.history},
        "links": dict(inputs.links),
        "sources": dict(inputs.sources),
        "tasks": {t.id: t.revision for ts in inputs.tasks.values() for t in ts},
        "actors": {i: a.revision for i, a in inputs.actors.items()},
        "decisions": inputs.decision_seq,
        "withdrawn": inputs.withdrawn,
        "dropped": inputs.dropped,
        "next_event": {"event_id": occurrence.event.id, "start": views.iso(occurrence.start),
                       "attendees": sorted(filter(None, (g["actor_id"] for g in occurrence.attendees)))}
        if occurrence else None,
    }


def fingerprint(deps: dict) -> str:
    return hashlib.sha256(json.dumps(deps, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


# ---------------------------------------------------------------- deterministic sections


def describe(row: ContextAssertion) -> str:
    text = MODALITY_PREFIX.get(row.modality, "") + row.statement
    return text + (" (inferred)" if not row.explicit else "")


def _day(value: str | datetime | None) -> str | None:
    if value is None:
        return None
    return (datetime.fromisoformat(value) if isinstance(value, str) else value).date().isoformat()


class _Section:
    def __init__(self, name: str):
        self.name, self.claims, self.gaps, self.extra = name, [], [], 0

    def claim(self, text: str, assertion_ids=(), task_ids=()) -> None:
        if len(self.claims) >= SECTION_LIMIT:
            self.extra += 1
            return
        self.claims.append(BriefClaim(text=text, assertion_ids=list(dict.fromkeys(assertion_ids)),
                                      task_ids=list(dict.fromkeys(task_ids))))

    def done(self, empty: str) -> BriefSection:
        gaps = list(self.gaps)
        if self.extra:
            gaps.append(f"{self.extra} more not shown.")
        if not self.claims and not gaps:
            gaps.append(empty)
        return BriefSection(name=self.name, claims=self.claims, gaps=gaps)


def _names(parties: list[dict]) -> str:
    return ", ".join(p["name"] or "someone" for p in parties) or "someone"


def loop_text(loop: dict) -> str:
    who = _names(loop["responsible"])
    lead = {"own": "You owe", "waiting_for": f"Waiting for {who}",
            "between_others": f"{who} owes {(loop['recipient'] or {}).get('name') or 'someone else'}",
            "collective": f"Shared ({who})"}[loop["bucket"]]
    text = f"{lead}: {loop['statement']}"
    if loop["due_at"]:
        text += f" (due {_day(loop['due_at'])})"
    if loop["inferred"]:
        text += " (inferred)"
    titles = ", ".join(f"“{t['title']}”" for t in loop["tasks"])
    if loop["task_state"] == "covered":
        text += f". Linked task {titles} is completed: that covers the tracked work, not necessarily the whole promise."
    elif loop["task_state"] == "open":
        text += f". Tracked in {titles}."
    elif loop["task_state"] == "cancelled":
        text += f". Linked task {titles} was cancelled."
    return text


def _state(inputs: Inputs) -> BriefSection:
    section = _Section("current_state_and_why")
    order = {k: i for i, k in enumerate(STATE_KINDS)}
    for row in sorted((r for r in inputs.rows if r.kind in order and r.lifecycle == "current"
                       and r.modality != "question"),
                      key=lambda r: (order[r.kind], -_effective(r).timestamp(), r.id)):
        label = {"rationale": "Why: ", "risk": "Risk: "}.get(row.kind, "")
        section.claim(label + describe(row), [row.id])
    return section.done("Nothing current is recorded about this yet.")


def _open_loops(inputs: Inputs) -> BriefSection:
    section = _Section("open_loops")
    for loop in inputs.loops:
        section.claim(loop_text(loop), [loop["assertion_id"]], [t["id"] for t in loop["tasks"]])
    return section.done("No open commitments are recorded.")


def _prep(inputs: Inputs) -> BriefSection:
    section = _Section("next_meeting_prep")
    occurrence = inputs.occurrence
    if occurrence is None:
        section.gaps.append(f"No upcoming calendar event involves this in the next {views.HORIZON_DAYS} days.")
        return section.done("")
    when = f"{occurrence.title} ({_day(occurrence.start)})"
    people = {g["actor_id"] for g in occurrence.attendees if g["actor_id"] and g["actor_id"] != inputs.me}
    for loop in inputs.loops:
        involved = {p["actor_id"] for p in loop["responsible"]} | {(loop["recipient"] or {}).get("actor_id")}
        if involved & people or inputs.entity.entity_type != "actor":
            section.claim(f"Raise at {when}: {loop_text(loop)}", [loop["assertion_id"]],
                          [t["id"] for t in loop["tasks"]])
    for row in inputs.rows:
        if row.kind == "open_question":
            section.claim(f"Ask at {when}: {describe(row)}", [row.id])
    if unresolved := sorted(filter(None, (g["name"] or g["email"] for g in occurrence.attendees
                                          if not g["actor_id"]))):
        section.gaps.append("Not matched to a person yet: " + ", ".join(unresolved) + ".")
    return section.done(f"Nothing open to raise at {when}.")


def _who(inputs: Inputs) -> BriefSection:
    section = _Section("who_is_who")
    cited: dict[str, list[str]] = defaultdict(list)
    roles: dict[str, list[str]] = defaultdict(list)
    for row in inputs.rows:
        if row.kind == "relationship":
            for kind, ident in ((row.subject_type, row.subject_id), (row.object_type, row.object_id)):
                if kind == "actor" and ident in inputs.actors:
                    roles[ident].append(describe(row))
                    cited[ident].append(row.id)
    owing: dict[str, list[str]] = defaultdict(list)
    for loop in inputs.loops:
        for party in loop["responsible"]:
            if party["actor_id"] in inputs.actors:
                owing[party["actor_id"]].append(loop["assertion_id"])
    for actor_id in sorted(set(cited) | set(owing), key=lambda i: (-len(cited[i]) - len(owing[i]),
                                                                     inputs.names.get(i, ""), i)):
        name = inputs.names[actor_id] + (" (you)" if actor_id == inputs.me else "")
        parts = list(dict.fromkeys(roles[actor_id]))
        if owing[actor_id]:
            parts.append(f"responsible for {len(owing[actor_id])} open item(s)")
        section.claim(f"{name}: " + "; ".join(parts), cited[actor_id] + owing[actor_id])
    unresolved = sorted({p["mention"] for loop in inputs.loops for p in loop["responsible"]
                         if not p["resolved"] and not p["collective"] and p["mention"]})
    if unresolved:
        section.gaps.append("Named but not matched to a person yet: " + ", ".join(unresolved) + ".")
    return section.done("No people are recorded in this context yet.")


def _changes(inputs: Inputs) -> BriefSection:
    section = _Section("recent_changes")
    previous = {r.id: r for r in inputs.history}
    for row in sorted(inputs.rows, key=lambda r: (-_effective(r).timestamp(), r.id)):
        if _effective(row) < inputs.window:
            continue
        if row.supersedes_id in previous:
            old = previous[row.supersedes_id]
            section.claim(f"Changed ({_day(_effective(row))}): {describe(old)} Now: {describe(row)}",
                          [old.id, row.id])
        elif row.kind in ("decision", "commitment", "fact", "risk", "open_question", "relationship"):
            section.claim(f"New ({_day(_effective(row))}): {describe(row)}", [row.id])
    if inputs.withdrawn:
        section.gaps.append(f"{inputs.withdrawn} earlier item(s) were withdrawn in the last {RECENT_DAYS} days.")
    return section.done(f"No changes in the last {RECENT_DAYS} days.")


def _unknowns(inputs: Inputs) -> BriefSection:
    section = _Section("unknowns")
    for row in inputs.rows:
        if row.kind == "open_question" or row.modality == "question":
            section.claim(f"Open question: {row.statement}", [row.id])
    contested: dict[tuple, list[ContextAssertion]] = defaultdict(list)
    for row in inputs.rows:
        if row.lifecycle == "contested":
            contested[temporal.chain_identity(row) or (row.subject_type, row.subject_id, row.predicate)].append(row)
    for rows in contested.values():
        section.claim("Contested: " + " / ".join(describe(r) for r in rows), [r.id for r in rows])
    for loop in inputs.loops:
        if not loop["due_at"]:
            section.claim(f"No due date agreed: {loop['statement']}", [loop["assertion_id"]])
    if inputs.dropped:
        section.gaps.append(f"{inputs.dropped} item(s) lost their supporting evidence and are left out.")
    return section.done("Nothing is recorded as unknown or contested.")


BUILDERS = {"current_state_and_why": _state, "open_loops": _open_loops, "next_meeting_prep": _prep,
            "who_is_who": _who, "recent_changes": _changes, "unknowns": _unknowns}


def deterministic(inputs: Inputs) -> list[BriefSection]:
    return [BUILDERS[name](inputs) for name in BRIEF_SECTIONS]


# ---------------------------------------------------------------- synthesis


def _due(task: Task) -> datetime | None:
    return datetime.combine(task.due_date, time.min, UTC) if task.due_date else None


def synthesize_request(db, inputs: Inputs) -> SynthesizeRequest:
    by_task: dict[str, tuple[Task, list[str]]] = {}
    for assertion_id, tasks in inputs.tasks.items():
        for task in tasks:
            by_task.setdefault(task.id, (task, []))[1].extend([assertion_id] if assertion_id else [])
    entity = inputs.entity.model_copy(update={"mention": (inputs.name or "")[:300] or None})
    return SynthesizeRequest(
        brief=BriefRequest(scope=ScopePredicate(owner_id=inputs.owner, entities=[inputs.entity]), entity=entity,
                           sections=[s for s in BRIEF_SECTIONS if s != "next_meeting_prep"]),
        assertions=[v for r in inputs.rows if (v := temporal.view(db, r))],
        history=[v for r in inputs.history if (v := temporal.view(db, r))],
        tasks=[TaskView(id=t.id, revision=t.revision, title=t.title, status=t.status, due_at=_due(t),
                        assertion_ids=ids) for t, ids in by_task.values()])


# ---------------------------------------------------------------- build (deterministic) and synthesis (split)


def latest(db, owner: str, entity_type: str, entity_id: str) -> ContextBrief | None:
    b = ContextBrief
    return db.scalar(select(b).where(b.owner_id == owner, b.entity_type == entity_type, b.entity_id == entity_id)
                     .order_by(b.version.desc()).limit(1).execution_options(populate_existing=True))


def _reusable(brief: ContextBrief | None, print_: str, engine: str) -> bool:
    """Same inputs never make a new version, unless a deterministic brief is being upgraded by synthesis."""
    return (brief is not None and brief.fingerprint == print_
            and (engine == "deterministic" or brief.watermarks.get("engine") != "deterministic"))


def _write(db, owner: str, inputs: Inputs, sections: list[BriefSection], *, engine: str, run_id: str | None,
           **marks) -> ContextBrief:
    """Store a version unless the latest one already covers these inputs (then just clear its stale flag)."""
    deps = dependencies(inputs)
    print_ = fingerprint(deps)
    entity_type, entity_id = inputs.entity.entity_type, inputs.entity.entity_id
    advisory(db, f"context-brief:{owner}:{entity_type}:{entity_id}")
    current = latest(db, owner, entity_type, entity_id)
    if _reusable(current, print_, engine):
        if current.stale:
            current.stale, current.updated_at = False, now()
            db.flush()
        return current
    watermarks = {"engine": engine, "format": FORMAT, "built_at": inputs.at.isoformat(),
                  "decisions": decisions.watermark(db, owner), "dependency_decisions": inputs.decision_seq,
                  **{k: v for k, v in marks.items() if v}}
    brief = ContextBrief(owner_id=owner, entity_type=entity_type, entity_id=entity_id,
                         sections=[s.model_dump(mode="json") for s in sections], dependencies=deps,
                         watermarks=watermarks, fingerprint=print_, version=(current.version if current else 0) + 1,
                         run_id=run_id)
    db.add(brief)
    db.flush()
    emit(db, owner, "context.changed", brief.id, brief.version)
    return brief


def _entity(entity_type: str, entity_id: str) -> EntityRef:
    if entity_type not in ENTITY_TYPES:
        raise DomainError("INVALID_ARGUMENT", f"Briefs cover {', '.join(ENTITY_TYPES)}.")
    return EntityRef(entity_type=entity_type, entity_id=entity_id)


def build_brief(db, owner: str, entity_type: str, entity_id: str, *, run_id: str | None = None,
                at: datetime | None = None) -> ContextBrief:
    """Deterministic rebuild in the caller's transaction (no model). Returns the latest version, which is the
    existing one when the dependency fingerprint is unchanged. For synthesis use `prepare_brief`,
    `run_synthesis` and `commit_brief`."""
    inputs = gather(db, owner, _entity(entity_type, entity_id), at=at)
    return _write(db, owner, inputs, deterministic(inputs), engine="deterministic", run_id=run_id)


@dataclass(frozen=True)
class BriefPlan:
    """What a synthesized brief will be built from, detached from any session: the fingerprint and dependencies
    of the inputs, the deterministic sections (the fallback, and next-meeting prep), and the synthesis request."""
    owner: str
    entity_type: str
    entity_id: str
    at: datetime
    fingerprint: str
    dependencies: dict
    deterministic: list[BriefSection]
    request: SynthesizeRequest


def prepare_brief(db, owner: str, entity_type: str, entity_id: str, *,
                  at: datetime | None = None) -> BriefPlan | None:
    """Read the inputs. None when the stored version is already synthesized from identical inputs (its stale
    flag, if any, is cleared). Close the session before running the plan's synthesis."""
    inputs = gather(db, owner, _entity(entity_type, entity_id), at=at)
    deps = dependencies(inputs)
    print_ = fingerprint(deps)
    current = latest(db, owner, entity_type, entity_id)
    if _reusable(current, print_, "synthesized"):
        if current.stale:
            current.stale, current.updated_at = False, now()
            db.flush()
        return None
    return BriefPlan(owner=owner, entity_type=entity_type, entity_id=entity_id, at=inputs.at, fingerprint=print_,
                     dependencies=deps, deterministic=deterministic(inputs),
                     request=synthesize_request(db, inputs))


def run_synthesis(plan: BriefPlan, synthesizer: Synthesizer) -> tuple[list[BriefSection] | None, str | None]:
    """Run the model outside any session: (grounded sections, None), or (None, reason) when it failed."""
    try:
        return ground(synthesizer.synthesize(plan.request), plan.request), None
    except (SynthesisFailed, DomainError) as error:
        reason = str(getattr(error, "code", None) or error)[:200]
        log.warning("brief synthesis failed, keeping deterministic sections: %s", reason)
        return None, reason


def commit_brief(db, owner: str, plan: BriefPlan, sections: list[BriefSection] | None, *,
                 run_id: str | None = None, error: str | None = None) -> ContextBrief:
    """Store the synthesized brief if the inputs are unchanged since `prepare_brief`. If they changed meanwhile,
    store the deterministic brief of the new inputs with `watermarks["resynthesize"] = True` (enqueue again).
    `sections=None` (synthesis failed or skipped) stores the deterministic sections with `error` recorded."""
    if owner != plan.owner:
        raise ValueError("plan belongs to another owner")
    inputs = gather(db, owner, _entity(plan.entity_type, plan.entity_id), at=plan.at)
    if fingerprint(dependencies(inputs)) != plan.fingerprint:
        return _write(db, owner, inputs, deterministic(inputs), engine="deterministic", run_id=run_id,
                      resynthesize=True)
    if sections is None:
        return _write(db, owner, inputs, plan.deterministic, engine="deterministic", run_id=run_id,
                      synthesis_error=error or "skipped")
    grounded = {s.name: s for s in ground(sections, plan.request)}
    merged = [grounded.get(s.name, s) if s.name != "next_meeting_prep" else s for s in plan.deterministic]
    return _write(db, owner, inputs, merged, engine="synthesized", run_id=run_id)


def serialize(brief: ContextBrief | None, inputs: Inputs, *, basis: str, sections: list[dict] | None = None) -> dict:
    occurrence = inputs.occurrence
    stale = basis == "current_view" and brief is not None
    return {
        "brief_id": brief.id if brief else None,
        "entity": {"entity_type": inputs.entity.entity_type, "entity_id": inputs.entity.entity_id,
                   "name": inputs.name},
        "version": brief.version if brief else None,
        "fingerprint": brief.fingerprint if brief else None,
        "engine": brief.watermarks.get("engine") if brief and not stale else "deterministic",
        "built_at": brief.watermarks.get("built_at") if brief and not stale else inputs.at.isoformat(),
        "stale": stale, "basis": basis, "notice": STALE_NOTICE if stale else None,
        "sections": sections if sections is not None else brief.sections,
        "next_meeting": occurrence.as_dict() if occurrence else None,
    }


def get_brief(db, owner: str, entity_type: str, entity_id: str, *, rebuild_if_stale: bool = True,
              at: datetime | None = None) -> dict:
    """The entity's current brief. A stored brief is served only when it is not stale and its fingerprint still
    matches the inputs; otherwise it is rebuilt (deterministically) or, with `rebuild_if_stale=False`, the
    deterministic current view is served, marked stale, without writing."""
    entity = _entity(entity_type, entity_id)
    inputs = gather(db, owner, entity, at=at)
    if inputs.name is None and not inputs.rows:
        raise DomainError("NOT_FOUND", "That item is no longer available.", 404)
    stored = latest(db, owner, entity_type, entity_id)
    if stored is not None and not stored.stale and stored.fingerprint == fingerprint(dependencies(inputs)):
        return serialize(stored, inputs, basis="stored")
    if rebuild_if_stale:
        return serialize(_write(db, owner, inputs, deterministic(inputs), engine="deterministic", run_id=None),
                         inputs, basis="rebuilt")
    return serialize(stored, inputs, basis="current_view",
                     sections=[s.model_dump(mode="json") for s in deterministic(inputs)])


def _stale_where(db, owner: str, clause) -> int:
    b = ContextBrief
    return db.execute(update(b).where(b.owner_id == owner, b.stale.is_(False), clause)
                      .values(stale=True, updated_at=now()).execution_options(synchronize_session="fetch")).rowcount


def mark_stale(db, owner: str, assertion_ids=(), *, source_id: str | None = None, task_ids=()) -> int:
    """Mark briefs depending on any of these assertions, on any assertion a source supports, on the source
    itself, or on these tasks stale, in the caller's transaction. Returns how many briefs changed."""
    ids = set(assertion_ids)
    count = 0
    b = ContextBrief
    if source_id:
        ids |= set(db.scalars(select(ContextEvidence.assertion_id).where(
            ContextEvidence.owner_id == owner, ContextEvidence.source_id == source_id)))
        count += _stale_where(db, owner, b.dependencies["sources"].has_key(source_id))
    if task_ids:
        count += _stale_where(db, owner, b.dependencies["tasks"].has_any(array(sorted(set(task_ids)))))
    return count + temporal.stale_briefs(db, owner, ids)
