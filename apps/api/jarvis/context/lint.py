"""Weekly context lint and incremental brief rebuilds (plan_final.md sections 5 and 8).

`due` uses the owner-local memory dream slot (`memory_review.period`: Sunday 03:00 in the owner's timezone, DST
included) gated the same way as the memory review (`memory_learning` and `deep_sleep_enabled`), plus the context
flag, the owner's pause and having any context source.

A pass:
1. Begin: `temporal.settle` brings keyed chains whose successor has since taken effect up to date, and the
   neighbourhoods changed since the cursor (subjects and parties of changed assertions, link targets, entities
   with a stale brief) are queued in `pending`.
2. Global checks without a model, once per pass (and again when a pass is carried into a new week): overdue
   commitments that no task, completion or proposal covers become task proposals; a mention left unresolved in
   two or more meetings becomes an "add this person" suggestion; B4 `merge_candidates` become merge suggestions.
   Lint proposals have no source and one fingerprint per finding, so a finding is proposed once and a dismissal
   is durable. Open lint proposals whose finding went away are superseded.
3. Each pending neighbourhood, under the slot's budget: live assertions about a changed subject go through the
   Reconciler and `temporal.apply_ops`, so conflicts become contested and `decisions.question_items` asks only
   when that is consequential. Assertions the owner confirmed by hand (a Question answered "both", or Confirm) are
   never contested or superseded again. The entity's existing brief is brought up to date without a model: a
   deterministic brief is rebuilt in place; a synthesized one whose inputs changed (or that is stale) gets one
   `context_brief` job, so synthesis stays with the pipeline and outside any transaction. A digest of the
   subject's assertion versions taken after processing lets an unchanged neighbourhood (lint's own writes
   included) skip the model next time.

`run` works in the caller's transaction and never commits. With `limit=None` it does the whole pass at once.
`perform` (the `context_lint` job) calls `run(limit=1)` in a fresh transaction per unit: a unit makes at most one
model call (the reconciler), before it takes any lock anyone else waits on, and an interruption loses one unit
at most: the next run resumes from `pending`. The reconciler is the native one, or the double when the owner's
settings say `"engine": "double"` (as in the pipeline).

State lives in `ContextOwnerState.settings["lint"]`: `cursor` (high-water mark of the last finished pass), the
pass in progress (`high`, `low`, `pending`, `run_id`, `checked`), `digests`, `spent`/`spent_slot` (the
slot's budget use), `completed` (slot of the last finished or budget-stopped pass) and `last` (the latest call's
summary). A pass the budget stops marks the slot done and resumes next week.
Ten runs without new evidence create no Questions, proposals or brief versions.
"""

import hashlib
import logging
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta

from sqlalchemy import func, select
from sqlalchemy.dialects.postgresql import array, insert

from ..db import session_scope
from ..domain import advisory, emit, enqueue_job, preferences
from ..memory_review import period
from ..models import Actor, Job
from ..models import now as clock
from . import briefs, decisions, identity, temporal, views
from . import evidence as survival
from . import settings as context_settings
from .doubles import DoubleReconciler
from .engine import Reconciler
from .models import (
    AMBIGUOUS_NAMESPACES,
    ContextActorIdentity,
    ContextAssertion,
    ContextBrief,
    ContextDecision,
    ContextEntityLink,
    ContextEvidence,
    ContextOwnerState,
    ContextProposal,
    ContextRun,
    ContextSource,
)
from .native.reconcile import ALLOWANCE_USD, NativeReconciler
from .schema import ENTITY_TYPES, EntityRef

log = logging.getLogger(__name__)

CONFIG_VERSION = "lint-v1"
DEFAULT_BUDGET_USD = 0.50  # per weekly slot; `ContextOwnerState.settings["lint_budget_usd"]` overrides
RECONCILE_RESERVE_USD = ALLOWANCE_USD  # the native reconciler's cap per model call
JOB_PENDING = ("queued", "dispatched", "running", "retrying")
RECONCILE_LIMIT = 50  # changed assertions per reconcile request
OVERLAP = timedelta(minutes=10)  # rows committed late by concurrent writers are still seen
RECURRING_SOURCES = 2
PERSON_ROLES = ("responsible", "responsible_mention", "recipient")
OPEN = ("pending", "later")
CHECKS = ("overdue_untracked", "recurring_mention", "duplicate_identity")


# ---------------------------------------------------------------- state and scheduling


def _state(db, owner: str, *, lock: bool = False) -> ContextOwnerState | None:
    query = select(ContextOwnerState).where(ContextOwnerState.owner_id == owner).execution_options(
        populate_existing=True)
    return db.scalar(query.with_for_update() if lock else query)


def _lint(state: ContextOwnerState | None) -> dict:
    return dict((state.settings or {}).get("lint") or {}) if state else {}


def _save(db, owner: str, **changes) -> dict:
    db.execute(insert(ContextOwnerState).values(owner_id=owner).on_conflict_do_nothing(index_elements=["owner_id"]))
    state = _state(db, owner, lock=True)
    lint = {**_lint(state), **changes}
    state.settings = {**(state.settings or {}), "lint": lint}
    state.updated_at = clock()
    db.flush()
    return lint


def slot(db, owner: str, at: datetime) -> str:
    """The weekly slot containing `at` (ISO start, UTC), the memory dream's."""
    return period(preferences(db, owner), at)[0].isoformat()


def due(db, owner: str, now: datetime | None = None) -> bool:
    """Whether this week's lint has not run yet for an owner whose context is on, unpaused and non-empty."""
    at = now or clock()
    if not context_settings.enabled():
        return False
    prefs = preferences(db, owner)
    if not prefs["memory_learning"] or not prefs["deep_sleep_enabled"]:
        return False
    state = _state(db, owner)
    if state is not None and state.paused:
        return False
    if db.scalar(select(ContextSource.id).where(ContextSource.owner_id == owner,
                                                ContextSource.deleted_at.is_(None)).limit(1)) is None:
        return False
    return _lint(state).get("completed") != period(prefs, at)[0].isoformat()


@dataclass
class _Budget:
    limit: float
    spent: float = 0.0

    def allows(self, reserve: float) -> bool:
        return self.limit - self.spent >= reserve

    def left(self) -> float:
        return max(0.0, self.limit - self.spent)


# ---------------------------------------------------------------- lint proposals


def _fingerprint(check: str, key: str) -> str:
    return hashlib.sha256(f"lint|{check}|{key}".encode()).hexdigest()


def _propose(db, owner: str, check: str, key: str, *, kind: str, title: str, payload: dict, at: datetime,
             responsible: str | None = None, due_at: datetime | None = None, assertion_ids=(),
             evidence=()) -> bool:
    """Insert a lint proposal unless this finding was ever proposed (whatever became of it)."""
    p = ContextProposal
    created = db.scalar(insert(p).values(
        owner_id=owner, source_id=None, kind=kind, fingerprint=_fingerprint(check, key), title=title[:500],
        payload={"origin": "lint", "check": check, "key": key, **payload}, responsible=responsible, due_at=due_at,
        assertion_ids=list(assertion_ids), evidence=list(evidence), created_at=at, updated_at=at,
    ).on_conflict_do_nothing(constraint="uq_context_proposals_fingerprint").returning(p.id))
    if created:
        emit(db, owner, "context.changed", created, 1)
    return created is not None


def _supersede(db, owner: str, check: str, keep: set[str]) -> int:
    """Close open lint proposals of `check` whose finding no longer holds."""
    p = ContextProposal
    count = 0
    for row in db.scalars(select(p).where(
            p.owner_id == owner, p.source_id.is_(None), p.payload["origin"].astext == "lint",
            p.payload["check"].astext == check, p.disposition.in_(OPEN)).order_by(p.id).with_for_update()):
        if row.payload.get("key") in keep:
            continue
        row.disposition, row.revision, row.updated_at = "superseded", row.revision + 1, clock()
        emit(db, owner, "context.changed", row.id, row.revision)
        count += 1
    db.flush()
    return count


def overdue(db, owner: str, at: datetime) -> dict[str, dict]:
    """Open-loop views of live, supported commitments past their due date that no linked task, completion
    evidence or non-lint proposal covers, keyed by assertion ID (commitments between other people excluded)."""
    a = ContextAssertion
    rows = views.supported(db, owner, list(db.scalars(select(a).where(
        a.owner_id == owner, a.kind == "commitment", a.lifecycle == "current", temporal.effective(a, at),
        a.due_at < at, a.modality.not_in(views.NOT_COMMITTED),
        func.jsonb_array_length(a.completion_evidence) == 0).order_by(a.due_at, a.id))))
    if not rows:
        return {}
    p = ContextProposal
    proposed = set()
    for cited, check in db.execute(select(p.assertion_ids, p.payload["check"].astext).where(
            p.owner_id == owner, p.assertion_ids.has_any(array([r.id for r in rows])))):
        if check != "overdue_untracked":
            proposed.update(cited)
    return {loop["assertion_id"]: loop for loop in views.loops(db, owner, [r for r in rows if r.id not in proposed])
            if loop["task_state"] == "none" and loop["status"] == "open" and loop["bucket"] != "between_others"}


def recurring_mentions(db, owner: str) -> dict[str, dict]:
    """People named in two or more live meetings whom no actor carries as a name or alias, keyed by mention."""
    link, e, a = ContextEntityLink, ContextEvidence, ContextAssertion
    rows = db.execute(select(link.id, link.mention, link.assertion_id, e.source_id)
                      .join(a, a.id == link.assertion_id)
                      .join(e, (e.assertion_id == link.assertion_id) & (e.role == "support"))
                      .where(link.owner_id == owner, link.status == "unresolved", link.target_id.is_(None),
                             link.role.in_(PERSON_ROLES), link.mention != "", a.lifecycle.in_(temporal.LIVE))
                      .order_by(link.id)).all()
    deleted = survival.deleted_sources(db, owner, {r.source_id for r in rows})
    groups: dict[str, dict] = defaultdict(lambda: {"names": Counter(), "sources": set(), "links": set(),
                                                   "assertions": set()})
    for link_id, mention, assertion_id, source_id in rows:
        key = decisions.mention_key(mention)
        if source_id in deleted or not key or temporal.normalized(key) in views.COLLECTIVE:
            continue
        group = groups[key]
        group["names"][mention.strip()] += 1
        group["sources"].add(source_id)
        group["links"].add(link_id)
        group["assertions"].add(assertion_id)
    known = {identity.normalize("alias", n) for n in db.scalars(select(Actor.name).where(
        Actor.owner_id == owner, Actor.archived.is_(False)))}
    i = ContextActorIdentity
    known |= set(db.scalars(select(i.normalized).where(i.owner_id == owner, i.namespace.in_(AMBIGUOUS_NAMESPACES))))
    return {key: {"name": min(g["names"], key=lambda n: (-g["names"][n], n)), "sources": sorted(g["sources"]),
                  "links": sorted(g["links"]), "assertions": sorted(g["assertions"])}
            for key, g in sorted(groups.items()) if len(g["sources"]) >= RECURRING_SOURCES and key not in known}


def _proposals(db, owner: str, at: datetime, summary: Counter) -> None:
    loops = overdue(db, owner, at)
    for assertion_id, loop in loops.items():
        who = ", ".join(p["name"] or "someone" for p in loop["responsible"]) or "someone"
        title = loop["statement"] if loop["bucket"] in ("own", "collective") else \
            f"Follow up with {who}: {loop['statement']}"
        summary["proposals_created"] += _propose(
            db, owner, "overdue_untracked", assertion_id, kind="create", title=title, at=at,
            payload={"statement": loop["statement"], "bucket": loop["bucket"], "due_at": loop["due_at"],
                     "description": f"Due {loop['due_at'][:10]} and no task tracks it."},
            responsible=next((p["actor_id"] for p in loop["responsible"] if p["actor_id"]), None),
            due_at=datetime.fromisoformat(loop["due_at"]), assertion_ids=[assertion_id], evidence=loop["evidence"])
    summary["proposals_superseded"] += _supersede(db, owner, "overdue_untracked", set(loops))

    mentions = recurring_mentions(db, owner)
    for key, found in mentions.items():
        summary["proposals_created"] += _propose(
            db, owner, "recurring_mention", key, kind="suggestion", at=at,
            title=f"Add {found['name']} as a person (named in {len(found['sources'])} meetings)",
            payload={"action": "actor.create", "name": found["name"], "source_ids": found["sources"],
                     "link_ids": found["links"], "mention_assertion_ids": found["assertions"]})
    summary["proposals_superseded"] += _supersede(db, owner, "recurring_mention", set(mentions))

    pairs = {":".join(sorted(m.actor_ids)): m for m in identity.merge_candidates(db, owner)}
    names = views.actor_names(db, owner, [i for m in pairs.values() for i in m.actor_ids])
    for key, merge in pairs.items():
        first, second = (names.get(i, "someone") for i in merge.actor_ids)
        summary["proposals_created"] += _propose(
            db, owner, "duplicate_identity", key, kind="suggestion", at=at, title=f"Same person? {first} and {second}",
            payload={"action": "actor.merge", "actor_ids": list(merge.actor_ids), "reason": merge.reason})
    summary["proposals_superseded"] += _supersede(db, owner, "duplicate_identity", set(pairs))


# ---------------------------------------------------------------- neighbourhoods


def _key(entity) -> str:
    return f"{entity[0]}\x1f{entity[1]}"


def changed(db, owner: str, low: datetime | None) -> list[tuple[str, str]]:
    """Entities whose neighbourhood changed after `low` (everything when None): subjects, objects and parties
    of changed assertions, targets of changed links, and entities with a stale brief."""
    a, link, b = ContextAssertion, ContextEntityLink, ContextBrief
    query = select(a.subject_type, a.subject_id, a.object_type, a.object_id, a.responsible, a.recipient).where(
        a.owner_id == owner)
    out: set[tuple[str, str]] = set()
    for subject_type, subject_id, object_type, object_id, responsible, recipient in db.execute(
            query.where(a.updated_at > low) if low else query):
        if subject_type != "source":
            out.add((subject_type, subject_id))
        if object_type in ENTITY_TYPES and object_id:
            out.add((object_type, object_id))
        out |= {("actor", r["entity_id"]) for r in responsible or []
                if r.get("entity_type") == "actor" and r.get("entity_id")}
        if recipient:
            out.add(("actor", recipient))
    links = select(link.target_type, link.target_id).where(link.owner_id == owner, link.target_type.is_not(None),
                                                           link.target_id.is_not(None))
    out |= {tuple(r) for r in db.execute(links.where(link.updated_at > low) if low else links)}
    out |= {tuple(r) for r in db.execute(select(b.entity_type, b.entity_id).where(b.owner_id == owner,
                                                                                  b.stale.is_(True)))}
    return sorted(out)


def digest(db, owner: str, entity: tuple[str, str]) -> str:
    """Fingerprint of every assertion version about this subject."""
    a = ContextAssertion
    rows = db.execute(select(a.id, a.current_version).where(
        a.owner_id == owner, a.subject_type == entity[0], a.subject_id == entity[1]).order_by(a.id)).all()
    return hashlib.sha256(";".join(f"{i}:{v}" for i, v in rows).encode()).hexdigest()[:24]


def _answered(db, owner: str, ids) -> set[str]:
    """Assertions with an active human Confirm: the owner already decided these hold."""
    d = ContextDecision
    return {r.target_id for r in db.scalars(temporal.active_decisions(owner).where(
        d.operation == "confirm", d.origin == "human", d.target_type == "assertion", d.target_id.in_(list(ids))))}


def _reconcile_rows(db, owner: str, entity: tuple[str, str], low: datetime | None, at: datetime):
    a = ContextAssertion
    query = select(a).where(a.owner_id == owner, a.subject_type == entity[0], a.subject_id == entity[1],
                            a.lifecycle.in_(temporal.LIVE), temporal.effective(a, at))
    if low:
        query = query.where(a.updated_at > low)
    return list(db.scalars(query.order_by(a.updated_at.desc(), a.id).limit(RECONCILE_LIMIT)))


def _brief_target(db, owner: str, entity: tuple[str, str]) -> ContextBrief | None:
    if entity[0] not in ENTITY_TYPES or entity[1].startswith("mention:"):
        return None
    return briefs.latest(db, owner, *entity)


def _brief_outdated(db, owner: str, entity, brief: ContextBrief, at: datetime) -> bool:
    inputs = briefs.gather(db, owner, EntityRef(entity_type=entity[0], entity_id=entity[1]), at=at)
    return brief.stale or brief.fingerprint != briefs.fingerprint(briefs.dependencies(inputs))


def _brief_job_pending(db, owner: str, entity) -> bool:
    return db.scalar(select(Job.id).where(
        Job.owner_id == owner, Job.kind == "context_brief", Job.status.in_(JOB_PENDING),
        Job.payload["entity_type"].astext == entity[0], Job.payload["entity_id"].astext == entity[1]).limit(1)) \
        is not None


@dataclass
class _Pass:
    owner: str
    at: datetime
    low: datetime | None
    run_id: str
    reconciler: Reconciler
    budget: _Budget
    digests: dict
    summary: Counter


def _neighbourhood(db, job: _Pass, entity: tuple[str, str]) -> bool:
    """Process one entity; False when the budget cannot cover it (nothing was changed)."""
    owner, at = job.owner, job.at
    rows = [] if job.digests.get(_key(entity)) == digest(db, owner, entity) else \
        _reconcile_rows(db, owner, entity, job.low, at)
    request, work_keys = temporal.reconcile_request(db, owner, [r.id for r in rows]) if rows else (None, {})
    if request is not None:
        if not job.budget.allows(RECONCILE_RESERVE_USD):
            return False
        before = getattr(job.reconciler, "calls", 0)
        ops = (job.reconciler.reconcile(request, work_keys) if isinstance(job.reconciler, NativeReconciler)
               else job.reconciler.reconcile(request))
        job.budget.spent += (getattr(job.reconciler, "calls", 0) - before) * RECONCILE_RESERVE_USD
        answered = _answered(db, owner, {t for op in ops for t in op.targets})
        kept = [op for op in ops if op.op not in ("conflict", "supersede") or not answered & set(op.targets)]
        job.summary["answered_skipped"] += len(ops) - len(kept)
        applied = temporal.apply_ops(db, owner, kept, run_id=job.run_id, at=at)
        for name in ("contested", "merged", "shadow", "superseded", "stale"):
            job.summary["stale_ops" if name == "stale" else name] += len(applied.get(name, []))
        job.summary["reconciled"] += len(rows)
    brief = _brief_target(db, owner, entity)
    if brief is not None and brief.watermarks.get("engine") == "synthesized":
        if _brief_outdated(db, owner, entity, brief, at) and not _brief_job_pending(db, owner, entity):
            enqueue_job(db, owner, "context_brief", {"entity_type": entity[0], "entity_id": entity[1]})
            job.summary["brief_jobs"] += 1
    elif brief is not None:
        built = briefs.build_brief(db, owner, *entity, run_id=job.run_id, at=at)
        job.summary["briefs_rebuilt"] += built.version != brief.version
    job.digests[_key(entity)] = digest(db, owner, entity)
    return True


def _queue(db, owner: str, lint: dict, since: datetime | None) -> list[tuple[str, str]]:
    pending = [tuple(e) for e in lint.get("pending") or []]
    queued = set(pending)
    return pending + [e for e in changed(db, owner, since) if e not in queued]


def _interrupted(db, owner: str, epoch: int) -> str | None:
    state = _state(db, owner)
    if state is None:
        return None
    return "cancelled" if state.paused else "stale" if state.epoch != epoch else None


def _engine(state: ContextOwnerState | None) -> Reconciler:
    double = state is not None and (state.settings or {}).get("engine") == "double"
    return DoubleReconciler() if double else NativeReconciler()


# ---------------------------------------------------------------- run


def run(db, owner: str, *, budget_usd: float | None = None, now: datetime | None = None,
        reconciler: Reconciler | None = None, limit: int | None = None) -> dict:
    """Advance the owner's lint pass in the caller's transaction (never commits). `limit` bounds the units done
    (beginning a pass, the global checks, or one neighbourhood); status "more" asks for another call in a new
    transaction. `budget_usd` caps model spend per weekly slot (default: the owner's `lint_budget_usd` setting,
    else DEFAULT_BUDGET_USD). `now` is the instant lint reasons about."""
    at = now or clock()
    state = _state(db, owner)
    if state is not None and state.paused:
        return {"status": "paused"}
    advisory(db, f"context-lint:{owner}")
    lint = _lint(state)
    week = slot(db, owner, at)
    cap = budget_usd if budget_usd is not None else ((state.settings or {}) if state else {}).get(
        "lint_budget_usd", DEFAULT_BUDGET_USD)
    budget = _Budget(float(cap), float(lint.get("spent", 0.0)) if lint.get("spent_slot") == week else 0.0)
    already = budget.spent
    epoch = state.epoch if state else 0
    summary, units = Counter(), 0
    record = db.get(ContextRun, lint["run_id"]) if lint.get("run_id") else None
    if not lint.get("high") or record is None:
        record = ContextRun(owner_id=owner, stage="lint", status="running", config_version=CONFIG_VERSION,
                            engine=type(reconciler or _engine(state)).__name__[:60], epoch=epoch,
                            decision_watermark=decisions.watermark(db, owner), started_at=clock())
        db.add(record)
        db.flush()
        summary["settled"] = len(temporal.settle(db, owner, at=at))
        high = clock()
        low = datetime.fromisoformat(lint["cursor"]) - OVERLAP if lint.get("cursor") else None
        lint |= {"high": high.isoformat(), "low": low.isoformat() if low else None, "run_id": record.id,
                 "checked": None, "pending": [list(e) for e in _queue(db, owner, {}, low)]}
        units += 1
    record.status = "running"
    if lint.get("checked") != week and (limit is None or units < limit):
        if lint.get("checked"):  # a pass carried into a new week also takes what changed since it began
            since = datetime.fromisoformat(lint["high"]) - OVERLAP
            lint |= {"high": clock().isoformat(), "pending": [list(e) for e in _queue(db, owner, lint, since)]}
        _proposals(db, owner, at, summary)
        lint["checked"] = week
        units += 1
    pending = [tuple(e) for e in lint.get("pending") or []]
    job = _Pass(owner=owner, at=at, low=datetime.fromisoformat(lint["low"]) if lint.get("low") else None,
                run_id=record.id, reconciler=reconciler or _engine(state), budget=budget,
                digests=dict(lint.get("digests") or {}), summary=summary)
    status = None
    while pending and (limit is None or units < limit):
        if stopped := _interrupted(db, owner, epoch):
            status = stopped
            break
        if not _neighbourhood(db, job, pending[0]):
            status, summary["budget_exhausted"] = "partial", 1
            break
        units += 1
        pending.pop(0)
        summary["neighbourhoods"] += 1
    status = status or ("succeeded" if not pending else "more")
    spent = budget.spent - already
    result = {"status": status, "run_id": record.id, "period": week, "pending": len(pending),
              "spent_usd": round(spent, 6), "slot_spent_usd": round(budget.spent, 6), **dict(sorted(summary.items()))}
    lint |= {"pending": [list(e) for e in pending], "digests": job.digests, "spent": budget.spent,
             "spent_slot": week, "last": result}
    if status == "succeeded":
        lint |= {"cursor": lint["high"], "high": None, "low": None, "run_id": None}
    if status in ("succeeded", "partial"):
        lint["completed"] = week
    _save(db, owner, **lint)
    record.status = {"more": "running", "cancelled": "cancelled", "stale": "stale"}.get(status, status)
    record.cost_usd = float(record.cost_usd or 0) + spent
    record.stage_results = {**(record.stage_results or {}), "last": result}
    record.updated_at = clock()
    if status in ("succeeded", "partial"):
        record.finished_at = record.updated_at
    db.flush()
    return result


def perform(owner: str, *, budget_usd: float | None = None, reconciler: Reconciler | None = None) -> dict:
    """The `context_lint` job: one transaction per unit until the pass finishes or stops. A failure marks the
    pass's run row failed and propagates; finished units stay done."""
    totals: Counter = Counter()
    while True:
        try:
            with session_scope() as db:
                result = run(db, owner, budget_usd=budget_usd, reconciler=reconciler, limit=1)
        except Exception as error:
            with session_scope() as db:
                run_id = _lint(_state(db, owner)).get("run_id")
                if record := db.get(ContextRun, run_id) if run_id else None:
                    record.status, record.error, record.updated_at = "failed", type(error).__name__, clock()
            log.warning("context lint failed for %s: %s", owner, type(error).__name__)
            raise
        totals.update({k: v for k, v in result.items()
                       if isinstance(v, int | float) and k not in ("pending", "slot_spent_usd")})
        if result["status"] != "more":
            return {**result, **{k: round(v, 6) if isinstance(v, float) else v for k, v in totals.items()}}
