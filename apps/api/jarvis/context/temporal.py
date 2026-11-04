"""Temporal model: effective intervals, immutable recorded history, as-of filters and keyed value chains.

Effective time is the half-open interval `[valid_from, valid_to)` with a precision. Transaction time lives in
`context_assertion_versions`: every change to an assertion's head closes the open version (`recorded_to`) and
appends a new one (`recorded_from`); versions are never rewritten otherwise.

Current state is lifecycle `current`/`contested` *and* effective now, so a future-effective successor never
displaces today's value: the predecessor keeps lifecycle `current` with `valid_to` set to the successor's start,
and becomes `superseded` once that start has passed (`rechain`, `settle`).

Keyed chains (plan section 5): only the agreed due date and the explicitly exclusive responsible person of a
linked work item are single-valued. A chain is every live assertion with one of `EXCLUSIVE_KEYS` as predicate
about the same task or Linear issue, or about the same prior commitment (`ContextAssertion.key` holds its
stable key). `rechain` orders a chain by effective time alone, never by arrival: values whose time windows
overlap (ties, coarse precision) or whose time is unknown are `contested`, never ordered. Running it over the
same set of assertions always yields the same intervals and lifecycles, whatever order they arrived in.

This module is also the core write path for reconciliation: `create` admits a candidate (durable negatives
retract a re-extracted wrong interpretation at once), `reconcile_request` gathers what an engine needs, and
`apply_ops` validates and applies an engine's `ReconcileOp`s under the policy above.
"""

import re
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from pydantic import ValidationError
from sqlalchemy import and_, exists, false, or_, select, update
from sqlalchemy.dialects.postgresql import array

from ..models import now
from .models import (
    ContextAssertion,
    ContextAssertionVersion,
    ContextBrief,
    ContextDecision,
    ContextEntityLink,
    ContextEvidence,
    ContextOwnerState,
    ContextSegment,
)
from .schema import (
    ENTITY_TYPES,
    EXCLUSIVE_KEYS,
    AssertionView,
    CandidateAssertion,
    Decision,
    EntityRef,
    EvidenceRef,
    ReconcileOp,
    ReconcileRequest,
    ScopePredicate,
    TimeBound,
)

WORK_TYPES = ("task", "linear_issue")
LIVE = ("current", "contested")
CHAINED = ("current", "contested", "superseded")
# Never single-valued (plan section 5): these never supersede or conflict, they only accumulate.
MULTI_VALUED_KINDS = ("preference", "relationship")
MULTI_VALUED_KEYS = ("member_of", "membership", "affiliation", "affiliated_with", "participant", "participation",
                     "participates_in", "works_on", "preference", "prefers")
_WORD = re.compile(r"\w+")
_STEP = {"day": timedelta(days=1), "week": timedelta(days=7)}
_MONTHS = {"month": 1, "quarter": 3, "year": 12}
_POINT = timedelta(microseconds=1)
_EPOCH = datetime(1, 1, 1, tzinfo=UTC)


def normalized(text: str) -> str:
    return " ".join(_WORD.findall(text.lower()))


# ---------------------------------------------------------------- spans and ordering


@dataclass(frozen=True)
class Span:
    """The window an effective start may fall in: `[start, end)`; `None` bounds mean unknown."""

    start: datetime | None
    end: datetime | None

    @property
    def known(self) -> bool:
        return self.start is not None and self.end is not None


def _add_months(at: datetime, months: int) -> datetime:
    index = at.month - 1 + months
    year, month = at.year + index // 12, index % 12 + 1
    days = [31, 29 if year % 4 == 0 and (year % 100 or year % 400 == 0) else 28, 31, 30, 31, 30, 31, 31, 30, 31,
            30, 31][month - 1]
    return at.replace(year=year, month=month, day=min(at.day, days))


def span(at: datetime | None, precision: str | None) -> Span:
    """The window of a start of the given precision; `at` is the start of the period (midnight for a day)."""
    if at is None or precision in (None, "unknown"):
        return Span(None, None)
    if precision in _MONTHS:
        return Span(at, _add_months(at, _MONTHS[precision]))
    return Span(at, at + _STEP.get(precision, _POINT))


def precedes(a: Span, b: Span) -> bool:
    """`a` certainly starts before `b`. Overlapping windows and unknown times are not ordered."""
    return a.known and b.known and a.end <= b.start


@dataclass(frozen=True)
class Member:
    """What ordering needs to know about one assertion."""

    id: str
    statement: str
    span: Span

    @property
    def order(self):
        """Effective order; unknown times sort last (callers never order them against known ones)."""
        start = self.span.start
        return start is None, start or _EPOCH, normalized(self.statement), self.id


def clusters(members: Iterable[Member]) -> list[list[Member]]:
    """Known-time members in effective order, grouped where their windows overlap (ties). Unknown ones are left
    out: callers treat them as unordered."""
    result, end = [], None
    for m in sorted((m for m in members if m.span.known), key=lambda m: m.order):
        if result and m.span.start < end:
            result[-1].append(m)
            end = max(end, m.span.end)
        else:
            result.append([m])
            end = m.span.end
    return result


def distinct(cluster: Iterable[Member]) -> dict[str, list[Member]]:
    """Members by normalized statement, each list in order (its first member is the canonical one)."""
    groups = defaultdict(list)
    for m in sorted(cluster, key=lambda m: m.order):
        groups[normalized(m.statement)].append(m)
    return dict(sorted(groups.items(), key=lambda kv: kv[1][0].order))


# ---------------------------------------------------------------- filters (usable by retrieval)


def effective(model, at: datetime):
    """`at` falls in `[model.valid_from, model.valid_to)`; works for assertions and versions."""
    return and_(or_(model.valid_from.is_(None), model.valid_from <= at),
                or_(model.valid_to.is_(None), model.valid_to > at))


def recorded(at: datetime):
    """The version Eridani believed at `at`: `[recorded_from, recorded_to)` contains it."""
    v = ContextAssertionVersion
    return and_(v.recorded_from <= at, or_(v.recorded_to.is_(None), v.recorded_to > at))


def lifecycle_clause(model, *, valid_at: datetime | None, include_superseded: bool, include_retracted: bool,
                     reference: datetime):
    """Lifecycle and effective-time filter. Without `valid_at`: live and effective at `reference` (now), plus
    superseded or retracted rows when asked for. With `valid_at`: what was in force then, superseded included
    (its closed interval decides), retracted only when asked for. Shadow rows are never included."""
    extra = [lc for lc, wanted in (("superseded", include_superseded), ("retracted", include_retracted)) if wanted]
    if valid_at is None:
        return or_(and_(model.lifecycle.in_(LIVE), effective(model, reference)),
                   model.lifecycle.in_(extra) if extra else false())
    return and_(model.lifecycle.in_([*CHAINED, *extra]), effective(model, valid_at))


def _subject_clauses(scope: ScopePredicate) -> list:
    a = ContextAssertion
    clauses = [a.owner_id == scope.owner_id]
    if scope.kinds:
        clauses.append(a.kind.in_(scope.kinds))
    if scope.entities:
        matches = []
        for e in scope.entities:
            matches += [and_(a.subject_type == e.entity_type, a.subject_id == e.entity_id),
                        and_(a.object_type == e.entity_type, a.object_id == e.entity_id)]
        link = ContextEntityLink
        matches.append(exists().where(
            link.assertion_id == a.id, link.owner_id == scope.owner_id, link.status != "rejected",
            or_(*[and_(link.target_type == e.entity_type, link.target_id == e.entity_id) for e in scope.entities])))
        clauses.append(or_(*matches))
    return clauses


def _flags(scope: ScopePredicate) -> dict:
    return {"include_superseded": scope.include_superseded, "include_retracted": scope.include_retracted}


def scope_clauses(scope: ScopePredicate, *, at: datetime | None = None) -> list:
    """WHERE clauses over `ContextAssertion` for a scope predicate (`at` stands in for now). With `known_at`,
    lifecycle and effective time come from the version recorded at that instant (`valid_at` defaulting to
    `known_at`); read that version's content with `select_versions`."""
    clauses = _subject_clauses(scope)
    if scope.known_at is None:
        clauses.append(lifecycle_clause(ContextAssertion, valid_at=scope.valid_at, reference=at or now(),
                                        **_flags(scope)))
    else:
        v = ContextAssertionVersion
        clauses.append(exists().where(
            v.assertion_id == ContextAssertion.id, recorded(scope.known_at),
            lifecycle_clause(v, valid_at=scope.valid_at, reference=scope.known_at, **_flags(scope))))
    return clauses


def select_assertions(scope: ScopePredicate, *, at: datetime | None = None):
    """Current heads matching the scope."""
    return select(ContextAssertion).where(*scope_clauses(scope, at=at)).order_by(ContextAssertion.id)


def select_versions(scope: ScopePredicate, *, at: datetime | None = None):
    """The versions recorded at `scope.known_at` (default: now) that match the scope: statement, lifecycle and
    interval as Eridani believed them then."""
    v = ContextAssertionVersion
    known = scope.known_at or at or now()
    return (select(v).join(ContextAssertion, ContextAssertion.id == v.assertion_id)
            .where(*_subject_clauses(scope), recorded(known),
                   lifecycle_clause(v, valid_at=scope.valid_at, reference=known, **_flags(scope)))
            .order_by(v.assertion_id))


def history(db, owner: str, assertion_id: str) -> list[ContextAssertionVersion]:
    return list(db.scalars(select(ContextAssertionVersion).where(
        ContextAssertionVersion.owner_id == owner, ContextAssertionVersion.assertion_id == assertion_id)
        .order_by(ContextAssertionVersion.version)))


# ---------------------------------------------------------------- versions

VERSIONED = ("statement", "lifecycle", "valid_from", "valid_to", "valid_precision")


def record_version(db, row: ContextAssertion, *, at: datetime | None = None, run_id: str | None = None,
                   decision_id: str | None = None, **changes) -> ContextAssertionVersion | None:
    """Apply head changes and append the matching version, closing the open one. No-op when nothing changes."""
    if unknown := set(changes) - set(VERSIONED):
        raise ValueError(f"unversioned fields {sorted(unknown)}")
    changes = {k: v for k, v in changes.items() if getattr(row, k) != v}
    if not changes:
        return None
    at = at or now()
    db.execute(update(ContextAssertionVersion).where(
        ContextAssertionVersion.assertion_id == row.id, ContextAssertionVersion.recorded_to.is_(None))
        .values(recorded_to=at))
    for key, value in changes.items():
        setattr(row, key, value)
    row.current_version += 1
    row.updated_at = at
    version = _version(row, at, run_id=run_id, decision_id=decision_id)
    db.add(version)
    db.flush()
    return version


def _version(row, at, *, run_id=None, decision_id=None):
    return ContextAssertionVersion(
        owner_id=row.owner_id, assertion_id=row.id, version=row.current_version, statement=row.statement,
        lifecycle=row.lifecycle, valid_from=row.valid_from, valid_to=row.valid_to,
        valid_precision=row.valid_precision, recorded_from=at, decision_id=decision_id, run_id=run_id)


def _subject(ref: EntityRef | None, fallback: tuple[str, str]) -> tuple[str, str]:
    if ref is None:
        return fallback
    return ref.entity_type, (ref.entity_id or "mention:" + normalized(ref.mention or ""))[:100]


def create(db, owner: str, candidate: CandidateAssertion, *, occurred_at: datetime, fallback_subject: tuple[str, str],
           run_id: str | None = None, decision_id: str | None = None, work_key: str | None = None,
           lifecycle: str = "current", at: datetime | None = None) -> ContextAssertion:
    """Insert an accepted candidate with version 1. `fallback_subject` names the subject when the candidate has
    none (for example `("source", source_id)`); `work_key` is a prior commitment's stable key. A candidate that
    repeats an interpretation the owner marked Wrong is stored already retracted, citing that decision. Attach
    evidence with `evidence.attach`."""
    at = at or now()
    start = candidate.effective_from(occurred_at)
    subject_type, subject_id = _subject(candidate.subject, fallback_subject)
    row = ContextAssertion(
        owner_id=owner, kind=candidate.kind, statement=candidate.statement, subject_type=subject_type,
        subject_id=subject_id, object_type=candidate.object.entity_type if candidate.object else None,
        object_id=(candidate.object.entity_id or candidate.object.mention)[:100] if candidate.object else None,
        predicate=candidate.key or candidate.kind, key=work_key, modality=candidate.modality,
        explicit=not candidate.inferred, valid_from=start.at, valid_precision=start.precision,
        valid_to=candidate.valid_to.at if candidate.valid_to else None, lifecycle=lifecycle, current_version=1,
        run_id=run_id, confidence=candidate.confidence,
        responsible=[r.model_dump(mode="json", exclude_none=True) for r in candidate.responsible],
        recipient=candidate.recipient.entity_id if candidate.recipient else None,
        due_at=candidate.due.at if candidate.due else None,
        due_precision=candidate.due.precision if candidate.due else None, created_at=at, updated_at=at)
    negative = negated_by(db, owner, row)
    if negative is not None:
        row.lifecycle, decision_id = "retracted", negative.id
    db.add(row)
    db.flush()
    db.add(_version(row, at, run_id=run_id, decision_id=decision_id))
    db.flush()
    return row


def active_decisions(owner: str):
    """Decisions not undone by a later compensating decision (undo rows themselves excluded)."""
    d, undo = ContextDecision, ContextDecision.__table__.alias("undo")
    return select(d).where(d.owner_id == owner, d.operation != "undo",
                           ~exists().where(undo.c.undoes_id == d.id))


def negated_by(db, owner: str, row: ContextAssertion) -> ContextDecision | None:
    """The active scoped negative (a Wrong decision) this interpretation repeats, if any."""
    d = ContextDecision
    for decision in db.scalars(active_decisions(owner).where(
            d.negative.is_(True), d.operation == "wrong", d.target_type == "assertion",
            d.before["subject_type"].astext == row.subject_type, d.before["subject_id"].astext == row.subject_id,
            d.before["kind"].astext == row.kind).order_by(d.seq)):
        if normalized(decision.before.get("statement", "")) == normalized(row.statement):
            return decision
    return None


# ---------------------------------------------------------------- keyed chains


def chain_identity(row: ContextAssertion) -> tuple[str, str, str] | None:
    """The single-valued slot this assertion fills, or None when it is not deterministically replaceable."""
    if row.predicate not in EXCLUSIVE_KEYS:
        return None
    if row.subject_type in WORK_TYPES and not row.subject_id.startswith("mention:"):
        return row.subject_type, row.subject_id, row.predicate
    if row.kind == "commitment" and row.key:
        return "commitment", row.key, row.predicate
    return None


def _chain_rows(db, owner, identity) -> list[ContextAssertion]:
    kind, ident, predicate = identity
    a = ContextAssertion
    match = (and_(a.kind == "commitment", a.key == ident) if kind == "commitment"
             else and_(a.subject_type == kind, a.subject_id == ident))
    rows = db.scalars(select(a).where(a.owner_id == owner, a.predicate == predicate, a.lifecycle.in_(CHAINED), match)
                      .order_by(a.id).with_for_update())
    return [r for r in rows if chain_identity(r) == identity]


def _intrinsic_end(db, row) -> datetime | None:
    """The end the assertion was stated with (version 1), before any chaining shortened it."""
    return db.scalar(select(ContextAssertionVersion.valid_to).where(
        ContextAssertionVersion.assertion_id == row.id, ContextAssertionVersion.version == 1))


def member(row) -> Member:
    return Member(row.id, row.statement, span(row.valid_from, row.valid_precision))


def rechain(db, owner: str, identity: tuple[str, str, str], *, at: datetime | None = None,
            run_id: str | None = None, decision_id: str | None = None) -> list[str]:
    """Recompute intervals, lifecycles and `supersedes_id` of one keyed chain from effective time alone.
    Returns the IDs whose head changed."""
    at = at or now()
    rows = {r.id: r for r in _chain_rows(db, owner, identity)}
    members = [member(r) for r in rows.values()]
    ordered = clusters(members)
    unknown = [m for m in members if not m.span.known]
    want: dict[str, dict] = {}
    previous = None
    for index, cluster in enumerate(ordered):
        groups = distinct(cluster)
        contested = len(groups) > 1
        later = ordered[index + 1] if index + 1 < len(ordered) else None
        if later is None:
            tail = set(groups)
            contested = contested or any(normalized(m.statement) not in tail for m in unknown)
        for m in cluster:
            if later is None:
                lifecycle, end = ("contested" if contested else "current"), _intrinsic_end(db, rows[m.id])
            else:
                start = min(n.span.start for n in later)
                end = start
                lifecycle = "superseded" if start <= at else "contested" if contested else "current"
            want[m.id] = {"lifecycle": lifecycle, "valid_to": end, "supersedes": previous}
        previous = min(cluster, key=lambda m: m.order).id
    tail = set(distinct(ordered[-1])) if ordered else set()
    for m in unknown:
        lifecycle = "current" if normalized(m.statement) in tail else "contested"
        want[m.id] = {"lifecycle": lifecycle, "valid_to": rows[m.id].valid_to, "supersedes": rows[m.id].supersedes_id}
    changed = []
    for ident, target in want.items():
        row = rows[ident]
        row.supersedes_id = target["supersedes"]
        if record_version(db, row, at=at, run_id=run_id, decision_id=decision_id, lifecycle=target["lifecycle"],
                          valid_to=target["valid_to"]):
            changed.append(ident)
    return changed


def settle(db, owner: str, *, at: datetime | None = None) -> list[str]:
    """Bring lifecycles up to date for keyed chains whose successor has since taken effect (maintenance)."""
    at = at or now()
    a = ContextAssertion
    rows = db.scalars(select(a).where(a.owner_id == owner, a.lifecycle.in_(LIVE), a.valid_to <= at,
                                      a.predicate.in_(EXCLUSIVE_KEYS)))
    changed = []
    for identity in sorted({i for r in rows if (i := chain_identity(r))}):
        changed += rechain(db, owner, identity, at=at)
    stale_briefs(db, owner, changed)
    return changed


# ---------------------------------------------------------------- briefs


def stale_briefs(db, owner: str, assertion_ids: Iterable[str]) -> int:
    """Mark briefs depending on any of the assertions stale, in the caller's transaction. A brief's
    `dependencies["assertions"]` is a list of IDs or an object keyed by ID."""
    ids = sorted(set(assertion_ids))
    if not ids:
        return 0
    b = ContextBrief
    result = db.execute(update(b).where(
        b.owner_id == owner, b.stale.is_(False), b.dependencies["assertions"].has_any(array(ids)))
        .values(stale=True, updated_at=now()).execution_options(synchronize_session="fetch"))
    return result.rowcount


# ---------------------------------------------------------------- views and requests


def view(db, row: ContextAssertion) -> AssertionView | None:
    """The engine view of an assertion, or None when it can no longer be shown with an exact quote (all its
    evidence was purged) or its stored shape is invalid."""
    refs = []
    for ev, seq in db.execute(select(ContextEvidence, ContextSegment.seq).join(
            ContextSegment, ContextSegment.id == ContextEvidence.segment_id).where(
            ContextEvidence.assertion_id == row.id, ContextEvidence.quote != "").order_by(ContextEvidence.created_at,
                                                                                         ContextEvidence.id)):
        try:
            refs.append(EvidenceRef(segment_seq=seq, quote=ev.quote, char_start=ev.char_start or 0,
                                    char_end=ev.char_end if ev.char_end is not None else len(ev.quote),
                                    role=ev.role, origin=ev.origin, source_revision_id=ev.revision_id))
        except ValidationError:
            continue
    subject = None
    if row.subject_type in ENTITY_TYPES:
        mention = row.subject_id.removeprefix("mention:") if row.subject_id.startswith("mention:") else None
        subject = EntityRef(entity_type=row.subject_type, entity_id=None if mention else row.subject_id,
                            mention=mention)
    try:
        return AssertionView(
            id=row.id, revision=row.current_version, lifecycle=row.lifecycle, kind=row.kind,
            statement=row.statement, subject=subject,
            object=EntityRef(entity_type=row.object_type, entity_id=row.object_id)
            if row.object_type in ENTITY_TYPES and row.object_id else None,
            key=row.predicate if row.predicate != row.kind else None, modality=row.modality,
            inferred=not row.explicit,
            valid_from=TimeBound(at=row.valid_from or row.created_at,
                                 precision=row.valid_precision if row.valid_from else "unknown"),
            valid_to=TimeBound(at=row.valid_to) if row.valid_to and row.valid_from and row.valid_to > row.valid_from
            else None,
            responsible=[EntityRef.model_validate(r) for r in row.responsible],
            recipient=EntityRef(entity_type="actor", entity_id=row.recipient) if row.recipient else None,
            due=TimeBound(at=row.due_at, precision=row.due_precision or "day") if row.due_at else None,
            evidence=refs, confidence=row.confidence, recorded_at=row.updated_at,
            source_revision_id=refs[0].source_revision_id if refs else None)
    except ValidationError:
        return None


def to_decision(row: ContextDecision) -> Decision:
    return Decision(id=row.id, seq=row.seq, operation=row.operation, origin=row.origin, target_type=row.target_type,
                    target_id=row.target_id, expected_revision=row.expected_revision,
                    statement=(row.before or {}).get("statement"), reason=row.reason, recorded_at=row.recorded_at)


def reconcile_request(db, owner: str, changed_ids: Iterable[str], *, limit: int = 200
                      ) -> tuple[ReconcileRequest | None, dict[str, str]]:
    """The engine request for newly applied assertions: the changed ones, every chained assertion about the same
    subjects or work keys (superseded included, so late arrivals can split intervals), and the active negative
    decisions about those subjects. Also returns `work_keys` (assertion ID -> stable commitment key)."""
    a = ContextAssertion
    changed = list(db.scalars(select(a).where(a.owner_id == owner, a.id.in_(list(changed_ids)),
                                              a.lifecycle.in_(LIVE)).order_by(a.id)))
    if not changed:
        return None, {}
    subjects = {(r.subject_type, r.subject_id) for r in changed}
    keys = {r.key for r in changed if r.kind == "commitment" and r.key}
    match = [and_(a.subject_type == t, a.subject_id == i) for t, i in sorted(subjects)]
    if keys:
        match.append(and_(a.kind == "commitment", a.key.in_(sorted(keys))))
    ids = {r.id for r in changed}
    related = [r for r in db.scalars(select(a).where(a.owner_id == owner, a.lifecycle.in_(CHAINED), or_(*match))
                                     .order_by(a.updated_at.desc(), a.id).limit(limit)) if r.id not in ids]
    d = ContextDecision
    negatives = db.scalars(active_decisions(owner).where(
        d.negative.is_(True), d.target_type == "assertion",
        or_(*[and_(d.before["subject_type"].astext == t, d.before["subject_id"].astext == i)
              for t, i in sorted(subjects)])).order_by(d.seq))
    changed_views = [v for r in changed if (v := view(db, r))]
    if not changed_views:
        return None, {}
    request = ReconcileRequest(owner_id=owner, changed=changed_views,
                               related=[v for r in related if (v := view(db, r))],
                               decisions=[to_decision(x) for x in negatives])
    work_keys = {r.id: r.key for r in [*changed, *related] if r.kind == "commitment" and r.key}
    return request, work_keys


# ---------------------------------------------------------------- applying engine ops


def semantic_supersede_enabled(db, owner: str) -> bool:
    state = db.get(ContextOwnerState, owner)
    return bool(state and (state.settings or {}).get("semantic_supersede"))


def multi_valued(row: ContextAssertion) -> bool:
    return row.kind in MULTI_VALUED_KINDS or row.predicate in MULTI_VALUED_KEYS


def apply_ops(db, owner: str, ops: Iterable[ReconcileOp], *, run_id: str | None = None,
              semantic_supersede: bool | None = None, at: datetime | None = None) -> dict:
    """Validate and apply an engine's ops; the engine proposes, this enforces policy:

    - ops computed against other revisions are skipped as `stale` (requeue the run);
    - keyed (exclusive) slots are recomputed by `rechain`, whatever pairwise op triggered it;
    - `equivalent` merges distinct evidence into the earliest assertion and retracts the duplicates, but only
      between tied values in a keyed chain (a later restatement is a new interval, not a duplicate);
    - semantic `supersede` needs certain temporal order (otherwise it is a conflict) and is recorded in shadow
      (the newer assertion becomes `shadow`, `supersedes_id` set, current value unchanged) unless the owner's
      `semantic_supersede` setting enables it; multi-valued kinds never supersede or conflict;
    - `conflict` marks live targets `contested`; `link_change` is left to identity linking.
    """
    from . import evidence

    at = at or now()
    if semantic_supersede is None:
        semantic_supersede = semantic_supersede_enabled(db, owner)
    ops = list(ops)
    ids = sorted({t for op in ops for t in op.targets})
    rows = {r.id: r for r in db.scalars(select(ContextAssertion).where(
        ContextAssertion.owner_id == owner, ContextAssertion.id.in_(ids)).order_by(ContextAssertion.id)
        .with_for_update())}
    summary = defaultdict(list)
    chains, touched = set(), set()

    def fresh(op):
        return all(t in rows and op.target_revisions.get(t, rows[t].current_version) == rows[t].current_version
                   for t in op.targets)

    for op in ops:
        if op.op == "no_change":
            continue
        if op.op == "link_change":
            summary["deferred"].append(op.targets)
            continue
        if not fresh(op):
            summary["stale"].append(op.targets)
            continue
        targets = [rows[t] for t in op.targets]
        identities = {chain_identity(r) for r in targets}
        keyed = None not in identities and len(identities) == 1
        if keyed:
            chains.add(identities.pop())
        if op.op == "equivalent":
            live = sorted((r for r in targets if r.lifecycle in LIVE), key=lambda r: member(r).order)
            if keyed and not all(not precedes(member(x).span, member(y).span) and
                                 not precedes(member(y).span, member(x).span) for x in live for y in live):
                continue
            for duplicate in live[1:]:
                evidence.merge(db, owner, live[0], duplicate)
                record_version(db, duplicate, at=at, run_id=run_id, lifecycle="retracted")
                touched.update((live[0].id, duplicate.id))
                summary["merged"].append([live[0].id, duplicate.id])
        elif keyed:
            continue
        elif any(multi_valued(r) for r in targets):
            summary["refused"].append(op.targets)
        elif op.op == "conflict" or not precedes(member(targets[0]).span, member(targets[1]).span):
            for r in targets:
                if r.lifecycle == "current" and record_version(db, r, at=at, run_id=run_id, lifecycle="contested"):
                    touched.add(r.id)
            summary["contested"].append(op.targets)
        else:
            old, new = targets
            if old.lifecycle not in LIVE or new.lifecycle not in LIVE:
                continue
            new.supersedes_id = old.id
            if semantic_supersede:
                end = new.valid_from if old.valid_to is None else min(old.valid_to, new.valid_from)
                record_version(db, old, at=at, run_id=run_id, valid_to=end,
                               lifecycle="superseded" if new.valid_from <= at else old.lifecycle)
                summary["superseded"].append(op.targets)
            else:
                record_version(db, new, at=at, run_id=run_id, lifecycle="shadow")
                summary["shadow"].append(op.targets)
            touched.update(op.targets)
    for identity in sorted(chains):
        touched.update(rechain(db, owner, identity, at=at, run_id=run_id))
    stale_briefs(db, owner, touched)
    summary["changed"] = sorted(touched)
    return dict(summary)

