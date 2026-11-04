"""Human, agent and system decisions about context: append-only `context_decisions` rows with their effects.

Every operation checks the target's expected revision under a row lock, allocates the owner's next decision
`seq`, applies its effect in the same transaction and marks dependent briefs stale. Origin is derived, never
claimed: `human` only when `routing.human_command` accepts the command ID (which also requires that no bot
credential is active); any other command is `agent`; work without a command is `system`. A caller may lower
human to agent, never raise anything to human.

- Wrong retracts the interpretation and records a scoped negative, so re-extraction cannot revive it.
- Changed adds a successor assertion with the supplied effective time and supersedes the old one.
- Undo appends a compensating decision (`undoes_id`) and reverses the effect; it refuses when a later decision
  on the same target depends on the one being undone.
- merge, unmerge and no_merge only record the decision: identity linking applies merges and reads negatives.

Contested context and consequential link ambiguity become Questions (`question_items`, kind `context`), answered
through `resolve_question` and deferred through `defer_question`.
"""

import hashlib
import unicodedata
from datetime import datetime, timedelta

from sqlalchemy import func, select
from sqlalchemy.dialects.postgresql import insert

from .. import routing
from ..bot_access import current_id
from ..domain import DomainError, emit
from ..models import now
from . import temporal
from .models import (
    ContextAssertion,
    ContextDecision,
    ContextEntityLink,
    ContextEvidence,
    ContextOwnerState,
    ContextProposal,
    ContextSegment,
    ContextSourceRevision,
)
from .schema import EXCLUSIVE_KEYS

PROPOSAL_DISPOSITIONS = {"approve": "approved", "dismiss": "dismissed", "later": "later"}
QUESTION_KINDS = ("commitment", "decision")


# ---------------------------------------------------------------- recording


def next_decision_seq(db, owner: str) -> int:
    """The owner's next decision seq, allocated under the owner-state row lock (created when absent)."""
    db.execute(insert(ContextOwnerState).values(owner_id=owner).on_conflict_do_nothing(index_elements=["owner_id"]))
    state = db.scalar(select(ContextOwnerState).where(ContextOwnerState.owner_id == owner).with_for_update()
                      .execution_options(populate_existing=True))
    state.decision_seq += 1
    state.updated_at = now()
    db.flush()
    return state.decision_seq


def origin_of(command_id: str | None, requested: str | None = None) -> str:
    if not command_id:
        return "agent" if current_id() else "system"
    derived = "human" if routing.human_command(command_id) else "agent"
    return "agent" if derived == "human" and requested == "agent" else derived


def record(db, owner: str, operation: str, target_type: str, target_id: str, *, command_id: str | None,
           principal: str, expected_revision: int | None = None, reason: str = "", negative: bool = False,
           before: dict | None = None, after: dict | None = None, effective_at: datetime | None = None,
           undoes_id: str | None = None, origin: str | None = None) -> ContextDecision:
    row = ContextDecision(
        owner_id=owner, seq=next_decision_seq(db, owner), operation=operation, target_type=target_type,
        target_id=target_id, expected_revision=expected_revision, reason=reason, negative=negative,
        origin=origin_of(command_id, origin), principal=principal or owner, command_id=command_id,
        before=before or {}, after=after or {}, effective_at=effective_at, undoes_id=undoes_id)
    db.add(row)
    db.flush()
    emit(db, owner, "context.changed", target_id)
    return row


def _conflict(message="This changed since you loaded it. Refresh and try again."):
    return DomainError("REVISION_CONFLICT", message, 409)


def _owned(db, model, ident, owner):
    row = db.scalar(select(model).where(model.id == ident, model.owner_id == owner).with_for_update())
    if row is None:
        raise DomainError("NOT_FOUND", "That context item is no longer available.", 404)
    return row


def _assertion(db, owner, assertion_id, expected_revision):
    row = _owned(db, ContextAssertion, assertion_id, owner)
    if row.current_version != expected_revision:
        raise _conflict()
    return row


def _iso(value):
    return value.isoformat() if value else None


def snapshot(row: ContextAssertion) -> dict:
    return {"version": row.current_version, "lifecycle": row.lifecycle, "statement": row.statement,
            "kind": row.kind, "subject_type": row.subject_type, "subject_id": row.subject_id,
            "predicate": row.predicate, "valid_from": _iso(row.valid_from), "valid_to": _iso(row.valid_to),
            "valid_precision": row.valid_precision, "supersedes_id": row.supersedes_id}


def _rechain(db, owner, row, decision):
    identity = temporal.chain_identity(row)
    return temporal.rechain(db, owner, identity, decision_id=decision.id) if identity else []


# ---------------------------------------------------------------- assertions


def wrong(db, owner: str, assertion_id: str, expected_revision: int, *, command_id: str | None, principal: str,
          reason: str = "") -> ContextDecision:
    """Retract an interpretation and record a scoped negative (same subject, kind and statement)."""
    row = _assertion(db, owner, assertion_id, expected_revision)
    if row.lifecycle == "retracted":
        raise _conflict("This is already marked wrong.")
    before = snapshot(row)
    decision = record(db, owner, "wrong", "assertion", row.id, command_id=command_id, principal=principal,
                      expected_revision=expected_revision, reason=reason, negative=True, before=before)
    temporal.record_version(db, row, decision_id=decision.id, lifecycle="retracted")
    changed = [row.id, *_rechain(db, owner, row, decision)]
    decision.after = {"version": row.current_version, "lifecycle": row.lifecycle}
    temporal.stale_briefs(db, owner, changed)
    return decision


def changed(db, owner: str, assertion_id: str, expected_revision: int, *, statement: str,
            effective_at: datetime | None = None, precision: str = "instant", due_at: datetime | None = None,
            command_id: str | None, principal: str, reason: str = "") -> ContextDecision:
    """Record that the value changed: a successor assertion effective from `effective_at` (default: now, as
    stated by the decision) supersedes the old one, which keeps its history."""
    old = _assertion(db, owner, assertion_id, expected_revision)
    if old.lifecycle not in temporal.LIVE:
        raise _conflict("Only current context can be changed.")
    statement = statement.strip()
    if not statement:
        raise DomainError("INVALID_ARGUMENT", "Say what it changed to.")
    at = now()
    start = effective_at or at
    if old.valid_from and start <= old.valid_from:
        raise DomainError("INVALID_ARGUMENT", "A change takes effect after the value it replaces.")
    before = snapshot(old)
    decision = record(db, owner, "changed", "assertion", old.id, command_id=command_id, principal=principal,
                      expected_revision=expected_revision, reason=reason, before=before, effective_at=start)
    new = ContextAssertion(
        owner_id=owner, kind=old.kind, statement=statement, subject_type=old.subject_type,
        subject_id=old.subject_id, object_type=old.object_type, object_id=old.object_id, predicate=old.predicate,
        key=old.key, modality="asserted", explicit=True, valid_from=start, valid_precision=precision,
        lifecycle="current", current_version=1, supersedes_id=old.id, responsible=list(old.responsible or []),
        recipient=old.recipient, due_at=due_at if due_at is not None else old.due_at,
        due_precision=old.due_precision, created_at=at, updated_at=at)
    db.add(new)
    db.flush()
    db.add(temporal._version(new, at, decision_id=decision.id))
    for ev in db.scalars(select(ContextEvidence).where(ContextEvidence.assertion_id == old.id,
                                                       ContextEvidence.quote != "")):
        db.add(ContextEvidence(owner_id=owner, assertion_id=new.id, assertion_version=1, source_id=ev.source_id,
                               revision_id=ev.revision_id, segment_id=ev.segment_id, quote=ev.quote,
                               char_start=ev.char_start, char_end=ev.char_end, role=ev.role, origin="derived",
                               derivation={"decision": decision.id, "from": ev.id}))
    db.flush()
    touched = [old.id, new.id]
    if temporal.chain_identity(old):
        touched += _rechain(db, owner, old, decision)
    else:
        end = start if old.valid_to is None else min(old.valid_to, start)
        temporal.record_version(db, old, at=at, decision_id=decision.id, valid_to=end,
                                lifecycle="superseded" if start <= at else old.lifecycle)
    decision.after = {"successor_id": new.id, "version": old.current_version}
    temporal.stale_briefs(db, owner, touched)
    return decision


def confirm(db, owner: str, assertion_id: str, expected_revision: int, *, command_id: str | None, principal: str,
            reason: str = "") -> ContextDecision:
    """Endorse an interpretation; a contested one outside a keyed chain becomes current."""
    row = _assertion(db, owner, assertion_id, expected_revision)
    if row.lifecycle not in temporal.LIVE:
        raise _conflict("Only current context can be confirmed.")
    decision = record(db, owner, "confirm", "assertion", row.id, command_id=command_id, principal=principal,
                      expected_revision=expected_revision, reason=reason, before=snapshot(row))
    if row.lifecycle == "contested" and not temporal.chain_identity(row):
        temporal.record_version(db, row, decision_id=decision.id, lifecycle="current")
        temporal.stale_briefs(db, owner, [row.id])
    decision.after = {"version": row.current_version, "lifecycle": row.lifecycle}
    return decision


# ---------------------------------------------------------------- links and identity


def _link(db, owner, link_id, expected_revision):
    row = _owned(db, ContextEntityLink, link_id, owner)
    if row.revision != expected_revision:
        raise _conflict()
    return row


def _link_state(row):
    return {"revision": row.revision, "status": row.status, "target_type": row.target_type,
            "target_id": row.target_id, "mention": row.mention, "role": row.role,
            "subject_type": row.subject_type, "subject_id": row.subject_id, "decision_id": row.decision_id}


def _set_link(db, owner, row, decision, **values):
    for key, value in values.items():
        setattr(row, key, value)
    row.revision += 1
    row.decision_id = decision.id
    row.updated_at = now()
    decision.after = _link_state(row)
    if row.assertion_id:
        temporal.stale_briefs(db, owner, [row.assertion_id])


def confirm_link(db, owner: str, link_id: str, expected_revision: int, *, command_id: str | None, principal: str,
                 target_type: str | None = None, target_id: str | None = None, reason: str = "") -> ContextDecision:
    row = _link(db, owner, link_id, expected_revision)
    target_type, target_id = target_type or row.target_type, target_id or row.target_id
    if not (target_type and target_id):
        raise DomainError("INVALID_ARGUMENT", "Choose who or what this refers to.")
    decision = record(db, owner, "confirm_link", "link", row.id, command_id=command_id, principal=principal,
                      expected_revision=expected_revision, reason=reason, before=_link_state(row))
    _set_link(db, owner, row, decision, status="confirmed", target_type=target_type, target_id=target_id)
    return decision


def mention_key(mention: str) -> str:
    """Identity's mention normal form: NFKC, casefolded, single-spaced."""
    return " ".join(unicodedata.normalize("NFKC", mention or "").casefold().split())


def _link_scope(db, row) -> str | None:
    """The source a segment-level link belongs to; other links are rejected everywhere (None)."""
    if row.subject_type != "segment":
        return None
    return db.scalar(select(ContextSourceRevision.source_id).join(
        ContextSegment, ContextSegment.revision_id == ContextSourceRevision.id).where(
        ContextSegment.id == row.subject_id))


def reject_link(db, owner: str, link_id: str, expected_revision: int, *, command_id: str | None, principal: str,
                reason: str = "") -> ContextDecision:
    """A durable negative in identity's shape (`identity.rejected_links` reads it): target
    `<entity_type>:<entity_id>`, before {mention, scope, entity_type, entity_id, link_id} plus the link state."""
    row = _link(db, owner, link_id, expected_revision)
    if not (row.target_type and row.target_id):
        raise DomainError("INVALID_ARGUMENT", "There is no suggested match to reject.")
    before = {**_link_state(row), "mention": mention_key(row.mention), "scope": _link_scope(db, row),
              "entity_type": row.target_type, "entity_id": row.target_id, "link_id": row.id}
    decision = record(db, owner, "reject_link", "link", f"{row.target_type}:{row.target_id}",
                      command_id=command_id, principal=principal, expected_revision=expected_revision,
                      reason=reason, negative=True, before=before)
    _set_link(db, owner, row, decision, status="rejected")
    return decision


def _pair(a: str, b: str) -> str:
    if a == b:
        raise DomainError("INVALID_ARGUMENT", "Choose two different people.")
    return ":".join(sorted((a, b)))


def no_merge(db, owner: str, actor_a: str, actor_b: str, *, command_id: str | None, principal: str,
             reason: str = "") -> ContextDecision:
    return record(db, owner, "no_merge", "actor_pair", _pair(actor_a, actor_b), command_id=command_id,
                  principal=principal, reason=reason, negative=True, before={"actors": sorted((actor_a, actor_b))})


def merge(db, owner: str, winner_id: str, loser_id: str, *, command_id: str | None, principal: str,
          merge_id: str | None = None, reason: str = "") -> ContextDecision:
    """Identity's shape: target the loser, after {merge_id, winner}. Identity performs the merge."""
    _pair(winner_id, loser_id)
    return record(db, owner, "merge", "actor", loser_id, command_id=command_id, principal=principal, reason=reason,
                  after={"merge_id": merge_id, "winner": winner_id})


def unmerge(db, owner: str, merge_id: str, merge_decision_id: str, *, command_id: str | None, principal: str,
            reason: str = "") -> ContextDecision:
    """Identity's shape: target the ContextActorMerge, `undoes_id` the merge decision (which stops being live)."""
    d = ContextDecision
    if db.scalar(select(d.id).where(d.owner_id == owner, d.undoes_id == merge_decision_id)):
        raise _conflict("This merge was already undone.")
    _owned(db, ContextDecision, merge_decision_id, owner)
    return record(db, owner, "unmerge", "actor_merge", merge_id, command_id=command_id, principal=principal,
                  reason=reason, undoes_id=merge_decision_id)


# ---------------------------------------------------------------- proposals and deferral


def _proposal_decision(db, owner, operation, proposal_id, expected_revision, *, command_id, principal, reason="",
                       until=None):
    row = _owned(db, ContextProposal, proposal_id, owner)
    if row.revision != expected_revision:
        raise _conflict("This proposal changed. Review the latest version.")
    if row.disposition in ("published", "superseded"):
        raise _conflict("This proposal is no longer open.")
    before = {"revision": row.revision, "disposition": row.disposition}
    decision = record(db, owner, operation, "proposal", row.id, command_id=command_id, principal=principal,
                      expected_revision=expected_revision, reason=reason, before=before, effective_at=until)
    row.disposition = PROPOSAL_DISPOSITIONS[operation]
    row.revision += 1
    row.updated_at = now()
    decision.after = {"revision": row.revision, "disposition": row.disposition}
    return decision


def approve(db, owner, proposal_id, expected_revision, *, command_id, principal, reason=""):
    """Record approval of this proposal revision (publication enqueues the write in the same transaction)."""
    return _proposal_decision(db, owner, "approve", proposal_id, expected_revision, command_id=command_id,
                              principal=principal, reason=reason)


def dismiss(db, owner, proposal_id, expected_revision, *, command_id, principal, reason=""):
    return _proposal_decision(db, owner, "dismiss", proposal_id, expected_revision, command_id=command_id,
                              principal=principal, reason=reason)


def later(db, owner, proposal_id, expected_revision, *, command_id, principal, until=None, reason=""):
    return _proposal_decision(db, owner, "later", proposal_id, expected_revision, command_id=command_id,
                              principal=principal, reason=reason, until=until)


# ---------------------------------------------------------------- undo


def undo(db, owner: str, decision_id: str, *, command_id: str | None, principal: str,
         reason: str = "") -> ContextDecision:
    """Append a compensating decision and reverse the effect of `decision_id`."""
    original = _owned(db, ContextDecision, decision_id, owner)
    if original.operation in ("undo", "unmerge"):
        raise DomainError("INVALID_ARGUMENT", "Make the decision again instead of undoing an undo.")
    if original.operation == "merge":
        raise DomainError("INVALID_ARGUMENT", "Unmerge these people instead.")
    d = ContextDecision
    if db.scalar(select(d.id).where(d.owner_id == owner, d.undoes_id == original.id)):
        raise _conflict("This decision was already undone.")
    link_id = original.before.get("link_id") or original.target_id
    targets = {original.target_id, original.after.get("successor_id"),
               link_id if original.target_type == "link" else None} - {None}
    later_ids = set(db.scalars(temporal.active_decisions(owner).where(d.seq > original.seq, d.target_id.in_(targets))))
    if later_ids:
        raise _conflict("A later decision depends on this one. Undo that first.")
    decision = record(db, owner, "undo", original.target_type, original.target_id, command_id=command_id,
                      principal=principal, reason=reason, undoes_id=original.id, before=original.after,
                      after=original.before)
    before = original.before
    touched = []
    if original.target_type == "assertion":
        row = _owned(db, ContextAssertion, original.target_id, owner)
        if original.operation == "changed":
            new = _owned(db, ContextAssertion, original.after["successor_id"], owner)
            temporal.record_version(db, new, decision_id=decision.id, lifecycle="retracted")
            touched.append(new.id)
        if original.operation in ("wrong", "changed", "confirm"):
            temporal.record_version(db, row, decision_id=decision.id, lifecycle=before["lifecycle"],
                                    valid_to=datetime.fromisoformat(before["valid_to"]) if before["valid_to"] else None)
            touched += [row.id, *_rechain(db, owner, row, decision)]
    elif original.target_type == "link":
        row = _owned(db, ContextEntityLink, link_id, owner)
        _set_link(db, owner, row, decision, status=before["status"], target_type=before["target_type"],
                  target_id=before["target_id"])
    elif original.target_type == "proposal":
        row = _owned(db, ContextProposal, original.target_id, owner)
        if row.revision != original.after.get("revision") or row.disposition == "published":
            raise _conflict("This proposal moved on; it cannot be undone here.")
        row.disposition = before["disposition"]
        row.revision += 1
        row.updated_at = now()
    temporal.stale_briefs(db, owner, touched)
    return decision


# ---------------------------------------------------------------- Questions inbox (kind "context")


def _key(*parts: str) -> str:
    return "context:" + hashlib.sha256("|".join(parts).encode()).hexdigest()[:32]


def _deferral(db, owner, key):
    d = ContextDecision
    rows = list(db.scalars(temporal.active_decisions(owner).where(d.target_type == "question", d.target_id == key)
                           .order_by(d.seq)))
    deferred = [r.effective_at for r in rows if r.operation == "later" and r.effective_at]
    return len(rows), (deferred[-1] if deferred else None)


def _quotes(db, ids):
    e = ContextEvidence
    quotes = {}
    for r in db.scalars(select(e).where(e.assertion_id.in_(ids), e.quote != "", e.role == "support")
                        .order_by(e.created_at, e.id)):
        quotes.setdefault(r.assertion_id, [])
        if len(quotes[r.assertion_id]) < 3:
            quotes[r.assertion_id].append({"source_id": r.source_id, "revision_id": r.revision_id, "quote": r.quote})
    return quotes


def _consequential(rows) -> bool:
    """Ask only when resolution affects work or decisions; other contested context stays inspectable."""
    return any(r.kind in QUESTION_KINDS or r.predicate in EXCLUSIVE_KEYS or r.subject_type in temporal.WORK_TYPES
               for r in rows)


def question_items(db, owner: str) -> list[dict]:
    """Open context Questions for `review_questions.items`: consequential contested groups and unresolved
    responsible-person links on current commitments."""
    a = ContextAssertion
    groups: dict[tuple, list] = {}
    for row in db.scalars(select(a).where(a.owner_id == owner, a.lifecycle == "contested")
                          .order_by(a.valid_from, a.id)):
        groups.setdefault((row.subject_type, row.subject_id, row.predicate), []).append(row)
    items = []
    for (subject_type, subject_id, predicate), rows in sorted(groups.items()):
        if not _consequential(rows):
            continue
        key = _key("contested", subject_type, subject_id, predicate)
        count, until = _deferral(db, owner, key)
        quotes = _quotes(db, [r.id for r in rows])
        items.append(_item(
            key, 1 + count + sum(r.current_version for r in rows), until, min(r.updated_at for r in rows),
            question="These meeting notes disagree about "
                     + ("the due date" if predicate == "due_date" else "who is responsible"
                        if predicate == "responsible" else "a " + rows[0].kind) + ". Which is right?",
            candidates=[{"assertion_id": r.id, "revision": r.current_version, "statement": r.statement,
                         "valid_from": r.valid_from, "evidence": quotes.get(r.id, [])} for r in rows],
            evidence={"subject_type": subject_type, "subject_id": subject_id, "predicate": predicate},
            answer_args={"assertion_ids": [r.id for r in rows], "actions": ["keep", "both"]
                         if predicate not in EXCLUSIVE_KEYS else ["keep"]}))
    link = ContextEntityLink
    for row, assertion in db.execute(select(link, a).join(a, a.id == link.assertion_id).where(
            link.owner_id == owner, link.role == "responsible", link.status == "unresolved",
            a.kind == "commitment", a.lifecycle.in_(temporal.LIVE)).order_by(link.created_at, link.id)):
        key = _key("link", row.id)
        count, until = _deferral(db, owner, key)
        if any(r.operation == "dismiss" for r in db.scalars(temporal.active_decisions(owner).where(
                ContextDecision.target_type == "question", ContextDecision.target_id == key))):
            continue
        items.append(_item(
            key, row.revision + count, until, row.created_at,
            question=f"Who is “{row.mention}” in “{assertion.statement}”?",
            candidates=[], evidence={"mention": row.mention, "assertion_id": assertion.id,
                                     "quotes": _quotes(db, [assertion.id]).get(assertion.id, [])},
            answer_args={"link_id": row.id, "actions": ["link", "dismiss"]}))
    return items


def _item(key, revision, until, created_at, *, question, candidates, evidence, answer_args):
    return {"key": key, "kind": "context", "revision": revision,
                "status": "deferred" if until and until > now() else "pending", "question": question,
                "candidates": candidates, "evidence": evidence, "source_id": key, "deferred_until": until,
                "created_at": created_at, "answer_tool": "context.resolve",
                "answer_args": {"question_key": key, "expected_revision": revision, **answer_args}}


def _question(db, owner, key, expected_revision):
    item = next((q for q in question_items(db, owner) if q["key"] == key), None)
    if item is None or item["revision"] != expected_revision:
        raise DomainError("STALE_REVIEW", "This question changed. Refresh Questions before answering.", 409)
    return item


def defer_question(db, owner: str, key: str, expected_revision: int, delta: timedelta, *,
                   command_id: str | None = None, principal: str | None = None) -> dict:
    _question(db, owner, key, expected_revision)
    until = now() + delta
    record(db, owner, "later", "question", key, command_id=command_id, principal=principal or current_id() or owner,
           expected_revision=expected_revision, effective_at=until)
    return {"question_key": key, "status": "deferred", "revision": _question_revision(db, owner, key),
            "deferred_until": until.isoformat()}


def _question_revision(db, owner, key):
    return next((q["revision"] for q in question_items(db, owner) if q["key"] == key), None)


def resolve_question(db, owner: str, *, question_key: str, expected_revision: int, action: str,
                     command_id: str | None, principal: str, assertion_id: str | None = None,
                     entity_type: str = "actor", entity_id: str | None = None) -> dict:
    """Answer a context Question. Contested: `keep` one assertion (the others are marked wrong) or, outside
    exclusive slots, `both` (each confirmed). Link: `link` to an entity, or `dismiss` the question."""
    item = _question(db, owner, question_key, expected_revision)
    args = item["answer_args"]
    if action not in args["actions"]:
        raise DomainError("INVALID_ARGUMENT", "That answer does not fit this question.")
    decisions = []
    if action == "keep":
        rows = {c["assertion_id"]: c["revision"] for c in item["candidates"]}
        if assertion_id not in rows:
            raise DomainError("INVALID_ARGUMENT", "Choose one of the listed statements.")
        for other, revision in rows.items():
            if other != assertion_id:
                decisions.append(wrong(db, owner, other, revision, command_id=command_id, principal=principal,
                                       reason="Question answered"))
        kept = db.get(ContextAssertion, assertion_id)
        decisions.append(confirm(db, owner, assertion_id, kept.current_version, command_id=command_id,
                                 principal=principal, reason="Question answered"))
    elif action == "both":
        for c in item["candidates"]:
            decisions.append(confirm(db, owner, c["assertion_id"], c["revision"], command_id=command_id,
                                     principal=principal, reason="Question answered"))
    elif action == "link":
        if not entity_id:
            raise DomainError("INVALID_ARGUMENT", "Choose who this refers to.")
        row = db.get(ContextEntityLink, args["link_id"])
        decisions.append(confirm_link(db, owner, row.id, row.revision, command_id=command_id, principal=principal,
                                      target_type=entity_type, target_id=entity_id))
    else:
        decisions.append(record(db, owner, "dismiss", "question", question_key, command_id=command_id,
                                principal=principal, expected_revision=expected_revision))
    return {"question_key": question_key, "status": "resolved", "decision_ids": [d.id for d in decisions]}


def watermark(db, owner: str) -> int:
    """The highest decision seq so far (`RunMeta.decision_watermark`)."""
    return db.scalar(select(func.coalesce(func.max(ContextDecision.seq), 0)).where(
        ContextDecision.owner_id == owner)) or 0
