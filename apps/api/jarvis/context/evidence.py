"""Evidence: exact spans that support or contradict an assertion, and the survival rule.

Roles are `support` and `contradiction`; origins are `independent` (read from a source) and `derived` (computed
from other evidence, `derivation` naming what it relied on). Survival (plan section 5): an assertion is
supported while at least one support group survives. Each live independent support span is its own group;
spans sharing `derivation["all_of"]` form one conjunctive group that needs every member. Contradictions and
derived spans are kept as provenance but never corroborate. Briefs and generated answers are never evidence.
An assertion the owner stated through a human Changed decision is supported by that decision.

Deleting a source (`propagate_source_deletion`) keeps assertions with surviving alternative support, retracts
the ones left unsupported, reports partially supported conjunctive ones for revalidation, scrubs the source's
quoted text from evidence (leaving a non-content tombstone) and marks dependent briefs stale.

Revising a source (`supersede_revisions`, called once a newer revision is completely extracted) applies the same
rule to what the meeting no longer says: evidence quoted from the source's earlier revisions stops counting
(`derivation["superseded_by"]`, the quote kept as provenance). Statements the new revision repeats carry its own
spans; the rest survive only on other sources or a human Changed decision, and are retracted otherwise.

Such support-loss retractions are recorded with neither a decision nor a run. When live support comes back (a
later revision says it again, or a deleted source is restored with new content), `revive` restores the assertion
to the lifecycle it had before, with a new version. A human Wrong on it, an active durable negative matching it,
or a retraction with any other cause (a decision, a reconcile merge) keeps it retracted.
"""

from collections import defaultdict
from collections.abc import Iterable

from sqlalchemy import exists, select

from ..models import now
from . import temporal
from .models import (
    ContextAssertion,
    ContextAssertionVersion,
    ContextDecision,
    ContextEvidence,
    ContextSegment,
    ContextSource,
    ContextSourceRevision,
)
from .schema import EvidenceRef


def attach(db, owner: str, assertion: ContextAssertion, refs: Iterable[EvidenceRef], *, revision_id: str,
           derivation: dict | None = None) -> list[ContextEvidence]:
    """Store exact quotes for the assertion's current version. A ref's own `source_revision_id` wins over
    `revision_id`. Quotes must match the segment text exactly; repeated spans are stored once."""
    rows = []
    for ref in refs:
        rev = db.scalar(select(ContextSourceRevision).where(
            ContextSourceRevision.owner_id == owner, ContextSourceRevision.id == (ref.source_revision_id or revision_id)))
        if rev is None:
            raise ValueError("evidence cites an unknown source revision")
        segment = db.scalar(select(ContextSegment).where(
            ContextSegment.revision_id == rev.id, ContextSegment.seq == ref.segment_seq))
        if segment is None or not ref.matches(segment.text):
            raise ValueError(f"evidence quote is not exact in segment {ref.segment_seq}")
        e = ContextEvidence
        if db.scalar(select(e.id).where(e.assertion_id == assertion.id, e.segment_id == segment.id,
                                        e.char_start == ref.char_start, e.char_end == ref.char_end,
                                        e.role == ref.role, e.origin == ref.origin)):
            continue
        row = ContextEvidence(owner_id=owner, assertion_id=assertion.id, assertion_version=assertion.current_version,
                              source_id=rev.source_id, revision_id=rev.id, segment_id=segment.id, quote=ref.quote,
                              char_start=ref.char_start, char_end=ref.char_end, role=ref.role, origin=ref.origin,
                              derivation=derivation or {})
        db.add(row)
        rows.append(row)
    db.flush()
    return rows


def merge(db, owner: str, canonical: ContextAssertion, duplicate: ContextAssertion) -> int:
    """Copy the duplicate's spans onto the canonical assertion (as provenance; whether they corroborate is decided
    by source independence at read time). Returns how many spans were new."""
    e = ContextEvidence
    have = {(r.source_id, r.segment_id, r.char_start, r.char_end, r.role)
            for r in db.scalars(select(e).where(e.assertion_id == canonical.id))}
    added = 0
    for r in db.scalars(select(e).where(e.assertion_id == duplicate.id).order_by(e.created_at, e.id)):
        span = (r.source_id, r.segment_id, r.char_start, r.char_end, r.role)
        if span in have:
            continue
        have.add(span)
        db.add(ContextEvidence(owner_id=owner, assertion_id=canonical.id, assertion_version=canonical.current_version,
                               source_id=r.source_id, revision_id=r.revision_id, segment_id=r.segment_id,
                               quote=r.quote, char_start=r.char_start, char_end=r.char_end, role=r.role,
                               origin=r.origin, derivation={**(r.derivation or {}), "merged_from": duplicate.id}))
        added += 1
    db.flush()
    return added


def _live(row: ContextEvidence, deleted: set[str]) -> bool:
    derivation = row.derivation or {}
    return row.source_id not in deleted and not derivation.get("source_deleted") and \
        not derivation.get("superseded_by")


def deleted_sources(db, owner: str, source_ids: Iterable[str]) -> set[str]:
    ids = sorted(set(source_ids))
    return set(db.scalars(select(ContextSource.id).where(
        ContextSource.owner_id == owner, ContextSource.id.in_(ids),
        (ContextSource.deleted_at.is_not(None)) | (ContextSource.status == "deleted")))) if ids else set()


def human_stated(db, assertion_id: str) -> bool:
    """The owner wrote this assertion through a Changed decision."""
    v, d = ContextAssertionVersion, ContextDecision
    return bool(db.scalar(select(exists().where(
        v.assertion_id == assertion_id, v.version == 1, d.id == v.decision_id, d.operation == "changed",
        d.origin == "human"))))


def support_status(db, owner: str, assertion: ContextAssertion, *, gone: Iterable[str] = ()) -> str:
    """`supported`, `revalidate` (only part of a conjunctive group survives) or `unsupported`. Sources in
    `gone` and deleted sources count as removed."""
    rows = list(db.scalars(select(ContextEvidence).where(
        ContextEvidence.assertion_id == assertion.id, ContextEvidence.role == "support",
        ContextEvidence.origin == "independent")))
    if not rows:
        return "supported" if human_stated(db, assertion.id) or not db.scalar(select(exists().where(
            ContextEvidence.assertion_id == assertion.id))) else "unsupported"
    deleted = set(gone) | deleted_sources(db, owner, {r.source_id for r in rows})
    groups = defaultdict(list)
    for r in rows:
        groups[(r.derivation or {}).get("all_of") or r.id].append(_live(r, deleted))
    if any(all(alive) for alive in groups.values()) or human_stated(db, assertion.id):
        return "supported"
    return "revalidate" if any(any(alive) for alive in groups.values()) else "unsupported"


def corroboration(db, owner: str, assertion_id: str) -> int:
    """Distinct live sources with independent support for the assertion (contradiction and derived spans and
    repeated quotes from one source do not add)."""
    rows = list(db.scalars(select(ContextEvidence).where(
        ContextEvidence.assertion_id == assertion_id, ContextEvidence.role == "support",
        ContextEvidence.origin == "independent")))
    deleted = deleted_sources(db, owner, {r.source_id for r in rows})
    return len({r.source_id for r in rows if _live(r, deleted)})


def _retract_unsupported(db, owner: str, ids: Iterable[str], *, gone: Iterable[str] = ()) -> tuple[dict, set]:
    """Retract the live assertions among `ids` left without support. Returns the outcome and the keyed chains
    that lost a member (the caller rechains them after its other writes)."""
    result = {"retracted": [], "kept": [], "revalidate": []}
    chains = set()
    at = now()
    for row in db.scalars(select(ContextAssertion).where(ContextAssertion.owner_id == owner,
                                                         ContextAssertion.id.in_(sorted(set(ids))))
                          .order_by(ContextAssertion.id).with_for_update()):
        if row.lifecycle == "retracted":
            continue
        status = support_status(db, owner, row, gone=gone)
        if status == "unsupported":
            record = temporal.record_version(db, row, at=at, lifecycle="retracted")
            if record and (identity := temporal.chain_identity(row)):
                chains.add(identity)
            result["retracted"].append(row.id)
        else:
            result["kept" if status == "supported" else "revalidate"].append(row.id)
    return result, chains


def _support_retraction(db, owner: str, row: ContextAssertion) -> str | None:
    """The lifecycle to restore when the row was retracted only for lack of support, else None."""
    v = ContextAssertionVersion
    head, previous = (list(db.scalars(select(v).where(v.assertion_id == row.id).order_by(v.version.desc())
                                      .limit(2))) + [None, None])[:2]
    if head is None or head.lifecycle != "retracted" or head.decision_id or head.run_id:
        return None
    d = ContextDecision
    wronged = db.scalar(temporal.active_decisions(owner).where(
        d.operation == "wrong", d.target_type == "assertion", d.target_id == row.id).limit(1))
    if wronged or temporal.negated_by(db, owner, row) is not None:
        return None
    restored = previous.lifecycle if previous is not None else "current"
    return "current" if restored == "retracted" else restored


def revive(db, owner: str, ids: Iterable[str], *, run_id: str | None = None) -> list[str]:
    """Restore assertions among `ids` that were retracted only for lack of support and are supported again.
    In the caller's transaction; idempotent. Returns the IDs whose head changed (revived ones and the keyed
    chain neighbours rechained with them)."""
    changed, chains = [], set()
    at = now()
    for row in db.scalars(select(ContextAssertion).where(
            ContextAssertion.owner_id == owner, ContextAssertion.id.in_(sorted(set(ids))),
            ContextAssertion.lifecycle == "retracted").order_by(ContextAssertion.id).with_for_update()):
        lifecycle = _support_retraction(db, owner, row)
        if lifecycle is None or support_status(db, owner, row) != "supported":
            continue
        if temporal.record_version(db, row, at=at, run_id=run_id, lifecycle=lifecycle):
            changed.append(row.id)
            if identity := temporal.chain_identity(row):
                chains.add(identity)
    for identity in sorted(chains):
        changed.extend(temporal.rechain(db, owner, identity, at=at, run_id=run_id))
    temporal.stale_briefs(db, owner, changed)
    return sorted(set(changed))


def supersede_revisions(db, owner: str, source_id: str, revision_id: str) -> dict:
    """Support from the source's revisions other than `revision_id` (its newest, completely extracted) no longer
    counts; assertions that lose all support are retracted. In the caller's transaction; idempotent. Returns the
    assertion IDs `retracted`, `kept`, needing `revalidate`, and `rechained` (chain neighbours whose head
    changed)."""
    e = ContextEvidence
    rows = [r for r in db.scalars(select(e).where(e.owner_id == owner, e.source_id == source_id,
                                                 e.revision_id != revision_id).with_for_update())
            if not (r.derivation or {}).get("superseded_by")]
    for r in rows:
        r.derivation = {**(r.derivation or {}), "superseded_by": revision_id}
    db.flush()
    ids = sorted({r.assertion_id for r in rows})
    result, chains = _retract_unsupported(db, owner, ids)
    db.flush()
    rechained = set()
    for identity in sorted(chains):
        rechained.update(temporal.rechain(db, owner, identity, at=now()))
    temporal.stale_briefs(db, owner, {*ids, *rechained})
    return {**result, "rechained": sorted(rechained)}


def propagate_source_deletion(db, owner: str, source_id: str) -> dict:
    """Apply a source deletion to dependent assertions in the caller's transaction (B1's deletion job calls it).
    Idempotent. Returns the assertion IDs `retracted`, `kept` and needing `revalidate`."""
    e = ContextEvidence
    ids = sorted(set(db.scalars(select(e.assertion_id).where(e.owner_id == owner, e.source_id == source_id))))
    result, chains = _retract_unsupported(db, owner, ids, gone={source_id})
    at = now()
    for r in db.scalars(select(e).where(e.owner_id == owner, e.source_id == source_id)):
        r.quote, r.char_start, r.char_end, r.segment_id = "", None, None, None
        r.derivation = {k: v for k, v in (r.derivation or {}).items() if k == "all_of"} | {"source_deleted": True}
    db.flush()
    for identity in sorted(chains):
        temporal.rechain(db, owner, identity, at=at)
    temporal.stale_briefs(db, owner, ids)
    return result
