"""Persisting engine results into the canonical context tables, under conditional writes (plan sections 5 and 9).

Engines propose; this module decides what lands. Every apply runs in one transaction after `guard` has taken
the owner's `context:{owner}` lock and verified that the source revision is still current, the owner's processing
epoch is the one the run started under, processing is not paused and (for model results) no decision was recorded
since the run read its inputs. A failed guard returns the reason; the caller marks the run `stale`.

- `apply_extraction` admits candidates through `temporal.create` (durable negatives retract a repeated Wrong
  interpretation at once) and `evidence.attach` (exact quotes only). A candidate the same source already produced,
  whatever that assertion's lifecycle (current, superseded, retracted, ...), only gains this revision's evidence:
  re-extracting a meeting never re-admits a superseded or retracted statement. Replaying a run adds nothing.
- `apply_speakers` writes `native.link.resolve_speakers` results onto segments; `apply_links` writes mention links
  for assertions. Neither overwrites a human decision (confirmed or rejected rows, or rows with a decision).
- `replace_chunks` swaps receipt's provisional chunks for the extraction chunker's.
"""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

from sqlalchemy import delete, exists, select

from ..domain import advisory
from ..models import now
from . import decisions, evidence, temporal
from .models import (
    ContextAssertion,
    ContextChunk,
    ContextEntityLink,
    ContextEvidence,
    ContextOwnerState,
    ContextRun,
    ContextSegment,
    ContextSource,
    ContextSourceRevision,
)
from .schema import Chunk, ExtractionResult, LinkProposal, Segment

DECIDED = ("confirmed", "rejected")


def lock(db, owner: str) -> None:
    advisory(db, f"context:{owner}")


def owner_state(db, owner: str) -> tuple[int, bool]:
    """(epoch, paused); an owner without a state row is at epoch 0, running."""
    state = db.get(ContextOwnerState, owner, populate_existing=True)
    return (state.epoch, state.paused) if state else (0, False)


def guard(db, owner: str, run: ContextRun, *, source_id: str | None = None, revision_id: str | None = None,
          watermark: bool = False) -> str | None:
    """Take the owner lock and check the run may still apply. Returns None, or why it is stale: `paused`, `epoch`,
    `deleted` (source gone), `revision` (a newer revision is current) or `decisions` (the watermark moved)."""
    lock(db, owner)
    epoch, paused = owner_state(db, owner)
    if paused:
        return "paused"
    if epoch != run.epoch:
        return "epoch"
    if source_id:
        source = db.get(ContextSource, source_id, populate_existing=True)
        if source is None or source.owner_id != owner or source.status == "deleted":
            return "deleted"
        if revision_id:
            current = db.scalar(select(ContextSourceRevision.id).where(
                ContextSourceRevision.source_id == source_id,
                ContextSourceRevision.revision == source.current_revision))
            if current != revision_id:
                return "revision"
    if watermark and decisions.watermark(db, owner) != run.decision_watermark:
        return "decisions"
    return None


# ---------------------------------------------------------------- segments and chunks


def segment_rows(db, revision_id: str) -> list[ContextSegment]:
    return list(db.scalars(select(ContextSegment).where(ContextSegment.revision_id == revision_id)
                           .order_by(ContextSegment.seq)))


def segments(db, revision_id: str) -> list[Segment]:
    """The revision's segments as engine contracts; empty when its content was purged."""
    rows = segment_rows(db, revision_id)
    if not rows or any(not r.text for r in rows):
        return []
    return [Segment(seq=r.seq, kind=r.kind, speaker_label=r.speaker_label, attribution=r.speaker_attribution,
                    text=r.text, start_ms=r.start_ms, end_ms=r.end_ms, char_start=r.char_start, char_end=r.char_end)
            for r in rows]


def replace_chunks(db, owner: str, revision_id: str, chunks: Sequence[Chunk]) -> bool:
    """Make the revision's chunk rows exactly `chunks`. Returns False when they already were (no rewrite, so
    embeddings already written stay)."""
    have = list(db.scalars(select(ContextChunk).where(ContextChunk.revision_id == revision_id)
                           .order_by(ContextChunk.seq)))
    if [(c.seq, c.segment_start, c.segment_end, c.text) for c in have] == \
            [(c.seq, c.segment_start, c.segment_end, c.text) for c in chunks]:
        return False
    db.execute(delete(ContextChunk).where(ContextChunk.revision_id == revision_id))
    db.add_all(ContextChunk(owner_id=owner, revision_id=revision_id, seq=c.seq, segment_start=c.segment_start,
                            segment_end=c.segment_end, token_count=c.token_count, text=c.text) for c in chunks)
    db.flush()
    return True


# ---------------------------------------------------------------- speakers


def apply_speakers(db, owner: str, revision_id: str, proposals: Iterable[LinkProposal], run_id: str,
                   owner_label: str = "me") -> int:
    """Write speaker-bucket resolutions onto the revision's segments. Returns how many segments changed."""
    by_label = {p.mention: p for p in proposals}
    changed = 0
    for row in db.scalars(select(ContextSegment).where(ContextSegment.revision_id == revision_id)
                          .order_by(ContextSegment.seq).with_for_update()):
        label = row.speaker_label or (owner_label if row.speaker_attribution == "owner" else None)
        proposal = by_label.get(label) if label else None
        if proposal is None or row.speaker_link_status in DECIDED or (row.speaker_provenance or {}).get("decision_id"):
            continue
        values = {"speaker_actor_id": proposal.candidate.entity_id, "speaker_link_status": proposal.status,
                  "speaker_provenance": {"via": "resolve_speakers", "run_id": run_id,
                                         "evidence": [e.model_dump(mode="json") for e in proposal.evidence]}}
        if (row.speaker_actor_id, row.speaker_link_status) != (values["speaker_actor_id"], proposal.status):
            changed += 1
        for key, value in values.items():
            setattr(row, key, value)
    db.flush()
    return changed


# ---------------------------------------------------------------- assertions


@dataclass
class Applied:
    assertion_ids: list[str | None] = field(default_factory=list)  # per candidate index; None when refused
    admitted: list[str] = field(default_factory=list)
    attached: list[str] = field(default_factory=list)
    rejected: list[dict] = field(default_factory=list)

    @property
    def live(self) -> list[str]:
        return list(dict.fromkeys(i for i in self.assertion_ids if i))


def work_keys(result: ExtractionResult) -> dict[int, str]:
    """Stable work keys for commitments the engine tied to an existing task or issue (update and link actions)."""
    keys = {}
    for action in result.actions:
        if action.kind in ("update", "link") and action.target and action.target.entity_id:
            for index in action.assertion_indexes:
                keys.setdefault(index, f"{action.target.entity_type}:{action.target.entity_id}")
    return keys


def _known(db, owner: str, source_id: str, kind: str, statement: str) -> ContextAssertion | None:
    """An assertion this source already produced with the same kind and statement, live ones first."""
    a, e = ContextAssertion, ContextEvidence
    rows = db.scalars(select(a).where(a.owner_id == owner, a.kind == kind,
                                      exists().where(e.assertion_id == a.id, e.source_id == source_id))
                      .order_by(a.created_at, a.id).with_for_update())
    key = temporal.normalized(statement)
    matches = [r for r in rows if temporal.normalized(r.statement) == key]
    return next((r for r in matches if r.lifecycle in temporal.LIVE), matches[0] if matches else None)


def apply_extraction(db, owner: str, run: ContextRun, source: ContextSource, revision: ContextSourceRevision,
                     result: ExtractionResult) -> Applied:
    """Admit the result's candidates (see the module docstring). The caller has passed `guard`."""
    texts = {r.seq: r.text for r in segment_rows(db, revision.id)}
    keys = work_keys(result)
    out = Applied()
    for index, candidate in enumerate(result.assertions):
        refs = list(candidate.evidence)
        if not all(r.source_revision_id in (None, revision.id) and r.segment_seq in texts
                   and r.matches(texts[r.segment_seq]) for r in refs):
            out.assertion_ids.append(None)
            out.rejected.append({"index": index, "reason": "inexact_evidence"})
            continue
        row = _known(db, owner, source.id, candidate.kind, candidate.statement)
        known = row is not None
        if not known:
            row = temporal.create(db, owner, candidate, occurred_at=revision.occurred_at or revision.received_at,
                                  fallback_subject=("source", source.id), work_key=keys.get(index), run_id=run.id)
            out.admitted.append(row.id)
        added = evidence.attach(db, owner, row, refs, revision_id=revision.id)
        if known and added and row.id not in out.admitted:
            out.attached.append(row.id)
        out.assertion_ids.append(row.id)
    out.attached = list(dict.fromkeys(out.attached))
    temporal.stale_briefs(db, owner, out.attached)
    return out


# ---------------------------------------------------------------- entity links


@dataclass(frozen=True)
class RoleLink:
    assertion_id: str
    role: str  # subject | object | recipient | responsible | responsible_mention
    proposal: LinkProposal


def apply_links(db, owner: str, links: Iterable[RoleLink], run_id: str) -> dict:
    """Upsert one link per (assertion, role, mention). Rows a human decided stay as they are."""
    counts = {"created": 0, "updated": 0, "kept": 0}
    stamp = now()
    for item in links:
        p = item.proposal
        mention = p.mention[:500]
        roles = ("responsible", "responsible_mention") if item.role.startswith("responsible") else (item.role,)
        row = db.scalar(select(ContextEntityLink).where(
            ContextEntityLink.owner_id == owner, ContextEntityLink.assertion_id == item.assertion_id,
            ContextEntityLink.role.in_(roles), ContextEntityLink.mention == mention)
            .order_by(ContextEntityLink.created_at).limit(1).with_for_update())
        values = {"role": item.role, "target_type": p.candidate.entity_type if p.candidate.entity_id else None,
                  "target_id": p.candidate.entity_id, "status": p.status, "score": p.score, "margin": p.margin,
                  "evidence": [e.model_dump(mode="json") for e in p.evidence]}
        if row is None:
            db.add(ContextEntityLink(owner_id=owner, assertion_id=item.assertion_id, subject_type="assertion",
                                     subject_id=item.assertion_id, mention=mention, run_id=run_id, created_at=stamp,
                                     updated_at=stamp, **values))
            counts["created"] += 1
        elif row.decision_id or row.status in DECIDED or all(getattr(row, k) == v for k, v in values.items()):
            counts["kept"] += 1
        else:
            for key, value in values.items():
                setattr(row, key, value)
            row.run_id, row.revision, row.updated_at = run_id, row.revision + 1, stamp
            counts["updated"] += 1
    db.flush()
    return counts
