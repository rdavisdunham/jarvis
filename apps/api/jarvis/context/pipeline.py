"""Context job entry point and stage pipeline. `worker.perform_job` sends every `context_*` job kind here (queue
`jarvis-context`); `perform(job_id)` dispatches on the kind through `HANDLERS` (later packages add one line).

Stages, one job each (payloads in brackets), each enqueuing the next in the transaction that records it:

    context_ingest {source_id, revision_id}       extraction chunks, identity seeding, speaker buckets
    context_extract {source_id, revision_id}      engine extraction, admission, coverage ledger
    context_link {source_id, revision_id, extract_run_id}       mention links, consequential Questions
    context_reconcile {source_id, revision_id, extract_run_id}  bounded reconciliation, source ready
    context_embed {source_id, revision_id}        chunk and assertion embeddings (projections)

plus `context_delete` {source_id, through_revision} (B1's purge), `context_granola_sync` and
`context_granola_backfill` {connection_id} (B2's poller, enqueued by `schedule_due`), `context_brief`
{entity_type, entity_id} (C3's `briefs.build_brief` when present), and `context_lint` and `context_proposals`,
which no-op until their packages land.

Run rules. `perform_job` runs inside a DBOS step retried up to ten times, so every handler is idempotent and the
job's status is set here. Each job has one `context_runs` row per stage; a replayed job whose run finished is a
no-op. Model and embedding calls happen outside database transactions, and their outputs are persisted on the
run (`stage_results`: each extraction chunk as it completes, then the whole result, or the reconcile ops) before
anything is applied, so a crash or retry never repeats a completed model call. Applying re-takes the owner's
`context:{owner}` lock and re-checks the source revision, processing epoch, pause flag and decision watermark
(`apply.guard`). A stale result is not applied: the run is marked `stale` and requeued when still relevant
(moved decisions), continued after resume (pause), or dropped (newer revision, deleted source). Work commands
(actors created while seeding identities) run as the owner, never under a bot credential of the calling context.

Engine choice: the native engines, or the deterministic doubles when the owner's `ContextOwnerState.settings`
has `"engine": "double"` (offline development and fixture evaluation).

Budget: the native extractor reserves every model call through `memory_learning` (owner budget) and caps the
whole revision at `context_source_budget_usd`, counting chunk costs spent by earlier attempts. Hitting that cap
leaves the source `awaiting_input` (`continue_source` resumes it with a larger cap); an owner budget refusal or
failed chunks leave it `partial`, continued by `schedule_due` with backoff.
"""

import hashlib
import json
import logging
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta

from sqlalchemy import or_, select
from sqlalchemy.dialects.postgresql import insert

from .. import access, bot_access
from ..cost_features import feature
from ..db import session_scope
from ..domain import enqueue_job
from ..models import Actor, Job, Project, now
from . import apply, calendar_link, decisions, identity, temporal
from . import settings as context_settings
from .models import (
    ContextActorIdentity,
    ContextAssertion,
    ContextConnection,
    ContextCursor,
    ContextDecision,
    ContextEntityLink,
    ContextOwnerState,
    ContextRun,
    ContextSource,
    ContextSourceRevision,
)
from .schema import (
    EntityCandidate,
    EntityRef,
    ExtractionResult,
    ExtractRequest,
    LinkProposal,
    Participant,
    ReconcileOp,
)

log = logging.getLogger("jarvis.context.pipeline")

PENDING = ("queued", "dispatched", "running", "retrying")
TERMINAL = ("succeeded", "failed", "cancelled", "expired")
FINISHED = ("succeeded", "partial", "stale", "failed", "cancelled")
SOURCE_STAGES = ("ingest", "extract", "link", "reconcile")
MAX_ATTEMPTS = 10  # perform_job's DBOS step attempts
MAX_REQUEUES = 5  # stale-on-decisions requeues of one stage before giving up
CONFIG_VERSION = "pipeline-v1"
CONTINUE_BACKOFF = (timedelta(minutes=5), timedelta(hours=6), timedelta(hours=24))  # then stop until asked
EXTRACT_CANDIDATES = 50
UNLINKABLE = {"we", "us", "team", "the team", "everyone", "all of us", "both of us", "unknown speaker", "unknown",
              "someone", "i", "me", "you"}
BRIEF_TYPES = ("actor", "project", "client", "area", "goal")


@dataclass(frozen=True)
class Work:
    job_id: str
    owner: str
    kind: str
    payload: dict
    attempts: int

    @property
    def base(self) -> dict:
        """The payload without perform's bookkeeping."""
        return {k: v for k, v in self.payload.items() if k != "attempts"}


# ---------------------------------------------------------------- entry point


@contextmanager
def as_owner():
    """Run work commands as the owner: clear any bot credential or delegated-work grant of the calling context."""
    tokens = [(bot_access.principal, bot_access.principal.set(None)), (access.execution, access.execution.set(None))]
    try:
        yield
    finally:
        for var, token in reversed(tokens):
            var.reset(token)


def perform(job_id):
    with session_scope() as db:
        job = db.scalar(select(Job).where(Job.id == job_id).with_for_update())
        if job is None or job.status in TERMINAL:
            return
        handler = HANDLERS.get(job.kind)
        attempts = int(job.payload.get("attempts", 0)) + 1
        job.payload = {**job.payload, "attempts": attempts}
        if handler is None:
            job.status, job.finished_at, job.result = "failed", now(), {"error": "Unsupported job kind"}
            return
        job.status = "running"
        work = Work(job.id, job.owner_id, job.kind, dict(job.payload), attempts)
    try:
        with as_owner():
            result = handler(work) or {}
    except Exception as exc:
        code = getattr(exc, "code", None) or type(exc).__name__
        final = attempts >= MAX_ATTEMPTS
        log.warning("context job %s (%s) attempt %s failed: %s", job_id, work.kind, attempts, code)
        with session_scope() as db:
            job = db.get(Job, job_id)
            job.status, job.result = ("failed" if final else "retrying"), {"error": code}
            job.finished_at = now() if final else None
            for run in db.scalars(select(ContextRun).where(ContextRun.job_id == job_id)):
                run.error, run.updated_at = code, now()
                if final and run.status not in FINISHED:
                    run.status, run.finished_at = "failed", now()
                    if run.stage in SOURCE_STAGES:
                        _source_status(db, run.source_id, "failed")
        if final:
            return
        raise
    with session_scope() as db:
        job = db.get(Job, job_id)
        if job.status not in TERMINAL:
            job.status, job.finished_at, job.result = "succeeded", now(), result


# ---------------------------------------------------------------- runs


def _run(db, work: Work, stage: str, *, source_id=None, revision_id=None) -> ContextRun:
    """This job's run for `stage`, created on the first attempt with the epoch and decision watermark."""
    run = db.scalar(select(ContextRun).where(ContextRun.job_id == work.job_id, ContextRun.stage == stage)
                    .with_for_update())
    stamp = now()
    if run is None:
        epoch, _ = apply.owner_state(db, work.owner)
        run = ContextRun(owner_id=work.owner, stage=stage, status="running", job_id=work.job_id, source_id=source_id,
                         revision_id=revision_id, epoch=epoch, decision_watermark=decisions.watermark(db, work.owner),
                         config_version=CONFIG_VERSION, stage_results={"payload": work.base}, attempts=0, started_at=stamp,
                         created_at=stamp)
        db.add(run)
    if run.status not in FINISHED:
        run.attempts += 1
    run.updated_at = stamp
    db.flush()
    return run


def _locked(db, run_id: str) -> ContextRun:
    return db.scalar(select(ContextRun).where(ContextRun.id == run_id).with_for_update())


def _results(run: ContextRun, **values) -> None:
    run.stage_results = {**(run.stage_results or {}), **values}
    run.updated_at = now()


def _finish(run: ContextRun, status: str = "succeeded", **results) -> dict:
    _results(run, **results)
    run.status, run.finished_at = status, now()
    return {"run_id": run.id, "status": status}


def _replay(run: ContextRun) -> dict | None:
    return {"run_id": run.id, "status": run.status, "replay": True} if run.status in FINISHED else None


def _backoff(work: Work) -> datetime | None:
    n = int(work.payload.get("continuation", 0))
    return now() + CONTINUE_BACKOFF[n] if n < len(CONTINUE_BACKOFF) else None


def _stale(db, work: Work, run: ContextRun, reason: str) -> dict:
    """Refuse the result. Pause and epoch changes continue after resume; moved decisions requeue now; a newer
    revision or a deleted source makes the work irrelevant."""
    run.status, run.error, run.finished_at, run.updated_at = "stale", f"stale: {reason}", now(), now()
    if reason in ("paused", "epoch"):
        _results(run, continue_after=now().isoformat())
    elif reason == "decisions":
        requeues = int(work.payload.get("requeues", 0))
        if requeues < MAX_REQUEUES:
            enqueue_job(db, work.owner, work.kind, {**work.base, "requeues": requeues + 1})
    return {"run_id": run.id, "status": "stale", "reason": reason}


def _source_status(db, source_id: str | None, status: str) -> None:
    source = db.get(ContextSource, source_id) if source_id else None
    if source is not None and source.status not in ("deleted", status):
        source.status, source.updated_at = status, now()


def _done(db, owner: str, stage: str, revision_id: str, run_id: str) -> bool:
    """Another job already completed this stage for the revision (duplicate delivery)."""
    return bool(db.scalar(select(ContextRun.id).where(
        ContextRun.owner_id == owner, ContextRun.stage == stage, ContextRun.revision_id == revision_id,
        ContextRun.status == "succeeded", ContextRun.id != run_id).limit(1)))


def _source(work: Work) -> tuple[str, str]:
    return work.payload["source_id"], work.payload["revision_id"]


def _attendees(db, revision_id: str) -> list[identity.IdentityCandidate]:
    """The identity candidates the revision's ingest run seeded."""
    run = db.scalar(select(ContextRun).where(ContextRun.revision_id == revision_id, ContextRun.stage == "ingest",
                                             ContextRun.status == "succeeded")
                    .order_by(ContextRun.created_at.desc()).limit(1))
    items = (run.stage_results or {}).get("attendees", []) if run else []
    return [identity.IdentityCandidate(**{k: tuple(v) if k in ("candidates", "duplicates") else v
                                          for k, v in item.items()}) for item in items]


def engine_name(db, owner: str) -> str:
    state = db.get(ContextOwnerState, owner)
    return "double" if state and (state.settings or {}).get("engine") == "double" else "native"


# ---------------------------------------------------------------- ingest


def ingest(work: Work) -> dict:
    from .native.chunking import chunk_segments
    from .native.link import resolve_speakers

    source_id, revision_id = _source(work)
    with session_scope() as db:
        run = _run(db, work, "ingest", source_id=source_id, revision_id=revision_id)
        if replay := _replay(run):
            return replay
        if reason := apply.guard(db, work.owner, run, source_id=source_id, revision_id=revision_id):
            return _stale(db, work, run, reason)
        if not work.payload.get("continue") and _done(db, work.owner, "ingest", revision_id, run.id):
            return _finish(run, skipped="duplicate")
        segments = apply.segments(db, revision_id)
        if not segments:
            return _stale(db, work, run, "deleted")
        chunks = [c.contract() for c in chunk_segments(segments)]
        apply.replace_chunks(db, work.owner, revision_id, chunks)
        seeded = calendar_link.seed_identities(db, work.owner, source_id) or []
        speakers = resolve_speakers(db, work.owner, source_id, segments, seeded)
        changed = apply.apply_speakers(db, work.owner, revision_id, speakers, run.id)
        _source_status(db, source_id, "processing")
        enqueue_job(db, work.owner, "context_extract", {"source_id": source_id, "revision_id": revision_id})
        return _finish(run, chunks=len(chunks), attendees=[asdict(c) for c in seeded], speakers_changed=changed,
                       speakers=[p.model_dump(mode="json") for p in speakers])


# ---------------------------------------------------------------- extract


def _candidates(db, owner: str, actor_ids: list[str]) -> list[EntityCandidate]:
    """Entities the extractor may name: the meeting's resolved people (with aliases) and active projects."""
    aliases: dict[str, set[str]] = {}
    for actor_id, value in db.execute(select(ContextActorIdentity.actor_id, ContextActorIdentity.value).where(
            ContextActorIdentity.owner_id == owner, ContextActorIdentity.actor_id.in_(actor_ids),
            ContextActorIdentity.namespace.in_(("alias", "calendar_name")))):
        aliases.setdefault(actor_id, set()).add(value[:300])
    out = [EntityCandidate(ref=EntityRef(entity_type="actor", entity_id=a.id), name=a.name,
                           aliases=sorted(aliases.get(a.id, set()) - {a.name}))
           for a in db.scalars(select(Actor).where(Actor.owner_id == owner, Actor.id.in_(actor_ids),
                                                   Actor.archived.is_(False)).order_by(Actor.name, Actor.id))]
    out += [EntityCandidate(ref=EntityRef(entity_type="project", entity_id=p.id), name=p.name[:300])
            for p in db.scalars(select(Project).where(Project.owner_id == owner, Project.archived.is_(False))
                                .order_by(Project.name, Project.id).limit(EXTRACT_CANDIDATES))]
    return out


def _request(db, owner: str, source: ContextSource, revision: ContextSourceRevision, segments) -> ExtractRequest:
    attendees = [a.actor_id for a in _attendees(db, revision.id) if a.actor_id]
    actor_ids = list(dict.fromkeys([identity.owner_actor(db, owner), *attendees]))
    d = ContextDecision
    negatives = db.scalars(temporal.active_decisions(owner).where(
        d.negative.is_(True), d.operation.in_(("wrong", "dismiss"))).order_by(d.seq.desc()).limit(200))
    return ExtractRequest(owner_id=owner, source_revision_id=revision.id,
                          occurred_at=revision.occurred_at or revision.received_at, timezone=revision.timezone,
                          title=source.title, segments=segments,
                          participants=[Participant.model_validate(p) for p in revision.participants or []],
                          candidates=_candidates(db, owner, actor_ids),
                          decisions=[temporal.to_decision(x) for x in negatives])


def _draft_dump(draft) -> dict:
    def dump(value):
        return None if value is None else value.model_dump(mode="json")
    return json.loads(json.dumps({
        "kind": draft.kind, "statement": draft.statement, "evidence": [dump(e) for e in draft.evidence],
        "assertion": dump(draft.assertion), "responsible": [dump(r) for r in draft.responsible],
        "due": dump(draft.due), "links": [dump(x) for x in draft.links], "derivations": draft.derivations,
        "notes": draft.notes}, default=str))


def _draft_load(data: dict):
    from .native.validate import Draft
    from .schema import CandidateAssertion, EvidenceRef, TimeBound
    return Draft(kind=data["kind"], statement=data["statement"],
                 evidence=[EvidenceRef.model_validate(e) for e in data["evidence"]],
                 assertion=CandidateAssertion.model_validate(data["assertion"]) if data["assertion"] else None,
                 responsible=[EntityRef.model_validate(r) for r in data["responsible"]],
                 due=TimeBound.model_validate(data["due"]) if data["due"] else None,
                 links=[LinkProposal.model_validate(x) for x in data["links"]], derivations=data["derivations"],
                 notes=data["notes"])


def _chunk_key(block: dict, candidates: list) -> str:
    from .native.prompts import PROMPT_VERSION
    raw = json.dumps({"prompt": PROMPT_VERSION, "block": block, "candidates": candidates}, sort_keys=True,
                     default=str)
    return hashlib.sha256(raw.encode()).hexdigest()


def resumable_extractor(cached: dict, save: Callable[[int, dict], None], budget_usd: float | None = None):
    """The native extractor with per-chunk persistence: a chunk whose inputs (prompt version, lines, context and
    candidates) match a saved one is restored without model calls, and a chunk completed now is saved at once.
    Skipped and failed chunks are never saved, so a continuation retries exactly those."""
    from .native.extract import NativeExtractor
    from .native.prompts import meeting_block
    from .native.validate import Rejection

    class ResumableExtractor(NativeExtractor):
        def _chunk(self, owner, run, ctx, meeting, candidates, spend, timeout):
            key = _chunk_key(meeting_block(meeting, run.chunk), candidates)
            saved = cached.get(str(run.chunk.seq))
            if saved and saved.get("key") == key:
                run.drafts = [_draft_load(d) for d in saved["drafts"]]
                run.rejected = [Rejection(r["code"], r["statement"], r["detail"], tuple(r["segment_seqs"]))
                                for r in saved["rejected"]]
                run.cost = float(saved.get("cost", 0.0))
                spend.spent += run.cost
                return
            super()._chunk(owner, run, ctx, meeting, candidates, spend, timeout)
            save(run.chunk.seq, {"key": key, "cost": run.cost, "calls": run.calls,
                                 "drafts": [_draft_dump(d) for d in run.drafts],
                                 "rejected": [r.as_dict() for r in run.rejected]})

    return ResumableExtractor(budget_usd=budget_usd)


def _cached_chunks(db, owner: str, revision_id: str) -> dict:
    """Saved chunk results from every extraction run of the revision, newest winning."""
    cached: dict = {}
    for row in db.scalars(select(ContextRun).where(ContextRun.owner_id == owner, ContextRun.stage == "extract",
                                                   ContextRun.revision_id == revision_id)
                          .order_by(ContextRun.created_at, ContextRun.id)):
        cached.update((row.stage_results or {}).get("chunks", {}))
    return cached


def _chunk_saver(run_id: str) -> Callable[[int, dict], None]:
    def save(seq: int, data: dict) -> None:
        with session_scope() as db:
            run = _locked(db, run_id)
            _results(run, chunks={**(run.stage_results or {}).get("chunks", {}), str(seq): data})
    return save


def run_extractor(engine: str, request: ExtractRequest, cached: dict, save, budget_usd=None) -> ExtractionResult:
    """Call the engine (outside any transaction)."""
    if engine == "double":
        from .doubles import DoubleExtractor
        return DoubleExtractor().extract(request)
    return resumable_extractor(cached, save, budget_usd).extract(request)


def extract(work: Work) -> dict:
    source_id, revision_id = _source(work)
    with session_scope() as db:
        run = _run(db, work, "extract", source_id=source_id, revision_id=revision_id)
        if replay := _replay(run):
            return replay
        if reason := apply.guard(db, work.owner, run, source_id=source_id, revision_id=revision_id):
            return _stale(db, work, run, reason)
        if not work.payload.get("continue") and _done(db, work.owner, "extract", revision_id, run.id):
            return _finish(run, skipped="duplicate")
        segments = apply.segments(db, revision_id)
        if not segments:
            return _stale(db, work, run, "deleted")
        source, revision = db.get(ContextSource, source_id), db.get(ContextSourceRevision, revision_id)
        request = _request(db, work.owner, source, revision, segments)
        engine = run.engine = engine_name(db, work.owner)
        stored = (run.stage_results or {}).get("result")
        cached = _cached_chunks(db, work.owner, revision_id)
        run_id = run.id

    if stored is not None:
        result = ExtractionResult.model_validate(stored)
    else:
        result = run_extractor(engine, request, cached, _chunk_saver(run_id), work.payload.get("budget_usd"))
        with session_scope() as db:
            run = _locked(db, run_id)
            diagnostics = result.diagnostics
            usage = diagnostics.get("usage") or {}
            run.model, run.prompt_version = diagnostics.get("model"), diagnostics.get("prompt_version")
            run.input_tokens, run.output_tokens = usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0)
            run.cost_usd = diagnostics.get("cost_usd", 0)
            _results(run, result=result.model_dump(mode="json"))

    with session_scope() as db:
        run = _locked(db, run_id)
        if replay := _replay(run):
            return replay
        if reason := apply.guard(db, work.owner, run, source_id=source_id, revision_id=revision_id, watermark=True):
            return _stale(db, work, run, reason)
        source, revision = db.get(ContextSource, source_id), db.get(ContextSourceRevision, revision_id)
        applied = apply.apply_extraction(db, work.owner, run, source, revision, result)
        run.coverage = {str(s.seq): result.coverage.get(s.seq, "failed") for s in segments}
        partial = any(v in ("skipped", "failed") for v in run.coverage.values())
        reasons = set((result.diagnostics.get("skipped") or {}).values())
        extra = {}
        if partial:
            _source_status(db, source_id, "awaiting_input" if "source_budget" in reasons else "partial")
            if "source_budget" not in reasons and (after := _backoff(work)):
                extra["continue_after"] = after.isoformat()
        actions = [{"action": a.model_dump(mode="json"),
                    "assertion_ids": [applied.assertion_ids[i] for i in a.assertion_indexes
                                      if applied.assertion_ids[i]]} for a in result.actions]
        enqueue_job(db, work.owner, "context_link", {"source_id": source_id, "revision_id": revision_id,
                                                     "extract_run_id": run.id})
        return _finish(run, "partial" if partial else "succeeded", assertion_ids=applied.assertion_ids,
                       admitted=applied.admitted, attached=applied.attached, refused=applied.rejected,
                       actions=actions, **extra)


# ---------------------------------------------------------------- link


@feature("context_linking")
def _mention_vectors(owner: str, texts: list[str]) -> dict[str, list[float]]:
    from .. import memory_learning
    try:
        return dict(zip(texts, memory_learning.embeddings(owner, texts), strict=True))
    except Exception as exc:  # noqa: BLE001 - linking falls back to lexical scores
        log.info("context mention embedding skipped: %s", getattr(exc, "code", None) or type(exc).__name__)
        return {}


def _roles(candidate) -> list[tuple[str, EntityRef]]:
    refs = [("subject", candidate.subject), ("object", candidate.object), ("recipient", candidate.recipient)]
    refs += [("responsible", r) for r in candidate.responsible]
    return [(role, ref) for role, ref in refs if ref is not None]


def _linkable(ref: EntityRef) -> bool:
    return bool(ref.mention) and ref.mention.casefold().strip() not in UNLINKABLE


def _speaker_map(db, revision_id: str) -> dict[str, tuple[str, str]]:
    """Normalized speaker label -> (actor ID, link status) for the revision's resolved speakers."""
    out: dict[str, tuple[str, str]] = {}
    for row in apply.segment_rows(db, revision_id):
        if row.speaker_label and row.speaker_actor_id and row.speaker_link_status in ("exact", "inferred", "confirmed"):
            out.setdefault(identity.normalize("alias", row.speaker_label),
                           (row.speaker_actor_id, row.speaker_link_status))
    return out


def link(work: Work) -> dict:
    from .native.link import LinkContext, Mention, link_mentions

    source_id, revision_id = _source(work)
    with session_scope() as db:
        run = _run(db, work, "link", source_id=source_id, revision_id=revision_id)
        if replay := _replay(run):
            return replay
        run_id = run.id
        extracted = db.get(ContextRun, work.payload["extract_run_id"])
        stored = (extracted.stage_results or {}) if extracted else {}
        result = ExtractionResult.model_validate(stored.get("result") or {})
        ids = stored.get("assertion_ids") or []
        texts = sorted({ref.mention for c in result.assertions for _, ref in _roles(c)
                        if not ref.entity_id and _linkable(ref)})
    vectors = _mention_vectors(work.owner, texts) if texts else {}

    with session_scope() as db:
        run = _locked(db, run_id)
        if replay := _replay(run):
            return replay
        if reason := apply.guard(db, work.owner, run, source_id=source_id, revision_id=revision_id):
            return _stale(db, work, run, reason)
        rejected = identity.rejected_links(db, work.owner)
        speakers = _speaker_map(db, revision_id)
        links: list[apply.RoleLink] = []
        pending: list[tuple[str, str, Mention]] = []
        for index, candidate in enumerate(result.assertions):
            assertion_id = ids[index] if index < len(ids) else None
            if not assertion_id:
                continue
            for role, ref in _roles(candidate):
                mention = ref.mention or ref.entity_id
                if ref.entity_id and not rejected.blocks(mention, ref.entity_type, ref.entity_id, source_id):
                    links.append(apply.RoleLink(assertion_id, role, LinkProposal(
                        mention=mention[:300], candidate=EntityRef(entity_type=ref.entity_type, entity_id=ref.entity_id),
                        status="inferred", evidence=candidate.evidence[:3])))
                    continue
                if not _linkable(ref):
                    continue
                spoken = speakers.get(identity.normalize("alias", ref.mention)) if ref.entity_type == "actor" else None
                if spoken and not rejected.blocks(ref.mention, "actor", spoken[0], source_id):
                    links.append(apply.RoleLink(assertion_id, role, LinkProposal(
                        mention=ref.mention[:300], candidate=EntityRef(entity_type="actor", entity_id=spoken[0]),
                        status="exact" if spoken[1] in ("exact", "confirmed") else "inferred")))
                    continue
                pending.append((assertion_id, role, Mention(text=ref.mention[:300], entity_types=(ref.entity_type,),
                                                            evidence=tuple(candidate.evidence[:3]),
                                                            consequential=role == "responsible")))
        if pending:
            context = LinkContext(scope=source_id, attendees=_attendees(db, revision_id),
                                  embed=(lambda xs: [vectors[x] for x in xs]) if vectors else None)
            proposals = link_mentions(db, work.owner, [m for *_, m in pending], context)
            for (assertion_id, role, _), proposal in zip(pending, proposals, strict=True):
                if role == "responsible" and proposal.status == "unresolved" and not getattr(proposal, "ask", False):
                    role = "responsible_mention"  # inspectable, but not worth a Question
                links.append(apply.RoleLink(assertion_id, role, LinkProposal.model_validate(
                    proposal.model_dump(include=set(LinkProposal.model_fields)))))
        counts = apply.apply_links(db, work.owner, links, run.id)
        enqueue_job(db, work.owner, "context_reconcile", {"source_id": source_id, "revision_id": revision_id,
                                                          "extract_run_id": work.payload["extract_run_id"]})
        questions = sum(1 for x in links if x.role == "responsible" and x.proposal.status == "unresolved")
        return _finish(run, links=counts, questions=questions)


# ---------------------------------------------------------------- reconcile


def run_reconciler(engine: str, request, work_keys: dict) -> list[ReconcileOp]:
    """Call the engine (outside any transaction)."""
    if engine == "double":
        from .doubles import DoubleReconciler
        return DoubleReconciler().reconcile(request)
    from .native.reconcile import NativeReconciler
    return NativeReconciler().reconcile(request, work_keys)


def _brief_targets(db, owner: str, assertion_ids: list[str]) -> set[tuple[str, str]]:
    if not assertion_ids:
        return set()
    a, link_ = ContextAssertion, ContextEntityLink
    targets = set()
    for row in db.scalars(select(a).where(a.owner_id == owner, a.id.in_(assertion_ids))):
        for kind, ident in ((row.subject_type, row.subject_id), (row.object_type, row.object_id)):
            if kind in BRIEF_TYPES and ident and not ident.startswith("mention:"):
                targets.add((kind, ident))
    targets |= set(db.execute(select(link_.target_type, link_.target_id).where(
        link_.owner_id == owner, link_.assertion_id.in_(assertion_ids), link_.target_type.in_(BRIEF_TYPES),
        link_.status.in_(("exact", "inferred", "confirmed")))).tuples())
    return targets


def _briefs_available() -> bool:
    try:
        from .briefs import build_brief  # noqa: F401 - package C3
    except ImportError:
        return False
    return True


def reconcile(work: Work) -> dict:
    source_id, revision_id = _source(work)
    with session_scope() as db:
        run = _run(db, work, "reconcile", source_id=source_id, revision_id=revision_id)
        if replay := _replay(run):
            return replay
        if reason := apply.guard(db, work.owner, run, source_id=source_id, revision_id=revision_id):
            return _stale(db, work, run, reason)
        extracted = db.get(ContextRun, work.payload["extract_run_id"])
        complete = extracted is not None and extracted.status == "succeeded"
        ids = [i for i in dict.fromkeys((extracted.stage_results or {}).get("assertion_ids") or []) if i] \
            if extracted else []
        request, work_keys = temporal.reconcile_request(db, work.owner, ids) if ids else (None, {})
        engine = run.engine = engine_name(db, work.owner)
        stored = (run.stage_results or {}).get("ops")
        run_id = run.id

    if request is not None and stored is None:
        ops = run_reconciler(engine, request, work_keys)
        with session_scope() as db:
            _results(_locked(db, run_id), ops=[op.model_dump(mode="json") for op in ops])
    else:
        ops = [ReconcileOp.model_validate(op) for op in stored or []]

    with session_scope() as db:
        run = _locked(db, run_id)
        if replay := _replay(run):
            return replay
        if reason := apply.guard(db, work.owner, run, source_id=source_id, revision_id=revision_id, watermark=True):
            return _stale(db, work, run, reason)
        summary = temporal.apply_ops(db, work.owner, ops, run_id=run.id) if ops else {}
        if summary.get("stale"):
            return {**_stale(db, work, run, "decisions"), "applied": summary}
        if complete:
            _source_status(db, source_id, "ready")
        enqueue_job(db, work.owner, "context_embed", {"source_id": source_id, "revision_id": revision_id})
        if _briefs_available():
            for entity_type, entity_id in sorted(_brief_targets(db, work.owner, ids + summary.get("changed", []))):
                if not _pending(db, work.owner, "context_brief", entity_type=entity_type, entity_id=entity_id):
                    enqueue_job(db, work.owner, "context_brief", {"entity_type": entity_type, "entity_id": entity_id})
        return _finish(run, summary=summary)


# ---------------------------------------------------------------- embed


def embed(work: Work) -> dict:
    from .native import embed as embedding

    revision_id = work.payload.get("revision_id")
    with session_scope() as db:
        run = _run(db, work, "embed", source_id=work.payload.get("source_id"), revision_id=revision_id)
        if replay := _replay(run):
            return replay
        if reason := apply.guard(db, work.owner, run):
            return _stale(db, work, run, reason)
        items = embedding.pending(db, work.owner, revision_id=revision_id)
        if not items:
            return _finish(run, embedded=0)
        run_id = run.id
    try:
        vectors, error = embedding.compute(work.owner, items), None
    except Exception as exc:  # noqa: BLE001 - recorded on the run; lexical retrieval keeps working
        vectors, error = None, getattr(exc, "code", None) or type(exc).__name__
    with session_scope() as db:
        run = _locked(db, run_id)
        if replay := _replay(run):
            return replay
        run.model = embedding.MODEL
        if error:
            run.error = f"embedding failed: {error}"
            extra = {"continue_after": after.isoformat()} if (after := _backoff(work)) else {"permanent": True}
            return _finish(run, "failed", pending=len(items), **extra)
        if reason := apply.guard(db, work.owner, run):
            return _stale(db, work, run, reason)
        written = embedding.write(db, work.owner, items, vectors)
        return _finish(run, embedded=written, pending=len(items) - written)


def retry_embeddings(db, owner: str) -> str | None:
    """Operator retry after a permanent embedding failure: one embedding run over everything unembedded."""
    if _pending(db, owner, "context_embed"):
        return None
    return enqueue_job(db, owner, "context_embed", {}).id


# ---------------------------------------------------------------- deletion, connectors, placeholders


def delete(work: Work) -> dict:
    from .sources import purge_source

    with session_scope() as db:
        return purge_source(db, work.owner, work.payload["source_id"],
                            through_revision=work.payload.get("through_revision"))


def _connection(db, work: Work) -> ContextConnection | None:
    connection = db.get(ContextConnection, work.payload.get("connection_id"))
    return connection if connection is not None and connection.owner_id == work.owner else None


def granola_sync(work: Work) -> dict:
    from .connectors import resolve

    with session_scope() as db:
        connection = _connection(db, work)
        if connection is None:
            return {"skipped": "connection not found"}
        return asdict(resolve("granola").poll(db, work.owner, connection))


def granola_backfill(work: Work) -> dict:
    from .connectors import resolve

    with session_scope() as db:
        connection = _connection(db, work)
        if connection is None:
            return {"skipped": "connection not found"}
        return asdict(resolve("granola").backfill(db, work.owner, connection, int(work.payload.get("days", 90))))


def synthesize_briefs(db, owner: str) -> bool:
    """Owner setting `synthesize_briefs` (default off) turns on model-written briefs in the pipeline."""
    state = db.get(ContextOwnerState, owner)
    return bool(state and (state.settings or {}).get("synthesize_briefs"))


def brief(work: Work) -> dict:
    """Rebuild one entity's brief: read inputs, synthesize outside any transaction, commit if inputs held still."""
    from .briefs import build_brief, commit_brief, prepare_brief, run_synthesis

    entity_type, entity_id = work.payload["entity_type"], work.payload["entity_id"]
    with session_scope() as db:
        run = _run(db, work, "brief")
        run.entity_type, run.entity_id = entity_type, entity_id
        if replay := _replay(run):
            return replay
        if reason := apply.guard(db, work.owner, run):
            return _stale(db, work, run, reason)
        engine = run.engine = engine_name(db, work.owner)
        if not synthesize_briefs(db, work.owner):
            # Deterministic briefs by default: model synthesis costs a call per entity per meeting.
            built = build_brief(db, work.owner, entity_type, entity_id, run_id=run.id)
            return _finish(run, brief_id=built.id)
        plan = prepare_brief(db, work.owner, entity_type, entity_id)
        if plan is None:
            return _finish(run, unchanged=True)
        run_id = run.id
    if engine == "double":
        from .doubles import DoubleSynthesizer
        synthesizer = DoubleSynthesizer()
    else:
        from .native.synthesize import NativeSynthesizer
        synthesizer = NativeSynthesizer()
    sections, error = run_synthesis(plan, synthesizer)
    with session_scope() as db:
        run = db.get(ContextRun, run_id, with_for_update=True)
        if replay := _replay(run):
            return replay
        if reason := apply.guard(db, work.owner, run):
            return _stale(db, work, run, reason)
        built = commit_brief(db, work.owner, plan, sections, run_id=run_id, error=error)
        if (built.watermarks or {}).get("resynthesize"):
            enqueue_job(db, work.owner, "context_brief", {"entity_type": entity_type, "entity_id": entity_id,
                                                          "resynthesize": run_id})
        return _finish(run, brief_id=built.id, synthesis_error=error)


def placeholder(work: Work) -> dict:
    log.info("%s is a placeholder until its package lands; nothing to do", work.kind)
    return {"skipped": "not implemented yet"}


HANDLERS: dict[str, Callable[[Work], dict]] = {
    "context_ingest": ingest,
    "context_extract": extract,
    "context_link": link,
    "context_reconcile": reconcile,
    "context_embed": embed,
    "context_delete": delete,
    "context_granola_sync": granola_sync,
    "context_granola_backfill": granola_backfill,
    "context_brief": brief,
    "context_lint": placeholder,  # weekly lint (Stage E)
    "context_proposals": placeholder,  # D1: proposals from extract runs' stage_results["actions"]
}


# ---------------------------------------------------------------- pause, continuation and scheduling


def _state_row(db, owner: str) -> ContextOwnerState:
    db.execute(insert(ContextOwnerState).values(owner_id=owner).on_conflict_do_nothing(index_elements=["owner_id"]))
    return db.scalar(select(ContextOwnerState).where(ContextOwnerState.owner_id == owner).with_for_update()
                     .execution_options(populate_existing=True))


def pause(db, owner: str) -> dict:
    """Stop automatic processing: bump the epoch so in-flight results are refused. Sources, reads, manual
    corrections and approved publications are unaffected."""
    apply.lock(db, owner)
    state = _state_row(db, owner)
    if not state.paused:
        state.paused, state.paused_at, state.epoch, state.updated_at = True, now(), state.epoch + 1, now()
    db.flush()
    return {"paused": True, "epoch": state.epoch}


def resume(db, owner: str) -> dict:
    """Resume processing and queue the work a pause interrupted."""
    apply.lock(db, owner)
    state = _state_row(db, owner)
    if state.paused:
        state.paused, state.paused_at, state.updated_at = False, None, now()
    db.flush()
    return {"paused": False, "epoch": state.epoch, "continued": _continue(db, now(), owner=owner)}


def continue_source(db, owner: str, source_id: str, budget_usd: float) -> str | None:
    """The explicit continuation for a source left `awaiting_input` by its budget cap: extract again with a
    larger cap. Completed chunks are reused, so only the remainder costs."""
    source = db.get(ContextSource, source_id)
    if source is None or source.owner_id != owner or source.status == "deleted":
        return None
    revision = db.scalar(select(ContextSourceRevision.id).where(
        ContextSourceRevision.source_id == source_id, ContextSourceRevision.revision == source.current_revision))
    return enqueue_job(db, owner, "context_extract", {"source_id": source_id, "revision_id": revision,
                                                      "continue": True, "budget_usd": float(budget_usd)}).id


def _pending(db, owner: str, kind: str | None = None, **match) -> bool:
    query = select(Job.id).where(Job.owner_id == owner, Job.status.in_(PENDING))
    query = query.where(Job.kind == kind) if kind else query.where(Job.kind.like("context\\_%"))
    for key, value in match.items():
        query = query.where(Job.payload[key].astext == str(value))
    return bool(db.scalar(query.limit(1)))


def _continue(db, at: datetime, owner: str | None = None) -> int:
    """Re-enqueue, once each, runs whose `continue_after` has passed (pause, owner budget, failed chunks,
    embedding retry), for owners that are not paused and sources still current."""
    paused = set(db.scalars(select(ContextOwnerState.owner_id).where(ContextOwnerState.paused.is_(True))))
    query = select(ContextRun).where(ContextRun.stage_results.has_key("continue_after"),
                                     ContextRun.status.in_(("stale", "partial", "failed")))
    if owner:
        query = query.where(ContextRun.owner_id == owner)
    queued = 0
    for run in db.scalars(query.order_by(ContextRun.created_at, ContextRun.id).with_for_update(skip_locked=True)):
        results = dict(run.stage_results or {})
        if run.owner_id in paused or datetime.fromisoformat(results["continue_after"]) > at:
            continue
        results.pop("continue_after")
        run.stage_results = results
        source = db.get(ContextSource, run.source_id) if run.source_id else None
        if run.source_id and (source is None or source.status == "deleted"):
            continue
        if run.revision_id and source is not None and db.scalar(select(ContextSourceRevision.revision).where(
                ContextSourceRevision.id == run.revision_id)) != source.current_revision:
            continue
        if run.source_id and _pending(db, run.owner_id, source_id=run.source_id):
            continue
        payload = {k: v for k, v in (results.get("payload") or {}).items() if k not in ("requeues", "attempts")}
        continuation = int(payload.get("continuation", 0)) + (run.status != "stale")
        enqueue_job(db, run.owner_id, f"context_{run.stage}", {**payload, "continue": True,
                                                                 "continuation": continuation})
        queued += 1
    db.flush()
    return queued


def _granola_due(db, connection: ContextConnection, kind: str, at: datetime, every: timedelta) -> bool:
    if _pending(db, connection.owner_id, kind, connection_id=connection.id):
        return False
    last = db.scalar(select(Job.created_at).where(Job.owner_id == connection.owner_id, Job.kind == kind,
                                                  Job.payload["connection_id"].astext == connection.id)
                     .order_by(Job.created_at.desc()).limit(1))
    return last is None or at - last >= every


def schedule_due(db, at: datetime | None = None) -> dict:
    """Housekeeping tick (the worker's housekeeping calls it): Granola polls every `context_poll_minutes`, the
    90-day Granola backfill (only while the owner has no ingestion or extraction waiting, so new meetings go
    first), and continuations. Does nothing unless `context_enabled`; Granola also needs
    `context_granola_enabled`."""
    at = at or now()
    settings = context_settings.current()
    out = {"granola_sync": 0, "granola_backfill": 0, "continued": 0}
    if not settings.enabled:
        return out
    if settings.granola_enabled:
        paused = set(db.scalars(select(ContextOwnerState.owner_id).where(ContextOwnerState.paused.is_(True))))
        every = timedelta(minutes=settings.poll_minutes)
        for connection in db.scalars(select(ContextConnection).where(
                ContextConnection.connector == "granola", ContextConnection.enabled.is_(True))
                .order_by(ContextConnection.created_at, ContextConnection.id)):
            if connection.owner_id in paused:
                continue
            if _granola_due(db, connection, "context_granola_sync", at, every):
                enqueue_job(db, connection.owner_id, "context_granola_sync", {"connection_id": connection.id})
                out["granola_sync"] += 1
            progress = db.scalar(select(ContextCursor.value).where(ContextCursor.connection_id == connection.id,
                                                                   ContextCursor.name == "backfill"))
            busy = db.scalar(select(Job.id).where(
                Job.owner_id == connection.owner_id, Job.status.in_(PENDING),
                or_(Job.kind == "context_ingest", Job.kind == "context_extract")).limit(1))
            if not (progress or {}).get("completed_at") and not busy and \
                    _granola_due(db, connection, "context_granola_backfill", at, every):
                enqueue_job(db, connection.owner_id, "context_granola_backfill",
                            {"connection_id": connection.id, "days": 90})
                out["granola_backfill"] += 1
    out["continued"] = _continue(db, at)
    db.flush()
    return out
